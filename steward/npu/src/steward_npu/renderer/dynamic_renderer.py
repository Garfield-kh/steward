"""Explicit experimental NPU renderer for live MuJoCo scenes; no CPU pixel fallback."""

import os, pickle, subprocess, time, json, sys, threading
from pathlib import Path
import mujoco
import numpy as np

_ORIGINAL = mujoco.Renderer
_worker = None
_worker_lock = threading.RLock()


def snapshot(model, scene, width, height):
    """Copy live scene metadata; no CPU-rendered pixels enter the NPU worker."""
    geoms = [scene.geoms[i] for i in range(scene.ngeom)]
    lights = [scene.lights[i] for i in range(scene.nlight)]
    a = {
        k: np.array([getattr(g, k) for g in geoms])
        for k in [
            "type",
            "size",
            "pos",
            "mat",
            "rgba",
            "objid",
            "objtype",
            "segid",
            "matid",
            "shininess",
            "specular",
            "reflectance",
            "emission",
        ]
    }
    for k in [
        "pos",
        "dir",
        "ambient",
        "diffuse",
        "specular",
        "attenuation",
        "type",
        "cutoff",
        "exponent",
        "headlight",
        "castshadow",
    ]:
        a["light_" + k] = np.array([getattr(l, k) for l in lights])
    for k in [
        "pos",
        "forward",
        "up",
        "frustum_center",
        "frustum_width",
        "frustum_bottom",
        "frustum_top",
        "frustum_near",
        "frustum_far",
        "orthographic",
    ]:
        a["camera_" + k] = np.mean(
            [np.array(getattr(c, k)) for c in scene.camera], axis=0
        )
    a.update(
        shadows=np.array(bool(scene.flags[int(mujoco.mjtRndFlag.mjRND_SHADOW)])),
        width=np.array(width),
        height=np.array(height),
        tex_data=model.tex_data.copy(),
    )
    for k in [
        "tex_adr",
        "tex_height",
        "tex_width",
        "tex_nchannel",
        "tex_type",
        "mat_texid",
        "mat_texrepeat",
        "mat_texuniform",
        "mat_rgba",
    ]:
        a[k] = getattr(model, k).copy()
    a["raster_samples"] = np.array(int(model.vis.quality.offsamples))
    return a


class DynamicNpuRenderer:
    def __init__(self, model, height=240, width=320, max_geom=10000):
        self._model = model
        self._scene = mujoco.MjvScene(model, maxgeom=max_geom)
        self._scene_option = mujoco.MjvOption()
        self.height = height
        self.width = width
        self._mode = 0
        self._buffers = None
        self._closed = False

    @property
    def scene(self):
        return self._scene

    def update_scene(self, data, camera=-1, scene_option=None):
        _ORIGINAL.update_scene(self, data, camera, scene_option)
        self._buffers = None

    def enable_depth_rendering(self):
        self._mode = 1

    def disable_depth_rendering(self):
        self._mode = 0

    def enable_segmentation_rendering(self):
        self._mode = 2

    def disable_segmentation_rendering(self):
        self._mode = 0

    def render(self, *, out=None):
        with _worker_lock:
            return self._render_locked(out=out)

    def _render_locked(self, *, out=None):
        global _worker
        if self._closed:
            raise RuntimeError("Renderer closed")
        if self._buffers is None:
            if _worker is None or _worker.poll() is not None:
                if "NPU_DEVICE" not in os.environ:
                    raise RuntimeError("Explicit NPU_DEVICE required")
                device = os.environ["NPU_DEVICE"]
                if not device.isdecimal():
                    raise ValueError(
                        "NPU_DEVICE must be one non-negative physical card index"
                    )
                env = os.environ.copy()
                env["ASCEND_RT_VISIBLE_DEVICES"] = device
                python = os.environ.get("STEWARD_NPU_PYTHON", sys.executable)
                _worker = subprocess.Popen(
                    [python, "-m", "steward_npu.renderer.render_worker"],
                    env=env,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                )
            started = time.perf_counter()
            try:
                pickle.dump(
                    snapshot(self._model, self.scene, self.width, self.height),
                    _worker.stdin,
                    protocol=5,
                )
                _worker.stdin.flush()
                result = pickle.load(_worker.stdout)
            except (BrokenPipeError, EOFError, OSError) as exc:
                shutdown_worker()
                raise RuntimeError(
                    "NPU render worker exited before returning a frame"
                ) from exc
            if "error" in result:
                raise RuntimeError("NPU renderer: " + result["error"])
            self._buffers = result["buffers"]
            self.last_timing = {
                **result["timing"],
                "adapter_wall_s": time.perf_counter() - started,
            }
            recording = os.environ.get("STEWARD_OPERATOR_RECORD_DIR")
            if recording:
                with (Path(recording).parent / "npu_render_timing.jsonl").open(
                    "a"
                ) as f:
                    f.write(
                        json.dumps(
                            {
                                "width": self.width,
                                "height": self.height,
                                **self.last_timing,
                            }
                        )
                        + "\n"
                    )
        result = self._buffers[self._mode]
        if out is not None:
            np.copyto(out, result)
            return out
        return result

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._buffers = None


def shutdown_worker():
    """Close the private child process; bound both graceful and forced shutdown."""
    global _worker
    with _worker_lock:
        worker, _worker = _worker, None
        if worker is None:
            return
        try:
            if worker.stdin is not None:
                try:
                    worker.stdin.close()
                except BrokenPipeError:
                    pass  # Child has already exited.
            try:
                worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                worker.terminate()
                try:
                    worker.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait()
        finally:
            if worker.stdout is not None:
                worker.stdout.close()


import atexit

atexit.register(shutdown_worker)


def install():
    if os.environ.get("STEWARD_RENDER_BACKEND") != "npu_dynamic":
        raise RuntimeError("STEWARD_RENDER_BACKEND=npu_dynamic required")
    mujoco.Renderer = DynamicNpuRenderer
