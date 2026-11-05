"""Bounded, simulation-clock robot RGB recordings containing only sensor evidence."""
from __future__ import annotations

import base64
import copy
import json
import math
import queue
import threading
import time
import uuid
from dataclasses import dataclass

import mujoco

from steward.glrender import encode_image
from steward.net import HttpError
from steward.world.sensor_cameras import CameraRenderer, _sensor_kinematics


@dataclass(frozen=True)
class RecordingConfig:
    max_frames: int = 256
    max_retained: int = 8
    queue_capacity: int = 16
    max_bytes: int = 64 * 1024 * 1024
    timeout_s: float = 180

    def __post_init__(self):
        if (self.max_frames < 2 or min(self.max_retained, self.queue_capacity, self.max_bytes) < 1
                or not math.isfinite(self.timeout_s) or self.timeout_s <= 0):
            raise ValueError("RGB recording limits must be positive, with max_frames at least two")


class SensorRecording:
    """One robot's exact-cadence sensor samples; no navigation or model call is performed."""

    def __init__(self, sim, robot: str, fps: float, max_frames: int, config: RecordingConfig):
        self.id = "rgb_" + uuid.uuid4().hex
        self.sim, self.robot, self.fps, self.max_frames, self.config = sim, robot, fps, max_frames, config
        self.camera_ids = ("head_camera",)
        self.lock = threading.RLock()
        self.queue = queue.Queue(maxsize=config.queue_capacity)
        self.finishing, self.aborting, self.done = threading.Event(), threading.Event(), threading.Event()
        self.frames, self.error, self.state = [], None, "recording"
        self.captured, self.byte_count = 0, 0
        self.deadline = time.monotonic() + config.timeout_s
        with sim.lock:
            self.source, self.epoch = sim.model, sim._epoch
            self.model = copy.deepcopy(sim.model)
            self.origin, self.first_step = sim.time, sim.step_count
            self.stride = round(1 / (fps * sim.timestep))
            _, sensor = sim.camera_pose(robot)
            self.intrinsics = dict(sensor["intrinsics"])
            self.unsubscribe = sim.subscribe_steps(self.sample)
            self.sample()
            if sim.freeze_idle and sim._thread is not None:
                try:
                    sim.wait_until(self.origin + (max_frames - 1) / fps, hold=self.id, timeout=0)
                except TimeoutError:
                    pass  # timeout=0 registers the recorder as a clock puller without waiting
        self.thread = threading.Thread(target=self.run, name=f"rgb-record-{self.id}", daemon=True)
        try:
            self.thread.start()
        except RuntimeError:
            self.unsubscribe()
            sim.release(self.id)
            raise

    def fail(self, message: str):
        with self.lock:
            self.error = message
            self.aborting.set()

    def sample(self):
        """Copy sensor state at an exact simulation step; overload never silently drops frames."""
        try:
            with self.lock:
                if self.finishing.is_set() or self.aborting.is_set():
                    return
                if self.sim.model is not self.source or self.sim._epoch != self.epoch:
                    self.fail("RGB recording interrupted by a scene reset or model replacement")
                    return
                expected = self.first_step + self.captured * self.stride
                if self.sim.step_count < expected:
                    return
                if self.sim.step_count != expected:
                    self.fail("RGB recording missed its simulation sampling step")
                    return
                data = mujoco.MjData(self.model)
                mujoco.mj_copyData(data, self.model, self.sim.data)
                packet = _sensor_kinematics(self.sim, self.robot)
                packet["odometry"] = self.sim.odometry(self.robot)
                packet["odometry_frame"] = self.sim.odometry_frame_id
                self.queue.put_nowait((data, packet))
                self.captured += 1
                if self.captured == self.max_frames:
                    self.finishing.set()
                    self.state = "finalizing"
        except queue.Full:
            self.fail("RGB recording render queue is full; reduce simulation speed or FPS and retry")
        except Exception as error:  # callback failures belong to this recording, not the physics loop
            self.fail(f"RGB sensor sampling failed: {error}")

    def status(self):
        with self.lock:
            stamps = [frame["stamp"] for frame in self.frames]
            return {"recording_id": self.id, "status": self.state, "fps": self.fps,
                    "camera_ids": list(self.camera_ids),
                    "camera_id": "head_camera", "input_modalities": ["rgb"], "captured_frames": self.captured, "written_frames": len(stamps),
                    "max_frames": self.max_frames, "first_stamp": stamps[0] if stamps else None,
                    "last_stamp": stamps[-1] if stamps else None, "error": self.error,
                    "pose_source": "robot_odometry_and_joint_encoders", "clock": "simulation_time"}

    def finish(self, cancel=False):
        with self.sim.lock:
            with self.lock:
                if not self.done.is_set():
                    self.finishing.set()
                    self.state = "finalizing"
                    if cancel:
                        self.aborting.set()
        self.done.wait(20)
        return self.status()

    def result(self):
        with self.lock:
            if self.state != "succeeded":
                raise HttpError(409, f"RGB recording is {self.state}; finish recording successfully before exporting")
            return {**self.status(), "format": "robot_rgb_recording_v1", "frames": list(self.frames)}

    def run(self):
        renderer = CameraRenderer()
        renderer.source = renderer.model = self.model
        try:
            while not self.finishing.is_set() or not self.queue.empty():
                self.sim.renew(self.id)
                if self.aborting.is_set():
                    break
                if time.monotonic() >= self.deadline:
                    self.fail("RGB recording timed out; stop it after navigation and confirm the simulation is advancing")
                    break
                try:
                    renderer.data, packet = self.queue.get(timeout=0.1)
                except queue.Empty:
                    continue  # no due sample yet; cancellation and timeout remain active
                batch = renderer.render_data(self.robot, self.camera_ids, {"head_camera": self.intrinsics}, self.epoch)
                images = {}
                for camera in self.camera_ids:
                    view = batch["views"][camera]
                    image = {key: view[key] for key in ("width", "height", "K")}
                    image["image"] = "data:image/jpeg;base64," + base64.b64encode(encode_image(view["rgb"], "jpeg")).decode("ascii")
                    images[camera] = image
                frame = {**packet, "middle": images["head_camera"]}
                expected_stamp = self.origin + len(self.frames) / self.fps
                if abs(frame["stamp"] - expected_stamp) > 1e-6:
                    raise ValueError("RGB timestamps differ from the requested FPS")
                size = len(json.dumps(frame).encode("utf-8"))
                with self.lock:
                    if self.byte_count + size > self.config.max_bytes:
                        raise ValueError("RGB recording exceeds its configured byte limit")
                    self.frames.append(frame)
                    self.byte_count += size
        except Exception as error:  # the failed recording remains visible through its status
            self.fail(f"RGB recording failed: {error}")
        finally:
            for cleanup in (self.unsubscribe, renderer.close, lambda: self.sim.release(self.id)):
                try:
                    cleanup()
                except Exception as error:
                    self.fail(f"RGB recording cleanup failed: {error}")
            with self.lock:
                while not self.queue.empty():
                    self.queue.get_nowait()
                self.state = "failed" if self.error else "cancelled" if self.aborting.is_set() else "succeeded"
                self.done.set()


class SensorRecordings:
    """Own bounded sensor recordings; completed handles are retained until their retention limit."""

    def __init__(self, sim, config: RecordingConfig | None = None):
        self.sim, self.config = sim, config or RecordingConfig()
        self.lock, self.closed, self.jobs, self.keys = threading.RLock(), False, {}, {}

    def start(self, robot: str, body: dict):
        fps, frames, key = body.get("fps", 2), body.get("max_frames", self.config.max_frames), body.get("request_key")
        if (type(fps) not in (int, float) or not math.isfinite(fps) or not 0 < fps <= 30
                or type(frames) is not int or not 2 <= frames <= self.config.max_frames
                or not isinstance(key, str) or not 8 <= len(key) <= 128):
            raise HttpError(400, "Use fps in (0,30], a bounded max_frames >= 2 and a stable 8–128 character request_key")
        sim = self.sim()
        period = 1 / (fps * sim.timestep)
        if abs(period - round(period)) > 1e-6:
            raise HttpError(400, "RGB sampling period must be an integer multiple of the simulation step")
        with self.lock:
            if self.closed:
                raise HttpError(503, "RGB recording service is closed")
            identity = (robot, key)
            if identity in self.keys:
                if self.keys[identity] not in self.jobs:
                    raise HttpError(410, "The recording for this request_key expired; do not repeat an uncertain start")
                job = self.jobs[self.keys[identity]]
                if job.fps != fps or job.max_frames != frames:
                    raise HttpError(409, "request_key already belongs to different recording options")
                return job.status()
            if any(job.robot == robot and not job.done.is_set() for job in self.jobs.values()):
                raise HttpError(409, "This robot already has an active RGB recording; query or stop it first")
            while len(self.jobs) >= self.config.max_retained:
                expired = next((id for id, job in self.jobs.items() if job.done.is_set()), None)
                if expired is None:
                    raise HttpError(409, "RGB recording capacity exhausted")
                del self.jobs[expired]
            job = SensorRecording(sim, robot, fps, frames, self.config)
            self.jobs[job.id], self.keys[identity] = job, job.id
            return job.status()

    def get(self, robot: str, id: str):
        with self.lock:
            job = self.jobs.get(id)
            if job is None or job.robot != robot:
                raise HttpError(404, "RGB recording not found; handles do not survive host restart or retention eviction")
            return job

    def interrupt(self, message: str):
        with self.lock:
            jobs = list(self.jobs.values())
        for job in jobs:
            if not job.done.is_set():
                job.fail(message)
        for job in jobs:
            job.thread.join()

    def close(self):
        with self.lock:
            self.closed = True
        self.interrupt("RGB recording service closed")
