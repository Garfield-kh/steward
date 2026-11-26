"""Private local pipe worker. Never accept pickle data from a network client."""

import os
import pickle
import sys
import time
import traceback


def main():
    import torch
    import torch_npu  # Registers the NPU backend before constructing tensors.
    from .kernels import TiledRenderer

    fraction = float(os.environ.get("STEWARD_NPU_RENDER_MEMORY_FRACTION", "0.25"))
    tile = int(os.environ.get("STEWARD_NPU_RENDER_TILE", "512"))
    threads = int(os.environ.get("STEWARD_NPU_RENDER_THREADS", "4"))
    if not 0 < fraction <= 1 or tile < 1 or threads < 1:
        raise ValueError("Invalid renderer memory fraction, tile size or thread count")
    torch.npu.set_device(
        0
    )  # Parent restricts visibility to the explicitly selected card.
    torch.set_num_threads(threads)
    torch.npu.set_per_process_memory_fraction(fraction, 0)
    while True:
        try:
            scene = pickle.load(sys.stdin.buffer)
        except EOFError:
            return
        try:
            started = time.perf_counter()
            renderer = TiledRenderer(scene, "npu", tile=tile)
            renderer.sync()
            prepared = time.perf_counter()
            buffers = renderer.render()
            depth_begin = time.perf_counter()
            depth_mode = "pixel_center"
            if os.environ.get("STEWARD_NPU_MATCH_A800_DEPTH") == "1":
                samples = int(scene.get("raster_samples", 4))
                if samples not in (0, 4):
                    raise ValueError(
                        "Only measured A800 0/4-sample depth modes supported"
                    )
                if samples == 4:
                    buffers = (
                        buffers[0],
                        renderer.render_depth_at(0.375, 0.875),
                        buffers[2],
                    )
                    depth_mode = "a800_msaa4_sample0"
            renderer.sync()
            done = time.perf_counter()
            result = {
                "buffers": buffers,
                "timing": {
                    "prepare_s": prepared - started,
                    "render_s": done - prepared,
                    "worker_s": done - started,
                    "depth_alignment_s": done - depth_begin,
                    "depth_sample_mode": depth_mode,
                },
            }
            del renderer
        except Exception as exc:
            result = {"error": f"{type(exc).__name__}: {exc}"}
            traceback.print_exc(file=sys.stderr)
        pickle.dump(result, sys.stdout.buffer, protocol=5)
        sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
