"""Optional hardware regression: compare with a separately supplied archived adapter.

Run with --original-dir, --fixtures, --device cpu|npu, --output report.json.
The archive and recorded fixtures are deliberately not distributed in this package.
"""

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from steward_npu.renderer.kernels import TiledRenderer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "npu"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.device == "npu":
        import torch_npu

        torch.npu.set_device(0)
        torch.npu.set_per_process_memory_fraction(0.25, 0)
    torch.set_num_threads(4)
    sys.path.insert(0, str(args.original_dir))
    previous = importlib.import_module("npu_render_optimized").TiledRenderer
    records = []
    for name in ("1649_head_camera.npz", "1649_wide_camera.npz", "824_hand_camera.npz"):
        scene = dict(np.load(args.fixtures / name))
        rendered = []
        for cls in (previous, TiledRenderer):
            begin = time.perf_counter()
            renderer = cls(scene, args.device, tile=512)
            buffers = renderer.render()
            depth = renderer.render_depth_at(0.375, 0.875)
            renderer.sync()
            rendered.append((*buffers, depth))
            del renderer
        equal = [bool(np.array_equal(a, b)) for a, b in zip(*rendered)]
        records.append(
            {
                "fixture": name,
                "device": args.device,
                "rgb_depth_ids_resolved_depth_bitwise_equal": equal,
            }
        )
        print(json.dumps(records[-1]), flush=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n")
    if not all(all(r["rgb_depth_ids_resolved_depth_bitwise_equal"]) for r in records):
        raise SystemExit("Renderer regression detected")


if __name__ == "__main__":
    main()
