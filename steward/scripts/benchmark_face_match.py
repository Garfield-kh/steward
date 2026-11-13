"""Measure the production face matcher on fixed printed textures; no robot or model calls.

Run from steward with .venv/bin/python scripts/benchmark_face_match.py --output FILE.
The reference contains the actual shipped texture at 24 cells per piece, with a box border.
This isolates matching arithmetic; it does not measure camera crops or task success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import statistics
import subprocess
import sys
import time
import tracemalloc
import types
from pathlib import Path

import numpy as np
from PIL import Image

from steward.ego.perception import match_face


def workload(name: str, grid: int) -> tuple[np.ndarray, list[tuple[np.ndarray, tuple]]]:
    """Fixed rotated pieces from a source image; expected locations are external test labels."""
    path = Path(__file__).resolve().parents[1] / "src/steward/assets/images" / name
    with Image.open(path) as source:
        source = source.convert("RGB")
        side = min(source.size) // grid * grid
        picture = np.asarray(source.crop((0, 0, side, side)).resize((grid * 24, grid * 24), Image.Resampling.LANCZOS), float)
    margin = 9 if grid == 3 else 10
    reference = np.empty((grid * 24 + 2 * margin, grid * 24 + 2 * margin, 3))
    reference[:] = (236, 232, 224)
    reference[margin:-margin, margin:-margin] = picture
    pieces = []
    for row in range(grid):
        for col in range(grid):
            turn = (row * grid + col) % 4
            face = np.rot90(picture[row * 24:(row + 1) * 24, col * 24:(col + 1) * 24], -turn).copy()
            pieces.append((face, (margin + row * 24 + 12, margin + col * 24 + 12, turn)))
    return reference, pieces


def main() -> None:
    """Save raw samples, input/output digests and memory measurements for one implementation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--revision", help="Load the unmodified perception module from this Git revision")
    parser.add_argument("--cases", nargs="+", choices=("cat", "greatwall", "dog"), default=["cat", "dog"])
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    module = Path(sys.modules[match_face.__module__].__file__)
    matcher, source = match_face, module.read_bytes()
    if args.revision:
        source = subprocess.check_output(["git", "show", f"{args.revision}:steward/src/steward/ego/perception.py"])
        baseline_module = types.ModuleType("_steward_face_match_baseline")
        sys.modules[baseline_module.__name__] = baseline_module
        exec(compile(source, f"git:{args.revision}:perception.py", "exec"), baseline_module.__dict__)
        matcher = baseline_module.match_face
    report = {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
              "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "perception_sha256": hashlib.sha256(source).hexdigest(), "source_revision": args.revision,
              "scope": "source-texture arithmetic, excluding camera, HTTP, actions and model; no task-success claim",
              "cases": []}
    for name in args.cases:
        grid = 4 if name == "dog" else 3
        image = "great-wall.jpg" if name == "greatwall" else name + ".jpg"
        reference, pieces = workload(image, grid)
        samples = []
        answers = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            answers = [matcher(face, reference) for face, _ in pieces]
            samples.append(time.perf_counter() - start)
        correct = sum(bool(answer["matches"]) and tuple(answer["matches"][0][key] for key in ("row", "col", "turns")) == expected
                      for answer, (_, expected) in zip(answers, pieces, strict=True))
        # Memory instrumentation is kept out of the runtime samples. Inputs and answers stay live.
        tracemalloc.start()
        memory_before = tracemalloc.get_traced_memory()[0]
        retained = [matcher(face, reference) for face, _ in pieces]
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        result = {"name": name, "pieces": len(pieces), "piece_cells": 24,
                  "reference_cells": list(reference.shape),
                  "input_sha256": hashlib.sha256(reference.tobytes() + b"".join(face.tobytes() for face, _ in pieces)).hexdigest(),
                  "runtime_samples_s": samples, "median_s": statistics.median(samples),
                  "correct_positions_and_turns": correct,
                  "traced_peak_bytes_above_baseline": peak - memory_before,
                  "traced_retained_bytes_above_baseline": current - memory_before,
                  "process_maxrss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  "answers": retained}
        report["cases"].append(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"{name}: median={result['median_s']:.4f}s correct={correct}/{len(pieces)} "
              f"peak={result['traced_peak_bytes_above_baseline']} retained={result['traced_retained_bytes_above_baseline']}", flush=True)


if __name__ == "__main__":
    main()
