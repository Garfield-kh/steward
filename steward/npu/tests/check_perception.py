"""Optional hardware check for eager preparation, graph capture and cache replay.

Use --image with a local PNG and --reference with a previous run's NPZ to check
bitwise output equality. Model weights and input images are supplied separately.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from steward_npu.perception import sam3_api_srv as service


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    config = service.parse_args(
        [
            "--ckpt",
            str(args.checkpoint),
            "--device-type",
            "npu",
            "--devices",
            "0",
            "--npu-text-cache",
            "16",
            "--npu-empty-box-fastpath",
            "--npu-empty-geometry",
            "--npu-shared-geometry",
            "--npu-batched-postprocess",
            "--npu-graphs",
        ]
    )
    service.initialize_runtime(config)
    replica = service.Replica(0)
    image = Image.open(args.image).convert("RGB")
    digest = hashlib.sha256(args.image.read_bytes()).hexdigest()
    prompts = [
        "airplane",
        "wing",
        "wheel",
        "screw",
        "screwdriver",
        "table",
        "robot",
        "gripper",
        "fuselage",
        "landing gear",
    ]
    arrays = {}
    for request in range(3):
        results = replica.segment(digest, image, prompts, 0.4)
        service.ACCEL.synchronize(replica.device)
        for prompt, result in enumerate(results):
            for label, data in zip(("scores", "boxes", "masks"), result):
                arrays[f"{request}_{prompt}_{label}"] = data
    np.savez_compressed(args.output, **arrays)
    report = {
        "arrays": len(arrays),
        "cache_hits": replica.cache_hits,
        "text_cache_hits": replica.text_cache_hits,
        "graph_replays": replica.graphs.replays,
    }
    if args.reference:
        with np.load(args.reference) as reference:
            same_keys = set(reference.files) == set(arrays)
            report["all_bitwise_equal"] = same_keys and all(
                np.array_equal(value, reference[key]) for key, value in arrays.items()
            )
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    if report.get("all_bitwise_equal") is False:
        raise SystemExit("Perception regression detected")


if __name__ == "__main__":
    main()
