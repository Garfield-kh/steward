# Validation — 2026-11-26

This directory separates refactoring checks from historical performance/episode
results. Tests used an isolated copy and a spare Ascend 910B3 card.

| Check | Result | Evidence |
| --- | --- | --- |
| CPU-only unit contracts | 10 passed | `python -m unittest discover -s tests -v` with `PYTHONPATH=src` |
| Inference algorithm syntax trees | 12 equal | `algorithm_equivalence.json` |
| Original vs cleaned kernels, CPU tensors | 3 cameras × 4 arrays, all bitwise equal | `renderer_cpu.json` |
| Original vs cleaned kernels, NPU tensors | 3 cameras × 4 arrays, all bitwise equal | `renderer_npu.json` |
| Real MuJoCo adapter → private NPU worker | RGB/depth/IDs match previous NPU archive on all 3 cameras; worker exits 0 | `adapter_smoke.json` |
| SAM original vs cleaned service | 3 requests × 10 prompts × scores/boxes/masks = 90 equal arrays | `sam_parity.json` |
| Cleaned service + pinned official SAM and compatibility patch | Same 90 arrays equal to original installation | `sam_pinned_parity.json` |
| Dependency patch | Clean apply to pinned commit; all 4 post-patch hashes match | `../dependencies/sam3-pin.json` |
| Packaging | Python wheel built successfully | `pyproject.toml` |

Syntax-tree comparison ignores comments/docstrings and relative import relocation;
it covers both rendering classes (excluding the removed unused baseline benchmark
render method), graph/decoder helpers and the four SAM inference methods. It does
not claim that launch, argument parsing or worker cleanup are unchanged. Those
changes are intentional and separately tested.

The four render arrays are RGB, pixel-centre depth, object-ID/type pairs and the
measured A800 resolved-depth sample. “Bitwise equal” means old NPU implementation
versus cleaned NPU implementation on the same backend, NOT NPU versus GPU or CPU
OpenGL. The CPU kernel check is a refactoring check, not a new CPU OpenGL benchmark.

Hardware test inputs were previously recorded head (2560×1920), wide (480×360)
and hand (848×480) camera states. SAM used the wide image with 10 fixed text prompts,
threshold 0.4, three identical requests. The first request captured one graph;
the next two exercised image/text caches and graph replay. Five detections per
request (three screws, two tables) produced non-empty masks. Scores, boxes and masks
were all compared, including empty detections for other prompts. This is a focused
regression sample, not a general SAM accuracy benchmark.

The original installed SAM source had no Git metadata. We inspected its compatibility
changes, extracted a four-file patch against official commit
`7f473cfd983d5eb43040ff608f66bff0d7bdcb53`, and ran the same inference with a separate
copy of that pinned source. We did not modify the installed SAM. The pinned version
and complete patch are shipped so the dependency is reproducible; no third-party
source tree or weights are copied into this proposed addition.

To reproduce kernel comparison, supply trusted local scene snapshots (NumPy archives
of MuJoCo scene metadata, not the raw simulation state archive):

```sh
PYTHONPATH=src python tests/compare_renderer.py \
  --original-dir /path/to/archived/renderer \
  --fixtures /path/to/scene/snapshots --device npu --output renderer_npu.json
```

For perception, load CANN, select an available card using
`ASCEND_RT_VISIBLE_DEVICES`, install the pinned patched SAM, then:

```sh
PYTHONPATH=src python tests/check_perception.py \
  --checkpoint /path/to/sam3.pt --image /path/to/wide.png \
  --output current.npz --reference /path/to/previous.npz
```

`historical_render_speed.csv` and `historical_parallel_speed.csv` summarize earlier
measurements, with hardware/methodology explained in the top-level README. The
CUDA service path was not part of this validation.
