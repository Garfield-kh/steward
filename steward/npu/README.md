# Optional Ascend NPU backends

English | [中文](README.zh.md)

This is an additive Python extension for the toy assembly profile, tested against
commit `07147e7764f5dce30838a9f3632d497f8067aeb8`. It does not modify the original
Steward/GPU source, assembly policy, skills, task or success thresholds. Importing
`steward_npu` does not enable a backend. This directory contains source, focused
tests and small validation summaries; model weights are not included.

## Components

- `renderer/kernels.py`: analytic box/cylinder intersections, tiled candidate culling,
  RGB shading, depth, object IDs and cubemap sampling.
- `renderer/dynamic_renderer.py`: opt-in MuJoCo adapter and private worker lifecycle.
- `renderer/render_worker.py`: NPU tensor execution in a separate Python process.
- `perception/sam3_api_srv.py`: configurable SAM 3 text-prompt image API, CUDA or NPU.
- `perception/graph_accel.py` and `decoder_meta.py`: optional NPU graph replay and
  host metadata caching. Optimizations are explicitly enabled with flags.

Physics and scene metadata remain on CPU. Pixel intersections, shadows, lighting
and texture sampling run on NPU; CPU OpenGL images are not inputs to that rendering.
SAM inference runs on the selected accelerator.

## Dependencies and installation

The tested accelerator environment is Ascend 910B3 (64 GiB), CANN 8.5.2,
Python 3.11, PyTorch 2.10.0, torch_npu 2.10.0 and MuJoCo 3.17.0. Use a matching CANN/driver/PyTorch/
torch_npu installation supplied for your machine; this package does not install or
upgrade those system components. SAM 3 and a locally provisioned `sam3.pt` checkpoint
are additional requirements for the perception service. No checkpoint is bundled.
Use the pinned official SAM commit and four-file compatibility patch described in
[`dependencies/README.md`](dependencies/README.md); an arbitrary unmodified SAM
installation is not equivalent to the tested environment.
The measured SAM installation identifies itself as version 0.1.0; see validation
notes for the exact scope of this environment-based test.

In each Python environment that will import this extension:

```sh
python -m pip install --no-deps -e /path/to/this/directory
```

Use `--no-deps` only after provisioning the dependencies from `pyproject.toml`.
Steward itself must also be installed in the simulation Python environment.
The worker environment needs NumPy, PyTorch and torch_npu; the simulation environment
needs MuJoCo and the original Steward dependencies. The two may be separate.
Load your CANN environment and required shared-library paths in your shell before
starting. No server address, username, checkpoint directory or conda path is embedded
in the Python implementation.

## Enable rendering

```sh
export STEWARD_ASSEMBLY_PROFILE=toy
export NPU_DEVICE=0
# Optional when the simulation and NPU environments are different:
export STEWARD_NPU_PYTHON=/path/to/npu/environment/bin/python
python -m steward_npu --help
# Replace --help with the same arguments used with python -m steward.cli.
```

This wrapper selects `STEWARD_RENDER_BACKEND=npu_dynamic` and defaults
`STEWARD_NPU_MATCH_A800_DEPTH=1`. Ordinary `python -m steward.cli` retains the original
backend. `NPU_DEVICE` selects one physical card before the child imports torch_npu;
the child uses logical device 0. Each simulation process owns one worker, shared by
its camera instances. Calls within a process are serialized; episode concurrency
uses separate processes. Do not share an adapter across a process fork.

Optional worker settings: `STEWARD_NPU_RENDER_TILE=512`,
`STEWARD_NPU_RENDER_THREADS=4`, `STEWARD_NPU_RENDER_MEMORY_FRACTION=0.25`.
The memory fraction is an allocator cap, not measured total device use or a scheduler
reservation. The optional `STEWARD_OPERATOR_RECORD_DIR` retains the existing timing
log convention (a JSONL file in its parent directory).

## Start perception

```sh
export ASCEND_RT_VISIBLE_DEVICES=0
python -m steward_npu.perception.sam3_api_srv \
  --device-type npu --devices 0 --ckpt /path/to/sam3.pt \
  --host 127.0.0.1 --port 18230 --memory-fraction 0.45 \
  --npu-text-cache 16 --npu-empty-box-fastpath --npu-empty-geometry \
  --npu-shared-geometry --npu-batched-postprocess --npu-graphs
```

`--devices` contains logical indices after visibility filtering (`--gpus` remains
an alias). One model replica is created per index. Text cache capacity is 0–32 and
graph capacity is 1–4 (`--npu-graph-cache`); unsupported graph shapes use eager NPU
inference. The current graph path covers 10, 11, 15 and 21 prompts. Without the
optimization flags the service uses its ordinary eager path. CUDA mode accepts
`--device-type cuda` with no NPU flags.

The API provides `GET /api/health` and `POST /api/segment`. POST is multipart with
`file` containing an image and `options` containing JSON, for example:
`{"prompts":[{"text":"screw"}],"threshold":0.4,"overlay":false}`.
Results contain scores, pixel XYXY boxes, normalized XYWH boxes and base64 PNG masks.
Point/box prompts and video sessions are unsupported. Bind to a trusted interface;
this development service has no authentication. `--log` writes optional request
metrics. `--profile-stages` synchronizes stages and changes measured latency.
`--save-dir`/`--save-n` optionally retain a bounded number of diagnostic requests.

## Measured scope and limitations

The renderer supports perspective cameras, opaque boxes/circular cylinders,
directional lights and RGB cubemap textures. Meshes, transparency, general OpenGL
materials, mipmaps and exact multisample RGB are not implemented. Unsupported
geometry fails rather than silently switching to CPU rendering. The A800 depth
sample mode was measured for 0/4 samples, not arbitrary GPU hardware. Supported
scenes are still approximate: edge depth errors can exceed one metre, despite high
object-ID agreement. Do not describe NPU images as pixel-identical to GPU/CPU images.

Earlier matched single-stream measurements (seconds per frame, scene update plus
RGB/depth/ID output copied to host, after warm-up):

| Camera | A800 SXM4 80GB | Kunpeng 920 CPU | Ascend 910B3 |
| --- | ---: | ---: | ---: |
| Head 2560×1920 | 0.2561 | 1.3095 | 0.6702 |
| Wide 480×360 | 0.00535 | 0.1410 | 0.1859 |
| Hand 848×480 | 0.01158 | 0.1600 | 0.2122 |

CPU rendering used OSMesa/llvmpipe with 8 threads on a four-socket Kunpeng 920
(192 cores); the A800 host used two Xeon Platinum 8358P CPUs (64 cores total).
NPU accelerated the large head image versus CPU,
but was slower for small images and slower than A800 in all three cases.

A separate one/two-stream NPU test measured per-client latency of 0.723/0.949 s
(head), 0.198/0.239 s (wide) and 0.237/0.265 s (hand). Aggregate throughput scaled
1.52×/1.58×/1.75×. Corresponding CPU latencies were 1.328/1.301, 0.171/0.155 and
0.191/0.168 s. No matched two-stream GPU result is available. Full CSVs are in
`validation/`; these pure-render tests do not predict end-to-end episode duration.

First batch with the unchanged assembly policy: 8 episodes, four cards, two episodes per card, up to
180 minutes per episode: **1 success under the 20 mm criterion; 0 under 6 mm**.
The successful episode first reached four screws after about 81.7 minutes and ended
after 131.1 minutes; maximum lateral error was 16.55 mm. A 32-episode study is in
progress.

## Validation

```sh
PYTHONPATH=src python -m unittest discover -s tests -v
```

`tests/compare_renderer.py` accepts an external original adapter directory and scene
snapshot directory, and checks RGB/depth/IDs plus the measured resolved-depth pass.
Hardware validation results and their scope are recorded in `validation/README.md`.
