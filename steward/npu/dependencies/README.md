# Reproducible SAM dependency

The server's pre-existing SAM installation includes local compatibility changes;
installing an arbitrary unmodified SAM release is not equivalent. This directory
makes that dependency explicit without vendoring the SAM source or weights.

Start with the official repository at the pinned commit:

```sh
git clone https://github.com/facebookresearch/sam3.git sam3
cd sam3
git checkout --detach 7f473cfd983d5eb43040ff608f66bff0d7bdcb53
git apply --check /path/to/steward-npu/dependencies/sam3-ascend-image.patch
git apply /path/to/steward-npu/dependencies/sam3-ascend-image.patch
# In the already provisioned NPU Python environment:
python -m pip install --no-deps -e .
```

Provision SAM's remaining dependencies for that environment separately. `--no-deps`
prevents pip from replacing the tested torch/torch_npu installation. The SAM package
and checkpoint retain their own upstream license and access requirements.

The four-file patch:

- selects NPU for positional/decoder metadata initialized on the accelerator;
- lets torch_npu select its attention implementation rather than forcing CUDA flash attention;
- permits image-only imports when CUDA Triton EDT is unavailable, while explicitly
  rejecting use of that unsupported video-tracker operation;
- uses `importlib.resources` for tokenizer assets instead of deprecated `pkg_resources`.

Register `torch_npu` before importing SAM (the service does this). The patch retains
CUDA behavior when the NPU runtime is unavailable. It does not add video tracking,
training or a general multi-backend SAM implementation.

`sam3-pin.json` records the official commit and pre/post hashes of all four changed
files. The package-level validation notes distinguish tests against the historical
installation from tests against this reproducible dependency. No original server
installation is modified by the validation process.
