---
kind: upgrade-guide
description: "Optional model preparation uses primary-camera RGB evidence without changing the robot camera layout."
---

# Steward RGB-only recording and NovelAnything inputs

English | [中文](guide.zh.md)

## Change

Optional model preparation uses only `head_camera` RGB with measured poses and intrinsics; NovelAnything ZIPs omit depth files and `depth_path`. The model predicts target RGB and depth. Existing head and wide RGB-D cameras retain their mounts, intrinsics and perception responsibilities; no camera is added or removed.

## Migration

1. Rebuild the Steward plugins with `pnpm --dir steward/dsh run build` from the repository root, then restart Steward and robot chat to load the Python and plugin changes. Restarting loses live recording handles; retain any needed local archives first.
2. Use `ego_rgb_recording_start/status/stop` for simulation-clock RGB evidence. Robot routes are `/sensors/rgb/record/*` and `/ego/rgb/record/*`; preparation uses `/ego/rgb/prepare-novelanything`. Recording handles begin with `rgb_`. The existing `/sensors/rgbd` interface remains the perception stream.
3. Use `novelanything_prepare_gripper_view` on a successful RGB recording and upload its returned ZIP. Confirm `metadata.input_modalities` equals `["rgb"]` and the archive contains no `depth_path` or depth images. The remote deployment must support RGB-only inputs (API 1.3.0 or later); do not add depth to work around an older server.
