---
description: "Generate novel-view RGBD from observations: examples, tool flow and package responsibilities."
---

# Perception model skills guide

English | [中文](README.zh.md)

## Summary

NovelAnything and rgbd_video_gen use observed images and camera poses to generate RGB and predicted depth at a specified virtual viewpoint, including positions or directions not directly covered by the original camera. The virtual camera position, orientation and intrinsics describe the target view; a gripper view is one application. The existing `head_camera` and `wide_camera` retain their mounts; the example below uses a head-camera RGB time sequence.

The rgbd_video_gen preparation tools accept explicit target positions and orientations in the reference-camera or gripper frame, including multiple targets. The current NovelAnything convenience preparation tool generates a fixed gripper view; other targets require a compliant scene ZIP containing their poses.

## Table of contents

- [Prerequisites](#prerequisites)
- [Copyable examples](#copyable-examples)
- [Virtual camera and sampling](#virtual-camera-and-sampling)
- [Flow and package reference](#flow-and-package-reference)
- [Results and recovery](#results-and-recovery)

<a id="prerequisites"></a>

## Prerequisites

Start Steward, load the scene for the task, and select the robot-1 chat in the Web console. The robot `ego` profile needs the selected model bundle, and both plugins and Python service must load the integrated version; see installation entries for [NovelAnything](../../dsh/packages/dsh-bundle-novelanything/README.md) and [rgbd_video_gen](../../dsh/packages/dsh-bundle-rgbd-video-gen/README.md). Check model readiness with `novelanything_health` or `rgbd_video_gen_capabilities`, respectively.

Table identification, direction estimates and route feasibility in the example depend on the current scene; the robot should report missing information or an inability to arrive safely.

<a id="copyable-examples"></a>

## Copyable examples

This example requests a novel view at the gripper position and can be pasted into the robot chat. For another view, change the virtual camera position and orientation and state the reference coordinate frame.

```text
输入：机器人移动到桌子正南方向，面向桌子，距桌 0.5 m；移动过程中，用头部相机按 2 FPS 连续采样 RGB 图像作为输入。
虚拟相机：到位后，置于手爪 link 原点，沿手爪 +X 方向看，画面上方对应 +Z，使用右手坐标系；内参复用头部相机。
生成：使用 rgbd_video_gen 生成该虚拟视角下的 RGBD 图像。
```

<a id="virtual-camera-and-sampling"></a>

## Virtual camera and sampling

| Item | Convention and example |
|---|---|
| Physical input | RGB from the same `head_camera` at different timestamps; `wide_camera` continues serving existing perception |
| Sampling clock | Simulation time: 2 FPS means 0.5 seconds between samples, independent of agent response latency |
| Pose source | Synchronized robot odometry, joint encoders and body kinematics; no World absolute truth fills missing evidence |
| Virtual position | Specified by the target viewpoint; this example uses the current gripper link origin when preparing input |
| Orientation | Specify viewing and image-up directions; this example looks along gripper +X with +Z up, making image-right gripper −Y |
| Camera coordinates | OpenCV: x right, y down, z forward; in this example they correspond to gripper −Y, −Z, +X and retain a right-handed frame |
| Intrinsics | Reuse head-camera pixel intrinsics; remote models may resize or crop for output specifications, so inspect result metadata |
| Model input count | NovelAnything uses 2–16 context images; rgbd_video_gen uses 1–8 source images; retained recording frames may exceed inference inputs |
| Generated content | RGB and predicted depth at the requested virtual viewpoint; generation adds no physical camera and does not modify Ego-State |

<a id="flow-and-package-reference"></a>

## Flow and package reference

Both models share head-camera recording and provide their own input preparation, remote calls and download tools. Paths below are relative to `steward/dsh/packages/`; the four roles are bundle installation/configuration, service definition, client execution and agent tool calls.

| Folder | Purpose | Main contents |
|---|---|---|
| [dsh-bundle-novelanything](../../dsh/packages/dsh-bundle-novelanything/README.md) | Installation and configuration | Installs the NovelAnything client, tools and `skills/novelanything/SKILL.md` in the robot Harness; configures endpoint and input/output directories |
| [dsh-novelanything](../../dsh/packages/dsh-novelanything/README.md) | Service definition | Defines health, upload, submission, status, result and download interfaces, plus parameter, result and error types |
| [dsh-novelanything-client](../../dsh/packages/dsh-novelanything-client/README.md) | Preparation and HTTP requests | Exports recording evidence through Ego, prepares a scene ZIP, connects to the service, recovers local request records, and downloads artifacts with a manifest |
| [dsh-tool-novelanything](../../dsh/packages/dsh-tool-novelanything/README.md) | Agent tools | Registers `novelanything_prepare_gripper_view`, `novelanything_upload`, `novelanything_submit`, `novelanything_status` and other tools, delegating execution to the service |
| [dsh-bundle-rgbd-video-gen](../../dsh/packages/dsh-bundle-rgbd-video-gen/README.md) | Installation and configuration | Installs the rgbd_video_gen client, tools and `skills/rgbd-video-gen/SKILL.md` in the robot Harness; configures endpoint and input/output directories |
| [dsh-rgbd-video-gen](../../dsh/packages/dsh-rgbd-video-gen/README.md) | Service definition | Defines capabilities, scene preparation, generation, status, result, download and cancellation interfaces, plus parameter, result and error types |
| [dsh-rgbd-video-gen-client](../../dsh/packages/dsh-rgbd-video-gen-client/README.md) | Preparation and HTTP requests | Prepares scene JSON from recording evidence, calls remote generation, handles recovery and downloads, and verifies sizes and SHA256 |
| [dsh-tool-rgbd-video-gen](../../dsh/packages/dsh-tool-rgbd-video-gen/README.md) | Agent tools | Registers `rgbd_video_gen_prepare_recording`, `rgbd_video_gen_submit`, `rgbd_video_gen_status` and other tools, delegating execution to the service |

The `tool` packages expose executable operations. The bundled `skill` explains their order, input restrictions, frame selection and result interpretation. Together they support a complete task from natural language to model output.

| Stage | Shared capabilities or NovelAnything | rgbd_video_gen |
|---|---|---|
| Record and move | `ego_rgb_recording_start` → existing `ego_predict` / `skill_run` / `skill_status` → `ego_rgb_recording_stop` / `ego_rgb_recording_status` | Same as left |
| Prepare | `novelanything_prepare_gripper_view` → `archive_path`, `context`, `target` | `rgbd_video_gen_prepare_recording` → `scene_path`, source indices and exact submission counts |
| Submit | `novelanything_upload` → `novelanything_submit` | `rgbd_video_gen_submit` |
| Track and download | `novelanything_status` → `novelanything_result` → `novelanything_download` | `rgbd_video_gen_status` → `rgbd_video_gen_result` → `rgbd_video_gen_download` |

Python capture and export use [sensor_cameras.py](../../src/steward/world/sensor_cameras.py), [rgb_recording.py](../../src/steward/world/rgb_recording.py), [recording_evidence.py](../../src/steward/ego/recording_evidence.py) and [novelanything_scene.py](../../src/steward/ego/novelanything_scene.py); `dsh-ego`, `dsh-ego-runtime-client` and `dsh-tool-ego` expose robot tools.

Loose prediction images use `source_reconstruction_00.jpg` for input-view reconstruction and `target_prediction_00.jpg` for the first requested target; target numbering starts at zero independently of source count. Remote `filename` remains the download selector; `local_filename` and `path` identify the local file. Re-downloading verified legacy `pred_*.jpg` files renames them without another transfer. Server `result.json`, ZIP entries and depth keys keep their original names; use manifest `targets` for the target RGB/depth mapping.

<a id="results-and-recovery"></a>

## Results and recovery

A success report includes terminal state `succeeded`, actual selected frame count and target timestamp, downloaded RGB/depth paths and `local_manifest_path`, explicitly identifying predictions. Default directories are relative to the robot workspace: inputs reside in `inputs/novelanything/` or `inputs/rgbd_video_gen/`; outputs reside in `outputs/<model>/<job_id>/`. Use returned absolute paths instead of concatenating or guessing filenames. The scene formats are not interchangeable; generated RGBD does not automatically become an additional physical-camera view in the console.

On recording failure, scene reset, lost evidence or generation failure, stop dependent steps and report the cause. Recording handles are process-local; prepare and save inputs before restart or retention eviction. Stopping recording does not stop robot motion. NovelAnything `UNKNOWN_OUTCOME` requires operator reconciliation rather than resubmission with a new key; its API has no remote cancellation. Explicitly request rgbd_video_gen cancellation with `rgbd_video_gen_cancel`; stopping a chat does not establish remote cancellation.
