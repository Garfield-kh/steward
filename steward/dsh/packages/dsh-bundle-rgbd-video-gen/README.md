---
description: "Add rgbd_video_gen novel-view RGB/depth generation to the robot Ego Harness using a remote service, without a local NPU installation."
kind: "package-bundle"
---

# @steward/dsh-bundle-rgbd-video-gen

English | [中文](README.zh.md)

## Summary

Add rgbd_video_gen novel-view RGB/depth generation to the robot Ego Harness using a remote service, without a local NPU installation.

## Table of Contents

- [Use this package](#use-this-package)
- [Understand the implementation](#understand-the-implementation)
- [Further Exploration](#further-exploration)
- [Model Experience](#model-experience)
- [Known Limitations and Deferred Work](#known-limitations-and-deferred-work)
- [Dev Note](#dev-note)

-----

<a id="use-this-package"></a>
## Use this package

From the Steward directory on Windows, run the setup script, then restart the robot Ego Harness chat. It installs the tools and bundled `rgbd-video-gen` skill only into the robot `ego` profile; `scripts/setup-dsh-profiles.sh` also includes it for fresh setups.

```powershell
.\scripts\setup-rgbd-video-gen.ps1
```

The default endpoint is `http://115.120.57.9/nvs`; `http://linxlab-research.com/nvs` is also reachable. Override `steward-rgbd-video-gen-api` with a profile patch containing `endpoint`, `inputRoot` and `stateDir`. Defaults are `.` and `var/rgbd_video_gen` for the two directories, resolved within `var/dsh/robot-1/workspace` when launched by Steward. Inputs must stay inside that robot workspace. [Client settings](../dsh-rgbd-video-gen-client/README.md) own transfer limits and deadlines.

Example request: “Use rgbd_video_gen on the server demo waymo080, trajectory mode, 30 steps and 2 target views. Wait for success, read the result and download the RGB video, depth video and meta.json.” This runs server-owned sample data; no local images are needed.

For local input, provide a `rgbd_videogen_scene_v1` JSON file with inline images or a ZIP containing `scene.json` and relative image files. The scene needs 2-64 ordered frames, pixel intrinsics K and right-handed OpenCV camera-to-world transforms in metres. First num_source_views frames supply images; later frames define target poses and may use image=null. Images must match camera dimensions. Set `num_source_views` explicitly for fewer than 6 evidence images (range 1-8); total frames must be at least `num_source_views+1`. For three camera images and one target pose, submit four ordered reference-camera frames with `num_source_views=3` and `max_target_views=1`. The model was trained on 6 evidence views, so other counts may reduce quality. The ZIP tool uses server defaults; use JSON when selecting a smaller evidence count.

Prepare current evidence with `ego_capture_cameras`, then call `rgbd_video_gen_prepare_scene` with explicit `targets`. It defaults to only middle-camera RGB; additional sources require an explicit `source_cameras` selection and matching intrinsics. The existing `wide_camera` has different intrinsics and cannot be combined by this helper. A gripper-forward target is `{frame:"gripper",position:[0,0,0],look_at:[1,0,0],up:[0,0,1]}`; other positions, directions and multiple targets are supported. Targets may also supply an exact `camera_to_frame` matrix, in reference or gripper coordinates. Submit the returned scene path with its exact `num_source_views` and `max_target_views`: the default one-source/one-target case has 2 frames and `num_source_views=1`. Preparation reuses middle-camera intrinsics and reads only selected RGB images. Download successful artifacts and retain the local manifest for downstream readers; no UI visualization or general file tools are needed.

The [Python demo](../../../tools/rgbd_video_gen_demo.py) accepts `--url`, `--demo`, `--demo-dir`, `--wait` and `--request-key`. Its local manifest mode reads `manifest_train.jsonl`; `--save-scene` exports JSON for the Harness without uploading. Use Steward Python dependencies for Pillow. It prints the request key before submission; retain it and identical parameters when retrying an uncertain POST.

Results include RGB/depth videos, frames.zip, depth.npz, comparison images and metadata. Target camera indices use the actual source count, not a fixed offset of six. Generated views are predictions. BEV demos have no target ground truth. Server preprocessing produces 378×672 images; inspect result metadata for the returned camera information.

For evidence throughout motion, start `ego_rgb_recording_start` before navigation and stop it afterwards. Pass the successful recording_id and explicit targets to `rgbd_video_gen_prepare_recording`; do not replace the route with a final snapshot. All middle-camera RGB images, timestamps and measured poses remain in the returned recording_manifest_path. The service declares at most 8 source views per job: recordings of at most eight frames use all frames; longer recordings default to eight uniformly spaced frames including both endpoints. Explicit sample counts or chronological source_frame_indices can override this selection. The result reports recorded_frames, selected indices and all_frames_used. Reference targets use the first recorded middle camera; gripper targets use the current gripper related through measured odometry. Submission must use the returned source/target counts; this preparation requires the injected robot Ego provider.

-----

<a id="understand-the-implementation"></a>
## Understand the implementation

<details>
<summary>Implementation details</summary>

[cordis.patch.yml](cordis.patch.yml) mounts the HTTP provider, tool consumer and [robot skill](skills/rgbd-video-gen/SKILL.md). World profiles do not install this bundle. Server paths remain `/v1/nvs/...`; the service input format remains `rgbd_videogen_scene_v1`. Package names use hyphens and tool names use the `rgbd_video_gen_` prefix.

The recorded-session test uses a private workspace and an HTTP fixture on an allocated loopback port for both RGBD and Ego APIs. It checks the composed Ego endpoint before starting the model and rejects robot observations, motion and operator alerts inside the fixture. Recording and replay therefore do not use a running robot service.

</details>

-----

<a id="further-exploration"></a>
## Further Exploration

[Steward](../../../README.md) · [Bundle](../dsh-bundle-rgbd-video-gen/README.md) · [Service](../dsh-rgbd-video-gen/README.md) · [HTTP provider](../dsh-rgbd-video-gen-client/README.md) · [Tools](../dsh-tool-rgbd-video-gen/README.md)

-----

<a id="model-experience"></a>
## Model Experience

Indirect: [tools](../dsh-tool-rgbd-video-gen/README.md) own model instructions and logged results.

-----

<a id="known-limitations-and-deferred-work"></a>
## Known Limitations and Deferred Work

- Robot inputs require prepared observations and estimated camera poses; automatic capture and pose estimation are not provided. Simulator ground truth from `world_capture` is not robot evidence. This layer calls an existing remote service; model weights, server startup and camera-pose estimation are outside its scope.

-----

<a id="dev-note"></a>
## Dev Note

None.
