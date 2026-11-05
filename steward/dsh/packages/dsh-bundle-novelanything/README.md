---
description: "Add remote NovelAnything novel-view synthesis from RGB-only inputs to the robot Ego Harness."
kind: "package-bundle"
---

# @steward/dsh-bundle-novelanything

English | [中文](README.zh.md)

## Summary

Add remote NovelAnything novel-view synthesis from RGB-only inputs to the robot Ego Harness.

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

From the Steward directory on Windows, run the setup script and restart the robot chat. The script installs this bundle only into robot-1's `ego` profile; fresh Unix setups include it through `setup-dsh-profiles.sh`. World profiles do not receive it.

```powershell
.\scripts\setup-novelanything.ps1
```

Example robot request: “Use NovelAnything with the supplied scene.zip, dataset scannetpp, context [0,1], target [2], metrics false. Wait for success and download all artifacts. Return the local manifest path.” Supply an existing compliant posed RGB-only ZIP in the robot workspace. Use only the middle RGB `head_camera` without any input depth; exclude `wide_camera` and other physical cameras. Multiple contexts are different timestamps/poses of that same camera, so a single synchronized capture is insufficient. Omit all depth files and depth_path fields. API 1.3.0 supports these inputs and still predicts RGB and depth. The [skill](skills/novelanything/SKILL.md) owns calibration, units and scene-format instructions.

For evidence captured during navigation, the robot agent composes independent tools: start `ego_rgb_recording_start` at the requested FPS before motion, use the existing predict/receipt/run navigation tools with `wait=false`, then finish with `ego_rgb_recording_stop`. The recorder uses simulation time and records only middle RGB, synchronized odometry and encoders. `novelanything_prepare_gripper_view` turns a successful recording into an RGB-only local ZIP and constructs a current gripper-link target looking along +X with image-up +Z and middle-camera intrinsics. It returns the archive and context/target indices for the existing upload/submit tools; preparation performs no motion or remote inference. Specify `dataset=scannetpp` for these metric robot scenes and label downloaded results `gripper_view_prediction`. An explicitly requested model is never substituted with rgbd_video_gen.

The ZIP retains all recorded RGB evidence and timestamps under `evidence/`, even when only up to 16 evenly spaced contexts are selected. Pose estimates use a shared robot odometry frame; simulator absolute poses are not required. Recording failures, queue overload, reset and timeout are visible through status. A recording that reaches its frame bound before navigation finishes does not cover the whole route. Live handles are bounded and do not survive host restart; prepare the local ZIP before restarting the host. The provider explicitly injects the robot `ego` service for preparation; rebuilding its plugin is sufficient for a fresh robot chat process and does not require discarding an existing recording.

The default endpoint is `http://115.120.57.9/novelanything`. Override `steward-novelanything-api` in the profile patch for another deployment. [Provider settings](../dsh-novelanything-client/README.md) configure local roots, deadlines, transfer bounds and optional token environment variable.

Inputs are retained under `var/dsh/robot-1/workspace/inputs/novelanything/`. Results preserve artifact subdirectories under `outputs/novelanything/<job_id>/` in that workspace. `manifest.json` records absolute local paths, local SHA256, input association and server result metadata. `job.json` retains the successful remote reply. Remote inference success and local download success are distinct.

-----

<a id="understand-the-implementation"></a>
## Understand the implementation

<details>
<summary>Implementation details</summary>

[cordis.patch.yml](cordis.patch.yml) inserts the provider, tool consumer and filesystem skill provider with only this bundled skill directory enabled. Tools use ordinary logged JSON results. The recorded Ego profile test exercises upload, submission and download against an isolated HTTP server and replays without model credentials.

</details>

-----

<a id="further-exploration"></a>
## Further Exploration

[Steward](../../../README.md) · [Bundle](../dsh-bundle-novelanything/README.md) · [Service](../dsh-novelanything/README.md) · [HTTP provider](../dsh-novelanything-client/README.md) · [Tools](../dsh-tool-novelanything/README.md)

-----

<a id="model-experience"></a>
## Model Experience

Indirect: [tools](../dsh-tool-novelanything/README.md) provide model-facing operations; the bundled skill explains robot evidence restrictions.

-----

<a id="known-limitations-and-deferred-work"></a>
## Known Limitations and Deferred Work

- Manual inputs require calibrated RGB and supplied target poses. Recorded gripper scenes use measured robot odometry and fixed rig calibration; they do not estimate camera motion from images or deploy the remote server. Predictions do not update Ego-State or authorize motion.
- No remote cancellation or idempotent POST is declared by the API. UNKNOWN_OUTCOME blocks resubmission and requires operator reconciliation.
- Optional SAM3 segmentation is not exposed by these synthesis tools. NovelAnything uses CC BY-NC-SA 4.0 for non-commercial use; see the [license](skills/novelanything/LICENSE.md).

-----

<a id="dev-note"></a>
## Dev Note

None.
