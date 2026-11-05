---
description: "Ask the robot Ego Harness to generate novel RGB/depth views, inspect progress and download verified outputs."
kind: "package-reference"
---

# @steward/dsh-tool-rgbd-video-gen

English | [中文](README.zh.md)

## Summary

Ask the robot Ego Harness to generate novel RGB/depth views, inspect progress and download verified outputs.

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

Install the [bundle](../dsh-bundle-rgbd-video-gen/README.md), then use `rgbd_video_gen_capabilities` before submission. `rgbd_video_gen_submit`, `rgbd_video_gen_upload` and `rgbd_video_gen_demo` select JSON, ZIP and server-demo input respectively.

Use `rgbd_video_gen_prepare_scene` with the exact capture manifest and explicit target descriptions. It defaults to middle-camera RGB only; `source_cameras` explicitly selects additional evidence. Targets use reference or gripper coordinates and either a rigid matrix or position/look-at/up. The tool saves an inline-image scene and returns exact evidence/target counts for submission without general file tools or delegation.

Follow `rgbd_video_gen_status` until a terminal state. Only `succeeded` permits `rgbd_video_gen_result`; choose exact returned filenames for `rgbd_video_gen_download`. `rgbd_video_gen_cancel` explicitly requests cancellation. Polling defaults to one query; stopping a chat does not cancel remote work. Exact parameters are in [src/index.ts](src/index.ts).

JSON scenes contain 2-64 frames and at least one target after the 1-8 evidence frames. Set `num_source_views` explicitly for fewer than 6 evidence images; three sources plus one target use four frames with `num_source_views=3`. Generation fields omitted from JSON/demo calls resolve from server capabilities. ZIP submission uses server defaults. BEV demo calls require explicit view count and altitude scale. Request keys identify logical operations and must be reused unchanged after disconnects.

For an entire-motion recording, use `rgbd_video_gen_prepare_recording` with a successful ego_rgb_recording handle and explicit targets. Every middle-camera RGB frame is saved locally; the scene uses either all frames up to eight or, by default for longer recordings, eight uniformly spaced frames including both endpoints. Explicit counts or chronological batch indices override this policy. The tool reports total and selected counts separately and never silently falls back to a final snapshot. The existing prepare_scene tool represents a single timestamp.

-----

<a id="understand-the-implementation"></a>
## Understand the implementation

<details>
<summary>Implementation details</summary>

The plugin registers tools with `ctx.effect` and delegates to `ctx.rgbdVideoGen`. The ordinary DSH pipeline logs and renders JSON results; downloads stay on disk.

</details>

-----

<a id="further-exploration"></a>
## Further Exploration

[Steward](../../../README.md) · [Bundle](../dsh-bundle-rgbd-video-gen/README.md) · [Service](../dsh-rgbd-video-gen/README.md) · [HTTP provider](../dsh-rgbd-video-gen-client/README.md) · [Tools](../dsh-tool-rgbd-video-gen/README.md)

-----

<a id="model-experience"></a>
## Model Experience

Descriptions distinguish acceptance, generation success and verified local files. RGB/depth are predictions, not observations. Target PSNR requires target ground truth. Target camera i corresponds to predicted frame num_source_views+i. Binary image content does not enter the tool transcript.

-----

<a id="known-limitations-and-deferred-work"></a>
## Known Limitations and Deferred Work

- No pose estimation, SLARM capture conversion, MuJoCo import or automatic robot action is provided.

-----

<a id="dev-note"></a>
## Dev Note

None.
