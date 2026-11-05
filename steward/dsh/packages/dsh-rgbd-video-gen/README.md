---
description: "Consume rgbd_video_gen generation through a typed service, with explicit submission identities, remote state and verified artifact receipts."
kind: "package-library"
---

# @steward/dsh-rgbd-video-gen

English | [中文](README.zh.md)

## Summary

Consume rgbd_video_gen generation through a typed service, with explicit submission identities, remote state and verified artifact receipts.

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

Import `RgbdVideoGenService`, `jobId` and request types from `@steward/dsh-rgbd-video-gen`; providers implement `ctx.rgbdVideoGen`. Applications install the [bundle](../dsh-bundle-rgbd-video-gen/README.md).

`prepareScene` creates a local scene from a saved robot capture, explicit target views and selected evidence cameras. Evidence defaults to middle `head_camera` RGB only. Targets accept a rigid OpenCV camera-to-frame matrix or position/look-at/up vectors, in either the captured reference or synchronized gripper frame. Reference-frame targets do not require a gripper pose. Preparation returns the scene path and exact evidence/target counts without submission. Before JSON or demo submission, `resolveOptions` fills generation defaults from the server; prepared scenes must explicitly pass their returned counts. Reuse the same request key and resolved parameters after a lost response. `RgbdVideoGenError` exposes code, retryability, HTTP status and details. See [source](src/index.ts) for cancellation and result semantics.

-----

<a id="understand-the-implementation"></a>
## Understand the implementation

<details>
<summary>Implementation details</summary>

Service definition, HTTP provider and tool consumer are independently injectable. Branded job ids separate remote identities from arbitrary strings.

</details>

-----

<a id="further-exploration"></a>
## Further Exploration

[Steward](../../../README.md) · [Bundle](../dsh-bundle-rgbd-video-gen/README.md) · [Service](../dsh-rgbd-video-gen/README.md) · [HTTP provider](../dsh-rgbd-video-gen-client/README.md) · [Tools](../dsh-tool-rgbd-video-gen/README.md)

-----

<a id="model-experience"></a>
## Model Experience

Indirect: the [tool package](../dsh-tool-rgbd-video-gen/README.md) logs JSON operation results.

-----

<a id="known-limitations-and-deferred-work"></a>
## Known Limitations and Deferred Work

- Local cancellation stops waiting or transfer; it does not cancel accepted server work.

-----

<a id="dev-note"></a>
## Dev Note

None.
