---
description: "Model-facing robot tools for hosted NovelAnything RGB/depth prediction."
kind: "package-reference"
---

# @steward/dsh-tool-novelanything

English | [中文](README.zh.md)

## Summary

Model-facing robot tools for hosted NovelAnything RGB/depth prediction.

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

Install the [bundle](../dsh-bundle-novelanything/README.md) into the robot profile. This plugin injects `tools` and `novelAnything`; `novelAnythingTools(service)` also provides the same definitions to tests. Tool outputs render as logged JSON text.

-----

<a id="understand-the-implementation"></a>
## Understand the implementation

<details>
<summary>Implementation details</summary>

[src/index.ts](src/index.ts) registers health, samples, upload, submit, status, result, log and download operations through `ctx.effect()`. Omitted metrics resolve to true; omitted dataset resolves to re10k; omitted wait_seconds resolves to zero.

</details>

-----

<a id="further-exploration"></a>
## Further Exploration

[Steward](../../../README.md) · [Bundle](../dsh-bundle-novelanything/README.md) · [Service](../dsh-novelanything/README.md) · [HTTP provider](../dsh-novelanything-client/README.md) · [Tools](../dsh-tool-novelanything/README.md)

-----

<a id="model-experience"></a>
## Model Experience

Direct: tools describe RGB-only evidence, frame selection and prediction limits, then return JSON through the standard tool pipeline. The [recorded-session test](../../test/novelanything-snapshot.test.mjs) checks the assembled robot experience.

-----

<a id="known-limitations-and-deferred-work"></a>
## Known Limitations and Deferred Work

- No filesystem or shell capability is added. Robot RGB evidence supports only the middle `head_camera` without input depth; `wide_camera` and other physical cameras are excluded. Missing posed RGB inputs must be supplied by the caller; a server sample is not a robot observation.

-----

<a id="dev-note"></a>
## Dev Note

None.
