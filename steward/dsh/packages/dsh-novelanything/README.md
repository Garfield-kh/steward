---
description: "Typed service definitions for remote NovelAnything synthesis."
kind: "package-library"
---

# @steward/dsh-novelanything

English | [中文](README.zh.md)

## Summary

Typed service definitions for remote NovelAnything synthesis.

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

Import `NovelAnythingService`, `Submission`, `remoteId` and `NovelAnythingError` from this package. The HTTP provider implements the service; the tool package consumes it. See [src/index.ts](src/index.ts) for cancellation, recovery and download guarantees.

-----

<a id="understand-the-implementation"></a>
## Understand the implementation

<details>
<summary>Implementation details</summary>

The `novelAnything` Cordis service carries operations without binding the model to an HTTP implementation. Remote identities are branded and validated.

</details>

-----

<a id="further-exploration"></a>
## Further Exploration

[Steward](../../../README.md) · [Bundle](../dsh-bundle-novelanything/README.md) · [Service](../dsh-novelanything/README.md) · [HTTP provider](../dsh-novelanything-client/README.md) · [Tools](../dsh-tool-novelanything/README.md)

-----

<a id="model-experience"></a>
## Model Experience

Indirect: the tool consumer owns model-visible descriptions and JSON results.

-----

<a id="known-limitations-and-deferred-work"></a>
## Known Limitations and Deferred Work

- Local cancellation never implies remote cancellation. An uncertain submission cannot be safely retried under the published API.

-----

<a id="dev-note"></a>
## Dev Note

None.
