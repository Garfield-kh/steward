---
description: "HTTP provider for remote NovelAnything jobs with confined local storage."
kind: "package-reference"
---

# @steward/dsh-novelanything-client

English | [中文](README.zh.md)

## Summary

HTTP provider for remote NovelAnything jobs with confined local storage.

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

Install the [bundle](../dsh-bundle-novelanything/README.md); it mounts this default service provider as `steward-novelanything-api`. Override its configuration in a profile patch. No model weights are installed locally. The provider requires the injected robot `ego` service for recording preparation.

| Setting | Default | Meaning |
|---|---|---|
| `endpoint` | `required` | HTTP(S) service root, including proxy prefix |
| `inputRoot` | `required` | Robot workspace; all local paths stay beneath it |
| `stateDir` | `var/novelanything` | Endpoint-specific POST receipts |
| `inputsDir` | `inputs/novelanything` | Content-addressed submitted ZIP archives |
| `outputsDir` | `outputs/novelanything` | Job outputs and local manifests |
| `tokenEnv` | `NOVELANYTHING_API_KEY` | Optional Bearer token environment variable |
| `requestTimeoutMs` | `15000` | Metadata request deadline |
| `transferTimeoutMs` | `300000` | Upload/download request deadline |
| `pollIntervalMs` | `2000` | Polling interval |
| `maxWaitSeconds` | `20` | Per-call polling bound, at most 60 |
| `maxMetadataBytes` | `2097152` | Metadata and log response byte limit |
| `maxUploadBytes` | `268435456` | Compressed ZIP byte limit |
| `maxArtifactBytes` | `2147483648` | Per-artifact download byte limit |

-----

<a id="understand-the-implementation"></a>
## Understand the implementation

<details>
<summary>Implementation details</summary>

POST reservations are published exclusively before sending. A saved accepted reply can be reused with identical inputs; an unresolved reservation blocks another POST, including after restart. GET failures propagate to the caller. Disposal aborts local requests and awaits settlement.

Artifact URLs must exactly match the configured job artifact route. Downloads stream to private temporary files, enforce byte limits, check server size/hash when present, then rename complete files and update the manifest after each file. Prior complete files survive later transfer failures.

</details>

-----

<a id="further-exploration"></a>
## Further Exploration

[Steward](../../../README.md) · [Bundle](../dsh-bundle-novelanything/README.md) · [Service](../dsh-novelanything/README.md) · [HTTP provider](../dsh-novelanything-client/README.md) · [Tools](../dsh-tool-novelanything/README.md)

-----

<a id="model-experience"></a>
## Model Experience

Indirect: [tools](../dsh-tool-novelanything/README.md) expose readiness, accepted jobs, terminal failures and local download metadata.

-----

<a id="known-limitations-and-deferred-work"></a>
## Known Limitations and Deferred Work

- The server validates ZIP scene contents. This client bounds and archives uploads; the robot Ego API prepares recorded scenes using measured odometry, fixed extrinsics and encoder-derived gripper poses.
- Remote artifact checksums are optional. A computed local SHA256 alone does not establish server checksum verification.
- Concurrent downloads to the same job directory are unsupported. Unknown POST outcomes require operator reconciliation; no automatic retry or cancellation is provided.

-----

<a id="dev-note"></a>
## Dev Note

None.
