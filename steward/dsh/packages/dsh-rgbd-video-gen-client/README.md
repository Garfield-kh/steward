---
description: "Configure remote rgbd_video_gen submissions, recover uncertain requests and retrieve integrity-checked RGB/depth outputs."
kind: "package-reference"
---

# @steward/dsh-rgbd-video-gen-client

English | [中文](README.zh.md)

## Summary

Configure remote rgbd_video_gen submissions, recover uncertain requests and retrieve integrity-checked RGB/depth outputs.

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

Install through the [bundle](../dsh-bundle-rgbd-video-gen/README.md). The `steward-rgbd-video-gen-api` profile row owns the settings below; `inputRoot` and `stateDir` resolve from the DSH working directory; `inputsDir` and `outputsDir` resolve beneath `inputRoot`.

| Field | Default | Meaning |
|---|---|---|
| `endpoint` | Required | HTTP(S) base URL, retaining `/nvs`; no credentials, query or fragment |
| `inputRoot` | Required | Existing root for local JSON/ZIP input |
| `stateDir` | Required | Endpoint-isolated request identities and recovery records |
| `inputsDir` | inputs/rgbd_video_gen | Submitted JSON, parameters and ZIP archives; relative to inputRoot |
| `outputsDir` | outputs/rgbd_video_gen | Results by remote task ID; relative to inputRoot |
| `requestTimeoutMs` | 15000 | Each metadata request deadline |
| `uploadTimeoutMs` | 1800000 | Each submission/upload deadline |
| `downloadTimeoutMs` | 300000 | Each artifact transfer deadline |
| `pollIntervalMs` | 2000 | Delay between status queries |
| `maxWaitSeconds` | 20 | Maximum requested poll duration, up to 60; one HTTP call can extend the wait |
| `retryCount` | 2 | Additional retryable HTTP attempts |
| `retryDelayMs` | 1000 | Initial exponential retry delay |
| `maxRetryDelayMs` | 30000 | Refuse retries exceeding this delay |
| `maxMetadataBytes` | 2097152 | Bound on JSON response bodies |
| `maxUploadBytes` | 536870912 | ZIP bytes, at most 512 MiB |
| `maxSceneBytes` | 67108864 | JSON submission bytes, at most 64 MiB |
| `maxArtifactBytes` | 2147483648 | Per-artifact download admission limit |
| `preparedSceneDir` | prepared | Single directory name beneath inputsDir for prepared scene files |
| `maxEvidenceImageBytes` | 16777216 | Per-JPEG preparation limit, at most 16 MiB |

Submissions persist an immutable payload digest and server idempotency key before POST. Reusing a key with changed input fails locally. The store retains accepted responses so restarts can recover jobs; keep `stateDir` to preserve this protection. Cancellation is limited to locally submitted or recovered jobs and has its own stable POST key.

`TRANSPORT` can mean an unknown POST outcome: retry the same operation key, never invent a replacement job id. Downloads remain `.part` until byte count and SHA256 match; a failed transfer does not invalidate remote generation or already verified files. Retry download with the same filenames.

Submission inputs are retained under `inputsDir`: JSON submissions keep `request.json` and `scene.json` with inline images; uploads keep `archive.zip` and their request metadata. Server demos retain only the selector and generation parameters. Each `outputsDir/<task_id>/` contains the full server `result.json` and a local `manifest.json` listing the input archive and verified downloaded files. RGBD artifacts are stored directly in that directory; `targets` maps target cameras to RGB paths, depth paths and NPZ keys. Repeated selections accumulate in the manifest; a later result query rechecks existing files. Missing local input history is represented by null, not reconstructed from server predictions. Downloads reuse verified complete files. Use the tool-returned `local_manifest_path` for downstream processing. Configure separate output directories for different servers whose task IDs can collide.

The provider requires the robot ego service. Recording preparation exports every RGB sample and measured pose through `/ego/rgb/record/export`, saves complete local evidence, and constructs a scene from explicit temporal indices and target descriptions. It performs no motion or remote inference. Source counts must match the prepared recording metadata at submission. The default selection uses all recorded frames up to eight, otherwise eight uniformly spaced frames including the first and last. Explicit counts or indices override this policy. Failed or cancelled preparation returns no completed scene; already saved local evidence may remain.

-----

Loose prediction images use `source_reconstruction_00.jpg` for input-view reconstruction and `target_prediction_00.jpg` for the first requested target; target numbering starts at zero independently of source count. Remote `filename` remains the download selector; `local_filename` and `path` identify the local file. Re-downloading verified legacy `pred_*.jpg` files renames them without another transfer. Server `result.json`, ZIP entries and depth keys keep their original names; use manifest `targets` for the target RGB/depth mapping.

<a id="understand-the-implementation"></a>
## Understand the implementation

<details>
<summary>Implementation details</summary>

The provider retains the reverse-proxy prefix, rejects foreign artifact URLs and follows no HTTP redirects. Plugin disposal aborts local I/O and waits for owned operations to settle. Metadata is bounded; binary downloads stream to disk. It does not forward DeepSeek credentials.

</details>

-----

<a id="further-exploration"></a>
## Further Exploration

[Steward](../../../README.md) · [Bundle](../dsh-bundle-rgbd-video-gen/README.md) · [Service](../dsh-rgbd-video-gen/README.md) · [HTTP provider](../dsh-rgbd-video-gen-client/README.md) · [Tools](../dsh-tool-rgbd-video-gen/README.md)

-----

<a id="model-experience"></a>
## Model Experience

Indirect: tools receive structured metadata, not image bytes. The full result manifest remains in the result directory.

-----

<a id="known-limitations-and-deferred-work"></a>
## Known Limitations and Deferred Work

- Geometry, image dimensions and ZIP contents are validated by the remote service. JSON submissions require inline images; remote absolute image paths are not supported by this client.

-----

<a id="dev-note"></a>
## Dev Note

None.
