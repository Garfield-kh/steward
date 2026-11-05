---
kind: upgrade-guide
description: "Steward rgbd_video_gen downloads distinguish source reconstruction images from target predictions in local filenames."
---

# RGBD local prediction filenames

English | [中文](guide.zh.md)

## Change

Steward rgbd_video_gen loose RGB downloads replace ambiguous `pred_<frame>.jpg` names with `source_reconstruction_<source>.jpg` and `target_prediction_<target>.jpg`. Both indices start at zero within their own group. For eight sources and one target, remote `pred_08.jpg` is saved as `target_prediction_00.jpg`.

Remote artifact `filename` remains the download selector. Result artifacts, downloaded file records and local manifest files expose `local_filename`; `path` and `targets[].rgb_path` point to the renamed local files. Raw server `result.json`, ZIP entries and NPZ depth keys retain their original identifiers and bytes.

## Migration

1. Replace local filename construction with returned `path` or manifest `targets[].rgb_path`. Continue passing remote `artifacts[].filename` to `rgbd_video_gen_download`.
2. Rebuild the Steward plugins. Download an existing successful job again to rename verified legacy loose predictions without retransferring their bytes. Missing or invalid local files still require normal verified downloads. Historical files are not renamed merely by querying a result.
3. Confirm that eight sources and one target produce local `source_reconstruction_00.jpg` through `source_reconstruction_07.jpg` and `target_prediction_00.jpg`. The target depth key remains `v08`; use manifest `targets` rather than deriving it from the new target filename.
