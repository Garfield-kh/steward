---
kind: upgrade-guide
description: "Steward model downloads use workspace output directories and retain submitted inputs separately from recovery state."
---

# Steward model input and output directories

English | [中文](guide.zh.md)

## Change

The optional Steward rgbd_video_gen and NovelAnything clients keep model inputs under `inputs/<model>/` and results under `outputs/<model>/<job_id>/` in the robot workspace. Tools return `local_manifest_path` for a local manifest containing downloaded paths and references to the submitted input. Recovery state remains separate from model artifacts. These integrations do not install SLARM.

Submissions archive JSON scenes and parameters or uploaded ZIPs under the corresponding `inputs/<model>/` directory. Gripper preparation defaults to `inputs/rgbd_video_gen/prepared/`. Server demos retain their selector, not server-owned evidence images. Historical submissions without local input records have a null input reference. Existing legacy files are not deleted or automatically moved.

## Migration

1. Install the desired bundle in the robot Ego profile, rebuild the Steward plugins and restart the robot Harness. Set `inputsDir` and `outputsDir` on the model client to override the defaults; relative paths resolve beneath `inputRoot`. Different endpoints with colliding task IDs need separate output directories.
2. Replace downstream path concatenation with the tool-returned `local_manifest_path`. Read `files` for verified downloads; read RGBD `targets` for target camera, RGB and depth array associations. A null path means that file has not been downloaded.
3. Query and download an existing successful task to publish it in the new directory, or copy historical files and verify them against the server result before using them. Retain existing files until consumers use the new manifest. A new submission preserves its inputs before POST.
4. Confirm that a downloaded task has `result.json`, `manifest.json` and the requested files under `outputs/<model>/`. The existing `var/` Git ignore rule covers caller workspaces; local runtime files and credentials must stay out of commits.
