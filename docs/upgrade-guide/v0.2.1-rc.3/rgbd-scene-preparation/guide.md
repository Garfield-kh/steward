---
kind: upgrade-guide
description: "Optional rgbd_video_gen scene preparation uses explicit targets and primary-camera RGB evidence by default."
---

# General RGBD scene preparation

English | [中文](guide.zh.md)

## Change

The optional `rgbd_video_gen_prepare_scene` tool and `prepareScene(captureManifest, options, signal)` service method prepare explicit target viewpoints from observed RGB. Preparation defaults to one `head_camera` image. No gripper target is chosen implicitly; existing head and wide perception cameras retain their layout and calibration.

## Migration

1. Supply `targets` with one to 24 entries. Each entry names `frame` (`reference` for the captured head OpenCV frame, or `gripper` for synchronized forward/left/up axes) and either a rigid OpenCV `camera_to_frame` matrix or all three `position`, `look_at`, `up` vectors. Positions are metres. Do not mix both forms.
2. For the former gripper-forward viewpoint, pass `targets:[{frame:"gripper",position:[0,0,0],look_at:[1,0,0],up:[0,0,1]}]`. Gripper targets require the same-timestamp embedded pose; reference targets do not require a gripper pose.
3. Omit `source_cameras` for `head_camera` RGB only. Only selected JPEGs are loaded, and selected sources must share the reference camera's resolution and intrinsics. The existing `wide_camera` has different intrinsics and must not be treated as interchangeable head-camera evidence.
4. Submit the returned `scene_path` with the exact returned `num_source_views` and `max_target_views`. One source and one target produce 2 frames and require `num_source_views=1`; the remote submission default remains 6. Scene files use `scene-<hash>.json` under the prepared input directory. Verify the saved scene metadata lists only `head_camera` for the default case.
5. For motion evidence, start and stop an RGB recording around independently controlled navigation, then call `rgbd_video_gen_prepare_recording` with the successful handle and explicit targets. The provider requires the robot Ego service and Python `/ego/rgb/record/export` endpoint; rebuild plugins and restart Steward/robot chat to load both changes. Save needed live recording evidence before restarting. Recordings of at most eight frames use all frames by default; longer recordings use eight uniformly spaced frames including both endpoints. Explicit counts or batch indices override the default; verify the prepared source indices and all_frames_used instead of treating local retention as model input.
