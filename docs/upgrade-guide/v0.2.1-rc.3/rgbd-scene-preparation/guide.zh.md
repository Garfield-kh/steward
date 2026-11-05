---
kind: upgrade-guide
description: "可选 rgbd_video_gen 场景准备使用显式目标视角，默认取原头部相机 RGB 证据。"
---

# 通用 RGBD 场景准备

[English](guide.md) | 中文

## 变更

可选的 `rgbd_video_gen_prepare_scene` 工具与 `prepareScene(captureManifest, options, signal)` 服务方法从观测 RGB 准备显式目标视角。默认输入是一张 `head_camera` 图像，不隐式选择手爪目标；现有头部与广角感知相机保留布局和标定。

## 迁移

1. 提供含 1–24 项的 `targets`。每项指定 `frame`（`reference` 为采集头部相机的 OpenCV 系，`gripper` 为同步的前/左/上轴），并提供刚性 OpenCV `camera_to_frame` 矩阵，或完整的 `position`、`look_at`、`up` 三个向量。位置单位是米，不混用两种形式。
2. 原手爪前视目标传入 `targets:[{frame:"gripper",position:[0,0,0],look_at:[1,0,0],up:[0,0,1]}]`。手爪目标要求同一时间戳的内嵌位姿；参考帧目标不要求手爪位姿。
3. 省略 `source_cameras` 时只取 `head_camera` RGB。只读取被选中的 JPEG，选中源必须与参考相机具有相同分辨率和内参。现有 `wide_camera` 的内参不同，不能视作可互换的头部相机证据。
4. 提交返回的 `scene_path` 时，使用工具返回的准确 `num_source_views` 和 `max_target_views`。单证据、单目标产生2帧，必须设置 `num_source_views=1`；远端提交默认值仍为6。场景文件以 `scene-<hash>.json` 命名，位于准备输入目录中。确认默认场景元数据只列出 `head_camera`。
5. 运动证据须围绕独立控制的导航开始和结束 RGB 录制，再将成功句柄及显式目标传给 `rgbd_video_gen_prepare_recording`。Provider 要求机器人 Ego 服务和 Python `/ego/rgb/record/export` 接口，须编译插件并重启 Steward/机器人聊天以加载两端变更。重启前保存所需的实时录制证据。不超过8帧的录制默认全部使用，超过时默认均匀选取包含首尾的8帧；显式数量或批次索引可覆盖默认值；检查准备结果的源帧索引和 all_frames_used，不能把本地保留数量当成模型输入数量。
