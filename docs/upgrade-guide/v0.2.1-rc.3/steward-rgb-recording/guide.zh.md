---
kind: upgrade-guide
description: "可选模型准备使用原头部相机的纯 RGB 证据，不改变机器人相机布局。"
---

# Steward 纯 RGB 录制与 NovelAnything 输入

[English](guide.md) | 中文

## 变更

可选模型准备仅使用 `head_camera` 的 RGB、测量位姿和内参；NovelAnything ZIP 不包含深度文件或 `depth_path`。模型预测目标 RGB 和深度。现有头部和广角 RGB-D 相机保留各自的安装位姿、内参和感知职责，不新增或移除相机。

## 迁移

1. 在仓库根目录运行 `pnpm --dir steward/dsh run build` 编译 Steward 插件，然后重启 Steward 与机器人聊天以加载 Python 和插件变更。重启会丢失实时录制句柄，请先保留需要的本地归档。
2. 使用 `ego_rgb_recording_start/status/stop` 按仿真时钟记录 RGB 证据。机器人路由为 `/sensors/rgb/record/*` 和 `/ego/rgb/record/*`；准备路由为 `/ego/rgb/prepare-novelanything`。录制句柄以 `rgb_` 开头。现有 `/sensors/rgbd` 接口仍是感知流。
3. 对成功的 RGB 录制调用 `novelanything_prepare_gripper_view`，上传其返回的 ZIP。确认 `metadata.input_modalities` 等于 `["rgb"]`，且归档不包含 `depth_path` 或深度图。远程部署须支持纯 RGB 输入（API 1.3.0 或更新）；不要添加深度来绕过旧服务限制。
