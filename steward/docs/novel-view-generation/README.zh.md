---
description: "基于观测图像生成新视角 RGBD：对话示例、工具流程与包职责。"
---

# 感知基模skill说明

[English](README.md) | 中文

## 概要

NovelAnything 和 rgbd_video_gen 根据观测图像及相机位姿，生成指定虚拟相机视角的 RGB 与预测深度，可用于观察原相机未直接覆盖的位置或方向。目标视角由虚拟相机的位置、朝向和内参描述，手爪视角是其中一个应用示例。现有 `head_camera` 与 `wide_camera` 保持原布局；以下示例使用头部相机的 RGB 时间序列。

rgbd_video_gen 的准备工具支持在参考相机或手爪坐标系中显式指定目标位置与朝向，也支持多个目标视角。NovelAnything 的当前便捷准备工具固定生成手爪视角；其他目标需提供含目标位姿的合规场景 ZIP。

## 目录

- [使用前提](#prerequisites)
- [可复制的使用示例](#copyable-examples)
- [虚拟相机与采样约定](#virtual-camera-and-sampling)
- [流程与包结构参考](#flow-and-package-reference)
- [结果检查与故障处理](#results-and-recovery)

<a id="prerequisites"></a>

## 使用前提

启动 Steward，加载任务所需场景，在网页面板选择“机器人 · robot-1”对话。机器人 `ego` profile 需要安装对应模型 bundle，插件与 Python 服务均需加载本次集成的版本；安装入口见 [NovelAnything](../../dsh/packages/dsh-bundle-novelanything/README.zh.md) 和 [rgbd_video_gen](../../dsh/packages/dsh-bundle-rgbd-video-gen/README.zh.md)。模型服务分别由 `novelanything_health`、`rgbd_video_gen_capabilities` 检查就绪状态。

示例中的桌子识别、方位判断与路径可达性由当前场景决定；信息不足或无法安全到达时，机器人应报告原因。

<a id="copyable-examples"></a>

## 可复制的使用示例

以下以手爪位置的新视角为例，可直接粘贴到机器人对话。其他视角可修改虚拟相机的位置和朝向，并明确所用参考坐标系。

```text
输入：机器人移动到桌子正南方向，面向桌子，距桌 0.5 m；移动过程中，用头部相机按 2 FPS 连续采样 RGB 图像作为输入。
虚拟相机：到位后，置于手爪 link 原点，沿手爪 +X 方向看，画面上方对应 +Z，使用右手坐标系；内参复用头部相机。
生成：使用 rgbd_video_gen 生成该虚拟视角下的 RGBD 图像。
```

<a id="virtual-camera-and-sampling"></a>

## 虚拟相机与采样约定

| 项目 | 约定与示例 |
|---|---|
| 真实输入 | 同一个 `head_camera` 在不同时刻采集的 RGB；`wide_camera` 继续服务原感知管线 |
| 采样时钟 | 仿真时间，2 FPS 即相邻样本相隔 0.5 秒；独立于智能体回复速度 |
| 位姿来源 | 同步的机器人里程计、关节编码器与本体运动学；不用 World 绝对真值补齐 |
| 虚拟相机位置 | 由目标视角指定；本例为准备输入时的当前手爪 link 原点 |
| 朝向 | 显式指定观察方向与画面上方；本例沿手爪 +X 看、+Z 向上，图像右方向为手爪 −Y |
| 相机坐标 | OpenCV：x 向右、y 向下、z 向前；本例分别对应手爪 −Y、−Z、+X，保持右手系 |
| 内参 | 复用头部相机的像素内参；远端模型可按其输出规格缩放或裁剪，结果以内含元数据为准 |
| 模型输入数量 | NovelAnything 为 2–16 张上下文图像；rgbd_video_gen 为 1–8 张源图像；保留的全部录制帧不一定全部参与推理 |
| 生成内容 | 指定虚拟视角的 RGB 与预测深度；生成不会增加实体相机或修改 Ego-State |

<a id="flow-and-package-reference"></a>

## 流程与包结构参考

两种模型共享头部相机录制能力，各自拥有输入准备、远端调用和下载工具。以下路径均相对 `steward/dsh/packages/`；四层职责分别为 bundle 安装配置、服务定义、client 执行和 tool 智能体调用。

| 文件夹 | 作用 | 主要内容 |
|---|---|---|
| [dsh-bundle-novelanything](../../dsh/packages/dsh-bundle-novelanything/README.zh.md) | 安装与配置入口 | 将 NovelAnything client、工具和 `skills/novelanything/SKILL.md` 装入机器人 Harness，配置服务地址及输入输出目录 |
| [dsh-novelanything](../../dsh/packages/dsh-novelanything/README.zh.md) | 服务接口定义 | 定义健康检查、上传、提交、查询、结果和下载接口，以及参数、结果和错误类型 |
| [dsh-novelanything-client](../../dsh/packages/dsh-novelanything-client/README.zh.md) | 执行准备与 HTTP 请求 | 调用 Ego 导出录制证据，准备场景 ZIP；连接服务、恢复本地请求记录、下载工件并保存清单 |
| [dsh-tool-novelanything](../../dsh/packages/dsh-tool-novelanything/README.zh.md) | 提供智能体工具 | 注册 `novelanything_prepare_gripper_view`、`novelanything_upload`、`novelanything_submit`、`novelanything_status` 等工具，将调用交给服务 |
| [dsh-bundle-rgbd-video-gen](../../dsh/packages/dsh-bundle-rgbd-video-gen/README.zh.md) | 安装与配置入口 | 将 rgbd_video_gen client、工具和 `skills/rgbd-video-gen/SKILL.md` 装入机器人 Harness，配置服务地址及输入输出目录 |
| [dsh-rgbd-video-gen](../../dsh/packages/dsh-rgbd-video-gen/README.zh.md) | 服务接口定义 | 定义能力查询、场景准备、生成、查询、结果、下载和取消接口，以及参数、结果和错误类型 |
| [dsh-rgbd-video-gen-client](../../dsh/packages/dsh-rgbd-video-gen-client/README.zh.md) | 执行准备与 HTTP 请求 | 从录制证据准备场景 JSON，调用远端生成服务，处理任务恢复、工件下载及大小和 SHA256 校验 |
| [dsh-tool-rgbd-video-gen](../../dsh/packages/dsh-tool-rgbd-video-gen/README.zh.md) | 提供智能体工具 | 注册 `rgbd_video_gen_prepare_recording`、`rgbd_video_gen_submit`、`rgbd_video_gen_status` 等工具，将调用交给服务 |

`tool` 提供可执行操作，bundle 内的 `skill` 说明使用顺序、输入限制、选帧与结果解释。两者共同支持从自然语言到模型结果的完整任务。

| 阶段 | 共享能力或 NovelAnything | rgbd_video_gen |
|---|---|---|
| 录制并移动 | `ego_rgb_recording_start` → 原有 `ego_predict`／`skill_run`／`skill_status` → `ego_rgb_recording_stop`／`ego_rgb_recording_status` | 同左 |
| 准备输入 | `novelanything_prepare_gripper_view` → `archive_path`、`context`、`target` | `rgbd_video_gen_prepare_recording` → `scene_path`、源帧索引和准确提交数量 |
| 提交 | `novelanything_upload` → `novelanything_submit` | `rgbd_video_gen_submit` |
| 跟踪与下载 | `novelanything_status` → `novelanything_result` → `novelanything_download` | `rgbd_video_gen_status` → `rgbd_video_gen_result` → `rgbd_video_gen_download` |

Python 采集与导出由 [sensor_cameras.py](../../src/steward/world/sensor_cameras.py)、[rgb_recording.py](../../src/steward/world/rgb_recording.py)、[recording_evidence.py](../../src/steward/ego/recording_evidence.py) 和 [novelanything_scene.py](../../src/steward/ego/novelanything_scene.py) 支持；机器人工具由 `dsh-ego`、`dsh-ego-runtime-client`、`dsh-tool-ego` 接入。

预测散图使用 `source_reconstruction_00.jpg` 表示输入视角重建，`target_prediction_00.jpg` 表示第一个目标视角；目标独立从零编号。远端 `filename` 仍用于选择下载，`local_filename` 和 `path` 表示本地文件。重新下载已校验的旧 `pred_*.jpg` 会直接更名，无需再次传输。服务器 `result.json`、ZIP 内部文件名和深度键名保持原样；目标 RGB／深度对应关系以清单 `targets` 为准。

<a id="results-and-recovery"></a>

## 结果检查与故障处理

成功报告应包含任务终态 `succeeded`、实际选帧数量和目标时间戳、下载后的 RGB／深度路径、`local_manifest_path`，并明确标记预测结果。默认目录相对机器人工作区：输入保存在 `inputs/novelanything/` 或 `inputs/rgbd_video_gen/`；输出保存在 `outputs/<model>/<job_id>/`。以工具返回的绝对路径为准，不拼接或猜测文件名。两个模型的场景格式不能互换；生成的 RGBD 不会自动显示成面板中的新增实体相机画面。

录制失败、场景重置、证据丢失或生成失败时，停止依赖该结果的步骤并报告原因。录制句柄是进程内数据，应在重启或保留期淘汰前准备并保存输入文件；停止录制不会停止机器人运动。NovelAnything 的 `UNKNOWN_OUTCOME` 需要人工核对，不能换请求键重复提交；其 API 没有远端取消能力。rgbd_video_gen 使用 `rgbd_video_gen_cancel` 显式请求取消，仅停止聊天不代表远端任务已取消。
