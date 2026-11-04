---
description: "让 机器人 Ego Harness 生成新视角 RGB/深度内容、查询进度并下载经过校验的结果。"
kind: "package-reference"
---

# @steward/dsh-tool-rgbd-video-gen

[English](README.md) | 中文

## 摘要

让 机器人 Ego Harness 生成新视角 RGB/深度内容、查询进度并下载经过校验的结果。

## 目录

- [使用此包](#use-this-package)
- [理解实现](#understand-the-implementation)
- [进一步探索](#further-exploration)
- [模型体验](#model-experience)
- [已知限制与后续工作](#known-limitations-and-deferred-work)
- [开发备注](#dev-note)

-----

<a id="use-this-package"></a>
## 使用此包

安装 [bundle](../dsh-bundle-rgbd-video-gen/README.zh.md)，提交前使用 `rgbd_video_gen_capabilities`。`rgbd_video_gen_submit`、`rgbd_video_gen_upload`、`rgbd_video_gen_demo` 分别选择 JSON、ZIP 和服务端 demo 输入。

使用 `rgbd_video_gen_prepare_scene`，传入准确的采集清单与显式目标描述。默认只使用中间相机 RGB；通过 `source_cameras` 显式选择其他证据。目标使用参考相机或手爪坐标，可用刚体矩阵或位置、朝向点、画面上方向描述。工具保存内联图像场景，返回提交所需的准确证据/目标数，无需通用文件工具或委派。

使用 `rgbd_video_gen_status` 查询至终态。只有 `succeeded` 才能读取 `rgbd_video_gen_result`，再将其返回的准确文件名交给 `rgbd_video_gen_download`。`rgbd_video_gen_cancel` 显式请求取消。轮询默认只查询一次，停止对话不会取消远端任务。具体参数见 [src/index.ts](src/index.ts)。

JSON 场景包含 2-64 帧，1-8 个证据帧之后至少有一个目标帧。证据图少于 6 张时显式设置 `num_source_views`；三张证据图加一个目标使用四帧并设置 `num_source_views=3`。JSON/demo 调用未提供的生成参数从服务端能力信息解析。ZIP 提交使用服务端默认值。BEV demo 调用必须显式提供视角数与高度比例。请求键标识逻辑操作，断线后重试须保持不变。

完整运动录制应使用 `rgbd_video_gen_prepare_recording`，传入成功的 ego_rgb_recording 句柄和显式目标。全部中间相机 RGB 帧保存在本地；场景可使用不超过8帧时的全部帧；更长录制默认均匀选取包含首尾的8帧。显式数量或有序批次索引可覆盖该策略。工具分别报告总数与选中数，不会静默退回最后一张快照。现有 prepare_scene 工具代表单个时间戳。

-----

<a id="understand-the-implementation"></a>
## 理解实现

<details>
<summary>实现详情</summary>

插件通过 `ctx.effect` 注册工具并调用 `ctx.rgbdVideoGen`。标准 DSH 管线记录并呈现 JSON 结果，下载文件保存在磁盘。

</details>

-----

<a id="further-exploration"></a>
## 进一步探索

[Steward](../../../README.zh.md) · [Bundle](../dsh-bundle-rgbd-video-gen/README.zh.md) · [Service](../dsh-rgbd-video-gen/README.zh.md) · [HTTP provider](../dsh-rgbd-video-gen-client/README.zh.md) · [Tools](../dsh-tool-rgbd-video-gen/README.zh.md)

-----

<a id="model-experience"></a>
## 模型体验

描述区分请求受理、生成成功与本地文件校验完成。RGB/深度属于预测而非观测。目标 PSNR 需要目标真值。目标相机 i 对应预测帧 num_source_views+i。二进制图片内容不进入工具记录。

-----

<a id="known-limitations-and-deferred-work"></a>
## 已知限制与后续工作

- 不提供位姿估计、SLARM 采集格式转换、MuJoCo 导入或自动机器人动作。

-----

<a id="dev-note"></a>
## 开发备注

无。
