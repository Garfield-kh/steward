---
description: "通过类型化服务调用 rgbd_video_gen，显式区分提交标识、远端状态与校验后的文件回执。"
kind: "package-library"
---

# @steward/dsh-rgbd-video-gen

[English](README.md) | 中文

## 摘要

通过类型化服务调用 rgbd_video_gen，显式区分提交标识、远端状态与校验后的文件回执。

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

从 `@steward/dsh-rgbd-video-gen` 导入 `RgbdVideoGenService`、`jobId` 和请求类型；提供者实现 `ctx.rgbdVideoGen`。应用安装 [bundle](../dsh-bundle-rgbd-video-gen/README.zh.md)。

`prepareScene` 从机器人采集清单、显式目标视角和选择的证据相机创建本地场景，默认只使用中间 `head_camera` 的 RGB。目标可使用刚体 OpenCV 相机到参考帧矩阵，或位置、朝向点、画面上方向，坐标可相对于采集参考相机或同步手爪帧。参考相机坐标系下的目标无需手爪位姿。准备操作返回场景路径及准确的证据/目标数，不提交任务。JSON 或 demo 提交前，`resolveOptions` 从服务器补齐生成默认值；准备场景必须显式传回工具返回的数量。响应丢失后复用相同请求键和已解析参数。`RgbdVideoGenError` 提供错误码、可重试性、HTTP 状态与详情。取消及结果语义见[源码](src/index.ts)。

-----

<a id="understand-the-implementation"></a>
## 理解实现

<details>
<summary>实现详情</summary>

服务定义、HTTP 提供者与工具消费者分别注入。品牌化任务 ID 将远端标识与任意字符串区分开。

</details>

-----

<a id="further-exploration"></a>
## 进一步探索

[Steward](../../../README.zh.md) · [Bundle](../dsh-bundle-rgbd-video-gen/README.zh.md) · [Service](../dsh-rgbd-video-gen/README.zh.md) · [HTTP provider](../dsh-rgbd-video-gen-client/README.zh.md) · [Tools](../dsh-tool-rgbd-video-gen/README.zh.md)

-----

<a id="model-experience"></a>
## 模型体验

间接影响：[工具包](../dsh-tool-rgbd-video-gen/README.zh.md) 记录 JSON 操作结果。

-----

<a id="known-limitations-and-deferred-work"></a>
## 已知限制与后续工作

- 本地取消只停止等待或传输，不会取消服务端已接受的任务。

-----

<a id="dev-note"></a>
## 开发备注

无。
