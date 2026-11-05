---
description: "调用远程 NovelAnything RGB/深度预测的机器人模型工具。"
kind: "package-reference"
---

# @steward/dsh-tool-novelanything

[English](README.md) | 中文

## Summary

调用远程 NovelAnything RGB/深度预测的机器人模型工具。

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

将 [bundle](../dsh-bundle-novelanything/README.zh.md) 安装到机器人 profile。本插件注入 `tools` 和 `novelAnything`；`novelAnythingTools(service)` 也向测试提供相同定义。工具输出渲染为带日志的 JSON 文本。

-----

<a id="understand-the-implementation"></a>
## Understand the implementation

<details>
<summary>实现细节</summary>

[src/index.ts](src/index.ts) 通过 `ctx.effect()` 注册健康、样例、上传、提交、状态、结果、日志和下载操作。省略 metrics 时为 true，省略 dataset 时为 re10k，省略 wait_seconds 时为零。

</details>

-----

<a id="further-exploration"></a>
## Further Exploration

[Steward](../../../README.zh.md) · [Bundle](../dsh-bundle-novelanything/README.zh.md) · [Service](../dsh-novelanything/README.zh.md) · [HTTP provider](../dsh-novelanything-client/README.zh.md) · [Tools](../dsh-tool-novelanything/README.zh.md)

-----

<a id="model-experience"></a>
## Model Experience

直接影响：工具说明纯 RGB 证据、帧选择和预测限制，再通过标准工具链返回 JSON。[录制会话测试](../../test/novelanything-snapshot.test.mjs) 验证组装后的机器人使用流程。

-----

<a id="known-limitations-and-deferred-work"></a>
## Known Limitations and Deferred Work

- 不增加文件系统或 shell 能力。机器人 RGB 证据仅支持中间 `head_camera`，不带输入深度，排除 `wide_camera` 和其他物理相机。缺少的带位姿 RGB 输入需由调用者提供；服务器样例不是机器人观测。

-----

<a id="dev-note"></a>
## Dev Note

无。
