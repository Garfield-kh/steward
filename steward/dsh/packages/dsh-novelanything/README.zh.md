---
description: "远程 NovelAnything 合成的类型化服务定义。"
kind: "package-library"
---

# @steward/dsh-novelanything

[English](README.md) | 中文

## Summary

远程 NovelAnything 合成的类型化服务定义。

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

从本包导入 `NovelAnythingService`、`Submission`、`remoteId` 和 `NovelAnythingError`。HTTP provider 实现服务，工具包消费服务。取消、恢复和下载保证见 [src/index.ts](src/index.ts)。

-----

<a id="understand-the-implementation"></a>
## Understand the implementation

<details>
<summary>实现细节</summary>

`novelAnything` Cordis 服务提供操作，不将模型绑定到具体 HTTP 实现。远程标识经过验证并使用品牌类型。

</details>

-----

<a id="further-exploration"></a>
## Further Exploration

[Steward](../../../README.zh.md) · [Bundle](../dsh-bundle-novelanything/README.zh.md) · [Service](../dsh-novelanything/README.zh.md) · [HTTP provider](../dsh-novelanything-client/README.zh.md) · [Tools](../dsh-tool-novelanything/README.zh.md)

-----

<a id="model-experience"></a>
## Model Experience

间接影响：工具消费者负责模型可见说明和 JSON 结果。

-----

<a id="known-limitations-and-deferred-work"></a>
## Known Limitations and Deferred Work

- 本地取消不代表远程取消。根据已发布 API，结果不明的提交不能安全重试。

-----

<a id="dev-note"></a>
## Dev Note

无。
