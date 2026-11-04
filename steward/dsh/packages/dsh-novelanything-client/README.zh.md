---
description: "具备本地存储限制的远程 NovelAnything 任务 HTTP provider。"
kind: "package-reference"
---

# @steward/dsh-novelanything-client

[English](README.md) | 中文

## Summary

具备本地存储限制的远程 NovelAnything 任务 HTTP provider。

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

安装 [bundle](../dsh-bundle-novelanything/README.zh.md)；它将默认服务 provider 挂载为 `steward-novelanything-api`。可在 profile patch 中覆盖配置。本地不安装模型权重。Provider 通过显式注入的机器人 `ego` 服务准备录制场景。

| 配置项 | 默认值 | 含义 |
|---|---|---|
| `endpoint` | `required` | HTTP(S) 服务根路径，含代理前缀 |
| `inputRoot` | `required` | 机器人工作区；所有本地路径均限制在此目录下 |
| `stateDir` | `var/novelanything` | 按端点隔离的 POST 回执 |
| `inputsDir` | `inputs/novelanything` | 按内容哈希归档的输入 ZIP |
| `outputsDir` | `outputs/novelanything` | 任务产物和本地 manifest |
| `tokenEnv` | `NOVELANYTHING_API_KEY` | 可选 Bearer token 环境变量 |
| `requestTimeoutMs` | `15000` | 元数据请求超时 |
| `transferTimeoutMs` | `300000` | 上传或下载请求超时 |
| `pollIntervalMs` | `2000` | 轮询间隔 |
| `maxWaitSeconds` | `20` | 单次轮询上限，最多 60 秒 |
| `maxMetadataBytes` | `2097152` | 元数据与日志响应字节上限 |
| `maxUploadBytes` | `268435456` | 压缩 ZIP 字节上限 |
| `maxArtifactBytes` | `2147483648` | 单个产物下载字节上限 |

-----

<a id="understand-the-implementation"></a>
## Understand the implementation

<details>
<summary>实现细节</summary>

发送前以排他方式写入 POST 预留记录。已保存的接受响应可在输入一致时复用；未解决的预留记录在重启后仍阻止再次 POST。GET 失败传递给调用者。释放 provider 会中止本地请求并等待结束。

产物 URL 必须与配置端点的任务产物路径完全一致。下载流写入私有临时文件，执行字节上限检查，在服务提供大小或哈希时校验，随后重命名完整文件并逐个更新 manifest。已完成文件不受后续传输失败影响。

</details>

-----

<a id="further-exploration"></a>
## Further Exploration

[Steward](../../../README.zh.md) · [Bundle](../dsh-bundle-novelanything/README.zh.md) · [Service](../dsh-novelanything/README.zh.md) · [HTTP provider](../dsh-novelanything-client/README.zh.md) · [Tools](../dsh-tool-novelanything/README.zh.md)

-----

<a id="model-experience"></a>
## Model Experience

间接影响：[工具](../dsh-tool-novelanything/README.zh.md) 暴露就绪状态、已接受任务、终止失败和本地下载元数据。

-----

<a id="known-limitations-and-deferred-work"></a>
## Known Limitations and Deferred Work

- 服务端校验 ZIP 场景内容。客户端限制并归档上传；机器人 Ego API 使用测量里程计、固定外参和编码器推导的手爪位姿准备录制场景。
- 远程产物校验和可选。仅计算本地 SHA256 不代表通过服务端校验和验证。
- 不支持同时向同一任务目录下载。结果不明的 POST 需操作者核对；不提供自动重试或取消。

-----

<a id="dev-note"></a>
## Dev Note

无。
