---
description: "配置远端 rgbd_video_gen 调用，恢复结果未知的请求，并下载经过完整性校验的 RGB/深度输出。"
kind: "package-reference"
---

# @steward/dsh-rgbd-video-gen-client

[English](README.md) | 中文

## 摘要

配置远端 rgbd_video_gen 调用，恢复结果未知的请求，并下载经过完整性校验的 RGB/深度输出。

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

通过 [bundle](../dsh-bundle-rgbd-video-gen/README.zh.md) 安装。`steward-rgbd-video-gen-api` profile 行管理下列设置；`inputRoot` 和 `stateDir` 相对于 DSH 工作目录解析；`inputsDir` 和 `outputsDir` 相对于 `inputRoot` 解析。

| 字段 | 默认值 | 含义 |
|---|---|---|
| `endpoint` | Required | HTTP(S) 基地址，保留 `/nvs`；不含凭据、查询参数或片段 |
| `inputRoot` | Required | 本地 JSON/ZIP 输入的现有根目录 |
| `stateDir` | 必填 | 按服务端点隔离的请求标识与恢复记录 |
| `inputsDir` | inputs/rgbd_video_gen | 提交的 JSON、参数与 ZIP 归档；相对于 inputRoot |
| `outputsDir` | outputs/rgbd_video_gen | 按远端任务 ID 保存结果；相对于 inputRoot |
| `requestTimeoutMs` | 15000 | 单次元数据请求期限 |
| `uploadTimeoutMs` | 1800000 | 单次提交或上传期限 |
| `downloadTimeoutMs` | 300000 | 单个文件下载期限 |
| `pollIntervalMs` | 2000 | 状态查询间隔 |
| `maxWaitSeconds` | 20 | 请求轮询时长上限，最高 60；单次 HTTP 请求可能延长等待 |
| `retryCount` | 2 | 可重试 HTTP 请求的额外次数 |
| `retryDelayMs` | 1000 | 指数退避的初始延迟 |
| `maxRetryDelayMs` | 30000 | 拒绝超过此延迟的重试 |
| `maxMetadataBytes` | 2097152 | JSON 响应体大小上限 |
| `maxUploadBytes` | 536870912 | ZIP 字节上限，最高 512 MiB |
| `maxSceneBytes` | 67108864 | JSON 提交字节上限，最高 64 MiB |
| `maxArtifactBytes` | 2147483648 | 单个结果文件的下载大小上限 |
| `preparedSceneDir` | prepared | 生成场景保存到 inputsDir 下的单层目录名 |
| `maxEvidenceImageBytes` | 16777216 | 场景打包时每张 JPEG 的上限，最大 16 MiB |

提交前持久化不可变的载荷摘要和服务端幂等键。相同键配合不同输入会在本地失败。存储保留已接受的响应以便重启后恢复任务；保留 `stateDir` 才能延续此保护。取消仅限本地提交或恢复的任务，使用独立且稳定的 POST 键。

`TRANSPORT` 可能表示 POST 结果未知：复用同一操作键重试，不要编造新任务 ID。文件在大小和 SHA256 均匹配前保持为 `.part`；传输失败不影响远端生成或已校验文件，可用相同文件名重试下载。

提交输入保存在 `inputsDir`：JSON 提交保留包含内联图像的 `request.json` 和 `scene.json`；上传保留 `archive.zip` 及请求元数据。服务器 demo 仅保留选择信息与生成参数。每个 `outputsDir/<task_id>/` 包含完整服务器 `result.json` 和本地 `manifest.json`，后者列出输入归档与已校验的下载文件。RGBD 工件直接存放在该目录；`targets` 将目标相机与 RGB 路径、深度路径及 NPZ 键名对应。多次选择下载的文件会累计进入清单；再次查询结果会重新校验已有文件。缺少本地输入历史时记录为 null，不从服务器预测中反推。下载会复用已校验的完整文件。后续流程使用工具返回的 `local_manifest_path`。不同服务器的任务 ID 可能冲突时，应分别配置输出目录。

Provider 要求机器人 ego 服务。录制准备通过 `/ego/rgb/record/export` 导出每张 RGB 及测量位姿，保存完整本地证据，并按显式时序索引和目标描述构建场景；不运动也不调用远端推理。提交证据数必须与准备的录制元数据一致。默认选择在不超过8帧时使用全部帧，否则均匀选取包含首尾的8帧。显式数量或索引可覆盖该策略。失败或取消的准备不返回完成场景；已经保存的本地证据可能保留。

-----

预测散图使用 `source_reconstruction_00.jpg` 表示输入视角重建，`target_prediction_00.jpg` 表示第一个目标视角；目标独立从零编号。远端 `filename` 仍用于选择下载，`local_filename` 和 `path` 表示本地文件。重新下载已校验的旧 `pred_*.jpg` 会直接更名，无需再次传输。服务器 `result.json`、ZIP 内部文件名和深度键名保持原样；目标 RGB／深度对应关系以清单 `targets` 为准。

<a id="understand-the-implementation"></a>
## 理解实现

<details>
<summary>实现详情</summary>

提供者保留反向代理前缀，拒绝外部结果 URL，也不跟随 HTTP 重定向。插件卸载会中止本地 I/O 并等待所属操作结束。元数据限制大小，二进制文件流式写入磁盘，不转发 DeepSeek 凭据。

</details>

-----

<a id="further-exploration"></a>
## 进一步探索

[Steward](../../../README.zh.md) · [Bundle](../dsh-bundle-rgbd-video-gen/README.zh.md) · [Service](../dsh-rgbd-video-gen/README.zh.md) · [HTTP provider](../dsh-rgbd-video-gen-client/README.zh.md) · [Tools](../dsh-tool-rgbd-video-gen/README.zh.md)

-----

<a id="model-experience"></a>
## 模型体验

间接影响：工具接收结构化元数据，不接收图片字节。完整结果清单保存在结果目录。

-----

<a id="known-limitations-and-deferred-work"></a>
## 已知限制与后续工作

- 几何、图片尺寸和 ZIP 内容由远端服务校验。JSON 提交要求内联图片，此客户端不支持远端绝对图片路径。

-----

<a id="dev-note"></a>
## 开发备注

无。
