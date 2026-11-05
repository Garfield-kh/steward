---
description: "为机器人 Ego Harness 提供仅以 RGB 输入的远程 NovelAnything 新视角合成。"
kind: "package-bundle"
---

# @steward/dsh-bundle-novelanything

[English](README.md) | 中文

## Summary

为机器人 Ego Harness 提供仅以 RGB 输入的远程 NovelAnything 新视角合成。

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

在 Windows 的 Steward 目录运行安装脚本，然后重启机器人聊天。脚本仅将 bundle 安装到 robot-1 的 `ego` profile；Unix 新环境通过 `setup-dsh-profiles.sh` 安装。World profile 不启用它。

```powershell
.\scripts\setup-novelanything.ps1
```

机器人请求示例：“使用 NovelAnything 处理提供的 scene.zip，dataset scannetpp，context [0,1]，target [2]，metrics false。等待成功后下载全部产物，返回本地 manifest 路径。” 需在机器人工作区提供已有的合规且带位姿的纯 RGB ZIP。RGB 仅支持中间 `head_camera`，不带任何输入深度；排除 `wide_camera` 和其他物理相机。多个上下文必须是同一中间相机在不同时刻/位姿的帧，因此单次同步快照仍不足。不包含任何深度文件或 depth_path 字段。API 1.3.0 支持这些输入，并仍预测 RGB 和深度。[skill](skills/novelanything/SKILL.md) 说明标定、单位和场景格式。

导航过程中的证据由机器人 agent 组合独立工具获得：运动前用 `ego_rgb_recording_start` 按要求的 FPS 开始采集，通过已有预测/收据/执行工具以 `wait=false` 导航，完成后用 `ego_rgb_recording_stop` 结束录制。录制按仿真时间采样，仅记录中间 RGB 和同步里程计/编码器。`novelanything_prepare_gripper_view` 将成功录制转换为纯 RGB 本地 ZIP，并构建当前手爪 link 原点处、沿 +X 看且画面上方为 +Z、复用中间相机内参的目标视角。它返回供现有上传/提交工具使用的归档路径和上下文/目标索引；准备工具不运动也不发起远程推理。这些米制机器人场景使用 `dataset=scannetpp`，下载结果标注为 `gripper_view_prediction`。明确指定的模型不得替换为 rgbd_video_gen。

ZIP 的 `evidence/` 中保留全部录制 RGB 证据及时间戳，即使生成只选择最多 16 个均匀分布的上下文。位姿估计采用统一的机器人里程计参考系，不需要仿真绝对位姿。录制失败、队列过载、重置和超时通过状态可见。若在导航完成前达到帧数上限，该录制不覆盖完整路线。实时句柄数量受限，宿主机重启后失效；应先准备并保存本地 ZIP 再重启宿主机。准备 provider 显式注入机器人 `ego` 服务；编译插件后新的机器人聊天进程即可加载，不需要丢弃已有录制。

默认端点为 `http://115.120.57.9/novelanything`。可在 profile patch 中覆盖 `steward-novelanything-api` 以连接其他部署。[Provider 配置](../dsh-novelanything-client/README.zh.md) 控制本地根目录、超时、传输大小限制及可选密钥环境变量。

输入保留在 `var/dsh/robot-1/workspace/inputs/novelanything/`。结果在该工作区的 `outputs/novelanything/<job_id>/` 下保留产物子目录。`manifest.json` 记录本地绝对路径、本地 SHA256、输入关联和服务结果元数据；`job.json` 保留成功的远程响应。远程推理成功与本地下载成功分别报告。

-----

<a id="understand-the-implementation"></a>
## Understand the implementation

<details>
<summary>实现细节</summary>

[cordis.patch.yml](cordis.patch.yml) 插入 provider、工具消费者和文件系统 skill provider，后者仅启用本 bundle 的 skill 目录。工具结果通过常规 JSON 日志输出。Ego profile 录制测试通过隔离 HTTP 服务执行上传、提交和下载，并在无模型凭据时回放。

</details>

-----

<a id="further-exploration"></a>
## Further Exploration

[Steward](../../../README.zh.md) · [Bundle](../dsh-bundle-novelanything/README.zh.md) · [Service](../dsh-novelanything/README.zh.md) · [HTTP provider](../dsh-novelanything-client/README.zh.md) · [Tools](../dsh-tool-novelanything/README.zh.md)

-----

<a id="model-experience"></a>
## Model Experience

间接影响：[工具](../dsh-tool-novelanything/README.zh.md) 提供模型操作，内置 skill 说明机器人证据限制。

-----

<a id="known-limitations-and-deferred-work"></a>
## Known Limitations and Deferred Work

- 手动输入需要带标定的 RGB 和目标位姿。录制的手爪场景使用机器人测量里程计和固定支架标定，不从图像估计相机运动，也不部署远程服务。预测不更新 Ego-State，也不授权运动。
- API 未声明远程取消或幂等 POST。UNKNOWN_OUTCOME 会阻止再次提交，需由操作者核对远程任务。
- 这些合成工具不暴露可选 SAM3 分割。NovelAnything 使用 CC BY-NC-SA 4.0，仅限非商业用途，见[许可证](skills/novelanything/LICENSE.md)。

-----

<a id="dev-note"></a>
## Dev Note

无。
