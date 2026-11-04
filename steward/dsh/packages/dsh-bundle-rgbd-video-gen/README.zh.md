---
description: "为 机器人 Ego Harness 添加 rgbd_video_gen 新视角 RGB/深度生成能力，通过远端服务调用，无需在本地安装 NPU 环境。"
kind: "package-bundle"
---

# @steward/dsh-bundle-rgbd-video-gen

[English](README.md) | 中文

## 摘要

为 机器人 Ego Harness 添加 rgbd_video_gen 新视角 RGB/深度生成能力，通过远端服务调用，无需在本地安装 NPU 环境。

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

在 Windows 的 Steward 目录运行安装脚本，然后重启 机器人 Ego Harness 对话。脚本仅将工具与内置 `rgbd-video-gen` skill 安装到机器人 `ego` profile；全新安装的 `scripts/setup-dsh-profiles.sh` 也包含此层。

```powershell
.\scripts\setup-rgbd-video-gen.ps1
```

默认端点为 `http://115.120.57.9/nvs`；`http://linxlab-research.com/nvs` 也可访问。可用包含 `endpoint`、`inputRoot` 和 `stateDir` 的 profile patch 覆盖 `steward-rgbd-video-gen-api`。两个目录的默认值为 `.` 与 `var/rgbd_video_gen`，由 Steward 启动时相对于 `var/dsh/robot-1/workspace` 解析。输入必须位于该机器人工作目录内。[客户端设置](../dsh-rgbd-video-gen-client/README.zh.md) 说明传输上限及期限。

示例请求：“使用 rgbd_video_gen 运行服务端 waymo080 demo，trajectory 模式、30 步、2 个目标视角。等待成功，读取结果并下载 RGB 视频、深度视频和 meta.json。”此流程使用服务端样例，无需本地图片。

本地输入可提供带内联图片的 `rgbd_videogen_scene_v1` JSON，或包含 `scene.json` 与相对路径图片的 ZIP。场景需要 2-64 个顺序帧、像素内参 K，以及以米为单位的右手 OpenCV 相机到世界变换。前 num_source_views 帧提供图片，之后的帧定义目标位姿，可设 image=null。图片必须匹配相机尺寸。证据图少于 6 张时须显式设置 `num_source_views`（范围 1-8），总帧数至少为 `num_source_views+1`。三相机图片加一个目标位姿应提交四个有序的参考相机帧，并设置 `num_source_views=3`、`max_target_views=1`。模型按 6 张证据视图训练，其他数量的质量可能下降。ZIP 工具使用服务端默认值；选择较少证据图时使用 JSON。

先通过 `ego_capture_cameras` 获取当前证据，再调用 `rgbd_video_gen_prepare_scene` 并显式给出 `targets`。默认只使用中间相机 RGB；额外源需要显式选择 `source_cameras` 且尺寸和内参相同。现有 `wide_camera` 的内参不同，此准备工具不能将其合并。手爪前视目标为 `{frame:"gripper",position:[0,0,0],look_at:[1,0,0],up:[0,0,1]}`，也支持其他位置、朝向和多个目标。目标还可在参考相机或手爪坐标中提供精确 `camera_to_frame` 矩阵。提交返回的场景路径时，必须使用工具返回的准确 `num_source_views` 和 `max_target_views`：默认单证据、单目标为2帧，`num_source_views=1`。准备操作复用中间相机内参，只读取选中的 RGB 图像。成功后下载工件并保留本地清单供后续流程读取，无需界面可视化或通用文件工具。

[Python demo](../../../tools/rgbd_video_gen_demo.py) 支持 `--url`、`--demo`、`--demo-dir`、`--wait` 和 `--request-key`。本地清单模式读取 `manifest_train.jsonl`；`--save-scene` 导出供 Harness 使用的 JSON，不执行上传。Pillow 来自 Steward Python 依赖。脚本在提交前输出请求键，POST 结果未知时须保留该键和相同参数重试。

结果包括 RGB/深度视频、frames.zip、depth.npz、对比图片及元数据。目标相机索引使用实际源视图数，而不是固定加六。生成视图属于预测；BEV demo 没有目标真值。服务端预处理输出 378×672 图片，请查看结果元数据中的相机信息。

运动过程证据应在导航前调用 `ego_rgb_recording_start`，完成后结束录制，再将成功的 recording_id 和显式目标传给 `rgbd_video_gen_prepare_recording`，不能用最后一张快照代替路线。返回的 recording_manifest_path 保留全部中间相机 RGB、时间戳和测量位姿。服务声明每任务最多8张源图：不超过8帧时默认全部使用，超过时默认在全程均匀选取8帧并保留首尾。可通过显式采样数或有序 source_frame_indices 覆盖此选择。结果报告 recorded_frames、选中索引及 all_frames_used。参考目标使用首个录制中间相机，手爪目标使用通过测量里程计关联的当前手爪。提交须使用返回的证据/目标数；准备操作要求注入机器人 Ego provider。

-----

<a id="understand-the-implementation"></a>
## 理解实现

<details>
<summary>实现详情</summary>

[cordis.patch.yml](cordis.patch.yml) 挂载 HTTP 提供者、工具消费者与[机器人 skill](skills/rgbd-video-gen/SKILL.md)。World profile 不安装此 bundle。服务端路径仍为 `/v1/nvs/...`，输入格式仍为 `rgbd_videogen_scene_v1`。包名使用连字符，工具名使用 `rgbd_video_gen_` 前缀。

会话录制测试使用独立工作目录，并将 RGBD 和 Ego API 都指向动态分配回环端口上的 HTTP 测试服务。启动模型前检查组合配置中的 Ego 地址，测试服务拒绝机器人观测、动作和操作员告警请求。因此，录制与回放均不使用正在运行的机器人服务。

</details>

-----

<a id="further-exploration"></a>
## 进一步探索

[Steward](../../../README.zh.md) · [Bundle](../dsh-bundle-rgbd-video-gen/README.zh.md) · [Service](../dsh-rgbd-video-gen/README.zh.md) · [HTTP provider](../dsh-rgbd-video-gen-client/README.zh.md) · [Tools](../dsh-tool-rgbd-video-gen/README.zh.md)

-----

<a id="model-experience"></a>
## 模型体验

间接影响：[工具包](../dsh-tool-rgbd-video-gen/README.zh.md) 负责模型指令与结果记录。

-----

<a id="known-limitations-and-deferred-work"></a>
## 已知限制与后续工作

- 机器人输入需要已准备的观测与估计相机位姿，不提供自动采集或位姿估计。`world_capture` 的仿真真值不属于机器人证据。此层调用已存在的远端服务，不负责模型权重、服务端启动或相机位姿估计。

-----

<a id="dev-note"></a>
## 开发备注

无。
