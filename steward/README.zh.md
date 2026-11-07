# Steward

[English](README.md) | 中文

机器人 Ego Harness 可选用 `rgbd_video_gen` 和 `NovelAnything` 两个 skill 调用远程新视角生成服务。`scripts/setup-dsh-profiles.sh`（Linux/macOS）和 `scripts/setup-dsh-profiles.ps1`（Windows）会在机器人 profile 中安装两者；Windows 上也可用 `scripts/setup-rgbd-video-gen.ps1`、`scripts/setup-novelanything.ps1` 加到已有 profile。使用方法见 [rgbd_video_gen](dsh/packages/dsh-bundle-rgbd-video-gen/README.zh.md) 和 [NovelAnything](dsh/packages/dsh-bundle-novelanything/README.zh.md)。

模型证据沿用原有 `head_camera`；连续录制按仿真时钟采样并保存机器人里程计和编码器，手爪目标由机器人自身运动学计算。快照工具使用现有头部／广角相机各自的标定，不改变两者的布局、视场或感知分工。此次集成不包含 SLARM 或新增三目相机。

可直接使用的机器人提示词、相机约定和包职责见[感知基模skill说明](docs/novel-view-generation/README.zh.md)。

Steward 是基于 DeepSeek Harness 的具身智能机器人系统。设计见仓库根目录的 [steward.md](../steward.md)，当前实现进度与运行命令见其中的「13. Implementation Progress」；Phase 1 感知如何从 RGB-D 得到几何与类别、以及泛化路线见 [13.7](../steward.md#137-phase-1-感知方法与泛化)。

## 快速开始

```sh
uv sync
.venv/bin/steward demo          # 构建厨房并运行「把杯子放到桌上」，按 World State 判定成败
.venv/bin/steward up --scenario phase0-shuttle --no-dispatch-task   # 常驻运行世界与机器人，搬运交给下面的 shuttle
open http://127.0.0.1:7400/console   # 浏览器控制台：斜视主视图、相机、地图、指标、控制
.venv/bin/steward shuttle --api http://127.0.0.1:7420 --steps 0   # 在餐桌、柜子、餐边柜之间循环搬运
./scripts/check.sh              # Python 测试、DSH 插件编译与测试、组合配置快照比对
.venv/bin/steward build-kitchen # 机器人模型改动后（如头部加装的广角相机）重建默认场景
```

机器人头部有两台 RGB-D 相机：`head_camera` 看前方，`wide_camera` 是短焦广角相机，看身前台面与夹爪；两台都进入感知，见 [steward.md 13.8](../steward.md#138-机器人本体的双相机)。

## 与 DeepSeek Harness 一起运行

```sh
echo 'DEEPSEEK_API_KEY=sk-...' > .env     # 已被 Git 忽略，运行脚本自动加载
.venv/bin/steward up                      # WCA :7400，机器人端口 :7410，Ego API :7420；场景任务自动交给 Ego Harness
.venv/bin/steward up --phase phase1-sam   # 机器人用 SAM 3.1 自行分割 RGB-D（设置在 config/segmentation.toml）
./scripts/setup-dsh-profiles.sh           # 构建插件，创建 ego、world、world-task profile
./scripts/run-ego.sh "把杯子放到桌上"       # 机器人智能体
./scripts/run-world.sh "按描述构建一间厨房……"  # World Harness 一次性任务
./scripts/run-world.sh --port 3091        # World Harness 浏览器界面
./scripts/install-desktop-app.sh          # 在桌面安装 Steward.app：选场景后启动世界、Ego Harness 并打开控制台
.venv/bin/steward services                # 检查远程感知服务（SAM 3.1、NovelAnything、NVS、SLARM）是否可达
```

## Windows

Windows 上使用同样的命令：`uv sync` 把 `steward` 可执行文件装在 `.venv\Scripts` 下，每个 `.sh` 脚本都有同名、同参数的 PowerShell 版本。如果 PowerShell 拒绝运行未签名脚本，先执行一次 `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`。

```powershell
uv sync
.venv\Scripts\steward up --phase phase1-sam
.\scripts\setup-dsh-profiles.ps1          # 构建插件，创建 ego、world、world-task profile
.\scripts\run-ego.ps1 "把杯子放到桌上"      # 机器人智能体
.\scripts\run-world.ps1 --port 3091       # World Harness 浏览器界面
.\scripts\check.ps1                       # Python 测试、DSH 插件编译与测试、组合配置快照比对
.\scripts\install-desktop-app.ps1         # 在桌面创建 Steward 快捷方式：选场景后启动世界、Ego Harness 并打开控制台
```

桌面启动器的日志写在 `%LOCALAPPDATA%\Steward\Logs`；macOS 上写在 `~/Library/Logs/Steward`。

`ego` profile 是机器人的 Ego Harness，只能经 Ego API 感知与行动，shell、文件、网页等通用工具已禁用；`world` 与 `world-task` profile 是 World Harness，经世界控制接口构建资产、组合场景、控制 Episode。机器人与世界使用不同的 `DSH_HOME`。

## 目录

| 路径 | 内容 |
|---|---|
| `src/steward/scene` | WS 与 ES 共用的 SceneState、校验器、MJCF 转换 |
| `src/steward/assets`、`authoring` | 资产生成、校验、预览、资产库；场景与 Scenario |
| `src/steward/world` | WSim 与 Steward 主机（WCA、机器人端口） |
| `src/steward/ego` | Ego Runtime：估计器、ESim、技能、Ego API |
| `src/steward/observer` | ES 与 World State 的对比评估 |
| `dsh/` | DeepSeek Harness 插件与 bundle（锁定 `@deepseek-ai/dsh@0.2.1-rc.3`） |
| `scripts/` | profile 创建、检查、多种子回归 |
| `config/perception-services.toml` | 远程感知服务清单（地址、健康检查、接口），见 steward.md 13.7.6 |
| `config/segmentation.toml` | `phase1-sam` 的机器人侧分割设置：词表、阈值、裁剪与语义融合，见 steward.md 13.7.7 |
| `compat/snapshots` | 组合配置快照 |

运行数据写入 `var/`，不纳入版本管理。
