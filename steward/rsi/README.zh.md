# steward-rsi：toy 装配任务的自我改进迭代

[English](README.md) | 中文

这个可选包是在迭代拼飞机的过程中产生的一套流程：跑一批整局 → 逐局按环节打分（零件放上、孔对上、螺丝进孔、全部完成）→ 请一个强模型指出最差环节和只改一处的建议 → 从中间状态开局验证这一处改动 → 合并前再跑整局回归。在多卡 GPU 或昇腾 NPU 机器上运行。只用 Python 标准库，不修改 Steward 任何文件。

## 这个包能用来做什么

1. **复现拼飞机。** 默认任务。在 GPU 上从头开局，四颗螺丝全进大约是 100 局里 6 到 7 局，所以按批来跑：8 卡机器跑 8、32、64 局，就能拿到成功的局来看。
2. **继续迭代。** 每一轮找出最弱的环节并改进它，做法见[一轮迭代怎么做](#一轮迭代怎么做)。下一个要攻的环节是起落架的孔位对准。
3. **试小车，或其他任务。** 小车就在同一场景的第二张桌子上：把 `[task].name` 改成 `"car"`，用同一套流程跑。
4. **做成通用的。** 把多个任务一起放进这个流程，只保留对所有任务都有帮助的改动。能适配多个任务的技能，就是通用的技能。同时留出一部分没见过的任务不参与迭代，只用来检验。

## 两个模型、两把密钥

| 角色 | 做什么 | 密钥写在哪里 |
| --- | --- | --- |
| **控制者**（controller） | 在每一局里一步步驱动机器人（机器人的 Ego Harness）。Steward 的 Harness 对接 DeepSeek，所以需要一把 **DeepSeek** API 密钥。 | `steward/.env`（见[第 2 步](#第-2-步控制者密钥和模型)） |
| **指导者**（advisor） | 一批跑完后读漏斗、事件摘要和画面，给出下一处改动建议。Claude、OpenAI（GPT）、DeepSeek **三选一**。 | `steward/rsi/rsi.toml`（见[第 3 步](#第-3-步指导者密钥)） |

两个角色可以共用同一把 DeepSeek 密钥。指导者从 `rsi.toml` 或 `api_key_env` 指定的环境变量读取密钥，不读取 `steward/.env`；所以把密钥写进 `steward/.env` 给控制者用，在 `rsi.toml` 里选 `provider = "deepseek"`，再把同一把密钥写进 `[deepseek].api_key`，或在运行 `steward_rsi` 的终端里导出 `DEEPSEEK_API_KEY`。

## 前提

- 一台多卡的 Linux 机器（示例配置按 8 卡写，卡号在 `[machine].devices` 里改）：支持 EGL 渲染的 NVIDIA GPU，或在仿真环境中装好可选包 [`steward-npu`](../npu/README.zh.md) 的昇腾 NPU。
- Steward 自身依赖（见[自行运行 toy 装配评测](../docs/assembly-profile.zh.md)）：Python 3.11+、`uv`、Node.js 和 pnpm、能渲染的 MuJoCo 环境。
- 自己部署、可访问的分割/感知服务（SAM 3），在 `steward/config/perception-services.toml` 和 `steward/config/segmentation.toml` 中配置；仓库不提供托管推理。多卡时可用 `[machine].segmentation_configs` 给每张卡指定各自的服务。
- 控制者用的 DeepSeek 密钥，以及所选指导者的密钥。
- 时间：每局默认墙钟上限 75 分钟（`[machine].wall_minutes`），资产构建和启动另计。

下面所有命令都从仓库根目录开始执行，所有路径都相对于仓库根目录。需要切换目录的代码块会写明，并在结尾回到根目录。

## 第 1 步：安装 Steward

```sh
cd steward
uv sync
./scripts/setup-dsh-profiles.sh     # 构建 Harness 插件，创建机器人和世界 profile
uv run steward services             # 检查感知服务是否可达
cd ..                               # 回到仓库根目录
```

`setup-dsh-profiles.sh` 会创建机器人 Harness 目录 `steward/var/dsh/robot-1`，其中 profile 名为 `ego`（若设置了 `STEWARD_DSH_VAR`，则在 `$STEWARD_DSH_VAR/robot-1`）。

## 第 2 步：控制者密钥和模型

1. 密钥。新建 `steward/.env`（已被 Git 忽略），写入：

   ```sh
   DEEPSEEK_API_KEY=sk-...
   ```

   `steward up` 给机器人派任务、启动 Harness 时读取这个文件。Harness 的 DeepSeek provider 使用 Anthropic Messages 协议，默认把请求发到 `https://api.deepseek.com/anthropic`。要改用另一个同样支持 Messages 协议的端点，在同一文件加 `DEEPSEEK_BASE_URL=...`。

2. 模型。随仓库提供的 Harness profile 默认使用 provider `deepseek-official`、模型 `deepseek-flash`。要换模型，在 `steward/var/dsh/robot-1/profiles/ego/cordis.patch.yml`（没有就新建）中加入：

   ```yaml
   - id: agent-default-model
     config:
       provider: deepseek-official
       model: deepseek-flash
       # reasoningEffort: high   # 可选；不写则用 provider 默认值
   ```

   机器人反思插件每一步的推理强度在 `steward/dsh/packages/dsh-bundle-ego/cordis.patch.yml`（`steward-ego-reflection` → `effort`，出厂全部为 `low`）。

`python -m steward_rsi check` 会报告 `steward/.env` 里是否有控制者密钥、`ego` profile 是否存在、是否覆盖了模型；不会打印密钥。

## 第 3 步：指导者密钥

```sh
cp steward/rsi/rsi.example.toml steward/rsi/rsi.toml    # rsi.toml 已被 Git 忽略
```

在 `rsi.toml` 里把 `[advisor].provider` 设为 `"claude"`、`"openai"` 或 `"deepseek"`，再填对应小节。每家二选一：把密钥直接写进 `api_key`；或保持 `api_key = ""`，导出 `api_key_env` 指定的环境变量（默认分别是 `ANTHROPIC_API_KEY`、`OPENAI_API_KEY`、`DEEPSEEK_API_KEY`）。文件里填了就优先用文件里的。密钥不会写进日志、提示词或报错信息。

| Provider | 请求 | 发送的字段 |
| --- | --- | --- |
| `claude` | `POST {base_url}/v1/messages`（Anthropic Messages API） | `model`、`max_tokens`、`system`、含 base64 `image` 块的 `messages`、`output_config.effort`、可选 `thinking: {type: "adaptive"}` |
| `openai` | `POST {base_url}/chat/completions` | `model`、含 `image_url` data URL 的 `messages`、`max_completion_tokens`、`reasoning_effort` |
| `deepseek` | `POST {base_url}/chat/completions`（OpenAI 兼容） | `model`、含 `image_url` data URL 的 `messages`、`max_tokens`、`reasoning_effort` |

`rsi.example.toml` 里的模型名是占位：`claude-opus-5-5`、`REPLACE_WITH_OPENAI_MODEL`、`deepseek-flash`，请换成你的账号可用的模型。`base_url` 可指向任何支持表中请求格式的网关。推理 token 也计入 `max_tokens`。模型拒答、回复为空或回复被 token 上限截断时，`review` 报错并写明原因，不生成 `advice.md`；被截断时调大 `max_tokens` 或调低 `effort`。

`rsi.toml` 里的相对路径（`output_dir`、`start_state_dir` 等）相对于文件所在目录 `steward/rsi/` 解析。同时设置 `[machine]`（后端、卡号、每卡并行局数、每局墙钟上限、仿真环境的 Python）和 `[task]`（`name = "airplane"` 或 `"car"`、种子；种子个数就是这一批的局数）。

## 第 4 步：安装本包

任意 Python 3.11+ 环境（用仿真环境即可）：

```sh
python -m pip install --no-deps -e steward/rsi
# 或者不安装：export PYTHONPATH=$PWD/steward/rsi/src
```

## 五分钟上手：check，然后跑一轮

`STEWARD_RSI_CONFIG` 为之后的每条命令指定配置文件（单条命令也可以用 `--config`）：

```sh
export STEWARD_RSI_CONFIG=$PWD/steward/rsi/rsi.toml
python -m steward_rsi check --offline   # 检查配置、Steward、控制者密钥与 profile、指导者密钥
python -m steward_rsi check             # 另外向指导者发一条最短请求
python -m steward_rsi round --dry-run   # 打印每条命令和环境变量，不启动任何东西
python -m steward_rsi round --name baseline
```

`round` 依次执行 `run`、`score`、`review`，结果在 `steward/rsi/runs/baseline/`。每一轮用一个新的 `--name`；轮次目录里已有局数据时，`run` 和 `round` 会拒绝执行。

```
steward/rsi/runs/baseline/
  plan.json, pids.jsonl                 启动了什么（runner 拥有的每个进程组）
  ep000-seed0/
    episode.json                        种子、卡号、端口、结束原因、机器人的错误/结束消息、墙钟时间
    world_initial.json, world_state.json, truth_samples.jsonl   World State 快照
    frames/*.jpg                        给指导者看的渲染画面
    receipts/                           技能回执与装配会话事件（STEWARD_RECEIPT_LOG_DIR）
    logs/build.log, up.log, runner.log
    root/, dsh/robot-1/                 本局独立的数据根目录和 Harness 目录
  score/funnel.md, score/score.json     漏斗和逐局结果
  score/review_prompt.md, score/advice.md   指导者看到的完整内容和它的回答
```

`run` 对每一局：在独立数据根目录构建 toy 装配资产；复制一份机器人 Harness 目录，避免会话和已保存能力在局间串扰；用 `--no-dispatch-task`、独立端口和指定的卡（`CUDA_VISIBLE_DEVICES`/`MUJOCO_EGL_DEVICE_ID`，或 `NPU_DEVICE`）启动 `python -m steward.cli up`（GPU）或 `python -m steward_npu up`（NPU）；通过控制台对话把任务文件发给机器人；每 30 秒采一次 World State；机器人这一轮结束（控制台已接受任务，之后报告机器人不忙）、世界进程退出或到达墙钟上限时结束本局；这一轮里控制台上机器人的错误和系统消息写进 `episode.json` 的 `endMessages`；然后保存终态，并停掉本局自己启动的进程组。不会向不是它启动的进程发信号。Ctrl-C 或 SIGTERM 会结束所有正在运行的局，包括还在构建资产的局。`--dry-run` 的输出和 `plan.json` 里，`[machine].env` 中名字含 `KEY`、`TOKEN`、`SECRET` 或 `PASSWORD` 的变量值显示为 `<redacted>`；各局进程拿到的是真实值。

## 一轮迭代怎么做

1. **基线。** 从头跑一轮（不设 `start_state_dir`），看 `score/funnel.md` 和 `score/advice.md`。
2. **只选一处改动。** 建议固定回答四个问题：最差环节及计数、最可能的原因、只改一处应该改什么以及改在哪里（任务文本、操作说明、技能参数或 `STEWARD_*` 设置）、怎么用中间状态开局验证。只做这一处改动。
3. **导出最差环节之前的状态。** `score.json` 里每局都有每个孔第一次装上螺丝的时间（`holeFirstSeatedWallS`）。选一局、选失败环节之前不久的时间点导出：

   ```sh
   python -m steward_rsi save-start steward/rsi/runs/baseline/ep003-seed3 --at 1500 --out steward/rsi/states/before-gear
   ```

   它按 `[task].seeds`（或 `--seeds 0,1,2`）为每个种子写一个 `<seed>.json`，格式就是 `STEWARD_START_STATE_DIR` 读取的格式。`--with-robot` 会同时保存机器人底盘位姿和关节值。那一刻机器人手里拿着的东西不会被重新抓住，命令会给出提醒。
4. **从这个状态验证。** 设 `[task].start_state_dir = "states/before-gear"`（相对于 `steward/rsi/`），跑一小轮（例如 4–8 局、调短 `wall_minutes`），只和改动前同一起点的结果比较最差环节的计数。
5. **回归。** 清空 `start_state_dir`，跑一轮整局。只有完成数没有下降、也没有新增对另一套玩具的干扰时才保留改动。中间状态开局的结果和从头开局的结果分开报告，不要混算。

## 打分规则

打分全部从场景定义推出，同一套代码给小车和飞机打分：

- 用 `ast` 读取 `steward/src/steward/authoring/service.py` 里的目标 `toy_assembly_goal()`（源码不在时用包内副本）。目标约束连成若干组，每组是一套玩具，约束的锚点就是该套的主体（`w:car-body`、`w:plane-fuselage`）。
- 螺丝孔来自 World State 里的 `screw-hole` 可供性。每个主体孔对应的零件资产，取目标偏移量离该孔（主体坐标系）最近的那条约束（小车：`wheel_0`…`wheel_3` → `car-wheel`；四个轮子可以互换）。
- 同一文件里的场景定义 `toy_assembly_spec()`（同样用 `ast` 读取，也有包内副本）写明每个实体放在（`on`）哪个支撑物上。该套的桌子是主体所放的那张桌子；授权螺丝是放在这张桌子上的螺丝。因此从装到一半导出的中间状态开局，得到的套件和从头开局相同。

每个主体孔：**placed**（正确零件的某个孔在目标位置容差 4 cm 内，且它的轴线与主体孔轴线同向、夹角 ≤ 20°，这也是仿真器把两个孔连上前要求的方向）、**aligned**（另外横向偏差 ≤ `threshold_mm`）、**seated**（另外有螺丝把这两个孔连上）。一局 **complete** 要求：每个孔都 seated（因而也满足 placed 和 aligned）且由授权螺丝装上、横向偏差 ≤ `threshold_mm`（默认 20 mm）、装配体经支撑关系落在自己的桌子上、机器人没有拿着它、没有用另一套的螺丝装它，并且在终态快照和所有 World State 采样中都没有看到另一套玩具的零件或螺丝被移动（> 2 cm 或 10°）或被机器人拿着。始终同时报告 `strict_mm`（6 mm）的严格结果。没有快照的局算基础设施故障，不进分母。

## 手工跑整局

也可以不用 `run`，按[自行运行 toy 装配评测](../docs/assembly-profile.zh.md)逐局手工运行，并收集 `score` 需要的文件。第一个终端，从仓库根目录：

```sh
EP=$PWD/steward/rsi/runs/manual/ep000-seed0
mkdir -p "$EP/receipts"
export STEWARD_ASSEMBLY_PROFILE=toy STEWARD_FASTEN_MODE=seat STEWARD_NOSLIP=5
export STEWARD_RECEIPT_LOG_DIR=$EP/receipts
(cd steward && uv run steward --root ./var/ep0 build-toy-assembly \
  && uv run steward --root ./var/ep0 up --scenario toy-assembly --phase phase1-sam --seed 0 --no-dispatch-task)
```

控制台起来后，在第二个终端从仓库根目录执行（世界端口默认 7400）：

```sh
EP=$PWD/steward/rsi/runs/manual/ep000-seed0
curl -s localhost:7400/wca/scene > "$EP/world_initial.json"
python3 -c 'import json,sys; print(json.dumps({"agent":"robot","message":open(sys.argv[1]).read()}))' \
  steward/rsi/src/steward_rsi/tasks/airplane.md | curl -s -H 'Content-Type: application/json' -d @- localhost:7400/console/chat
# 机器人这一轮结束或到你的时间上限时：
curl -s localhost:7400/wca/scene > "$EP/world_state.json"
```

然后仍在仓库根目录、按上文设好 `STEWARD_RSI_CONFIG`，执行 `python -m steward_rsi score steward/rsi/runs/manual` 和 `python -m steward_rsi review steward/rsi/runs/manual/score`。可选：`truth_samples.jsonl`（每行一个 `{"elapsed_s": ..., "state": <scene>}`）可以算里程碑和全程干扰检查；`frames/final-kit.jpg` 给指导者一张图（`/wca/render?camera=overview&format=jpeg`）。

## 任务说明

任务文本随包提供，在 `steward/rsi/src/steward_rsi/tasks/`。`airplane.md` 是仓库里的[操作说明](../docs/toy-assembly-operator-notes.md)加上一段飞机任务描述。`car.md` 按同样规则改写为小车（轮毂孔对车身侧面孔、用小车桌上的四颗螺丝、不做螺丝刀动作、`finalPose: "as_is"`）。文件原样作为操作员消息发出，控制台会自动加上机器人的标准简介。

## 说明

- 每局结束时 runner 先向本局 `steward up` 的进程组发 SIGTERM，60 秒后仍未退出则发 SIGKILL。Steward 会把机器人 Harness 放在单独的进程组里启动，正常情况下收到 SIGTERM 时会一并停掉；如果 `steward up` 被强杀，Harness 进程可能残留。中断过的轮次请检查 `pids.jsonl` 和 `logs/`。
- 目标里小车轮子的偏移量是在平面上围着车身摆的；打分器只用它把孔对应到零件资产，不把它当成功判据。
- 每局各自构建资产，启动慢一些，但各局相互独立。
- 打分器读取 World State 快照，这是机器人看不到的真值；它是评测器，不是机器人的输入。
- 对另一套玩具的干扰按终态快照和每 `[run].sample_every_s` 秒一次的采样判断；在两次采样之间拿起又放回的零件不会出现在这些快照里。
- 每局的 `dsh/robot-1/` 是机器人 Harness 目录的副本，包括其中的凭据文件。指向 Harness 目录之外的相对软链接，在副本中保存为绝对链接。

## 测试

```sh
PYTHONPATH=steward/rsi/src python -m unittest discover -s steward/rsi/tests -v
```

测试覆盖配置、运行计划（`--dry-run`）、Harness 目录复制、对模拟控制台对话的单局控制、在一局真实成功的飞机记录上的打分（从头开局和以它自己的终态作开局两种）、提示词组装，以及三家接口的请求和回复处理；不需要加速卡、网络或安装 Steward（有两个测试比对包内的目标副本、场景定义副本和仓库源码，源码不在时跳过）。
