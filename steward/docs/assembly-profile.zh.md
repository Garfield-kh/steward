# Toy 装配配置

[English](assembly-profile.md) | 中文

默认运行时、机器人模型、传感器标定及世界规则沿用上游配置。在启动 World 和 Ego 前设置 `STEWARD_ASSEMBLY_PROFILE=toy`，即可启用装配几何、相机、感知、规划、执行及可选世界机制。不设置、设为空值或设为 `off` 时使用上游实现，包括钻头朝下的螺钉插入/拧紧技能及 Cobot Ego Harness 默认配置。

导入 `steward` 时，`steward/__init__.py` 会立即将所选配置的默认值写入 `os.environ`，已有显式环境值优先。`STEWARD_ASSEMBLY_PROFILE` 为非法值时，即使没有启动服务器，导入也会抛出 `ValueError`。不支持导入后切换配置；应重启进程，并按目标设置在独立数据根目录重新构建资产。

## 设置

| 设置 | Toy 默认值 |
| --- | --- |
| `SUBGOAL_SUCCESS` | `0` |
| `STEWARD_WRIST_CAMERA` | `1` |
| `STEWARD_RGBD_STORED` | `1` |
| `STEWARD_PERCEPTION_HZ` | `5` |
| `STEWARD_ASSEMBLY_DRIVE_BUDGET_S` | `600` |
| `STEWARD_ASSEMBLY_LANDING_CHECK` | `0` |
| `STEWARD_ASSEMBLY_DESTINATION_CHECK` | `0` |
| `STEWARD_ASSEMBLY_RELEASE_OBSERVATION` | `0` |
| `STEWARD_ASSEMBLY_COMPONENT_LANDING_CHECK` | `0` |
| `STEWARD_ASSEMBLY_SESSION_FACES` | `0` |
| `STEWARD_ASSEMBLY_INSERTION_SEARCH` | `0` |
| `STEWARD_ASSEMBLY_SESSION_INVENTORY` | `0` |
| `STEWARD_ASSEMBLY_RELEASE_CONDITIONS` | `0` |
| `STEWARD_ASSEMBLY_RELEASE_RETENTION` | `0` |
| `STEWARD_ASSEMBLY_RELEASE_PLANE` | `0` |
| `STEWARD_ASSEMBLY_FIRST_INTERFACE` | `0` |
| `STEWARD_ASSEMBLY_SESSION_IDENTITY` | `0` |

其它装配默认值见[能力目录](../src/steward/ego/assembly_capabilities.json)。非 toy 模式下布尔能力默认关闭，感知沿用上游 10 Hz。支撑固定机制仅在 toy 下默认开启，可用 `STEWARD_SUPPORT_FIXTURE=0` 关闭。两种配置的紧固模式均默认 `off`。评测显式设置 `STEWARD_FASTEN_MODE=seat` 和 `STEWARD_NOSLIP=5`；`STEWARD_FINGER_VISUAL=1` 仅改变观察者看到的显示。`STEWARD_START_STATE_DIR` 提供中间状态初始位姿；从头运行时应取消设置。

Toy 扩展理想夹持附着假设并选择对应夹爪几何，不模拟真实手指接触与摩擦。Toy 的 `drive_screw` 和 `regrasp_upright` 使用配套装配规划器与执行器。该配置面向 toy 装配，应用于家务或 Cobot 任务时不保证上游行为。两套实现仍并排保留。

## 接口与启动行为

Node Harness 在注册工具及反思钩子前，从所连接的 Ego 解析配置。每个 HTTP provider 启动时额外发送一次只读 `GET /ego/control`，并发初始化共享该请求。Toy 新增 `assemblyProfile: "toy"`；字段缺失时选择 default，兼容旧 Ego。该既有、不记录输入的路由不获取控制权、不调用 `runtime.apply`、不推进仿真时间、不消耗随机数。默认响应字段及每次工具调用的请求数不变。

运行时配置决定工具、反思/续行、相机元数据解析及 explore 超时。Harness 环境冲突时，stderr 输出一次 `STEWARD_HARNESS_PROFILE`，以运行时为准。Ego 不可达或配置非法时，启动在工具注册前失败。连接缓存该配置，运行时配置变更后需重启 Harness。独立工厂、本地 provider 及反思配置 schema 仍使用启动环境。

两种配置均公开 13 个技能。Toy 有 38 个 API 路由和 20 个核心 Ego 工具；default 为 39 个路由和 21 个工具。Toy 通过 `ego_predict` 提供 `complete_assembly`；default 提供 `insert_screw_into_hole`、`ego_screw_holes` 及 `/ego/screw-holes`。`complete_assembly` 是技能值，不是独立工具。[运行时](../tests/expected/runtime-profile-surface.json)、[已注册工具](../tests/expected/registered-tool-surfaces.json)及[启动](../tests/expected/launch-profile-surface.json)夹具保存独立参考，包括描述、schema、SDK 和机器人简介。

## 自行运行 toy 装配评测

前提：Python 3.11 或以上及 Steward 依赖、可渲染的 MuJoCo 环境、Node.js 和 pnpm、Harness 环境中的模型凭据，以及自行提供的可达分割/感知服务。将 `steward/config/perception-services.toml` 指向这些服务；仓库不提供托管推理资源。从仓库根目录执行：

```sh
cd steward
uv sync
./scripts/setup-dsh-profiles.sh
uv run steward services
export STEWARD_ASSEMBLY_PROFILE=toy
export STEWARD_FASTEN_MODE=seat
export STEWARD_NOSLIP=5
unset STEWARD_START_STATE_DIR
uv run steward --root ./var/toy-evaluation build-toy-assembly
uv run steward --root ./var/toy-evaluation up --scenario toy-assembly \
  --phase phase1-sam --seed 0 --turn-budget-s 4500
```

初始化脚本安装并构建 Steward Harness 插件，创建机器人与世界 profile。启动前在这些 profile 中设置模型及推理强度。`up` 默认启动 World、Ego 并派发场景任务；启动输出给出运行目录和控制台/API 地址。向模型提供评测说明时，原样使用仓库中的[操作说明](toy-assembly-operator-notes.md)，并随运行记录保存确切的场景任务及额外指令。

每局评测按最多 4,500 秒（75 分钟）墙钟时间预留，安装与资产构建另计。这是评测预算，不是实测典型完成时间；延迟取决于渲染、模型及感知服务。`--turn-budget-s` 限制每个 Harness 回合，而非 World 进程寿命。需在外部执行整局墙钟限制、记录耗时，并在预算耗尽时停止运行。

读取控制台及启动输出所指运行目录、配置的 Harness home 中保存的会话/回执日志。检查 `complete_assembly` 作业的 decision、reason、各接口账本以及 `released` / `confirmed` / `required` 计数。`complete_observed` 表示机器人观察支持完成；`executed_unverified` 不代表成功。底层作业的 `succeeded` 和释放计数不能等同于装配成功。外部几何评测应独立报告，注明 2 cm 或 6 mm 阈值、分子/分母、种子、初始状态、模型/推理强度、服务版本及耗时。中间状态运行不能与从头运行混计。

Toy 启动输出 `STEWARD_EGO_INTERFACE` 和 `STEWARD_HARNESS_INTERFACE` 诊断。使用相同环境和实际 profile 目录核对已安装的机器人配置：

```sh
STEWARD_ASSEMBLY_PROFILE=toy node dsh/scripts/check-ego-profile.mjs \
  --expect toy --profile-dir "$STEWARD_DSH_VAR/robot-1/profiles/ego"
```

若未设置 `STEWARD_DSH_VAR`，使用 `./var/dsh/robot-1/profiles/ego`。检查运行真实的离线校验器，不产生动作或网络请求。检查 default 时取消 profile 环境变量并使用 `--expect off`。代码变更后重新构建、安装插件。Toy 启动器读取 `.env` 中的 Harness 凭据时保留继承的 `STEWARD_*` 和 `SUBGOAL_SUCCESS`；default 沿用上游 dotenv 优先级。`up` 会拒绝模型设置与当前配置不符的 toy 缓存资产。

## 离线回归检查

[默认期望夹具](../tests/fixtures/upstream-defaults.json)由上游参考树生成；[生成说明](../tests/fixtures/README.zh.md)记录流程与 sideboard 修正。测试读取已提交的期望值、驱动轨迹、默认分支 AST 指纹及 Harness 参考；运行测试无需基线提交或 Git 历史。缺少夹具会失败，不会跳过。

不启用 toy 运行上游测试。正常收集钩子在 default 下会于导入前排除[扩展测试列表](../tests/assembly_profile_tests.json)，这些测试需启用 toy。保留的装配测试使用随仓库提供或合成的输入。依赖未提供录制、外部脚本或私有实现的比较已移出，待可分发夹具齐备后恢复。

```sh
PYTHONPATH=src python -m pytest --noconftest -p no:cacheprovider -q \
  tests/test_assembly_profile_defaults.py tests/test_profile_runtime_surface.py \
  tests/test_profile_launch.py
env -u STEWARD_ASSEMBLY_PROFILE node dsh/test/profile-interface.test.mjs
STEWARD_ASSEMBLY_PROFILE=toy node dsh/test/profile-interface.test.mjs
node dsh/test/runtime-profile.test.mjs
```

这些离线检查需要 NumPy、pytest 及已构建的 Harness 依赖。它们覆盖构造、预测/派发、HTTP 与工具注册、校验器以及两种配置，并替换渲染与运动末端调用；不验证物理、外部感知或模型整局运行。
