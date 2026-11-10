# Steward

English | [中文](README.zh.md)

For `steward-bot`, `place` with `flip: true` turns the held object over before release on either a support or the floor. The default roll cap permits a half turn and is intersected with the robot model's declared joint limits; explicit narrower caps remain effective. Floor rotation checks the extension, wrist rotation and lowering path from measured joints. An unreachable path is rejected before execution; a successful floor placement does not silently discard `flip`.

The robot Ego Harness can use the `rgbd_video_gen` and `NovelAnything` skills for remote novel-view synthesis. `scripts/setup-dsh-profiles.sh` (Linux/macOS) and `scripts/setup-dsh-profiles.ps1` (Windows) install both into the robot profile; on Windows, `scripts/setup-rgbd-video-gen.ps1` and `scripts/setup-novelanything.ps1` add them to an existing one. See the [rgbd_video_gen guide](dsh/packages/dsh-bundle-rgbd-video-gen/README.md) and [NovelAnything guide](dsh/packages/dsh-bundle-novelanything/README.md).

Model evidence uses the existing `head_camera`. Continuous recording samples in simulation time with measured robot odometry and encoders; target gripper views use robot-owned forward kinematics. Snapshots retain the head/wide cameras and their individual calibration, layout, field of view and perception responsibilities. These capabilities do not install SLARM or extra cameras.

See the [perception model skills guide](docs/novel-view-generation/README.md) for copyable robot prompts, camera conventions and package responsibilities.

Steward is an embodied robotics system built on DeepSeek Harness. See [steward.md](../steward.md) for the design, its “13. Implementation Progress” section for progress and launch commands, and [13.7](../steward.md#137-phase-1-感知方法与泛化) for Phase 1 RGB-D geometry, classification and generalization.

## Quick start

Face matching captures copied robot observations at one recorded world instant, then releases input scheduling and the world-clock hold before correlation. State queries can run during correlation. Closing the HTTP connection cancels calculation between pieces and quarter turns; acquiring the observation snapshot still completes. Results retain the snapshot timestamp and reference pose, so observe again if objects moved. `/ego/inputs` records snapshot acquisition, including requests later cancelled, and is not a completion log.

To diagnose puzzle face matching on a running Ego API, spread the pieces and let the robot observe them and the lid, then stop other agent/tool requests and run `python3 steward/scripts/diagnose_match.py --api http://127.0.0.1:7420 --output /tmp/match-diagnosis.json` from the repository root. The standard-library script discovers `puzzle_piece` IDs and a unique `lid`, measures three state-query baselines, then tests a 15-second match deadline and a separate disconnect after 2 seconds. `--px 32` increases matching resolution; `--reference ID` and `--entities ID ...` override discovery. It does not move the robot. JSON includes responses, concurrent and post-disconnect state latency, and applied input counts; failed recovery, unconfirmed settlement after disconnect, or invalid returned matches exit with status 2 and stop further matching. A false observation does not establish a fix, and HTTP evidence cannot establish continuous CPU computation. The API must be a direct HTTP origin, without a path prefix; each recovery probe waits up to `--recovery-timeout` seconds (default 120), so total runtime can exceed that limit.

```sh
uv sync
.venv/bin/steward demo          # 构建厨房并运行「把杯子放到桌上」，按 World State 判定成败
.venv/bin/steward up --scenario phase0-shuttle --no-dispatch-task   # 常驻运行世界与机器人，搬运交给下面的 shuttle
open http://127.0.0.1:7400/console   # 浏览器控制台：斜视主视图、相机、地图、指标、控制
.venv/bin/steward shuttle --api http://127.0.0.1:7420 --steps 0   # 在餐桌、柜子、餐边柜之间循环搬运
./scripts/check.sh              # Python 测试、DSH 插件编译与测试、组合配置快照比对
.venv/bin/steward build-kitchen # 机器人模型改动后（如头部加装的广角相机）重建默认场景
```

Focus either preview with a click or Tab: WASD moves the shared view horizontally, Shift accelerates, dragging turns in place, and scrolling zooms. Release the keys, leave the preview or press Escape to stop. Presets recenter the view. These controls do not move the robot or its sensor cameras.

The console's current-thinking panel labels each block with its model step, separates the text from subsequent tool calls, and highlights the latest step. The highlight identifies the latest recorded step, not whether the model is still generating. Blocks up to 2,000 characters appear unchanged; longer blocks show the last 2,000 characters with a truncation label and expand/collapse controls. Full text loads on demand from `GET /console/thinking?robot=<id>&entry=<id>`; expansion survives polling for the same entry and clears when that entry leaves the visible window. A failed load retains the preview and offers retry; episode resets exclude old sessions from full-text reads too.

The robot has two head RGB-D cameras: `head_camera` looks ahead; the short-focal-length `wide_camera` sees the nearby worktop and gripper. Both contribute to perception; see [steward.md 13.8](../steward.md#138-机器人本体的双相机).

The `cobot-magic` arms start with joint1–joint6 at zero radians and their fingers open. The `cobot-magic` grippers use simplified palm, guide, slider and finger geometry. Their fingers extend along the tool axis and open symmetrically to a 100 mm inner aperture; the grasp point lies between the pads, 5 mm behind the tips. These visual primitives are not manufacturer CAD or contact-based gripping.

The `cobot-magic` arm/lift geometry follows manufacturer `robot.usd`; the [embodiment profile](embodiments/cobot-magic.yaml) records coordinate mappings, unit assumptions and retained simulation approximations. Runtime assets come from `src/steward/assets/cobot.py`, not the descriptive YAML. See the [migration guide](../docs/upgrade-guide/v0.2.2-alpha.2/cobot-lift-coordinates/guide.md) for saved scenes and raw lift commands.

AgileX preserves its arm posture until docking succeeds. Placement checks the required wrist orientation and IK continuity from measured joints; these checks do not replace full collision or dynamics planning. After successful work, both empty arms return their six joints to zero before the job completes, leaving lift and fingers unchanged. A held object or handle prevents homing; failures and cancellation do not initiate it. Homing failures are reported separately from the completed action.

Passing `--misnames-config ''` disables name mappings. A switch to `phase0-misnamed` without a mapping is rejected, and the running world and later episodes retain the previous phase.

## Import a USD robot

The optional `usd` extra uses MuJoCo 4.0's native OpenUSD decoder. Install it in a separate Python 3.12 environment to keep the regular simulation environment unchanged. From the repository root:

```sh
uv venv --python 3.12 steward/var/usd-env
uv pip install --python steward/var/usd-env/bin/python -e './steward[usd]'
steward/var/usd-env/bin/steward import-usd steward/assets/agilex/robot.usd --name agilex-usd \
  --meters-per-unit 1 --orient-joints --fixed-base \
  --bound-mass 0.001 --bound-inertia 0.000001 --inertia-from-bounds --drive-frequency 5
steward/.venv/bin/steward world --scenario agilex-usd --phase phase0 \
  --wca-port 7450 --robot-port 7460 --ego-api '' --world-harness '' --robot-harness ''
```

Open `http://127.0.0.1:7450/console`. Import creates a floor inspection scene and writes `template.json`, binary meshes and `report.json` under `var/steward/imports/<name>/`. Existing names are refused. Keep this directory at its original location: mesh references are absolute. Viewing the result also works with MuJoCo 3.17 and does not read the source USD. Source files are never edited.

The example options address the supplied AgileX asset: its centimeter metadata describes meter-valued geometry, one base joint is reversed, and inertias are missing. Bounds, uniform-box inertia estimates and 5 Hz critically damped drives are explicit simulation approximations, recorded in the report. Omit these options for a correctly authored asset; inspect the report before relying on dynamics. Drives hold the imported rest pose at startup, rather than applying USD drive targets.

The importer accepts one rigid robot tree with fixed, revolute and prismatic joints, meshes and basic solids, force position drives and scalar PhysX mimic constraints. Joint limits, pivots and explicit inertias survive conversion. Mesh collisions use MuJoCo convex hulls. Shader networks, textures and cameras are not transferred; missing MDL shaders are reported, while missing geometry dependencies abort import. Closed loops and unsupported dynamics are rejected. A generic import has no assigned task controller; named joints can be driven through the robot port's `/command` endpoint. The supplied dual-Piper model has a separate registered adapter, `agilex-usd`, described below.


After importing the bundled [robot.usd](assets/agilex/robot.usd) as `agilex-usd`, run it in a task scene:

```sh
steward/.venv/bin/steward up --scenario cobot-table --body agilex-usd --phase phase0 \
  --speed 1 --no-dispatch-task
```

Open `http://127.0.0.1:7400/console` and select **AgileX USD 双 Piper** in the body selector. It can also replace the robot in kitchen scenarios. `up` starts its Ego Runtime; navigation, pick and place use the same prediction receipts and skill API as the other bodies. The raw import inspection scene remains separate and unregistered.

[The simulation calibration](config/agilex-usd.toml) locates the template and defines servo gains, lift home and simulated camera pitch. Paths are relative to that file; set `STEWARD_AGILEX_USD_CONFIG` to another profile when importing under a different data root. Missing or incompatible templates fail explicitly. The adapter preserves USD meshes, arm transforms, limits, inertias and finger coupling, adds grasp points and maps the mobile base to planar position control. Wheel joints are fixed. Both dual-Piper embodiments (`cobot-magic` and `agilex-usd`) start with joint1–joint6 of each arm at zero radians. Two simulated RGB-D views mount on `top_camera_link` and move with the lift; their calibration is not imported from USD. Phase 0 uses ideal attachment, not contact-based gripping. This supports simulation tasks, not real-hardware control or validated wheel dynamics.

For the local asset integration test, run `STEWARD_TEST_AGILEX_USD=1 steward/.venv/bin/python -m pytest -q steward/tests/test_agilex.py`. It navigates, picks a bowl and places it elsewhere on the table through the Ego API, then checks the world position. The kinematics fixture contains source joint data with small box stand-ins; the integration test uses the actual imported meshes and is opt-in because it requires the import step above.

## Run with DeepSeek Harness

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

Steward runs on Windows with the same commands: `uv sync` installs the `steward` executable under `.venv\Scripts`, and every `.sh` script has a PowerShell counterpart with the same name and arguments. If PowerShell refuses unsigned scripts, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

```powershell
uv sync
.venv\Scripts\steward up --phase phase1-sam
.\scripts\setup-dsh-profiles.ps1          # 构建插件，创建 ego、world、world-task profile
.\scripts\run-ego.ps1 "把杯子放到桌上"      # 机器人智能体
.\scripts\run-world.ps1 --port 3091       # World Harness 浏览器界面
.\scripts\check.ps1                       # Python 测试、DSH 插件编译与测试、组合配置快照比对
.\scripts\install-desktop-app.ps1         # 在桌面创建 Steward 快捷方式：选场景后启动世界、Ego Harness 并打开控制台
```

The desktop launcher writes its logs to `%LOCALAPPDATA%\Steward\Logs`; on macOS they go to `~/Library/Logs/Steward`.

The robot `ego` profile observes and acts through the Ego API; general shell, file and web tools are disabled. The `world` and `world-task` profiles use the World Control API for assets, scenes and episodes. Robot and World Harnesses use separate `DSH_HOME` directories.

## Layout

| Path | Contents |
|---|---|
| `src/steward/scene` | Shared SceneState, validation and MJCF conversion |
| `src/steward/assets`, `authoring` | Asset generation, validation, previews and catalog; scenes and scenarios |
| `src/steward/world` | WSim and Steward host, WCA and robot ports |
| `src/steward/ego` | Ego Runtime: estimation, ESim, skills and Ego API |
| `src/steward/observer` | Ego-State evaluation against World State |
| `dsh/` | DeepSeek Harness plugins and bundles pinned to `@deepseek-ai/dsh@0.2.1-rc.3` |
| `scripts/` | Profile setup, checks and multi-seed regressions |
| `config/perception-services.toml` | Remote perception service endpoints, health checks and APIs; steward.md 13.7.6 |
| `config/segmentation.toml` | Robot segmentation vocabulary, thresholds, cropping and geometry fusion for `phase1-sam`; steward.md 13.7.7 |
| `compat/snapshots` | Composed configuration snapshots |

Runtime data is written to `var/` and excluded from version control.
