# Toy assembly profile

English | [中文](assembly-profile.zh.md)

The default runtime, robot model, sensor calibration and world rules follow the upstream configuration. Select `STEWARD_ASSEMBLY_PROFILE=toy` before starting World and Ego to opt into assembly geometry, cameras, perception, planning, execution and optional world mechanisms. Unset it, use an empty value, or use `off` for the upstream implementation, including tip-down screw insertion/driving and the Cobot Ego Harness default.

Importing `steward` applies the selected profile defaults to `os.environ` immediately through `steward/__init__.py`. Explicit environment values win. An unsupported `STEWARD_ASSEMBLY_PROFILE` value raises `ValueError` during import, even when no server is started. Changing the profile after import is unsupported; restart the process and rebuild assets in a separate data root with the intended settings.

## Settings

| Setting | Toy default |
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

The [capability catalogue](../src/steward/ego/assembly_capabilities.json) supplies other assembly defaults. Without toy, boolean capabilities default off and perception runs at the upstream 10 Hz. The support fixture defaults on only in toy; disable it with `STEWARD_SUPPORT_FIXTURE=0`. Fastening defaults to `off` in both profiles. `STEWARD_FASTEN_MODE=seat` and `STEWARD_NOSLIP=5` explicitly select the evaluation world assumptions. `STEWARD_FINGER_VISUAL=1` is an optional observer-only display setting. `STEWARD_START_STATE_DIR` supplies intermediate initial poses; leave it unset for runs from scratch.

Toy extends the ideal gripper attachment assumptions and selects its gripper geometry; it does not simulate realistic finger contact and friction. Toy `drive_screw` and `regrasp_upright` use matching assembly planners and executors. The profile targets toy assembly and does not promise upstream behavior when applied to household or Cobot tasks. The two implementations remain alongside one another.

## Interface and startup behavior

The Node Harness resolves its profile from the connected Ego before registering tools and reflection hooks. Each HTTP provider makes one additional read-only `GET /ego/control` at startup, shared by concurrent initialization. Toy adds `assemblyProfile: "toy"`; an absent field selects default, including older Ego servers. This existing unrecorded route does not acquire control, call `runtime.apply`, advance simulation time or draw randomness. Default response fields and per-tool request counts remain unchanged.

The runtime profile selects tools, reflection/continuation, camera metadata parsing and the explore timeout. A conflicting Harness environment produces one `STEWARD_HARNESS_PROFILE` stderr line; the runtime wins. An unreachable Ego or an unsupported profile fails startup before tool registration. The connection caches the profile; restart Harness after a runtime profile change. Standalone factories and local providers still use the environment; the reflection configuration schema also uses the launch environment.

Both profiles expose 13 public skills. Toy has 38 API routes and 20 core Ego tools; default has 39 routes and 21 tools. Toy exposes `complete_assembly` through `ego_predict`; default exposes `insert_screw_into_hole`, `ego_screw_holes` and `/ego/screw-holes`. `complete_assembly` is a skill value, not a separate tool. The [runtime](../tests/expected/runtime-profile-surface.json), [registered-tool](../tests/expected/registered-tool-surfaces.json) and [launch](../tests/expected/launch-profile-surface.json) fixtures record the independent references, including descriptions, schemas, SDK and robot brief.

## Running the toy assembly evaluation yourself

Prerequisites: Python 3.11 or later with the Steward dependencies, a rendering-capable MuJoCo environment, Node.js and pnpm, model credentials in the Harness environment, and reachable segmentation/perception services that you provide. Edit `steward/config/perception-services.toml` to point at those services; the repository does not supply hosted inference capacity. From the repository root:

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

The setup script installs/builds the Steward Harness plugins and creates the robot/world profiles. Configure the model and reasoning effort in those profiles before launching. `up` starts World and Ego and dispatches the scenario task by default; its startup output identifies the run directory and console/API addresses. Use the bundled [operator instructions](toy-assembly-operator-notes.md) verbatim when supplying evaluation instructions to the model; record the exact scenario task and any additional instructions with the run.

Plan for up to 4,500 wall-clock seconds (75 minutes) per evaluated episode, excluding installation and asset building. This is the evaluation budget, not a measured typical completion time; latency depends on rendering, model and perception services. `--turn-budget-s` bounds each Harness turn, not the lifetime of the World process. Enforce the episode wall-clock limit externally, record elapsed time, and stop the run when the budget is reached.

Read the console and saved session/receipt logs in the reported run directory and configured Harness home. Inspect the `complete_assembly` job decision, reason, per-interface ledger and `released` / `confirmed` / `required` counts. `complete_observed` means robot observations support completion; `executed_unverified` does not establish success. Low-level `succeeded` jobs and release counts are not assembly success. Report external geometric evaluation separately, including the 2 cm or 6 mm threshold, numerator/denominator, seed, initial state, model/effort, service versions and elapsed time. Do not mix intermediate-state runs with runs from scratch.

Toy startup emits `STEWARD_EGO_INTERFACE` and `STEWARD_HARNESS_INTERFACE` diagnostics. Verify the installed robot profile with the same environment; use the actual profile directory:

```sh
STEWARD_ASSEMBLY_PROFILE=toy node dsh/scripts/check-ego-profile.mjs \
  --expect toy --profile-dir "$STEWARD_DSH_VAR/robot-1/profiles/ego"
```

If `STEWARD_DSH_VAR` is unset, use `./var/dsh/robot-1/profiles/ego`. The check executes real offline validators without motion or network calls. Use `--expect off` with the variable unset for default. Rebuild/reinstall plugins after code changes. Toy launchers preserve inherited `STEWARD_*` and `SUBGOAL_SUCCESS` while reading Harness credentials from `.env`; default retains upstream dotenv precedence. `up` rejects cached toy robot assets whose model settings differ from the active configuration.

## Offline regression checks

The [default expectation fixture](../tests/fixtures/upstream-defaults.json) is generated from the upstream reference tree; [generation instructions](../tests/fixtures/README.md) describe the procedure and the sideboard correction. Tests read the committed expected values, drive trace, default-branch AST fingerprints and Harness reference. No base commit or Git history is needed at test runtime. Missing fixtures fail instead of skipping.

Run upstream tests without toy. The [extension test list](../tests/assembly_profile_tests.json) is excluded by the normal collection hook in default mode before import; run those tests with toy. Included assembly tests use bundled or synthetic inputs. Comparisons needing unshipped recordings, external scripts or private implementations are excluded and can return when distributable fixtures are available.

```sh
PYTHONPATH=src python -m pytest --noconftest -p no:cacheprovider -q \
  tests/test_assembly_profile_defaults.py tests/test_profile_runtime_surface.py \
  tests/test_profile_launch.py
env -u STEWARD_ASSEMBLY_PROFILE node dsh/test/profile-interface.test.mjs
STEWARD_ASSEMBLY_PROFILE=toy node dsh/test/profile-interface.test.mjs
node dsh/test/runtime-profile.test.mjs
```

These offline checks require NumPy, pytest and built Harness dependencies. They exercise constructors, prediction/dispatch, HTTP and tool registration, validators and both profile choices; they replace rendering and motion leaves. They do not validate physics, external perception or complete model episodes.
