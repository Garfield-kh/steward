# steward-rsi: self-improvement rounds for the toy-assembly task

English | [中文](README.zh.md)

This optional package is the loop that came out of iterating on the toy airplane: run a batch of whole episodes, score every episode stage by stage (part placed, holes aligned, screw seated, finished), ask a strong model which stage is worst and what single change to try, verify that change from an intermediate state, then run a whole-episode regression before keeping it. It runs on a multi-card GPU or Ascend NPU machine. It uses only the Python standard library and changes nothing in Steward.

## What you can do with it

1. **Reproduce the airplane.** The default task. From scratch on GPU, all four screws are seated in about 6 to 7 of 100 episodes, so run a batch: 8, 32 or 64 episodes on an 8-card machine give you successful episodes to look at.
2. **Keep iterating.** Each round finds the weakest stage and improves it; see [One improvement iteration](#one-improvement-iteration). The next stage to work on is aligning the landing-gear holes.
3. **Try the car, or another task.** The car kit sits on the second table of the same scene: set `[task].name = "car"` and run the same loop.
4. **Make it general.** Put several tasks into the loop and keep only the changes that help all of them. Skills that carry over to many tasks are general skills. Hold out some unseen tasks that take no part in the iteration and use them only for checking.

## Two models, two keys

| Role | What it does | Where its key goes |
| --- | --- | --- |
| **Controller** | Drives the robot step by step inside each episode (the robot's Ego Harness). Steward's Harness talks to DeepSeek, so you need a **DeepSeek** API key. | `steward/.env` (see [Step 2](#step-2-controller-key-and-model)) |
| **Advisor** | After a batch, reads the funnel, event digests and frames and suggests the next change. Choose **one** of Claude, OpenAI (GPT) or DeepSeek. | `steward/rsi/rsi.toml` (see [Step 3](#step-3-advisor-key)) |

The two roles may share one DeepSeek key. The advisor reads its key from `rsi.toml` or from the environment variable named in `api_key_env`, and does not read `steward/.env`; so put the key in `steward/.env` for the controller, select `provider = "deepseek"` in `rsi.toml`, and either write the same key into `[deepseek].api_key` or export `DEEPSEEK_API_KEY` in the shell that runs `steward_rsi`.

## Prerequisites

- A multi-card Linux machine (the example configuration assumes 8 cards; change `[machine].devices` to match yours): NVIDIA GPUs with EGL rendering, or Ascend NPUs with the optional [`steward-npu`](../npu/README.md) package installed in the simulation environment.
- Steward's own requirements (see [Running the toy-assembly evaluation yourself](../docs/assembly-profile.md)): Python 3.11+, `uv`, Node.js and pnpm, a MuJoCo setup that can render.
- A reachable segmentation/perception service (SAM 3) that you host yourself, configured in `steward/config/perception-services.toml` and `steward/config/segmentation.toml`. The repository ships no hosted inference. With several cards you may give each card its own service via `[machine].segmentation_configs`.
- A DeepSeek API key for the controller and a key for the advisor you choose.
- Budget: an episode is limited to 75 minutes of wall-clock time by default (`[machine].wall_minutes`); asset building and start-up come on top.

Every command below starts in the repository root, and every path is relative to it. A block that changes directory says so and returns to the root at its end.

## Step 1: install Steward

```sh
cd steward
uv sync
./scripts/setup-dsh-profiles.sh     # builds the Harness plugins and creates the robot and world profiles
uv run steward services             # checks that your perception services are reachable
cd ..                               # back to the repository root
```

`setup-dsh-profiles.sh` creates the robot Harness home `steward/var/dsh/robot-1` with the profile `ego` (or under `$STEWARD_DSH_VAR/robot-1` if you set that variable).

## Step 2: controller key and model

1. Key. Create `steward/.env` (git-ignored) with:

   ```sh
   DEEPSEEK_API_KEY=sk-...
   ```

   `steward up` reads this file when it starts the robot's Harness for a task. The Harness's DeepSeek provider speaks the Anthropic Messages protocol and sends requests to `https://api.deepseek.com/anthropic` by default. To use another endpoint that speaks the same Messages protocol, add `DEEPSEEK_BASE_URL=...` to the same file.

2. Model. The shipped Harness profiles use provider `deepseek-official` with model `deepseek-flash`. To choose another model, add this entry to `steward/var/dsh/robot-1/profiles/ego/cordis.patch.yml` (create the file if it does not exist):

   ```yaml
   - id: agent-default-model
     config:
       provider: deepseek-official
       model: deepseek-flash
       # reasoningEffort: high   # optional; omitted means the provider default
   ```

   The per-step reasoning effort of the robot's reflection plugin is set in `steward/dsh/packages/dsh-bundle-ego/cordis.patch.yml` (`steward-ego-reflection` → `effort`, all `low` as shipped).

`python -m steward_rsi check` reports whether the controller key is present in `steward/.env` and whether the `ego` profile exists and overrides the model; it never prints the key.

## Step 3: advisor key

```sh
cp steward/rsi/rsi.example.toml steward/rsi/rsi.toml    # rsi.toml is git-ignored
```

In `rsi.toml`, set `[advisor].provider` to `"claude"`, `"openai"` or `"deepseek"`, then fill the matching section. For each provider either write the key into `api_key`, or leave `api_key = ""` and export the variable named in `api_key_env` (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `DEEPSEEK_API_KEY` by default). A key written in the file wins over the environment. Keys are never written to logs, prompts or error messages.

| Provider | Request | Fields sent |
| --- | --- | --- |
| `claude` | `POST {base_url}/v1/messages` (Anthropic Messages API) | `model`, `max_tokens`, `system`, `messages` with base64 `image` blocks, `output_config.effort`, optional `thinking: {type: "adaptive"}` |
| `openai` | `POST {base_url}/chat/completions` | `model`, `messages` with `image_url` data URLs, `max_completion_tokens`, `reasoning_effort` |
| `deepseek` | `POST {base_url}/chat/completions` (OpenAI-compatible) | `model`, `messages` with `image_url` data URLs, `max_tokens`, `reasoning_effort` |

The model names in `rsi.example.toml` are placeholders: `claude-opus-5-5`, `REPLACE_WITH_OPENAI_MODEL` and `deepseek-flash`. Replace them with models your account can use. `base_url` can point to any gateway that speaks the request format in the table. Reasoning tokens count against `max_tokens`. A refusal, an empty reply or a reply cut off by the token limit stops `review` with an error that names the cause, and no `advice.md` is written; for a cut-off reply, raise `max_tokens` or lower `effort`.

Relative paths inside `rsi.toml` (`output_dir`, `start_state_dir`, ...) are resolved against `steward/rsi/`, the directory that holds the file. Also set `[machine]` (backend, devices, episodes per device, wall-clock limit, the simulation Python) and `[task]` (`name = "airplane"` or `"car"`, seeds; the number of seeds is the number of episodes in the batch).

## Step 4: install this package

In any Python 3.11+ environment (the simulation environment is fine):

```sh
python -m pip install --no-deps -e steward/rsi
# or, without installing:  export PYTHONPATH=$PWD/steward/rsi/src
```

## Five-minute start: check, then a round

`STEWARD_RSI_CONFIG` names the configuration file for every command (`--config` does the same per command):

```sh
export STEWARD_RSI_CONFIG=$PWD/steward/rsi/rsi.toml
python -m steward_rsi check --offline   # configuration, Steward, controller key and profile, advisor key
python -m steward_rsi check             # also sends one short request to the advisor
python -m steward_rsi round --dry-run   # prints every command and environment variable; starts nothing
python -m steward_rsi round --name baseline
```

`round` runs `run`, then `score`, then `review`. The results are in `steward/rsi/runs/baseline/`. Each round needs a new `--name`; `run` and `round` refuse a round directory that already holds episode data.

```
steward/rsi/runs/baseline/
  plan.json, pids.jsonl                 what was started (every process group the runner owns)
  ep000-seed0/
    episode.json                        seed, device, ports, end reason, robot error/end messages, wall time
    world_initial.json, world_state.json, truth_samples.jsonl   World State snapshots
    frames/*.jpg                        rendered views for the advisor
    receipts/                           skill receipts and assembly-session events (STEWARD_RECEIPT_LOG_DIR)
    logs/build.log, up.log, runner.log
    root/, dsh/robot-1/                 this episode's own data root and Harness home
  score/funnel.md, score/score.json     funnel and per-episode results
  score/review_prompt.md, score/advice.md   exactly what the advisor saw, and its answer
```

What `run` does per episode: builds the toy-assembly assets in a private data root; copies the robot Harness home so sessions and saved capabilities do not leak between episodes; starts `python -m steward.cli up` (GPU) or `python -m steward_npu up` (NPU) with `--no-dispatch-task`, its own ports and its card (`CUDA_VISIBLE_DEVICES`/`MUJOCO_EGL_DEVICE_ID`, or `NPU_DEVICE`); sends the task file to the robot through the console chat; samples the World State every 30 s; and ends the episode when the robot's turn ends (the chat accepted the task and then reports the robot not busy), the world exits, or the wall-clock limit is reached. The robot's error and system messages from the chat during the turn go into `episode.json` as `endMessages`. It then saves the final state and stops the process group it started. It never signals processes it did not start. Ctrl-C or SIGTERM ends every running episode, including one that is still building assets. In `--dry-run` output and `plan.json`, values of `[machine].env` variables whose names contain `KEY`, `TOKEN`, `SECRET` or `PASSWORD` appear as `<redacted>`; the episodes receive the real values.

## One improvement iteration

1. **Baseline.** Run a round from scratch (no `start_state_dir`). Read `score/funnel.md` and `score/advice.md`.
2. **Pick one change.** The advice answers four fixed questions: the worst stage with counts, the most likely cause, exactly one change and where to make it (task text, operator note, skill parameter or `STEWARD_*` setting), and how to verify it from an intermediate state. Make that one change only.
3. **Export the state just before the worst stage.** `score.json` lists, per episode, when each hole was first seated (`holeFirstSeatedWallS`). Pick an episode and a time shortly before the stage that fails and export it:

   ```sh
   python -m steward_rsi save-start steward/rsi/runs/baseline/ep003-seed3 --at 1500 --out steward/rsi/states/before-gear
   ```

   This writes `<seed>.json` for each seed in `[task].seeds` (`--seeds 0,1,2` to choose) in the format read by `STEWARD_START_STATE_DIR`. `--with-robot` also stores the robot base pose and joint values. Objects held by the robot at that moment are not re-grasped; the command warns you.
4. **Verify from that state.** Set `[task].start_state_dir = "states/before-gear"` (relative to `steward/rsi/`), run a small round (for example 4–8 episodes, shorter `wall_minutes`), and compare only the counts of the worst stage with the same start state before the change.
5. **Regression.** Clear `start_state_dir` and run a whole-episode round. Keep the change only if completion did not drop and no new interference with the other kit appeared. Report intermediate-state rounds separately from from-scratch rounds; never pool them.

## Scoring

The scorer derives everything from the scenario definition, so the same code scores the car and the airplane:

- The goal `toy_assembly_goal()` in `steward/src/steward/authoring/service.py` is read with `ast` (a bundled copy is used if the source is absent). Its constraints form groups; each group is a kit, and the constraint anchor is the kit body (`w:car-body`, `w:plane-fuselage`).
- Screw holes come from the `screw-hole` affordances in the World State. Each body hole is assigned the part asset whose goal offset is nearest to the hole's position in the body frame (for the car: `wheel_0`…`wheel_3` → `car-wheel`; wheels are interchangeable).
- The scene definition `toy_assembly_spec()` in the same file (also read with `ast`, with a bundled copy) places each entity `on` a support. The kit table is the support the body is placed on; authorized screws are the screws placed on that table. Kits from a start state exported mid-assembly are therefore the same as from scratch.

Per body hole: **placed** (a hole of the right part within the goal position tolerance, 4 cm, its axis pointing the same way as the body-hole axis within 20°, as the simulator requires before it fastens two holes), **aligned** (also lateral offset ≤ `threshold_mm`), **seated** (also a screw fastens the two holes). An episode is **complete** when every hole is seated (and so also placed and aligned) with an authorized screw and lateral error ≤ `threshold_mm` (20 mm by default), the assembly rests on its own table, the robot holds none of it, no screw of the other kit fastens it, and no part or screw of the other kit is seen moved (> 2 cm or 10°) or held by the robot in the final snapshot or in any World State sample. The strict result at `strict_mm` (6 mm) is always reported too. Episodes without snapshots count as infrastructure failures and are excluded from denominators.

## Running episodes by hand

You can also skip `run`: start each episode as described in [Running the toy-assembly evaluation yourself](../docs/assembly-profile.md) and collect the files `score` needs. In the first shell, from the repository root:

```sh
EP=$PWD/steward/rsi/runs/manual/ep000-seed0
mkdir -p "$EP/receipts"
export STEWARD_ASSEMBLY_PROFILE=toy STEWARD_FASTEN_MODE=seat STEWARD_NOSLIP=5
export STEWARD_RECEIPT_LOG_DIR=$EP/receipts
(cd steward && uv run steward --root ./var/ep0 build-toy-assembly \
  && uv run steward --root ./var/ep0 up --scenario toy-assembly --phase phase1-sam --seed 0 --no-dispatch-task)
```

In a second shell, from the repository root, once the console is up (default World port 7400):

```sh
EP=$PWD/steward/rsi/runs/manual/ep000-seed0
curl -s localhost:7400/wca/scene > "$EP/world_initial.json"
python3 -c 'import json,sys; print(json.dumps({"agent":"robot","message":open(sys.argv[1]).read()}))' \
  steward/rsi/src/steward_rsi/tasks/airplane.md | curl -s -H 'Content-Type: application/json' -d @- localhost:7400/console/chat
# when the robot's turn has ended or your time limit is reached:
curl -s localhost:7400/wca/scene > "$EP/world_state.json"
```

Then, still from the repository root and with `STEWARD_RSI_CONFIG` set as above, `python -m steward_rsi score steward/rsi/runs/manual` and `python -m steward_rsi review steward/rsi/runs/manual/score`. Optional extras: `truth_samples.jsonl` (one `{"elapsed_s": ..., "state": <scene>}` per line) enables milestones and interference checks over time; `frames/final-kit.jpg` gives the advisor an image (`/wca/render?camera=overview&format=jpeg`).

## Task texts

The task texts ship inside the package, in `steward/rsi/src/steward_rsi/tasks/`. `airplane.md` is the repository's [operator notes](../docs/toy-assembly-operator-notes.md) with a one-paragraph task statement for the airplane kit. `car.md` applies the same rules to the car kit (wheel hub to side body hole, the four car screws, no screwdriver action, `finalPose: "as_is"`). The file is sent unchanged as the operator message; the console adds the robot's standard brief.

## Notes

- The runner stops each episode's `steward up` process group with SIGTERM and, after 60 s, SIGKILL. Steward starts the robot's Harness in its own process group and normally stops it on SIGTERM; if `steward up` has to be killed, a Harness process may outlive it. Check `pids.jsonl` and `logs/` after interrupted rounds.
- The goal's car offsets place the wheels around the body in the plane; the scorer uses them only to assign holes to part assets, not as a success criterion.
- Each episode builds its own assets; this costs start-up time but keeps episodes independent.
- The scorer reads World State snapshots, which robots never see; it is an evaluator, not a robot input.
- Interference with the other kit is judged from the final snapshot and the samples taken every `[run].sample_every_s`; a part picked up and put back between two samples does not appear in them.
- Each episode's `dsh/robot-1/` is a copy of the robot Harness home, including its credential files. Relative symbolic links that point outside the Harness home are stored as absolute links in the copy.

## Tests

```sh
PYTHONPATH=steward/rsi/src python -m unittest discover -s steward/rsi/tests -v
```

The tests cover configuration, the run plan (`--dry-run`), Harness copying, episode control against a simulated console chat, scoring on a recorded successful airplane episode (from scratch and from its own final state), prompt assembly and the three provider requests and replies. They need no accelerator, network or Steward installation (two tests compare the bundled goal and scene definition with the repository source and are skipped if that source is absent).
