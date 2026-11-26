"""Plan and run a batch of whole episodes in parallel.

Episode ``i`` uses slot ``i % slots``; a slot is bound to one device and runs its episodes one
after another. Every episode gets its own Steward data root, Harness home, receipt directory and
port triple. Each started process gets its own session (process group); at the wall-clock limit
the runner signals only the process groups it started and recorded in ``pids.jsonl``.

A round directory that already holds episode data is refused. Environment values whose names
contain KEY, TOKEN, SECRET or PASSWORD are shown as ``<redacted>`` in ``--dry-run`` output and in
``plan.json``; the episode processes receive the real values.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .config import Config

SCENARIO = "toy-assembly"
PORT_STRIDE = 10
STOP_GRACE_S = 60.0
POLL_S = 5.0
STARTUP_POLL_S = 2.0
BUILD_POLL_S = 1.0
SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD")
REDACTED = "<redacted>"
END_MESSAGES_KEPT = 20


class RoundExistsError(Exception):
    """The round directory already holds episode data from an earlier run."""


def is_secret_name(name: str) -> bool:
    upper = name.upper()
    return any(marker in upper for marker in SECRET_MARKERS)


def masked_env(env: dict[str, str]) -> dict[str, str]:
    """``env`` with the value of every secret-looking name replaced by ``<redacted>``."""
    return {k: (REDACTED if is_secret_name(k) else v) for k, v in env.items()}


def check_round_dir(round_dir: Path) -> None:
    """Raise :class:`RoundExistsError` when ``round_dir`` holds a plan, a pid record or episode directories."""
    if not round_dir.is_dir():
        return
    used = sorted(p.name for p in round_dir.iterdir()
                  if p.name in ("plan.json", "pids.jsonl") or (p.is_dir() and p.name.startswith("ep")))
    if used:
        shown = ", ".join(used[:4]) + (", ..." if len(used) > 4 else "")
        raise RoundExistsError(f"round directory {round_dir} already holds episode data ({shown}); "
                               f"choose another --name")


def copy_harness(template: Path, dest: Path) -> None:
    """Copy the Harness home ``template`` to ``dest``, without its ``sessions`` and ``workspace``.

    Symbolic links stay links. A relative link whose target lies inside ``template`` keeps its text,
    so the copy's link points into the copy. A relative link whose target lies outside ``template``
    is rewritten to that target's absolute path, which stays valid at any directory depth. Absolute
    links are copied unchanged.
    """
    root = os.path.normpath(os.path.abspath(template))
    shutil.copytree(root, dest, symlinks=True, ignore=shutil.ignore_patterns("sessions", "workspace"))
    for dirpath, dirnames, filenames in os.walk(dest):
        source_dir = os.path.normpath(os.path.join(root, os.path.relpath(dirpath, dest)))
        for name in dirnames + filenames:
            link = os.path.join(dirpath, name)
            if not os.path.islink(link):
                continue
            target = os.readlink(link)
            if os.path.isabs(target):
                continue
            resolved = os.path.normpath(os.path.join(source_dir, target))
            if os.path.commonpath([resolved, root]) == root:
                continue
            os.unlink(link)
            os.symlink(resolved, link)


@dataclasses.dataclass
class EpisodePlan:
    index: int
    slot: int
    device: int
    seed: int
    dir: Path
    root: Path
    harness_home: Path
    ports: dict[str, int]
    env_set: dict[str, str]
    env_unset: list[str]
    build_cmd: list[str]
    up_cmd: list[str]
    task_file: Path

    def to_dict(self) -> dict:
        """JSON-ready plan with secret-looking environment values redacted."""
        d = dataclasses.asdict(self)
        d["env_set"] = masked_env(self.env_set)
        return json.loads(json.dumps(d, default=str))


def _entry(cfg: Config) -> list[str]:
    return ["-m", "steward_npu"] if cfg.machine.backend == "npu" else ["-m", "steward.cli"]


def device_env(cfg: Config, device: int) -> dict[str, str]:
    if cfg.machine.backend == "npu":
        return {"NPU_DEVICE": str(device), "MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa"}
    return {"CUDA_VISIBLE_DEVICES": str(device), "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl",
            "MUJOCO_EGL_DEVICE_ID": str(device)}


def plan(cfg: Config, round_dir: Path) -> list[EpisodePlan]:
    m = cfg.machine
    slots = len(m.devices) * m.episodes_per_device
    plans = []
    for index, seed in enumerate(cfg.task.seeds):
        slot = index % slots
        device = m.devices[slot // m.episodes_per_device]
        ep_dir = round_dir / f"ep{index:03d}-seed{seed}"
        root = ep_dir / "root"
        harness = ep_dir / "dsh" / "robot-1" if cfg.run.isolate_harness else cfg.run.robot_harness_template
        base = m.base_port + PORT_STRIDE * slot
        ports = {"wca": base, "robot": base + 1, "api": base + 2}
        env = {"STEWARD_ASSEMBLY_PROFILE": "toy", "STEWARD_FASTEN_MODE": "seat", "STEWARD_NOSLIP": "5",
               "STEWARD_RECEIPT_LOG_DIR": str(ep_dir / "receipts"), "PYTHONUNBUFFERED": "1",
               **device_env(cfg, device), **m.env}
        unset = []
        if cfg.task.start_state_dir:
            env["STEWARD_START_STATE_DIR"] = str(cfg.task.start_state_dir)
        else:
            unset.append("STEWARD_START_STATE_DIR")
        py = [m.python, *_entry(cfg), "--root", str(root)]
        up = [*py, "up", "--scenario", SCENARIO, "--phase", cfg.task.phase, "--seed", str(seed),
              "--no-dispatch-task", "--wca-port", str(ports["wca"]), "--robot-port", str(ports["robot"]),
              "--api-port", str(ports["api"]), "--robot-harness", str(harness), "--world-harness", "",
              "--turn-budget-s", str(int(m.wall_minutes * 60))]
        if not cfg.run.pause_on_perception_fault:
            up.append("--no-pause-on-perception-fault")
        if m.segmentation_configs:
            up += ["--segmentation-config", m.segmentation_configs[m.devices.index(device)]]
        plans.append(EpisodePlan(index=index, slot=slot, device=device, seed=seed, dir=ep_dir, root=root,
                                 harness_home=harness, ports=ports, env_set=env, env_unset=unset,
                                 build_cmd=[*py, "build-toy-assembly"], up_cmd=up, task_file=cfg.task.task_file))
    return plans


def dry_run_text(cfg: Config, plans: list[EpisodePlan]) -> str:
    """Human-readable plan: one block per episode with its environment changes and commands."""
    import shlex

    m = cfg.machine
    lines = [f"backend={m.backend} devices={m.devices} episodes_per_device={m.episodes_per_device} "
             f"wall_minutes={m.wall_minutes} episodes={len(plans)} "
             f"slots={len(m.devices) * m.episodes_per_device}", f"cwd: {cfg.steward_dir}"]
    if cfg.run.isolate_harness:
        lines.append(f"harness template copied per episode from: {cfg.run.robot_harness_template}")
    for p in plans:
        lines += ["", f"# episode {p.index} slot {p.slot} device {p.device} seed {p.seed} ports {p.ports}",
                  f"#   dir {p.dir}"]
        lines += [f"export {k}={shlex.quote(v)}" for k, v in sorted(masked_env(p.env_set).items())]
        lines += [f"unset {k}" for k in p.env_unset]
        lines.append(shlex.join(p.build_cmd))
        lines.append(shlex.join(p.up_cmd))
        lines.append(f"# then POST http://127.0.0.1:{p.ports['wca']}/console/chat "
                     f"{{\"agent\": \"robot\", \"message\": <contents of {p.task_file}>}}")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ HTTP to the local world
def _http(method: str, url: str, body: dict | None = None, timeout: float = 30.0) -> bytes:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, method=method,
                                     headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _get_json(url: str, timeout: float = 30.0):
    return json.loads(_http("GET", url, timeout=timeout))


# ------------------------------------------------------------------ process control
class Recorder:
    """Append-only record of every process the runner started (``pids.jsonl``)."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()

    def add(self, **row):
        with self.lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"t": time.time(), **row}) + "\n")


def stop_group(proc: subprocess.Popen, grace_s: float = STOP_GRACE_S) -> str:
    """Signal the process group this runner created for ``proc`` (its pid is the group id).

    The group is signalled only while ``proc`` has not been reaped, so the id cannot belong to an
    unrelated process. Returns what was done.
    """
    if proc.poll() is not None:
        return f"already exited rc={proc.returncode}"
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=grace_s)
        return f"SIGTERM rc={proc.returncode}"
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        return "SIGKILL"


class Episode:
    def __init__(self, cfg: Config, p: EpisodePlan, recorder: Recorder, stop: threading.Event):
        self.cfg, self.p, self.recorder, self.stop = cfg, p, recorder, stop
        self.meta: dict = {"seed": p.seed, "task": cfg.task.name, "device": p.device, "slot": p.slot,
                           "ports": p.ports, "startStateDir": str(cfg.task.start_state_dir or "") or None,
                           "backend": cfg.machine.backend, "wallLimitS": cfg.machine.wall_minutes * 60}
        self.wca = f"http://127.0.0.1:{p.ports['wca']}"

    def save_meta(self, **update):
        self.meta.update(update)
        (self.p.dir / "episode.json").write_text(json.dumps(self.meta, indent=2))

    def env(self) -> dict:
        env = dict(os.environ)
        env.update(self.p.env_set)
        for key in self.p.env_unset:
            env.pop(key, None)
        return env

    def spawn(self, cmd: list[str], log_name: str, role: str) -> subprocess.Popen:
        log = (self.p.dir / "logs" / log_name).open("w", encoding="utf-8")
        proc = subprocess.Popen(cmd, cwd=self.cfg.steward_dir, env=self.env(), stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        log.close()
        self.recorder.add(episode=self.p.index, role=role, pid=proc.pid, pgid=proc.pid, cmd=cmd[:4])
        return proc

    def scene(self) -> dict:
        return _get_json(f"{self.wca}/wca/scene", timeout=60)

    def frames(self, tag: str, lookat: list[float] | None):
        out = self.p.dir / "frames"
        out.mkdir(exist_ok=True)
        views = {"overview": {"camera": "overview"}}
        if lookat:
            views["kit"] = {"camera": "free", "lookat": ",".join(f"{v:.3f}" for v in lookat), "distance": "0.9",
                            "azimuth": "90", "elevation": "-55"}
        for name, query in views.items():
            query = {**query, "width": "640", "height": "480", "format": "jpeg"}
            try:
                data = _http("GET", f"{self.wca}/wca/render?{urllib.parse.urlencode(query)}", timeout=120)
                (out / f"{tag}-{name}.jpg").write_bytes(data)
            except (urllib.error.URLError, OSError) as error:
                self.log(f"frame {tag}-{name} failed: {error}")

    def log(self, text: str):
        with (self.p.dir / "logs" / "runner.log").open("a", encoding="utf-8") as handle:
            handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {text}\n")

    def run(self):
        p = self.p
        (p.dir / "logs").mkdir(parents=True, exist_ok=True)
        self.save_meta(status="starting", startedAt=time.time())
        if self.cfg.run.isolate_harness:
            template = self.cfg.run.robot_harness_template
            if not (template / "profiles" / "ego").is_dir():
                self.save_meta(status="error", error=f"robot harness template has no profiles/ego: {template}")
                return
            copy_harness(template, p.harness_home)
        if not self._build():
            return
        if self.stop.is_set():
            self.save_meta(status="error", error="interrupted before the world started")
            return
        up = self.spawn(p.up_cmd, "up.log", "up")
        self.save_meta(status="starting-world", upPid=up.pid)
        try:
            self._drive(up)
        finally:
            how = stop_group(up)
            self.save_meta(stop=how, finishedAt=time.time())

    def _build(self) -> bool:
        """Build the toy assets; False (with the reason saved) on failure, timeout or stop."""
        build = self.spawn(self.p.build_cmd, "build.log", "build")
        deadline = time.monotonic() + self.cfg.run.build_timeout_s
        while (rc := build.poll()) is None:
            if self.stop.is_set():
                self.save_meta(status="error", error="interrupted during the asset build",
                               buildStop=stop_group(build, 10))
                return False
            if time.monotonic() > deadline:
                self.save_meta(status="error", error="asset build timed out", buildStop=stop_group(build, 10))
                return False
            self.stop.wait(BUILD_POLL_S)
        if rc != 0:
            self.save_meta(status="error", error=f"asset build exited with {rc}; see logs/build.log")
            return False
        return True

    def _drive(self, up: subprocess.Popen):
        cfg, p = self.cfg, self.p
        deadline = time.monotonic() + cfg.run.startup_timeout_s
        while True:
            if up.poll() is not None:
                self.save_meta(status="error", error=f"steward up exited with {up.returncode} during startup")
                return
            try:
                chat = _get_json(f"{self.wca}/console/chat", timeout=10)
                robot = chat.get("agents", {}).get("robot", {})
                if robot.get("available"):
                    break
            except (urllib.error.URLError, OSError, ValueError):
                pass
            if self.stop.is_set():
                self.save_meta(status="error", error="interrupted during world startup")
                return
            if time.monotonic() > deadline:
                self.save_meta(status="error", error="world or robot chat not ready before startup timeout")
                return
            self.stop.wait(STARTUP_POLL_S)
        initial = self.scene()
        (p.dir / "world_initial.json").write_text(json.dumps(initial))
        lookat = None
        try:
            from .scene import World, derive_kits, load_goal, load_spec, select_kit

            world = World(initial)
            kits = derive_kits(load_goal(cfg.repo_root)[0], load_spec(cfg.repo_root)[0], world)
            kit = select_kit(kits, world, cfg.task.name, cfg.task.kit_body)
            lookat = list(world.pose(kit.body)["p"])
        except (ValueError, KeyError, LookupError) as error:
            self.log(f"cannot locate the kit for frames: {error}")
        self.frames("start", lookat)
        task_text = p.task_file.read_text(encoding="utf-8").strip()
        try:
            accepted = json.loads(_http("POST", f"{self.wca}/console/chat", {"agent": "robot", "message": task_text})
                                  or b"{}")
        except urllib.error.HTTPError as error:
            detail = error.read()[:500].decode("utf-8", "replace")
            self.save_meta(status="error", endReason="task-rejected",
                           error=f"POST /console/chat returned HTTP {error.code}: {detail}")
            return
        # The server marks the robot busy before it answers the POST, so once the task is accepted
        # busy=false means this turn is over. Transcript entries after the task entry belong to it.
        since = int(accepted.get("id", 0)) if isinstance(accepted, dict) else 0
        started = time.monotonic()
        self.save_meta(status="running", taskSentAt=time.time())
        next_sample = next_frame = 0.0
        end = "unknown"
        end_messages: list[dict] = []
        samples = (p.dir / "truth_samples.jsonl").open("a", encoding="utf-8")
        try:
            while True:
                elapsed = time.monotonic() - started
                if self.stop.is_set():
                    end = "interrupted"
                    break
                if up.poll() is not None:
                    end = f"world-exited rc={up.returncode}"
                    break
                if elapsed >= cfg.machine.wall_minutes * 60:
                    end = "wall-limit"
                    break
                if elapsed >= next_sample:
                    next_sample = elapsed + cfg.run.sample_every_s
                    try:
                        state = self.scene()
                        samples.write(json.dumps({"elapsed_s": round(elapsed, 3), "state": state}) + "\n")
                        samples.flush()
                        (p.dir / "world_latest.json").write_text(json.dumps(state))
                    except (urllib.error.URLError, OSError, ValueError) as error:
                        self.log(f"sample failed: {error}")
                if cfg.run.frame_every_min > 0 and elapsed >= next_frame:
                    next_frame = elapsed + cfg.run.frame_every_min * 60
                    self.frames(f"t{int(elapsed):06d}", lookat)
                try:
                    chat = _get_json(f"{self.wca}/console/chat?since={since}", timeout=10)
                    for entry in chat.get("messages", []):
                        since = max(since, int(entry.get("id", since)))
                        if entry.get("agent") == "robot" and entry.get("role") in ("error", "system"):
                            end_messages.append({"role": entry["role"], "text": str(entry.get("text", ""))[:1000],
                                                 "t": entry.get("t")})
                    if not chat.get("agents", {}).get("robot", {}).get("busy"):
                        end = "turn-ended"
                        break
                except (urllib.error.URLError, OSError, ValueError) as error:
                    self.log(f"chat poll failed: {error}")
                self.stop.wait(POLL_S)
        finally:
            samples.close()
        wall = time.monotonic() - started
        try:
            (p.dir / "world_state.json").write_text(json.dumps(self.scene()))
            self.frames("final", lookat)
        except (urllib.error.URLError, OSError, ValueError) as error:
            self.log(f"final snapshot failed: {error}; world_latest.json is the last sample")
        self.save_meta(status="finished", endReason=end, endMessages=end_messages[-END_MESSAGES_KEPT:],
                       wallS=round(wall, 1))


def execute(cfg: Config, plans: list[EpisodePlan], round_dir: Path) -> None:
    """Run ``plans``; one thread per slot. Ctrl-C or SIGTERM ends every episode and stops its processes.

    Raises :class:`RoundExistsError` before starting anything when ``round_dir`` holds episode data.
    """
    check_round_dir(round_dir)
    round_dir.mkdir(parents=True, exist_ok=True)
    (round_dir / "plan.json").write_text(json.dumps([p.to_dict() for p in plans], indent=2))
    recorder = Recorder(round_dir / "pids.jsonl")
    stop = threading.Event()
    by_slot: dict[int, list[EpisodePlan]] = {}
    for p in plans:
        by_slot.setdefault(p.slot, []).append(p)

    def worker(items):
        for p in items:
            if stop.is_set():
                return
            p.dir.mkdir(parents=True, exist_ok=True)
            episode = Episode(cfg, p, recorder, stop)
            try:
                episode.run()
            except Exception as error:  # one failed episode must not stop the other slots
                episode.save_meta(status="error", error=f"{type(error).__name__}: {error}")

    previous = signal.signal(signal.SIGTERM, lambda *_: stop.set())
    threads = [threading.Thread(target=worker, args=(items,), name=f"slot-{slot}") for slot, items in by_slot.items()]
    for t in threads:
        t.start()
    try:
        while any(t.is_alive() for t in threads):
            for t in threads:
                t.join(timeout=1.0)
    except KeyboardInterrupt:
        stop.set()
        for t in threads:
            t.join()
    finally:
        signal.signal(signal.SIGTERM, previous)
