"""Steward command line.

    steward build-kitchen            generate, validate and publish the Phase 0 kitchen
    steward world                    run WSim + Observer + WCA + robot ports
    steward ego                      run one robot's Ego Runtime + Ego API
    steward up                       run the world, and the robot's Ego Runtime in its own process
    steward demo                     scripted "cup to table" mission, checked against World State
    steward shuttle                  carry objects around table → cabinet → sideboard, step by step
    steward chores                   sort tableware, stack it on the sideboard, swap drawer contents; repeat
    steward services                 list the remote perception services and check that each is reachable
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from steward.phases import LATEST_PHASE, PHASES

STEWARD_DIR = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = STEWARD_DIR / "var" / "steward"
SEGMENTATION_CONFIG = STEWARD_DIR / "config" / "segmentation.toml"


def _robot_model() -> dict:
    from steward.assets.generators import steward_bot

    return steward_bot().to_dict()


def _wait_forever(stoppers: list, pump=None) -> None:
    """Block until SIGINT/SIGTERM; the main thread serves render requests through ``pump``."""
    done = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: done.set())
    signal.signal(signal.SIGTERM, lambda *_: done.set())
    while not done.is_set():
        if pump is not None:
            pump(0.05)
        else:
            done.wait(0.5)
    for stop in stoppers:
        stop()


def cmd_build_kitchen(args) -> int:
    from steward.authoring.service import AuthoringService, build_default_kitchen

    out = build_default_kitchen(AuthoringService(args.root))
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


def _host(args, run_dir: Path, start: bool = True):
    from steward.world.server import StewardHost

    ports = {"w:robot-1": args.robot_port} if getattr(args, "robot_port", 0) else {}
    host = StewardHost(args.root, scenario=args.scenario, speed=args.speed, seed=args.seed, run_dir=run_dir,
                       wca_port=args.wca_port, robot_ports=ports, chat=_chat_config(args),
                       phase=args.phase, bind_wait_s=getattr(args, "bind_wait", 0.0),
                       pause_on_perception_fault=getattr(args, "pause_on_perception_fault", False),
                       freeze_idle=args.freeze_idle)
    return host.start() if start else host


def _chat_config(args):
    """Operator chat for the console: the steward-local DSH launcher, the World Harness home and
    the robot's Ego Harness home. ``steward up`` fills in its own Ego API address after start."""
    from steward.world.chat import ChatConfig

    world, robot = getattr(args, "world_harness", None), getattr(args, "robot_harness", None)
    if not world and not robot:
        return None
    _dsh_bin = STEWARD_DIR / "dsh" / "node_modules" / ".bin"
    # Windows cannot exec the POSIX sh launcher; dsh.CMD is the npm-generated cmd wrapper
    _dsh = _dsh_bin / ("dsh.CMD" if os.name == "nt" and (_dsh_bin / "dsh.CMD").exists() else "dsh")
    return ChatConfig(dsh=_dsh, cwd=STEWARD_DIR,
                      env_file=STEWARD_DIR / ".env", world_home=Path(world) if world else None,
                      robot_home=Path(robot) if robot else None, ego_api=getattr(args, "ego_api", None))


def cmd_world(args) -> int:
    run_dir = Path(args.root) / "runs" / time.strftime("%Y%m%d-%H%M%S")
    host = _host(args, run_dir)
    if args.ego_api:
        host.attach_ego_api("w:robot-1", args.ego_api)
    print(json.dumps({"console": f"{host.wca.url}/console", "wca": host.wca.url, "robots": {k: v.url for k, v in host.robot_servers.items()},
                      "runDir": str(run_dir)}), flush=True)
    _wait_forever([host.stop], host.pump)
    return 0


def _segmenter(config: str, root: str):
    """The robot-side segmenter from ``config`` (``''``: none), caching results under ``<root>/ego/segmentation``."""
    if not config:
        return None
    from steward.ego.segmenter import Segmenter

    return Segmenter.from_files(Path(config), cache_dir=Path(root) / "ego" / "segmentation")


def _start_ego(robot_url: str, api_port: int, out_dir: Path, bind_wait_s: float = 0.0, segmenter=None):
    from steward.ego.runtime import EgoRuntime, RobotLink
    from steward.ego.server import ego_api

    runtime = EgoRuntime(RobotLink(robot_url), _robot_model(), out_dir=out_dir, segmenter=segmenter)
    runtime.start()
    api = ego_api(runtime, port=api_port, bind_wait_s=bind_wait_s).start()
    return runtime, api


def cmd_ego(args) -> int:
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.root) / "ego" / "observations"
    runtime, api = _start_ego(args.robot_url, args.api_port, out_dir, args.bind_wait,
                              _segmenter(args.segmentation_config, args.root))
    print(json.dumps({"egoApi": api.url, "robot": args.robot_url}), flush=True)
    if args.parent_pid:
        _exit_with_parent(args.parent_pid)
    _wait_forever([api.stop, runtime.stop])
    return 0


def _exit_with_parent(parent_pid: int) -> None:
    """Stop this process (as SIGTERM would) once the process that launched it is gone.

    On Windows, ``start_new_session=True`` breaks ``os.getppid()`` identity with the
    launcher, so the old ``getppid() == parent_pid`` watch exited immediately (status 15).
    Poll whether ``parent_pid`` still exists instead.
    """
    def parent_alive() -> bool:
        if os.name == "nt":
            import ctypes
            handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(parent_pid))
            if not handle:
                return False
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        try:
            os.kill(parent_pid, 0)
        except OSError:
            return False
        return True

    def watch() -> None:
        while parent_alive():
            time.sleep(1.0)
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=watch, name="parent-watch", daemon=True).start()


class _EgoProcess:
    """The robot's Ego Runtime, Ego API and Ego-State view in a process of its own, as on a robot:
    nothing of it shares memory or OpenGL contexts with the world. Its output is passed through."""

    def __init__(self, args, robot_url: str, out_dir: Path):
        cmd = [sys.executable, "-m", "steward.cli", "--root", str(args.root), "ego", "--robot-url", robot_url,
               "--api-port", str(args.api_port), "--out-dir", str(out_dir), "--parent-pid", str(os.getpid()),
               "--bind-wait", str(args.bind_wait), "--segmentation-config", args.segmentation_config]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, encoding="utf-8",
                                     errors="replace", start_new_session=True)  # NDJSON is UTF-8; a locale
        # decode error here would silently kill the reader (and backpressure the child) on Windows
        first = self.proc.stdout.readline()
        if not first:
            raise SystemExit(f"the Ego Runtime process exited with status {self.proc.wait()} before starting")
        self.url = json.loads(first)["egoApi"]
        self._stopping = False
        self._reported = False
        threading.Thread(target=self._relay, name="ego-output", daemon=True).start()

    def _relay(self) -> None:
        for line in self.proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()

    def check(self) -> None:
        """Report once on stderr when the Ego Runtime process ended without being stopped."""
        if not self._stopping and not self._reported and self.proc.poll() is not None:
            self._reported = True
            print(f"steward up: the Ego Runtime process exited with status {self.proc.returncode}; "
                  "the robot no longer perceives or acts", file=sys.stderr, flush=True)

    def stop(self) -> None:
        self._stopping = True
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()


def cmd_up(args) -> int:
    run_dir = Path(args.root) / "runs" / time.strftime("%Y%m%d-%H%M%S")
    # The robot port serves from construction; the console's WCA opens only once the Ego API is
    # attached, so the console never asks for an Ego-State view no robot provides yet.
    host = _host(args, run_dir, start=False)
    robot_url = host.robot_servers["w:robot-1"].url
    try:
        ego = _EgoProcess(args, robot_url, run_dir / "ego")
    except BaseException:
        host.stop()
        raise
    host.attach_ego_api("w:robot-1", ego.url)
    if host.chat.config is not None:
        host.chat.config.ego_api = ego.url
    host.start()
    print(json.dumps({"console": f"{host.wca.url}/console", "wca": host.wca.url, "robot": robot_url, "egoApi": ego.url, "runDir": str(run_dir),
                      "task": host.scenario["task"]}, ensure_ascii=False), flush=True)

    def pump(timeout: float) -> None:
        host.pump(timeout)
        ego.check()

    _wait_forever([ego.stop, host.stop], pump)
    return 0


def cmd_demo(args) -> int:
    from steward.authoring.service import AuthoringService, build_default_kitchen
    from steward.mission import MissionFailed, cup_to_table

    if "phase0-kitchen" not in AuthoringService(args.root).list_scenarios():
        build_default_kitchen(AuthoringService(args.root))
    run_dir = Path(args.root) / "runs" / ("demo-" + time.strftime("%Y%m%d-%H%M%S"))
    host = _host(args, run_dir)
    runtime, api = _start_ego(host.robot_servers["w:robot-1"].url, 0, run_dir / "ego",
                              segmenter=_segmenter(args.segmentation_config, args.root))
    t0 = time.monotonic()
    outcome: dict = {}

    def mission() -> None:
        try:
            outcome["result"] = cup_to_table(api.url)
        except MissionFailed as exc:
            outcome["error"] = str(exc)

    worker = threading.Thread(target=mission, name="mission", daemon=True)
    worker.start()
    while worker.is_alive():
        host.pump(0.05)
    ok_mission = "result" in outcome
    result = outcome.get("result") or {"error": outcome.get("error")}
    time.sleep(1.0 / max(args.speed, 0.1))
    ws = host.sim.world_state()
    on_table = any(r["type"] == "supported-by" and r["a"]["entity"] == "w:cup-1" and r["b"]["entity"] == "w:table-1"
                   for r in ws["relations"])
    (run_dir / "overview-final.png").write_bytes(host.sim.render("overview", 800, 600))
    report = {
        "missionCompleted": ok_mission,
        "groundTruthCupOnTable": on_table,
        "wallSeconds": round(time.monotonic() - t0, 1),
        "simSeconds": round(host.sim.time, 1),
        "mission": result.get("log", result),
        "observer": host.observer.report(),
        "egoObjects": [(o["id"], o["category"], o["position"], o["uncertaintyM"], o["existence"])
                       for o in runtime.summary()["objects"]],
        "observerMatches": host.observer.latest.get("w:robot-1", {}).get("matches"),
        "events": [e for e in host.sim.events if e["type"].startswith("grasp/")],
        "runDir": str(run_dir),
    }
    api.stop()
    runtime.stop()
    host.stop()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if ok_mission and on_table else 1


def _ground_truth_locations(scene: dict, supports: list[str], objects: list[str]) -> dict[str, str | None]:
    """Support category under each carried object category, from World State relations."""
    category = {e["id"]: e["category"]["value"] for e in scene["entities"]}
    out: dict[str, str | None] = {cat: None for cat in objects}
    for rel in scene["relations"]:
        a, b = category.get(rel["a"]["entity"]), category.get(rel["b"]["entity"])
        if rel["type"] == "supported-by" and a in out and b in supports:
            out[a] = b
    return out


SUPPORT_NAMES = {"table": "餐桌", "cabinet": "柜子", "sideboard": "餐边柜", "counter": "操作台"}
OBJECT_NAMES = {"cup": "杯子", "bowl": "碗", "plate": "盘子"}


def cmd_shuttle(args) -> int:
    """Shuttle objects between table, cabinet and sideboard, checking each move against World State."""
    from steward.mission import MissionFailed, shuttle
    from steward.net import call

    supports, objects = ["table", "cabinet", "sideboard"], ["cup", "bowl"]
    host = runtime = api = None
    if args.api:
        api_url = args.api.rstrip("/")
        def world_scene():
            return call("GET", f"{args.wca.rstrip('/')}/wca/scene")
    else:
        from steward.authoring.service import AuthoringService, build_default_kitchen

        if args.scenario not in AuthoringService(args.root).list_scenarios():
            build_default_kitchen(AuthoringService(args.root))
        run_dir = Path(args.root) / "runs" / ("shuttle-" + time.strftime("%Y%m%d-%H%M%S"))
        host = _host(args, run_dir)
        runtime, api = _start_ego(host.robot_servers["w:robot-1"].url, 0, run_dir / "ego",
                                  segmenter=_segmenter(args.segmentation_config, args.root))
        api_url = api.url
        def world_scene():
            return host.sim.world_state()
        print(json.dumps({"console": f"{host.wca.url}/console"}), flush=True)
    t0 = time.monotonic()
    results: list[bool] = []

    def on_step(index: int, move: dict) -> None:
        time.sleep(0.5)
        truth = _ground_truth_locations(world_scene(), supports, objects)
        ok = truth.get(move["object"]) == move["to"]
        results.append(ok)
        if args.steps > 0:
            filled = int(30 * (index + 1) / args.steps)
            progress = f"[{'#' * filled}{'-' * (30 - filled)}] {index + 1}/{args.steps}"
        else:
            progress = f"[第 {index + 1} 步，第 {index // 3 + 1} 轮]"
        print(f"{progress}  "
              f"{OBJECT_NAMES.get(move['object'], move['object'])}：{SUPPORT_NAMES.get(move['from'], '其他位置')} → "
              f"{SUPPORT_NAMES[move['to']]}  World State {'确认' if ok else '不一致'} "
              f"{ {OBJECT_NAMES.get(k, k): SUPPORT_NAMES.get(v, v) for k, v in truth.items()} }  "
              f"({time.monotonic() - t0:.0f}s)", flush=True)

    def on_failure(index: int, message: str) -> None:
        print(f"[第 {index + 1} 步失败，重新观察后继续]  {message}  ({time.monotonic() - t0:.0f}s)", flush=True)

    def on_wait(owner: str) -> None:
        print(f"[暂停：机器人由 {owner} 接管，交还控制后继续]  ({time.monotonic() - t0:.0f}s)", flush=True)

    outcome: dict = {}

    def mission() -> None:
        try:
            outcome["result"] = shuttle(api_url, args.steps, tuple(supports), tuple(objects), on_step=on_step,
                                        on_failure=on_failure, on_wait=on_wait)
        except MissionFailed as exc:
            outcome["error"] = str(exc)

    worker = threading.Thread(target=mission, name="shuttle", daemon=True)
    worker.start()
    interrupted = False
    try:
        while worker.is_alive():
            if host is not None:
                host.pump(0.05)
            else:
                worker.join(0.2)
    except KeyboardInterrupt:
        interrupted = True
        if host is None:
            call("POST", f"{api_url}/skill/stop", {})
    finished = "result" in outcome or (interrupted and "error" not in outcome)
    ok = finished and (args.steps <= 0 or len(results) == args.steps) and all(results)
    print(json.dumps({"completedSteps": len(results), "allConfirmedByWorldState": ok,
                      "error": outcome.get("error")}, ensure_ascii=False), flush=True)
    if host is not None:
        api.stop()
        runtime.stop()
        host.stop()
    return 0 if ok else 1


def cmd_chores(args) -> int:
    """Repeat the chores rounds, checking every completed phase against World State."""
    from steward.chores import PHASE_NAMES, chores
    from steward.mission import MissionFailed
    from steward.net import call
    from steward.world.chores_check import check_sideboard, check_swap, check_table, drawer_contents

    host = runtime = api = None
    if args.api:
        api_url = args.api.rstrip("/")
        wca = args.wca.rstrip("/")

        def truth() -> tuple[dict, dict]:
            boxes = {g["id"]: g for g in call("GET", f"{wca}/console/state")["groundTruth"]}
            return call("GET", f"{wca}/wca/scene"), boxes
    else:
        from steward.authoring.service import AuthoringService, build_default_kitchen
        from steward.world.console import console_state

        if args.scenario not in AuthoringService(args.root).list_scenarios():
            build_default_kitchen(AuthoringService(args.root))
        run_dir = Path(args.root) / "runs" / ("chores-" + time.strftime("%Y%m%d-%H%M%S"))
        host = _host(args, run_dir)
        runtime, api = _start_ego(host.robot_servers["w:robot-1"].url, 0, run_dir / "ego",
                                  segmenter=_segmenter(args.segmentation_config, args.root))
        api_url = api.url

        def truth() -> tuple[dict, dict]:
            return host.sim.world_state(), {g["id"]: g for g in console_state(host)["groundTruth"]}
        print(json.dumps({"console": f"{host.wca.url}/console"}), flush=True)
    t0 = time.monotonic()
    results: list[bool] = []
    drawers_before = {"contents": drawer_contents(truth()[0])}

    def stamp() -> str:
        return f"({time.monotonic() - t0:.0f}s)"

    def on_step(record: dict) -> None:
        print(f"    · {record['text']}  {stamp()}", flush=True)

    def on_phase(index: int, phase: str) -> None:
        time.sleep(0.5)
        scene, boxes = truth()
        if phase == "table":
            ok, detail = check_table(scene, boxes)
        elif phase == "sideboard":
            ok, detail = check_sideboard(scene, boxes)
        else:
            ok, detail = check_swap(drawers_before["contents"], scene)
            drawers_before["contents"] = drawer_contents(scene)
        results.append(ok)
        print(f"[第 {index + 1} 轮] {PHASE_NAMES[phase]}完成  World State {'确认' if ok else '不一致'}：{detail}  {stamp()}",
              flush=True)

    def on_failure(index: int, phase: str, message: str) -> None:
        print(f"[第 {index + 1} 轮] {PHASE_NAMES[phase]}失败，放下手中物品、重新观察后继续：{message}  {stamp()}", flush=True)
        drawers_before["contents"] = drawer_contents(truth()[0])

    def on_wait(owner: str) -> None:
        print(f"[暂停：机器人由 {owner} 接管，交还控制后继续]  {stamp()}", flush=True)

    outcome: dict = {}

    def mission() -> None:
        try:
            outcome["result"] = chores(api_url, args.rounds, on_step=on_step, on_phase=on_phase, on_failure=on_failure,
                                       on_wait=on_wait)
        except MissionFailed as exc:
            outcome["error"] = str(exc)

    worker = threading.Thread(target=mission, name="chores", daemon=True)
    worker.start()
    interrupted = False
    try:
        while worker.is_alive():
            if host is not None:
                host.pump(0.05)
            else:
                worker.join(0.2)
    except KeyboardInterrupt:
        interrupted = True
        if host is None:
            call("POST", f"{api_url}/skill/stop", {})
    finished = "result" in outcome or (interrupted and "error" not in outcome)
    ok = finished and (args.rounds <= 0 or len(results) == 3 * args.rounds) and all(results)
    print(json.dumps({"completedPhases": len(results), "allConfirmedByWorldState": ok, "error": outcome.get("error")},
                     ensure_ascii=False), flush=True)
    if host is not None:
        api.stop()
        runtime.stop()
        host.stop()
    return 0 if ok else 1


def cmd_services(args) -> int:
    from steward import perception_services as ps

    services = ps.load(Path(args.config))
    healthy = True
    for svc in services.values():
        if args.no_check:
            print(json.dumps({"service": svc.name, "title": svc.title, "urls": list(svc.urls), "docs": svc.docs},
                             ensure_ascii=False))
            continue
        health = ps.check(svc)
        healthy &= health.ok
        print(json.dumps({"service": svc.name, "title": svc.title, "ok": health.ok, "url": health.url,
                          "detail": health.detail}, ensure_ascii=False), flush=True)
    return 0 if healthy else 1


def _speed(text: str) -> float:
    return math.inf if text == "max" else float(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="steward")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="Steward data root (assets, scenes, runs)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build-kitchen").set_defaults(fn=cmd_build_kitchen)
    defaults = {"demo": "phase0-kitchen", "shuttle": "phase0-shuttle"}
    for name, fn in (("world", cmd_world), ("up", cmd_up), ("demo", cmd_demo), ("shuttle", cmd_shuttle),
                     ("chores", cmd_chores)):
        p = sub.add_parser(name)
        p.add_argument("--scenario", default=defaults.get(name, "phase0-chores"))
        in_process = name in ("demo", "shuttle", "chores")
        p.add_argument("--speed", type=_speed, nargs="?", const=math.inf, default=4.0 if in_process else 1.0,
                       help="simulation speed as a multiple of real time; without a value (or 'max'), as fast as possible")
        p.add_argument("--freeze-idle", action="store_true",
                       help="no world time passes while no robot acts, e.g. while the agents think")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--phase", choices=sorted(PHASES), default=LATEST_PHASE,
                       help="phase level of the replaceable modules (switchable at run time from the console)")
        p.add_argument("--wca-port", type=int, default=0 if in_process else 7400)
        p.add_argument("--robot-port", type=int, default=0 if in_process else 7410)
        if not in_process:
            p.add_argument("--bind-wait", type=float, default=10.0,
                           help="seconds to wait for a fixed port still held by a service that just stopped")
            p.add_argument("--pause-on-perception-fault", action=argparse.BooleanOptionalAction, default=True,
                           help="pause the world when the Observer finds a gross perception fault (phantom object, "
                                "robot inside an object) until the operator acknowledges the alert")
            p.add_argument("--world-harness", default=str(STEWARD_DIR / "var" / "dsh" / "world"),
                           help="DSH home of the World Harness used by the console chat; '' disables it")
            p.add_argument("--robot-harness", default=str(STEWARD_DIR / "var" / "dsh" / "robot-1"),
                           help="DSH home of the robot's Ego Harness used by the console chat; '' disables it")
        if name == "world":
            p.add_argument("--ego-api", default="http://127.0.0.1:7420",
                           help="Ego API of the robot, for robot tasks from the console chat")
        if name == "up":
            p.add_argument("--api-port", type=int, default=7420)
        if name != "world":
            p.add_argument("--segmentation-config", default=str(SEGMENTATION_CONFIG),
                           help="robot-side segmenter settings, used at phase levels without world masks; '' disables it")
        if name == "chores":
            p.add_argument("--rounds", type=int, default=1, help="number of rounds; 0 repeats until Ctrl-C")
            p.add_argument("--api", help="Ego API of a running `steward up`; omit to run in-process")
            p.add_argument("--wca", default="http://127.0.0.1:7400", help="WCA of the running world (with --api)")
        if name == "shuttle":
            p.add_argument("--steps", type=int, default=6, help="number of moves; 0 repeats until Ctrl-C")
            p.add_argument("--api", help="Ego API of a running `steward up`; omit to run in-process")
            p.add_argument("--wca", default="http://127.0.0.1:7400", help="WCA of the running world (with --api)")
        p.set_defaults(fn=fn)
    p = sub.add_parser("ego")
    p.add_argument("--robot-url", default="http://127.0.0.1:7410")
    p.add_argument("--api-port", type=int, default=7420)
    p.add_argument("--out-dir", help="where the runtime writes its observations (default: <root>/ego/observations)")
    p.add_argument("--parent-pid", type=int, help="exit when this process is gone (set by `steward up`)")
    p.add_argument("--bind-wait", type=float, default=10.0,
                   help="seconds to wait for a fixed port still held by a service that just stopped")
    p.add_argument("--segmentation-config", default=str(SEGMENTATION_CONFIG),
                   help="robot-side segmenter settings, used at phase levels without world masks; '' disables it")
    p.set_defaults(fn=cmd_ego)
    p = sub.add_parser("services")
    p.add_argument("--config", default=str(STEWARD_DIR / "config" / "perception-services.toml"),
                   help="perception services file")
    p.add_argument("--no-check", action="store_true", help="only list the services, without contacting them")
    p.set_defaults(fn=cmd_services)
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
