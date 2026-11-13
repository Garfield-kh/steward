"""Measure chores stability under a pinned phase level (Iteration A baseline).

Records per-phase World State checks and Observer metrics so later perception/association
changes can be compared against the same seeds and settings.

    .venv/bin/python scripts/chores_baseline.py --phase phase1 --seeds 5 --speed 10
    .venv/bin/python scripts/chores_baseline.py --phase phase0 --seeds 5 --speed 10  # control

Writes ``<root>/baselines/chores-<phase>-<stamp>/`` with ``config.json``, per-seed JSON, and
``summary.json``. Each seed also keeps residual-failure diagnostics:

* ``chores.log.json`` / ``chores.log.jsonl`` — doing / skip / pick-receipt / place-receipt / clear
* ``supports-<phase>.json`` — World + Ego ``supported-by`` slice at phase end (bowls, wine, mug, …)
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import threading
import time
from pathlib import Path

from steward.assets.generators import steward_bot
from steward.authoring.service import AuthoringService, build_default_kitchen
from steward.chores import PHASE_NAMES, PHASES, Chores, _explore
from steward.ego.runtime import EgoRuntime, RobotLink
from steward.ego.server import ego_api
from steward.mission import MissionFailed
from steward.net import HttpError
from steward.phases import PHASES as PHASE_LEVELS, phase_modules
from steward.net import call
from steward.world.chores_check import check_sideboard, check_swap, check_table, drawer_contents
from steward.world.chores_diagnostics import (
    append_jsonl,
    phase_support_diagnostics,
    write_json,
)
from steward.world.chores_metrics import OBSERVER_KEYS, observer_brief, summarize, visible_recall, closed_drawer_object_count
from steward.world.console import console_state
from steward.world.server import StewardHost


ROBOT = "w:robot-1"
SEGMENTATION_CONFIG = Path(__file__).resolve().parents[1] / "config" / "segmentation.toml"


def _segmenter_for_phase(phase: str, root: Path):
    """Attach the robot-side SAM segmenter when the phase sends no world masks (``phase1-sam``)."""
    if phase_modules(phase).get("masks", True):
        return None
    from steward.ego.segmenter import Segmenter

    return Segmenter.from_files(SEGMENTATION_CONFIG, cache_dir=root / "ego" / "segmentation")


def bar(done: int, total: int, width: int = 30) -> str:
    filled = int(width * done / total)
    return f"[{'#' * filled}{'-' * (width - filled)}] {done}/{total}"


def perception_snapshot(host: StewardHost) -> dict:
    record = host.observer.latest.get(ROBOT)
    scene = host.sim.world_state()
    brief = observer_brief(record) or {}
    brief["visibleRecall"] = visible_recall(record, scene)
    brief["closedDrawerObjects"] = closed_drawer_object_count(scene)
    brief["leaks"] = 1 if record and "leak" in record else 0
    return brief


def truth(host: StewardHost) -> tuple[dict, dict]:
    return host.sim.world_state(), {g["id"]: g for g in console_state(host)["groundTruth"]}


def run_seed(root: Path, out_dir: Path, seed: int, phase: str, speed: float, timeout_s: float,
             only_phases: list[str] | None = None) -> dict:
    run_dir = out_dir / f"seed-{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    planned = list(only_phases) if only_phases else list(PHASES)
    host = StewardHost(root, scenario="phase0-chores", speed=speed, seed=seed, run_dir=run_dir,
                       phase=phase).start()
    runtime = EgoRuntime(RobotLink(host.robot_servers[ROBOT].url), steward_bot().to_dict(),
                         segmenter=_segmenter_for_phase(phase, root))
    runtime.start()
    api = ego_api(runtime).start()
    stop = threading.Event()

    def pump() -> None:
        while not stop.is_set():
            host.pump(0.05)

    worker = threading.Thread(target=pump, name=f"pump-{seed}", daemon=True)
    worker.start()
    t0 = time.monotonic()
    phases: list[dict] = []
    error: str | None = None
    after_explore: dict | None = None
    report = None
    log_jsonl = run_dir / "chores.log.jsonl"
    runner: Chores | None = None
    try:
        def on_step(record: dict) -> None:
            append_jsonl(log_jsonl, record)

        runner = Chores(api.url, timeout_s=timeout_s, on_step=on_step)
        explore = _explore(api.url, timeout_s)
        time.sleep(0.5)
        after_explore = perception_snapshot(host)
        after_explore["exploreState"] = explore.get("state")
        _write_partial(run_dir, seed, phase, after_explore, phases, error, time.monotonic() - t0)
        before_drawers = drawer_contents(truth(host)[0])
        checks = {
            "table": lambda: check_table(*truth(host)),
            "sideboard": lambda: check_sideboard(*truth(host)),
            "drawers": lambda: check_swap(before_drawers, truth(host)[0]),
        }
        actions = {
            "table": runner.arrange_on_table,
            "sideboard": runner.stack_on_sideboard,
            "drawers": runner.swap_drawers,
        }
        for name in planned:
            entry: dict = {"name": name, "label": PHASE_NAMES[name]}
            try:
                actions[name]()
                time.sleep(0.5)
                ok, detail = checks[name]()
                entry.update({"completed": True, "wsOk": ok, "detail": detail,
                              "observer": perception_snapshot(host)})
            except (MissionFailed, HttpError, OSError) as exc:
                entry.update({"completed": False, "wsOk": False, "error": str(exc)[:500],
                              "observer": perception_snapshot(host)})
                entry["supports"] = _record_phase_supports(run_dir, name, host, api.url, entry)
                _flush_chore_log(run_dir, runner)
                phases.append(entry)
                error = f"{name}: {exc}"
                _write_partial(run_dir, seed, phase, after_explore, phases, error, time.monotonic() - t0)
                break
            entry["supports"] = _record_phase_supports(run_dir, name, host, api.url, entry)
            _flush_chore_log(run_dir, runner)
            phases.append(entry)
            _write_partial(run_dir, seed, phase, after_explore, phases, error, time.monotonic() - t0)
            if not entry["wsOk"]:
                error = f"{name} world-state mismatch: {entry['detail']}"
                _write_partial(run_dir, seed, phase, after_explore, phases, error, time.monotonic() - t0)
                break
    finally:
        stop.set()
        worker.join(timeout=2.0)
        report = host.observer.report().get(ROBOT)
        wall_s = round(time.monotonic() - t0, 1)
        if runner is not None:
            _flush_chore_log(run_dir, runner)
        for closer in (api.stop, runtime.stop, host.stop):
            try:
                closer()
            except Exception as exc:  # EGL teardown can fail after RGB-D; keep the measured result
                print(f"warning: shutdown {getattr(closer, '__qualname__', closer)}: {exc}", flush=True)

    ok = error is None and len(phases) == len(planned) and all(p.get("wsOk") for p in phases)
    result = {
        "seed": seed,
        "phase": phase,
        "ok": ok,
        "wallS": wall_s,
        "error": error,
        "onlyPhases": planned,
        "afterExplore": after_explore,
        "phases": phases,
        "observerReport": report,
        "diagnostics": {
            "choresLog": "chores.log.json",
            "choresLogJsonl": "chores.log.jsonl",
            "supports": [f"supports-{p['name']}.json" for p in phases],
        },
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def _ego_summary(api_url: str) -> dict:
    try:
        return call("GET", f"{api_url}/ego/summary")
    except (HttpError, OSError):  # noqa: BLE001 — diagnostics must not fail the seed
        return {}


def _record_phase_supports(run_dir: Path, phase: str, host: StewardHost, api_url: str,
                           entry: dict) -> dict:
    """World + Ego support slice at phase end; also written to ``supports-<phase>.json``."""
    scene, _boxes = truth(host)
    packet = phase_support_diagnostics(
        scene, _ego_summary(api_url),
        ws_ok=entry.get("wsOk"), detail=entry.get("detail") or entry.get("error"),
    )
    write_json(run_dir / f"supports-{phase}.json", packet)
    return packet


def _flush_chore_log(run_dir: Path, runner: Chores) -> None:
    """Full structured chore trail (includes skips and receipts not sent through ``on_step``)."""
    write_json(run_dir / "chores.log.json", runner.log)


def _write_partial(run_dir: Path, seed: int, phase: str, after_explore: dict | None,
                   phases: list[dict], error: str | None, wall_s: float) -> None:
    """Checkpoint mid-seed so a shutdown fault does not erase measured phases."""
    payload = {
        "seed": seed,
        "phase": phase,
        "ok": False,
        "partial": True,
        "wallS": round(wall_s, 1),
        "error": error,
        "afterExplore": after_explore,
        "phases": phases,
    }
    (run_dir / "result.partial.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                                                 encoding="utf-8")


def load_seed_result(out_dir: Path, seed: int) -> dict | None:
    path = out_dir / f"seed-{seed}" / "result.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run_seed_subprocess(script: Path, root: Path, out_dir: Path, seed: int, phase: str,
                        speed: float, timeout_s: float, only_phases: list[str] | None = None) -> dict:
    """Run one seed in a child process so an EGL segfault cannot abort the sweep."""
    import subprocess
    import sys

    cmd = [sys.executable, str(script), "--one-seed", str(seed), "--phase", phase,
           "--speed", str(speed), "--timeout-s", str(timeout_s), "--root", str(root), "--out", str(out_dir)]
    if only_phases:
        cmd += ["--only-phases", *only_phases]
    proc = subprocess.run(cmd, check=False)
    result = load_seed_result(out_dir, seed)
    if result is not None:
        return result
    return {
        "seed": seed,
        "phase": phase,
        "ok": False,
        "wallS": None,
        "error": f"seed subprocess exited {proc.returncode} without result.json",
        "afterExplore": None,
        "phases": [],
        "observerReport": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=sorted(PHASE_LEVELS), default="phase1",
                        help="pinned phase level (Iteration A uses phase1)")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--first", type=int, default=0, help="first seed index")
    parser.add_argument("--one-seed", type=int, default=None,
                        help="run only this seed in-process (used by the parent sweep)")
    parser.add_argument("--speed", type=float, default=10.0)
    parser.add_argument("--timeout-s", type=float, default=240.0, help="per-skill Ego API wait")
    parser.add_argument("--root", type=Path, default=None,
                        help="Steward data root; default: a fresh temp kitchen under /tmp")
    parser.add_argument("--out", type=Path, default=None,
                        help="baseline output directory; default: <root>/baselines/chores-...")
    parser.add_argument("--resume", action="store_true",
                        help="skip seeds that already have result.json under --out")
    parser.add_argument("--only-phases", nargs="+", choices=list(PHASES), default=None,
                        help="run a subset of chore phases")
    args = parser.parse_args()
    only_phases = args.only_phases

    if args.root is None:
        root = Path(tempfile.mkdtemp(prefix="steward-chores-baseline-"))
        build_default_kitchen(AuthoringService(root))
    else:
        root = args.root
        root.mkdir(parents=True, exist_ok=True)
        if "phase0-chores" not in AuthoringService(root).list_scenarios():
            build_default_kitchen(AuthoringService(root))

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = args.out or (root / "baselines" / f"chores-{args.phase}-{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.one_seed is not None:
        result = run_seed(root, out_dir, args.one_seed, args.phase, args.speed, args.timeout_s,
                          only_phases=only_phases)
        status = "ok" if result["ok"] else f"FAIL {result['error']}"
        explore = result.get("afterExplore") or {}
        print(json.dumps({"seed": args.one_seed, "status": status,
                          "visibleRecall": explore.get("visibleRecall"),
                          "posErr": explore.get("meanPositionErrorM"),
                          "wallS": result["wallS"]},
                         ensure_ascii=False), flush=True)
        return 0 if result["ok"] else 1

    config = {
        "phase": args.phase,
        "scenario": "phase0-chores",
        "seeds": args.seeds,
        "first": args.first,
        "speed": args.speed,
        "timeoutS": args.timeout_s,
        "root": str(root),
        "purpose": "iteration-A-baseline",
        "subprocessPerSeed": True,
        "onlyPhases": only_phases or list(PHASES),
        "metrics": {
            "fullRoundOk": "all three phases completed and confirmed by World State",
            "wsRate": "fraction of seeds whose phase passed World State check",
            "visibleRecall": "matchedObjects / (gtObjects - closed-drawer contents)",
            "observer": list(OBSERVER_KEYS),
        },
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"outDir": str(out_dir), **{k: config[k] for k in ("phase", "seeds", "speed")},
                      "onlyPhases": config["onlyPhases"]},
                     ensure_ascii=False), flush=True)

    script = Path(__file__).resolve()
    results: list[dict] = []
    t0 = time.monotonic()
    for i in range(args.seeds):
        seed = args.first + i
        existing = load_seed_result(out_dir, seed) if args.resume else None
        if existing is not None:
            result = existing
            note = "resume"
        else:
            result = run_seed_subprocess(script, root, out_dir, seed, args.phase, args.speed, args.timeout_s,
                                         only_phases=only_phases)
            note = "ran"
        results.append(result)
        status = "ok" if result["ok"] else f"FAIL {result.get('error')}"
        explore = result.get("afterExplore") or {}
        print(f"{bar(i + 1, args.seeds)}  seed {seed} [{note}]: {status}  "
              f"visibleRecall={explore.get('visibleRecall')}  "
              f"posErr={explore.get('meanPositionErrorM')}  "
              f"({time.monotonic() - t0:.0f}s)", flush=True)

    summary = summarize(results, config)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "outDir": str(out_dir),
        "fullRoundRate": summary["fullRoundRate"],
        "phases": {k: v["wsRate"] for k, v in summary["phases"].items()},
        "afterExplore": summary["afterExplore"],
        "failures": summary["failures"],
    }, ensure_ascii=False, indent=2), flush=True)
    return 0 if summary["fullRoundRate"] == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
