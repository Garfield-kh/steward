"""Command line: ``python -m steward_rsi {check,run,score,review,save-start,round}``."""

from __future__ import annotations

import argparse
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

from . import providers
from .config import Config, ConfigError, load, resolve_api_key
from .review import review
from .runner import RoundExistsError, dry_run_text, execute, plan
from .scene import load_goal, load_spec
from .score import find_runs, score_runs
from .start_state import save_start


def _cfg(args) -> Config:
    return load(args.config)


def _advisor(cfg: Config) -> providers.Provider:
    key, _ = resolve_api_key(cfg.advisor)
    return providers.make(cfg.advisor, key)


# ------------------------------------------------------------------ check
def controller_checks(cfg: Config, environ=None) -> list[tuple[str, str, str]]:
    """Checks for the controller model (the robot's Ego Harness). No value of any key is returned."""
    environ = os.environ if environ is None else environ
    out = []
    env_file = cfg.steward_dir / ".env"
    in_file = False
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            key, _, value = line.strip().removeprefix("export ").partition("=")
            if key.strip() == "DEEPSEEK_API_KEY" and value.strip().strip("'\""):
                in_file = True
    if in_file:
        out.append(("OK", "controller key", "DEEPSEEK_API_KEY is set in steward/.env"))
    elif environ.get("DEEPSEEK_API_KEY"):
        out.append(("WARN", "controller key", "DEEPSEEK_API_KEY is only in this shell's environment; "
                    "steward up passes it on, but steward/.env is the documented place"))
    else:
        out.append(("FAIL", "controller key", "no DEEPSEEK_API_KEY in steward/.env or the environment"))
    profile = cfg.run.robot_harness_template / "profiles" / "ego"
    if not profile.is_dir():
        out.append(("FAIL", "controller profile", f"{profile} is missing; run steward/scripts/setup-dsh-profiles.sh"))
    else:
        patch = profile / "cordis.patch.yml"
        model = None
        if patch.is_file():
            match = re.search(r"id:\s*agent-default-model\s*\n(?:\s+.*\n)*?\s+model:\s*['\"]?([\w.\-/]+)",
                              patch.read_text(encoding="utf-8"))
            model = match.group(1) if match else None
        detail = (f"model {model} set in {patch}" if model else
                  "no agent-default-model override in the profile; the Harness default applies "
                  "(deepseek-official / deepseek-flash in the shipped templates)")
        out.append(("OK", "controller profile", f"{profile}; {detail}"))
    return out


def _port_free(port: int) -> bool:
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def cmd_check(args) -> int:
    rows: list[tuple[str, str, str]] = []
    try:
        cfg = _cfg(args)
    except ConfigError as error:
        print(f"FAIL config: {error}")
        return 1
    rows.append(("OK", "config", f"{cfg.path} (advisor={cfg.advisor.name}, backend={cfg.machine.backend}, "
                                  f"task={cfg.task.name}, devices={cfg.machine.devices})"))
    service = cfg.repo_root / "steward" / "src" / "steward" / "authoring" / "service.py"
    rows.append(("OK" if service.is_file() else "FAIL", "repository", str(cfg.repo_root)))
    try:
        goal, source = load_goal(cfg.repo_root)
        rows.append(("OK", "scene goal", f"{len(goal['constraints'])} constraints from {source}"))
        spec, source = load_spec(cfg.repo_root)
        rows.append(("OK", "scene definition", f"{len(spec['entities'])} entities from {source}"))
    except (LookupError, ValueError, SyntaxError) as error:
        rows.append(("FAIL", "scene goal", str(error)))
    rows.append(("OK" if cfg.task.task_file.is_file() else "FAIL", "task file", str(cfg.task.task_file)))
    if cfg.task.start_state_dir:
        missing = [s for s in cfg.task.seeds if not (cfg.task.start_state_dir / f"{s}.json").is_file()]
        rows.append(("WARN" if missing else "OK", "start state",
                     f"{cfg.task.start_state_dir}" + (f"; no file for seeds {missing} (they start from scratch)"
                                                       if missing else "")))
    module = "steward_npu" if cfg.machine.backend == "npu" else "steward.cli"
    if Path(cfg.machine.python).exists() or "/" not in cfg.machine.python:
        probe = subprocess.run([cfg.machine.python, "-c", f"import importlib.util,sys; "
                                f"sys.exit(0 if importlib.util.find_spec('{module.split('.')[0]}') else 3)"],
                               capture_output=True, text=True, cwd=cfg.steward_dir, timeout=120)
        rows.append(("OK" if probe.returncode == 0 else "FAIL", "simulation python",
                     f"{cfg.machine.python} finds {module}" if probe.returncode == 0
                     else f"{cfg.machine.python} cannot import {module}: {probe.stderr.strip()[-300:]}"))
    else:
        rows.append(("FAIL", "simulation python", f"{cfg.machine.python} does not exist"))
    perception = cfg.steward_dir / "config" / "perception-services.toml"
    rows.append(("OK" if perception.is_file() else "WARN", "perception config",
                 f"{perception} (run `steward services` to test reachability)"))
    rows += controller_checks(cfg)
    slots = len(cfg.machine.devices) * cfg.machine.episodes_per_device
    busy = [cfg.machine.base_port + 10 * s + k for s in range(slots) for k in range(3)
            if not _port_free(cfg.machine.base_port + 10 * s + k)]
    rows.append(("WARN" if busy else "OK", "ports", f"busy: {busy}" if busy else
                 f"{slots} slots from {cfg.machine.base_port} are free"))
    try:
        key, origin = resolve_api_key(cfg.advisor)
        rows.append(("OK", "advisor key", f"{cfg.advisor.name} key read from {origin}"))
    except ConfigError as error:
        rows.append(("FAIL", "advisor key", str(error)))
        key = None
    if key is not None and not args.offline:
        try:
            t0 = time.time()
            reply = providers.make(cfg.advisor, key).complete("Reply with the single word OK.", "ping")
            rows.append(("OK", "advisor request", f"{cfg.advisor.name} {reply.model} answered "
                         f"{reply.text.strip()[:40]!r} in {time.time() - t0:.1f}s"))
        except providers.ProviderError as error:
            rows.append(("FAIL", "advisor request", str(error)))
    for status, name, detail in rows:
        print(f"{status:4} {name}: {detail}")
    return 1 if any(r[0] == "FAIL" for r in rows) else 0


# ------------------------------------------------------------------ run / score / review
def _round_dir(cfg: Config, name: str | None) -> Path:
    return cfg.output_dir / (name or time.strftime("round-%Y%m%d-%H%M%S"))


def cmd_run(args) -> int:
    cfg = _cfg(args)
    round_dir = _round_dir(cfg, args.name)
    plans = plan(cfg, round_dir)
    if args.dry_run:
        sys.stdout.write(dry_run_text(cfg, plans))
        return 0
    execute(cfg, plans, round_dir)
    print(f"round directory: {round_dir}")
    return 0


def _score(cfg: Config, dirs: list[Path], out_dir: Path) -> dict:
    goal, _ = load_goal(cfg.repo_root)
    spec, _ = load_spec(cfg.repo_root)
    results, summary = score_runs(dirs, goal, spec, cfg.task.name, cfg.task.threshold_mm, cfg.task.strict_mm,
                                  cfg.task.kit_body, out_dir)
    print((out_dir / "funnel.md").read_text())
    return summary


def cmd_score(args) -> int:
    cfg = _cfg(args)
    dirs = [Path(d) for d in args.runs]
    out = Path(args.out) if args.out else (dirs[0] if len(dirs) == 1 else Path.cwd()) / "score"
    _score(cfg, dirs, out)
    print(f"wrote {out / 'score.json'} and {out / 'funnel.md'}")
    return 0


def _review(cfg: Config, score_dir: Path, run_paths: list[Path]) -> Path:
    run_dirs = {p.name: p for p in find_runs(run_paths)}
    task_text = cfg.task.task_file.read_text(encoding="utf-8")
    advice = review(score_dir / "score.json", run_dirs, _advisor(cfg), task_text, cfg.language, score_dir)
    print(advice.read_text())
    return advice


def cmd_review(args) -> int:
    cfg = _cfg(args)
    score_dir = Path(args.score)
    runs = [Path(r) for r in args.runs] if args.runs else [score_dir.parent]
    advice = _review(cfg, score_dir, runs)
    print(f"wrote {advice}")
    return 0


def cmd_save_start(args) -> int:
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    if seeds is None:
        seeds = _cfg(args).task.seeds
    written, warnings, source = save_start(Path(args.run), Path(args.out), seeds, args.at, args.with_robot)
    for w in warnings:
        print(f"WARN {w}")
    print(f"exported {source} to {', '.join(str(p) for p in written)}")
    print(f"use: STEWARD_START_STATE_DIR={Path(args.out).resolve()} (or [task].start_state_dir in rsi.toml)")
    return 0


def cmd_round(args) -> int:
    cfg = _cfg(args)
    round_dir = _round_dir(cfg, args.name)
    plans = plan(cfg, round_dir)
    if args.dry_run:
        sys.stdout.write(dry_run_text(cfg, plans))
        return 0
    execute(cfg, plans, round_dir)
    _score(cfg, [round_dir], round_dir / "score")
    if args.no_review:
        return 0
    _review(cfg, round_dir / "score", [round_dir])
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m steward_rsi", description=__doc__)
    parser.add_argument("--config", default=os.environ.get("STEWARD_RSI_CONFIG", "rsi.toml"),
                        help="configuration file (default: ./rsi.toml or $STEWARD_RSI_CONFIG)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check", help="validate configuration, keys and the advisor connection")
    p.add_argument("--offline", action="store_true", help="do not send the advisor test request")
    p.set_defaults(fn=cmd_check)
    for name, fn, text in (("run", cmd_run, "start a batch of whole episodes"),
                           ("round", cmd_round, "run, score and review in one go")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--name", help="round directory name under [paths].output_dir")
        p.add_argument("--dry-run", action="store_true", help="print commands and environment; start nothing")
        if name == "round":
            p.add_argument("--no-review", action="store_true", help="stop after scoring")
        p.set_defaults(fn=fn)
    p = sub.add_parser("score", help="score episode directories and write the funnel")
    p.add_argument("runs", nargs="+", help="a round directory or episode directories")
    p.add_argument("--out", help="output directory (default: <run>/score)")
    p.set_defaults(fn=cmd_score)
    p = sub.add_parser("review", help="send the funnel and question sheet to the advisor")
    p.add_argument("score", help="directory holding score.json and funnel.md")
    p.add_argument("--runs", nargs="*", help="episode or round directories (default: parent of the score dir)")
    p.set_defaults(fn=cmd_review)
    p = sub.add_parser("save-start", help="export an intermediate state for STEWARD_START_STATE_DIR")
    p.add_argument("run", help="episode directory")
    p.add_argument("--out", required=True, help="directory to write <seed>.json into")
    p.add_argument("--at", type=float, help="seconds after the task was sent (default: final state)")
    p.add_argument("--seeds", help="comma-separated seeds (default: [task].seeds)")
    p.add_argument("--with-robot", action="store_true", help="also export robot base pose and joint values")
    p.set_defaults(fn=cmd_save_start)
    args = parser.parse_args(argv)
    try:
        return args.fn(args)
    except (ConfigError, providers.ProviderError, RoundExistsError, FileNotFoundError, LookupError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
