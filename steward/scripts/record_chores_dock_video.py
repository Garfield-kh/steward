"""Re-run chores drawers for one seed while recording a free-camera World State video.

Aimed at watching drawer place docks that collide with the sideboard body (seed 2).

    .venv/bin/python scripts/record_chores_dock_video.py --one-seed 2 --speed 10

Writes frames + ``dock.mp4`` under ``--out`` (default: baselines/chores-phase1-drawer-dock-video-seedN).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

# Allow ``from chores_baseline import …`` when launched as ``python scripts/….py``.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from steward.assets.generators import steward_bot
from steward.authoring.service import AuthoringService, build_default_kitchen
from steward.chores import PHASE_NAMES, Chores, _explore
from steward.ego.runtime import EgoRuntime, RobotLink
from steward.ego.server import ego_api
from steward.mission import MissionFailed
from steward.net import HttpError
from steward.world.chores_check import check_swap, drawer_contents
from steward.world.chores_diagnostics import append_jsonl, write_json
from steward.world.server import StewardHost

from chores_baseline import (  # noqa: E402
    ROBOT,
    _flush_chore_log,
    _record_phase_supports,
    _segmenter_for_phase,
    _write_partial,
    perception_snapshot,
    truth,
)


def _fetch_jpeg(url: str, timeout: float = 5.0) -> bytes | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read()
    except Exception:
        return None


def _encode_mp4(frames_dir: Path, out_mp4: Path, fps: float) -> None:
    pattern = str(frames_dir / "frame_%05d.jpg")
    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps), "-i", pattern,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(out_mp4),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--one-seed", type=int, default=2)
    parser.add_argument("--phase", default="phase1")
    parser.add_argument("--speed", type=float, default=10.0)
    parser.add_argument("--timeout-s", type=float, default=400.0)
    parser.add_argument("--root", type=Path, default=Path("var/steward"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=2.0, help="wall-clock capture rate")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    # Sideboard-centred free camera (kitchen chores layout: board near x≈-2.7).
    parser.add_argument("--lookat", default="-2.7,0.6,0.55")
    parser.add_argument("--distance", type=float, default=3.2)
    parser.add_argument("--azimuth", type=float, default=145.0)
    parser.add_argument("--elevation", type=float, default=-28.0)
    args = parser.parse_args()

    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    if "phase0-chores" not in AuthoringService(root).list_scenarios():
        build_default_kitchen(AuthoringService(root))

    out_dir = args.out or (root / "baselines" / f"chores-{args.phase}-drawer-dock-video-seed{args.one_seed}")
    run_dir = out_dir / f"seed-{args.one_seed}"
    frames_dir = run_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    log_jsonl = run_dir / "chores.log.jsonl"

    host = StewardHost(root, scenario="phase0-chores", speed=args.speed, seed=args.one_seed,
                       run_dir=run_dir, phase=args.phase).start()
    runtime = EgoRuntime(RobotLink(host.robot_servers[ROBOT].url), steward_bot().to_dict(),
                         segmenter=_segmenter_for_phase(args.phase, root))
    runtime.start()
    api = ego_api(runtime).start()
    stop = threading.Event()
    record_stop = threading.Event()
    frame_n = {"n": 0}

    def pump() -> None:
        while not stop.is_set():
            host.pump(0.05)

    query = urllib.parse.urlencode({
        "camera": "free",
        "lookat": args.lookat,
        "distance": args.distance,
        "azimuth": args.azimuth,
        "elevation": args.elevation,
        "width": args.width,
        "height": args.height,
        "format": "jpeg",
    })
    render_url = f"{host.wca.url}/wca/render?{query}"

    def record_loop() -> None:
        interval = 1.0 / max(args.fps, 0.1)
        while not record_stop.is_set():
            t0 = time.monotonic()
            data = _fetch_jpeg(render_url)
            if data:
                frame_n["n"] += 1
                (frames_dir / f"frame_{frame_n['n']:05d}.jpg").write_bytes(data)
            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, interval - elapsed))

    worker = threading.Thread(target=pump, name="pump-video", daemon=True)
    worker.start()
    recorder = threading.Thread(target=record_loop, name="record-video", daemon=True)
    # Start recording once drawers begin so the file focuses on dock attempts.
    t0 = time.monotonic()
    phases: list[dict] = []
    error: str | None = None
    after_explore: dict | None = None
    runner: Chores | None = None
    try:
        def on_step(record: dict) -> None:
            append_jsonl(log_jsonl, record)

        runner = Chores(api.url, timeout_s=args.timeout_s, on_step=on_step)
        explore = _explore(api.url, args.timeout_s)
        time.sleep(0.5)
        after_explore = perception_snapshot(host)
        after_explore["exploreState"] = explore.get("state")
        before_drawers = drawer_contents(truth(host)[0])
        _write_partial(run_dir, args.one_seed, args.phase, after_explore, phases, error,
                        time.monotonic() - t0)

        recorder.start()
        entry: dict = {"name": "drawers", "label": PHASE_NAMES["drawers"]}
        try:
            runner.swap_drawers()
            time.sleep(0.5)
            ok, detail = check_swap(before_drawers, truth(host)[0])
            entry.update({"completed": True, "wsOk": ok, "detail": detail,
                          "observer": perception_snapshot(host)})
            if not ok:
                error = f"drawers world-state mismatch: {detail}"
        except (MissionFailed, HttpError, OSError) as exc:
            entry.update({"completed": False, "wsOk": False, "error": str(exc)[:500],
                          "observer": perception_snapshot(host)})
            error = f"drawers: {exc}"
        entry["supports"] = _record_phase_supports(run_dir, "drawers", host, api.url, entry)
        phases.append(entry)
    finally:
        record_stop.set()
        recorder.join(timeout=5.0)
        stop.set()
        worker.join(timeout=2.0)
        wall_s = round(time.monotonic() - t0, 1)
        if runner is not None:
            _flush_chore_log(run_dir, runner)
        for closer in (api.stop, runtime.stop, host.stop):
            try:
                closer()
            except Exception as exc:
                print(f"warning: shutdown {closer}: {exc}", flush=True)

    ok = error is None and phases and phases[0].get("wsOk")
    result = {
        "seed": args.one_seed,
        "phase": args.phase,
        "ok": bool(ok),
        "wallS": wall_s,
        "error": error,
        "onlyPhases": ["drawers"],
        "afterExplore": after_explore,
        "phases": phases,
        "video": {
            "frames": frame_n["n"],
            "fps": args.fps,
            "lookat": args.lookat,
            "distance": args.distance,
            "azimuth": args.azimuth,
            "elevation": args.elevation,
        },
    }
    write_json(run_dir / "result.json", result)
    write_json(out_dir / "config.json", {
        "seed": args.one_seed, "phase": args.phase, "speed": args.speed,
        "timeout_s": args.timeout_s, "fps": args.fps, "camera": result["video"],
    })

    mp4 = run_dir / "dock.mp4"
    if frame_n["n"] > 0:
        try:
            _encode_mp4(frames_dir, mp4, args.fps)
            print(json.dumps({
                "seed": args.one_seed,
                "status": "ok" if ok else f"FAIL {error}",
                "frames": frame_n["n"],
                "mp4": str(mp4),
                "wallS": wall_s,
            }, ensure_ascii=False), flush=True)
        except subprocess.CalledProcessError as exc:
            err = exc.stderr.decode(errors="replace")[:500] if exc.stderr else str(exc)
            print(f"ffmpeg failed: {err}", flush=True)
            print(json.dumps({"seed": args.one_seed, "frames": frame_n["n"], "mp4": None,
                              "error": error}), flush=True)
            return 1
    else:
        print(json.dumps({"seed": args.one_seed, "frames": 0,
                          "error": error or "no frames"}), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
