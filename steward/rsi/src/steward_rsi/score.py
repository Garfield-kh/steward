"""Score finished episodes from World State snapshots and build a stage funnel.

Per body hole of the selected kit (an *interface*), stages are cumulative:

1. ``placed``  - a hole of a part of the expected asset is within the goal position tolerance of
   the body hole, with its axis pointing the same way as the body-hole axis (within 20 degrees),
   the direction the simulator requires before it fastens two holes.
2. ``aligned`` - additionally, its lateral offset from the body-hole axis is within the threshold.
3. ``seated``  - additionally, a screw fastens this body hole to a hole of a part of the expected asset.

An episode is ``complete`` when every interface is ``seated`` (so also placed and aligned) with an
authorized screw and lateral error within the threshold, the assembly rests on the kit table
through support relations, the robot holds no kit entity, no screw of another kit fastens it, and
no entity of another kit is seen moved or grasped in the final snapshot or in any sample of
``truth_samples.jsonl``. ``complete`` is computed for ``threshold_mm`` and, separately, ``strict_mm``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from .scene import World, Kit, derive_kits, dot, norm, quat_angle_deg, select_kit, sub

COLLINEAR_COS = math.cos(math.radians(20.0))
MOVE_M = 0.02
MOVE_DEG = 10.0
STAGES = ("placed", "aligned", "seated")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def iter_samples(path: Path):
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue  # a partially written last line
            if isinstance(row, dict) and "state" in row:
                yield row


def _pair(world: World, body: str, hole: str, part: str, part_hole: str) -> dict:
    a, axis = world.hole_world(body, hole)
    b, paxis = world.hole_world(part, part_hole)
    d = sub(b, a)
    along = dot(d, axis)
    lateral = norm([d[i] - axis[i] * along for i in range(3)])
    return {"part": part, "partHole": part_hole, "distanceM": norm(d), "axialM": abs(along),
            "lateralMm": lateral * 1000.0, "collinear": dot(axis, paxis) >= COLLINEAR_COS}


def interfaces(world: World, kit: Kit, threshold_mm: float) -> list[dict]:
    """One row per body hole of ``kit`` in ``world``."""
    fasteners = world.fasteners()
    asset_parts: dict[str, list[str]] = {}
    for part in kit.parts:
        asset_parts.setdefault(world.asset(part), []).append(part)
    rows: dict[str, dict] = {}
    used: set[tuple[str, str]] = set()
    for hole, asset in sorted(kit.hole_asset.items()):
        rows[hole] = {"hole": hole, "expectedAsset": asset, "screw": None, "screwAuthorized": None,
                      "seatedPartAsset": None, "match": None}
        for screw, holes in fasteners.items():
            if not any(h["entity"] == kit.body and h["hole"] == hole for h in holes):
                continue
            others = [h for h in holes if h["entity"] != kit.body]
            row = rows[hole]
            row["screw"] = screw
            row["screwAuthorized"] = screw in kit.screws
            if others:
                other = others[0]
                row["seatedPartAsset"] = world.asset(other["entity"]) if other["entity"] in world.entities else None
                if other["entity"] in world.entities and other["hole"] in world.holes(other["entity"]):
                    row["match"] = _pair(world, kit.body, hole, other["entity"], other["hole"])
                    used.add((other["entity"], other["hole"]))
            break
    # Holes without a fastened partner: greedy nearest unused hole of an expected-asset part.
    candidates = []
    for hole, row in rows.items():
        if row["match"] is not None:
            continue
        for part in asset_parts.get(row["expectedAsset"], []):
            for part_hole in world.holes(part):
                if (part, part_hole) in used:
                    continue
                pair = _pair(world, kit.body, hole, part, part_hole)
                candidates.append((pair["distanceM"], hole, pair))
    for _, hole, pair in sorted(candidates, key=lambda c: c[0]):
        if rows[hole]["match"] is not None or (pair["part"], pair["partHole"]) in used:
            continue
        rows[hole]["match"] = pair
        used.add((pair["part"], pair["partHole"]))
    out = []
    for hole, row in rows.items():
        m = row["match"]
        placed = bool(m and m["collinear"] and m["distanceM"] <= kit.position_tol_m)
        aligned = placed and m["lateralMm"] <= threshold_mm
        seated_raw = row["screw"] is not None and row["seatedPartAsset"] == row["expectedAsset"]
        out.append({**row, "lateralMm": round(m["lateralMm"], 3) if m else None,
                    "match": {k: (round(v, 5) if isinstance(v, float) else v) for k, v in m.items()} if m else None,
                    "placed": placed, "aligned": aligned, "seatedRaw": seated_raw,
                    "seated": aligned and seated_raw})
    return out


def _moved(initial: World, world: World, ids) -> dict[str, dict]:
    out = {}
    for eid in ids:
        if eid not in world.entities or eid not in initial.entities:
            continue
        a, b = initial.pose(eid), world.pose(eid)
        t = norm(sub(b["p"], a["p"]))
        r = quat_angle_deg(a["q"], b["q"])
        if t > MOVE_M or r > MOVE_DEG:
            out[eid] = {"translationM": round(t, 4), "rotationDeg": round(r, 2)}
    return out


def score_episode(run_dir: Path, goal: dict, spec: dict, task_name: str, threshold_mm: float, strict_mm: float,
                  kit_body: str = "") -> dict:
    """Score one episode directory. Missing snapshots give ``{"valid": false, "error": ...}``."""
    run_dir = Path(run_dir)
    meta = _read_json(run_dir / "episode.json") if (run_dir / "episode.json").is_file() else {}
    final_path = next((p for p in (run_dir / "world_state.json", run_dir / "world_latest.json") if p.is_file()), None)
    samples_path = run_dir / "truth_samples.jsonl"
    initial_state = None
    if (run_dir / "world_initial.json").is_file():
        initial_state = _read_json(run_dir / "world_initial.json")
    else:
        first = next(iter_samples(samples_path), None)
        initial_state = first["state"] if first else None
    base = {"run": run_dir.name, "seed": meta.get("seed"), "endReason": meta.get("endReason"),
            "wallS": meta.get("wallS"), "startState": meta.get("startStateDir")}
    if final_path is None or initial_state is None:
        return {**base, "valid": False, "error": "missing world_state.json or initial snapshot"}
    initial = World(initial_state)
    final = World(_read_json(final_path))
    kits = derive_kits(goal, spec, initial)
    kit = select_kit(kits, initial, task_name, kit_body)
    foreign_kits = [k for k in kits if k.body != kit.body]
    foreign = set().union(*(k.members | set(k.screws) for k in foreign_kits)) if foreign_kits else set()
    others = {eid for eid in initial.entities if initial.is_free_object(eid)} - foreign - kit.members - set(kit.screws)
    others -= {kit.table, *(k.table for k in foreign_kits)}

    rows = interfaces(final, kit, threshold_mm)
    lateral = [r["lateralMm"] for r in rows if r["seatedRaw"] and r["lateralMm"] is not None]
    max_lateral = max(lateral) if len(lateral) == len(rows) and rows else None

    foreign_moved = _moved(initial, final, foreign)
    other_moved = _moved(initial, final, others)
    foreign_grasped = set(final.held_by_robot()) & foreign
    milestones: dict[str, float] = {}
    hole_seated: dict[str, float] = {}
    samples = 0
    for row in iter_samples(samples_path):
        samples += 1
        w = World(row["state"])
        for eid, move in _moved(initial, w, foreign).items():
            best = foreign_moved.get(eid)
            if best is None or move["translationM"] > best["translationM"]:
                foreign_moved[eid] = move
        foreign_grasped |= w.held_by_robot() & foreign
        seated_now = [r["hole"] for r in interfaces(w, kit, threshold_mm) if r["seatedRaw"]]
        at = round(float(row.get("elapsed_s", 0.0)), 1)
        for k in range(1, len(seated_now) + 1):
            milestones.setdefault(str(k), at)
        for hole in seated_now:
            hole_seated.setdefault(hole, at)

    members = kit.members
    foreign_screws_used = sorted({r["screw"] for r in rows if r["screw"] and not r["screwAuthorized"]})
    supported = kit.table in final.support_closure(members)
    held = sorted((members | set(kit.screws)) & final.held_by_robot())
    clean = not foreign_moved and not foreign_grasped and not foreign_screws_used
    all_seated = bool(rows) and all(r["seated"] and r["screwAuthorized"] for r in rows)
    base_ok = all_seated and supported and not held and clean

    def complete(limit):
        return bool(base_ok and max_lateral is not None and max_lateral <= limit)

    return {**base, "valid": True, "task": task_name, "kit": {"body": kit.body, "parts": kit.parts,
            "table": kit.table, "screws": kit.screws, "holeAsset": kit.hole_asset},
            "interfaces": rows,
            "counts": {s: sum(1 for r in rows if r[s]) for s in STAGES} | {"total": len(rows)},
            "maxLateralMm": round(max_lateral, 3) if max_lateral is not None else None,
            "supportedByKitTable": supported, "kitHeldByRobot": held,
            "foreignMoved": foreign_moved, "foreignGrasped": sorted(foreign_grasped),
            "foreignScrewsUsed": foreign_screws_used, "otherMoved": other_moved,
            "seatedMilestonesWallS": milestones, "holeFirstSeatedWallS": hole_seated, "samplesRead": samples,
            "thresholdMm": threshold_mm, "strictMm": strict_mm,
            "complete": complete(threshold_mm), "completeStrict": complete(strict_mm)}


def funnel(results: list[dict]) -> dict:
    """Aggregate per-interface stage counts and per-episode completion over valid episodes."""
    valid = [r for r in results if r.get("valid")]
    total = sum(r["counts"]["total"] for r in valid)
    reached = {s: sum(r["counts"][s] for r in valid) for s in STAGES}
    all_seated = sum(1 for r in valid if r["counts"]["seated"] == r["counts"]["total"] and r["counts"]["total"])
    complete = sum(1 for r in valid if r["complete"])
    complete_strict = sum(1 for r in valid if r["completeStrict"])
    rows = []
    previous = total
    for s in STAGES:
        rows.append({"stage": s, "unit": "interface", "reached": reached[s], "of": previous,
                     "conversion": (reached[s] / previous) if previous else None})
        previous = reached[s]
    rows.append({"stage": "complete", "unit": "episode", "reached": complete, "of": all_seated,
                 "conversion": (complete / all_seated) if all_seated else None})
    rows.append({"stage": "complete_strict", "unit": "episode", "reached": complete_strict, "of": complete,
                 "conversion": (complete_strict / complete) if complete else None})
    # The strict row only decides when every earlier stage converts fully.
    worst = None
    for row in rows[:-1]:
        if row["conversion"] is None or row["conversion"] >= 1.0:
            continue
        if worst is None or row["conversion"] < worst["conversion"]:
            worst = row
    if worst is None and rows[-1]["conversion"] is not None and rows[-1]["conversion"] < 1.0:
        worst = rows[-1]
    return {"episodes": len(results), "validEpisodes": len(valid),
            "invalid": [{"run": r["run"], "error": r.get("error")} for r in results if not r.get("valid")],
            "interfaces": total, "stages": rows, "worstStage": worst["stage"] if worst else None,
            "complete": complete, "completeStrict": complete_strict,
            "thresholdMm": valid[0]["thresholdMm"] if valid else None,
            "strictMm": valid[0]["strictMm"] if valid else None,
            "foreignInterference": sum(1 for r in valid if r["foreignMoved"] or r["foreignGrasped"]
                                       or r["foreignScrewsUsed"])}


def _pct(x):
    return "-" if x is None else f"{100 * x:.0f}%"


def markdown(results: list[dict], summary: dict) -> str:
    lines = ["# RSI funnel", "",
             f"Episodes: {summary['episodes']} (valid {summary['validEpisodes']}); "
             f"interfaces: {summary['interfaces']}.",
             f"Complete at {summary['thresholdMm']} mm: {summary['complete']}/{summary['validEpisodes']}; "
             f"at {summary['strictMm']} mm (strict): {summary['completeStrict']}/{summary['validEpisodes']}.",
             f"Episodes with interference with another kit: {summary['foreignInterference']}.",
             f"Worst stage: **{summary['worstStage']}**.", "",
             "| Stage | Unit | Reached | Of | Conversion |", "| --- | --- | ---: | ---: | ---: |"]
    for row in summary["stages"]:
        lines.append(f"| {row['stage']} | {row['unit']} | {row['reached']} | {row['of']} | {_pct(row['conversion'])} |")
    lines += ["", "## Episodes", "",
              "| Run | Seed | End | Placed | Aligned | Seated | Max lateral mm | Complete | Strict | Not allowed |",
              "| --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- |"]
    for r in results:
        if not r.get("valid"):
            lines.append(f"| {r['run']} | {r.get('seed')} | {r.get('endReason')} | - | - | - | - | invalid | - | {r.get('error')} |")
            continue
        c = r["counts"]
        bad = []
        if r["foreignMoved"]:
            bad.append("moved " + ",".join(sorted(r["foreignMoved"])))
        if r["foreignGrasped"]:
            bad.append("grasped " + ",".join(r["foreignGrasped"]))
        if r["foreignScrewsUsed"]:
            bad.append("used " + ",".join(r["foreignScrewsUsed"]))
        lines.append(f"| {r['run']} | {r.get('seed')} | {r.get('endReason')} | {c['placed']}/{c['total']} | "
                     f"{c['aligned']}/{c['total']} | {c['seated']}/{c['total']} | {r['maxLateralMm']} | "
                     f"{r['complete']} | {r['completeStrict']} | {'; '.join(bad) or '-'} |")
    if summary["invalid"]:
        lines += ["", "Invalid episodes are infrastructure failures and are excluded from the denominators."]
    return "\n".join(lines) + "\n"


def find_runs(paths: list[Path]) -> list[Path]:
    """Episode directories: each given directory that holds a snapshot, else its children that do."""
    markers = ("world_state.json", "world_latest.json", "episode.json")
    out = []
    for path in paths:
        if any((path / m).is_file() for m in markers):
            out.append(path)
        elif path.is_dir():
            out += sorted(p for p in path.iterdir() if p.is_dir() and any((p / m).is_file() for m in markers))
    return out


def score_runs(paths: list[Path], goal: dict, spec: dict, task_name: str, threshold_mm: float, strict_mm: float,
               kit_body: str = "", out_dir: Path | None = None) -> tuple[list[dict], dict]:
    results = []
    for run in find_runs(paths):
        try:
            results.append(score_episode(run, goal, spec, task_name, threshold_mm, strict_mm, kit_body))
        except (ValueError, KeyError, LookupError) as error:
            results.append({"run": run.name, "valid": False, "error": f"{type(error).__name__}: {error}"})
    summary = funnel(results)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "score.json").write_text(json.dumps({"summary": summary, "episodes": results}, indent=2))
        (out_dir / "funnel.md").write_text(markdown(results, summary))
    return results, summary
