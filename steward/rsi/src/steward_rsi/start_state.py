"""Export an intermediate state for ``STEWARD_START_STATE_DIR``.

The file format is the one ``steward/world/start_state.py`` reads: ``<seed>.json`` holding
``{entity id: [x, y, z, qw, qx, qy, qz]}`` for free graspable objects and, optionally, a
``"__robot__"`` entry ``{"id", "base": [x, y, yaw], "joints": {name: value}}``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from .scene import World
from .score import iter_samples


def pick_state(run_dir: Path, at_s: float | None) -> tuple[dict, str]:
    """The final snapshot, or the last sample at or before ``at_s`` seconds after the task was sent."""
    run_dir = Path(run_dir)
    if at_s is None:
        for name in ("world_state.json", "world_latest.json"):
            if (run_dir / name).is_file():
                return json.loads((run_dir / name).read_text()), name
        raise FileNotFoundError(f"no world_state.json or world_latest.json in {run_dir}")
    chosen = None
    for row in iter_samples(run_dir / "truth_samples.jsonl"):
        if float(row.get("elapsed_s", 0.0)) <= at_s:
            chosen = row
    if chosen is None:
        raise LookupError(f"no sample at or before {at_s} s in {run_dir / 'truth_samples.jsonl'}")
    return chosen["state"], f"truth_samples.jsonl@{chosen['elapsed_s']}s"


def _yaw(q) -> float:
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def start_poses(state: dict, with_robot: bool = False) -> tuple[dict, list[str]]:
    """Poses for every free object with a ``graspable`` affordance, and warnings."""
    world = World(state)
    out: dict = {}
    warnings = []
    for eid, ent in world.entities.items():
        if not world.is_free_object(eid):
            continue
        graspable = any(item.get("value") == "graspable"
                        for link in ent.get("body", {}).get("links", []) for item in link.get("affordances", []))
        if not graspable:
            continue
        pose = world.pose(eid)
        out[eid] = [*map(float, pose["p"]), *map(float, pose["q"])]
    held = world.held_by_robot()
    if held:
        warnings.append("the robot holds " + ", ".join(sorted(held)) + "; a start state does not restore grasps, "
                        "so held objects will fall when the episode starts")
    if with_robot:
        robots = [eid for eid, e in world.entities.items() if e.get("role") == "robot"]
        if len(robots) != 1:
            warnings.append(f"expected one robot, found {len(robots)}; robot pose not exported")
        else:
            rid = robots[0]
            pose = world.pose(rid)
            joints = {}
            for joint in world.entities[rid].get("body", {}).get("joints", []):
                q = (joint.get("q") or {}).get("value") or []
                if len(q) == 1 and (joint.get("type") or {}).get("value") in ("prismatic", "revolute"):
                    joints[joint["id"]] = float(q[0])
            out["__robot__"] = {"id": rid, "base": [float(pose["p"][0]), float(pose["p"][1]), _yaw(pose["q"])],
                                "joints": joints}
    return out, warnings


def save_start(run_dir: Path, out_dir: Path, seeds: list[int], at_s: float | None = None,
               with_robot: bool = False) -> tuple[list[Path], list[str], str]:
    state, source = pick_state(run_dir, at_s)
    poses, warnings = start_poses(state, with_robot)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for seed in seeds:
        path = out_dir / f"{seed}.json"
        path.write_text(json.dumps(poses, indent=1) + "\n")
        written.append(path)
    (out_dir / "SOURCE.json").write_text(json.dumps({"run": str(run_dir), "state": source, "seeds": seeds,
                                                     "withRobot": with_robot, "warnings": warnings}, indent=2))
    return written, warnings, source
