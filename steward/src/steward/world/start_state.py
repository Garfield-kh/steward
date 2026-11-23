"""Optional world-side initialization from saved object and robot poses.

When STEWARD_START_STATE_DIR is set and holds <seed>.json ({entity id: [x,y,z,qw,qx,qy,qz]}),
the listed free bodies are put at those poses right after the scene loads, and fasteners that
already sit through two mating holes are joined at once. An optional "__robot__" entry
{"id": robot entity id, "base": [x, y, yaw], "joints": {name: value}} also puts the robot base and
arm joints (positions and controller targets) at the recorded values. Off by default. The robot never reads this.
"""
import dataclasses
import json
import os
import sys

import mujoco


def _place_robot(sim, spec) -> str:
    names = sim.build.names[spec["id"]]
    done = []
    for joint, key, value in zip(names.root_joints, ("root_x", "root_y", "root_yaw"), spec["base"], strict=True):
        jid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        sim.data.qpos[sim.model.jnt_qposadr[jid]] = float(value)
        sim.data.qvel[sim.model.jnt_dofadr[jid]] = 0.0
        sim._set_ctrl(names.actuators[key], float(value))
        done.append(key)
    for name, value in (spec.get("joints") or {}).items():
        if name not in names.joints or name not in names.actuators:
            continue
        jid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, names.joints[name])
        sim.data.qpos[sim.model.jnt_qposadr[jid]] = float(value)
        sim.data.qvel[sim.model.jnt_dofadr[jid]] = 0.0
        sim._set_ctrl(names.actuators[name], float(value))
        done.append(name)
    return ",".join(done)


def _separate(sim, ids) -> int:
    """Recorded poses are rounded, so bodies that rested on each other come back interpenetrating by ~0.1 mm.
    A weld created in that state fights the contact and the assembly creeps (measured 1-2 cm per 2 min).
    Raise the upper body of every interpenetrating pair of placed bodies by a small clearance before welding."""
    clearance = float(os.environ.get("STEWARD_START_CLEARANCE_MM", "0.3")) / 1000.0
    if clearance <= 0:
        return 0
    roots = {}
    for entity_id in ids:
        ent = sim.template.find(entity_id)
        if ent is not None and ent.role not in ("robot", "actor") and ent.rigid.root.value == "free":
            roots[sim._root_body(ent.id)] = ent
    def owner(body):
        while body and body not in roots:
            body = sim.model.body_parentid[body]
        return body if body in roots else None
    raise_ids = set()
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        if c.dist >= 0:
            continue
        a, b = owner(sim.model.geom_bodyid[c.geom1]), owner(sim.model.geom_bodyid[c.geom2])
        if a is None or b is None or a == b:
            continue
        raise_ids.add(a if sim.data.xpos[a][2] >= sim.data.xpos[b][2] else b)
    for body in raise_ids:
        ent = roots[body]
        pose = sim._entity_pose(ent.id)
        p = list(pose["p"]); p[2] += clearance
        sim._set_root_pose(ent, {"p": p, "q": list(pose["q"])})
    if raise_ids:
        mujoco.mj_forward(sim.model, sim.data)
    return len(raise_ids)

def apply_start_state(sim) -> None:
    folder = os.environ.get("STEWARD_START_STATE_DIR")
    if not folder:
        return
    path = os.path.join(folder, f"{sim._seed}.json")
    if not os.path.isfile(path):
        return
    with open(path) as handle:
        poses = json.load(handle)
    robot = poses.pop("__robot__", None)
    moved = 0
    for entity_id, value in poses.items():
        ent = sim.template.find(entity_id)
        if ent is None or ent.role in ("robot", "actor") or ent.rigid.root.value != "free":
            continue
        sim._set_root_pose(ent, {"p": value[:3], "q": value[3:7]})
        moved += 1
    placed = ""
    if robot:
        try:
            placed = _place_robot(sim, robot)
        except Exception as error:  # evaluation aid only: report and continue with the default robot pose
            placed = f"FAILED {type(error).__name__}: {error}"
    mujoco.mj_forward(sim.model, sim.data)
    lifted = _separate(sim, set(poses))
    joined = 0
    if sim.fasten.mode != "off":
        original = sim.fasten
        sim.fasten = dataclasses.replace(original, dwell=1e-13, gap=0.05)
        try:
            for _ in range(8):
                before = len(sim._fastening.welds)
                sim._fastening.update()
                if len(sim._fastening.welds) == before:
                    break
            joined = len(sim._fastening.welds)
        finally:
            sim.fasten = original
    print(f"[start_state] seed {sim._seed}: moved {moved} bodies, separated {lifted}, welds {joined}, robot [{placed}] ({path})", file=sys.stderr, flush=True)
