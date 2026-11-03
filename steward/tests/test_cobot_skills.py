"""Cobot skills end to end in the simulation: pick a bowl off a table and put it down elsewhere."""

import numpy as np
import pytest

from steward.assets.cobot import TCP_OFFSET, cobot_magic
from steward.assets.generators import bowl, building, dining_table, plate
from steward.ego.cobot_skills import CobotSkills, MotionConfig
from steward.world.sim import WorldSim

# The left arm's tabletop band on the mobile twin, measured by scanning tool-down IK: the chassis
# front is at x = 0.34, and the arm reaches x 0.38-0.60 at y +-0.25 (y = 0 sits between the arms).
TABLE_TOP = 0.75
BOWL_AT = [0.40, 0.20, TABLE_TOP]
PLATE_AT = [0.62, 0.20, TABLE_TOP]


def _scene() -> dict:
    room = building(width=4.0, depth=4.0, height=2.6).to_dict()
    room["id"] = "w:room"

    robot = cobot_magic().to_dict()
    robot["id"] = "w:cobot"

    table = dining_table(width=1.2, depth=0.9, height=TABLE_TOP).to_dict()
    table["id"] = "w:table"
    table["pose"]["value"]["p"] = [0.92, 0.0, 0.0]

    dish = bowl().to_dict()
    dish["id"] = "w:bowl-1"
    dish["pose"]["value"]["p"] = list(BOWL_AT)

    saucer = plate().to_dict()
    saucer["id"] = "w:plate-1"
    saucer["pose"]["value"]["p"] = list(PLATE_AT)
    return {"mode": "world", "entities": [room, robot, table, dish, saucer]}


@pytest.fixture()
def rig():
    sim = WorldSim(_scene())
    sim.start()
    skills = CobotSkills(command=lambda cmd: sim.command("w:cobot", cmd),
                         state=lambda: sim.sensors("w:cobot"),
                         config=MotionConfig(tick_s=0.01))
    try:
        yield sim, skills
    finally:
        sim.stop()


def _bowl(sim: WorldSim) -> list[float]:
    state = sim.world_state()
    return next(e for e in state["entities"] if e["id"] == "w:bowl-1")["pose"]["value"]["p"]


def test_tcp_sits_between_the_finger_bodies(rig) -> None:
    sim, skills = rig
    tcp = np.asarray(skills.tcp("L"))
    left = np.asarray(sim.data.xpos[sim.model.body("w:cobot/L_link7").id])
    right = np.asarray(sim.data.xpos[sim.model.body("w:cobot/L_link8").id])
    assert np.linalg.norm(tcp - (left + right) / 2) < 5e-3, "TCP must be the finger midpoint"
    assert np.linalg.norm(np.asarray(TCP_OFFSET)) == pytest.approx(0.135, abs=1e-6)


def test_pick_and_place_moves_the_object(rig) -> None:
    """The full verb pair: descend, close, lift, carry, lower, open, retreat."""
    sim, skills = rig
    bowl_before = _bowl(sim)

    job = skills.start("pick", {"id": "r1", "plan": {"arm": "L", "at": BOWL_AT}})
    done = skills.wait(job.id, timeout=90)
    assert done.state == "succeeded", f"pick failed: {done.message}\n" + "\n".join(done.log)
    assert any(e["type"] == "grasp/attach" for e in sim.events), "the gripper closed on nothing"

    carried = skills.tcp("L")
    # Forward along the arm's own line: everything behind the chassis front is out of reach now.
    target = [carried[0] + 0.10, carried[1], carried[2] - skills.config.retreat_height]
    job = skills.start("place", {"id": "r2", "plan": {"arm": "L", "at": target}})
    done = skills.wait(job.id, timeout=90)
    assert done.state == "succeeded", f"place failed: {done.message}\n" + "\n".join(done.log)

    sim.step(60)
    bowl_after = _bowl(sim)
    moved = np.linalg.norm(np.subtract(bowl_after[:2], bowl_before[:2]))
    assert moved > 0.05, f"the bowl did not move on the table (displacement {moved:.3f} m)"
    assert any(e["type"] == "grasp/release" for e in sim.events), "the gripper never opened"


def test_pick_where_there_is_nothing_reports_a_failed_grasp(rig) -> None:
    sim, skills = rig
    # Table top, midway between the bowl and the plate: reachable with a vertical tool, and both
    # are farther than the 6 cm capture radius (computed, so a layout change cannot invalidate it).
    empty = [(BOWL_AT[0] + PLATE_AT[0]) / 2, (BOWL_AT[1] + PLATE_AT[1]) / 2, TABLE_TOP]
    assert min(np.hypot(empty[0] - BOWL_AT[0], empty[1] - BOWL_AT[1]),
               np.hypot(empty[0] - PLATE_AT[0], empty[1] - PLATE_AT[1])) > 0.07
    job = skills.start("pick", {"id": "r3", "plan": {"arm": "L", "at": empty}})
    done = skills.wait(job.id, timeout=90)
    assert done.state == "failed"
    assert "grabbed nothing" in done.message


def test_move_joints_verb_reaches_the_requested_posture(rig) -> None:
    sim, skills = rig
    target = dict(zip(skills.ik.joint_names("R"), skills.ik.neutral("R"), strict=True))
    job = skills.start("move_joints", {"id": "r4", "plan": {"arm": "R", "joints": target}})
    done = skills.wait(job.id, timeout=90)
    assert done.state == "succeeded", f"{done.message}\n" + "\n".join(done.log)
    for name, value in target.items():
        assert sim.sensors("w:cobot")["joints"][name] == pytest.approx(value, abs=0.06)


def test_joint_target_outside_the_urdf_range_is_refused(rig) -> None:
    sim, skills = rig
    target = dict(zip(skills.ik.joint_names("L"), skills.ik.neutral("L"), strict=True))
    target["L_joint2"] = 9.0
    job = skills.start("move_joints", {"id": "r5", "plan": {"arm": "L", "joints": target}})
    done = skills.wait(job.id, timeout=30)
    assert done.state == "failed"
    assert "outside" in done.message


def test_park_verb_returns_the_arm_to_the_rest_posture(rig) -> None:
    from steward.assets.cobot import HOME

    sim, skills = rig
    job = skills.start("park", {"id": "r6", "plan": {"arm": "L"}})
    done = skills.wait(job.id, timeout=90)
    assert done.state == "succeeded", f"{done.message}\n" + "\n".join(done.log)
    joints = sim.sensors("w:cobot")["joints"]
    for name, value in zip(skills.ik.joint_names("L"), HOME, strict=True):
        assert joints[name] == pytest.approx(value, abs=0.06)


def test_the_base_drives_to_a_waypoint_and_turns() -> None:
    """The mobile half of the twin: the chassis goes where it is told and says so in odometry.

    An empty room on purpose: turning in place beside a table sweeps the arms through it, which is
    the body's real constraint and belongs in a docking test, not in a chassis test."""
    from steward.assets.generators import building

    room = building(width=4.0, depth=4.0, height=2.6).to_dict()
    room["id"] = "w:room"
    robot = cobot_magic().to_dict()
    robot["id"] = "w:cobot"
    sim = WorldSim({"mode": "world", "entities": [room, robot]})
    sim.start()
    skills = CobotSkills(command=lambda cmd: sim.command("w:cobot", cmd),
                         state=lambda: sim.sensors("w:cobot"),
                         config=MotionConfig(tick_s=0.01))
    try:
        job = skills.start("navigate", {"id": "r5", "plan": {"at": [-0.35, 0.20, 0.6]}})
        done = skills.wait(job.id, timeout=90)
        assert done.state == "succeeded", f"{done.message}\n" + "\n".join(done.log)
        o = sim.sensors("w:cobot")["odometry"]
        assert np.hypot(o["x"] + 0.35, o["y"] - 0.20) < 0.05
        assert abs(o["yaw"] - 0.6) < 0.06
    finally:
        sim.stop()


def test_the_torso_rises_to_a_requested_height(rig) -> None:
    sim, skills = rig
    job = skills.start("lift", {"id": "r6", "plan": {"height": 0.95}})
    done = skills.wait(job.id, timeout=60)
    assert done.state == "succeeded", f"{done.message}\n" + "\n".join(done.log)
    assert sim.sensors("w:cobot")["joints"]["lift"] == pytest.approx(0.95, abs=0.02)
