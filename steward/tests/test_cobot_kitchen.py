"""The twin in the chores kitchen: the coloured bowls on a table its two fixed arms can reach.

The chores kitchen is built for a mobile robot that drives to a sideboard and a cabinet. This
scene keeps the tableware and drops the mobile base, so what is under test is that the layout sits
inside the twin's reachable tabletop band and that the nesting mission really stacks the bowls.
"""

import math

import numpy as np
import pytest

from steward.assets.cobot import cobot_magic
from steward.authoring.service import AuthoringService, build_cobot_kitchen
from steward.ego.cobot_mission import nest_bowls
from steward.ego.cobot_skills import CobotSkills, MotionConfig, _tool_down
from steward.ego.estimator import EgoEstimator
from steward.world.sim import WorldSim


@pytest.fixture(scope="module")
def rig(tmp_path_factory):
    root = tmp_path_factory.mktemp("cobot-kitchen")
    build_cobot_kitchen(AuthoringService(root))
    scene = AuthoringService(root).load_scenario("cobot-kitchen")["sceneState"]
    sim = WorldSim(scene, speed=3.0)
    sim.start()
    robot = next(e["id"] for e in scene["entities"] if e["role"] == "robot")
    skills = CobotSkills(command=lambda cmd: sim.command(robot, cmd),
                         state=lambda: sim.sensors(robot), config=MotionConfig(tick_s=0.01))
    est = EgoEstimator(robot_model=cobot_magic().to_dict())
    try:
        yield sim, robot, skills, est
    finally:
        sim.stop()


def _objects(sim: WorldSim) -> dict[str, dict]:
    """Tableware by id: everything that is not the table, the room or the robot."""
    out = {}
    for ent in sim.world_state()["entities"]:
        if ent["id"].startswith("w:") and ent["category"]["value"] in ("bowl", "wine_glass", "mug"):
            out[ent["id"]] = ent
    return out


def test_the_three_bowls_and_the_other_tableware_sit_within_the_twins_reach(rig) -> None:
    sim, _, skills, _ = rig
    objects = _objects(sim)
    assert {"w:bowl-red", "w:bowl-green", "w:bowl-blue", "w:wine-glass", "w:mug"} <= set(objects)
    radii = {eid: float(sim.entity_obb(eid).size[0]) / 2.0 for eid in objects}
    # The task grasps the three bowls: each must be reachable by the arm on its side, at the pose
    # it stands in. Asked of the solver, not of a hardcoded band, so a layout change cannot pass
    # this test while being unreachable.
    ik, tool = skills.ik, _tool_down()
    ik.set_base(0.0, 0.0, 0.0, float(sim.sensors("w:cobot-1")["joints"]["lift"]))
    for eid in ("w:bowl-red", "w:bowl-green", "w:bowl-blue"):
        x, y, z = (float(v) for v in objects[eid]["pose"]["value"]["p"])
        ik.solve_tool("L" if y >= 0 else "R", [x, y, z + 0.03], tool, max_tilt=0.7)
    # The glass and the mug are left on the table; they stand beyond the arms' strip, in view.
    for eid in ("w:wine-glass", "w:mug"):
        assert float(objects[eid]["pose"]["value"]["p"][0]) > 0.60
    # No two pieces of tableware overlap: they must each be seen and grasped on their own.
    ids = sorted(objects)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            pa = objects[a]["pose"]["value"]["p"]
            pb = objects[b]["pose"]["value"]["p"]
            gap = math.hypot(float(pa[0]) - float(pb[0]), float(pa[1]) - float(pb[1]))
            assert gap >= radii[a] + radii[b] - 0.005, f"{a} and {b} overlap ({gap:.3f} m apart)"


def test_the_largest_bowl_sits_on_the_line_both_arms_can_reach(rig) -> None:
    sim, _, _, _ = rig
    objects = _objects(sim)
    bowls = sorted((eid for eid in objects if objects[eid]["category"]["value"] == "bowl"),
                   key=lambda eid: -sim.entity_obb(eid).size[0])
    assert [round(sim.entity_obb(eid).size[0], 2) for eid in bowls] == [0.30, 0.20, 0.12]
    # The nesting target is this bowl's centre: the arm carrying either smaller bowl has to reach
    # it, so it stands on the shared centre line between the arms.
    y = float(objects[bowls[0]]["pose"]["value"]["p"][1])
    assert abs(y) <= 0.05, f"the largest bowl is at y={y:.2f}, off the shared centre line"


def test_the_twin_stacks_the_three_bowls_largest_first(rig) -> None:
    """The whole live path: RGB-D only, anonymous masks, and bowls it measured itself."""
    from steward.ego.perception import decode_rgbd

    sim, robot, skills, est = rig
    frame = lambda: decode_rgbd(sim.rgbd(robot, camera="top_camera"))
    result = nest_bowls(est, skills, frame, retries=1, rgbd=True)
    assert result["ok"], "\n".join(result["log"])
    sim.step(120)
    heights = {eid: float(e["pose"]["value"]["p"][2]) for eid, e in _objects(sim).items()}
    red, green, blue = heights["w:bowl-red"], heights["w:bowl-green"], heights["w:bowl-blue"]
    # Each bowl's pose is at its own base: red on the table, green on red's floor, blue on green's.
    assert green - red == pytest.approx(0.008, abs=0.006), f"green at {green:.3f} on red at {red:.3f}"
    assert blue - green == pytest.approx(0.008, abs=0.006), f"blue at {blue:.3f} on green at {green:.3f}"


def test_the_mission_reports_a_missing_bowl_instead_of_guessing(rig) -> None:
    """A look that sees no bowls must not send an arm at a remembered position."""
    sim, robot, skills, est = rig
    blocked = lambda: {**sim.sensors(robot), "detections": []}
    result = nest_bowls(est, skills, blocked, retries=1)
    assert result["ok"] is False
    assert any("bowl(s) in view" in line for line in result["log"]), result["log"]
