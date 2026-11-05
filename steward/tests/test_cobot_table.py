"""The twin at its own workstation: the bowl-on-plate errand on the real cobot-table layout.

This runs the scene the console actually loads, not a hand-set test layout. When the scene was
re-laid for the mobile body the plate landed on the right arm's side while the errand is one
arm's job -- the left arm picked the bowl and then could not cross the body to the plate, and
because no test exercised the real spec, the mission failed live on the console twice in a row
before anyone noticed.
"""

import pytest

from steward.assets.cobot import cobot_magic
from steward.authoring.service import AuthoringService, build_cobot_workstation
from steward.ego.cobot_mission import bowl_onto_plate
from steward.ego.cobot_skills import CobotSkills, MotionConfig, _tool_down
from steward.ego.estimator import EgoEstimator
from steward.world.sim import WorldSim


@pytest.fixture(scope="module")
def rig(tmp_path_factory):
    root = tmp_path_factory.mktemp("cobot-table")
    build_cobot_workstation(AuthoringService(root))
    scene = AuthoringService(root).load_scenario("cobot-table")["sceneState"]
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


def _entity(sim: WorldSim, entity_id: str) -> dict:
    return next(e for e in sim.world_state()["entities"] if e["id"] == entity_id)


def test_the_bowl_and_the_plate_stand_where_one_arm_can_work_both(rig) -> None:
    """The errand is one arm's job: both pieces on that arm's side, at poses the solver accepts."""
    sim, robot, skills, _ = rig
    bowl = _entity(sim, "w:bowl-1")["pose"]["value"]["p"]
    plate = _entity(sim, "w:plate-1")["pose"]["value"]["p"]
    same_side = (bowl[1] > 0) == (plate[1] > 0)
    assert same_side, f"bowl at y={bowl[1]:.2f} and plate at y={plate[1]:.2f} split the arms"
    arm = "L" if bowl[1] >= 0 else "R"
    ik = skills.ik
    ik.set_base(0.0, 0.0, 0.0, float(sim.sensors("w:cobot-1")["joints"]["lift"]))
    for at in (bowl, plate):
        ik.solve_tool(arm, [at[0], at[1], at[2] + 0.03], _tool_down(), max_tilt=0.7)


def test_the_twin_puts_the_bowl_on_the_plate(rig) -> None:
    """The whole live path against the real scene: RGB-D, its own estimates, World State checked."""
    from steward.ego.perception import decode_rgbd

    sim, robot, skills, est = rig
    frame = lambda: decode_rgbd(sim.rgbd(robot, "top_camera"))
    result = bowl_onto_plate(est, skills, frame, retries=1, rgbd=True)
    assert result["ok"], "\n".join(result["log"])
    sim.step(120)

    support = [r for r in sim.world_state()["relations"]
               if r["type"] == "supported-by" and r["a"]["entity"] == "w:bowl-1"]
    assert support and support[0]["b"]["entity"] == "w:plate-1", support
    bowl, plate = _entity(sim, "w:bowl-1"), _entity(sim, "w:plate-1")
    assert bowl["pose"]["value"]["p"][:2] == pytest.approx(plate["pose"]["value"]["p"][:2], abs=0.05), \
        f"bowl at {bowl['pose']['value']['p']}, plate at {plate['pose']['value']['p']}"
