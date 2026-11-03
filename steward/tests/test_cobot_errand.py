"""The mobile twin's errand: take the bowl from a table, drive across the room, put it on a counter.

This is the mission the fixed-arm twin could not do at all: both the distance and the 15 cm height
difference between the stations are outside an arm's reach, so a pass means the chassis drove and
the torso rose -- checked against World State, not against the mission's own log.
"""

import math

import pytest

from steward.assets.cobot import cobot_magic
from steward.authoring.service import COBOT_ERRAND_MISSION, AuthoringService, build_cobot_errand
from steward.ego.cobot_mission import carry_to_station
from steward.ego.cobot_skills import CobotSkills, MotionConfig
from steward.ego.estimator import EgoEstimator
from steward.ego.perception import decode_rgbd
from steward.world.sim import WorldSim

COUNTER_TOP = 0.90


@pytest.fixture(scope="module")
def rig(tmp_path_factory):
    root = tmp_path_factory.mktemp("cobot-errand")
    build_cobot_errand(AuthoringService(root))
    scene = AuthoringService(root).load_scenario("cobot-errand")["sceneState"]
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


def test_the_counter_is_metres_away_and_higher_than_the_table(rig) -> None:
    sim, robot, _, _ = rig
    bowl = _entity(sim, "w:bowl-1")["pose"]["value"]["p"]
    assert bowl[2] == pytest.approx(0.75, abs=0.01), "the bowl stands on the 0.75 m table"
    place = COBOT_ERRAND_MISSION["place"]
    odo = sim.sensors(robot)["odometry"]
    # The drop point is 2.8 m from where the twin starts: no arm pose reaches it from here.
    assert math.hypot(place[0] - odo["x"], place[1] - odo["y"]) > 2.0
    # And it is 15 cm higher than the tabletop, so the drop is also a change of working height.
    assert place[2] - COUNTER_TOP == pytest.approx(0.06, abs=0.01)
    assert COUNTER_TOP - 0.75 == pytest.approx(0.15, abs=0.01)
    # The route goes around the table rather than through it: a straight line would cross the
    # tabletop's footprint (x 0.30-1.50), and the waypoints dip to y = -1.10 to clear it.
    assert min(p[1] for p in COBOT_ERRAND_MISSION["drive"]["path"]) < -1.0


def test_the_twin_carries_the_bowl_to_the_counter(rig) -> None:
    """Perception for the bowl, the map for the drop: the whole live path, including the drive."""
    sim, robot, skills, est = rig
    frame = lambda: decode_rgbd(sim.rgbd(robot, camera="top_camera"))
    result = carry_to_station(est, skills, frame, rgbd=True, retries=1, **COBOT_ERRAND_MISSION)
    assert result["ok"], "\n".join(result["log"])
    sim.step(150)  # let the released bowl settle

    odo = sim.sensors(robot)["odometry"]
    assert odo["x"] == pytest.approx(COBOT_ERRAND_MISSION["drive"]["at"][0], abs=0.03)
    # The counter is above the shoulders' home height, so the torso rose for the place.
    assert sim.sensors(robot)["joints"]["lift"] == pytest.approx(COBOT_ERRAND_MISSION["lift"], abs=0.01)
    bowl = _entity(sim, "w:bowl-1")["pose"]["value"]["p"]
    place = COBOT_ERRAND_MISSION["place"]
    # Where the bowl *lands* repeats the offset between the centre it was perceived at and the
    # pinch point that held it, so this is a grasp-accuracy bound, not a drive bound: 5 cm on a
    # 1.8 m counter, for a bowl of 16 cm.
    assert bowl[0] == pytest.approx(place[0], abs=0.05), f"bowl landed at {bowl}"
    assert bowl[1] == pytest.approx(place[1], abs=0.05), f"bowl landed at {bowl}"
    assert bowl[2] == pytest.approx(COUNTER_TOP, abs=0.01), "the bowl rests on the counter's top"
    relations = sim.world_state()["relations"]
    support = [r for r in relations if r["type"] == "supported-by" and r["a"]["entity"] == "w:bowl-1"]
    assert support and support[0]["b"]["entity"] == "w:counter-1", support
