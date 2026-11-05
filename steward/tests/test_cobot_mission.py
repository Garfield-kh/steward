"""The closed loop: the twin picks up the bowl where it *believes* it is, and puts it on the plate."""

import numpy as np
import pytest

from steward.assets.cobot import cobot_magic
from steward.ego.cobot_mission import bowl_onto_plate, perceive
from steward.ego.cobot_skills import CobotSkills, MotionConfig
from steward.ego.estimator import EgoEstimator
from steward.world.sim import WorldSim

from test_cobot_skills import _scene


@pytest.fixture()
def rig():
    sim = WorldSim(_scene())
    sim.start()
    skills = CobotSkills(command=lambda cmd: sim.command("w:cobot", cmd),
                         state=lambda: sim.sensors("w:cobot"), config=MotionConfig(tick_s=0.01))
    est = EgoEstimator(robot_model=cobot_magic().to_dict())
    try:
        yield sim, est, skills
    finally:
        sim.stop()


def _truth(sim: WorldSim, entity_id: str) -> list[float]:
    from steward.assets.cobot import TCP_OFFSET  # noqa: F401 - documentation of the grasp frame

    state = sim.world_state()
    pose = next(e for e in state["entities"] if e["id"] == entity_id)["pose"]["value"]["p"]
    return list(pose)


def test_the_robot_sees_the_tableware_where_it_is(rig) -> None:
    sim, est, _ = rig
    seen = perceive(est, lambda: sim.sensors("w:cobot"), frames=10)
    assert "bowl" in seen and "plate" in seen, f"categories seen: {sorted(seen)}"
    for name, entity in (("bowl", "w:bowl-1"), ("plate", "w:plate-1")):
        truth = np.asarray(_truth(sim, entity))
        error = float(np.linalg.norm(seen[name].pos[:2] - truth[:2]))
        assert error < 0.02, f"{name} believed {seen[name].pos} vs truth {truth} ({error*100:.1f} cm)"


def test_a_look_reports_only_what_this_look_saw(rig) -> None:
    """A stale track keeps its position and decays by 3% a frame, so it stays "alive" for ~100
    frames. Reporting it sends the next grasp to where the object used to be: the fingers close on
    nothing while the log reads "at the bowl"."""
    sim, est, _ = rig
    assert "bowl" in perceive(est, lambda: sim.sensors("w:cobot"), frames=6)
    blocked = lambda: {**sim.sensors("w:cobot"), "detections": []}  # an arm in front of the camera
    assert "bowl" not in perceive(est, blocked, frames=6)
    assert any(track.existence > 0.4 for track in est.tracks.values()), \
        "the track should still be alive in the belief; otherwise this test proves nothing"


def test_mission_picks_from_belief_and_lands_on_the_plate(rig) -> None:
    sim, est, skills = rig
    plate_truth = _truth(sim, "w:plate-1")
    result = bowl_onto_plate(est, skills, lambda: sim.sensors("w:cobot"))
    assert result["ok"], "\n".join(result["log"])
    sim.step(60)
    bowl = _truth(sim, "w:bowl-1")
    plate = _truth(sim, "w:plate-1")
    assert np.linalg.norm(np.subtract(bowl[:2], plate[:2])) < 0.06, f"bowl {bowl} is not on the plate {plate}"
    assert bowl[2] > plate[2] + 0.005, "the bowl must rest on top of the plate, not beside it"
    # The plate never moved: whatever the robot believed, it gripped at plate height.
    assert np.linalg.norm(np.subtract(plate[:2], plate_truth[:2])) < 0.02
