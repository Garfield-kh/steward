"""Place flips reverse a held board's physical normal before release, without a model call."""

import threading

import numpy as np
import pytest

from steward.assets.generators import building, dining_table, printed_board, steward_bot
from steward.ego.esim import Esim
from steward.ego.estimator import EgoEstimator, Track
from steward.ego.skills import MotionConfig, SkillExecutor
from steward.math3d import quat_rotate
from steward.world.sim import PerceptionConfig, WorldSim


class SteppedClock:
    """Advance real MuJoCo physics only when the skill requests its next clock instant."""

    def __init__(self, sim):
        self.sim = sim

    def arm(self, *args, **kwargs):
        pass  # One skill owns stepping; there are no competing clock clients.

    def release(self, *args):
        pass  # No clock lease is acquired by this single-client fixture.

    def wait_until(self, at=None, *args, **kwargs):
        if at is not None:
            self.sim.step_seconds(max(0, at - self.sim.time))
        return self.sim.time


@pytest.mark.parametrize("destination", ["table", "floor"])
@pytest.mark.parametrize("face_up", [False, True])
@pytest.mark.parametrize("grasp_offset", [0.0, 0.02])
def test_place_flip_reverses_the_physical_face(destination, face_up, grasp_offset, record_property):
    room, robot, table, piece = (factory().to_dict() for factory in
                                 (building, steward_bot, dining_table, printed_board))
    for entity, eid in zip((room, robot, table, piece), ("w:room", "w:robot", "w:table", "w:piece"), strict=True):
        entity["id"] = eid
    table["pose"]["value"]["p"] = [0.9, 0.0, 0.0]
    piece["pose"]["value"] = {"p": [0.36, grasp_offset, 0.95], "q": [1.0, 0.0, 0.0, 0.0] if face_up else [0.0, 1.0, 0.0, 0.0]}
    next(j for j in robot["body"]["joints"] if j["id"] == "lift")["q"]["value"] = [0.95]
    sim = WorldSim({"mode": "world", "entities": [room, robot, table, piece]},
                   perception=PerceptionConfig(odom_pos_sigma=0, odom_yaw_sigma=0))
    try:
        sim.command("w:robot", {"gripper": "close"})
        sim.step_seconds(0.05)
        assert sim.sensors("w:robot")["gripper"]["holding"], "fixture must start with the board grasped"
        est = EgoEstimator(robot_model=robot)
        frame = {**sim.sensors("w:robot"), "detections": []}
        est.update(frame)
        for eid, category, position, size in (("e:table", "table", [0.9, 0.0, 0.375], [1.2, 0.8, 0.75]),
                                               ("e:piece", "board", [0.36, grasp_offset, 0.95], [0.12, 0.12, 0.006])):
            est.tracks[eid] = Track(eid, {category: 5.0}, np.array(position), np.full(3, 1e-6), 0.0,
                                    1e-4, np.array(size), 10, 0.0, sim.time, existence=1.0)
        est.held_id = "e:piece"
        est.tracks["e:piece"].held = True
        esim = Esim(robot)
        params = {"on": "e:table", "offset": [-0.3, 0.0]} if destination == "table" else {"at": [0.0, -0.8]}
        receipt = esim.predict(est.to_scene(), "place", {**params, "flip": True})
        assert receipt["feasible"], receipt["issues"]
        before = float(quat_rotate(sim._entity_pose("w:piece")["q"], [0, 0, 1])[2])
        settled = threading.Event()
        executor = SkillExecutor(lambda command: sim.command("w:robot", command),
                                 lambda: esim.arm.virtualize(sim.sensors("w:robot")), lambda eid: eid,
                                 config=MotionConfig(tick_s=0.02), clock=SteppedClock(sim),
                                 on_finish=lambda job: settled.set())
        try:
            job = executor.start("place", receipt)
            assert settled.wait(30), "physical skill did not settle"
            assert job.state == "succeeded", (job.message, job.log)
            sim.step_seconds(1.0)
            after = float(quat_rotate(sim._entity_pose("w:piece")["q"], [0, 0, 1])[2])
            record_property("before_normal_z", before)
            record_property("after_normal_z", after)
            record_property("destination", destination)
            record_property("grasp_offset_m", grasp_offset)
            assert before * after < -0.9, f"place returned success without a half turn: normal {before:.3f} -> {after:.3f}"
            assert not sim.sensors("w:robot")["gripper"]["holding"]
            z = sim._entity_pose("w:piece")["p"][2]
            assert z > 0.7 if destination == "table" else z < 0.02
            expected = receipt["plan"].get("spot", receipt["plan"].get("floor"))
            assert np.linalg.norm(np.array(sim._entity_pose("w:piece")["p"][:2]) - expected) < 0.03
        finally:
            executor.cancel()
            assert settled.wait(30), "skill cleanup did not settle"
    finally:
        sim.stop()
