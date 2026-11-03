from __future__ import annotations

import numpy as np

from steward.assets.generators import steward_bot
from steward.ego.esim import Esim
from steward.ego.estimator import EgoEstimator
from steward.scene.validate import validate_ego
from steward.world.sim import WorldSim


def _observe(sim: WorldSim, est: EgoEstimator, frames: int) -> None:
    for _ in range(frames):
        est.update(sim.sensors("w:robot-1"))
        sim.step_seconds(0.1)


def _cup_track(est: EgoEstimator):
    """The track most often labelled cup; the detector occasionally confuses cups and bowls."""
    return max((t for t in est.tracks.values() if "cup" in t.category_counts),
               key=lambda t: t.category_counts["cup"])


def test_ego_state_converges_toward_ground_truth(kitchen_scene):
    sim = WorldSim(kitchen_scene, seed=3)
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    sim.command("w:robot-1", {"base": [-0.6, -0.2, 0.9]})
    sim.step_seconds(3.0)
    gt = np.asarray(sim.entity_obb("w:cup-1").center)
    _observe(sim, est, 3)
    early = float(np.linalg.norm(_cup_track(est).pos - gt))
    early_sigma = _cup_track(est).sigma
    _observe(sim, est, 57)
    late = float(np.linalg.norm(_cup_track(est).pos - gt))
    assert late < early
    assert _cup_track(est).sigma < early_sigma / 3
    assert late < 0.02


def test_ego_state_is_partial_and_leak_free(kitchen_scene):
    sim = WorldSim(kitchen_scene, seed=4)
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    _observe(sim, est, 5)
    es = est.to_scene().to_dict()
    assert validate_ego(es) == []
    categories = {e["category"]["value"] for e in es["entities"]}
    assert not categories & {"cup", "counter"}, "the counter and its cup are outside the initial field of view"


def test_esim_refuses_uncertain_pick_and_accepts_after_observation(kitchen_scene):
    sim = WorldSim(kitchen_scene, seed=5)
    model = steward_bot().to_dict()
    est = EgoEstimator(robot_model=model)
    esim = Esim(model)
    sim.command("w:robot-1", {"base": [-1.8, -1.2, 1.2]})
    sim.step_seconds(3.0)
    _observe(sim, est, 4)
    cup_id = _cup_track(est).id
    far = esim.predict(est.to_scene(), "pick", {"target": cup_id})
    assert far["feasible"] and far["risk"] > 0.35
    sim.command("w:robot-1", {"base": [-0.5, 1.0, 1.5708]})
    sim.step_seconds(4.0)
    _observe(sim, est, 30)
    near = esim.predict(est.to_scene(), "pick", {"target": cup_id})
    assert near["feasible"] and near["risk"] < 0.05
    assert near["plan"]["support"] != cup_id


def test_translate_keeps_heading_in_the_odometry_yaw_frame():
    from steward.ego.skills import SkillExecutor

    commands: list[dict] = []
    stamp = {"t": 0.0}

    def state() -> dict:
        stamp["t"] += 1.0
        return {"stamp": stamp["t"], "odometry": {"x": 0.0, "y": 0.0, "yaw": 2 * 3.14159265 + 0.1}, "joints": {},
                "gripper": {}, "entities": {}, "toolOffset": 0.26}

    executor = SkillExecutor(commands.append, state, lambda _id: None)
    executor.config.tick_s = 0.0
    executor._translate_to(0.0, 0.0, 0.1)
    assert abs(commands[-1]["base"][2] - (2 * 3.14159265 + 0.1)) < 1e-6


def test_esim_place_requires_holding(kitchen_scene):
    sim = WorldSim(kitchen_scene, seed=6)
    model = steward_bot().to_dict()
    est = EgoEstimator(robot_model=model)
    _observe(sim, est, 3)
    table = next(t for t in est.tracks.values() if t.category == "table")
    receipt = Esim(model).predict(est.to_scene(), "place", {"on": table.id})
    assert not receipt["feasible"]
    assert "not holding" in receipt["issues"][0]


def test_motion_state_reads_the_live_body_not_the_perception_clock(kitchen_scene):
    """A real segmenter can be seconds behind. Motion grades itself on the body's own header;
    the belief only supplies where the objects are."""
    from steward.ego.runtime import EgoRuntime, RobotLink

    runtime = EgoRuntime(RobotLink("http://127.0.0.1:1"), steward_bot().to_dict())
    runtime.estimator.stamp = 5.0
    runtime.estimator.odometry = {"x": 0.0, "y": 0.0, "yaw": 0.0}

    class Live:
        @staticmethod
        def sensors() -> dict:
            return {"stamp": 9.0, "odometry": {"x": 1.0, "y": 2.0, "yaw": 0.5},
                    "joints": {"lift": 0.4, "reach": 0.2},
                    "gripper": {"state": "closed", "holding": True}}

    class Dead:
        @staticmethod
        def sensors() -> dict:
            raise OSError("robot port unavailable")

    runtime.link = Live()
    state = runtime._motion_state()
    assert state["stamp"] == 9.0 and state["odometry"] == {"x": 1.0, "y": 2.0, "yaw": 0.5}
    assert state["joints"] == {"lift": 0.4, "reach": 0.2} and state["gripper"]["holding"]
    runtime.link = Dead()
    assert runtime._motion_state()["stamp"] == 5.0, "an unreachable body leaves the belief as the answer"
