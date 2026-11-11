"""Ideal detections report screw holes; Ego-State exposes them as screwHoles."""

from __future__ import annotations

import math

import numpy as np

from steward.assets.generators import airplane_fuselage, steward_bot
from steward.ego.estimator import EgoEstimator, Track
from steward.ego.runtime import EgoRuntime
from steward.math3d import quat_from_yaw
class _Link:
    def __init__(self, sensors):
        self._sensors = sensors

    def sensors(self):
        return self._sensors()


def test_world_entity_screw_holes_are_read_from_template_affordances():
    from steward.world.sim import WorldSim

    holes = WorldSim._screw_holes(airplane_fuselage(), None)
    assert {h["id"] for h in holes} == {"wing_0", "wing_1", "gear_0", "gear_1"}


def test_estimator_fuses_detection_screw_holes_onto_the_track():
    model = steward_bot().to_dict()
    est = EgoEstimator(robot_model=model)
    est.stamp = 1.0
    det = {
        "category": "airplane_fuselage", "size": [0.18, 0.045, 0.04], "color": "yellow",
        "screwHoles": [{"id": "wing_0", "p": [0.02, 0.012, 0.038]},
                       {"id": "wing_1", "p": [0.02, -0.012, 0.038]}],
    }
    track = est._new_track(det, np.array([0.0, 0.2, 0.77]), 0.0, 0.01)
    assert {h["id"] for h in track.screw_holes} == {"wing_0", "wing_1"}
    est._fuse(track, {**det, "screwHoles": [{"id": "gear_0", "p": [0.0, 0.012, 0.002]}]},
              track.pos, 0.0, 0.01)
    assert {h["id"] for h in track.screw_holes} == {"gear_0"}
    ent = est.to_scene().entity(track.id)
    observed = [a for a in ent.rigid.links[0].affordances
                if isinstance(a.value, dict) and "screw-hole" in a.value]
    assert {a.value["screw-hole"]["id"] for a in observed} == {"gear_0"}
    assert all(a.src == "observed" for a in observed)


def test_summary_view_splits_string_affordances_from_screw_holes():
    from steward.scene.schema import Est
    aff = [Est(value="graspable", src="prior"),
           Est(value={"screw-hole": {"id": "mount_0", "p": [0.0, 0.012, 0.008]}}, src="observed"),
           Est(value={"screw-hole": {"id": "mount_1", "p": [0.0, -0.012, 0.008]}}, src="observed")]
    holes = EgoRuntime._screw_holes_view(aff)
    assert {h["id"] for h in holes} == {"mount_0", "mount_1"}
    assert all(h["source"] == "observed" for h in holes)


def test_prior_holes_appear_until_a_detection_reports_them():
    model = steward_bot().to_dict()
    est = EgoEstimator(robot_model=model)
    est.tracks["e:1"] = Track(
        id="e:1", category_counts={"airplane_wing": 5.0}, pos=np.array([0.0, 0.2, 0.75]),
        var=np.full(3, 1e-4), yaw=0.0, yaw_var=1e-3, size=np.array([0.055, 0.20, 0.01]),
        observations=5, first_seen=0.0, last_seen=0.0, existence=1.0,
    )
    ent = est.to_scene().entity("e:1")
    holes = [a for a in ent.rigid.links[0].affordances
             if isinstance(a.value, dict) and "screw-hole" in a.value]
    assert {a.value["screw-hole"]["id"] for a in holes} == {"mount_0", "mount_1"}
    assert all(a.src == "prior" for a in holes)


def test_observe_forwards_detection_screw_holes():
    sensors = {
        "stamp": 2.0,
        "detections": [
            {"category": "car_wheel", "range": 0.6, "camera": "head_camera",
             "screwHoles": [{"id": "hub", "p": [0.0, 0.0, 0.008]}]},
            {"category": "table", "range": 1.2, "camera": "head_camera"},
        ],
    }
    runtime = EgoRuntime.__new__(EgoRuntime)
    runtime.link = _Link(lambda: sensors)
    runtime.out_dir = None
    runtime.perception_mode = "detections"
    runtime._cameras = lambda _frame: ["head_camera"]
    out = EgoRuntime.observe(runtime)
    wheel = next(v for v in out["visible"] if v["category"] == "car_wheel")
    assert wheel["screwHoles"] == [{"id": "hub", "p": [0.0, 0.0, 0.008]}]
    assert "screwHoles" not in next(v for v in out["visible"] if v["category"] == "table")
