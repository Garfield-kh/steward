"""Place seats mating screw holes instead of a free-hand offset."""

from __future__ import annotations

import math

import numpy as np
import pytest

from steward.assets.generators import steward_bot
from steward.ego.arm import Arm
from steward.ego.esim import Esim
from steward.ego.estimator import EgoEstimator, Track
from steward.ego.priors import screw_holes
from steward.math3d import yaw_from_quat


def _track(eid: str, category: str, pos: list[float], size: list[float], yaw: float = 0.0) -> Track:
    return Track(id=eid, category_counts={category: 5.0}, pos=np.array(pos, dtype=float),
                 var=np.full(3, 1e-4), yaw=yaw, yaw_var=1e-3, size=np.array(size, dtype=float),
                 observations=10, first_seen=0.0, last_seen=0.0, existence=1.0)


def _assembly_scene(held: str = "e:wing"):
    """Robot south of a table, same docking geometry as the wrist placement fixtures."""
    model = steward_bot().to_dict()
    est = EgoEstimator(robot_model=model)
    est.odometry = {"x": 0.0, "y": 1.0, "yaw": -math.pi / 2, "sigma": {"pos": 0.0, "yaw": 0.0}}
    est.joints = {"lift": 0.85, "reach": 0.1, "wrist_yaw": 0.0, "wrist_roll": 0.0}
    est.tracks["e:table"] = _track("e:table", "table", [0.0, 0.0, 0.375], [0.8, 0.8, 0.75])
    est.tracks["e:fuse"] = _track("e:fuse", "airplane_fuselage", [0.0, 0.1, 0.77], [0.18, 0.045, 0.04])
    est.tracks["e:wing"] = _track("e:wing", "airplane_wing", [0.15, 0.35, 0.755], [0.055, 0.20, 0.01], yaw=0.3)
    est.tracks["e:gear"] = _track("e:gear", "airplane_landing_gear", [-0.15, 0.35, 0.762], [0.03, 0.08, 0.025])
    est.tracks["e:body"] = _track("e:body", "car_body", [0.25, 0.1, 0.772], [0.16, 0.10, 0.045])
    est.tracks["e:wheel"] = _track("e:wheel", "car_wheel", [0.35, 0.3, 0.758], [0.05, 0.05, 0.016])
    est.held_id = held
    est.tracks[held].held = True
    return model, est.to_scene()


def _esim(model) -> Esim:
    esim = Esim(model)
    esim.arm = Arm(esim.arm.geometry)
    return esim


def test_category_priors_list_the_toy_screw_holes():
    assert {h["id"] for h in screw_holes("airplane_fuselage")} == {"wing_0", "wing_1", "gear_0", "gear_1"}
    assert {h["id"] for h in screw_holes("airplane_wing")} == {"mount_0", "mount_1"}
    assert len(screw_holes("car_body")) == 4
    assert screw_holes("car_wheel")[0]["id"] == "hub"


def test_ego_state_carries_screw_hole_affordances_from_priors():
    _, scene = _assembly_scene()
    holes = {a.value["screw-hole"]["id"]
             for a in scene.entity("e:fuse").rigid.links[0].affordances
             if isinstance(a.value, dict) and "screw-hole" in a.value}
    assert holes == {"wing_0", "wing_1", "gear_0", "gear_1"}


def test_place_wing_on_fuselage_seats_the_wing_mount_holes():
    model, scene = _assembly_scene("e:wing")
    receipt = _esim(model).predict(scene, "place", {"on": "e:fuse"})
    assert receipt["feasible"], receipt["issues"]
    assert receipt["plan"]["alignTo"] == "e:fuse"
    assert receipt["plan"]["mate"] is True
    assert any("aligned screw holes" in i for i in receipt["issues"])
    fuse = scene.entity("e:fuse")
    fx, fy = fuse.pose.value["p"][0], fuse.pose.value["p"][1]
    fyaw = yaw_from_quat(fuse.pose.value["q"])
    spot = receipt["plan"]["spot"]
    # Wing centre should sit 2 cm forward of the fuselage centre (wing holes at x=0.02).
    lx = math.cos(-fyaw) * (spot[0] - fx) - math.sin(-fyaw) * (spot[1] - fy)
    ly = math.sin(-fyaw) * (spot[0] - fx) + math.cos(-fyaw) * (spot[1] - fy)
    assert lx == pytest.approx(0.02, abs=1e-3)
    assert ly == pytest.approx(0.0, abs=1e-3)


def test_over_support_allows_a_wide_mate_whose_footprint_hangs_past_the_part():
    """A wing on a fuselage fails footprint containment; mate mode only needs the centre on it."""
    from steward.ego.skills import SkillExecutor

    class _Fake:
        def _gripper_xy(self):
            return np.array([0.02, 0.0])

    plan = {
        "spot": [0.02, 0.0], "gripSpot": [0.02, 0.0], "heldHalf": 0.10,
        "supportBox": {"cx": 0.0, "cy": 0.0, "yaw": 0.0, "hx": 0.09, "hy": 0.0225},
    }
    assert not SkillExecutor._over_support(_Fake(), plan)
    plan["mate"] = True
    assert SkillExecutor._over_support(_Fake(), plan)


def test_place_gear_on_table_aligns_to_fuselage_gear_holes():
    model, scene = _assembly_scene("e:gear")
    receipt = _esim(model).predict(scene, "place", {"on": "e:table", "alignTo": "e:fuse", "hole": "gear"})
    assert receipt["feasible"], receipt["issues"]
    fuse = scene.entity("e:fuse")
    spot = receipt["plan"]["spot"]
    assert spot[0] == pytest.approx(fuse.pose.value["p"][0], abs=1e-3)
    assert spot[1] == pytest.approx(fuse.pose.value["p"][1], abs=1e-3)


def test_place_wheel_aligns_hub_to_named_car_mount():
    model, scene = _assembly_scene("e:wheel")
    receipt = _esim(model).predict(scene, "place", {"on": "e:table", "alignTo": "e:body", "hole": "wheel_1"})
    assert receipt["feasible"], receipt["issues"]
    body = scene.entity("e:body")
    byaw = yaw_from_quat(body.pose.value["q"])
    bx, by = body.pose.value["p"][0], body.pose.value["p"][1]
    # wheel_1 is at [0.05, -0.048] in the body frame.
    want = (bx + math.cos(byaw) * 0.05 - math.sin(byaw) * (-0.048),
            by + math.sin(byaw) * 0.05 + math.cos(byaw) * (-0.048))
    spot = receipt["plan"]["spot"]
    assert spot[0] == pytest.approx(want[0], abs=1e-3)
    assert spot[1] == pytest.approx(want[1], abs=1e-3)


def test_align_holes_false_keeps_a_manual_offset_on_a_table():
    model, scene = _assembly_scene("e:wing")
    receipt = _esim(model).predict(scene, "place", {"on": "e:table", "alignTo": "e:fuse", "alignHoles": False,
                                                    "offset": [-0.28, -0.28], "clearance": 0.0})
    assert receipt["feasible"], receipt["issues"]
    assert "alignTo" not in receipt["plan"]
    table = scene.entity("e:table")
    tyaw = yaw_from_quat(table.pose.value["q"])
    tx, ty = table.pose.value["p"][0], table.pose.value["p"][1]
    spot = receipt["plan"]["spot"]
    lx = math.cos(-tyaw) * (spot[0] - tx) - math.sin(-tyaw) * (spot[1] - ty)
    ly = math.sin(-tyaw) * (spot[0] - tx) + math.cos(-tyaw) * (spot[1] - ty)
    assert lx == pytest.approx(-0.28, abs=1e-3)
    assert ly == pytest.approx(-0.28, abs=1e-3)
