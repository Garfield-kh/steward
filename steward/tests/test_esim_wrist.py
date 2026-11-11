"""Held-object planning preserves orientation and checks the route from physical encoders."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from steward.assets.generators import steward_bot
from steward.ego.arm import Arm
from steward.ego.esim import Esim
from steward.ego.estimator import EgoEstimator, Track
from steward.ego.planner import entity_box
from steward.scene.schema import exact


def _scene(yaw=0.0, roll=0.0):
    model = steward_bot().to_dict()
    est = EgoEstimator(robot_model=model)
    est.odometry = {"x": 0.0, "y": 1.0, "yaw": -math.pi / 2, "sigma": {"pos": 0.0, "yaw": 0.0}}
    est.joints = {"lift": 0.85, "reach": 0.1, "wrist_yaw": yaw, "wrist_roll": roll}
    for eid, category, pos, size in [("e:4", "table", [0.0, 0.0, 0.375], [0.6, 0.6, 0.75]),
                                     ("e:3", "puzzle_piece", [0.0, 0.6, 0.85], [0.115, 0.115, 0.006])]:
        est.tracks[eid] = Track(id=eid, category_counts={category: 5.0}, pos=np.array(pos),
                               var=np.full(3, 1e-4), yaw=-math.pi / 2, yaw_var=1e-3, size=np.array(size),
                               observations=10, first_seen=0.0, last_seen=0.0, existence=1.0)
    est.held_id, est.tracks["e:3"].held = "e:3", True
    return model, est.to_scene()


class OrientationArm(Arm):
    def __init__(self, geometry, accepts=lambda yaw, roll: True):
        super().__init__(geometry)
        self.accepts = accepts
        self.paths = []

    def reachable(self, lift, reach, wrist_yaw=0.0, wrist_roll=0.0, wrist_pitch=0.0):
        return (super().reachable(lift, reach, wrist_yaw, wrist_roll, wrist_pitch)
                and self.accepts(wrist_yaw, wrist_roll))

    def reachable_path(self, targets, *, start_joints=None):
        self.paths.append((targets, start_joints))
        return super().reachable_path(targets, start_joints=start_joints)


@pytest.mark.parametrize("case,yaw,roll", [("yaw", 1.1, 0.0), ("roll", 0.0, 0.4)])
def test_held_orientation_can_make_a_zero_orientation_endpoint_unreachable(case, yaw, roll):
    model, scene = _scene(yaw, roll)
    esim = Esim(model)
    esim.arm = OrientationArm(esim.arm.geometry, lambda y, r: abs(y) < 0.1 and abs(r) < 0.1)
    assert esim.arm.reachable(0.8, 0.3)
    receipt = esim.predict(scene, "place", {"on": "e:4", "offset": [0.0, 0.1]})
    expected = json.loads((Path(__file__).parent / "expected/wrist-placement.json").read_text())
    assert {"feasible": receipt["feasible"], "issues": receipt["issues"]} == expected[case]


def test_dock_selection_uses_required_tool_yaw_not_only_its_scalar_limit():
    model, scene = _scene()
    esim = Esim(model)
    esim.arm = OrientationArm(esim.arm.geometry, lambda yaw, roll: abs(yaw) < 0.6)
    receipt = esim.predict(scene, "place", {"on": "e:4", "offset": [0.0, 0.1], "yaw": 0.0})
    assert receipt["feasible"], receipt["issues"]
    assert receipt["plan"]["wristYaw"] == pytest.approx(0.0)
    assert receipt["plan"]["yaw"] == pytest.approx(0.0)


def test_placement_checks_rotation_before_lowering_from_current_encoders():
    model, scene = _scene(0.3, 0.2)
    esim = Esim(model)
    arm = esim.arm = OrientationArm(esim.arm.geometry)
    receipt = esim.predict(scene, "place", {"on": "e:4", "offset": [0.0, 0.1], "yaw": 0.2})
    assert receipt["feasible"], receipt["issues"]
    transitions = [(path, start) for path, start in arm.paths if start is not None]
    assert transitions
    for path, start in transitions:
        assert path[0]["wrist_yaw"] == start["wrist_yaw"] == 0.3
        assert all(p["wrist_roll"] == 0.2 for p in path)
        lowered = next(i for i, p in enumerate(path) if p["lift"] < 0.85)
        assert all(p["wrist_yaw"] == pytest.approx(path[-1]["wrist_yaw"]) for p in path[lowered:])
        assert all(abs(b["wrist_yaw"] - a["wrist_yaw"]) <= esim.config.wrist_check_step + 1e-9
                   for a, b in zip(path, path[1:]))


def test_endpoint_reachability_does_not_override_a_blocked_rotation_path():
    model, scene = _scene()
    esim = Esim(model)

    class BlockedPath(OrientationArm):
        def reachable_path(self, targets, *, start_joints=None):
            return start_joints is None and super().reachable_path(targets)

    esim.arm = BlockedPath(esim.arm.geometry)
    receipt = esim.predict(scene, "place", {"on": "e:4", "offset": [0.0, 0.1], "yaw": 0.2})
    assert not receipt["feasible"]
    assert "no continuous arm approach" in "; ".join(receipt["issues"])


def test_pouring_preserves_held_yaw_while_sampling_roll():
    model, scene = _scene(0.3)
    scene.entity("e:3").rigid.links[0].affordances.append(exact("pourable"))
    esim = Esim(model)
    arm = esim.arm = OrientationArm(esim.arm.geometry)
    receipt = esim.predict(scene, "pour", {"on": "e:4"})
    assert receipt["feasible"], receipt["issues"]
    tips = [path for path, _ in arm.paths if any(abs(t["wrist_roll"]) > 1.0 for t in path)]
    assert tips
    assert all(t["wrist_yaw"] == pytest.approx(0.3) for path in tips for t in path)


@pytest.mark.parametrize("planned_yaw", [None, 1.1])
def test_turnover_checks_the_planned_or_held_yaw(monkeypatch, planned_yaw):
    model, scene = _scene(0.3)
    esim = Esim(model)
    esim.config.flip_pitch_min = 0.0
    checked = []

    def turn_over(lift, reach, pitch, floor, surface, edge, wrist_yaw=0.0):
        checked.append(wrist_yaw)
        return floor

    monkeypatch.setattr(esim.arm, "turn_over", turn_over)
    plan = {"reach": 0.3, "dock": [0.0, 1.0], "yaw": -math.pi / 2}
    if planned_yaw is not None:
        plan["wristYaw"] = planned_yaw
    box, _, top = entity_box(scene.entity("e:4"))
    assert esim._turn_over_plan(scene, scene.entity("e:3"), plan, 0.8, top, box)["lift"] == 0.8
    assert checked == [0.3 if planned_yaw is None else planned_yaw]
