"""Ego runtime and estimator: the robot and the load in its hand as the body is now, and state
that belongs to one episode."""

from __future__ import annotations

import math

import numpy as np
import pytest


def _held_setup():
    from steward.assets.generators import steward_bot
    from steward.ego.estimator import EgoEstimator, Track

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    grip = est.gripper_position()

    def track(oid, cat, pos, size):
        return Track(id=oid, category_counts={cat: 5.0}, pos=np.array(pos, float), var=np.full(3, 1e-6), yaw=0.0,
                     yaw_var=1e-4, size=np.array(size, float), observations=5, first_seen=0.0, last_seen=0.0, existence=1.0)

    est.tracks["e:1"] = track("e:1", "bowl", grip + [0.05, 0.0, 0.0], [0.2, 0.2, 0.06])
    est.tracks["e:2"] = track("e:2", "cup", grip + [0.08, 0.03, 0.0], [0.05, 0.05, 0.05])
    return est


def test_a_load_renamed_during_the_approach_is_found_at_the_gripper_of_the_grasp_body():
    """A2: the fallback for a load merged away during the approach looked near the gripper of the last
    perception update (a drive old), so a grasp after a 1 m drive raised 'no longer in the ego state'."""
    est = _held_setup()
    moved = {**est.odometry, "x": float(est.odometry.get("x", 0.0)) + 1.0}
    body = {"odometry": moved, "joints": dict(est.joints)}
    est.tracks["e:1"].pos = est.gripper_position(None, body["joints"], moved) + [0.02, 0.0, 0.0]
    assert est.mark_held("e:99", since=est.stamp, body=body) == "e:1"


def test_a_held_load_and_its_contents_turn_with_the_live_body():
    """B2: to_scene placed the load at the live body but kept its heading (and what lies in it) from
    the last look: the base turned 90 deg, the load read as unturned, and the place wrist angle was
    planned from that."""
    est = _held_setup()
    body = {"odometry": dict(est.odometry), "joints": dict(est.joints)}
    est.mark_held("e:1", since=est.stamp, body=body)
    est.tracks["e:2"].carried_offset = est.gripper_rotation().T @ (est.tracks["e:2"].pos - est.tracks["e:1"].pos)
    turned = {**body["odometry"], "yaw": float(body["odometry"].get("yaw", 0.0)) + math.pi / 2}
    es = est.to_scene({"odometry": turned, "joints": body["joints"], "stamp": est.stamp})
    from steward.math3d import yaw_from_quat

    load, cup = es.find("e:1"), es.find("e:2")
    assert abs(math.remainder(yaw_from_quat(load.pose.value["q"]) - math.pi / 2, 2 * math.pi)) < 1e-6
    want = est.held_pose(body["joints"], turned)["e:2"][0]
    assert cup.pose.value["p"] == pytest.approx(list(want), abs=1e-9)
    assert math.dist(cup.pose.value["p"][:2], est.tracks["e:2"].pos[:2]) > 0.05, "the content moved with the body"


def test_a_load_tipped_by_the_live_wrist_does_not_carry_its_contents():
    """B2 (review): held_pose judged tipping by the belief's old wrist: a live wrist rolled 2 rad
    still carried the contents along in the scene."""
    est = _held_setup()
    body = {"odometry": dict(est.odometry), "joints": dict(est.joints)}
    est.mark_held("e:1", since=est.stamp, body=body)
    est.tracks["e:2"].carried_offset = est.gripper_rotation().T @ (est.tracks["e:2"].pos - est.tracks["e:1"].pos)
    rolled = {**body["joints"], "wrist_roll": 2.0}
    assert "e:2" not in est.held_pose(rolled, body["odometry"])
    es = est.to_scene({"odometry": body["odometry"], "joints": rolled, "stamp": est.stamp})
    assert not any(r.type == "supported-by" and r.a["entity"] == "e:2" and r.b["entity"] == "e:1" for r in es.relations)
    assert "e:2" in est.held_pose(body["joints"], body["odometry"])  # upright, it is carried
