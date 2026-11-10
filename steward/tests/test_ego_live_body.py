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
