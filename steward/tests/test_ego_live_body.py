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


class _Link:
    def __init__(self, lock, live, on_read=None):
        self.lock, self.live, self.on_read = lock, live, on_read

    def sensors(self):
        if self.on_read is not None:
            self.on_read()
        return dict(self.live)


class _Arm:
    @staticmethod
    def virtualize(state):
        return state


def _runtime(live, on_read=None):
    import threading

    from steward.assets.generators import steward_bot
    from steward.ego.estimator import EgoEstimator
    from steward.ego.runtime import EgoRuntime

    rt = EgoRuntime.__new__(EgoRuntime)
    rt._lock = threading.RLock()
    rt.estimator = EgoEstimator(robot_model=steward_bot().to_dict())
    rt._episode, rt._generation, rt._motion_belief = None, 0, None
    rt.arm = _Arm()
    rt.link = _Link(rt._lock, live, on_read)
    return rt


def _self_x(scene):
    return scene.find(scene.self_id).pose.value["p"][0]


def test_ego_state_composes_a_live_body_newer_than_the_belief():
    """5: the self pose of the Ego State is the body read now, not the last perception update's."""
    rt = _runtime({})
    rt.link.live = {"odometry": {"x": 0.3, "y": 0.0, "yaw": 0.0}, "stamp": rt.estimator.stamp + 1.0,
                    "joints": dict(rt.estimator.joints)}
    assert _self_x(rt.ego_state()) == pytest.approx(0.3)


def test_ego_state_does_not_mix_a_body_read_across_a_reset():
    """4/5: a reset between reading the body and composing the answer: the old episode's body is
    not put with the new episode's objects (and the sensor read does not hold the belief lock)."""
    rt = _runtime({})
    rt.link.live = {"odometry": {"x": 0.3, "y": 0.0, "yaw": 0.0}, "stamp": rt.estimator.stamp + 1.0,
                    "joints": dict(rt.estimator.joints)}

    def reset():
        assert not rt._lock._is_owned()
        rt._generation += 1

    rt.link.on_read = reset
    assert _self_x(rt.ego_state()) == pytest.approx(_self_x(rt.estimator.to_scene()))


def test_a_motion_state_across_a_reset_is_one_episode_whole():
    """4: the reset check compared the live stamp with itself; then only the clock was put back,
    leaving a new body with the old objects. A body behind the belief's clock, or of another
    episode, leaves the belief whole."""
    rt = _runtime({"stamp": 1.0, "odometry": {"x": 1.0, "y": 0.0, "yaw": 0.0}})
    belief = {"generation": 0, "episode": None, "stamp": 5.0, "odometry": {"x": 0.0, "y": 0.0, "yaw": 0.0},
              "joints": {}, "gripper": {}, "grippers": {}, "entities": {"e:1": {"p": [1.0, 2.0, 0.0]}}}
    rt._keep_motion_belief = lambda: belief
    rt._motion_belief = belief
    out = rt._motion_state()
    assert out["stamp"] == 5.0 and out["odometry"]["x"] == 0.0 and out["entities"]["e:1"]["p"] == [1.0, 2.0, 0.0]
    rt.link.live = {"stamp": 6.0, "episode": 3, "odometry": {"x": 1.0, "y": 0.0, "yaw": 0.0}}
    belief["episode"] = 2
    assert rt._motion_state()["odometry"]["x"] == 0.0
    belief["episode"] = 3
    assert rt._motion_state()["odometry"]["x"] == 1.0


def test_a_grasp_felt_across_a_reset_is_not_applied_to_the_new_episode():
    """4: a skill callback that read the body before a reset must not mark e:1 of the next episode."""
    rt = _runtime({"stamp": 1.0, "odometry": {"x": 0.0, "y": 0.0, "yaw": 0.0}})

    class _Est:
        stamp, called = 0.0, []

        def mark_held(self, *a, **k):
            self.called.append(a)
            return a[0]

        def release_at(self, *a, **k):
            self.called.append(a)

    class _Geo:
        gripper = None

    rt.estimator = _Est()
    rt.arm.geometry = _Geo()

    def reset():
        rt._generation += 1

    rt.link.on_read = reset
    rt._mark_held("e:1")
    rt._released("e:1", [0.0, 0.0, 0.0])
    assert rt.estimator.called == []


def test_a_skill_predicted_before_a_reset_does_not_mark_the_new_episode():
    """4 (review r3): the generation was read when the callback ran, so an old skill thread that
    paused across the reset and only then called back passed the check and marked the new e:1."""
    import threading
    from types import SimpleNamespace

    rt = _runtime({"stamp": 1.0, "odometry": {"x": 0.0, "y": 0.0, "yaw": 0.0}})
    called = []
    rt.estimator = SimpleNamespace(stamp=0.0, mark_held=lambda *a, **k: called.append(a) or a[0],
                                   release_at=lambda *a, **k: called.append(a))
    rt.arm.geometry = SimpleNamespace(gripper=None)
    rt._receipt_generation = {"rcpt-1": 0}
    rt.executor = SimpleNamespace(jobs={"job-1": SimpleNamespace(receipt="rcpt-1")})
    rt._generation = 1  # the reset came before the callback
    rt._receipt_generation.clear()

    def old_skill():
        rt._mark_held("e:1")
        rt._released("e:1", [0.0, 0.0, 0.0])

    t = threading.Thread(target=old_skill, name="skill-job-1")
    t.start()
    t.join()
    assert called == []
    # a skill of this episode is applied
    rt._receipt_generation["rcpt-2"] = 1
    rt.executor.jobs["job-2"] = SimpleNamespace(receipt="rcpt-2")
    t = threading.Thread(target=lambda: rt._mark_held("e:1"), name="skill-job-2")
    t.start()
    t.join()
    assert len(called) == 1
