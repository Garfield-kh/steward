"""The estimator's own kinematics must agree with the simulation, for every camera and pose.

This is what replaces the mobile robot's fixed head-mount assumption: a wrist camera has to move
with the arm, and the pinch point has to come out of the joint encoders.
"""

import numpy as np
import pytest

from steward.assets.cobot import cobot_magic
from steward.ego.estimator import EgoEstimator
from steward.world.sim import WorldSim

from test_cobot_skills import _scene


@pytest.fixture()
def rig():
    sim = WorldSim(_scene())
    est = EgoEstimator(robot_model=cobot_magic().to_dict())
    yield sim, est
    sim.stop()


def _sync(est: EgoEstimator, sim: WorldSim) -> None:
    """Give the estimator a clean frame: encoders from the sim, odometry the base's true pose (an
    arm swinging round nudges the chassis by millimetres)."""
    from steward.math3d import yaw_from_quat

    frame = sim.sensors("w:cobot")
    base = sim._entity_pose("w:cobot")
    frame["odometry"] = {"x": base["p"][0], "y": base["p"][1], "yaw": yaw_from_quat(base["q"]),
                         "sigma": {"pos": 0.0, "yaw": 0.0}}
    est.update(frame)


def test_every_camera_matches_the_simulation(rig) -> None:
    sim, est = rig
    _sync(est, sim)
    poses = est.camera_poses()
    assert set(poses) == {"head_camera", "wide_camera"}
    for sensor_id, pose in poses.items():
        truth, _ = sim.camera_pose("w:cobot", sensor_id)
        # 1 mm is far tighter than any perception error and absorbs the float path difference
        # between MuJoCo's body quaternions and this joint-chain composition.
        assert np.linalg.norm(np.subtract(pose["p"], truth["p"])) < 1e-3, sensor_id
        assert abs(float(np.dot(pose["q"], truth["q"]))) > 0.99999, sensor_id


def test_the_head_stays_where_it_is_when_the_arms_and_the_lift_move(rig) -> None:
    sim, est = rig
    _sync(est, sim)
    before = est.camera_pose(camera="head_camera")["p"]
    sim.command("w:cobot", {"joints": {"L_joint1": 1.0, "L_joint2": 0.4, "L_joint5": 0.5, "lift": 0.9}})
    sim.step(400)
    _sync(est, sim)
    after = est.camera_pose(camera="head_camera")["p"]
    assert np.linalg.norm(np.subtract(after, before)) < 1e-4, "only the chassis's own creep as the arms swing round"
    truth, _ = sim.camera_pose("w:cobot", "head_camera")
    assert np.linalg.norm(np.subtract(after, truth["p"])) < 1e-3


def test_pinch_point_comes_out_of_the_encoders(rig) -> None:
    sim, est = rig
    for joints in ({"L_joint1": 0.0, "L_joint2": 1.57, "L_joint3": -1.35, "L_joint4": 0.0, "L_joint5": 0.0, "L_joint6": 0.0},
                   {"L_joint1": -0.6, "L_joint2": 0.9, "L_joint3": -2.0, "L_joint4": 0.0, "L_joint5": 0.0, "L_joint6": 0.0}):
        sim.command("w:cobot", {"joints": joints})
        sim.step(1500)  # from the resting posture the arm swings a long way: let it settle
        _sync(est, sim)
        truth = sim.grasp_point("w:cobot", "L_link6")["p"]
        got = est.gripper_position("L_link6")
        assert np.linalg.norm(got - np.asarray(truth)) < 1e-3, f"joints {joints}"


def test_single_camera_robot_still_uses_the_first_sensor(rig) -> None:
    from steward.assets.generators import steward_bot

    sim, _ = rig
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    frame = {"stamp": 0.0, "odometry": {"x": 1.0, "y": 2.0, "yaw": 0.3, "sigma": {"pos": 0.0, "yaw": 0.0}},
             "joints": {"lift": 0.6, "reach": 0.1}, "gripper": {"state": "open", "holding": False},
             "detections": [], "camera": {"id": "head_camera", "intrinsics": {}}}
    est.update(frame)
    # base (1, 2) yawed 0.3, head mount (0.16, 0, 1.50) and camera mount position (0.05, 0, 0),
    # both offsets turned by the base yaw; the mount's own 30 deg pitch turns the image only.
    pose = est.camera_pose()
    reach = 0.16 + 0.05
    assert pose["p"][0] == pytest.approx(1.0 + reach * np.cos(0.3), abs=1e-6)
    assert pose["p"][1] == pytest.approx(2.0 + reach * np.sin(0.3), abs=1e-6)
    assert pose["p"][2] == pytest.approx(1.50, abs=1e-6)
    # The optical axis (+x of the mount frame) points 30 degrees down.
    from steward.math3d import quat_rotate

    forward = quat_rotate(pose["q"], [1.0, 0.0, 0.0])
    assert forward[2] == pytest.approx(-np.sin(np.radians(30)), abs=1e-6)


def test_the_ego_scene_uses_the_joints_it_is_given(rig) -> None:
    """The console's Ego State must show the arm where the encoders say it is.

    The mission looks at the table a few times a round; between looks the estimator holds the
    joints of the last frame, so a panel built from them alone freezes for the whole motion.
    """
    sim, est = rig
    _sync(est, sim)
    frame = sim.proprioception("w:cobot")
    moved = {name: value + 0.1 for name, value in frame["joints"].items()}
    frame["joints"] = moved
    scene = est.to_scene(frame)
    robot = next(ent for ent in scene.entities if ent.role == "robot")
    shown = {joint.id: joint.q.value[0] for joint in robot.body.joints if joint.q and joint.q.value}
    assert shown["L_joint1"] == pytest.approx(moved["L_joint1"])
    assert shown["L_joint1"] != pytest.approx(est.joints["L_joint1"])
    # without a live frame the same scene describes what the estimator itself last saw
    stale = {joint.id: joint.q.value[0] for joint in est.to_scene().entities[0].body.joints
             if joint.q and joint.q.value}
    assert stale["L_joint1"] == pytest.approx(est.joints["L_joint1"])


def test_proprioception_leaves_the_perception_noise_stream_alone(rig) -> None:
    """A console watching the twin must not change what the twin perceives."""
    sim, _ = rig
    baseline = sim.sensors("w:cobot")["odometry"]
    for _ in range(5):
        sim.proprioception("w:cobot")
    assert sim.sensors("w:cobot")["odometry"] == baseline


def test_a_dual_arm_load_follows_the_arm_that_holds_it(rig) -> None:
    """Two grippers, no single "gripper" field: the load must follow the arm that took it.

    Before this, a dual-arm frame left the estimator with an open gripper for ever, so the Ego
    State never showed a bowl in the hand -- the panel had the arm moving empty while the world had
    it carrying one.
    """
    from steward.ego.estimator import Track

    sim, est = rig
    _sync(est, sim)
    at = est.gripper_position("R_link6")
    track = Track(id="e:loaded", category_counts={"bowl": 1.0}, pos=at.copy(), var=np.full(3, 1e-4),
                  yaw=0.0, yaw_var=1e-4, size=np.array([0.2, 0.2, 0.06]), observations=3,
                  first_seen=0.0, last_seen=est.stamp)
    est.tracks[track.id] = track
    est.mark_held(track.id, gripper="R_link6", at=at)

    def frame_with_right_holding() -> dict:
        frame = sim.sensors("w:cobot")
        frame["grippers"] = {"L_link6": {"state": "open", "holding": False, "holdingHandle": False},
                             "R_link6": {"state": "closed", "holding": True, "holdingHandle": False}}
        return frame

    est.update(frame_with_right_holding())
    assert est.gripper["state"] == "closed" and est.gripper["holding"] is True
    assert np.linalg.norm(track.pos - est.gripper_position("R_link6")) < 1e-6

    # Move the right arm: the load goes with it, and the left arm's pose stays irrelevant.
    sim.command("w:cobot", {"joints": {"R_joint1": -0.6, "R_joint2": 0.9, "R_joint3": -2.0}})
    sim.step(400)
    est.update(frame_with_right_holding())
    assert np.linalg.norm(track.pos - est.gripper_position("R_link6")) < 1e-3
    assert np.linalg.norm(track.pos - est.gripper_position("L_link6")) > 0.10

    # And the rendered Ego State puts it there too, without the track being rewritten.
    live = sim.proprioception("w:cobot")
    scene = est.to_scene(live)
    rendered = next(ent for ent in scene.entities if ent.id == track.id)
    assert np.linalg.norm(np.subtract(rendered.pose.value["p"], est.gripper_position("R_link6", live["joints"]))) < 1e-3

    # A grasp recorded with the pinch point it closed at needs no fresh encoders: the offset is
    # measured there, so a body that looks only between moves still carries the object by the hand.
    track.pos = np.array([0.31, 0.29, 0.78])
    est.mark_held(track.id, gripper="R_link6", at=np.array([0.30, 0.30, 0.78]))
    assert np.linalg.norm(est.gripper_rotation("R_link6") @ track.held_offset - np.array([0.01, -0.01, 0.0])) < 1e-9

    # Opening that gripper puts the load down where it is.
    frame = frame_with_right_holding()
    frame["grippers"]["R_link6"] = {"state": "open", "holding": False, "holdingHandle": False}
    est.update(frame)
    assert est.held_id is None and est.held_gripper is None
    assert track.held is False
    # It is believed where the hand let go of it, not where it was picked up -- otherwise the
    # next look measures a bowl 30 cm from the stale position and starts a second track for it.
    assert np.linalg.norm(track.pos - est.gripper_position("R_link6") - np.array([0.01, -0.01, 0.0])) < 1e-3
