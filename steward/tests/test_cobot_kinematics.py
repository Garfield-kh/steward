"""Static FK against raw USD joint anchors; no physics stepping or USD runtime required."""

import json
import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

from steward.assets.cobot import cobot_magic
from steward.ego.cobot_arm import PiperArm
from steward.math3d import matrix_from_quat, quat_from_axis_angle, quat_from_yaw, quat_normalize
from steward.scene.mjcf import build_mjcf
from steward.scene.schema import SceneState, exact, pose


def _transform(p, q):
    matrix = np.eye(4)
    matrix[:3, :3] = matrix_from_quat(quat_normalize(q))
    matrix[:3, 3] = p
    return matrix


@pytest.mark.parametrize("share", [0.0, 0.5, 1.0])
def test_compiled_link_poses_match_usd_joint_frames(share):
    reference = json.loads((Path(__file__).parent / "fixtures/cobot-usd-joints.json").read_text())
    robot = cobot_magic()
    robot.pose = exact(pose([1.0, -2.0, 0.0], quat_from_yaw(0.4)))
    build = build_mjcf(SceneState(mode="world", entities=[robot]))
    model = build.compile()
    data = mujoco.MjData(model)
    build.apply_initial(model, data)
    names = build.names[robot.id]
    base = _transform(robot.pose.value["p"], robot.pose.value["q"])
    reference_base = base @ _transform([0.0, 0.0, -reference["wheel_min_z"]], [1, 0, 0, 0])
    frames = {"base": reference_base}
    for record in reference["joints"]:
        motion = np.eye(4)
        kind = record["type"]
        if kind != "PhysicsFixedJoint":
            lo, hi = record["lowerLimit"], record["upperLimit"]
            if kind == "PhysicsRevoluteJoint":
                lo, hi = math.radians(lo), math.radians(hi)
            # Opposite arms take different postures; fingers use symmetric openings.
            fraction = 1 - share if record["id"].startswith("R_joint") else share
            if record["id"].endswith("joint8"):
                fraction = 1 - fraction
            value = lo + fraction * (hi - lo)
            joint = model.joint(names.joints[record["id"]])
            shift = 0.60 if record["id"] == "lift" else 0.0
            np.testing.assert_allclose(joint.range, [lo + shift, hi + shift], atol=1e-6, rtol=0)
            data.qpos[joint.qposadr[0]] = value + shift
            axis = np.eye(3)["XYZ".index(record["axis"])]
            if kind == "PhysicsRevoluteJoint":
                motion = _transform([0, 0, 0], quat_from_axis_angle(axis, value))
            else:
                motion[:3, 3] = axis * value
        anchor0 = _transform(record["localPos0"], record["localRot0"])
        anchor1 = _transform(record["localPos1"], record["localRot1"])
        frames[record["child"]] = frames[record["parent"]] @ anchor0 @ motion @ np.linalg.inv(anchor1)
    mujoco.mj_forward(model, data)
    for link, expected in frames.items():
        if link == "base":
            continue  # Steward retains its floor-level chassis frame.
        actual = data.body(names.bodies[link])
        np.testing.assert_allclose(actual.xpos, expected[:3, 3], atol=5e-6, rtol=0)
        np.testing.assert_allclose(actual.xmat.reshape(3, 3), expected[:3, :3], atol=5e-6, rtol=0)


def test_custom_mount_height_drives_arm_geometry_and_lift_preference():
    robot = cobot_magic(separation=0.70, mount_height=1.10)
    arm = PiperArm(robot.to_dict())
    assert arm.geometry.tool_offset == pytest.approx(0.36848)
    assert arm.geometry.lateral == pytest.approx(0.35)
    target = {"lift": 1.0, "wrist_pitch": 0.0}
    assert arm._prefer(target) == pytest.approx(1.0 + arm.config.shoulder_above - 1.10)
    assert all(0 <= height <= 0.60 for height, _ in arm._working_seeds(target))
