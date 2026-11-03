"""Cobot Magic embodiment twin: URDF fidelity, MJCF compilability, and WSim command/sensor flow."""

import io

import mujoco
import numpy as np
import pytest

from steward.assets.cobot import LIFT_HOME, cobot_magic
from steward.assets.generators import bowl
from steward.scene.mjcf import build_mjcf
from steward.scene.schema import SceneState
from steward.scene.validate import validate_world
from steward.world.sim import WorldSim


@pytest.fixture()
def robot() -> dict:
    return cobot_magic().to_dict()


def test_valid_world_entity(robot: dict) -> None:
    assert validate_world({"mode": "world", "entities": [robot]}) == []


def test_urdf_joint_fidelity(robot: dict) -> None:
    """Joint1..joint6 of both arms carry the Piper URDF origins, axes and limits."""
    joints = {j["id"]: j for j in robot["body"]["joints"]}
    spec = {
        "L_joint1": ([0.0, 0.0, 0.123], -2.618, 2.618),
        "L_joint3": ([0.2836, 0.0287, 0.0], -2.697, 0.0),
        "L_joint6": ([0.0, 0.091, 0.0014], -3.14, 3.14),
        "R_joint4": ([-0.2422, 0.0685, 0.0], -1.832, 1.832),
    }
    for jid, (xyz, lo, hi) in spec.items():
        j = joints[jid]
        assert j["type"]["value"] == "revolute"
        assert np.allclose(j["origin"]["value"]["p"], xyz, atol=1e-4)
        assert j["limits"]["value"] == {"lo": lo, "hi": hi}
    fingers = [joints[f"{a}_joint{i}"] for a in ("L", "R") for i in (7, 8)]
    for j in fingers:
        assert j["type"]["value"] == "prismatic"
        assert abs(j["limits"]["value"]["hi"] - j["limits"]["value"]["lo"]) == pytest.approx(0.05)


def test_compiles_and_steps() -> None:
    build = build_mjcf(SceneState(mode="world", entities=[cobot_magic()]), timestep=0.005)
    model = mujoco.MjModel.from_xml_string(build.xml)
    data = mujoco.MjData(model)
    # Planar root (2 slides + hinge), the lift, and 2 arms x (6 revolute + 2 prismatic fingers).
    assert model.njnt == 20
    assert model.nu == 20
    # Top camera plus one wrist camera per arm.
    cam_names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)}
    assert any(n.endswith("top_camera") for n in cam_names)
    assert any(n.endswith("L_wrist_camera") for n in cam_names)
    assert any(n.endswith("R_wrist_camera") for n in cam_names)
    # Holds the zero pose (targets default to qpos) without blowing up.
    for _ in range(200):
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all()
    assert data.qvel.shape == (20,)


def _cobot_scene(bowl_at: list[float] | None = None) -> dict:
    ent = cobot_magic().to_dict()
    ent["id"] = "w:cobot"
    entities = [ent]
    if bowl_at is not None:
        b = bowl().to_dict()
        b["id"] = "w:bowl-1"
        b["pose"]["value"]["p"] = list(bowl_at)
        entities.append(b)
    return {"mode": "world", "entities": entities}


def test_sim_generic_joints_and_dual_gripper_grasp() -> None:
    # First load locates the left pinch point in the world; the second places a bowl on it.
    probe = WorldSim(_cobot_scene())
    grip = probe.grasp_point("w:cobot", "L_link6")
    bowl_at = [grip["p"][0] + 0.02, grip["p"][1], grip["p"][2] - 0.03]

    sim = WorldSim(_cobot_scene(bowl_at))
    sim.command("w:cobot", {"gripper": {"L_link6": "close"}})
    kinds = [e["type"] for e in sim.events]
    assert "grasp/attach" in kinds, f"expected an attach, events: {sim.events}"
    sim.step(20)
    frame = sim.sensors("w:cobot")
    # Parked at the scene's pose: odometry reads there (plus sensor noise); joints cover the
    # chassis, the lift and both arms.
    assert abs(frame["odometry"]["x"]) < 0.01 and abs(frame["odometry"]["yaw"]) < 0.01
    assert frame["joints"]["lift"] == pytest.approx(LIFT_HOME, abs=0.02)
    assert {"L_joint1", "L_joint7", "R_joint6", "R_joint8"} <= set(frame["joints"])
    assert set(frame["grippers"]) == {"L_link6", "R_link6"}
    assert "gripper" not in frame, "multi-gripper robots must not fake a single gripper field"
    assert frame["grippers"]["L_link6"]["holding"] is True
    assert frame["grippers"]["R_link6"]["holding"] is False

    # Generic joint command moves an arm joint through the position actuator.
    sim.command("w:cobot", {"joints": {"L_joint2": 1.5}})
    for _ in range(400):
        sim.step(1)
    moved = sim.sensors("w:cobot")["joints"]["L_joint2"]
    assert moved > 0.8, f"L_joint2 reached only {moved:.3f} rad"

    sim.command("w:cobot", {"gripper": {"L_link6": "open"}})
    assert any(e["type"] == "grasp/release" for e in sim.events)

    with pytest.raises(KeyError):
        sim.command("w:cobot", {"joints": {"L_nosuch": 0.0}})
    with pytest.raises(KeyError):
        sim.command("w:cobot", {"gripper": "close"})  # ambiguous gripper must demand a link id


def test_sim_rgbd_per_camera() -> None:
    sim = WorldSim(_cobot_scene())
    for camera, expect in ((None, "top_camera"), ("L_wrist_camera", "L_wrist_camera")):
        blob = sim.rgbd("w:cobot", camera=camera)
        frame = np.load(io.BytesIO(blob))
        assert frame["rgb"].shape == (480, 640, 3)
        assert frame["depth"].shape == (480, 640)
        assert frame["mask"].shape == (480, 640)
        import json
        header = json.loads(str(frame["header"]))
        assert header["camera"]["id"] == expect
