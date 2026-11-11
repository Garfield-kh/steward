"""Compiled gripper geometry: aperture, TCP, attachment and visual envelope."""

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from steward.assets.cobot import TCP_OFFSET, cobot_magic
from steward.scene.mjcf import build_mjcf
from steward.scene.schema import SceneState


def test_arms_start_at_zero_with_open_fingers():
    joints = {joint.id: joint for joint in cobot_magic().rigid.joints}
    for arm in ("L", "R"):
        for joint in range(1, 7):
            assert joints[f"{arm}_joint{joint}"].q.value == [0.0]
        assert joints[f"{arm}_joint7"].q.value == [0.05]
        assert joints[f"{arm}_joint8"].q.value == [-0.05]


def _bounds(model, data, wrist, geom_name):
    """Box bounds in the wrist frame, including the URDF finger rotations."""
    geom = model.geom(geom_name)
    rotation = wrist.xmat.reshape(3, 3).T
    center = rotation @ (data.geom_xpos[geom.id] - wrist.xpos)
    half = np.abs(rotation @ data.geom_xmat[geom.id].reshape(3, 3)) @ geom.size
    return np.array([center - half, center + half])


@pytest.mark.parametrize("arm", ["L", "R"])
@pytest.mark.parametrize("travel", [0.0, 0.025, 0.05])
@pytest.mark.parametrize("wrist_angle", [0.0, 0.7])
def test_fingers_stay_connected_and_close_at_tcp(arm, travel, wrist_angle):
    entity = cobot_magic()
    build = build_mjcf(SceneState(mode="world", entities=[entity]))
    model = build.compile()
    data = mujoco.MjData(model)
    build.apply_initial(model, data)
    names = build.names[entity.id]
    for joint, value in ((6, wrist_angle), (7, travel), (8, -travel)):
        data.qpos[model.joint(names.joints[f"{arm}_joint{joint}"]).qposadr[0]] = value
    mujoco.mj_forward(model, data)
    wrist = data.body(names.bodies[f"{arm}_link6"])
    guide = _bounds(model, data, wrist, names.bodies[f"{arm}_link6"] + "::v2")
    pads = []
    snapshot = {}
    for finger in (7, 8):
        body_name = names.bodies[f"{arm}_link{finger}"]
        slider, shank, pad = [_bounds(model, data, wrist, f"{body_name}::v{i}") for i in range(3)]
        # Connected solids from guide to pad throughout the stroke, with no free-floating finger.
        for first, second in ((guide, slider), (slider, shank), (shank, pad)):
            assert np.all(np.minimum(first[1], second[1]) - np.maximum(first[0], second[0]) >= -2e-6)
        assert shank[1, 2] - shank[0, 2] > 4 * (shank[1, 0] - shank[0, 0])
        assert pad[0, 2] < TCP_OFFSET[2] < pad[1, 2]
        assert pad[1, 2] - TCP_OFFSET[2] == pytest.approx(0.005, abs=2e-6)
        for i in range(3):
            geom = model.geom(f"{body_name}::v{i}")
            assert geom.contype == 0 and geom.conaffinity == 0
        pads.append(pad)
        snapshot[f"finger{finger}"] = {
            "slider": slider.round(5).tolist(), "shank": shank.round(5).tolist(), "pad": pad.round(5).tolist(),
        }
    assert pads[0][1, 0] == pytest.approx(-travel, abs=2e-6)
    assert pads[1][0, 0] == pytest.approx(travel, abs=2e-6)
    assert pads[1][0, 0] - pads[0][1, 0] == pytest.approx(2 * travel, abs=2e-6)
    assert (pads[1][0, 0] + pads[0][1, 0]) / 2 == pytest.approx(TCP_OFFSET[0], abs=2e-6)
    # The housing and guide end behind the gripping region, leaving it unobstructed.
    assert guide[1, 2] < min(pad[0, 2] for pad in pads)
    expected = json.loads((Path(__file__).parent / "expected/cobot-gripper.json").read_text())
    assert snapshot == expected[str(travel)]
