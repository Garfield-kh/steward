"""Local mesh geometry retains visual pose, explicit inertia and collision ownership."""

import copy
import struct

import mujoco
import numpy as np
import pytest

from steward.scene.mjcf import build_mjcf
from steward.scene.schema import SceneState
from steward.scene.validate import validate_world
from steward.world.sim import WorldSim


@pytest.fixture
def mesh_scene(tmp_path):
    vertices = np.array([[0, 0, 0], [0.2, 0, 0], [0, 0.3, 0], [0, 0, 0.4]], dtype="<f4")
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype="<i4")
    mesh = tmp_path / "tetra.msh"
    mesh.write_bytes(struct.pack("<4i", 4, 0, 0, 4) + vertices.tobytes() + faces.tobytes())
    geom = {"kind": "mesh", "ref": str(mesh), "pose": {"p": [0.3, 0.1, 0.2], "q": [1, 0, 0, 0]},
            "rgba": [0.8, 0.5, 0.2, 1]}
    return {"mode": "world", "entities": [{"id": "w:mesh", "role": "object", "category": {"value": "tetra"},
             "pose": {"value": {"p": [1, 2, 3], "q": [1, 0, 0, 0]}}, "body": {
                 "kind": "rigid", "root": {"value": "free"}, "joints": [], "links": [{
                     "id": "base", "inertial": {"value": {"mass": 2.0, "diagonal": [0.02, 0.03, 0.04],
                         "pose": {"p": [0.1, 0.2, 0.3], "q": [1, 0, 0, 0]}}},
                     "geometry": {"collision": [geom], "visual": [copy.deepcopy(geom)]}}]}}]}


def test_mesh_assets_are_embedded_once_and_keep_explicit_inertia(mesh_scene):
    assert validate_world(mesh_scene) == []
    build = build_mjcf(SceneState.from_dict(mesh_scene))
    assert len(build.assets) == 1
    model = build.compile()
    data = mujoco.MjData(model)
    build.apply_initial(model, data)
    assert model.nmesh == 1 and model.ngeom == 2
    assert model.body_mass[1] == 2
    np.testing.assert_allclose(model.body_inertia[1], [0.02, 0.03, 0.04])
    np.testing.assert_allclose(model.body_ipos[1], [0.1, 0.2, 0.3])
    assert model.geom_contype.tolist() == [1, 0]
    mujoco.mj_step(model, data, nstep=100)
    assert not np.any(data.warning.number)


def test_mesh_bounds_contain_transformed_vertices(mesh_scene):
    sim = WorldSim(mesh_scene)
    try:
        box = sim.entity_obb("w:mesh")
        low = np.array(box.center) - np.array(box.size) / 2
        high = np.array(box.center) + np.array(box.size) / 2
        vertices = np.array([[0, 0, 0], [0.2, 0, 0], [0, 0.3, 0], [0, 0, 0.4]]) + [1.3, 2.1, 3.2]
        assert np.all(vertices >= low - 1e-6) and np.all(vertices <= high + 1e-6)
        assert np.all(np.array(box.size) < 1.0)
    finally:
        sim.stop()


@pytest.mark.parametrize("inertia", [None, {"mass": 2}, {"mass": -1, "diagonal": [1, 1, 1]},
                                      {"mass": 2, "diagonal": [1, 0, 1]},
                                      {"mass": 2, "diagonal": [1, float("nan"), 1]}])
def test_mesh_collision_requires_valid_explicit_inertia(mesh_scene, inertia):
    mesh_scene["entities"][0]["body"]["links"][0]["inertial"] = None if inertia is None else {"value": inertia}
    assert any("inertial" in error for error in validate_world(mesh_scene))
