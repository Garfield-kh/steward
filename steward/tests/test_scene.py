from __future__ import annotations

import copy

import mujoco
import pytest

from steward.assets.generators import GENERATORS, steward_bot
from steward.scene.mjcf import build_mjcf
from steward.scene.schema import SceneState
from steward.scene.validate import validate_ego, validate_world


def test_world_scene_round_trips_and_validates(kitchen_scene):
    assert validate_world(kitchen_scene) == []
    assert SceneState.from_dict(kitchen_scene).to_dict() == kitchen_scene


def test_world_state_rejects_estimate_metadata(kitchen_scene):
    scene = copy.deepcopy(kitchen_scene)
    scene["entities"][1]["pose"]["unc"] = {"kind": "gaussian", "cov": [0.01] * 36}
    assert any("must not carry 'unc'" in e for e in validate_world(scene))


def test_ego_state_reports_world_identifier_as_leak():
    robot = steward_bot().to_dict()
    robot["id"] = "e:self"
    ego = {"mode": "ego", "frame": "map", "self": "e:self", "entities": [robot],
           "relations": [{"type": "supported-by", "a": {"entity": "e:self"}, "b": {"entity": "w:table-1"},
                          "derived": True}]}
    errors = validate_ego(ego)
    assert any("leaked into ego state" in e for e in errors)


def test_rigid_tree_rejects_unconnected_link_and_bad_geometry():
    ent = GENERATORS["cabinet_with_drawer"]().to_dict()
    ent["id"] = "w:cab"
    ent["body"]["joints"] = []
    ent["body"]["links"][0]["geometry"]["collision"][0]["size"] = [1.0, 1.0]
    errors = validate_world({"mode": "world", "entities": [ent]})
    assert any("not connected by a joint" in e for e in errors)
    assert any("needs 3 positive size values" in e for e in errors)


@pytest.mark.parametrize("name", sorted(GENERATORS))
def test_every_generator_compiles_to_mjcf(name):
    ent = GENERATORS[name]()
    ent.id = "w:x"
    model = build_mjcf(SceneState(mode="world", entities=[ent])).compile()
    assert model.nbody >= 2


def test_textured_appearance_dresses_furniture_and_casts_a_key_light():
    ent = GENERATORS["counter"]()
    ent.id = "w:x"
    scene = SceneState(mode="world", entities=[ent])
    flat = build_mjcf(scene)
    rich = build_mjcf(scene, appearance="textured")
    assert "castshadow" not in flat.xml and not flat.shadows
    assert 'castshadow="true"' in rich.xml and rich.shadows
    assert "<material " in rich.xml
    assert rich.compile().nmat >= 1
    glass = GENERATORS["cup"]()
    glass.id = "w:g"
    glass_build = build_mjcf(SceneState(mode="world", entities=[glass]), appearance="textured")
    assert "<material " not in glass_build.xml


def test_planar_root_starts_at_entity_pose(kitchen_scene):
    scene = SceneState.from_dict(kitchen_scene)
    build = build_mjcf(scene)
    model = build.compile()
    data = mujoco.MjData(model)
    build.apply_initial(model, data)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, build.names["w:robot-1"].bodies["base"])
    assert data.xpos[bid][:2] == pytest.approx([-1.8, -1.2], abs=1e-6)
