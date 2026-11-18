"""The bundled Merom layout loads in a fresh data root without a USD importer."""

import copy
import hashlib
import json
from pathlib import Path
import shutil
import struct
from zipfile import ZipFile

import mujoco
import numpy as np
import pytest

from steward.authoring import merom
from steward.authoring.service import AuthoringService
from steward.world.sim import WorldSim


def test_bundle_meshes_and_layout_match_the_conversion_record():
    manifest = json.loads((merom.BUNDLE / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((merom.BUNDLE / name).read_bytes()).hexdigest() == digest
    scene = json.loads((merom.BUNDLE / "scene.json").read_text())
    assert len(scene["entities"]) == 132
    assert all(e["body"]["root"]["value"] == "fixed" for e in scene["entities"])
    with ZipFile(merom.BUNDLE / "meshes.zip") as archive:
        refs = set()
        for role, expected in [("visual", (199, 85345)), ("collision", (1163, 108504))]:
            geoms = [g for e in scene["entities"] for link in e["body"]["links"]
                     for g in link["geometry"].get(role, [])]
            triangles = 0
            for geom in geoms:
                if geom["kind"] != "mesh":
                    continue
                ref = geom["ref"]
                data = archive.read(ref)
                assert ref == hashlib.sha256(data).hexdigest() + ".msh"
                triangles += struct.unpack("<4i", data[:16])[3]
                refs.add(ref)
            assert (len(geoms), triangles) == expected
        assert set(archive.namelist()) == refs
        assert len(refs) == manifest["unique_meshes"]
    by_id = {e["id"]: e for e in scene["entities"]}
    for name in ["w:prep_obj_bread2", "w:prep_obj_bread4"]:
        assert by_id[name]["body"]["links"][0]["geometry"]["visual"][0]["rgba"] == [0.8, 0.54, 0.27, 1]


def test_catalog_builds_a_self_contained_robot_free_scene(tmp_path, monkeypatch):
    relocated = tmp_path / "relocated-bundle"
    shutil.copytree(merom.BUNDLE, relocated)
    monkeypatch.setattr(merom, "BUNDLE", relocated)
    service = AuthoringService(tmp_path / "world")
    entry = next(e for e in service.scenario_catalog() if e["name"] == "merom-house")
    assert entry["robots"] == {} and not entry["built"]
    service.ensure_scenario("merom-house")
    scenario = service.load_scenario("merom-house")
    assert scenario["robots"] == [] and scenario["phase"] == "phase0"
    assert scenario["view"]["elevation"] == -90
    for ent in scenario["sceneState"]["entities"]:
        for link in ent["body"]["links"]:
            for group in link["geometry"].values():
                for geom in group:
                    if geom["kind"] == "mesh":
                        assert Path(geom["ref"]).is_relative_to(service.root)
                        assert Path(geom["ref"]).is_file()
    shutil.rmtree(relocated)
    service.ensure_scenario("merom-house")
    assert service.load_scenario("merom-house") == scenario
    sim = WorldSim(scenario["sceneState"])
    try:
        assert sim.model.ngeom == 1362 and sim.model.nq == 0
        sim.step(100)
        assert not np.any(sim.data.warning.number)
    finally:
        sim.stop()


def test_worktable_retains_metre_scale_support_and_open_space(tmp_path):
    service = AuthoringService(tmp_path)
    service.ensure_scenario("merom-house")
    scene = service.load_scene("merom-house")
    table = copy.deepcopy(next(e for e in scene["entities"] if e["id"] == "w:prep_obj_instance"))
    assert len(table["body"]["links"][0]["geometry"]["collision"]) == 12
    x, y, _ = table["pose"]["value"]["p"]
    probes = []
    for index, (dx, dy) in enumerate([(-0.25, -0.3), (0, 0), (0.25, 0.3)]):
        probes.append({"id": f"w:probe-{index}", "role": "object", "category": {"value": "sphere"},
                       "pose": {"value": {"p": [x + dx, y + dy, 1.1], "q": [1, 0, 0, 0]}},
                       "body": {"kind": "rigid", "root": {"value": "free"}, "joints": [], "links": [
                           {"id": "base", "geometry": {"collision": [
                               {"kind": "primitive", "shape": "sphere", "size": [0.03],
                                "pose": {"p": [0, 0, 0], "q": [1, 0, 0, 0]}}]}}]}})
    sim = WorldSim({"mode": "world", "entities": [table, *probes]})
    try:
        sim.step(1500)
        np.testing.assert_allclose(sim.data.qpos.reshape(-1, 7)[:, 2], 0.73574 + 0.03, atol=0.002)
        assert sim.data.ncon == 3 and not np.any(sim.data.warning.number)
        # At floor level the table centre is open; the former solid box filled this clearance.
        sim.data.qpos[7:10] = [x, y, 0.2]
        mujoco.mj_forward(sim.model, sim.data)
        name = sim.build.names["w:probe-1"].collision_geoms[0]
        gid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert all(gid not in (contact.geom1, contact.geom2) for contact in sim.data.contact)
    finally:
        sim.stop()


def test_corrupt_mesh_does_not_publish_a_scenario(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    shutil.copytree(merom.BUNDLE, bundle)
    monkeypatch.setattr(merom, "BUNDLE", bundle)
    with ZipFile(bundle / "meshes.zip") as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    files[next(iter(files))] = b"damaged mesh"
    with ZipFile(bundle / "meshes.zip", "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    service = AuthoringService(tmp_path / "world")
    with pytest.raises(ValueError, match="content hash"):
        service.ensure_scenario("merom-house")
    assert service.list_scenes() == [] and service.list_scenarios() == []
