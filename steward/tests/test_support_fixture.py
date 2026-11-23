"""World workholding contracts; physics cases skip explicitly without MuJoCo."""
from __future__ import annotations

import copy
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from steward.assets.generators import (airplane_fuselage, airplane_wing, car_body,
                                       airplane_landing_gear, toy_screw)
from steward.scene.mjcf import build_mjcf
from steward.scene.schema import Entity, Link, RigidTree, SceneState, exact, pose, primitive
from steward.world.support_fixture import candidates, enabled_from_env, reserve


def scene():
    body = airplane_fuselage()
    body.id = "w:anonymous"
    body.category = exact("unrelated-label")
    body.pose = exact(pose([0, 0, .201]))
    table = Entity("w:platform", "object", exact("platform"), exact(pose()),
        RigidTree(exact("fixed"), [Link("top", "top",
            geometry={"collision": [primitive("box", [.8, .8, .2], pose([0, 0, .1]))]},
            affordances=[exact("support-surface")])]))
    robot = Entity("w:fixture", "robot", exact("fixture"), exact(pose([0, 0, .23])),
        RigidTree(exact("fixed"), [Link("grip", "grip", affordances=[exact("grasp-point")])]))
    return SceneState(mode="world", entities=[body, table, robot])


class MetadataTests(unittest.TestCase):
    def test_metadata_not_names_and_upper_parts_are_excluded(self):
        world = scene()
        for i, factory in enumerate((airplane_wing, airplane_landing_gear, toy_screw, car_body)):
            ent = factory(); ent.id = f"w:part{i}"; ent.category = exact("arbitrary")
            world.entities.append(ent)
        self.assertEqual(candidates(world), {"w:anonymous", "w:part3"})
        for ent in world.entities:
            for link in ent.rigid.links:
                for aff in link.affordances:
                    if isinstance(aff.value, dict) and "screw-hole" in aff.value:
                        aff.value["screw-hole"]["id"] = "renamed-" + aff.value["screw-hole"]["id"]
        self.assertEqual(candidates(world), {"w:anonymous", "w:part3"})

    def test_explicit_groups_allow_coplanar_interfaces(self):
        world = scene()
        rows = [a.value["screw-hole"] for a in world.entities[0].rigid.links[0].affordances
                if isinstance(a.value, dict)]
        for i, row in enumerate(rows):
            row.update(axis=[0, 0, 1], group=str(i // 2))
        self.assertEqual(candidates(world), {"w:anonymous"})
        for row in rows:
            row["group"] = "single"
        self.assertEqual(candidates(world), set())

    def test_switch_default_and_exact_disabled_xml_and_ego_isolation(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertTrue(enabled_from_env())
        with patch.dict("os.environ", {"STEWARD_SUPPORT_FIXTURE": "0"}):
            self.assertFalse(enabled_from_env())
        world = scene(); build = build_mjcf(world); original = build.xml
        self.assertEqual(reserve(build, world, False), {})
        self.assertEqual(build.xml, original)
        self.assertEqual(len(reserve(build, world, True)), 1)
        world.mode = "ego"; build = build_mjcf(world); original = build.xml
        self.assertEqual(reserve(build, world, True), {})
        self.assertEqual(build.xml, original)


@unittest.skipUnless(importlib.util.find_spec("mujoco"), "MuJoCo is not installed in this Python")
class PhysicsTests(unittest.TestCase):
    def make(self, enabled=True):
        from steward.world.sim import WorldSim
        from steward.scene.fasteners import FastenConfig
        sim = WorldSim(scene().to_dict(), support_fixture=enabled, fasten=FastenConfig(mode="off"))
        self.addCleanup(sim.stop)
        return sim

    def place(self, sim, entity="w:anonymous"):
        import mujoco
        at = sim._entity_pose(entity)
        sim._set_root_pose(sim.template.entity("w:fixture"), at)
        mujoco.mj_forward(sim.model, sim.data)
        sim.command("w:fixture", {"gripper": "close"})
        self.assertEqual(sim.robots["w:fixture"].attached, entity)
        sim.command("w:fixture", {"gripper": "open"})
        sim.step_seconds(1.)

    def perturb(self, sim):
        import mujoco
        from steward.math3d import yaw_from_quat
        bid = sim._root_body("w:anonymous")
        before = sim._entity_pose("w:anonymous")
        maximum = np.zeros(2)
        sim.data.xfrc_applied[bid] = [.8, .2, 0, 0, 0, .003]
        for step in range(round(.5 / sim.timestep)):
            if step == round(.05 / sim.timestep):
                sim.data.xfrc_applied[bid] = 0  # short lateral/torsional impulse
            sim.step(); mujoco.mj_forward(sim.model, sim.data)
            after = sim._entity_pose("w:anonymous")
            maximum = np.maximum(maximum, [np.linalg.norm(np.asarray(after["p"]) - before["p"]),
                abs(yaw_from_quat(after["q"]) - yaw_from_quat(before["q"]))])
        sim.data.xfrc_applied[bid] = 0
        sim.step_seconds(.5)
        mujoco.mj_forward(sim.model, sim.data)
        return maximum

    def test_release_settle_perturb_and_regrasp(self):
        import mujoco
        from steward.math3d import yaw_from_quat
        sim = self.make(); sim.step_seconds(.5)
        self.assertFalse(sim._support_fixture.active, "initial placement is not a robot release")
        self.place(sim)
        self.assertEqual(len(sim._support_fixture.active), 1)
        before = sim._entity_pose("w:anonymous")
        displacement, yaw = self.perturb(sim)
        self.assertLess(displacement, 1e-4)
        self.assertLess(yaw, 1e-3)
        after = sim._entity_pose("w:anonymous")
        self.assertLess(np.linalg.norm(np.array(before["p"]) - after["p"]), 1e-4)
        self.assertLess(abs(yaw_from_quat(before["q"]) - yaw_from_quat(after["q"])), 1e-3)
        sim._set_root_pose(sim.template.entity("w:fixture"), after)
        mujoco.mj_forward(sim.model, sim.data)
        sim.command("w:fixture", {"gripper": "close"})
        self.assertFalse(sim._support_fixture.active)
        target = copy.deepcopy(after); target["p"][0] += .1; target["p"][2] += .15
        sim._set_root_pose(sim.template.entity("w:fixture"), target)
        mujoco.mj_forward(sim.model, sim.data)
        sim.step(2); mujoco.mj_forward(sim.model, sim.data)
        np.testing.assert_allclose(sim._entity_pose("w:anonymous")["p"], target["p"], atol=1e-6)
        # A release in the air must not lock it there.
        sim.command("w:fixture", {"gripper": "open"}); sim.step(2)
        self.assertFalse(sim._support_fixture.active)

    def test_disabled_has_no_weld_and_still_slides(self):
        sim = self.make(False); self.place(sim)
        self.assertEqual(sim.model.neq, 0)
        self.assertFalse(sim._support_fixture.pending)
        before = sim._entity_pose("w:anonymous")["p"]
        self.perturb(sim)
        self.assertGreater(np.linalg.norm(np.array(before[:2]) - sim._entity_pose("w:anonymous")["p"][:2]), .001)

    def test_rebuild_preserves_fixture_and_reset_clears_it(self):
        sim = self.make(); self.place(sim)
        original = copy.deepcopy(sim._support_fixture.active)
        sim._rebuild(SceneState.from_dict(sim.world_state()))
        self.assertEqual(sim._support_fixture.active, original)
        self.assertEqual(sum(sim.data.eq_active), 1)
        self.perturb(sim)
        sim.reset()
        self.assertFalse(sim._support_fixture.active)
        self.assertFalse(sim._support_fixture.pending)
        self.assertFalse(any(sim.data.eq_active))

    def test_fixture_is_relative_to_support_not_global_origin(self):
        import mujoco
        from steward.math3d import pose_compose, pose_inverse, quat_from_yaw
        sim = self.make(); self.place(sim)
        relative = pose_compose(pose_inverse(sim._entity_pose("w:platform")), sim._entity_pose("w:anonymous"))
        moved = pose([.02, .01, 0], quat_from_yaw(.15))
        sim._set_root_pose(sim.template.entity("w:platform"), moved)
        mujoco.mj_forward(sim.model, sim.data)
        sim.step_seconds(.5); mujoco.mj_forward(sim.model, sim.data)
        expected = pose_compose(moved, relative)
        np.testing.assert_allclose(sim._entity_pose("w:anonymous")["p"], expected["p"], atol=1e-4)
        self.assertEqual(len(sim._support_fixture.active), 1)

    def test_loose_upper_and_screw_do_not_arm_fixture(self):
        sim = self.make()
        for factory in (airplane_wing, toy_screw):
            sim.edit_spawn("w:loose", factory().to_dict(), {"at": dict(x=.2, y=0, z=.21)})
            self.place(sim, "w:loose")
            self.assertFalse(sim._support_fixture.pending)
            self.assertFalse(sim._support_fixture.active)
            sim.edit_remove("w:loose")

    def test_inverted_fastened_component_locks_through_upper_contact(self):
        from steward.assets.generators import electric_screwdriver
        from steward.scene.fasteners import FastenConfig, records
        from steward.world.sim import WorldSim
        spec = importlib.util.spec_from_file_location("support_fixture_demo", Path(__file__).parents[1] / "scripts/fasten_demo.py")
        demo = importlib.util.module_from_spec(spec); spec.loader.exec_module(demo)
        world = scene()
        world.entities[0].id = demo.BASE
        world.entities[2].id = demo.FIXTURE
        for factory, eid, x, y in ((airplane_wing, demo.WING, .2, 0),
                                   (toy_screw, demo.SCREWS[0], -.2, .1),
                                   (toy_screw, demo.SCREWS[1], -.2, -.1),
                                   (electric_screwdriver, demo.TOOL, .2, .2)):
            ent = factory(); ent.id = eid; ent.pose = exact(pose([x, y, .201]))
            world.entities.append(ent)
        sim = WorldSim(world.to_dict(), fasten=FastenConfig(), support_fixture=True)
        self.addCleanup(sim.stop)
        demo.hold(sim, demo.BASE, sim._entity_pose(demo.BASE))
        demo.release(sim); sim.step_seconds(1.)
        self.assertEqual(len(sim._support_fixture.active), 1)
        # Existing seating/fastening must still work with the base clamped.
        demo.prepare(sim)
        for screw in demo.SCREWS:
            self.assertTrue(demo.tighten(sim, screw))
        self.assertEqual(len(records(sim.template)), 2)
        demo.release(sim); demo.set_pose(sim, demo.TOOL, pose([.6, 0, .3]))
        demo.set_pose(sim, demo.FIXTURE, sim._entity_pose(demo.WING))
        sim.command(demo.FIXTURE, {"gripper": "close"})
        self.assertIn(sim.robots[demo.FIXTURE].attached, sim._fastening.component(demo.BASE))
        self.assertFalse(sim._support_fixture.active)
        demo.hold(sim, demo.BASE, pose([0, 0, .26], [0, 1, 0, 0]))
        demo.release(sim); sim.step_seconds(2.)
        self.assertEqual(len(sim._support_fixture.active), 1)
        self.assertEqual(sum(sim.data.eq_active), 4)  # three fastener welds plus the fixture
        self.assertTrue(any(r.a["entity"] == demo.WING and r.b["entity"] == "w:platform"
                            for r in sim._relations() if r.type == "supported-by"))
        before = copy.deepcopy(sim._entity_pose(demo.BASE))
        bid = sim._root_body(demo.BASE)
        sim.data.xfrc_applied[bid] = [.8, .2, 0, 0, 0, .003]
        sim.step_seconds(.5); sim.data.xfrc_applied[bid] = 0; sim.step_seconds(.5)
        self.assertLess(np.linalg.norm(np.array(before["p"]) - sim._entity_pose(demo.BASE)["p"]), 1e-4)
        self.assertEqual(len(records(sim.template)), 2)

    def test_repeated_restore_regrasp_release_relocks_connected_assembly(self):
        """Exercise a second flip cycle with a connected member bearing the load."""
        from steward.assets.generators import electric_screwdriver
        from steward.scene.fasteners import FastenConfig, records
        from steward.world.sim import WorldSim
        spec = importlib.util.spec_from_file_location("restore_demo", Path(__file__).parents[1] / "scripts/fasten_demo.py")
        demo = importlib.util.module_from_spec(spec); spec.loader.exec_module(demo)
        world = scene(); world.entities[0].id = demo.BASE; world.entities[2].id = demo.FIXTURE
        for factory, eid, x, y in ((airplane_wing, demo.WING, .2, 0),
                (toy_screw, demo.SCREWS[0], -.2, .1), (toy_screw, demo.SCREWS[1], -.2, -.1),
                (electric_screwdriver, demo.TOOL, .2, .2)):
            ent = factory(); ent.id = eid; ent.pose = exact(pose([x, y, .201])); world.entities.append(ent)
        sim = WorldSim(world.to_dict(), fasten=FastenConfig(), support_fixture=True)
        self.addCleanup(sim.stop)
        demo.prepare(sim)
        for screw in demo.SCREWS: self.assertTrue(demo.tighten(sim, screw))
        demo.release(sim); demo.set_pose(sim, demo.TOOL, pose([.6, 0, .3]))
        for orientation in ([0, 1, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]):
            # Real gripper command must detach the existing fixture before transport.
            demo.set_pose(sim, demo.FIXTURE, sim._entity_pose(demo.BASE))
            sim.command(demo.FIXTURE, {"gripper": "close"})
            self.assertIsNotNone(sim.robots[demo.FIXTURE].attached)
            self.assertFalse(sim._support_fixture.active)
            demo.hold(sim, demo.BASE, pose([0, 0, .26], orientation))
            demo.release(sim); sim.step_seconds(2.)
            self.assertEqual(len(sim._support_fixture.active), 1)
            self.assertEqual(sum(sim.data.eq_active), 4)
            self.assertEqual(len(records(sim.template)), 2)
        self.assertTrue(any(r.a["entity"] == demo.WING and r.b["entity"] == "w:platform"
                            for r in sim._relations() if r.type == "supported-by"))


if __name__ == "__main__":
    unittest.main()
