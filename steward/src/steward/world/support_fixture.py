"""World-only, release-armed workholding; never exported as robot capability/state.

An ideal fixture welds a settled assembly to its actual supporting link. It does
not alter contact masks, fastener records, or any unattached part's dynamics.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import numpy as np

from steward.math3d import pose_compose, pose_inverse
from steward.scene.fasteners import features, unit


def enabled_from_env():
    from steward.assembly_profile import enabled
    value = os.environ.get("STEWARD_SUPPORT_FIXTURE", "1" if enabled() else "0")
    if value not in ("0", "1"):
        raise ValueError("STEWARD_SUPPORT_FIXTURE must be 0 or 1")
    return value == "1"


def candidates(scene):
    """Free rigid objects with multiple interface groups, independent of names.

    Explicit hole ``group`` metadata wins when complete. Older assets have no
    groups: distinct (link, outward axis) families define their interfaces.
    A single two-hole upper plate therefore does not qualify.
    """
    holes = {}
    for feature in features(scene)["screw-hole"]:
        holes.setdefault(feature.entity, []).append(feature)
    result = set()
    for eid, rows in holes.items():
        if all(f.spec.get("group") is not None for f in rows):
            groups = {str(f.spec["group"]) for f in rows}
        else:
            groups = {(f.link, *np.round(unit(f.spec["axis"]), 6)) for f in rows}
        if len(groups) >= 2:
            result.add(eid)
    return result


def reserve(build, scene, enabled):
    """Add inactive welds only to the world build, before model compilation.

    One slot per candidate/support link avoids rebinding MuJoCo body IDs live.
    Disabled is an exact no-op, including the generated XML.
    """
    slots = {}
    if not enabled or scene.mode != "world":
        return slots
    eligible = candidates(scene)
    supports = [(e.id, link.id) for e in scene.entities if e.role in ("object", "environment")
                for link in e.rigid.links if link.geometry.get("collision") and
                (e.role == "environment" or any(a.value == "support-surface" for a in link.affordances))]
    root = ET.fromstring(build.xml)
    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    for eid in sorted(eligible):
        for support, link in supports:
            if eid == support:
                continue
            name = f"support-fixture-{len(slots)}"
            slots[eid, support, link] = name
            ET.SubElement(equality, "weld", name=name,
                          body1=build.names[support].bodies[link],
                          body2=next(iter(build.names[eid].bodies.values())),
                          active="false", solref="0.004 1", solimp="0.999 0.9999 0.001",
                          torquescale="0.1")
    if slots:
        build.xml = ET.tostring(root, encoding="unicode")
    return slots


class SupportFixture:
    # Simulated-time bounded pose dwell tolerates contact solver chatter without
    # mistaking a translating/rotating body for a resting one.
    dwell = .25
    position_tolerance = .0005
    angle_tolerance = .01

    def __init__(self, sim, slots, previous=None):
        import mujoco
        self.sim = sim
        self.slots = {key: mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
                      for key, name in slots.items()}
        self.eligible = {key[0] for key in slots}
        self.pending = {}
        self.active = {}
        if previous is not None:
            self.pending = {e: None for e in previous.pending if e in self.eligible}
            for key, rel in previous.active.items():
                if key in self.slots:
                    self._activate(key, rel, emit=False)

    def released(self, entity):
        for eid in self.sim._fastening.component(entity) & self.eligible:
            self.pending[eid] = None

    def detach(self, entity):
        members = self.sim._fastening.component(entity)
        for eid in members:
            self.pending.pop(eid, None)
        for key in list(self.active):
            if key[0] in members:
                self.sim.data.eq_active[self.slots[key]] = False
                del self.active[key]
                self.sim.events.append(dict(t=self.sim.time, type="support-fixture/release", entity=key[0]))

    def _activate(self, key, rel, emit=True):
        eq = self.slots[key]
        self.sim.model.eq_data[eq, :3] = 0
        self.sim.model.eq_data[eq, 3:10] = [*rel["p"], *rel["q"]]
        self.sim.data.eq_active[eq] = True
        self.active[key] = rel
        if emit:
            self.sim.events.append(dict(t=self.sim.time, type="support-fixture/lock", entity=key[0],
                                        support=key[1], link=key[2]))

    def _pose(self, bid):
        return dict(p=self.sim.data.xpos[bid].tolist(), q=self.sim.data.xquat[bid].tolist())

    def _stationary(self, before, after):
        if before.keys() != after.keys():
            return False
        for eid, at in after.items():
            old = before[eid]
            distance = np.linalg.norm(np.asarray(at["p"]) - old["p"])
            angle = 2 * np.arccos(np.clip(abs(np.dot(at["q"], old["q"])), 0, 1))
            if (not np.isfinite(distance + angle) or distance > self.position_tolerance
                    or angle > self.angle_tolerance):
                return False
        return True

    def update(self):
        """Called with fresh position/velocity/contact caches after a physics step."""
        sim = self.sim
        for eid in list(self.pending):
            members = sim._fastening.component(eid)
            if any(rs.holding_any(e) for rs in sim.robots.values() for e in members):
                self.detach(eid)
                continue
            # Only one fixture per rigidly fastened component.
            if any(key[0] in members for key in self.active):
                self.pending.pop(eid)
                continue
            contacts = {}
            for con in sim.data.contact[:sim.data.ncon]:
                nz = float(con.frame[2])
                if abs(nz) < .7 or con.dist > .001:
                    continue
                lower, upper = (con.geom1, con.geom2) if nz > 0 else (con.geom2, con.geom1)
                lb, ub = (int(sim.model.geom_bodyid[g]) for g in (lower, upper))
                support = sim._body_entity.get(lb)
                key = (eid, support, sim._body_link.get(lb))
                if sim._body_entity.get(ub) in members and support not in members and key in self.slots:
                    contacts[key] = lb
            if not contacts:
                self.pending[eid] = None
                continue
            key = sorted(contacts)[0]
            support_inverse = pose_inverse(self._pose(contacts[key]))
            relative = {e: pose_compose(support_inverse, self._pose(sim._root_body(e))) for e in members}
            previous = self.pending[eid]
            if previous is None or previous[0] != key or not self._stationary(previous[2], relative):
                self.pending[eid] = (key, sim.time, relative)
            elif sim.time - previous[1] >= self.dwell:
                self._activate(key, relative[eid])
                del self.pending[eid]
