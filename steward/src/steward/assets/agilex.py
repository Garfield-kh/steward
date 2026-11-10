"""Task-ready adapter for the imported split_aloha_mid_360_with_piper USD asset.

Meshes, inertias, arm transforms and joint limits come from the imported template. The simulation
profile supplies servos, a planar mobile-base abstraction and an RGB-D rig. Wheel joints are fixed;
this is a task simulator, not a wheel-contact or real-hardware controller.
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import tomllib

from steward.assets.cobot import ROOT_ACTUATOR
from steward.assets.generators import head_link
from steward.scene.schema import Entity, Joint, Link, SceneState, exact, pose
from steward.scene.validate import require_valid


def agilex_usd(config_path: Path | None = None) -> dict:
    """Build the registered USD body from its local template and simulation calibration.

    ``config_path`` defaults to ``STEWARD_AGILEX_USD_CONFIG`` or the shipped TOML profile.
    Both arms start with their six revolute joints at zero radians.
    A missing asset fails with import instructions; no substitute geometry is loaded.
    """
    path = config_path or Path(os.environ.get("STEWARD_AGILEX_USD_CONFIG",
                               str(Path(__file__).resolve().parents[3] / "config/agilex-usd.toml")))
    cfg = tomllib.loads(path.read_text())
    positive = ("arm_kp", "arm_damping", "arm_effort", "lift_kp", "lift_damping", "lift_effort", "finger_kp", "finger_damping")
    for key in (*positive, "lift_home", "survey_pitch_deg", "work_pitch_deg"):
        value = cfg.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or (key in positive and value <= 0):
            raise ValueError(f"Invalid {key} in {path}: expected a finite {'positive ' if key in positive else ''}number")
    template = (path.parent / cfg["template"]).resolve()
    if not template.is_file():
        raise ValueError(f"USD embodiment template missing: {template}. Run steward import-usd with --name agilex-usd; "
                         "see README's USD embodiment instructions or set STEWARD_AGILEX_USD_CONFIG.")
    entity = json.loads(template.read_text())
    body = entity["body"]
    link_names = {}
    for link in body["links"]:
        parts = link["id"].split("/")
        name = parts[-1]
        if len(parts) > 1 and parts[-2] in ("fl", "fr"):
            name = ("L_" if parts[-2] == "fl" else "R_") + name
        link_names[link["id"]] = {"base_link": "base", "lifting_link": "frame"}.get(name, name)
    required = {"base", "frame", "top_camera_link", *[f"{a}_link{i}" for a in ("L", "R") for i in range(1, 9)]}
    if not required <= set(link_names.values()) or len(set(link_names.values())) != len(link_names):
        raise ValueError("USD template is not the expected dual-Piper mobile robot (missing or duplicate links)")
    removed = {key for key, name in link_names.items() if name.startswith("dummy_base_")}
    joint_names = {}
    joints = []
    for joint in body["joints"]:
        if joint["parent"] in removed or joint["child"] in removed:
            continue
        old = joint["id"]
        name = old.split("/")[-1]
        for prefix, arm in (("fl_joint", "L_joint"), ("fr_joint", "R_joint")):
            if name.startswith(prefix):
                name = name.replace(prefix, arm, 1)
        joint["parent"], joint["child"] = link_names[joint["parent"]], link_names[joint["child"]]
        if joint["child"] in ("L_arm_base", "R_arm_base"):
            name = joint["child"][0] + "_mount"
        if name == "lifting_joint":
            name = "lift"
        joint["id"] = joint_names[old] = name
        if "wheel" in joint["child"]:
            joint.update(type={"value": "fixed"}, q={"value": []}, qd={"value": []})
            for key in ("axis", "anchor", "limits", "actuator", "dynamics"):
                joint.pop(key, None)
        elif name == "lift" or name.startswith(("L_joint", "R_joint")):
            finger = name.endswith(("7", "8"))
            group = "lift" if name == "lift" else "finger" if finger else "arm"
            if "actuator" in joint:
                joint["actuator"] = {"kind": "position", "gains": {"kp": cfg[f"{group}_kp"]},
                                     "limits": {"effort": cfg.get(f"{group}_effort", cfg["arm_effort"])}}
            joint["dynamics"] = {"value": {"damping": cfg[f"{group}_damping"]}}
            if name == "lift":
                value = cfg["lift_home"]
            elif finger:
                value = 0.05 if name.endswith("7") else -0.05
            else:
                value = 0.0
            limits = joint["limits"]["value"]
            if not limits["lo"] <= value <= limits["hi"]:
                raise ValueError(f"Initial {name}={value} lies outside the USD limits {limits}")
            joint["q"] = {"value": [value]}
        joints.append(joint)
    links = []
    for link in body["links"]:
        if link["id"] in removed:
            continue
        link["id"] = link_names[link["id"]]
        # USD visual/collision duplicates share one mesh. Keep one visible copy of each part.
        geometry = link["geometry"]
        visual = {json.dumps(g, sort_keys=True): g for g in geometry.get("visual", [])}
        geometry["visual"] = list(visual.values())
        if link["id"] in ("base", "box_link", "frame") or link["id"] in {f"{a}_link{i}" for a in ("L", "R") for i in range(1, 6)}:
            geometry["collision"] = copy.deepcopy(geometry["visual"])
        if link["id"] in ("L_link6", "R_link6"):
            finger = next(j for j in joints if j["id"] == link["id"][0] + "_joint7")
            link["affordances"] = [{"value": "grasp-point"},
                {"value": {"tcp-offset": pose(finger["origin"]["value"]["p"])}}]
        if link["id"] == "top_camera_link":
            link["sensors"] = head_link(camera_pitch_deg=cfg["survey_pitch_deg"],
                                        wide_pitch_deg=cfg["work_pitch_deg"], head_size=(320, 240)).to_dict()["sensors"]
        links.append(link)
    # Put the floor at entity z=0 so scenario replacement preserves the spawn pose.
    links.insert(0, Link(id="ground", name="planar base reference", inertial=exact({"mass": 0.001})).to_dict())
    joints.insert(0, Joint(id="base_height", parent="ground", child="base", type=exact("fixed"),
                           origin=exact(entity["pose"]["value"]), q=exact([]), qd=exact([])).to_dict())
    body.update(links=links, joints=joints, root={"value": "planar"}, rootActuator=copy.deepcopy(ROOT_ACTUATOR))
    for coupling in body.get("couplings", []):
        for key in ("joint1", "joint2"):
            coupling[key] = joint_names[coupling[key]]
    entity.update(id="w:agilex-1", category={"value": "agilex-usd"}, asset={"id": "agilex-usd"}, pose={"value": pose()})
    require_valid(SceneState(mode="world", entities=[Entity.from_dict(entity)]).to_dict())
    return entity
