"""Steward Console: a browser page on the World Control API port.

The console belongs to the Steward plane: it shows World State and every robot's Ego-State side
by side, with Observer metrics, events and clock/episode controls. It is never served on a robot
port.
"""

from __future__ import annotations

import math
from pathlib import Path

from steward.math3d import quat_rotate, yaw_from_quat
from steward.world.thinking import thinking_state

CONSOLE_HTML = Path(__file__).with_name("console.html")

UPRIGHT_COS = math.cos(math.radians(20.0))  # same threshold the relative-poses goal uses


def _box(entity_id: str, category: str, role: str, obb) -> dict:
    """One ground-truth box for the console. ``yaw`` is the heading of the entity's frame; for an
    object lying face-down (its top face against the support) that heading reads as the mirrored
    one, so ``upright`` says whether the entity's +z axis still points up and the yaw can be read
    as a heading at all."""
    return {"id": entity_id, "category": category, "role": role, "center": [round(v, 4) for v in obb.center],
            "size": [round(v, 4) for v in obb.size], "yaw": round(yaw_from_quat(obb.quat), 4),
            "upright": quat_rotate(obb.quat, [0.0, 0.0, 1.0])[2] > UPRIGHT_COS}


def console_state(host) -> dict:
    """Everything one console refresh needs, in a single response."""
    sim = host.sim
    if sim is None:
        return {"loaded": False, "scenario": None}
    ground_truth = []
    for ent in sim.template.entities:
        if ent.role == "environment":
            for link in ent.rigid.links:
                if link.id != "floor":
                    ground_truth.append(_box(f"{ent.id}/{link.id}", "wall", "environment", sim.entity_obb(ent.id, [link.id])))
                else:
                    floor = sim.entity_obb(ent.id, [link.id])
                    ground_truth.append(_box(f"{ent.id}/floor", "floor", "environment", floor))
        else:
            ground_truth.append(_box(ent.id, ent.category.value, ent.role, sim.entity_obb(ent.id)))
    status = sim.status()
    robots = {}
    # The robots of the loaded scenario, not every robot port the host has ever opened: after a
    # switch to a scenario with another body the old robot's port is still bound, but there is
    # nothing behind it to show.
    for rid in (host.scenario or {}).get("robots", list(host.robot_servers)):
        ego = host.observer.latest_ego.get(rid)
        record = host.observer.latest.get(rid, {})
        robots[rid] = {
            "ego": _ego_view(ego) if ego else None,
            "metrics": {k: record.get(k) for k in ("t", "recall", "falsePositives", "meanPositionErrorM",
                                                    "meanSizeErrorM", "meanNees3", "categoryAccuracy",
                                                    "selfPositionErrorM", "phase")},
            "matches": record.get("matches", {}),
            "errorHistory": [[round(h["t"], 2), h["meanPositionErrorM"], h.get("meanSizeErrorM"), h.get("recall"),
                              h.get("phase")]
                             for h in host.observer.history.get(rid, []) if h.get("meanPositionErrorM") is not None][-300:],
            "camera": f"robot:{rid}",
            "embodiment": host.embodiments.get(rid),
        }
    floor = next((g for g in ground_truth if g["category"] == "floor"), None)
    cx, cy = (floor["center"][0], floor["center"][1]) if floor else (0.0, 0.0)
    span = max(floor["size"][0], floor["size"][1]) if floor else 6.0
    # A scenario that says where its work happens is framed on it: the room-sized default puts a
    # tabletop robot five metres away, where a three-centimetre grasp reads as no motion at all.
    view = {"lookat": [round(cx, 3), round(cy, 3), 0.5], "distance": round(span * 1.35, 3), "azimuth": 90.0,
            "elevation": -50.0, "fovy": sim.free_camera_fovy}
    if (host.scenario or {}).get("view"):
        view.update({key: value for key, value in host.scenario["view"].items() if key in view})
    return {
        "loaded": True,
        "view": view,
        "scenario": (host.scenario or {}).get("name"),
        "embodiments": host.embodiments,
        "phase": host.phase_state(),
        "task": (host.scenario or {}).get("task"),
        "goal": host.goal_progress(),
        "t": round(status["t"], 2),
        "paused": status["paused"],
        "speed": status["speed"],
        "groundTruth": ground_truth,
        "holding": status["robots"],
        "events": status["events"],
        "robots": robots,
        "thinking": _thinking(host),
        "perceptionAlerts": host.active_perception_alerts(),
    }


def _thinking(host) -> dict[str, dict]:
    """The robot agent's live thinking, from its Harness home when the console knows it."""
    config = host.chat.config if host.chat is not None else None
    home = getattr(config, "robot_home", None)
    if home is None:
        return {}
    return thinking_state({"w:robot-1": str(home)})


def _ego_view(ego: dict) -> dict:
    objects = []
    self_pose = None
    for ent in ego["entities"]:
        p = ent["pose"]["value"]
        cov = (ent["pose"].get("unc") or {}).get("cov")
        sigma = math.sqrt((cov[0] + cov[7]) / 2) if cov else 0.0
        if ent["id"] == ego.get("self"):
            self_pose = {"x": p["p"][0], "y": p["p"][1], "yaw": yaw_from_quat(p["q"]), "sigma": sigma}
            continue
        size = ent["body"]["links"][0]["geometry"]["collision"][0]["size"]
        objects.append({"id": ent["id"], "category": ent["category"]["value"], "center": p["p"], "size": size,
                        "yaw": yaw_from_quat(p["q"]), "sigma": sigma, "existence": ent.get("existence", 1.0),
                        "color": (ent.get("stateVars", {}).get("color") or {}).get("value"),
                        "role": (ent.get("stateVars", {}).get("role") or {}).get("value")})
    held = [r["a"]["entity"] for r in ego.get("relations", []) if r["type"] == "attached"]
    return {"version": ego.get("version"), "t": ego.get("stamp"), "self": self_pose, "objects": objects, "holding": held}
