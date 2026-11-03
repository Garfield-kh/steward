"""Embodiments: which robot body a scenario puts in the world, registered in one place.

A scene's robot entity carries the identity of its body in ``asset.id`` and ``category`` (the
generators publish ``steward-bot`` and ``cobot-magic``). This registry maps that id to what the
rest of the system needs to know about a body without opening its model: a label for the console,
the ``steward`` subcommand that runs its Ego Runtime, and the five-layer profile document under
``steward/embodiments/``. Like :mod:`steward.phases`, one entry per implementation; a body the
registry does not know is shown by its asset id and gets no Ego Runtime.

Switching bodies is never a live switch: the body is compiled into the world model, so a scenario
with another body means a new episode and another Ego Runtime process.
"""

from __future__ import annotations

EMBODIMENTS: dict[str, dict] = {
    "steward-bot": {
        "label": "Steward 移动厨房机器人",
        "description": "平面移动底盘 + 升降柱 + 直线伸出臂 + 平行爪；1.5 m 固定头部 RGB-D 相机（俯视 30°）",
        "profile": "steward-bot.yaml",
        "ego": "ego",
        # Its missions are clients of the Ego API (`steward chores --api`, the Ego Harness).
        "runs_mission": False,
        "skills": True,
    },
    "cobot-magic": {
        "label": "AgileX Cobot Magic 双 Piper",
        "description": "固定机架 + 双 Piper 6-DoF 臂 + 平行爪；顶部巡视 RGB-D + 双腕部 RGB-D 相机",
        "profile": "cobot-magic.yaml",
        "ego": "cobot-ego",
        # The twin's Ego API has no skill endpoints yet, so the scenario's scripted mission runs
        # inside its runtime (`steward cobot-ego --task`).
        "runs_mission": True,
        # ...and an Ego Harness task from the console would have nothing to call.
        "skills": False,
    },
}


def embodiment_of(entity: dict) -> str | None:
    """The registered embodiment id of a robot entity (a scene-state dict), else ``None``."""
    asset = entity.get("asset")
    candidates = [asset.get("id") if isinstance(asset, dict) else asset,
                  (entity.get("category") or {}).get("value")]
    return next((c for c in candidates if c in EMBODIMENTS), None)


def body_id_of(entity: dict) -> str | None:
    """What the entity calls its body, registered or not: the asset id, else the category."""
    asset = entity.get("asset")
    if isinstance(asset, dict):
        asset = asset.get("id")
    return asset or (entity.get("category") or {}).get("value")


def describe(embodiment_id: str | None, body_id: str | None = None) -> dict:
    """``{id, label, description, registered, skills}`` for the console; an unregistered body is
    named by its asset id so the operator sees what is there rather than a blank."""
    if embodiment_id in EMBODIMENTS:
        entry = EMBODIMENTS[embodiment_id]
        return {"id": embodiment_id, "label": entry["label"], "description": entry["description"],
                "registered": True, "skills": entry["skills"]}
    shown = body_id or embodiment_id or "unknown"
    return {"id": shown, "label": f"未注册本体 · {shown}", "description": "这个机器人资产没有登记在本体注册表里，"
            "控制台能显示它，但没有 Ego Runtime 会去驱动它", "registered": False, "skills": False}


def scenario_embodiments(scenario: dict) -> dict[str, dict]:
    """Robot id -> :func:`describe` entry for every robot a loaded scenario lists."""
    entities = {e["id"]: e for e in scenario["sceneState"]["entities"]}
    out = {}
    for rid in scenario["robots"]:
        ent = entities.get(rid, {})
        out[rid] = describe(embodiment_of(ent), body_id_of(ent))
    return out


def embodiment_catalog() -> list[dict]:
    """Every registered body with its label and description."""
    return [describe(k) for k in EMBODIMENTS]
