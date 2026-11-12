"""Diagnostics for chores baseline seeds: support slices and compact skill receipts.

Persisted next to ``result.json`` so a World-State mismatch can be traced to skipped moves,
wrong ``on`` targets, or clearing that left tableware off the sideboard — without re-running.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from steward.world.chores_check import _category, _support

# Sideboard / table residual failures are about these entities.
FOCUS_CATEGORIES = ("bowl", "wine_glass", "mug", "cup", "sideboard", "table", "cabinet")


def receipt_brief(receipt: dict) -> dict:
    """Compact Prediction Receipt fields for a chores log line."""
    plan = receipt.get("plan") or {}
    blockers = []
    for b in receipt.get("blockers") or []:
        blockers.append({
            "id": b.get("id"),
            "category": b.get("category"),
            "movable": b.get("movable"),
            "parkAt": b.get("parkAt"),
        })
    return {
        "skill": receipt.get("skill"),
        "feasible": receipt.get("feasible"),
        "withinRiskLimit": receipt.get("withinRiskLimit"),
        "risk": receipt.get("risk"),
        "issues": list(receipt.get("issues") or [])[:8],
        "blockers": blockers[:6],
        "planSpot": plan.get("spot"),
        "planTarget": plan.get("target"),
        "planOn": plan.get("on"),
    }


def support_slice_world(scene: dict) -> list[dict]:
    """World ``supported-by`` rows for tableware and the furniture they rest on."""
    cats = _category(scene)
    support = _support(scene)
    focus = {eid for eid, cat in cats.items() if cat in FOCUS_CATEGORIES}
    for eid in list(focus):
        parent = support.get(eid)
        if parent and parent.get("entity"):
            focus.add(parent["entity"])
    rows = []
    for eid in sorted(focus):
        on = support.get(eid)
        rows.append({
            "id": eid,
            "category": cats.get(eid),
            "on": None if on is None else {"entity": on.get("entity"), "link": on.get("link")},
        })
    return rows


def support_slice_ego(summary: dict) -> list[dict]:
    """Ego summary objects that matter for sideboard nesting and tableware placement."""
    rows = []
    for o in summary.get("objects") or []:
        role = o.get("role") or o.get("category")
        cat = o.get("category")
        if role not in FOCUS_CATEGORIES and cat not in FOCUS_CATEGORIES:
            continue
        rows.append({
            "id": o.get("id"),
            "category": cat,
            "role": o.get("role"),
            "on": o.get("on"),
            "position": o.get("position"),
            "sizeM": o.get("extentM") or o.get("sizeM"),
            "color": o.get("color"),
            "existence": o.get("existence"),
        })
    return rows


def phase_support_diagnostics(scene: dict, summary: dict | None, *, ws_ok: bool | None = None,
                              detail: str | None = None) -> dict:
    """One phase-end packet: World slice, Ego slice, and the check outcome."""
    return {
        "wsOk": ws_ok,
        "detail": detail,
        "world": support_slice_world(scene),
        "ego": support_slice_ego(summary or {}),
    }


def write_json(path: Path, payload: Any) -> None:
    """Write ``payload`` as UTF-8 JSON with a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_jsonl(path: Path, record: dict) -> None:
    """Append one JSON object as a line (crash-safe chore trail)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
