"""Chores baseline residual-failure diagnostics: support slices and receipt briefs."""

from __future__ import annotations

from steward.world.chores_diagnostics import (
    phase_support_diagnostics,
    receipt_brief,
    support_slice_ego,
    support_slice_world,
)


def test_receipt_brief_keeps_blockers_and_plan_spot():
    brief = receipt_brief({
        "skill": "place",
        "feasible": True,
        "withinRiskLimit": True,
        "risk": 0.1,
        "issues": ["ok"],
        "blockers": [{"id": "e:9", "category": "chair", "movable": True, "parkAt": [[0.1, 0.2]]}],
        "plan": {"spot": [1.0, 2.0], "on": "e:1", "target": "e:2"},
    })
    assert brief["skill"] == "place" and brief["planSpot"] == [1.0, 2.0]
    assert brief["blockers"][0]["id"] == "e:9" and brief["blockers"][0]["parkAt"] == [[0.1, 0.2]]


def test_support_slice_world_lists_bowls_wine_and_parents():
    scene = {
        "entities": [
            {"id": "w:sideboard-1", "category": {"value": "sideboard"}},
            {"id": "w:table-1", "category": {"value": "table"}},
            {"id": "w:bowl-red", "category": {"value": "bowl"}},
            {"id": "w:bowl-green", "category": {"value": "bowl"}},
            {"id": "w:bowl-blue", "category": {"value": "bowl"}},
            {"id": "w:wine-glass", "category": {"value": "wine_glass"}},
            {"id": "w:mug", "category": {"value": "mug"}},
            {"id": "w:teddy", "category": {"value": "teddy_bear"}},
        ],
        "relations": [
            {"type": "supported-by", "a": {"entity": "w:bowl-red"}, "b": {"entity": "w:sideboard-1"}},
            {"type": "supported-by", "a": {"entity": "w:bowl-green"}, "b": {"entity": "w:bowl-red"}},
            {"type": "supported-by", "a": {"entity": "w:bowl-blue"}, "b": {"entity": "w:table-1"}},
            {"type": "supported-by", "a": {"entity": "w:wine-glass"}, "b": {"entity": "w:table-1"}},
            {"type": "supported-by", "a": {"entity": "w:mug"}, "b": {"entity": "w:sideboard-1"}},
            {"type": "supported-by", "a": {"entity": "w:teddy"},
             "b": {"entity": "w:sideboard-1", "link": "drawer"}},
        ],
    }
    rows = {r["id"]: r for r in support_slice_world(scene)}
    assert "w:teddy" not in rows, "drawer toys are out of the sideboard residual slice"
    assert rows["w:bowl-blue"]["on"]["entity"] == "w:table-1"
    assert rows["w:bowl-green"]["on"]["entity"] == "w:bowl-red"
    assert rows["w:wine-glass"]["on"]["entity"] == "w:table-1"
    assert rows["w:sideboard-1"]["on"] is None


def test_support_slice_ego_keeps_on_and_size_for_tableware():
    summary = {
        "objects": [
            {"id": "e:1", "category": "sideboard", "role": "sideboard", "on": None,
             "position": [0, 0, 0.4], "sizeM": [0.5, 1.2, 0.8], "existence": 1.0},
            {"id": "e:2", "category": "bowl", "role": "bowl", "on": "e:1", "color": "blue",
             "position": [0.1, 0, 0.9], "extentM": [0.13, 0.13, 0.05], "existence": 0.9},
            {"id": "e:3", "category": "wall", "role": "wall", "on": None, "existence": 1.0},
        ],
    }
    rows = support_slice_ego(summary)
    assert [r["id"] for r in rows] == ["e:1", "e:2"]
    assert rows[1]["on"] == "e:1" and rows[1]["sizeM"] == [0.13, 0.13, 0.05]


def test_phase_support_diagnostics_bundles_check_outcome():
    scene = {
        "entities": [{"id": "w:table-1", "category": {"value": "table"}}],
        "relations": [],
    }
    packet = phase_support_diagnostics(scene, {"objects": []}, ws_ok=False, detail="wine missing")
    assert packet["wsOk"] is False and packet["detail"] == "wine missing"
    assert packet["world"][0]["id"] == "w:table-1"


def test_stack_logs_plan_and_skips_when_everything_is_already_placed(monkeypatch):
    """Residual failures need an explicit skip trail when ``on`` already matches the target."""
    from steward import chores as chores_mod
    from steward.chores import Chores

    board = {"id": "e:board", "category": "sideboard", "role": "sideboard", "on": None,
             "position": [0, 0, 0.4], "extentM": [0.5, 1.2, 0.8], "sizeM": [0.5, 1.2, 0.8],
             "yawDeg": 0.0, "existence": 1.0, "categories": {"sideboard": 1.0}, "uncertaintyM": 0.01}
    red = {"id": "e:r", "category": "bowl", "role": "bowl", "on": "e:board", "color": "red",
           "position": [0, 0, 0.9], "extentM": [0.22, 0.22, 0.07], "sizeM": [0.22, 0.22, 0.07],
           "yawDeg": 0.0, "existence": 1.0, "categories": {"bowl": 1.0}, "uncertaintyM": 0.01,
           "lastSeen": 1e9}
    green = {**red, "id": "e:g", "on": "e:r", "color": "green",
             "extentM": [0.17, 0.17, 0.06], "sizeM": [0.17, 0.17, 0.06]}
    blue = {**red, "id": "e:b", "on": "e:g", "color": "blue",
            "extentM": [0.13, 0.13, 0.05], "sizeM": [0.13, 0.13, 0.05]}
    wine = {"id": "e:w", "category": "wine_glass", "role": "wine_glass", "on": "e:board",
            "color": "clear", "position": [0.2, 0, 0.9], "extentM": [0.06, 0.06, 0.15],
            "sizeM": [0.06, 0.06, 0.15], "yawDeg": 0.0, "existence": 1.0,
            "categories": {"wine_glass": 1.0}, "uncertaintyM": 0.01, "lastSeen": 1e9}
    summary = {"objects": [board, red, green, blue, wine],
               "self": {"x": 0, "y": 0, "yawDeg": 0, "holding": None}}

    monkeypatch.setattr(chores_mod, "_summary", lambda api: summary)
    monkeypatch.setattr(chores_mod, "_settle", lambda api, seconds=25.0, want=2: summary)
    monkeypatch.setattr(chores_mod, "_approach_to_observe", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "simtime_sleep", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "_refresh",
                        lambda api, obj: next(o for o in summary["objects"] if o["id"] == obj["id"]))
    monkeypatch.setattr(chores_mod, "_refresh_or_recover",
                        lambda api, obj: next(o for o in summary["objects"] if o["id"] == obj["id"]))
    runner = Chores("http://unused")
    monkeypatch.setattr(runner, "_confirmed", lambda role: board)
    monkeypatch.setattr(runner, "_with_known_support", lambda obj, expect=None: obj)
    monkeypatch.setattr(runner, "_doing", lambda *a, **k: None)
    runner.stack_on_sideboard()
    assert runner.log[0]["step"] == "observe-before-plan"
    assert any(e.get("step") == "plan-sideboard" for e in runner.log)
    assert sum(1 for e in runner.log if e.get("step") == "skip") == 4
    assert any(e.get("reason") == "already-nested" and e.get("object") == "e:b" for e in runner.log)
