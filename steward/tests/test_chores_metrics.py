"""Baseline metric definitions used to compare Iteration A before/after runs."""

from __future__ import annotations

from steward.world.chores_metrics import (
    closed_drawer_object_count,
    stack_drawer_metrics,
    summarize,
    visible_recall,
)


def test_visible_recall_excludes_closed_drawer_contents():
    scene = {
        "entities": [
            {"id": "w:sideboard-1", "category": {"value": "sideboard"}, "body": {"joints": [
                {"id": "drawer_slide", "q": {"value": [0.0]}}]}},
            {"id": "w:cabinet-1", "category": {"value": "cabinet"}, "body": {"joints": [
                {"id": "drawer_slide", "q": {"value": [0.0]}}]}},
        ],
        "relations": [
            {"type": "supported-by", "a": {"entity": "w:teddy"}, "b": {"entity": "w:sideboard-1", "link": "drawer"}},
            {"type": "supported-by", "a": {"entity": "w:plane"}, "b": {"entity": "w:cabinet-1", "link": "drawer"}},
            {"type": "supported-by", "a": {"entity": "w:bowl"}, "b": {"entity": "w:table-1"}},
        ],
    }
    assert closed_drawer_object_count(scene) == 2
    assert visible_recall({"matchedObjects": 8, "gtObjects": 10}, scene) == 1.0
    assert visible_recall({"matchedObjects": 7, "gtObjects": 10}, scene) == 0.875


def test_visible_recall_keeps_open_drawer_contents_in_the_denominator():
    scene = {
        "entities": [
            {"id": "w:sideboard-1", "category": {"value": "sideboard"}, "body": {"joints": [
                {"id": "drawer_slide", "q": {"value": [0.3]}}]}},
        ],
        "relations": [
            {"type": "supported-by", "a": {"entity": "w:teddy"}, "b": {"entity": "w:sideboard-1", "link": "drawer"}},
        ],
    }
    assert closed_drawer_object_count(scene) == 0
    assert visible_recall({"matchedObjects": 9, "gtObjects": 10}, scene) == 0.9


def test_stack_drawer_metrics_score_nested_bowls_and_drawer_inventory():
    ego = {
        "entities": [
            {"id": "e:1", "category": {"value": "sideboard"}},
            {"id": "e:2", "category": {"value": "bowl"}},
            {"id": "e:3", "category": {"value": "bowl"}},
            {"id": "e:4", "category": {"value": "drawer"}},
            {"id": "e:5", "category": {"value": "teddy_bear"}},
        ],
        "relations": [
            {"type": "supported-by", "a": {"entity": "e:2"}, "b": {"entity": "e:1"}},
            {"type": "supported-by", "a": {"entity": "e:3"}, "b": {"entity": "e:2"}},
            {"type": "supported-by", "a": {"entity": "e:5"}, "b": {"entity": "e:4"}},
        ],
    }
    world = {
        "entities": [
            {"id": "w:sideboard-1", "category": {"value": "sideboard"}, "body": {"joints": [
                {"id": "drawer_slide", "q": {"value": [0.2]}}]}},
            {"id": "w:cabinet-1", "category": {"value": "cabinet"}, "body": {"joints": [
                {"id": "drawer_slide", "q": {"value": [0.0]}}]}},
            {"id": "w:bowl-red", "category": {"value": "bowl"}, "body": {"joints": []}},
            {"id": "w:bowl-green", "category": {"value": "bowl"}, "body": {"joints": []}},
            {"id": "w:teddy", "category": {"value": "teddy_bear"}, "body": {"joints": []}},
        ],
        "relations": [
            {"type": "supported-by", "a": {"entity": "w:bowl-red"}, "b": {"entity": "w:sideboard-1"}},
            {"type": "supported-by", "a": {"entity": "w:bowl-green"}, "b": {"entity": "w:bowl-red"}},
            {"type": "supported-by", "a": {"entity": "w:teddy"}, "b": {"entity": "w:sideboard-1", "link": "drawer"}},
        ],
    }
    matches = {"e:2": "w:bowl-red", "e:3": "w:bowl-green", "e:5": "w:teddy", "e:1": "w:sideboard-1"}
    metrics = stack_drawer_metrics(ego, world, matches)
    assert metrics["nestedBowlSupports"] == 1
    assert metrics["gtNestedBowlSupports"] == 1
    assert metrics["nestedBowlSupportRecall"] == 1.0
    assert metrics["drawersTracked"] == 1
    assert metrics["gtDrawers"] == 2
    assert metrics["drawerContentsTracked"] == 1
    assert metrics["gtDrawerContents"] == 1
    assert metrics["drawerContentRecall"] == 1.0


def test_summarize_reports_per_phase_world_state_rates():
    results = [
        {"seed": 0, "ok": True, "wallS": 10.0,
         "afterExplore": {"visibleRecall": 1.0, "recall": 0.8, "meanPositionErrorM": 0.02,
                          "meanSizeErrorM": 0.03, "categoryAccuracy": 1.0, "falsePositives": 0, "leaks": 0},
         "phases": [{"name": "table", "wsOk": True, "completed": True},
                    {"name": "sideboard", "wsOk": True, "completed": True},
                    {"name": "drawers", "wsOk": True, "completed": True}]},
        {"seed": 1, "ok": False, "wallS": 12.0, "error": "sideboard: nested bowls merged",
         "afterExplore": {"visibleRecall": 0.9, "recall": 0.7, "meanPositionErrorM": 0.04,
                          "meanSizeErrorM": 0.05, "categoryAccuracy": 0.9, "falsePositives": 1, "leaks": 0},
         "phases": [{"name": "table", "wsOk": True, "completed": True},
                    {"name": "sideboard", "wsOk": False, "completed": True}]},
    ]
    summary = summarize(results, {"phase": "phase1"})
    assert summary["fullRoundOk"] == 1 and summary["fullRoundRate"] == 0.5
    assert summary["phases"]["table"]["wsRate"] == 1.0
    assert summary["phases"]["sideboard"]["wsRate"] == 0.5
    assert summary["phases"]["drawers"]["wsRate"] == 0.5
    assert summary["afterExplore"]["meanVisibleRecall"] == 0.95
    assert summary["failures"] == [{"seed": 1, "error": "sideboard: nested bowls merged"}]
