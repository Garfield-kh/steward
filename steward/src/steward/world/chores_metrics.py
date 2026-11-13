"""Comparable chores baseline metrics (Iteration A): World State phase rates and Observer views.

Used by ``scripts/chores_baseline.py`` so before/after runs share one definition of success and
of visible-object recall (closed-drawer contents excluded from the denominator). Observer also
records stack/drawer specials from :func:`stack_drawer_metrics` on every scored snapshot.
"""

from __future__ import annotations

from steward.chores import PHASES
from steward.world.chores_check import drawer_contents, drawers_closed

OBSERVER_KEYS = (
    "recall", "falsePositives", "meanPositionErrorM", "meanSizeErrorM", "categoryAccuracy",
    "matchedObjects", "gtObjects", "esObjects", "selfPositionErrorM", "selfYawErrorRad", "phase",
    "nestedBowlSupports", "gtNestedBowlSupports", "nestedBowlSupportRecall",
    "drawersTracked", "gtDrawers", "drawerContentsTracked", "gtDrawerContents",
    "drawerContentRecall",
)

STACK_DRAWER_KEYS = (
    "nestedBowlSupports", "gtNestedBowlSupports", "nestedBowlSupportRecall",
    "drawersTracked", "gtDrawers", "drawerContentsTracked", "gtDrawerContents",
    "drawerContentRecall",
)


def observer_brief(record: dict | None) -> dict | None:
    """Keep the Observer fields that Iteration A compares across baseline runs."""
    if not record:
        return None
    return {k: record.get(k) for k in OBSERVER_KEYS if k in record}


def closed_drawer_object_count(scene: dict) -> int:
    """GT objects that stay invisible while both drawers are shut."""
    if not drawers_closed(scene):
        return 0
    return sum(len(items) for items in drawer_contents(scene).values())


def visible_recall(record: dict | None, scene: dict) -> float | None:
    """``matchedObjects / (gtObjects - closed-drawer contents)``, capped at 1."""
    if not record or record.get("matchedObjects") is None or record.get("gtObjects") is None:
        return None
    visible_gt = max(record["gtObjects"] - closed_drawer_object_count(scene), 0)
    if visible_gt == 0:
        return None
    return round(min(record["matchedObjects"], visible_gt) / visible_gt, 4)


def _category(entity: dict) -> str:
    cat = entity.get("category")
    return cat["value"] if isinstance(cat, dict) else str(cat)


def _support_map(scene: dict) -> dict[str, str]:
    """Map each supported entity id → the entity that holds it (link ignored)."""
    out: dict[str, str] = {}
    for rel in scene.get("relations", []):
        if rel.get("type") != "supported-by":
            continue
        a, b = rel.get("a", {}), rel.get("b", {})
        if "entity" in a and "entity" in b:
            out[a["entity"]] = b["entity"]
    return out


def _bowl_ids(scene: dict) -> set[str]:
    return {e["id"] for e in scene.get("entities", []) if _category(e) == "bowl"}


def stack_drawer_metrics(ego: dict, world: dict, matches: dict[str, str]) -> dict:
    """Observer specials for nested bowls and drawer inventory (Iteration A).

    ``matches`` maps Ego ids to World ids from the Observer association. Nested-bowl support recall
    is the share of GT bowl-on-bowl supports whose both ends are matched and whose ES support link
    also runs between those tracks. Drawer-content recall is matched drawer-held ES objects over
    GT objects resting in a drawer link (0 when drawers are shut and nothing is tracked inside).
    """
    ego_support = _support_map(ego)
    world_support = _support_map(world)
    ego_bowls = _bowl_ids(ego)
    world_bowls = _bowl_ids(world)
    nested_es = sum(1 for oid, parent in ego_support.items() if oid in ego_bowls and parent in ego_bowls)
    nested_gt = sum(1 for oid, parent in world_support.items() if oid in world_bowls and parent in world_bowls)
    reverse = {gid: eid for eid, gid in matches.items()}
    recovered = 0
    for child, parent in world_support.items():
        if child not in world_bowls or parent not in world_bowls:
            continue
        es_child, es_parent = reverse.get(child), reverse.get(parent)
        if es_child and es_parent and ego_support.get(es_child) == es_parent:
            recovered += 1
    nested_recall = round(recovered / nested_gt, 4) if nested_gt else None

    ego_drawers = {e["id"] for e in ego.get("entities", []) if _category(e) == "drawer"}
    gt_drawers = sum(1 for e in world.get("entities", []) if _category(e) == "drawer")
    # World drawer parts are often ``w:cabinet-1/drawer`` style perception ids; count body joints too.
    if gt_drawers == 0:
        gt_drawers = sum(
            1 for e in world.get("entities", [])
            for j in e.get("body", {}).get("joints", []) if j.get("id") == "drawer_slide")
    drawer_contents_es = sum(1 for oid, parent in ego_support.items() if parent in ego_drawers)
    gt_in_drawers = sum(len(v) for v in drawer_contents(world).values())
    matched_drawer_contents = 0
    for eid, gid in matches.items():
        if ego_support.get(eid) in ego_drawers and any(gid in items for items in drawer_contents(world).values()):
            matched_drawer_contents += 1
    content_recall = round(matched_drawer_contents / gt_in_drawers, 4) if gt_in_drawers else None

    return {
        "nestedBowlSupports": nested_es,
        "gtNestedBowlSupports": nested_gt,
        "nestedBowlSupportRecall": nested_recall,
        "drawersTracked": len(ego_drawers),
        "gtDrawers": gt_drawers,
        "drawerContentsTracked": drawer_contents_es,
        "gtDrawerContents": gt_in_drawers,
        "drawerContentRecall": content_recall,
    }


def summarize(results: list[dict], config: dict) -> dict:
    """Aggregate per-seed chores baseline records into rates and mean Observer stats."""
    n = len(results)
    phase_stats: dict[str, dict] = {}
    for name in PHASES:
        attempted = [r for r in results if any(p["name"] == name for p in r["phases"])]
        confirmed = [r for r in attempted if next(p for p in r["phases"] if p["name"] == name).get("wsOk")]
        completed = [r for r in attempted if next(p for p in r["phases"] if p["name"] == name).get("completed")]
        phase_stats[name] = {
            "attempted": len(attempted),
            "completed": len(completed),
            "wsConfirmed": len(confirmed),
            "wsRate": round(len(confirmed) / n, 4) if n else None,
            "completeRate": round(len(completed) / n, 4) if n else None,
        }

    def mean(values: list[float | None]) -> float | None:
        xs = [v for v in values if v is not None]
        return round(sum(xs) / len(xs), 4) if xs else None

    explores = [r.get("afterExplore") or {} for r in results]
    return {
        "config": config,
        "seeds": n,
        "fullRoundOk": sum(1 for r in results if r["ok"]),
        "fullRoundRate": round(sum(1 for r in results if r["ok"]) / n, 4) if n else None,
        "meanWallS": mean([r.get("wallS") for r in results]),
        "phases": phase_stats,
        "afterExplore": {
            "meanVisibleRecall": mean([e.get("visibleRecall") for e in explores]),
            "meanRecall": mean([e.get("recall") for e in explores]),
            "meanPositionErrorM": mean([e.get("meanPositionErrorM") for e in explores]),
            "meanSizeErrorM": mean([e.get("meanSizeErrorM") for e in explores]),
            "meanCategoryAccuracy": mean([e.get("categoryAccuracy") for e in explores]),
            "meanFalsePositives": mean([float(e["falsePositives"]) for e in explores
                                        if e.get("falsePositives") is not None]),
            "meanDrawersTracked": mean([float(e["drawersTracked"]) for e in explores
                                        if e.get("drawersTracked") is not None]),
            "leaks": sum(int(e.get("leaks") or 0) for e in explores),
        },
        "failures": [{"seed": r["seed"], "error": r.get("error")} for r in results if not r.get("ok")],
    }
