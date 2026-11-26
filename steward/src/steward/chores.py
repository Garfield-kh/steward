"""Scripted chores over the Ego API, repeated round after round.

One round has three phases:

1. **table**: the tableware (bowls, wine glass, mug) stands in one row on the dining table,
   ordered from the smallest to the largest footprint;
2. **sideboard**: the tableware moves to the sideboard top; bowls nest largest to smallest in
   one stack, the other pieces stand in a row beside it (pieces already in place stay);
3. **drawers**: both drawers are opened, the contents of the sideboard drawer and the cabinet
   drawer are exchanged, and both drawers are closed.

Everything is located in the Ego-State by role (the scene-level reading of each object's category), colour and estimated size; placements use
explicit offsets in the support's own frame, and every motion goes through a Prediction Receipt.
"""

from __future__ import annotations

from steward.assembly_profile import enabled as _assembly_enabled


import itertools
import math
import re
import time

from steward.diagnosis import CATEGORY_ZH, Diagnosis, describe, diagnose
from steward.mission import (
    Blocked,
    MissionFailed,
    _approach_to_observe,
    _explore,
    _footprint_m,
    _refresh,
    _run,
    _safe_receipt,
    _wait_for_control,
    call,
    infeasible,
    simtime_sleep,
)
from steward.net import HttpError
from steward.world.chores_diagnostics import receipt_brief

TABLEWARE = ("bowl", "wine_glass", "mug", "cup")
# Deployed contents of the two chores drawers (phase0-chores). Swap only these; tableware and
# look-alike toy parts misread in an open drawer must not be carried.
DRAWER_SWAP_ROLES = ("teddy_bear", "pistol", "toy_airplane", "pill_bottle", "perfume_bottle")
# Meal pieces that are never stored in these drawers: skip without a close look when their role
# says so. ``cup`` is omitted — a perfume bottle is often read as a cup until a nearer view.
DRAWER_MEAL_SKIP = ("bowl", "wine_glass", "mug")
DRAWER_CONTENT_MARGIN_M = 0.08  # XY slack around the open drawer's measured box
DRAWER_FLOOR_TOL_M = 0.06  # content base may sit slightly under a tall measured drawer floor
DRAWER_CONFIRM_VIEWS = 2  # close looks before skipping an in-drawer track that is not yet whitelisted
DRAWER_CONFIRM_DISTANCE = 0.9
# Temporary staging on the furniture top: keep drawer approach clear while contents wait there.
# Spots stay inward of the rim: ESim rejects places ~4 cm from the edge under support uncertainty.
# Keep staged pieces inward of the long ends: edge spots are hard to grasp (seed2f pistol miss).
STAGE_ALONG_FRAC = 0.32  # spread staged pieces across this share of the long half-length
STAGE_BACK_M = 0.06  # nudge away from the open drawer along the short axis
STAGE_EDGE_CLEAR_M = 0.12  # keep at least this much of board half-size as rim clearance
STAGE_RECOVER_RADIUS_M = 0.35  # rematch a staged drawer item only within this of its stage XY
# Second staged match pass when the tight neighbourhood is empty (renumbered centre drift).
STAGE_RECOVER_RADIUS_WIDE_M = 0.50
# Stage rematch accepts slightly soft tracks (cabinet leg can weaken a top piece to ~0.4–0.6).
STAGE_MATCH_MIN_EXISTENCE = 0.40
STAGE_PICK_DISTANCE = 0.75  # closer than confirm distance so pick sees a usable pose on the top
STAGE_PICK_CLOSE_DISTANCE = 0.55  # when existence is soft, step in before committing the grasp
STAGE_PICK_MISS_DISTANCE = 0.45  # after a staged grasp miss, freshen from even closer
STAGE_PICK_MIN_EXISTENCE = 0.75  # below this, freshen closer / skip a doomed pick receipt
STAGE_CONFIRM_VIEWS = 2  # headings for the first post-place stage look
STAGE_RECONFIRM_VIEWS = 2  # second pass when the first look does not rebind a whitelist track
# After placing into an open drawer, rematch near the release spot inside that drawer box.
DRAWER_LAND_RECOVER_RADIUS_M = 0.45
# Unstaged drawer-carry footprint rematch (cabinet→sideboard leg): same role, same neighbourhood.
# Wider than stage recover so a renumbered in-drawer track still matches; far shorter than the
# room so a staged piece on the opposite top cannot be adopted as the cabinet carry.
DRAWER_FOOTPRINT_REMATCH_M = 1.5
PLACE_BASE_BLOCKED = ("base is blocked", "no progress")
PLACE_BLOCKED_TRIES = 2  # place attempts including the first, after a short retreat while holding
# Staged / drawer-carry picks: distinct freshen headings, then closer looks after soft existence.
PICK_BASE_BLOCKED_TRIES = 4
PICK_GRASP_RETRY = ("missed", "not closing", "re-plan", "no longer in the ego state")
# Soft pick receipts: existence ghosts should not be grasped; mild pose uncertainty may still
# be within the risk limit and must not burn every retry on look-only cycles.
PICK_EXISTENCE_ISSUES = ("existence probability",)
PICK_OBSERVE_ISSUES = ("observe the target more closely",)
# Cap look-only deferrals so soft receipts cannot consume every staged pick attempt.
# Pose uncertainty: one closer look is enough. Soft existence often needs a rematch + second
# look (seed2c teddy stayed at 0.60 after one defer, then a doomed grasp missed).
PICK_OBSERVE_DEFER_MAX = 1
PICK_EXISTENCE_DEFER_MAX = 2
# Open-drawer place often fails from across the room: stand at the drawer mouth first, then retry
# dock-infeasible predictions from a new heading after a short retreat while holding.
DRAWER_PLACE_ALIGN_DISTANCE = 0.95
PLACE_DOCK_INFEASIBLE = ("no feasible approach", "dock pose collides")
# False 0 cm opening (missing host / swollen seat) is not a dock collide; retry after refresh.
PLACE_DRAWER_TOO_SHUT = ("too little", "open it first")
# Open part has no room left (crowding / false occupancy): retreat, re-align, try again.
PLACE_DRAWER_FULL = ("no free spot",)
PLACE_DRAWER_TRIES = 3  # align + up to two retreat/re-align cycles for drawer places
# Close skill can report success from base travel while the slide is still ajar. Re-read
# openingM and retry while above the world gate (:func:`drawers_closed` uses 0.02 m).
DRAWER_CLOSE_OK_M = 0.02  # match :func:`drawers_closed` tolerance in chores_check
DRAWER_CLOSE_TRIES = 3  # first close + up to two verify/retry cycles
# ES openingM can sit under the world gate while the joint is still a few centimetres out
# (seed2: verified at 0.5 cm, world ``drawer_slide`` still > 2 cm). One explicit seat push
# after the gate reading drives the slide home; predict must get ``distance`` or it refuses
# as already closed when opening < 5 cm.
DRAWER_CLOSE_SEAT_M = 0.10
# Same-role tracks whose footprints differ by at most this are treated as ghost duplicates.
# A much larger same-label track is often a misread of a different deployed piece — keep both.
DRAWER_ROLE_DUPE_FP_TOL_M = 0.04
# Same-role tracks count as ghosts only when also this close in XY (distinct bottles stay).
DRAWER_ROLE_DUPE_XY_M = 0.12
# Unconfirmed in-drawer tracks at least this wide get one extra confirm pass (airplane-scale
# look-alikes); smaller bottles already had DRAWER_CONFIRM_VIEWS.
DRAWER_CONFIRM_LARGE_M = 0.12
# Max leftover occupants to move out of a drawer before placing the other set into it.
DRAWER_EVACUATE_MAX = 4
TOP_SUPPORT_ROLES = ("sideboard", "cabinet", "table", "counter")
# Drawer hosts used as temporary stage tops: never rematch these onto a counter when the
# remembered role was sideboard/cabinet (kitchen counter can be nearer in XY after a role flip).
STAGE_HOST_ROLES = ("sideboard", "cabinet")
STAGE_HOST_REMATCH_ROLES = ("sideboard", "cabinet", "table")
ROW_GAP = 0.06
FLOOR_LEVEL_M = 0.15  # tableware whose base is lower than this lies on the floor
FRESH_S = 45.0  # a piece planned into today's row must have been seen this recently
TABLE_ROW_Y = 0.18
VIEW_DISTANCE = 1.1
# Close-look rescan: tracks that may be tableware but are not yet planned as such.
WEAK_TABLEWARE_MIN = 0.15  # categories mass worth a closer look for a named tableware label
STEMWARE_MIN_H_M = 0.14  # height of a stemware-like box (wine glass prior is ~0.19 m)
STEMWARE_MAX_FOOT_M = 0.14  # long footprint side of that box
RESCAN_SETTLE_S = 2.0
RESCAN_MAX_VIEWS = 3  # distinct headings tried before leaving a doubtful track unplanned
# Preferred approach headings (rad) for successive close looks; distinct sides of the same object.
RESCAN_VIEW_OFFSETS = (0.0, 1.0, -1.0)


def _summary(api: str) -> dict:
    return call("GET", f"{api}/ego/summary")


def _objects(summary: dict, category: str) -> list[dict]:
    """Objects whose role is ``category``: the scene-level reading, so that furniture or tableware
    whose own best category is a look-alike (a sideboard read as a counter) is still found."""
    return [o for o in summary["objects"] if o.get("role", o["category"]) == category and o["existence"] >= 0.5]


def _fresh_objects(summary: dict, category: str) -> list[dict]:
    """Objects of a category that were seen recently enough to act on.

    A track outlives the view it came from: last seen a couple of minutes ago, at the pose the
    robot had then, it still reads as tableware in the ego state. Planning a pick on it spends
    the attempt on empty space. Everything the next minutes are planned around — the pieces to
    arrange, the bowls to stack — has just been looked at from close by, so age is a cheap and
    honest filter."""
    return [o for o in _objects(summary, category) if o.get("lastSeenAgoS", 0.0) <= FRESH_S]


def _supported_first(objects: list[dict]) -> list[dict]:
    """The objects known to stand on something, or all of them when nothing is."""
    supported = [o for o in objects if o.get("on") is not None]
    return supported or objects


def _off_floor(objects: list[dict]) -> list[dict]:
    """Drop tracks whose base is on the floor (mats/markers misread as tableware)."""
    return [o for o in objects if _bottom(o) >= FLOOR_LEVEL_M]


def _near_support_surface(obj: dict, summary: dict, margin: float = 0.15) -> bool:
    """Whether ``obj`` already has ``on``, or its XY lies on/near a furniture support surface.

    Used when planning moves: a fresh track with ``on is None`` (common for thin or transparent
    pieces) stays in the list if it sits over a table or sideboard; a wall-corner fragment far
    from every support surface stays out."""
    if obj.get("on") is not None:
        return True
    surfaces = [o for cat in ("table", "sideboard", "cabinet", "counter") for o in _objects(summary, cat)]
    if not surfaces:
        return True  # no surface known yet — keep candidates rather than emptying the plan
    ox, oy = obj["position"][0], obj["position"][1]
    for surf in surfaces:
        e = surf.get("extentM") or surf["sizeM"]
        yaw = math.radians(surf.get("yawDeg", 0.0))
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = ox - surf["position"][0], oy - surf["position"][1]
        local_x, local_y = c * dx + s * dy, -s * dx + c * dy
        if abs(local_x) <= e[0] / 2 + margin and abs(local_y) <= e[1] / 2 + margin:
            return True
    return False


def _sideboard_plan_bowls(summary: dict) -> list[dict]:
    """Bowls for the sideboard plan/inventory: fresh, off the floor, near a support if ``on`` unknown.

    Unlike :func:`_supported_first`, a missing ``on`` does not drop a piece when siblings already
    have support — pick time uses :meth:`Chores._with_known_support` instead."""
    bowls, _others, _rejected = _sideboard_plan_inventory(summary)
    return bowls


def _sideboard_plan_others(summary: dict) -> list[dict]:
    """Non-bowl tableware for the sideboard plan/inventory (same filters as bowls)."""
    _bowls, others, _rejected = _sideboard_plan_inventory(summary)
    return others


def _tableware_plan_candidates(summary: dict) -> list[dict]:
    """Objects that look like tableware for sideboard planning or rejection diagnostics.

    Includes tracks whose ``role`` or ``category`` is a tableware label, and tracks whose category
    distribution still puts mass on a tableware label (mis-roled pieces that planning would miss)."""
    out, seen = [], set()
    for o in summary.get("objects", []):
        oid = o.get("id")
        if oid in seen:
            continue
        role = o.get("role", o.get("category"))
        cat = o.get("category")
        cats = o.get("categories") or {}
        if role in TABLEWARE or cat in TABLEWARE or any(c in TABLEWARE for c in cats):
            seen.add(oid)
            out.append(o)
    return out


def _sideboard_reject_reasons(obj: dict, summary: dict) -> list[str]:
    """Why ``obj`` is left out of the sideboard move list (empty ⇒ it would be selected).

    Reason codes: ``role`` (not a planned tableware role), ``existence``, ``stale``, ``floor``,
    ``far`` (no ``on`` and XY not near a support surface)."""
    role = obj.get("role", obj.get("category"))
    reasons: list[str] = []
    if role not in TABLEWARE:
        reasons.append("role")
        return reasons  # planning never considers it; further filters are moot for the plan slot
    if obj.get("existence", 0.0) < 0.5:
        reasons.append("existence")
    if obj.get("lastSeenAgoS", 0.0) > FRESH_S:
        reasons.append("stale")
    if _bottom(obj) < FLOOR_LEVEL_M:
        reasons.append("floor")
    if not _near_support_surface(obj, summary):
        reasons.append("far")
    return reasons


def _reject_brief(obj: dict, reasons: list[str]) -> dict:
    """Compact rejected-candidate row for ``plan-sideboard`` diagnostics."""
    return {
        **_plan_obj(obj),
        "lastSeenAgoS": obj.get("lastSeenAgoS"),
        "reasons": reasons,
    }


def _sideboard_plan_inventory(summary: dict) -> tuple[list[dict], list[dict], list[dict]]:
    """Bowls (large→small), other tableware, and rejected tableware candidates with reasons.

    Selection matches :func:`_sideboard_plan_bowls` / :func:`_sideboard_plan_others`; rejected rows
    explain every other tableware-like track so open-plan omissions are greppable from the log."""
    bowls, others, rejected = [], [], []
    for obj in _tableware_plan_candidates(summary):
        reasons = _sideboard_reject_reasons(obj, summary)
        if reasons:
            rejected.append(_reject_brief(obj, reasons))
            continue
        role = obj.get("role", obj.get("category"))
        if role == "bowl":
            bowls.append(obj)
        else:
            others.append(obj)
    bowls = sorted(bowls, key=_area, reverse=True)
    others = sorted(others, key=_area, reverse=True)
    rejected = sorted(rejected, key=lambda r: r.get("id") or "")
    return bowls, others, rejected


def _stemware_shaped(obj: dict) -> bool:
    """Whether the measured box is tall and narrow like a wine glass (geometry-only hint)."""
    size = obj.get("extentM") or obj.get("sizeM") or [0.0, 0.0, 0.0]
    if len(size) < 3:
        return False
    foot = float(max(size[0], size[1]))
    height = float(size[2])
    return height >= STEMWARE_MIN_H_M and 0.03 < foot <= STEMWARE_MAX_FOOT_M


def _doubtful_tableware_for_rescan(summary: dict) -> list[dict]:
    """Tracks worth a close look before planning: unlabeled or weakly labeled tableware on a support.

    Covers (1) ``role`` rejected pieces that still carry tableware mass in ``categories``, and
    (2) stemware-shaped boxes labeled ``unknown`` with no tableware mass yet (transparent glass
    often starts that way). Already-selected plan bowls/others are skipped."""
    bowls, others, rejected = _sideboard_plan_inventory(summary)
    planned = {o["id"] for o in bowls} | {o["id"] for o in others}
    out, seen = [], set()
    for obj in rejected:
        if obj["id"] in planned or obj["id"] in seen:
            continue
        if "role" in (obj.get("reasons") or []):
            seen.add(obj["id"])
            out.append(obj)
    for obj in summary.get("objects", []):
        oid = obj.get("id")
        if oid in planned or oid in seen or obj.get("existence", 0.0) < 0.5:
            continue
        if obj.get("lastSeenAgoS", 0.0) > FRESH_S or _bottom(obj) < FLOOR_LEVEL_M:
            continue
        if not _near_support_surface(obj, summary):
            continue
        role = obj.get("role", obj.get("category"))
        cat = obj.get("category")
        cats = obj.get("categories") or {}
        weak = any(float(cats.get(t, 0.0)) >= WEAK_TABLEWARE_MIN for t in TABLEWARE)
        unlabeled = role == "unknown" or cat == "unknown"
        if unlabeled and (weak or _stemware_shaped(obj)):
            seen.add(oid)
            out.append(obj)
        elif weak and role not in TABLEWARE and cat not in TABLEWARE:
            seen.add(oid)
            out.append(obj)
    return sorted(out, key=lambda o: o.get("id") or "")


def _plan_ready_tableware(obj: dict, summary: dict) -> bool:
    """Whether ``obj`` would enter the sideboard move list: a confirmed tableware role and filters."""
    return not _sideboard_reject_reasons(obj, summary)


def _summary_object(summary: dict, oid: str) -> dict | None:
    """The summary row for ``oid``, or None when the track is gone."""
    return next((o for o in summary.get("objects", []) if o.get("id") == oid), None)


def _one(summary: dict, category: str) -> dict:
    found = _objects(summary, category)
    if not found:
        raise MissionFailed(f"no {category} in the ego state")
    return max(found, key=lambda o: o["existence"])


def _area(obj: dict) -> float:
    e = obj["extentM"] or obj["sizeM"]
    return e[0] * e[1]


def _width(obj: dict) -> float:
    e = obj["extentM"] or obj["sizeM"]
    return max(e[0], e[1])


def tableware(summary: dict) -> list[dict]:
    """Tableware tracks ordered from the smallest to the largest footprint.

    Things that cost a pick per round and never yield are left out: what lies on the floor (a flat
    thing on the floor read as a bowl or plate is a mat or a marker), what has not been seen for a
    while (a track outlives the view it came from), and what stands on nothing (a cup at table
    height with no support under it is a corner of a wall or the edge of a leg). When no support
    surface has been read at all, nothing is supported, and then all of them are kept rather than
    none."""
    items = [o for cat in TABLEWARE for o in _fresh_objects(summary, cat) if _bottom(o) >= FLOOR_LEVEL_M]
    supported = [o for o in items if o.get("on") is not None]
    return sorted(supported or items, key=_area)


def _bottom(obj: dict) -> float:
    extent = obj.get("extentM") or obj["sizeM"]
    return obj["position"][2] - extent[2] / 2


def row_offsets(widths: list[float], gap: float = ROW_GAP) -> list[float]:
    """Centres of a row of items with the given widths, centred on 0."""
    total = sum(widths) + gap * (len(widths) - 1)
    out, x = [], -total / 2
    for w in widths:
        out.append(x + w / 2)
        x += w + gap
    return out


def _offset(support: dict, along: float, across: float) -> list[float]:
    """An offset in the support's frame with ``along`` on its longer horizontal side, whichever
    local axis that is (asset frames and fitted boxes differ)."""
    e = support["extentM"] or support["sizeM"]
    return [along, across] if e[0] >= e[1] else [across, along]


def _world(support: dict, offset: list[float]) -> list[float]:
    yaw = math.radians(support["yawDeg"])
    c, s = math.cos(yaw), math.sin(yaw)
    return [support["position"][0] + c * offset[0] - s * offset[1], support["position"][1] + s * offset[0] + c * offset[1]]


def _plan_obj(obj: dict) -> dict:
    """Compact object fields recorded when a phase plans what it will move."""
    return {
        "id": obj.get("id"),
        "category": obj.get("category"),
        "role": obj.get("role"),
        "on": obj.get("on"),
        "color": obj.get("color"),
        "sizeM": obj.get("extentM") or obj.get("sizeM"),
        "position": obj.get("position"),
        "existence": obj.get("existence"),
    }


def _match_by_footprint(summary: dict, obj: dict, *, require_deployed: bool = False) -> dict | None:
    """Best summary object matching ``obj``'s footprint (and preferably role/color).

    Used after a close look when :func:`_refresh` still fails because the renumbered track drifted
    beyond the local search radius (stacked bowls can be re-seen a metre away after a drive).

    When ``require_deployed`` is true, only :data:`DRAWER_SWAP_ROLES` candidates with the same
    expected role (``stagedRole`` / role / category) within :data:`DRAWER_FOOTPRINT_REMATCH_M` of
    ``obj``'s last position are considered. Without the role and neighbourhood gates, a cabinet
    airplane rematch adopted a staged pistol on the opposite sideboard top (similar footprint).
    """
    want = _footprint_m(obj)
    if want <= 0:
        return None
    role = obj.get("stagedRole") or obj.get("role") or obj.get("category")
    color = obj.get("color")
    fp_tol = max(0.04, 0.35 * want)
    pos = obj.get("position")
    cands = []
    for o in summary.get("objects", []):
        if o["existence"] < 0.5 or abs(_footprint_m(o) - want) > fp_tol:
            continue
        if require_deployed:
            if not _drawer_deployed(o):
                continue
            o_role = o.get("role") or o.get("category")
            if role not in (o_role, o.get("category")):
                continue
            if pos is not None and math.dist(o["position"][:2], pos[:2]) > DRAWER_FOOTPRINT_REMATCH_M:
                continue
        cands.append(o)
    if not cands:
        return None

    def rank(o: dict) -> tuple:
        role_ok = role in (o.get("role"), o["category"])
        color_ok = bool(color) and o.get("color") == color
        dist = math.dist(o["position"][:2], pos[:2]) if pos is not None else 0.0
        return (0 if role_ok else 1, 0 if color_ok else 1, abs(_footprint_m(o) - want), dist)

    return min(cands, key=rank)


def _stage_xy_of(obj: dict) -> list[float] | None:
    """World XY recorded when a drawer item was temporarily placed on a furniture top."""
    xy = obj.get("stageXy")
    if isinstance(xy, (list, tuple)) and len(xy) >= 2:
        return [float(xy[0]), float(xy[1])]
    return None


def _drawer_land_xy_of(obj: dict) -> list[float] | None:
    """World XY recorded when a whitelist piece was released into an open drawer."""
    xy = obj.get("drawerLandXy")
    if isinstance(xy, (list, tuple)) and len(xy) >= 2:
        return [float(xy[0]), float(xy[1])]
    return None


def _anchor_xy_of(obj: dict) -> list[float] | None:
    """Best XY hint for rematch: stage top, then in-drawer landing, then last pose."""
    return (_stage_xy_of(obj) or _drawer_land_xy_of(obj)
            or (list(obj["position"][:2]) if obj.get("position") else None))


def _match_in_drawer(summary: dict, expected: dict, drawer: dict,
                     *, spot: list[float] | None = None) -> dict | None:
    """Whitelist track inside ``drawer`` matching ``expected``'s role/footprint, nearest ``spot``.

    Prefers ``on`` = drawer (:func:`_contents`), then any deployed role whose pose lies in the
    open box (support links often lag a fresh release).
    """
    want_role = expected.get("stagedRole") or expected.get("role") or expected.get("category")
    want_fp = _footprint_m(expected)
    fp_tol = max(0.04, 0.4 * want_fp) if want_fp > 0 else 0.08

    def role_fp_ok(o: dict) -> bool:
        role = o.get("role") or o.get("category")
        if want_role not in (role, o.get("category")):
            return False
        if want_fp > 0 and abs(_footprint_m(o) - want_fp) > fp_tol:
            return False
        return True

    cands = [o for o in _contents(summary, drawer) if role_fp_ok(o)]
    if not cands:
        cands = [o for o in summary.get("objects", [])
                 if o.get("existence", 0.0) >= 0.5 and _drawer_deployed(o) and role_fp_ok(o)
                 and _in_open_drawer_volume(o, drawer)]
    if not cands:
        return None
    if spot is not None and len(spot) >= 2:
        near = [o for o in cands
                if math.dist(o["position"][:2], spot[:2]) <= DRAWER_LAND_RECOVER_RADIUS_M]
        if near:
            cands = near
        return min(cands, key=lambda o: math.dist(o["position"][:2], spot[:2]))
    return max(cands, key=_area)


def _match_staged_drawer_item(summary: dict, expected: dict, xy: list[float] | None,
                              *, radius: float = STAGE_RECOVER_RADIUS_M,
                              require_role: bool = True) -> dict | None:
    """Whitelist track nearest ``xy`` that fits ``expected``'s role/footprint (never tableware).

    When ``require_role`` is true (default), a nearby teddy is not accepted for a staged pistol —
    that used to overwrite the staged id and lose the gun's ``stageXy`` anchor. Existence may be
    as low as :data:`STAGE_MATCH_MIN_EXISTENCE`; among matches, higher existence wins.
    """
    if xy is None or len(xy) < 2:
        return None
    want_role = expected.get("stagedRole") or expected.get("role") or expected.get("category")
    want_fp = _footprint_m(expected)
    fp_tol = max(0.04, 0.4 * want_fp) if want_fp > 0 else 0.08
    cands = []
    for o in summary.get("objects", []):
        if o.get("existence", 0.0) < STAGE_MATCH_MIN_EXISTENCE or not _drawer_deployed(o):
            continue
        if math.dist(o["position"][:2], xy[:2]) > radius:
            continue
        if want_fp > 0 and abs(_footprint_m(o) - want_fp) > fp_tol:
            continue
        role = o.get("role") or o.get("category")
        if require_role and want_role not in (role, o.get("category")):
            continue
        cands.append(o)
    if not cands:
        return None

    def rank(o: dict) -> tuple:
        role = o.get("role") or o.get("category")
        role_ok = want_role in (role, o.get("category"))
        fp = abs(_footprint_m(o) - want_fp) if want_fp > 0 else 0.0
        # Prefer correct role, then stronger existence, then footprint / distance.
        return (0 if role_ok else 1, -float(o.get("existence") or 0.0), fp,
                math.dist(o["position"][:2], xy[:2]))

    return min(cands, key=rank)


def _match_staged_near(summary: dict, expected: dict, xy: list[float] | None,
                       *, radius: float = STAGE_RECOVER_RADIUS_M) -> dict | None:
    """:func:`_match_staged_drawer_item` then a wider staged radius if the tight neighbourhood is empty."""
    found = _match_staged_drawer_item(summary, expected, xy, radius=radius)
    if found is None and radius + 1e-9 < STAGE_RECOVER_RADIUS_WIDE_M:
        found = _match_staged_drawer_item(
            summary, expected, xy, radius=STAGE_RECOVER_RADIUS_WIDE_M)
    return found


def _refresh_or_recover(api: str, obj: dict, *, require_deployed: bool = False) -> dict:
    """Like :func:`_refresh`, but after a lost id: look again, retry refresh, then match by footprint
    anywhere in the summary. Stacking must not abort solely because a bowl was renumbered mid-drive.

    ``require_deployed`` restricts footprint recovery to :data:`DRAWER_SWAP_ROLES` (drawer carry).
    """
    try:
        return _refresh(api, obj)
    except MissionFailed as first:
        try:
            _approach_to_observe(api, obj, distance=0.9)
        except MissionFailed:
            pass
        try:
            return _refresh(api, obj)
        except MissionFailed:
            pass
        found = _match_by_footprint(_summary(api), obj, require_deployed=require_deployed)
        if found is not None:
            return found
        raise first


def _with_stage_anchor(obj: dict, source: dict) -> dict:
    """Copy stage / in-drawer land anchors and ``stagedRole`` from ``source`` onto ``obj``."""
    out = dict(obj)
    xy = _stage_xy_of(source)
    if xy is not None:
        out["stageXy"] = xy
    land = _drawer_land_xy_of(source)
    if land is not None:
        out["drawerLandXy"] = land
    if source.get("drawerId"):
        out["drawerId"] = source["drawerId"]
    role = source.get("stagedRole") or (
        source.get("role") or source.get("category") if land is not None else None)
    if role:
        out["stagedRole"] = role
    if "stageConfirmed" in source:
        out["stageConfirmed"] = bool(source["stageConfirmed"])
    return out


def _expected_drawer_role(obj: dict) -> str | None:
    """Role used when rematching a drawer whitelist piece."""
    role = obj.get("stagedRole") or obj.get("role") or obj.get("category")
    return role if role in DRAWER_SWAP_ROLES else None


def _role_matches_expected(obj: dict, expected_role: str | None) -> bool:
    """Whether ``obj``'s live role/category equals a planned whitelist ``expected_role``."""
    if not expected_role:
        return _drawer_deployed(obj)
    return expected_role in (obj.get("role"), obj.get("category"))


def _held_drawer_match(api: str, expected: dict) -> dict | None:
    """If the gripper already holds a whitelist piece matching ``expected``, return that track."""
    want = _expected_drawer_role(expected)
    if want is None:
        return None
    s = _summary(api)
    held_id = s.get("self", {}).get("holding")
    if not held_id:
        return None
    held = _summary_object(s, held_id)
    if held is None or not _drawer_deployed(held):
        return None
    role = held.get("role") or held.get("category")
    if want not in (role, held.get("category")):
        return None
    return _with_stage_anchor(held, expected)


def _protect_stage(api: str, obj: dict, log: list) -> None:
    """Tell the estimator a staged whitelist piece at ``stageXy`` is real, not a furniture smear.

    Uses an XY anchor (not a long-lived pin) so later freshen views can still update the track.
    """
    xy = _stage_xy_of(obj)
    if xy is None:
        return
    role = obj.get("stagedRole") or obj.get("role") or obj.get("category")
    body = {"xy": xy, "role": role, "radius": STAGE_RECOVER_RADIUS_M}
    try:
        result = call("POST", f"{api}/ego/stage-protect", body)
        log.append({"step": "stage-protect", "xy": xy, "role": role,
                    "anchors": len(result.get("anchors") or [])})
    except Exception as exc:  # noqa: BLE001 - staging must continue even if the ego route is old
        log.append({"step": "stage-protect-failed", "xy": xy, "role": role, "error": str(exc)})


def _clear_stage_protect(api: str, obj: dict | None, log: list) -> None:
    """Drop the stage anchor for ``obj`` (or every anchor when ``obj`` is None)."""
    xy = _stage_xy_of(obj) if obj is not None else None
    body: dict = {"clear": True}
    if xy is not None:
        body["xy"] = xy
    try:
        result = call("POST", f"{api}/ego/stage-protect", body)
        log.append({"step": "stage-protect-clear", "xy": xy, "cleared": result.get("cleared")})
    except Exception as exc:  # noqa: BLE001 - clearing is best-effort
        log.append({"step": "stage-protect-clear-failed", "xy": xy, "error": str(exc)})


def _freshen_drawer_pick(api: str, obj: dict, log: list, *,
                         prefer_offset: float | None = None,
                         distance: float | None = None) -> dict:
    """Near look + settle so a staged whitelist pick uses a freshly observed pose.

    Rematch only restores identity; place-unverified tracks often keep a coarse centre until a
    closer view updates them. Keeps ``stageXy`` / ``stagedRole`` on the returned track.
    Soft existence after a farther look triggers a second approach at
    :data:`STAGE_PICK_CLOSE_DISTANCE` before returning — even when the caller already passed an
    explicit far ``distance`` (staged ``_move`` always passes one).

    A refreshed id is accepted only when its live role still matches the planned whitelist role —
    otherwise a renumbered / reclassified track (e.g. furniture smear) must rematch near ``stageXy``.
    """
    stage_xy = _stage_xy_of(obj)
    expected_role = _expected_drawer_role(obj)
    look = obj
    if stage_xy is not None:
        z = float((obj.get("position") or [0.0, 0.0, 0.85])[2])
        look = {**obj, "position": [stage_xy[0], stage_xy[1], z]}
    dist = STAGE_PICK_DISTANCE if distance is None else float(distance)

    def _once(approach_distance: float) -> dict:
        try:
            _approach_to_observe(api, look, distance=approach_distance, prefer_offset=prefer_offset)
        except MissionFailed:
            pass
        simtime_sleep(api, RESCAN_SETTLE_S)
        summary = _summary(api)
        try:
            fresh = _refresh(api, obj)
            if (_drawer_deployed(fresh)
                    and _role_matches_expected(fresh, expected_role)):
                out = _with_stage_anchor(fresh, obj)
                log.append({"step": "drawer-freshen", "id": out["id"], "offset": prefer_offset,
                            "distance": approach_distance,
                            "existence": out.get("existence"),
                            "role": out.get("role") or out.get("category")})
                return out
            if not _role_matches_expected(fresh, expected_role):
                log.append({"step": "drawer-freshen", "reason": "role-mismatch",
                            "id": fresh.get("id"), "from": obj.get("id"),
                            "role": fresh.get("role") or fresh.get("category"),
                            "expectedRole": expected_role, "offset": prefer_offset,
                            "distance": approach_distance})
        except MissionFailed:
            pass
        xy = stage_xy or (list(obj["position"][:2]) if obj.get("position") else None)
        found = _match_staged_near(summary, obj, xy)
        if found is None:
            found = _match_by_footprint(summary, obj, require_deployed=True)
        if found is None:
            raise MissionFailed(
                f"lost staged drawer item {obj.get('id')} near {xy} after freshen look")
        out = _with_stage_anchor(found, obj)
        log.append({"step": "drawer-freshen", "id": out["id"], "from": obj.get("id"),
                    "offset": prefer_offset, "distance": approach_distance,
                    "existence": out.get("existence"),
                    "role": out.get("role") or out.get("category")})
        return out

    out = _once(dist)
    # Step in whenever existence is soft and this look was farther than the close distance —
    # callers that pass an explicit far ``distance`` still need the auto-close path.
    if (float(out.get("existence") or 0.0) < STAGE_PICK_MIN_EXISTENCE
            and dist > STAGE_PICK_CLOSE_DISTANCE + 1e-6):
        out = _once(STAGE_PICK_CLOSE_DISTANCE)
    return out


def _recover_drawer_carry(api: str, item: dict, log: list) -> dict:
    """Refresh a drawer whitelist item; rematch near stage / in-drawer land anchors.

    Staged pieces and drawer landings often lose their track id after place. Global footprint
    matching then binds tableware; this path only accepts :data:`DRAWER_SWAP_ROLES` and prefers
    ``stageXy``, then ``drawerLandXy`` / ``drawerId``, then the last pose. When ``stageXy`` is set,
    a closer freshen look follows so pick does not reuse the coarse post-release pose.

    A successful rematch is not aborted by a single freshen miss: retry from another heading, then
    keep the rematched track so :func:`_move` can freshen again (staged ids often churn under a
    nearer look right after rematch).
    """
    stage_xy = _stage_xy_of(item)
    land_xy = _drawer_land_xy_of(item)
    anchor_xy = _anchor_xy_of(item)
    drawer_id = item.get("drawerId")
    expected_role = item.get("stagedRole") or item.get("role") or item.get("category")
    want_role = expected_role if expected_role in DRAWER_SWAP_ROLES else None
    found: dict | None = None
    try:
        obj = _refresh(api, item)
        if _drawer_deployed(obj) and _role_matches_expected(obj, want_role):
            found = _with_stage_anchor(obj, item)
        else:
            log.append({"step": "drawer-rematch",
                        "reason": ("role-mismatch" if _drawer_deployed(obj)
                                   else "refreshed-non-deployed"),
                        "from": item.get("id"), "to": obj.get("id"),
                        "role": obj.get("role") or obj.get("category"),
                        "expectedRole": expected_role, "stageXy": stage_xy,
                        "drawerLandXy": land_xy, "drawerId": drawer_id})
    except MissionFailed as exc:
        log.append({"step": "drawer-rematch", "reason": "refresh-missed",
                    "from": item.get("id"), "error": str(exc),
                    "expectedRole": expected_role, "stageXy": stage_xy,
                    "drawerLandXy": land_xy, "drawerId": drawer_id})

    if found is None:
        look = dict(item)
        if anchor_xy is not None:
            z = float((item.get("position") or [0, 0, 0.85])[2]) if item.get("position") else 0.85
            look = {**item, "position": [anchor_xy[0], anchor_xy[1], z]}
        try:
            _approach_to_observe(api, look, distance=DRAWER_CONFIRM_DISTANCE)
        except MissionFailed:
            pass
        simtime_sleep(api, RESCAN_SETTLE_S)
        summary = _summary(api)
        matched = None
        # Stage top first: a stamped source ``drawerId`` from confirm must not pull a staged
        # teddy back into the open box (or onto the drawer track) when rematching for the
        # board→cabinet leg.
        # Footprint neighbourhood must be the stage spot when set — the pre-stage drawer pose is
        # metres away, so a rematch centred there never sees the top (seed2e pistol).
        footprint_seed = item
        if stage_xy is not None:
            z = float((item.get("position") or [0.0, 0.0, 0.85])[2])
            footprint_seed = {**item, "position": [stage_xy[0], stage_xy[1], z]}
        if stage_xy is not None:
            matched = _match_staged_near(summary, item, stage_xy)
        if matched is None and drawer_id:
            drawer = _summary_object(summary, drawer_id)
            if drawer is not None:
                matched = _match_in_drawer(summary, item, drawer, spot=land_xy or anchor_xy)
        if matched is None and stage_xy is None:
            matched = _match_staged_drawer_item(
                summary, item, anchor_xy, radius=DRAWER_LAND_RECOVER_RADIUS_M)
        if matched is None:
            matched = _match_by_footprint(summary, footprint_seed, require_deployed=True)
        if matched is None and stage_xy is not None:
            # One more stage look before failing: post-release tracks often appear only after a
            # second heading at the stage spot (same gate as :func:`_confirm_stage_landing`).
            log.append({"step": "drawer-rematch", "reason": "stage-reconfirm",
                        "from": item.get("id"), "expectedRole": expected_role,
                        "stageXy": stage_xy})
            _confirm_stage_landing(api, item, stage_xy, log, views=STAGE_RECONFIRM_VIEWS)
            summary = _summary(api)
            matched = _match_staged_near(summary, item, stage_xy)
            if matched is None:
                matched = _match_by_footprint(summary, footprint_seed, require_deployed=True)
        if matched is None:
            raise MissionFailed(
                f"lost staged drawer item {item.get('id')} "
                f"(expected {expected_role} near {anchor_xy}): not in {DRAWER_SWAP_ROLES}")
        found = _with_stage_anchor(matched, item)
        log.append({"step": "drawer-rematch", "reason": "matched",
                    "from": item.get("id"), "to": found["id"],
                    "role": found.get("role") or found.get("category"),
                    "expectedRole": expected_role, "stageXy": stage_xy,
                    "drawerLandXy": land_xy, "drawerId": drawer_id})

    if stage_xy is not None:
        # Re-assert the anchor (and pin the rematched id) so freshen views cannot drop it again.
        _protect_stage(api, found, log)
        try:
            return _freshen_drawer_pick(api, found, log, prefer_offset=RESCAN_VIEW_OFFSETS[0])
        except MissionFailed as freshen_exc:
            retry_offset = RESCAN_VIEW_OFFSETS[min(1, len(RESCAN_VIEW_OFFSETS) - 1)]
            log.append({"step": "drawer-freshen", "reason": "retry-heading",
                        "id": found.get("id"), "error": str(freshen_exc),
                        "offset": retry_offset, "distance": STAGE_PICK_CLOSE_DISTANCE})
            try:
                return _freshen_drawer_pick(
                    api, found, log, prefer_offset=retry_offset,
                    distance=STAGE_PICK_CLOSE_DISTANCE)
            except MissionFailed as freshen_exc2:
                # Keep the rematched whitelist track; _move will freshen from new headings.
                log.append({"step": "drawer-freshen", "reason": "soft-keep",
                            "id": found.get("id"), "error": str(freshen_exc2),
                            "stageXy": stage_xy})
                return found
    return found


def _confirm_drawer_landing(api: str, obj: dict, drawer: dict, spot: list[float],
                            log: list) -> dict:
    """Look into ``drawer`` after a whitelist place and rebind the released track.

    Place often reports unverified release while the segmenter renumbers the piece. Recording
    ``drawerLandXy`` / ``drawerId`` lets a later :func:`_recover_drawer_carry` find it inside the
    open box instead of failing on the stale id.
    """
    land = [round(float(spot[0]), 3), round(float(spot[1]), 3)]
    role = obj.get("stagedRole") or obj.get("role") or obj.get("category")
    anchored = {**obj, "drawerLandXy": land, "drawerId": drawer.get("id"),
                "stagedRole": role, "position": [land[0], land[1],
                                                float((obj.get("position") or [0, 0, 0.55])[2])]}
    try:
        _approach_to_observe(api, drawer, distance=DRAWER_PLACE_ALIGN_DISTANCE)
    except MissionFailed:
        pass
    simtime_sleep(api, RESCAN_SETTLE_S)
    summary = _summary(api)
    live_drawer = _summary_object(summary, drawer.get("id")) or drawer
    found = _match_in_drawer(summary, anchored, live_drawer, spot=land)
    if found is None:
        found = _match_staged_drawer_item(summary, anchored, land,
                                          radius=DRAWER_LAND_RECOVER_RADIUS_M)
    if found is None:
        try:
            found = _refresh(api, anchored)
            if not _drawer_deployed(found):
                found = None
        except MissionFailed:
            found = None
    if found is None:
        log.append({"step": "drawer-land-unconfirmed", "from": obj.get("id"),
                    "drawer": drawer.get("id"), "xy": land, "role": role})
        return anchored
    out = _with_stage_anchor(found, anchored)
    out["drawerLandXy"] = land
    out["drawerId"] = live_drawer.get("id") or drawer.get("id")
    log.append({"step": "drawer-land-confirm", "from": obj.get("id"), "to": out["id"],
                "drawer": out["drawerId"], "xy": land,
                "role": out.get("role") or out.get("category")})
    return out


def _confirm_stage_landing(api: str, obj: dict, stage_xy: list[float], log: list,
                           *, views: int = 2) -> dict:
    """Look at a furniture-top stage spot after place and rebind the released whitelist track.

    Staging often reports unverified release (``release track unavailable``). Without an immediate
    look, a later board→cabinet rematch finds an empty ``stageXy``. Same role gate as
    :func:`_match_staged_drawer_item`; retries from distinct headings like drawer confirm.
    """
    xy = [round(float(stage_xy[0]), 3), round(float(stage_xy[1]), 3)]
    role = obj.get("stagedRole") or obj.get("role") or obj.get("category")
    z = float((obj.get("position") or [0.0, 0.0, 0.85])[2])
    anchored = {**obj, "stageXy": xy, "stagedRole": role, "position": [xy[0], xy[1], z]}
    anchored.pop("drawerId", None)
    anchored.pop("drawerLandXy", None)
    found: dict | None = None
    tried = 0
    view_budget = max(1, views)
    for offset in RESCAN_VIEW_OFFSETS[:view_budget]:
        tried += 1
        look = {**anchored, "position": [xy[0], xy[1], z]}
        # First headings at the normal pick distance; later ones step in if still unbound.
        distance = STAGE_PICK_DISTANCE if tried <= STAGE_CONFIRM_VIEWS else STAGE_PICK_CLOSE_DISTANCE
        try:
            _approach_to_observe(api, look, distance=distance, prefer_offset=offset)
        except MissionFailed:
            pass
        simtime_sleep(api, RESCAN_SETTLE_S)
        summary = _summary(api)
        found = _match_staged_near(summary, anchored, xy)
        if found is None:
            try:
                fresh = _refresh(api, anchored)
                if _drawer_deployed(fresh):
                    found = fresh
                else:
                    found = None
            except MissionFailed:
                found = None
        if found is not None:
            break
    if found is None:
        anchored["stageConfirmed"] = False
        log.append({"step": "stage-land-unconfirmed", "from": obj.get("id"),
                    "xy": xy, "role": role, "views": tried})
        return anchored
    out = _with_stage_anchor(found, anchored)
    out["stageXy"] = xy
    out["stagedRole"] = role
    out["stageConfirmed"] = True
    out.pop("drawerId", None)
    out.pop("drawerLandXy", None)
    log.append({"step": "stage-land-confirm", "from": obj.get("id"), "to": out["id"],
                "xy": xy, "role": out.get("role") or out.get("category"), "views": tried})
    return out


def _live_top_support(api: str, support: dict, log: list | None = None) -> dict:
    """Current furniture/table track for a top place: same id if still a top support, else rematch.

    A stage-protect radius can cover a sideboard centre; a mis-applied rename then turns the host
    into the staged piece and place fails with "neither a surface nor a receptacle". Prefer a live
    :data:`TOP_SUPPORT_ROLES` track near the remembered pose (same role when known).

    When the remembered host is a drawer stage (:data:`STAGE_HOST_ROLES`), rematch only among
    sideboard / cabinet / table — never a kitchen ``counter`` that happens to be nearer in XY
    after the sideboard track flips to ``wall``.
    """
    want_role = support.get("role") or support.get("category")
    pos = support.get("position")
    try:
        live = _refresh(api, support)
        role = live.get("role") or live.get("category")
        if role in TOP_SUPPORT_ROLES:
            return live
        if log is not None:
            log.append({"step": "top-support-rematch", "reason": "role-lost",
                        "from": support.get("id"), "was": role, "want": want_role})
        if pos is None:
            pos = live.get("position")
    except MissionFailed as exc:
        if log is not None:
            log.append({"step": "top-support-rematch", "reason": "refresh-missed",
                        "from": support.get("id"), "error": str(exc)})
    summary = _summary(api)
    if want_role in STAGE_HOST_ROLES:
        role_order = [want_role, *(r for r in STAGE_HOST_REMATCH_ROLES if r != want_role)]
    elif want_role in TOP_SUPPORT_ROLES:
        role_order = [want_role, *(r for r in TOP_SUPPORT_ROLES if r != want_role)]
    else:
        role_order = list(TOP_SUPPORT_ROLES)
    cands: list[dict] = []
    seen: set[str] = set()
    for role in role_order:
        for o in _objects(summary, role):
            oid = o["id"]
            if oid in seen:
                continue
            seen.add(oid)
            cands.append(o)
    if not cands:
        raise MissionFailed(
            f"{support.get('id')} is no longer a top support and no furniture/table was found")
    if pos is not None:
        found = min(cands, key=lambda o: math.dist(o["position"][:2], pos[:2]))
    else:
        found = max(cands, key=_area)
    if log is not None:
        log.append({"step": "top-support-rematch", "reason": "matched",
                    "from": support.get("id"), "to": found["id"],
                    "role": found.get("role") or found.get("category")})
    return found


def _place_at(api: str, support: dict, xy: list[float], log: list | None = None) -> dict:
    """Place params for a fixed world spot on a support, in the support's current frame (a fitted box
    may turn half way round between views; the spot must not move with it)."""
    current = _live_top_support(api, support, log)
    return {"on": current["id"], "offset": [round(v, 4) for v in _local(current, xy)]}


def _local(support: dict, xy: list[float]) -> tuple[float, float]:
    yaw = math.radians(support["yawDeg"])
    dx, dy = xy[0] - support["position"][0], xy[1] - support["position"][1]
    c, s = math.cos(-yaw), math.sin(-yaw)
    return c * dx - s * dy, s * dx + c * dy


def _stage_top_xy(board: dict, drawer: dict | None, index: int, count: int) -> list[float]:
    """World XY on ``board``'s top for a temporary drawer item, clear of the drawer approach.

    Pieces spread along the long axis toward the ends, but stay inward of
    :data:`STAGE_EDGE_CLEAR_M` so place risk stays under the rim gate. When ``drawer`` is known,
    they sit slightly toward the back of the top (opposite the drawer) so a later dock at the
    open drawer is less likely to stall against the staged pile.
    """
    extent = board.get("extentM") or board["sizeM"]
    long_a, short_a = float(extent[0]), float(extent[1])
    if short_a > long_a:
        long_a, short_a = short_a, long_a
    half_long = long_a / 2
    half_short = short_a / 2
    along_span = max(0.0, half_long - STAGE_EDGE_CLEAR_M) * STAGE_ALONG_FRAC
    if count <= 1:
        along = 0.0
    else:
        along = (index / (count - 1) * 2 - 1) * along_span
    across = 0.0
    if drawer is not None and drawer.get("position") is not None:
        lx, ly = _local(board, drawer["position"][:2])
        # Short-axis local of the drawer: stage on the opposite side of the top.
        short = ly if float(extent[0]) >= float(extent[1]) else lx
        back = min(STAGE_BACK_M, max(0.0, half_short - STAGE_EDGE_CLEAR_M))
        across = -math.copysign(back, short if short != 0.0 else 1.0)
    return _world(board, _offset(board, along, across))


def _is_base_blocked(exc: BaseException) -> bool:
    """Whether a skill failure is a stuck base (dock / navigate stall), worth a short retreat."""
    msg = str(exc)
    return any(token in msg for token in PLACE_BASE_BLOCKED)


def _is_drawer_support(obj: dict) -> bool:
    """Whether ``obj`` is the open drawer used as a place support."""
    return (obj.get("role") or obj.get("category")) == "drawer"


def _is_place_dock_failure(exc: BaseException) -> bool:
    """Whether place failed because ESim found no legal dock at the support."""
    msg = str(exc).lower()
    return any(token in msg for token in PLACE_DOCK_INFEASIBLE)


def _is_drawer_too_shut(exc: BaseException) -> bool:
    """Whether place failed because ESim read the drawer opening as too small (often a false 0 cm)."""
    msg = str(exc).lower()
    return any(token in msg for token in PLACE_DRAWER_TOO_SHUT)


def _is_drawer_full(exc: BaseException) -> bool:
    """Whether place failed because the open drawer part reports no free spot."""
    msg = str(exc).lower()
    return any(token in msg for token in PLACE_DRAWER_FULL)


def _recover_drawer_opening(api: str, drawer: dict, log: list, timeout_s: float) -> dict:
    """Refresh the drawer (and its host when known) after a false shut opening; re-open if empty-handed.

    Place runs while holding, so ``open`` is skipped then — ESim's pulled-open floor should already
    clear a remembered pull. When the gripper is empty (or the hold was dropped), a real re-open
    recovers a renumbered drawer that lost that memory.
    """
    drawer = _refresh_or_recover(api, drawer)
    host_id = drawer.get("on")
    if host_id:
        host = _summary_object(_summary(api), host_id)
        if host is not None:
            try:
                _approach_to_observe(api, host, distance=0.95)
            except MissionFailed:
                pass
            simtime_sleep(api, 1.0)
            drawer = _refresh_or_recover(api, drawer)
    held = _summary(api)["self"].get("holding")
    if held:
        log.append({"step": "drawer-opening-recover", "drawer": drawer.get("id"),
                    "openingM": drawer.get("openingM"), "held": held, "reopen": False})
        return drawer
    if drawer.get("openingM", 0.0) >= 0.2:
        log.append({"step": "drawer-opening-recover", "drawer": drawer.get("id"),
                    "openingM": drawer.get("openingM"), "reopen": False})
        return drawer
    try:
        receipt = call("POST", f"{api}/ego/predict",
                       {"skill": "open", "params": {"target": drawer["id"]}})
        if not receipt.get("withinRiskLimit"):
            log.append({"step": "drawer-reopen-skipped", "drawer": drawer.get("id"),
                        "issues": list(receipt.get("issues") or [])})
            return drawer
        _run(api, receipt, timeout_s)
        simtime_sleep(api, 1.5)
        drawer = _refresh_or_recover(api, drawer)
        log.append({"step": "drawer-reopen", "drawer": drawer.get("id"),
                    "openingM": drawer.get("openingM")})
    except MissionFailed as exc:
        log.append({"step": "drawer-reopen-failed", "drawer": drawer.get("id"), "error": str(exc)})
    return drawer


def _align_drawer_place(api: str, drawer: dict, log: list, *,
                        prefer_offset: float | None = None) -> None:
    """Stand in front of the open drawer before predicting place into it.

    After picking from the opposite cabinet the base often still faces that side of the room;
    predicting place from there yields ``no feasible approach`` against the furniture body.
    """
    try:
        _approach_to_observe(api, drawer, distance=DRAWER_PLACE_ALIGN_DISTANCE,
                             prefer_offset=prefer_offset)
        log.append({"step": "drawer-place-align", "drawer": drawer.get("id"),
                    "offset": prefer_offset})
    except MissionFailed as exc:
        log.append({"step": "drawer-place-align-failed", "drawer": drawer.get("id"),
                    "offset": prefer_offset, "error": str(exc)})
    simtime_sleep(api, 1.0)


def _retreat_while_holding(api: str, log: list, timeout_s: float) -> bool:
    """Back away a short distance without turning, while possibly holding an object.

    Used after a place dock stalls so the next place prediction can pick another approach. Returns
    whether the base moved.
    """
    me = _summary(api)["self"]
    heading = math.radians(me["yawDeg"])
    x = me["x"] - UNSTICK_M * math.cos(heading)
    y = me["y"] - UNSTICK_M * math.sin(heading)
    receipt = call("POST", f"{api}/ego/predict",
                   {"skill": "navigate", "params": {"x": x, "y": y, "yaw": heading}})
    log.append({"step": "retreat-receipt", "params": {"x": x, "y": y}, **receipt_brief(receipt)})
    if not receipt["withinRiskLimit"]:
        return False
    try:
        _run(api, receipt, timeout_s)
    except MissionFailed as exc:
        log.append({"step": "retreat-failed", "error": str(exc)})
        return False
    return True


def _landed(api: str, obj: dict, spot: list[float]) -> dict:
    """``obj`` as the ego state knows it after it was put down at ``spot`` (world x, y).

    With a real segmenter the track of a carried object is often rebuilt under a new id when the
    gripper lets go. Then the piece of its role nearest the spot, within 0.25 m of it and within a
    quarter of the object's width, is taken for it: the bowl just nested stands on the same spot
    as the wider bowl under it. Drawer whitelist pieces also accept any deployed role near the spot
    with a similar footprint (a pistol must not be rebound to a mug on the same top). When no such
    piece is seen there yet, ``obj`` is returned at ``spot``, for a later refresh to find it by.
    """
    objects = _summary(api)["objects"]
    found = next((o for o in objects if o["id"] == obj["id"]), None)
    if found is not None:
        return found
    role, width = obj.get("role", obj.get("category")), _width(obj)
    there = [o for o in objects if o["existence"] >= 0.5 and role in (o.get("role"), o["category"])
             and math.dist(o["position"][:2], spot[:2]) <= 0.25 and abs(_width(o) - width) <= 0.25 * width]
    if there:
        return min(there, key=lambda o: math.dist(o["position"][:2], spot[:2]))
    if _drawer_deployed(obj):
        staged = _match_staged_drawer_item({"objects": objects}, obj, list(spot[:2]),
                                           radius=STAGE_RECOVER_RADIUS_M)
        if staged is not None:
            return staged
    return {**obj, "position": [spot[0], spot[1], obj["position"][2]]}


def _drawer_host_furniture(api: str, drawer: dict) -> dict | None:
    """Furniture top that holds ``drawer`` (``on`` link, else nearest stage-host by XY).

    Rematched open-drawer tracks often lose ``on`` after a long swap; parking a held piece still
    needs a sideboard/cabinet top near the drawer mouth.
    """
    s = _summary(api)
    host_id = drawer.get("on")
    if host_id:
        host = _summary_object(s, host_id)
        if host is not None:
            return host
    pos = drawer.get("position")
    if pos is None:
        return None
    hosts = [
        o for o in s["objects"]
        if (o.get("role") or o.get("category")) in STAGE_HOST_ROLES
        and o.get("existence", 0.0) >= 0.5
        and o.get("position") is not None
    ]
    if not hosts:
        return None
    return min(hosts, key=lambda o: math.dist(o["position"][:2], pos[:2]))


def _park_held_clear_drawer_occupant(
        api: str, held: dict, drawer: dict, evacuate_to: dict, log: list,
        timeout_s: float) -> tuple[dict, dict]:
    """Park the held track on the drawer host top, move one occupant to ``evacuate_to``, re-grasp.

    Used once after a drawer-full place failure while the jaws still hold the inbound piece.
    Occupant choice prefers deployed swap roles, then largest footprint.

    @param held: track currently in the gripper
    @param drawer: open drawer that reported no free spot
    @param evacuate_to: support that should receive the cleared occupant (usually the other drawer)
    @returns: ``(held after re-grasp, refreshed drawer)``; on park/clear failure returns best-effort
    """
    drawer = _refresh_or_recover(api, drawer)
    host = _drawer_host_furniture(api, drawer)
    if host is None:
        log.append({"step": "drawer-full-clear-skip", "reason": "no-host",
                    "drawer": drawer.get("id")})
        return held, drawer
    s = _summary(api)
    occupants = [
        o for o in _drawer_on_holder(s, drawer)
        if o.get("id") != held.get("id")
        and not _drawer_meal_skip(o)
        and _in_open_drawer_volume(o, drawer)
        # Prefer clearing deployed leftovers; skip soft existence ghosts that mirror the held role.
        and (o.get("role") or o.get("category")) != (
            held.get("stagedRole") or held.get("role") or held.get("category"))
    ]
    if not occupants:
        log.append({"step": "drawer-full-clear-skip", "reason": "no-occupant",
                    "drawer": drawer.get("id")})
        return held, drawer
    occupants = sorted(
        occupants, key=lambda o: (0 if _drawer_deployed(o) else 1, -_area(o)))
    occupant = occupants[0]
    park_xy = _stage_top_xy(host, drawer, 0, 1)
    try:
        park_place = _place_at(api, host, park_xy, log)
        receipt = _safe_receipt(api, "place", park_place, host, log)
        _run(api, receipt, timeout_s)
        spot = receipt.get("plan", {}).get("spot") or park_xy
        held = _landed(api, held, spot)
        held = {
            **held,
            "stageXy": [round(park_xy[0], 3), round(park_xy[1], 3)],
            "stagedRole": held.get("stagedRole") or held.get("role") or held.get("category"),
        }
        log.append({"step": "drawer-full-park", "object": held.get("id"), "on": host["id"],
                    "xy": held["stageXy"], "cleared": occupant.get("id")})
    except MissionFailed as exc:
        log.append({"step": "drawer-full-park-failed", "object": held.get("id"),
                    "error": str(exc)})
        return held, drawer
    try:
        dest = _refresh_or_recover(api, evacuate_to)
        _move(api, occupant, {"on": dest["id"]}, dest, log, timeout_s)
        log.append({"step": "drawer-full-clear", "cleared": occupant.get("id"),
                    "to": dest.get("id"), "drawer": drawer.get("id")})
    except MissionFailed as exc:
        log.append({"step": "drawer-full-clear-failed", "cleared": occupant.get("id"),
                    "error": str(exc)})
    try:
        held = _recover_drawer_carry(api, held, log)
    except MissionFailed as exc:
        # Fall back to a direct pick of the parked track.
        try:
            held = _refresh_or_recover(api, held, require_deployed=False)
            _approach_to_observe(api, held, distance=VIEW_DISTANCE)
            receipt = _safe_receipt(api, "pick", {"target": held["id"]}, held, log)
            _run(api, receipt, timeout_s)
            log.append({"step": "drawer-full-regrasp", "object": held.get("id")})
        except MissionFailed as pick_exc:
            log.append({"step": "drawer-full-regrasp-failed", "object": held.get("id"),
                        "error": str(pick_exc), "recover": str(exc)})
            raise MissionFailed(
                f"drawer-full clear parked {held.get('id')} but re-grasp failed: {pick_exc}"
            ) from pick_exc
    drawer = _refresh_or_recover(api, drawer)
    return held, drawer


def _move(api: str, obj: dict, place: dict, observe: dict, log: list, timeout_s: float,
          attempts: int | None = None, drawer_evacuate_to: dict | None = None) -> dict:
    """Pick ``obj`` (observing it again and retrying after a missed grasp) and place it with
    ``place`` params; return the object as it stands after the place (:func:`_landed`).

    Staged drawer pieces (``stageXy`` set) freshen from a distinct heading each attempt so a miss
    or a base-blocked dock does not reuse the same approach. Whitelist picks that lose their id
    rematch via :func:`_recover_drawer_carry` (and adopt a matching held track when the jaws
    already closed).     Soft pose-uncertainty receipts share one look-only deferral; soft existence may defer up to
    :data:`PICK_EXISTENCE_DEFER_MAX` times and rematches near ``stageXy`` between looks so a
    weak ghost is not grasped. Grasp misses on staged pieces rematch near ``stageXy`` before
    the next heading. Drawer places look into the open box to confirm the landed track.
    When ``drawer_evacuate_to`` is set, the first drawer-full failure parks the held piece,
    moves one occupant onto that support, re-grasps, and retries the place.
    """
    staged = _stage_xy_of(obj) is not None
    drawer_carry = bool(_expected_drawer_role(obj) or _drawer_deployed(obj)
                        or obj.get("stagedRole") in DRAWER_SWAP_ROLES
                        or _drawer_land_xy_of(obj) is not None)
    if attempts is None:
        attempts = PICK_BASE_BLOCKED_TRIES if (staged or drawer_carry) else 2
    observe_defers = 0
    existence_defers = 0
    drawer_full_cleared = False
    for attempt in range(attempts):
        # Look before moving: from here the class-prompted fresh look at pick time sees the
        # object whole (below ~0.4 m every prompt fails), and the receipt is predicted from the
        # pose the pick really starts at. A failed approach is not a failed pick — the robot
        # just looks from where it stands.
        if staged:
            offset = RESCAN_VIEW_OFFSETS[min(attempt, len(RESCAN_VIEW_OFFSETS) - 1)]
            # Far → close after soft receipts → even closer after a grasp miss heading.
            if attempt == 0:
                freshen_distance = STAGE_PICK_DISTANCE
            elif attempt == 1:
                freshen_distance = STAGE_PICK_CLOSE_DISTANCE
            else:
                freshen_distance = STAGE_PICK_MISS_DISTANCE
            obj = _freshen_drawer_pick(api, obj, log, prefer_offset=offset, distance=freshen_distance)
            want = _expected_drawer_role(obj)
            if want is not None and not _role_matches_expected(obj, want):
                log.append({"step": "pick-identity", "reason": "role-mismatch",
                            "object": obj.get("id"),
                            "role": obj.get("role") or obj.get("category"),
                            "expectedRole": want, "attempt": attempt})
                obj = _recover_drawer_carry(api, obj, log)
        else:
            try:
                # After a soft-observe defer, step in so the next receipt is not the same far pose.
                look_distance = (0.9 if attempt == 0 else STAGE_PICK_CLOSE_DISTANCE)
                _approach_to_observe(api, obj, distance=look_distance)
            except MissionFailed:
                pass
            # The approach takes seconds, and with a real segmenter a track can be renumbered while
            # the robot drives: the id planned from is not the id that exists now. Adopt whatever of
            # the same role stands there before asking for a pick of it.
            if drawer_carry:
                # Prefer stage / drawer-land / source-drawer rematch over bare footprint recovery,
                # which can adopt a same-role sibling still sitting in the open box.
                obj = _recover_drawer_carry(api, obj, log)
            else:
                obj = _refresh(api, obj)
        want = _expected_drawer_role(obj) if (staged or drawer_carry) else None
        if want is not None and not _role_matches_expected(obj, want):
            raise MissionFailed(
                f"pick failed: target {obj.get('id')} is "
                f"{obj.get('role') or obj.get('category')}, expected {want}")
        pick = _safe_receipt(api, "pick", {"target": obj["id"]}, obj, log)
        log.append({"step": "pick-receipt", "object": obj["id"], "attempt": attempt, **receipt_brief(pick)})
        issues_text = " ".join(str(x) for x in (pick.get("issues") or []))
        soft_existence = any(token in issues_text for token in PICK_EXISTENCE_ISSUES)
        soft_observe = any(token in issues_text for token in PICK_OBSERVE_ISSUES)
        # Soft existence: rematch + closer look (up to PICK_EXISTENCE_DEFER_MAX). Pose-only
        # uncertainty: a single look-only deferral. Applies to staged tops and in-drawer carries
        # (seed2b second perfume: 4 cm / 46% miss with no defer because it was not staged).
        # Neither may consume the final attempt.
        whitelist_pick = staged or drawer_carry
        if (whitelist_pick and attempt + 1 < attempts and soft_existence
                and existence_defers < PICK_EXISTENCE_DEFER_MAX):
            existence_defers += 1
            before_id = obj.get("id")
            try:
                obj = _recover_drawer_carry(api, obj, log)
            except MissionFailed as rematch_exc:
                log.append({"step": "pick-retry", "object": before_id, "attempt": attempt,
                            "reason": "existence", "issues": list(pick.get("issues") or []),
                            "existenceDefers": existence_defers,
                            "rematchError": str(rematch_exc)})
                continue
            log.append({"step": "pick-retry", "object": before_id, "attempt": attempt,
                        "reason": "existence", "issues": list(pick.get("issues") or []),
                        "existenceDefers": existence_defers,
                        "rematched": obj.get("id")})
            continue
        if (whitelist_pick and attempt + 1 < attempts and observe_defers < PICK_OBSERVE_DEFER_MAX
                and soft_observe and not soft_existence):
            observe_defers += 1
            log.append({"step": "pick-retry", "object": obj["id"], "attempt": attempt,
                        "reason": "observe", "issues": list(pick.get("issues") or []),
                        "observeDefers": observe_defers})
            continue
        # In-drawer soft ghosts: after rematch budget is spent, skip instead of a doomed grasp that
        # aborts the whole swap (seed2h perfume e:247 stayed at existence 0.50 through 2 rematches).
        if (drawer_carry and not staged and soft_existence
                and existence_defers >= PICK_EXISTENCE_DEFER_MAX
                and float(obj.get("existence") or 0.0) < 0.55):
            role = _expected_drawer_role(obj) or obj.get("role") or obj.get("category")
            log.append({"step": "pick-retry", "object": obj.get("id"), "attempt": attempt,
                        "reason": "soft-skip", "issues": list(pick.get("issues") or []),
                        "existence": obj.get("existence"),
                        "existenceDefers": existence_defers})
            raise MissionFailed(
                f"lost staged drawer item {obj.get('id')} "
                f"(expected {role}): soft existence {obj.get('existence')} "
                f"after {existence_defers} rematches")
        try:
            job = _run(api, pick, timeout_s)
            log.append({"step": "pick", "object": obj["id"], "jobState": job.get("state"),
                        "message": job.get("message")})
            if drawer_carry:
                held = _held_drawer_match(api, obj)
                if held is not None:
                    obj = held
                else:
                    # Grasp closed on furniture / drawer track — treat as a miss so place does not
                    # demand a drawer-sized opening for a whitelist piece.
                    msg = str(job.get("message") or "")
                    if msg.startswith("holding ") and obj["id"] not in msg:
                        s = _summary(api)
                        held_id = s.get("self", {}).get("holding")
                        held_obj = _summary_object(s, held_id) if held_id else None
                        if held_obj is not None and not _drawer_deployed(held_obj):
                            raise MissionFailed(
                                f"pick failed: holding {held_id} "
                                f"({held_obj.get('role') or held_obj.get('category')}) "
                                f"instead of {obj.get('stagedRole') or obj.get('role') or obj.get('category')}")
            break
        except MissionFailed as exc:
            # Missed grasp / vanished identity: second look. Base stall on the pick dock (common
            # after a cross-room drawer place with both drawers open): short retreat, then a new
            # freshen heading for staged pieces so ESim can choose another approach. Grasp misses
            # on the top also retreat so the next heading is not from the same dock pose.
            retry_grasp = any(w in str(exc) for w in PICK_GRASP_RETRY) or (
                drawer_carry and "instead of" in str(exc))
            retry_base = _is_base_blocked(exc)
            if attempt + 1 == attempts or not (retry_grasp or retry_base):
                raise
            log.append({"step": "pick-retry", "object": obj["id"], "error": str(exc),
                        "attempt": attempt,
                        "reason": "base-blocked" if retry_base else "grasp"})
            if retry_base or (staged and retry_grasp):
                _retreat_while_holding(api, log, timeout_s)
                simtime_sleep(api, 0.5)
            if "instead of" in str(exc):
                # Jaws closed on a non-whitelist track; put it down before the next pick attempt.
                try:
                    s = _summary(api)
                    held_id = s.get("self", {}).get("holding")
                    held_obj = _summary_object(s, held_id) if held_id else None
                    if held_obj is not None:
                        receipt = call(
                            "POST", f"{api}/ego/predict",
                            {"skill": "place", "params": {"at": held_obj["position"][:2]}})
                        if receipt.get("withinRiskLimit"):
                            _run(api, receipt, timeout_s)
                            log.append({"step": "pick-wrong-hold-release", "held": held_id})
                except MissionFailed as release_exc:
                    log.append({"step": "pick-wrong-hold-release-failed",
                                "error": str(release_exc)})
            # Staged grasp miss: reconfirm the stage spot (pose may have drifted), then rematch
            # near stageXy before the next freshen heading so a reclassified id is not re-grasped.
            if staged:
                if retry_grasp:
                    xy = _stage_xy_of(obj)
                    if xy is not None:
                        try:
                            obj = _confirm_stage_landing(
                                api, obj, xy, log, views=STAGE_RECONFIRM_VIEWS)
                            _protect_stage(api, obj, log)
                        except MissionFailed as confirm_exc:
                            log.append({"step": "pick-stage-reconfirm-failed",
                                        "object": obj.get("id"), "error": str(confirm_exc),
                                        "attempt": attempt})
                    try:
                        obj = _recover_drawer_carry(api, obj, log)
                    except MissionFailed as rematch_exc:
                        log.append({"step": "pick-rematch-failed", "object": obj.get("id"),
                                    "error": str(rematch_exc), "attempt": attempt})
                continue
            if drawer_carry and ("no longer in the ego state" in str(exc) or "instead of" in str(exc)):
                held = _held_drawer_match(api, obj)
                if held is not None:
                    obj = held
                    log.append({"step": "pick-held-fallback", "object": obj["id"],
                                "role": obj.get("role") or obj.get("category")})
                    break
                obj = _recover_drawer_carry(api, obj, log)
            else:
                obj = _refresh(api, obj)
    # Pick can take long enough for a nested support bowl to be renumbered; rebind ``on`` before place.
    # A crowded sideboard top or an open drawer often makes the first dock stall with the base
    # blocked; retreat while holding and re-predict once so another approach can be chosen.
    # Placing into a drawer after working at the opposite cabinet also needs an explicit align at
    # the drawer mouth: otherwise ESim reports no feasible approach against the furniture body.
    receipt = None
    drawer_place = _is_drawer_support(observe)
    place_tries = PLACE_DRAWER_TRIES if drawer_place else PLACE_BLOCKED_TRIES
    for place_try in range(place_tries):
        live_place = dict(place)
        if live_place.get("on") is not None:
            obs_role = observe.get("role") or observe.get("category")
            # Top staging: rematch furniture when the host id was renamed to a staged piece.
            if obs_role in TOP_SUPPORT_ROLES or (
                    obs_role in DRAWER_SWAP_ROLES and not _is_drawer_support(observe)):
                observe = _live_top_support(api, observe, log)
            else:
                observe = _refresh_or_recover(api, observe)
            live_place = {**live_place, "on": observe["id"]}
            drawer_place = drawer_place or _is_drawer_support(observe)
        if drawer_place:
            offset = RESCAN_VIEW_OFFSETS[min(place_try, len(RESCAN_VIEW_OFFSETS) - 1)]
            _align_drawer_place(api, observe, log, prefer_offset=offset)
            observe = _refresh_or_recover(api, observe)
            if live_place.get("on") is not None:
                live_place = {**live_place, "on": observe["id"]}
        try:
            receipt = _safe_receipt(api, "place", live_place, observe, log)
            log.append({"step": "place-receipt", "object": obj["id"], "params": live_place,
                        "attempt": place_try, **receipt_brief(receipt)})
            job = _run(api, receipt, timeout_s)
            log.append({"step": "place", "object": obj["id"], **live_place, "jobState": job.get("state"),
                        "message": job.get("message")})
            break
        except MissionFailed as exc:
            retry_dock = drawer_place and _is_place_dock_failure(exc)
            retry_base = _is_base_blocked(exc)
            retry_shut = drawer_place and _is_drawer_too_shut(exc)
            retry_full = drawer_place and _is_drawer_full(exc)
            if place_try + 1 == place_tries or not (
                    retry_dock or retry_base or retry_shut or retry_full):
                raise
            reason = ("drawer-full" if retry_full else
                      "drawer-too-shut" if retry_shut else
                      "dock-infeasible" if retry_dock else "base-blocked")
            log.append({"step": "place-retry", "object": obj["id"], "error": str(exc),
                        "attempt": place_try, "reason": reason})
            if retry_shut:
                observe = _recover_drawer_opening(api, observe, log, timeout_s)
                if live_place.get("on") is not None:
                    live_place = {**live_place, "on": observe["id"]}
            if (retry_full and not drawer_full_cleared and drawer_evacuate_to is not None
                    and _summary(api)["self"].get("holding")):
                _retreat_while_holding(api, log, timeout_s)
                obj, observe = _park_held_clear_drawer_occupant(
                    api, obj, observe, drawer_evacuate_to, log, timeout_s)
                drawer_full_cleared = True
                if live_place.get("on") is not None:
                    live_place = {**live_place, "on": observe["id"]}
                simtime_sleep(api, 0.5)
                continue
            _retreat_while_holding(api, log, timeout_s)
            simtime_sleep(api, 0.5)
    simtime_sleep(api, 0.8)
    plan = receipt["plan"]
    spot = plan["spot"] if "spot" in plan else plan["floor"]
    landed = _landed(api, obj, spot)
    if drawer_place and drawer_carry:
        landed = _confirm_drawer_landing(api, landed, observe, spot, log)
    return landed


def _safe_explore(api: str, timeout_s: float) -> str:
    """One look around for the opening of a round; the state it leaves behind, or the reason
    it could not be taken. The first task approaches and observes its own target anyway, so a
    failed opening look must not keep the round from starting."""
    try:
        return _explore(api, timeout_s)["state"]
    except (MissionFailed, HttpError) as exc:
        return f"failed: {exc}"


def _settle(api: str, seconds: float = 45.0, want: int = 2) -> dict:
    """Wait for the belief to hold a room before planning against it.

    A real segmenter delivers a frame every few seconds, so the ego state right after an
    exploration contains a corner of the room and little else; a round planned on it fails on its
    own emptiness ("no table in the ego state", "one piece of tableware"). Nothing is moved here:
    the robot looks, and the round starts once there is something to plan about."""
    if _assembly_enabled():
        from steward.ego import action_budget as ab
        sensor_clock = ab.Clock()
        deadline = ab.Deadline(sensor_clock, seconds, 'caller observation settling interval') if ab.enabled() else time.monotonic()+seconds
    else:
        deadline = time.monotonic() + seconds
    s: dict = {"objects": [], "relations": []}
    if _assembly_enabled():
        while not ab.expired(deadline):
            try:
                s = _summary(api)
                if ab.enabled() and 't' in s:
                    first = not sensor_clock.initialized
                    sensor_clock.observe(float(s['t']))
                    if first: deadline.start = sensor_clock.stamp
            except (HttpError, OSError):
                time.sleep(2.0)  # the robot is not answering yet; the phases report a dead API themselves
                continue
            if _objects(s, "table") and len(tableware(s)) >= want:
                return s
            time.sleep(2.0)
    else:
        while time.monotonic() < deadline:
            try:
                s = _summary(api)
            except (HttpError, OSError):
                time.sleep(2.0)  # the robot is not answering yet; the phases report a dead API themselves
                continue
            if _objects(s, "table") and len(tableware(s)) >= want:
                return s
            time.sleep(2.0)
    return s


def _view(api: str, furniture: dict, timeout_s: float) -> None:
    """Stand in front of a piece of furniture, facing it, to see its drawer up close."""
    yaw = math.radians(furniture.get("frontDeg", furniture["yawDeg"]))
    n = (math.cos(yaw), math.sin(yaw))
    half = furniture.get("frontHalfM", (furniture["extentM"] or furniture["sizeM"])[0] / 2)
    x = furniture["position"][0] + n[0] * (half + VIEW_DISTANCE)
    y = furniture["position"][1] + n[1] * (half + VIEW_DISTANCE)
    receipt = call("POST", f"{api}/ego/predict",
                   {"skill": "navigate", "params": {"x": x, "y": y, "yaw": math.atan2(-n[1], -n[0])}})
    if not receipt["withinRiskLimit"]:
        raise infeasible(f"navigate to the front of {furniture['id']}", receipt)
    _run(api, receipt, timeout_s)
    simtime_sleep(api, 1.5)


def _in_open_drawer_volume(obj: dict, drawer: dict) -> bool:
    """Whether ``obj`` lies in the measured box of an open ``drawer``.

    When the drawer track has no size or pose yet, the gate cannot run and returns True so a
    whitelist-only filter still applies. Tableware on the furniture top that was wrongly given
    ``on`` = drawer is rejected by height and footprint against that box.
    """
    size = drawer.get("extentM") or drawer.get("sizeM")
    pos = drawer.get("position")
    if not size or not pos or len(size) < 3 or len(pos) < 3:
        return True
    ox, oy, oz = (float(obj["position"][0]), float(obj["position"][1]), float(obj["position"][2]))
    half_x = float(size[0]) / 2 + DRAWER_CONTENT_MARGIN_M
    half_y = float(size[1]) / 2 + DRAWER_CONTENT_MARGIN_M
    if abs(ox - float(pos[0])) > half_x or abs(oy - float(pos[1])) > half_y:
        return False
    obj_size = obj.get("extentM") or obj.get("sizeM") or (0.0, 0.0, 0.0)
    bottom = oz - float(obj_size[2]) / 2
    drawer_bottom = float(pos[2]) - float(size[2]) / 2
    drawer_top = float(pos[2]) + float(size[2]) / 2
    return drawer_bottom - DRAWER_FLOOR_TOL_M <= bottom <= drawer_top + 0.05


def _drawer_deployed(obj: dict) -> bool:
    """Whether the scene-level reading names a deployed drawer item (:data:`DRAWER_SWAP_ROLES`)."""
    role = obj.get("role") or obj.get("category")
    cat = obj.get("category")
    return role in DRAWER_SWAP_ROLES or cat in DRAWER_SWAP_ROLES


def _drawer_meal_skip(obj: dict) -> bool:
    """Whether ``obj`` is meal tableware that this home never keeps in drawers.

    ``cup`` is not included: bottles are often read as cups until a closer view.
    """
    role = obj.get("role") or obj.get("category")
    return role in DRAWER_MEAL_SKIP


def _drawer_on_holder(summary: dict, holder: dict) -> list[dict]:
    """Tracks whose support id is ``holder``, likely enough to exist."""
    return [o for o in summary["objects"] if o.get("on") == holder["id"] and o.get("existence", 0.0) >= 0.5]


def _drawer_content_reject_reasons(obj: dict, drawer: dict) -> list[str]:
    """Hard reasons not to carry ``obj`` (empty: it may be kept or confirmed by a close look).

    Reason codes: ``existence``, ``tableware`` (bowl / wine / mug mis-associated to the drawer),
    ``outside`` (centre/base not in the open box). Non-whitelist look-alikes (``cup``,
    ``car_body``, …) are not hard-rejected here — :meth:`Chores._confirm_drawer_contents` looks
    closer and then skips if they still are not :data:`DRAWER_SWAP_ROLES`.
    """
    reasons: list[str] = []
    if obj.get("existence", 0.0) < 0.5:
        reasons.append("existence")
    if _drawer_meal_skip(obj):
        reasons.append("tableware")
    if not _in_open_drawer_volume(obj, drawer):
        reasons.append("outside")
    return reasons


def _contents(summary: dict, holder: dict) -> list[dict]:
    """Deployed drawer items on ``holder`` that already pass hard gates, largest footprint first.

    A fading fragment is not something to carry. A place takes the free spot nearest the middle
    of the open part: a large object put there leaves room for the small ones on both sides;
    two bottles in the middle leave none for the toy airplane.

    Only :data:`DRAWER_SWAP_ROLES` inside the open drawer box are kept without a further look.
    Uncertain in-drawer tracks are handled by :meth:`Chores._confirm_drawer_contents`.
    """
    inside = _drawer_on_holder(summary, holder)
    kept = [o for o in inside
            if not _drawer_content_reject_reasons(o, holder) and _drawer_deployed(o)]
    return sorted(kept, key=_area, reverse=True)


def _dedupe_drawer_roles(kept: list[dict]) -> tuple[list[dict], list[dict]]:
    """Drop same-role ghost tracks; keep larger footprint first.

    Open-drawer perception can mint a second perfume / pill track overlapping the real one.
    Carrying the ghost stalls the swap on a grasp miss and never reaches close. A ghost must
    match footprint within :data:`DRAWER_ROLE_DUPE_FP_TOL_M` **and** lie within
    :data:`DRAWER_ROLE_DUPE_XY_M` of the kept track — spatially distinct same-role pieces
    (two real bottles) are kept so they leave the drawer and free place spots.

    @param kept: candidates already sorted largest-footprint first
    @returns: ``(unique, dropped)`` where ``dropped`` are the discarded same-role ghosts
    """
    unique: list[dict] = []
    dropped: list[dict] = []
    for obj in kept:
        role = obj.get("stagedRole") or obj.get("role") or obj.get("category")
        if role not in DRAWER_SWAP_ROLES:
            dropped.append(obj)
            continue
        fp = _footprint_m(obj)
        pos = obj.get("position")
        ghost = False
        for prior in unique:
            prior_role = prior.get("stagedRole") or prior.get("role") or prior.get("category")
            if prior_role != role:
                continue
            if abs(fp - _footprint_m(prior)) > DRAWER_ROLE_DUPE_FP_TOL_M:
                continue
            prior_pos = prior.get("position")
            if pos is not None and prior_pos is not None:
                if math.dist(pos[:2], prior_pos[:2]) > DRAWER_ROLE_DUPE_XY_M:
                    # Far same-role with a soft existence is usually a drifted ghost, not a
                    # second real bottle (seed2b: e:277 perfume sibling of e:400).
                    if float(obj.get("existence") or 0.0) < STAGE_PICK_MIN_EXISTENCE:
                        ghost = True
                        break
                    continue
            ghost = True
            break
        if ghost:
            dropped.append(obj)
            continue
        unique.append(obj)
    return unique, dropped


def _uncertain_drawer_tracks(summary: dict, holder: dict) -> list[dict]:
    """In-drawer tracks that are not meal/outside rejects and not yet whitelisted."""
    inside = _drawer_on_holder(summary, holder)
    out = [o for o in inside
           if not _drawer_content_reject_reasons(o, holder) and not _drawer_deployed(o)]
    return sorted(out, key=_area, reverse=True)


def _reject_brief_drawer(obj: dict, reasons: list[str]) -> dict:
    """One rejected / unconfirmed drawer-content row for the chores log."""
    return {
        "id": obj.get("id"),
        "category": obj.get("category"),
        "role": obj.get("role", obj.get("category")),
        "reasons": reasons,
    }


def _contents_rejected(summary: dict, holder: dict) -> list[dict]:
    """Brief rows for tracks with ``on`` = ``holder`` that hard gates refuse (not unconfirmed)."""
    inside = _drawer_on_holder(summary, holder)
    out = [_reject_brief_drawer(obj, reasons)
           for obj in inside
           if (reasons := _drawer_content_reject_reasons(obj, holder))]
    return sorted(out, key=lambda r: r.get("id") or "")


def _drawer_of(summary: dict, furniture: dict) -> dict | None:
    """The drawer that belongs to ``furniture``.

    Search radius scales with the furniture footprint: a wide sideboard's closed drawer centre can
    sit more than 0.6 m from the fitted box centre, which used to yield "no drawer seen". Falls
    back to lower-existence drawer tracks (still labelled drawer) when role filtering finds none.
    """
    extent = furniture.get("extentM") or furniture["sizeM"]
    radius = max(0.9, max(extent[0], extent[1]) / 2 + 0.5)
    fx, fy = furniture["position"][:2]

    def near(cands: list[dict]) -> dict | None:
        hit = [d for d in cands if math.dist(d["position"][:2], [fx, fy]) < radius]
        return min(hit, key=lambda d: math.dist(d["position"][:2], [fx, fy])) if hit else None

    found = near(_objects(summary, "drawer"))
    if found is not None:
        return found
    weak = [o for o in summary["objects"]
            if (o.get("role") == "drawer" or o["category"] == "drawer") and o["existence"] >= 0.2]
    return near(weak)


class Chores:
    """Runs the chores phases against one robot's Ego API."""

    def __init__(self, api: str, timeout_s: float = 180.0, on_step=None):
        self.api = api
        self.timeout_s = timeout_s
        self.on_step = on_step
        self.log: list[dict] = []
        self.plan = {"goal": GOAL, "source": "家务循环", "round": None, "current": "", "difficulty": "",
                     "state": "running", "status": {phase: "pending" for phase in PHASES}}
        self.attempt = ""
        self.failures: list[dict] = []

    def publish(self, **changes) -> None:
        """Update the published plan (goal, phase status, current step, latest difficulty) that the
        console shows. The plan is informational: a failed publish is logged and work goes on."""
        status = changes.pop("status", {})
        self.plan.update(changes)
        self.plan["status"].update(status)
        body = {key: value for key, value in self.plan.items() if key != "status"}
        body["steps"] = [{"name": f"{PHASE_NAMES[phase]}：{PHASE_DETAILS[phase]}", "status": st} for phase, st in self.plan["status"].items()]
        try:
            call("POST", f"{self.api}/ego/plan", body, timeout=2.0)
        except (HttpError, OSError) as exc:
            self.log.append({"step": "plan-publish-failed", "error": str(exc)})

    def _clearing(self, phase: str, action):
        """Run ``action``; when it is blocked by movable entities, move each one out of the way
        (at most two) and try again."""
        cleared: set[str] = set()
        while True:
            try:
                return action()
            except Blocked as exc:
                movable = [b for b in exc.blockers if b.get("movable") and b["id"] not in cleared]
                if not movable or len(cleared) >= 2:
                    raise
                self._clear(phase, movable[0])
                cleared.add(movable[0]["id"])

    def _clear(self, phase: str, blocker: dict) -> None:
        """Carry a blocking floor object to a parking spot out of the way."""
        obj = _refresh(self.api, {"id": blocker["id"], "category": blocker["category"]})
        try:
            _approach_to_observe(self.api, obj, distance=0.9)
        except MissionFailed:
            pass
        spots = blocker.get("parkAt") or []
        if not spots:
            raise MissionFailed(f"{blocker['id']} ({blocker['category']}) is in the way and there is no free spot to move it to")
        self._doing(f"清障：把挡住操作路径的 {describe(obj)} 搬开")
        self.log.append({"step": "clear-start", "phase": phase, "object": obj["id"],
                         "category": obj.get("category"), "parkAt": spots})
        pick = _safe_receipt(self.api, "pick", {"target": obj["id"]}, obj, self.log)
        self.log.append({"step": "pick-receipt", "object": obj["id"], "reason": "clear", **receipt_brief(pick)})
        _run(self.api, pick, self.timeout_s)
        for spot in spots:
            receipt = call("POST", f"{self.api}/ego/predict", {"skill": "place", "params": {"at": spot}})
            self.log.append({"step": "place-receipt", "object": obj["id"], "reason": "clear",
                             "params": {"at": spot}, **receipt_brief(receipt)})
            if receipt["withinRiskLimit"]:
                _run(self.api, receipt, self.timeout_s)
                self._step(phase, f"清障完成：{describe(obj)} 放到 ({spot[0]:.2f}, {spot[1]:.2f})",
                           object=blocker["id"], parkAt=spot)
                return
        raise MissionFailed(f"picked {blocker['id']} but none of the parking spots {spots} is reachable")

    def _confirmed(self, role: str) -> dict:
        """The object playing ``role`` in the Ego-State, made sure of before it is acted on: while its
        role is doubtful the robot looks at it from closer, and when no object has the role it looks
        at the likeliest candidate, instead of failing on a confused category."""
        zh = CATEGORY_ZH.get(role, role)
        searched = False
        for attempt in range(CONFIRM_TRIES + 1):
            s = _summary(self.api)
            found = _objects(s, role)
            last = attempt == CONFIRM_TRIES
            if not found:
                candidates = [o for o in s["objects"] if o["existence"] >= 0.5 and o["categories"].get(role, 0.0) >= CANDIDATE_MIN]
                if not candidates and not searched:
                    # Nothing looks like it: it has not been seen yet, so look for it before guessing.
                    searched = True
                    self._search(role)
                    continue
                if not candidates or last:
                    raise MissionFailed(f"no {role} in the ego state")
                cand = max(candidates, key=lambda o: o["categories"][role])
                self._doing(f"寻找{zh}：没有物体被认作{zh}，靠近最可能的候选{describe(cand)}（是{zh}的概率 "
                            f"{cand['categories'][role]:.2f}）观察")
                _approach_to_observe(self.api, cand, distance=CONFIRM_DISTANCE)
                simtime_sleep(self.api, 1.5)
                continue
            obj = max(found, key=lambda o: o["existence"])
            confidence = obj.get("roleConfidence", 1.0)
            if confidence >= CONFIRM_CONFIDENCE or last:
                return obj
            self._doing(f"确认{describe(obj)}：它是{zh}的把握只有 {confidence:.2f}，先靠近观察")
            _approach_to_observe(self.api, obj, distance=CONFIRM_DISTANCE)
            simtime_sleep(self.api, 1.5)
        raise MissionFailed(f"no {role} in the ego state")

    def _search(self, role: str) -> None:
        """Tour the room's viewpoints, looking around at each, until some object takes ``role``."""
        zh = CATEGORY_ZH.get(role, role)
        for x, y in self._viewpoints():
            self._doing(f"寻找{zh}：还没有看到像{zh}的物体，到 ({x:.1f}, {y:.1f}) 环顾一周")
            receipt = call("POST", f"{self.api}/ego/predict", {"skill": "navigate", "params": {"x": x, "y": y}})
            if receipt["withinRiskLimit"]:
                try:
                    _run(self.api, receipt, self.timeout_s)
                except MissionFailed as exc:
                    self.log.append({"step": "search-move-failed", "error": str(exc)})
            _explore(self.api, self.timeout_s)
            if _objects(_summary(self.api), role):
                return

    def _step(self, phase: str, text: str, **extra) -> None:
        record = {"phase": phase, "text": text, **extra}
        self.log.append(record)
        self.publish(current=text)
        if self.on_step is not None:
            self.on_step(record)

    def _doing(self, text: str) -> None:
        """Record and publish the action about to start, so a failure names what was being attempted."""
        self.attempt = text
        self.log.append({"step": "doing", "text": text})
        self.publish(current=f"正在{text}")
        if self.on_step is not None:
            self.on_step({"step": "doing", "text": text})

    def note_failure(self, phase: str, message: str) -> None:
        """Keep a failed phase with the action it was attempting and where the robot stood."""
        try:
            me = _summary(self.api)["self"]
            pose = [me["x"], me["y"]]
        except (HttpError, OSError):  # the pose is supporting evidence only; the failure is kept without it
            pose = None
        self.failures.append({"phase": phase, "step": self.attempt, "message": message, "pose": pose})
        del self.failures[:-FAILURE_HISTORY]

    def diagnose(self) -> Diagnosis:
        """Explain the failures since the last completed phase from this run's log and the Ego-State."""
        needs: dict[str, int] = {}
        for failure in self.failures:
            needs.update(PHASE_NEEDS[failure["phase"]])
        try:
            summary = _summary(self.api)
        except (HttpError, OSError):  # diagnose from the failure log alone
            summary = None
        return diagnose(self.failures, summary, needs)

    # ------------------------------------------------------------------ table
    def arrange_on_table(self) -> list[str]:
        table = self._confirmed("table")
        # Plan the row only after a close look: a transparent glass is not detected in every frame.
        _approach_to_observe(self.api, table, distance=1.3)
        simtime_sleep(self.api, 2.0)
        s = _settle(self.api, seconds=25.0)
        table = _refresh(self.api, table)
        items = tableware(s)
        # The task is to arrange what stands on the table. When at least two pieces are seen
        # resting on it, those are the row — a fragment of the tabletop's edge that happens to
        # read as a bowl has no support and stays out.
        on_table = [o for o in items if o.get("on") == table["id"]]
        if len(on_table) >= 2:
            items = on_table
        if len(items) < 2:
            raise MissionFailed(f"only {len(items)} pieces of tableware are known")
        offsets = row_offsets([_width(o) for o in items])
        order = []
        for obj, x in zip(items, offsets, strict=True):
            obj = _refresh(self.api, obj)
            order.append(obj["id"])
            spot = _world(table, _offset(table, x, TABLE_ROW_Y))
            if obj["on"] == table["id"] and math.dist(obj["position"][:2], spot) < 0.03:
                continue
            self._doing(f"把{describe(obj)} 放到餐桌第 {len(order)} 位")
            self._clearing("table", lambda o=obj, xy=spot: _move(self.api, o, _place_at(self.api, table, xy),
                                                                  table, self.log, self.timeout_s))
            self._step("table", f"{describe(obj)} → 餐桌第 {len(order)} 位", object=obj["id"])
        return order

    # -------------------------------------------------------------- sideboard
    def stack_on_sideboard(self) -> None:
        board = self._confirmed("sideboard")
        # Plan only after a close look at the surface tableware still stands on (same idea as
        # arrange_on_table): small bowls and transparent glasses often lack ``on`` until seen
        # from nearby, and ``_supported_first`` would then drop them from the stacking list.
        s0 = _summary(self.api)
        source = max(_objects(s0, "table"), key=lambda o: o["existence"], default=None)
        if source is not None:
            self._doing(f"近看{describe(source)} 上的餐具再规划叠放")
            self.log.append({"step": "observe-before-plan", "surface": source["id"]})
            try:
                _approach_to_observe(self.api, source, distance=1.3)
            except MissionFailed:
                pass
            simtime_sleep(self.api, 2.0)
        else:
            self.log.append({"step": "observe-before-plan", "surface": None})
        s = _settle(self.api, seconds=25.0, want=2)
        board = _refresh_or_recover(self.api, board)
        # Transparent glass and other look-alikes often sit as ``unknown`` until nearer multi-view
        # looks raise a plan-ready tableware role. Stop early per track once ``role`` clears inventory
        # filters; after RESCAN_MAX_VIEWS still-unconfirmed tracks are left out of others (no
        # geometry-forced planning).
        doubtful = _doubtful_tableware_for_rescan(s)
        if doubtful:
            self.log.append({"step": "rescan-before-plan",
                             "targets": [o["id"] for o in doubtful]})
        unconfirmed: list[dict] = []
        for obj in doubtful:
            oid = obj["id"]
            confirmed = False
            views = 0
            last_role = obj.get("role", obj.get("category"))
            for offset in RESCAN_VIEW_OFFSETS[:RESCAN_MAX_VIEWS]:
                self._doing(f"近看{describe(obj)} 以确认类别")
                try:
                    _approach_to_observe(self.api, obj, distance=0.9, prefer_offset=offset)
                except MissionFailed:
                    pass
                views += 1
                simtime_sleep(self.api, RESCAN_SETTLE_S)
                s = _summary(self.api)
                fresh = _summary_object(s, oid)
                if fresh is None:
                    continue
                obj = fresh
                last_role = fresh.get("role", fresh.get("category"))
                if _plan_ready_tableware(fresh, s):
                    confirmed = True
                    break
            self.log.append({"step": "rescan-object", "id": oid, "views": views,
                             "confirmed": confirmed, "role": last_role})
            if not confirmed:
                unconfirmed.append({"id": oid, "role": last_role, "views": views})
        if doubtful:
            s = _settle(self.api, seconds=12.0, want=1)
            board = _refresh_or_recover(self.api, board)
        # Inventory by freshness / off-floor / near a support — not by ``on``. Thin or transparent
        # pieces often lack ``on`` after a table look while siblings already have one; dropping them
        # here left them on the table. Unknown support is resolved at pick via ``_with_known_support``.
        bowls, others, rejected = _sideboard_plan_inventory(s)
        planned_ids = {o["id"] for o in bowls} | {o["id"] for o in others}
        rejected_ids = {r["id"] for r in rejected}
        for row in unconfirmed:
            if row["id"] in planned_ids:
                continue
            if row["id"] in rejected_ids:
                for r in rejected:
                    if r["id"] == row["id"]:
                        reasons = list(r.get("reasons") or [])
                        if "unconfirmed" not in reasons:
                            reasons.append("unconfirmed")
                        r["reasons"] = reasons
                        r["views"] = row["views"]
                        break
                continue
            track = _summary_object(s, row["id"])
            if track is not None:
                brief = {**_reject_brief(track, ["unconfirmed"]), "views": row["views"]}
            else:
                brief = {"id": row["id"], "role": row.get("role"), "reasons": ["unconfirmed"],
                         "views": row["views"]}
            rejected.append(brief)
            rejected_ids.add(row["id"])
        rejected = sorted(rejected, key=lambda r: r.get("id") or "")
        plan = {
            "step": "plan-sideboard",
            "board": board["id"],
            "bowls": [_plan_obj(b) for b in bowls],
            "others": [_plan_obj(o) for o in others],
            "rejected": rejected,
        }
        self.log.append(plan)
        if self.on_step is not None:
            self.on_step(plan)
        widths = ([_width(bowls[0])] if bowls else []) + [_width(o) for o in others]
        offsets = row_offsets(widths, gap=0.08)
        slot = iter(offsets)
        below = None
        for bowl in bowls:
            bowl = self._with_known_support(bowl)
            if below is None:
                offset = next(slot)
                if bowl["on"] != board["id"]:
                    spot = _world(board, _offset(board, offset, 0.0))
                    self._doing(f"把{describe(bowl)} 放到餐边柜上作叠放底座")
                    was = bowl
                    bowl = self._clearing("sideboard", lambda b=bowl, xy=spot: _move(
                        self.api, b, _place_at(self.api, board, xy), board, self.log, self.timeout_s))
                    self._step("sideboard", f"{describe(was)} → 餐边柜（叠放底座）", object=bowl["id"])
                else:
                    self.log.append({"step": "skip", "phase": "sideboard", "reason": "already-on-board",
                                     "object": bowl["id"], "on": bowl.get("on"), "board": board["id"]})
            else:
                below = _refresh_or_recover(self.api, below)
                bowl = self._with_known_support(bowl, expect=below)
                if bowl["on"] != below["id"]:
                    self._doing(f"把{describe(bowl)} 叠进{describe(below)}")
                    was = bowl
                    bowl = self._clearing("sideboard", lambda b=bowl, u=below: _move(
                        self.api, b, {"on": u["id"]}, u, self.log, self.timeout_s))
                    self._step("sideboard", f"{describe(was)} → 叠进{describe(below)}",
                               object=bowl["id"], below=below["id"])
                else:
                    self.log.append({"step": "skip", "phase": "sideboard", "reason": "already-nested",
                                     "object": bowl["id"], "on": bowl.get("on"), "below": below["id"]})
            below = bowl
        for obj in others:
            obj = self._with_known_support(obj)
            offset = next(slot)
            if obj["on"] == board["id"]:
                self.log.append({"step": "skip", "phase": "sideboard", "reason": "already-on-board",
                                 "object": obj["id"], "on": obj.get("on"), "board": board["id"]})
                continue
            spot = _world(board, _offset(board, offset, 0.0))
            self._doing(f"把{describe(obj)} 放到餐边柜上")
            self._clearing("sideboard", lambda o=obj, xy=spot: _move(
                self.api, o, _place_at(self.api, board, xy), board, self.log, self.timeout_s))
            self._step("sideboard", f"{describe(obj)} → 餐边柜", object=obj["id"])

    def _with_known_support(self, obj: dict, expect: dict | None = None) -> dict:
        """Refresh ``obj``; if its support is still unknown (or not ``expect``), look closer once.

        Phase 1 nesting often leaves ``on`` empty until a near look updates the bowl–bowl link;
        picking blind then fails or places on the wrong surface."""
        obj = _refresh_or_recover(self.api, obj)
        need = obj.get("on") is None or (expect is not None and obj.get("on") != expect["id"])
        if not need:
            return obj
        self._doing(f"近看{describe(obj)} 以确认它放在什么上面")
        try:
            _approach_to_observe(self.api, obj, distance=0.9)
        except MissionFailed:
            pass
        simtime_sleep(self.api, 1.5)
        return _refresh_or_recover(self.api, obj)

    # ----------------------------------------------------------------- drawers
    def _open(self, furniture: dict) -> dict:
        return self._clearing("drawers", lambda: self._open_once(furniture))

    def _open_once(self, furniture: dict) -> dict:
        self._doing(f"走到{describe(furniture)} 前观察并打开它的抽屉")
        _view(self.api, furniture, self.timeout_s)
        s = _summary(self.api)
        drawer = _drawer_of(s, furniture)
        if drawer is None:
            # One more close look: the first stop sometimes sees only the furniture body.
            _approach_to_observe(self.api, furniture, distance=0.95)
            simtime_sleep(self.api, 1.5)
            drawer = _drawer_of(_summary(self.api), furniture)
        if drawer is None:
            raise MissionFailed(f"no drawer seen in front of {furniture['id']}")
        if drawer.get("openingM", 0.0) < 0.2:
            receipt = call("POST", f"{self.api}/ego/predict", {"skill": "open", "params": {"target": drawer["id"]}})
            if not receipt["withinRiskLimit"]:
                raise infeasible(f"open {drawer['id']}", receipt)
            _run(self.api, receipt, self.timeout_s)
            simtime_sleep(self.api, 1.5)
            self._step("drawers", f"打开{describe(furniture)} 的抽屉", object=drawer["id"])
        # Look into the open drawer so young content tracks gather observations before picking.
        drawer = _refresh(self.api, drawer)
        _approach_to_observe(self.api, drawer, distance=0.95)
        simtime_sleep(self.api, 1.5)
        live = _refresh(self.api, drawer)
        # Remember the host so a later close can re-find the drawer after a long cross-room swap.
        return {**live, "hostId": furniture["id"],
                "hostRole": furniture.get("role") or furniture.get("category")}

    def _live_drawer(self, drawer: dict, furniture: dict | None = None) -> dict | None:
        """Current drawer track for close: same id, else the drawer of ``furniture`` after a look.

        After a long swap the open drawer id is often gone while the piece still stands out of the
        carcass. Bare :func:`_refresh` then fails; re-acquire via the host like :meth:`_open_once`.
        """
        try:
            return _refresh(self.api, drawer)
        except MissionFailed as exc:
            self.log.append({"step": "drawer-close-rematch", "reason": "refresh-missed",
                             "from": drawer.get("id"), "error": str(exc)})
        host = furniture
        if host is None and drawer.get("hostId"):
            host = _summary_object(_summary(self.api), drawer["hostId"])
        if host is None:
            role = drawer.get("hostRole")
            if role in TOP_SUPPORT_ROLES:
                try:
                    host = self._confirmed(role)
                except MissionFailed:
                    host = None
        if host is None:
            return None
        try:
            host = _refresh_or_recover(self.api, host)
        except MissionFailed:
            pass
        self._doing(f"近看{describe(host)} 以重新找到要关的抽屉")
        try:
            _approach_to_observe(self.api, host, distance=0.95)
        except MissionFailed:
            pass
        simtime_sleep(self.api, 1.5)
        found = _drawer_of(_summary(self.api), host)
        if found is None:
            try:
                _view(self.api, host, self.timeout_s)
            except MissionFailed:
                pass
            found = _drawer_of(_summary(self.api), host)
        if found is None:
            self.log.append({"step": "drawer-close-rematch", "reason": "no-drawer",
                             "from": drawer.get("id"), "host": host.get("id")})
            return None
        self.log.append({"step": "drawer-close-rematch", "reason": "matched",
                         "from": drawer.get("id"), "to": found["id"], "host": host.get("id"),
                         "openingM": found.get("openingM")})
        return {**found, "hostId": host.get("id"),
                "hostRole": host.get("role") or host.get("category") or drawer.get("hostRole")}

    def _seat_closed_drawer(self, drawer: dict, furniture: dict | None,
                            *, attempt: int | None = None) -> None:
        """One short explicit close after ES already reads under the world gate.

        Protrusion ``openingM`` can sit at a few millimetres while the prismatic joint is still
        above :func:`drawers_closed`. Passing :data:`DRAWER_CLOSE_SEAT_M` forces a push that the
        bare close would refuse as already closed. Seat refusal or a vanished track still accepts
        the under-gate reading — do not reopen.
        """
        live = drawer
        offset = 0.0 if attempt is None else RESCAN_VIEW_OFFSETS[min(attempt, len(RESCAN_VIEW_OFFSETS) - 1)]
        try:
            _approach_to_observe(self.api, live, distance=DRAWER_PLACE_ALIGN_DISTANCE,
                                 prefer_offset=offset)
        except MissionFailed as align_exc:
            self.log.append({"step": "drawer-close-seat-align-failed", "drawer": live.get("id"),
                             "error": str(align_exc)})
        self._doing(f"坐实关上抽屉 {live['id']}")
        receipt = call("POST", f"{self.api}/ego/predict",
                       {"skill": "close",
                        "params": {"target": live["id"], "distance": DRAWER_CLOSE_SEAT_M}})
        self.log.append({"step": "drawer-close-seat", "drawer": live["id"],
                         "distance": DRAWER_CLOSE_SEAT_M,
                         "openingM": live.get("openingM"),
                         "feasible": bool(receipt.get("withinRiskLimit")),
                         "issues": list(receipt.get("issues") or [])})
        if not receipt.get("withinRiskLimit"):
            opening = float(live.get("openingM") or 0.0)
            self.log.append({"step": "drawer-close-verified", "drawer": live["id"],
                             "openingM": opening, "reason": "seat-skipped"})
            self._step("drawers", "关上抽屉", object=live["id"])
            return
        _run(self.api, receipt, self.timeout_s)
        simtime_sleep(self.api, RESCAN_SETTLE_S)
        checked = self._live_drawer(live, furniture)
        if checked is None:
            self.log.append({"step": "drawer-close-verified", "drawer": live["id"],
                             "openingM": 0.0, "reason": "seat-no-drawer"})
            self._step("drawers", "关上抽屉", object=live["id"])
            return
        self.log.append({"step": "drawer-close-verified", "drawer": checked["id"],
                         "openingM": checked.get("openingM"), "reason": "seat"})
        self._step("drawers", "关上抽屉", object=checked["id"])

    def _close(self, drawer: dict, furniture: dict | None = None) -> None:
        """Push ``drawer`` shut; re-find it on ``furniture`` when the open-track id was dropped.

        After each close, re-read ``openingM`` and keep pushing while above the 0.02 m gate of
        :func:`drawers_closed`. When ES first reads under that gate, run one seat push
        (:meth:`_seat_closed_drawer`) before accepting — the estimate alone under-reads the joint.
        Do not reopen for a sub-gate residual when the skill already reports closed.
        """
        live = self._live_drawer(drawer, furniture)
        if live is None:
            # No drawer track after a host look: typically shut and merged into the furniture face.
            self.log.append({"step": "drawer-close-skip", "reason": "no-drawer",
                             "from": drawer.get("id"),
                             "host": (furniture or {}).get("id") or drawer.get("hostId")})
            return
        if float(live.get("openingM") or 0.0) <= DRAWER_CLOSE_OK_M:
            self.log.append({"step": "drawer-close-under-gate", "drawer": live.get("id"),
                             "openingM": live.get("openingM"), "reason": "already-shut"})
            self._seat_closed_drawer(live, furniture)
            return
        last_id = live["id"]
        last_opening = float(live.get("openingM") or 0.0)
        last_receipt: dict | None = None
        for attempt in range(DRAWER_CLOSE_TRIES):
            # Face the slide before predicting close (same idea as drawer place align).
            offset = RESCAN_VIEW_OFFSETS[min(attempt, len(RESCAN_VIEW_OFFSETS) - 1)]
            try:
                _approach_to_observe(self.api, live, distance=DRAWER_PLACE_ALIGN_DISTANCE,
                                     prefer_offset=offset)
                self.log.append({"step": "drawer-close-align", "drawer": live["id"],
                                 "offset": offset, "attempt": attempt})
            except MissionFailed as align_exc:
                self.log.append({"step": "drawer-close-align-failed", "drawer": live["id"],
                                 "offset": offset, "attempt": attempt, "error": str(align_exc)})
            self._doing(f"关上抽屉 {live['id']}")
            receipt = call("POST", f"{self.api}/ego/predict",
                           {"skill": "close", "params": {"target": live["id"]}})
            last_receipt = receipt
            if not receipt["withinRiskLimit"]:
                issues = list(receipt.get("issues") or [])
                already_closed = any("already closed" in str(i).lower() for i in issues)
                self.log.append({"step": "drawer-close-retry", "drawer": live["id"],
                                 "attempt": attempt, "openingM": live.get("openingM"),
                                 "tries": DRAWER_CLOSE_TRIES,
                                 "reason": ("already-closed" if already_closed else "infeasible"),
                                 "issues": issues})
                if already_closed:
                    # Skill refuses while ES still shows a sub-world-gate residual — seat if under
                    # the gate, else rematch; do not reopen (that can widen a nearly-shut slide).
                    checked = self._live_drawer(live, furniture)
                    if checked is None or float(checked.get("openingM") or 0.0) <= DRAWER_CLOSE_OK_M:
                        self._seat_closed_drawer(checked or live, furniture, attempt=attempt)
                        return
                    # Skill and ES disagree above the world gate: rematch and try again.
                    _retreat_while_holding(self.api, self.log, self.timeout_s)
                    ghost = {**checked, "id": f"stale:{checked['id']}"}
                    rematched = self._live_drawer(ghost, furniture)
                    if rematched is None or float(rematched.get("openingM") or 0.0) <= DRAWER_CLOSE_OK_M:
                        self._seat_closed_drawer(rematched or checked, furniture, attempt=attempt)
                        return
                    live = rematched
                    last_id = live["id"]
                    last_opening = float(live.get("openingM") or 0.0)
                    if attempt + 1 >= DRAWER_CLOSE_TRIES:
                        raise MissionFailed(
                            f"drawer {last_id} skill says closed but opening "
                            f"{last_opening * 100:.0f} cm after rematch")
                    continue
                # Renumbered drawer collides with the carcass — step back, re-acquire on the host
                # from a new heading, and try again (or accept shut).
                _retreat_while_holding(self.api, self.log, self.timeout_s)
                ghost = {**live, "id": f"stale:{live['id']}"}
                rematched = self._live_drawer(ghost, furniture)
                if rematched is None:
                    self.log.append({"step": "drawer-close-verified", "drawer": live["id"],
                                     "attempt": attempt, "openingM": 0.0,
                                     "reason": "infeasible-no-drawer"})
                    self._step("drawers", "关上抽屉", object=live["id"])
                    return
                if float(rematched.get("openingM") or 0.0) <= DRAWER_CLOSE_OK_M:
                    self._seat_closed_drawer(rematched, furniture, attempt=attempt)
                    return
                live = rematched
                last_id = live["id"]
                last_opening = float(live.get("openingM") or 0.0)
                if attempt + 1 >= DRAWER_CLOSE_TRIES:
                    raise infeasible(f"close {last_id}", receipt)
                continue
            _run(self.api, receipt, self.timeout_s)
            last_id = live["id"]
            simtime_sleep(self.api, RESCAN_SETTLE_S)
            checked = self._live_drawer(live, furniture)
            if checked is None:
                self.log.append({"step": "drawer-close-verified", "drawer": last_id,
                                 "attempt": attempt, "openingM": 0.0, "reason": "no-drawer"})
                self._step("drawers", "关上抽屉", object=last_id)
                return
            last_opening = float(checked.get("openingM") or 0.0)
            # Same gate as drawers_closed, then one seat push for joint residual.
            if last_opening <= DRAWER_CLOSE_OK_M:
                self._seat_closed_drawer(checked, furniture, attempt=attempt)
                return
            self.log.append({"step": "drawer-close-retry", "drawer": checked["id"],
                             "attempt": attempt, "openingM": last_opening,
                             "tries": DRAWER_CLOSE_TRIES, "reason": "still-open"})
            live = checked
        if last_receipt is not None and not last_receipt.get("withinRiskLimit"):
            raise infeasible(f"close {last_id}", last_receipt)
        raise MissionFailed(
            f"drawer {last_id} still open {last_opening * 100:.0f} cm after {DRAWER_CLOSE_TRIES} close attempts")

    def _confirm_drawer_contents(self, drawer: dict, furniture: str) -> list[dict]:
        """Carry only :data:`DRAWER_SWAP_ROLES`; look closer at uncertain in-drawer tracks, else skip.

        Hard rejects (meal tableware / outside the open box) never get a close look. Tracks that
        are not whitelisted yet (``cup``, ``car_body``, …) get up to :data:`DRAWER_CONFIRM_VIEWS`
        nearer views; a view that names a deployed role keeps them, otherwise they are logged as
        ``unconfirmed`` and left alone.
        """
        drawer = _refresh_or_recover(self.api, drawer)
        s = _summary(self.api)
        kept = list(_contents(s, drawer))
        rejected = list(_contents_rejected(s, drawer))
        unconfirmed: list[dict] = []

        def _confirm_one(obj: dict, *, view_budget: int) -> tuple[dict | None, dict, int, list[str] | None]:
            """Near looks for one in-drawer track. Returns (promoted, last, views, hard_reasons)."""
            oid = obj["id"]
            views = 0
            hard_reasons: list[str] | None = None
            last = obj
            nonlocal drawer, s
            for offset in RESCAN_VIEW_OFFSETS[:view_budget]:
                self._doing(f"近看{describe(obj)} 以确认是否为抽屉物品")
                try:
                    _approach_to_observe(self.api, obj, distance=DRAWER_CONFIRM_DISTANCE,
                                         prefer_offset=offset)
                except MissionFailed:
                    pass
                views += 1
                simtime_sleep(self.api, RESCAN_SETTLE_S)
                s = _summary(self.api)
                drawer = _summary_object(s, drawer["id"]) or drawer
                fresh = _summary_object(s, oid)
                if fresh is None:
                    continue
                last = fresh
                obj = fresh
                hard_reasons = _drawer_content_reject_reasons(fresh, drawer) or None
                if hard_reasons:
                    return None, last, views, hard_reasons
                if _drawer_deployed(fresh):
                    return fresh, last, views, None
            return None, last, views, hard_reasons

        for obj in _uncertain_drawer_tracks(s, drawer):
            promoted, last, views, hard_reasons = _confirm_one(obj, view_budget=DRAWER_CONFIRM_VIEWS)
            oid = obj["id"]
            if promoted is not None:
                kept.append(promoted)
                self.log.append({"step": "drawer-confirm", "drawer": drawer["id"],
                                 "furniture": furniture, "id": oid, "views": views,
                                 "confirmed": True,
                                 "role": last.get("role", last.get("category"))})
                continue
            reasons = list(hard_reasons or ["unconfirmed"])
            brief = {**_reject_brief_drawer(last, reasons), "views": views}
            rejected.append(brief)
            if "unconfirmed" in reasons:
                unconfirmed.append(last)
            self.log.append({"step": "drawer-confirm", "drawer": drawer["id"],
                             "furniture": furniture, "id": oid, "views": views,
                             "confirmed": False,
                             "role": last.get("role", last.get("category")),
                             "reasons": reasons})
        # One extra pass on the largest still-unconfirmed in-box track (airplane-scale look-alikes
        # often need a third heading; small bottles already had the normal budget).
        large = [o for o in unconfirmed if _width(o) >= DRAWER_CONFIRM_LARGE_M]
        if large:
            candidate = max(large, key=_area)
            promoted, last, views, hard_reasons = _confirm_one(
                candidate, view_budget=max(1, DRAWER_CONFIRM_VIEWS - 1))
            oid = candidate["id"]
            if promoted is not None:
                kept.append(promoted)
                rejected = [r for r in rejected if r.get("id") != oid]
                self.log.append({"step": "drawer-confirm", "drawer": drawer["id"],
                                 "furniture": furniture, "id": oid, "views": views,
                                 "confirmed": True, "extra": True,
                                 "role": last.get("role", last.get("category"))})
            else:
                self.log.append({"step": "drawer-confirm", "drawer": drawer["id"],
                                 "furniture": furniture, "id": oid, "views": views,
                                 "confirmed": False, "extra": True,
                                 "role": last.get("role", last.get("category")),
                                 "reasons": list(hard_reasons or ["unconfirmed"])})
        kept = sorted(kept, key=_area, reverse=True)
        # Stamp the source drawer so a later pick rematch prefers in-box neighbours over a
        # same-role sibling stolen via global footprint search (two pills, or a misnamed airplane).
        kept = [{**o, "drawerId": drawer["id"],
                 "stagedRole": o.get("role") or o.get("category")} for o in kept]
        kept, role_dupes = _dedupe_drawer_roles(kept)
        for dupe in role_dupes:
            rejected.append(_reject_brief_drawer(
                dupe, ["duplicate-role", dupe.get("stagedRole") or dupe.get("role") or ""]))
            self.log.append({"step": "drawer-role-dupe", "drawer": drawer["id"],
                             "furniture": furniture, "id": dupe.get("id"),
                             "role": dupe.get("stagedRole") or dupe.get("role"),
                             "kept": [o["id"] for o in kept]})
        rejected = sorted(rejected, key=lambda r: r.get("id") or "")
        self.log.append({"step": "drawer-contents", "drawer": drawer["id"], "furniture": furniture,
                         "kept": [o["id"] for o in kept], "rejected": rejected})
        return kept

    def _evacuate_drawer_leftovers(
            self, drawer: dict, dest: dict, how: str, *,
            collect: list[dict] | None = None,
            away_from: dict | None = None,
            skip_roles: set[str] | None = None) -> int:
        """Move leftover deployed swap-role occupants out of ``drawer`` onto/into ``dest``.

        Perception often leaves a real bottle after role confirm/dedupe; those still occupy place
        spots. Only :data:`DRAWER_SWAP_ROLES` are evacuated — unconfirmed look-alikes (``car_body``,
        …) burn grasp probes without clearing truth occupancy. ``skip_roles`` excludes roles that
        are already staged on a furniture top so a mis-associated stage ghost is not re-grasped
        out of the emptied cabinet drawer. Deployed leftovers use :meth:`_transfer` (and may join
        ``collect`` when dest is a stage top).

        @param drawer: open drawer to empty of leftovers
        @param dest: support that receives them (other drawer or furniture top)
        @param how: Chinese phrase for the doing line / log
        @param collect: when dest is a top, append successfully staged landings here
        @param away_from: stage-spot bias for top destinations (usually the open drawer)
        @param skip_roles: staged roles that must not be evacuated from this drawer
        @returns: number of successful evacuations
        """
        drawer = _refresh_or_recover(self.api, drawer)
        s = _summary(self.api)
        skip = skip_roles or set()
        leftovers = [
            o for o in _drawer_on_holder(s, drawer)
            if _drawer_deployed(o) and _in_open_drawer_volume(o, drawer)
            and (o.get("role") or o.get("category")) not in skip
        ]
        leftovers = sorted(leftovers, key=_area, reverse=True)[:DRAWER_EVACUATE_MAX]
        if not leftovers:
            return 0
        moved = 0
        dest_role = dest.get("role") or dest.get("category")
        staging = dest_role in TOP_SUPPORT_ROLES
        for i, item in enumerate(leftovers):
            try:
                stamped = {
                    **item,
                    "drawerId": drawer["id"],
                    "stagedRole": item.get("role") or item.get("category"),
                }
                if staging and collect is not None:
                    landed = self._transfer(
                        stamped, dest, how,
                        stage_index=len(collect),
                        stage_count=max(len(collect) + len(leftovers) - i, 1),
                        away_from=away_from or drawer)
                    if landed is not None:
                        collect.append(landed)
                        moved += 1
                        self.log.append({
                            "step": "drawer-evacuate", "from": drawer["id"],
                            "to": dest.get("id"), "id": landed.get("id"),
                            "role": landed.get("stagedRole") or landed.get("role"),
                            "staged": True})
                else:
                    landed = self._transfer(stamped, dest, how)
                    if landed is not None:
                        moved += 1
                        self.log.append({
                            "step": "drawer-evacuate", "from": drawer["id"],
                            "to": dest.get("id"), "id": landed.get("id"),
                            "role": landed.get("stagedRole") or landed.get("role")})
            except MissionFailed as exc:
                self.log.append({
                    "step": "drawer-evacuate-failed", "from": drawer.get("id"),
                    "id": item.get("id"),
                    "role": item.get("role") or item.get("category"),
                    "error": str(exc)})
        return moved

    def swap_drawers(self) -> None:
        """Exchange the contents of the sideboard and cabinet drawers. Only the pulled-out part of a
        drawer can take an object, so one drawer never holds both sets: the sideboard drawer's items
        wait on the sideboard top while the cabinet's move across."""
        board, cabinet = self._confirmed("sideboard"), self._confirmed("cabinet")
        a = self._open(board)
        a_items = self._confirm_drawer_contents(a, "sideboard")
        b = self._open(cabinet)
        b_items = self._confirm_drawer_contents(b, "cabinet")
        n_stage = len(a_items)
        staged: list[dict] = []
        for i, item in enumerate(a_items):
            landed = self._transfer(item, board, f"暂放到{describe(board)} 台面上",
                                    stage_index=i, stage_count=n_stage, away_from=a)
            if landed is not None:
                staged.append(landed)
        # Anything still in the sideboard drawer blocks incoming cabinet pieces.
        self._evacuate_drawer_leftovers(
            a, board, f"从餐边柜抽屉清到{describe(board)} 台面",
            collect=staged, away_from=a)
        for item in b_items:
            self._transfer(item, a, f"从柜子的抽屉移到餐边柜的抽屉 {a['id']}")
        # Leftovers that confirm/dedupe skipped must leave before staged pieces return.
        # Do not evacuate roles waiting on the stage top (mis-bound stage ghosts look in-drawer).
        staged_roles = {
            o.get("stagedRole") or o.get("role") or o.get("category")
            for o in staged
            if (o.get("stagedRole") or o.get("role") or o.get("category")) in DRAWER_SWAP_ROLES
        }
        self._evacuate_drawer_leftovers(
            b, a, f"从柜子抽屉清到餐边柜的抽屉 {a['id']}",
            skip_roles=staged_roles)
        # Re-assert stage tracks after a long cabinet→sideboard leg (whitelist ids often vanish).
        restaged: list[dict] = []
        for item in staged:
            xy = _stage_xy_of(item)
            if xy is None:
                restaged.append(item)
                continue
            confirmed = _confirm_stage_landing(
                self.api, item, xy, self.log, views=STAGE_CONFIRM_VIEWS)
            if any(e.get("step") == "stage-land-unconfirmed" and e.get("from") == item.get("id")
                   for e in self.log[-3:]):
                confirmed = _confirm_stage_landing(
                    self.api, confirmed, xy, self.log,
                    views=STAGE_CONFIRM_VIEWS + STAGE_RECONFIRM_VIEWS)
            _protect_stage(self.api, confirmed, self.log)
            restaged.append(confirmed)
        for item in restaged:
            self._transfer(item, b, f"从台面移到柜子的抽屉 {b['id']}",
                           drawer_evacuate_to=a)
        _clear_stage_protect(self.api, None, self.log)
        # Pass the host: after a long swap the open-drawer id is often gone from the ego state.
        self._clearing("drawers", lambda: self._close(b, cabinet))
        self._clearing("drawers", lambda: self._close(a, board))

    def _transfer(self, item: dict, dst: dict, how: str, *,
                  stage_index: int | None = None, stage_count: int = 1,
                  away_from: dict | None = None,
                  drawer_evacuate_to: dict | None = None) -> dict | None:
        """Carry ``item`` onto or into ``dst``; return it as it stands there (:func:`_landed`).

        Skips (returns ``None``) only when the listed track was never a deployed drawer role (stale
        cup / car-body). A previously staged whitelist piece is rematched near its ``stageXy`` and
        must not silently become tableware — that raises :class:`MissionFailed` instead of
        pretending the transfer finished. Temporary staging onto a furniture top (``stage_index``
        set) uses an edge spot away from ``away_from`` (the open drawer) so later docks at that
        drawer are not boxed in by the staged pile. ``drawer_evacuate_to`` is forwarded to
        :func:`_move` so a drawer-full place can clear one occupant onto that support and retry.
        """
        leaving_stage = _stage_xy_of(item) is not None and stage_index is None
        # Unconfirmed stage landings get one more close look before the board→cabinet pick.
        # If the look still fails, rematch/soft-keep via recover so a weak top track can still be
        # grasped — abort only when rematch also cannot find the whitelist role near stageXy.
        if leaving_stage and item.get("stageConfirmed") is False:
            xy = _stage_xy_of(item)
            if xy is not None:
                item = _confirm_stage_landing(
                    self.api, item, xy, self.log,
                    views=STAGE_CONFIRM_VIEWS + STAGE_RECONFIRM_VIEWS)
                _protect_stage(self.api, item, self.log)
                if item.get("stageConfirmed") is False:
                    try:
                        recovered = _recover_drawer_carry(self.api, item, self.log)
                        self.log.append({
                            "step": "stage-unconfirmed-recover",
                            "from": item.get("id"), "to": recovered.get("id"),
                            "role": recovered.get("stagedRole") or recovered.get("role"),
                            "stageXy": xy,
                            "stageConfirmed": bool(recovered.get("stageConfirmed"))})
                        item = recovered
                    except MissionFailed as rematch_exc:
                        raise MissionFailed(
                            f"staged {item.get('stagedRole') or item.get('id')} at {xy} "
                            f"still unconfirmed before transfer"
                        ) from rematch_exc
        expected_deployed = (_drawer_deployed(item) or _stage_xy_of(item) is not None
                             or item.get("stagedRole") in DRAWER_SWAP_ROLES)
        if expected_deployed:
            try:
                obj = _recover_drawer_carry(self.api, item, self.log)
            except MissionFailed as exc:
                # Staged pieces must rematch (stageXy). A source-drawer track that vanished mid-swap
                # after a sibling was already carried is skipped so the rest of the exchange continues.
                if leaving_stage or _stage_xy_of(item) is not None:
                    raise
                self.log.append({"step": "skip", "phase": "drawers", "reason": "lost-drawer-item",
                                 "object": item.get("id"),
                                 "role": item.get("stagedRole") or item.get("role") or item.get("category"),
                                 "drawerId": item.get("drawerId"), "error": str(exc)})
                return None
        else:
            obj = _refresh_or_recover(self.api, item, require_deployed=False)
            role = obj.get("role") or obj.get("category")
            cat = obj.get("category")
            if _drawer_meal_skip(obj) or not _drawer_deployed(obj):
                self.log.append({"step": "skip", "phase": "drawers", "reason": "non-drawer-content",
                                 "object": obj.get("id"), "role": role, "category": cat})
                return None
        if stage_index is not None:
            dst = _live_top_support(self.api, dst, self.log)
        else:
            dst = _refresh_or_recover(self.api, dst)
        dst_role = dst.get("role") or dst.get("category")
        stage_xy: list[float] | None = None
        staged_role = obj.get("role") or obj.get("category")
        if stage_index is not None and dst_role in TOP_SUPPORT_ROLES:
            stage_xy = _stage_top_xy(dst, away_from, stage_index, max(stage_count, 1))
            place = _place_at(self.api, dst, stage_xy, self.log)
            self.log.append({"step": "stage-spot", "object": obj["id"], "on": dst["id"],
                             "xy": [round(stage_xy[0], 3), round(stage_xy[1], 3)],
                             "index": stage_index, "count": stage_count})
        else:
            place = {"on": dst["id"]}
        self._doing(f"把{describe(obj)} {how}")
        try:
            landed = self._clearing(
                "drawers",
                lambda o=obj, p=place, d=dst, ev=drawer_evacuate_to: _move(
                    self.api, o, p, d, self.log, self.timeout_s,
                    drawer_evacuate_to=ev))
        except MissionFailed as exc:
            # ``_move`` re-runs recover after the approach; a source-drawer track can vanish
            # there even when the pre-pick recover succeeded.
            if (expected_deployed and stage_index is None and not leaving_stage
                    and "lost staged drawer item" in str(exc)):
                self.log.append({"step": "skip", "phase": "drawers", "reason": "lost-drawer-item",
                                 "object": obj.get("id"),
                                 "role": obj.get("stagedRole") or obj.get("role") or obj.get("category"),
                                 "drawerId": obj.get("drawerId"), "error": str(exc),
                                 "where": "move"})
                return None
            raise
        if stage_xy is not None:
            # Immediate look+rebind (like drawer land confirm): unverified release often leaves no
            # track id; waiting until board→cabinet finds an empty stageXy.
            seed = {**landed, "stagedRole": staged_role}
            landed = _confirm_stage_landing(
                self.api, seed, stage_xy, self.log, views=STAGE_CONFIRM_VIEWS)
            if any(e.get("step") == "stage-land-unconfirmed" and e.get("from") == seed.get("id")
                   for e in self.log[-3:]):
                # Second pass uses closer looks (see :func:`_confirm_stage_landing` distance ramp).
                landed = _confirm_stage_landing(
                    self.api, {**landed, "stagedRole": staged_role}, stage_xy, self.log,
                    views=STAGE_CONFIRM_VIEWS + STAGE_RECONFIRM_VIEWS)
            landed = {
                **landed,
                "stageXy": [round(stage_xy[0], 3), round(stage_xy[1], 3)],
                "stagedRole": staged_role,
                "position": [stage_xy[0], stage_xy[1],
                             (landed.get("position") or obj.get("position") or [0, 0, 0.85])[2]],
            }
            # Source-drawer id was for in-box rematch before staging; drop it so later recover
            # prefers the stage XY (not the open drawer that now holds the other cabinet set).
            landed.pop("drawerId", None)
            landed.pop("drawerLandXy", None)
            # Register before later perception cycles can drop a top-of-board pistol as a false end.
            _protect_stage(self.api, landed, self.log)
        if leaving_stage:
            _clear_stage_protect(self.api, item, self.log)
        self._step("drawers", f"{describe(obj)}：{how}", object=landed["id"], to=dst["id"])
        return landed

    # ---------------------------------------------------------------- recovery
    def recover(self, message: str = "") -> None:
        """After a failed step: put down a held object (a floor-standing one where it is, anything
        else on the table), and look around again only when something was missing from view."""
        try:
            s = _summary(self.api)
            held_id = s["self"].get("holding")
            if held_id:
                held = next((o for o in s["objects"] if o["id"] == held_id), None)
                extent = (held or {}).get("extentM") or [0, 0, 0]
                if held is not None and held["position"][2] - extent[2] / 2 < 0.15:
                    receipt = call("POST", f"{self.api}/ego/predict",
                                   {"skill": "place", "params": {"at": held["position"][:2]}})
                else:
                    table = _one(s, "table")
                    receipt = _safe_receipt(self.api, "place", {"on": table["id"]}, table, self.log)
                if receipt["withinRiskLimit"]:
                    _run(self.api, receipt, self.timeout_s)
            if any(word in message for word in PERCEPTION_FAILURES):
                _explore(self.api, self.timeout_s)
        except (MissionFailed, HttpError, OSError) as exc:
            self.log.append({"step": "recover-failed", "error": str(exc)})
            simtime_sleep(self.api, 2.0)

    def escape(self) -> bool:
        """Break out of a failure loop: back away from whatever the base is pressed against, then
        tour the room and look around so that the Ego-State is rebuilt from fresh views. Returns
        whether the robot could move at all."""
        moved = self._unstick()
        for x, y in self._viewpoints():
            receipt = call("POST", f"{self.api}/ego/predict", {"skill": "navigate", "params": {"x": x, "y": y}})
            if not receipt["withinRiskLimit"]:
                continue
            try:
                _run(self.api, receipt, self.timeout_s)
            except MissionFailed as exc:
                self.log.append({"step": "escape-tour-failed", "error": str(exc)})
                continue
            moved = True
            self._step("escape", f"自救：到 ({x:.1f}, {y:.1f}) 环顾一周，刷新对房间的感知")
            _explore(self.api, self.timeout_s)
        return moved

    def _unstick(self) -> bool:
        """Translate a short way, keeping the heading, in the free direction farthest from known objects."""
        s = _summary(self.api)
        me = s["self"]
        heading = math.radians(me["yawDeg"])
        centres = [o["position"][:2] for o in s["objects"] if o["existence"] >= 0.5 and o["id"] != me.get("holding")]
        candidates = []
        for k in range(8):
            ang = k * math.pi / 4
            spot = (me["x"] + UNSTICK_M * math.cos(ang), me["y"] + UNSTICK_M * math.sin(ang))
            clearance = min((math.dist(spot, c) for c in centres), default=math.inf)
            candidates.append((clearance, spot))
        for _, (x, y) in sorted(candidates, reverse=True):
            receipt = call("POST", f"{self.api}/ego/predict",
                           {"skill": "navigate", "params": {"x": x, "y": y, "yaw": heading}})
            if not receipt["withinRiskLimit"]:
                continue
            try:
                _run(self.api, receipt, self.timeout_s)
            except MissionFailed as exc:
                self.log.append({"step": "unstick-failed", "error": str(exc)})
                continue
            self._step("escape", f"自救：底盘后撤到 ({x:.2f}, {y:.2f})，离开贴住的障碍")
            return True
        self._step("escape", "自救：四周都没有可行的后撤方向")
        return False

    def _viewpoints(self) -> list[tuple[float, float]]:
        """Points between the room centre and its corners, from the walls in the Ego-State (or the
        furniture when no wall is known yet)."""
        s = _summary(self.api)
        walls = _objects(s, "wall")
        pts = [o["position"][:2] for o in (walls or s["objects"])]
        if not pts:
            return []
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        hx, hy = (max(xs) - min(xs)) / 4, (max(ys) - min(ys)) / 4
        return [(cx, cy), (cx - hx, cy - hy), (cx + hx, cy + hy)]


GOAL = "家务循环：整理餐桌 → 餐边柜叠放 → 抽屉对调，周而复始"
PHASES = ("table", "sideboard", "drawers")
PHASE_NAMES = {"table": "餐桌按大小排列", "sideboard": "餐边柜叠放", "drawers": "抽屉对调"}
PHASE_DETAILS = {"table": "餐具在餐桌上按占地大小排成一排", "sideboard": "碗按大小叠放，其余餐具在旁边并排",
                 "drawers": "打开两个抽屉，对调里面的物品，再关上"}
PHASE_NEEDS = {"table": {"table": 1}, "sideboard": {"sideboard": 1}, "drawers": {"sideboard": 1, "cabinet": 1, "drawer": 2}}
UNSTICK_M = 0.35
CONFIRM_CONFIDENCE = 0.6  # a role less sure than this is checked from closer before acting on it
CONFIRM_TRIES = 2
CONFIRM_DISTANCE = 1.3
CANDIDATE_MIN = 0.3  # least probability of the wanted category for an object worth a closer look;
# below it the object has more likely not been seen yet and the room is searched instead
FAILURE_HISTORY = 8
PERCEPTION_FAILURES = ("ego state", "no drawer seen", "not seen", "lost")
LIVELOCK_REPEATS = 2
LIVELOCK_WAIT_S = 180.0


def _signature(phase: str, message: str) -> str:
    """A failure with its numbers and ids stripped, to recognise the same failure repeating."""
    return phase + ":" + re.sub(r"[-\d.]+|e:\d+|rcpt-\d+", "#", message)[:200]


def _report_livelock(api: str, report: dict, on_wait=None) -> None:
    """Raise an operator alert (``report`` carries ``message`` and optionally ``goal``, ``step`` and
    ``diagnosis``) and wait without moving until it is acknowledged or a while passes."""
    try:
        alert = call("POST", f"{api}/ego/alert", {**report, "source": "mission", "level": "blocked"})
    except HttpError:
        time.sleep(LIVELOCK_WAIT_S)
        return
    if on_wait is not None:
        on_wait("operator（活锁告警待处理）")
    deadline = time.monotonic() + LIVELOCK_WAIT_S
    while time.monotonic() < deadline:
        active = call("GET", f"{api}/ego/alerts")["active"]
        if all(a["id"] != alert["id"] for a in active):
            return
        time.sleep(3.0)


def chores(api: str, rounds: int, on_step=None, on_phase=None, on_failure=None, on_wait=None,
           timeout_s: float = 180.0) -> dict:
    """Run ``rounds`` chores rounds (``<= 0`` repeats until the process stops).

    ``on_phase(round, phase)`` is called after each completed phase and ``on_failure(round, phase,
    message)`` after a failed one; a failed phase is followed by recovery and the next phase.

    A livelock is the same phase failing twice for the same reason, or a whole round of phases
    failing for any reasons. The first one triggers :meth:`Chores.escape`; another one before any
    phase completes raises an operator alert and waits for its acknowledgement. The goal, phase
    status, current step and latest difficulty are published to ``/ego/plan`` throughout.
    """
    runner = Chores(api, timeout_s, on_step)
    runner.publish(current="开局：环顾一周，建立对房间的感知")
    runner.log.append({"step": "explore", "state": _safe_explore(api, timeout_s)})
    _settle(api)  # a slow segmenter needs a few frames before there is a room to plan in
    completed = []
    streak: dict[str, tuple[str, int]] = {}
    failures = 0  # consecutive failed phases, whatever their reasons
    escaped = False
    for index in (range(rounds) if rounds > 0 else itertools.count()):
        runner.publish(round=index + 1, status={phase: "pending" for phase in PHASES})
        for phase in PHASES:
            _wait_for_control(api, on_wait)
            runner.publish(status={phase: "active"}, state="running", current=f"开始：{PHASE_NAMES[phase]}")
            try:
                {"table": runner.arrange_on_table, "sideboard": runner.stack_on_sideboard,
                 "drawers": runner.swap_drawers}[phase]()
            except (MissionFailed, HttpError, OSError) as exc:
                message = str(exc)
                runner.publish(status={phase: "failed"}, state="recovering", difficulty=f"{PHASE_NAMES[phase]}：{message}")
                if on_failure is not None:
                    on_failure(index, phase, message)
                if rounds > 0:
                    raise
                _wait_for_control(api, on_wait)
                runner.note_failure(phase, message)
                runner.recover(message)
                sig = _signature(phase, message)
                previous, count = streak.get(phase, ("", 0))
                streak[phase] = (sig, count + 1 if sig == previous else 1)
                failures += 1
                if streak[phase][1] < LIVELOCK_REPEATS and failures < len(PHASES):
                    continue
                why = (f"「{PHASE_NAMES[phase]}」连续 {streak[phase][1]} 次因同一原因失败" if streak[phase][1] >= LIVELOCK_REPEATS
                       else f"连续 {failures} 个阶段失败、没有任何进展")
                streak, failures = {}, 0
                diagnosis = runner.diagnose()
                if not escaped:
                    escaped = True
                    runner.publish(state="escaping", difficulty=f"疑似活锁：{why}，先自救。\n自我诊断：{diagnosis.text}\n最近一次：{message}")
                    try:
                        runner.escape()
                    except (MissionFailed, HttpError, OSError) as escape_exc:
                        runner.log.append({"step": "escape-failed", "error": str(escape_exc)})
                    continue
                escaped = False
                runner.publish(state="waiting-operator", difficulty=f"活锁：{why}，自救后仍无进展。\n自我诊断：{diagnosis.text}")
                _report_livelock(api, {
                    "goal": f"{GOAL}（第 {index + 1} 轮 · {PHASE_NAMES[phase]}：{PHASE_DETAILS[phase]}）",
                    "step": runner.attempt or f"开始：{PHASE_NAMES[phase]}",
                    "diagnosis": diagnosis.text,
                    "message": f"家务循环卡住：{why}，自救（后撤、巡视并重新感知）后仍无进展，已停止重试。最近失败："
                               + "；".join(f"[{PHASE_NAMES[f['phase']]}] {f['step'] or '—'} → {f['message'][:160]}"
                                          for f in runner.failures[-3:])
                               + "。请处理后点「知道了」继续。"}, on_wait)
                runner.failures.clear()
                continue
            streak.pop(phase, None)
            failures, escaped = 0, False
            runner.failures.clear()
            runner.publish(status={phase: "done"}, state="running", difficulty="")
            completed.append((index, phase))
            if on_phase is not None:
                on_phase(index, phase)
    return {"completed": completed, "log": runner.log}
