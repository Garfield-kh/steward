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


def _match_by_footprint(summary: dict, obj: dict) -> dict | None:
    """Best summary object matching ``obj``'s footprint (and preferably role/color), any distance.

    Used after a close look when :func:`_refresh` still fails because the renumbered track drifted
    beyond the local search radius (stacked bowls can be re-seen a metre away after a drive)."""
    want = _footprint_m(obj)
    if want <= 0:
        return None
    role = obj.get("role", obj.get("category"))
    color = obj.get("color")
    fp_tol = max(0.04, 0.35 * want)
    pos = obj.get("position")
    cands = []
    for o in summary.get("objects", []):
        if o["existence"] < 0.5 or abs(_footprint_m(o) - want) > fp_tol:
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


def _refresh_or_recover(api: str, obj: dict) -> dict:
    """Like :func:`_refresh`, but after a lost id: look again, retry refresh, then match by footprint
    anywhere in the summary. Stacking must not abort solely because a bowl was renumbered mid-drive."""
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
        found = _match_by_footprint(_summary(api), obj)
        if found is not None:
            return found
        raise first


def _place_at(api: str, support: dict, xy: list[float]) -> dict:
    """Place params for a fixed world spot on a support, in the support's current frame (a fitted box
    may turn half way round between views; the spot must not move with it)."""
    current = _refresh_or_recover(api, support)
    return {"on": current["id"], "offset": [round(v, 4) for v in _local(current, xy)]}


def _local(support: dict, xy: list[float]) -> tuple[float, float]:
    yaw = math.radians(support["yawDeg"])
    dx, dy = xy[0] - support["position"][0], xy[1] - support["position"][1]
    c, s = math.cos(-yaw), math.sin(-yaw)
    return c * dx - s * dy, s * dx + c * dy


def _landed(api: str, obj: dict, spot: list[float]) -> dict:
    """``obj`` as the ego state knows it after it was put down at ``spot`` (world x, y).

    With a real segmenter the track of a carried object is often rebuilt under a new id when the
    gripper lets go. Then the piece of its role nearest the spot, within 0.25 m of it and within a
    quarter of the object's width, is taken for it: the bowl just nested stands on the same spot
    as the wider bowl under it. When no such piece is seen there yet, ``obj`` is returned at
    ``spot``, for a later :func:`steward.mission._refresh` to find it by."""
    objects = _summary(api)["objects"]
    found = next((o for o in objects if o["id"] == obj["id"]), None)
    if found is not None:
        return found
    role, width = obj.get("role", obj.get("category")), _width(obj)
    there = [o for o in objects if o["existence"] >= 0.5 and role in (o.get("role"), o["category"])
             and math.dist(o["position"][:2], spot[:2]) <= 0.25 and abs(_width(o) - width) <= 0.25 * width]
    if there:
        return min(there, key=lambda o: math.dist(o["position"][:2], spot[:2]))
    return {**obj, "position": [spot[0], spot[1], obj["position"][2]]}


def _move(api: str, obj: dict, place: dict, observe: dict, log: list, timeout_s: float, attempts: int = 2) -> dict:
    """Pick ``obj`` (observing it again and retrying after a missed grasp) and place it with
    ``place`` params; return the object as it stands after the place (:func:`_landed`)."""
    for attempt in range(attempts):
        # Look before moving: from here the class-prompted fresh look at pick time sees the
        # object whole (below ~0.4 m every prompt fails), and the receipt is predicted from the
        # pose the pick really starts at. A failed approach is not a failed pick — the robot
        # just looks from where it stands.
        try:
            _approach_to_observe(api, obj, distance=0.9)
        except MissionFailed:
            pass
        # The approach takes seconds, and with a real segmenter a track can be renumbered while
        # the robot drives: the id planned from is not the id that exists now. Adopt whatever of
        # the same role stands there before asking for a pick of it.
        obj = _refresh(api, obj)
        pick = _safe_receipt(api, "pick", {"target": obj["id"]}, obj, log)
        log.append({"step": "pick-receipt", "object": obj["id"], "attempt": attempt, **receipt_brief(pick)})
        try:
            job = _run(api, pick, timeout_s)
            log.append({"step": "pick", "object": obj["id"], "jobState": job.get("state"),
                        "message": job.get("message")})
            break
        except MissionFailed as exc:
            # A missed grasp is worth a second look and a second try; an infeasible approach or a
            # blocked path is not, and repeating it only spends the round's time.
            if attempt + 1 == attempts or not any(w in str(exc) for w in ("missed", "not closing", "re-plan")):
                raise
            log.append({"step": "pick-retry", "object": obj["id"], "error": str(exc)})
            obj = _refresh(api, obj)
    # Pick can take long enough for a nested support bowl to be renumbered; rebind ``on`` before place.
    if place.get("on") is not None:
        observe = _refresh_or_recover(api, observe)
        place = {**place, "on": observe["id"]}
    receipt = _safe_receipt(api, "place", place, observe, log)
    log.append({"step": "place-receipt", "object": obj["id"], "params": place, **receipt_brief(receipt)})
    job = _run(api, receipt, timeout_s)
    log.append({"step": "place", "object": obj["id"], **place, "jobState": job.get("state"),
                "message": job.get("message")})
    simtime_sleep(api, 0.8)
    plan = receipt["plan"]
    return _landed(api, obj, plan["spot"] if "spot" in plan else plan["floor"])


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
    deadline = time.monotonic() + seconds
    s: dict = {"objects": [], "relations": []}
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


def _contents(summary: dict, holder: dict) -> list[dict]:
    """The objects resting on ``holder`` that are likely to exist (a fading fragment is not
    something to carry), the largest footprint first: they are carried to another drawer in this
    order, and a place takes the free spot nearest the middle of the open part. A large object
    put there leaves room for the small ones on both sides; two bottles in the middle leave none
    for the toy airplane."""
    inside = [o for o in summary["objects"] if o["on"] == holder["id"] and o["existence"] >= 0.5]
    return sorted(inside, key=_area, reverse=True)


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
        return _refresh(self.api, drawer)

    def _close(self, drawer: dict) -> None:
        drawer = _refresh(self.api, drawer)
        if drawer.get("openingM", 0.0) < 0.05:
            return
        self._doing(f"关上抽屉 {drawer['id']}")
        receipt = call("POST", f"{self.api}/ego/predict", {"skill": "close", "params": {"target": drawer["id"]}})
        if not receipt["withinRiskLimit"]:
            raise infeasible(f"close {drawer['id']}", receipt)
        _run(self.api, receipt, self.timeout_s)
        self._step("drawers", "关上抽屉", object=drawer["id"])

    def swap_drawers(self) -> None:
        """Exchange the contents of the sideboard and cabinet drawers. Only the pulled-out part of a
        drawer can take an object, so one drawer never holds both sets: the sideboard drawer's items
        wait on the sideboard top while the cabinet's move across."""
        board, cabinet = self._confirmed("sideboard"), self._confirmed("cabinet")
        a = self._open(board)
        a_items = _contents(_summary(self.api), a)
        b = self._open(cabinet)
        b_items = _contents(_summary(self.api), b)
        a_items = [self._transfer(item, board, f"暂放到{describe(board)} 台面上") for item in a_items]
        for item in b_items:
            self._transfer(item, a, f"从柜子的抽屉移到餐边柜的抽屉 {a['id']}")
        for item in a_items:
            self._transfer(item, b, f"从台面移到柜子的抽屉 {b['id']}")
        self._clearing("drawers", lambda: self._close(b))
        self._clearing("drawers", lambda: self._close(a))

    def _transfer(self, item: dict, dst: dict, how: str) -> dict:
        """Carry ``item`` onto or into ``dst``; return it as it stands there (:func:`_landed`)."""
        obj = _refresh(self.api, item)
        dst = _refresh(self.api, dst)
        self._doing(f"把{describe(obj)} {how}")
        landed = self._clearing("drawers", lambda o=obj, d=dst: _move(self.api, o, {"on": d["id"]}, d, self.log,
                                                                      self.timeout_s))
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
