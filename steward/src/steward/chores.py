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
    _refresh,
    _run,
    _safe_receipt,
    _wait_for_control,
    infeasible,
    simtime_sleep,
)
from steward.net import HttpError, call

TABLEWARE = ("bowl", "wine_glass", "mug", "cup")
ROW_GAP = 0.06
FLOOR_LEVEL_M = 0.15  # tableware whose base is lower than this lies on the floor
FRESH_S = 45.0  # a piece planned into today's row must have been seen this recently
TABLE_ROW_Y = 0.18
VIEW_DISTANCE = 1.1


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


def _place_at(api: str, support: dict, xy: list[float]) -> dict:
    """Place params for a fixed world spot on a support, in the support's current frame (a fitted box
    may turn half way round between views; the spot must not move with it)."""
    current = _refresh(api, support)
    return {"on": current["id"], "offset": [round(v, 4) for v in _local(current, xy)]}


def _local(support: dict, xy: list[float]) -> tuple[float, float]:
    yaw = math.radians(support["yawDeg"])
    dx, dy = xy[0] - support["position"][0], xy[1] - support["position"][1]
    c, s = math.cos(-yaw), math.sin(-yaw)
    return c * dx - s * dy, s * dx + c * dy


def _move(api: str, obj: dict, place: dict, observe: dict, log: list, timeout_s: float, attempts: int = 2) -> dict:
    """Pick ``obj`` (observing it again and retrying after a missed grasp) and place it with
    ``place`` params; return the place job."""
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
        try:
            _run(api, pick, timeout_s)
            break
        except MissionFailed as exc:
            # A missed grasp is worth a second look and a second try; an infeasible approach or a
            # blocked path is not, and repeating it only spends the round's time.
            if attempt + 1 == attempts or not any(w in str(exc) for w in ("missed", "not closing", "re-plan")):
                raise
            log.append({"step": "pick-retry", "object": obj["id"], "error": str(exc)})
            obj = _refresh(api, obj)
    receipt = _safe_receipt(api, "place", place, observe, log)
    job = _run(api, receipt, timeout_s)
    log.append({"step": "place", "object": obj["id"], **place})
    simtime_sleep(api, 0.8)
    return job


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


def _contents(summary: dict, holder: dict) -> list[str]:
    """Ids of the objects resting on ``holder`` that are likely to exist (a fading fragment is not
    something to carry)."""
    return [o["id"] for o in summary["objects"] if o["on"] == holder["id"] and o["existence"] >= 0.5]


def _drawer_of(summary: dict, furniture: dict) -> dict | None:
    drawers = _objects(summary, "drawer")
    near = [d for d in drawers if math.dist(d["position"][:2], furniture["position"][:2]) < 0.6]
    return min(near, key=lambda d: math.dist(d["position"][:2], furniture["position"][:2])) if near else None


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
        pick = _safe_receipt(self.api, "pick", {"target": obj["id"]}, obj, self.log)
        _run(self.api, pick, self.timeout_s)
        for spot in spots:
            receipt = call("POST", f"{self.api}/ego/predict", {"skill": "place", "params": {"at": spot}})
            if receipt["withinRiskLimit"]:
                _run(self.api, receipt, self.timeout_s)
                self._step(phase, f"清障完成：{describe(obj)} 放到 ({spot[0]:.2f}, {spot[1]:.2f})",
                           object=blocker["id"])
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
        self.publish(current=f"正在{text}")

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
        s = _summary(self.api)
        # A piece standing on nothing is a fragment of a wall corner, not a bowl — but when the
        # support surface itself has not been read yet, no piece has one, and dropping them all
        # would leave nothing to stack. Prefer supported pieces; fall back to all of them.
        bowls = _supported_first(_fresh_objects(s, "bowl"))
        others = _supported_first([o for cat in TABLEWARE if cat != "bowl"
                                   for o in _fresh_objects(s, cat)])
        bowls = sorted(bowls, key=_area, reverse=True)
        others = sorted(others, key=_area, reverse=True)
        widths = ([_width(bowls[0])] if bowls else []) + [_width(o) for o in others]
        offsets = row_offsets(widths, gap=0.08)
        slot = iter(offsets)
        below = None
        for bowl in bowls:
            bowl = _refresh(self.api, bowl)
            if below is None:
                offset = next(slot)
                if bowl["on"] != board["id"]:
                    spot = _world(board, _offset(board, offset, 0.0))
                    self._doing(f"把{describe(bowl)} 放到餐边柜上作叠放底座")
                    self._clearing("sideboard", lambda b=bowl, xy=spot: _move(
                        self.api, b, _place_at(self.api, board, xy), board, self.log, self.timeout_s))
                    self._step("sideboard", f"{describe(bowl)} → 餐边柜（叠放底座）", object=bowl["id"])
            else:
                below = _refresh(self.api, below)
                if bowl["on"] != below["id"]:
                    self._doing(f"把{describe(bowl)} 叠进{describe(below)}")
                    self._clearing("sideboard", lambda b=bowl, u=below: _move(
                        self.api, b, {"on": u["id"]}, u, self.log, self.timeout_s))
                    self._step("sideboard", f"{describe(bowl)} → 叠进{describe(below)}",
                               object=bowl["id"])
            below = bowl
        for obj in others:
            obj = _refresh(self.api, obj)
            offset = next(slot)
            if obj["on"] == board["id"]:
                continue
            spot = _world(board, _offset(board, offset, 0.0))
            self._doing(f"把{describe(obj)} 放到餐边柜上")
            self._clearing("sideboard", lambda o=obj, xy=spot: _move(
                self.api, o, _place_at(self.api, board, xy), board, self.log, self.timeout_s))
            self._step("sideboard", f"{describe(obj)} → 餐边柜", object=obj["id"])

    # ----------------------------------------------------------------- drawers
    def _open(self, furniture: dict) -> dict:
        return self._clearing("drawers", lambda: self._open_once(furniture))

    def _open_once(self, furniture: dict) -> dict:
        self._doing(f"走到{describe(furniture)} 前观察并打开它的抽屉")
        _view(self.api, furniture, self.timeout_s)
        s = _summary(self.api)
        drawer = _drawer_of(s, furniture)
        if drawer is None:
            raise MissionFailed(f"no drawer seen in front of {furniture['id']}")
        if drawer.get("openingM", 0.0) < 0.2:
            receipt = call("POST", f"{self.api}/ego/predict", {"skill": "open", "params": {"target": drawer["id"]}})
            if not receipt["withinRiskLimit"]:
                raise infeasible(f"open {drawer['id']}", receipt)
            _run(self.api, receipt, self.timeout_s)
            simtime_sleep(self.api, 1.5)
            self._step("drawers", f"打开{describe(furniture)} 的抽屉", object=drawer["id"])
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
        for oid in a_items:
            self._transfer(oid, board, f"暂放到{describe(board)} 台面上")
        for oid in b_items:
            self._transfer(oid, a, f"从柜子的抽屉移到餐边柜的抽屉 {a['id']}")
        for oid in a_items:
            self._transfer(oid, b, f"从台面移到柜子的抽屉 {b['id']}")
        self._clearing("drawers", lambda: self._close(b))
        self._clearing("drawers", lambda: self._close(a))

    def _transfer(self, oid: str, dst: dict, how: str) -> None:
        obj = _refresh(self.api, {"id": oid, "category": "item"})
        dst = _refresh(self.api, dst)
        self._doing(f"把{describe(obj)} {how}")
        self._clearing("drawers", lambda o=obj, d=dst: _move(self.api, o, {"on": d["id"]}, d, self.log, self.timeout_s))
        self._step("drawers", f"{describe(obj)}：{how}", object=oid, to=dst["id"])

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
