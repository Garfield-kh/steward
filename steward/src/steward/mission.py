"""Scripted Phase 0 mission over the Ego API: put the cup on the table.

The script uses exactly the operations the Ego Harness tools expose (explore, summary, predict,
run, wait), so it doubles as an end-to-end check of the Ego API without a language model.
"""

from __future__ import annotations

import itertools
import math
import time

from steward.net import HttpError, call


class MissionFailed(RuntimeError):
    pass


class Blocked(MissionFailed):
    """A skill is infeasible because entities are in the way; ``blockers`` comes from the receipt
    (id, category, movable, parkAt)."""

    def __init__(self, message: str, blockers: list[dict]):
        super().__init__(message)
        self.blockers = blockers


def infeasible(skill: str, receipt: dict) -> MissionFailed:
    """The error for an infeasible receipt: :class:`Blocked` when movable entities are in the way."""
    message = f"{skill} infeasible: {receipt['issues']}"
    if any(b.get("movable") for b in receipt.get("blockers", [])):
        return Blocked(message, receipt["blockers"])
    return MissionFailed(message)


def _find(summary: dict, category: str, min_probability: float = 0.25) -> dict | None:
    """The most likely object of a category, using the full category distribution so that an
    object briefly mislabelled by perception is still found."""
    cands = [o for o in summary["objects"]
             if o["categories"].get(category, 0.0) >= min_probability and o["existence"] >= 0.5]
    return max(cands, key=lambda o: (o["categories"][category], -o["uncertaintyM"])) if cands else None


def _refresh(api: str, obj: dict) -> dict:
    """The same object, as the ego state knows it now.

    The estimator renumbers a track whenever it merges or rebuilds it, so an id held across a
    navigation or a prediction can name something that no longer exists. Take over whatever of
    the same role (or of the same size, for furniture whose label went missing) stands where this
    one stood."""
    objects = call("GET", f"{api}/ego/summary")["objects"]
    found = next((o for o in objects if o["id"] == obj["id"]), None)
    if found is not None:
        return found
    pos = obj.get("position")
    if pos is not None:
        role = obj.get("role", obj.get("category"))
        size = obj.get("sizeM") or [0.0, 0.0]
        reach = 0.6 if max(size[:2]) >= 0.5 else 0.25
        same = [o for o in objects if o["existence"] >= 0.5
                and role in (o.get("role"), o["category"])
                and math.dist(o["position"][:2], pos[:2]) <= reach]
        if same:
            return min(same, key=lambda o: math.dist(o["position"][:2], pos[:2]))
        if max(size[:2]) >= 0.5:
            # The label can go missing before the thing does: a table seen only as its top slab
            # is re-read as "unknown" in the same place with the same footprint.
            there = [o for o in objects if o["existence"] >= 0.5
                     and max((o.get("extentM") or o["sizeM"])[:2]) >= 0.4
                     and math.dist(o["position"][:2], pos[:2]) <= reach]
            if there:
                return min(there, key=lambda o: math.dist(o["position"][:2], pos[:2]))
    raise MissionFailed(f"{obj['id']} ({obj['category']}) is no longer in the ego state")


def simtime_sleep(api: str, seconds: float) -> None:
    """Let ``seconds`` of world time pass, e.g. for perception to settle after a motion."""
    call("POST", f"{api}/clock/sleep", {"seconds": seconds}, timeout=3600.0)


def _predict_until_safe(api: str, skill: str, params: dict, attempts: int = 5, settle_s: float = 1.0) -> dict:
    for _ in range(attempts):
        receipt = call("POST", f"{api}/ego/predict", {"skill": skill, "params": params})
        if receipt["withinRiskLimit"]:
            return receipt
        if not receipt["feasible"]:
            raise infeasible(skill, receipt)
        simtime_sleep(api, settle_s)
    raise MissionFailed(f"{skill} stayed above the risk limit: {receipt['risk']} {receipt['issues']}")


def _approach_to_observe(api: str, target: dict, distance: float = 1.1, timeout_s: float = 120.0) -> dict | None:
    """Drive to a floor pose about ``distance`` from the target, facing it, to observe it closely."""
    me = call("GET", f"{api}/ego/summary")["self"]
    tx, ty = target["position"][:2]
    base = math.atan2(me["y"] - ty, me["x"] - tx)
    for offset in (0.0, 0.5, -0.5, 1.0, -1.0, 1.6, -1.6):
        ang = base + offset
        x, y = tx + distance * math.cos(ang), ty + distance * math.sin(ang)
        receipt = call("POST", f"{api}/ego/predict",
                       {"skill": "navigate", "params": {"x": x, "y": y, "yaw": math.atan2(ty - y, tx - x)}})
        if receipt["withinRiskLimit"]:
            _run(api, receipt, timeout_s)
            simtime_sleep(api, 1.5)
            return receipt
    return None


def _run(api: str, receipt: dict, timeout_s: float) -> dict:
    job = call("POST", f"{api}/skill/run", {"receipt": receipt["id"]})
    job = call("POST", f"{api}/skill/wait", {"job": job["id"], "timeoutS": timeout_s}, timeout=timeout_s + 10)
    if job["state"] != "succeeded":
        raise MissionFailed(f"{receipt['skill']} {job['state']}: {job['message']}")
    return job


def _explore(api: str, timeout_s: float) -> dict:
    job = call("POST", f"{api}/ego/explore")
    return call("POST", f"{api}/skill/wait", {"job": job["id"], "timeoutS": timeout_s}, timeout=timeout_s + 10)


def _safe_receipt(api: str, skill: str, params: dict, observe: dict, log: list) -> dict:
    """Predict a skill; when it is above the risk limit or needs a better view, move closer to
    ``observe`` (the object or surface concerned) and predict again until it is safe."""
    first = call("POST", f"{api}/ego/predict", {"skill": skill, "params": params})
    if not first["feasible"] and params.get("target") and any("unknown target" in i for i in first["issues"]):
        # The id was renumbered between choosing the object and planning the pick.
        fresh = _refresh(api, observe)
        params = {**params, "target": fresh["id"]}
        first = call("POST", f"{api}/ego/predict", {"skill": skill, "params": params})
    if first["withinRiskLimit"]:
        return first
    if not first["feasible"] and any(b.get("movable") for b in first.get("blockers", [])):
        raise infeasible(skill, first)
    moved = _approach_to_observe(api, observe)
    log.append({"step": "observe-closer", "target": observe["id"], "navigated": moved is not None,
                "issues": first["issues"]})
    return _predict_until_safe(api, skill, params)


def _move(api: str, obj: dict, support: dict, log: list, timeout_s: float) -> None:
    try:
        _approach_to_observe(api, obj, distance=0.9)  # look before moving: fresh look needs the distance
    except MissionFailed:
        pass
    pick = _safe_receipt(api, "pick", {"target": obj["id"]}, obj, log)
    log.append({"step": "pick", "target": obj["id"], "risk": pick["risk"], "message": _run(api, pick, timeout_s)["message"]})
    place = _safe_receipt(api, "place", {"on": support["id"]}, support, log)
    log.append({"step": "place", "on": support["id"], "risk": place["risk"], "spot": place["plan"]["spot"],
                "message": _run(api, place, timeout_s)["message"]})
    simtime_sleep(api, 1.0)
    placed = next((o for o in call("GET", f"{api}/ego/summary")["objects"] if o["id"] == obj["id"]), None)
    if placed is None or placed["on"] != support["id"]:
        where = "lost" if placed is None else f"on {placed['on']} at {placed['position']}"
        raise MissionFailed(f"after placing, {obj['id']} is not on {support['id']} in the ego state ({where})")


def cup_to_table(api: str, timeout_s: float = 180.0) -> dict:
    log = [{"step": "explore", "state": _explore(api, timeout_s)["state"]}]
    summary = call("GET", f"{api}/ego/summary")
    cup, table = _find(summary, "cup"), _find(summary, "table")
    if cup is None or table is None:
        raise MissionFailed(f"after exploring, cup={cup is not None} table={table is not None}")
    log.append({"step": "found", "cup": cup["id"], "table": table["id"], "cupUncertaintyM": cup["uncertaintyM"]})
    _move(api, cup, table, log, timeout_s)
    return {"log": log, "cup": cup["id"], "table": table["id"], "summary": call("GET", f"{api}/ego/summary")}


def shuttle(api: str, steps: int, supports: tuple[str, ...] = ("table", "cabinet", "sideboard"),
            objects: tuple[str, ...] = ("cup", "bowl"), on_step=None, timeout_s: float = 180.0,
            on_failure=None, on_wait=None) -> dict:
    """Carry objects around the cycle of support categories, one move per step; ``steps <= 0``
    repeats until the caller stops the process.

    With fewer objects than supports one support is always free: each step moves the object
    whose next support in the cycle is empty, so no support ever holds two carried objects.
    An object found anywhere else (for example moved by a world edit) is brought back to a free
    support. Objects and supports are located in the Ego-State by category at every step.
    ``on_step(index, record)`` is called after each completed move. A bounded run stops at the
    first failed step; an endless run reports it through ``on_failure(index, message)``, puts
    down anything still held, looks around again and continues, so the world may be edited
    while it runs. Before each step the shuttle waits while another client (an operator task)
    holds exclusive control of the robot, reporting ``on_wait(owner)`` once per pause.
    """
    if len(objects) >= len(supports):
        raise ValueError("the shuttle needs more supports than objects")
    log = [{"step": "explore", "state": _explore(api, timeout_s)["state"]}]
    moves = []
    for index in (range(steps) if steps > 0 else itertools.count()):
        _wait_for_control(api, on_wait)
        if steps > 0:
            moves.append(_shuttle_step(api, index, supports, objects, log, timeout_s))
        else:
            try:
                moves.append(_shuttle_step(api, index, supports, objects, log, timeout_s))
            except (MissionFailed, HttpError) as exc:
                log.append({"step": index + 1, "failed": str(exc)})
                if on_failure is not None:
                    on_failure(index, str(exc))
                _wait_for_control(api, on_wait)
                _recover(api, supports, objects, log, timeout_s)
                continue
        if on_step is not None:
            on_step(index, moves[-1])
    return {"log": log, "moves": moves, "summary": call("GET", f"{api}/ego/summary")}


def _wait_for_control(api: str, on_wait=None, poll_s: float = 1.0) -> None:
    owner = call("GET", f"{api}/ego/control")["owner"]
    if owner is None:
        return
    if on_wait is not None:
        on_wait(owner)
    while call("GET", f"{api}/ego/control")["owner"] is not None:
        time.sleep(poll_s)


def _recover(api: str, supports: tuple[str, ...], objects: tuple[str, ...], log: list, timeout_s: float) -> None:
    """After a failed step: put a held object on a free support, then look around again."""
    try:
        summary = call("GET", f"{api}/ego/summary")
        held = summary["self"].get("holding")
        if held is not None:
            occupied = {o["on"] for o in summary["objects"] if o["on"] and o["id"] != held
                        and any(o["categories"].get(c, 0.0) >= 0.25 for c in objects)}
            free = [s for s in (_find(summary, c) for c in supports) if s is not None and s["id"] not in occupied]
            if free:
                place = _safe_receipt(api, "place", {"on": free[0]["id"]}, free[0], log)
                _run(api, place, timeout_s)
                log.append({"step": "recover-place", "on": free[0]["id"]})
        log.append({"step": "recover-explore", "state": _explore(api, timeout_s)["state"]})
    except (MissionFailed, HttpError) as exc:
        log.append({"step": "recover-failed", "error": str(exc)})
        simtime_sleep(api, 2.0)


def _shuttle_step(api: str, index: int, supports: tuple[str, ...], objects: tuple[str, ...], log: list,
                  timeout_s: float) -> dict:
    """One shuttle move chosen from the current Ego-State; raises :class:`MissionFailed` on failure."""
    summary = call("GET", f"{api}/ego/summary")
    found_supports = {cat: _find(summary, cat) for cat in supports}
    found_objects = {cat: _find(summary, cat) for cat in objects}
    missing = [c for c, v in {**found_supports, **found_objects}.items() if v is None]
    if missing:
        log.append({"step": "explore", "missing": missing, "state": _explore(api, timeout_s)["state"]})
        summary = call("GET", f"{api}/ego/summary")
        found_supports = {cat: _find(summary, cat) for cat in supports}
        found_objects = {cat: _find(summary, cat) for cat in objects}
        missing = [c for c, v in {**found_supports, **found_objects}.items() if v is None]
        if missing:
            raise MissionFailed(f"step {index + 1}: cannot find {missing} in the ego state")
    support_cat = {v["id"]: cat for cat, v in found_supports.items()}
    location = {cat: support_cat.get(obj["on"]) for cat, obj in found_objects.items()}
    for cat in [c for c, loc in location.items() if loc is None]:
        _approach_to_observe(api, found_objects[cat])
        refreshed = _find(call("GET", f"{api}/ego/summary"), cat)
        if refreshed is not None:
            found_objects[cat] = refreshed
            location[cat] = support_cat.get(refreshed["on"])
        log.append({"step": "observe-closer", "target": cat, "foundOn": location[cat]})
    occupied = {loc for loc in location.values() if loc is not None}
    choice = None
    for cat in objects:
        here = location[cat]
        if here is None:
            continue
        nxt = supports[(supports.index(here) + 1) % len(supports)]
        if nxt not in occupied:
            choice = (cat, here, nxt)
            break
    if choice is None:
        free = [s for s in supports if s not in occupied]
        stray = next((c for c in objects if location[c] is None), None)
        if free and stray is not None:
            choice = (stray, None, free[0])
    if choice is None:
        raise MissionFailed(f"step {index + 1}: no object can move to a free support; locations {location}")
    cat, here, nxt = choice
    _move(api, found_objects[cat], found_supports[nxt], log, timeout_s)
    record = {"step": index + 1, "object": cat, "from": here, "to": nxt, "objectId": found_objects[cat]["id"],
              "supportId": found_supports[nxt]["id"]}
    return record
