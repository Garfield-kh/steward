"""A Cobot Magic mission driven by what the robot sees, not by ground truth.

The twin's skills take positions; this module is what supplies them. It feeds sensor frames to
the Ego estimator, reads the resulting tracks, and hands the *estimated* object positions to the
skills. That is the whole point of the exercise: a pick succeeds or fails on the robot's own
perception of where the bowl is.

Phase 0 detection frames carry labelled boxes from the world; the RGB-D phase replaces them with
:meth:`WorldSim.rgbd` frames and the same estimator, which is why nothing here names a simulator
concept beyond ``frame()``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from steward.assets.cobot import ARM_MOUNT_X
from steward.ego.cobot_skills import ARM_OF_GRIPPER, CobotSkills, SkillFailed, SkillStopped
from steward.ego.estimator import EgoEstimator


@dataclass
class Sighting:
    track_id: str
    category: str
    pos: np.ndarray
    size: np.ndarray
    existence: float
    range_m: float


def perceive(est: EgoEstimator, frame: Callable[[], dict], frames: int = 8,
             pause_s: float = 0.02, min_existence: float = 0.4,
             rgbd: bool = False) -> dict[str, Sighting]:
    """Update the estimator ``frames`` times and report the best track per category.

    A track counts only if these frames saw it. Existence decays slowly, so an object that an arm
    carried out of view and set down elsewhere is still alive in the belief for many frames — and
    reporting its last position sends the next grasp to where it used to be, which closes on
    nothing while the log says the fingers are exactly on it.

    ``rgbd=True`` means ``frame()`` returns a decoded RGB-D frame (rgb, depth, anonymous masks,
    header) instead of labelled detections: the robot then has to find and measure the objects
    itself, which is the whole point of the exercise.
    """
    best: dict[str, Sighting] = {}
    for sighting in sightings(est, frame, frames=frames, pause_s=pause_s, min_existence=min_existence,
                              rgbd=rgbd):
        if sighting.category not in best or sighting.existence > best[sighting.category].existence:
            best[sighting.category] = sighting
    return best


def sightings(est: EgoEstimator, frame: Callable[[], dict], frames: int = 8,
              pause_s: float = 0.02, min_existence: float = 0.4,
              rgbd: bool = False) -> list[Sighting]:
    """Like :func:`perceive`, but one entry per object: a table with three bowls needs all three.

    Freshness uses ``observations`` (not ``last_seen``): with ``freeze_idle`` the world stamp can
    stay flat across frames, so a last_seen gate would drop every track that an earlier look()
    already fused at the same stamp.
    """
    before = {track_id: track.observations for track_id, track in est.tracks.items()}
    for _ in range(frames):
        payload = frame()
        if rgbd:
            est.update_rgbd(payload)
        else:
            est.update(payload)
        if pause_s:
            time.sleep(pause_s)
    seen: list[Sighting] = []
    cam = est.camera_pose()
    for track in est.tracks.values():
        if not track.category_counts or track.existence < min_existence:
            continue
        if before.get(track.id, 0) == track.observations:
            continue
        seen.append(Sighting(track_id=track.id, category=max(track.category_counts, key=track.category_counts.get),
                             pos=np.array(track.pos, dtype=float), size=np.array(track.size, dtype=float),
                             existence=float(track.existence),
                             range_m=float(np.linalg.norm(sighting_distance(track.pos, cam)))))
    return seen


def sighting_distance(pos, cam_pose: dict) -> np.ndarray:
    from steward.math3d import pose_compose, pose_inverse

    return np.asarray(pose_compose(pose_inverse(cam_pose), {"p": list(pos), "q": [1.0, 0.0, 0.0, 0.0]})["p"])


def bowl_onto_plate(est: EgoEstimator, skills: CobotSkills, frame: Callable[[], dict],
                    grip: str | None = None, retries: int = 2, rgbd: bool = False) -> dict:
    """Look at the table, pick up the bowl where the robot believes it is, and set it on the plate.

    One arm does the whole errand, chosen by the side of the table the bowl stands on — the same
    arm has to reach the plate too, so a layout that splits the pair across the two arms is a
    scene error the first place will expose, not something to hand off mid-air. ``grip`` overrides
    the choice. A missed grasp is fed back: the robot looks again before the next attempt, so a
    stale or wrong belief costs one retry instead of the mission.
    """
    log: list[str] = []
    for attempt in range(1, retries + 1):
        seen = perceive(est, frame, rgbd=rgbd, frames=10 if rgbd else 8)
        missing = [c for c in ("bowl", "plate") if c not in seen]
        if missing:
            log.append(f"attempt {attempt}: not seen: {', '.join(missing)}")
            continue
        bowl, plate = seen["bowl"], seen["plate"]
        arm, grip = _arm_for(float(bowl.pos[1])) if grip is None else (ARM_OF_GRIPPER[grip], grip)
        log.append(f"attempt {attempt}: bowl {_fmt(bowl.pos)} (e:{bowl.track_id}, {bowl.range_m:.2f} m), "
                   f"plate {_fmt(plate.pos)}")
        if not _arm_can_reach(skills, bowl.pos, arm):
            log.append(f"attempt {attempt}: the bowl at {_fmt(bowl.pos)} is outside the {arm} arm's workspace")
            continue
        # Set the bowl down on the plate's top surface, from the sizes the robot measured: placing
        # it at the plate's *centre* puts the bowl's base through the plate, and every round then
        # shoves the plate a couple of centimetres until the bowl hides it from the camera.
        plate_top = float(plate.pos[2]) + 0.5 * _extent(plate.size, 2, 0.01)
        target = [float(plate.pos[0]), float(plate.pos[1]),
                  plate_top + 0.5 * _extent(bowl.size, 2, 0.05) + 0.005]
        try:
            _run(skills, "pick", {"grip": grip, "arm": arm, "at": [float(v) for v in bowl.pos]}, log)
            est.mark_held(bowl.track_id, gripper=grip, at=bowl.pos)
            _run(skills, "place", {"grip": grip, "arm": arm, "at": target}, log)
            est.release_held(at=target)
        except SkillStopped:
            raise
        except SkillFailed as exc:
            log.append(f"attempt {attempt} failed: {exc}")
            if (est.gripper or {}).get("holding") or _holding(skills, grip):
                # Put the load down where it is: a retry that reaches for an object still in the
                # gripper can only fail, and leaving it held hides the failure from the operator.
                _run(skills, "open", {"gripper": grip}, log)
                est.release_held()
                log.append(f"attempt {attempt}: put the load down where it was held")
            _park(skills, grip, log)
            continue
        return {"ok": True, "attempts": attempt, "bowl": bowl.pos.tolist(), "plate": plate.pos.tolist(), "log": log}
    return {"ok": False, "log": log}


def nest_bowls(est: EgoEstimator, skills: CobotSkills, frame: Callable[[], dict],
               retries: int = 2, rgbd: bool = False) -> dict:
    """Look at the table, find the three nesting bowls, and stack them largest at the bottom.

    Each bowl is carried by the arm on its own side of the table, and every one is put down on
    the largest bowl's centre line — the one strip of the tabletop both arms can reach — so the
    smaller bowl always ends up inside the larger one. The bowls are released a few centimetres
    above the floor of the bowl below: a parallel gripper cannot lower a bowl into another bowl
    without its fingers reaching the bottom first, and a real person lets go and lets it drop in.
    """
    log: list[str] = []
    for attempt in range(1, retries + 1):
        seen = sightings(est, frame, frames=10 if rgbd else 8, rgbd=rgbd)
        bowls = sorted((s for s in seen if s.category == "bowl"), key=lambda s: -_footprint(s))
        if len(bowls) < 3:
            log.append(f"attempt {attempt}: only {len(bowls)} bowl(s) in view, the table has three")
            _park_both(skills, log)
            continue
        log.append(f"attempt {attempt}: bowls " + "; ".join(f"{_fmt(b.pos)} {_footprint(b) * 1e4:.0f} cm2"
                                                           for b in bowls))
        base, grip, placed = bowls[0], None, []
        try:
            for small in bowls[1:]:
                arm, grip = _arm_for(float(small.pos[1]))
                floor = _inner_floor(base)
                rest = floor + 0.5 * _extent(small.size, 2, 0.05)
                target = [float(base.pos[0]), float(base.pos[1]), rest + _NEST_DROP]
                _run(skills, "pick", {"arm": arm, "gripper": grip, "at": [float(v) for v in small.pos]}, log)
                # The robot knows what it just picked up: the load follows that arm from here on,
                # so the Ego State shows the bowl in the hand instead of leaving it on the table.
                est.mark_held(small.track_id, gripper=grip, at=small.pos)
                _run(skills, "place", {"arm": arm, "gripper": grip, "at": target}, log)
                est.release_held(at=target)
                placed.append(small.track_id)
                # After it drops, the bowl rests on the floor it was released over.
                base = Sighting(track_id=small.track_id, category=small.category,
                                pos=np.array([target[0], target[1], rest]), size=small.size,
                                existence=small.existence, range_m=small.range_m)
        except SkillStopped:
            raise
        except SkillFailed as exc:
            log.append(f"attempt {attempt} failed: {exc}")
            if grip is not None and _holding(skills, grip):
                _run(skills, "open", {"gripper": grip}, log)
                est.release_held()
                log.append(f"attempt {attempt}: put the bowl down where it was held")
            _park_both(skills, log)
            continue
        return {"ok": True, "attempts": attempt, "base": bowls[0].track_id, "nested": placed,
                "top": round(float(base.pos[2]), 4), "log": log}
    return {"ok": False, "log": log}


def carry_to_station(est: EgoEstimator, skills: CobotSkills, frame: Callable[[], dict],
                     drive: dict, place: list[float], lift: float | None = None,
                     grip: str = "L_link6", retries: int = 2, rgbd: bool = False) -> dict:
    """Take the bowl from this station and put it down at another one the robot drives to.

    What it moves is perceived -- the bowl's position comes from the robot's own look at the
    table it starts at. Where it goes is known: ``drive`` is the route (waypoints and final
    heading) and ``place`` the world point on the destination's surface, the way a deployed
    robot knows its furniture and measures what stands on it. ``lift`` raises the torso on
    arrival, for a destination higher than the table it left.

    The load is parked before the drive: an arm held out over the table would sweep the room
    for two metres, and folding it is what a person does with a tray.
    """
    log: list[str] = []
    arm = ARM_OF_GRIPPER.get(grip)
    if arm is None:
        return {"ok": False, "log": [f"unknown gripper {grip!r}"]}
    for attempt in range(1, retries + 1):
        seen = perceive(est, frame, rgbd=rgbd, frames=10 if rgbd else 8)
        if "bowl" not in seen:
            log.append(f"attempt {attempt}: no bowl in view")
            continue
        bowl = seen["bowl"]
        log.append(f"attempt {attempt}: bowl {_fmt(bowl.pos)} (e:{bowl.track_id}, {bowl.range_m:.2f} m)")
        try:
            _run(skills, "pick", {"arm": arm, "gripper": grip, "at": [float(v) for v in bowl.pos]}, log)
            est.mark_held(bowl.track_id, gripper=grip, at=bowl.pos)
            _park(skills, grip, log)
            _run(skills, "navigate", {"at": [float(v) for v in drive["at"]],
                                      "path": [[float(p[0]), float(p[1])] for p in drive.get("path") or []]},
                 log)
            if lift is not None:
                _run(skills, "lift", {"height": float(lift)}, log)
            _run(skills, "place", {"arm": arm, "gripper": grip, "at": [float(v) for v in place]}, log)
            est.release_held(at=[float(v) for v in place])
        except SkillStopped:
            raise
        except SkillFailed as exc:
            log.append(f"attempt {attempt} failed: {exc}")
            if _holding(skills, grip):
                _run(skills, "open", {"gripper": grip}, log)
                est.release_held()
                log.append(f"attempt {attempt}: put the bowl down where it was held")
            _park_both(skills, log)
            continue
        return {"ok": True, "attempts": attempt, "bowl": bowl.pos.tolist(),
                "station": [float(v) for v in place], "base": skills.tcp_base(), "log": log}
    return {"ok": False, "log": log}


# A nesting bowl's inner floor: the 8 mm base disc plus the 2 mm the wall segments are lifted by.
_BOWL_FLOOR = 0.010
# Released this far above the floor of the bowl below; see the docstring of nest_bowls.
_NEST_DROP = 0.030


def _inner_floor(base: Sighting) -> float:
    """Where the next bowl comes to rest: the base bowl's own floor, measured from its pose."""
    return float(base.pos[2]) - 0.5 * _extent(base.size, 2, 0.06) + _BOWL_FLOOR


def _footprint(sighting: Sighting) -> float:
    return float(sighting.size[0]) * float(sighting.size[1]) if len(sighting.size) > 1 else 0.0


def _arm_for(y: float) -> tuple[str, str]:
    """The arm that works the side of the table ``y`` is on."""
    return ("L", "L_link6") if y >= 0.0 else ("R", "R_link6")


def _park_both(skills: CobotSkills, log: list[str]) -> None:
    for grip in ("L_link6", "R_link6"):
        _park(skills, grip, log)


def _park(skills: CobotSkills, grip: str, log: list[str]) -> None:
    """Take the arm out of the head cameras' view before the next look.

    A failed attempt leaves the hand where it gave up — typically over the plate it was lowering
    onto — so the camera sees a forearm instead of the tabletop. Parking first is what makes the
    retry's look a real one rather than a decision on a stale belief.
    """
    arm = ARM_OF_GRIPPER.get(grip, "L")
    try:
        _run(skills, "park", {"arm": arm}, log)
    except SkillFailed as exc:
        log.append(f"park failed: {exc}")


def _run(skills: CobotSkills, name: str, plan: dict, log: list[str]) -> None:
    job = skills.start(name, {"id": f"mission-{name}", "plan": plan})
    while job.state == "running":
        time.sleep(0.02)
    log.extend(f"{name}: {line}" for line in job.log)
    if job.state != "succeeded":
        raise SkillFailed(job.message or job.state)


def _extent(size, axis: int, fallback: float) -> float:
    """One dimension of a measured box, with a fallback when the estimate did not carry sizes."""
    try:
        value = float(size[axis])
    except (IndexError, TypeError, ValueError):
        return fallback
    return value if value > 1e-4 else fallback


def _arm_can_reach(skills: CobotSkills, pos: np.ndarray, arm: str = "L") -> bool:
    """A crude gate before commanding: is this anywhere near the arm's measured workspace? Judged
    in the robot's frame, ahead of the arm mounts."""
    o = skills._state().get("odometry") or {}
    bx, by, byaw = (float(o.get(k, 0.0)) for k in ("x", "y", "yaw"))
    dx, dy = float(pos[0]) - bx, float(pos[1]) - by
    ahead = math.cos(byaw) * dx + math.sin(byaw) * dy - ARM_MOUNT_X
    y = (-math.sin(byaw) * dx + math.cos(byaw) * dy) * (1.0 if arm == "L" else -1.0)
    return 0.10 <= ahead <= 0.60 and 0.02 <= y <= 0.55 and 0.55 <= float(pos[2]) <= 1.05


def _holding(skills: CobotSkills, grip: str) -> bool:
    st = skills._state()
    return bool((st.get("grippers") or {}).get(grip, {}).get("holding"))


def _fmt(p) -> str:
    return "(" + ", ".join(f"{float(v):.3f}" for v in p) + ")"


# Board body-frame hole centres (same layout as screw_board / screw_board_goal).
# +y first: left arm inserts more reliably on that side; right arm still reaches y≈±0.05 to drive.
_BOARD_HOLES = ((0.07, 0.05), (-0.07, 0.05), (0.07, -0.05), (-0.07, -0.05))


def _free_screws(board: Sighting, screws: list[Sighting], driven: list[str]) -> list[Sighting]:
    """Screws still beside the board (outside its footprint), not already driven.

    Table screws rest near table height — below the board top — so a z > board_top filter
    wrongly drops every free screw. Lateral clearance from the board AABB is the gate.
    """
    half_w = 0.5 * _extent(board.size, 0, 0.28) + 0.03
    half_d = 0.5 * _extent(board.size, 1, 0.20) + 0.03
    bx, by = float(board.pos[0]), float(board.pos[1])
    done = set(driven)
    free: list[Sighting] = []
    for s in screws:
        if s.track_id in done:
            continue
        if abs(float(s.pos[0]) - bx) <= half_w and abs(float(s.pos[1]) - by) <= half_d:
            continue  # on / over the board → seated or in a hole
        free.append(s)
    return free


def drive_screws(est: EgoEstimator, skills: CobotSkills, frame: Callable[[], dict],
                 retries: int = 2, rgbd: bool = False, max_screws: int = 4) -> dict:
    """Dual-arm screw-board mission: L inserts each screw, R drives it, both use wrist cameras.

    Left arm picks a free screw, refines the empty hole with ``L_wrist_camera``, then
    ``insert_screw_into_hole``. Right arm picks the electric screwdriver once, refines the seated
    screw head with ``R_wrist_camera``, then ``drive_screw`` (driver stays in the right hand).
    Perception supplies positions; wrist refine nudges planar aim from a closer look.
    """
    log: list[str] = []
    driven: list[str] = []
    driver_grip = "R_link6"
    screw_grip = "L_link6"
    holding_driver = False

    def look() -> dict[str, Sighting]:
        return perceive(est, frame, frames=8 if not rgbd else 10, rgbd=rgbd)

    def hole_world(board: Sighting, index: int) -> list[float]:
        ox, oy = _BOARD_HOLES[index]
        # Board yaw is not on Sighting; treat as axis-aligned for the twin's table layout.
        return [float(board.pos[0]) + ox, float(board.pos[1]) + oy,
                float(board.pos[2]) + 0.5 * _extent(board.size, 2, 0.018)]

    for attempt in range(1, retries + 1):
        try:
            seen = look()
            if "screw_board" not in seen:
                log.append(f"attempt {attempt}: no screw_board in view")
                _park_both(skills, log)
                continue
            board = seen["screw_board"]
            screws = [s for s in sightings(est, frame, frames=4, rgbd=rgbd)
                      if s.category == "toy_screw"]
            free = _free_screws(board, screws, driven)
            if not free and len(driven) >= max_screws:
                return {"ok": True, "driven": driven, "log": log}
            if not free:
                if not screws:
                    log.append(f"attempt {attempt}: no toy_screw in view")
                else:
                    log.append(f"attempt {attempt}: no free screws beside the board "
                               f"({len(screws)} seen, {len(driven)} driven)")
                break

            for hole_i in range(min(max_screws, len(_BOARD_HOLES))):
                if len(driven) >= max_screws:
                    break
                seen = look()
                board = seen.get("screw_board") or board
                screws = [s for s in sightings(est, frame, frames=4, rgbd=rgbd)
                          if s.category == "toy_screw"]
                free = _free_screws(board, screws, driven)
                if not free:
                    break
                screw = min(free, key=lambda s: float(s.range_m))
                hole_at = hole_world(board, hole_i)
                seat_z = hole_at[2] + 0.02
                seat = [hole_at[0], hole_at[1], seat_z]

                # --- L: pick screw and insert with left wrist refine ---
                _run(skills, "pick", {"arm": "L", "gripper": screw_grip,
                                      "at": [float(v) for v in screw.pos]}, log)
                est.mark_held(screw.track_id, gripper=screw_grip, at=screw.pos)
                # Prefer holes on L's side of the board (positive local y) when still free.
                look()
                board2 = perceive(est, frame, frames=4, rgbd=rgbd).get("screw_board", board)
                refine_hole = hole_world(board2, hole_i)
                refine_seat = [refine_hole[0], refine_hole[1], seat_z]
                # Tip-down + near-vertical tool: a tilted insert lays the shaft in the hole and
                # the screw falls over on release (goal upright=False).
                _run(skills, "insert_screw_into_hole", {
                    "arm": "L", "gripper": screw_grip, "at": refine_seat,
                    "refine": True, "wristCamera": "L_wrist_camera", "refineAt": refine_seat,
                    "releaseLingerS": 0.4,
                }, log)
                est.release_held(at=refine_seat)
                log.append(f"hole_{hole_i}: inserted tip-down {screw.track_id}")

                # --- R: pick driver once, refine on screw head, drive ---
                if not holding_driver:
                    drivers = [s for s in sightings(est, frame, frames=4, rgbd=rgbd)
                               if s.category == "electric_screwdriver"]
                    if not drivers and "electric_screwdriver" in (seen := look()):
                        drivers = [seen["electric_screwdriver"]]
                    if not drivers:
                        raise SkillFailed("no electric_screwdriver in view to drive with")
                    driver = min(drivers, key=lambda s: float(s.range_m))
                    _run(skills, "pick", {"arm": "R", "gripper": driver_grip,
                                          "at": [float(v) for v in driver.pos]}, log)
                    est.mark_held(driver.track_id, gripper=driver_grip, at=driver.pos)
                    holding_driver = True
                    log.append(f"holding driver {driver.track_id} in R")

                # Keep R's hover low: high tool-down poses over the midline exceed Piper reach.
                head_z = refine_seat[2] + 0.03
                head = [refine_seat[0], refine_seat[1], head_z]
                look()
                # Closest screw near the hole is the one just seated.
                seated = [s for s in sightings(est, frame, frames=4, rgbd=rgbd)
                          if s.category == "toy_screw"
                          and math.hypot(float(s.pos[0]) - head[0], float(s.pos[1]) - head[1]) < 0.06]
                if seated:
                    s0 = min(seated, key=lambda s: math.hypot(float(s.pos[0]) - head[0],
                                                              float(s.pos[1]) - head[1]))
                    refine_head = [float(s0.pos[0]), float(s0.pos[1]), head_z]
                else:
                    refine_head = head
                press = [refine_head[0], refine_head[1], refine_seat[2] + 0.008]
                _run(skills, "drive_screw", {
                    "arm": "R", "gripper": driver_grip, "at": head, "pressAt": press,
                    "refine": True, "wristCamera": "R_wrist_camera", "refineAt": refine_head,
                    "driveS": 1.5,
                }, log)
                driven.append(screw.track_id)
                log.append(f"hole_{hole_i}: drove {screw.track_id}")
                # Park L between screws; keep R holding the driver out of the way.
                _park(skills, screw_grip, log)

            _park_both(skills, log)
            return {"ok": len(driven) > 0, "driven": driven, "log": log,
                    "holdingDriver": holding_driver}
        except SkillStopped:
            raise
        except SkillFailed as exc:
            log.append(f"attempt {attempt} failed: {exc}")
            for grip in (screw_grip, driver_grip):
                if _holding(skills, grip):
                    _run(skills, "open", {"gripper": grip}, log)
                    if grip == screw_grip:
                        est.release_held()
                    if grip == driver_grip:
                        holding_driver = False
                        est.release_held()
            _park_both(skills, log)
    return {"ok": False, "driven": driven, "log": log}
