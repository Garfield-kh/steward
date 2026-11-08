"""Skill execution: turn a Prediction Receipt's plan into timed actuator commands.

Motion uses only the robot's own estimates (odometry, encoders, ES). Base motion ramps a
position setpoint along each path segment at a bounded speed in simulation time taken from sensor
stamps; joints settle to targets with one sag correction. Control ticks follow the world clock when
one is given, so a skill moves the same at every simulation speed. A stop request interrupts
the current skill between control ticks and holds the current pose.
"""

from __future__ import annotations

import itertools
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from steward.math3d import wrap_angle

if TYPE_CHECKING:
    from steward.ego.runtime import RobotLink


class SkillStopped(Exception):
    pass


class SkillFailed(Exception):
    pass


class OutOfReach(SkillFailed):
    """The arm cannot put its gripper where a command asks; the command was not sent."""


@dataclass
class MotionConfig:
    linear_speed: float = 1.4
    angular_speed: float = 3.6
    look_angular_speed: float = 1.0
    """Turn rate of a look-around while perception keeps keyframes. At the perception rate this
    is one keyframe step, so a long table that is fully inside the head camera for only ~0.2 rad
    is sampled. The faster ``angular_speed`` steps ~0.36 rad and can land only on the truncated
    views, which do not start a track. A look-around that fuses every frame keeps
    ``angular_speed``."""
    gentle_speed: float = 0.3
    """Base speed while carrying a container with its contents (plans marked ``gentle``)."""
    gentle_turn_speed: float = 0.5
    gentle_lift_speed: float = 0.1
    """Lift and reach speed with a gentle plan: a container accelerated faster leaves its contents behind."""
    arm_speed: float | None = None
    """Lift and reach speed of an arm whose joints, servoed straight to a solved posture, carry the
    grasp point off the line between the two postures (an arm driven by inverse kinematics sweeps its
    wrist below that line, through a drawer front): the move is ramped so the grasp point stays on the
    line. None: the joint goes straight to its target (a slide moves along the line anyway)."""
    wrist_pitch_speed: float = 1.0
    """Rate at which a tool tilts forward (``wrist_pitch``) on its way out to a grasp, and back after."""
    drawer_speed: float = 0.35
    """Base speed while pulling or pushing a drawer by its handle: faster, its contents tip over."""
    position_tol: float = 0.03
    yaw_tol: float = 0.04
    joint_tol: float = 0.01
    tick_s: float = 0.04
    retract_reach: float = 0.10
    # After a failure the arm first rises this far, then turns the wrist upright in place, then draws
    # back: drawn back at once, a box tipped over a table caught its edge and turned the table over.
    failure_raise: float = 0.08
    reach_max: float = 0.70  # the wrist slide's upper joint limit; asking beyond it only fails
    stow_lift: float = 0.60
    dock_tol: float = 0.06
    dock_position_tol: float = 0.01
    """How near its docking pose the base settles before the arm moves (``position_tol`` leaves it
    2-3 cm short, and the arm reaches that much farther than planned)."""
    dock_yaw_tol: float = 0.12
    align_min: float = 0.01
    align_max: float = 0.15
    stall_s: float = 2.0
    # Offsets from the estimate at which a pick closes the gripper before giving up: the first
    # probe is at the estimate itself, then forward/back, then left/right, then other heights.
    grasp_probe_xy: tuple[float, ...] = (0.035, -0.035)
    grasp_probe_z: tuple[float, ...] = (-0.025, 0.025)
    settle_s: float = 0.4
    """Robot time after a joint command at which a joint still short of its target (sagging under
    a load) gets one corrected setpoint."""
    grasp_wait_s: float = 0.1
    release_wait_s: float = 0.15
    release_joint_tol: float = 0.002
    """How near its setpoint each arm joint comes before a place lets go: ``joint_tol`` ends a move
    up to a centimetre short, and the load is put down where the gripper is."""
    release_settle_s: float = 0.6
    """Most robot time a place waits for its joints to come within ``release_joint_tol``; then it lets go anyway."""
    handle_release_wait_s: float = 0.08
    look_dwell_s: float = 0.6
    wrist_turn_s: float = 0.6
    """Robot time to turn the wrist yaw to a placing heading."""
    pour_tip_s: float = 2.5
    """Robot time to tip a container over and to set it upright again: slow, so the contents slide
    out rather than fly."""


@dataclass
class Job:
    id: str
    skill: str
    receipt: str
    state: str = "running"
    message: str = ""
    started: float = 0.0
    finished: float | None = None
    log: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"id": self.id, "skill": self.skill, "receipt": self.receipt, "state": self.state,
                "message": self.message, "started": self.started, "finished": self.finished, "log": self.log[-20:]}


class _Stall:
    """Detects a motion that stopped making progress: the error has not improved by ``step``
    within ``window`` seconds of robot time and over at least ``samples`` fresh state updates
    (at a high simulation speed one update can span a large part of the window)."""

    def __init__(self, t0: float, window: float, step: float, samples: int = 8):
        self.window, self.step, self.samples = window, step, samples
        self.best = math.inf
        self.since = t0
        self.last_stamp = t0
        self.fresh = 0

    def stuck(self, stamp: float, err: float) -> bool:
        if stamp != self.last_stamp:
            self.last_stamp = stamp
            self.fresh += 1
        if err < self.best - self.step:
            self.best, self.since, self.fresh = err, stamp, 0
        return stamp - self.since > self.window and self.fresh >= self.samples


class SkillExecutor:
    """Runs at most one skill at a time in a worker thread."""

    def __init__(self, command: Callable[[dict], None], state: Callable[[], dict], mark_held: Callable[[str], str],
                 config: MotionConfig | None = None, shift: Callable[[str, list[float]], None] | None = None,
                 on_error: Callable[[str], None] | None = None, on_finish: Callable[[Job], None] | None = None,
                 on_begin: Callable[[str, dict], None] | None = None,
                 on_release: Callable[[str, list[float]], None] | None = None,
                 clock: RobotLink | None = None):
        """``clock`` is the robot link whose world clock times the control ticks (one hold per job,
        ``skill/<job id>``); without it a tick sleeps ``tick_s`` of wall time."""
        self._on_error = on_error
        self._clock = clock
        self._now = 0.0
        self._hold = ""
        self._on_finish = on_finish
        self._on_begin = on_begin
        self._on_release = on_release
        self._command = command
        self._state = state
        self._mark_held = mark_held
        self._shift = shift
        self.config = config or MotionConfig()
        self._ids = itertools.count(1)
        self._gentle = False  # the running plan moves a container with its contents
        self._setpoints: dict[str, float] = {}  # the last target each joint was moved to
        self._look = False  # the running skill is a look-around
        self._keyframed = False  # that look-around turns at look_angular_speed only while perception keeps keyframes
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.jobs: dict[str, Job] = {}
        self.current: Job | None = None

    # ------------------------------------------------------------------ public
    def start(self, skill: str, receipt: dict) -> Job:
        with self._lock:
            if self.current is not None and self.current.state == "running":
                raise RuntimeError(f"skill {self.current.skill} ({self.current.id}) is still running")
            job = Job(id=f"job-{next(self._ids)}", skill=skill, receipt=receipt["id"], started=self._state()["stamp"])
            self.jobs[job.id] = job
            self.current = job
            self._stop.clear()
        if self._clock is not None:
            self._clock.arm(f"skill/{job.id}")
        threading.Thread(target=self._run, args=(job, skill, receipt), name=f"skill-{job.id}", daemon=True).start()
        return job

    def cancel(self) -> None:
        """Interrupt the current skill without commanding the body (used when the world restarts)."""
        self._stop.set()

    def stop(self) -> dict:
        self._stop.set()
        running = self.current if self.current and self.current.state == "running" else None
        if running is not None and self._clock is not None:
            # A skill waiting on a paused world clock ends now instead of when the world resumes.
            self._release_clock(f"skill/{running.id}")
        st = self._state()
        o = st["odometry"]
        self._command({"base": [o["x"], o["y"], o["yaw"]]})
        return {"stopped": running.id if running else None}

    def wait(self, job_id: str, timeout: float) -> Job:
        deadline = time.monotonic() + timeout
        job = self.jobs[job_id]
        while job.state == "running" and time.monotonic() < deadline:
            time.sleep(0.05)
        return job

    # ---------------------------------------------------------------- runner
    def _begin(self, job: Job, skill: str, receipt: dict) -> None:
        """Run the caller's preparation inside the job thread, so ``start`` returns at once.

        Preparation may take tens of seconds (a fresh segmentation of the scene). The caller that
        waits on the job has a budget for that; the caller that issues the run only has its
        transport timeout, and must not be made to hold it."""
        if self._on_begin is None:
            return
        try:
            self._on_begin(skill, receipt)
        except SkillFailed:
            raise  # the skill itself reports it; a preparation that found the plan stale is not an alert
        except Exception as exc:  # noqa: BLE001 - an unexpected preparation error is an alert, and fails the skill
            if self._on_error is not None:
                self._on_error(f"技能 {skill}（{job.id}）准备阶段出错：{type(exc).__name__}: {exc}")
            raise

    def _run(self, job: Job, skill: str, receipt: dict) -> None:
        try:
            if self._clock is not None:
                self._hold = f"skill/{job.id}"
                self._now = self._clock.wait_until(None, self._hold)
            plan = receipt.get("plan", {})
            self._begin(job, skill, receipt)
            if self._stop.is_set():
                raise SkillStopped()
            self._gentle = bool(plan.get("gentle"))
            if skill == "look_around":
                self._look = True
                try:
                    self._look_around(job)
                finally:
                    self._look = False
            elif skill == "navigate":
                self._drive(job, plan["path"], plan["yaw"])
            elif skill == "pick":
                self._pick(job, plan)
            elif skill == "place" and "floor" in plan:
                self._place_floor(job, plan)
            elif skill == "place":
                self._place(job, plan)
            elif skill in ("open", "close"):
                self._drawer(job, plan)
            elif skill == "pour":
                self._pour(job, plan)
            else:
                raise SkillFailed(f"unknown skill {skill}")
            job.state, job.message = "succeeded", job.message or f"{skill} completed"
        except SkillStopped:
            job.state, job.message = "stopped", "stopped on request"
        except SkillFailed as exc:
            job.state, job.message = "failed", str(exc)
            self._safe_retract()
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller through the job record
            job.state, job.message = "failed", f"{type(exc).__name__}: {exc}"
            if self._on_error is not None:
                self._on_error(f"技能 {skill}（{job.id}）出现未预料的异常：{type(exc).__name__}: {exc}")
            self._safe_retract()
        finally:
            job.finished = self._state()["stamp"]
            if self._clock is not None:
                self._release_clock(f"skill/{job.id}")
            if self._on_finish is not None:
                self._on_finish(job)

    def _released(self, object_id: str, spot: list[float]) -> None:
        if self._on_release is not None:
            self._on_release(object_id, list(spot))

    def _release_clock(self, hold: str) -> None:
        try:
            self._clock.release(hold)
        except Exception:  # noqa: BLE001 - the world is gone or restarting; its hold lease expires on its own
            pass

    def _safe_retract(self) -> None:
        """Rise ``failure_raise``, turn the wrist upright in place, then draw the arm back; each step
        is tried even where the one before it failed."""
        lift = self._state()["joints"].get("lift")
        steps = [lambda: self._set_joint("lift", lift + self.config.failure_raise)] if lift is not None else []
        steps += [lambda: self._ramp_joint("wrist_roll", 0.0, self.config.wrist_turn_s),
                  lambda: self._set_joint("reach", self.config.retract_reach)]
        for step in steps:
            try:
                step()
            except Exception:  # noqa: BLE001 - best effort after a failure that is already reported
                continue

    def _tick(self) -> dict:
        if self._stop.is_set():
            raise SkillStopped()
        if self._clock is None:
            time.sleep(self.config.tick_s)
        else:
            self._now = self._clock.wait_until(self._now + self.config.tick_s, self._hold)
        return self._state()

    # ------------------------------------------------------------- primitives
    def _turn_to(self, yaw_target: float) -> None:
        st = self._state()
        o = st["odometry"]
        x, y, yaw0 = o["x"], o["y"], o["yaw"]
        delta = wrap_angle(yaw_target - yaw0)
        if self._look and self._keyframed:
            speed = self.config.look_angular_speed
        elif self._gentle:
            speed = self.config.gentle_turn_speed
        else:
            speed = self.config.angular_speed
        duration = abs(delta) / speed
        t0 = st["stamp"]
        deadline = t0 + duration * 3 + 5.0
        stall = _Stall(t0, self.config.stall_s, 0.02)
        while True:
            st = self._tick()
            frac = self._profile(1.0 if duration == 0 else min(1.0, (st["stamp"] - t0) / duration))
            self._command({"base": [x, y, yaw0 + delta * frac]})
            err = abs(wrap_angle(yaw_target - st["odometry"]["yaw"]))
            if frac >= 1.0 and err < self.config.yaw_tol:
                return
            if stall.stuck(st["stamp"], err):
                raise SkillFailed(f"turn is blocked: no progress for {self.config.stall_s:.0f} s (yaw error {err:.2f} rad)")
            if st["stamp"] > deadline:
                raise SkillFailed(f"turn did not converge (yaw error {err:.2f} rad)")

    def _translate_to(self, x1: float, y1: float, yaw: float, speed: float | None = None,
                      tol: float | None = None) -> None:
        tol = self.config.position_tol if tol is None else tol
        st = self._state()
        x0, y0 = st["odometry"]["x"], st["odometry"]["y"]
        # The base yaw joint is continuous: express the heading in the same unwrapped frame as the
        # odometry, otherwise the position controller spins the base by a full turn.
        current = st["odometry"]["yaw"]
        yaw = current + wrap_angle(yaw - current)
        length = math.hypot(x1 - x0, y1 - y0)
        duration = length / (speed or (self.config.gentle_speed if self._gentle else self.config.linear_speed))
        t0 = st["stamp"]
        deadline = t0 + duration * 3 + 5.0
        stall = _Stall(t0, self.config.stall_s, 0.01)
        while True:
            st = self._tick()
            frac = self._profile(1.0 if duration == 0 else min(1.0, (st["stamp"] - t0) / duration))
            self._command({"base": [x0 + (x1 - x0) * frac, y0 + (y1 - y0) * frac, yaw]})
            err = math.hypot(x1 - st["odometry"]["x"], y1 - st["odometry"]["y"])
            if frac >= 1.0 and err < tol:
                return
            if stall.stuck(st["stamp"], err):
                raise SkillFailed(f"base is blocked: no progress for {self.config.stall_s:.0f} s ({err:.2f} m from the waypoint)")
            if st["stamp"] > deadline:
                raise SkillFailed(f"base did not reach the waypoint (error {err:.2f} m)")

    def _drive(self, job: Job, path: list, final_yaw: float) -> None:
        for wx, wy in path[1:]:
            o = self._state()["odometry"]
            if math.hypot(wx - o["x"], wy - o["y"]) < self.config.position_tol:
                continue
            heading = math.atan2(wy - o["y"], wx - o["x"])
            self._turn_to(heading)
            self._translate_to(wx, wy, heading)
            job.log.append(f"reached waypoint ({wx:.2f}, {wy:.2f})")
        self._turn_to(final_yaw)

    def _set_joint(self, name: str, target: float, timeout: float = 8.0, ramp: bool = True) -> None:
        self._setpoints[name] = target
        speed = self.config.gentle_lift_speed if self._gentle else self.config.arm_speed
        if ramp and speed and name in ("lift", "reach"):
            st = self._state()
            # An arm at rest stands outside the planar arm's workspace: its first move goes straight in.
            q0 = target if st.get("armResting") else st["joints"].get(name, target)
            seconds = abs(target - q0) / speed
            if seconds > self.config.tick_s:
                self._ramp_joint(name, target, seconds)
                return
        self._command({name: target})
        t0 = self._state()["stamp"]
        corrected = False
        while True:
            st = self._tick()
            q = st["joints"][name]
            if abs(q - target) < self.config.joint_tol:
                return
            settled = st["stamp"] - t0 > self.config.settle_s
            if settled and not corrected:
                # LOCAL EXPERIMENT (uncommitted): re-issue the absolute target instead of
                # target + (target - q). In the split-process topology the command takes an
                # HTTP round trip to reach the servo, so at the 1.5 s stamp the joint may not
                # have started moving yet; inflating by the full remaining error then pegs the
                # joint at ~2x the intended travel (observed: reach 0.101->0.310 landed at
                # 0.519, lift 0.593->0.790 landed at 0.980 — exactly target + (target - rest)).
                self._command({name: target})
                corrected = True
            if st["stamp"] - t0 > timeout:
                raise SkillFailed(f"joint {name} did not reach {target:.3f} (at {q:.3f})")

    def _settle_for_release(self) -> None:
        """Wait until the lift and reach are within ``release_joint_tol`` of their setpoints, for at
        most ``release_settle_s``; a joint that stays farther (sagging under the load) does not fail the place."""
        t0 = self._state()["stamp"]
        while True:
            st = self._tick()
            if all(abs(st["joints"].get(name, target) - target) < self.config.release_joint_tol
                   for name, target in self._setpoints.items() if name in ("lift", "reach")):
                return
            if st["stamp"] - t0 > self.config.release_settle_s:
                return

    def _profile(self, frac: float) -> float:
        """Share of a move done at time share ``frac``: steady, or for a gentle plan starting and
        ending at rest (smoothstep), so a carried container's contents are not flung."""
        return frac * frac * (3.0 - 2.0 * frac) if self._gentle else frac

    def _ramp_joint(self, name: str, target: float, seconds: float) -> None:
        """Move a joint to ``target`` at a steady rate over ``seconds`` of robot time, then settle."""
        self._ramp_joints({name: target}, seconds)

    def _ramp_joints(self, targets: dict[str, float], seconds: float) -> None:
        """Move joints to ``targets`` together over ``seconds`` of robot time, starting and ending
        at rest, then settle each."""
        st = self._state()
        q0, t0 = {name: st["joints"].get(name, 0.0) for name in targets}, st["stamp"]
        while True:
            st = self._tick()
            frac = min(1.0, (st["stamp"] - t0) / seconds) if seconds > 0 else 1.0
            share = frac * frac * (3.0 - 2.0 * frac)
            self._command({name: q0[name] + (target - q0[name]) * share for name, target in targets.items()})
            if frac >= 1.0:
                break
        for name, target in targets.items():
            self._set_joint(name, target, ramp=False)

    def _reach_over(self, plan: dict, xy: list[float], lift: float) -> None:
        """Put the gripper over ``xy`` at ``lift``; with the plan's ``approachLift`` (the spot lies
        below the rim of what holds it) the arm reaches out above the rim and goes straight down."""
        over = plan.get("approachLift")
        if over is not None and over > lift:
            self._set_joint("lift", over)
            self._reach_out(xy, plan.get("wristPitch"))
        self._set_joint("lift", lift)
        self._reach_out(xy, plan.get("wristPitch"))

    def _reach_out(self, xy: list[float], pitch: float | None) -> None:
        """Reach the gripper out to ``xy``, tilting the tool to ``pitch`` on the way (from where it
        is retracted the tilted tool might not reach at all, and tilted out there it would swing
        through what it is reaching into)."""
        reach = max(0.05, self._reach_to(xy))
        joints = self._state()["joints"]
        if pitch is None or abs(joints.get("wrist_pitch", 0.0) - pitch) < self.config.joint_tol:
            self._set_joint("reach", reach)
            return
        seconds = max(abs(reach - joints["reach"]) / (self.config.arm_speed or self.config.gentle_lift_speed),
                      abs(pitch - joints.get("wrist_pitch", 0.0)) / self.config.wrist_pitch_speed)
        self._ramp_joints({"reach": reach, "wrist_pitch": pitch}, seconds)

    def _untilt(self) -> None:
        """Turn a tool tilted forward for a grasp back to pointing down, once the arm is retracted."""
        pitch = self._state()["joints"].get("wrist_pitch", 0.0)
        if abs(pitch) >= self.config.joint_tol:
            self._ramp_joints({"wrist_pitch": 0.0}, abs(pitch) / self.config.wrist_pitch_speed)

    def _withdraw(self, plan: dict) -> None:
        """Retract the arm, first back up above the rim it reached over (the plan's ``approachLift``)."""
        over = plan.get("approachLift")
        if over is not None and self._state()["joints"]["lift"] < over:
            self._set_joint("lift", over)
        self._retract(plan)

    def _retract(self, plan: dict) -> None:
        """Bring the gripper back to the retracted reach, or only as near as the plan's ``retractReach``
        (a tool tilted forward cannot come all the way back without turning the wrist over)."""
        self._set_joint("reach", max(self.config.retract_reach, plan.get("retractReach", 0.0)))

    def _wait(self, seconds: float) -> None:
        t0 = self._state()["stamp"]
        while self._tick()["stamp"] - t0 < seconds:
            pass

    def _dock(self, job: Job, plan: dict, retract: bool = True) -> None:
        if retract:
            # Only what the arm holds is held at the plan's tilt; an empty gripper points down.
            self._retract(plan if self._state()["gripper"].get("holding") else {})
        self._drive(job, plan["path"], plan["yaw"])
        self._translate_to(plan["dock"][0], plan["dock"][1], plan["yaw"], tol=self.config.dock_position_tol)
        o = self._state()["odometry"]
        dist = math.hypot(o["x"] - plan["dock"][0], o["y"] - plan["dock"][1])
        dyaw = abs(wrap_angle(o["yaw"] - plan["yaw"]))
        if dist > self.config.dock_tol or dyaw > self.config.dock_yaw_tol:
            self._backoff(plan)
            raise SkillFailed(f"docking missed the planned pose by {dist:.2f} m / {math.degrees(dyaw):.0f} deg")
        job.log.append("docked")

    def _gripper_xy(self) -> np.ndarray:
        """Gripper position from the robot's own odometry and reach encoder, on the arm's line
        (``armLateral`` left of the base centre)."""
        st = self._state()
        o = st["odometry"]
        ahead, side = st["toolOffset"] + st["joints"]["reach"], st.get("armLateral", 0.0)
        c, s = math.cos(o["yaw"]), math.sin(o["yaw"])
        return np.array([o["x"] + ahead * c - side * s, o["y"] + ahead * s + side * c])

    def _align_lateral(self, target_xy: list[float]) -> None:
        """Shift the base sideways so the target lies on the arm's line of reach."""
        st = self._state()
        o = st["odometry"]
        left = np.array([-math.sin(o["yaw"]), math.cos(o["yaw"])])
        lateral = float((np.array(target_xy) - np.array([o["x"], o["y"]])) @ left) - st.get("armLateral", 0.0)
        if self.config.align_min < abs(lateral) <= self.config.align_max:
            self._translate_to(o["x"] + lateral * left[0], o["y"] + lateral * left[1], o["yaw"])

    def _backoff(self, plan: dict) -> None:
        self._translate_to(plan["predock"][0], plan["predock"][1], plan["yaw"])

    def _reach_to(self, target_xy: list[float]) -> float:
        """Wrist extension that puts the gripper over ``target_xy``, within the joint's travel.

        The plan's reach and the pose the base actually stopped at differ by a few centimetres; a
        target a little beyond the arm used to fail the whole pick with "joint reach did not
        reach", when all it costs is that the gripper closes short — inside its capture radius."""
        st = self._state()
        o = st["odometry"]
        heading = np.array([math.cos(o["yaw"]), math.sin(o["yaw"])])
        rel = np.array(target_xy) - np.array([o["x"], o["y"]])
        return min(self.config.reach_max, float(rel @ heading) - st["toolOffset"])

    def _over_support(self, plan: dict) -> bool:
        """True when the held object's footprint lies within the support top: the object's centre is
        where the gripper is, offset as the plan's ``spot`` is from its ``gripSpot``."""
        box = plan["supportBox"]
        grip = plan.get("gripSpot", plan["spot"])
        g = self._gripper_xy() + np.array([plan["spot"][0] - grip[0], plan["spot"][1] - grip[1]])
        dx, dy = g[0] - box["cx"], g[1] - box["cy"]
        c, s = math.cos(-box["yaw"]), math.sin(-box["yaw"])
        lx, ly = c * dx - s * dy, s * dx + c * dy
        margin = plan.get("heldHalf", 0.05) * 0.6
        return abs(lx) <= box["hx"] - margin and abs(ly) <= box["hy"] - margin

    # ---------------------------------------------------------------- skills
    def _look_around(self, job: Job) -> None:
        yaw0 = self._state()["odometry"]["yaw"]
        for k in range(1, 5):
            self._turn_to(wrap_angle(yaw0 + k * math.pi / 2))
            self._wait(self.config.look_dwell_s)
            job.log.append(f"looked toward {math.degrees(wrap_angle(yaw0 + k * math.pi / 2)):.0f} deg")

    def _pick(self, job: Job, plan: dict) -> None:
        self._dock(job, plan)
        target = self._state()["entities"].get(plan["target"])
        if target is None and plan.get("targetP") is not None:
            # The track was renumbered while docking; the predicted position is seconds old and
            # the gripper is already beside it — go on with the position.
            target = {"p": list(plan["targetP"])}
        if target is None:
            raise SkillFailed(f"target {plan['target']} is no longer in the ego state")
        # A large object may be taken by its edge (``graspOffset`` from its centre, in the world), a
        # container on its rim (``graspDz`` above its centre).
        ox, oy = plan.get("graspOffset") or (0.0, 0.0)
        dz = plan.get("graspDz", 0.0)

        def aimed(entity: dict) -> dict:
            return {"p": [entity["p"][0] + ox, entity["p"][1] + oy, entity["p"][2] + dz]}

        target = aimed(target)
        self._align_lateral(target["p"][:2])
        # Off the floor the planned height (its base at floor level) outranks the noisy live estimate.
        self._reach_over(plan, target["p"][:2], plan["lift"] if plan.get("support") == "floor" else target["p"][2])
        miss = float(np.linalg.norm(self._gripper_xy() - np.array(target["p"][:2])))
        if miss > plan.get("captureRadius", 0.05):
            self._withdraw(plan)
            self._untilt()
            self._backoff(plan)
            raise SkillFailed(f"gripper is {miss * 100:.0f} cm from {plan['target']}; not closing")
        # A real object's estimate is a few centimetres off in any direction as often as not, and
        # the hand only feels the object when it closes. Probe around the estimate the way a person
        # feels for a cup in the dark: at the estimate, then a little forward and back (the arm
        # alone moves), then to either side (the base shifts), then at other heights. The estimate
        # is re-read before each probe -- it is best while the camera is right over the object.
        # Taken by its edge (docked facing that side), forward lies over what is inside it and back
        # over nothing: the probes keep along the edge.
        probes = [(0.0, 0.0, 0.0)]
        if not (ox or oy):
            probes += [(dx, 0.0, 0.0) for dx in self.config.grasp_probe_xy]
        probes += [(0.0, dy, 0.0) for dy in self.config.grasp_probe_xy]
        probes += [(0.0, 0.0, dz) for dz in self.config.grasp_probe_z]
        grasped = False
        for n, (forward, sideways, dz) in enumerate(probes):
            seen = self._state()["entities"].get(plan["target"])
            latest = aimed(seen) if seen is not None else target
            o = self._state()["odometry"]
            heading = np.array([math.cos(o["yaw"]), math.sin(o["yaw"])])
            left = np.array([-heading[1], heading[0]])
            goal = np.array(latest["p"][:2]) + forward * heading + sideways * left
            if n:
                self._command({"gripper": "open"})
                self._wait(self.config.grasp_wait_s)
                self._align_lateral(goal.tolist())
                try:
                    self._set_joint("lift", latest["p"][2] + dz)
                    self._set_joint("reach", max(0.05, self._reach_to(goal.tolist())))
                except OutOfReach as e:
                    job.log.append(f"skipped probe {n}: {e}")
                    continue
            self._command({"gripper": "close"})
            self._wait(self.config.grasp_wait_s)
            if self._state()["gripper"]["holding"]:
                if n:
                    job.log.append(f"grasped {plan['target']} on probe {n} "
                                   f"(forward {forward * 100:+.0f} cm, left {sideways * 100:+.0f} cm, "
                                   f"height {dz * 100:+.1f} cm)")
                grasped = True
                break
        if not grasped:
            self._command({"gripper": "open"})
            self._wait(0.2)
        if not grasped:
            self._withdraw(plan)
            self._untilt()
            self._backoff(plan)
            raise SkillFailed(f"grasp missed {plan['target']} ({len(probes)} probes)")
        try:
            held = self._mark_held(plan["target"])
        except KeyError as exc:
            raise SkillFailed(str(exc).strip("'")) from None
        job.log.append(f"grasped {held}")
        self._set_joint("lift", plan["carryLift"])
        if not plan.get("keepReach"):
            # A large object carried low (a chair off the floor) keeps the arm out so it clears the base.
            self._retract(plan)
        if "retreat" in plan:
            self._translate_to(plan["retreat"][0], plan["retreat"][1], plan["yaw"])
        else:
            self._backoff(plan)
        job.message = f"holding {held}"

    def _place(self, job: Job, plan: dict) -> None:
        self._dock(job, plan)
        if "wristYaw" in plan:
            self._ramp_joint("wrist_yaw", plan["wristYaw"], self.config.wrist_turn_s)
        grip = plan.get("gripSpot", plan["spot"])
        if "turnLift" in plan:
            # Turned over in the air above the spot, then lowered as far as the arm reaches so.
            self._align_lateral(grip)
            self._set_joint("lift", plan["turnLift"])
            self._reach_out(grip, plan.get("wristPitch"))
            self._command({"wrist_roll": plan["wristRoll"], "turnOver": True})
            self._set_joint("wrist_roll", plan["wristRoll"], ramp=False)
        elif "wristRoll" in plan:
            self._ramp_joint("wrist_roll", plan["wristRoll"], self.config.wrist_turn_s)
        self._align_lateral(grip)
        self._reach_over(plan, grip, plan["lift"])
        if not self._over_support(plan):
            self._withdraw(plan)
            self._backoff(plan)
            raise SkillFailed(f"gripper is not over {plan['support']}; keeping {plan['object']}")
        self._settle_for_release()
        self._command({"gripper": "open"})
        self._released(plan["object"], plan["spot"])
        self._wait(self.config.release_wait_s)
        job.log.append(f"released {plan['object']} on {plan['support']}")
        if "turnLift" in plan:
            self._set_joint("lift", plan["turnLift"])
            self._command({"wrist_roll": 0.0, "turnOver": True})
            self._set_joint("wrist_roll", 0.0, ramp=False)
        self._withdraw(plan)
        self._untilt()
        self._set_joint("lift", max(plan["lift"], self.config.stow_lift))
        if "wristYaw" in plan:
            self._set_joint("wrist_yaw", 0.0)
        if "wristRoll" in plan:
            self._set_joint("wrist_roll", 0.0)
        self._backoff(plan)
        job.message = f"placed {plan['object']} on {plan['support']}"

    def _pour(self, job: Job, plan: dict) -> None:
        """Hold the container low over the spot, tip it slowly about the arm, wait for the contents
        to fall out, raise it ``rise`` off the pile, set it upright again and back away still holding it."""
        self._dock(job, plan)
        grip = plan.get("gripSpot", plan["spot"])
        self._align_lateral(grip)
        self._set_joint("lift", plan["lift"])
        self._set_joint("reach", max(0.05, self._reach_to(grip)))
        self._tip(plan["lift"], plan["pivot"], 0.0, plan["roll"])
        self._wait(plan["pourS"])
        job.log.append(f"poured {plan['object']} over {plan['support']}")
        raised = plan["lift"] + plan["rise"]
        self._set_joint("lift", raised + plan["pivot"] * abs(math.sin(plan["roll"])))
        self._tip(raised, plan["pivot"], plan["roll"], 0.0)
        self._retract(plan)
        self._set_joint("lift", max(plan["lift"], self.config.stow_lift))
        self._backoff(plan)
        job.message = f"poured {plan['object']} over {plan['support']}"

    def _tip(self, lift0: float, pivot: float, roll0: float, roll1: float) -> None:
        """Turn the wrist roll from ``roll0`` to ``roll1`` about the held container's lower edge, ``pivot``
        metres from the gripper: the lift rises by pivot * |sin roll|, so that edge stays just above
        the surface and the contents slide out from low down instead of dropping."""
        t0 = self._state()["stamp"]
        while True:
            st = self._tick()
            frac = min(1.0, (st["stamp"] - t0) / self.config.pour_tip_s)
            roll = roll0 + (roll1 - roll0) * frac * frac * (3.0 - 2.0 * frac)
            self._command({"wrist_roll": roll, "lift": lift0 + pivot * abs(math.sin(roll))})
            if frac >= 1.0:
                break
        self._set_joint("wrist_roll", roll1)

    def _drawer(self, job: Job, plan: dict) -> None:
        """Grasp a drawer handle, move the base along the drawer axis to slide it, release and back away."""
        self._dock(job, plan)
        handle = plan["handle"]
        self._align_lateral(handle[:2])
        self._set_joint("lift", handle[2])
        self._set_joint("reach", max(0.05, self._reach_to(handle[:2])))
        miss = float(np.linalg.norm(self._gripper_xy() - np.array(handle[:2])))
        if miss > plan.get("captureRadius", 0.05):
            self._set_joint("reach", self.config.retract_reach)
            self._backoff(plan)
            raise SkillFailed(f"gripper is {miss * 100:.0f} cm from the handle of {plan['drawer']}; not closing")
        self._command({"gripper": "close"})
        self._wait(self.config.grasp_wait_s)
        if not self._state()["gripper"].get("holdingHandle"):
            self._command({"gripper": "open"})
            self._set_joint("reach", self.config.retract_reach)
            self._backoff(plan)
            raise SkillFailed(f"missed the handle of {plan['drawer']}")
        job.log.append(f"grasped the handle of {plan['drawer']}")
        start = self._state()["odometry"]
        self._translate_to(plan["end"][0], plan["end"][1], plan["yaw"], self.config.drawer_speed)
        end = self._state()["odometry"]
        self._command({"gripper": "open"})
        self._wait(self.config.handle_release_wait_s)
        moved = (end["x"] - start["x"]) * plan["axis"][0] + (end["y"] - start["y"]) * plan["axis"][1]
        if self._shift is not None:
            self._shift(plan["drawer"], [moved * plan["axis"][0], moved * plan["axis"][1], 0.0])
        verb = "opened" if plan["direction"] > 0 else "closed"
        job.log.append(f"{verb} {plan['drawer']} by {abs(moved) * 100:.0f} cm")
        self._set_joint("reach", self.config.retract_reach)
        self._translate_to(plan["viewing"][0], plan["viewing"][1], plan["yaw"])
        self._set_joint("lift", self.config.stow_lift)
        job.message = f"{verb} {plan['drawer']}"

    def _place_floor(self, job: Job, plan: dict) -> None:
        """Drive in with the held object, lower it to just above the floor, release and back away.

        The arm keeps its carrying reach, or extends to the plan's ``reach`` where the object would
        otherwise land on the base: retracting it would pull the object over the base."""
        self._dock(job, plan, retract=False)
        if "reach" in plan:
            self._set_joint("reach", plan["reach"])
        self._set_joint("lift", plan["lift"])
        self._command({"gripper": "open"})
        self._released(plan["object"], plan["floor"])
        self._wait(self.config.release_wait_s)
        job.log.append(f"released {plan['object']} on the floor at ({plan['floor'][0]:.2f}, {plan['floor'][1]:.2f})")
        self._backoff(plan)
        self._set_joint("lift", self.config.stow_lift)
        job.message = f"placed {plan['object']} on the floor"
