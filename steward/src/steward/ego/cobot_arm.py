"""The Cobot Magic's planar arm (:mod:`steward.ego.arm`): one Piper arm and the torso lift.

The virtual joints place the pinch point in the robot's own frame -- ``tool_offset + reach``
ahead of the base centre on the arm's own line (``lateral`` = the arm mount's y), ``lift`` above
the floor -- with the tool pointing down, tilted ``wrist_pitch`` forward, turned ``wrist_yaw``
about the vertical and tipped ``wrist_roll`` about the forward axis. The Piper's wrist bends only
+-70 deg (joint5), so with the tool straight down the arm reaches 0.50 m ahead of its mount and
tips a container less than 1 rad out there; tilted 60 deg forward it reaches 0.70 m and turns
the tool about its own axis instead (joint6, a full turn). A command on them is solved by inverse kinematics in the
base frame (so odometry noise never enters the arm), at the torso height the target ``lift``
sets, so the torso moves as smoothly as the skill moves the pinch point; other heights are tried
only where the arm cannot reach from that one.

Encoder readings are turned back into virtual joints by forward kinematics, except that an arm
standing within servo tolerance of the joints solved for the last command reports that command:
a solution meets its pose to a few millimetres and hundredths of a radian, and the skills wait for
their own targets, not for the solver's residual.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

import numpy as np

from steward.assets.cobot import ARM_MOUNT_X, CHASSIS, HOME, LIFT_HOME, LIFT_RANGE, TCP_OFFSET
from steward.ego.arm import VIRTUAL_JOINTS, Arm, ArmGeometry
from steward.ego.ik import CobotIK, IkError
from steward.ego.skills import SkillFailed
from steward.math3d import matrix_from_quat, pose_compose, quat_from_axis_angle, quat_mul

TOOL_DOWN = [0.0, 1.0, 0.0, 0.0]  # link6 +z (through the pinch point) along world -z


@dataclass
class PiperArmConfig:
    arm: str = "L"  # the arm that works; the other stays parked
    # The torso is solved with the arm; it is drawn to a shoulder this high above the pinch point
    # with the tool down, lower by ``shoulder_drop_per_pitch`` per radian of forward tilt (measured:
    # tilted 60 deg, a shoulder 0.25 m below the pinch point tips a container 1.2 rad anywhere from
    # 0.35 to 0.65 m out), so it follows the pinch point as smoothly as a skill moves it.
    shoulder_above: float = 0.15
    shoulder_drop_per_pitch: float = 0.38
    pitch_tolerance: float = 0.03  # how close ``pitch_for`` comes to the least tilt that holds the grasp point
    torso_seed_inset: float = 0.05  # the low and high torso seeds stand this far inside its travel
    # Pinch point ahead of the arm mount: tool down 0.50, tilted 60 deg 0.70; planned short of that,
    # where a small step does not make the solver turn the wrist over.
    reach: tuple[float, float] = (0.22, 0.66)
    lift: tuple[float, float] = (0.30, 1.15)  # pinch point above the floor, tool down
    wrist_yaw_max: float = 3.1
    # Bounds only; what a posture allows is solved (``reachable``): the tool straight down tips
    # less than 1 rad, tilted forward a full turn.
    wrist_roll_max: float = 2.5
    wrist_pitch_max: float = 1.05
    pitch_unseen_cos: float = 0.1  # |cos roll| below which the tool pitch cannot be read from the posture
    # Holding something, a small step of the grasp point (a ramp, a tip) that the solver can only
    # follow by turning the wrist over to another posture -- a joint moving this far, or the torso
    # this far -- is refused: the arm would swing what it holds about on the way (measured: tipping
    # a held box past 1.4 rad, joint4 meets its limit and the solution jumps 3.5 rad).
    max_joint_step: float = 0.6
    max_torso_step: float = 0.06
    small_step: tuple[float, float] = (0.05, 0.2)  # metres, radians: a command this close to the last is a step
    arrived_rad: float = 0.04  # encoders this close to the last solution report its command
    arrived_lift_m: float = 0.01


class PiperArm(Arm):
    def __init__(self, robot_model: dict, config: PiperArmConfig | None = None):
        self.config = cfg = config or PiperArmConfig()
        self.ik = CobotIK(robot_model)
        mount = next(j for j in robot_model["body"]["joints"] if j["id"] == f"{cfg.arm}_mount")
        lateral = float(mount["origin"]["value"]["p"][1])
        super().__init__(ArmGeometry(tool_offset=ARM_MOUNT_X, lateral=lateral, reach_min=cfg.reach[0],
                                     reach_max=cfg.reach[1], lift_min=cfg.lift[0], lift_max=cfg.lift[1],
                                     wrist_yaw_max=cfg.wrist_yaw_max, wrist_roll_max=cfg.wrist_roll_max,
                                     gripper=f"{cfg.arm}_link6", wrist_pitch_max=cfg.wrist_pitch_max))
        self.names = self.ik.joint_names(cfg.arm)
        self._lock = threading.Lock()
        self._target: dict | None = None  # the virtual joints last commanded
        self._solved: tuple[float, list[float]] | None = None  # (torso, arm joints) solved for them
        self._last: dict = {}  # the latest sensor header seen

    # ------------------------------------------------------------- reading
    def virtualize(self, state: dict) -> dict:
        self._last = state
        out = super().virtualize(state)
        out["joints"] = {**(state.get("joints") or {}), **self.joints_of(state.get("joints") or {}, state.get("odometry"))}
        return out

    def resting(self) -> bool:
        """Until its first command the arm stands parked, folded upright and turned aside."""
        with self._lock:
            return self._target is None

    def joints_of(self, joints: dict, odometry: dict | None = None) -> dict:
        if not all(n in joints for n in self.names) or "lift" not in joints:
            return {}
        arm = [float(joints[n]) for n in self.names]
        torso = float(joints["lift"])
        with self._lock:
            if self._target is not None and self._solved is not None and abs(torso - self._solved[0]) < self.config.arrived_lift_m \
                    and max(abs(a - b) for a, b in zip(arm, self._solved[1], strict=True)) < self.config.arrived_rad:
                return dict(self._target)
            return self._fk(torso, arm)

    def _fk(self, torso: float, arm: list[float]) -> dict:
        self.ik.set_base(0.0, 0.0, 0.0, torso)
        tcp = pose_compose(self.ik.fk(self.config.arm, arm), {"p": TCP_OFFSET, "q": [1.0, 0.0, 0.0, 0.0]})
        # m = Rz(yaw) Rx(roll) Ry(-pitch); the pitch never reaches a quarter turn, so the tipped
        # (roll beyond a quarter turn) reading is the one turned back by half a turn. Tipped a
        # quarter turn the pitch does not show in the posture: it is the one last commanded.
        m = matrix_from_quat(tcp["q"]) @ matrix_from_quat(TOOL_DOWN).T
        if math.hypot(m[2, 0], m[2, 2]) < self.config.pitch_unseen_cos:
            pitch = self._target["wrist_pitch"] if self._target is not None else 0.0
        else:
            pitch = math.atan2(m[2, 0], m[2, 2])
            if abs(pitch) > math.pi / 2:
                pitch = math.atan2(-m[2, 0], -m[2, 2])
        m = m @ matrix_from_quat(quat_from_axis_angle([0.0, 1.0, 0.0], pitch))
        return {"lift": float(tcp["p"][2]), "reach": float(tcp["p"][0]) - self.geometry.tool_offset,
                "wrist_yaw": math.atan2(m[1, 0], m[0, 0]), "wrist_roll": math.atan2(m[2, 1], m[2, 2]),
                "wrist_pitch": pitch}

    # ----------------------------------------------------------- commanding
    def command(self, cmd: dict) -> dict:
        out = {k: v for k, v in cmd.items() if k == "base"}
        if "gripper" in cmd:
            grip = cmd["gripper"]
            out["gripper"] = grip if isinstance(grip, dict) else {self.geometry.gripper: grip}
        wanted = {k: float(cmd[k]) for k in VIRTUAL_JOINTS if k in cmd}
        if wanted:
            torso, arm = self._solve(wanted)
            out["joints"] = {"lift": torso, **dict(zip(self.names, arm, strict=True))}
        return out

    def _solve(self, wanted: dict) -> tuple[float, list[float]]:
        joints = self._last.get("joints") or {}
        with self._lock:
            target = dict(self._target) if self._target is not None else None
            if target is None:
                if not all(n in joints for n in self.names) or "lift" not in joints:
                    raise SkillFailed("the arm's encoders have not been read yet")
                # From the resting posture the first command starts with the tool pointing down, as
                # the kitchen robot's wrist starts at zero, and inside the workspace: the resting arm
                # stands folded upright, its tool sideways and above the highest working point.
                g = self.geometry
                here = self._fk(float(joints["lift"]), [float(joints[n]) for n in self.names])
                target = {"lift": min(max(here["lift"], g.lift_min), g.lift_max),
                          "reach": min(max(here["reach"], g.reach_min), g.reach_max), "wrist_yaw": 0.0, "wrist_roll": 0.0,
                          "wrist_pitch": 0.0}
            previous, prior = dict(target), self._solved
            target.update(wanted)
            solved, last = self._ik(target, joints, prior)
            holding = ((self._last.get("grippers") or {}).get(self.geometry.gripper) or {}).get("holding")
            if (solved is not None and holding and prior is not None and self._step(previous, target)
                    and self._jumps(prior, solved)):
                raise SkillFailed(f"the {self.config.arm} arm would have to turn its wrist over to follow its gripper "
                                  f"{target['reach']:.2f} m ahead at {target['lift']:.2f} m (roll {target['wrist_roll']:.2f}, "
                                  f"pitch {target['wrist_pitch']:.2f}); stopped before it swings what it holds")
            if solved is not None:
                self._target, self._solved = target, solved
                return solved
        raise SkillFailed(f"the {self.config.arm} arm cannot put its gripper {target['reach']:.2f} m ahead at "
                          f"{target['lift']:.2f} m height (wrist yaw {target['wrist_yaw']:.2f}, roll "
                          f"{target['wrist_roll']:.2f}, pitch {target['wrist_pitch']:.2f}) at any torso height: {last}")

    def _step(self, a: dict, b: dict) -> bool:
        """Whether virtual-joint targets ``b`` lie a small step from ``a``."""
        metres, radians = self.config.small_step
        return (max(abs(a[k] - b[k]) for k in ("lift", "reach")) <= metres
                and max(abs(a[k] - b[k]) for k in ("wrist_yaw", "wrist_roll", "wrist_pitch")) <= radians)

    def _jumps(self, a: tuple[float, list[float]], b: tuple[float, list[float]]) -> bool:
        """Whether going from solution ``a`` to ``b`` (torso, arm joints) turns the arm over."""
        return (abs(a[0] - b[0]) > self.config.max_torso_step
                or max(abs(x - y) for x, y in zip(a[1], b[1], strict=True)) > self.config.max_joint_step)

    def _ik(self, target: dict, joints: dict, prior: tuple[float, list[float]] | None = None
            ) -> tuple[tuple[float, list[float]] | None, IkError | None]:
        """Torso height and arm joints for virtual joints ``target`` in one whole-body solve, seeded
        from ``prior`` (the last solution, so a motion keeps one posture), then ``joints``, then the
        working posture. The caller holds the lock."""
        cfg = self.config
        pitch = target["wrist_pitch"]
        tcp = [self.geometry.tool_offset + target["reach"], self.geometry.lateral, target["lift"]]
        tool = quat_mul(quat_mul(quat_from_axis_angle([0.0, 0.0, 1.0], target["wrist_yaw"]),
                                 quat_from_axis_angle([1.0, 0.0, 0.0], target["wrist_roll"])),
                        quat_mul(quat_from_axis_angle([0.0, 1.0, 0.0], -pitch), TOOL_DOWN))
        prefer = target["lift"] + cfg.shoulder_above - cfg.shoulder_drop_per_pitch * pitch - CHASSIS["height"]
        encoders = ([(float(joints["lift"]), [float(joints[n]) for n in self.names])]
                    if "lift" in joints and all(n in joints for n in self.names) else [])
        # Then the working posture at the preferred torso height and near either end of the torso's
        # travel: one solve moves the torso only so far from where it starts.
        lo, hi = LIFT_RANGE
        seeds = ([prior] if prior is not None else []) + encoders + [
            (min(max(h, lo), hi), list(HOME)) for h in (prefer, lo + cfg.torso_seed_inset, hi - cfg.torso_seed_inset)]
        last: IkError | None = None
        self.ik.set_base(0.0, 0.0, 0.0, None)
        for lift, q in seeds:
            try:
                solved = self.ik.solve_body(cfg.arm, tcp, tool, q, lift, prefer)
            except IkError as exc:
                last = exc
                continue
            return solved, None
        return None, last

    def reachable(self, lift: float, reach: float, wrist_yaw: float = 0.0, wrist_roll: float = 0.0,
                  wrist_pitch: float = 0.0) -> bool:
        if not super().reachable(lift, reach, wrist_yaw, wrist_roll, wrist_pitch):
            return False
        with self._lock:
            solved, _ = self._ik({"lift": lift, "reach": reach, "wrist_yaw": wrist_yaw, "wrist_roll": wrist_roll,
                                  "wrist_pitch": wrist_pitch}, self._last.get("joints") or {})
        return solved is not None

    def pitch_for(self, lift: float, reach: float, wrist_yaw: float = 0.0, wrist_roll: float = 0.0) -> float | None:
        g = self.geometry
        if not super().reachable(lift, reach, wrist_yaw, wrist_roll, 0.0):
            return None
        joints = self._last.get("joints") or {}
        at = {"lift": lift, "reach": reach, "wrist_yaw": wrist_yaw, "wrist_roll": wrist_roll}
        with self._lock:
            if self._ik({**at, "wrist_pitch": 0.0}, joints)[0] is not None:
                return 0.0
            # A tilt that holds it there, from the most down; the least one lies between it and none.
            lo, hi = 0.0, None
            for share in (1.0, 0.75, 0.5):
                if self._ik({**at, "wrist_pitch": g.wrist_pitch_max * share}, joints)[0] is not None:
                    hi = g.wrist_pitch_max * share
                    break
            if hi is None:
                return None
            while hi - lo > self.config.pitch_tolerance:
                mid = (lo + hi) / 2
                if self._ik({**at, "wrist_pitch": mid}, joints)[0] is not None:
                    hi = mid
                else:
                    lo = mid
            return hi

    def reachable_path(self, targets) -> bool:
        if not super().reachable_path(targets):
            return False
        with self._lock:
            joints = self._last.get("joints") or {}
            prior = None
            for t in targets:
                full = {"wrist_yaw": 0.0, "wrist_roll": 0.0, "wrist_pitch": 0.0, **t}
                solved, _ = self._ik(full, joints, prior)
                if solved is None or (prior is not None and self._jumps(prior, solved)):
                    return False
                prior = solved
        return True



def workspace(arm: PiperArm, reaches: np.ndarray, lifts: np.ndarray, wrist_yaw: float = 0.0,
              wrist_roll: float = 0.0, wrist_pitch: float = 0.0) -> np.ndarray:
    """Which (reach, lift) pairs the arm reaches from its parked posture with the tool down, tilted,
    turned and tipped as given: a boolean grid, rows by lift."""
    ok = np.zeros((len(lifts), len(reaches)), dtype=bool)
    for i, lift in enumerate(lifts):
        for j, reach in enumerate(reaches):
            arm._target, arm._solved = None, None
            arm._last = {"joints": {"lift": LIFT_HOME, **dict(zip(arm.names, HOME, strict=True))}}
            try:
                arm._solve({"lift": float(lift), "reach": float(reach), "wrist_yaw": wrist_yaw, "wrist_roll": wrist_roll,
                            "wrist_pitch": wrist_pitch})
            except SkillFailed:
                continue
            ok[i, j] = True
    return ok
