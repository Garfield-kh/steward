"""The Cobot Magic's planar arm (:mod:`steward.ego.arm`): one Piper arm and the torso lift.

The virtual joints place the pinch point in the robot's own frame -- ``tool_offset + reach``
ahead of the base centre on the arm's own line (``lateral`` = the arm mount's y), ``lift`` above
the floor -- with the tool pointing down, turned ``wrist_yaw`` about the vertical and tipped
``wrist_roll`` about the forward axis. A command on them is solved by inverse kinematics in the
base frame (so odometry noise never enters the arm), at the first torso height that admits a
solution, the current one first: the torso moves only when the arm alone cannot get there.

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
from steward.math3d import matrix_from_quat, pose_compose, pose_inverse, quat_from_axis_angle, quat_mul

TOOL_DOWN = [0.0, 1.0, 0.0, 0.0]  # link6 +z (through the pinch point) along world -z


@dataclass
class PiperArmConfig:
    arm: str = "L"  # the arm that works; the other stays parked
    # Shoulder heights above the pinch point tried for the torso, after the current torso height.
    shoulder_above: tuple[float, ...] = (0.15, 0.25, 0.05, 0.35, -0.05)
    reach: tuple[float, float] = (0.22, 0.50)  # pinch point ahead of the arm mount, tool down
    lift: tuple[float, float] = (0.30, 1.15)  # pinch point above the floor, tool down
    wrist_yaw_max: float = 3.1
    wrist_roll_max: float = 1.2
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
                                     gripper=f"{cfg.arm}_link6"))
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
        m = matrix_from_quat(tcp["q"]) @ matrix_from_quat(TOOL_DOWN).T
        return {"lift": float(tcp["p"][2]), "reach": float(tcp["p"][0]) - self.geometry.tool_offset,
                "wrist_yaw": math.atan2(m[1, 0], m[0, 0]), "wrist_roll": math.atan2(m[2, 1], m[2, 2])}

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
                # From the parked posture the first command starts with the tool pointing down, as
                # the kitchen robot's wrist starts at zero: the parked tool is folded sideways.
                target = {**self._fk(float(joints["lift"]), [float(joints[n]) for n in self.names]),
                          "wrist_yaw": 0.0, "wrist_roll": 0.0}
            target.update(wanted)
            tcp = [self.geometry.tool_offset + target["reach"], self.geometry.lateral, target["lift"]]
            tool = quat_mul(quat_mul(quat_from_axis_angle([0.0, 0.0, 1.0], target["wrist_yaw"]),
                                     quat_from_axis_angle([1.0, 0.0, 0.0], target["wrist_roll"])), TOOL_DOWN)
            wrist = pose_compose({"p": tcp, "q": tool}, pose_inverse({"p": TCP_OFFSET, "q": [1.0, 0.0, 0.0, 0.0]}))
            seed = [float(joints[n]) for n in self.names] if all(n in joints for n in self.names) else None
            now = float(joints["lift"]) if "lift" in joints else None
            last: IkError | None = None
            for torso in self._torso_heights(target["lift"], now):
                self.ik.set_base(0.0, 0.0, 0.0, torso)
                try:
                    arm = self.ik.solve(self.config.arm, wrist["p"], wrist["q"], q_init=seed, orientation_tolerance=0.02)
                except IkError as exc:
                    last = exc
                    continue
                self._target, self._solved = target, (torso, arm)
                return torso, arm
        raise SkillFailed(f"the {self.config.arm} arm cannot put its gripper {target['reach']:.2f} m ahead at "
                          f"{target['lift']:.2f} m height (wrist yaw {target['wrist_yaw']:.2f}, roll "
                          f"{target['wrist_roll']:.2f}) at any torso height: {last}")

    def _torso_heights(self, lift: float, now: float | None) -> list[float]:
        lo, hi = LIFT_RANGE
        out = [] if now is None else [min(max(now, lo), hi)]
        for above in self.config.shoulder_above:
            q = min(max(lift + above - CHASSIS["height"], lo), hi)
            if all(abs(q - o) > 0.02 for o in out):
                out.append(q)
        return out


def workspace(arm: PiperArm, reaches: np.ndarray, lifts: np.ndarray, wrist_yaw: float = 0.0,
              wrist_roll: float = 0.0) -> np.ndarray:
    """Which (reach, lift) pairs the arm reaches from its parked posture with the tool down, turned
    and tipped as given: a boolean grid, rows by lift."""
    ok = np.zeros((len(lifts), len(reaches)), dtype=bool)
    for i, lift in enumerate(lifts):
        for j, reach in enumerate(reaches):
            arm._target, arm._solved = None, None
            arm._last = {"joints": {"lift": LIFT_HOME, **dict(zip(arm.names, HOME, strict=True))}}
            try:
                arm._solve({"lift": float(lift), "reach": float(reach), "wrist_yaw": wrist_yaw, "wrist_roll": wrist_roll})
            except SkillFailed:
                continue
            ok[i, j] = True
    return ok
