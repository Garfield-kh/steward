"""Cobot Magic inverse kinematics on the robot's own kinematic model.

The twin's model is the vendored Piper URDF (see :mod:`steward.assets.cobot`), compiled to
MJCF and solved with mink: 6 revolute joints per arm, no analytic IK, and 6-DoF is exactly the
case PyRoki refuses, so the solver is mink exactly as on the YAM arm.

The solver freezes every DOF outside the arm being solved — the other arm's joints and both
fingers — so a left-arm goal never drags the right arm along. Joint limits come from the model
(:class:`mink.ConfigurationLimit`), so a solution is always inside the URDF ranges.

Frames: the end-effector frame of an arm is its last revolute link (``L_link6`` / ``R_link6``).
Its +z axis points out through the pinch point, 0.135 m away (``assets.cobot.TCP_OFFSET``), and
the fingers separate along its y axis; the result is measured, not assumed — see
``tests/test_cobot.py::test_tcp_is_where_the_fingers_pinch``.
"""

from __future__ import annotations

import math

import numpy as np

from steward.assets.cobot import HOME, TCP_OFFSET, cobot_magic
from steward.math3d import quat_angle_between
from steward.scene.mjcf import build_mjcf
from steward.scene.schema import SceneState

import mink

ARMS = ("L", "R")
_ENT = "w:robot"  # entity id used to compile the robot-only model; MJCF names are prefixed with it
_DOF_PER_ARM = 6

# Tool poses tried when the requested one is out of the wrist's range: turns about the tool's own
# axis (they move the wrist, not the pinch point), then tilts from a ring of directions.
_TURNS = (0.5, -0.5, 1.0, -1.0, 1.6, -1.6, math.pi)
_TILT_FRACTIONS = (0.35, 0.6, 0.85, 1.0)
_TILT_AZIMUTHS = 8


class IkError(RuntimeError):
    """No joint solution reached the requested pose within the iteration budget."""


class CobotIK:
    """FK/IK for one Cobot Magic twin, on the robot's own (URDF-derived) model."""

    def __init__(self, robot: dict | None = None, timestep: float = 0.002) -> None:
        self.robot = robot if robot is not None else cobot_magic().to_dict()
        # Only the robot is compiled: IK must not depend on scene furniture.
        ent = dict(self.robot)
        ent["id"] = _ENT
        build = build_mjcf(SceneState(mode="world", entities=[_as_entity(ent)]), timestep=timestep)
        self.model = build.compile()
        self.data = mink.Configuration(self.model)
        # MuJoCo's qpos0 is the zero posture, not the robot's posture: the lift's legal range does
        # not even contain zero, so a solver seeded from qpos0 stands outside its own limits and
        # every QP is infeasible. Seed from what the model was built with instead -- chassis pose
        # from the entity, arms at their home posture, torso at its home height.
        q = np.array(self.model.qpos0, dtype=float)
        for jname, values in build.initial_qpos.items():
            adr = int(self.model.jnt_qposadr[_joint_id(self.model, jname)])
            for offset, value in enumerate(values):
                q[adr + offset] = float(value)
        self.data.update(q)
        self.dt = timestep
        self._arms = {}
        for arm in ARMS:
            plain = [f"{arm}_joint{i}" for i in range(1, _DOF_PER_ARM + 1)]
            names = [f"{_ENT}::{p}" for p in plain]
            qadrs = []
            dofadrs = []
            for name in names:
                jid = _joint_id(self.model, name)
                qadrs.append(int(self.model.jnt_qposadr[jid]))
                dofadrs.append(int(self.model.jnt_dofadr[jid]))
            self._arms[arm] = {
                "joints": plain,
                "qadr": qadrs,
                "dofadr": dofadrs,
                "ee": f"{_ENT}/{arm}_link6",
                "limits": [(float(self.model.jnt_range[_joint_id(self.model, n)][0]),
                            float(self.model.jnt_range[_joint_id(self.model, n)][1])) for n in names],
            }
        # Every DOF that is not part of a solved arm is frozen: the base's x, y and yaw, the lift,
        # the other arm and both fingers. Freezing the base matters as much as freeing the arm --
        # an IK that may drive the chassis can "reach" any table by walking the robot there.
        all_dofs = [int(self.model.jnt_dofadr[j]) for j in range(self.model.njnt)]
        self._freeze = {}
        for arm in ARMS:
            owned = set(self._arms[arm]["dofadr"])
            self._freeze[arm] = mink.DofFreezingTask(
                self.model, dof_indices=[d for d in all_dofs if d not in owned])
        self._posture = mink.PostureTask(self.model, cost=1e-2)
        self._limit = mink.ConfigurationLimit(self.model)

    # ------------------------------------------------------------------ queries
    def joint_names(self, arm: str) -> list[str]:
        return list(self._arms[self._check(arm)]["joints"])

    def set_base(self, x: float = 0.0, y: float = 0.0, yaw: float = 0.0,
                 lift: float | None = None) -> None:
        """Place the model where the robot is: the chassis pose and the torso height.

        Targets are then given in the world frame, as everything the robot acts on is. A solver
        whose root stayed at the origin would be solving for a robot that never drove.
        """
        q = np.array(self.data.q, dtype=float)
        for name, value in (("root_x", x), ("root_y", y), ("root_yaw", yaw), ("lift", lift)):
            if name == "lift" and value is None:
                continue
            try:
                jid = _joint_id(self.model, f"{_ENT}::{name}")
            except KeyError:  # a fixed-root model has no chassis joints
                continue
            value = float(value)
            lo, hi = self.model.jnt_range[jid]
            if self.model.jnt_limited[jid]:
                value = float(min(max(value, lo), hi))
            q[self.model.jnt_qposadr[jid]] = value
        self.data.update(q)

    def joint_limits(self, arm: str) -> list[tuple[float, float]]:
        return list(self._arms[self._check(arm)]["limits"])

    def neutral(self, arm: str) -> list[float]:
        """A mid-range starting posture: the middle of each joint's URDF range."""
        return [0.5 * (lo + hi) for lo, hi in self.joint_limits(arm)]

    def current(self, arm: str) -> list[float]:
        spec = self._arms[self._check(arm)]
        q = self.data.q
        return [float(q[a]) for a in spec["qadr"]]

    def set_joints(self, arm: str, joints: list[float]) -> None:
        """Seed the solver from measured joint angles (the robot's encoders, not a past solution).

        Measured values are clamped to the URDF ranges. A servo holding against a hard stop
        reports a few ten-thousandths of a radian past the limit, and mink's limit constraint
        reads the whole configuration, not just the arms being solved: one such value in the
        *other* arm makes every solve infeasible (an arm resting on its stop blocked the other
        arm's grasps in the kitchen stacking mission).
        """
        spec = self._arms[self._check(arm)]
        _check_len(joints, _DOF_PER_ARM, f"{arm} arm joints")
        q = np.array(self.data.q, dtype=float)
        for qadr, value in zip(spec["qadr"], self._within_limits(arm, joints), strict=True):
            q[qadr] = value
        self.data.update(q)

    def fk(self, arm: str, joints: list[float] | None = None) -> dict:
        """End-effector pose ``{p: [x, y, z], q: wxyz}`` in the world frame."""
        spec = self._arms[self._check(arm)]
        q = np.array(self.data.q, dtype=float)
        if joints is not None:
            _check_len(joints, _DOF_PER_ARM, f"{arm} arm joints")
            for qadr, value in zip(spec["qadr"], joints, strict=True):
                q[qadr] = value
        self.data.update(q)
        pose = self.data.get_transform_frame_to_world(spec["ee"], "body")
        p = np.asarray(pose.translation())
        rot = np.asarray(pose.rotation().as_matrix())
        return {"p": [float(v) for v in p], "q": _quat_from_matrix(rot)}

    # --------------------------------------------------------------------- ik
    def solve(self, arm: str, position: list[float], orientation: list[float] | None = None,
              q_init: list[float] | None = None, iterations: int = 120,
              tolerance: float = 5e-3, orientation_tolerance: float = 0.05) -> list[float]:
        """Joint angles (6) that put the arm's end-effector at ``position`` (metres, world).

        ``orientation`` is a wxyz quaternion; when omitted only position is constrained.
        Raises :class:`IkError` when the goal is unreachable within ``tolerance``.
        """
        arm = self._check(arm)
        spec = self._arms[arm]
        if q_init is not None:
            _check_len(q_init, _DOF_PER_ARM, f"{arm} arm seed")
        start = self.current(arm) if q_init is None else self._within_limits(arm, q_init)

        q = np.array(self.data.q, dtype=float)
        for qadr, value in zip(spec["qadr"], start, strict=True):
            q[qadr] = value
        self.data.update(q)

        if orientation is not None:
            _check_len(orientation, 4, "orientation quaternion")
        # A position-only goal still needs a target pose; the zero orientation cost ignores it.
        task = mink.FrameTask(spec["ee"], "body", position_cost=1.0,
                              orientation_cost=0.0 if orientation is None else 1.0,
                              gain=0.6, lm_damping=1e-1)
        task.set_target(_se3(position, orientation if orientation is not None else [1.0, 0.0, 0.0, 0.0]))
        self._posture.set_target(self.data.q)
        tasks = [task, self._posture]

        best, best_err, oerr = None, float("inf"), 0.0
        for _ in range(iterations):
            try:
                velocity = mink.solve_ik(self.data, tasks, self.dt, solver="daqp",
                                         limits=[self._limit], constraints=[self._freeze[arm]])
            except mink.NoSolutionFound as exc:  # solver gave up: report the best pose so far
                raise IkError(f"{arm} arm IK found no solution toward {list(position)}") from exc
            self.data.integrate_inplace(velocity, self.dt)
            err, oerr = self._errors(arm, position, orientation)
            if err <= tolerance and oerr <= orientation_tolerance:
                return self.current(arm)
            if err < best_err:
                best, best_err = self.current(arm), err
        raise IkError(
            f"{arm} arm IK missed {list(position)}: {best_err * 1000:.1f} mm after {iterations} steps"
            + ("" if best is None else f" (best q={[round(v, 3) for v in best]}, rotation {oerr:.3f} rad)")
        )

    def reachable(self, arm: str, position: list[float], orientation: list[float] | None = None,
                  q_init: list[float] | None = None) -> bool:
        try:
            self.solve(arm, position, orientation, q_init)
        except IkError:
            return False
        return True

    def solve_tool(self, arm: str, position: list[float], orientation: list[float],
                   max_tilt: float = 0.6, q_init: list[float] | None = None) -> tuple[list[float], float]:
        """Reach ``position`` preferring ``orientation``, but accepting a tilt up to ``max_tilt``.

        The Piper wrist has a +-70 deg roll joint (joint5), so a strictly vertical tool is only
        reachable in a narrow band in front of the base; a real pick turns the tool about its own
        axis, or tilts it a little, instead of refusing. Returns ``(joints, tilt_radians)``; the
        tilt is 0 when the strict orientation was reached.

        What is held fixed is the *pinch point* — ``position`` plus ``TCP_OFFSET`` along the tool
        axis — not the end-effector origin. A tilted tool has to move its wrist the other way by
        ``TCP_OFFSET * sin(tilt)``, or the fingers close up to 8.7 cm away from the object while
        every position report still reads "at the target". That displacement is what used to make
        a reachable place miss by 5 cm and then walk its descent into an unreachable waypoint.

        Turning the tool about its own axis moves the wrist a long way and the pinch point not at
        all, so the turn candidates come first; the tilt sweep follows only if every turn fails.
        Seeds are tried in order — the caller's posture, then home — because a 6-DoF solve started
        from a folded posture can miss a reachable goal entirely.
        """
        seeds = [self.current(arm) if q_init is None else list(q_init), list(HOME), self.neutral(arm)]
        strict_errors: list[str] = []
        for seed in seeds:
            try:
                return self.solve(arm, position, orientation, q_init=seed), 0.0
            except IkError as exc:
                strict_errors.append(str(exc))
        pinch = _along(orientation, position, 1.0)
        failures: list[str] = []
        for turn, tilt, azimuth in _tool_candidates(max_tilt):
            tool = _turn_and_tilt(orientation, turn, tilt, azimuth)
            wrist = _along(tool, pinch, -1.0)
            for seed in seeds[:2]:
                try:
                    # A tight orientation tolerance here is what keeps the pinch point on target:
                    # the wrist target was computed from the candidate tool pose, so an orientation
                    # that lands 0.05 rad off moves the fingers 7 mm.
                    joints = self.solve(arm, wrist, tool, q_init=seed, iterations=80,
                                        orientation_tolerance=0.02)
                except IkError as exc:
                    failures.append(str(exc))
                    continue
                return joints, float(quat_angle_between(self.fk(arm, joints)["q"], orientation))
        detail = (failures or strict_errors)[-1]
        raise IkError(f"{arm} arm cannot reach {[round(v, 3) for v in position]} in a usable tool "
                      f"orientation: {detail}")

    # ------------------------------------------------------------------ private
    def _within_limits(self, arm: str, joints: list[float]) -> list[float]:
        return [min(max(float(v), lo), hi)
                for v, (lo, hi) in zip(joints, self._arms[arm]["limits"])]

    def _errors(self, arm: str, position: list[float], orientation: list[float] | None) -> tuple[float, float]:
        pose = self.data.get_transform_frame_to_world(self._arms[arm]["ee"], "body")
        err = float(np.linalg.norm(np.asarray(pose.translation()) - np.asarray(position)))
        if orientation is None:
            return err, 0.0
        target = _quat_from_matrix(np.asarray(_se3(position, orientation).rotation().as_matrix()))
        got = _quat_from_matrix(np.asarray(pose.rotation().as_matrix()))
        dot = abs(float(np.dot(target, got)))
        return err, 2.0 * float(np.arccos(min(1.0, dot)))

    @staticmethod
    def _check(arm: str) -> str:
        if arm not in ARMS:
            raise KeyError(f"unknown arm {arm!r}; expected one of {ARMS}")
        return arm


def _joint_id(model, name: str) -> int:
    import mujoco

    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise KeyError(f"model has no joint {name!r}")
    return jid


def _as_entity(d: dict):
    from steward.scene.schema import Entity

    return Entity.from_dict(d)


def _check_len(values, n: int, what: str) -> None:
    if len(values) != n:
        raise ValueError(f"{what} must have {n} values, got {len(values)}")


def _se3(position: list[float], orientation: list[float]) -> "mink.SE3":
    from steward.math3d import matrix_from_quat

    rot = np.asarray(matrix_from_quat(list(orientation)), dtype=float).reshape(3, 3)
    m = np.eye(4)
    m[:3, :3] = rot
    m[:3, 3] = np.asarray(position, dtype=float)
    return mink.SE3.from_matrix(m)


def _quat_from_matrix(m: np.ndarray) -> list[float]:
    """Rotation matrix to a wxyz quaternion, normalised."""
    from steward.math3d import quat_from_matrix as _qf

    return [float(v) for v in _qf(np.asarray(m, dtype=float))]


def _along(orientation: list[float], point: list[float], sign: float) -> list[float]:
    """``point`` moved ``sign`` times the tool offset along the orientation's +z axis.

    The pinch point is ``TCP_OFFSET`` along the end-effector's +z, so a tool pose and its pinch
    point are exchanged by this one displacement with ``sign`` +1 or -1.
    """
    from steward.math3d import quat_rotate

    reach = quat_rotate(orientation, TCP_OFFSET)
    return [float(p) + sign * float(r) for p, r in zip(point, reach, strict=True)]


def _turn_and_tilt(orientation: list[float], turn: float, tilt: float, azimuth: float) -> list[float]:
    """``orientation`` with the tool turned about its own axis and then tilted off it."""
    from steward.math3d import quat_from_axis_angle, quat_mul

    tool = quat_mul(orientation, quat_from_axis_angle([0.0, 0.0, 1.0], turn))
    if tilt == 0.0:
        return [float(v) for v in tool]
    axis = [math.cos(azimuth), math.sin(azimuth), 0.0]
    return [float(v) for v in quat_mul(tool, quat_from_axis_angle(axis, tilt))]


def _tool_candidates(max_tilt: float):
    """(turn, tilt, azimuth) tool poses to try, cheapest and most upright first."""
    for turn in _TURNS:
        yield turn, 0.0, 0.0
    for fraction in _TILT_FRACTIONS:
        for i in range(_TILT_AZIMUTHS):
            yield 0.0, max_tilt * fraction, 2.0 * math.pi * i / _TILT_AZIMUTHS
