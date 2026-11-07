"""The planar arm: how the planner and the skills see a body's gripper, whatever the body.

ESim plans and the skills execute in the terms of the mobile kitchen robot's arm: the grasp
point stands ``tool_offset + reach`` ahead of the base centre and ``lateral`` to its left, at
height ``lift`` above the floor; ``wrist_yaw`` turns what the gripper holds about the vertical and
``wrist_roll`` tips it about the forward axis. A body supplies those four virtual joints from its
own encoders (:meth:`Arm.virtualize`) and turns commands on them into commands for its own
actuators (:meth:`Arm.command`). For the kitchen robot both are its real joints; a body with a
6-DoF arm solves inverse kinematics instead (:mod:`steward.ego.cobot_arm`).
"""

from __future__ import annotations

from dataclasses import dataclass

VIRTUAL_JOINTS = ("lift", "reach", "wrist_yaw", "wrist_roll")


@dataclass(frozen=True)
class ArmGeometry:
    """Where the grasp point can go, in planar-arm terms (metres, radians)."""

    tool_offset: float  # grasp point ahead of the base centre at reach 0
    lateral: float  # grasp point to the left of the base centre line
    reach_min: float
    reach_max: float
    lift_min: float
    lift_max: float
    wrist_yaw_max: float
    wrist_roll_max: float
    gripper: str  # the grasp-point link that holds what this arm carries


class Arm:
    """The kitchen robot's own arm: lift, reach slide and a two-axis wrist are real joints, so the
    planar arm is the body itself."""

    def __init__(self, geometry: ArmGeometry):
        self.geometry = geometry

    def virtualize(self, state: dict) -> dict:
        """``state`` (a sensor header: odometry, joints, gripper or grippers) in planar-arm terms:
        ``joints`` holds the four virtual joints, ``gripper`` the arm's gripper, and ``toolOffset``
        / ``armLateral`` its geometry. Remembers the header for :meth:`command`."""
        joints = state.get("joints") or {}
        out = {**state, "toolOffset": self.geometry.tool_offset, "armLateral": self.geometry.lateral,
               "joints": {**joints, **{k: joints[k] for k in VIRTUAL_JOINTS if k in joints}}}
        grippers = state.get("grippers") or {}
        if self.geometry.gripper in grippers:
            out["gripper"] = grippers[self.geometry.gripper]
        return out

    def command(self, cmd: dict) -> dict:
        """A command on the planar arm (``base``, the virtual joints, ``gripper`` open/close) as a
        command for the body."""
        return cmd

    def reachable(self, lift: float, reach: float, wrist_yaw: float = 0.0, wrist_roll: float = 0.0) -> bool:
        """Whether the arm can hold its grasp point at ``lift`` and ``reach`` turned and tipped so."""
        g = self.geometry
        return (g.lift_min <= lift <= g.lift_max and g.reach_min <= reach <= g.reach_max
                and abs(wrist_yaw) <= g.wrist_yaw_max and abs(wrist_roll) <= g.wrist_roll_max)

    def joints_of(self, joints: dict, odometry: dict) -> dict:
        """The virtual joints for encoder readings ``joints`` at ``odometry`` (planning reads the
        Ego-State's self entity this way)."""
        return {k: float(joints[k]) for k in VIRTUAL_JOINTS if k in joints}


def slide_arm(robot_model: dict, reach: tuple[float, float], lift: tuple[float, float],
              wrist_yaw_max: float, wrist_roll_max: float) -> Arm:
    """The kitchen robot's arm, its tool offset read from the model (lift column plus gripper mount)."""
    tree = robot_model["body"]
    lift_joint = next(j for j in tree["joints"] if j["id"] == "lift")
    mount = next(j for j in tree["joints"] if j["id"] == "gripper_mount")
    gripper = next(link["id"] for link in tree["links"]
                   if any((a if isinstance(a, str) else a.get("value")) == "grasp-point"
                          for a in link.get("affordances", [])))
    return Arm(ArmGeometry(tool_offset=lift_joint["origin"]["value"]["p"][0] + mount["origin"]["value"]["p"][0],
                           lateral=0.0, reach_min=reach[0], reach_max=reach[1], lift_min=lift[0], lift_max=lift[1],
                           wrist_yaw_max=wrist_yaw_max, wrist_roll_max=wrist_roll_max, gripper=gripper))
