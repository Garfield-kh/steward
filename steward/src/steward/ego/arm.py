"""The planar arm: how the planner and the skills see a body's gripper, whatever the body.

ESim plans and the skills execute in the terms of the mobile kitchen robot's arm: the grasp
point stands ``tool_offset + reach`` ahead of the base centre and ``lateral`` to its left, at
height ``lift`` above the floor; ``wrist_yaw`` turns what the gripper holds about the vertical and
``wrist_roll`` tips it about the forward axis. A body supplies those four virtual joints from its
own encoders (:meth:`Arm.virtualize`) and turns commands on them into commands for its own
actuators (:meth:`Arm.command`). For the kitchen robot both are its real joints; a body with a
6-DoF arm solves inverse kinematics instead (:mod:`steward.ego.cobot_arm`).

Such an arm may also tilt its tool forward, ``wrist_pitch`` about the lateral axis
(``ArmGeometry.wrist_pitch_max``; the kitchen robot's gripper always points down): the tool comes
in at an angle, which reaches farther and leaves the wrist free to tip a container far. A grasp
taken at a pitch is carried at it -- changing the pitch tilts what the gripper holds -- so only a
pick chooses it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

VIRTUAL_JOINTS = ("lift", "reach", "wrist_yaw", "wrist_roll", "wrist_pitch")


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
    wrist_pitch_max: float = 0.0  # forward tilt of the tool; 0: it always points down



class Arm:
    """The kitchen robot's own arm: lift, reach slide and a two-axis wrist are real joints, so the
    planar arm is the body itself."""

    def __init__(self, geometry: ArmGeometry):
        self.geometry = geometry

    def virtualize(self, state: dict) -> dict:
        """``state`` (a sensor header: odometry, joints, gripper or grippers) in planar-arm terms:
        ``joints`` holds the four virtual joints, ``gripper`` the arm's gripper, ``toolOffset`` /
        ``armLateral`` its geometry, and ``armResting`` whether the arm stands in a rest posture
        outside the planar arm's workspace (the next command takes it straight there). Remembers the
        header for :meth:`command`."""
        joints = state.get("joints") or {}
        out = {**state, "toolOffset": self.geometry.tool_offset, "armLateral": self.geometry.lateral,
               "armResting": self.resting(),
               "joints": {**joints, **{k: joints[k] for k in VIRTUAL_JOINTS if k in joints}}}
        grippers = state.get("grippers") or {}
        if self.geometry.gripper in grippers:
            out["gripper"] = grippers[self.geometry.gripper]
        return out

    def command(self, cmd: dict) -> dict:
        """A command on the planar arm (``base``, the virtual joints, ``gripper`` open/close) as a
        command for the body. ``turnOver`` marks the command that turns what the gripper holds over
        in place (:meth:`turn_over`); the slide's wrist rolls there like any other command."""
        return {k: v for k, v in cmd.items() if k != "turnOver"}

    def turn_over(self, lift: float, reach: float, wrist_pitch: float, floor: float, surface_top: float,
                  surface_edge: float) -> float | None:
        """The lowest height, no lower than ``floor``, to which the arm lowers what it holds after
        turning it over (a half roll) at ``lift`` and ``reach`` with the tool tilted ``wrist_pitch``,
        none of its links coming down onto what it is set down on (its top at ``surface_top``, its
        near edge ``surface_edge`` ahead of the base centre); None where it cannot turn it over there. The slide's wrist rolls a half turn anywhere the
        slide reaches: ``floor``."""
        g = self.geometry
        return floor if g.lift_min <= floor <= g.lift_max and g.reach_min <= reach <= g.reach_max else None

    def resting(self) -> bool:
        """Whether the arm stands in a rest posture outside the planar arm's workspace: the slide's
        joints are the planar arm, so never."""
        return False

    def reachable(self, lift: float, reach: float, wrist_yaw: float = 0.0, wrist_roll: float = 0.0,
                  wrist_pitch: float = 0.0) -> bool:
        """Whether the arm can hold its grasp point at ``lift`` and ``reach`` turned, tipped and
        tilted so."""
        g = self.geometry
        return (g.lift_min <= lift <= g.lift_max and g.reach_min <= reach <= g.reach_max
                and abs(wrist_yaw) <= g.wrist_yaw_max and abs(wrist_roll) <= g.wrist_roll_max
                and 0.0 <= wrist_pitch <= g.wrist_pitch_max)

    def pitch_for(self, lift: float, reach: float, wrist_yaw: float = 0.0, wrist_roll: float = 0.0) -> float | None:
        """A forward tilt of the tool at which the arm holds its grasp point at ``lift`` and ``reach``
        turned and tipped so: none where it holds it pointing down, else a little more than the least
        that does (the body's margin); None where no tilt does. A tool that always points down: 0."""
        return 0.0 if self.reachable(lift, reach, wrist_yaw, wrist_roll) else None

    def reachable_path(self, targets: Sequence[dict]) -> bool:
        """Whether the arm can move its grasp point through ``targets`` (virtual joints; turn, tip
        and tilt default to 0) in order without a jump between two of them: the slide's joints
        move straight through each."""
        return all(self.reachable(t["lift"], t["reach"], t.get("wrist_yaw", 0.0), t.get("wrist_roll", 0.0),
                                  t.get("wrist_pitch", 0.0)) for t in targets)

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
