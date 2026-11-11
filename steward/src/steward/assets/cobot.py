"""AgileX twin with manufacturer USD arm/lift kinematics and primitive geometry.

``embodiments/cobot-magic.yaml`` records the source digest, joint mapping and unit
assumption. The USD base frame is raised to put the lowest wheel mesh point on the
floor; this is a geometric estimate, not a measured real-robot calibration. USD
joint-frame rotations are converted to child-link frames, with sub-micrometre
mount residuals rounded away. The lift reads metres above its lowest position.

The planar chassis, fixed two-camera head, masses, servo gains and simplified
collision geometry are simulation approximations. Fingers remain visual-only;
grasping uses ideal attachment and an operational, uncalibrated TCP.
"""

from __future__ import annotations

import math

import numpy as np

from steward.assets.generators import HEAD_HEIGHT_M, ROBOT_ARM, ROBOT_BODY, STEEL, TEMPLATE_ID, _fixed, head_link
from steward.math3d import quat_from_axis_angle, quat_normalize
from steward.scene.schema import Entity, Joint, Link, RigidTree, exact, pose, primitive

GRIPPER_RGBA = [0.10, 0.10, 0.10, 1.0]

# USD fl/fr joint1..joint6; quaternions are wxyz, angles and limits are radians.
_PIPER_CHAIN: list[dict] = [
    dict(jid="joint1", parent="arm_base", child="link1", xyz=[0.0, 0.0, 0.123], quat=[0.7071055174, 0.0, 0.0, -0.7071080804],
         lo=-2.618, hi=2.618),
    dict(jid="joint2", parent="link1", child="link2", xyz=[0.0, 0.0, 0.0], quat=[0.5086483955, 0.4911975563, -0.5086519718, -0.4911975563],
         lo=0.0, hi=3.14),
    dict(jid="joint3", parent="link2", child="link3", xyz=[0.28358, 0.028726, 0.0], quat=[0.9994548559, 0.0, 0.0, 0.0330157019],
         lo=-2.697, hi=0.0),
    dict(jid="joint4", parent="link3", child="link4", xyz=[-0.24221, 0.068514, 0.0], quat=[0.5447667241, -0.5447686911, -0.4508091807, 0.4508075416],
         lo=-1.832, hi=1.832),
    dict(jid="joint5", parent="link4", child="link5", xyz=[0.0, 0.0, 0.0], quat=[0.7071055174, 0.7071080804, 0.0, 0.0],
         lo=-1.22, hi=1.22),
    dict(jid="joint6", parent="link5", child="link6", xyz=[0.0, 0.091, 0.0014165], quat=[0.4999982119, -0.5, -0.5, -0.5000018477],
         lo=-3.14, hi=3.14),
]

# Fingers: prismatic joint7/joint8 out of link6 (position control, intermediate aperture).
_PIPER_FINGERS: list[dict] = [
    dict(jid="joint7", child="link7", xyz=[0.0, 0.0, 0.13503], quat=[0.4999982119, 0.5, 0.5000018477, 0.5], axis=[0.0, 0.0, -1.0],
         lo=0.0, hi=0.05),
    dict(jid="joint8", child="link8", xyz=[0.0, 0.0, 0.13503], quat=[0.4999982119, -0.5, -0.5000018477, 0.5], axis=[0.0, 0.0, 1.0],
         lo=-0.05, hi=0.0),
]

# Where the head (the two head cameras the mobile kitchen robot carries) sits on the chassis: a
# mast at the front centre line, between the arm mounts, at the same height as on that robot.
HEAD_AT = [0.16, 0.0]
# USD base_link -> Steward base: identity rotation, z from the wheel mesh lower bound.
USD_BASE_HEIGHT = 0.32052979804575443
USD_LIFT_ORIGIN = [0.22448, 0.0, 1.1147]
ARM_MOUNT_OFFSET = [0.144, 0.306, 0.138]  # left arm relative to the lift carriage
ARM_MOUNT_X = USD_LIFT_ORIGIN[0] + ARM_MOUNT_OFFSET[0]
ARM_SEPARATION = 2 * ARM_MOUNT_OFFSET[1]

# Grasp geometry, measured in the simulation at the open-aperture posture (see tests/test_cobot.py):
# the operational pinch point lies in the finger-joint plane along link6's +z;
# the fingers separate along link6's x axis,
# 0.05 m per finger. Closed means both finger joints at 0 (the URDF ranges are 0..0.05 and
# -0.05..0), so the aperture is the distance travelled from 0.
TCP_OFFSET = list(_PIPER_FINGERS[0]["xyz"])  # operational TCP, not a manufacturer calibration
FINGER_OPEN = 0.05               # per finger, metres from closed
FINGER_CLOSED = 0.0

# A working posture (joint1..joint6): the arm folds forward and slightly up, clear of the table.
# The all-zero URDF posture is a folded singularity where IK has no useful gradient, so solves
# are seeded here; joints 7/8 start open. It stands in front of the head cameras, so it is not
# where an arm rests.
HOME = [0.0, 1.57, -1.35, 0.0, 0.0, 0.0]
# Where an arm rests (the left arm; the right turns the other way): turned out a quarter turn and
# folded upright over its mount, inside the chassis width, so neither head camera sees more of it
# than the mount (measured: the arms covered 6 % of the survey camera's frame and 23 % of the work
# camera's at HOME, none and 5 % here).
PARK = [1.57, 0.9, -2.6, 0.0, 1.0, 0.0]


def park(arm: str) -> list[float]:
    """The rest posture of arm ``L`` or ``R`` (joint1..joint6)."""
    return list(PARK) if arm == "L" else [-PARK[0], *PARK[1:]]

# TRACER 2.0 platform (product sheet: 685 x 570 x 155 mm, 100 kg payload, 1.6 m/s). MEASURE:
# only the envelope is taken from the datasheet; mass and the sim's actuator gains are tuned to
# behave like the mobile kitchen robot's base, not measured on the real unit.
CHASSIS = dict(length=0.685, width=0.570, height=0.155, mass=30.0)
# lift = q_USD + 0.60; zero is the lowest carriage position, in metres.
LIFT_RANGE = (0.0, 0.60)
LIFT_HOME = 0.0
ARM_MOUNT_HEIGHT = USD_BASE_HEIGHT + USD_LIFT_ORIGIN[2] - LIFT_RANGE[1] + ARM_MOUNT_OFFSET[2]
# The base drives like the mobile kitchen robot: position control on absolute x, y, yaw.
ROOT_ACTUATOR = {
    "kind": "position",
    # Larger than the kitchen robot's: this base carries a 40 kg torso and two arms on a wider
    # footprint, and a yaw servo sized for the other robot simply stalls (measured: 0.32 rad short
    # with the effort saturated).
    "limits": {"effort": [1200.0, 1200.0, 700.0], "velocity": [0.5, 0.5, 1.0]},
    "gains": {"kp": [10000.0, 10000.0, 3500.0], "damping": [1800.0, 1800.0, 500.0]},
}

# Per-link capsule spec: link id -> (offset to the next joint origin in this link's frame,
# capsule radius, mass). None -> a sphere at the origin. Fingers are visual-only.
_PIPER_LINKS: dict[str, tuple] = {
    "arm_base": ([0.0, 0.0, 0.123], 0.050, 1.2),
    "link1": (None, 0.050, 0.9),
    "link2": (_PIPER_CHAIN[2]["xyz"], 0.045, 0.9),
    "link3": (_PIPER_CHAIN[3]["xyz"], 0.040, 0.7),
    "link4": (None, 0.035, 0.5),
    "link5": (_PIPER_CHAIN[5]["xyz"], 0.030, 0.4),
    "link6": (TCP_OFFSET, 0.035, 0.35),
    "link7": (None, 0.012, 0.06),
    "link8": (None, 0.012, 0.06),
}


def _quat_z_to(direction: np.ndarray) -> list[float]:
    """wxyz quaternion rotating the +z axis onto a unit direction."""
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, direction)
    s = float(np.linalg.norm(axis))
    c = float(np.dot(z, direction))
    if s < 1e-9:
        return [0.0, 1.0, 0.0, 0.0] if c < 0 else [1.0, 0.0, 0.0, 0.0]
    return quat_from_axis_angle(axis / s, math.atan2(s, c))


def _link_geometry(spec: tuple, rgba: list[float], collision: bool = True) -> dict[str, list[dict]]:
    key = "collision" if collision else "visual"
    offset, radius, _ = spec
    if offset is None:
        return {key: [primitive("sphere", [radius], None, rgba)]}
    a = np.zeros(3)
    b = np.asarray(offset, dtype=float)
    length = float(np.linalg.norm(b - a))
    if length < 1e-6:
        return {key: [primitive("sphere", [radius], None, rgba)]}
    center = (a + b) / 2.0
    at = pose([float(v) for v in center], _quat_z_to(b / length))
    return {key: [primitive("capsule", [radius, length], at, rgba)]}


def _palm_geometry() -> dict[str, list[dict]]:
    """Wrist flange, palm housing and transverse guide in the link6 frame (metres)."""
    return {"visual": [
        primitive("cylinder", [0.027, 0.025], pose([0.0, 0.0, 0.0125]), ROBOT_ARM),
        primitive("box", [0.138, 0.048, 0.044], pose([0.0, 0.0, 0.041]), GRIPPER_RGBA),
        primitive("box", [0.138, 0.024, 0.008], pose([0.0, 0.0, 0.067]), STEEL),
    ]}


def _finger_geometry(name: str) -> dict[str, list[dict]]:
    """Slider, axial finger and inward pad in the mirrored URDF link7/link8 frames.

    Local +/-y points back toward the palm; local -z points away from the pinch
    axis on both fingers. Pad inner faces meet at zero travel, with the TCP
    between the pads, 5 mm behind their tips. Dimensions are in metres.
    """
    toward_palm = -1.0 if name == "link7" else 1.0
    return {"visual": [
        primitive("box", [0.028, 0.014, 0.022],
                  pose([0.0, toward_palm * 0.064, -0.011]), GRIPPER_RGBA),
        primitive("box", [0.024, 0.064, 0.010],
                  pose([0.0, toward_palm * 0.027, -0.008]), STEEL),
        primitive("box", [0.020, 0.026, 0.003],
                  pose([0.0, toward_palm * 0.008, -0.0015]), GRIPPER_RGBA),
    ]}


def _piper_arm(prefix: str, mount_p: list[float]) -> tuple[list[Link], list[Joint]]:
    """One Piper arm (arm_base..link6 + two fingers), link/joint ids prefixed ``L_``/``R_``."""
    links: list[Link] = []
    joints: list[Joint] = []
    for name, spec in _PIPER_LINKS.items():
        if name in ("link7", "link8"):
            continue
        lid = f"{prefix}_{name}"
        affordances = [exact("grasp-point"), exact({"tcp-offset": {"p": TCP_OFFSET, "q": [1.0, 0.0, 0.0, 0.0]}})] \
            if name == "link6" else None
        # The gripper is visual-only; grasping uses the link6 TCP and ideal attachment.
        geometry = _palm_geometry() if name == "link6" else _link_geometry(spec, ROBOT_ARM)
        link = Link(id=lid, name=f"{prefix} {name}", geometry=geometry,
                    inertial=exact({"mass": spec[2]}), affordances=affordances or [])
        links.append(link)
        if name == "arm_base":
            joints.append(Joint(id=f"{prefix}_mount", parent="frame", child=lid, type=exact("fixed"),
                                origin=exact(pose(mount_p)), q=exact([]), qd=exact([])))
    for j in _PIPER_CHAIN:
        joints.append(Joint(
            id=f"{prefix}_{j['jid']}", parent=f"{prefix}_{j['parent']}", child=f"{prefix}_{j['child']}",
            type=exact("revolute"), origin=exact(pose(j["xyz"], quat_normalize(j["quat"]))),
            axis=exact([0.0, 0.0, 1.0]), limits=exact({"lo": j["lo"], "hi": j["hi"]}),
            q=exact([0.0]), qd=exact([0.0]), dynamics=exact({"damping": 60.0}),
            actuator={"kind": "position", "limits": {"effort": 100.0, "velocity": 3.0}, "gains": {"kp": 1500.0}}))
    for name, spec in _PIPER_LINKS.items():
        if name not in ("link7", "link8"):
            continue
        lid = f"{prefix}_{name}"
        links.append(Link(id=lid, name=f"{prefix} finger {name[-1]}",
                          geometry=_finger_geometry(name),
                          inertial=exact({"mass": spec[2]})))
    for j in _PIPER_FINGERS:
        joints.append(Joint(
            id=f"{prefix}_{j['jid']}", parent=f"{prefix}_link6", child=f"{prefix}_{j['child']}",
            type=exact("prismatic"), origin=exact(pose(j["xyz"], quat_normalize(j["quat"]))),
            axis=exact(j["axis"]), limits=exact({"lo": j["lo"], "hi": j["hi"]}),
            q=exact([FINGER_OPEN if int(j["jid"][-1]) == 7 else -FINGER_OPEN]), qd=exact([0.0]),
            dynamics=exact({"damping": 5.0}),
            actuator={"kind": "position", "limits": {"effort": 100.0, "velocity": 1.0}, "gains": {"kp": 100.0}}))
    return links, joints


def cobot_magic(separation: float = ARM_SEPARATION, mount_height: float = ARM_MOUNT_HEIGHT) -> Entity:
    """Cobot Magic twin: mobile chassis, lifting torso, two Piper arms, and the mobile kitchen
    robot's head (:func:`~steward.assets.generators.head_link`: a survey camera and a
    short-focal-length work camera) on a mast fixed to the chassis.
    Both arms start with joint1–joint6 at zero radians and their fingers open.

    ``separation`` is the arm-root spacing; ``mount_height`` is their height above the floor
    at ``LIFT_HOME``. Both are metres and override the USD-derived defaults. The lift reads
    displacement above its lowest position, not arm height. The head stays fixed to the chassis.
    """
    mast = HEAD_HEIGHT_M - CHASSIS["height"]
    chassis = Link(
        id="base", name="tracer chassis",
        geometry={"collision": [primitive("box", [CHASSIS["length"], CHASSIS["width"], CHASSIS["height"]],
                                          pose([0.0, 0.0, CHASSIS["height"] / 2]), ROBOT_BODY)],
                  "visual": [primitive("box", [CHASSIS["length"], CHASSIS["width"], CHASSIS["height"]],
                                       pose([0.0, 0.0, CHASSIS["height"] / 2]), ROBOT_BODY),
                             primitive("box", [0.05, 0.05, mast],
                                       pose([HEAD_AT[0] - 0.04, HEAD_AT[1], CHASSIS["height"] + mast / 2]), STEEL)]},
        inertial=exact({"mass": CHASSIS["mass"]}),
        # Wheels roll: a slab dragging on the floor at the robot body's friction cannot be turned
        # in place, which a differential-drive base does all the time.
        material=exact({"friction": 0.1}),
    )
    frame = Link(
        id="frame", name="cobot magic lift carriage",
        # The carriage is a slim back plate behind the arms, not a full-depth box: the workspace in
        # front of the arms is table, and a vertical tool can only reach down onto it about
        # 0.2-0.35 m out, so the torso must stay behind that band.
        # Kept clear of the arm mounts: plates that start at the mount face collide with the folded
        # arm (measured: the shoulder could not return to HOME, held 0.04 rad short by a contact).
        # Between the mounts and inside the chassis width: the planner keeps the chassis footprint
        # clear, and a plate wider than it struck walls the base was cleared to pass.
        geometry={"collision": [primitive("box", [0.12, min(separation - 0.12, CHASSIS["width"]), 0.45],
                                          pose([ARM_MOUNT_OFFSET[0] - 0.13, 0.0, ARM_MOUNT_OFFSET[2] + 0.20]), ROBOT_BODY)]},
        inertial=exact({"mass": 40.0}),
    )
    links = [chassis, frame, head_link()]
    # The USD base offset and lift zero shift are folded into the carriage origin.
    joints: list[Joint] = [_fixed("head_mount", "base", "head", pose([HEAD_AT[0], HEAD_AT[1], HEAD_HEIGHT_M])), Joint(
        id="lift", parent="base", child="frame", type=exact("prismatic"),
        origin=exact(pose([USD_LIFT_ORIGIN[0], 0.0, mount_height - LIFT_HOME - ARM_MOUNT_OFFSET[2]])),
        axis=exact([0.0, 0.0, 1.0]),
        limits=exact({"lo": LIFT_RANGE[0], "hi": LIFT_RANGE[1]}), q=exact([LIFT_HOME]), qd=exact([0.0]),
        dynamics=exact({"damping": 3000.0}),
        # A lead screw holds its height; a soft position spring sags 7 cm under the torso's weight.
        actuator={"kind": "position", "limits": {"effort": 4000.0, "velocity": 0.3}, "gains": {"kp": 120000.0}})]
    for prefix, side in (("L", 1.0), ("R", -1.0)):
        arm_links, arm_joints = _piper_arm(prefix, [ARM_MOUNT_OFFSET[0], side * separation / 2.0, ARM_MOUNT_OFFSET[2]])
        links.extend(arm_links)
        joints.extend(arm_joints)
    return Entity(
        id=TEMPLATE_ID,
        role="robot",
        category=exact("cobot-magic"),
        pose=exact(pose()),
        body=RigidTree(root=exact("planar"), links=links, joints=joints, root_actuator=ROOT_ACTUATOR),
        state_vars={"gripper_left": exact("open"), "gripper_right": exact("open")},
    )
