"""Cobot Magic embodiment: the AgileX dual Piper mobile manipulation platform.

The real machine is a Mobile-ALOHA-style platform: a TRACER 2.0 differential-drive chassis
carrying a height-adjustable torso with two Piper arms (the kit has four — two leader arms for
teleoperation, which autonomy does not use), two wrist cameras and one overhead depth camera.
The twin models the autonomous half of that: a planar base, a lifting carriage, and the two
follower arms.

Arm kinematics are transcribed verbatim from the vendored URDF
``third_party/piper_description/urdf/piper_description_v100_camera.urdf``
(pinned upstream commit in ``third_party/piper_description/PROVENANCE.md``).
Every joint origin, axis and limit matches the URDF; link collision geometry is
approximated by capsules between consecutive joint origins at the zero pose.

Chassis and lift dimensions are MEASURE placeholders pending the physical survey
of the agilex workstation; see ``steward/embodiments/cobot-magic.yaml``. The
chassis is a planar root (absolute x, y, yaw); the carriage rides a prismatic
lift joint above it, and everything the real torso carries — both arm mounts and
the overhead camera — moves with the carriage.
"""

from __future__ import annotations

import math

import numpy as np

from steward.assets.generators import ROBOT_ARM, ROBOT_BODY, TEMPLATE_ID
from steward.math3d import quat_from_axis_angle
from steward.scene.schema import Entity, Joint, Link, RigidTree, exact, pose, primitive

GRIPPER_RGBA = [0.10, 0.10, 0.10, 1.0]

# (id, parent, child, xyz, rpy, lo, hi) — URDF joint1..joint6, all revolute about z.
_PIPER_CHAIN: list[dict] = [
    dict(jid="joint1", parent="arm_base", child="link1", xyz=[0.0, 0.0, 0.123], rpy=[0.0, 0.0, -1.5708],
         lo=-2.618, hi=2.618),
    dict(jid="joint2", parent="link1", child="link2", xyz=[0.0, 0.0, 0.0], rpy=[1.5708, -0.0349, -1.5708],
         lo=0.0, hi=3.14),
    dict(jid="joint3", parent="link2", child="link3", xyz=[0.2836, 0.0287, 0.0], rpy=[0.0, 0.0, 0.066],
         lo=-2.697, hi=0.0),
    dict(jid="joint4", parent="link3", child="link4", xyz=[-0.2422, 0.0685, 0.0], rpy=[-1.5708, 0.0, 1.3826],
         lo=-1.832, hi=1.832),
    dict(jid="joint5", parent="link4", child="link5", xyz=[0.0, 0.0, 0.0], rpy=[1.5708, 0.0, 0.0],
         lo=-1.22, hi=1.22),
    dict(jid="joint6", parent="link5", child="link6", xyz=[0.0, 0.091, 0.0014], rpy=[-1.5708, -1.5708, 0.0],
         lo=-3.14, hi=3.14),
]

# Fingers: prismatic joint7/joint8 out of link6 (position control, intermediate aperture).
_PIPER_FINGERS: list[dict] = [
    dict(jid="joint7", child="link7", xyz=[0.0, 0.0, 0.135], rpy=[1.5708, 0.0, 1.5708], axis=[0.0, 0.0, -1.0],
         lo=0.0, hi=0.05),
    dict(jid="joint8", child="link8", xyz=[0.0, 0.0, 0.135], rpy=[-1.5708, 0.0, 1.5708], axis=[0.0, 0.0, 1.0],
         lo=-0.05, hi=0.0),
]

# hand_camera_joint, verbatim from the URDF (mount frame convention: looks along +x).
_PIPER_HAND_CAMERA = dict(xyz=[0.0, 0.075, 0.03], rpy=[0.0, -0.3, 1.57])

# Grasp geometry, measured in the simulation at the open-aperture posture (see tests/test_cobot.py):
# the pinch point sits 0.135 m along link6's +z, and the fingers separate along link6's y axis,
# 0.05 m per finger. Closed means both finger joints at 0 (the URDF ranges are 0..0.05 and
# -0.05..0), so the aperture is the distance travelled from 0.
TCP_OFFSET = [0.0, 0.0, 0.135]   # in the link6 frame, for either arm
FINGER_OPEN = 0.05               # per finger, metres from closed
FINGER_CLOSED = 0.0

# A natural parked posture (joint1..joint6): the arm folds forward and slightly up, clear of the
# table. The all-zero URDF posture is a folded singularity where IK has no useful gradient, so the
# twin starts here instead; joints 7/8 start open.
HOME = [0.0, 1.57, -1.35, 0.0, 0.0, 0.0]

# TRACER 2.0 platform (product sheet: 685 x 570 x 155 mm, 100 kg payload, 1.6 m/s). MEASURE:
# only the envelope is taken from the datasheet; mass and the sim's actuator gains are tuned to
# behave like the mobile kitchen robot's base, not measured on the real unit.
CHASSIS = dict(length=0.685, width=0.570, height=0.155, mass=30.0)
# Height-adjustable torso: the carriage's height above the chassis top, and its travel. MEASURE.
# The home height is a *working* height, not a stowed one: with the arms only just above a 0.75 m
# tabletop a forearm sweeping over the table edge goes through it (measured: the arm could not
# even fold back to HOME while parked at the table). Arms a quarter metre above the work surface
# is what the lift is for, and it is the posture the real machine works in.
LIFT_RANGE = (0.35, 1.05)
LIFT_HOME = 0.70           # arms at 0.855 m: a hand's width above a 0.75 m table, and the reach
                           # is measurably better than with the torso fully up (probed, not guessed)
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
    "link2": ([0.2836, 0.0287, 0.0], 0.045, 0.9),
    "link3": ([-0.2422, 0.0685, 0.0], 0.040, 0.7),
    "link4": (None, 0.035, 0.5),
    "link5": ([0.0, 0.091, 0.0014], 0.030, 0.4),
    "link6": ([0.0, 0.0, 0.135], 0.035, 0.35),
    "link7": (None, 0.012, 0.06),
    "link8": (None, 0.012, 0.06),
}


def _rpy_quat(rpy: list[float]) -> list[float]:
    """URDF fixed-axis roll-pitch-yaw (radians) to a wxyz quaternion."""
    r, p, y = rpy
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    return [cy * cp * cr + sy * sp * sr,
            cy * cp * sr - sy * sp * cr,
            cy * sp * cr + sy * cp * sr,
            sy * cp * cr - cy * sp * sr]


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


def _piper_arm(prefix: str, mount_p: list[float], camera: bool = True,
               fovy: float = 60.0, width: int = 640, height: int = 480) -> tuple[list[Link], list[Joint]]:
    """One Piper arm (arm_base..link6 + two fingers), link/joint ids prefixed ``L_``/``R_``."""
    links: list[Link] = []
    joints: list[Joint] = []
    for name, spec in _PIPER_LINKS.items():
        if name in ("link7", "link8"):
            continue
        lid = f"{prefix}_{name}"
        affordances = [exact("grasp-point"), exact({"tcp-offset": {"p": TCP_OFFSET, "q": [1.0, 0.0, 0.0, 0.0]}})] \
            if name == "link6" else None
        # The wrist (link6) is visual only: its capsule spans the gap to the pinch point, so a
        # colliding wrist would shove aside the very object the fingers are closing on. The same
        # reasoning the mobile robot uses for its whole arm.
        link = Link(id=lid, name=f"{prefix} {name}", geometry=_link_geometry(spec, ROBOT_ARM, collision=name != "link6"),
                    inertial=exact({"mass": spec[2]}), affordances=affordances or [])
        if name == "link6" and camera:
            cam = _PIPER_HAND_CAMERA
            link.sensors.append({
                "id": f"{prefix}_wrist_camera", "kind": "rgbd",
                "mount": pose(cam["xyz"], _rpy_quat(cam["rpy"])),
                "intrinsics": {"width": width, "height": height, "fovy": fovy}, "rateHz": 10,
            })
        links.append(link)
        if name == "arm_base":
            joints.append(Joint(id=f"{prefix}_mount", parent="frame", child=lid, type=exact("fixed"),
                                origin=exact(pose(mount_p)), q=exact([]), qd=exact([])))
    for j in _PIPER_CHAIN:
        index = int(j["jid"][-1]) - 1
        joints.append(Joint(
            id=f"{prefix}_{j['jid']}", parent=f"{prefix}_{j['parent']}", child=f"{prefix}_{j['child']}",
            type=exact("revolute"), origin=exact(pose(j["xyz"], _rpy_quat(j["rpy"]))),
            axis=exact([0.0, 0.0, 1.0]), limits=exact({"lo": j["lo"], "hi": j["hi"]}),
            q=exact([HOME[index]]), qd=exact([0.0]), dynamics=exact({"damping": 60.0}),
            actuator={"kind": "position", "limits": {"effort": 100.0, "velocity": 3.0}, "gains": {"kp": 1500.0}}))
    for name, spec in _PIPER_LINKS.items():
        if name not in ("link7", "link8"):
            continue
        lid = f"{prefix}_{name}"
        links.append(Link(id=lid, name=f"{prefix} finger {name[-1]}",
                          geometry={"visual": [primitive("box", [0.024, 0.016, 0.06],
                                                         pose([0.0, 0.0, -0.03]), GRIPPER_RGBA)]},
                          inertial=exact({"mass": spec[2]})))
    for j in _PIPER_FINGERS:
        joints.append(Joint(
            id=f"{prefix}_{j['jid']}", parent=f"{prefix}_link6", child=f"{prefix}_{j['child']}",
            type=exact("prismatic"), origin=exact(pose(j["xyz"], _rpy_quat(j["rpy"]))),
            axis=exact(j["axis"]), limits=exact({"lo": j["lo"], "hi": j["hi"]}),
            q=exact([FINGER_OPEN if int(j["jid"][-1]) == 7 else -FINGER_OPEN]), qd=exact([0.0]),
            dynamics=exact({"damping": 5.0}),
            actuator={"kind": "position", "limits": {"effort": 100.0, "velocity": 1.0}, "gains": {"kp": 100.0}}))
    return links, joints


def cobot_magic(separation: float = 0.62, mount_height: float = 0.855,
                top_camera: list[float] | None = None, camera_fovy: float = 60.0) -> Entity:
    """Cobot Magic twin: mobile chassis, lifting torso, two Piper arms, top + dual wrist RGB-D cameras.

    ``separation``/``mount_height``/``top_camera``/``CHASSIS``/``LIFT_*`` are MEASURE placeholders
    (see module docstring); URDF-derived joint data is exact. ``mount_height`` is the arm mounting
    height above the floor with the lift at its home height; the top camera is given in the same
    floor frame and rides the lift, so raising the torso keeps the camera over the table.
    """
    # Centred over the table it watches, so a whole tabletop fits in the frame at this height. A
    # camera whose view cuts the table is not just coarse: a region cut by the image border is only
    # part of something and never becomes a track, so the table would be invisible to the robot's
    # own state for ever -- and raising the camera to fit a bigger table costs the small objects
    # the accuracy they are grasped with (measured: a plate's estimate wanders 2.3 cm at 1.85 m).
    top_camera = list(top_camera) if top_camera is not None else [0.42, 0.05, 1.60]
    chassis = Link(
        id="base", name="tracer chassis",
        geometry={"collision": [primitive("box", [CHASSIS["length"], CHASSIS["width"], CHASSIS["height"]],
                                          pose([0.0, 0.0, CHASSIS["height"] / 2]), ROBOT_BODY)],
                  "visual": [primitive("box", [CHASSIS["length"], CHASSIS["width"], CHASSIS["height"]],
                                       pose([0.0, 0.0, CHASSIS["height"] / 2]), ROBOT_BODY)]},
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
        geometry={"collision": [primitive("box", [0.12, separation + 0.55, 0.45],
                                          pose([-0.13, 0.0, 0.20]), ROBOT_BODY)]},
        inertial=exact({"mass": 40.0}),
        sensors=[{
            "id": "top_camera", "kind": "rgbd",
            # Straight down: the mount's +x (the optical axis) rotates onto world -z. Expressed
            # relative to the carriage, so the camera rises and falls with the torso.
            "mount": pose([top_camera[0], top_camera[1], top_camera[2] - mount_height],
                          quat_from_axis_angle([0, 1, 0], math.pi / 2)),
            "intrinsics": {"width": 640, "height": 480, "fovy": camera_fovy}, "rateHz": 10,
        }],
    )
    links = [chassis, frame]
    # The lift carries the arms and the overhead camera; home keeps the arms where they were
    # before the machine had a torso.
    joints: list[Joint] = [Joint(
        id="lift", parent="base", child="frame", type=exact("prismatic"),
        origin=exact(pose([0.0, 0.0, CHASSIS["height"]])), axis=exact([0.0, 0.0, 1.0]),
        limits=exact({"lo": LIFT_RANGE[0], "hi": LIFT_RANGE[1]}), q=exact([LIFT_HOME]), qd=exact([0.0]),
        dynamics=exact({"damping": 3000.0}),
        # A lead screw holds its height; a soft position spring sags 7 cm under the torso's weight.
        actuator={"kind": "position", "limits": {"effort": 4000.0, "velocity": 0.3}, "gains": {"kp": 120000.0}})]
    for prefix, side in (("L", 1.0), ("R", -1.0)):
        arm_links, arm_joints = _piper_arm(prefix, [0.0, side * separation / 2.0, 0.0], fovy=camera_fovy)
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
