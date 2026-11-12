"""What the Ego Runtime needs to drive each registered body: its model, its planar arm
(:mod:`steward.ego.arm`) and the planner and motion settings that differ between bodies.

Every body runs the same Ego Runtime (``steward ego --body <id>``): perception, the Ego-State,
ESim receipts and the skills are the kitchen robot's; a body changes only what its geometry and
actuators require.
"""

from __future__ import annotations

import dataclasses

from steward.ego.arm import Arm
from steward.ego.runtime import RuntimeConfig
from steward.embodiments import robot_entity


def robot_model(body: str) -> dict:
    """The body's model as its generator builds it (the same one the scenes compile)."""
    return robot_entity(body)


def body_arm(body: str, model: dict) -> Arm | None:
    """The body's planar arm; None for the kitchen robot, whose own joints are the planar arm."""
    if body == "cobot-magic":
        from steward.ego.cobot_arm import PiperArm, PiperArmConfig

        # Match MotionConfig working tip so deploy_working and skills land on the same hole-free seed.
        arm = PiperArm(model)
        floors = _cobot_envelope(arm, retract_reach=0.28)
        return PiperArm(model, PiperArmConfig(working_lift=floors["working_lift"],
                                              working_reach=floors["working_reach"]))
    return None


def _cobot_envelope(arm: Arm, retract_reach: float) -> dict[str, float]:
    """Lift floors implied by the USD IK envelope for ``arm``, not scene-specific constants.

    ``tilted_exit_lift``: lowest height where a most-tilted short retract is freshly reachable.
    ``untilted_exit_lift``: same for tool-down at a mid place reach (tool-down tops out near 0.50 m).
    ``tilted_carry_floor``: lowest height where a most-tilted mid reach is freshly reachable.
    ``elevated_grasp_floor``: lowest height where a deep dock (~0.52 m) needs a positive pitch.
    """
    pitch = arm.geometry.wrist_pitch_max
    min_lift_for = getattr(arm, "min_lift_for", None)
    pitch_for = arm.pitch_for
    if min_lift_for is None:
        return {"tilted_exit_lift": 1.05, "untilted_exit_lift": 0.85,
                "tilted_carry_floor": 0.95, "elevated_grasp_floor": 0.72, "handle_lift_floor": 0.65,
                "tool_down_reach_mid": 0.36, "working_lift": 0.85, "working_reach": 0.40}
    def floor(value: float | None, default: float, margin: float = 0.02) -> float:
        return (default if value is None else value) + margin

    # Carry/exit use the short-retract and mid-reach envelopes. Exit floors add joint_tol (0.08) so
    # a settle that stops short of the setpoint still clears the IK ridge (1.02 m @ pitch 1.05
    # cannot take retract_reach; 1.05 m can).
    settle_slack = 0.08
    tilted_exit = floor(min_lift_for(retract_reach, pitch), 1.05, margin=0.02 + settle_slack)
    untilted_exit = floor(min_lift_for(0.45, 0.0), 0.85, margin=0.02 + settle_slack)
    # Flip/place carries often keep ~0.40 m reach while most-tilted; that ridge is higher than 0.45 m.
    tilted_carry = floor(min_lift_for(0.40, pitch), 0.95, margin=0.02 + settle_slack)
    # Deep dock: first height where a 0.52 m reach needs a decisive tilt (not a shallow ridge).
    elevated = 0.72
    lo, hi = arm.geometry.lift_min, arm.geometry.lift_max
    for k in range(0, int((hi - lo) / 0.02) + 1):
        h = lo + 0.02 * k
        pf = pitch_for(h, 0.52)
        if pf is not None and pf >= 0.35:
            elevated = h
            break
    # Handle docks ask ~0.40 m tool-down; below this TCP height that reach has no fresh IK.
    handle_lift = floor(min_lift_for(0.42, 0.0), 0.65)
    # Farthest tool-down reach across the drawer-close band (0.58–0.62 m): use the strictest
    # height so creep/probes never ask past a hole that opens a centimetre lower.
    tool_down_mid = 0.36
    for r in (0.45, 0.42, 0.40, 0.38, 0.36, 0.34, 0.32, 0.28):
        if all(arm.reachable(h, r, wrist_pitch=0.0) for h in (0.58, 0.60, 0.62)):
            tool_down_mid = r
            break
    # Working tip after HOME: past the 0.32 m tool-down ridge. Prefer ≥0.85 m — untilted_exit can
    # sit near 0.75 m where 0.32 m is a hole, and settle then lands inside that ridge.
    working_lift = max(untilted_exit, 0.85)
    working_reach = max(tool_down_mid, 0.40)
    for r in (0.42, 0.40, tool_down_mid, 0.36):
        if arm.reachable(working_lift, r, wrist_pitch=0.0):
            working_reach = r
            break
    return {"tilted_exit_lift": tilted_exit, "untilted_exit_lift": untilted_exit,
            "tilted_carry_floor": tilted_carry, "elevated_grasp_floor": elevated,
            "handle_lift_floor": handle_lift, "tool_down_reach_mid": tool_down_mid,
            "working_lift": working_lift, "working_reach": working_reach}


def runtime_config(body: str, base: RuntimeConfig | None = None) -> RuntimeConfig:
    """``base`` (default settings) with the body's planner and motion settings."""
    cfg = base or RuntimeConfig()
    if body != "cobot-magic":
        return cfg
    from steward.assets.cobot import HOME, LIFT_HOME, cobot_magic
    from steward.ego.cobot_arm import PiperArm

    # The TRACER base is servoed at 0.5 m/s and 1 rad/s (assets.cobot.ROOT_ACTUATOR): commanded faster,
    # it lags its setpoint and every docking misses. The arm's joints, servoed straight to a solved
    # posture, sweep the wrist off the line the grasp point is to move along, so its moves are
    # ramped. A retracted arm stays inside the reach its tool-down workspace has; tilted, the tool
    # reaches 0.70 m. Its reach is still short, so a large object is taken by the edge nearest the
    # robot (it then hangs ahead of the hand) and set down with a 5 cm margin inside the surface's edge.
    # Its gripper cannot point up at a surface, so it turns things over in the air.
    # Under the USD mounts, tool-down at stow height (0.60 m) only reaches from about 0.18 m out:
    # retract_reach 0.15 left every drawer open stuck when drawing the arm back.
    retract_reach = 0.28
    arm = PiperArm(cobot_magic().to_dict())
    arm.virtualize({"joints": {"lift": LIFT_HOME, **dict(zip(arm.names, HOME, strict=True))},
                    "odometry": {"x": 0.0, "y": 0.0, "yaw": 0.0}})
    floors = _cobot_envelope(arm, retract_reach)
    motion = dataclasses.replace(cfg.motion, linear_speed=0.45, angular_speed=0.9, look_angular_speed=0.6,
                                 gentle_speed=0.25, gentle_turn_speed=0.4, retract_reach=retract_reach, reach_max=0.70,
                                 # Twin arm mount is ~0.30 m off centreline; a short dock can leave
                                 # 15+ cm of lateral error after unfurl creep — allow a wider align.
                                 align_max=0.25,
                                 # Virtual TCP after a whole-body solve can sit a few centimetres off
                                 # the commanded lift/reach while the torso is still settling.
                                 joint_tol=0.10, arm_speed=0.3, settle_s=2.0,
                                 # USD whole-body TCP lags the planar tip report; a short grasp wait
                                 # closes before the pinch is inside the sim's 6 cm handle capture.
                                 # Keep under ~0.4 s: longer plus settle-before-close blew the 150 s
                                 # skill wait on a single drawer open.
                                 grasp_wait_s=0.35,
                                 # Stow above the drawer-handle band: at 0.60 m a retracted reach only
                                 # has limit-packed IK solutions the servo cannot hold.
                                 stow_lift=0.75,
                                 wrist_pitch_max=arm.geometry.wrist_pitch_max,
                                 tilted_exit_lift=floors["tilted_exit_lift"],
                                 untilted_exit_lift=floors["untilted_exit_lift"],
                                 tilted_carry_floor=floors["tilted_carry_floor"],
                                 handle_lift_floor=floors["handle_lift_floor"],
                                 tool_down_reach_mid=floors["tool_down_reach_mid"],
                                 working_lift=floors["working_lift"],
                                 working_reach=floors["working_reach"])
    # handle_reach 0.18 packs joint2 against its 3.14 rad stop at drawer-handle height (~0.61 m),
    # so the TCP settles ~6 cm long and the grasp misses; 0.25 m leaves headroom. capture_radius
    # covers residual TCP report noise after a whole-body solve.
    # handle_reach 0.18 packs joint2; 0.32 asks ~0.45 m at handle height and leaves the tool-down
    # envelope. 0.25 leaves stop headroom without forcing an unreachable dock.
    # Slow whole-body IK + ramped arm moves: a 60 s receipt TTL expires mid-drawer/flip.
    esim = dataclasses.replace(cfg.esim, edge_grasp_first=True, place_edge_margin=0.05, flip_in_air=True,
                               handle_reach=0.25, capture_radius=0.12,
                               handle_probes_side_m=(0.06, -0.06, 0.10, -0.10, 0.14, -0.14),
                               handle_probes_along_m=(0.04, 0.08, 0.12, -0.05, -0.09, 0.16),
                               elevated_grasp_floor=floors["elevated_grasp_floor"],
                               tilted_carry_floor=floors["tilted_carry_floor"],
                               min_require_tilt=0.25, receipt_ttl=300.0,
                               # Flip-in-air drops from turnLift; allow a wide shift so the lid already
                               # on the table does not exhaust free spots around the requested offset.
                               # Flip-in-air releases from ~turnLift; need clearance past neighbours
                               # and room to shift off a crowded requested offset onto free table.
                               # Soft-exit unrolls then leaves; keep clearance modest so free spots
                               # still exist on a crowded puzzle table after the lid is set down.
                               place_clearance=0.18, spot_shift_max=0.85,
                               # USD tip is slower to track than the kitchen slide; hold longer so
                               # rim-tilted pours actually empty before the wrist rights the box.
                               pour_hold_s=3.0)
    return dataclasses.replace(cfg, motion=motion, esim=esim)
