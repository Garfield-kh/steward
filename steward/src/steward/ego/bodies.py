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
from steward.embodiments import EMBODIMENTS


def robot_model(body: str) -> dict:
    """The body's model as its generator builds it (the same one the scenes compile)."""
    if body == "steward-bot":
        from steward.assets.generators import steward_bot

        return steward_bot().to_dict()
    if body == "cobot-magic":
        from steward.assets.cobot import cobot_magic

        return cobot_magic().to_dict()
    raise ValueError(f"unknown body {body!r}; known: {', '.join(EMBODIMENTS)}")


def body_arm(body: str, model: dict) -> Arm | None:
    """The body's planar arm; None for the kitchen robot, whose own joints are the planar arm."""
    if body == "cobot-magic":
        from steward.ego.cobot_arm import PiperArm

        return PiperArm(model)
    return None


def runtime_config(body: str, base: RuntimeConfig | None = None) -> RuntimeConfig:
    """``base`` (default settings) with the body's planner and motion settings."""
    cfg = base or RuntimeConfig()
    if body != "cobot-magic":
        return cfg
    # The TRACER base is servoed at 0.5 m/s and 1 rad/s (assets.cobot.ROOT_ACTUATOR): commanded faster,
    # it lags its setpoint and every docking misses. The Piper wrist tips a container to 1.2 rad,
    # so it pours at 1.1; a retracted arm stays inside the reach its tool-down workspace has.
    motion = dataclasses.replace(cfg.motion, linear_speed=0.45, angular_speed=0.9, look_angular_speed=0.6,
                                 gentle_speed=0.25, gentle_turn_speed=0.4, retract_reach=0.15, reach_max=0.48,
                                 joint_tol=0.02)
    esim = dataclasses.replace(cfg.esim, pour_roll=1.1)
    return dataclasses.replace(cfg, motion=motion, esim=esim)
