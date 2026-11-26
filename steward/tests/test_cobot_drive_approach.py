"""Replay a rejected screw-board belief through real Piper IK, without model or world truth."""
import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest

from steward.ego.bodies import body_arm, robot_model, runtime_config
from steward.ego.cobot_arm import PiperArm
from steward.ego.esim import Esim
from steward.ego.fasten_plan import tool_estimate
from steward.ego.skills import Job, SkillExecutor
from steward.math3d import matrix_from_quat, quat_from_yaw
from steward.scene.schema import SceneState


def test_recorded_right_driver_has_a_continuous_bit_down_press_and_withdrawal():
    # 2026-11-16 phase0 run: Ego-State 493.328, before rejected receipt rcpt-8.
    scene = SceneState.from_dict(json.loads((Path(__file__).parent / 'fixtures/cobot-held-driver.json').read_text()))
    model = robot_model('cobot-magic')
    arm = PiperArm(model, dataclasses.replace(body_arm('cobot-magic', model).config, arm='R'))
    planner = Esim(model, runtime_config('cobot-magic').esim, arm)
    receipt = planner.predict(scene, 'drive_screw', {'target': 'e:1', 'arm': 'R'})
    assert receipt['feasible'], receipt['issues']
    plan = receipt['plan']
    held = planner._held(scene)
    feature = tool_estimate(held, scene.stamp)
    q = planner._arm_joints(scene)
    heading = planner._robot_xy_yaw(scene)[2]
    original = matrix_from_quat(quat_from_yaw(heading)) @ arm.tool_rotation(
        q['wrist_yaw'], q['wrist_roll'], q['wrist_pitch'])
    axis = original.T @ feature['axis']
    tip = original.T @ (np.asarray(feature['tip']) - feature['grip'])
    rotation = matrix_from_quat(quat_from_yaw(plan['yaw'])) @ arm.tool_rotation(
        plan['wristYaw'], plan['wristRoll'], plan['wristPitch'])
    assert (rotation @ axis)[2] <= -math.cos(planner.config.drive_max_tilt)
    press_tip = np.array([*plan['gripSpot'], plan['pressLift']]) + rotation @ tip
    np.testing.assert_allclose(press_tip[:2], plan['spot'], atol=1e-6)
    assert plan['armApproach'][-1]['wrist_pitch'] == plan['wristPitch']
    assert plan['armApproach'][-1]['wrist_roll'] == plan['wristRoll']
    assert plan['armApproach'][0]['wrist_roll'] == pytest.approx(q['wrist_roll'])


def test_drive_executes_checked_route_and_returns_without_opening_driver_hand(monkeypatch):
    state = {'joints': {'lift': .9}, 'stamp': 0.}
    commands = []
    executor = SkillExecutor(commands.append, lambda: state, lambda eid: eid)
    waypoints = [dict(lift=.9, reach=.3), dict(lift=1., reach=.3), dict(lift=1., reach=.5)]
    plan = dict(armApproach=waypoints, gripSpot=[.5, 0], pressLift=.92, lift=1.,
                target='e:screw', object='e:driver', driveS=.2)
    actions = []
    monkeypatch.setattr(executor, '_dock', lambda job, plan, retract: actions.append(('dock', retract)))
    monkeypatch.setattr(executor, '_align_lateral', lambda _: None)
    monkeypatch.setattr(executor, '_restore_planned_reach', lambda *_: None)
    monkeypatch.setattr(executor, '_ramp_joints', lambda target, _: actions.append(('move', target)))
    monkeypatch.setattr(executor, '_set_joint', lambda name, val: actions.append((name, val)))
    monkeypatch.setattr(executor, '_wait', lambda seconds: actions.append(('dwell', seconds)))
    monkeypatch.setattr(executor, '_backoff', lambda _: actions.append(('backoff',)))
    executor._drive_screw(Job('j', 'drive_screw', 'r'), plan)
    assert actions == [('dock', False), ('move', waypoints[1]), ('move', waypoints[2]),
                       ('lift', .92), ('dwell', .2), ('lift', 1.),
                       ('move', waypoints[1]), ('move', waypoints[0]), ('backoff',)]
    assert commands == []  # In particular, no gripper-open command outside the checked route.
