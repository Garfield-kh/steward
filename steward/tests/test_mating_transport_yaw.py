"""An unobservable heading stays unknown while encoder transport is cancelled."""
import math

import numpy as np
import pytest

from steward.assets.generators import steward_bot
from steward.ego.estimator import EgoEstimator, Track
from steward.ego.esim import Esim
from steward.math3d import wrap_angle
from steward.scene.schema import Est
from test_hole_geometry import holes, obj


def tracked_part():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    track = Track('part', {'car_wheel': 1}, np.array([.4, 0., .8]), np.ones(3)*.001,
                  .63, .01, np.array([.05, .05, .016]), 5, 0., 1.)
    est.tracks[track.id] = track
    return est, track


def test_grasp_snapshot_survives_transport_and_is_cleared_on_release():
    est, track = tracked_part()
    est.mark_held(track.id, at=track.pos.tolist())
    assert track.grasp_world_yaw == pytest.approx(.63)
    # The snapshot is fixed, whereas the held pose follows encoders.
    track.yaw = 2.1
    orientation = est._track_entity(track).state_vars['orientation'].value
    assert orientation['graspWorldYawRad'] == pytest.approx(.63)
    assert orientation['directedYawObserved'] is False
    est.release_held()
    assert track.grasp_world_yaw is None
    assert 'graspWorldYawRad' not in est._track_entity(track).state_vars['orientation'].value
    track.yaw = -.8
    est.mark_held(track.id, at=track.pos.tolist())
    assert track.grasp_world_yaw == pytest.approx(-.8)


@pytest.mark.parametrize('transport', [.4, -1.9, math.pi])
def test_single_offcentre_hole_undoes_transport_without_reading_initial_true_yaw(transport):
    held, mate = obj(.63 + transport), obj(-.2)
    held.state_vars = {'orientation': Est(value={'graspWorldYawRad': .63, 'directedYawObserved': False})}
    sim = Esim.__new__(Esim)
    hh, mh = holes([[.02, .01]]), holes([[.06, -.04]])
    offset, yaw, _ = sim._mate_offset_yaw(held, hh, mate, mh)
    assert yaw == pytest.approx(.63)
    # Unknown initial frame bias is not recovered: only the transport delta is undone.
    current_robot, dock = -.7, 1.1
    sim._robot_xy_yaw = lambda es: (0, 0, current_robot)
    sim._arm_joints = lambda es: {'wrist_yaw': .2}
    sim.arm = type('Arm', (), {'geometry': type('Geometry', (), {'wrist_yaw_max': math.pi})()})()
    wrist = sim._wrist_for(None, held, yaw, dock)
    applied = wrap_angle(dock + wrist - current_robot - .2)
    assert wrap_angle(applied + transport) == pytest.approx(0.)
    requested = sim._mate_offset_yaw(held, hh, mate, mh, -1.)
    assert requested[1] == pytest.approx(-1.)
    assert requested[0] != pytest.approx(offset)


def test_multiple_holes_still_constrain_rotation_not_grasp_snapshot():
    held, mate = obj(1.4), obj()
    held.state_vars = {'orientation': Est(value={'graspWorldYawRad': .63})}
    _, yaw, _ = Esim.__new__(Esim)._mate_offset_yaw(held, holes([[-.03, 0], [.03, 0]]),
                                                 mate, holes([[0, -.03], [0, .03]]))
    assert abs(yaw) == pytest.approx(math.pi/2)
