"""Coherent per-view boxes rather than mixed union-yaw/median-size geometry."""
import math
from itertools import product

import numpy as np
import pytest

from steward.ego.hole_registration import coherent_interface_box
from steward.ego.priors import SCREW_HOLE_FRAMES, screw_holes
from steward.ego.estimator import EgoEstimator, _voxelize
from steward.assets.generators import steward_bot
from test_place_align_holes import _track
from steward.math3d import wrap_angle


def rows(angles, swap=False):
    return [(np.array([i*.001, 0, .8]), np.array([.18, .045, .08]) if not swap else np.array([.045, .18, .08]),
             1., math.radians(a)-(math.pi/2 if swap else 0), True) for i, a in enumerate(angles)]


def solve(r):
    return coherent_interface_box(r, SCREW_HOLE_FRAMES['airplane_fuselage'], screw_holes('airplane_fuselage'))


@pytest.mark.parametrize('angle', [-178, -71, 0, 23, 87, 175])
@pytest.mark.parametrize('swap', [False, True])
def test_rotation_and_axis_order_equivariance(angle, swap):
    c, size, yaw = solve(rows([angle-1, angle, angle+1], swap))
    assert abs((yaw-math.radians(angle)+math.pi/2) % math.pi-math.pi/2) < 1e-8
    assert size == pytest.approx([.18, .045, .08])


def test_wrap_outlier_and_far_view_do_not_replace_consensus():
    r = rows([179, -179, 180, 178, 90])
    r.append((np.zeros(3), np.array([.18, .045, .08]), 2., math.pi/2, True))
    assert abs(abs(solve(r)[2])-math.pi) < math.radians(2)


@pytest.mark.parametrize('r', [rows([0, 0, 90, 90]), rows([0, 0]),
                             [(a, b, c, d, False) for a,b,c,d,_ in rows([0,0,0])]])
def test_insufficient_or_contradictory_axes_stay_unregistered(r):
    assert solve(r) is None


def test_real_rebuild_ignores_perpendicular_union_axis_and_registers_consensus():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    b = _track('body', 'airplane_fuselage', [0, 0, .8], [.18, .045, .08])
    # Masks individually long along X, displaced along Y in the accumulated cloud.
    points = np.array(list(product(np.linspace(-.09,.09,7), np.linspace(-.12,.12,13), [.76,.84])))
    b.voxels = _voxelize(points)
    b.recent.extend(rows([0, 1, -1, 0]))
    est.tracks[b.id] = b
    est._rebuild_geometry(b, np.array([0, -1, 1.2]), 1.)
    assert b.interface_axis_consistent is True
    assert abs(wrap_angle(b.yaw)) < .01
    info = est._track_entity(b).state_vars['orientation'].value
    assert info['holeRegistration']['status'] == 'axis_registered'
    b.recent.clear(); b.recent.extend(rows([0, 0, 90, 90]))
    est._rebuild_geometry(b, np.array([0, -1, 1.2]), 1.)
    assert b.interface_axis_consistent is False
    info = est._track_entity(b).state_vars['orientation'].value['holeRegistration']
    assert info['status'] == 'axis_registered' and info['warnings']
