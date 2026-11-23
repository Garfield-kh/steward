"""Offline posture and two-feature orientation regressions; no physics/HTTP."""
import math

import pytest

from steward.ego.hole_registration import assembly_posture
from steward.math3d import quat_from_yaw, wrap_angle
from steward.scene.schema import Est
from test_place_align_holes import _assembly_scene, _esim


def test_shape_alarm_preserves_vertical_axis_but_never_certifies_upright():
    template = [.24, .04, .09]
    assert assembly_posture([.04, .24, .09], template)['dimensionsCompatible'] is True
    assert assembly_posture(template, template)['upright'] is None
    side = assembly_posture([.24, .09, .04], template, slack_m=.005)
    assert side['status'] == 'inconsistent'
    assert side['upright'] is None  # May also be occlusion/merged mask, not a measured roll.
    assert assembly_posture([.24, .04, float('nan')], template)['status'] == 'unknown'


@pytest.mark.parametrize('params', [{'flip': True}, {'roll': math.pi / 2}])
def test_planar_mating_refuses_to_turn_the_load_over(params):
    model, scene = _assembly_scene()
    receipt = _esim(model).predict(scene, 'place', {'on': 'e:fuse', **params})
    assert not receipt['feasible']
    assert 'RSI_ASSEMBLY_POSTURE' in ' '.join(receipt['issues'])


@pytest.mark.parametrize('ordinary', [False, True])
def test_suspect_mate_cannot_be_bypassed_by_manual_placement(ordinary):
    model, scene = _assembly_scene()
    scene.entity('e:fuse').state_vars['orientation'].value['posture']['status'] = 'inconsistent'
    result = _esim(model).predict(scene, 'place', {'on': 'e:fuse', 'alignHoles': not ordinary})
    assert result['feasible'], result['issues']
    assert any(w['reason'] == 'suspect_posture' for w in result['warnings'])
    # Recovery parks on an independent support, without claiming mating.
    parked = _esim(model).predict(scene, 'place', {'on': 'e:table', 'alignHoles': False})
    assert parked['feasible'], parked['issues']


def test_indirect_mate_and_suspect_held_part_are_checked():
    model, scene = _assembly_scene('e:gear')
    sim = _esim(model)
    for entity in ('e:fuse', 'e:gear'):
        posture = scene.entity(entity).state_vars['orientation'].value['posture']
        posture['status'] = 'inconsistent'
        result = sim.predict(scene, 'place', {'on': 'e:table', 'alignTo': 'e:fuse', 'hole': 'gear'})
        assert result['feasible'], result['issues']
        assert any(w['reason'] == 'suspect_posture' for w in result['warnings'])
        posture['status'] = 'unknown'


@pytest.mark.parametrize('transport', [math.pi, -math.pi/2, .7])
def test_complete_gear_pair_uses_template_and_pretransport_heading(transport):
    model, scene = _assembly_scene('e:gear')
    gear = scene.entity('e:gear')
    gear.state_vars['orientation'] = Est(value={'graspWorldYawRad': .1})
    gear.pose.value['q'] = quat_from_yaw(transport)
    result = _esim(model).predict(scene, 'place', {'on': 'e:fuse', 'hole': 'gear'})
    assert result['feasible'], result['issues']
    assert abs(wrap_angle(result['plan']['placementTarget']['yawRad'])) < .01
    assert 'mount_0' in ' '.join(result['issues']) and 'mount_1' in ' '.join(result['issues'])
    assert any('RSI_MATING_FRAME' in i for i in result['issues'])
