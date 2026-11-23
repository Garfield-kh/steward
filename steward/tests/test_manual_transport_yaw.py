import math
import pytest
from steward.scene.schema import Est
from test_place_align_holes import _assembly_scene, _esim


@pytest.mark.parametrize('heading', [-.6, 0., .8])
@pytest.mark.parametrize('requested', [None, .3])
def test_manual_interface_placement_restores_heading_unless_explicit(heading, requested):
    model, scene = _assembly_scene('e:wheel')
    wheel = scene.entity('e:wheel')
    wheel.state_vars['orientation'] = Est(value={'graspWorldYawRad': heading, 'directedYawObserved': False})
    params = {'on': 'e:table', 'alignHoles': False, 'offset': [0., -.15], 'clearance': 0}
    if requested is not None:
        params['yaw'] = requested
    r = _esim(model).predict(scene, 'place', params)
    assert r['feasible'], r['issues']
    assert r['plan']['placementTarget']['yawRad'] == pytest.approx(heading if requested is None else requested)
    assert 'alignTo' not in r['plan']
    assert any('RSI_TRANSPORT_YAW' in i for i in r['issues']) == (requested is None)


def test_missing_snapshot_keeps_manual_placement_contract():
    model, scene = _assembly_scene('e:wheel')
    r = _esim(model).predict(scene, 'place', {'on': 'e:table', 'alignHoles': False, 'offset': [0., -.15]})
    assert r['feasible'], r['issues']
    assert 'placementTarget' not in r['plan']
