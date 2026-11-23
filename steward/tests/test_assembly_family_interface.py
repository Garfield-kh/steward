"""Family labels may expose a registered component hypothesis, never a merged shape."""
import numpy as np
import pytest

from steward.ego.estimator import EgoEstimator, Track
from steward.ego.esim import Esim


def candidate(size):
    track = Track('candidate', {'toy_airplane': 1}, np.zeros(3), np.ones(3)*.000001,
                  .4, .001, np.array(size), 3, 0., 1.)
    return EgoEstimator.__new__(EgoEstimator)._track_entity(track)


@pytest.mark.parametrize('size', [(.188, .044, .08), (.044, .188, .08)])
def test_slender_family_has_registered_component_interface_without_relabel(size):
    entity = candidate(size)
    assert entity.category.value == 'toy_airplane'
    holes = Esim._screw_holes(entity)
    assert {h['group'] for h in holes} == {'fuselage.wing', 'fuselage.gear'}
    assert all(h['source'] == 'prior' and h['frame'] == 'entity' for h in holes)
    info = entity.state_vars['orientation'].value['holeRegistration']
    assert info['templateCategory'] == 'airplane_fuselage'
    assert info['categoryHypothesis'] and not info['directedYawObserved']


@pytest.mark.parametrize('size', [(.306, .146, .07), (.188, .094, .083), (.2, .2, .07)])
def test_broad_family_cannot_borrow_fuselage_holes(size):
    entity = candidate(size)
    assert Esim._screw_holes(entity)
    assert entity.state_vars['orientation'].value['holeRegistration']['warnings']
