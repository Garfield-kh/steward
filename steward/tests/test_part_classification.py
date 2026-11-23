"""Measured boxes and competing image labels, without scene state or a segmenter service."""
import numpy as np
import pytest

from steward.assets.generators import steward_bot
from steward.ego.classify import _color_score, part_semantics
from steward.ego.estimator import EgoEstimator
from steward.ego.segmenter import load_config


@pytest.mark.parametrize('size,color,labels,expected', [
    ([.203, .062, .078], [.8, .65, .1], {'toy_airplane': .8}, 'airplane_fuselage'),
    ([.227, .083, .015], [.8, .15, .15], {'puzzle_piece': .8, 'airplane_wing': .5}, 'airplane_wing'),
    ([.17, .05, .039], [.9, .4, .08], {'toy_airplane': .8, 'electric_screwdriver': .5}, 'electric_screwdriver'),
    ([.085, .04, .03], [.7, .72, .75], {'bowl': .8, 'airplane_landing_gear': .5}, 'airplane_landing_gear'),
    ([.016, .016, .022], [.55, .58, .62], {'toy_screw': .7}, 'toy_screw'),
    ([.22, .20, .06], [.9, .8, .1], {'toy_airplane': .8}, 'toy_airplane'),
    ([.17, .17, .06], [.7, .72, .75], {'bowl': .8, 'airplane_landing_gear': .5}, 'bowl'),
    ([.24, .24, .02], [.9, .9, .9], {'plate': .8, 'airplane_wing': .5}, 'plate'),
    ([.065, .04, .10], [.9, .4, .6], {'perfume_bottle': .8}, 'perfume_bottle'),
])
def test_part_evidence_and_household_counterexamples(size, color, labels, expected):
    est = EgoEstimator(steward_bot().to_dict())
    probs = est.category_probs(np.array(size), .75, np.array(color), 0., True, labels)
    assert max(probs, key=probs.get) == expected, probs


def test_airplane_family_evidence_needs_a_slender_observation():
    evidence = {'toy_airplane': .8}
    assert 'airplane_fuselage' not in part_semantics(evidence, np.array([.22, .20, .06]))
    assert 'airplane_fuselage' in part_semantics(evidence, np.array([.05, .18, .06]))
    assert evidence == {'toy_airplane': .8}


def test_color_rules_distinguish_orange_tool_from_yellow_airplane():
    orange, yellow = np.array([.9, .4, .08]), np.array([.9, .8, .1])
    assert _color_score('orange', orange, 0) >= 0 > _color_score('yellow', orange, 0)
    assert _color_score('yellow', yellow, 0) >= 0 > _color_score('orange', yellow, 0)
    assert _color_score('red', np.array([.8, .1, .1]), 0) >= 0 > _color_score('red', yellow, 0)


def test_fuselage_is_asked_on_the_whole_image_and_in_crops():
    cfg = load_config()
    assert cfg.vocabulary['airplane fuselage'] == 'airplane_fuselage'
    assert 'airplane fuselage' in cfg.crop_prompts


@pytest.mark.parametrize('size,labels,expected', [
    # Depth extents from p2 dropped records and uninflated, uncompleted small boxes.
    ([.035, .014, .030], None, 'toy_screw'),
    ([.0272, .0272, .030], None, 'toy_screw'),
    ([.0382, .0152, .027], None, 'toy_screw'),
    ([.046, .021, .026], {'toy_screw': .7}, 'toy_screw'),
    ([.073, .014, .022], None, 'airplane_landing_gear'),
    ([.080, .030, .025], {'toy_screw': .7, 'airplane_landing_gear': .5}, 'airplane_landing_gear'),
    ([.05, .05, .016], {'car_wheel': .7}, 'car_wheel'),
])
def test_noisy_screws_without_renaming_real_gear_or_wheels(size, labels, expected):
    est = EgoEstimator(steward_bot().to_dict())
    color = np.array([.15, .15, .15]) if expected == 'car_wheel' else np.array([.55, .58, .62])
    probs = est.category_probs(np.array(size), .75, color, 0., True, labels)
    assert max(probs, key=probs.get) == expected, probs


@pytest.mark.parametrize('size,color,labels,expected', [
    ([.035, .014, .03], [.55, .58, .62], {'toy_screw': .7}, 'toy_screw'),
    ([.085, .04, .03], [.7, .72, .75], {'bowl': .8, 'airplane_landing_gear': .5}, 'airplane_landing_gear'),
    ([.203, .062, .078], [.8, .65, .1], {'toy_airplane': .8}, 'airplane_fuselage'),
    ([.12, .07, 0.], [.9, .9, .9], {'picture': .8}, 'picture'),
    ([.23, .22, 0.], [.9, .9, .9], {'manual': .8}, 'manual'),
    ([.17, .17, .06], [.7, .72, .75], {'bowl': .8}, 'bowl'),
    ([.22, .20, .06], [.9, .8, .1], {'toy_airplane': .8}, 'toy_airplane'),
    ([.065, .04, .10], [.9, .4, .6], {'perfume_bottle': .8}, 'perfume_bottle'),
])
@pytest.mark.parametrize('bottom', [.4, .75, 1.1])
def test_occluded_base_does_not_turn_support_elevation_into_object_height(size, color, labels, expected, bottom):
    est = EgoEstimator(steward_bot().to_dict())
    probs = est.category_probs(np.array(size), bottom, np.array(color), 0., False, labels)
    assert max(probs, key=probs.get) == expected, probs


def test_occluded_furniture_still_uses_top_height():
    est = EgoEstimator(steward_bot().to_dict())
    probs = est.category_probs(np.array([1.2, .8, .01]), .74,
                               np.array([.62, .50, .38]), 0., False, {'table': .8})
    assert max(probs, key=probs.get) == 'table', probs
