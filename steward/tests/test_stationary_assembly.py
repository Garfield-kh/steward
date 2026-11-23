"""Frozen planning reference survives the qa6 failure modes without inventing observations."""
from copy import deepcopy

import numpy as np
import pytest

from steward.ego.stationary_assembly import StationaryAssemblies
from steward.ego.estimator import EgoEstimator
from test_assembly_reference_memory import fixture, observe
from test_place_align_holes import _assembly_scene, _esim


def frozen():
    model, scene = _assembly_scene('e:gear')
    sim = _esim(model)
    receipt = sim.predict(scene, 'place', {'on': 'e:fuse', 'hole': 'gear'})
    target = receipt['plan']['placementTarget']
    ref = target['mateReference']
    memory = StationaryAssemblies()
    memory.freeze(scene.entity('e:fuse'), scene.entity(ref['support']), ref, {'views': 3})
    return memory, model, scene, sim, target


@pytest.mark.parametrize('change', ['gone', 'category', 'id', 'unregistered', 'existence'])
def test_second_fit_uses_frozen_frame_without_fresh_registered_body(change):
    m, model, scene, sim, target = frozen()
    before = sim.predict(scene, 'place', {'on': 'e:fuse', 'hole': 'gear'})
    body = scene.entity('e:fuse')
    if change == 'gone': scene.entities.remove(body)
    if change == 'id': body.id = 'merged'
    if change == 'category': body.category.value = 'toy_airplane'
    if change == 'existence': body.existence = .01
    if change == 'unregistered': body.rigid.links[0].affordances = []
    m.released('first', target['mateReference']['position'], target, 10)
    m.overlay(scene.entities, scene.relations, {}, 1000)
    after = sim.predict(scene, 'place', {'on': 'e:fuse', 'hole': 'gear'})
    assert after['feasible'], after['issues']
    assert after['plan']['spot'] == pytest.approx(before['plan']['spot'])
    assert after['plan']['placementTarget']['yawRad'] == pytest.approx(target['yawRad'])
    assert after['plan']['lift'] == pytest.approx(before['plan']['lift'])
    info = scene.entity('e:fuse').state_vars['orientation'].value
    assert info['stationaryAssembly']['parts'][0]['status'] == 'released_unverified'
    assert scene.entity('e:fuse').pose.t == m.records['e:fuse']['entity'].pose.t


@pytest.mark.parametrize('what', ['body', 'support'])
def test_grasp_invalidates_and_cannot_silently_refreeze(what):
    m, _, scene, sim, target = frozen()
    ref = target['mateReference']
    m.invalidate('e:fuse' if what == 'body' else ref['support'])
    m.overlay(scene.entities, scene.relations, {}, 100)
    assert sim.predict(scene, 'place', {'on': 'e:fuse', 'hole': 'gear'})['feasible']
    assert m.freeze(scene.entity('e:fuse'), scene.entity(ref['support']), ref, {})['warnings']


def test_clear_motion_needs_distinct_observations_and_survives_missing_track():
    old, body, target = fixture()
    est = EgoEstimator(robot_model=_assembly_scene('e:gear')[0])
    est.tracks[body.id] = body
    ent = est.to_scene().entity(body.id)
    ref = target['mateReference']
    m = StationaryAssemblies(); m.freeze(ent, ent, ref, {})
    m.first_grasp_at = 10
    observe(body, centre=(2.3,-3,.8), size=(.18,.06,.08), stamp=11)
    body.assembly_observation['rangeM'] = 1.
    m.update({body.id:body}, 11); m.update({body.id:body}, 11)
    assert m.records[body.id]['status'] == 'frozen'
    observe(body, centre=(2.3,-3,.8), size=(.18,.06,.08), stamp=12)
    body.assembly_observation['rangeM'] = 1.
    m.update({body.id:body}, 12)
    assert m.records[body.id]['status'] == 'invalidated'


def test_merged_axis_and_partial_extent_are_not_body_motion():
    _, body, target = fixture()
    est = EgoEstimator(robot_model=_assembly_scene('e:gear')[0]); est.tracks[body.id] = body
    ent = est.to_scene().entity(body.id)
    m = StationaryAssemblies(); m.freeze(ent, ent, target['mateReference'], {})
    for stamp in (11, 12):
        observe(body, size=(.25,.12,.07), yaw=.35, stamp=stamp)
        m.update({body.id:body}, stamp)
    assert m.records[body.id]['status'] == 'frozen'


def test_episode_reset_clears_frozen_reference():
    est = EgoEstimator(robot_model=_assembly_scene('e:gear')[0])
    est.stationary_assemblies = frozen()[0]
    est.forget_objects()
    assert not est.stationary_assemblies.records


def test_invalidated_body_keeps_live_moved_pose_for_recovery():
    m, _, scene, _, _ = frozen()
    body=scene.entity('e:fuse'); body.pose.value['p'][0] += .4
    m.records[body.id].update(status='invalidated', hardMotion=True, reason='direct near fresh body displacement')
    m.overlay(scene.entities,scene.relations,{},100)
    assert scene.entity(body.id).pose.value['p']==body.pose.value['p']
