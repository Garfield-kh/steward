"""Release/observation/reference lifecycle; synthetic Ego geometry only."""
import copy
import math

import numpy as np
import pytest

from steward.ego.assembly_memory import AssemblyMemory
from steward.ego.estimator import EgoEstimator
from test_place_align_holes import _track, _assembly_scene, _esim
from steward.ego.priors import screw_holes, SCREW_HOLE_FRAMES
from steward.ego.hole_registration import register_hole_template


def fixture():
    m = AssemblyMemory()
    b = _track('body', 'airplane_fuselage', [2, -3, .8], [.18, .06, .08], .0)
    holes, reg = register_hole_template(screw_holes(b.category), SCREW_HOLE_FRAMES[b.category], b.size)
    ref = {'entity': b.id, 'category': b.category, 'family': 'toy_airplane',
           'position': b.pos.tolist(), 'size': b.size.tolist(), 'yaw': b.yaw,
           'holes': holes, 'holeSource': 'prior', 'registration': reg, 'support': 'table', 'registeredAt': 1}
    target = {'mateReference': ref, 'sizeM': [.06, .22, .02], 'yawRad': 0., 'partCategory': 'airplane_wing'}
    m.released('wing', [2.02, -3, .85], target, 10)
    return m, b, target


def observe(b, size=(.18, .22, .1), centre=(2, -3, .81), stamp=11, yaw=0, **kw):
    b.size, b.pos, b.yaw, b.last_seen = np.array(size), np.array(centre), yaw, stamp
    b.assembly_observation = {'stamp': stamp, 'source': 'rgbd_segment', 'centreM': list(centre),
                             'visibleSizeM': list(size), 'boxYawRad': yaw,
                             'truncated': False, 'invalidDepthFraction': 0, **kw}


def test_release_alone_does_not_verify_or_unlock_merged_body():
    m, b, _ = fixture()
    v = m.views({b.id: b}, 11)[b.id]
    assert v['status'] == 'unresolved'
    assert v['parts'][0]['status'] == 'released_unverified'


def test_known_union_preserves_all_holes_frame_and_cannot_certify_upright():
    m, b, target = fixture()
    observe(b)
    v = m.views({b.id: b}, 11)[b.id]
    assert v['status'] == 'stationary_envelope_observed'
    assert v['reference'] == target['mateReference']
    assert v['parts'][0]['status'] == 'envelope_observed'
    assert v['upright'] is None and v['directedYawVerified'] is False
    assert v['collisionSize'][1] >= .22
    # Returned evidence is isolated from the ledger.
    v['reference']['position'][0] = -100
    assert m.records[0]['reference']['position'][0] == 2


@pytest.mark.parametrize('change', ['moved', 'tipped', 'unknown_piece', 'stale', 'partial', 'invalid_depth'])
def test_unexplained_changes_or_bad_observations_cannot_unlock(change):
    m, b, _ = fixture()
    observe(b)
    if change == 'moved': observe(b, centre=(2.12, -3, .81))
    if change == 'tipped': observe(b, size=(.18, .1, .22))
    if change == 'unknown_piece': observe(b, size=(.35, .30, .1))
    if change == 'stale': observe(b, stamp=8)
    if change == 'partial': observe(b, truncated=True)
    if change == 'invalid_depth': observe(b, invalidDepthFraction=.1)
    assert m.views({b.id: b}, 11)[b.id]['status'] == 'unresolved'


def test_unique_family_reassociation_but_not_ambiguous_neighbour():
    m, b, _ = fixture()
    observe(b)
    b.id, b.category_counts = 'new', {'toy_airplane': 1}
    assert m.views({b.id: b}, 11)[b.id]['reference']['entity'] == 'body'
    other = copy.deepcopy(b); other.id = 'duplicate'
    assert m.views({b.id: b, other.id: other}, 11)[b.id]['status'] == 'unresolved'


@pytest.mark.parametrize('entity', ['body', 'wing', 'table'])
def test_moving_any_known_member_or_support_invalidates_reference(entity):
    m, b, _ = fixture()
    observe(b)
    m.invalidate(entity)
    assert m.views({b.id: b}, 11)[b.id]['status'] == 'invalidated'
    assert m.records[0]['status'] == 'invalidated'


def test_second_release_keeps_original_registration_and_observes_new_union():
    m, b, target = fixture()
    observe(b)
    m.views({b.id: b}, 11)
    m.released('gear', [2, -3, .85], {**target, 'sizeM': [.03, .08, .025]}, 12)
    observe(b, stamp=13)
    v = m.views({b.id: b}, 13)[b.id]
    assert len(v['parts']) == 2
    assert v['reference'] == target['mateReference']


def test_planner_uses_original_body_height_and_frame_for_second_interface():
    model, scene = _assembly_scene('e:gear')
    sim = _esim(model)
    before = sim.predict(scene, 'place', {'on': 'e:fuse', 'hole': 'gear'})
    assert before['feasible']
    target = before['plan']['placementTarget']
    est = EgoEstimator(robot_model=model)
    m, b, _ = fixture()
    # Publish through the real estimator, with an observed merged union.
    observe(b)
    est.tracks[b.id] = b; est.assembly_memory = m; est.stamp = 11
    ent = est.to_scene().entity(b.id)
    info = ent.state_vars['orientation'].value
    assert info['posture']['status'] == 'unknown'
    assert ent.pose.value['p'] == [2, -3, .8]
    assert len(sim._screw_holes(ent)) == 4
    assert info['holeRegistration'] == m.records[0]['reference']['registration']
    fuse = scene.entity('e:fuse')
    fuse.state_vars['orientation'].value['assembly'] = {'reference': target['mateReference']}
    fuse.rigid.links[0].geometry['collision'][0]['size'][2] += .1
    after = sim.predict(scene, 'place', {'on': 'e:fuse', 'hole': 'gear'})
    assert after['feasible'], after['issues']
    assert after['plan']['lift'] == pytest.approx(before['plan']['lift'])
    assert after['plan']['spot'] == pytest.approx(before['plan']['spot'])


def test_union_named_after_installed_part_is_reassociated_without_relabelling():
    m, b, _ = fixture()
    observe(b)
    b.id, b.category_counts = 'wing', {'airplane_wing': 1}
    v = m.views({b.id: b}, 11)[b.id]
    assert v['status'] == 'stationary_envelope_observed'
    assert v['reference']['entity'] == 'body'
    assert v['parts'][0]['status'] == 'envelope_observed'


def test_independent_part_displacement_invalidates_reference_and_remains_visible():
    m, b, _ = fixture()
    observe(b, size=(.18, .06, .08), centre=(2, -3, .8))
    wing = _track('wing', 'airplane_wing', [2.2,-3,.85], [.06,.22,.02])
    observe(wing, size=(.06,.22,.02), centre=(2.2,-3,.85))
    tracks = {b.id:b, wing.id:wing}
    assert m.views(tracks,11)[b.id]['status'] == 'invalidated'
    assert m.views(tracks,11)[b.id]['status'] == 'invalidated'
