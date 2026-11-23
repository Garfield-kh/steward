"""Manipulation protocol only: no physics or service calls."""
import pytest
from steward.ego.skills import SkillExecutor, SkillFailed, SkillStopped, Job
from test_mating_transport_yaw import tracked_part


@pytest.mark.parametrize('failure', [None, SkillFailed, SkillStopped, RuntimeError])
def test_protection_spans_docking_and_is_released_on_every_exit(failure):
    est, track = tracked_part()
    events = []
    def pin(eid, value):
        assert eid == track.id
        events.append(value)
        track.pinned = value
    executor = SkillExecutor(lambda x: None, lambda: {}, lambda eid: eid, pin=pin)
    def attempt(job, plan):
        est._drop(track, 'occluded')
        est._fuse_cloud(track, {}, None)
        assert track.id in est.tracks
        assert track.category == 'car_wheel' and track.yaw == .63
        if failure:
            raise failure('interrupted')
        est.mark_held(track.id, at=track.pos.tolist())
        assert track.grasp_world_yaw == .63
    executor._pick_target = attempt
    if failure:
        with pytest.raises(failure):
            executor._pick(Job('j', 'pick', 'r'), {'target': track.id, 'protectGraspFrame': True})
    else:
        executor._pick(Job('j', 'pick', 'r'), {'target': track.id, 'protectGraspFrame': True})
    assert events == [True, False]
    assert not track.pinned


def test_protection_does_not_forge_observation_time():
    est, track = tracked_part()
    track.last_seen, est.stamp = 3., 50.
    track.pinned, track.pin_observed_age = True, 2.
    est.mark_held(track.id, at=track.pos.tolist())
    a = track.grasp_association
    assert a['observedAt'] == 3. and a['recordedAt'] == 50.
    assert a['protectedObservedAgeS'] == 2.
    assert a['identityVerified'] is False


def test_same_spot_duplicate_does_not_mutate_protected_track():
    import copy
    est, track = tracked_part()
    other = copy.deepcopy(track)
    other.id = 'duplicate'
    est.tracks[other.id] = other
    track.pinned = True
    est._merge_same_spot()
    assert set(est.tracks) == {track.id, other.id}
    assert track.observations == other.observations == 5


@pytest.mark.parametrize('age,registered,protected', [(2., True, True), (21., True, True), (2., False, False)])
def test_planner_only_protects_recent_registered_interfaces(age, registered, protected):
    from test_place_align_holes import _assembly_scene, _esim
    model, scene = _assembly_scene('e:wheel')
    sim = _esim(model)
    sim._held = lambda es: None
    target = scene.entity('e:wheel')
    scene.stamp = 30.
    target.pose.t = scene.stamp - age
    if not registered:
        for aff in target.rigid.links[0].affordances:
            if isinstance(aff.value, dict) and 'screw-hole' in aff.value:
                aff.value['screw-hole']['frame'] = 'unregistered'
    r = sim.predict(scene, 'pick', {'target': target.id})
    assert r['feasible'], r['issues']
    assert r['plan'].get('protectGraspFrame', False) == protected
    if age > 20:
        assert any(w['reason'] == 'stale_pose_estimate' and w['ageS'] == age for w in r['warnings'])
