"""Cross-process persistence and admission tests using anonymous robot estimates.

No world, judge, scenario, mechanical attachment truth or rendering is used.
"""
from collections import deque
from copy import deepcopy
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace as NS

import numpy as np
import pytest

from steward.assets.generators import steward_bot
from steward.ego.estimator import EgoEstimator, Track, Appearance
from steward.ego import memory_checkpoint as mem
from steward.ego import assembly_session as sessions
from steward.scene.schema import Est


def runtime():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    return NS(estimator=est, _lock=threading.RLock(), _generation=7, plan=None, _roles={})


def body(rt, stamp=None):
    est = rt.estimator
    return dict(stamp=est.stamp if stamp is None else stamp, odometry=deepcopy(est.odometry),
                joints=deepcopy(est.joints), gripper=deepcopy(est.gripper), grippers=deepcopy(est.grippers))


def populate(rt):
    e = rt.estimator; e.stamp = 40.; e.joints = {'axis': .4}
    def track(eid, offset):
        return Track(eid, {'unknown': 8.}, np.array([offset, 0., .5]), np.full(3, .0001),
            .2, .0001, np.array([.3, .2, .1]), 8, 10., 35., existence=1.)
    e.tracks = {'e:1': track('e:1', 0.), 'e:2': track('e:2', 1.), 'e:3': track('e:3', 2.)}
    e._ids = itertools.count(17)
    t = e.tracks['e:1']
    t.appearance = Appearance(np.full((2, 3, 3), np.nan), (.3, .2), 32.)
    t.interface_estimate_at = 33.; t.interface_motion_at = 30.
    t.screw_holes = [dict(id='port-A', p=[0., 0., .05], frame='entity', observedAt=33.)]
    t.interface_views.append(dict(stamp=34., points=np.ones((3, 3))))
    t.manipulation_view = dict(stamp=35., source='rgbd_segment', points=np.ones((3, 3)),
        centreM=t.pos.tolist(), visibleSizeM=t.size.tolist(), boxYawRad=t.yaw, rangeM=1.,
        truncated=False, invalidDepthFraction=0.)
    ent = e._track_entity(t)
    ent.state_vars['orientation'] = Est(value={}, src='inferred')
    support = e._track_entity(e.tracks['e:3'])
    ref = dict(entity=t.id, category='unknown', family='unknown', position=t.pos.tolist(),
        size=t.size.tolist(), yaw=t.yaw, support=support.id, registeredAt=30., holes=t.screw_holes,
        holeSource='observed', registration={}, rotationM=np.diag([1., -1., -1.]).tolist(),
        faceResolution=dict(status='estimated', source='executed_release', since=37., exposedHoles=['port-B']))
    e.stationary_assemblies.freeze(ent, support, ref, dict(observedAt=30., source='robot_rgbd'))
    e.stationary_assemblies.first_grasp_at = 25.
    e.fastening_attempts[('e:1', 'port-A')] = dict(body='e:1', part='e:2', hole='port-A', screw='e:9',
        releaseExecuted=True, seatedAt=31., seatConfirmed=False, releaseHistory=[dict(stamp=31.)])
    e.fastener_origins['e:9'] = dict(kitSupport='e:3', observedAt=20.,
        driverRelease=dict(track=deepcopy(e.tracks['e:2']), releasedAt=31.))
    e.released_interfaces['e:2'] = dict(entity='e:2', body='e:1', observedAt=28., releasedAt=31.,
        holes=deepcopy(t.screw_holes), holePairs=[dict(bodyHole='port-A', partHole='port-A')])
    e._seat_registrations = {'e:2': dict(body='e:1', observedAt=32., invalidated=None, point=[0., 0., .5])}
    e._interface_entry_budgets = {('e:1', 'e:2'): {'orientation', 'view'}}
    goal = dict(interfaces=[dict(key='0', body='e:1', part='e:2', holes=['port-A'], heldHoles=['port-A'], dependencies=[]),
        dict(key='1', body='e:1', part='e:3', holes=['port-B'], heldHoles=['port-B'], dependencies=[])],
        materials={}, tools=dict(mode='none'), budgetSimS=100., finalPose='as_is')
    state = dict(id='saved-session', generation=3, goal=goal, aliases={}, sequence=6,
        retries={'0': 1}, interfaceCalls={'0': 1}, releaseEpochs={'0/port-A': 1}, contradictions={},
        completed=['0'], flipped=['1'], decision='running', controller=None,
        phase='interfaces', finalPose=dict(policy='as_is', achieved=True, targets=[]),
        flipHistory=[dict(body='e:1', intentional=True, completedAt=37., faceResolution=ref['faceResolution'])],
        releasedUnconfirmedInterfaces=['0'], outcomeRefills={'0': 1}, outcomeComponentRetries={'0': 1},
        initialFaceInvalidatedAt={'e:1': 37.}, reviewedHoles=['0/port-A'])
    e.assembly_sessions = {state['id']: state}
    rt.plan = dict(current='continue remaining interface', t=39.)
    rt._roles = {'e:1': 'object'}
    rt._support_hole_watch = deepcopy(e._seat_registrations['e:2'])
    return rt


def save(rt, directory, monkeypatch):
    monkeypatch.setenv('STEWARD_MEMORY_CHECKPOINT_DIR', str(directory))
    monkeypatch.delenv('STEWARD_MEMORY_RESUME', raising=False)
    m = rt._memory_checkpoint = mem.create(rt)
    m.before_observation(body(rt))
    path = m.save('fixture', force=True)
    m.flush() if hasattr(m, 'flush') else None
    assert path is not None, m.receipts
    return path


def restore(path, monkeypatch, *, moved=False):
    monkeypatch.setenv('STEWARD_MEMORY_RESUME', str(path))
    monkeypatch.delenv('STEWARD_MEMORY_CHECKPOINT_DIR', raising=False)
    rt = runtime(); rt.estimator.joints = {'axis': .4}
    m = rt._memory_checkpoint = mem.create(rt)
    live = body(rt, 2.)
    if moved:
        live['odometry']['x'] += .5; live['joints']['axis'] += .2
    m.before_observation(live)
    return rt, m, live


def test_roundtrip_preserves_registration_release_ledger_and_session(tmp_path, monkeypatch):
    rt = populate(runtime()); path = save(rt, tmp_path, monkeypatch)
    fresh, m, live = restore(path, monkeypatch)
    assert m.restore_report['clockShiftS'] == -38.
    e = fresh.estimator
    assert set(e.tracks) == set(rt.estimator.tracks)
    assert next(e._ids) == 17
    assert e.assembly_sessions['saved-session']['completed'] == ['0']
    assert e.assembly_sessions['saved-session']['flipped'] == ['1']
    assert e.assembly_sessions['saved-session']['retries'] == {'0': 1}
    assert e.assembly_sessions['saved-session']['outcomeRefills'] == {'0': 1}
    assert e.assembly_sessions['saved-session']['generation'] == fresh._generation
    assert e.assembly_sessions['saved-session']['controller'] is None
    assert e.fastening_attempts[('e:1', 'port-A')]['releaseExecuted']
    assert isinstance(e.fastener_origins['e:9']['driverRelease']['track'], Track)
    assert e._interface_entry_budgets == rt.estimator._interface_entry_budgets
    assert e.stationary_assemblies.records['e:1']['reference']['rotationM'] == np.diag([1., -1., -1.]).tolist()
    assert e._seat_registrations['e:2']['observedAt'] == -6.
    assert fresh._support_hole_watch['observedAt'] == -6.
    assert m.restore_report['bodyCheck']['exceedsNoise'] is False


def test_repeated_restart_preserves_pending_controller_memory(tmp_path, monkeypatch):
    rt = populate(runtime())
    rt.estimator.assembly_sessions['saved-session']['controller'] = NS(
        _seat_retry_counts={'e:1': 2}, _tool_sources={'e:2': {'observedAt': 30.}})
    first = save(rt, tmp_path / 'first', monkeypatch)
    resumed, _, _ = restore(first, monkeypatch)
    second = save(resumed, tmp_path / 'second', monkeypatch)
    twice, m, _ = restore(second, monkeypatch)
    state = twice.estimator.assembly_sessions['saved-session']
    assert state['savedControllerMemory']['_seat_retry_counts'] == {'e:1': 2}
    assert state['savedControllerMemory']['_tool_sources']['e:2']['observedAt'] == -8.
    assert state['completed'] == ['0']
    assert m.restore_report['clockShiftS'] == 0.


def test_clock_translation_preserves_age_and_never_restores_new_frames(tmp_path, monkeypatch):
    original = populate(runtime()); path = save(original, tmp_path, monkeypatch)
    rt, m, _ = restore(path, monkeypatch)
    e = rt.estimator; t = e.tracks['e:1']
    assert e.stamp-t.last_seen == original.estimator.stamp-original.estimator.tracks[t.id].last_seen
    assert t.last_seen == -3. and t.first_seen == -28.
    assert t.interface_estimate_at == -5. and t.interface_motion_at == -8.
    assert t.appearance.stamp == -6. and np.isnan(t.appearance.rgb).all()
    assert e.fastening_attempts[('e:1', 'port-A')]['releaseHistory'][0]['stamp'] == -7.
    assert e.assembly_sessions['saved-session']['initialFaceInvalidatedAt']['e:1'] == -1.
    assert e.assembly_sessions['saved-session']['goal']['budgetSimS'] == 100.
    assert t.manipulation_view is None and t.assembly_observation is None and not t.interface_views
    scene = e.to_scene(); assert scene.find('e:1').pose.src == 'prior'
    assert 'contactView' not in scene.find('e:1').state_vars
    assert e.perception_cycles == 0 and not e.perception_stamps


def test_body_is_live_and_mismatch_marks_prior_not_a_block(tmp_path, monkeypatch):
    path = save(populate(runtime()), tmp_path, monkeypatch)
    rt, m, live = restore(path, monkeypatch, moved=True)
    assert rt.estimator.odometry == live['odometry'] and rt.estimator.joints == live['joints']
    check = m.restore_report['bodyCheck']
    assert check['exceedsNoise'] and check['positionDeltaM'] == .5
    assert check['jointDeltas']['axis'] == pytest.approx(.2)
    assert all(t.restored_prior['needsRecheck'] for t in rt.estimator.tracks.values())
    assert rt.estimator._seat_registrations['e:2']['invalidated'] == 'memory_body_changed'
    assert mem.progress(rt, rt.estimator.assembly_sessions['saved-session'])['unconfirmed']['bodyDependentMemory']


def test_first_detection_wins_over_low_variance_and_frozen_overlay(tmp_path, monkeypatch):
    path = save(populate(runtime()), tmp_path, monkeypatch)
    rt, m, _ = restore(path, monkeypatch)
    e = rt.estimator; t = e.tracks['e:1']; new = t.pos + np.array([.09, 0., 0.])
    e.stamp += 1.; e._note_perception(e.stamp)
    e._fuse(t, dict(category='unknown', size=t.size.tolist()), new, .5, .01)
    np.testing.assert_allclose(t.pos, new)
    assert t.yaw == pytest.approx(.5)
    assert m.conflicts == 1 and not hasattr(t, 'restored_prior')
    assert e.stationary_assemblies.records[t.id]['status'] == 'invalidated'
    np.testing.assert_allclose(e.to_scene().find(t.id).pose.value['p'], new)
    m.after_observation()
    assert m.restore_report['firstObservationStatus'] == 'processed'
    assert m.restore_report['firstObservationConflicts'] == 1
    # The next ordinary observation is ordinary fusion, not another restore conflict.
    e._fuse(t, dict(category='unknown', size=t.size.tolist()), new, .5, .01)
    assert m.conflicts == 1


def test_first_category_observation_is_not_vetoed_by_old_label(tmp_path, monkeypatch):
    original = populate(runtime()); original.estimator.tracks['e:1'].label = 'old-label'
    path = save(original, tmp_path, monkeypatch)
    rt, m, _ = restore(path, monkeypatch)
    t = rt.estimator.tracks['e:1']
    assert t in rt.estimator._candidates('unknown', t.pos, .01, t.footprint)
    rt.estimator._fuse(t, dict(category='unknown', size=t.size.tolist()), t.pos.copy(), t.yaw, .01)
    assert t.category == 'unknown' and m.conflicts == 1


def test_summary_and_subset_goal_continue_same_session(tmp_path, monkeypatch):
    path = save(populate(runtime()), tmp_path, monkeypatch)
    rt, _, _ = restore(path, monkeypatch)
    state = rt.estimator.assembly_sessions['saved-session']
    request = dict(interfaces=[dict(part='e:3', body='e:1', hole=['port-B'])], budgetSimS=999.)
    assert mem.owner(rt, request) == state['id']
    rt.esim = NS(_receipt=lambda scene, skill, params, risk, issues, plan, **kw: dict(plan=plan, **kw))
    receipt = sessions.predict(rt, NS(), request)
    assert receipt['plan'] == dict(resumeSession=state['id'])
    summary = receipt['progressSummary']
    assert [i['key'] for i in summary['completedInterfaces']] == ['0']
    assert summary['nextStep']['interface']['key'] == '1'
    assert summary['bodies'][0]['intentionalFlip']
    assert summary['bodies'][0]['bodyUpZEstimate'] == -1.
    assert summary['unconfirmed']['interfaces'] == ['0']
    assert state['goal']['budgetSimS'] == 100.


@pytest.mark.parametrize('mode', ['no_env', 'disabled'])
def test_zero_effect_without_opt_in_or_with_switch_off(tmp_path, monkeypatch, mode):
    monkeypatch.delenv('STEWARD_MEMORY_CHECKPOINT_DIR', raising=False)
    monkeypatch.delenv('STEWARD_MEMORY_RESUME', raising=False)
    if mode == 'disabled':
        monkeypatch.setenv('STEWARD_ASSEMBLY_MEMORY_CHECKPOINT', '0')
        monkeypatch.setenv('STEWARD_MEMORY_CHECKPOINT_DIR', str(tmp_path/'unused'))
        monkeypatch.setenv('STEWARD_MEMORY_RESUME', str(tmp_path/'missing'))
    rt = populate(runtime()); before = mem.encode(rt.estimator.tracks)
    assert mem.create(rt) is None
    assert mem.event(rt, 'release_completed') is None
    assert mem.result_fields(rt) == {}
    assert mem.encode(rt.estimator.tracks) == before
    assert not (tmp_path/'unused').exists()


@pytest.mark.parametrize('damage', ['truncated', 'version', 'checksum', 'type', 'structure'])
def test_bad_archive_rejected_transactionally_and_perception_can_continue(tmp_path, monkeypatch, damage):
    path = save(populate(runtime()), tmp_path, monkeypatch)
    row = json.loads(path.read_text())
    if damage == 'truncated': path.write_text('{')
    else:
        if damage == 'version': row['schema'] = 'future-version'
        if damage == 'checksum': row['sha256'] = 'bad'
        if damage in ('type', 'structure'):
            row['payload'] = {'$': 'executable', 'value': []} if damage == 'type' else mem.encode({'estimator': {}})
            import hashlib
            row['sha256'] = mem._checksum(row)
        path.write_text(json.dumps(row))
    rt, m, live = restore(path, monkeypatch)
    assert m.restore_report['status'] == 'rejected'
    assert not rt.estimator.tracks and not hasattr(rt.estimator, 'assembly_sessions')
    rt.estimator.update({**live, 'detections': []})
    assert rt.estimator.stamp == live['stamp'] and rt.estimator.perception_cycles == 1


def test_periodic_robot_clock_atomic_names_and_receipts(tmp_path, monkeypatch):
    logs = tmp_path/'receipts'; monkeypatch.setenv('STEWARD_RECEIPT_LOG_DIR', str(logs))
    original = populate(runtime()); path = save(original, tmp_path/'memory', monkeypatch)
    m = original._memory_checkpoint
    assert m.save() is None
    original.estimator.stamp += mem.PERIOD_S
    second = m.save(); assert second is not None and second.name > path.name
    # Events at the same stamp have unique ordered sequence suffixes.
    third = m.save('flip_completed', force=True)
    assert third != second and third.name > second.name
    m.flush() if hasattr(m, 'flush') else None
    assert not list((tmp_path/'memory').glob('*.tmp'))
    rt, resumed, _ = restore(second, monkeypatch)
    resumed.flush() if hasattr(resumed, 'flush') else None
    rows = [json.loads(s) for p in logs.glob('memory-*.jsonl') for s in p.read_text().splitlines()]
    assert sum(r['event'] == 'checkpoint_saved' for r in rows) == 3
    assert any(r['event'] == 'memory_restored' and r['sections']['tracks'] == 3 for r in rows)


def test_process_boundary(tmp_path, monkeypatch):
    """Writer and reader are separate fresh interpreters, sharing only a file."""
    env = dict(os.environ, STEWARD_ASSEMBLY_MEMORY_CHECKPOINT='1', STEWARD_MEMORY_CHECKPOINT_DIR=str(tmp_path/'mem'))
    env.pop('STEWARD_MEMORY_RESUME', None)
    code = '''
from test_memory_checkpoint import *
rt = populate(runtime())
m = rt._memory_checkpoint = mem.create(rt)
m.before_observation(body(rt))
print(m.save('process_exit', force=True))
'''
    env['PYTHONPATH'] = os.pathsep.join([str(Path(__file__).parent), *sys.path])
    writer = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True, check=True)
    path = Path(writer.stdout.strip()); assert path.is_file()
    env.pop('STEWARD_MEMORY_CHECKPOINT_DIR'); env['STEWARD_MEMORY_RESUME'] = str(path)
    code = '''
from test_memory_checkpoint import *
rt = runtime(); rt.estimator.joints = {'axis': .4}
m = rt._memory_checkpoint = mem.create(rt)
m.before_observation(body(rt, 2.))
e=rt.estimator
print(json.dumps(dict(ids=list(e.tracks), observed=e.tracks['e:1'].last_seen,
    release=e.fastening_attempts[('e:1','port-A')]['seatedAt'],
    progress=mem.progress(rt,e.assembly_sessions['saved-session']), report=m.restore_report)))
'''
    reader = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True, check=True)
    result = json.loads(reader.stdout)
    assert result['ids'] == ['e:1', 'e:2', 'e:3']
    assert result['observed'] == -3. and result['release'] == -7.
    assert result['progress']['nextStep']['interface']['key'] == '1'
    assert result['report']['bodyCheck']['exceedsNoise'] is False


def test_snapshot_clock_uses_later_encoder_tick_not_stale_perception(tmp_path, monkeypatch):
    rt = populate(runtime()); save(rt, tmp_path, monkeypatch)
    m = rt._memory_checkpoint
    live = body(rt, 46.); live['joints']['axis'] = .6
    m.note_body(live)
    rt.estimator.fastening_attempts[('e:1', 'port-A')]['seatedAt'] = 45.
    path = m.save('release_completed', force=True)
    m.flush() if hasattr(m, 'flush') else None
    assert json.loads(path.read_text())['robotTime'] == 46.
    resumed, restored, _ = restore(path, monkeypatch)
    assert restored.restore_report['clockShiftS'] == -44.
    assert resumed.estimator.tracks['e:1'].last_seen == -9.
    assert resumed.estimator.fastening_attempts[('e:1', 'port-A')]['seatedAt'] == 1.
    assert restored.restore_report['bodyCheck']['jointDeltas']['axis'] == pytest.approx(.2)


def test_held_state_is_restored_but_open_live_hand_wins(tmp_path, monkeypatch):
    rt = populate(runtime()); est = rt.estimator
    est.gripper = dict(state='closed', holding=True)
    est.held_id = 'e:2'; est.held_since = 36.
    t = est.tracks['e:2']; t.held = True; t.held_offset = np.zeros(3)
    t.held_interface = dict(observedAt=35., registration=dict(source='robot_measurement'))
    path = save(rt, tmp_path, monkeypatch)
    monkeypatch.setenv('STEWARD_MEMORY_RESUME', str(path))
    monkeypatch.delenv('STEWARD_MEMORY_CHECKPOINT_DIR')
    new = runtime(); new.estimator.joints = {'axis': .4}
    new.estimator.gripper = dict(state='closed', holding=True)
    memory = new._memory_checkpoint = mem.create(new)
    memory.before_observation(body(new, 2.))
    assert new.estimator.held_id == 'e:2' and new.estimator.tracks['e:2'].held
    assert new.estimator.held_since == -2.
    assert new.estimator.tracks['e:2'].held_interface['observedAt'] == -3.
    opened, m, _ = restore(path, monkeypatch)
    assert opened.estimator.held_id is None and not opened.estimator.tracks['e:2'].held
    assert m.restore_report['bodyCheck']['gripperChanged']


def test_real_runtime_first_poll_restores_before_fusion_and_exposes_summary(tmp_path, monkeypatch):
    path = save(populate(runtime()), tmp_path/'store', monkeypatch)
    monkeypatch.setenv('STEWARD_MEMORY_RESUME', str(path))
    monkeypatch.delenv('STEWARD_MEMORY_CHECKPOINT_DIR')
    from steward.ego.runtime import EgoRuntime, RuntimeConfig
    source = runtime(); source.estimator.joints = {'axis': .4}
    header = body(source, 2.)
    class Link:
        def sensors(self): return deepcopy(header)
    # The optional UI renderer is never started; perception/serialization stay real.
    monkeypatch.setitem(sys.modules, 'steward.ego.view', NS(EgoViewRenderer=lambda: NS(close=lambda: None)))
    rt = EgoRuntime(Link(), source.estimator.robot_model, RuntimeConfig(decision_observation=False))
    rt._poll_once()
    assert rt._generation == 0  # no spurious old-clock episode reset
    assert rt.estimator.assembly_sessions['saved-session']['generation'] == 0
    assert rt._memory_checkpoint.restore_report['firstObservationStatus'] == 'processed'
    result = rt.summary()
    assert result['progressSummary'][0]['nextStep']['interface']['key'] == '1'
    assert result['memoryResume']['clockShiftS'] == -38.
    assert rt._decision_frames == {} and not rt.receipts
    rt._skill_finished(NS(state='succeeded', skill='place', message='', postcondition={}, id='test'))


@pytest.mark.parametrize('historically_complete', [False, True, 'snapshot_boundary'])
def test_session_scheduler_skips_completed_interface_without_resetting_ledgers(tmp_path, monkeypatch, historically_complete):
    original = populate(runtime())
    if historically_complete:
        original.estimator.assembly_sessions['saved-session']['completed'] = ['0', '1']
        for item in original.estimator.assembly_sessions['saved-session']['goal']['interfaces']:
            original.estimator.fastening_attempts[(item['body'], item['holes'][0])] = dict(
                body=item['body'], part=item['part'], hole=item['holes'][0], screw='e:9',
                releaseExecuted=True, seatedAt=31., seatConfirmed=True,
                seatCheck=dict(passed=True, observedAt=40. if historically_complete == 'snapshot_boundary' else 35.))
    path = save(original, tmp_path, monkeypatch)
    rt, _, _ = restore(path, monkeypatch)
    from steward.ego import action_budget as ab, assembly_final_pose, session_initial, first_interface, assembly_outcome
    from steward.ego.skills import Job
    monkeypatch.setattr(first_interface, 'enabled', lambda: False)
    monkeypatch.setattr(first_interface, 'initialize', lambda s: None)
    monkeypatch.setattr(first_interface, 'repair', lambda s, i: None)
    monkeypatch.setattr(first_interface, 'summary', lambda s, r: r)
    monkeypatch.setattr(session_initial, 'refresh', lambda s: None)
    monkeypatch.setattr(session_initial, 'review', lambda s: None)
    monkeypatch.setattr(assembly_final_pose, 'restore', lambda s: True)
    monkeypatch.setattr(assembly_outcome, 'enabled', lambda: True)
    monkeypatch.setattr(assembly_outcome, 'summary', lambda s: dict(repairPending=False, holeOutcomes=[]))
    from steward.ego import session_observation, session_faces, axial_release
    monkeypatch.setattr(session_observation, 'enabled', lambda: False)
    monkeypatch.setattr(session_faces, 'enabled', lambda: False)
    monkeypatch.setattr(axial_release, 'enabled', lambda: False)
    clock = ab.Clock(); clock.observe(rt.estimator.stamp)
    rt.executor = NS(_stop=threading.Event(), _action_clock=clock,
        _state=lambda: body(rt), stop=lambda: None)
    calls = []
    monkeypatch.setattr(sessions.Session, 'access', lambda s, i: dict(status='estimated_accessible'))
    monkeypatch.setattr(sessions.Session, 'execute_interface', lambda s, i: calls.append(i['key']))
    monkeypatch.setattr(sessions.Session, 'complete', lambda s, i: True)
    monkeypatch.setattr(sessions.Session, 'review_releases', lambda s: None)
    monkeypatch.setattr(sessions.Session, 'execute_tools', lambda s: None)
    monkeypatch.setattr(sessions.Session, 'emit', lambda s, event, **kw: None)
    monkeypatch.setattr(sessions.Session, 'summary', lambda s, *a, **kw: mem.progress(rt, s.state))
    job = Job('continuation', 'complete_assembly', 'new-receipt', started=2., postcondition={})
    session = sessions.Session(rt)
    session.run(job, dict(resumeSession='saved-session'))
    assert calls == ([] if historically_complete else ['1'])
    state = rt.estimator.assembly_sessions['saved-session']
    if historically_complete:
        assert state['decision'] == 'executed_unverified', state.get('reason')
        assert all(a['seatConfirmed'] for a in rt.estimator.fastening_attempts.values())
        assert len(job.postcondition['unconfirmed']['holes']) == 2

    assert state['completed'] == ['0', '1'] and state['flipped'] == ['1']
    assert state['outcomeRefills'] == {'0': 1} and state['outcomeComponentRetries'] == {'0': 1}
    assert state['retries'] == {'0': 1} and state['reviewedHoles'] == ['0/port-A']


def test_first_face_observation_conflict_is_reported_without_restoring_old_orientation(tmp_path, monkeypatch):
    path = save(populate(runtime()), tmp_path, monkeypatch)
    rt, m, _ = restore(path, monkeypatch)
    ref = rt.estimator.stationary_assemblies.records['e:1']['reference']
    ref['faceResolution'] = dict(status='observed', source='current_rgbd_signature_registered_face',
                                since=3., exposedHoles=['port-A'])
    m.reconcile_faces()
    assert m.conflicts == 1
    assert m.restore_report['conflicts'][0]['kind'] == 'registered_face'
    assert ref['faceResolution']['exposedHoles'] == ['port-A']
    m.reconcile_faces(); assert m.conflicts == 1


def test_embedded_recovery_tracks_remain_priors_and_have_no_current_frames(tmp_path, monkeypatch):
    rt = populate(runtime())
    rt.estimator.assembly_sessions['saved-session']['firstInterfaceSources'] = deepcopy(rt.estimator.tracks)
    path = save(rt, tmp_path, monkeypatch)
    new, _, _ = restore(path, monkeypatch)
    sources = new.estimator.assembly_sessions['saved-session']['firstInterfaceSources']
    assert sources['e:1'].manipulation_view is None
    assert sources['e:1'].restored_prior['originalLastSeen'] == -3.
    new.estimator.tracks['e:1'] = deepcopy(sources['e:1'])
    assert new.estimator._track_entity(new.estimator.tracks['e:1']).pose.src == 'prior'


def test_observation_override_precedes_history_based_size_guard(tmp_path, monkeypatch):
    path = save(populate(runtime()), tmp_path, monkeypatch)
    rt, m, _ = restore(path, monkeypatch)
    from steward.ego import supply_reliability as sr
    monkeypatch.setattr(sr, 'size_enabled', lambda: True)
    seen = []
    def reject(est, track, size, stage, category=None):
        seen.append(track.category)
        return track.category == 'old-label'
    monkeypatch.setattr(sr, 'reject', reject)
    t = rt.estimator.tracks['e:1']; t.label = 'old-label'
    rt.estimator._fuse(t, dict(category='unknown', size=t.size.tolist()), t.pos.copy(), t.yaw, .01)
    assert seen and all(c == 'unknown' for c in seen)
    assert t.last_seen == rt.estimator.stamp and m.conflicts == 1


def test_write_failure_is_advisory_and_keeps_existing_snapshot(tmp_path, monkeypatch):
    rt = populate(runtime()); path = save(rt, tmp_path, monkeypatch)
    original = path.read_bytes()
    monkeypatch.setattr(mem.os, 'replace', lambda *args: (_ for _ in ()).throw(OSError('write failed')))
    pending = rt._memory_checkpoint.save('release_completed', force=True)
    rt._memory_checkpoint.flush() if hasattr(rt._memory_checkpoint, 'flush') else None
    assert pending is None or not pending.exists()
    assert path.read_bytes() == original
    assert rt._memory_checkpoint.receipts[-1]['event'] == 'checkpoint_rejected'
    assert not list(tmp_path.glob('*.tmp'))


def test_header_clock_corruption_is_rejected(tmp_path, monkeypatch):
    path = save(populate(runtime()), tmp_path, monkeypatch)
    row = json.loads(path.read_text()); row['robotTime'] += 1.
    path.write_text(json.dumps(row))
    rt, m, _ = restore(path, monkeypatch)
    assert m.restore_report['status'] == 'rejected'
    assert not rt.estimator.tracks


def test_restore_preserves_watch_cache_alias_for_current_hole_invalidation(tmp_path, monkeypatch):
    rt = populate(runtime())
    registration = rt.estimator._seat_registrations['e:2']
    registration.update(entity='e:2', hole='port-A')
    rt._support_hole_watch = registration
    path = save(rt, tmp_path, monkeypatch)
    new, _, _ = restore(path, monkeypatch)
    assert new._support_hole_watch is new.estimator._seat_registrations['e:2']
    new._support_hole_watch['invalidated'] = 'support_features_moved'
    assert new.estimator._seat_registrations['e:2']['invalidated'] == 'support_features_moved'


def test_conflicting_retained_reference_cannot_override_current_geometry(tmp_path, monkeypatch):
    original = populate(runtime())
    original.estimator.stationary_assemblies.records['e:1']['edgeRetained'] = True
    path = save(original, tmp_path, monkeypatch)
    rt, memory, _ = restore(path, monkeypatch)
    t = rt.estimator.tracks['e:1']; position = t.pos + [.08, 0., 0.]
    rt.estimator.stamp += 1.
    rt.estimator._fuse(t, dict(category='unknown', size=t.size.tolist()), position, t.yaw, .01)
    record = rt.estimator.stationary_assemblies.records['e:1']
    assert record['edgeRetained']
    np.testing.assert_allclose(record['reference']['position'], position)
    assert record['reference']['rotationM'][2][2] == -1.  # face prior survives position correction
    assert record['reference']['registeredAt'] == -8.  # no fresh hole registration was invented
    assert rt.estimator._seat_registrations['e:2']['invalidated'] == 'memory_observation_conflict'


def test_timestamp_tuples_translate_but_durations_and_geometric_before_do_not():
    value = dict(content_hits=[(8., [1., 2.], .9)],
        _wrist_landing_history=[dict(release=('e:4', 9.), stamp=9.)],
        before=.1, simS=2., ageS=4., wallTime=123., observedAt=10.)
    shifted = mem.rebase(deepcopy(value), -12.)
    assert shifted['content_hits'][0][0] == -4.
    assert shifted['_wrist_landing_history'][0]['release'] == ('e:4', -3.)
    assert shifted['observedAt'] == -2.
    assert all(shifted[k] == value[k] for k in ('before', 'simS', 'ageS', 'wallTime'))


def test_archive_header_is_directly_inspectable_for_selection(tmp_path, monkeypatch):
    rt = populate(runtime()); path = save(rt, tmp_path, monkeypatch)
    header = json.loads(path.read_text())
    assert header['robotTime'] == header['bodySnapshotStamp'] == 40.
    assert header['held'] is None
    assert header['progressSummary'][0]['bodies'][0]['intentionalFlip']
    assert header['progressSummary'][0]['nextStep']['interface']['key'] == '1'
    assert header['sections']['assembly_sessions'] == 1


def test_first_rgbd_measurement_discards_old_union_and_pinning(tmp_path, monkeypatch):
    from itertools import product
    from steward.ego.perception import Segment
    rt = populate(runtime()); t = rt.estimator.tracks['e:1']
    t.pinned = True; t.voxels = np.ones((8, 3))*100
    path = save(rt, tmp_path, monkeypatch)
    new, memory, _ = restore(path, monkeypatch)
    est = new.estimator; t = est.tracks['e:1']
    shift = 6*(est.config.pos_sigma_base+est.config.pos_sigma_per_m)
    position = t.pos + [shift, 0., 0.]
    points = np.array(list(product(*[[-v/2, v/2] for v in t.size]))) + position
    segment = Segment(points=points, color=np.full(3, .5), pixels=len(points),
                      invalid_frac=0., border=False, range=1.)
    est.stamp += 1.
    est._fuse_cloud(t, dict(seg=segment, probs={'unknown': 1.}, centre=position,
                           size=t.size.copy(), yaw=t.yaw), np.zeros(3))
    assert not t.pinned and not hasattr(t, 'restored_prior')
    assert memory.conflicts == 1 and t.last_seen == est.stamp
    assert np.linalg.norm(t.pos-position) < est.config.pos_sigma_base
    assert t.assembly_observation['stamp'] == est.stamp


def test_late_decode_failure_does_not_commit_any_partial_runtime_state(tmp_path, monkeypatch):
    path = save(populate(runtime()), tmp_path, monkeypatch)
    row = json.loads(path.read_text()); data = mem.decode(row['payload'])
    data['stationary_assemblies']['records']['e:1']['reference'] = None
    row['payload'] = mem.encode(data); row['sha256'] = mem._checksum(row)
    path.write_text(json.dumps(row))
    rt, m, _ = restore(path, monkeypatch)
    assert m.restore_report['status'] == 'rejected'
    assert rt.plan is None and not rt._roles and not rt.estimator.tracks


def test_standalone_flip_history_survives_restart_without_session(tmp_path, monkeypatch):
    rt = populate(runtime()); rt.estimator.assembly_sessions = {}
    save(rt, tmp_path, monkeypatch)
    rt._memory_checkpoint.skill_finished(NS(skill='flip_assembly', finished=40.,
        postcondition=dict(body='e:1', releaseExecuted=True, commandedTurnOver=True,
                           flipped=None, confirmation='unconfirmed')))
    rt._memory_checkpoint.flush() if hasattr(rt._memory_checkpoint, 'flush') else None
    path = max(tmp_path.glob('memory-*.json'))
    new, _, _ = restore(path, monkeypatch)
    result = mem.result_fields(new)
    assert not result['progressSummary']
    assert result['orientationHistory'][0]['intentional']
    assert result['orientationHistory'][0]['confirmation'] == 'unconfirmed'
    assert result['orientationHistory'][0]['completedAt'] == 2.
