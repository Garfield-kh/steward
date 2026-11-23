"""Self-contained assembly regressions using bundled inputs and synthetic fixtures."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import threading
import time

import numpy as np
import pytest

from test_memory_checkpoint import runtime, populate, body, restore
from steward.ego import memory_checkpoint as mem


def memory(rt, path, monkeypatch):
    monkeypatch.setenv('STEWARD_MEMORY_CHECKPOINT_DIR', str(path))
    monkeypatch.delenv('STEWARD_MEMORY_RESUME', raising=False)
    m = rt._memory_checkpoint = mem.Memory(rt)
    m.before_observation(body(rt))
    return m


def loaded_runtime(samples=12000):
    rt = populate(runtime())
    rng = np.random.default_rng(96)
    for t in rt.estimator.tracks.values():
        # Exact persistent shape data must survive; raw per-view data does not.
        t.voxels = rng.integers(-100, 100, (6000, 3), dtype=np.int64)
        t.span = rng.random((8, 3))
        t.recent.extend([(rng.random(3), rng.random(3), 1.) for _ in range(5)])
        for i in range(12):
            t.interface_views.append(dict(stamp=30.+i, points=rng.random((samples, 3))))
        t.interface_best_near = dict(stamp=35., points=rng.random((samples, 3)))
        t.manipulation_view = dict(stamp=35., points=rng.random((samples, 3)))
        t.assembly_observation = dict(stamp=35., points=rng.random((samples, 3)))
    return rt


def test_long_episode_archives_plateau_and_keep_semantic_summaries(tmp_path, monkeypatch):
    rt = populate(runtime()); m = memory(rt, tmp_path, monkeypatch)
    session = rt.estimator.assembly_sessions['saved-session']
    sizes = []
    for count in (100, 1000, 10000):
        rt.estimator.orientation_history = [dict(body='e:1', completedAt=1., intentional=i == 0) for i in range(count)]
        session['flipHistory'] = deepcopy(rt.estimator.orientation_history)
        session['flipDurations'] = [dict(body='e:1', simS=1.+i % 8) for i in range(count)]
        rt.estimator.stationary_assemblies.motion_events = [dict(entity='e:1', stamp=1.) for _ in range(count)]
        rt.estimator.fastening_attempts[('e:1', 'port-A')]['driveHistory'] = [dict(driveStatus='unknown') for _ in range(count)]
        path = m.save('long_episode', force=True); m.flush()
        sizes.append(path.stat().st_size)
        data = mem.decode(json.loads(path.read_text())['payload'])
        assert len(data['estimator']['orientation_history']) == mem.HISTORY_LIMIT+1
        assert data['estimator']['orientation_history'][0]['intentional']
        s = data['estimator']['assembly_sessions']['saved-session']
        assert s['flipDurations'] == [dict(body='e:1', simS=8.)]
        assert len(data['stationary_assemblies']['motion_events']) == 1
        assert len(data['estimator']['fastening_attempts'][('e:1', 'port-A')]['driveHistory']) == mem.HISTORY_LIMIT
        assert s['completed'] == ['0'] and s['releaseEpochs'] == {'0/port-A': 1}
    assert max(sizes) < 1_000_000
    assert max(sizes)-min(sizes) < 128


def test_main_thread_does_not_serialize_or_wait_for_blocked_disk(tmp_path, monkeypatch):
    rt = loaded_runtime(); m = memory(rt, tmp_path, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    owner = threading.get_ident()
    original_encode, original_fsync = mem.encode, mem.os.fsync
    def encode(value):
        assert threading.get_ident() != owner
        return original_encode(value)
    def fsync(fd):
        assert threading.get_ident() != owner
        entered.set()
        assert release.wait(5.)
        return original_fsync(fd)
    monkeypatch.setattr(mem, 'encode', encode)
    monkeypatch.setattr(mem.os, 'fsync', fsync)
    try:
        start = time.perf_counter(); path = m.save(force=True)
        assert time.perf_counter()-start < .050
        assert entered.wait(5.)
        # A second event preserves its own snapshot while the first is blocked.
        with rt._lock:
            rt.estimator.stamp += 1.
            start = time.perf_counter(); second = m.save('release_completed', force=True)
            assert time.perf_counter()-start < .050
        assert not path.exists() and not second.exists()
    finally:
        release.set(); m.flush(timeout=10.)
    assert path.exists() and second.exists()
    rows = [r for r in m.receipts if r['event'] == 'checkpoint_saved']
    assert len(rows) == 2
    for r in rows:
        assert r['bytes'] == Path(r['file']).stat().st_size
        assert 0 <= r['mainThreadMs'] < 50.
        assert r['backgroundWriteMs'] >= 0 and r['backgroundTotalMs'] >= r['backgroundWriteMs']


def test_event_snapshot_is_detached_and_preserves_every_event(tmp_path, monkeypatch):
    rt = populate(runtime()); m = memory(rt, tmp_path, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    original = m._serialize
    def slow(row):
        entered.set(); assert release.wait(5.)
        return original(row)
    monkeypatch.setattr(m, '_serialize', slow)
    try:
        first = m.save('interface_released', force=True); assert entered.wait(5.)
        rt.estimator.tracks['e:1'].pos[:] = [9., 8., 7.]
        rt.estimator.assembly_sessions['saved-session']['completed'].append('1')
        second = m.save('final_pose_result', force=True)
        assert first != second
    finally:
        release.set(); m.flush(timeout=10.)
    a = mem.decode(json.loads(first.read_text())['payload'])['estimator']
    b = mem.decode(json.loads(second.read_text())['payload'])['estimator']
    np.testing.assert_array_equal(a['tracks']['e:1'].pos, [0., 0., .5])
    np.testing.assert_array_equal(b['tracks']['e:1'].pos, [9., 8., 7.])
    assert a['assembly_sessions']['saved-session']['completed'] == ['0']
    assert b['assembly_sessions']['saved-session']['completed'] == ['0', '1']


@pytest.mark.parametrize('failure', ['serialize', 'write', 'fsync', 'rename'])
def test_background_failure_does_not_interrupt_run_and_can_recover(tmp_path, monkeypatch, failure):
    rt = populate(runtime()); m = memory(rt, tmp_path, monkeypatch)
    good = m.save('before', force=True); m.flush(); saved = good.read_bytes()
    def fail(*args, **kwargs): raise OSError('injected backend failure')
    with monkeypatch.context() as patch:
        if failure == 'serialize': patch.setattr(m, '_serialize', fail)
        elif failure == 'write': patch.setattr(mem.tempfile, 'NamedTemporaryFile', fail)
        else: patch.setattr(mem.os, 'fsync' if failure == 'fsync' else 'replace', fail)
        pending = m.save('release_completed', force=True)
        assert pending is not None
        rt.estimator.stamp += 1.
        m.flush()
        assert m.receipts[-1]['event'] == 'checkpoint_rejected'
        assert all(k in m.receipts[-1] for k in ('bytes', 'mainThreadMs', 'backgroundWriteMs'))
    assert good.read_bytes() == saved
    assert not list(tmp_path.glob('.memory-*.tmp'))
    recovered = m.save('recovered', force=True); m.flush()
    assert recovered.exists()


def test_period_configuration_retention_and_events(tmp_path, monkeypatch):
    monkeypatch.setenv('STEWARD_MEMORY_CHECKPOINT_PERIOD_S', '7.5')
    monkeypatch.setenv('STEWARD_MEMORY_CHECKPOINT_KEEP', '2')
    rt = populate(runtime()); m = memory(rt, tmp_path, monkeypatch)
    periods, events = [], []
    for _ in range(5):
        periods.append(m.save()); m.flush()
        events.append(m.save('interface_released', force=True)); m.flush()
        rt.estimator.stamp += 7.49
        assert m.save() is None
        rt.estimator.stamp += .01
    assert all(p.exists() for p in events)
    assert [p.exists() for p in periods] == [False, False, False, True, True]
    assert len(list(tmp_path.glob('memory-*.json'))) == 7


@pytest.mark.parametrize('period', ['nan', 'inf', '-1', '0', 'bad'])
def test_invalid_period_defaults_safely(tmp_path, monkeypatch, period):
    monkeypatch.setenv('STEWARD_MEMORY_CHECKPOINT_PERIOD_S', period)
    m = memory(populate(runtime()), tmp_path, monkeypatch)
    assert m.period_s == 30.


def test_only_one_periodic_pending_and_events_bypass_period(tmp_path, monkeypatch):
    rt = populate(runtime()); m = memory(rt, tmp_path, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    original = m._serialize
    def slow(row):
        entered.set(); assert release.wait(5.)
        return original(row)
    monkeypatch.setattr(m, '_serialize', slow)
    try:
        first = m.save(); assert entered.wait(5.)
        rt.estimator.stamp += 60.
        assert m.save() is None
        event = m.save('flip_completed', force=True)
    finally:
        release.set(); m.flush(timeout=10.)
    assert first.exists() and event.exists()
    rt.estimator.stamp += 30.
    last = m.save(); m.flush(); assert last.exists()


def test_latest_motion_preserves_every_admission_threshold():
    rows = [dict(entity=e, stamp=float(t)) for t in range(50) for e in ('e:1', 'e:2')]
    compact = mem.latest_motion(rows)
    for eid in ('e:1', 'e:2', 'e:3'):
        for stamp in (-1, 0, 1, 48, 49, 50):
            assert any(e['entity'] == eid and e['stamp'] > stamp for e in rows) == any(
                e['entity'] == eid and e['stamp'] > stamp for e in compact)
            assert any(e['entity'] == eid and e['stamp'] >= stamp for e in rows) == any(
                e['entity'] == eid and e['stamp'] >= stamp for e in compact)


def test_nested_tuple_tracks_pruned_and_distinct_transcripts_not_aliased():
    from collections import deque
    rt = populate(runtime()); track = rt.estimator.tracks['e:1']
    track.interface_views = deque([dict(points=np.zeros((1000, 3)))], maxlen=7)
    state = dict(nested=(track,), first={'driveHistory': [dict(number=1)] * 20},
                 second={'driveHistory': [dict(number=2)] * 20})
    frozen = mem.snapshot_copy(state)
    assert frozen['nested'][0].interface_views.maxlen == 7
    assert not frozen['nested'][0].interface_views
    assert len(track.interface_views) == 1
    assert frozen['first']['driveHistory'][0]['number'] == 1
    assert frozen['second']['driveHistory'][0]['number'] == 2


def test_retention_default_unlimited_zero_and_run_isolation(tmp_path, monkeypatch):
    monkeypatch.delenv('STEWARD_MEMORY_CHECKPOINT_KEEP', raising=False)
    rt = populate(runtime()); m = memory(rt, tmp_path, monkeypatch)
    assert m.keep is None
    first = m.save(); m.flush()
    rt.estimator.stamp += 30.; second = m.save(); m.flush()
    assert first.exists() and second.exists()
    monkeypatch.setenv('STEWARD_MEMORY_CHECKPOINT_KEEP', '0')
    other = memory(populate(runtime()), tmp_path, monkeypatch)
    removed = other.save(); event = other.save('release_completed', force=True); other.flush()
    assert not removed.exists() and event.exists()
    assert first.exists() and second.exists()


def test_all_io_including_receipts_is_off_the_runtime_lock(tmp_path, monkeypatch):
    monkeypatch.setenv('STEWARD_RECEIPT_LOG_DIR', str(tmp_path / 'logs'))
    rt = populate(runtime()); m = memory(rt, tmp_path / 'memory', monkeypatch)
    original_open = Path.open
    owner = threading.get_ident()
    def opened(path, *args, **kwargs):
        assert threading.get_ident() != owner, path
        assert not rt._lock._is_owned()
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', opened)
    m.emit('memory_test')
    path = m.save('release_completed', force=True)
    m.flush()
    assert path.exists()


def test_stationary_motion_latest_timestamp_handles_out_of_order_updates():
    from steward.ego.stationary_assembly import StationaryAssemblies
    memory = StationaryAssemblies()
    memory.invalidate('e:1', stamp=5.)
    memory.invalidate('e:1', stamp=2.)
    memory.invalidate('e:2', stamp=3.)
    memory.invalidate('e:1', stamp=8.)
    assert {e['entity']: e['stamp'] for e in memory.motion_events} == {'e:1': 8., 'e:2': 3.}
    assert len(memory.motion_events) == 2


def test_interface_observation_rollup_keeps_exact_counts_across_resume(monkeypatch):
    from types import SimpleNamespace as NS
    from steward.ego import interface_entry, assembly_session
    monkeypatch.setattr(assembly_session, 'current', lambda: None)
    rows = [dict(orientation=dict(measured=i % 2 == 0, matches=i % 3 != 0),
                 support=dict(measured=False, matches=None)) for i in range(10000)]
    original = dict(schema='interface_entry.v1', id='check', body='e:1', holes=['port-A'],
                    corrections={}, observations=rows)
    old_job = NS(postcondition={})
    interface_entry.publish(NS(rt=NS()), old_job, deepcopy(original))
    saved = mem.snapshot_copy(original)
    assert len(saved['observations']) == mem.HISTORY_LIMIT
    assert saved['observationArchive']['count'] == 10000-mem.HISTORY_LIMIT
    new_job = NS(postcondition={})
    interface_entry.publish(NS(rt=NS(_memory_checkpoint=True)), new_job, saved)
    assert old_job.postcondition == new_job.postcondition
    saved['observations'].append(rows[0])
    interface_entry.publish(NS(rt=NS(_memory_checkpoint=True)), new_job, saved)
    result = new_job.postcondition['interfaceEntryChecks'][0]
    assert result['observationCount'] == 10001
    assert result['observationCounts']['orientation']['measured'] == 5001
    assert len(saved['observations']) == mem.HISTORY_LIMIT
