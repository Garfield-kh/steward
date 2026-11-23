"""Versioned robot-owned memory. No simulator, task catalogue or executable pickle.

Capture under the runtime lock; restore transactionally before the first fusion.
Sensor pixels, live controllers, commands and observation admission tokens are
never reactivated. Persisted evidence retains its age and becomes a prior.
"""
from collections import deque
from copy import deepcopy, copy
import base64
from concurrent.futures import ThreadPoolExecutor
import dataclasses
from functools import wraps, lru_cache
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import tempfile
import uuid
import time
import threading
import zlib

import numpy as np

from steward.ego import capabilities
from steward.math3d import wrap_angle

SCHEMA = 'steward.robot_memory.v1'
PERIOD_S = 30.0  # robot time, independent of wall-clock / perception throughput
# State ownership is explicit: adding an estimator cache never silently persists it.
ESTIMATOR_FIELDS = (
    'tracks', 'held_id', 'held_gripper', 'held_since', 'rigid_carry',
    'manipulation_envelopes', 'flip_capture', 'fastening_attempts',
    'released_interfaces', 'fastener_origins', 'fastener_kits', 'seat_assemblies',
    'fastener_releases', 'assembly_sessions', 'orientation_history', '_opened_drawers', '_pulled_open',
    '_pushed_shut', '_handle', '_handle_travel', '_shut_follow', 'content_hits',
    'holding_belief', 'geometry_rejections', '_seat_registrations',
    '_component_alignment_attempts', '_interface_entry_budgets',
    '_interface_entry_reports', '_wrist_landing_history',
)
CONTROLLER_FIELDS = ('_tool_sources', '_driver_misses', '_screw_misses',
    '_seat_retry_counts', '_seat_retry_budget', '_retired_screws', '_missed_screws',
    '_supply_supports', '_stance_retry_used')
TIME_FIELDS = {'t', 'stamp', 'simTime', 'since', 'after', 'started', 'finished',
    'first_seen', 'last_seen', 'front_seen', 'held_since', 'interface_estimate_at',
    'interface_motion_at', 'first_grasp_at'}
TIME_LISTS = {'frameStamps', 'stamps', 'perception_stamps'}
TIME_MAPS = {'initialFaceInvalidatedAt', 'releaseObservedAt'}


# Only fields already cleared by v1 restore are omitted. In particular voxels,
# appearance, registration geometry and action accounting are NOT disposable.
TRACK_TRANSIENT = dict(manipulation_view=None, assembly_observation=None,
    interface_views=deque(maxlen=48), interface_best_near=None,
    interface_sample_status={}, absent_views=[], pinned=False, pin_observed_age=None)
HISTORY_LIMIT = 8


def recent_orientations(rows):
    """Latest eight per body plus its last intentional flip (sticky provenance)."""
    counts, intentional, keep = {}, set(), []
    for row in reversed(rows):
        body = row.get('body')
        n = counts.get(body, 0)
        relevant = row.get('intentional') and body not in intentional
        if n < HISTORY_LIMIT or relevant:
            keep.append(row)
        counts[body] = n + 1
        if row.get('intentional'): intentional.add(body)
    return list(reversed(keep))


def longest_flips(rows):
    # Final-pose allowance consumes max(simS) per body, never sample order/count.
    best = {}
    for row in rows:
        body = row['body']
        if body not in best or row['simS'] >= best[body]['simS']: best[body] = row
    return list(best.values())


def latest_motion(rows):
    # All admission readers use any/max stamp for an entity; no trajectory reader.
    best = {}
    for row in rows:
        eid = row['entity']
        if eid not in best or row['stamp'] >= best[eid]['stamp']: best[eid] = row
    return list(best.values())


def compact_entry_report(report):
    """Bound repeated interface observations while retaining exact public counts."""
    rows = report.get('observations', [])
    if len(rows) <= HISTORY_LIMIT: return report
    older = rows[:-HISTORY_LIMIT]
    totals = deepcopy(report.get('observationArchive', {}))
    totals['count'] = totals.get('count', 0) + len(older)
    for kind in ('orientation', 'support'):
        previous = totals.get(kind, {})
        totals[kind] = dict(
            measured=previous.get('measured', 0) + sum(bool(o[kind]['measured']) for o in older),
            mismatch=previous.get('mismatch', 0) + sum(o[kind]['matches'] is False for o in older))
    return dict(report, observations=rows[-HISTORY_LIMIT:], observationArchive=totals)


def snapshot_copy(value, memo=None):
    """Copy owned mutable state, skipping dead observation buffers before traversal.

    One memo preserves shared recovery Tracks instead of repeatedly copying them.
    No JSON, compression, hashing, public export or filesystem work happens here.
    """
    if memo is None: memo = {}
    if value is None or isinstance(value, (str, bool, int, float, np.generic)):
        return value
    if id(value) in memo: return memo[id(value)]
    # Keep temporary slices alive so CPython cannot recycle a memoized id.
    memo.setdefault('_sources', []).append(value)
    if dataclasses.is_dataclass(value):
        result = copy(value); memo[id(value)] = result
        attrs = vars(value)
        if type(value).__name__ == 'Track':
            attrs = dict(attrs, **TRACK_TRANSIENT)
            attrs['interface_views'] = deque(maxlen=value.interface_views.maxlen)
        for k, v in attrs.items(): setattr(result, k, snapshot_copy(v, memo))
        return result
    if isinstance(value, dict):
        result = {}; memo[id(value)] = result
        source = compact_entry_report(value) if value.get('schema') == 'interface_entry.v1' else value
        for k, v in source.items():
            # Diagnostic-only drive transcript; current drive status is separate.
            if k == 'driveHistory' and isinstance(v, list): v = v[-HISTORY_LIMIT:]
            result[k] = snapshot_copy(v, memo)
        return result
    if isinstance(value, list):
        result = []; memo[id(value)] = result
        result.extend(snapshot_copy(v, memo) for v in value)
        return result
    if isinstance(value, deque):
        result = deque(maxlen=value.maxlen); memo[id(value)] = result
        result.extend(snapshot_copy(v, memo) for v in value)
        return result
    if isinstance(value, (tuple, set)):
        result = type(value)(snapshot_copy(v, memo) for v in value)
        memo[id(value)] = result
        return result
    return deepcopy(value, memo)


def _setting(name, default, integer=False):
    try:
        value = int(os.environ[name]) if integer else float(os.environ[name])
        if value < 0 or not math.isfinite(value) or (not integer and value == 0):
            raise ValueError(name)
        return value
    except (KeyError, ValueError, OverflowError):
        return default


def enabled():
    return capabilities.get('STEWARD_ASSEMBLY_MEMORY_CHECKPOINT') != '0'


@lru_cache(maxsize=1)
def _types():
    from steward.ego.estimator import Track, Appearance, _HandleHold, _ShutFollow
    from steward.scene.schema import Entity, Est, Link, Joint, RigidTree, Relation
    return {c.__name__: c for c in (Track, Appearance, _HandleHold, _ShutFollow,
        Entity, Est, Link, Joint, RigidTree, Relation)}


def encode(value):
    """Small closed data codec; arbitrary classes/code cannot be loaded."""
    if isinstance(value, np.generic):
        return encode(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {'$': 'float', 'value': str(value)}
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in 'biuf':
            raise ValueError('non-numeric memory array')
        return {'$': 'ndarray', 'dtype': value.dtype.str, 'shape': list(value.shape),
                'value': base64.b64encode(value.tobytes(order='C')).decode('ascii')}
    if dataclasses.is_dataclass(value) and type(value).__name__ in _types():
        return {'$': type(value).__name__, 'value': encode(vars(value))}
    if isinstance(value, dict):
        return {'$': 'dict', 'value': [[encode(k), encode(v)] for k, v in list(value.items())]}
    if isinstance(value, (list, tuple, set, deque)):
        kind = type(value).__name__
        return {'$': kind, 'value': [encode(v) for v in value],
                **({'maxlen': value.maxlen} if isinstance(value, deque) else {})}
    raise ValueError('unsupported memory type: ' + type(value).__name__)


def decode(value):
    if not isinstance(value, dict):
        if isinstance(value, (list,)) or not (value is None or isinstance(value, (str, bool, int, float))):
            raise ValueError('invalid memory leaf')
        return value
    kind, rows = value['$'], value['value']
    if kind == 'zlib-json':
        return decode(json.loads(zlib.decompress(base64.b64decode(rows, validate=True))))
    if kind == 'float':
        if rows not in ('nan', 'inf', '-inf'): raise ValueError('invalid float')
        return float(rows)
    if kind == 'dict': return {decode(k): decode(v) for k, v in rows}
    if kind in ('list', 'tuple', 'set', 'deque'):
        items = [decode(v) for v in rows]
        return {'list': list, 'tuple': tuple, 'set': set, 'deque': lambda a: deque(a, maxlen=value['maxlen'])}[kind](items)
    if kind == 'ndarray':
        dtype = np.dtype(value['dtype'])
        if dtype.kind not in 'biuf': raise ValueError('non-numeric memory array')
        return np.frombuffer(base64.b64decode(rows, validate=True), dtype=dtype).reshape(value['shape']).copy()
    if kind == 'array':
        dtype = np.dtype(value['dtype'])
        if dtype.kind not in 'biuf': raise ValueError('non-numeric memory array')
        return np.asarray(decode(rows), dtype=dtype)
    cls = _types().get(kind)
    if cls is None: raise ValueError('unknown memory type: ' + str(kind))
    attrs = decode(rows)
    fields = {f.name for f in dataclasses.fields(cls) if f.init}
    result = cls(**{k: v for k, v in attrs.items() if k in fields})
    # Track extensions contain robot-owned registration/grasp estimates.
    if kind != 'Track' and set(attrs) - fields: raise ValueError('unexpected data fields')
    vars(result).update(attrs)
    return result


def _timestamp(key):
    return isinstance(key, str) and (key in TIME_FIELDS or key.endswith(('At', 'Stamp', '_at', 'Since')))


def rebase(value, delta, key=None):
    """Translate timestamps, not durations, ages, geometry, counters or wall time."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value + delta if _timestamp(key) and math.isfinite(value) else value
    if dataclasses.is_dataclass(value):
        for k, v in vars(value).items(): setattr(value, k, rebase(v, delta, k))
    elif isinstance(value, dict):
        for k, v in list(value.items()):
            value[k] = rebase(v, delta, 'stamp' if key in TIME_MAPS else k)
    elif isinstance(value, (list, tuple, deque)):
        if key in ('absent_views', 'content_hits'):
            rows = [(r[0]+delta, *r[1:]) for r in value]
        elif key == 'release' and isinstance(value, tuple) and len(value) == 2:
            rows = [value[0], rebase(value[1], delta, 'stamp')]
        else:
            rows = [rebase(v, delta, 'stamp' if key in TIME_LISTS else key) for v in value]
        if isinstance(value, deque): return deque(rows, maxlen=value.maxlen)
        return tuple(rows) if isinstance(value, tuple) else rows
    return value


def _prior(value, restored_at=None, needs_recheck=False):
    """Tag evidence without falsifying historical action/confirmation ledgers."""
    from steward.ego.estimator import Track
    from steward.scene.schema import Est
    if isinstance(value, Est):
        if value.src in ('observed', 'fused'): value.src = 'prior'
        _prior(value.value, restored_at, needs_recheck)
    elif dataclasses.is_dataclass(value):
        for v in list(vars(value).values()): _prior(v, restored_at, needs_recheck)
        if isinstance(value, Track):
            value.restored_prior = dict(restoredAt=restored_at, needsRecheck=needs_recheck,
                originalLastSeen=value.last_seen, originalPoseSource=(
                    'grasp_relative_plus_encoders' if value.held else
                    'fused' if value.observations > 1 else 'observed'))
            value.pinned = False; value.pin_observed_age = None
            # Also clear embedded recovery-source Tracks: resurrecting an old
            # source must never resurrect a current-frame cache.
            value.manipulation_view = value.assembly_observation = None
            value.interface_views.clear(); value.interface_best_near = None
            value.interface_sample_status = {}; value.absent_views.clear()
    elif isinstance(value, dict):
        if any(_timestamp(k) for k in value): value['restoredPrior'] = True
        for v in list(value.values()): _prior(v, restored_at, needs_recheck)
    elif isinstance(value, (list, tuple, deque)):
        for v in value: _prior(v, restored_at, needs_recheck)


def _body(est):
    return {k: deepcopy(getattr(est, k)) for k in ('stamp', 'odometry', 'joints', 'gripper', 'grippers')}


def body_check(saved, live):
    a, b = saved['odometry'], live['odometry']
    distance = math.hypot(a['x']-b['x'], a['y']-b['y'])
    yaw = abs(wrap_angle(a['yaw']-b['yaw']))
    sigma_a, sigma_b = a.get('sigma', {}), b.get('sigma', {})
    pos_noise = math.hypot(sigma_a.get('pos', 0.), sigma_b.get('pos', 0.))
    yaw_noise = math.hypot(sigma_a.get('yaw', 0.), sigma_b.get('yaw', 0.))
    # Encoder covariance already used by the robot's published self-state.
    joint_noise = math.sqrt(2e-6)
    keys = set(saved['joints']) | set(live['joints'])
    joints = {k: abs(saved['joints'][k]-live['joints'][k])
              if k in saved['joints'] and k in live['joints'] else None for k in keys}
    grip = any(saved.get(k, {}) != live.get(k, {}) for k in ('gripper', 'grippers'))
    mismatch = (distance > 3*pos_noise or yaw > 3*yaw_noise or grip or
                any(v is None or v > 3*joint_noise for v in joints.values()))
    return dict(positionDeltaM=distance, yawDeltaRad=yaw, jointDeltas=joints,
        gripperChanged=grip, exceedsNoise=mismatch,
        noise=dict(positionSigmaM=pos_noise, yawSigmaRad=yaw_noise, jointSigma=joint_noise))


def _counts(data):
    result = {k: len(v) if isinstance(v, (dict, list, set, tuple, deque)) else int(v is not None)
              for k, v in data['estimator'].items()}
    result.update(assembly_memory=len(data['assembly_memory']['records']),
                  stationary_assemblies=len(data['stationary_assemblies']['records']))
    return result


def _checksum(row):
    return hashlib.sha256(_json({k: v for k, v in row.items() if k != 'sha256'}).encode()).hexdigest()


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _track_issues(track):
    """Local numeric defects are not evidence that the archive is corrupt."""
    issues = []
    for field, shape, lower, inclusive in (
            ('pos', (3,), None, True), ('size', (3,), 0, False),
            ('var', (3,), 0, True), ('yaw', (), None, True),
            ('yaw_var', (), 0, True)):
        value = getattr(track, field)
        try:
            a = np.asarray(value)
        except (ValueError, TypeError):
            issues.append(dict(field=field, value=repr(value), reason='invalid numeric shape/type'))
            continue
        reason = None
        if a.shape != shape or a.dtype.kind not in 'iuf':
            reason = 'expected finite numeric ' + ('3-vector' if shape else 'scalar')
        elif not np.isfinite(a).all():
            reason = 'non-finite value'
        elif lower is not None and np.any(a < lower if inclusive else a <= lower):
            reason = 'must be nonnegative' if inclusive else 'must be strictly positive'
        if reason:
            # JSON-safe original values, including NaN/Inf, without losing zeros.
            def safe(v):
                if isinstance(v, list): return [safe(x) for x in v]
                if isinstance(v, float) and not math.isfinite(v): return str(v)
                return v if v is None or isinstance(v, (str, bool, int, float)) else repr(v)
            issues.append(dict(field=field, value=safe(a.tolist()), reason=reason))
    return issues


def _sanitize_tracks(data, phase):
    """Quarantine invalid geometry, including embedded recovery-source Tracks.

    Never invent a pose/extent. Identity and container errors remain fatal. A
    record owning a required Track (e.g. driverRelease.track) is removed with it;
    an ID -> Track cache loses only that entry. Historical action ledgers remain.
    """
    from steward.ego.estimator import Track
    adjustments = []
    drop = object()

    def visit(value, path):
        if isinstance(value, Track):
            if not isinstance(value.id, str): raise ValueError('invalid track identity')
            issues = _track_issues(value)
            if issues:
                adjustments.append(dict(track=value.id, path=path, action='dropped',
                    phase=phase, issues=issues))
                return drop
            return value
        if isinstance(value, dict):
            discard_record = False
            for k, v in list(value.items()):
                cleaned = visit(v, path + '.' + str(k))
                if cleaned is drop:
                    del value[k]
                    if isinstance(v, Track) and k != v.id:
                        discard_record = True
                        adjustments[-1]['droppedContainer'] = path
                else:
                    value[k] = cleaned
            return drop if discard_record else value
        if isinstance(value, (list, tuple, deque)):
            rows = [visit(v, path + '.' + str(i)) for i, v in enumerate(value)]
            rows = [v for v in rows if v is not drop]
            return deque(rows, maxlen=value.maxlen) if isinstance(value, deque) else type(value)(rows)
        return value

    visit(data, 'payload')
    state = data['estimator']
    dropped = {r['track'] for r in adjustments if r['path'].startswith('payload.estimator.tracks.')}
    # A removed estimate must not reappear through a frozen scene overlay.
    for eid, record in list(data['stationary_assemblies']['records'].items()):
        affected = dropped.intersection((eid, record['support'].id))
        if affected:
            del data['stationary_assemblies']['records'][eid]
            for row in adjustments:
                if row['track'] in affected:
                    row.setdefault('dependentMemoryRemoved', []).append('stationary_assemblies.records.' + eid)
    for record in data['assembly_memory']['records']:
        if dropped.intersection([record['entity'], record['support'], record['reference']['entity'],
                                 *(p['entity'] for p in record['parts'])]):
            record['status'] = 'invalidated'
    if state.get('held_id') in dropped:
        state['held_id'] = None
        state['rigid_carry'] = {}
    for field in ('_seat_registrations',):
        for eid, record in state.get(field, {}).items():
            if eid in dropped or record.get('body') in dropped:
                record.update(needsRecheck=True, invalidated='memory_track_dropped')
    watch = data['runtime'].get('supportHoleWatch')
    if watch and dropped.intersection((watch.get('entity'), watch.get('body'))):
        watch.update(needsRecheck=True, invalidated='memory_track_dropped')
    return adjustments


class Memory:
    def __init__(self, rt):
        self.rt = rt
        self.directory = os.environ.get('STEWARD_MEMORY_CHECKPOINT_DIR') if enabled() else None
        self.resume = os.environ.get('STEWARD_MEMORY_RESUME') if enabled() else None
        self.initialized = False
        self.last_save = -math.inf
        self.sequence = 0
        self.run_id = uuid.uuid4().hex
        self.receipts = []
        self.restore_report = None
        self.latest_body = None
        self.pending_faces = {}
        self.period_s = _setting('STEWARD_MEMORY_CHECKPOINT_PERIOD_S', PERIOD_S)
        self.keep = _setting('STEWARD_MEMORY_CHECKPOINT_KEEP', None, integer=True)
        self._writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix='memory-writer')
        self._receipt_lock = threading.Lock()
        self._periodic_future = None
        self._periodic_files = {}
        self.receipt_directory = os.environ.get('STEWARD_RECEIPT_LOG_DIR')

    def emit(self, event, **data):
        row = dict(schema=SCHEMA, event=event, **{'robotTime': self.clock_stamp(), **data})
        with self._receipt_lock:
            self.receipts.append(row)
            self.receipts[:] = self.receipts[-32:]
        if self.receipt_directory:
            if threading.current_thread().name.startswith('memory-writer'):
                self._write_receipt(row)
            else:
                self._writer.submit(self._write_receipt, row)
        return row

    def _write_receipt(self, row):
        try:
            path = Path(self.receipt_directory); path.mkdir(parents=True, exist_ok=True)
            with (path / f'memory-{os.getpid()}.jsonl').open('a', encoding='utf-8') as out:
                out.write(_json(row)+'\n')
        except (OSError, ValueError, TypeError):
            pass

    def flush(self, timeout=None):
        """Explicit durability barrier for shutdown/tests; never used by control."""
        self._writer.submit(lambda: None).result(timeout=timeout)

    def skill_finished(self, job):
        outcome = getattr(job, 'postcondition', None) or {}
        if job.skill == 'flip_assembly' and outcome.get('releaseExecuted'):
            with self.rt._lock:
                self.pending_faces.pop(outcome.get('body'), None)
                history = getattr(self.rt.estimator, 'orientation_history', [])
                history.append(dict(body=outcome.get('body'), completedAt=job.finished,
                    intentional=outcome.get('commandedTurnOver', True),
                    observedFlipped=outcome.get('flipped'), confirmation=outcome.get('confirmation'),
                    faceResolution=deepcopy(outcome.get('faceResolution'))))
                self.rt.estimator.orientation_history = recent_orientations(history)
        self.save('skill_finished:' + job.skill, force=True)

    def reset(self):
        # A later sensor-episode reset is not another request to replay this file.
        self.resume = None
        self.restore_report = None
        self.latest_body = None
        self.pending_faces.clear()
        self.run_id = uuid.uuid4().hex
        self.sequence = 0
        self.last_save = -math.inf

    def note_body(self, header):
        if (header and header.get('stamp') is not None and header.get('odometry') is not None
                and header.get('joints') is not None):
            sample = {k: deepcopy(header.get(k, {})) for k in
                      ('stamp', 'odometry', 'joints', 'gripper', 'grippers')}
            if self.latest_body is None or sample['stamp'] >= self.latest_body['stamp']:
                self.latest_body = sample

    def clock_stamp(self):
        return max(self.rt.estimator.stamp, (self.latest_body or {}).get('stamp', -math.inf))

    def capture(self, reason):
        est = self.rt.estimator
        state = {k: getattr(est, k) for k in ESTIMATOR_FIELDS if hasattr(est, k)}
        sessions = {}
        for sid, original in state.get('assembly_sessions', {}).items():
            row = {k: v for k, v in original.items() if k not in ('controller', 'deadline', 'heartbeatWall')}
            controller = original.get('controller')
            if controller is not None:
                row['savedControllerMemory'] = {k: getattr(controller, k) for k in CONTROLLER_FIELDS if hasattr(controller, k)}
            else:
                row.setdefault('savedControllerMemory', {})
            if original.get('deadline') is not None: row['savedBudget'] = original['deadline'].report()
            sessions[sid] = row
        state['assembly_sessions'] = sessions
        if 'orientation_history' in state:
            state['orientation_history'] = recent_orientations(state['orientation_history'])
        for session in sessions.values():
            if 'flipHistory' in session:
                session['flipHistory'] = recent_orientations(session['flipHistory'])
            if 'flipDurations' in session:
                session['flipDurations'] = longest_flips(session['flipDurations'])
        saved_body = self.latest_body if self.latest_body and self.latest_body['stamp'] >= est.stamp else _body(est)
        data = dict(estimator=state, assembly_memory=vars(est.assembly_memory),
            stationary_assemblies=vars(est.stationary_assemblies), body=saved_body,
            runtime=dict(plan=getattr(self.rt, 'plan', None), roles=getattr(self.rt, '_roles', {}),
                         supportHoleWatch=getattr(self.rt, '_support_hole_watch', None)),
            nextTrackId=est._ids.__reduce__()[1][0])
        # motionSamples/contactView are also discarded by _prepare on restore.
        stationary = dict(data['stationary_assemblies'])
        records = {}
        for key, record in stationary['records'].items():
            record = dict(record, motionSamples={})
            for field in ('entity', 'support'):
                entity = copy(record[field])
                entity.state_vars = {k: v for k, v in entity.state_vars.items() if k != 'contactView'}
                record[field] = entity
            records[key] = record
        stationary['records'] = records
        stationary['motion_events'] = latest_motion(stationary.get('motion_events', []))
        data['stationary_assemblies'] = stationary
        detached = snapshot_copy(data)
        return dict(schema=SCHEMA, robotTime=float(self.clock_stamp()), run=self.run_id,
            sequence=self.sequence, reason=reason, sections=_counts(data), payload=detached,
            bodySnapshotStamp=float(saved_body['stamp']), held=est.held_id)

    def _serialize(self, row):
        from types import SimpleNamespace
        from steward.ego.public import public_json
        data = row['payload']
        row['trackAdjustments'] = self._validate(data, phase='write')
        row['sections'] = _counts(data)
        est = SimpleNamespace(**data['estimator'])
        est.stationary_assemblies = SimpleNamespace(**data['stationary_assemblies'])
        rt = SimpleNamespace(estimator=est)
        row['progressSummary'] = public_json([progress(rt, s) for s in est.assembly_sessions.values()])
        row['orientationHistory'] = public_json(getattr(est, 'orientation_history', []))
        # Lossless compression retains geometry, IDs, budgets and all release ledgers.
        row['payload'] = {'$': 'zlib-json', 'value': base64.b64encode(
            zlib.compress(_json(encode(data)).encode('utf-8'), level=1)).decode('ascii')}
        row['sha256'] = _checksum(row)
        blob = (_json(row)+'\n').encode('utf-8')
        # Exercise the actual reader, including preparation, on the exact bytes
        # before publication. Runs on the writer thread, never under control lock.
        header = dict(data['body'], stamp=row['robotTime'])
        candidate = self._restore_candidate(json.loads(blob), header)
        if candidate[-1]: raise ValueError('memory readback changed serialized tracks')
        return blob

    def save(self, reason='periodic', force=False):
        if not enabled() or not self.directory or not self.initialized: return None
        start = time.perf_counter()
        with self.rt._lock:
            now = self.clock_stamp()
            if not force and now-self.last_save < self.period_s: return None
            # At most one periodic snapshot in flight on a slow shared volume.
            # Events are never coalesced: each owns its state at the event boundary.
            if not force and self._periodic_future is not None and not self._periodic_future.done():
                return None
            self.sequence += 1
            try:
                row = self.capture(reason)
                path = Path(self.directory) / f'memory-{now:020.6f}-{self.run_id}-{self.sequence:08d}.json'
                metrics = dict(firstObservationConflicts=self.conflicts)
                ready = threading.Event()
                future = self._writer.submit(self._write, row, path, reason == 'periodic' and not force, metrics, ready)
                if not force: self._periodic_future = future
                self.last_save = now
                metrics['mainThreadMs'] = (time.perf_counter()-start)*1000
                ready.set()
                return path  # scheduled destination; checkpoint_saved is the durable acknowledgement
            except Exception as exc:
                self.emit('checkpoint_rejected', reason=str(exc), bytes=0,
                          mainThreadMs=(time.perf_counter()-start)*1000, backgroundWriteMs=0.)
                return None

    def _write(self, row, destination, periodic, metrics, ready):
        ready.wait()
        start = time.perf_counter(); temporary = None; size = 0; write_start = None
        try:
            blob = self._serialize(row); size = len(blob)
            write_start = time.perf_counter()
            path = destination.parent; path.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode='wb', dir=path, prefix='.memory-', suffix='.tmp', delete=False) as out:
                temporary = Path(out.name)
                out.write(blob); out.flush(); os.fsync(out.fileno())
            os.replace(temporary, destination)
            descriptor = os.open(path, os.O_RDONLY)
            try: os.fsync(descriptor)
            finally: os.close(descriptor)
            if periodic and self.keep is not None:
                files = self._periodic_files.setdefault(row['run'], deque())
                files.append(destination)
                while len(files) > self.keep:
                    files[0].unlink(missing_ok=True); files.popleft()
            self.emit('checkpoint_saved', file=str(destination), reason=row['reason'],
                robotTime=row['robotTime'], sections=row['sections'], bytes=size,
                trackAdjustments=row['trackAdjustments'], readBackChecked=True,
                backgroundWriteMs=(time.perf_counter()-write_start)*1000,
                backgroundTotalMs=(time.perf_counter()-start)*1000,
                clockShiftS=0., bodyCheck=None, **metrics)
        except Exception as exc:
            self.emit('checkpoint_rejected', file=str(destination), reason=str(exc),
                robotTime=row['robotTime'], checkpointReason=row['reason'], bytes=size,
                backgroundWriteMs=(time.perf_counter()-(write_start or start))*1000,
                backgroundTotalMs=(time.perf_counter()-start)*1000, **metrics)
        finally:
            if temporary is not None:
                try: temporary.unlink(missing_ok=True)
                except OSError: pass

    @property
    def conflicts(self):
        return len((self.restore_report or {}).get('conflicts', []))

    def before_observation(self, header):
        # Serialize initialization with perception and API state reads. No thread
        # may fuse a new observation while another is still installing priors.
        with self.rt._lock:
            self.note_body(header)
            if self.initialized: return
            self.initialized = True
            if not self.resume or not enabled(): return
            try:
                row = json.loads(Path(self.resume).read_text(encoding='utf-8'))
                data, check, delta, estimator, watch, adjustments = self._restore_candidate(row, header)
                now = float(header['stamp'])
                faces = {eid: deepcopy(r['reference'].get('faceResolution'))
                    for eid, r in estimator.stationary_assemblies.records.items()
                    if r['reference'].get('faceResolution')}
                report = dict(file=self.resume, sections=_counts(data), clockShiftS=delta,
                    trackAdjustments=[*row.get('trackAdjustments', []), *adjustments],
                    bodyCheck=check, conflicts=[], firstObservationConflicts=0,
                    firstObservationStatus='pending', restoredAt=now)
                # Commit only after decoding, rebasing, validation and preparation
                # all succeed. No fallible parsing follows these assignments.
                self.rt.estimator = estimator
                self.rt.plan = data['runtime']['plan']
                self.rt._roles = data['runtime']['roles']
                self.rt._support_hole_watch = watch
                self.pending_faces = faces
                self.restore_report = report
                self.emit('memory_restored', **deepcopy(report))
            except Exception as exc:  # Reject the whole archive, retaining the empty estimator.
                self.restore_report = dict(file=self.resume, status='rejected', reason=str(exc),
                    firstObservationConflicts=0, sections={})
                self.emit('memory_resume_rejected', **self.restore_report)

    def _restore_candidate(self, row, header):
        if row.get('schema') != SCHEMA: raise ValueError('unsupported memory version')
        if _checksum(row) != row['sha256']: raise ValueError('memory checksum mismatch')
        old, now = float(row['robotTime']), float(header['stamp'])
        if not all(math.isfinite(t) for t in (old, now)): raise ValueError('invalid robot clock')
        if not isinstance(row.get('trackAdjustments', []), list):
            raise ValueError('invalid track adjustment log')
        data = decode(row['payload'])
        adjustments = self._validate(data)
        check = body_check(data['body'], header)
        delta = now-old
        rebase(data, delta)
        _prior(data, now, check['exceedsNoise'])
        estimator, watch = self._prepare(data, header, check)
        return data, check, delta, estimator, watch, adjustments

    def _validate(self, data, phase='restore'):
        from steward.ego.estimator import Track
        state = data['estimator']
        if not isinstance(data['runtime']['roles'], dict): raise ValueError('invalid role memory')
        if data['runtime']['plan'] is not None and not isinstance(data['runtime']['plan'], dict):
            raise ValueError('invalid progress memory')
        if set(state)-set(ESTIMATOR_FIELDS): raise ValueError('unknown memory section')
        for eid, t in state['tracks'].items():
            if not isinstance(t, Track) or t.id != eid: raise ValueError('invalid track identity')
        for sid, s in state['assembly_sessions'].items():
            if s['id'] != sid or not isinstance(s['goal']['interfaces'], list): raise ValueError('invalid session')
            if not math.isfinite(s['goal']['budgetSimS']) or s['goal']['budgetSimS'] <= 0: raise ValueError('invalid budget')
        consumed = [int(e[2:]) for e in state['tracks'] if e.startswith('e:') and e[2:].isdigit()]
        if not isinstance(data['nextTrackId'], int) or data['nextTrackId'] <= max(consumed, default=0):
            raise ValueError('invalid track sequence')
        if not isinstance(data['assembly_memory']['records'], list): raise ValueError('invalid assembly records')
        if not isinstance(data['stationary_assemblies']['records'], dict): raise ValueError('invalid stationary records')
        return _sanitize_tracks(data, phase)

    def _prepare(self, data, header, check):
        from steward.ego.estimator import EgoEstimator
        original = self.rt.estimator
        est = EgoEstimator(robot_model=original.robot_model, config=original.config)
        est.precision_probe_path = getattr(original, 'precision_probe_path', None)
        # Prepare everything before mutating the running estimator.
        state = data['estimator']
        for s in state['assembly_sessions'].values():
            s['generation'] = self.rt._generation
            s['controller'] = None
            s['memoryResumed'] = True
            s['memoryRestoredAt'] = float(header['stamp'])
            s['action'] = None
            s['reviewingReleases'] = False
            s['memoryNeedsRecheck'] = check['exceedsNoise']
        from steward.ego import geometry_relocalization as geometry
        geometry.historical_receipts(data, float(header['stamp']))
        for record in data['stationary_assemblies']['records'].values():
            geometry.pending(record['reference'], float(header['stamp']), 'memory_restored')
            record['restoredPrior'] = True
            record['needsRecheck'] = check['exceedsNoise']
            record['motionSamples'] = {}
            for entity in (record['entity'], record['support']):
                entity.state_vars.pop('contactView', None)
            if check['exceedsNoise']:
                record['reference'].get('componentHoleRegistration', {}).update(
                    invalidated='memory_body_changed', needsRecheck=True)
        # IDs consumed before dropped tracks must not be reused.
        ids = itertools.count(data['nextTrackId'])
        for k, v in state.items(): setattr(est, k, v)
        vars(est.assembly_memory).update(data['assembly_memory'])
        vars(est.stationary_assemblies).update(data['stationary_assemblies'])
        from steward.ego import session_identity
        if session_identity.enabled():
            for session in est.assembly_sessions.values():
                session_identity.restored(est, session)
        est._ids = ids
        est._assembly_views = {}
        est.stamp = float(header['stamp'])
        est.odometry = deepcopy(header['odometry']); est.joints = deepcopy(header['joints'])
        est._read_grippers(header)
        est.posed = True
        if est.gripper.get('holding') is False:
            est._unmark_held(est.tracks.get(est.held_id))
            est.rigid_carry = {}
        for registration in getattr(est, '_seat_registrations', {}).values():
            registration['needsRecheck'] = check['exceedsNoise'] or bool(registration.get('invalidated'))
            if check['exceedsNoise']: registration['invalidated'] = 'memory_body_changed'
        watch = data['runtime'].get('supportHoleWatch')
        if watch is not None:
            watch['needsRecheck'] = check['exceedsNoise'] or bool(watch.get('invalidated'))
            if check['exceedsNoise']: watch['invalidated'] = 'memory_body_changed'
        if watch and watch.get('entity') in getattr(est, '_seat_registrations', {}):
            cached = est._seat_registrations[watch['entity']]
            if all(cached.get(k) == watch.get(k) for k in ('body', 'hole', 'observedAt')):
                watch = cached  # feature-motion invalidation must reach the reusable registration
        est._memory_checkpoint = self
        return est, watch

    def observed(self, track, position, size, yaw, category, sigma):
        prior = getattr(track, 'restored_prior', None)
        if prior is None: return
        distance = float(np.linalg.norm(track.pos-position))
        # Box heading has no directed face: a half-turn is the same axis.
        turned = abs(wrap_angle(2*(yaw-track.yaw)))/2
        noise = 3*math.sqrt(float(np.max(track.var))+sigma**2)
        size_error = float(np.max(abs(track.size-np.asarray(size))))
        conflict = (distance > noise or turned > 3*math.sqrt(track.yaw_var) or
                    size_error > noise+self.rt.estimator.config.hole_registration_extent_slack_m or
                    category != track.category)
        if conflict:
            row = dict(entity=track.id, stamp=self.rt.estimator.stamp, distanceM=distance,
                yawDeltaRad=turned, sizeDeltaM=size_error, categoryChanged=category != track.category, resolution='observation_wins')
            self.restore_report['conflicts'].append(row)
            self.restore_report['firstObservationConflicts'] = self.conflicts
            # Frozen overlays must not put the superseded prior back on top.
            for r in self.rt.estimator.stationary_assemblies.records.values():
                if track.id in (r['entity'].id, r['support'].id, *r.get('aliases', [])):
                    r.update(status='invalidated', reason='restored_prior_observation_conflict', needsRecheck=True)
                    self._correct_reference(r, track.id, position, size, yaw, category)
            registrations = list(getattr(self.rt.estimator, '_seat_registrations', {}).values())
            watch = getattr(self.rt, '_support_hole_watch', None)
            if watch: registrations.append(watch)
            for reg in registrations:
                if track.id in (reg.get('entity'), reg.get('body')):
                    reg.update(invalidated='memory_observation_conflict', needsRecheck=True)
            for frame in self.rt.estimator.released_interfaces.values():
                if track.id in (frame.get('entity'), frame.get('body')):
                    frame.pop('registeredSeatTargets', None)
                    frame['needsRecheck'] = True
            self.rt.estimator.assembly_memory.invalidate(track.id)
            self.emit('memory_observation_conflict', **row, firstObservationConflicts=self.conflicts)
        # Remove history's votes/union/heading lock before ordinary fusion. The
        # current measurement is authoritative even if the saved variance is tiny.
        track.pos = np.asarray(position, float).copy(); track.size = np.asarray(size, float).copy()
        track.yaw = float(yaw); track.var = np.full(3, sigma**2); track.yaw_var = max(track.yaw_var, sigma**2)
        track.voxels = track.span = None; track.recent.clear(); track.yaw_conflicts = 0
        track.category_counts = {category: 1.}; track.semantic_counts = {}; track.observations = 0
        if conflict and track.label != category: track.label = None
        track.interface_estimate_at = None
        del track.restored_prior

    def _correct_reference(self, record, eid, position, size, yaw, category):
        """Ordinary detections propose geometry; they cannot renew precision."""
        from steward.ego import geometry_relocalization as geometry
        if eid not in (record['entity'].id, record['support'].id, *record.get('aliases', [])): return
        ref = record['reference']
        record.setdefault('memoryPriorReference', deepcopy(ref))
        ref['observationHypothesis'] = dict(position=np.asarray(position).tolist(),
            size=np.asarray(size).tolist(), yaw=float(yaw), category=category,
            observedAt=self.rt.estimator.stamp, source='ordinary_object_observation')
        geometry.pending(ref, self.rt.estimator.stamp, 'ordinary_observation_conflict')
        # Translation-only compatible observations may improve a coarse estimate.
        # Neither their box heading nor dimensions are rigid-interface evidence.
        compatible = (category == ref['category'] and
            np.allclose(size, ref['size'], atol=self.rt.estimator.config.hole_registration_extent_slack_m) and
            abs(wrap_angle(2*(yaw-ref['yaw'])))/2 <= math.radians(10))
        if compatible and eid in (record['entity'].id, *record.get('aliases', [])):
            delta = np.asarray(position)-np.asarray(ref['position'])
            ref.update(position=np.asarray(position).tolist(),
                localizationSource='coarse_observation', poseObservedAt=self.rt.estimator.stamp)
            record['entity'].pose.value['p'] = list(ref['position'])
            for part in record.get('parts', []):
                part['position'] = (np.asarray(part['position'])+delta).tolist()
                part['needsRecheck'] = True

    def reconcile_faces(self):
        """Current registered face evidence wins through the normal face registrar.

        Its first contradiction is audited too, even if the face becomes visible
        after the first wide frame (the geometry fusion alone cannot measure roll).
        """
        if not self.restore_report or self.restore_report.get('status') == 'rejected': return
        for eid, old in list(self.pending_faces.items()):
            record = self.rt.estimator.stationary_assemblies.records.get(eid, {})
            new = record.get('reference', {}).get('faceResolution', {})
            if (not new or new.get('restoredPrior') or
                    new.get('status') != 'observed' or
                    new.get('since', new.get('observedAt', -math.inf)) < self.restore_report['restoredAt']):
                continue
            self.pending_faces.pop(eid)
            if set(new.get('exposedHoles', [])) != set(old.get('exposedHoles', [])):
                row = dict(entity=eid, stamp=self.rt.estimator.stamp, kind='registered_face',
                    resolution='observation_wins', previousExposedHoles=old.get('exposedHoles', []),
                    observedExposedHoles=new.get('exposedHoles', []))
                self.restore_report['conflicts'].append(row)
                self.restore_report['firstObservationConflicts'] = self.conflicts
                self.emit('memory_observation_conflict', **row, firstObservationConflicts=self.conflicts)

    def after_observation(self):
        self.reconcile_faces()
        report = self.restore_report
        if report and report.get('firstObservationStatus') == 'pending' and self.rt.estimator.perception_cycles:
            report['firstObservationStatus'] = 'processed'
            report['unobservedEntities'] = [t.id for t in self.rt.estimator.tracks.values() if hasattr(t, 'restored_prior')]
            self.emit('memory_first_observation', firstObservationConflicts=self.conflicts,
                      unobservedEntities=report['unobservedEntities'], clockShiftS=report['clockShiftS'],
                      bodyCheck=report['bodyCheck'])
        self.save()


def create(rt):
    if not enabled() or not (os.environ.get('STEWARD_MEMORY_CHECKPOINT_DIR') or os.environ.get('STEWARD_MEMORY_RESUME')):
        return None
    return Memory(rt)


def event(rt, reason):
    memory = getattr(rt, '_memory_checkpoint', None)
    if memory is not None:
        with rt._lock:
            memory.reconcile_faces()
            return memory.save(reason, force=True)


def observation_wins(fn):
    """Outside the usual history-based fusion guard; sensor validity still applies."""
    @wraps(fn)
    def call(est, track, *args, **kw):
        memory = getattr(est, '_memory_checkpoint', None)
        if memory is not None and hasattr(track, 'restored_prior'):
            if fn.__name__ == '_fuse_cloud':
                m = args[0]
                category = max(m['probs'], key=m['probs'].get) if m['probs'] else track.category
                sigma = est.config.pos_sigma_base + est.config.pos_sigma_per_m * m['seg'].range
                memory.observed(track, m['centre'], m['size'], m['yaw'], category, sigma)
            else:
                det, pos, yaw, sigma = args
                memory.observed(track, pos, det['size'], yaw, det['category'], sigma)
        return fn(est, track, *args, **kw)
    return call


def progress(rt, state):
    """Model-facing progress is data from the ledger, never a new observation."""
    items = state['goal']['interfaces']; completed = state.get('completed', [])
    aliases = state.get('aliases', {})
    def canonical(eid):
        seen = set()
        while eid in aliases and eid not in seen:
            seen.add(eid); eid = aliases[eid]
        return eid
    remaining = [deepcopy(i) for i in items if i['key'] not in completed]
    ready = [i for i in remaining if set(i.get('dependencies', [])) <= set(completed)]
    next_item = next((i for i in ready if i['key'] == state.get('interface')),
                     (ready or remaining or [None])[0])
    bodies = []
    history = [*getattr(rt.estimator, 'orientation_history', []), *state.get('flipHistory', [])]
    records = rt.estimator.stationary_assemblies.records
    for body in dict.fromkeys(canonical(i['body']) for i in items):
        r = records.get(body, {}); ref = r.get('reference', {})
        track = rt.estimator.tracks.get(body)
        bodies.append(dict(body=body, poseSource='restored_prior' if state.get('memoryResumed') else 'robot_estimate',
            yawRad=ref.get('yaw', getattr(track, 'yaw', None)),
            faceResolution=deepcopy(ref.get('faceResolution')), bodyUpZEstimate=ref.get('bodyUpZEstimate',
                float(np.asarray(ref['rotationM'])[2, 2]) if 'rotationM' in ref else None),
            intentionalFlip=any(canonical(f.get('body')) == body and f.get('intentional') for f in history),
            flipHistory=[deepcopy(f) for f in history if canonical(f.get('body')) == body]))
    from steward.ego.session_identity import public as identity_public
    identity_fields = identity_public(rt.estimator, state)
    next_step = dict(action='continue_interface', interface=next_item) if next_item else dict(action='review_results_and_final_pose')
    if state.get('memoryResumed') and identity_fields:
        next_step.update(deepcopy(identity_fields['identityNextStep']))
        next_step['interface'] = next_item  # Preserve the stable interface schema.
    return dict(**identity_fields, session=state['id'], resumed=bool(state.get('memoryResumed')),
        completedInterfaces=[deepcopy(i) for i in items if i['key'] in completed],
        remainingInterfaces=remaining, bodies=bodies, finalPose=deepcopy(state.get('finalPose', {})),
        nextStep=next_step,
        aliases=deepcopy(aliases),
        unconfirmed=dict(interfaces=deepcopy(state.get('releasedUnconfirmedInterfaces', [])),
            holes=[dict(body=a.get('body'), hole=a.get('hole'), releaseExecuted=bool(a.get('releaseExecuted')))
                for a in rt.estimator.fastening_attempts.values()
                if canonical(a.get('body')) in {b['body'] for b in bodies} and (not a.get('seatConfirmed') or
                    state.get('memoryResumed') and ((a.get('seatCheck') or {}).get('restoredPrior') or
                        (a.get('seatCheck') or {}).get('observedAt', -math.inf) < state['memoryRestoredAt']))],
            initialHoles=[dict(interface=k, hole=h) for k, r in state.get('initialInterfaces', {}).items()
                for h, evidence in r.get('holes', {}).items() if evidence.get('occupied') and not evidence.get('confirmed')],
            priorEntities=[t.id for t in rt.estimator.tracks.values() if hasattr(t, 'restored_prior')],
            details=deepcopy(state.get('unconfirmedInterfaces', {})),
            bodyDependentMemory=bool(state.get('memoryNeedsRecheck'))),
        guidance='Continue recorded progress; restored estimates are priors. Use new observations to correct conflicts. '
                 'If unobservable, change view at most once, execute the best estimate, mark unconfirmed and review after completion.')


def result_fields(rt):
    memory = getattr(rt, '_memory_checkpoint', None)
    if memory is None or memory.restore_report is None: return {}
    with rt._lock:
        memory.reconcile_faces()
    return dict(memoryResume=deepcopy(memory.restore_report),
        orientationHistory=deepcopy(getattr(rt.estimator, 'orientation_history', [])),
        progressSummary=[progress(rt, s) for s in getattr(rt.estimator, 'assembly_sessions', {}).values()])


def owner(rt, params):
    """Match a restated goal (including its remaining subset) by bound identities."""
    if not enabled(): return None
    requested = params.get('interfaces', [])
    if not requested: return None
    matches = []
    for s in getattr(rt.estimator, 'assembly_sessions', {}).values():
        if not s.get('memoryResumed') or s.get('generation') != rt._generation: continue
        def canonical(e):
            seen = set()
            while e in s.get('aliases', {}) and e not in seen:
                seen.add(e); e = s['aliases'][e]
            return e
        def same(r, i):
            return (canonical(r.get('part')) == canonical(i['part']) and
                    canonical(r.get('body')) == canonical(i['body']) and
                    (not r.get('hole') or set(r['hole'] if isinstance(r['hole'], list) else [r['hole']]) == set(i['holes'])) and
                    (not r.get('heldHole') or set(r['heldHole'] if isinstance(r['heldHole'], list) else [r['heldHole']]) == set(i['heldHoles'])))
        if all(isinstance(r, dict) and any(same(r, i) for i in s['goal']['interfaces']) for r in requested): matches.append(s['id'])
    return matches[0] if len(matches) == 1 else None
