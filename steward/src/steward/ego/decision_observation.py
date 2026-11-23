"""Decision-boundary camera scheduling and auditable frame consumption.

Only explicitly declared transit is sparse. Unknown manipulation phases retain
periodic perception. Body reads alone never count as reading visual evidence.
"""
from collections import Counter
from contextlib import contextmanager
from functools import wraps
import json
import os
from pathlib import Path
import threading
import uuid


def enabled(rt):
    return bool(getattr(rt.config, 'decision_observation', False))


def context(ex):
    stack = getattr(ex, '_observation_stack', ())
    if stack:
        return stack[-1]
    job = getattr(ex, 'current', None)
    if job is not None and job.state == 'running':
        return (job.skill, 'transport' if job.skill == 'navigate' else
                'survey' if job.skill == 'look_around' else 'precision',
                getattr(ex, '_observation_thread', None))
    return ('idle', 'idle', None)


def decision(ex, reason, skill=None):
    callback = getattr(ex, '_observe_decision', None)
    if callback is not None:
        callback(reason, skill)


@contextmanager
def scope(ex, skill, *, transport=False):
    stack = getattr(ex, '_observation_stack', ())
    phase = 'transport' if transport else 'survey' if skill == 'look_around' else 'precision'
    ex._observation_stack = (*stack, (skill, phase, threading.get_ident()))
    try:
        decision(ex, 'action_start', skill)
        yield
    finally:
        try:
            decision(ex, 'action_end', skill)
        finally:
            ex._observation_stack = stack


def action(skill, *, transport=False):
    def decorate(fn):
        @wraps(fn)
        def run(ex, *args, **kwargs):
            with scope(ex, skill, transport=transport):
                return fn(ex, *args, **kwargs)
        return run
    return decorate


def controller_action(fn):
    """Compound controllers dispatch child actions on the same skill thread."""
    @wraps(fn)
    def run(controller, *args, **kwargs):
        # Session.action(skill, params), Fasten.step(job, skill, params).
        skill = kwargs.get('skill')
        if skill is None:
            skill = args[0] if isinstance(args[0], str) else args[1]
        with scope(controller.rt.executor, skill, transport=skill == 'navigate'):
            return fn(controller, *args, **kwargs)
    return run


class Evidence(dict):
    """Track access to the visual part of a motion snapshot, not encoder ticks."""
    def __init__(self, values, read):
        super().__init__(values)
        self._read = read

    def __getitem__(self, key):
        self._read()
        return super().__getitem__(key)

    def get(self, *args):
        self._read()
        return super().get(*args)

    def items(self):
        self._read()
        return super().items()

    def values(self):
        self._read()
        return super().values()

    def __iter__(self):
        self._read()
        return super().__iter__()

    def __contains__(self, key):
        self._read()
        return super().__contains__(key)

    def __len__(self):
        self._read()
        return super().__len__()

    def keys(self):
        self._read()
        return super().keys()


class Audit:
    def __init__(self):
        self.run = uuid.uuid4().hex
        self.lock = threading.RLock()
        self.sequence = 0
        self.frames = {}
        self.latest = {}
        self.reads = set()
        self.body_updates = 0
        self.errors = 0

    def emit(self, event, **fields):
        directory = os.environ.get('STEWARD_RECEIPT_LOG_DIR')
        if not directory:
            return
        try:
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            with (path / f'observations-{os.getpid()}.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(dict(schema='decision_observation.v1', run=self.run,
                    event=event, **fields), allow_nan=False)+'\n')
        except (OSError, ValueError, TypeError):
            pass

    def processed(self, raw, *, skill, phase, reason, fused=False):
        with self.lock:
            identity = raw.get('_observation_id')
            if identity is None:
                self.sequence += 1
                identity = self.sequence
                raw['_observation_id'] = identity
                row = dict(id=identity, camera=raw['camera'], stamp=raw['header']['stamp'],
                    skill=skill, phase=phase, reason=reason,
                    duringAction=skill != 'idle' and reason in ('periodic', 'targeted_request'))
                self.frames[identity] = row
                self.emit('processed', **row)
            if fused:
                self.latest[raw['camera']] = identity
            return identity

    def consume(self, skill, ids=None, reason='scene_read'):
        if skill == 'idle':
            return
        with self.lock:
            ids = tuple(self.latest.values()) if ids is None else tuple(ids)
            fresh = [i for i in ids if i in self.frames and (skill, i) not in self.reads]
            self.reads.update((skill, i) for i in fresh)
            if fresh:
                self.emit('read', skill=skill, frames=fresh, reason=reason)

    def snapshot(self):
        with self.lock:
            by = {}
            for row in self.frames.values():
                group = by.setdefault(row['skill'], Counter())
                group['fullFrames'] += 1
                group['duringActionFrames'] += int(row['duringAction'])
                group['precisionFrames'] += int(row['phase'] == 'precision')
            for skill, _ in self.reads:
                by.setdefault(skill, Counter())['decisionReadFrames'] += 1
            for group in by.values():
                for key in ('fullFrames', 'duringActionFrames', 'precisionFrames', 'decisionReadFrames'):
                    group.setdefault(key, 0)
            return dict(fullFrames=len(self.frames),
                duringActionFrames=sum(r['duringAction'] for r in self.frames.values()),
                precisionFrames=sum(r['phase'] == 'precision' for r in self.frames.values()),
                decisionReadFrames=len({i for _, i in self.reads}),
                byAction={k:dict(v) for k,v in by.items()}, bodyUpdates=self.body_updates,
                observationErrors=self.errors, run=self.run,
                coverage='recorded', unit='camera_frame; reads deduplicated per action type')
