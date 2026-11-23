"""Passive job timing. Sensor stamps are copied only from reads already requested.

Rows are inclusive (nested rows must not be added); exclusive values partition
the job. Wall seconds are measured, never inferred from the simulation ratio.
"""
from steward.ego import capabilities
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import time

_active = ContextVar('phase_timing', default=None)


def enabled():
    return capabilities.get('STEWARD_ASSEMBLY_PHASE_TIMING') != '0'


@contextmanager
def phase(name):
    session = _active.get()
    if session is None:
        yield
        return
    start, stamp = time.monotonic(), session['stamp']
    frame = [0., 0.]
    session['stack'].append(frame)
    try:
        yield
    finally:
        wall, sim = time.monotonic()-start, max(0., session['stamp']-stamp)
        session['stack'].pop()
        if session['stack']:
            session['stack'][-1][0] += wall
            session['stack'][-1][1] += sim
        row = session['rows'].setdefault(name, dict(count=0, simS=0., wallS=0.,
            exclusiveSimS=0., exclusiveWallS=0.))
        row['count'] += 1
        for key, value in dict(simS=sim, wallS=wall, exclusiveSimS=max(0., sim-frame[1]),
                               exclusiveWallS=max(0., wall-frame[0])).items():
            row[key] += value


def timed(name):
    def decorate(fn):
        @wraps(fn)
        def call(*args, **kwargs):
            with phase(name):
                return fn(*args, **kwargs)
        return call
    return decorate


def state_reader(fn):
    @wraps(fn)
    def read(*args, **kwargs):
        with phase('sensor_read'):
            state = fn(*args, **kwargs)
            session = _active.get()
            if session is not None and state.get('stamp') is not None:
                session['stamp'] = max(session['stamp'], float(state['stamp']))
            return state
    return read


def command_writer(fn):
    @wraps(fn)
    def send(command):
        if command.get('gripper') != 'open':
            return fn(command)
        with phase('release_command'):
            return fn(command)
    return send


@contextmanager
def job_timing(job):
    if not enabled():
        yield
        return
    rows = {}
    job.phaseTiming = dict(source='existing_sensor_stamps_and_monotonic_clock',
        accounting='inclusive_rows_do_not_sum; exclusive_rows_partition_job', phases=rows)
    token = _active.set(dict(stamp=job.started, rows=rows, stack=[]))
    try:
        with phase('job'):
            yield
    finally:
        _active.reset(token)


def snapshot(job):
    """Columnar public receipt keeps every phase inside the existing size budget."""
    columns = ['simS', 'wallS']
    return dict(columns=columns, phases={name:[(round(row[k], 3) or 0) for k in columns]
                                        for name,row in list(job.phaseTiming['phases'].items())})
