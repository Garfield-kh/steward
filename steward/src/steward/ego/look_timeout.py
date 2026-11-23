"""A wall budget scoped to one exploration worker, including robot-port RPCs."""
from steward.ego import capabilities
from contextlib import contextmanager
from contextvars import ContextVar
import math
import time
from steward.ego import action_budget as ab

ENABLED = capabilities.get('STEWARD_LOOK_TIMEOUT') != '0'
_deadline = ContextVar('look_deadline', default=None)


class LookExpired(TimeoutError):
    pass


def active():
    return _deadline.get() is not None


def remaining(default):
    from steward.ego.fasten_sequence import remaining as drive_remaining
    default = drive_remaining(default)
    deadline = _deadline.get()
    if deadline is None:
        return default
    left = ab.left(deadline)
    if left <= 0:
        raise LookExpired('exploration wall budget exhausted')
    return min(default, left)


@contextmanager
def budget(deadline):
    token = _deadline.set(deadline)
    try:
        yield
    finally:
        _deadline.reset(token)


def capture(ex, state):
    ex._look_last_state = state
    for eid, row in state.get('entities', {}).items():
        ex._look_observations[eid] = dict(id=eid, **{k:row[k] for k in
            ('p','category','lastSeenAt') if k in row})


def run(ex, job, receipt):
    """Timed exploration is rotation only; no after-scan arm homing/inspection."""
    from steward.ego.skills import SkillStopped
    plan = receipt['plan']; deadline = plan['_lookDeadline']
    wall_start = time.monotonic()
    if ab.enabled():
        speed = ex.config.look_angular_speed * (1. if ex.config.inspect_closed_drawers else .5)
        seconds = min(plan['timeoutS'], 3*2*math.pi/speed+4*(5.+ex.config.look_dwell_s))
        deadline = ab.make(ex, seconds, 'four quarter-turns: 3*angle/speed + 5s convergence per turn + dwell')
    # Reserve part of the caller's budget for holding the current base pose and
    # releasing the clock. This is inside timeoutS, not an extension of it.
    reserve = min(1., plan['timeoutS']*.1)
    ex._look = True; ex._gentle = False; ex._arm_engaged = False
    ex._look_observations = {}
    capture(ex, plan['_lookInitialState'])
    ex._stamp_watchdog = (None, time.monotonic())
    ex._job_deadline = ex._primitive_deadline = math.inf
    state, reason, message = 'succeeded', 'complete', 'look_around completed'
    try:
        with budget(ab.reserve(deadline,reserve)):
            if ex._clock is not None:
                ex._hold = f'skill/{job.id}'
                ex._clock.arm(ex._hold)
                ex._now = ex._clock.wait_until(None, ex._hold)
            if ex._stop.is_set():
                raise SkillStopped()
            ex._begin(job, 'look_around', receipt)
            ex._look_around(job)
    except LookExpired:
        reason, message = 'timeout', ('look_around simulation budget or deadlock guard reached; returning partial observations'
                                    if ab.enabled() else 'look_around wall budget reached; returning partial observations')
    except SkillStopped:
        state, reason, message = 'stopped', 'cancelled', 'stopped on request'
    except Exception as exc:
        state, reason, message = 'failed', 'motion_failed', str(exc)
    finally:
        # Never poll fresh sensors or home the arms after the deadline. The last
        # encoder state is sufficient for a best-effort stop setpoint.
        with budget(deadline):
            try:
                o = ex._look_last_state['odometry']
                ex._command({'base':[o['x'],o['y'],o['yaw']]})
            except Exception:
                pass
            if ex._clock is not None:
                ex._release_clock(f'skill/{job.id}')
        ex._look = False
        completion = (job.postcondition or {}).get('lookCompletion')
        if reason == 'complete' and completion and completion.get('stopReason'):
            reason = completion['stopReason']
        job.finished = ex._look_last_state['stamp']
        job.postcondition = dict(timeoutS=plan['timeoutS'], timedOut=reason=='timeout',
            coverageComplete=reason in ('complete', 'angular_coverage'), stopReason=reason,
            observations=list(ex._look_observations.values()), observationSource='ego_memory',
            wallS=time.monotonic()-wall_start if ab.enabled() else time.monotonic()-(deadline-plan['timeoutS']))
        if ab.enabled():
            job.postcondition['actionBudget'] = ab.report(deadline)
        if completion:
            job.postcondition['lookCompletion'] = completion
        job.state, job.message = state, message
        if ex._on_finish is not None:
            ex._on_finish(job)
