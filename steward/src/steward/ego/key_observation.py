"""Actionable checkpoint telemetry; contains only robot evidence."""
from copy import deepcopy
import uuid


def begin(owner, phase, *, kind=None):
    if kind == 'release':
        existing = next((r for r in owner.get('keyObservations', [])
                         if r.get('releaseKey') == [owner.get('body'), owner.get('hole'),
                             owner.get('screw'), owner.get('seatedAt')]), None)
        if existing is not None:
            existing.pop('failure',None)
            return existing
    row = dict(schema='key_observation.v1', id=uuid.uuid4().hex, phase=phase,
               kind=kind or owner.get('kind'), triggered=True, measured=False,
               action=None, reason=None)
    if kind == 'release':
        row['releaseKey'] = [owner.get('body'), owner.get('hole'), owner.get('screw'), owner.get('seatedAt')]
    owner.setdefault('keyObservations', []).append(row)
    publish(row)
    return row


def publish(row):
    from steward.ego.capability_usage import record
    row['revision'] = row.get('revision', 0)+1
    record('key_observation', event_id=row['id'], stage=row['phase'],
           measured=row['measured'], action=bool(row['action']),
           reason=row.get('reason'), evidence=row)


def finish(row, *, measured=None, action=None, reason=None):
    if row is None:
        return
    if measured is not None:
        row['measured'] = bool(measured)
    if action:
        row['action'] = action
    if reason is not None:
        row['reason'] = reason
    publish(row)


def seating_result(row, attempt, check):
    passed = check.get('passed')
    status = 'seated' if passed is True else 'not_seated' if passed is False else 'unclear'
    history = attempt.get('seatHistory', [])
    retried = bool(history)
    replaced = any(event.get('seatedAt') != attempt.get('seatedAt') for event in history)
    initial = history[0].get('check', {}) if history else check
    row.update(result=status, check=deepcopy(check),
               holeObservationId=(attempt.get('holeSeatObservation') or {}).get('id'),
               initialResult='not_seated' if initial.get('passed') is False else
                   'seated' if initial.get('passed') is True else 'unclear',
               recovery='replaced_same_object' if replaced else 'same_object_replay_failed' if retried else
                   (attempt.get('landingRecovery') or {}).get('status', 'deferred_reserved_hole' if attempt.get('seatDeferred') else 'reported_and_stopped') if passed is not True else 'none',
               recoveryFailure=attempt.get('seatReplayFailure'),
               recoveryResult=status)
    row['releaseResults'] = []
    for event in [*history, dict(screw=attempt.get('screw'), seatedAt=attempt.get('seatedAt'), check=check)]:
        observed = event.get('check', {})
        value = observed.get('passed')
        row['releaseResults'].append(dict(key=[attempt.get('body'),attempt.get('hole'),event.get('screw'),event.get('seatedAt')],
            result='seated' if value is True else 'not_seated' if value is False else 'unclear',
            state=observed.get('landingState', observed.get('reason')),
            reason=observed.get('reason', 'unspecified'),
            handling=row['recovery'], handlingResult=status))
    # Report geometry in the original check (axis/height/lateral mismatch), not
    # an inferred physical label when the visible evidence cannot distinguish it.
    from steward.ego import axial_release
    finish(row, measured=passed is not None,
           action=('continue' if passed is True else 'continue_unconfirmed') if axial_release.enabled() else
           'defer_hole_continue' if passed is None and attempt.get('seatDeferred') else 'hold_interface' if passed is not True else 'stop_after_failure' if row.get('failure') else 'continue',
           reason=check.get('reason'))


def observed_seat(function):
    """Persist an outcome even if observation, re-pick, or replacement raises."""
    from functools import wraps
    @wraps(function)
    def wrapped(controller, job, attempt, *args, **kwargs):
        from steward.ego import axial_release
        if axial_release.handled(attempt):
            return function(controller, job, attempt, *args, **kwargs)
        row = begin(attempt, 'after_action', kind='release')
        try:
            return function(controller, job, attempt, *args, **kwargs)
        except BaseException as exc:
            row['failure'] = str(exc) or type(exc).__name__
            raise
        finally:
            seating_result(row, attempt, attempt.get('seatCheck') or
                           dict(passed=None, status='unknown', reason='observation_interrupted'))
            job.postcondition.setdefault('keyObservations', []).append(deepcopy(row))
    return wrapped
