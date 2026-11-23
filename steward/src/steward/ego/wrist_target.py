"""Transactional wrist target proposals; all distances come from robot estimates."""
from steward.ego import capabilities
from copy import deepcopy
import numpy as np


def enabled():
    return capabilities.get('STEWARD_ASSEMBLY_WRIST_TARGET') != '0'


def adoption_enabled():
    return capabilities.get('STEWARD_ASSEMBLY_WRIST_TARGET_ADOPTION') != '0'


def entry_check(es, params, target, registered):
    """Validate wrist time against live self sensing, without refreshing object beliefs."""
    now = es.stamp
    robot = es.find(es.self_id)
    live = robot.pose.t if robot is not None else None
    wrist = registered.get('measurementSource') in ('wrist_rgbd_empty_recess', 'preplace_layout_rgbd')
    if adoption_enabled() and wrist and live is not None and np.isfinite(live):
        now = max(now, live)
    observed = registered.get('observedAt', -float('inf'))
    cached = target.get('componentFrame', {}).get('registeredSeatTargets', {}).get(
        params.get('hole'), {}).get('observedAt')
    checks = dict(operation=params.get('operation') == 'seat',
        entity=registered.get('entity') == params.get('part'),
        body=registered.get('body') == params.get('body'),
        hole=registered.get('hole') == params.get('hole'),
        observation_age_or_persistent_stamp=bool(0 <= now-observed <= 90 or
            registered.get('persistent') and cached == observed))
    return dict(mismatches=[k for k,v in checks.items() if not v],
        sceneStamp=es.stamp, selfStamp=live, admissionStamp=now,
        observedAt=observed, observationAgeS=now-observed, persistentStamp=cached)


def local_range_enabled():
    return capabilities.get('STEWARD_ASSEMBLY_WRIST_TARGET_RANGE') != '0'


def audit(row):
    if row.get('_usageId'):
        from steward.ego.capability_usage import record
        record('wrist_hole', event_id=row['_usageId'], measured=True,
               reason=None if row.get('targetReplaced') else row.get('reason'))
    if not adoption_enabled():
        return
    event = {k:deepcopy(row[k]) for k in ('samples', 'observedAt', 'measuredHole',
        'targetDecision', 'targetReplaced', 'reason', 'targetPointM', 'originalTargetPointM',
        'targetDeltaM', 'targetDeltaLimitM', 'endpointUnavailable', 'receiptId') if k in row}
    history = row.setdefault('targetEvents', [])
    if history and history[-1].get('samples') == row.get('samples'):
        history[-1] = event
    else:
        history.append(event)


def propose(reg, selected, stamp, row):
    from steward.ego.fasten_target import spacing
    distance = spacing(reg.get('layout', reg.get('holes', [])))
    uncertainty = float(reg.get('uncertaintyM', 0.)) + float(selected.get('uncertaintyM', 0.))
    delta = float(np.linalg.norm(np.asarray(selected['point'])-reg.get('executionPoint', reg['point'])))
    row.update(targetDecision='kept_original', targetReplaced=False,
               targetDeltaM=delta, targetDeltaLimitM=None,
               targetDeltaBasis='layout_consistency_no_motion_cap',
               targetPointM=deepcopy(selected['point']), observedAt=stamp,
               originalTargetPointM=deepcopy(reg.get('executionPoint', reg['point'])))
    if adoption_enabled():
        for key in ('endpointUnavailable', 'receiptId', 'replanInput', 'endpoint'):
            row.pop(key, None)
        layout = reg.get('layout', reg.get('holes', []))
        row.update(wrist_target_adoption=True, measuredHole=selected['bodyHole'],
            correspondence=dict(requestedHole=reg['hole'],
                layout=[dict(hole=h.get('bodyHole'), point=h['point']) for h in layout],
                selectedLayoutPoint=next((h['point'] for h in layout
                    if h.get('bodyHole') == selected['bodyHole']), None),
                registeredPoint=deepcopy(reg['point']), holeSpacingM=distance))
    row.update(holeSpacingM=distance, targetUncertaintyM=uncertainty)
    if selected['bodyHole'] != reg['hole']:
        row['reason'] = 'different_hole_requires_fresh_admission'
    elif not np.isfinite(delta):
        row['reason'] = 'nonfinite_observed_target'
    else:
        candidate = deepcopy(reg)
        candidate.update(point=list(selected['point']), executionPoint=list(selected['point']),
                         observedAt=stamp, uncertaintyM=selected['uncertaintyM'],
                         measurementSource='wrist_rgbd_empty_recess', measuredHole=deepcopy(selected))
        row['reason'] = 'awaiting_endpoint_admission'
        audit(row)
        return candidate
    audit(row)
    return None


def allow_return(controller, job):
    """A failed candidate alone is never a reason to carry a load away."""
    if not enabled():
        return True
    ctx = getattr(controller, '_registered_alignment', {}) or {}
    report = ctx.get('report') or {}
    wrist = report.get('wristAlignment') or {}
    if (not wrist.get('samples') or report.get('releaseExecuted')
            or report.get('object') != controller.rt.estimator.held_id):
        return True
    from steward.ego.registered_alignment import residual
    try:
        measured = residual(controller, report)
    except (KeyError, AttributeError):
        measured = None
    from steward.ego.seat_safety import limits
    tolerance = limits(controller, report)['lateralLimitM']
    from steward.ego.wrist_fasten import MAX_SAMPLES
    exhausted = report.get('kinematicAlignmentExecutions', 0) >= 3 or wrist.get('samples', 0) >= MAX_SAMPLES
    misaligned = measured is not None and (measured['residualM'] > tolerance or measured['axisErrorDeg'] > 8.)
    allowed = bool(misaligned and exhausted)
    job.postcondition['wristReturn'] = dict(allowed=allowed, estimate=measured,
        correctionsExhausted=exhausted, reason='misaligned_corrections_exhausted' if allowed else
        'retain_at_target_until_descent_or_correction_can_complete')
    return allowed
