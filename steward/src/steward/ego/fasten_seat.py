"""Seat continuation receipts and posterior recovery (robot evidence).

Ordinary placement remains the missing-observation fallback. The close-view
precision loop requests the checked endpoint path after measuring the held tip.
"""
from steward.ego.phase_timing import timed
from contextlib import contextmanager
import math
import uuid
import numpy as np
from steward.ego.esim import PredictionError


def endpoint_event(row, stage):
    from steward.ego.capability_usage import record
    record('seat_endpoint', entity=row['object'], stamp=row.get('releasedAt', row['arrivedAt']),
           stage=stage, event_id=row['id'], action=row['releaseExecuted'],
           reason=row.get('failure'), evidence=row)


@contextmanager
def endpoint_attempt(ex, job, outer):
    """Count completed descents and actual opens, including interrupted exits."""
    row = dict(schema='seat_endpoint.v1', id=uuid.uuid4().hex,
        object=outer['object'], body=outer.get('seatBody'), hole=outer.get('seatHole'),
        arrivedAt=ex._state()['stamp'], policy='bounded_wait_then_release',
        releaseExecuted=False, endpointBlocked=False)
    outer['seatEndpointOutcome'] = row
    if getattr(job, 'postcondition', None) is not None:
        job.postcondition.setdefault('seatEndpointAttempts', []).append(row)
    endpoint_event(row, 'arrived')
    try:
        yield row
    except BaseException as exc:
        row['failure'] = str(exc) or type(exc).__name__
        row['endpointBlocked'] = not outer.get('releaseExecuted', False) and str(exc).startswith(
            ('seat endpoint XY did not settle', 'FK endpoint outside release tolerance'))
        raise
    finally:
        row['releaseExecuted'] = bool(outer.get('releaseExecuted'))
        if row['releaseExecuted']:
            row['releasedAt'] = outer['releasedAt']
        endpoint_event(row, 'finished')


def refine_plan(planner, scene, params, plan):
    """Keep ordinary place admission, then prefer a checked upright endpoint path."""
    from steward.ego.fasten_plan import contact
    try:
        receipt = contact(planner, scene, dict(part=params['on'], body=params['seatBody'],
            hole=params['seatHole'], screw=plan['object'], operation='seat', refineSeatSurface=True, precisionInsert=bool(params.get('precisionInsert')), registeredEntry=params.get('registeredEntry')))
    except PredictionError as exc:
        # Still execute the admitted low release, then inspect the actual landing.
        plan['seatMotion'] = dict(mode='low_release', reason=str(exc), contactObserved=False)
        if hasattr(exc, 'wrist_admission'):
            plan['wristAdmission'] = exc.wrist_admission
        return
    plan['seatEndpoint'] = receipt['plan']
    plan['spot'] = list(receipt['plan']['targetEvidence']['point'][:2])
    plan['seatTarget'] = receipt['plan']['targetEvidence']
    plan['seatMotion'] = dict(mode='axial_endpoint', goal=receipt['plan']['goal'],
        targetSource=plan['seatTarget'].get('targetSource'), contactObserved=False,
        pressureControl='position_only_no_force_sensor')


def lift_to(ex, target, *, ramp=True, tolerance=.001, floor=None):
    """Reach ``target`` at the requested tolerance, compensating a settled load offset.

    A loaded lift can settle a few millimetres from its setpoint. After a timed-out move and two
    stationary readings, the setpoint is shifted by the measured residual (bounded) and the result
    is checked against the TARGET, not the shifted setpoint. A moving joint or a larger residual
    is not compensated. Defaults preserve the ordinary executor; a release-plane caller can
    request finer arrival precision and bound every setpoint by its estimated surface floor.
    """
    from steward.ego.skills import SkillFailed
    # An offset compensated once is remembered: under the same load the next setpoint misses by
    # the same amount, and waiting out the full joint timeout before every 2 mm step makes one
    # lowering last longer than a registration stays valid. The remembered offset is only a first
    # command; the result is still verified against the TARGET within 1 mm, else the full path runs.
    bounded = lambda q: max(q, floor) if floor is not None else q
    target = bounded(target)
    known = getattr(ex, '_lift_settled_offset', 0.)
    if known:
        try:
            ex._set_joint('lift', bounded(target+known), ramp=ramp, tol=.013)
            ex._wait(max(.2, ex.config.settle_s))
            q = ex._state()['joints']['lift']
            ex._wait(.2)
            now = ex._state()['joints']['lift']
            if abs(now-target) < tolerance and abs(now-q) < min(.0005, tolerance):
                return
        except SkillFailed:
            pass
        ex._lift_settled_offset = 0.
    from steward.ego.fasten_delivery import ENABLED
    local = ENABLED and abs(ex._state()['joints']['lift']-target) <= .02
    try:
        ex._set_joint('lift', target, ramp=False if local else ramp, tol=tolerance,
                      **({'timeout':2.} if local else {}))
        return
    except SkillFailed as exc:
        if not str(exc).startswith('joint lift did not reach '):
            raise
        failure = exc
    sent = target
    for _ in range(2):
        q = ex._state()['joints']['lift']
        ex._wait(.2)
        now = ex._state()['joints']['lift']
        error = target-now
        if abs(error) < tolerance:
            return
        command = bounded(sent+error)
        if abs(now-q) > .0005 or abs(error) > .012 or abs(command-target) > .012:
            break
        try:
            ex._set_joint('lift', command, ramp=False, tol=.013)
        except SkillFailed:
            break
        ex._wait(max(.2, ex.config.settle_s))
        sent = command
        q = ex._state()['joints']['lift']
        ex._wait(.2)
        now = ex._state()['joints']['lift']
        if abs(now-target) < tolerance and abs(now-q) < min(.0005, tolerance):
            ex._lift_settled_offset = command-target
            return
    raise failure


@timed("descent")
def lower(ex, plan):
    """Reach clearance, observe once, then settle at the updated absolute endpoint."""
    from steward.ego.skills import SkillFailed
    hover = min(plan.get('safeLift', plan['lift']+.012), plan['lift']+.012)
    try:
        lift_to(ex, hover)
    except SkillFailed as exc:
        # The start height of the descent is a clearance. A lift that settles short of it but
        # still above the endpoint starts the descent from where it is.
        reached = ex._state()['joints'].get('lift')
        if (not str(exc).startswith('joint lift did not reach ') or reached is None
                or not plan['lift']+.003 <= reached <= plan['lift']+.02):
            raise
        hover = reached
    observe=getattr(ex,'_precision_descent',None)
    if observe:
        observe(plan,initialize=True)
        observe(plan,reached=hover)
    # One absolute endpoint through the proven loaded-lift executor. Observations
    # update plan['lift'] before this command; no open-loop microstep chain.
    from steward.ego import axial_release
    if axial_release.enabled():
        axial_release.lower(ex, float(plan['lift']))
    else:
        lift_to(ex, float(plan['lift']))


def endpoint_place(ex, job, outer):
    """Execute the endpoint path within the consumed place receipt; retain release on exit failure."""
    from steward.ego.skills import SkillFailed
    p = outer['seatEndpoint']
    ex._dock(job, p, retract=False)
    ex._set_joint('lift', p['safeLift'])
    ex._set_joint('reach', p['retractReach'])
    for key, name in [('wrist_yaw','wristYaw'), ('wrist_roll','wristRoll'), ('wrist_pitch','wristPitch')]:
        if key in ex._state()['joints']:
            ex._set_joint(key, p[name], tol=.02)
    ex._align_lateral(p['gripSpot'], tol=.001)
    ex._set_joint('reach', ex._reach_to(p['gripSpot']), tol=.001)
    floor = getattr(ex, '_precision_release_floor', None)
    if floor: floor(outer, p)
    precision = getattr(ex, '_precision_hover', None)
    if precision:
        try:
            ex._set_joint('lift', p['lift']+.012, tol=.001)
        except SkillFailed as exc:
            # The hover is a clearance above the endpoint, not the endpoint itself. A lift that
            # settles short of it but still clear of the endpoint may continue; the lowering and
            # the release checks that follow keep their own tolerances.
            reached = ex._state()['joints'].get('lift')
            if (not str(exc).startswith('joint lift did not reach ') or reached is None
                    or not p['lift']+.003 <= reached <= p['lift']+.02):
                raise
            outer.setdefault('seatMotion', {})['hoverLiftShortM'] = round(p['lift']+.012-reached, 4)
        if precision(outer):
            return
    tip_entry = getattr(ex, '_precision_tip_entry', None)
    if tip_entry:
        tip_entry(outer)
    clearance = getattr(ex, '_precision_release_clearance', None)
    clearance = float(clearance()) if clearance else 0.
    # The descent still starts from the hover already reached above the endpoint.
    lower(ex, dict(p, lift=p['lift']+clearance, safeLift=min(p.get('safeLift', p['lift']+.012), p['lift']+.012))
          if clearance > 0 else p)
    if p.get('descentTiming'):
        outer.setdefault('precision', {}).setdefault('seatDelivery', {})['descentTiming'] = p['descentTiming']
    with endpoint_attempt(ex, job, outer) as outcome:
        finish_endpoint_place(ex, job, outer, outcome)


def finish_endpoint_place(ex, job, outer, outcome):
    p = outer['seatEndpoint']
    ex._settle_for_release()
    inserted = getattr(ex, '_precision_insert', None)
    if inserted:
        inserted(outer)
    ex._tick()
    # This is a single FK-to-command residual, not tip/hole distance or motion
    # stability. The ordinary bounded settle already permits a loaded offset;
    # do not turn that offset into a failed seat and a trip back to the supply.
    outcome['gripperResidualM'] = float(np.linalg.norm(ex._gripper_xy()-p['gripSpot']))
    # This is FK positioning, never force/contact or visual confirmation.
    if ex._validate_placement is not None:
        ex._validate_placement(outer)
    before_release = getattr(ex, '_precision_before_release', None)
    if before_release:
        before_release(outer)
    ex._command({'gripper':'open'})
    outer['releasedAt'] = ex._state()['stamp']
    outer['releaseExecuted'] = True
    from steward.ego import release_conditions
    release_conditions.opened(outer)
    outcome.update(releaseExecuted=True, releasedAt=outer['releasedAt'])
    endpoint_event(outcome, 'released')
    from steward.ego.grasp_endpoints import released
    outer['endpointReleaseEvidence'] = released(p, outer['releasedAt'])
    ex._released(outer['object'], outer['spot'])
    ex._wait(.6)
    outer['seatMotion']['settledUntil'] = ex._state()['stamp']
    release_conditions.held_open(ex, outer)
    from steward.ego.fasten_completion import CONFIRM
    after_release = getattr(ex, '_precision_after_release', None)
    if after_release and (CONFIRM or getattr(after_release, 'wrist_capture', False)):
        after_release(outer)
    release_conditions.withdrawing(ex, outer)
    ex._set_joint('lift', p['safeLift'])
    ex._withdraw(p)
    release_conditions.withdrawn(ex, outer)
    for key in ('wrist_roll','wrist_yaw','wrist_pitch'):
        if key in ex._state()['joints']:
            ex._set_joint(key, 0., tol=.02)
    if not outer.get('stayNear'):
        ex._backoff(p)
    ex._report_placement(job, outer['object'])


def summary(part, body, rows):
    from steward.ego.fasten_completion import REPORT, progress
    if REPORT:
        return progress(rows, part, body)['interfaceSummary']
    pending = [r['hole'] for r in rows if not r.get('released')]
    confirmed = sum(r['status'] == 'confirmed' for r in rows)
    prefix = f'接口 {part} → {body}：'
    if not pending:
        return prefix+f'各孔均已放置一颗（确认 {confirmed}/{len(rows)}，其余未能目视确认）；建议进入下一步。'
    return (prefix+f'确认 {confirmed}/{len(rows)}；尚未释放孔 '+', '.join(pending)+
            '；仅续做未释放孔，保留已放上件；已释放孔不再补料。')


def flip_advice(frames, attempts, body):
    attempts = list(attempts)
    holes = ({p['bodyHole'] for f in frames if f['body'] == body for p in f.get('holePairs', [])}
             | {a['hole'] for a in attempts if a['body'] == body and a.get('hole')})
    released = {a['hole'] for a in attempts if a['body'] == body and a.get('releaseExecuted')}
    from steward.ego.fasten_completion import REPORT
    if REPORT:
        confirmed = {a['hole'] for a in attempts if a['body']==body and (a.get('seatConfirmed') or a.get('seatCheck',{}).get('passed') is True)}
        pending = sorted(holes-confirmed)
        return dict(remainingHoles=pending, advisory=True, releasedHoles=sorted(released),
            confirmedSeatedHoles=sorted(confirmed), message='未确认入座孔 '+', '.join(pending) if pending else '各孔已确认入座。')
    pending = sorted(holes-released)
    return dict(remainingHoles=pending, advisory=True,
        message=('尚未释放孔 '+', '.join(pending)+'；可续做这些孔，保留已放上件。') if pending else
                '各孔均已放置一颗；建议进入下一步，已释放但未确认的孔不再补料。')


def grasp_check(scene, held_id, requested):
    """After picking, reject a separately identified different load, not missing pixels."""
    from steward.ego.fasten_supply import measured_fastener
    held = scene.find(held_id)
    if held is None:
        return dict(status='unknown', reason='held_track_unavailable')
    measured = measured_fastener(held, scene.stamp)
    if measured:
        return dict(status='supported', reason='independent_fastener_endpoints', source=measured['source'])
    # held_fastener_kind: a held load identified as something other than a screw is never seated, even if it is the requested track
    if held.category.value != 'toy_screw':
        return dict(status='contradicted', reason='different_identified_load', actual=held_id,
                    category=held.category.value)
    return dict(status='unknown', reason='no_independent_held_shape')


def surface_target(scene, part, target):
    """Refine only local entry Z from a fresh separate patch, preserving hole correspondence."""
    from copy import deepcopy
    from steward.ego.fasten_plan import value
    from steward.ego.fasten_geometry import surface_patch
    view = value(scene.find(part), 'contactView')
    if (not view or view.get('source') != 'rgbd_segment' or view.get('truncated', True)
            or view.get('invalidDepthFraction', 1) > .05
            or not 0 <= scene.stamp-view.get('stamp', -1) <= 20
            or view['stamp'] <= target.get('releasedAt', -1)):
        return target
    patch = surface_patch(view.get('points', []), target['point'][:2])
    if patch is None or abs(patch['point'][2]-target['point'][2]) > .012:
        return target
    result = deepcopy(target)
    result.update(point=patch['point'], surfaceEvidence=dict(patch, stamp=view['stamp'],
        source='rgbd_local_entry_surface'), targetSource='interface_xy_plus_local_surface_z')
    return result


def replay_release(controller, attempt, check, params):
    """Try the existing close release once, only after an observed lower-hole miss.

    Unknown, displaced components, upper interfaces and powered driving retain
    ordinary placement. An inadmissible refinement falls back before any motion.
    No endpoint servo, new contact check or extra replay budget is introduced.
    """
    if not (getattr(controller, '_seat_only', False)
            and getattr(controller, '_lower_interface', False)
            and check.get('status') == 'failed' and check.get('passed') is False
            and check.get('reason') == 'independent_screw_geometry'
            and check.get('observedAt', -1) is not None
            and attempt.get('seatedAt', float('inf')) < check['observedAt'] <= controller.rt.ego_state().stamp
            and controller.rt.ego_state().stamp-check['observedAt'] <= 20
            and not attempt.get('closeReplayTried') and not params.get('closeRelease')):
        return params, None
    controller.check()
    attempt['closeReplayTried'] = True
    refined = dict(params, closeRelease=True)
    receipt = controller.rt.predict('place', refined)
    attempt['replayRelease'] = dict(mode='ordinary_close_release' if receipt.get('withinRiskLimit')
        else 'ordinary_release_fallback', reason='observed_miss_after_release',
        observedAt=check['observedAt'], issues=list(receipt.get('issues', [])))
    if receipt.get('withinRiskLimit'):
        return refined, receipt
    return params, None
