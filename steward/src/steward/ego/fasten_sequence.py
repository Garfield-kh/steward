"""Place all fasteners before a wall-bounded powered phase."""
from steward.ego import capabilities
from contextlib import contextmanager
from contextvars import ContextVar
import math
import time
from steward.ego import action_budget as ab

ENABLED = capabilities.get('STEWARD_ASSEMBLY_DRIVE_SEQUENCE') != '0'
PER_HOLE = capabilities.get('STEWARD_ASSEMBLY_PER_HOLE_DRIVE_GATE') != '0'
BUDGET_S = float(capabilities.get('STEWARD_ASSEMBLY_DRIVE_BUDGET_S'))
_deadline = ContextVar('fasten_drive_deadline', default=None)
_parents = ContextVar('fasten_parent_deadlines', default=())


class DriveExpired(TimeoutError):
    pass


def active():
    return _deadline.get() is not None


def remaining(default):
    deadline = _deadline.get()
    if deadline is None:
        return default
    left = min(ab.left(d) for d in (*_parents.get(), deadline))
    if left <= 0:
        raise DriveExpired('powered phase simulation budget or deadlock guard exhausted' if isinstance(deadline, (ab.Deadline, ab.Limits)) else 'powered phase wall budget exhausted')
    return min(default, left)


@contextmanager
def budget(deadline):
    previous = _deadline.get()
    parents = _parents.set((*_parents.get(), previous) if ab.enabled() and previous is not None else ())
    token = _deadline.set(deadline)
    try:
        remaining(math.inf)
        yield
    finally:
        _deadline.reset(token)
        _parents.reset(parents)


def bounded(controller, job, operation, support=None, *, budget_s=None, hole_count=1):
    """One phase budget, including acquisition, stroke, observation and parking."""
    if not ENABLED or active():
        return operation()
    seconds = BUDGET_S if budget_s is None else budget_s
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('STEWARD_ASSEMBLY_DRIVE_BUDGET_S must be finite and positive')
    basis = None
    if ab.enabled():
        seconds, basis = ab.drive_estimate(controller, hole_count)
    start = time.monotonic(); deadline = ab.make(controller.ex, seconds, basis)
    reserve = min(30., seconds/6)
    report = dict(budgetS=seconds, cleanupReserveS=reserve, timedOut=False, wallS=0.)
    job.postcondition['driveBudget'] = report
    failure = None
    work_deadline=ab.reserve(deadline,reserve)
    try:
        with budget(work_deadline):
            operation()
    except DriveExpired as exc:
        failure = exc
        report.update(timedOut=True, reason='powered_phase_sim_timeout_or_deadlock' if ab.enabled() else 'powered_phase_wall_timeout',
                      stopStage=job.postcondition.get('driveStage') or
                      job.postcondition.get('driverMotion', {}).get('stage'))
        # Bounded cleanup is part of the same budget. Do not park sideways out
        # of contact if the stroke failed to establish withdrawal clearance.
        if controller.rt.estimator.held_id and job.postcondition.get('driverClear') is not False:
            try:
                with budget(deadline):
                    if support is None:
                        from steward.ego.fasten_plan import support_anchor
                        scene = controller.rt.ego_state()
                        support = support_anchor(controller.rt.esim, scene,
                            scene.entity(controller.body_id), []).id
                    controller.park_driver(job, support)
            except Exception as cleanup:
                from steward.ego.skills import SkillStopped
                if isinstance(cleanup, SkillStopped):
                    raise
                report['cleanupFailure'] = str(cleanup)
    finally:
        report.update(wallS=time.monotonic()-start, held=controller.rt.estimator.held_id)
        report.update(ab.report(deadline))
        if failure is not None and ab.enabled():
            report['stopReason']=ab.report(work_deadline).get('stopReason') or 'parent_or_child_budget'
        job.postcondition['driveBudget'] = report
    if failure:
        from steward.ego.skills import SkillFailed
        raise SkillFailed(str(failure))


def drive_all(controller, job, plan, holes, failures):
    """Keep misses/unknowns explicit, but never interleave a drive with supply."""
    from steward.ego.skills import SkillFailed
    from steward.ego.interface_sequence import ready
    if not ready(controller, job, plan, holes, failures):
        return
    if PER_HOLE:
        return drive_each(controller, job, plan, holes, failures)
    def operation():
        for hole in holes:
            remaining(math.inf)
            a = controller.rt.estimator.fastening_attempts.get((plan['body'], hole['id']))
            if (not a or not a.get('releaseExecuted') or a.get('driveStatus') == 'driven'
                    or not plan.get('driveExplicit') and a.get('verified')):
                continue
            if a.get('seatCheck', {}).get('passed') is False:
                failures.append(dict(hole=hole['id'], skill='drive_screw', failure='known_not_seated'))
                continue
            try:
                controller.step(job, 'drive_screw', dict(screw=a['screw'],body=plan['body'],
                    hole=hole['id'], method=plan.get('method','simple'),
                    correctOnce=bool(plan.get('correctOnce',True))))
            except SkillFailed as exc:
                controller.check()
                failures.append(dict(hole=hole['id'], skill='drive_screw', failure=str(exc)))
                if controller.rt.estimator.held_id:
                    break
    try:
        bounded(controller, job, operation, plan['support'])
    except SkillFailed as exc:
        controller.check()
        failures.append(dict(hole=None, skill='drive_screw', failure=str(exc)))


def eligibility(attempt):
    if not attempt or not attempt.get('releaseExecuted'):
        return 'unattempted'
    if attempt.get('seatCheck', {}).get('passed') is False or attempt.get('seatStatus') == 'failed':
        return 'known_not_seated'
    return None


def require_attempt(controller, job, plan, attempt=None):
    """Revalidate exact body/hole/identity, including after acquiring the tool."""
    if not PER_HOLE:
        return
    if attempt is None:
        attempt = controller.rt.estimator.fastening_attempts.get((plan.get('body'), plan.get('hole')))
        if attempt is None:
            matches = [a for a in controller.rt.estimator.fastening_attempts.values()
                if a.get('screw') == plan.get('screw') and
                (not plan.get('body') or a['body'] == plan['body']) and
                (not plan.get('hole') or a['hole'] == plan['hole'])]
            attempt = matches[0] if len(matches) == 1 else None
    reason = eligibility(attempt)
    if attempt and plan.get('screw') not in (None, attempt['screw'], *attempt.get('screwAliases', [])):
        reason = 'wrong_fastener_for_hole'
    if reason:
        from steward.ego.skills import SkillFailed
        job.postcondition['driveGate'] = dict(reason=reason, eligibleHoles=[], wallS=0.,
            nextAction=dict(skill='seat_screw', body=plan.get('body'), hole=plan.get('hole'),
                            reason='complete_seating_before_drive'))
        raise SkillFailed('drive skipped: '+reason)


def drive_each(controller, job, plan, holes, failures):
    from copy import deepcopy
    from steward.ego.skills import SkillFailed
    candidates, skipped = [], []
    for hole in holes:
        a = controller.rt.estimator.fastening_attempts.get((plan['body'], hole['id']))
        reason = eligibility(a)
        if reason:
            skipped.append(dict(hole=hole['id'], reason=reason))
        elif a.get('driveStatus') != 'driven' and (plan.get('driveExplicit') or not a.get('verified')):
            candidates.append(a)
    next_action = (dict(skill='seat_screw', body=plan['body'], part=plan.get('part'),
        hole=skipped[0]['hole'], reason=skipped[0]['reason']) if skipped else None)
    gate = dict(eligibleHoles=[a['hole'] for a in candidates], skippedHoles=skipped,
        reason='eligible_releases' if candidates else 'no_eligible_released_holes',
        nextAction=next_action, wallS=0.)
    job.postcondition['driveGate'] = gate
    report = dict(budgetS=BUDGET_S, perHole=[], timedOut=False, wallS=0.)
    job.postcondition['driveBudget'] = report
    if not candidates:
        return
    from steward.ego import driver_batch
    if driver_batch.enabled() and plan.get('method','simple') == 'simple':
        return driver_batch.run(controller,job,plan,candidates,failures)
    share = BUDGET_S/len(candidates)
    if ab.enabled():
        share, basis = ab.drive_estimate(controller)
        report.update(clock='simulation', budgetS=share*len(candidates),
                      budgetSimS=share*len(candidates), basis=basis)
    report['perHoleBudgetS'] = share
    start = time.monotonic()
    for a in candidates:
        try:
            require_attempt(controller, job, dict(plan, screw=a['screw'], hole=a['hole']), a)
            bounded(controller, job, lambda:controller.step(job, 'drive_screw',
                dict(screw=a['screw'], body=plan['body'], hole=a['hole'],
                    method=plan.get('method','simple'), correctOnce=bool(plan.get('correctOnce',True)))),
                plan['support'], budget_s=share)
        except SkillFailed as exc:
            failures.append(dict(hole=a['hole'], skill='drive_screw', failure=str(exc)))
        finally:
            per_hole = job.postcondition.get('driveBudget', {})
            if per_hole is not report:
                child_report=deepcopy(per_hole)
                if ab.enabled(): child_report.pop('basis',None)
                report['perHole'].append(dict(child_report, hole=a['hole']))
                report['timedOut'] |= per_hole.get('timedOut', False)
                if ab.enabled():
                    report['simS'] = sum(r.get('simS',0.) for r in report['perHole'])
            report.update(wallS=time.monotonic()-start, held=controller.rt.estimator.held_id)
            gate['wallS'] = report['wallS']
            job.postcondition.update(driveBudget=report, driveGate=gate)
        controller.check()
        if controller.rt.estimator.held_id:
            break
