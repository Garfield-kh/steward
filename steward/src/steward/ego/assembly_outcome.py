"""Current robot observations override release memory; bounded, resumable repair."""
from copy import deepcopy
from steward.ego import capabilities
from steward.ego.skills import SkillFailed


def enabled():
    return capabilities.get('STEWARD_ASSEMBLY_OUTCOME') != '0'


def failure_current(attempt):
    from steward.ego.session_identity import current_failure, enabled
    return not enabled() or current_failure(attempt.get('seatCheck') or {}, attempt.get('identityEvidenceAfter', -float('inf')))


def empty(attempt):
    check = attempt.get('seatCheck') or {}
    return failure_current(attempt) and check.get('passed') is False and check.get('landingState') == 'empty_hole'


def refill_ready(attempt):
    authorization = attempt.get('refillAuthorized') or {}
    check = attempt.get('seatCheck') or {}
    observed = check.get('observedAt') or -1
    return bool(enabled() and empty(attempt) and authorization.get('observedAt') == observed
        and observed > max(attempt.get('seatedAt', -1), attempt.get('transportedAt', -1)))


def check_is_history(attempt):
    from steward.ego.session_identity import enabled, current_failure
    check = attempt.get('seatCheck') or {}
    return enabled() and (check.get('restoredPrior') or 'identityEvidenceAfter' in attempt) and not current_failure(
        check, attempt.get('identityEvidenceAfter', -float('inf')))


def state(attempt):
    from steward.ego.grasp_binding import enabled as binding_enabled
    if binding_enabled() and attempt.get('bindingUnconfirmed') and not attempt.get('transportCommitted'):
        return 'observed_empty' if empty(attempt) else 'executed_unverified'
    from steward.ego.hole_seating import confirmed
    if empty(attempt): return 'observed_empty'
    if confirmed(attempt) and not check_is_history(attempt):
        check = attempt.get('seatCheck') or {}
        if ((check.get('observedAt') or -1) > attempt.get('transportedAt', -1)
                and check.get('currentPoseConfirmed') is not False):
            return 'confirmed_occupied'
    check = attempt.get('seatCheck') or {}
    if check.get('passed') is False and failure_current(attempt): return 'observed_not_seated'
    return 'executed_unverified' if attempt.get('releaseExecuted') or attempt.get('releaseHistory') else 'unattempted'


def apply(session, item, report, phase):
    """Never turn absence of a detection into evidence of an empty hole."""
    for hole, attempt in zip(item['holes'], session.rows(item)):
        if not attempt: continue
        from steward.ego import axial_release
        if axial_release.checked(attempt): continue
        from steward.ego import released_object
        success = released_object.current_success(attempt)
        if success and phase != 'after_release': continue
        observation = report.get('holes', {}).get(hole, {})
        check = observation.get('check') or {}
        if released_object.enabled() and (attempt.get('holeSeatObservation') or {}).get('context'):
            # A mouth/layout match cannot overwrite same-object evidence with
            # a different protrusion. Reuse the acquired frame, never add a view.
            refreshed = None
            if not success and phase != 'after_release':
                try:
                    refreshed = released_object.refresh(session.controller,attempt,report)
                except (AttributeError,KeyError,TypeError,ValueError):
                    pass
            observation = (dict(occupied=refreshed['passed'],check=refreshed,source=released_object.SOURCE)
                           if refreshed is not None else {})
            check = observation.get('check') or {}
        after = max(attempt.get('seatedAt', -1), attempt.get('transportedAt', -1))
        if (check.get('observedAt') or -1) > after and observation.get('occupied') is not None:
            old = attempt.get('seatCheck') or {}
            if check['observedAt'] >= (old.get('observedAt') or -1):
                attempt.update(seatCheck=deepcopy(check), seatConfirmed=check.get('passed') is True and not (attempt.get('bindingUnconfirmed') and not attempt.get('transportCommitted')),
                               seatStatus=check.get('status', 'unknown'))
                if empty(attempt):
                    initial = session.state.get('initialInterfaces', {}).get(item['key'], {}).get('holes', {})
                    if hole in initial: initial[hole].update(occupied=False, confirmed=False)
        if axial_release.enabled():
            axial_release.record(session, attempt, item['key'])
            continue
        session.emit('outcome_checked', interface=item['key'], hole=hole,
            body=session.canonical(item['body']), screw=attempt.get('screw'),
            phase=phase, outcome=state(attempt), seatedAt=attempt.get('seatedAt'),
            check=deepcopy(attempt.get('seatCheck', {})), observation=deepcopy(observation))
    support = report.get('support') or {}
    from steward.ego.session_identity import enabled as identity_enabled, current_failure
    if (identity_enabled() and session.state.get('memoryResumed') and support.get('matches') is False
            and not current_failure(support, max(session.state.get('memoryRestoredAt', 0), session.state.get('disturbedAt', -1)))):
        support = {}
    if support.get('measured'):
        session.state.setdefault('outcomeComponents', {})[item['key']] = deepcopy(support)
        session.emit('component_outcome', interface=item['key'], phase=phase, check=support)


def support_current(session, item):
    from steward.ego import interface_entry
    try:
        from steward.ego import session_identity
        member = session.state.get('members', {}).get(item['part'], {})
        if session_identity.enabled() and member.get('installed') and not member.get('binding', {}).get('track'):
            return interface_entry.unknown('installed_member_without_independent_track')
        # This detector compares a visible upper bearing face. After a flip,
        # an attached underside is not a displaced upper component.
        if session.access(item)['status'] == 'inaccessible':
            return interface_entry.unknown('selected_face_hidden')
        return interface_entry.support(session.controller, session.canonical(item['part']),
                                       session.canonical(item['body']), -1)
    except (AttributeError, KeyError, TypeError, ValueError):
        return interface_entry.unknown('support_view_unavailable')


def observe(session, item, *, change_view=False):
    from steward.ego import session_initial
    params = dict(body=session.canonical(item['body']), part=session.canonical(item['part']))
    if change_view: params['changeView'] = True
    acquired = session.action('_assembly_observe', params)
    current = dict(item, **params)
    if session.access(item)['status'] == 'inaccessible':
        # The visible opposite face must not be matched to this hidden layout.
        # Keep its best estimate; observing is never a reason to flip the body.
        report = dict(holes={h:dict(occupied=None, reason='selected_face_hidden') for h in item['holes']})
    else:
        report = session_initial.observe(session.controller, current)
    report['support'] = support_current(session, item)
    report['acquisition'] = deepcopy((acquired or {}).get('acquisition', {}))
    return report


def park(session):
    """Return an interrupted recovery load through an ordinary admitted place."""
    if not session.rt.estimator.held_id: return
    source = session.source_support(None)
    try:
        if session.controller.park(session.job, source): return
    except SkillFailed as exc:
        from steward.ego.assembly_session import check_budget
        check_budget()
        session.emit('outcome_return_unconfirmed', reason=str(exc))
    # The old wrist-confidence gate can refuse even a return to the source.
    # A new ordinary placement receipt still owns reach/collision admission.
    if session.rt.estimator.held_id and source:
        session.controller.step(session.job, 'place', dict(on=source, alignHoles=False))


def restore_component(session, item):
    """Replay ordinary pick/place at the independently observed component pose."""
    support = session.state.get('outcomeComponents', {}).get(item['key'], {})
    if support.get('matches') is not False: return False
    from steward.ego.session_identity import current_failure, enabled
    if enabled() and session.state.get('memoryResumed') and not current_failure(support, max(
            session.state.get('memoryRestoredAt', 0), session.state.get('disturbedAt', -1))): return False
    used = session.state.setdefault('outcomeComponentRetries', {})
    if used.get(item['key'], 0) >= 1: return False
    evidence_at = support.get('observedAt') or -1
    consumed = session.state.setdefault('outcomeComponentEvidence', {})
    if evidence_at <= consumed.get(item['key'], -1): return False
    consumed[item['key']] = evidence_at
    used[item['key']] = used.get(item['key'], 0)+1
    part, body = (session.canonical(item[k]) for k in ('part', 'body'))
    session.emit('outcome_component_repair', interface=item['key'], check=support)
    park(session)
    from steward.ego.fasten_plan import value
    scene = session.rt.ego_state()
    view = value(scene.find(part), 'contactView') or {}
    est = session.rt.estimator
    with session.rt._lock:
        track = est.tracks.get(part)
        if track is not None and view.get('stamp') == support.get('observedAt'):
            import numpy as np
            track.pos = np.asarray(view['centreM'])
        record = getattr(getattr(est, 'stationary_assemblies', None), 'records', {}).get(body)
        if record: record['reference'].pop('componentHoleRegistration', None)
        getattr(est, '_seat_registrations', {}).pop(part, None)
        session.controller._registered_alignment = {}
    try:
        session.controller.step(session.job, 'pick', dict(target=part, stayNear=True))
        session.controller.step(session.job, 'place', dict(on=body, alignHoles=True, hole=item['holes'],
            heldHole=item['heldHoles'], stayNear=True, closeRelease=True))
    finally:
        park(session)
    # Execution removes an old contradiction, without claiming observed success.
    session.state['outcomeComponents'][item['key']] = dict(measured=False, matches=None,
        reason='component_replaced_unverified')
    session.state['contradictions'].pop(item['key'], None)
    for attempt in session.rows(item):
        if attempt:
            from steward.ego import first_interface
            if first_interface.applies(session, item) and attempt.get('releaseExecuted'):
                attempt['firstInterfaceNeedsReseat'] = True
            attempt.update(seatConfirmed=False, seatStatus='unknown',
                transportedAt=session.rt.ego_state().stamp,
                seatCheck=dict(status='unknown', passed=None, reason='component_replaced_recheck_hole'))
    from steward.ego import first_interface
    if first_interface.applies(session, item):
        # The ordinary place owns the post-release view; consume that current
        # frame rather than adding an observation for the first-face repair.
        apply(session, item, dict(support=support_current(session, item)), 'after_component_repair')
    else:
        apply(session, item, observe(session, item), 'after_component_repair')
    return True


def stock_stamp(session):
    """Current bound-source loose stock can renew an otherwise unchanged miss."""
    from steward.ego import session_initial
    from steward.ego.fasten_plan import value
    try:
        scene = session.rt.ego_state()
        available = session_initial.inventory(session)['entities']
        return max([(value(scene.find(e), 'contactView') or {}).get('stamp') or
                    scene.find(e).pose.t or -1 for e in available] or [-1])
    except (AttributeError, KeyError, TypeError, ValueError):
        return -1


def repair_empty(session, item):
    """One refill per hole per invocation; a new invocation obtains new evidence."""
    if session.state.get('outcomeComponents', {}).get(item['key'], {}).get('matches') is False:
        return  # Put the upper layer back before supplying any hole.
    used = session.state.setdefault('outcomeRefills', {})
    for hole in item['holes']:
        attempt = next((a for a in session.rows(item) if a.get('hole') == hole), {})
        key = item['key']+'/'+hole
        if not empty(attempt) or used.get(key, 0) >= 1: continue
        evidence_at = (attempt.get('seatCheck') or {}).get('observedAt', -1)
        history = session.state.setdefault('outcomeRefillEvidence', {})
        stock_at = stock_stamp(session)
        stock_history = session.state.setdefault('outcomeRefillStockEvidence', {})
        if evidence_at <= history.get(key, -1) and stock_at <= stock_history.get(key, -1):
            continue  # Repeated text/counters alone do not renew a repair.
        history[key] = evidence_at
        stock_history[key] = stock_at
        from steward.ego.assembly_session import check_budget
        check_budget()
        used[key] = used.get(key, 0)+1
        # Keep releaseExecuted and the entire history as action facts. A separate
        # fresh-empty authorization revokes occupancy, allowing current loose stock.
        attempt['refillAuthorized'] = deepcopy(attempt['seatCheck'])
        attempt.update(seatConfirmed=False, verified=False)
        session.state['retries'].pop(item['key'], None)
        session.emit('outcome_refill', interface=item['key'], hole=hole,
            body=session.canonical(item['body']), screw=attempt.get('screw'), seatedAt=attempt.get('seatedAt'),
            check=deepcopy(attempt['seatCheck']), previousRelease=attempt.get('seatedAt'),
            looseStockObservedAt=stock_at if stock_at >= 0 else None)
        try:
            session.action('seat_screw', dict(body=session.canonical(item['body']),
                part=session.canonical(item['part']), hole=hole, deferVerify=True))
        except SkillFailed as exc:
            check_budget()
            session.emit('outcome_repair_pending', interface=item['key'], hole=hole, reason=str(exc))
        finally:
            park(session)
        # The new release's native hole view has already run. Retain its result
        # before any further source trip.
        latest = next((a for a in session.rows(item) if a.get('hole') == hole), attempt)
        if (latest.get('releaseExecuted') and
                ((latest.get('seatCheck') or {}).get('observedAt') or -1) <= latest.get('seatedAt', -1)):
            try:
                apply(session, item, observe(session, item), 'after_refill')
            except (SkillFailed, AttributeError, KeyError, TypeError, ValueError) as exc:
                check_budget()
                session.emit('outcome_view_unavailable', reason=str(exc))
        session.emit('outcome_refill_result', interface=item['key'], hole=hole,
            outcome=state(latest), released=bool(latest.get('releaseExecuted')),
            seatedAt=latest.get('seatedAt'))


def repair_not_seated(session, item):
    """Keep the existing same-object recovery for a visible bad landing.

    Unlike an empty mouth, a raised/tilted head never authorizes new stock.
    A fresh observation can renew the local replay budget without erasing its
    historical count; the existing identity/departure checks still choose it.
    """
    if session.state.get('outcomeComponents', {}).get(item['key'], {}).get('matches') is False:
        return
    used = session.state.setdefault('outcomeRefills', {})
    consumed = session.state.setdefault('outcomeReseatEvidence', {})
    for attempt in session.rows(item):
        if state(attempt) != 'observed_not_seated': continue
        from steward.ego import released_object
        if (released_object.enabled() and (attempt.get('seatCheck') or {}).get('source') == released_object.SOURCE
                and attempt.get('seatRetryCount', 0) >= 1): continue
        key = item['key']+'/'+attempt['hole']
        stamp = (attempt.get('seatCheck') or {}).get('observedAt') or -1
        if (used.get(key, 0) or stamp <= consumed.get(key, -1) or
                stamp <= max(attempt.get('seatedAt', -1), attempt.get('transportedAt', -1))): continue
        used[key] = 1; consumed[key] = stamp
        old_count = attempt.get('seatRetryCount', 0)
        attempt['seatRetryCount'] = 0
        previous_mode = session.state.get('reviewingReleases', False)
        session.state['reviewingReleases'] = True
        session.controller._seat_retry_budget = [1]
        session.emit('outcome_reseat', interface=item['key'], hole=attempt['hole'],
                     check=deepcopy(attempt.get('seatCheck')))
        try:
            session.action('seat_screw', dict(body=session.canonical(item['body']),
                part=session.canonical(item['part']), hole=attempt['hole']))
        except SkillFailed as exc:
            from steward.ego.assembly_session import check_budget
            check_budget()
            session.emit('outcome_repair_pending', interface=item['key'], hole=attempt['hole'], reason=str(exc))
        finally:
            latest = next((a for a in session.rows(item) if a.get('hole') == attempt['hole']), attempt)
            latest['seatRetryCount'] = old_count+latest.get('seatRetryCount', 0)
            session.state['reviewingReleases'] = previous_mode
            park(session)


def after_interface(session, item, post=None):
    """Consume the already captured post-release result before the next interface."""
    if not enabled() or session.state.get('outcomeRepairing'): return
    from steward.ego import axial_release
    if axial_release.enabled():
        for attempt in session.rows(item):
            if not attempt or axial_release.handled(attempt): continue
            if axial_release.checked(attempt):
                session.controller.confirm_seat(session.job, attempt, dict(method='simple'), observe=False)
                continue
            observation = attempt.get('holeSeatObservation') or {}
            direct = observation.get('check') if observation.get('releasedAt') == attempt.get('seatedAt') else None
            if direct:
                attempt.update(seatCheck=deepcopy(direct), seatConfirmed=direct.get('passed') is True)
            if attempt.get('releaseExecuted'):
                # A place can finish before its parent skill is interrupted.
                # Finish that same release locally; do not defer it to a tour.
                check = attempt.get('seatCheck') or {}
                if check and (check.get('passed') is None or
                              (check.get('observedAt') or -1) > attempt.get('seatedAt', -1)):
                    axial_release.record(session, attempt, item['key'])
                session.controller.confirm_seat(session.job, attempt, dict(method='simple'), observe=False)
        return
    session.state['outcomeRepairing'] = True
    try:
        # Do not replace a definitive native empty-hole check with old memory.
        for a in session.rows(item):
            direct = a.get('holeSeatObservation') or {}
            check = direct.get('check') or {}
            if (check.get('observedAt') or -1) > max(a.get('seatedAt', -1), a.get('transportedAt', -1)):
                if (check.get('observedAt') or -1) >= ((a.get('seatCheck') or {}).get('observedAt') or -1):
                    a.update(seatCheck=deepcopy(check), seatConfirmed=check.get('passed') is True)
        missing = [a for a in session.rows(item) if a.get('releaseExecuted')
            and ((a.get('seatCheck') or {}).get('observedAt') or -1) <= a.get('seatedAt', -1)
            and not any(f.get('capture', {}).get('frameCaptured') and
                        (f.get('stamp') or -1) > a.get('seatedAt', -1)
                        for f in (a.get('holeSeatObservation') or {}).get('frames', []))]
        report = dict(support=support_current(session, item))
        contradicted = any(((post or {}).get(k) or {}).get('passed') is False
                           for k in ('placementCheck', 'placementReadiness', 'supportTargetCheck'))
        if missing or contradicted:
            try:
                report = observe(session, item)
            except (SkillFailed, AttributeError, KeyError, TypeError, ValueError) as exc:
                from steward.ego.assembly_session import check_budget
                check_budget()
                session.emit('outcome_view_unavailable', reason=str(exc))
        apply(session, item, report, 'after_release')
        for a in session.rows(item):
            check = a.get('seatCheck') or {}
            if check.get('reason') == 'component_displaced_after_release':
                session.state.setdefault('outcomeComponents', {})[item['key']] = dict(
                    measured=True, matches=False, observedAt=check.get('componentObservedAt'),
                    source='independent_component_rgbd', reason=check['reason'])
        if session.state.get('outcomeComponents', {}).get(item['key'], {}).get('matches') is False:
            restore_component(session, item)
        repair_empty(session, item)
        from steward.ego import released_object
        if released_object.enabled() and any((a.get('seatCheck') or {}).get('source') == released_object.SOURCE
                                             for a in session.rows(item)):
            repair_not_seated(session, item)
    finally:
        session.state['outcomeRepairing'] = False


def review(session):
    """Fresh bounded review on every invocation; no persistent 'already looked' veto."""
    from steward.ego.assembly_session import check_budget
    from steward.ego import axial_release
    if axial_release.enabled():
        session.state['outcomeReviewedAt'] = session.rt.ego_state().stamp
        return
    for item in sorted(session.state['goal']['interfaces'],
                       key=lambda i: session.access(i)['status'] == 'inaccessible'):
        # Initial occupancy has its own late review; do not disturb that
        # interface merely to re-observe a release which this session never made.
        if not any(a.get('releaseExecuted') or a.get('releaseHistory') for a in session.rows(item)):
            continue
        from steward.ego.released_object import current_success
        if all(current_success(a) for a in session.rows(item)): continue
        session.state['interface'] = item['key']
        for changed in (False, True):
            if changed and not any(state(a) == 'executed_unverified' for a in session.rows(item)):
                break
            try:
                report = observe(session, item, change_view=changed)
                apply(session, item, report, 'changed_view' if changed else 'before_return')
            except (SkillFailed, AttributeError, KeyError, TypeError, ValueError) as exc:
                check_budget()
                session.emit('outcome_review_unconfirmed', reason=str(exc), changedView=changed)
        # An unavailable alternate view does not suppress a repair supported by
        # the first view. Act on the best retained result, then inspect the action.
        try:
            restore_component(session, item)
            repair_empty(session, item)
            repair_not_seated(session, item)
        except (SkillFailed, AttributeError, KeyError, TypeError, ValueError) as exc:
            check_budget()
            park(session)
            session.emit('outcome_repair_pending', reason=str(exc))
    session.state['outcomeReviewedAt'] = session.rt.ego_state().stamp


def summary(session):
    holes = []
    for item in session.state['goal']['interfaces']:
        from steward.ego import session_faces
        initial_row = session.state.get('initialInterfaces', {}).get(item['key'], {})
        initial = initial_row.get('holes', {})
        for hole, attempt in zip(item['holes'], session.rows(item)):
            label = state(attempt)
            if not attempt and session_faces.protected(initial_row):
                label = 'existing_structure_unconfirmed'
            if not attempt and initial.get(hole, {}).get('occupied'):
                label = 'confirmed_occupied' if initial[hole].get('confirmed') else 'executed_unverified'
            check = attempt.get('seatCheck') or initial.get(hole, {}).get('check') or {}
            holes.append(dict(interface=item['key'], body=session.canonical(item['body']), hole=hole,
                state=label, released=bool(attempt.get('releaseExecuted') or attempt.get('releaseHistory')),
                observedAt=check.get('observedAt'), historicalConfirmation=bool(attempt.get('seatConfirmed')),
                currentPoseConfirmed=check.get('currentPoseConfirmed')))
    components = deepcopy(session.state.get('outcomeComponents', {}))
    pending = any(h['state'] in ('observed_empty', 'observed_not_seated') for h in holes)
    pending |= any(r.get('matches') is False for r in components.values())
    return dict(holeOutcomes=holes, componentOutcomes=components, repairPending=pending,
        canContinue=True, taskComplete=False,
        nextInternalAction='restore_component_then_refill_observed_empty' if pending else 'review_unconfirmed_results',
        resumeSuggestion=dict(skill='complete_assembly', params=dict(session=session.state['id'])),
        guidance='释放是动作记录。已确认占用、已执行未确认、观察为空分别列出；继续同一会话，先复位离位上件，再补观察为空的孔。看不清按最好估计继续，完成动作后复查。')
