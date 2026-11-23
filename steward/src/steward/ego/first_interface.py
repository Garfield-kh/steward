"""Finish the exposed interface before transporting its loose assembly.

Consumes existing robot views and physical release records. This module never
acquires a view, infers occupancy from missing stock, or treats a failed release
as a successful seat. Its recovery allowance survives explicit resumes.
"""
from copy import deepcopy
from steward.ego import capabilities
from steward.ego.skills import SkillFailed


def enabled():
    return capabilities.get('STEWARD_ASSEMBLY_FIRST_INTERFACE') != '0'


def initialize(session):
    if not enabled() or 'firstInterfaceCandidates' in session.state:
        return
    access = {i['key']: access_status(session, i) for i in session.state['goal']['interfaces']}
    session.state['firstInterfaceInitialAccess'] = access
    session.state['firstInterfaceCandidates'] = [k for k,v in access.items() if v != 'inaccessible']
    tracks = getattr(session.rt.estimator, 'tracks', {})
    session.state['firstInterfaceSources'] = {eid: deepcopy(tracks[eid])
        for i in session.state['goal']['interfaces'] if i['key'] in session.state['firstInterfaceCandidates']
        for eid in (i['body'], i['part']) if eid in tracks}


def restore_sources(session, item):
    """A lost mask retains its bound prior, with its original observation age."""
    sources = session.state.get('firstInterfaceSources', {})
    est = session.rt.estimator
    with session.rt._lock:
        for original in (item['body'], item['part']):
            eid = session.canonical(original)
            if any(session.canonical(f.get('entity')) == eid for f in
                   getattr(est, 'released_interfaces', {}).values()):
                continue  # Never resurrect the loose pose after an actual release.
            if eid not in est.tracks and original in sources:
                track = deepcopy(sources[original])
                track.id = eid
                est.tracks[eid] = track
                if hasattr(est, 'version'):
                    est.version += 1


def access_status(session, item):
    try:
        return session.access(item)['status']
    except (SkillFailed, AttributeError, KeyError, IndexError, TypeError, ValueError):
        # A retired mask cannot erase the bound interface or demand another view.
        return session.state.get('firstInterfaceInitialAccess', {}).get(item['key'], 'unknown')


def applies(session, item):
    return enabled() and item['key'] in session.state.get('firstInterfaceCandidates', [])


def ready(session, item):
    """Unknown after actual execution is allowed; a known miss is not a seat."""
    if session.state.get('outcomeComponents', {}).get(item['key'], {}).get('matches') is False:
        return False
    if any(c.get('passed') is False for c in session.state.get('contradictions', {}).get(item['key'], [])):
        return False
    initial = session.state.get('initialInterfaces', {}).get(item['key'], {}).get('holes', {})
    for hole, attempt in zip(item['holes'], session.rows(item)):
        if attempt:
            if (not attempt.get('releaseExecuted') or attempt.get('firstInterfaceNeedsReseat') or
                    (attempt.get('seatCheck') or {}).get('passed') is False):
                return False
        elif not initial.get(hole, {}).get('occupied'):
            return False
    return bool(item['holes'])


def pending(session, body=None):
    return [i for i in session.state['goal']['interfaces'] if applies(session, i)
        and (body is None or session.canonical(i['body']) == session.canonical(body))
        and not ready(session, i)]


def guard_turn(session, body):
    if not enabled():
        return
    blockers = pending(session, body)
    if blockers:
        if all(i['key'] == session.state.get('interface') and
               access_status(session, i) == 'inaccessible' for i in blockers):
            return  # This unfinished interface itself needs the opposite face.
        # The scheduler already used the local repair. Never start a second
        # controller recursively from inside an in-flight flip receipt.
        raise SkillFailed('first_interface_pending_before_turn: ' + ','.join(i['key'] for i in blockers))


def guard_runtime_turn(rt, body):
    """The public flip primitive cannot bypass a known session prerequisite."""
    if not enabled():
        return
    from steward.ego.assembly_session import Session, sessions
    for state in sessions(rt).values():
        if state.get('generation') == rt._generation:
            owner = Session(rt); owner.state = state
            guard_turn(owner, body)


def resume_owner(rt, requests):
    """A smaller restatement cannot abandon this body's unfinished first face."""
    if not enabled():
        return None
    from steward.ego.assembly_session import Session, sessions
    bodies = {r.get('body') for r in requests if isinstance(r, dict)}
    for state in sessions(rt).values():
        if state.get('generation') != rt._generation:
            continue
        owner = Session(rt); owner.state = state
        if any(pending(owner, b) for b in bodies):
            return state['id']
    return None


def repair(session, item):
    """One local action pass, using native release views already in the ledger."""
    if not applies(session, item) or ready(session, item):
        return
    from steward.ego import assembly_outcome, axial_release
    from steward.ego.assembly_session import check_budget
    from steward.ego.fasten_release import released_interface
    used = session.state.setdefault('firstInterfaceRepairs', {})
    key = item['key']
    session.state['interface'] = key
    body, part = (session.canonical(item[k]) for k in ('body', 'part'))
    common = dict(body=body, part=part)

    def act(label, skill, params, execute=None):
        token = key + '/' + label
        if token in used:
            return False
        used[token] = 'attempted'  # reserve before motion, including interruption
        session.state['action'] = skill
        session.emit('first_interface_repair', repair=label, skill=skill, params=params)
        try:
            if execute is not None:
                execute()
            else:
                session.action(skill, params)
            if label in ('component', 'component_best_estimate'):
                if session.rt.estimator.held_id:
                    raise SkillFailed('component_release_not_executed')
                session.state.get('contradictions', {}).pop(key, None)
                session.state.setdefault('outcomeComponents', {})[key] = dict(
                    measured=False, matches=None, reason='component_replaced_unverified')
            used[token] = 'executed'
            return True
        except (SkillFailed, StopIteration, AttributeError, KeyError, IndexError, TypeError, ValueError) as exc:
            check_budget()
            used[token] = str(exc) or type(exc).__name__
            session.emit('first_interface_repair_unconfirmed', repair=label, reason=used[token])
            return False
        finally:
            if session.rt.estimator.held_id:
                assembly_outcome.park(session)

    if session.rt.estimator.held_id:
        assembly_outcome.park(session)
    # Failure before release must not become 'done' merely because the outer
    # interface controller returned. Use its ordinary primitives as fallback.
    restore_sources(session, item)
    scene = session.rt.ego_state()
    frame = released_interface(scene, part, body)
    retained = getattr(session.rt.estimator, 'released_interfaces', {}).get(part)
    if not frame and retained and session.canonical(retained.get('body')) == body:
        frame = retained  # An actual release supersedes its old loose-source prior.
    initial = session.state.get('initialInterfaces', {}).get(key, {})
    if (not frame and not any(a.get('releaseExecuted') for a in session.rows(item))
            and initial.get('component', {}).get('onBody') is not True):
        # Re-running the compound skill would repeat its registration views.
        # Its bound source estimate instead supplies ordinary pick/place motion.
        def place_estimate():
            session.controller.step(session.job, 'pick', dict(target=part, stayNear=True))
            session.controller.step(session.job, 'place', dict(on=body, alignHoles=True,
                hole=item['holes'], heldHole=item['heldHoles'], stayNear=True, closeRelease=True))
        if not act('component_best_estimate', 'pick_then_place', common, place_estimate):
            session.state.setdefault('firstInterfaceRepairFinished', {})[key] = 'component_motion_failed'
            return
    support = assembly_outcome.support_current(session, item)
    if support.get('measured'):
        session.state.setdefault('outcomeComponents', {})[key] = deepcopy(support)
    if support.get('matches') is False:
        if any(a.get('releaseExecuted') and (a.get('seatCheck') or {}).get('passed') is not False
               for a in session.rows(item)):
            # Do not pull an upper layer against a possibly occupied hole.
            session.state.setdefault('firstInterfaceRepairFinished', {})[key] = 'occupied_component_preserved'
            return
        # Reuse the existing component repair and its release-owned observation.
        token = key + '/component_restore'
        if token not in used:
            used[token] = 'attempted'
            try:
                assembly_outcome.restore_component(session, item)
            except (SkillFailed, KeyError, TypeError, ValueError) as exc:
                check_budget()
                session.emit('first_interface_repair_unconfirmed', repair='component_restore', reason=str(exc))
            finally:
                assembly_outcome.park(session)
        if session.state.get('outcomeComponents', {}).get(key, {}).get('matches') is False:
            session.state.setdefault('firstInterfaceRepairFinished', {})[key] = 'component_repair_still_contradicted'
            return
    for hole, attempt in zip(item['holes'], session.rows(item)):
        check_budget()
        if not attempt and initial.get('holes', {}).get(hole, {}).get('occupied'):
            continue
        if attempt.get('releaseExecuted'):
            check = attempt.get('seatCheck') or {}
            if check.get('passed') is not False and not attempt.get('firstInterfaceNeedsReseat'):
                continue  # Unknown reserves the actual release, never double-fill.
            token = key + '/seat/' + hole
            if token in used:
                continue
            if assembly_outcome.empty(attempt):
                attempt['refillAuthorized'] = deepcopy(check)
            else:
                # Same original object only. Existing identity and departure
                # admission still governs pickup; no new stock for a raised head.
                attempt['firstInterfaceRetryGranted'] = True
                attempt.setdefault('firstInterfaceRetryBase', attempt.get('seatRetryCount', 0))
                attempt.pop('axialReleaseHandled', None)
            act('seat/' + hole, 'seat_screw', dict(common, hole=hole, deferVerify=True))
        else:
            act('seat/' + hole, 'seat_screw', dict(common, hole=hole, deferVerify=True))
    session.emit('first_interface_result', readyForTurn=ready(session, item),
                 unconfirmedHoles=[h for h,a in zip(item['holes'], session.rows(item))
                                   if not a.get('seatConfirmed')])
    session.state.setdefault('firstInterfaceRepairFinished', {})[key] = 'local_actions_finished'


def summary(session, result):
    if not enabled():
        return result
    rows = pending(session)
    if not rows:
        return result
    result['firstInterfacePending'] = [dict(interface=i['key'], body=session.canonical(i['body']),
        holes=[h for h,a in zip(i['holes'], session.rows(i))
               if not a.get('releaseExecuted') or a.get('firstInterfaceNeedsReseat') or
               (a.get('seatCheck') or {}).get('passed') is False]) for i in rows]
    result['nextInternalAction'] = session.state.get('action') or 'finish_current_interface_before_turn'
    result['guidance'] = ('先完成当前可装接口的上件和所有孔，再翻面。缺少观察按保留估计执行并标注未确认；'
        '已释放且看不清的孔不重复投料。失败孔在当前面用同一物体纠正，明确空孔才补料。'
        '不要缩小目标重开会话或重复观察；恢复沿用本会话的剩余动作与预算。')
    result['firstInterfaceRepairs'] = deepcopy(session.state.get('firstInterfaceRepairs', {}))
    if (result.get('decision') in ('continue_assembly', 'needs_input', 'executed_unverified', 'incomplete_after_actions') and
            all(i['key'] in session.state.get('firstInterfaceRepairFinished', {}) for i in rows)):
        # Unknown vision alone is ready() after a real release and never enters
        # this branch. These are actual refused/failed actions or observed misses
        # after the local correction. Do not advertise an identical no-op resume.
        result.update(decision='incomplete_after_actions', canContinue=False, taskComplete=False,
            nextInternalAction='current_interface_actions_exhausted',
            guidance='当前接口已按现有估计执行并做完有限补救，仍有明确未完成项；保留当前面。'
                     '未确认不等于成功；相同状态续跑不会产生新动作，不要重复观察、缩小目标或求助以跳过这些孔。')
        result.pop('resumeSuggestion', None)
    return result


def same_release(left, right, canonical=lambda x: x):
    """Two keys for one physical release are aliases, not competing occupants."""
    return bool(enabled() and left.get('releaseExecuted') and right.get('releaseExecuted')
        and left.get('seatedAt') is not None and left.get('seatedAt') == right.get('seatedAt')
        and left.get('screw') is not None and canonical(left['screw']) == canonical(right.get('screw'))
        and left.get('hole') == right.get('hole'))


def merge_release(left, right):
    """Keep the newer observation of the identical release and its histories."""
    check_time = lambda row: ((row.get('seatCheck') or {}).get('observedAt') or -1)
    older, newer = sorted((left, right), key=check_time)
    merged = dict(older, **newer)
    for key in ('releaseHistory', 'seatHistory'):
        history = []
        for entry in [*left.get(key, []), *right.get(key, [])]:
            if entry not in history:
                history.append(deepcopy(entry))
        if history:
            merged[key] = history
    return merged
