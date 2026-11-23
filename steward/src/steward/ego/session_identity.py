"""Persistent assembly membership, with replaceable robot-observation bindings.

No scene catalogue, world state or task-specific IDs. Association is coarse;
ordinary action admission and ASSEMBLY_GEOMETRY_RELOCALIZATION alone own shape and precision authority.
"""
from copy import deepcopy
from types import SimpleNamespace
import math
import hashlib
import numpy as np
from steward.ego import capabilities
from steward.ego.skills import SkillFailed


def enabled():
    return capabilities.get('STEWARD_ASSEMBLY_SESSION_IDENTITY') != '0'


def alias(state, eid):
    seen = set()
    while eid in state.get('aliases', {}) and eid not in seen:
        seen.add(eid)
        eid = state['aliases'][eid]
    return eid


def ledger(est, state):
    members = state.setdefault('members', {})
    for item in state['goal']['interfaces']:
        for role in ('body', 'part'):
            eid = item[role]
            row = members.setdefault(eid, dict(id=eid, role=role, interfaces=[]))
            if item['key'] not in row['interfaces']: row['interfaces'].append(item['key'])
            if role == 'part':
                row['assembly'] = item['body']
                row['installed'] = bool(row.get('installed') or item['key'] in state.get('completed', []) or
                    any(r.get('entity', r.get('part')) in (alias(state, eid), target(state, eid))
                        and r.get('body') in (alias(state, item['body']), target(state, item['body']))
                        for r in getattr(est, 'released_interfaces', {}).values()))
    for eid, material in state['goal'].get('materials', {}).items():
        members.setdefault(eid, dict(id=eid, role='material', interfaces=[], support=material['source']))
    tool = state['goal'].get('tools', {}).get('entity')
    if tool:
        members.setdefault(tool, dict(id=tool, role='tool', interfaces=[]))
    for source in {v['source'] for v in state['goal'].get('materials', {}).values()}:
        members.setdefault(source, dict(id=source, role='support', interfaces=[]))
    records = getattr(getattr(est, 'stationary_assemblies', None), 'records', {})
    for row in list(members.values()):
        record = records.get(alias(state, row.get('assembly', row['id'])), {})
        support = record.get('support')
        if getattr(support, 'id', None):
            members.setdefault(support.id, dict(id=support.id, role='support', interfaces=[]))
    for eid, row in list(members.items()):
        current = alias(state, eid)
        track = getattr(est, 'tracks', {}).get(current)
        record = records.get(alias(state, row.get('assembly', eid)), {})
        support = record.get('support')
        support_id = getattr(support, 'id', None) or row.get('support')
        if support_id:
            row['support'] = support_id
            members.setdefault(support_id, dict(id=support_id, role='support', interfaces=[]))
        released = next((r for r in getattr(est, 'released_interfaces', {}).values()
            if r.get('entity') in (eid, current)), {})
        if 'prior' not in row and released.get('position') and released.get('size'):
            row['prior'] = dict(position=list(released['position']), size=list(released['size']),
                variance=[.0004]*3, category=released.get('category'))
        if 'prior' not in row and track is not None:
            row['prior'] = dict(position=np.asarray(track.pos).tolist(), size=np.asarray(track.size).tolist(),
                variance=np.asarray(track.var).tolist(), category=track.category)
            st = getattr(est, 'tracks', {}).get(support_id)
            if st is not None:
                offset = np.asarray(track.pos)-st.pos
                row['prior']['supportOffset'] = offset.tolist()
                c, sn = math.cos(st.yaw), math.sin(st.yaw)
                row['prior']['supportLocalOffset'] = [c*offset[0]+sn*offset[1], -sn*offset[0]+c*offset[1], offset[2]]
                row['prior']['supportVariance'] = np.asarray(st.var).tolist()
        if row.get('prior') and row.get('assembly') and row.get('installed'):
            bt = getattr(est, 'tracks', {}).get(alias(state, row['assembly']))
            if bt is not None:
                row.setdefault('assemblyOffset', (np.asarray(row['prior']['position'])-bt.pos).tolist())
        row.setdefault('binding', dict(track=current if track is not None else None,
            status='prior' if state.get('memoryResumed') else 'observed', confirmed=False))
        row.setdefault('consumedCandidates', [])
    return members


def target(state, eid):
    row = state.get('members', {}).get(eid)
    if row:
        b = row.get('binding', {})
        if b.get('track'): return alias(state, b['track'])
        # A composite resolves membership, never a pick target for the hidden part.
        # Callers inspect assemblyTrack explicitly when they need the assembly.
    return alias(state, eid)


def current_failure(evidence, after=-math.inf):
    stamp = evidence.get('observedAt')
    return bool(evidence and not evidence.get('restoredPrior') and evidence.get('currentPoseConfirmed') is not False
                and isinstance(stamp, (int, float)) and stamp >= 0 and stamp > after)


def retire_history(est, state):
    """Retain action facts and history, removing only stale scheduling authority."""
    if not enabled() or not state.get('memoryResumed'): return
    after = max(state.get('memoryRestoredAt', 0), state.get('disturbedAt', -math.inf))
    bodies = {alias(state, i['body']) for i in state['goal']['interfaces']} | {target(state, i['body']) for i in state['goal']['interfaces']}
    for a in getattr(est, 'fastening_attempts', {}).values():
        if alias(state, a.get('body')) not in bodies: continue
        a['identityEvidenceAfter'] = max(after, a.get('transportedAt', -math.inf))
        check = a.get('seatCheck') or {}
        if check.get('passed') is False and not current_failure(check, a['identityEvidenceAfter']):
            a['historicalSeatCheck'] = deepcopy(check)
            a['seatCheck'] = dict(passed=None, status='unknown', reason='historical_failure_currently_unconfirmed',
                currentPoseConfirmed=False, restoredPrior=True)
            a.pop('firstInterfaceNeedsReseat', None)
    for key, row in state.get('outcomeComponents', {}).items():
        if row.get('matches') is False and not current_failure(row, after):
            state.setdefault('historicalComponentOutcomes', {})[key] = deepcopy(row)
            row.update(matches=None, measured=False, reason='historical_failure_currently_unconfirmed')
    for key, rows in list(state.get('contradictions', {}).items()):
        checks = rows if isinstance(rows, list) else [rows]
        if not any(current_failure(c, after) for c in checks if isinstance(c, dict)):
            state.setdefault('historicalContradictions', {})[key] = state['contradictions'].pop(key)


def restored(est, state):
    if not enabled(): return
    ledger(est, state)
    state.pop('identityViewAttempted', None)
    state.pop('identityPending', None)
    for row in state['members'].values():
        row.pop('viewAttempted', None)
        row.pop('pickPending', None)
        row['binding'].update(status='prior', confirmed=False, precisionEligible=False)
    retire_history(est, state)


def dropped(est, track, into=None):
    if not enabled(): return
    for state in getattr(est, 'assembly_sessions', {}).values():
        members = ledger(est, state)
        affected = [r for e, r in members.items() if target(state, e) == track.id]
        if not affected: continue
        others = [r for e, r in members.items() if target(state, e) == getattr(into, 'id', None) and r not in affected]
        if into is not None and not others:
            from steward.ego.assembly_session import track_renamed
            track_renamed(est, track.id, into.id)
            for row in affected:
                row['binding'] = dict(track=into.id, status='associated', confirmed=False)
        else:
            for row in affected:
                row['binding'] = dict(track=None, status='track_absent', confirmed=False)
                if into is not None:
                    row['binding'].update(assemblyTrack=into.id, status='assembly_member_unconfirmed')
                    state.setdefault('assemblyObservations', {})[into.id] = dict(
                        members=list(dict.fromkeys([r['id'] for r in affected+others])),
                        interpretation='composite', confirmed=False)


def migrated(est, old, new):
    """Move observation dependencies, never stable member keys or action history."""
    fields = {'entity','body','part','screw','support','kitSupport','source','target','object','track','assemblyTrack','requested','held','body_id'}
    def visit(obj):
        if isinstance(obj, dict):
            for key, value in list(obj.items()):
                if key in fields and isinstance(value, str) and value == old: obj[key] = new
                elif key not in ('releaseHistory', 'prior', 'goal', 'aliases'): visit(value)
        elif isinstance(obj, list):
            for value in obj: visit(value)
    for state in getattr(est, 'assembly_sessions', {}).values():
        if alias(state, old) == new:
            for key in ('members','initialInterfaces','outcomeComponents','supplyEntities','savedControllerMemory','finalPose'):
                visit(state.get(key, {}))
            # Estimator callbacks may run outside the action ContextVar. The
            # persistent session owns the live controller in either thread.
            controller = state.get('controller')
            if getattr(controller, 'body_id', None) == old: controller.body_id = new
            if controller is not None:
                for key in ('_registered_alignment', '_tool_sources'):
                    visit(getattr(controller, key, None))
                sources = getattr(controller, '_source_tracks', {})
                if old in sources and new not in sources:
                    sources[new] = sources.pop(old)
                    sources[new].id = new
    for name in ('_seat_registrations','seat_assemblies','rigid_carry','fastener_releases',
                 '_interface_entry_reports','_interface_entry_budgets','_component_alignment_attempts'):
        mapping = getattr(est, name, None)
        if isinstance(mapping, dict):
            for key in list(mapping):
                replacement = new if key == old else tuple(new if x == old else x for x in key) if isinstance(key, tuple) else key
                if replacement != key and replacement not in mapping: mapping[replacement] = mapping.pop(key)
            visit(mapping)
    visit(getattr(getattr(est, 'assembly_memory', None), 'records', []))
    stationary = getattr(getattr(est, 'stationary_assemblies', None), 'records', {})
    for record in stationary.values():
        visit(record)
        for key in ('entity', 'support'):
            entity = record.get(key)
            if getattr(entity, 'id', None) == old: entity.id = new
        if 'aliases' in record:
            record['aliases'] = list(dict.fromkeys(new if e == old else e for e in record['aliases']))
    for track in getattr(est, 'tracks', {}).values():
        for name in ('support', 'parent'):
            if getattr(track, name, None) == old: setattr(track, name, new)
        for name in ('placement', 'held_interface', 'grasp_association'):
            visit(getattr(track, name, None))
    from steward.ego.assembly_session import current
    active = current()
    if active is not None and active.rt.estimator is est:
        visit(getattr(active.rt, '_support_hole_watch', None))
        controller = getattr(active, 'controller', None)
        if getattr(controller, 'body_id', None) == old: controller.body_id = new
        if controller is not None:
            for name in ('_registered_alignment', '_tool_sources'):
                visit(getattr(controller, name, None))


def predicted(est, state, row):
    prior = row.get('prior', {})
    position = prior.get('position')
    support = getattr(est, 'tracks', {}).get(alias(state, row.get('support')))
    if support is not None and 'supportLocalOffset' in prior:
        x, y, z = prior['supportLocalOffset']; c, sn = math.cos(support.yaw), math.sin(support.yaw)
        position = np.asarray(support.pos)+[c*x-sn*y, sn*x+c*y, z]
    elif support is not None and 'supportOffset' in prior:
        position = np.asarray(support.pos)+prior['supportOffset']
    return None if position is None else np.asarray(position, float)


def candidates(est, state, row):
    position = predicted(est, state, row)
    if position is None: return []
    prior = row['prior']; expected = np.sort(prior['size'])
    variance = np.maximum(np.asarray(prior.get('variance', [.01]*3)), .0001)
    support = getattr(est, 'tracks', {}).get(alias(state, row.get('support')))
    if support is not None:
        variance = variance+np.asarray(support.var)+np.asarray(prior.get('supportVariance', [0.]*3))
    others = {target(state, e) for e, r in state['members'].items() if r is not row and r['role'] != 'support'}
    consumed = {alias(state, e) for e in row['consumedCandidates']}
    results = []
    for t in getattr(est, 'tracks', {}).values():
        if t.id in consumed or t.id in others or getattr(t, 'held', False): continue
        if t.last_seen < state.get('memoryRestoredAt', 0): continue
        size = np.asarray(t.size, float)
        if not np.isfinite(size).all() or np.any(size <= 0): continue
        dist = float(np.linalg.norm((np.asarray(t.pos)-position)/np.sqrt(variance+np.asarray(t.var)+.0004)))
        shape = float(np.linalg.norm(np.log(np.sort(size)/np.maximum(expected, .001))))
        interpretations = [dict(kind='independent', shapeError=shape)]
        attached = [m for m in state['members'].values() if m.get('assembly') == row['id']
                    and m.get('installed') and m.get('prior') and 'assemblyOffset' in m]
        if row['role'] == 'body' and attached:
            lo, hi = -np.asarray(prior['size'])/2, np.asarray(prior['size'])/2
            for member in attached:
                p, extent = np.asarray(member['assemblyOffset']), np.asarray(member['prior']['size'])/2
                lo, hi = np.minimum(lo, p-extent), np.maximum(hi, p+extent)
            composite_error = float(np.linalg.norm(np.log(np.sort(size)/np.maximum(np.sort(hi-lo), .001))))
            interpretations.append(dict(kind='composite', shapeError=composite_error, members=[m['id'] for m in attached]))
        shape = min(h['shapeError'] for h in interpretations)
        if dist > 8 or shape > 1.5: continue  # Association envelope, not action admission.
        visible = getattr(t, 'manipulation_view', None) or {}
        geometry = bool(visible and not visible.get('truncated', True))
        score = -dist-shape*3 + .25*(t.category == prior.get('category')) + .5*geometry
        results.append(dict(track=t.id, score=score, distance=dist, shape=shape,
            geometry=geometry, interpretations=interpretations, position=np.asarray(t.pos).tolist()))
    return sorted(results, key=lambda r: (-r['score'], r['track']))


def reconcile(est, state):
    if not enabled(): return
    members = ledger(est, state)
    retire_history(est, state)
    for eid, row in members.items():
        b = row['binding']; tid = alias(state, b.get('track') or eid)
        track = getattr(est, 'tracks', {}).get(tid)
        if row.get('installed'):
            if track is None:
                body = target(state, row['assembly'])
                row['binding'] = dict(track=None, assemblyTrack=body, status='assembly_member_unconfirmed', confirmed=False)
            continue
        if track is not None and b.get('status') == 'grasp_supported':
            continue
        if row['role'] in ('support', 'material', 'tool'):
            # Bound stock/tool ownership still uses its existing source policy;
            # the assembly identity search must not repurpose it as a component.
            if track is None:
                row['binding'] = dict(track=None, status='track_absent', confirmed=False)
            continue
        ranked = candidates(est, state, row)
        row['candidates'] = ranked[:8]
        if not ranked:
            row['binding'] = dict(track=None, status='local_observation_required', confirmed=False)
            if (row['role'] in ('body', 'part') and track is not None and state.get('identityViewAttempted')
                    and tid not in {alias(state, e) for e in row['consumedCandidates']}):
                # A still-existing coarse pose is an executable best estimate
                # after the one view, not fresh evidence. Ordinary pick/shape
                # admission remains responsible for whether it can be used.
                row['binding'] = dict(track=tid, status='tentative', source='ledger_prediction',
                    confirmed=False, precisionEligible=False)
            continue
        best = ranked[0]
        same = best['track'] == alias(state, eid)
        unambiguous = len(ranked) == 1 or best['score']-ranked[1]['score'] > 1.
        status = 'observed_track' if same and unambiguous else 'tentative'
        composite = next((h for h in best['interpretations'] if h['kind'] == 'composite'), None)
        if composite:
            state.setdefault('assemblyObservations', {})[best['track']] = dict(
                members=[eid, *composite['members']], interpretations=deepcopy(best['interpretations']), confirmed=False)
        # Ambiguity remains explicit even when a single actionable candidate wins.
        row['binding'] = dict(track=best['track'], status=status, confirmed=False,
            score=best['score'], precisionEligible=False)
        if b.get('track') != best['track'] or status == 'tentative' and b.get('status') != 'tentative':
            records = getattr(getattr(est, 'stationary_assemblies', None), 'records', {})
            body = target(state, row.get('assembly', eid))
            if body in records:
                from steward.ego.geometry_relocalization import pending
                pending(records[body]['reference'], est.stamp, 'tentative_identity_binding')
        if row['role'] == 'body' and best['track'] != alias(state, eid) and b.get('track') != best['track']:
            records = getattr(getattr(est, 'stationary_assemblies', None), 'records', {})
            old = records.get(alias(state, eid))
            if old and best['track'] not in records:
                records[best['track']] = deepcopy(old)
                records[best['track']]['entity'].id = best['track']
                records[best['track']]['reference']['entity'] = best['track']
                records[best['track']]['aliases'] = [best['track']]
            if best['track'] in records:
                from steward.ego.geometry_relocalization import pending
                candidate = est.tracks[best['track']]
                # Keep ASSEMBLY_GEOMETRY_RELOCALIZATION's original extent/category/axis compatibility rule.
                # Incompatible/composite regions remain a coarse prior, not a
                # fabricated current independent body outline.
                from steward.ego.memory_checkpoint import Memory
                proxy = SimpleNamespace(rt=SimpleNamespace(estimator=est))
                Memory._correct_reference(proxy, records[best['track']], best['track'],
                    candidate.pos, candidate.size, candidate.yaw, candidate.category)
                pending(records[best['track']]['reference'], est.stamp, 'tentative_identity_binding')
    state['identityReassociatedAt'] = getattr(est, 'stamp', 0.)


def acquire_region(session, row):
    """A missing entity can still request native pixels at its predicted region."""
    rt, est = session.rt, session.rt.estimator
    point = predicted(est, session.state, row)
    candidate = est.tracks.get(row['binding'].get('track'))
    if candidate is not None: point = candidate.pos
    if point is None or not hasattr(rt, 'request_precision_view'):
        session.emit('identity_region_unavailable', member=row['id'], reason='no_located_region')
        return
    from steward.ego import action_budget as ab
    from steward.ego.assembly_session import check_budget
    since = rt.executor._state()['stamp']
    with rt._lock:
        request = rt.request_precision_view([np.asarray(point).tolist()], since)
    try:
        deadline = ab.make(rt.executor, ab.observation(rt.executor) if ab.enabled() else 8., 'identity_local_region')
        while request.get('cameras') and not request.get('completed') and not ab.expired(deadline):
            check_budget()
            rt.executor._wait(.1)
            if rt.executor._state()['stamp']-since >= (ab.observation(rt.executor) if ab.enabled() else 8.): break
        row['localObservation'] = dict(region=np.asarray(point).tolist(), requestedAt=since,
            completed=bool(request.get('completed')), reason=request.get('reason'))
    finally:
        rt.finish_precision_view(request)


def local_view(session, row):
    """One directed view per recovery; an unavailable trip consumes the budget."""
    if row.get('viewAttempted') or session.state.get('identityViewAttempted'): return
    row['viewAttempted'] = True
    session.state['identityViewAttempted'] = True
    est = session.rt.estimator
    point = predicted(est, session.state, row)
    candidate = getattr(est, 'tracks', {}).get(row['binding'].get('track'))
    if candidate is not None: point = candidate.pos
    if not hasattr(session.rt, 'ego_state'):
        session.emit('identity_local_view_unavailable', member=row['id'], reason='observation_runtime_unavailable')
        return
    scene = session.rt.ego_state()
    support = scene.find(alias(session.state, row.get('support')))
    if support is None:
        support = scene.find(session.source_support(None))
    if support is None or point is None:
        session.emit('identity_local_view_unavailable', member=row['id'], reason='support_or_prediction_unavailable')
        return
    from steward.ego.planner import entity_box
    box, _, _ = entity_box(support)
    odom = session.rt.executor._state()['odometry']
    session.controller.observation_target = np.asarray(point).tolist()
    try:
        session.controller.view(session.job, support, math.atan2(odom['y']-box.cy, odom['x']-box.cx), math.pi/3)
        acquire_region(session, row)
    except (SkillFailed, KeyError, ValueError, AttributeError) as exc:
        from steward.ego.assembly_session import check_budget
        check_budget()
        session.emit('identity_local_view_unconfirmed', member=row['id'], reason=str(exc))


def prepare(session):
    if not enabled() or not session.state.get('memoryResumed'): return
    session.state.pop('identityPending', None)
    with session.rt._lock: reconcile(session.rt.estimator, session.state)
    # At most one physical view for this entry; each unresolved member gets at
    # most one across resumes. Remaining candidates are tried immediately.
    unresolved = [r for r in session.state['members'].values() if not r.get('installed')
        and r['role'] in ('body','part') and r['binding']['status'] in ('tentative','local_observation_required')]
    unresolved.sort(key=lambda r: r['role'] != 'part')
    for row in unresolved:
        if not row.get('viewAttempted'):
            local_view(session, row)
            break
    with session.rt._lock: reconcile(session.rt.estimator, session.state)
    if session.state.get('identityViewAttempted'):
        session.state['initialViewChanges'] = max(1, session.state.get('initialViewChanges', 0))
    session.emit('identity_reassociated', nextStep=public(session.rt.estimator, session.state)['identityNextStep'])


def prepare_item(session, item):
    """Resolve before ordinary fasten_onto; that action owns registration/pick/place."""
    if not enabled() or not session.state.get('memoryResumed'): return True
    est = session.rt.estimator
    with session.rt._lock: reconcile(est, session.state)
    row = session.state['members'][item['part']]
    if row.get('installed'): return True
    if not row['binding'].get('track'):
        local_view(session, row)
        # Repeated entry with all candidates consumed obtains local evidence,
        # never another orbit or another call with the old missing identity.
        if session.state.get('identityViewAttempted'):
            acquire_region(session, row)
        with session.rt._lock: reconcile(est, session.state)
    # Scores rank identities; ordinary pick admission decides whether their
    # geometry is operable. Inspect at most the bounded candidate shortlist.
    if row['binding'].get('track') and row['binding'].get('status') != 'grasp_supported' and hasattr(session.rt, 'predict'):
        for _ in range(8):
            candidate = row['binding'].get('track')
            if not candidate: break
            receipt = session.rt.predict('pick', dict(target=candidate, stayNear=True))
            if receipt.get('withinRiskLimit'): break
            row['consumedCandidates'].append(candidate)
            row.setdefault('candidateRejections', {})[candidate] = list(receipt.get('issues', []))
            row['binding'] = dict(track=None, status='candidate_consumed', confirmed=False)
            with session.rt._lock: reconcile(est, session.state)
        else:
            row['binding'] = dict(track=None, status='local_observation_required', confirmed=False)
    if not row['binding'].get('track'):
        acquire_region(session, row)
        session.state.setdefault('unconfirmedInterfaces', {})[item['key']] = 'no_operable_observation_after_local_view'
        session.state['identityPending'] = True
        return False
    return True


def pick_started(session, eid):
    if not enabled(): return
    for row in session.state.get('members', {}).values():
        if row['binding'].get('track') != eid and alias(session.state, row['id']) != eid: continue
        if row.get('installed'):
            # An absent member has no authority to trigger disassembly.
            check = session.state.get('outcomeComponents', {}).get(session.state.get('interface'), {})
            if session.state.get('memoryResumed') and not current_failure(check, max(
                    session.state.get('memoryRestoredAt', 0), session.state.get('disturbedAt', -1))):
                raise SkillFailed('installed_member_without_current_displacement_evidence')
        if session.state.get('memoryResumed') and row['role'] in ('body', 'part') and not row.get('installed'):
            if eid in {alias(session.state, e) for e in row['consumedCandidates']}:
                raise SkillFailed('identity_candidate_already_consumed')
            row['consumedCandidates'].append(eid)
            row['pickPending'] = eid


def action_finished(session):
    if not enabled(): return
    est = session.rt.estimator
    for row in session.state.get('members', {}).values():
        candidate = row.pop('pickPending', None)
        if not candidate: continue
        candidate = alias(session.state, candidate)
        track = est.tracks.get(candidate)
        evidence = deepcopy(getattr(track, 'grasp_association', {}) or row.get('graspEvidence') or {})
        # Identity support requires both pinch and motion from the robot, and
        # still grants neither observed assembly completion nor precision.
        held = est.held_id == candidate
        row['graspEvidence'] = evidence
        if held or alias(session.state, evidence.get('requested')) == candidate and evidence.get('recordedAt', -1) > session.state.get('memoryRestoredAt', 0):
            row['binding'].update(status='grasp_supported', confirmed=False, precisionEligible=False)
        else:
            row['binding'] = dict(track=None, status='candidate_consumed', confirmed=False)
            session.state['identityPending'] = True
        session.emit('identity_candidate_result', member=row['id'], candidate=candidate,
            held=held, evidence=evidence, binding=deepcopy(row['binding']))


def failure(session, item, fingerprint, reason):
    if not enabled() or not session.state.get('memoryResumed') or not reason: return
    version = hashlib.sha256(fingerprint.encode()).hexdigest()
    session.state.setdefault('identityFailures', {})[item['key']] = dict(evidence=fingerprint, evidenceVersion=version, reason=reason)
    ledger(session.rt.estimator, session.state)
    row = session.state['members'][item['part']]
    candidate = row['binding'].get('track')
    if not row.get('installed') and candidate:
        if candidate not in row['consumedCandidates']: row['consumedCandidates'].append(candidate)
        row['binding'] = dict(track=None, status='candidate_consumed', confirmed=False)
    session.state['identityPending'] = True


def repeated(session, item):
    if not enabled() or not session.state.get('memoryResumed'): return False
    previous = session.state.get('identityFailures', {}).get(item['key'])
    if previous and previous['evidence'] == session.fingerprint(item):
        row = session.state['members'][item['part']]
        old = row['binding'].get('track')
        if old and old not in row['consumedCandidates']: row['consumedCandidates'].append(old)
        row['binding'] = dict(track=None, status='candidate_consumed', confirmed=False)
        session.state['identityPending'] = True
        return True
    return False



def retain_grasp(est, track):
    """Capture robot gripping/transport evidence before release clears the Track."""
    if not enabled() or track is None or not getattr(track, 'grasp_association', None): return
    for state in getattr(est, 'assembly_sessions', {}).values():
        for row in state.get('members', {}).values():
            if row['binding'].get('track') == track.id:
                row['graspEvidence'] = deepcopy(track.grasp_association)
                row['graspEvidence']['gripperHolding'] = getattr(est, 'gripper', {}).get('holding')
                row['graspEvidence']['transportPositionM'] = np.asarray(track.pos).tolist()
                row['graspEvidence']['transportSource'] = 'gripper_encoder_prediction'
                row['graspEvidence']['identityVerified'] = False
                measured = (getattr(track, 'held_interface', None) or {}).get('inhandMeasurement')
                if measured and measured.get('observedAt', -1) > row['graspEvidence'].get('recordedAt', -1):
                    row['graspEvidence']['transportObservation'] = deepcopy(measured)
                    row['graspEvidence']['transportSource'] = 'native_inhand_geometry_and_gripper_encoders'


def public(est, state):
    if not enabled() or not state.get('members'): return {}
    members = []
    for eid, row in state['members'].items():
        b = row['binding']
        members.append(dict(member=eid, role=row['role'], installed=bool(row.get('installed')),
            trackPresent=b.get('track') in getattr(est, 'tracks', {}), physicalExistence='not_disproved',
            trackAbsenceMeaning='current_track_absent_not_part_absent',
            currentConfirmation=False, binding=deepcopy(b)))
    remaining = [i for i in state['goal']['interfaces'] if i['key'] not in state.get('completed', [])
        or i['key'] in state.get('contradictions', {})]
    item = (remaining or [None])[0]
    binding = state['members'][item['part']]['binding'] if item else {}
    next_step = dict(skill='complete_assembly', params=dict(session=state['id']),
        branch='identity_continuation', interface=item['key'] if item else None,
        evidenceVersion=state.get('identityFailures', {}).get(item['key'], {}).get('evidenceVersion') if item else None,
        member=item['part'] if item else None, binding=deepcopy(binding),
        region=deepcopy(state['members'][item['part']].get('prior', {}).get('position')) if item else None,
        action='bounded_candidate_pick_then_remaining_interface' if binding.get('track') else
            'observe_ledger_region' if item else 'review_results_and_final_pose')
    return dict(assemblyMembers=members, identityNextStep=next_step)
