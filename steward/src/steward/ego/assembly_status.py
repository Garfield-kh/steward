"""Actionable assembly state from robot registrations and independent evidence only."""
from copy import deepcopy
import numpy as np

from steward.ego.fasten_plan import value, interface_holes


def face_advice(check):
    """Relative face reversal, explicitly separate from absolute uprightness."""
    passed = check.get('passed') if check else None
    if check and check.get('reason') == 'body_still_at_pickup':
        passed = None  # Position alone does not resolve the face.
    return dict(flipState='flipped' if passed is True else 'not_flipped' if passed is False else 'uncertain',
        label='已翻面' if passed is True else '未翻面' if passed is False else '翻面不确定',
        absoluteUpright=None, advisory=True,
        observationSuggestion='从能同时看到本体侧面特征和已装件的斜侧方观察一次；比较翻面前后的凸起在上/在下，'
            '或本体与已装件的高低顺序。两者均看不清则保留不确定，不以动作成功或估计 upZ 代替。',
        nextAction='continue_selected_interface' if passed is True else
                   'retry_flip_assembly' if passed is False else 'observe_face_once')


def interface_access(planner, body, selected, scene=None):
    """Opposed parallel interface layers need the exposed layer, not a top-box fit.

    Layer order comes from the robot's existing interface catalogue. It is only
    an orientation hypothesis unless a post-motion face check supports it.
    Single-layer/side interfaces without normals remain unresolved, not blocked.
    """
    from steward.ego import working_face
    observed = working_face.assess(planner, body, selected, scene)
    if observed is not None:
        return observed
    from steward.ego import session_observation
    observed = session_observation.access(body.id, selected) if body is not None else None
    if observed is not None:
        return observed
    info = value(body, 'orientation', {})
    ledger = info.get('stationaryAssembly', info.get('assembly', {}))
    ref = ledger.get('reference', {})
    from steward.ego import placement_support_margin
    if placement_support_margin.enabled() and ledger.get('edgeRetained'):
        resolution = ref.get('faceResolution', {})
        if resolution.get('status') not in ('estimated', 'observed'):
            return dict(status='unknown', reason='retained_release_face_unconfirmed',
                nextAction='assemble_onto', source='released_registration',
                bodyRetained=True, unconfirmed=True)
    check = ref.get('faceCheck')
    if check and check.get('passed') is None and ref.get('faceBefore'):
        from steward.ego.assembly_motion import flip_check
        view = value(body, 'contactView')
        stamp = view['stamp'] if view else ref.get('faceSince', 0.)
        refreshed = flip_check(ref['faceBefore'], view, {}, {}, ref.get('faceSince', stamp), stamp)
        if refreshed['passed'] is not None:
            check = refreshed
            # If unchanged, the nominal half turn must not determine accessibility.
            if check['passed'] is False:
                ref = dict(ref, rotationM=ref.get('rotationBefore', np.eye(3)))
    resolution = ref.get('faceResolution', {})
    current_face = (resolution.get('source') == 'current_rgbd_signature_registered_face'
                    and resolution.get('since', -1) > ref.get('faceSince', -1))
    if (resolution.get('status') in ('estimated', 'observed')
            and (current_face or not check or check.get('passed') is not False)):
        exposed = set(resolution['exposedHoles'])
        known = {h['id'] for h in ref.get('holes', [])}
        if selected and selected <= known:
            blocked = not bool(selected & exposed)
            return dict(status='inaccessible' if blocked else 'estimated_accessible',
                reason='selected_interface_on_opposite_face' if blocked else 'selected_interface_on_exposed_layer',
                nextAction='flip_assembly_then_verify_face' if blocked else 'assemble_onto',
                source=resolution['source'], faceResolution=deepcopy(resolution))
    if check and check.get('passed') is None:
        return dict(status='unknown', reason='face_unresolved_after_motion',
                    nextAction='observe_body_face_then_retry', source='post_motion_rgbd',
                    faceAdvice=face_advice(check))
    holes = interface_holes(planner, body)
    levels = [float(h['p'][2]) for h in holes]
    chosen = [float(h['p'][2]) for h in holes if h['id'] in selected]
    up = float(np.asarray(ref.get('rotationM', np.eye(3)))[2, 2])
    if not levels or not chosen or max(levels)-min(levels) <= .01 or abs(up) < .8:
        return dict(status='unknown', reason='no_parallel_face_estimate', source='interface_geometry')
    mid = (max(levels)+min(levels))/2
    facing = [(z-mid)*up for z in chosen]
    blocked = all(z < -.005 for z in facing)
    return dict(status='inaccessible' if blocked else 'estimated_accessible',
                reason='selected_interface_on_opposite_face' if blocked else 'selected_interface_on_exposed_layer',
                nextAction='flip_assembly_then_verify_face' if blocked else 'assemble_onto',
                bodyUpZEstimate=up, source='registered_interface_layers')


def require_interface_access(planner, body, selected):
    # Catalogue layers and nominal motion are advisory, not measured normals.
    return dict(interface_access(planner, body, selected), advisory=True)


def assembly_state(rt, plan, post):
    """A catalogue inventory, never a claim to know the operator's whole task."""
    scene = rt.ego_state()
    body_id = plan.get('body', post.get('body'))
    body = scene.find(body_id) if body_id else None
    records = rt.estimator.stationary_assemblies.records
    record = records.get(body_id, {})
    ref = record.get('reference', {})
    attempts = [a for a in rt.estimator.fastening_attempts.values() if a['body'] == body_id]
    frames = [f for f in getattr(rt.estimator, 'released_interfaces', {}).values() if f['body'] == body_id]
    parts = {p['entity']:p for p in record.get('parts', [])}
    parts.update({f['entity']:f for f in frames})
    parts.update({a['part']:{} for a in attempts if a['part'] not in parts})
    installed = []
    occupied = set()
    for eid in parts:
        frame = next((f for f in frames if f['entity'] == eid), {})
        rows = [a for a in attempts if a['part'] == eid]
        holes = {p['bodyHole'] for p in frame.get('holePairs', [])} | {a['hole'] for a in rows}
        occupied.update(holes)
        confirmed = {a['hole'] for a in rows if a.get('seatConfirmed')}
        unattempted = holes-{a['hole'] for a in rows if a.get('releaseExecuted')}
        contradicted = {a['hole'] for a in rows if a.get('seatCheck', {}).get('passed') is False}
        installed.append(dict(unattemptedHoles=sorted(unattempted), failedHoles=sorted(contradicted), part=eid, released=bool(frame or parts[eid].get('releasedAt') is not None or
            any(a.get('releaseExecuted') for a in rows)), holes=sorted(holes),
            confirmedHoles=sorted(confirmed), remainingHoles=sorted(holes-confirmed),
            attachmentObserved=any(a.get('verified') for a in rows),
            allLandingsConfirmed=bool(holes) and holes <= confirmed))
    remaining = []
    if body:
        groups = {}
        for h in interface_holes(rt.esim, body):
            groups.setdefault(h.get('group', h['id']), []).append(h)
        for group, holes in groups.items():
            if all(h['id'] in occupied for h in holes):
                continue
            candidates = []
            for ent in scene.entities:
                if ent.id == body_id or ent.id in parts:
                    continue
                ph = interface_holes(rt.esim, ent)
                if any(h.get('group') in p.get('mates', ()) and p.get('group') in h.get('mates', ())
                       for h in holes for p in ph):
                    candidates.append(ent.id)
            remaining.append(dict(interface=group, holes=[h['id'] for h in holes],
                candidateParts=candidates, access=interface_access(rt.esim, body, {h['id'] for h in holes})))
    face = post.get('orientationCheck', ref.get('faceCheck', {}))
    orientation = dict(status=face.get('status', 'unknown'), check=deepcopy(face),
        **face_advice(face),
        upZEstimate=ref.get('commandedUpZ'), source=ref.get('source', 'perceived_outline'),
        observed=face.get('passed') is not None if face else False)
    held = rt.estimator.held_id
    pending = next((p for p in installed if p['unattemptedHoles'] and body and
                    interface_access(rt.esim, body, set(p['holes']))['status'] != 'inaccessible'), None)
    next_interface = next((r for r in remaining if r['candidateParts']), None)
    if held:
        action = dict(skill='place', reason='park_held_load_then_retry', target=held)
    elif post.get('placementReadiness', {}).get('passed') is False:
        action = dict(skill='observe', body=body_id, part=plan.get('part'),
            reason='restore_component_on_body_before_fastener_supply')
    elif plan.get('fastenSkill') == 'flip_assembly' and face.get('passed') is False:
        action = dict(skill='observe', reason='verify_face_before_retrying_flip_or_dependent_assembly', body=body_id)
    elif post.get('failure') and plan.get('part') and plan['part'] not in parts:
        action = dict(skill='fasten_onto', part=plan['part'], body=body_id,
                      reason='observe_part_and_reachable_station_then_retry')
    elif post.get('failedHoles') and post.get('interfaceSummary'):
        action = (dict(post['resumeSuggestion'], reason='resume_unreleased_or_same_screw_replay_once')
                  if post.get('resumeSuggestion') else dict(skill='observe', body=body_id,
                  reason='known_landing_failure_replay_exhausted', holes=post['failedHoles']))
    elif post.get('resumeSuggestion'):
        action = dict(post['resumeSuggestion'], reason='resume_only_unreleased_holes')
    elif pending:
        action = dict(skill='fasten_onto', part=pending['part'], body=body_id,
                      reason='resume_unreleased_holes_without_replacing_component')
    elif next_interface:
        access = next_interface['access']
        action = dict(skill='flip_assembly' if access['status']=='inaccessible' else
                      'fasten_onto',
                      body=body_id, part=next_interface['candidateParts'][0], reason=access['reason'])
        if access.get('faceAdvice'):
            action['observationSuggestion'] = access['faceAdvice']['observationSuggestion']
    else:
        action = dict(skill='observe', reason='continue_remaining_task_steps_and_check_final_orientation')
    from steward.ego import placement_support_margin
    saved = placement_support_margin.retained(records, body_id)
    if saved and not held:
        keep = placement_support_margin.guidance(saved)
        action = keep['nextAction']
        if pending:
            action.update(part=pending['part'], holes=pending['unattemptedHoles'])
        elif next_interface:
            action.update(part=next_interface['candidateParts'][0], holes=next_interface['holes'])
        orientation.update(nextAction='continue_remaining_interfaces_in_place',
            observationSuggestion=keep['guidance'])
    return dict(body=body_id, orientation=orientation, components=installed,
        remainingInterfaces=remaining, nextAction=action, retryable=True,
        guidance='Unknown is unobserved, not failed. Actions already executed may be followed by the next task step; do not end the task or repeatedly refill solely for unknown.',
        taskComplete=None, scope='observed_parts_and_catalogue_interfaces_only')
