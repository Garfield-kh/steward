"""Admission evidence for top-down mating, using robot-owned interface records."""
from steward.ego.assembly_status import interface_access
from steward.ego.fasten_plan import interface_holes, mating_holes, value


def assess(planner, body, selected, *, part=None, observed=None, occupancy=True, scene=None):
    selected = set(selected)
    info = value(body, 'orientation', {})
    ledger = info.get('stationaryAssembly', info.get('assembly', {}))
    ref = ledger.get('reference', {})
    evidence = ledger.get('registrationEvidence', {})
    check = ref.get('faceCheck') or {}
    after_flip = bool(check or 'rotationBefore' in ref or ref.get('source') == 'commanded_release')
    low = bool(evidence.get('confidence') == 'low' or evidence.get('lowConfidence'))
    access = interface_access(planner, body, selected, scene)
    # A batch cannot make a hidden hole reachable by also selecting an exposed
    # one. Reuse the same geometric test for each requested hole.
    if access['status'] == 'estimated_accessible' and any(
            interface_access(planner, body, {hole})['status'] == 'inaccessible' for hole in selected):
        access = dict(access, status='inaccessible', reason='selected_interface_on_opposite_face')
    holes = interface_holes(planner, body)
    levels = [float(h['p'][2]) for h in holes]
    mid = (min(levels)+max(levels))/2 if levels else None
    faces = sorted({'+z' if h['p'][2] > mid else '-z' if h['p'][2] < mid else 'unresolved'
                    for h in holes if h['id'] in selected})
    result = dict(status=access['status'], reason=access['reason'], holes=sorted(selected),
        catalogueFaces=faces, source='registered_interface_layers',
        orientationSource=ref.get('source', 'registered_rotation' if 'rotationM' in ref
                                  else 'initial_outline_catalogue_assumption'),
        lowConfidenceAfterFlip=after_flip and low,
        normalObserved=False)
    if 'bodyUpZEstimate' in access:
        result['bodyUpZEstimate'] = access['bodyUpZEstimate']
    if check:
        result['faceCheck'] = {k:check[k] for k in ('passed', 'reason') if k in check}
    uncertainty = []
    if access['status'] == 'unknown':
        uncertainty.append(access['reason'])
    if ledger.get('status') == 'invalidated':
        uncertainty.append('stationary_registration_invalidated')
    if info.get('posture', {}).get('status') == 'inconsistent':
        uncertainty.append('perceived_posture_inconsistent')
    # Initial outline confidence describes position/axis quality, not a failed
    # face reversal. After a flip the same low confidence belongs to a propagated
    # orientation hypothesis and must never become a hard face gate.
    if low and after_flip:
        uncertainty.append('low_confidence_registration')
    if after_flip and check.get('passed') is None:
        uncertainty.append('flip_face_unconfirmed')
    # Position at the pickup does not establish a face, even if motion failed.
    if check.get('reason') == 'body_still_at_pickup':
        uncertainty.append('position_only_flip_check')
    if selected - {h['id'] for h in holes}:
        uncertainty.append('selected_holes_missing_from_catalogue')
    if uncertainty:
        result.update(status='unknown', uncertaintySources=list(dict.fromkeys(uncertainty)))

    # The existing fresh RGB-D semantic face check outranks a nominal frame,
    # including when it resolves a previously unconfirmed flip.
    if observed and observed['status'] in ('satisfied', 'blocked'):
        result.update(status='estimated_accessible' if observed['status'] == 'satisfied' else 'inaccessible',
                      reason=observed['reasonCode'], orientationSource='fresh_rgbd_semantic_face',
                      observedAt=observed['observedAt'])
        result.pop('bodyUpZEstimate', None)
        result.pop('uncertaintySources', None)

    from steward.ego import working_face
    current_face = working_face.assess(planner, body, selected, scene)
    if current_face is not None:
        same_semantic_measurement = (result['orientationSource'] == 'fresh_rgbd_semantic_face'
            and current_face['evidenceSources'] == ['current_protrusion_and_catalogue_bearing'])
        result.update(status=current_face['status'], reason=current_face['reason'],
            currentFace=current_face, orientationSource=current_face['source'],
            normalObserved='current_registered_entrance_normal' in current_face['evidenceSources'])
        if current_face['faceObserved']:
            result.pop('uncertaintySources', None)
            result['bodyUpZEstimate'] = current_face['bodyUpZEstimate']
        else:
            result['uncertaintySources'] = [current_face['reason']]
        if same_semantic_measurement:
            result['orientationSource'] = 'fresh_rgbd_semantic_face'
            result.pop('bodyUpZEstimate', None)

    # An installed component's registered correspondence is positive occupancy
    # evidence. Its own through-hole remains usable for seating/resumption.
    occupied = sorted({f['entity'] for f in value(body, 'releasedInterfaces', [])
        if f['body'] == body.id and f['entity'] != part and
        selected & {p['bodyHole'] for p in f.get('holePairs', [])}})
    # Seating targets the released component's own through-hole; an id that was
    # re-identified since release must not turn that component into an obstruction.
    if occupied and not occupancy:
        result['occupancyAdvisory'] = occupied; occupied = []
    if occupied:
        result.update(status='inaccessible', reason='interface_occupied_by_other_component',
                      obstructingParts=occupied)
    result['nextAction'] = ('inspect_and_clear_interface_then_predict' if occupied else
        'flip_assembly_then_verify_face' if result['status'] == 'inaccessible' else
        'observe_body_face_then_predict' if result['status'] == 'unknown' else 'continue_selected_interface')
    if current_face is not None and not occupied and result['status'] == 'unknown':
        result['nextAction'] = 'assemble_current_upward_face_then_review'
    return result


def apply(receipt, access):
    """Keep uncertainty advisory and make both denial and evidence model-visible."""
    receipt['matingFace'] = access
    if access['status'] == 'inaccessible':
        receipt.update(feasible=False, risk=1.0)
        if access.get('obstructingParts'):
            issue = ('目标配合孔已被另一已登记上件占用；先检查并移开遮挡上件，使目标界面露出后重新 predict，'
                     '不要把新上件或紧固件叠放在遮挡件上。')
        else:
            issue = ('目标配合孔位于本体当前朝下的一面，不能从上方装配或入座；'
                     '先翻面（调用 flip_assembly 将该配合面翻到朝上），确认后重新 predict。')
        receipt['issues'].insert(0, issue)
    elif access['status'] == 'unknown':
        receipt['warnings'].append(dict(access, reason='mating_face_uncertain'))
        if access.get('nextAction') == 'assemble_current_upward_face_then_review':
            receipt['issues'].append('配合面未确认；保持本体，按当前朝上面试装，完成后复查补救。')
            return receipt
        receipt['issues'].append('配合面朝向不确定（' + ', '.join(access['uncertaintySources']) +
            '）；先观察本体侧面及已装件确认目标面朝向；本项仅警告，继续由现有风险门槛决定。')
    return receipt


def admit(rt, scene, skill, params, receipt):
    """One gate for model predictions and the staged controller's fresh receipts.

    Keep the geometric planner/release paths intact. A missing catalogue can
    still use a saved release target; it supplies uncertainty, not a new veto.
    """
    if skill not in ('assemble_onto', 'fasten_onto', 'seat_screw'):
        return receipt
    body = scene.find(params.get('body'))
    if body is None:
        return receipt
    from steward.ego.esim import PredictionError
    from steward.ego.fasten_release import released_interface
    from steward.ego.preconditions import face_check
    planner = rt.esim
    part = receipt.get('plan', {}).get('part', params.get('part'))
    frame = released_interface(scene, part, body.id, allow_alias=skill == 'fasten_onto'
                               and not params.get('drive', True))
    try:
        if skill == 'seat_screw':
            selected = {h['id'] for h in planner._select_holes(
                interface_holes(planner, body), params.get('hole'), body.id)}
        elif frame:
            part = frame['entity']
            selected = {p['bodyHole'] for p in frame['holePairs']}
        else:
            plan = {**params, **receipt.get('plan', {})}
            # A precondition denial has no plan yet; matching is read-only.
            selected = {h['id'] for h in mating_holes(planner, scene, plan)}
    except (PredictionError, KeyError):
        selector = params.get('hole')
        selected = set(selector if isinstance(selector, (list, tuple)) else
                       [str(selector)] if selector is not None else [])
    observed = face_check(rt, scene, part, body.id, dict(params, hole=sorted(selected))) if selected else None
    return apply(receipt, assess(planner, body, selected, part=part, observed=observed,
        occupancy=skill != 'seat_screw', scene=scene))
