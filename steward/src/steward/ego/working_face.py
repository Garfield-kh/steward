"""Current RGB-D decides the working face; catalogue yaw is not a face sensor.

No acquisition, episode state or simulator connection data. Missing evidence
retains the old turn policy except for a flat body overhanging its support.
"""
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
from steward.ego.fasten_plan import value


def interface_holes(planner, body):
    from steward.ego.fasten_plan import interface_holes as read
    return read(planner, body)


def enabled():
    from steward.ego import capabilities
    return capabilities.get('STEWARD_ASSEMBLY_WORKING_FACE') != '0'


def side(planner, body, selected):
    holes = interface_holes(planner, body)
    levels = [h['p'][2] for h in holes]
    if not levels or max(levels) == min(levels):
        return None
    middle = (max(levels)+min(levels))/2
    chosen = [h['p'][2]-middle for h in holes if h['id'] in selected]
    return (1 if min(chosen) > 0 else -1 if max(chosen) < 0 else None) if chosen else None


def view(entity, stamp):
    v = value(entity, 'contactView') or {}
    if (v.get('source') == 'rgbd_segment' and 0 <= stamp-v.get('stamp', -1) <= 2
            and not v.get('truncated', True) and v.get('invalidDepthFraction', 1) <= .05):
        return v
    return None


def regional_votes(planner, scene, body, bv, below_interfaces):
    """Recover merged attached layers from the already cached segmented cloud.

    Catalogue members are shape hypotheses only. A hypothesis votes only after
    two distinct current horizontal bands match touching body/member outlines.
    """
    points = np.asarray(bv.get('points', []), float)
    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 30:
        return []
    from steward.ego import session_faces as faces, session_inventory as inventory
    from steward.ego import session_observation as so, assembly_precision as ap
    proxy = SimpleNamespace(rt=SimpleNamespace(ego_state=lambda:scene, esim=planner),
                            ex=SimpleNamespace(config=getattr(planner, 'config', SimpleNamespace())))
    model_body = deepcopy(body)
    category = value(body, 'orientation', {}).get('holeRegistration', {}).get('templateCategory')
    if category:
        model_body.category.value = category
    spec = faces.model(proxy, model_body)
    items, models = inventory.hypotheses(proxy, model_body)
    margin = ap.tolerance(proxy)
    bands = faces.bands(points, margin)
    votes = []
    for bi, band in enumerate(bands):
        base = faces.fit_band(band, spec, margin, bv['centreM'])
        if base is None:
            continue
        for pi, other in enumerate(bands):
            if pi == bi:
                continue
            matches = []
            for item in items:
                part_spec = faces.model(proxy, models[item['part']])
                member = faces.fit_band(other, part_spec, margin, base['position'], base['yaw'])
                if member is None:
                    continue
                on, exposed, _ = so.stacked(proxy, member, base)
                delta = member['position'][:2]-base['position'][:2]
                c,s = np.cos(base['yaw']), np.sin(base['yaw'])
                local = delta @ np.array([[c,-s],[s,c]])
                if on is True and exposed is not None and np.all(abs(local) <= base['size'][:2]/2+margin):
                    hs = side(planner, body, set(item['holes']))
                    if hs:
                        matches.append(hs*(1 if exposed else -1))
                        if exposed is False:
                            below_interfaces.append(set(item['holes']))
            if matches and len(set(matches)) == 1:
                votes.append(matches[0])
    return votes


def edge_context(planner, scene, body):
    """Scope missing-face fallback to a body already at the support boundary.

    Interior from-zero assembly retains its existing phase-one/two contract.
    The test uses current perceived footprints, so it survives ledger loss.
    """
    if scene is None:
        return False
    from steward.ego.fasten_plan import support_anchor
    from steward.ego.planner import entity_box
    try:
        support = support_anchor(planner, scene, body, [])
        table, _, table_top = entity_box(support)
        box, body_low, body_top = entity_box(body)
    except (AttributeError, KeyError, TypeError, ValueError):
        return False  # No located support boundary; not an identified third phase.
    c,s = np.cos(table.yaw), np.sin(table.yaw)
    centre = (np.array([box.cx,box.cy])-[table.cx,table.cy]) @ np.array([[c,-s],[s,c]])
    angle = box.yaw-table.yaw
    extent = abs(np.array([[np.cos(angle),-np.sin(angle)],
                           [np.sin(angle),np.cos(angle)]])) @ [box.hx,box.hy]
    margin = np.array([table.hx,table.hy])-abs(centre)-extent
    # A historic keep flag and proximity to an edge are not observations of
    # overhang. Use the current perceived envelope, including its thickness.
    flat = body_top-body_low <= min(2*box.hx, 2*box.hy)
    return bool(flat and body_top >= table_top and np.any(margin < 0)
                and np.all(abs(centre)-extent < [table.hx,table.hy]))


def assess(planner, body, selected, scene=None):
    """Resolve absolute side from independent current layers or shape semantics."""
    if not enabled() or body is None or not selected:
        return None
    target = side(planner, body, selected)
    if target is None:
        return None
    stamp = scene.stamp if scene is not None else body.pose.t
    if stamp is None:
        return None
    bv = view(body, stamp)
    votes, sources, below_members, below_interfaces = [], [], [], []
    target_entrance_up = False
    below_sides = set()
    if bv:
        from steward.ego.fasten_geometry import vertical_signature
        from steward.ego.classify import SHAPES
        category = value(body, 'orientation', {}).get('holeRegistration', {}).get('templateCategory', body.category.value)
        shape = SHAPES.get(category, {})
        dims, bearing = shape.get('dims'), shape.get('bearing_height')
        signature = vertical_signature(bv) if bv.get('points') is not None else None
        if dims and bearing is not None and 0 < bearing < dims[2] and signature and 'visibleSizeM' in bv:
            slack = float(shape.get('extent_slack', 0))+3*float(bv.get('uncertaintyM', 0))
            if np.all(abs(np.sort(bv['visibleSizeM'])-np.sort(dims)) <= slack):
                votes.append(1 if signature == 'detail_above' else -1)
                sources.append('current_protrusion_and_catalogue_bearing')
        # A measured, catalogue-associated entrance normal can establish its
        # own face. Prior hole locations never qualify as visible entrances.
        for h in interface_holes(planner, body):
            if (h.get('source') != 'observed' or h.get('normalFrame') != 'world'
                    or h.get('observedAt') != bv['stamp']):
                continue
            normal = np.asarray(h.get('normal', []), float)
            hs = side(planner, body, {h['id']})
            if hs and normal.shape == (3,) and abs(normal[2]) > np.linalg.norm(normal[:2]):
                votes.append(hs*(1 if normal[2] > 0 else -1))
                sources.append('current_registered_entrance_normal')
                if h['id'] in selected and normal[2] > 0:
                    target_entrance_up = True
    if scene is not None and bv and all(k in bv for k in ('centreM', 'visibleSizeM')):
        from steward.ego import session_observation as so
        from steward.ego.planner import entity_box
        proxy = SimpleNamespace(rt=SimpleNamespace(ego_state=lambda:scene),
            ex=SimpleNamespace(config=getattr(planner, 'config', SimpleNamespace())))
        base = so.layer(proxy, body)
        catalogue = interface_holes(planner, body)
        for part in scene.entities:
            if part.id == body.id or not view(part, stamp):
                continue
            ph = interface_holes(planner, part)
            matched = {h['id'] for h in catalogue if any(p.get('group') in h.get('mates', []) for p in ph)}
            ps = side(planner, body, matched) if matched else None
            if ps is None:
                continue
            upper = so.layer(proxy, part)
            if base is None or upper is None or base['stamp'] != upper['stamp']:
                continue
            on, exposed, _ = so.stacked(proxy, upper, base)
            if on is True and exposed is not None:
                votes.append(ps*(1 if exposed else -1))
                sources.append('current_independent_member_height_and_catalogue')
                if exposed is False:
                    below_sides.add(ps)
                    if set(selected) <= matched:
                        below_members.append(part.id)
        if not votes:
            recovered = regional_votes(planner, scene, body, bv, below_interfaces)
            votes.extend(recovered)
            if recovered:
                sources.append('current_merged_cloud_touching_catalogue_layers')
    up = votes[0] if votes and len(set(votes)) == 1 else None
    at_edge = edge_context(planner, scene, body)
    below_sides.update(side(planner, body, ids) for ids in below_interfaces)
    positive_up = target_entrance_up or (at_edge and -target in below_sides)
    # Shape/height hypotheses can reject a down-facing target. They cannot
    # authorize an upward placement without a visible target entrance or an
    # actually observed supporting member under an overhanging flat body.
    if up is not None and target*up > 0 and not positive_up:
        up = None
    # Only the opposed lower layer changes the old initial-interface fallback.
    # Measured evidence, however, wins for either face.
    if up is None and (target > 0 or not at_edge):
        return None
    exposed = target*up > 0 if up is not None else None
    return dict(status='unknown' if exposed is None else 'estimated_accessible' if exposed else 'inaccessible',
        reason='current_face_unresolved_try_in_place' if exposed is None else
               'remaining_interface_already_up' if exposed else 'current_remaining_interface_down',
        source='current_rgbd_and_catalogue', faceObserved=up is not None,
        bodyUpZEstimate=up, observedAt=bv['stamp'] if bv else None,
        evidenceSources=sorted(set(sources)), unconfirmed=up is None,
        retentionEvidence=('visible_target_entrance' if exposed and target_entrance_up else
            'current_underbody_member_at_edge' if exposed else
            'flat_overhang_face_unresolved' if exposed is None else None),
        preserveExisting=bool(at_edge and exposed is False and (below_members or
            any(set(selected) <= holes for holes in below_interfaces))),
        currentMembersBelow=below_members,
        currentInterfacesBelow=[sorted(holes) for holes in below_interfaces],
        nextAction='flip_assembly_then_verify_face' if exposed is False else 'assemble_onto',
        extraObservations=0, bodyRetained=exposed is not False)


def selected_for_flip(planner, scene, body, params):
    selector = params.get('hole', params.get('selectedHoleIds'))
    if selector:
        return {h['id'] for h in planner._select_holes(interface_holes(planner, body), selector, body.id)}
    from steward.ego.assembly_session import current
    session = current()
    if session:
        items = session.state.get('goal', {}).get('interfaces', [])
        active = [i for i in items if i.get('key') == session.state.get('interface')
                  and session.canonical(i['body']) == body.id]
        if active:
            return set(active[0]['holes'])
    occupied = {p['bodyHole'] for f in value(body, 'releasedInterfaces', []) for p in f.get('holePairs', [])}
    pending = {h['id'] for h in interface_holes(planner, body)}-occupied
    if side(planner, body, pending) is not None:
        return pending
    # With no action history, match loose perceived members to the catalogue.
    candidates = []
    for part in scene.entities:
        if part.id == body.id:
            continue
        ph = interface_holes(planner, part)
        ids = {h['id'] for h in interface_holes(planner, body)
               if any(p.get('group') in h.get('mates', []) for p in ph)}
        if ids and side(planner, body, ids) is not None:
            from steward.ego.planner import entity_box, rectangle_separation
            if rectangle_separation(entity_box(body)[0], entity_box(part)[0]) > 0:
                candidates.append(ids)
    return candidates[0] if candidates and all(c == candidates[0] for c in candidates) else set()


def flip_guard(planner, scene, params):
    if not enabled() or params.get('turnOver') is False or params.get('_edgeAdjusted'):
        return None
    body = scene.find(params.get('body'))
    if body is None:
        return None
    selected = selected_for_flip(planner, scene, body, params)
    evidence = assess(planner, body, selected, scene)
    if evidence is None or evidence['status'] == 'inaccessible' and not evidence.get('preserveExisting'):
        return None
    if evidence.get('preserveExisting'):
        selected = {h['id'] for h in interface_holes(planner, body)
                    if side(planner, body, {h['id']}) != side(planner, body, selected)}
    return dict(evidence, flipSkipped=True, motionAttempted=False, canContinue=True,
        nextAction=dict(skill='fasten_onto', body=body.id, hole=sorted(selected)),
        guidance=('已处于可装配姿态，无需翻面；下一步在原位放上件并逐孔放入紧固件。'
                  if evidence['faceObserved'] else
                  '当前朝向未确认，保持本体；下一步按当前朝上面试装上件并逐孔放入紧固件，完成后复查补救。'))


def prepare(controller, body_id, selected):
    """Install a working frame, with an explicit estimate when vision is absent."""
    if not enabled():
        return None
    scene = controller.rt.ego_state(); body = scene.find(body_id)
    result = assess(controller.rt.esim, body, set(selected), scene)
    if result is None or result['status'] == 'inaccessible':
        return result
    records = getattr(getattr(controller.rt.estimator, 'stationary_assemblies', None), 'records', {})
    record = records.get(body_id)
    if record:
        from steward.math3d import matrix_from_quat, quat_from_matrix
        ref = record['reference']
        up = result['bodyUpZEstimate'] or side(controller.rt.esim, body, set(selected))
        rotation = np.asarray(ref.get('rotationM', matrix_from_quat(record['entity'].pose.value['q'])))
        if rotation[2, 2]*up < 0:
            rotation = rotation @ np.diag([1., -1., -1.])
        ref['rotationM'] = rotation.tolist()
        record['entity'].pose.value['q'] = quat_from_matrix(rotation)
        ref['workingFaceEvidence'] = dict(result, selectedHoles=sorted(selected))
        # This is a working estimate, never promoted into a visual normal.
        record['workingFaceKeepBody'] = True
        controller.rt.estimator.version += 1
    return result


def ordinary_path(planner, scene, body, selected):
    if not enabled() or body is None or side(planner, body, set(selected)) != -1:
        return False
    evidence = assess(planner, body, set(selected), scene)
    return evidence is not None and evidence['status'] != 'inaccessible'


def preserve_existing(session, item):
    if not enabled():
        return False
    try:
        scene = session.rt.ego_state()
        body = scene.find(session.canonical(item['body']))
        evidence = assess(session.rt.esim, body, set(item['holes']), scene)
    except (AttributeError, KeyError, TypeError, ValueError):
        return False
    if evidence and evidence.get('preserveExisting') and edge_context(session.rt.esim, scene, body):
        session.state.setdefault('workingFacePreservedInterfaces', {})[item['key']] = evidence
        return True
    return False


def support_report(planner, scene, body, points):
    from steward.ego.fasten_plan import support_anchor
    from steward.ego.planner import entity_box
    support = support_anchor(planner, scene, body, [])
    box = entity_box(support)[0]
    c,s = np.cos(box.yaw),np.sin(box.yaw)
    local = (np.asarray(points)[:, :2]-[box.cx,box.cy]) @ np.array([[c,-s],[s,c]])
    margins = np.min([box.hx,box.hy]-abs(local), axis=1).tolist()
    from steward.ego.edge_geometry import force_report
    ledger = value(body, 'orientation', {}).get('stationaryAssembly', {})
    bearing = force_report(ledger, points) if ledger.get('reference') else None
    if bearing and bearing.get('marginM') is not None:
        margins.append(bearing['marginM'])
    return dict(direction=[0.,0.,-1.], inside=bool(np.all(np.asarray(margins) >= 0)),
        marginM=float(min(margins)), source='perceived_support_and_registered_entries',
        observed=False, bodyMoved=False)


def placement_plan(planner, scene, plan, params):
    """Constrain only the opposed-face placement, before the receipt is issued."""
    if not enabled():
        return
    # alignTo alone is an XY reference for placement on a separate support.
    # Face admission applies to the actual receiving body or seating target.
    body = scene.find(params.get('seatBody') or params.get('on'))
    if body is None:
        return
    selector = params.get('seatHole', params.get('hole'))
    if selector is None:
        selector = [p['bodyHole'] for p in plan.get('placementTarget', {}).get('holePairs', [])]
    if not selector:
        return
    holes = interface_holes(planner, body)
    if len({h['p'][2] for h in holes}) < 2:
        return  # A support table or single-face placement is outside this phase.
    selected = planner._select_holes(holes, selector, body.id)
    ids = {h['id'] for h in selected}
    from steward.ego.assembly_status import interface_access
    # Preserve standalone geometric fitting. A second-component action has an
    # actual prior release or a live assembly owner; current measured down-face
    # evidence also applies without history (e.g. a resumed physical assembly).
    from steward.ego.assembly_session import current
    session = current()
    owned = bool(session and any(session.canonical(i['body']) == body.id
        for i in session.state.get('goal', {}).get('interfaces', [])))
    target = side(planner, body, ids)
    prior = any(side(planner, body, {p['bodyHole'] for p in frame.get('holePairs', [])}) == -target
        for frame in value(body, 'releasedInterfaces', [])) if target else False
    observed = assess(planner, body, ids, scene)
    access = interface_access(planner, body, ids, scene) if owned or prior else observed
    if access and access['status'] == 'inaccessible':
        from steward.ego.esim import PredictionError
        raise PredictionError('目标接口位于当前朝下的一面；先翻面（flip_assembly），再放置上件或紧固件。')
    if not ordinary_path(planner, scene, body, ids):
        return
    point = plan.get('seatTarget', {}).get('point', plan['spot'])
    force = support_report(planner, scene, body, [point])
    plan['supportForce'] = force
    if not force['inside']:
        from steward.ego.esim import PredictionError
        raise PredictionError('current placement load point is outside perceived support; keep body in place')
    opening = planner.arm.geometry.jaw_opening
    clearance = opening/4 if opening else getattr(planner.config, 'interface_precision_m', .0035)
    plan.update(workingFaceStationaryPlacement=True, stayNear=True, gentle=True,
                edgeLowRelease=True, edgeDescentM=clearance,
                edgeVerticalClearanceM=min(clearance, max(0., planner.lift_max-plan['lift'])))
    # XY refinements belong at the existing hover; do not push sideways after
    # descending to the working face. Normal precision callbacks remain shared.
    plan.pop('refineReleaseXY', None)


@contextmanager
def placement(controller, job, plan, params):
    """Same ordinary placement, vertical final descent and cached drift receipt."""
    if not enabled():
        yield
        return
    body_id = params.get('seatBody') or params.get('on')
    scene = controller.rt.ego_state()
    body = scene.find(body_id)
    selected = params.get('seatHole') or params.get('hole')
    selected = [selected] if isinstance(selected, str) else selected or []
    if not ordinary_path(controller.rt.esim, scene, body, selected):
        yield
        return
    before = deepcopy(value(body, 'contactView'))
    position = list(body.pose.value['p'])
    # The admitted ordinary place plan already contains the vertical approach.
    force = plan.get('supportForce') or support_report(controller.rt.esim, scene, body, [plan['spot']])
    job.postcondition['supportForce'] = force
    try:
        yield
    finally:
        after_scene = controller.rt.ego_state(); after = after_scene.find(body_id)
        av = value(after, 'contactView') if after is not None else None
        fresh = bool(before and av and av.get('stamp', -1) > before.get('stamp', -1)
                     and av.get('source') == before.get('source') == 'rgbd_segment')
        delta = (np.asarray(av['centreM'])-before['centreM'] if fresh else
                 np.asarray(after.pose.value['p'])-position if after is not None else None)
        report = dict(deltaM=delta.tolist() if delta is not None else None,
            source='independent_rgbd_centres' if fresh else 'perceived_pose_estimate',
            observed=fresh, unconfirmed=not fresh, extraObservations=0)
        plan['bodyPositionChange'] = report
        job.postcondition['bodyPositionChange'] = report
