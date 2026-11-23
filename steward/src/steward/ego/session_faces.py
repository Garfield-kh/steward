"""Resolve the observed stack's side BEFORE assigning entry identities.

Catalogue faces describe correspondence, never world orientation. Search current
depth around the bound body, rather than requiring a previously placed component
pose. No observation of the exposed side is evidence for the hidden side.
"""
from copy import deepcopy
import math
import numpy as np

from steward.ego.fasten_plan import interface_holes
from steward.ego import session_observation as current
from steward.ego.esim import PredictionError
from steward.ego.skills import SkillFailed


def enabled():
    from steward.ego import capabilities
    return current.enabled() and capabilities.get('STEWARD_ASSEMBLY_SESSION_FACES') != '0'


def model(controller, entity):
    """Existing robot catalogue, with perceived geometry only as a fallback."""
    from steward.ego.priors import SCREW_HOLE_FRAMES, screw_holes
    from steward.ego.classify import SHAPES
    p, size, yaw = current.prior_box(entity)
    template = SCREW_HOLE_FRAMES.get(entity.category.value)
    holes = screw_holes(entity.category.value)
    shape = SHAPES.get(entity.category.value, {})
    if template and holes:
        xy = np.asarray(template['sizeXYM'], float)
        thickness = shape.get('bearing_height', max(h['p'][2] for h in holes))
        if thickness <= 0:
            thickness = shape.get('dims', size)[2]
        return dict(size=np.r_[xy, thickness], holes=holes, yaw=yaw,
                    slack=shape.get('extent_slack', 0), centre=p)
    return dict(size=size, holes=interface_holes(controller.rt.esim, entity),
                yaw=yaw, slack=0., centre=p)


def sides(controller, body, items):
    catalogue = model(controller, body)['holes']
    levels = [h['p'][2] for h in catalogue]
    if not levels or max(levels) == min(levels):
        return {}
    mid = (max(levels)+min(levels))/2
    result = {}
    for item in items:
        selected = [h['p'][2]-mid for h in catalogue if h['id'] in item['holes']]
        if selected and (all(z > 0 for z in selected) or all(z < 0 for z in selected)):
            result[item['key']] = 1 if selected[0] > 0 else -1
    return result


def bands(points, margin):
    """Horizontal visible surfaces, including disconnected exposed overhangs."""
    if len(points) < 30:
        return []
    bins = np.floor(points[:, 2]/margin).astype(int)
    keys, counts = np.unique(bins, return_counts=True)
    rows = []
    for key in keys[np.argsort(counts)[::-1]]:
        z = float(np.median(points[bins == key, 2]))
        if any(abs(z-r['top']) <= margin for r in rows):
            continue
        band = points[abs(points[:, 2]-z) <= margin]
        if len(band) < 30:
            continue
        centre = np.mean(band[:, :2], axis=0)
        design = np.c_[band[:, :2]-centre, np.ones(len(band))]
        if np.linalg.matrix_rank(design) < 3:
            continue
        fit = np.linalg.lstsq(design, band[:, 2], rcond=None)[0]
        spread = max(np.ptp(band[:, :2], axis=0))
        if np.linalg.norm(fit[:2])*spread > margin:
            continue
        rows.append(dict(points=band, top=float(np.median(band[:, 2]))))
    return rows


def _fit_outline(band, spec, margin):
    """Match an outline without assuming its directed yaw or its face."""
    from steward.ego.hole_registration import register_hole_template
    points = band['points']
    cov = np.cov(points[:, :2].T)
    axis = np.linalg.eigh(cov)[1][:, -1]
    yaw = math.atan2(axis[1], axis[0])
    rotation = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    local = points[:, :2] @ rotation
    lo, hi = np.quantile(local, [.02, .98], axis=0)
    size = hi-lo
    template = dict(sizeXYM=spec['size'][:2], centerXYM=[0., 0.])
    _, registration = register_hole_template(spec['holes'], template, size,
        extent_slack_m=spec['slack']+margin)
    if registration['status'] != 'axis_registered':
        return None
    error = float(np.linalg.norm(np.sort(size)-np.sort(spec['size'][:2])))
    centre = rotation @ ((hi+lo)/2)
    return dict(position=np.r_[centre, band['top']-spec['size'][2]/2],
        size=np.r_[size, spec['size'][2]], yaw=yaw, top=band['top'],
        points=points, independent=True, source='current_body_region_depth',
        outlineErrorM=error, contactEstimated=True)


def fit_band(band, spec, margin, anchor=None, yaw=None):
    """Remove disjoint coplanar clutter without turning crop edges into outlines."""
    candidates = [_fit_outline(band, spec, margin)]
    if anchor is not None:
        angle = spec['yaw'] if yaw is None else yaw
        half = spec['size'][:2]/2+2*(spec['slack']+margin)
        for turn in (angle, angle+math.pi/2):
            c,s = math.cos(turn), math.sin(turn)
            rotation = np.array([[c,-s],[s,c]])
            local = (band['points'][:, :2]-np.asarray(anchor)[:2]) @ rotation
            keep = np.all(abs(local) < half, axis=1)
            if keep.sum() < 30 or keep.all():
                continue
            # A clipped broad plane is not a component-sized outline. Require
            # a measured end to the returns inside every search boundary.
            if np.any(np.max(abs(local[keep]), axis=0) >= half-margin):
                continue
            candidates.append(_fit_outline(dict(band, points=band['points'][keep]), spec, margin))
    return min((r for r in candidates if r is not None),
               key=lambda r:r['outlineErrorM'], default=None)


def regional_layers(controller, body, items, samples):
    """Search each current body region jointly; loose bound poses are not gates."""
    from steward.ego import wrist_observation as wo
    from steward.ego.assembly_precision import tolerance
    scene = controller.rt.ego_state()
    members = {i['key']:scene.find(i['part']) for i in items}
    base = model(controller, body)
    specs = {k:model(controller, e) for k,e in members.items() if e is not None}
    margin = tolerance(controller)
    radius = max([np.linalg.norm(base['size']), *[np.linalg.norm(s['size']) for s in specs.values()]])+margin
    best, best_score = {}, None
    for raw, cam in samples:
        if 'depth' not in raw:
            continue
        points, _, _ = wo.cloud(raw, cam, wo.window(raw, cam, base['centre'], radius))
        keep = np.isfinite(points).all(axis=1)
        keep &= np.linalg.norm(points[:, :2]-base['centre'][:2], axis=1) <= radius
        keep &= abs(points[:, 2]-base['centre'][2]) <= sum(s['size'][2] for s in specs.values())+base['size'][2]+margin
        try:
            from steward.ego.planner import entity_box
            support = controller.rt.esim._anchor(scene, body)
            if support.id != body.id:
                keep &= points[:, 2] > entity_box(support)[2]+margin
        except (AttributeError, KeyError, TypeError, ValueError):
            pass
        surfaces = bands(points[keep], margin)
        for bi, surface in enumerate(surfaces):
            lower = fit_band(surface, base, margin, base['centre'])
            if lower is None:
                continue
            # Reject a nearby unrelated object that merely resembles the body.
            if np.linalg.norm(lower['position'][:2]-base['centre'][:2]) > min(base['size'][:2])+margin:
                continue
            candidates = []
            for key, spec in specs.items():
                for pi, plane in enumerate(surfaces):
                    if pi == bi:
                        continue
                    upper = fit_band(plane, spec, margin, lower['position'], lower['yaw'])
                    if upper is None:
                        continue
                    on, exposed, reason = current.stacked(controller, upper, lower)
                    if on is not True:
                        continue
                    # Contact requires the component centre over the body, not
                    # merely two neighbouring footprints touching at an edge.
                    delta = upper['position'][:2]-lower['position'][:2]
                    c,s = math.cos(lower['yaw']), math.sin(lower['yaw'])
                    local = delta @ np.array([[c,-s],[s,c]])
                    if np.any(abs(local) > lower['size'][:2]/2+margin):
                        continue
                    candidates.append((upper['outlineErrorM'], key, pi, upper, exposed))
            found, used = {}, set()
            for error, key, pi, upper, exposed in sorted(candidates, key=lambda x:x[0]):
                if key in found or pi in used:
                    continue
                if any(other != key and plane == pi and abs(cost-error) <= margin and
                       np.all(abs(np.sort(specs[other]['size'][:2])-np.sort(specs[key]['size'][:2])) <= margin)
                       for cost,other,plane,_,_ in candidates):
                    continue  # Equal silhouettes do not identify a catalogue face.
                used.add(pi)
                upper['stamp'] = raw['header']['stamp']
                found[key] = upper, exposed
            score = (-len(found), lower['outlineErrorM']+sum(r[0]['outlineErrorM'] for r in found.values()))
            if found and (best_score is None or score < best_score):
                best, best_score = found, score
    return best


def prepare(controller, items, component_reader, samples):
    reports = {}
    for item in items:
        try:
            c, frame = component_reader(controller, item)
        except (AttributeError, KeyError, TypeError, ValueError, np.linalg.LinAlgError, PredictionError, SkillFailed):
            c, frame = dict(onBody=None, source='unavailable', reason='current_stack_unavailable'), None
        reports[item['key']] = dict(component=c, frame=frame,
            holes={h:dict(occupied=None, confirmed=False, source='unavailable') for h in item['holes']})
    for body_id in {i['body'] for i in items}:
        group = [i for i in items if i['body'] == body_id]
        body = controller.rt.ego_state().find(body_id)
        if body is None:
            continue
        try:
            side = sides(controller, body, group)
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
        if not side:
            continue
        if not all(reports[i['key']]['component'].get('faceObserved') and
                   reports[i['key']]['component'].get('onBody') is True for i in group):
            try:
                recovered = regional_layers(controller, body, group, samples)
            except (AttributeError, KeyError, TypeError, ValueError, np.linalg.LinAlgError):
                recovered = {}
            for item in group:
                if item['key'] not in recovered:
                    continue
                upper, exposed = recovered[item['key']]
                try:
                    frame = make_frame(controller, item, upper)
                except (AttributeError, KeyError, TypeError, ValueError):
                    # Classification/tracks may change while a native frame is
                    # being read. A missing model is unavailable evidence, not
                    # a reason to abort the entire interface initialization.
                    continue
                reports[item['key']].update(component=dict(onBody=True, confirmed=False,
                    source='current_rgbd_stack', reason='body_region_outline_and_touching_surface',
                    observedAt=upper['stamp'], faceExposed=exposed, faceObserved=True,
                    contactConfirmed=False, bestEstimate=True,
                    layerSources=[upper['source']], outlineErrorM=upper['outlineErrorM']),
                    frame=frame)
        votes = [side[k]*(1 if reports[k]['component']['faceExposed'] else -1) for k in side
                 if reports[k]['component'].get('faceObserved') and
                 reports[k]['component'].get('faceExposed') is not None and
                 reports[k]['component'].get('onBody') is not False]
        evidence = [dict(interface=k, catalogueSide=side[k],
                         observedSide=reports[k]['component'].get('faceExposed'),
                         source=reports[k]['component']['source'],
                         observedAt=reports[k]['component'].get('observedAt')) for k in side
                    if reports[k]['component'].get('faceObserved')]
        up = votes[0] if votes and len(set(votes)) == 1 else None
        for item in group:
            if item['key'] not in side:
                continue
            row = reports[item['key']]
            exposed = side[item['key']]*up > 0 if up is not None else None
            row['face'] = dict(exposed=exposed, bodyUpZEstimate=up,
                source='current_stack_side_and_catalogue' if up is not None else 'current_face_unresolved',
                confirmed=up is not None, resolvedBeforeHoles=True,
                votes=deepcopy(evidence))
            row['component']['faceExposed'] = exposed
            row['holeMapping'] = dict(method='catalogue_face_then_visible_component_layout',
                selectedHoles=list(item['holes']), faceExposed=exposed,
                acceptedVisibleHoles=[], observedEntries=[], rejectedReason=None if exposed is True else
                'selected_face_hidden' if exposed is False else 'body_face_unresolved')
    from steward.ego import session_inventory
    if session_inventory.enabled():
        inventory = getattr(controller, '_session_inventory_context', None)
        if inventory is None:
            # Outcome/re-entry observers also need the layer reference when
            # segmentation still has no independent member track.
            from steward.ego.assembly_session import current as active_session
            session = active_session()
            if (session is not None and getattr(session, 'controller', None) is controller and
                    session.state.get('stackInventory')):
                inventory = session_inventory.prepare(session, session.state['goal']['interfaces'])
        reports.update({k:deepcopy(v) for k,v in (inventory or {}).items() if k in reports})
    return reports


def make_frame(controller, item, layer):
    """Observed layer frame; catalogue holes carry correspondence, not pose."""
    from steward.ego.hole_registration import register_hole_template
    scene = controller.rt.ego_state()
    part = scene.find(item['part'])
    spec = model(controller, part)
    catalogue, _ = register_hole_template(spec['holes'],
        dict(sizeXYM=spec['size'][:2], centerXYM=[0., 0.]), layer['size'], best_effort=True)
    c,s = math.cos(layer['yaw']), math.sin(layer['yaw'])
    rotation = np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
    holes = [deepcopy(h) for h in catalogue if h['id'] in item['heldHoles']]
    top = max(h['p'][2] for h in holes)
    for h in holes:
        local = np.asarray(h['p'], float)
        local[2] -= top-spec['size'][2]/2
        h['p'] = local.tolist()
        h['point'] = (layer['position']+rotation @ local).tolist()
    return dict(entity=part.id, body=item['body'], category=part.category.value,
        bodyPoseAtObservation=deepcopy(scene.find(item['body']).pose.value),
        position=layer['position'].tolist(), size=layer['size'].tolist(), rotation=rotation.tolist(), holes=holes,
        holePairs=[dict(partHole=p,bodyHole=b) for p,b in zip(item['heldHoles'],item['holes'])],
        observedAt=layer['stamp'], releasedAt=layer['stamp'], preExisting=True, observed=False,
        source='initial_current_rgbd_stack', heightSource='current_body_region_depth')


def observe_all(controller, items, observer, component_reader, sample_reader):
    """Keep the existing observer hook, but supply a jointly resolved context."""
    if not enabled():
        return {i['key']:observer(controller, i) for i in items}
    previous = getattr(controller, '_session_face_context', None)
    try:
        try:
            samples = list(sample_reader(controller))
        except (AttributeError, KeyError, TypeError, ValueError):
            samples = []
        controller._session_face_context = prepare(controller, items, component_reader, samples)
        return {i['key']:observer(controller, i) for i in items}
    finally:
        if previous is None:
            controller.__dict__.pop('_session_face_context', None)
        else:
            controller._session_face_context = previous


def priority(session, item):
    row = session.state.get('initialInterfaces', {}).get(item['key'], {})
    return (0 if row.get('status') == 'ready_to_fasten' else
            1 if row.get('status') == 'partial' and row.get('component', {}).get('onBody') else 2,
            session.access(item)['status'] == 'inaccessible')


def protected(row):
    return bool(enabled() and row and row.get('preserveExisting') and
                row['component'].get('onBody') is True and
                row.get('face', {}).get('exposed') is False)


def hidden_stack(row):
    return bool(enabled() and row.get('face', {}).get('exposed') is False and
                row['component'].get('onBody') is not False and row.get('frame'))


def finish(row):
    """Hidden attachment is preserved, without inventing occupied-hole evidence."""
    if row.get('face', {}).get('resolvedBeforeHoles'):
        # A late pose-only overlap estimate cannot overrule the current face
        # solution (or manufacture one when the solution remained unknown).
        row['component']['faceExposed'] = row['face']['exposed']
    if row.get('face', {}).get('resolvedBeforeHoles') and row['face'].get('exposed') is False:
        if row['component'].get('onBody') is True:
            row['preserveExisting'] = True
        for h in row['holes'].values():
            h.update(occupied=None, confirmed=False, source='unavailable',
                     reason='selected_face_hidden_preserve_structure_unconfirmed')
        return True
    return False


def use_execution_face(row, body):
    """A bare/separate start retains the ordinary placement/flip lifecycle."""
    if not enabled() or row.get('face', {}).get('exposed') is not None:
        return False
    if row['component'].get('onBody') is False:
        return True
    from steward.ego.assembly_session import current as active_session
    session = active_session()
    if session is None:
        return False
    return any(f.get('body') == body and not f.get('preExisting')
               for f in getattr(session.rt.estimator, 'released_interfaces', {}).values())


def retain_face(session, item, before, now):
    """A missing re-entry view does not erase a side previously observed."""
    face = before.get('face', {})
    cutoff = session.state.get('initialFaceInvalidatedAt', {}).get(session.canonical(item['body']), -math.inf)
    if (not enabled() or not face.get('resolvedBeforeHoles') or face.get('exposed') is None or
            now.get('face', {}).get('exposed') is not None or
            now['component'].get('onBody') is False or
            cutoff >= before['component'].get('observedAt', -1)):
        return
    now['face'] = dict(deepcopy(face), confirmed=False, source='retained_current_stack_face_estimate')
    now['component']['faceExposed'] = face['exposed']
    if now.get('holeMapping'):
        now['holeMapping'].update(faceExposed=face['exposed'],
            rejectedReason='selected_face_hidden' if face['exposed'] is False else 'current_entries_unavailable')
