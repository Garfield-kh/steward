"""Initial assembly geometry from robot RGB-D, independent of release history.

Entity poses delimit search windows, not observations. A native depth return or
an independent segmented layer supplies the measurement. Unresolved overlapping
layers preserve the existing structure as an explicitly unconfirmed estimate.
"""
from copy import deepcopy
import numpy as np

from steward.ego.fasten_plan import value, interface_holes
from steward.math3d import matrix_from_quat


def enabled():
    from steward.ego import capabilities
    return capabilities.get('STEWARD_ASSEMBLY_SESSION_OBSERVATION') != '0'


def prior_box(entity):
    from steward.ego.planner import entity_box
    box, low, high = entity_box(entity)
    return (np.asarray(entity.pose.value['p'], float),
            np.asarray([2*box.hx, 2*box.hy, high-low]), float(box.yaw))


def layer(controller, entity, sample=None):
    """Read an independent mask or native points in a bound member's window."""
    from steward.ego.assembly_precision import tolerance
    from steward.ego import wrist_observation as wo
    now = controller.rt.ego_state().stamp
    view = value(entity, 'contactView') or {}
    if (view.get('source') == 'rgbd_segment' and 0 <= now-view.get('stamp', -1) <= 2
            and not view.get('truncated', False) and view.get('invalidDepthFraction', 0) <= .05):
        return dict(position=np.asarray(view['centreM']), size=np.asarray(view['visibleSizeM']),
            yaw=view.get('boxYawRad', 0), stamp=view['stamp'], points=np.asarray(view.get('points', [])),
            source='current_rgbd_layers', independent=True)
    if sample is None:
        return None
    raw, cam = sample
    p, size, yaw = prior_box(entity)
    margin = tolerance(controller)
    region = wo.window(raw, cam, p, float(np.linalg.norm(size)/2+margin))
    points, _, _ = wo.cloud(raw, cam, region)
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    local = (points[:, :2]-p[:2]) @ rotation
    keep = np.all(abs(local) <= size[:2]/2+margin, axis=1)
    keep &= abs(points[:, 2]-p[2]) <= size[2]+margin
    points = points[keep & np.isfinite(points).all(axis=1)]
    # Reject the support plane. A table return inside an object's predicted
    # volume must not become evidence that the object is there.
    try:
        from steward.ego.planner import entity_box
        scene = controller.rt.ego_state()
        support = controller.rt.esim._anchor(scene, entity)
        if support.id != entity.id:
            points = points[points[:, 2] > entity_box(support)[2]+margin]
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    if len(points) < 12:
        return None
    local = (points[:, :2]-p[:2]) @ rotation
    lo, hi = np.quantile(local, [.02, .98], axis=0)
    if np.any(hi-lo < size[:2]/2):
        return None
    bottom, top = np.quantile(points[:, 2], [.02, .98])
    thickness = max(top-bottom, size[2])
    centre = np.r_[p[:2]+rotation @ ((lo+hi)/2), top-thickness/2]
    # Visible depth does not measure the hidden thickness. Retain the prior
    # extent explicitly, only the horizontal surface is directly observed.
    return dict(position=centre, size=np.r_[hi-lo, thickness], yaw=yaw,
        stamp=raw['header']['stamp'], points=points, source='current_native_depth_window', independent=False)


def stacked(controller, upper, body):
    from steward.ego.planner import Box2D, rectangle_separation
    from steward.ego.assembly_precision import tolerance
    from steward.ego.interface_entry import bearing_plane
    margin = tolerance(controller)
    boxes = [Box2D(str(i), *r['position'][:2], r['yaw'], *r['size'][:2]/2)
             for i, r in enumerate((upper, body))]
    if rectangle_separation(*boxes) > margin:
        return False, None, 'separate_current_footprints'
    p, b = upper['position'], body['position']
    separated = False
    # Distinct independent masks may establish either the upper or lower layer.
    if upper['independent'] and body['independent']:
        gap = abs(p[2]-b[2])-(upper['size'][2]+body['size'][2])/2
        if abs(gap) <= margin and abs(p[2]-b[2]) > margin:
            return True, bool(p[2] > b[2]), 'current_touching_layers'
        separated = gap > margin
    # Local broad planes survive protrusions/merged envelopes. Two copies of
    # the same pixels cannot establish a layer relationship.
    a, b = upper['points'], body['points']
    if (a.ndim == b.ndim == 2 and a.shape[1:] == b.shape[1:] == (3,)
            and len(a) >= 30 and len(b) >= 30 and not np.array_equal(a, b)):
        pa = bearing_plane(a, p[:2], margin)
        # Exclude the upper footprint from the body's bearing evidence when
        # association came from a depth window rather than separate masks.
        if not (upper['independent'] and body['independent']):
            yaw = upper['yaw']; r = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            outside = np.any(abs((b[:, :2]-p[:2]) @ r) > upper['size'][:2]/2, axis=1)
            b = b[outside]
        pb = bearing_plane(b, p[:2], margin) if len(b) >= 30 else None
        if pa and pb:
            dz = pa['point'][2]-pb['point'][2]
            if dz > margin and abs(dz-upper['size'][2]) <= margin:
                return True, True, 'current_bearing_layers'
            if dz < -margin and abs(-dz-body['size'][2]) <= margin:
                return True, False, 'current_bearing_layers'
            if margin < abs(dz) <= max(upper['size'][2], body['size'][2])+margin:
                return None, bool(dz > 0), 'current_plane_order_contact_unconfirmed'
    if separated:
        return False, None, 'separated_current_heights'
    return None, None, 'overlapping_layers_unresolved'


def make_frame(controller, item, upper, hole_reader=None):
    from steward.ego.fasten_release import grasp_frame
    scene = controller.rt.ego_state()
    part = scene.find(item['part'])
    rotation = matrix_from_quat(part.pose.value['q'])
    geometry = grasp_frame(part, upper['position'], np.eye(3)) if getattr(part, 'rigid', None) else None
    catalogue = geometry['holes'] if geometry else (hole_reader or interface_holes)(controller.rt.esim, part)
    holes = [deepcopy(h) for h in catalogue if h['id'] in item['heldHoles']]
    for h in holes:
        h['point'] = (upper['position']+rotation @ h['p']).tolist()
    return dict(entity=part.id, body=item['body'], category=part.category.value,
        bodyPoseAtObservation=deepcopy(scene.find(item['body']).pose.value),
        position=upper['position'].tolist(), size=upper['size'].tolist(), rotation=rotation.tolist(), holes=holes,
        holePairs=[dict(partHole=p, bodyHole=b) for p,b in zip(item['heldHoles'], item['holes'])],
        observedAt=upper['stamp'], releasedAt=upper['stamp'], preExisting=True, observed=False,
        source=('initial_stack_prior_estimate' if upper['source']=='last_perceived_overlap_estimate'
                else 'initial_current_rgbd_stack'), heightSource=upper['source'])


def component(controller, item, samples, hole_reader=None):
    """Return None only when no current RGB-D relationship can be extracted."""
    scene = controller.rt.ego_state()
    entities = [scene.find(item[k]) for k in ('part', 'body')]
    if any(e is None for e in entities):
        return None
    best = None
    for sample in [None, *samples]:
        layers = [layer(controller, e, sample) for e in entities]
        if any(r is None for r in layers): continue
        on_body, exposed, reason = stacked(controller, *layers)
        report = dict(onBody=on_body, source='current_rgbd_stack', reason=reason,
            observedAt=min(r['stamp'] for r in layers), confirmed=on_body is not None,
            faceExposed=exposed, faceObserved=exposed is not None,
            layerSources=[r['source'] for r in layers])
        frame = make_frame(controller, item, layers[0], hole_reader) if on_body is not False else None
        best = report, frame
        if on_body is not None:
            return best
    return best


def estimate(controller, item, report):
    """After the one view, use only the bound members' last perceived geometry.

    This is a non-destructive hypothesis, never a new observation. A separate
    footprint remains separate; a merged overlapping mask is not pulled apart.
    """
    if (report['component'].get('onBody') is True and report.get('frame') and
            not actionable_stack(report)):
        body = controller.rt.ego_state().find(item['body'])
        dz = report['frame']['position'][2]-body.pose.value['p'][2] if body is not None else 0
        from steward.ego.assembly_precision import tolerance
        margin = tolerance(controller)
        report['component'].update(source='last_perceived_overlap_estimate', confirmed=False, bestEstimate=True,
            reason='current_view_unresolved_preserve_last_stack',
            faceExposed=True if dz > margin else False if dz < -margin else None)
        return
    if report['component'].get('onBody') is not None or report.get('frame'):
        return
    from steward.ego.assembly_precision import tolerance
    scene = controller.rt.ego_state()
    part, body = [scene.find(item[k]) for k in ('part', 'body')]
    if part is None or body is None: return
    p, size, yaw = prior_box(part)
    b, base, angle = prior_box(body)
    r = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    margin = tolerance(controller)
    if (np.any(abs((p[:2]-b[:2]) @ r) > base[:2]/2+margin) or
            abs(p[2]-b[2]) > (size[2]+base[2])/2+margin):
        return
    stamp = min(part.pose.t if part.pose.t is not None else -1,
                body.pose.t if body.pose.t is not None else -1)
    upper = dict(position=p, size=size, yaw=yaw, stamp=stamp, source='last_perceived_overlap_estimate')
    report['frame'] = make_frame(controller, item, upper)
    report['component'] = dict(onBody=True, confirmed=False, bestEstimate=True,
        source=upper['source'], reason='one_view_unresolved_preserve_overlapping_members', observedAt=stamp,
        faceExposed=True if p[2]-b[2] > margin else False if b[2]-p[2] > margin else None)


def finish(report):
    """Decide an unresolved overlapping stack without dismantling it."""
    component = report['component']
    if (component.get('source') == 'current_rgbd_stack' and component.get('onBody') is None
            and report.get('frame')):
        component.update(onBody=True, confirmed=False, reason='overlapping_current_layers_preserve_structure',
                         bestEstimate=True)
    from steward.ego import session_faces
    if session_faces.enabled() and session_faces.finish(report):
        return report
    if component.get('onBody') is True and component.get('faceExposed') is False:
        for hole in report['holes'].values():
            if hole.get('occupied') is None:
                hole.update(occupied=True, confirmed=False, source='hidden_attached_layer_estimate',
                            reason='preserve_existing_underbody_structure_unconfirmed')
    if report.get('observationPolicy') == 'current_rgbd' and component.get('onBody') is not True:
        for hole in report['holes'].values():
            if hole.get('occupied') is True:
                hole.update(occupied=None, confirmed=False, reason='head_seen_component_stack_unresolved')
    return report


def resolve_faces(controller, items, reports):
    """An observed member's side resolves the other catalogue face too.

    The catalogue supplies only relative interface layers. The sign of up
    comes from the actual current member/body ordering, never default yaw.
    """
    for body_id in {i['body'] for i in items}:
        body = controller.rt.ego_state().find(body_id)
        if body is None: continue
        catalogue = interface_holes(controller.rt.esim, body)
        levels = [h['p'][2] for h in catalogue]
        if not levels or max(levels) == min(levels): continue
        middle = (max(levels)+min(levels))/2
        sides = {}
        for item in items:
            if item['body'] != body_id: continue
            chosen = [h['p'][2]-middle for h in catalogue if h['id'] in item['holes']]
            if chosen and (all(z > 0 for z in chosen) or all(z < 0 for z in chosen)):
                sides[item['key']] = 1 if chosen[0] > 0 else -1
        votes = []
        for key, side in sides.items():
            c = reports[key]['component']
            if (c.get('source') == 'current_rgbd_stack' and c.get('onBody') is not False
                    and c.get('faceExposed') is not None and c.get('faceObserved')):
                votes.append(side*(1 if c['faceExposed'] else -1))
        if not votes or len(set(votes)) != 1: continue
        for key, side in sides.items():
            row = reports[key]
            from steward.ego import session_faces
            if session_faces.enabled() and row.get('face', {}).get('resolvedBeforeHoles'):
                continue
            exposed = side*votes[0] > 0
            row['face'] = dict(exposed=exposed, source='current_member_side_and_catalogue_layers',
                               confirmed=True, bodyUpZEstimate=votes[0])
            row['component']['faceExposed'] = exposed
            if not exposed:
                # A visually matched feature on the other face is not evidence
                # for a hidden interface with a similar hole pattern.
                for h in row['holes'].values():
                    h.update(occupied=None, confirmed=False, source='unavailable', reason='selected_face_hidden')


def initial_row(body, holes, part=None):
    from steward.ego.assembly_session import current
    session = current()
    if not enabled() or session is None: return None
    body = session.canonical(body)
    for item in session.state.get('goal', {}).get('interfaces', []):
        if session.canonical(item['body']) == body and set(item['holes']) == set(holes) and (
                part is None or session.canonical(item['part']) == session.canonical(part)):
            row = session.state.get('initialInterfaces', {}).get(item['key'])
            cutoff = session.state.get('initialFaceInvalidatedAt', {}).get(body)
            if cutoff is not None and (row is None or row['component'].get('source') != 'current_rgbd_stack'
                    or row['component'].get('observedAt', -1) <= cutoff):
                return None
            return row
    return None


def transported_frame(controller, frame):
    """A moved body invalidates old world points; retain only relative geometry."""
    old = frame.get('bodyPoseAtObservation')
    body = controller.rt.ego_state().find(frame['body'])
    if old is None or body is None: return None
    pose = body.pose.value
    rotation = matrix_from_quat(pose['q']) @ matrix_from_quat(old['q']).T
    shift = np.asarray(pose['p'])-rotation @ old['p']
    result = deepcopy(frame)
    result['position'] = (rotation @ frame['position']+shift).tolist()
    result['rotation'] = (rotation @ np.asarray(frame['rotation'])).tolist()
    for hole in result['holes']:
        hole['point'] = (rotation @ hole['point']+shift).tolist()
    result.update(observed=False, heightSource='body_pose_transport_estimate',
                  transportedAt=controller.rt.ego_state().stamp)
    return result


def actionable_stack(row):
    return bool(row and row.get('frame') and row['component'].get('onBody') is True and (
        row['component'].get('source') in ('current_rgbd_stack', 'last_perceived_overlap_estimate') or
        any(h.get('source') == 'current_hole_rgbd' and h.get('occupied') is not None
            for h in row.get('holes', {}).values())))


def preserve_component(body, holes, part=None):
    row = initial_row(body, holes, part)
    return actionable_stack(row)


def face(controller, body, holes):
    row = initial_row(body, holes)
    if row is None: return None
    if row.get('face'):
        from steward.ego import session_faces
        if session_faces.use_execution_face(row, body):
            return None
        if row['face'].get('exposed') is None:
            return dict(measured=False, matches=None, source=row['face']['source'],
                        reason='current_body_face_unresolved_continue_unconfirmed')
        if (session_faces.enabled() and row['component'].get('source') == 'last_perceived_overlap_estimate'
                and not row['component'].get('confirmed')):
            return dict(measured=False, matches=row['face']['exposed'],
                bodyUpZEstimate=row['face']['bodyUpZEstimate'], source=row['component']['source'],
                observedAt=row['component'].get('observedAt', -1),
                reason='preserved_stack_face_estimate_unconfirmed')
        return dict(measured=row['face'].get('confirmed', True) if session_faces.enabled() else True,
            matches=row['face']['exposed'],
            bodyUpZEstimate=row['face']['bodyUpZEstimate'], observedAt=row['component'].get('observedAt', -1),
            source=row['face']['source'], reason='current_member_side_relative_to_catalogue')
    component = row['component']
    exposed = component.get('faceExposed')
    if component.get('onBody') is not True or exposed is None: return None
    catalogue = interface_holes(controller.rt.esim, controller.rt.ego_state().find(body))
    levels = [h['p'][2] for h in catalogue]
    if not levels or max(levels) == min(levels): return None
    mid = (max(levels)+min(levels))/2
    selected = [h['p'][2]-mid for h in catalogue if h['id'] in holes]
    if not selected or not (all(z > 0 for z in selected) or all(z < 0 for z in selected)): return None
    up = (1 if selected[0] > 0 else -1)*(1 if exposed else -1)
    return dict(measured=component.get('confirmed', False), matches=exposed, bodyUpZEstimate=up,
        observedAt=component['observedAt'], source=component['source'],
        reason='selected_component_layer_relative_to_body', confirmed=component.get('confirmed', False))


def access(body, holes):
    row = initial_row(body, holes)
    if row and row.get('face'):
        from steward.ego import session_faces
        if session_faces.use_execution_face(row, body):
            return None
        exposed = row['face']['exposed']
        if exposed is None:
            return dict(status='unknown', source=row['face']['source'],
                reason='current_body_face_unresolved', nextAction='continue_best_estimate_unconfirmed')
        return dict(status='estimated_accessible' if exposed else 'inaccessible',
            source=row['face']['source'], reason='current_observed_interface_side',
            nextAction=('preserve_existing_structure_unconfirmed' if session_faces.protected(row) else
                        'assemble_onto' if exposed else 'flip_assembly_then_verify_face'),
            faceObserved=row['face'].get('confirmed', True))
    if not actionable_stack(row): return None
    if row['component'].get('source') not in ('current_rgbd_stack', 'last_perceived_overlap_estimate'): return None
    component = row['component']
    if component.get('onBody') is not True: return None
    exposed = component.get('faceExposed')
    if exposed is None:
        return dict(status='unknown', source=component['source'],
            reason='preserve_existing_stack_face_unconfirmed', nextAction='fasten_existing_component_unconfirmed')
    return dict(status='estimated_accessible' if exposed else 'inaccessible',
        source=component['source'], reason='current_stack_on_exposed_face' if exposed else 'current_stack_on_hidden_face',
        observedAt=component.get('observedAt'), faceObserved=component.get('confirmed', False),
        nextAction='fasten_existing_component' if exposed else 'flip_assembly_then_verify_face')


def summary(session, *, snapshot=True):
    """Compact model-visible evidence; never hide the initial state in a checkpoint."""
    if snapshot and 'initialInterfaceSnapshot' in session.state:
        return deepcopy(session.state['initialInterfaceSnapshot'])
    return {key:dict(status=row['status'], component=deepcopy(row['component']),
        holes={h:{k:deepcopy(e[k]) for k in ('occupied', 'confirmed', 'source', 'reason') if k in e}
               for h,e in row['holes'].items()},
        nextAction=('preserve_existing_structure_unconfirmed' if row['status']=='existing_unconfirmed' else
                    'skip_completed_interface' if row['status']=='completed' else
                    'fasten_existing_component' if row['component'].get('onBody') is True else 'place_component_then_fasten'),
        face=deepcopy(row.get('face')),
        **(dict(holeMapping=deepcopy(row['holeMapping'])) if 'holeMapping' in row else {}),
        unconfirmed=not row['component'].get('confirmed', False) or any(not e.get('confirmed') for e in row['holes'].values()))
        for key,row in session.state.get('initialInterfaces', {}).items()}
