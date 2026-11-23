"""Perceived protrusion overhang with an estimated, supported centre of mass.

All geometry is in robot registrations/RGB-D, never a scene asset or world pose.
The mass estimate uses envelope volumes; the margin represents unknown density.
"""
from itertools import product
import math
import numpy as np

from steward.ego.planner import entity_box
from steward.math3d import matrix_from_quat, quat_from_yaw


def corners(item, origin, delta):
    r = np.asarray(item.get('rotation', item.get('rotationM', matrix_from_quat(quat_from_yaw(item.get('yaw', 0.))))))
    return np.array([delta @ (np.asarray(item['position'])-origin+r @ (np.asarray(item['size'])*s/2))
                     for s in product((-1, 1), repeat=3)])


def support_polygon(points):
    """Convex contact polygon; never use an axis-aligned projected box."""
    pts = sorted(set(map(tuple, np.asarray(points)[:, :2])))
    def cross(a, b, c):
        return np.cross(np.subtract(b, a), np.subtract(c, a)).item()
    def chain(seq):
        out = []
        for point in seq:
            while len(out) >= 2 and cross(out[-2], out[-1], point) <= 0:
                out.pop()
            out.append(point)
        return out
    return np.asarray(chain(pts)[:-1]+chain(pts[::-1])[:-1])


def supports(points, com, margin):
    polygon = support_polygon(points)
    if len(polygon) < 3:
        return False
    edges = np.roll(polygon, -1, axis=0)-polygon
    inward = np.c_[-edges[:, 1], edges[:, 0]]/np.linalg.norm(edges, axis=1)[:, None]
    return bool(np.all(np.sum((np.asarray(com)[:2]-polygon)*inward, axis=1) >= margin-1e-9))


def contact_margin(face, record):
    """Contact-scale reserve plus position/COM uncertainty; never cap uncertainty."""
    polygon = support_polygon(face)
    if len(polygon) < 3:
        return math.inf
    edges = np.roll(polygon, -1, axis=0)-polygon
    inward = np.c_[-edges[:, 1], edges[:, 0]]/np.linalg.norm(edges, axis=1)[:, None]
    width = float(np.min(np.ptp(polygon @ inward.T, axis=0)))
    items = [record['reference'], *record.get('parts', [])]
    weights = np.array([max(float(np.prod(p['size'])), 1e-9) for p in items])
    centres = np.array([p['position'][:2] for p in items])
    centre = np.average(centres, axis=0, weights=weights)
    # Unknown relative density: a factor-of-two interval for each envelope.
    # Enumerate vertices for small assemblies; a conservative bound otherwise.
    if len(items) <= 8:
        spread = max(float(np.linalg.norm(np.average(centres, axis=0, weights=weights*f)-centre))
                     for f in product((.5, 2.), repeat=len(items)))
    else:
        spread = float(3*np.average(np.linalg.norm(centres-centre, axis=1), weights=weights))
    uncertainty = max(.002+spread, float(record.get('comUncertaintyM', 0.)))
    return uncertainty+.10*width


def geometry(record, turn_over, view=None, *, body_face=False, stable_bearing=False, allow_flat=False, diagnostics=None):
    diag = diagnostics if diagnostics is not None else {}
    ref = record['reference']
    if not all(k in ref for k in ('position', 'size', 'yaw')):
        return None
    if any(not all(k in p for k in ('position', 'size')) for p in record.get('parts', [])):
        return None
    origin = np.asarray(ref['position'])
    r = matrix_from_quat(quat_from_yaw(ref['yaw']))
    delta = r @ np.diag([1., -1., -1.]) @ r.T if turn_over else np.eye(3)
    core = corners(ref, origin, delta)
    bearing = float(core[:, 2].min())
    raw = None
    if view and view.get('source') in ('rgbd_segment', 'rgbd_assembly', 'rgbd_local') and not view.get('truncated', True) and view.get('invalidDepthFraction', 1) <= .05:
        points = np.asarray(view.get('points', []))
        if points.ndim == 2 and points.shape[1] == 3 and len(points) >= 12 and np.isfinite(points).all():
            raw = (points-origin) @ delta.T
            from steward.ego.fasten_geometry import surface_patch
            patch = surface_patch(raw, [0., 0.], max_tilt_deg=25)
            if stable_bearing:
                from steward.ego.assembly_edge_observation import body_patch
                broad_diag = {}
                broad = body_patch(raw, dict(ref, position=[0., 0., 0.]), broad_diag)
                if broad and broad['normal'][2] >= math.cos(math.radians(25)):
                    patch = broad
                diag['faceFit'] = broad_diag
            diag['patchFound'] = patch is not None
            if patch:
                bearing = max(bearing, patch['point'][2])
    parts = [(p, corners(p, origin, delta)) for p in record.get('parts', [])]
    items = [ref, *record.get('parts', [])]
    weights = [max(float(np.prod(p['size'])), 1e-9) for p in items]
    com = np.average([delta @ (np.asarray(p['position'])-origin) for p in items], axis=0, weights=weights)
    contact = core[np.isclose(core[:, 2], core[:, 2].min(), atol=1e-6)]
    margin = contact_margin(contact, record)
    role = 'body_face'
    if stable_bearing:
        # Only a level contact face containing the entire COM uncertainty disk
        # can replace the body face. Tilted corners/lines cannot bear the load.
        for _, pts in parts:
            low = float(pts[:, 2].min())
            face = pts[np.isclose(pts[:, 2], low, atol=1e-6)]
            face_margin = contact_margin(face, record)
            if low < bearing and supports(face, com, face_margin):
                bearing, contact, role, margin = low, face, 'spanning_member', face_margin
    elif not body_face:
        # Retain the generic stack estimate for existing non-flip callers.
        for part, pts in parts:
            low, high = pts.min(axis=0), pts.max(axis=0)
            if (np.all(low[:2] <= 0) and np.all(high[:2] >= 0)
                    and np.prod(high[:2]-low[:2]) >= .25*np.prod(np.ptp(core[:, :2], axis=0))):
                bearing = min(bearing, float(low[2]))
    protrusions = []
    diag.update(bearingZM=round(bearing, 6), bodyBottomZM=round(float(core[:, 2].min()), 6),
                bearingRole=role, rawPoints=0 if raw is None else len(raw),
                registeredParts=len(parts), requiredComMarginM=round(margin, 6))
    if raw is not None:
        diag.update(rawLowZM=round(float(raw[:, 2].min()), 6),
                    rawHighZM=round(float(raw[:, 2].max()), 6),
                    belowBearingPoints=int(np.sum(raw[:, 2] < bearing-.006)),
                    depthBelowBearingM=round(float(bearing-raw[:, 2].min()), 6),
                    xyCoverage=np.round(np.ptp(raw[:, :2], axis=0)/np.maximum(np.ptp(core[:, :2], axis=0), 1e-9), 3).tolist())
    if raw is not None:
        protrusions.extend(raw[raw[:, 2] < bearing-.006])
    for part, pts in parts:
        if pts[:, 2].min() < bearing-.006:
            protrusions.extend(pts)
    diag.update(protrusionPoints=len(protrusions),
                reason='protrusions_found' if protrusions else 'no_points_below_bearing',
                unseenGeometry='not_resolved_by_visible_cloud')
    if not protrusions and not allow_flat:
        return None
    all_points = np.vstack([core, *[pts for _, pts in parts], np.asarray(protrusions).reshape(-1, 3),
                            raw if raw is not None else np.empty((0, 3))])
    return dict(protrusions=np.asarray(protrusions).tolist(), envelope=all_points.tolist(),
        core=core.tolist(), bearingZM=float(bearing), upperBodyZM=float(core[:, 2].max()), centreOfMass=com.tolist(), comUncertaintyM=margin,
        source=(view['source'] if raw is not None else 'registered_envelopes'), massSource='uniform_volume_estimate',
        bearingRole=role, bearingPolygon=contact.tolist(),
        yaw=ref['yaw'], turnOver=bool(turn_over))


def motion_bounds(record, low, high):
    """Include observed/inferred protrusions in the normal collision/sweep bounds."""
    shape = record.get('edgeGeometry')
    if not shape:
        return low, high
    r = matrix_from_quat(quat_from_yaw(record['reference']['yaw']))
    delta = r @ np.diag([1., -1., -1.]) @ r.T if shape['turnOver'] else np.eye(3)
    local = np.asarray(shape['envelope']) @ delta @ r
    return np.minimum(low, local.min(axis=0)), np.maximum(high, local.max(axis=0))


def preflip_view(controller, entity, view):
    """One acquisition, or cached geometry in the ASSEMBLY_EDGE_RESULT flip observation scope."""
    if getattr(controller, '_edge_reuse_frames', False):
        from steward.ego.edge_result import pre_view
        return pre_view(controller, entity, view)
    def reason(v, after=-math.inf):
        now = controller.ex._state()['stamp']
        after = max(after, record['reference'].get('ownReleaseAt', -math.inf)) if record else after
        if not v or v.get('source') != 'rgbd_segment': return 'no_rgbd_segment'
        if not after < v.get('stamp', -1) <= now or now-v['stamp'] > 20: return 'no_fresh_frame'
        if v.get('truncated', True): return 'truncated_view'
        if v.get('invalidDepthFraction', 1) > .05: return 'invalid_depth'
        points = np.asarray(v.get('points', []))
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 12 or not np.isfinite(points).all():
            return 'no_local_points'
        return None
    from steward.ego.assembly_edge_observation import merged_view, native_view
    controller._edge_observation_diag = {}
    record = controller.rt.estimator.stationary_assemblies.records.get(entity)
    why = reason(view)
    if why and record:
        merged = merged_view(controller, record)
        if merged:
            return merged, dict(attempts=0, reason='fresh_local_geometry', source=merged['source'], diagnostics=controller._edge_observation_diag)
    if why is None:
        return view, dict(attempts=0, reason='fresh_local_geometry')
    from steward.ego.assembly_precision import acquire
    from steward.ego.skills import SkillFailed
    after = controller.ex._state()['stamp']
    try:
        acquire(controller, [entity])
    except SkillFailed:
        controller.check()
        return None, dict(attempts=1, reason='acquisition_failed')
    view = controller.sample(entity)
    why = reason(view, after)
    if why and record:
        alternate = merged_view(controller, record, after) or native_view(controller, record, after)
        if alternate:
            return alternate, dict(attempts=1, reason='fresh_local_geometry', source=alternate['source'], diagnostics=controller._edge_observation_diag)
    return (None if why else view), dict(attempts=1, reason=why or 'fresh_local_geometry', diagnostics=controller._edge_observation_diag)


def _reject(diagnostics, reason):
    if diagnostics is not None:
        counts = diagnostics.setdefault('rejected', {})
        counts[reason] = counts.get(reason, 0)+1


def _candidates(record, table, sigma=0., diagnostics=None, edge_filter=None):
    shape = record.get('edgeGeometry')
    if shape is None:
        return []
    box, _, top = entity_box(table)
    r = matrix_from_quat(quat_from_yaw(box.yaw))[:2, :2]
    protrusions = np.asarray(shape['protrusions'])[:, :2] @ r
    envelope = np.asarray(shape['envelope'])[:, :2] @ r
    com = r.T @ np.asarray(shape['centreOfMass'])[:2]
    half = np.array([box.hx, box.hy])
    contact = shape.get('bearingPolygon', shape.get('core'))
    if contact and not supports(contact, shape['centreOfMass'], shape['comUncertaintyM']):
        _reject(diagnostics, 'com')
        if diagnostics is not None:
            diagnostics['comGate'] = 'bearing'
            polygon = support_polygon(contact)
            if len(polygon) >= 3:
                vectors = np.roll(polygon, -1, axis=0)-polygon
                inward = np.c_[-vectors[:, 1], vectors[:, 0]]/np.linalg.norm(vectors, axis=1)[:, None]
                clearance = float(np.min(np.sum((np.asarray(shape['centreOfMass'])[:2]-polygon)*inward, axis=1)))
                diagnostics['bearingShortfallM'] = round(max(0., shape['comUncertaintyM']-clearance), 6)
        return []
    gap = max(.008, 2*sigma)+record.get('edgeClearanceExtraM', 0.)
    margin = shape['comUncertaintyM']+2*sigma
    result = []
    for axis, sign in product(range(2), (-1, 1)):
        if edge_filter is not None and (axis, sign) != edge_filter:
            continue
        tangent = 1-axis
        # Every low protrusion clears the same edge, while the estimated COM
        # and its uncertainty stay inside all four edges.
        normal = sign*(half[axis]+gap)-sign*np.min(sign*protrusions[:, axis])
        lo = -half[tangent]-envelope[:, tangent].min()+gap
        hi = half[tangent]-envelope[:, tangent].max()-gap
        far_clearance = float(np.min(sign*(normal+envelope[:, axis])))
        if lo > hi or far_clearance < -half[axis]+gap:
            _reject(diagnostics, 'bounds')
            if diagnostics is not None:
                shortfall = max(lo-hi, -half[axis]+gap-far_clearance)
                diagnostics['boundsShortfallM'] = round(min(shortfall, diagnostics.get('boundsShortfallM', math.inf)), 6)
            continue
        root = r.T @ (np.asarray(record['reference']['position'])[:2]-[box.cx, box.cy])
        along = float(np.clip(root[tangent], lo, hi))
        for value in dict.fromkeys([along, (lo+hi)/2, lo, hi]):
            offset = np.zeros(2); offset[axis] = normal; offset[tangent] = value
            clearance = float(np.min(half-abs(offset+com)))
            if clearance < margin:
                _reject(diagnostics, 'com')
                if diagnostics is not None:
                    diagnostics['comGate'] = 'edge'
                    diagnostics['requiredComMarginM'] = round(margin, 6)
                    diagnostics['bestRejectedComMarginM'] = round(max(clearance, diagnostics.get('bestRejectedComMarginM', -math.inf)), 6)
                    diagnostics['comShortfallM'] = round(margin-diagnostics['bestRejectedComMarginM'], 6)
                    from steward.ego.edge_result import final_contact
                    diagnostics['orientationUnion'] = not final_contact(shape)
                continue
            world = np.array([box.cx, box.cy])+r @ offset
            result.append(dict(offset=offset.tolist(), spot=world.tolist(), edgeAxis=axis,
                edgeSign=sign, overhangM=float(np.max(sign*(offset[axis]+envelope[:, axis]))-half[axis]),
                comMarginM=clearance, requiredComMarginM=margin, protrusionClearanceM=gap,
                bodyCentreZM=top-shape['bearingZM'], bearingZM=shape['bearingZM'],
                bodyGraspZM=top+(shape['upperBodyZM']-shape['bearingZM'])/2,
                comOffsetWorldM=shape['centreOfMass'][:2], massSource=shape['massSource'],
                bearingRole=shape.get('bearingRole', 'body_face'),
                estimatedTiltDeg=0., observed=False))
    return sorted(result, key=lambda c:np.linalg.norm(np.asarray(c['spot'])-record['reference']['position'][:2]))


def candidates(record, table, sigma=0., diagnostics=None):
    """Put the direction from COM to obstruction perpendicular to an edge.

    Rotate geometry and the commanded body together. If the low geometry
    straddles the COM there is no stable solution at a single straight edge.
    """
    from copy import deepcopy
    from steward.ego.edge_result import final_contact
    from steward.ego import edge_geometry
    if edge_geometry.active(record.get('edgeGeometry')):
        return edge_geometry.candidates(record, table, sigma, diagnostics)
    shape = record.get('edgeGeometry')
    if not shape:
        if diagnostics is not None: diagnostics['candidateCount'] = 0
        return []
    box, _, _ = entity_box(table)
    centre = np.asarray(shape['protrusions'])[:, :2].mean(axis=0)
    direction = centre-np.asarray(shape['centreOfMass'])[:2]
    if np.linalg.norm(direction) < 1e-6:
        _reject(diagnostics, 'com')
        if diagnostics is not None: diagnostics['candidateCount'] = 0
        return []
    heading = math.atan2(direction[1], direction[0])
    results = []
    for edge_heading in (box.yaw, box.yaw+math.pi/2, box.yaw+math.pi, box.yaw-math.pi/2):
        angle = (edge_heading-heading+math.pi) % (2*math.pi)-math.pi
        rot = matrix_from_quat(quat_from_yaw(angle))
        moved = deepcopy(record); geometry = moved['edgeGeometry']
        for key in ('protrusions', 'envelope', 'core', 'bearingPolygon'):
            if key in geometry:
                geometry[key] = (np.asarray(shape[key]) @ rot.T).tolist()
        # Include the unturned envelope too: orientation uncertainty may retain
        # some of its footprint during the controlled wrist turn.
        if not final_contact(shape):
            geometry['protrusions'] += shape['protrusions']
            geometry['envelope'] += shape['envelope']
        geometry['centreOfMass'] = (rot @ shape['centreOfMass']).tolist()
        edge_axis = int(abs(math.sin(edge_heading-box.yaw)) > .5)
        edge_sign = 1 if math.cos(edge_heading-box.yaw-edge_axis*math.pi/2) > 0 else -1
        for option in _candidates(moved, table, sigma, diagnostics, (edge_axis, edge_sign)):
            option.update(yaw=record['reference']['yaw']+angle, yawDeltaRad=angle)
            results.append(option)
    if final_contact(shape):
        for option in _candidates(record, table, sigma, diagnostics):
            option.update(yaw=record['reference']['yaw'], yawDeltaRad=0.)
            results.append(option)
        unique = {}
        for c in results:
            unique.setdefault(tuple(np.round([*c['offset'], c['yaw']], 9)), c)
        results = sorted(unique.values(), key=lambda c:(abs(c['yawDeltaRad']) > 1e-6,
            np.linalg.norm(np.asarray(c['spot'])-record['reference']['position'][:2])))
        if diagnostics is not None: diagnostics['candidateCount'] = len(results)
        return results
    if diagnostics is not None: diagnostics['candidateCount'] = len(results)
    return sorted(results, key=lambda c:(abs(c['yawDeltaRad']),
        np.linalg.norm(np.asarray(c['spot'])-record['reference']['position'][:2])))


def validate(record, table, offset, sigma=0., yaw=None):
    """Recompute at prediction time; callers cannot inject arbitrary overhangs."""
    from steward.ego import edge_geometry
    if edge_geometry.active(record.get('edgeGeometry')):
        return edge_geometry.validate(record, table, offset, sigma, yaw)
    from steward.ego.edge_result import final_contact
    if final_contact(record.get('edgeGeometry')):
        from steward.ego.edge_result import validate as continuous
        return continuous(record, table, offset, sigma, yaw)
    return next((c for c in candidates(record, table, sigma)
                 if offset is not None and np.allclose(offset, c['offset'], atol=1e-6)
                 and (yaw is None or abs((yaw-c['yaw']+math.pi) % (2*math.pi)-math.pi) < 1e-6)), None)


def release_inside(plan, xy):
    edge = plan['edgePlacement']; box = plan['supportBox']
    from steward.ego import placement_support_margin
    if placement_support_margin.enabled() and edge.get('inwardOnly'):
        delta=np.asarray(xy)-plan['spot']
        return bool(delta @ np.asarray(edge['edgeNormalWorld']) <= 0
            and np.linalg.norm(delta) <= edge.get('comMarginM', 0.))
    r = matrix_from_quat(quat_from_yaw(box['yaw']))[:2, :2]
    com = r.T @ (np.asarray(xy)+edge['comOffsetWorldM']-[box['cx'], box['cy']])
    from steward.ego.edge_geometry import enabled
    if enabled() and edge.get('contactWindow'):
        # The commanded best-margin negative fallback is deliberate. Encoder
        # error still triggers one refinement, not a permanent held-load gate.
        return bool(np.linalg.norm(np.asarray(xy)-plan['spot']) <= .008)
    return bool(np.min([box['hx'], box['hy']]-abs(com)) >= edge['requiredComMarginM']
                and np.linalg.norm(np.asarray(xy)-plan['spot']) <= .008)


def level(view, xy, after, stamp):
    """One independent local face measurement; missing pixels remain unknown."""
    result = dict(estimatedTiltDeg=None, observed=False, level=None, reason='no_fresh_local_face')
    if (not view or view.get('source') not in ('rgbd_segment', 'rgbd_assembly', 'rgbd_local') or view.get('truncated', True)
            or view.get('invalidDepthFraction', 1) > .05
            or not after < view.get('stamp', -1) <= stamp or stamp-view['stamp'] > 20):
        return result
    from steward.ego.fasten_geometry import surface_patch
    patch = surface_patch(view.get('points', []), xy, max_tilt_deg=35)
    if patch is None:
        return result
    tilt = math.degrees(math.acos(np.clip(patch['normal'][2], -1, 1)))
    return dict(estimatedTiltDeg=tilt, observed=True, level=tilt <= 5.+1e-6,
                reason='rgbd_local_upward_face', observedAt=view['stamp'], patch=patch)


def observe_level(controller, reference, view, after):
    """Measure a broad body face, continuing to native depth after a bad mask."""
    from steward.ego import edge_geometry, edge_face
    if edge_geometry.enabled() and getattr(controller, "_edge_contact_measure", False):
        measured = edge_face.measure(controller, reference, view, after)
        if measured["observed"]:
            return measured
        local = level(view, reference["position"][:2], after, controller.ex._state()["stamp"])
        return dict(local, unconfirmed=True) if local["observed"] else measured
    from steward.ego.assembly_edge_observation import merged_view, native_view, body_patch, usable
    controller._edge_observation_diag = diagnostics = {}
    # Preserve the existing independent local-face measurement first. The broad
    # and native fits below recover it when the narrow centre patch is absent.
    initial = level(view, reference['position'][:2], after, controller.ex._state()['stamp'])
    if initial['observed']:
        return dict(initial, observationDiagnostics=dict(separate=dict(reason='local_face_fit',
            points=initial.get('patch', {}).get('points', 0))))
    result = dict(estimatedTiltDeg=None, observed=False, level=None, reason='no_fresh_local_face',
                  observationDiagnostics=diagnostics)
    surface = controller.rt.ego_state().find(reference['support'])
    target_z = reference.get('landingTopM', reference['position'][2]+reference['size'][2]/2)
    tolerance = reference['size'][2]+.5*max(reference['size'][:2])*math.sin(math.radians(35))
    def measure(sample, key, native=False):
        diag = diagnostics.setdefault(key, {})
        now = controller.ex._state()['stamp']
        if sample is None:
            diag.setdefault('reason', 'no_view')
            return None
        if not native and not usable(sample, now, after):
            diag['reason'] = ('frame_not_fresh_after_release' if not after < sample.get('stamp', -1) <= now
                              or now-sample.get('stamp', -1) > 20 else 'view_quality')
            return None
        # The native face ROI can be partly clipped or have missing pixels;
        # only its visible, spatially extended fitted inliers establish a face.
        if native and (not after < sample.get('stamp', -1) <= now or now-sample.get('stamp', -1) > 20):
            diag['reason'] = 'frame_not_fresh_after_release'
            return None
        patch = body_patch(sample['points'], reference, diag)
        if patch is None:
            return None
        diag['heightErrorM'] = round(patch['point'][2]-target_z, 5)
        if (surface is None or patch['point'][2] <= entity_box(surface)[2]+.004
                or abs(patch['point'][2]-target_z) > tolerance):
            diag['reason'] = 'local_patch_not_body_face'
            return None
        tilt = math.degrees(math.acos(np.clip(patch['normal'][2], -1, 1)))
        return dict(estimatedTiltDeg=tilt, observed=True, level=tilt <= 5.+1e-6,
            reason=key.removesuffix('Fit')+'_rgbd_local_body_face', observedAt=sample['stamp'],
            faceSource=sample['source'], patch=patch, observationDiagnostics=diagnostics)
    measured = measure(view, 'separate')
    if measured:
        return measured
    record = dict(reference=reference)
    sample = merged_view(controller, record, after)
    measured = measure(sample, 'mergedFit')
    if measured:
        return measured
    # A usable merged envelope is not necessarily a usable face. Do not let it
    # short-circuit the independent native acquisition when its fit failed.
    from steward.ego import edge_result
    sample = (edge_result.cached_view(controller, record, after, face=True) if getattr(controller, '_edge_reuse_frames', False)
              else native_view(controller, record, after, face=True))
    measured = measure(sample, 'nativeFit', native=True)
    return measured or result


def rebase_height(new, old, delta, edge):
    shape = new.get('edgeGeometry')
    if shape:
        rot = matrix_from_quat(quat_from_yaw(edge.get('yawDeltaRad', 0.)))
        for key in ('protrusions', 'envelope', 'core', 'bearingPolygon'):
            if key in shape:
                shape[key] = (np.asarray(shape[key]) @ rot.T).tolist()
        shape['centreOfMass'] = (rot @ shape['centreOfMass']).tolist()
        for key in ('outwardAxis', 'bodyLongAxis'):
            if key in shape: shape[key] = (rot @ shape[key]).tolist()
        shape.update(yaw=new['reference']['yaw'], turnOver=False)
        if 'outwardAxis' in shape:
            new['reference']['edgeGraspOutward'] = shape['outwardAxis']
    centre_z = edge['bodyCentreZM']
    if old.get('forcePoints'):
        new['forcePoints'] = (np.asarray(old['forcePoints']) @ delta.T).tolist()
    ref = new['reference']
    if old.get('contactReference'):
        contact = new['contactReference']
        contact['position'] = (np.r_[ref['position'][:2], centre_z]+delta @
            (np.asarray(old['contactReference']['position'])-old['reference']['position'])).tolist()
        axis = delta @ matrix_from_quat(quat_from_yaw(old['contactReference']['yaw']))[:,0]
        contact['yaw'] = math.atan2(axis[1], axis[0])
        contact['transportedAt'] = ref['ownReleaseAt']
    ref['position'][2] = centre_z
    ref['landingTopM'] = centre_z+(shape['upperBodyZM'] if shape and shape.get('contactWindow') else ref['size'][2]/2)
    ref['graspHeightM'] = edge['bodyGraspZM']
    ref['heightSource'] = 'edge_bearing_geometry'
    new['entity'].pose.value['p'] = list(ref['position'])
    for part, previous in zip(new['parts'], old['parts']):
        part['position'] = (np.asarray(ref['position'])+delta @
                            (np.asarray(previous['position'])-old['reference']['position'])).tolist()


def tilt_geometry(record, measured):
    """Infer a high-end obstruction from the observed slope, not an unseen outline."""
    if not measured.get('observed') or not measured.get('estimatedTiltDeg', 0) > 5:
        return None
    patch = measured.get('patch')
    if not patch:
        return None
    normal = np.asarray(patch['normal'], float)
    direction = -normal[:2]
    if normal[2] <= 0 or np.linalg.norm(direction) == 0:
        return None
    slope = np.linalg.norm(direction)/normal[2]
    direction /= np.linalg.norm(direction)
    shape = geometry(record, False, body_face=True, stable_bearing=True, allow_flat=True)
    if shape is None:
        return None
    core = np.asarray(shape['core'])
    along = core[:, :2] @ direction
    span = float(np.ptp(along))
    band = max(.20*span, 2*patch.get('residualM', 0.)/slope)
    if band >= .5*span:
        return None
    # Clear an uncertain outer fifth, rather than just its highest tip.
    polygon = support_polygon(core)
    threshold = along.max()-band
    end = []
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        da, db = a @ direction-threshold, b @ direction-threshold
        if da >= 0:
            end.append(a)
        if (da >= 0) != (db >= 0):
            end.append(a+(b-a)*da/(da-db))
    protrusions = np.c_[np.asarray(end), np.zeros(len(end))]
    protrusions[:, 2] = shape['bearingZM']-span*slope
    known = np.asarray(shape['protrusions']).reshape(-1, 3)
    shape['protrusions'] = np.vstack([known, protrusions]).tolist()
    shape['envelope'] += protrusions.tolist()
    shape.update(source='post_tilt', turnOver=False)
    return shape


def adjust_once(controller, job, plan):
    """At most one body reposition, including a tilt-derived no-geometry fallback."""
    from copy import deepcopy
    from steward.ego.skills import SkillFailed
    post = job.postcondition; edge = post['edgePlacement']
    edge.update(fallback=False, fallbackResult='not_needed', tiltBefore=edge.get('estimatedTiltDeg'), tiltAfter=None)
    record = controller.rt.estimator.stationary_assemblies.records[plan['body']]
    if not record.get('edgeGeometry'):
        edge['fallback'] = True
        stamp = edge.get('observedAt', -1)
        now = controller.ex._state()['stamp']
        shape = (tilt_geometry(record, edge) if record['reference'].get('ownReleaseAt', -1) < stamp <= now
                 and now-stamp <= 20 else None)
        if shape is None:
            edge.update(adjustment='no_suitable_geometry', fallbackResult='no_tilt_geometry')
            return
        record['edgeGeometry'] = shape
        edge['source'] = shape['source']
    record['edgeClearanceExtraM'] = 0. if edge['fallback'] else .008
    table = controller.rt.ego_state().find(record['reference']['support'])
    options = candidates(record, table, controller.rt.esim._sigma_xy(table)) if table else []
    if not options:
        edge.update(adjustment='no_supported_adjustment', fallbackResult='no_candidate')
        return
    post['levelAlignment']['adjustmentAttempts'] += 1
    child = {}; job.postcondition = child
    try:
        controller.flip(job, dict(plan, turnOver=False, _edgeAdjusted=True))
        post['edgePlacementBeforeAdjustment'] = deepcopy({k:v for k,v in edge.items() if k in (
            'used', 'source', 'hint', 'reason', 'estimatedTiltDeg', 'observed', 'levelReason',
            'fallback', 'fallbackResult', 'geometryDiagnostics', 'observationDiagnostics',
            'candidateCount', 'rejected', 'comGate', 'comShortfallM')})
        result = child['edgePlacement']
        post['edgePlacement'] = dict(result, adjustment='one_body_reposition',
            fallback=edge['fallback'], fallbackResult='used' if result['used'] else 'not_admitted',
            tiltBefore=edge['tiltBefore'], tiltAfter=result.get('estimatedTiltDeg'))
        post['reference'] = child['reference']
        post.setdefault('stages', []).extend(child.get('stages', []))
    except SkillFailed as exc:
        controller.check()
        edge.update(adjustment=str(exc), fallbackResult='failed')
        if controller.rt.estimator.held_id:
            raise
    finally:
        job.postcondition = post
