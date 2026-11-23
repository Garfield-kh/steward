"""Optional, dimension-free caller prior for a downward end after a body turn."""
from copy import deepcopy
import numpy as np
from steward.math3d import matrix_from_quat, quat_from_yaw
from steward.ego import assembly_edge as edge

ENDS = {'+x': [1., 0.], '-x': [-1., 0.], '+y': [0., 1.], '-y': [0., -1.]}


def validate_hint(hint):
    from steward.ego.esim import PredictionError
    if hint is None:
        return
    if (not isinstance(hint, dict) or set(hint)-{'end', 'uncertain', 'edge'}
            or hint.get('end') not in (*ENDS, 'unknown')
            or not isinstance(hint.get('uncertain', False), bool)):
        raise PredictionError('downwardProtrusion needs end +x/-x/+y/-y/unknown in the registered body frame and optional uncertain boolean')
    choice = hint.get('edge')
    if choice is not None and (not isinstance(choice, dict) or set(choice) != {'axis', 'sign'}
            or choice['axis'] not in (0, 1) or choice['sign'] not in (-1, 1)):
        raise PredictionError('downwardProtrusion.edge needs support-frame axis 0/1 and sign -1/1 from edgeOptions')


def apply(record, turn_over, shape, hint):
    """No hint is an exact identity. Geometry always wins over a direction prior."""
    if hint is None:
        return shape, None
    validate_hint(hint)
    report = dict(source='automatic_geometry', end=hint['end'], uncertain=hint.get('uncertain', False))
    if hint['end'] == 'unknown':
        return shape, dict(report, reason='no_direction_prior')
    direction = np.array(ENDS[hint['end']])
    if turn_over:
        direction[1] *= -1
    direction = matrix_from_quat(quat_from_yaw(record['reference']['yaw']))[:2, :2] @ direction
    if shape is not None:
        observed = np.asarray(shape['protrusions'])[:, :2]-np.asarray(shape['centreOfMass'])[:2]
        projection = observed @ direction
        reason = 'geometry_preferred'
        if projection.max() < 0:
            reason = 'hint_contradicts_geometry'
        elif projection.min() > 0:
            reason = 'hint_agrees_with_geometry'
        return shape, dict(report, reason=reason)
    shape = edge.geometry(record, turn_over, body_face=True, stable_bearing=True, allow_flat=True)
    if shape is None:
        return None, dict(report, reason='no_registered_body_geometry')
    core = np.asarray(shape['core']); along = core[:, :2] @ direction
    # The prior identifies an end, not a tip. Clear an outer band of the body's
    # measured extent. Uncertainty increases that band without reducing reserves.
    fraction = .35 if hint.get('uncertain') else .20
    threshold = along.max()-fraction*np.ptp(along)
    polygon = edge.support_polygon(core); end = []
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        da, db = a @ direction-threshold, b @ direction-threshold
        if da >= -1e-12:
            end.append(a)
        if (da >= 0) != (db >= 0):
            end.append(a+(b-a)*da/(da-db))
    # Depth is unknown. A conservative registered-height envelope is only for
    # motion bounds, never claimed as a measured protrusion or levelness.
    points = np.c_[end, np.full(len(end), shape['bearingZM']-record['reference']['size'][2])]
    shape['protrusions'] = points.tolist(); shape['envelope'] += points.tolist()
    shape.update(source='caller_hint', hintedEnd=hint['end'])
    return shape, dict(report, source='caller_hint', reason='end_prior', endBandFraction=fraction,
                       depthObserved=False)


def options(record, table, sigma, hint, diagnostics):
    rows = edge.candidates(record, table, sigma, diagnostics)
    if hint and hint.get('edge'):
        chosen = hint['edge']
        selected = lambda r:(r['edgeAxis'], r['edgeSign']) == (chosen['axis'], chosen['sign'])
        from steward.ego.edge_result import final_contact
        if final_contact(record.get('edgeGeometry')):
            rows.sort(key=lambda r:not selected(r))  # preference, not a dead end when unreachable
        else:
            rows = [r for r in rows if selected(r)]
        diagnostics['candidateCount'] = len(rows)
    if not rows and hint and record.get('edgeGeometry'):
        diagnostics['hintRejection'] = 'no_stable_edge_for_direction'
        shape = record['edgeGeometry']
        pts = np.asarray(shape['protrusions'])[:, :2]; com = np.asarray(shape['centreOfMass'])[:2]
        direction = pts.mean(axis=0)-com
        if np.linalg.norm(direction):
            direction /= np.linalg.norm(direction)
            available = float(np.min((pts-com) @ direction))-max(.008, 2*sigma)
            diagnostics['comShortfallM'] = max(diagnostics.get('comShortfallM', 0.),
                diagnostics.get('bearingShortfallM', 0.), shape['comUncertaintyM']+2*sigma-available)
    return rows


def preview(planner, scene, body, params):
    """Read-only geometric options; motion admission is refreshed while held."""
    from steward.ego.fasten_plan import value
    hint = params.get('downwardProtrusion'); validate_hint(hint)
    record = deepcopy(value(body, 'orientation', {}).get('stationaryAssembly'))
    result = dict(options=[], motionChecks='pending_execution', source='automatic_geometry')
    if not record:
        return dict(result, reason='no_registered_body_geometry')
    table = scene.find(record['reference']['support'])
    if table is None:
        return dict(result, reason='no_support_geometry')
    view = value(body, 'contactView')
    from steward.ego.assembly_edge_observation import usable
    if not usable(view, scene.stamp, record['reference'].get('ownReleaseAt', -float('inf'))):
        view = None
    shape = edge.geometry(record, params.get('turnOver', True), view, body_face=True, stable_bearing=True)
    shape, report = apply(record, params.get('turnOver', True), shape, hint)
    from steward.ego import edge_result
    from steward.ego import edge_geometry
    if edge_geometry.enabled() and (shape or edge_geometry.requires_prior(record)):
        shape = edge_geometry.prepare(record, params.get('turnOver', True), view, shape, {})
    elif edge_result.enabled():
        shape = edge_result.prepare(record, params.get('turnOver', True), view, shape, {})
    record['edgeGeometry'] = shape
    if report:
        result['hint'] = report
    if not shape:
        return dict(result, reason='no_protrusion_geometry')
    result['source'] = shape['source']; diagnostics = {}
    rows = options(record, table, planner._sigma_xy(table), hint, diagnostics)
    seen = set()
    for row in rows:
        key = (row['edgeAxis'], row['edgeSign'])
        if key in seen:
            continue
        seen.add(key)
        result['options'].append(dict(edge=dict(axis=key[0], sign=key[1]),
            end=shape.get('hintedEnd', observed_end(record, shape)),
            overhangM=row['overhangM'], comMarginM=row['comMarginM'],
            requiredComMarginM=row['requiredComMarginM'], yawDeltaRad=row['yawDeltaRad']))
    result['diagnostics'] = diagnostics
    return result


def observed_end(record, shape):
    direction = np.mean(shape['protrusions'], axis=0)[:2]-np.asarray(shape['centreOfMass'])[:2]
    local = matrix_from_quat(quat_from_yaw(record['reference']['yaw']))[:2, :2].T @ direction
    if shape['turnOver']:
        local[1] *= -1
    axis = int(np.argmax(abs(local)))
    return ('+' if local[axis] >= 0 else '-')+'xy'[axis]
