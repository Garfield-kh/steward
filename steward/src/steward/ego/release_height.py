"""Release floor and tilt diagnostics from registered geometry, RGB-D and FK."""
from copy import deepcopy
import math
import numpy as np
from steward.ego.fasten_geometry import transform_feature, surface_patch
from steward.ego.fasten_plan import value


def geometry(controller, current):
    est = controller.rt.estimator
    track = est.tracks.get(current.get('object'))
    feature = getattr(track, 'held_tool_feature', None) or getattr(track, 'held_grasp_geometry', None)
    target = current.get('seatTarget') or {}
    if not feature or 'point' not in target or est.held_id != current.get('object'):
        return None
    state = controller.ex._state()
    f = transform_feature(feature,
        est.gripper_rotation(est.held_gripper, state['joints'], state['odometry']),
        est.gripper_position(est.held_gripper, state['joints'], state['odometry']))
    entry = float(target['point'][2]); floor = entry
    sources = ['registered_entry']
    reg = (getattr(controller, '_registered_alignment', None) or {}).get('registration') or {}
    if (not reg.get('invalidated') and reg.get('hole') is not None
            and reg.get('hole') == current.get('seatHole')
            and reg.get('body') == current.get('seatBody') and reg.get('point')):
        floor = max(floor, float(reg['point'][2]))
        sources.append('registered_geometry')
    # A same-frame visible bearing surface may raise the bound, never lower it.
    servo = current.get('tipEntryServo', {})
    last = (servo.get('frames') or [{}])[-1]
    if last.get('stamp') == servo.get('lastStamp') and last.get('bothVisible'):
        floor = max(floor, float(last['entry']['point'][2])); sources.append('same_frame_entry')
    part = current.get('seatPart') or target.get('entity')
    scene = controller.rt.ego_state() if part else None
    view = value(scene.find(part), 'contactView') if scene else None
    if (view and view.get('source') == 'rgbd_segment' and view.get('stamp') == state['stamp']
            and not view.get('truncated', True) and view.get('invalidDepthFraction', 1) <= .05):
        patch = surface_patch(view.get('points', []), target['point'][:2])
        if patch:
            floor = max(floor, float(patch['point'][2])); sources.append('same_frame_bearing_surface')
    axis = np.asarray(f['axis'], float); axis /= np.linalg.norm(axis)
    observed_basis = feature.get('source') == 'rgbd_endpoint_width' or (
        feature.get('offsetEvidence') or {}).get('status') == 'observed_basis'
    measured = bool(last.get('bothVisible') and last.get('stamp') == state['stamp']
        and last.get('lowerEnd', {}).get('axis') is not None
        and last.get('lift') == state['joints']['lift'])
    if measured:
        axis = np.asarray(last['lowerEnd']['axis'], float); axis /= np.linalg.norm(axis)
    return dict(entryM=entry, floorM=floor, tipM=float(f['tip'][2]),
        lift=float(state['joints']['lift']), floorSources=sources,
        tiltDeg=math.degrees(math.acos(float(np.clip(axis[2], -1, 1)))),
        tiltSource='same_frame_lower_shaft_axis' if measured else
            'observed_grasp_axis_plus_fk' if observed_basis else 'inferred_grasp_axis_plus_fk',
        tiltMeasured=measured, observedAt=state['stamp'])


def guard(controller, current, path):
    """Clamp before movement, accounting for real TCP height rather than lift=Z."""
    try: g = geometry(controller, current)
    except (AttributeError, KeyError, TypeError, ValueError): g = None
    row = current.setdefault('releaseHeight', {})
    if g is None:
        row.update(status='unknown', tipRelativeToEntryM=None, reason='no_held_endpoint_geometry')
        return
    from steward.ego.fasten_delivery import RELEASE_HEIGHT_M
    minimum = g['lift'] + g['floorM'] + RELEASE_HEIGHT_M - g['tipM']
    original = float(path['lift']); end = max(original, minimum)
    state = controller.ex._state()
    if end != original and not controller.rt.esim.arm.reachable_path(
            [dict(state['joints']), dict(state['joints'], lift=end)]):
        # Keep the current checked pose, finish release unconfirmed, then inspect.
        end = float(state['joints']['lift'])
        row['reason'] = 'floor_endpoint_unreachable_release_at_current_pose'
    path['lift'] = end
    if 'safeLift' in path: path['safeLift'] = max(float(path['safeLift']), end)
    row.update(status='estimated', **g, requestedLift=original, boundedLift=end,
        heightClamped=end != original,
        predictedTipRelativeToEntryM=g['tipM']+end-g['lift']-g['entryM'])
    if current.get('seatEndpoint') is not None:
        current['seatEndpoint']['lift'] = end


def record(controller, current, report):
    from steward.ego import seating_geometry
    if seating_geometry.enabled() and current.get('seatingGeometryDelivery', {}).get('tipRelativeToEntryM') is not None:
        row = current['seatingGeometryDelivery']
        current['releaseHeight'] = {k:deepcopy(row[k]) for k in
            ('policy', 'tipRelativeToEntryM', 'tiltDeg', 'entryObservedAt',
             'entrySource', 'endpointSource', 'afterXYM', 'unconfirmed', 'reason') if k in row}
        report['releaseHeight'] = deepcopy(current['releaseHeight'])
        return
    if current.get('releaseHeight', {}).get('policy') == 'observed_plane_height_only':
        report['releaseHeight'] = deepcopy(current['releaseHeight'])
        return
    try: g = geometry(controller, current)
    except (AttributeError, KeyError, TypeError, ValueError): g = None
    row = current.setdefault('releaseHeight', {})
    if g:
        row.update(status='estimated', **g, tipRelativeToEntryM=g['tipM']-g['entryM'])
    else:
        row.update(status='unknown', tipRelativeToEntryM=None, tiltMeasured=False)
    keep = ('status', 'tipRelativeToEntryM', 'tiltDeg', 'tiltSource', 'tiltMeasured',
            'floorSources', 'heightClamped', 'reason')
    compact = {k:round(v, 6) if isinstance(v, float) else v for k,v in row.items() if k in keep}
    current['releaseHeight'] = compact
    report['releaseHeight'] = deepcopy(compact)
