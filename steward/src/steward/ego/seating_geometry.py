"""Consistent entry geometry and bounded delivery using existing robot evidence.

No acquisition, world state, semantic part names or task dimensions. A missing
measurement retains the best admitted estimate and never vetoes an open.
"""
from copy import deepcopy
import math
import numpy as np
from steward.ego import capabilities
from steward.ego.fasten_geometry import transform_feature


def enabled():
    return capabilities.get('STEWARD_ASSEMBLY_SEATING_GEOMETRY') != '0'


def applies(current):
    """Only an identified fastener mouth owns the bounded delivery policy."""
    return (enabled() and bool(current.get('seatHole') and current.get('seatPart'))
            and current.get('entryTargetKind') != 'head')


def point(value):
    return value is not None and np.shape(value) == (3,) and np.isfinite(value).all()


def remember(controller, current, frame, header):
    """Bind an existing image to its own encoders; synchronize only this mouth."""
    if not enabled(): return
    est = controller.rt.estimator
    if (not current.get('seatHole') or not current.get('seatPart')
            or not hasattr(est, 'released_interfaces')): return
    stamp = header['stamp']
    if est.held_id != current.get('object') or stamp < est.held_since: return
    frame['stamp'] = stamp  # Entry-only frames also need their true acquisition time.
    entry = frame.get('entry') or {}
    target = current.get('seatTarget') or {}
    saved = target.get('componentFrame') or {}
    live = est.released_interfaces.get(current['seatPart'], {})
    epoch = max(component.get(k, -math.inf) for component in (saved, live)
                for k in ('releasedAt', 'alignmentMovedAt', 'transportedAt'))
    if stamp < epoch: return
    identity = current.get('seatHole')
    if (point(entry.get('point')) and not frame.get('targetFromLayout')
            and not entry.get('targetFromLayout')
            and entry.get('targetHole', identity) == identity):
        current['seatTarget'] = dict(target, **deepcopy(entry), observedAt=stamp,
            targetSource='existing_frame_upper_entry', entryObserved=True)
        update = dict(point=list(entry['point']), upperPoint=list(entry['point']),
            observedAt=stamp, targetSource='existing_frame_upper_entry',
            pairDistanceM=(target.get('pairAlignment') or {}).get('distanceXYM'))
        # Persist the accepted entry without translating unseen holes or changing
        # their identity. A new component release/motion invalidates it normally.
        frames = [saved, live]
        ctx = getattr(controller, '_registered_alignment', {}) or {}
        reg = ctx.get('registration') or {}
        cache = getattr(est, '_seat_registrations', {}).get(current.get('seatPart')) or {}
        for registration in (reg, cache):
            if (registration.get('body') != current.get('seatBody') or
                    registration.get('entity') != current.get('seatPart')): continue
            layout = registration.get('layout', [])
            selected = next((row for row in layout if row.get('bodyHole') == identity), {})
            old_point = selected.get('point')
            raw_holes = registration.get('holes', [])
            if old_point is not None and raw_holes:
                # Keep the passive motion watch in the same local reference;
                # updating one visible mouth must not refresh unseen peers.
                nearest = min(raw_holes, key=lambda h:np.linalg.norm(np.asarray(h['point'])-old_point))
                from steward.ego.fasten_target import spacing
                separation = spacing(layout)
                if separation and np.linalg.norm(np.asarray(nearest['point'])-old_point) < separation/2:
                    nearest.update(point=list(entry['point']), observedAt=stamp)
            for row in layout:
                row.setdefault('observedAt', registration.get('observedAt'))
                if row.get('bodyHole') == identity:
                    row.update(point=list(entry['point']), observedAt=stamp, measured=True,
                               targetSource='existing_frame_upper_entry')
            if registration.get('hole') == identity:
                registration.update(point=list(entry['point']), executionPoint=list(entry['point']),
                                    observedAt=stamp)
                ctx.pop('correctionPoint', None)
            frames.append(registration.get('frame') or {})
        for component in frames:
            if component:
                component.setdefault('registeredSeatTargets', {})[identity] = deepcopy(update)
        current['seatingGeometryEntry'] = dict(point=list(entry['point']), observedAt=stamp,
            source=entry.get('source', 'existing_frame_recess'), unconfirmed=True)
    lower = frame.get('lowerEnd') or {}
    if frame.get('bothVisible') and point(lower.get('tip')) and point(lower.get('head')):
        rotation = np.asarray(est.gripper_rotation(est.held_gripper, header['joints'], header['odometry']))
        grip = np.asarray(est.gripper_position(est.held_gripper, header['joints'], header['odometry']))
        local = transform_feature(lower, rotation.T, -rotation.T @ grip)
        current['_seatingGeometryEndpoint'] = dict(local=local, heldSince=est.held_since,
                                      observedAt=stamp, source='existing_frame_lower_end')


def endpoint(controller, current):
    est = controller.rt.estimator
    if est.held_id != current.get('object'): return None
    saved = current.get('_seatingGeometryEndpoint') or {}
    local = saved.get('local') if saved.get('heldSince') == est.held_since else None
    track = est.tracks.get(current.get('object'))
    local = local or getattr(track, 'held_tool_feature', None) or getattr(track, 'held_grasp_geometry', None)
    if not local: return None
    state = controller.ex._state()
    feature = transform_feature(local,
        est.gripper_rotation(est.held_gripper, state['joints'], state['odometry']),
        est.gripper_position(est.held_gripper, state['joints'], state['odometry']))
    if not point(feature.get('tip')) or not point(feature.get('head')): return None
    return state, feature, ('existing_frame_lower_end_plus_fk' if saved.get('local') is local
                            else 'held_geometry_plus_fk')


def delivery_scale(feature, target):
    """Precision follows the measured shaft and length, never a snap radius."""
    length = float(np.linalg.norm(np.asarray(feature['head'])-feature['tip']))
    radius = float(feature.get('shaftRadiusM') or feature.get('radiusM') or length/4)
    if length <= 0 or radius <= 0: raise ValueError('endpoint_scale_unresolved')
    numeric = np.finfo(float).eps * max(1., abs(target['point'][2]))
    noise = float(target.get('planeNoiseM') or target.get('residualM') or radius/8)
    gap = max(numeric, min(noise, length/32, radius/4))
    return length, radius, gap


def release_geometry(controller, current, prior):
    """Use the delivered endpoint as the result-view search prior, not truth.

    The lower-shaft image can omit the head. Preserve the acquired full length
    and radius, explicitly extrapolating the head from the transported lower end.
    """
    if not enabled() or not current.get('_seatingGeometryEndpoint'): return prior
    read = endpoint(controller, current)
    if read is None or read[2] != 'existing_frame_lower_end_plus_fk': return prior
    feature = read[1]
    axis = np.asarray(feature['head'])-feature['tip']
    length = np.linalg.norm(np.asarray(prior['head'])-prior['tip'])
    if np.linalg.norm(axis) <= 0 or length <= 0: return prior
    axis /= np.linalg.norm(axis)
    return dict(prior, tip=list(feature['tip']),
        head=(np.asarray(feature['tip'])+length*axis).tolist(), axis=axis.tolist(),
        source='existing_frame_lower_end_plus_held_length_fk', headExtrapolated=True)


def finish(controller, current, report):
    """One final alignment from existing geometry, then a close axial release.

    A bounded encoder correction is not another camera observation. Failed
    motion remains unconfirmed and the caller still executes the normal open.
    """
    if not enabled() or current.get('seatingGeometryDelivery'): return
    if current.get('releaseHeight', {}).get('policy') == 'observed_plane_height_only':
        return  # An explicitly enabled observed-plane controller already owns this endpoint.
    from steward.ego.skills import SkillFailed
    from steward.ego.fasten_seat import lift_to
    from steward.ego.tip_entry_servo import move
    row = dict(policy='existing_geometry_bounded_delivery', unconfirmed=True,
               observationsAdded=0, motions=[])
    current['seatingGeometryDelivery'] = row
    target = current.get('seatTarget') or {}
    try:
        read = endpoint(controller, current)
        if read is None or not point(target.get('point')):
            row['reason'] = 'endpoint_unavailable_best_estimate_release'
            return
        state, feature, source = read
        length, radius, gap = delivery_scale(feature, target)
        entry = np.asarray(target['point'], float)
        delta = entry[:2]-np.asarray(feature['tip'])[:2]
        row.update(entry=entry.tolist(), entryObservedAt=target.get('observedAt'),
            entrySource=target.get('targetSource'), endpointSource=source,
            beforeXYM=float(np.linalg.norm(delta)), beforeHeightM=float(feature['tip'][2]-entry[2]))
        if np.linalg.norm(delta) > radius/4:
            # Clear the mouth and neighbours before any planar correction. The
            # tool orientation stays fixed; no sweep through the newly placed head.
            track = controller.rt.estimator.tracks.get(current.get('object'))
            acquired = (getattr(track, 'held_tool_feature', None) or
                        getattr(track, 'held_grasp_geometry', None) or {})
            full_length = (float(np.linalg.norm(np.asarray(acquired['head'])-acquired['tip']))
                if point(acquired.get('head')) and point(acquired.get('tip')) else length)
            # A close frame may expose only a short lower shaft. Clearance must
            # still cover the full carried length acquired before that occlusion.
            clearance = max(length, full_length, controller.rt.esim.arm.geometry.jaw_opening/2,
                            float(np.linalg.norm(delta)))
            raised = state['joints']['lift'] + max(0., entry[2]+clearance-feature['tip'][2])
            if raised > state['joints']['lift']:
                joints = dict(state['joints'], lift=raised)
                if controller.rt.esim.arm.reachable_path([state['joints'], joints], start_joints=state['joints']):
                    try:
                        lift_to(controller.ex, raised, ramp=False, tolerance=gap)
                    except SkillFailed as exc:
                        controller.check()
                        row['clearanceFailure'] = str(exc)
                        reached = endpoint(controller, current)
                        if reached is None or reached[1]['tip'][2] < entry[2]+clearance:
                            delta = np.zeros(2)
                else:
                    row['reason'] = 'clearance_unreachable_best_estimate_release'
                    delta = np.zeros(2)
            if np.linalg.norm(delta):
                # One finite correction; never chase vision or spend more stock.
                bound = min(length, controller.rt.esim.arm.geometry.jaw_opening/4)
                delta *= min(1., bound/np.linalg.norm(delta))
                motion = move(controller, delta, radius/4)
                row['motions'].append(motion)
        read = endpoint(controller, current)
        if read is None: return
        state, feature, _ = read
        desired = state['joints']['lift'] + entry[2]+gap-feature['tip'][2]
        # Every actuator command is bounded against the same selected entry.
        floor = desired-gap-length/32
        path = [state['joints'], dict(state['joints'], lift=desired)]
        if controller.rt.esim.arm.reachable_path(path, start_joints=state['joints']):
            # A timed-out command may have moved. Retry the same endpoint once,
            # using reached encoders rather than treating the setpoint as truth.
            for _ in range(2):
                motion = dict(kind='height', targetLift=desired, completed=False)
                row['motions'].append(motion)
                try:
                    lift_to(controller.ex, desired, ramp=False, tolerance=gap/2, floor=floor)
                    motion['completed'] = True
                    break
                except SkillFailed as exc:
                    controller.check()
                    motion['reason'] = str(exc)
        else:
            row['reason'] = 'height_unreachable_best_estimate_release'
    except (AttributeError, KeyError, TypeError, ValueError, SkillFailed) as exc:
        controller.check()  # Cancellation and exhausted action budgets still propagate.
        row['reason'] = 'delivery_unconfirmed:'+str(exc)
    finally:
        try:
            read = endpoint(controller, current)
            if read and point(target.get('point')):
                state, feature, source = read
                axis = np.asarray(feature['head'])-feature['tip']; axis /= np.linalg.norm(axis)
                row.update(afterXYM=float(np.linalg.norm(np.asarray(feature['tip'])[:2]-target['point'][:2])),
                    tipRelativeToEntryM=float(feature['tip'][2]-target['point'][2]),
                    tiltDeg=math.degrees(math.acos(float(np.clip(axis[2], -1., 1.)))),
                    endpointSource=source, checkedAt=state['stamp'])
                current['releaseGapM'] = row['tipRelativeToEntryM']
        except (AttributeError, KeyError, TypeError, ValueError): pass
        current['nearContactUnconfirmed'] = True
        if point(target.get('point')):
            current['spot'] = list(target['point'][:2])
        report['seatingGeometryDelivery'] = deepcopy(row)


def pick_clearance(planner, scene, target, support, height):
    """Raise the empty hand above the observed stock before extending it."""
    if not enabled() or support is None: return None
    from steward.ego.planner import entity_box
    from steward.ego.priors import prior
    box, _, _ = entity_box(support)
    opening = planner.arm.geometry.jaw_opening
    if not opening: return None
    tops = [entity_box(target)[2]]
    for other in scene.entities:
        if other.id in (target.id, support.id, scene.self_id): continue
        if prior(other.category.value).get('assemblyRole') != 'fastener': continue
        b, _, high = entity_box(other)
        xy = np.asarray([b.cx-box.cx, b.cy-box.cy])
        c,s = math.cos(box.yaw), math.sin(box.yaw)
        if abs(c*xy[0]+s*xy[1]) <= box.hx and abs(-s*xy[0]+c*xy[1]) <= box.hy:
            tops.append(high)
    return min(planner.lift_max, max(height, max(tops)+opening))
