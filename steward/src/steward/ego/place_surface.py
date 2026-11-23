"""Fresh native contact surfaces for assembly placement, with explicit fallback.

Planes are observations; gravity settling and a held part's underside remain
estimates. A measured normal sets the clearance envelope, not an invented yaw.
"""
from steward.ego import capabilities
from steward.ego import action_budget as ab
from copy import deepcopy
import math
import numpy as np
from steward.math3d import matrix_from_quat, quat_from_yaw
from steward.ego.skills import SkillFailed


class SurfaceMoved(SkillFailed):
    """A measured plane followed by positive body motion evidence."""

MAX_AGE_S = 90.
LEVEL_LIMIT_DEG = 5.


def height_at(patch, xy):
    p, n = np.asarray(patch['point']), np.asarray(patch['normal'])
    return float(p[2]-np.dot(n[:2], np.asarray(xy)-p[:2])/n[2])


# Experiment switch: 0 keeps every placement on its pre-measurement path.
ENABLED = capabilities.get('STEWARD_PLACE_SURFACE') != '0'
FASTENER_ENTRY_SURFACE = False  # fastener release heights keep their verified registered-entry path


def fallback_margin(record):
    """Unknown support posture: bound vertical rise by what can stand under the body.

    A body resting on one of its registered horizontal faces is lifted by at
    most the vertical extent by which attached parts protrude beyond those
    faces (either face may be the lower one after a turn). No task-specific
    protrusion or pose constants enter this bound.
    """
    from steward.ego.assembly_edge import corners
    ref = record['reference']; origin = np.asarray(ref['position'])
    cov = (getattr(record.get('entity'), 'pose', None).unc or {}).get('cov') if record.get('entity') else None
    position = 2*math.sqrt(max(0., float(cov[14]))) if cov and len(cov) > 14 else 0.
    base = max(position, .002)
    # Before any transport use the registration's own positional uncertainty;
    # there is no new unseen placement to extrapolate.
    if ref.get('ownReleaseAt') is None and not record.get('parts') and not record.get('surfaceMotion'):
        return base
    body = corners(ref, origin, np.eye(3))[:, 2]
    parts = [corners(item, origin, np.eye(3))[:, 2] for item in record.get('parts', [])]
    protrusion = max([0.]+[max(float(z.max()-body.max()), float(body.min()-z.min())) for z in parts])
    return base+protrusion+.003


def observe(controller, record, after, raw=None):
    from steward.ego import assembly_edge_observation as obs
    ref = record['reference']
    record = dict(record, surfaceMotion=any(e['entity'] == ref['entity']
        for e in controller.rt.estimator.stationary_assemblies.motion_events))
    nominal = float(ref.get('landingTopM', ref['position'][2]+ref['size'][2]/2))
    controller._edge_observation_diag = {}
    result = dict(source='assumed', observed=False, nominalHeightM=nominal,
        observedMinusAssumedM=None, normal=None, tiltDeg=None, levelFirst=None,
        uncertaintyM=fallback_margin(record), observedAt=None,
        referenceAt=ref['registeredAt'], entity=ref['entity'], checkedAt=controller.ex._state()['stamp'])
    # The carried load is an occluder too, even if it projects into this ROI.
    est = controller.rt.estimator
    held = est.tracks.get(est.held_id)
    boxes = []
    if held is not None:
        pose = est.held_pose().get(held.id)
        if pose is not None:
            centre, yaw = pose
            interface = held.held_interface
            rotation = (est.gripper_rotation(est.held_gripper) @ np.asarray(interface['rotation'])
                        if interface else matrix_from_quat(quat_from_yaw(yaw)))
            boxes.append((np.asarray(centre), held.size/2+math.sqrt(float(np.max(held.var))), rotation))
    from steward.ego.skills import SkillFailed
    try:
        view = obs.native_view(controller, record, after, face=True, raw=raw, exclude_boxes=boxes)
    except (SkillFailed, ValueError, KeyError, TypeError) as exc:
        controller.check()
        result.update(reason='native_acquisition_failed', detail=type(exc).__name__)
        return result
    diag = controller._edge_observation_diag
    fit_diag = {}; patch = obs.body_patch(view['points'], ref, fit_diag) if view else None
    result['diagnostics'] = dict(native=diag.get('native', {}), fit=fit_diag)
    if patch is None:
        result['reason'] = fit_diag.get('reason') or diag.get('native', {}).get('reason', 'no_local_plane')
        return result
    normal = np.asarray(patch['normal'])
    tilt = math.degrees(math.acos(float(np.clip(normal[2], -1, 1))))
    result.update(source='observed_native_plane', observed=True, patch=patch,
        observedAt=view['stamp'], normal=normal.tolist(), tiltDeg=tilt,
        levelFirst=tilt > LEVEL_LIMIT_DEG, thresholdDeg=LEVEL_LIMIT_DEG,
        observedMinusAssumedM=patch['point'][2]-nominal,
        uncertaintyM=max(patch['residualM'], 1e-6), reason='fresh_local_plane')
    return result


def valid(surface, reference, stamp):
    # A direct native capture can be newer than the asynchronous scene clock.
    stamp = max(stamp, (surface or {}).get('checkedAt', -math.inf))
    return bool(surface and surface.get('referenceAt') == reference['registeredAt']
        and 0 <= stamp-surface.get('checkedAt', -math.inf) <= MAX_AGE_S
        and (not surface.get('observed') or
             reference.get('ownReleaseAt', -math.inf) < surface['observedAt'] <= stamp
             and stamp-surface['observedAt'] <= MAX_AGE_S))


def placement_envelope(surface, xy, half, yaw, nominal, gap, held_uncertainty, *, advisory=False):
    """Horizontal load clears the highest part of the measured contact plane.

    No wrist attitude is guessed: steep surfaces request levelling. Unknown
    surfaces retain the registered height plus an explicit geometry bound.
    """
    report = deepcopy(surface)
    centre = float(nominal)
    rise = 0.
    if surface['observed']:
        centre = height_at(surface['patch'], xy)
        n = np.asarray(surface['normal'])
        slope = -n[:2]/n[2]
        r = matrix_from_quat(quat_from_yaw(yaw))[:2, :2]
        rise = float(abs(r.T @ slope) @ np.asarray(half))
    top = centre+rise
    # Includes the load's inferred underside and the existing joint move tolerance.
    clearance = max(gap, .002)+surface['uncertaintyM']+held_uncertainty
    if advisory:
        # XY grasp uncertainty is not a vertical clearance estimate. The added
        # plane allowance is twice its residual, capped at the existing gap.
        extra = min(2*surface['uncertaintyM'], max(gap, .002)) if surface['observed'] else 0.
        clearance = gap+extra
        report.update(addedMarginM=extra, marginSource='bounded_plane_residual'
                      if surface['observed'] else 'baseline_unchanged')
    report.update(surfaceHeightM=centre, highestContactM=top,
        footprintRiseM=rise, observedMinusAssumedM=centre-nominal if surface['observed'] else None,
        releaseMarginM=clearance, releaseBottomM=top+clearance,
        heldGeometryUncertaintyM=held_uncertainty,
        orientationSource='observed_surface_normal' if surface['observed'] else 'assumed_horizontal')
    return top, clearance, report


def refresh(controller, job, plan, params, *, after=None, raw=None):
    """Observe at the destination, install evidence, and consume a new risk receipt."""
    from steward.ego.skills import SkillFailed
    rt, est = controller.rt, controller.rt.estimator
    from steward.ego import placement_xy_compensation
    if placement_xy_compensation.applies(plan) and getattr(controller, '_placement_xy_compensation_replay', False):
        plan['surfaceRefreshed'] = True
        return plan
    body_id = params.get('seatBody') or params.get('on')
    record = est.stationary_assemblies.records.get(body_id)
    if not ENABLED:
        return plan
    if record is None or params.get('flip') or params.get('roll') or params.get('assemblyBody'):
        return plan
    # Seating existing frames stays on its original path. New component releases
    # carry this surface provenance and refresh their own upper face here.
    frame = None
    if params.get('seatBody'):
        if not FASTENER_ENTRY_SURFACE:
            return plan
        frame = est.released_interfaces.get(params.get('on'))
        if not frame or not frame.get('placementSurface'):
            return plan
        ref = dict(entity=frame['entity'], position=frame['position'], size=frame['size'],
            yaw=math.atan2(frame['rotation'][1][0], frame['rotation'][0][0]),
            registeredAt=frame['releasedAt'], ownReleaseAt=frame['releasedAt'],
            support=record['reference']['support'])
        measured_record = dict(reference=ref, parts=[])
    else:
        measured_record = record
    if raw is None or after is None or raw['header']['stamp'] <= after:
        if getattr(rt, 'segmentation', None) is not None or getattr(rt, '_precision_native_frames', None):
            controller.ex._dock(job, plan)
        after = max(after if after is not None else -math.inf, controller.ex._state()['stamp'])
        raw = None
    after = max(after, measured_record['reference'].get('ownReleaseAt', -math.inf),
                max((e['stamp'] for e in est.stationary_assemblies.motion_events
                     if e['entity'] in (body_id, params.get('on'))), default=-math.inf))
    if raw is not None and raw['header']['stamp'] <= after:
        raw = None
    controller.check()
    measured = observe(controller, measured_record, after, raw)
    measured['body'] = body_id
    measured['motionAfter'] = after
    measured['checkedAt'] = controller.ex._state()['stamp']
    with rt._lock:
        saved = {k:v for k,v in measured.items() if k != 'diagnostics'}
        if frame is not None:
            frame['entrySurface'] = saved
        else:
            record['reference']['placementSurface'] = saved
        est.version += 1
    job.postcondition['placementSurface'] = deepcopy(measured)
    if measured.get('levelFirst'):
        controller.warn('contact_surface_requires_levelling', tiltDeg=measured['tiltDeg'], thresholdDeg=LEVEL_LIMIT_DEG)
    receipt = rt.predict('place', params)
    if not receipt.get('withinRiskLimit'):
        from steward.ego.assembly_destination import ENABLED as destination_enabled
        if destination_enabled and measured.get('observed'):
            # A known raised/tilted contact face cannot become missing evidence
            # just because reaching it is inconvenient. Keep it for the caller.
            measured.update(reason='surface_corrected_place_not_admitted')
            job.postcondition['placementSurface'] = deepcopy(measured)
            job.postcondition['supportTargetCheck'] = dict(status='failed', passed=False,
                reason='observed_surface_target_not_admitted', body=body_id, issues=receipt.get('issues', []))
            raise SkillFailed('surface-corrected place not admitted; observed support retained: '+
                              '; '.join(receipt.get('issues', [])))
        # Missing metrology retains the existing best-estimate fallback.
        measured.update(observed=False, source='assumed', reason='surface_corrected_place_not_admitted')
        with rt._lock:
            destination = frame if frame is not None else record['reference']
            destination['entrySurface' if frame is not None else 'placementSurface'] = deepcopy(measured)
            est.version += 1
        job.postcondition['placementSurface'] = deepcopy(measured)
        receipt = rt.predict('place', params)
        if not receipt.get('withinRiskLimit'):
            raise SkillFailed('surface-corrected and baseline place not admitted: '+'; '.join(receipt.get('issues', [])))
    replacement = rt._consume_receipt(receipt)
    deadline = getattr(controller.ex, '_primitive_deadline', None)
    controller.ex._begin(job, 'place', receipt)
    if deadline is not None:
        controller.ex._primitive_deadline = ab.limited(deadline, controller.ex._primitive_deadline)
    replacement['surfaceRefreshed'] = True
    job.postcondition['placementSurface'] = deepcopy(replacement.get('placementSurface', measured))
    return replacement


def validate(rt, plan):
    """Only positive body motion AFTER a plane observation can veto lowering.

    Age, reference revisions, old hardMotion flags, support/held motion and
    asynchronous perception lag are not evidence that this measured body moved.
    """
    surface = plan.get('placementSurface')
    if not ENABLED or not surface or not surface.get('observed'):
        return
    est = rt.estimator
    record = est.stationary_assemblies.records.get(surface['body']) or {}
    after = surface['observedAt']
    identities = {surface['body'], *record.get('aliases', [])}
    events = [e for e in est.stationary_assemblies.motion_events
              if e['entity'] in identities and e['stamp'] > after]
    displacement = record.get('displacementEvidence', {})
    condition = ('body_motion_event_after_observation' if events else
                 'body_displacement_after_observation'
                 if displacement.get('stamp', -math.inf) > after else None)
    surface['validation'] = dict(condition=condition, observedAt=after,
        estimatorAt=est.stamp, checkedAt=surface['checkedAt'])
    if condition:
        surface['validation']['motionAt'] = (max(e['stamp'] for e in events)
                                             if events else displacement['stamp'])
        raise SurfaceMoved('contact surface changed: '+condition+'; before lowering/release')
