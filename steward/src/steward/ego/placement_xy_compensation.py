"""Observed entrance pairs, grasp-aware placement, one vertical correction.

All targets are robot measurements. Missing imagery is advisory; no acquisition
is performed here. Raw centres survive fitting so residuals remain independent.
"""
from copy import deepcopy
import math
import numpy as np
from steward.ego import capabilities
from steward.ego.component_registration import rotation


def enabled():
    return capabilities.get('STEWARD_ASSEMBLY_PLACEMENT_XY_COMPENSATION') != '0'


def applies(plan):
    return enabled() and len(plan.get('placementTarget', {}).get('holePairs', [])) == 2


def solve(local, points, preferred_yaw):
    """Fixed correspondences: explicitly score the fitted and half-turn poses.

    Unlabelled pair permutations are scored by the caller. A true two-point
    symmetry cannot resolve a directed axis; a tied heading remains a prior.
    """
    a, b = np.asarray(local, float), np.asarray(points, float)[:, :2]
    if a.shape != (2, 2) or b.shape != a.shape or not np.isfinite([a, b]).all():
        return None
    da, db = a[1]-a[0], b[1]-b[0]
    if min(np.linalg.norm(da), np.linalg.norm(db)) <= np.finfo(float).eps:
        return None
    yaw = math.atan2(db[1], db[0])-math.atan2(da[1], da[0])
    candidates = []
    for angle in (yaw, yaw+math.pi):
        r = rotation(angle); centre = b.mean(0)-r @ a.mean(0)
        delta = a @ r.T+centre-b
        residual = np.linalg.norm(delta, axis=1)
        candidates.append(dict(position=centre.tolist(), yaw=angle,
            plannedDeltaM=delta.tolist(), plannedResidualM=residual.tolist(),
            residualSumM=float(residual.sum())))
    best = min(candidates, key=lambda c:(c['residualSumM'],
        abs((c['yaw']-preferred_yaw+math.pi) % (2*math.pi)-math.pi)))
    return dict(best, yawCandidates=deepcopy(candidates))


def retain(reg, points):
    """Save compact raw entrances, never reconstructed fit predictions."""
    reg['observedBodyHoles'] = [dict(bodyHole=p['bodyHole'], **{
        k:deepcopy(h[k]) for k in ('point', 'normal', 'radiusM', 'uncertaintyM', 'relUncM', 'pixel', 'nativeEvidence') if k in h})
        for p, h in zip(reg['holePairs'], points)]
    fit = solve(reg['local'], [h['point'] for h in points], reg['yaw'])
    if fit and reg.get('mode') != 'xy':
        reg.update(fit)
        odom = reg['baseOdometry']
        reg.update(basePositionM=(rotation(odom['yaw']).T @
            (np.asarray(reg['position'])-[odom['x'], odom['y']])).tolist(),
            baseYawRad=reg['yaw']-odom['yaw'])
        reg['yawCandidates'] = [{k:c[k] for k in ('yaw','residualSumM')} for c in reg['yawCandidates']]


def observe(controller, current, sample):
    """Reuse the already acquired hover frame, restricted to the exposed face."""
    from steward.ego import component_registration as cr, component_measurement_admission
    target = current['placementTarget']; ref = target['mateReference']
    pairs = target['holePairs']
    own = {h['id']:h for h in target.get('releasedFrame', {}).get('holes', [])}
    if len(pairs) != 2 or any(p['partHole'] not in own for p in pairs):
        return None, dict(reason='registered_upper_pair_unavailable')
    raw, cam = sample
    local = [own[p['partHole']]['p'][:2] for p in pairs]
    expected = np.asarray(local) @ rotation(target['yawRad']).T+current['spot']
    height = ref.get('landingTopM', ref['position'][2]+ref['size'][2]/2)
    found = cr.detect_native(raw, cam, expected, height)
    # Layer admission precedes visibility; a fitted catalogue never supplies a rim.
    stages = {}
    if component_measurement_admission.enabled():
        reasons = {}; visible = []
        for h in found['holes']:
            clear, why = component_measurement_admission.visibility(controller.rt.estimator, h, raw)
            reasons[why] = reasons.get(why, 0)+1
            if clear: visible.append(h)
        stages = dict(afterDetectionCount=found.get('afterDetectionCount', 0),
            afterFaceCount=len(found['holes']), afterOcclusionCount=len(visible),
            occlusionReasons=reasons, holeRejections=found['holeRejections'],
            detectionReason=found.get('depthReason', found['reason']), searchWindow=found.get('searchWindow'),
            stamp=raw['header']['stamp'], camera=raw['camera'])
        found['holes'] = visible
    else:
        found['holes'] = [h for h in found['holes'] if cr.entry_clear(
            controller.rt.estimator, h['point'], raw['camera'])]
    if raw.get('_stationary_entrance_layers'):
        stages['observedHoles'] = deepcopy(found['holes'])
    fitted, reason = cr.fit_layout(local, found['holes'], expected, target['yawRad'],
                                   controller.ex.config.interface_precision_m)
    if fitted is None:
        return None, dict(reason=(found.get('depthReason', found['reason']) or reason)
            if component_measurement_admission.enabled() else (reason or found['reason']),
            detectedHoles=len(found['holes']), **stages)
    reg = dict(fitted, local=local, holePairs=deepcopy(pairs), body=ref['entity'],
        part=current['object'], observedAt=raw['header']['stamp'], referenceAt=ref['registeredAt'],
        nominalPosition=current['spot'], nominalYaw=target['yawRad'],
        baseOdometry=deepcopy(raw['header']['odometry']), source='current_upward_entrance_pair',
        frame='base_at_observation')
    retain(reg, fitted['points'])
    reg.pop('points', None); reg.pop('indices', None)
    return reg, dict(reason=None, detectedHoles=len(found['holes']), **stages)


def annotate(plan):
    if not applies(plan): return
    target = plan['placementTarget']; reg = target.get('componentHoleRegistration') or {}
    row = dict(schema='placement_xy_compensation.v1', targetSource=reg.get('source', 'baseline_registered_prior'),
        observedAt=reg.get('observedAt'), observedHoleCount=len(reg.get('observedBodyHoles', [])),
        plannedResidualM=None, plannedDeltaM=None, postResidualM=None, replayed=False,
        replayResidualM=None, confirmed=False, reason='unconfirmed_continue',
        gripOffsetM=(np.asarray(plan['gripSpot'])-plan['spot']).tolist())
    if reg.get('observedBodyHoles'):
        actual = np.asarray(reg['local']) @ rotation(target['yawRad']).T+plan['spot']
        lower = np.array([h['point'][:2] for h in reg['observedBodyHoles']])
        row.update(plannedDeltaM=(actual-lower).tolist(),
            plannedResidualM=np.linalg.norm(actual-lower, axis=1).tolist(),
            yawCandidates=deepcopy(reg.get('yawCandidates', [])))
    row.update({k:v for k,v in plan.get('placement_xy_compensation', {}).items() if k.startswith('gripResidual')})
    plan['placement_xy_compensation'] = row


def measure(controller, plan):
    """Current upper entrances or independent outline against stationary bare holes."""
    from steward.ego import component_registration as cr, assembly_precision as ap
    from steward.ego.fasten_release import released_interface
    from steward.ego.fasten_align import outline
    rt = controller.rt; est = rt.estimator
    reg = plan.get('placementTarget', {}).get('componentHoleRegistration') or {}
    unknown = dict(status='unknown', residualM=None, reason='observed_body_pair_unavailable')
    from steward.ego.geometry_relocalization import eligible, current_reference
    ref = current_reference(plan.get('placementTarget'), controller.rt.estimator)
    if ref.get('geometryRelocalization') and not eligible(ref, reg):
        return dict(unknown, reason='geometry_relocalization_required')
    if len(reg.get('observedBodyHoles', [])) != 2: return unknown
    if cr.invalid_reason(est, reg): return dict(unknown, reason='body_moved_since_bare_observation')
    scene = rt.ego_state(); frame = released_interface(scene, plan['object'], reg['body'])
    if not frame: return dict(unknown, reason='release_frame_unavailable')
    after = max(plan['releasedAt'], frame.get('alignmentMovedAt', plan['releasedAt']))
    lower = np.array([h['point'][:2] for h in reg['observedBodyHoles']])
    local = np.array(reg['local']); pairs = reg['holePairs']
    pose = outline(ap.value(scene.find(plan['object']), 'contactView'), frame['position'],
                   np.asarray(frame['rotation']), frame['size'], after, scene.stamp)
    upper = None; source = None; uncertainty = None
    raws = {**getattr(rt, '_precision_native_frames', {}), **getattr(rt, '_latest_native_frames', {})}
    for raw in sorted(raws.values(), key=lambda r:r['header']['stamp'], reverse=True):
        if not after < raw['header']['stamp'] <= scene.stamp: continue
        from steward.ego.inhand import camera_pose
        cam = camera_pose(est, raw['header'], raw['camera'])
        own = {h['id']:h for h in frame['holes']}
        expected = np.array([own[p['partHole']]['point'] for p in pairs])
        found = cr.detect_native(raw, cam, expected[:, :2], float(expected[:, 2].mean()))
        # Upper openings need their own surface: visible lower rims through or
        # beside the upper piece must not falsely report a perfect placement.
        lower_z = np.mean([h['point'][2] for h in reg['observedBodyHoles']])
        found['holes'] = [h for h in found['holes'] if
            h['point'][2]-lower_z > h.get('relUncM', h.get('uncertaintyM', 0.))]
        fit, _ = cr.fit_layout(local, found['holes'], expected[:, :2],
                              plan['placementTarget']['yawRad'], controller.ex.config.interface_precision_m)
        if fit:
            upper = np.array([h['point'][:2] for h in fit['points']])
            source = 'existing_frame_upper_entrances_vs_bare_body_entrances'
            uncertainty = fit['worldUncM']+reg.get('worldUncM', 0.)
            pose = dict(position=[*fit['position'], float(np.mean([h['point'][2] for h in fit['points']])-
                np.mean([own[p['partHole']]['p'][2] for p in pairs]))],
                rotation=[[math.cos(fit['yaw']), -math.sin(fit['yaw']), 0.],
                          [math.sin(fit['yaw']), math.cos(fit['yaw']), 0.], [0., 0., 1.]],
                observedAt=raw['header']['stamp'], source=source)
            break
    if upper is None and pose:
        own = {h['id']:h for h in frame['holes']}
        upper = np.array([np.asarray(pose['position'])+np.asarray(pose['rotation']) @
                          own[p['partHole']]['p'] for p in pairs])[:, :2]
        source = 'existing_upper_outline_vs_bare_body_entrances'
        uncertainty = pose['positionUncertaintyM']+reg.get('worldUncM', 0.)
    if upper is None: return dict(unknown, reason='upper_pair_unseen_continue')
    errors = np.linalg.norm(upper-lower, axis=1)
    diameters = [2*h['radiusM'] for h in reg['observedBodyHoles'] if h.get('radiusM', 0.) > 0]
    limit = min(diameters) if len(diameters) == 2 else ap.tolerance(controller)
    return dict(status='estimated', residualM=float(max(errors)), perHoleM=errors.tolist(),
        deltaM=(upper-lower).tolist(), source=source, observedAt=pose['observedAt'],
        uncertaintyM=uncertainty, limitM=limit,
        limitSource='observed_entrance_diameter' if len(diameters)==2 else 'robot_interface_precision',
        needsCorrection=bool(max(errors)>limit), confirmed=bool(max(errors)+uncertainty<=limit), partPose=pose)


def post(controller, job, plan, params):
    """One shared per-interface replay budget, reserved before any pickup."""
    from steward.ego import assembly_precision as ap
    from steward.ego.skills import SkillFailed
    est = controller.rt.estimator; annotate(plan)
    row = plan['placement_xy_compensation']; reg = plan['placementTarget'].get('componentHoleRegistration') or {}
    body = reg.get('body', params.get('on')); part = plan['object']
    budgets = getattr(est, '_component_alignment_attempts', {})
    est._component_alignment_attempts = budgets; key = (body, part)
    try: measurement = measure(controller, plan)
    except (KeyError, ValueError, TypeError, AttributeError) as exc:
        measurement = dict(status='unknown', residualM=None, reason='measurement_unavailable', detail=type(exc).__name__)
    row.update(postResidualM=measurement['residualM'], post=ap.compact(measurement),
        confirmed=measurement.get('confirmed', False), replayAttempts=budgets.get(key, 0),
        acquisition=deepcopy(getattr(controller, '_precision_acquisition', {})))
    occupied = any(a.get('releaseExecuted') and a.get('part')==part for a in est.fastening_attempts.values())
    from steward.ego.session_initial import occupied as initial
    occupied |= any(initial(body, p['bodyHole']) for p in plan['placementTarget']['holePairs'])
    if measurement.get('needsCorrection') and not budgets.get(key) and not occupied and not est.held_id:
        budgets[key] = 1; row['replayAttempts'] = 1
        with controller.rt._lock:
            ap.install_pose(est, part, measurement['partPose'])
            est.version += 1
        controller._placement_xy_compensation_replay = True
        try:
            controller.step(job, 'pick', dict(target=part, stayNear=True, topGrasp=True))
            corrected = controller.step(job, 'place', dict(params, stayNear=True, closeRelease=True))
            row['replayed'] = bool(corrected.get('releaseExecuted'))
            result = corrected.get('placement_xy_compensation') or {}
            row.update(replayResidualM=result.get('postResidualM'), replayPost=deepcopy(result.get('post')),
                confirmed=result.get('confirmed', False))
        except SkillFailed as exc:
            controller.check(); row.update(replayFailure=str(exc), confirmed=False)
            # Missing imagery never throws; a physical admission failure remains
            # distinct. If pickup succeeded, finish the already admitted estimate.
            if est.held_id:
                receipt = controller.rt.predict('place', dict(params, stayNear=True, closeRelease=True))
                if receipt.get('withinRiskLimit'):
                    corrected = controller.step(job, 'place', dict(params, stayNear=True, closeRelease=True), receipt=receipt)
                    row['replayed'] = bool(corrected.get('releaseExecuted'))
                else:
                    raise
        finally:
            controller._placement_xy_compensation_replay = False
    row['reason'] = 'within_observed_control_bound' if row['confirmed'] else 'unconfirmed_continue'
    from steward.ego.geometry_relocalization import provenance
    row.update(provenance(plan.get('placementTarget'), plan.get('releasedAt'), est))
    job.postcondition['placementXYCompensation'] = deepcopy(row)
    plan.setdefault('precision', {}).update(placementXYCompensation=deepcopy(row), postRelease=deepcopy(row),
        postReleaseResidualM=row['postResidualM'], postCorrectionCount=int(row['replayed']))
    stored = est.released_interfaces.get(part)
    if stored is not None: stored['placementXYCompensation'] = deepcopy(row)


def finish(ex, plan):
    """One encoder correction above the body, followed by vertical descent."""
    if not applies(plan): return
    row = plan.setdefault('placement_xy_compensation', {}); goal = np.asarray(plan.get('gripSpot', plan.get('spot')))
    try:
        row['gripResidualBeforeM'] = float(np.linalg.norm(ex._gripper_xy()-goal))
    except (KeyError, AttributeError, TypeError, ValueError):
        row.update(reason='encoder_geometry_unavailable_continue', confirmed=False)
        return
    if row['gripResidualBeforeM'] > ex.config.interface_precision_m:
        try:
            lift = float(ex._state()['joints']['lift'])
            clear = plan['placementXYCompensationClearLift']
        except (KeyError, AttributeError, TypeError):
            row.update(reason='encoder_geometry_unavailable_continue', confirmed=False)
            return
        ex._set_joint('lift', max(lift, clear))
        ex._align_lateral(goal, tol=ex.config.interface_precision_m)
        ex._set_joint('reach', ex._reach_to(goal), tol=ex.config.interface_precision_m)
        ex._set_joint('lift', lift)
    row['gripResidualAfterM'] = float(np.linalg.norm(ex._gripper_xy()-goal))


def clear_before_dock(ex, plan):
    if not applies(plan) or plan.get('placementXYCompensationClearLift') is None: return
    try: lift = float(ex._state()['joints']['lift'])
    except (KeyError, AttributeError, TypeError):
        plan.setdefault('placement_xy_compensation', {}).update(reason='encoder_geometry_unavailable_continue', confirmed=False)
        return
    ex._set_joint('lift', max(lift, plan['placementXYCompensationClearLift']))
