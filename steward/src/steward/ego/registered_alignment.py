"""Time-separated RGB-D registration and encoder-only release alignment.

All positions are in Ego's odometry map; a base-frame snapshot is also retained.
No contact, insertion or mechanical fastening is inferred from FK residuals.
"""
from steward.ego.phase_timing import timed
from steward.ego import action_budget as ab
from copy import deepcopy
import math
import time
import numpy as np

from steward.ego.fasten_plan import value
from steward.ego.fasten_release import released_interface
from steward.ego.fasten_align import outline
from steward.ego.fasten_geometry import surface_patch, transform_feature
from steward.ego.skills import SkillFailed
from steward.math3d import matrix_from_quat, quat_from_yaw


MOVEMENT_M = .004
# Age alone is a weak staleness signal: motion of the support, the interface or the
# registered features invalidates a registration on its own. The limit only bounds
# unobserved drift, and must outlast one carry-dock-align-lower cycle of the arm.
MAX_AGE_S = 300.



def arm_clear(est, point, camera):
    """Robot collision geometry must leave the camera-to-entry ray clear."""
    origin = np.asarray(est.camera_pose(camera=camera)['p'])
    distance = np.linalg.norm(np.asarray(point)-origin)
    if distance <= .10:
        return False
    ray = origin + np.linspace(.08/distance, .98, 100)[:, None]*(np.asarray(point)-origin)
    for centre, half, rotation in est.body_boxes():
        if np.any(np.all(abs((ray-centre) @ rotation) <= np.asarray(half)+.003, axis=1)):
            return False
    return True


def pixel_evidence(raw, cam, point):
    """Native image scale and local dark-region span; not a hole-centre detector.

    No physical hole diameter is present in the catalogue. A bounded RGB contrast
    region is diagnostic only and may be a mark/shadow. Missing contrast is null.
    """
    rgb = raw['rgb']; height, width = rgb.shape[:2]
    local = matrix_from_quat(cam['q']).T @ (np.asarray(point)-cam['p'])
    out = dict(camera=raw['camera'], width=int(width), height=int(height),
               native=True, holeObserved=False, darkRegionSpanPx=None)
    if local[0] <= 0:
        return out
    focal = height/(2*math.tan(math.radians(raw['fovy'])/2))
    out['metresPerPixel'] = float(local[0]/focal)
    u, v = width/2-focal*local[1]/local[0], height/2-focal*local[2]/local[0]
    radius = max(3, min(24, math.ceil(.008/out['metresPerPixel'])))
    x, y = int(round(u)), int(round(v))
    if x-radius < 0 or y-radius < 0 or x+radius >= width or y+radius >= height:
        return out
    crop = rgb[y-radius:y+radius+1, x-radius:x+radius+1].astype(float).mean(axis=2)
    threshold = np.median(crop)*.45
    dark = crop < threshold
    # Report a bounded dark-pixel span, never a contrast region crossing the crop edge.
    yy, xx = np.where(dark)
    if len(xx) and not (dark[0].any() or dark[-1].any() or dark[:,0].any() or dark[:,-1].any()):
        out['darkRegionSpanPx'] = [int(np.ptp(xx)+1), int(np.ptp(yy)+1)]
    return out


def sample(controller, plan, after):
    scene = controller.rt.ego_state(); est = controller.rt.estimator
    target_track = est.tracks.get(plan['part'])
    if target_track is None or target_track.held:
        return None, 'target_not_released_or_lost'
    f = released_interface(scene, plan['part'], plan['body'])
    if not f:
        return None, 'no_released_interface'
    view = value(scene.find(plan['part']), 'contactView')
    pose = outline(view, f['position'], np.asarray(f['rotation']), f['size'],
                   max(after, f['releasedAt']), scene.stamp)
    if not pose:
        return None, 'no_complete_post_settle_outline'
    camera = view.get('camera')
    native = view.get('nativeResolution')
    if not camera or not native:
        return None, 'native_resolution_unknown'
    pair = next((p for p in f['holePairs'] if p['bodyHole'] == plan['hole']), None)
    hole = next((h for h in f['holes'] if pair and h['id'] == pair['partHole']), None)
    if not hole:
        return None, 'hole_correspondence_missing'
    point = np.asarray(pose['position'])+np.asarray(pose['rotation']) @ hole['p']
    if not arm_clear(est, point, camera):
        return None, 'robot_occludes_entry_ray'
    patch = surface_patch(view['points'], point[:2])
    if not patch:
        return None, 'no_local_entry_surface'
    point = np.asarray(patch['point'])
    raw = getattr(controller.rt, '_precision_native_frames', {}).get(camera)
    pixels = (pixel_evidence(raw, est.camera_pose(raw['header']['odometry'], camera), point)
              if raw is not None and raw['header']['stamp'] == view['stamp'] else
              dict(camera=camera, width=native[0], height=native[1], native=True,
                   holeObserved=False, darkRegionSpanPx=None, metresPerPixel=None))
    radius = float(np.linalg.norm(hole['p'][:2]))
    sigma = pose['positionUncertaintyM']+radius*pose['angleUncertaintyRad']
    pixel = pixels.get('metresPerPixel')
    sigma += pixel if pixel is not None else .003
    result = deepcopy(f)
    result.update(position=pose['position'], rotation=pose['rotation'], observedAt=pose['observedAt'],
                  source=pose['source'], outlineObserved=True, observed=False)
    for h in result['holes']:
        h['point'] = (np.asarray(pose['position'])+np.asarray(pose['rotation']) @ h['p']).tolist()
    o = controller.ex._state()['odometry']; base = np.array([o['x'],o['y'],0.])
    return dict(point=point.tolist(), basePointM=(matrix_from_quat(quat_from_yaw(o['yaw'])).T @ (point-base)).tolist(),
        baseOdometry=deepcopy(o), frame=result, pose=pose, pixels=pixels,
        observedAt=pose['observedAt'], uncertaintyM=sigma, surfaceUncertaintyM=patch['residualM'],
        rawCentreM=deepcopy(view.get('centreM')), rawBearingRad=bearing(view),
        source='rgbd_outline_plus_local_hole_template', holeObserved=False,
        entity=plan['part'], body=plan['body'], hole=plan['hole']), None


def closer_view(controller, job, plan):
    """Use the existing checked observation approach when no camera is close enough."""
    from steward.ego.fasten_plan import support_anchor
    from steward.ego.planner import entity_box
    scene = controller.rt.ego_state()
    support = support_anchor(controller.rt.esim, scene, scene.entity(plan['body']), [])
    box, _, _ = entity_box(support)
    o = controller.ex._state()['odometry']
    previous_target = getattr(controller, 'observation_target', None)
    previous_close = getattr(controller, '_close_view', False)
    controller.observation_target = scene.entity(plan['part']).pose.value['p']
    controller._close_view = True
    try:
        controller.view(job, support, math.atan2(o['y']-box.cy, o['x']-box.cx), 0.)
    finally:
        controller._close_view = previous_close
        if previous_target is None:
            del controller.observation_target
        else:
            controller.observation_target = previous_target


@timed("hole_registration")
def register(controller, job, plan):
    from steward.ego.support_registration import register as support_register
    ctx = support_register(controller, job, plan)
    from steward.ego.fasten_target import apply
    with controller.rt._lock:
        apply(controller, job, plan, ctx)
    from steward.ego.assembly_precision import publish
    publish(job)
    return ctx


def register_outline(controller, job, plan):
    """Two fresh, arm-clear native samples after release; renewed before each pick."""
    from steward.ego.assembly_precision import acquire
    context = dict(mode='registered_hole_fk_v1', holeRegistrationAttempts=1,
        holeRegistrationSuccesses=0, tipOffsetAttempts=0, tipOffsetMeasuredCount=0,
        registration=None, tip=None, reason=None)
    job._registration_serial = getattr(job, '_registration_serial', 0)+1
    context['registrationId'] = str(job.id)+':'+str(job._registration_serial)
    controller._registered_alignment = context
    # Keep attempted registrations visible even if picking or prediction fails.
    report = dict(kind='screw_registration', object=None, body=plan['body'], hole=plan['hole'],
                  observations=[], corrections=[], releaseExecuted=False)
    context['report'] = report
    if not hasattr(job, '_precision_reports'):
        job._precision_reports = []
    job._precision_reports.append(report)
    attach(controller, report)
    rows = job.postcondition.setdefault('registrationDiagnostics', [])
    row = dict(entity=plan['part'], hole=plan['hole'], status='unknown')
    rows.append(row)
    samples = []
    for _ in range(2):
        controller.check()
        after = acquire(controller, [plan['part']])
        request = getattr(controller, '_precision_acquisition', {})
        if (not samples and not request.get('cameras') and not row.get('closerViewAttempted')
                and callable(getattr(getattr(controller.rt, 'link', None), 'rgbd', None))):
            row['closerViewAttempted'] = True
            try:
                closer_view(controller, job, plan)
                after = acquire(controller, [plan['part']])
                row['closerViewExecuted'] = True
            except SkillFailed as exc:
                controller.check()
                row['closerViewFailure'] = str(exc)
        measured, reason = sample(controller, plan, after)
        row['acquisition'] = deepcopy(getattr(controller, '_precision_acquisition', {}))
        if measured is None:
            context['reason'] = row['reason'] = reason
            attach(controller, report)
            return context
        samples.append(measured)
        if len(samples) == 1:
            controller.ex._wait(.25)
    first, last = samples
    shift = np.linalg.norm(np.asarray(last['point'])-first['point'])
    rotation = np.linalg.norm(np.asarray(last['pose']['rotation'])-first['pose']['rotation'])
    if shift > .002 or rotation > .035 or last['observedAt'] <= first['observedAt']:
        context['reason'] = row['reason'] = 'target_not_settled'
        attach(controller, report)
        return context
    last['uncertaintyM'] += float(shift)
    # Record motion epochs and the fused pose separately from the raw outline.
    last['tracks'] = {eid: dict(position=est_track.pos.tolist(), yaw=est_track.yaw, motionAt=est_track.interface_motion_at)
        for eid in (plan['part'], plan['body'])
        if (est_track := controller.rt.estimator.tracks.get(eid)) is not None}
    context.update(registration=last, holeRegistrationSuccesses=1)
    row.update(status='registered', observedAt=last['observedAt'], uncertaintyM=last['uncertaintyM'],
               source=last['source'], pixels=last['pixels'])
    attach(controller, report)
    return context


def bearing(view):
    size = view.get('visibleSizeM', [])
    if len(size) != 3 or 'boxYawRad' not in view:
        return None
    return float(view['boxYawRad'])+(math.pi/2 if size[1] > size[0] else 0.)


def invalid_reason(controller, registration):
    if not registration:
        return 'hole_registration_unknown'
    if registration.get('source') == 'native_rgbd_support_holes':
        from steward.ego.support_registration import invalid_reason as support_invalid
        return support_invalid(controller, registration)
    est = controller.rt.estimator; now = controller.ex._state()['stamp']
    if not 0 <= now-registration['observedAt'] <= MAX_AGE_S:
        return 'hole_registration_expired'
    current = est.released_interfaces.get(registration['entity'])
    if not current or current['releasedAt'] != registration['frame']['releasedAt']:
        return 'interface_released_again'
    if current.get('alignmentMovedAt', -math.inf) > registration['observedAt']:
        return 'interface_motion_after_registration'
    for eid, saved in registration['tracks'].items():
        track = est.tracks.get(eid)
        if track is None or track.held or track.interface_motion_at > saved['motionAt']:
            return 'target_touched_or_lost'
        if (np.linalg.norm(track.pos-saved['position']) > MOVEMENT_M or
                abs((track.yaw-saved['yaw']+math.pi) % (2*math.pi)-math.pi) > .035):
            return 'target_pose_changed'
    # Inspect new raw observations even when stationary-memory fusion freezes pose.
    scene = controller.rt.ego_state(); view = value(scene.find(registration['entity']), 'contactView')
    f = registration['frame']
    if view and view.get('stamp', -1) > registration['observedAt']:
        centre, angle = view.get('centreM'), bearing(view)
        old_centre, old_angle = registration.get('rawCentreM'), registration.get('rawBearingRad')
        if ((centre is not None and old_centre is not None and
             np.linalg.norm(np.asarray(centre)-old_centre) > MOVEMENT_M) or
            (angle is not None and old_angle is not None and
             abs((angle-old_angle+math.pi/2) % math.pi-math.pi/2) > .035)):
            return 'raw_outline_moved'
    pose = outline(view, f['position'], np.asarray(f['rotation']), f['size'], registration['observedAt'], scene.stamp)
    if pose:
        old = registration['pose']
        if (np.linalg.norm(np.asarray(pose['position'])-old['position']) > MOVEMENT_M or
                np.linalg.norm(np.asarray(pose['rotation'])-old['rotation']) > .05):
            return 'raw_outline_moved'
    events = controller.rt.estimator.stationary_assemblies.motion_events
    if any(e['entity'] in (registration['entity'], registration['body']) and
           e['stamp'] > registration['observedAt'] for e in events):
        return 'observed_motion_event'
    return None


@timed("registration_refresh")
def refresh_if_moved(controller, job, plan):
    """Before transport, a changed target gets another arm-clear registration.

    After transport, an invalid registration holds the load instead of sending
    it to a stale target. The next invocation can register again away from it.
    """
    old = getattr(controller, '_registered_alignment', None)
    if not old or not old.get('registration') or not invalid_reason(controller, old['registration']):
        return
    new = register(controller, job, plan)
    job._precision_reports.remove(old['report'])
    for key in ('tip', 'tipHeldSince', 'tipOffsetAttempts', 'tipOffsetMeasuredCount', 'tipOffsetSource'):
        if key in old:
            new[key] = old[key]
    for key in ('holeRegistrationAttempts', 'holeRegistrationSuccesses'):
        new[key] += old[key]
    if 'registrationEvents' in old:
        for event in old['registrationEvents']:
            if event.get('failureReason') is None:
                event['invalidated'] = True
        new['registrationEvents'] = old['registrationEvents']+new['registrationEvents']
    posture = old['report'].get('seatDelivery', {}).get('posture')
    if posture is not None:
        new['report'].setdefault('seatDelivery', {})['posture'] = deepcopy(posture)
        job.postcondition.setdefault('seatDelivery', {})['posture'] = deepcopy(posture)
    attach(controller, new['report'])
    if new['mode'] == 'support_hole_fk_v2':
        from steward.ego.assembly_precision import publish
        publish(job)


def attach(controller, report):
    ctx = controller._registered_alignment
    reg = ctx.get('registration')
    report.update(alignmentBasis='registered_hole_fk' if reg and ctx.get('tip') else 'ordinary_place_unknown',
        alignmentMode=ctx['mode'], registrationId=ctx['registrationId'],
        kinematicResidualAxes='xy', tipOffsetUncertaintyM=None, alignmentUncertaintyM=None,
        holeRegistrationAttempts=ctx['holeRegistrationAttempts'],
        holeRegistrationSuccesses=ctx['holeRegistrationSuccesses'],
        tipOffsetAttempts=ctx['tipOffsetAttempts'], tipOffsetMeasuredCount=ctx['tipOffsetMeasuredCount'],
        tipOffsetMeasured=bool(ctx.get('tip') and ctx['tipOffsetMeasuredCount']), kinematicAlignmentExecutions=0,
        kinematicBeforeResidualM=None, kinematicAfterResidualM=None,
        registrationAgeS=None, registrationUncertaintyM=reg['uncertaintyM'] if reg else None,
        registrationPixels=deepcopy(reg['pixels']) if reg else None,
        registrationSource=reg['source'] if reg else None,
        fallback=ctx.get('reason'), insertionObserved=None)
    if ctx['mode'] == 'support_hole_fk_v2':
        report.update(registrationEvents=deepcopy(ctx['registrationEvents']),
            registrationFailureReason=ctx.get('registrationFailureReason'), tipOffsetSource=ctx.get('tipOffsetSource'),
            kinematicMeasuredOffsetExecutions=0, kinematicInferredOffsetExecutions=0,
            tipOffsetUncertaintyM=(ctx.get('tip') or {}).get('uncertaintyM'),
            registeredHoles=deepcopy(reg['holes']) if reg else [],
            registeredPointM=deepcopy(reg['point']) if reg else None,
            registeredBasePointM=deepcopy(reg['basePointM']) if reg else None)
        if reg and ctx.get('tip'):
            report['alignmentUncertaintyM'] = reg['uncertaintyM']+(ctx['tip'].get('uncertaintyM') or 0.)


def residual(controller, report):
    ctx = controller._registered_alignment; reg = ctx.get('registration')
    if reg:
        report['registrationAgeS'] = controller.ex._state()['stamp']-reg['observedAt']
    reason = invalid_reason(controller, reg)
    if not reason and not ctx.get('tip') and ctx['mode'] == 'support_hole_fk_v2':
        # The close post-grasp sample may never have been taken. The geometry captured at the
        # grasp is the same inferred offset the sampling step would have fallen back to.
        from steward.ego.support_registration import grasp_geometry, grasp_geometry_missing
        inferred = grasp_geometry(controller, report['object'])
        if inferred:
            ctx.update(tip=inferred, tipHeldSince=controller.rt.estimator.held_since,
                       tipOffsetSource='grasp_geometry_inferred')
            report.update(tipOffsetSource='grasp_geometry_inferred',
                          tipOffsetUncertaintyM=inferred['uncertaintyM'])
            if reg:
                report['alignmentUncertaintyM'] = reg['uncertaintyM']+inferred['uncertaintyM']
        else:
            report['tipInferenceFailure'] = grasp_geometry_missing(controller, report['object'])
    if reason or not ctx.get('tip'):
        report['fallback'] = reason or 'tip_offset_unknown'
        if reason == 'registration_invalidated':
            report['registrationFailureReason'] = reason
        return None
    est = controller.rt.estimator; state = controller.ex._state()
    if est.held_id != report['object'] or est.held_since != ctx['tipHeldSince']:
        report['fallback'] = 'grasp_changed_after_measurement'
        return None
    grip = est.gripper_position(est.held_gripper, state['joints'], state['odometry'])
    rotation = est.gripper_rotation(est.held_gripper, state['joints'], state['odometry'])
    f = transform_feature(ctx['tip'], rotation, grip)
    delta = np.asarray(reg.get('executionPoint', reg['point']))-f['tip']
    return dict(residualM=float(np.linalg.norm(delta[:2])), axisErrorDeg=math.degrees(
        math.acos(float(np.clip(f['axis'][2],-1,1)))), tipHeightM=float(-delta[2]))


def hover(controller, job, current, params, report):
    from steward.ego.wrist_target import audit
    candidate = report.pop('_wristCandidate', None)
    if candidate is not None and report.get('kinematicAlignmentExecutions', 0) >= 3:
        report['wristAlignment'].update(targetDecision='kept_original', targetReplaced=False,
            reason='existing_correction_limit_exhausted')
        report['fallbackAction'] = 'continue_original_descent'
        audit(report['wristAlignment'])
        return False
    measured = residual(controller, report)
    if measured is None:
        # Missing pre-transport evidence retains the original admitted placement.
        # A formerly valid registration invalidated by motion must not be reused.
        from steward.ego import seating_geometry
        if seating_geometry.enabled():
            report['fallbackAction'] = 'best_estimate_release_unconfirmed'
            current['nearContactUnconfirmed'] = True
            return False
        if report['alignmentBasis'] == 'registered_hole_fk':
            reg = (controller._registered_alignment.get('registration') or {})
            raise SkillFailed('registered target changed; withdraw and register again before release'
                              f" ({report.get('fallback')}; invalidated={reg.get('invalidated')})")
        return False
    report['fallback'] = None
    if report['kinematicBeforeResidualM'] is None:
        report['kinematicBeforeResidualM'] = measured['residualM']
    if report['kinematicAlignmentExecutions'] and candidate is None:
        wrist = report.get('wristAlignment', {})
        if ((not report.get('_wristMeasured') or wrist.get('status') == 'converged')
                and not report.get('_wristTargetChanged')
                or report['kinematicAlignmentExecutions'] >= 3):
            return False
    # The registered hole may differ from the saved layout by the admitted layout shift, so the
    # correction that brings the tip over it has the same reach, plus the tip estimate's own error.
    wrist_target = report.get('wristAlignment', {}).get('targetReplaced', False)
    ctx = controller._registered_alignment
    reg = candidate or ctx['registration']
    with controller.rt._lock:
        controller.rt.estimator.released_interfaces[params['on']] = dict(
            deepcopy(reg['frame']), **{k:v for k,v in controller.rt.estimator.released_interfaces.get(
                params['on'], {}).items() if k == 'registeredSeatTargets'})
        controller.rt.estimator.version += 1
    revised = dict(params, controlledSeat=True, precisionInsert=True,
                   registeredEntry=dict({k:deepcopy(reg[k]) for k in ('entity','body','hole','point','observedAt','source')},
                       point=deepcopy(reg['point'] if candidate else ctx.get('correctionPoint', reg.get('executionPoint', reg['point']))), persistent=bool(reg.get('persistent'))))
    prediction_started = time.monotonic()
    if candidate is not None:
        from steward.ego.wrist_target import adoption_enabled
        if adoption_enabled():
            revised['registeredEntry']['measurementSource'] = candidate['measurementSource']
        state = controller.ex._state()
        report['wristAlignment']['replanInput'] = dict(
            registeredEntry=deepcopy(revised['registeredEntry']),
            graspOffset=deepcopy(ctx.get('tip')), joints=deepcopy(state.get('joints')),
            odometry=deepcopy(state.get('odometry')))
        from steward.ego.wrist_target import local_range_enabled
        if local_range_enabled() or adoption_enabled():
            report['wristAlignment']['replanInput']['placementRequest'] = {
                k:deepcopy(revised.get(k)) for k in ('on', 'seatBody', 'seatHole')}
    receipt = controller.rt.predict('place', revised)
    if candidate is not None and adoption_enabled():
        planned = receipt.get('plan', {})
        report['wristAlignment']['replanInput']['entryCheck'] = deepcopy(
            planned.get('wristAdmission') or planned.get('seatTarget', {}).get('wristAdmission'))
    report.setdefault('seatDelivery', {})['endpointPredictWallS'] = round(time.monotonic()-prediction_started,3)
    row = dict(receiptId=receipt.get('id'), executed=False, residualBeforeM=measured['residualM'], source='registered_hole_fk')
    report['corrections'].append(row)
    if not receipt.get('withinRiskLimit') or not receipt.get('plan', {}).get('seatEndpoint'):
        report['fallback'] = 'registered_endpoint_path_unavailable'
        # Say which admission failed: the risk limit of the place, or the endpoint planner's reason.
        motion = receipt.get('plan', {}).get('seatMotion') or {}
        why = dict(withinRiskLimit=bool(receipt.get('withinRiskLimit')), risk=receipt.get('risk'),
                   issues=[str(i)[:160] for i in receipt.get('issues', [])[:4]],
                   endpointReason=str(motion.get('reason'))[:200] if motion.get('reason') else None)
        row['endpointUnavailable'] = why
        report['endpointUnavailable'] = why
        if candidate is not None:
            report['wristAlignment'].update(targetDecision='kept_original', targetReplaced=False,
                reason='endpoint_admission_failed', endpointUnavailable=why)
            report['fallbackAction'] = 'continue_original_descent'
            audit(report['wristAlignment'])
            return False
        if wrist_target:
            report['fallbackAction'] = 'retain_load_observed_target_unreachable'
            from steward.ego import seating_geometry
            if seating_geometry.enabled():
                report['fallbackAction'] = 'best_estimate_release_unconfirmed'
                return False
            raise SkillFailed('wrist hole target could not be replanned; old head target is superseded')
        report['fallbackAction'] = 'ordinary_place'
        # The ordinary placement is judged as an ordinary placement: the registered-hole release
        # check applies only to a load that was actually brought over the registered hole.
        report['alignmentBasis'] = 'ordinary_place_after_refusal'
        return False
    controller.check()
    if invalid_reason(controller, reg):
        report['fallback'] = 'registration_changed_during_planning'
        if ctx['mode'] == 'support_hole_fk_v2':
            report['registrationFailureReason'] = 'registration_invalidated'
        from steward.ego import seating_geometry
        if seating_geometry.enabled():
            report['fallbackAction'] = 'best_estimate_release_unconfirmed'
            return False
        raise SkillFailed('target registration changed during planning')
    replacement = controller.rt._consume_receipt(receipt)
    if candidate is not None:
        ctx['registration'].clear()
        ctx['registration'].update(candidate)
        reg = ctx['registration']
        ctx.pop('correctionPoint', None)
        report['_wristTargetChanged'] = True
        report['_wristEmptyEntry'] = dict(point=deepcopy(reg['point']), stamp=reg['observedAt'],
            radiusM=(ctx.get('tip') or {}).get('radiusM'))
        report['wristAlignment'].update(targetDecision='replaced', targetReplaced=True,
            reason='endpoint_admitted', receiptId=receipt.get('id'),
            endpoint=deepcopy(replacement.get('seatEndpoint')))
        audit(report['wristAlignment'])
    replacement.update(precision=report, **{k:current[k] for k in ('seatBody','seatHole','seatPart') if k in current})
    deadline = getattr(controller.ex, '_primitive_deadline', None)
    controller.ex._begin(job, 'place', receipt)
    if deadline is not None:
        controller.ex._primitive_deadline = ab.limited(deadline, controller.ex._primitive_deadline)
    row['executed'] = True
    if report.get('landingCompensation',{}).get('status')=='estimated':
        report['landingCompensation'].update(attempted=True,receiptId=receipt.get('id'))
    if ctx.get('correctionPoint'):
        report.setdefault('seatDelivery', {})['correctionAttempted'] = True
    report['kinematicAlignmentExecutions'] += 1
    wrist_target_changed = report.pop('_wristTargetChanged', False)
    if report.get('_wristMeasured') or wrist_target_changed:
        report['wristAlignment']['corrections'] += 1
    if ctx['mode'] == 'support_hole_fk_v2':
        field = ('kinematicMeasuredOffsetExecutions' if ctx.get('tipOffsetSource') == 'measured'
                 else 'kinematicInferredOffsetExecutions')
        report[field] += 1
        from steward.ego.assembly_precision import publish
        publish(job)
    try:
        usage_id = report.get('wristAlignment', {}).get('_usageId')
        if usage_id and (report.get('_wristMeasured') or wrist_target_changed):
            from steward.ego.capability_usage import record
            record('wrist_hole', event_id=usage_id, measured=True, action=True)
        controller.ex._place(job, replacement)
        if ctx.get('correctionPoint'):
            report.setdefault('seatDelivery', {})['correctionApplied'] = True
    finally:
        if ctx.get('correctionPoint') and replacement.get('releaseExecuted'):
            report.setdefault('seatDelivery', {})['correctionApplied'] = True
        if (replacement.get('releaseExecuted') and report.get('landingCompensation',{}).get('status')=='estimated'):
            report['landingCompensation']['applied']=True
        current.update(replacement)
    return True


def release_clearance(controller, report):
    """No uncertainty-height offset in fasten_delivery; retain the old rule behind its switch."""
    from steward.ego.fasten_delivery import ENABLED
    if ENABLED:
        return 0.
    if report.get('alignmentBasis') != 'registered_hole_fk' or report.get('tipOffsetSource') == 'measured':
        return 0.
    tip = (controller._registered_alignment.get('tip') or {})
    return float(np.clip(float(tip.get('uncertaintyM') or 0.), 0., .012))


def before_release(controller, current, report):
    fallback = current.get('_fallbackReleaseEndpoint')
    if fallback:
        est = controller.rt.estimator
        if est.held_id != current['object'] or est.held_since != fallback['heldSince']:
            raise SkillFailed('held_object_changed_before_release')
        report['releaseGeometryBasis']='pre_descent_corrected_target_near_contact_unconfirmed'
        return
    if report['alignmentBasis'] != 'registered_hole_fk':
        return
    observed=current.get('_observedReleaseEndpoint')
    if observed:
        est=controller.rt.estimator
        if (est.held_id != current['object'] or est.held_since != observed['heldSince'] or
                abs(controller.ex._state()['joints']['lift']-observed['lift']) > observed['uncertaintyM']):
            raise SkillFailed('observed_endpoint_changed_before_release')
        report['releaseGeometryBasis']='same_frame_visible_gap_and_settled_lift'
        return
    measured = residual(controller, report)
    if measured is None:
        # Name the cause: which invalidation, or which grasp change, and how old the registration is.
        ctx = controller._registered_alignment; reg = ctx.get('registration') or {}
        est = controller.rt.estimator
        cause = dict(fallback=report.get('fallback'), invalidated=reg.get('invalidated'),
                     heldIsObject=bool(est.held_id == report.get('object')),
                     heldSinceChanged=bool(est.held_since != ctx.get('tipHeldSince')),
                     ageS=round(float(report.get('registrationAgeS') or 0.), 1))
        report['releaseBlockedBy'] = cause
        raise SkillFailed('registration or measured grasp offset invalid before release '
                          f"({cause['fallback']}; invalidated={cause['invalidated']};"
                          f" held_is_object={cause['heldIsObject']}; held_since_changed={cause['heldSinceChanged']};"
                          f" age={cause['ageS']}s)")
    report.update(kinematicAfterResidualM=round(measured['residualM'], 5),
                  kinematicAxisErrorDeg=round(measured['axisErrorDeg'], 3),
                  kinematicTipHeightM=round(measured['tipHeightM'], 5))
    # fasten_delivery reports the axial criterion without disguising uncertainty as clearance.
    # The disabled path retains the legacy uncertainty-sized height window.
    tip = (controller._registered_alignment.get('tip') or {})
    above = (max(.002, float(tip.get('uncertaintyM') or 0.))+.002
             if report.get('tipOffsetSource') != 'measured' else .002)
    from steward.ego.fasten_delivery import ENABLED
    if ENABLED:
        above = .003
    # Likewise the lateral bound of an inferred tip is not tighter than its own uncertainty
    # (capped): external runs seated every such release with the real tip 10-14 mm off.
    lateral = (max(report['toleranceM'], min(float(tip.get('uncertaintyM') or 0.), .008))
               if report.get('tipOffsetSource') != 'measured' else report['toleranceM'])
    lateral_miss = measured['residualM'] > lateral
    if current.get('seatEndpointOutcome') or any(
            c.get('executed') for c in current.get('tipEntryServo', {}).get('corrections', [])):
        # Completed endpoint descents (seat_endpoint), and differential translations
        # measured from one frame (tip_entry_servo), retain nominal-offset XY as advisory.
        # It cannot veto a correction that deliberately bypasses that offset.
        if lateral_miss:
            report.setdefault('seatDelivery', {})['lateralResidualAdvisoryM'] = measured['residualM']
        lateral_miss = False
    if (lateral_miss or measured['axisErrorDeg'] > 8
            or not -.004 <= measured['tipHeightM'] <= above):
        report['fallback'] = 'kinematic_endpoint_not_reached'
        from steward.ego.fasten_delivery import ENABLED
        if ENABLED and measured['axisErrorDeg'] <= 10. and not report.get('wristAlignment', {}).get('targetReplaced'):
            report.setdefault('seatDelivery', {})['endpointWarning'] = 'kinematic_endpoint_not_reached'
            return  # Keep the admitted low release; posterior remains independent.
        raise SkillFailed('FK endpoint outside release tolerance'
                          f" (lateral {1000*measured['residualM']:.1f} mm of {1000*lateral:.1f},"
                          f" axis {measured['axisErrorDeg']:.1f} deg, tip height {1000*measured['tipHeightM']:.1f} mm"
                          f" of -4..{1000*above:.0f}); load remains held")
