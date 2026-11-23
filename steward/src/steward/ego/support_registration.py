"""Acquire centred support holes before picking; never wait for vision at release."""
from copy import deepcopy
from itertools import product
import math
import time
from steward.ego import action_budget as ab
import numpy as np
from PIL import Image
from steward.ego import support_holes as holes
from steward.ego.skills import SkillFailed
from steward.math3d import matrix_from_quat, quat_from_yaw

MODE = 'support_hole_fk_v2'


def envelope(controller, frame):
    corners = np.array(list(product((-1,1), repeat=3)))
    if frame.get('viewBoundsM') is not None:
        lo, hi = np.asarray(frame['viewBoundsM'])
        return (lo+hi)/2+corners*(hi-lo)/2
    points = corners*np.asarray(frame['size'])/2 @ np.asarray(frame['rotation']).T+frame['position']
    body = controller.rt.estimator.tracks.get(frame['body'])
    if body is not None:
        other = corners*body.size/2 @ matrix_from_quat(quat_from_yaw(body.yaw)).T+body.pos
        points = np.vstack([points, other])
    return points


# Planned viewpoints stay this far inside the framing band; a failed arrival is retried elsewhere.
FRAMING_MARGIN = .08
VIEW_ATTEMPTS = 3

# Registration compares consecutive frames; it gives up after this many, and never accepts more drift than this.
SETTLE_FRAMES = 5
SETTLE_SHIFT_MAX = .008

def view_geometry(controller, frame, odometry=None):
    est = controller.rt.estimator; camera = est.primary_camera
    cam = est.camera_pose(odometry, camera)
    intr = est.camera_intrinsics(camera)
    points = envelope(controller, frame)
    uv, forward = holes.project(points, cam, intr)
    result = holes.framing(uv, (intr['height'], intr['width']))
    result['rangeM'] = round(float(np.linalg.norm(np.asarray(frame['position'])-cam['p'])),6)
    reasons = []
    if not np.all(forward > .05): reasons.append('behind_camera')
    if result['rangeM'] >= holes.PerceptionConfig().max_range: reasons.append('sensor_range')
    if result['border']: reasons.append('border')
    if not result['centred']: reasons.append('not_centred')
    result['pixelCheck'] = 'requires_native_frame'
    result['rejections'] = reasons
    result['usable'] = not reasons
    result['camera'] = camera
    return result


def adjust_view(controller, job, frame, event=None):
    """Search outwards, admitting each move through the existing navigator.

    Metric samples scale with perceived support geometry;
    range terminates only at the sensor's range. Unknown hole scale is checked
    on arrival, never supplied from a task-specific hole diameter.
    """
    from steward.ego.fasten_plan import support_anchor
    from steward.ego.planner import entity_box
    event = event if event is not None else {}
    records = event.setdefault('viewCandidates', [])
    scene = controller.rt.ego_state()
    body = scene.find(frame['body'])
    if body is None:
        supports = [e for e in scene.entities if controller.rt.esim._affords(e, 'support-surface')]
        if not supports:
            event['viewSearchFailure'] = 'no_perceived_support'
            raise SkillFailed('no perceived support for registration view')
        support = min(supports, key=lambda e: np.linalg.norm(np.asarray(e.pose.value['p'])-frame['position']))
    else:
        support = support_anchor(controller.rt.esim, scene, body, [])
    box, _, _ = entity_box(support)
    odom = controller.ex._state()['odometry']
    origin = math.atan2(odom['y']-box.cy, odom['x']-box.cx)
    target = np.asarray(frame['position']); choices = []
    # A bounded discrete search, including all sides and farther viewpoints.
    # Reaching a sample still requires the navigator's full path/collision check.
    scale = max(.05, min(box.hx, box.hy)/2)
    clearances = []
    distance = scale
    while distance < holes.PerceptionConfig().max_range:
        clearances.append(distance)
        distance *= 1.35
    for turn in np.arange(16)*math.tau/16:
        angle = origin+turn
        edge = min(box.hx/max(abs(math.cos(angle-box.yaw)), 1e-9),
                   box.hy/max(abs(math.sin(angle-box.yaw)), 1e-9))
        for clearance in clearances:
            radius = edge+clearance
            x, y = box.cx+radius*math.cos(angle), box.cy+radius*math.sin(angle)
            params = dict(x=x, y=y, yaw=math.atan2(target[1]-y, target[0]-x))
            choices.append((math.hypot(x-odom['x'],y-odom['y']), params))
    for _, params in sorted(choices, key=lambda c:c[0]):
        controller.check()
        geometry = view_geometry(controller, frame, params)
        record = dict(pose=params, rejections=list(geometry['rejections']),
                      centreFraction=geometry['centreFraction'], edgePixels=geometry['edgePixels'],
                      rangeM=geometry['rangeM'], pixelCheck=geometry['pixelCheck'], navigation='not_checked')
        records.append(record)
        if not geometry['usable']:
            continue
        # Arrival differs from the planned pose by a few centimetres and a few hundredths of a
        # radian. A viewpoint that only just passes the framing test fails it on arrival, so the
        # plan keeps a margin inside the acceptance band.
        if (min(geometry['centreFraction']) < .25+FRAMING_MARGIN
                or max(geometry['centreFraction']) > .75-FRAMING_MARGIN):
            record['rejections'].append('framing_margin')
            continue
        if any(math.hypot(params['x']-q['x'], params['y']-q['y']) < .03 for q in event.get('viewsTried', [])):
            record['rejections'].append('already_tried')
            continue
        receipt = controller.rt.predict('navigate', params)
        record.update(navigation='predicted', issues=receipt.get('issues', []), risk=receipt.get('risk'))
        if not receipt.get('withinRiskLimit'):
            text = ' '.join(str(v) for v in record['issues']).lower()
            reason = 'collision_risk' if 'collid' in text else (
                'unreachable' if not receipt.get('feasible', True) else 'risk_limit')
            record['rejections'].append(reason)
            continue
        try:
            controller.step(job, 'navigate', params, receipt=receipt)
        except SkillFailed:
            record['rejections'].append('navigation_failed')
            raise  # Never continue from an uncertain partially executed move.
        record['navigation'] = 'executed'
        event.setdefault('viewsTried', []).append(dict(x=float(params['x']), y=float(params['y'])))
        event['selectedViewpoint'] = {k: round(float(v),6) for k,v in controller.ex._state()['odometry'].items() if k in ('x','y','yaw')}
        event['selectedViewCandidate'] = len(records)-1
        return
    event['viewSearchFailure'] = 'sampled_viewpoints_exhausted'
    raise SkillFailed('no admitted centred registration viewpoint')


def acquire(controller, frame):
    """Force a native primary-camera batch even when the part id has disappeared."""
    rt, ex = controller.rt, controller.ex
    after = ex._state()['stamp']; start = time.monotonic()
    deadline = ab.make(ex, ab.observation(ex) if ab.enabled() else 8., 'existing eight-sensor-second registration frame window')
    with rt._lock:
        request = rt.request_precision_view([frame['position']], after)
        primary = rt.estimator.primary_camera
        # Only this registration request bypasses the independent-part .8 m
        # chooser. Ordinary precision requests and their quality gates are intact.
        configured = getattr(rt.config, 'cameras', None)
        request['cameras'] = [primary] if (configured is None or primary in configured) else []
        request['reason'] = 'support_registration_native_frame' if request['cameras'] else 'camera_not_configured'
        request['entities'] = [frame['entity'], frame['body']]
        request['supportHoleRegistration'] = True
    controller._precision_acquisition = request
    try:
        while request['cameras'] and not request.get('completed'):
            controller.check()
            if ab.expired(deadline) or ex._state()['stamp']-after >= (ab.observation(ex) if ab.enabled() else 8):
                break
            ex._wait(.1)
        raw = getattr(rt, '_precision_native_frames', {}).get(primary)
        if (not request.get('completed') or raw is None or raw['header']['stamp'] <= after):
            return None
        return raw
    finally:
        rt.finish_precision_view(request)


def save_frame(rt, raw):
    if raw is None:
        return dict(error='no_registration_frame')
    try:
        if rt.out_dir is None:
            return dict(error='runtime_output_directory_unavailable')
        path = rt.out_dir / f"register-{raw['header']['stamp']:.6f}-{raw['camera']}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(raw['rgb']).save(path, format='PNG')
        return dict(filename=path.name, stamp=raw['header']['stamp'], camera=raw['camera'],
                    width=int(raw['rgb'].shape[1]), height=int(raw['rgb'].shape[0]))
    except Exception as exc:
        return dict(error=type(exc).__name__)


def diagnostic(rt, row):
    """Same local file as the gate probe; never consumed by control or model."""
    import json
    try:
        if rt.out_dir is not None:
            rt.out_dir.mkdir(parents=True, exist_ok=True)
            with (rt.out_dir / 'precision_probe.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(kind='support_hole_registration', **row))+ '\n')
    except Exception:
        pass


def register(controller, job, plan):
    from steward.ego.registered_alignment import attach, arm_clear
    rt = controller.rt; est = rt.estimator
    job._registration_serial = getattr(job, '_registration_serial', 0)+1
    ctx = dict(mode=MODE, registrationId=str(job.id)+':'+str(job._registration_serial),
        holeRegistrationAttempts=1, holeRegistrationSuccesses=0, tipOffsetAttempts=0,
        tipOffsetMeasuredCount=0, tipOffsetSource=None, registration=None, tip=None, reason=None,
        registrationEvents=[])
    report = dict(kind='screw_registration', object=None, body=plan['body'], hole=plan['hole'],
        observations=[], corrections=[], releaseExecuted=False)
    ctx['report'] = report; controller._registered_alignment = ctx
    if not hasattr(job, '_precision_reports'): job._precision_reports = []
    job._precision_reports.append(report)
    event = dict(registrationId=ctx['registrationId'], before=None, after=None, frames=[],
        holeCount=0, selectedHole=None, targetDistanceM=None, failureReason=None,
        viewCandidates=[], selectedViewpoint=None, selectedHoleSpanPixels=None,
        selectedHoleRecessM=None, depthEvidence=None)
    ctx['registrationEvents'].append(event)
    with rt._lock:
        rt._support_hole_watch = None
        frame = deepcopy(est.released_interfaces.get(plan['part']))
    try:
        if not frame or frame.get('body') != plan['body']:
            event['failureReason'] = 'no_saved_interface'
            return ctx
        if est.held_id in (plan['part'], plan['body']):
            event['failureReason'] = 'registration_invalidated'
            return ctx
        from steward.ego.fasten_target import reuse
        cached = reuse(controller, plan)
        if cached:
            ctx.update(registration=cached, holeRegistrationSuccesses=1, targetSource='reused_registration')
            event['reused'] = True
            with rt._lock:
                rt._support_hole_watch = cached
            return ctx
        reference_pair = next((p for p in frame['holePairs'] if p['bodyHole']==plan['hole']), None)
        reference = next((h for h in frame['holes'] if reference_pair and h['id']==reference_pair['partHole']), None)
        if reference is None:
            event['failureReason'] = 'hole_target_mismatch'
            return ctx
        event['before'] = view_geometry(controller, frame)
        for _ in range(VIEW_ATTEMPTS):
            if view_geometry(controller, frame)['usable']:
                break
            try:
                adjust_view(controller, job, frame, event)
            except SkillFailed:
                controller.check()
                event['failureReason'] = 'view_adjustment_failed'
                event['after'] = view_geometry(controller, frame)
                return ctx
        event['after'] = view_geometry(controller, frame)
        if not event['after']['usable']:
            event['failureReason'] = 'view_adjustment_failed'
            return ctx
        event['selectedViewpoint'] = {k: round(float(v),6) for k,v in controller.ex._state()['odometry'].items() if k in ('x','y','yaw')}
        samples = []; shifts = []
        actual_retry = False
        while True:
            controller.check()
            raw = acquire(controller, frame)
            saved = save_frame(rt, raw); event['frames'].append(saved)
            if raw is None:
                event['failureReason'] = 'no_registration_frame'
                return ctx
            cam = est.camera_pose(raw['header']['odometry'], raw['camera'])
            found = holes.detect(raw, cam, reference['point'])
            saved.update(framing=found['framing'], holeCount=len(found['holes']),
                         mask=found['mask'], failureReason=found['reason'])
            if found.get('holeRejections'):
                saved['holeRejections'] = found['holeRejections']
            if event.get('selectedViewCandidate') is not None:
                candidate = event['viewCandidates'][event['selectedViewCandidate']]
                candidate['frameValidation'] = found['reason'] or 'holes_detected'
                if found['reason']:
                    candidate['rejections'].append(found['reason'])
                    candidate['rejections'].extend(found.get('holeRejections', {}))
            event['holeCount'] = len(found['holes'])
            if found['reason'] == 'view_adjustment_failed' and not actual_retry:
                actual_retry = True
                event['observedBeforeAdjustment'] = found['framing']
                frame['viewBoundsM'] = found['regionBoundsM']
                try:
                    adjust_view(controller, job, frame, event)
                except SkillFailed:
                    controller.check()
                    event['failureReason'] = 'view_adjustment_failed'
                    return ctx
                event['after'] = view_geometry(controller, frame)
                if not event['after']['usable']:
                    event['failureReason'] = 'view_adjustment_failed'
                    return ctx
                samples.clear()
                continue
            if found['reason']:
                event['failureReason'] = found['reason']
                return ctx
            # Resolved protrusions fail the depth veto; released ledger entries
            # additionally exclude their locations without changing accounting.
            occupied = [np.asarray(a['entry']) for a in est.fastening_attempts.values()
                        if a.get('body') == plan['body'] and a.get('releaseExecuted') and 'entry' in a]
            for h in found['holes']:
                h['available'] = not any(np.linalg.norm(np.asarray(h['point'])-p) <= .006 for p in occupied)
            selected, distance = holes.correspond(found['holes'], frame, plan['hole'], occupied, diagnostics=event)
            event.update(selectedHole=selected, targetDistanceM=distance)
            saved.update(selectedHole=selected, targetDistanceM=distance)
            if selected is None:
                event['failureReason'] = 'hole_target_mismatch'
                saved['failureReason'] = 'hole_target_mismatch'
                return ctx
            h = found['holes'][selected]
            evidence = dict(selectedHoleSpanPixels=h['spanPixels'], selectedHoleRecessM=h['recessM'],
                            depthEvidence=h['depthEvidence'], selectedHolePlaneNoiseM=h['planeNoiseM'])
            event.update(evidence)
            saved.update(evidence)
            if not arm_clear(est, h['point'], raw['camera']):
                event['failureReason'] = 'robot_occludes_entry_ray'
                return ctx
            samples.append((raw, found, h))
            if len(samples) >= 2:
                # Two consecutive frames must agree within what the measurement itself can resolve.
                # A fixed 2 mm bound is below the centre noise of a small hole seen from a metre away.
                first, last = samples[-2], samples[-1]
                shift = float(np.linalg.norm(np.asarray(first[2]['point'])-last[2]['point']))
                tolerance = float(np.clip(first[2]['uncertaintyM']+last[2]['uncertaintyM'], .002, SETTLE_SHIFT_MAX))
                shifts.append(round(shift, 4))
                if shift <= tolerance and last[0]['header']['stamp'] > first[0]['header']['stamp']:
                    break
                if len(samples) >= SETTLE_FRAMES:
                    event.update(failureReason='support_not_settled', settleShiftsM=shifts,
                                 settleToleranceM=round(tolerance, 4))
                    return ctx
            controller.ex._wait(.25)
        first, last = samples[-2], samples[-1]
        shift = float(np.linalg.norm(np.asarray(first[2]['point'])-last[2]['point']))
        raw, found, h = last; odom = deepcopy(raw['header']['odometry'])
        base = np.array([odom['x'], odom['y'], 0.])
        for entry in found['holes']:
            entry['basePointM'] = (matrix_from_quat(quat_from_yaw(odom['yaw'])).T @
                                  (np.asarray(entry['point'])-base)).tolist()
        reg = dict(point=h['point'], uncertaintyM=h['uncertaintyM']+shift,
            observedAt=float(raw['header']['stamp']), source='native_rgbd_support_holes',
            entity=plan['part'], body=plan['body'], hole=plan['hole'], frame=frame,
            basePointM=(matrix_from_quat(quat_from_yaw(odom['yaw'])).T @ (np.asarray(h['point'])-base)).tolist(),
            baseOdometry=odom, holes=found['holes'], invalidated=None,
            pixels=dict(camera=raw['camera'], width=raw['rgb'].shape[1], height=raw['rgb'].shape[0],
                        native=True, holeObserved=True, holeSpanPixels=h['spanPixels']))
        ctx.update(registration=reg, holeRegistrationSuccesses=1)
        with rt._lock:
            rt._support_hole_watch = reg
        # Preserve the prior frame as prior. Only the selected entry is measured;
        # neither the assembly mask nor this point overwrites any object's pose.
        return ctx
    finally:
        if not ctx['holeRegistrationSuccesses'] and event['failureReason'] is None:
            event['failureReason'] = 'registration_interrupted'
        event['registered'] = bool(ctx['holeRegistrationSuccesses'])
        ctx['reason'] = event['failureReason']
        ctx['registrationFailureReason'] = event['failureReason']
        diagnostic(rt, event)
        attach(controller, report)
        from steward.ego.assembly_precision import publish
        publish(job)


def motion_due(rt, raw):
    """Gate native-mask preparation, retaining the original sticky expiration."""
    reg = getattr(rt, '_support_hole_watch', None)
    if not reg or reg['invalidated'] or raw['header']['stamp'] <= reg['observedAt']:
        return False
    from steward.ego.registered_alignment import MAX_AGE_S
    if not reg.get('persistent') and raw['header']['stamp']-reg['observedAt'] > MAX_AGE_S:
        reg.update(invalidated='expired', invalidatedAt=raw['header']['stamp'])
        return False
    return True


def observe_motion(rt, raw):
    """Observe support features in ordinary frames; invalidation stays sticky."""
    if not motion_due(rt, raw):
        return
    reg = rt._support_hole_watch
    cam = rt.estimator.camera_pose(raw['header']['odometry'], raw['camera'])
    found = holes.detect(raw, cam, reg['point'])
    if found['reason']:
        return  # Occlusion is absence of evidence, never a new position.
    old = np.array([h['point'] for h in reg['holes']])
    new = np.array([h['point'] for h in found['holes']])
    distances = np.linalg.norm(old[:,None]-new[None,:], axis=2)
    # A feature must remain near its own previous neighbourhood. A newly covered
    # hole cannot be matched to an adjacent one and called assembly movement.
    for i, row in enumerate(distances):
        j = int(row.argmin())
        separation = min((np.linalg.norm(old[i]-p) for k,p in enumerate(old) if k!=i), default=.06)
        if row[j] < min(.03, separation/2) and distances[:,j].argmin() == i:
            if row[j] > .004 + reg['holes'][i]['uncertaintyM'] + found['holes'][j]['uncertaintyM']:
                reg.update(invalidated='support_features_moved', invalidatedAt=raw['header']['stamp'])
                return


def invalid_reason(controller, reg):
    from steward.ego.registered_alignment import MAX_AGE_S
    if reg.get('invalidated'):
        return 'registration_invalidated'
    est = controller.rt.estimator; now = controller.ex._state()['stamp']
    reason = None
    if not reg.get('persistent') and not 0 <= now-reg['observedAt'] <= MAX_AGE_S:
        reason = 'expired'
    current = est.released_interfaces.get(reg['entity'])
    if not current or current['releasedAt'] != reg['frame']['releasedAt']:
        reason = 'interface_released_again'
    if current and current.get('alignmentMovedAt', -math.inf) > reg['observedAt']:
        reason = 'interface_motion'
    if est.held_id in (reg['entity'], reg['body']):
        reason = 'support_held'
    events = est.stationary_assemblies.motion_events
    if any(e['entity'] in (reg['entity'],reg['body']) and e['stamp']>reg['observedAt'] for e in events):
        reason = 'support_motion_event'
    if reason:
        reg.update(invalidated=reason, invalidatedAt=now)
        return 'registration_invalidated'
    return None


def grasp_geometry_missing(controller, eid):
    """Why no grasp-time geometry is available for this held object."""
    est = controller.rt.estimator; track = est.tracks.get(eid)
    if track is None:
        return 'no_track'
    if not track.held or est.held_id != eid:
        return 'not_the_held_object'
    if not track.held_grasp_geometry:
        return f'no_geometry_for_category:{track.category}'
    return 'invalid_geometry'


def grasp_geometry(controller, eid):
    """Use the feature captured at grasp, never reconstruct from a merged mask."""
    est = controller.rt.estimator; track = est.tracks.get(eid)
    feature = deepcopy(track.held_grasp_geometry) if track is not None and track.held else None
    if not feature or est.held_id != eid:
        return None
    if (not all(np.asarray(feature[k]).shape == (3,) and np.isfinite(feature[k]).all()
                for k in ('tip','head','axis')) or feature.get('lengthM', 0) <= 0):
        return None
    from steward.ego.grasp_endpoints import emit
    emit(feature.get('offsetEvidence'), 'offset')
    feature['source'] = 'grasp_geometry_inferred'
    # Explicit conservative prior uncertainty, independent of hole precision.
    feature['uncertaintyM'] = float(np.sqrt(np.max(track.var)))
    with controller.rt._lock:
        # Endpoint planning and residual evaluation must use the same geometry.
        track.held_tool_feature = deepcopy(feature)
        est.version += 1
    return feature
