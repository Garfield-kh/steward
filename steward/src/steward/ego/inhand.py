"""Post-grasp component holes relative to synchronized encoder FK.

Only native RGB-D, the robot model, and the already acquired hole catalogue are
inputs. Failure leaves the original grasp transform and ordinary place available.
"""
from steward.ego import capabilities
from copy import copy, deepcopy
import math
import time
import numpy as np

from steward.ego import component_registration as cr, support_holes as sh
from steward.math3d import matrix_from_quat, quat_from_yaw
from steward.net import HttpError

PRECISION_M = .002
FRAME_WAIT_S = 2.


def load_pose(est, held, state):
    r = est.gripper_rotation(est.held_gripper, state['joints'], state['odometry'])
    g = est.gripper_position(est.held_gripper, state['joints'], state['odometry'])
    return g+r @ held.held_offset, r @ np.asarray(held.held_interface['rotation']), g, r


def camera_pose(est, state, camera):
    proxy = copy(est)
    proxy.joints, proxy.odometry = state['joints'], state['odometry']
    return proxy.camera_pose(camera=camera)


def layout(held):
    frame = held.held_interface
    if not frame or not frame.get('holes'):
        return None
    try:
        points = np.asarray([h['p'] for h in frame['holes']], float)
    except (ValueError, KeyError, TypeError):
        return None
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        return None
    # Multiple catalogue layers are not interchangeable visible hole groups.
    if np.ptp(points[:, 2]) > PRECISION_M:
        return None
    return points


def ray_clear(est, camera, points, state):
    """Include visual fingers/hand/arm: these have no collision primitives."""
    proxy = copy(est)
    proxy.joints, proxy.odometry = state['joints'], state['odometry']
    origin = np.asarray(proxy.camera_pose(camera=camera)['p'])
    for point in points:
        for centre, half, rotation in proxy.body_boxes():
            a, b = (np.asarray([origin, point])-centre) @ rotation
            delta = b-a
            low, high = 0., 1.
            for axis in range(3):
                if abs(delta[axis]) < 1e-12:
                    if abs(a[axis]) > half[axis]:
                        low, high = 1., 0.
                        break
                else:
                    x, y = sorted(((-half[axis]-a[axis])/delta[axis],
                                   (half[axis]-a[axis])/delta[axis]))
                    low, high = max(low, x), min(high, y)
            if low < high and high > 0 and low < 1:
                return False
    return True


def geometry(est, held, state, camera, intr=None):
    local = layout(held)
    if local is None:
        return None
    centre, rotation, _, _ = load_pose(est, held, state)
    # This detector observes an upward, approximately horizontal exposed face.
    # Do not rotate/flip the load to manufacture an unobserved opposite face.
    if rotation[2, 2] < math.cos(math.radians(25)):
        return None
    face = local.copy()
    face[:, 2] = held.size[2]/2
    points = face @ rotation.T+centre
    cam = camera_pose(est, state, camera)
    intr = intr or est.camera_intrinsics(camera)
    pad = sh.MAX_LAYOUT_SHIFT
    corners = [[x, y, z] for x in (points[:, 0].min()-pad, points[:, 0].max()+pad)
               for y in (points[:, 1].min()-pad, points[:, 1].max()+pad)
               for z in (points[0, 2]-.03, points[0, 2]+.03)]
    uv, forward = sh.project(corners, cam, intr)
    if not np.isfinite(uv).all() or np.any(forward <= .05):
        return None
    framing = sh.framing(uv, (intr['height'], intr['width']))
    if framing['border'] or not framing['centred']:
        return None
    ray = np.asarray(cam['p'])-points.mean(axis=0)
    distance = float(np.linalg.norm(ray))
    cosine = float(ray @ rotation[:, 2])/distance
    if cosine <= 0:
        return None
    focal = intr['height']/(2*math.tan(math.radians(intr['fovy'])/2))
    pixel = distance/(focal*cosine)
    if pixel*.25 > PRECISION_M:
        return None
    return dict(rangeM=distance, incidenceDeg=math.degrees(math.acos(cosine)),
                pixelFootprintM=pixel, camera=camera, points=points,
                predictedClear=ray_clear(est, camera, points, state))


def predict(planner, scene, params):
    """Internal arm-only motion, using ordinary receipt/risk/IK admission.

    The load's swept box is checked against perceived geometry at the existing
    lift/wrist sampling resolution. No base move, release, or task-specific pose.
    """
    from steward.ego.esim import PredictionError
    from steward.ego.planner import entity_box, Box2D, rects_overlap
    held = planner._held(scene)
    if held is None:
        raise PredictionError('inhand view requires a held component')
    current = planner._arm_joints(scene)
    goal = params['joints']
    if set(goal) - {'lift', 'reach', 'wrist_yaw'} or not all(math.isfinite(v) for v in goal.values()):
        raise PredictionError('invalid inhand view joints')
    # Raise before extending/turning; returning uses the reverse sequence.
    order = ('lift', 'reach', 'wrist_yaw') if goal.get('lift', current['lift']) >= current['lift'] else ('wrist_yaw', 'reach', 'lift')
    at = dict(current); path = [at]; stages = []
    for key in order:
        target = goal.get(key, at[key])
        if target == at[key]:
            continue
        step = planner.config.wrist_check_step if key == 'wrist_yaw' else planner.config.lift_check_step
        count = max(1, math.ceil(abs(target-at[key])/step))
        path.extend(dict(at, **{key:at[key]+(target-at[key])*i/count}) for i in range(1, count+1))
        stages.append([key, target]); at = path[-1]
    joints = {j.id:j.q.value[0] for j in planner._self(scene).rigid.joints if j.q.value}
    if not planner.arm.reachable_path(path, start_joints=joints):
        raise PredictionError('inhand view arm path unreachable')
    x, y, yaw = planner._robot_xy_yaw(scene)
    base = np.array([x, y, 0.]); rb = matrix_from_quat(quat_from_yaw(yaw))
    def grip(q):
        return base+rb @ [planner.tool_offset+q['reach'], planner.lateral, q['lift']]
    def rotation(q):
        return rb @ planner.arm.tool_rotation(q.get('wrist_yaw', 0.), q.get('wrist_roll', 0.), q.get('wrist_pitch', 0.))
    r0 = rotation(current)
    offset = r0.T @ (np.asarray(held.pose.value['p'])-grip(current))
    relative = r0.T @ matrix_from_quat(held.pose.value['q'])
    box, bottom, top = entity_box(held)
    half = np.array([box.hx, box.hy, (top-bottom)/2])
    # Swept interpolation allowance: half a sample step plus the held posterior.
    cov = (held.pose.unc or {}).get('cov', [])
    sigma = math.sqrt(max([float(cov[i]) for i in (0, 7, 14)] or [0.])) if len(cov) >= 15 else 0.
    pad = sigma+planner.config.lift_check_step/2+np.linalg.norm(half+abs(offset))*planner.config.wrist_check_step/2
    obstacles = [(e, entity_box(e)) for e in scene.entities if e.id not in (held.id, scene.self_id)
                 and e.rigid and e.rigid.links and e.rigid.links[0].geometry.get('collision')]
    for q in path:
        r = rotation(q); centre = grip(q)+r @ offset
        extent = abs(r @ relative) @ half+pad
        load = Box2D(held.id, centre[0], centre[1], 0., extent[0], extent[1])
        for ent, (obstacle, low, high) in obstacles:
            if centre[2]-extent[2] < high and centre[2]+extent[2] > low and rects_overlap(load, obstacle):
                raise PredictionError('inhand view load path intersects perceived geometry')
    return planner._receipt(scene, '_inhand_view', params, .02, [], dict(stages=stages))


def move(controller, goal):
    receipt = controller.rt.predict('_inhand_view', dict(joints=goal))
    if not receipt.get('withinRiskLimit'):
        return False
    plan = controller.rt._consume_receipt(receipt)
    for key, value in plan['stages']:
        controller.check()
        controller.ex._set_joint(key, value, timeout=FRAME_WAIT_S,
                                 tol=controller.ex.config.release_joint_tol)
    return True


def acquire(controller, camera, point):
    """One bounded native request; no semantic/model call or instance mask."""
    from steward.ego.perception import decode_rgbd
    rt, ex = controller.rt, controller.ex
    after = ex._state()['stamp']
    controller.check()
    # Direct bounded sensor read bypasses segmentation and all model workers.
    ex._wait(.1)
    raw = decode_rgbd(rt.link.rgbd(camera, timeout=FRAME_WAIT_S))
    controller.check()
    return raw if raw['header']['stamp'] > after else None


def measure(est, held, raw, report, *, yaw_limit=None):
    cache = {} if capabilities.get('STEWARD_ASSEMBLY_PARTIAL_RIM_REGISTRATION') != '0' else None
    result = _measure(est, held, raw, report, yaw_limit=yaw_limit, detection_cache=cache)
    if (result is not None or capabilities.get('STEWARD_ASSEMBLY_PARTIAL_RIM_REGISTRATION') == '0'
            or report.get('reason') in ('frame_not_from_current_grasp',
                'no_single_face_layout', 'unsupported_face_orientation')):
        return result
    local = layout(held)
    if local is None or len(local) != 2:
        return None
    original = deepcopy(report)
    context = dict(kind='existing_inhand_interface', measuredCentresRequired=len(local),
                   precisionM=PRECISION_M)
    result = _measure(est, held, raw, report, yaw_limit=yaw_limit, measurement_context=context,
                      detection_cache=cache)
    report['partial_rim_registration'] = dict(attempted=True, originalReason=original.get('reason'),
                         originalHoleCount=original.get('holeCount'), recovered=result is not None)
    return result


def _measure(est, held, raw, report, *, yaw_limit=None, measurement_context=None, detection_cache=None):
    header = raw['header']
    if (not header.get('joints') or not header.get('odometry')
            or header['stamp'] <= est.held_since or not header.get('gripper', {}).get('holding')):
        report['reason'] = 'frame_not_from_current_grasp'
        return None
    local = layout(held)
    if local is None:
        report['reason'] = 'no_single_face_layout'
        return None
    centre, rotation, grip, gr = load_pose(est, held, header)
    if rotation[2, 2] < math.cos(math.radians(25)):
        report['reason'] = 'unsupported_face_orientation'
        return None
    expected = local @ rotation.T+centre
    face = local.copy(); face[:, 2] = held.size[2]/2
    height = float((face @ rotation.T+centre)[:, 2].mean())
    cam = camera_pose(est, header, raw['camera'])
    context = measurement_context
    if detection_cache is not None:
        context = (dict(context, priorDetection=detection_cache.get('found', {})) if context else
                   dict(kind='existing_inhand_interface', admitAppearance=False))
    args = (context,) if context else ()
    found = cr.detect_native(raw, cam, expected[:, :2], height, *args)
    if detection_cache is not None and not measurement_context:
        detection_cache['found'] = found
    report.update(holeCount=len(found['holes']), reason=found['reason'],
        holeRejections=found.get('holeRejections', {}), framing=found.get('framing'))
    if capabilities.get('STEWARD_ASSEMBLY_COMPONENT_MEASUREMENT_ADMISSION') != '0':
        report.update(afterDetectionCount=found.get('afterDetectionCount', 0),
            afterFaceCount=len(found['holes']), searchWindow=found.get('searchWindow'),
            afterOcclusionCount=0, occlusionReasons={})
    # Native ROI depth diagnostics, using the detector's existing search band.
    # A clear nominal ray says nothing about valid depth around a closed rim.
    intr = dict(width=raw['depth'].shape[1], height=raw['depth'].shape[0], fovy=raw['fovy'])
    pad = sh.MAX_LAYOUT_SHIFT
    corners = [[x,y,z] for x in (expected[:,0].min()-pad, expected[:,0].max()+pad)
               for y in (expected[:,1].min()-pad, expected[:,1].max()+pad)
               for z in (height-.03, height+.03)]
    uv, forward = sh.project(corners, cam, intr)
    if np.isfinite(uv).all() and np.all(forward > 0):
        lo = np.maximum(np.floor(uv.min(axis=0)).astype(int), 0)
        hi = np.minimum(np.ceil(uv.max(axis=0)).astype(int)+1, [intr['width'], intr['height']])
        depth = raw['depth'][lo[1]:hi[1], lo[0]:hi[0]]
        if depth.size:
            report['roiInvalidDepthFraction'] = round(float(np.mean(~np.isfinite(depth) | (depth <= 0))), 4)
    if raw.get('_stationary_entrance_layers') or measurement_context:
        report['observedHoles'] = deepcopy(found['holes'])
    if found['reason']:
        return None
    # Preserve the grasp's existing tilt and catalogue Z while fitting XY/yaw.
    # Project in the nominal yaw frame; using local XY directly would double
    # count a tilted feature's Z contribution when reconstructing its TCP pose.
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    projected = (local @ rotation.T)[:, :2] @ cr.rotation(yaw)
    fit, reason = cr.fit_layout(projected, found['holes'], expected[:, :2], yaw,
                                tolerance=PRECISION_M)
    if measurement_context:
        report['candidateAmbiguity'] = dict(candidateCount=len(found['holes']),
            uniqueLayout=fit is not None, reason=reason,
            symmetricYaw=fit.get('symmetricYaw') if fit else None)
    if fit is None:
        report['reason'] = reason
        return None
    if not ray_clear(est, raw['camera'], [h['point'] for h in fit['points']], header):
        report['reason'] = 'robot_occludes_holes'
        if capabilities.get('STEWARD_ASSEMBLY_COMPONENT_MEASUREMENT_ADMISSION') != '0':
            report['occlusionReasons'] = {'robot_ray_occluded': len(fit['points'])}
        return None
    if capabilities.get('STEWARD_ASSEMBLY_COMPONENT_MEASUREMENT_ADMISSION') != '0':
        report.update(afterOcclusionCount=len(fit['points']), occlusionReasons={'ray_clear': len(fit['points'])})
    angle = 0. if fit['mode'] == 'xy' else (fit['yaw']-yaw+math.pi) % math.tau-math.pi
    corrected = centre.copy(); corrected[:2] = fit['position']
    rr = matrix_from_quat(quat_from_yaw(angle)) @ rotation
    offset = gr.T @ (corrected-grip)
    delta = offset-held.held_offset
    uncertainty = fit['uncertaintyM']+fit['residualM']
    report.update(measured=True, deltaMm=np.round(delta*1000, 3).tolist(),
                  uncertaintyMm=round(uncertainty*1000, 3), mode=fit['mode'],
                  yawDeltaDeg=round(math.degrees(angle), 3), observedAt=header['stamp'],
                  minHoleSpanPx=min(min(h['spanPixels']) for h in fit['points']))
    if (np.linalg.norm(delta) > sh.MAX_LAYOUT_SHIFT or abs(angle) >
            (cr.MAX_YAW_CORRECTION_RAD if yaw_limit is None else yaw_limit)):
        report['reason'] = 'correction_limit'
        return None
    if uncertainty > PRECISION_M:
        report['reason'] = 'inhand_uncertainty_exceeded'
        return None
    report['reason'] = None
    if measurement_context:
        for h in fit['points']:
            h['identityStatus'] = 'admitted_by_existing_interface_layout'
            h['candidateAmbiguity'] = deepcopy(report['candidateAmbiguity'])
        report['selectedHoles'] = deepcopy(fit['points'])
    return dict(offset=offset, rotation=gr.T @ rr,
                yaw=math.atan2(rr[1, 0], rr[0, 0])-math.atan2(gr[1, 0], gr[0, 0]),
                uncertaintyM=uncertainty, observedAt=header['stamp'],
                rawHoles=deepcopy(fit['points']),
                holesTcpM=[(gr.T @ (np.asarray(h['point'])-grip)).tolist() for h in fit['points']])


def apply(controller, held, measurement, params, report):
    """Transactional update: the corrected place must pass original risk checks."""
    rt = controller.rt; est = rt.estimator
    with rt._lock:
        if est.held_id != held.id or not held.held or est.held_since >= measurement['observedAt']:
            report['reason'] = 'grasp_changed'
            return
        delta = measurement['offset']-held.held_offset
        angle = (measurement['yaw']-held.held_yaw+math.pi) % math.tau-math.pi
        if np.linalg.norm(delta) > sh.MAX_LAYOUT_SHIFT or abs(angle) > cr.MAX_YAW_CORRECTION_RAD:
            report['reason'] = 'correction_limit'
            return
        if measurement['uncertaintyM'] > PRECISION_M:
            report['reason'] = 'inhand_uncertainty_exceeded'
            return
        report['deltaMm'] = np.round(delta*1000, 3).tolist()
        saved = deepcopy((held.held_offset, held.held_yaw, held.held_interface, held.pos, held.yaw))
        held.held_offset = measurement['offset'].copy()
        held.held_yaw = measurement['yaw']
        held.held_interface.update(offset=measurement['offset'].tolist(), rotation=measurement['rotation'].tolist(),
            observedAt=measurement['observedAt'], source='native_inhand_holes_and_synchronized_fk',
            inhandMeasurement=dict(uncertaintyM=measurement['uncertaintyM'], holesTcpM=measurement['holesTcpM']))
        est._follow_held(); est.version += 1
    try:
        receipt = rt.predict('place', params)
        report['applied'] = bool(receipt.get('withinRiskLimit'))
        report['reason'] = None if report['applied'] else 'corrected_place_not_admitted'
    finally:
        if not report['applied']:
            with rt._lock:
                held.held_offset, held.held_yaw, held.held_interface, held.pos, held.yaw = saved
                est.version += 1


def observe(controller, job, held, params):
    """Use a visible current native frame before paying for a confirming capture.

    No speculative arm tour: the single-hand model does not certify that a
    nominally clear centre ray exposes a closed rim. Missing evidence is cheap.
    """
    from steward.ego.skills import SkillFailed
    rt, ex = controller.rt, controller.ex; est = rt.estimator
    report = dict(measured=False, holeCount=0, deltaMm=None, uncertaintyMm=None,
                  applied=False, reason='no_native_stream', frames=0, viewMotion=False)
    job.postcondition['inhandAlignment'] = report
    if capabilities.get('STEWARD_INHAND') == '0':
        report['reason'] = 'disabled'
        return
    if (not str(getattr(rt, 'perception_mode', '')).startswith('rgbd')
            and getattr(rt, 'segmentation', None) is None
            and not getattr(rt, '_precision_native_frames', None)) or layout(held) is None:
        report['reason'] = 'no_single_face_layout' if layout(held) is None else 'no_native_stream'
        return
    start = time.monotonic()
    try:
        now = ex._state()['stamp']
        # Acquisition is already done by perception. Require this grasp and use
        # the existing native wait budget as the cache freshness budget as well.
        frames = {**getattr(rt, '_precision_native_frames', {}),
                  **getattr(rt, '_latest_native_frames', {})}
        frames = [raw for raw in frames.values()
                  if est.held_since < raw['header']['stamp'] <= now
                  and now-raw['header']['stamp'] <= FRAME_WAIT_S]
        if not frames:
            report['reason'] = 'no_current_native_frame'
            return
        raw = max(frames, key=lambda f:f['header']['stamp'])
        header = raw['header']
        if not header.get('joints') or not header.get('odometry'):
            report['reason'] = 'missing_motion_header'
            return
        intr = dict(width=raw['depth'].shape[1], height=raw['depth'].shape[0], fovy=raw['fovy'])
        geo = geometry(est, held, header, raw['camera'], intr)
        if geo is None:
            report['reason'] = 'current_face_not_visible'
            return
        report.update({k:round(v, 6) if isinstance(v, float) else v for k,v in geo.items() if k != 'points'})
        if not geo['predictedClear']:
            report['reason'] = 'robot_occludes_holes'
            return
        first = measure(est, held, raw, report)
        report['frames'] = 1
        if first is None:
            return
        # Only an actually measured first frame justifies a bounded second read.
        raw = acquire(controller, geo['camera'], geo['points'].mean(axis=0))
        if raw is None:
            report['reason'] = 'no_fresh_native_frame'
            return
        last = measure(est, held, raw, report)
        report['frames'] = 2
        if last is None:
            return
        drift = float(np.max(np.linalg.norm(np.asarray(first['holesTcpM'])-last['holesTcpM'], axis=1)))
        if last['observedAt'] <= first['observedAt'] or drift > first['uncertaintyM']+last['uncertaintyM']:
            report['reason'] = 'inhand_measurement_not_stable'
            return
        last['offset'] = (first['offset']+last['offset'])/2
        last['uncertaintyM'] = max(first['uncertaintyM'], last['uncertaintyM'])+drift/2
        report['uncertaintyMm'] = round(last['uncertaintyM']*1000, 3)
        if last['uncertaintyM'] > PRECISION_M:
            report['reason'] = 'inhand_uncertainty_exceeded'
            return
        apply(controller, held, last, params, report)
        if report['applied'] and 'componentHoleAlignment' in job.postcondition:
            job.postcondition['componentHoleAlignment']['grasp'] = cr.grasp_evidence(
                est, held, dict(local=layout(held)[:, :2]))
    except (SkillFailed, ValueError, KeyError, TypeError, AttributeError, OSError, HttpError) as exc:
        controller.check()
        report.update(applied=False, reason='observation_unavailable', detail=type(exc).__name__)
    finally:
        report['wallS'] = round(time.monotonic()-start, 3)


def refresh_evidence(controller, job, entity):
    """Re-read evidence after observation; the release-time snapshot is premature."""
    with controller.rt._lock:
        evidence = controller.rt.estimator.placement_evidence(entity)
    job.postcondition['placementEvidence'] = evidence
    if evidence.get('deltaXYM') is not None:
        job.log.append('[RSI_PLACE_EVIDENCE] '+entity+' '+evidence['status']+
                       ' deltaXYM='+str(evidence['deltaXYM'])+'; ok=null')
