"""Bounded, pre-release relative entrance servo. Robot RGB-D and encoder FK only.

Two observation configurations, four corrections, one execution retry. A failed
measurement returns to the admitted ASSEMBLY_PLACEMENT_XY_COMPENSATION placement; it never retains the load.
The independent raw entrance samples are kept alongside every fitted command.
"""
from copy import copy, deepcopy
import math
import uuid
import numpy as np
from steward.ego import capabilities, component_registration as cr, inhand, component_measurement_admission, stationary_entrance_layers, base_command_ledger, stationary_reacquisition

MAX_ROUNDS = 4
MAX_PROPAGATION_S = 8.  # short manipulation horizon, not registration-cache age
DAMPING = .65


def enabled():
    return capabilities.get('STEWARD_ASSEMBLY_RELATIVE_ENTRANCE_SERVO') != '0'


def applies(plan):
    return (enabled() and 'placementXYCompensationClearLift' in plan
            and len(plan.get('placementTarget', {}).get('holePairs', [])) == 2)


def wrap(angle):
    return (angle+math.pi) % math.tau-math.pi


def project_axes(points, normals, plane_point, plane_normal):
    """Intersect measured 3-D hole axes with the landing plane, not camera rays."""
    p, a = np.asarray(points, float), np.asarray(normals, float)
    n, q = np.asarray(plane_normal, float), np.asarray(plane_point, float)
    d = a @ n
    if not np.isfinite([p, a]).all() or not np.isfinite([n, q]).all() or np.any(abs(d) < .5):
        raise ValueError('hole_axis_not_transverse_to_landing_plane')
    return p + a * (((q-p) @ n)/d)[:, None]


def pair_error(upper, lower, upper_normals, lower_normals, prior_axis):
    """Score both half-turn correspondences; preserve directed registered identity.

    Fitting scores correspondence only. Control residuals always use raw axes.
    """
    low = np.asarray(lower, float); high = np.asarray(upper, float)
    n = np.mean(lower_normals, axis=0); n /= np.linalg.norm(n)
    u = project_axes(high, upper_normals, low.mean(0), n)[:, :2]
    candidates = []
    prior = math.atan2(prior_axis[1], prior_axis[0])
    for order in ([0, 1], [1, 0]):
        b = low[order, :2]; du, db = u[1]-u[0], b[1]-b[0]
        if min(np.linalg.norm(du), np.linalg.norm(db)) <= np.finfo(float).eps:
            raise ValueError('degenerate_entrance_pair')
        yaw = wrap(math.atan2(db[1], db[0])-math.atan2(du[1], du[0]))
        fitted = (u-u.mean(0)) @ cr.rotation(yaw).T+b.mean(0)
        residual = float(np.max(np.linalg.norm(fitted-b, axis=1)))
        identity = abs(wrap(math.atan2(db[1], db[0])-prior))
        delta = b-u
        candidates.append(dict(order=order, dxM=float(delta[:, 0].mean()),
            dyM=float(delta[:, 1].mean()), dpsiRad=yaw,
            perHoleDeltaM=delta.tolist(), residualM=float(np.max(np.linalg.norm(delta, axis=1))),
            correspondenceFitResidualM=residual, priorAngleRad=identity,
            consistentWithRegistration=bool(identity < math.pi/2)))
    eligible = [c for c in candidates if c['consistentWithRegistration']]
    if not eligible:
        raise ValueError('hole_identity_unresolved')
    chosen = min(eligible, key=lambda c:(c['correspondenceFitResidualM'], c['priorAngleRad']))
    return dict(chosen, candidates=candidates, projectedUpperM=u.tolist(),
                planePointM=low.mean(0).tolist(), planeNormal=n.tolist())


def pose(est, state):
    return (est.gripper_position(est.held_gripper, state['joints'], state['odometry']),
            est.gripper_rotation(est.held_gripper, state['joints'], state['odometry']))


def body_moved(est, body, since):
    ledger = est.stationary_assemblies
    record = ledger.records.get(body, {})
    return (not record or bool(record.get('hardMotion'))
        or record.get('displacementEvidence', {}).get('stamp', -math.inf) > since
        or any(e['entity'] == body and e['stamp'] > since for e in ledger.motion_events))


def measure(controller, plan, camera, after):
    """One synchronous frame supplies both independent entrance sets and TCP FK."""
    if isinstance(camera, dict):
        return stationary_entrance_layers.measure(controller, plan, camera, after)
    from steward.ego import wrist_observation as wo, action_budget as ab, placement_xy_compensation
    ex, rt = controller.ex, controller.rt; est = rt.estimator
    controller.check()
    ex._wait(ex.config.tick_s)
    wo.context(controller, 'relative_entrance_servo', plan['spot'], None)
    if component_measurement_admission.enabled() and capabilities.get('STEWARD_WRIST_DUMP_DIR'):
        controller._wristContext['measureCall'] = component_measurement_admission.replay_context(controller, plan, camera, after)
    sample = wo.capture(controller, ab.make(ex, ab.frame_timeout(ex, 4.), 'relative entrance frame'),
                        camera=camera, check_view=False)
    detail = dict(camera=camera, capture=deepcopy(getattr(controller, '_wristCapture', {})))
    if sample is None:
        return None, dict(detail, reason='fresh_frame_unavailable')
    raw, cam = sample
    try:
        result, outcome = measure_sample(controller, plan, sample, after, detail)
        detail.update(outcome)
        return result, outcome
    finally:
        if component_measurement_admission.enabled() and capabilities.get('STEWARD_WRIST_DUMP_DIR'):
            context = deepcopy(raw.get('_archiveContext', controller._wristContext))
            context['camera'] = camera
            context['measureDetail'] = deepcopy(detail)
            wo._dump(raw, cam, raw['header'], context)


def measure_sample(controller, plan, sample, after, detail=None):
    """Pure measurement half of measure(), shared by live frames and replay."""
    if stationary_entrance_layers.enabled() and '_stationary_entrance_layers' in plan:
        return stationary_entrance_layers.measure_sample(controller, plan, sample, after, detail)
    from steward.ego import placement_xy_compensation
    ex, rt = controller.ex, controller.rt; est = rt.estimator
    raw, cam = sample; header = raw['header']
    camera = raw['camera']
    if detail is None: detail = dict(camera=camera)
    if component_measurement_admission.enabled() and not math.isfinite(header['stamp']):
        return None, dict(detail, reason='invalid_frame_timestamp')
    if header['stamp'] <= max(after, est.held_since):
        return None, dict(detail, reason='frame_not_after_motion')
    held = est.tracks.get(plan['object'])
    # The visibility veto must use the same robot state as the raster too.
    proxy = copy(est); proxy.joints = header['joints']; proxy.odometry = header['odometry']
    proxy.stamp = header['stamp']
    reader = copy(controller); reader.rt = copy(rt); reader.rt.estimator = proxy
    reg, why = placement_xy_compensation.observe(reader, plan, sample)
    detail['body'] = why
    if not stationary_entrance_layers.enabled() and reg is None:
        return None, dict(detail, reason='body_pair_unseen')
    if not stationary_entrance_layers.enabled() and body_moved(est, reg['body'], header['stamp']):
        return None, dict(detail, reason='body_not_stationary')
    hand_report = {}
    hand = inhand.measure(est, held, raw, hand_report, yaw_limit=cr.AXIS_LIMIT)
    detail.update(inhand=hand_report, upperObservation=stationary_entrance_layers.native(hand), lowerObservation=stationary_entrance_layers.native(reg),
                  upperReason=hand_report.get('reason'), lowerReason=why.get('reason'))
    if reg is None:
        return None, dict(detail, reason='body_pair_unseen')
    if body_moved(est, reg['body'], header['stamp']):
        return None, dict(detail, reason='body_not_stationary')
    if hand is None:
        return None, dict(detail, reason=hand_report.get('reason', 'upper_pair_unseen'))
    ids = [h['id'] for h in held.held_interface['holes']]
    indices = [ids.index(p['partHole']) for p in plan['placementTarget']['holePairs']]
    upper = [hand['rawHoles'][i] for i in indices]
    lower = reg['observedBodyHoles']
    # Reject the same rims supplying both layers. Surface separation must be
    # resolved by the actual depth uncertainty, not by a nominal catalogue Z.
    if any(u['point'][2]-b['point'][2] <= u.get('relUncM', u['uncertaintyM'])+
           b.get('relUncM', b.get('uncertaintyM', 0.)) for u, b in zip(upper, lower)):
        return None, dict(detail, reason='entrance_layers_not_separated')
    local = np.asarray(reg['local']); prior_axis = cr.rotation(plan['placementTarget']['yawRad']) @ (local[1]-local[0])
    grip, rotation = pose(est, header)
    # Predict the commanded vertical descent before intersecting axes. For a
    # tilted bore, projecting at hover and then translating vertically do NOT
    # commute. These remain transformed raw samples, never fitted catalogue holes.
    release_state = deepcopy(header); release_state['joints']['lift'] = plan['lift']
    release_grip, release_rotation = pose(est, release_state)
    raw_tcp = np.array([hand['holesTcpM'][i] for i in indices])
    normals_tcp = np.array([rotation.T @ h['normal'] for h in upper])
    error = pair_error(raw_tcp @ release_rotation.T+release_grip, [h['point'] for h in lower],
                       normals_tcp @ release_rotation.T, [h['normal'] for h in lower], prior_axis)
    observation = dict(stamp=header['stamp'], camera=camera, cameraPose=deepcopy(cam),
        robotState={k:deepcopy(header[k]) for k in ('stamp', 'joints', 'odometry')},
        upper=deepcopy(upper), lower=deepcopy(lower), error=error, body=reg['body'],
        bodyRegistration=deepcopy(reg),
        graspSince=est.held_since, gripM=grip.tolist(), gripRotation=rotation.tolist(),
        releaseLift=plan['lift'],
        holesTcpM=[hand['holesTcpM'][i] for i in indices],
        heldOffsetTcpM=hand['offset'].tolist(), heldRotationTcp=hand['rotation'].tolist(),
        heldYawRad=float(hand['yaw']),
        normalsTcp=[(rotation.T @ h['normal']).tolist() for h in upper],
        uncertaintyM=float(hand['uncertaintyM']+reg['uncertaintyM']))
    return observation, dict(detail, reason=None)


def configurations(controller):
    from steward.ego.tip_entry_servo import camera_for
    rt = controller.rt
    if not getattr(getattr(rt, 'estimator', None), 'cameras', None):
        return []
    plan = getattr(controller, '_componentMeasurementAdmissionPlan', None)
    if component_measurement_admission.enabled() and plan is not None:
        available = [c for c in getattr(rt, '_provided_camera_ids', ()) if c in rt.estimator.cameras
                     and rt.estimator.cameras[c][1].get('intrinsics')]
        if stationary_entrance_layers.enabled() and available:
            return [stationary_entrance_layers.configurations(controller, plan, available)]
        ranking = component_measurement_admission.camera_ranking(controller, plan, available)
        plan['relative_entrance_servo']['cameraRanking'] = ranking
        return [r['camera'] for r in ranking[:2]]
    primary = camera_for(rt)
    available = [c for c in getattr(rt, '_provided_camera_ids', ()) if c in rt.estimator.cameras
                 and rt.estimator.cameras[c][1].get('intrinsics')]
    if primary:
        link = rt.estimator.cameras[primary][0]
        available.sort(key=lambda c:rt.estimator.cameras[c][0] == link)
    return list(dict.fromkeys(([primary] if primary else [])+available))[:2]


def predict_holes(est, state, observation):
    landing = (stationary_entrance_layers.local_state(state) if observation.get('coordinateFrame') == 'stationary_robot_base'
               else deepcopy(state))
    if 'releaseLift' in observation:
        landing['joints']['lift'] = observation['releaseLift']
    grip, rotation = pose(est, landing)
    points = np.asarray(observation['holesTcpM']) @ rotation.T+grip
    normals = np.asarray(observation['normalsTcp']) @ rotation.T
    error = observation['error']
    return project_axes(points, normals, error['planePointM'], error['planeNormal'])[:, :2]


def command(controller, observation):
    """Damped numerical FK Jacobian includes wrist/extension/base coupling."""
    ex, est = controller.ex, controller.rt.estimator
    state = ex._state()
    if observation.get('coordinateFrame') == 'stationary_robot_base':
        reason = stationary_entrance_layers.propagation_reason(controller, observation)
        if reason: return None, reason
    base = predict_holes(est, state, observation)
    epsilon = math.sqrt(np.finfo(float).eps)
    columns = []
    for dof in ('reach', 'lateral', 'wrist_yaw'):
        shifted = deepcopy(state)
        if dof == 'lateral' and observation.get('coordinateFrame') == 'stationary_robot_base':
            shifted['_stationaryEntranceLayersLateral'] = epsilon
        elif dof == 'lateral':
            yaw = shifted['odometry']['yaw']
            shifted['odometry']['x'] -= math.sin(yaw)*epsilon
            shifted['odometry']['y'] += math.cos(yaw)*epsilon
        else:
            shifted['joints'][dof] += epsilon
        columns.append(((predict_holes(est, shifted, observation)-base)/epsilon).ravel())
    jac = np.array(columns).T
    span = np.linalg.norm(base[1]-base[0])
    scale = np.array([1., 1., 1./max(span, ex.config.interface_precision_m)])
    scaled = jac*scale
    if np.linalg.matrix_rank(scaled) < 3:
        return None, 'available_dofs_do_not_control_pair'
    delta = np.asarray(observation['error']['perHoleDeltaM']).ravel()
    dq = scale*np.linalg.solve(scaled.T @ scaled+np.eye(3)*(1-DAMPING)**2, scaled.T @ delta)*DAMPING
    opening = controller.rt.esim.arm.geometry.jaw_opening
    step = opening/4
    factor = min(1., step/max(np.linalg.norm(dq[:2]), np.finfo(float).eps),
                 cr.AXIS_LIMIT/max(abs(dq[2]), np.finfo(float).eps))
    return dict(coordinateFrame=observation.get('coordinateFrame', 'world'),
                reachM=float(dq[0]*factor), lateralM=float(dq[1]*factor),
                wristYawRad=float(dq[2]*factor), damping=DAMPING,
                angularToleranceRad=ex.config.interface_precision_m/(2*lever(observation, opening))), None


def lever(observation, opening):
    return max(opening, float(np.max(np.linalg.norm(np.asarray(observation['holesTcpM'])[:, :2], axis=1))))


def move(controller, correction):
    """Use the existing local servo envelope and arm path admission."""
    from steward.ego.skills import SkillFailed
    ex, arm = controller.ex, controller.rt.esim.arm; state = ex._state()
    start = dict(state['joints']); end = dict(start,
        reach=start['reach']+correction['reachM'],
        wrist_yaw=start['wrist_yaw']+correction['wristYawRad'])
    # Actual execution is sequential: test both swept stages, not a diagonal.
    middle = dict(start, wrist_yaw=end['wrist_yaw'])
    path = [dict(start, wrist_yaw=float(v)) for v in np.linspace(start['wrist_yaw'], end['wrist_yaw'], 5)]
    path += [dict(middle, reach=float(v)) for v in np.linspace(start['reach'], end['reach'], 5)]
    if not arm.reachable_path(path, start_joints=state['joints']):
        return False, 'correction_path_unreachable'
    yaw = state['odometry']['yaw']; tol = ex.config.interface_precision_m/2
    local = correction.get('coordinateFrame') == 'stationary_robot_base'
    precision = local and base_command_ledger.enabled()
    basis = None
    if precision:
        receipt = correction['baseCommand'] = base_command_ledger.begin_receipt(ex, correction['lateralM'])
        basis = receipt['baseHoldBefore']
        if not basis['valid'] and correction['lateralM'] != 0.:
            receipt['reason'] = 'precision_base_hold_unavailable'
            base_command_ledger.finish_receipt(ex, receipt)
            return False, receipt['reason']
    current_grip = pose(controller.rt.estimator, state)[0][:2] if local else ex._gripper_xy()
    target = current_grip+np.array([-math.sin(yaw), math.cos(yaw)])*correction['lateralM']
    commanded = deepcopy(state); commanded['joints'] = end
    commanded['odometry']['x'] -= math.sin(yaw)*correction['lateralM']
    commanded['odometry']['y'] += math.cos(yaw)*correction['lateralM']
    if precision and basis['valid']:
        commanded['odometry'].update(zip(('x', 'y', 'yaw'),
            base_command_ledger.target(basis['base'], correction['lateralM'])))
    correction['targetState'] = commanded
    correction['gripTargetM'] = (pose(controller.rt.estimator, commanded)[0][:2].tolist() if local else
        (target+np.array([math.cos(yaw), math.sin(yaw)])*correction['reachM']).tolist())
    try:
        ex._set_joint('wrist_yaw', end['wrist_yaw'], ramp=False,
                      tol=correction.get('angularToleranceRad', tol/arm.geometry.jaw_opening))
        if local:
            # Convert the measured local increment into one actuator setpoint.
            # Re-reading two noisy world poses must not manufacture a motion.
            if precision and correction['lateralM'] != 0.:
                o = commanded['odometry']
                ex._translate_to(o['x'], o['y'], o['yaw'], tol=tol, command_start=basis)
            elif not precision and abs(correction['lateralM']) > np.finfo(float).eps:
                o = commanded['odometry']
                ex._translate_to(o['x'], o['y'], o['yaw'], tol=tol)
        else:
            ex._align_lateral(target.tolist(), tol=tol)
        ex._set_joint('reach', end['reach'], ramp=False, tol=tol)
        return True, None
    except SkillFailed as exc:
        controller.check()
        return False, 'correction_execution_incomplete:'+str(exc)
    finally:
        if precision:
            base_command_ledger.finish_receipt(ex, receipt)


def restore(controller, plan):
    """Return to the original admitted ASSEMBLY_PLACEMENT_XY_COMPENSATION target at clearance, then continue."""
    from steward.ego.skills import SkillFailed
    state = plan['_relative_entrance_servo']; ex = controller.ex
    if not state.get('moved'):
        plan.update(deepcopy(state['original']))
        state['active'] = False
        return
    try:
        ex._set_joint('lift', plan['placementXYCompensationClearLift'])
        if state['original'].get('wristYaw') is not None:
            ex._set_joint('wrist_yaw', state['original']['wristYaw'])
        goal = state['original']['gripSpot']
        ex._align_lateral(goal, tol=ex.config.interface_precision_m)
        ex._set_joint('reach', ex._reach_to(goal), tol=ex.config.interface_precision_m)
    except SkillFailed as exc:
        controller.check()
        plan['relative_entrance_servo']['fallbackMotionFailure'] = str(exc)
    plan.update(deepcopy(state['original']))
    state['active'] = False


def fail(controller, plan, reason, *, restore_target=True):
    plan['relative_entrance_servo'].update(closedLoop=False, reason=reason, fallback='placement_xy_compensation_admitted_placement')
    if component_measurement_admission.enabled():
        plan['relative_entrance_servo'].setdefault('fallbackStartedAt', controller.ex._state()['stamp'])
    if restore_target:
        restore(controller, plan)


def hover(controller, job, plan, report):
    """Run once per actual placement; no recursive replan or confirmation gate."""
    if '_relative_entrance_servo' in plan:
        return
    controller.ex._liftEndpointCompensationArm = controller.rt.esim.arm
    original = {k:deepcopy(plan[k]) for k in ('spot', 'gripSpot', 'wristYaw', 'placementTarget') if k in plan}
    original.setdefault('gripSpot', deepcopy(plan['spot']))
    row = dict(schema='relative_entrance_servo.v1', id=uuid.uuid4().hex, object=plan['object'],
        body=plan['placementTarget'].get('mateReference', {}).get('entity'), closedLoop=False,
        reason='not_established', configurations=[], observations=[], rounds=[],
        lastVisibleError=None, preReleaseExecutionResidual=None, propagation=None,
        executionRetries=0, releaseExecuted=False)
    if stationary_reacquisition.enabled():
        row['placementTarget'] = dict(holePairs=deepcopy(plan['placementTarget']['holePairs']))
    plan['relative_entrance_servo'] = report['relativeEntranceServo'] = row
    plan['_relative_entrance_servo'] = dict(original=original, active=False, last=None, moved=False)
    from steward.ego.geometry_relocalization import eligible, current_reference
    target = plan['placementTarget']; ref = current_reference(target, controller.rt.estimator)
    if ref.get('geometryRelocalization') and not eligible(ref, target.get('componentHoleRegistration')):
        row.update(referencePrecision='coarse', geometryUnconfirmed=True, converged=False)
        return fail(controller, plan, 'geometry_relocalization_required')
    try:
        controller._componentMeasurementAdmissionPlan = plan
        cameras = configurations(controller)
        if not cameras:
            plan['_relative_entrance_servo']['legacy'] = True
            return fail(controller, plan, 'calibrated_camera_unavailable')
        observation = None
        for camera in cameras:
            observation, detail = measure(controller, plan, camera, -math.inf)
            row['configurations'].append(dict(camera=camera, hoverLift=plan['placementXYCompensationClearLift'], detail=detail))
            if observation: break
        if observation is None:
            row['observationFailure'] = deepcopy(row['configurations'])
            return fail(controller, plan, 'both_configurations_unmeasurable')
        row['camera'] = camera
        row['observations'].append(observation)
        row['lastVisibleError'] = deepcopy(observation['error'])
        plan['_relative_entrance_servo']['last'] = observation
        # Even an initially aligned pair is remeasured before claiming a loop.
        for index in range(MAX_ROUNDS):
            previous = observation['error']['residualM']
            correction, reason = command(controller, observation)
            if correction is None and stationary_reacquisition.enabled():
                fresh = stationary_reacquisition.refresh(controller, plan, observation)
                if fresh is not None:
                    observation = fresh
                    previous = observation['error']['residualM']
                    correction, reason = command(controller, observation)
            if correction is None:
                return fail(controller, plan, reason)
            if previous <= controller.ex.config.interface_precision_m:
                correction.update(reachM=0., lateralM=0., wristYawRad=0.)
            attempt = dict(index=index+1, before=deepcopy(observation['error']), correction=correction,
                           after=None, executed=False)
            row['rounds'].append(attempt)
            before_state = controller.ex._state()
            plan['_relative_entrance_servo']['moved'] = True
            executed, reason = move(controller, correction)
            attempt.update(executed=executed, reason=reason)
            after_state = controller.ex._state()
            attempt['execution'] = dict(fromStamp=before_state['stamp'], toStamp=after_state['stamp'],
                reachM=after_state['joints']['reach']-before_state['joints']['reach'],
                wristYawRad=wrap(after_state['joints']['wrist_yaw']-before_state['joints']['wrist_yaw']),
                baseDeltaM=[after_state['odometry'][k]-before_state['odometry'][k] for k in ('x', 'y')])
            if base_command_ledger.enabled():
                attempt['execution']['sampling'] = 'immediately_after_move_return_before_settle'
            if not executed:
                return fail(controller, plan, reason)
            after = controller.ex._state()['stamp']
            observation, detail = measure(controller, plan, camera, after)
            attempt['measurement'] = detail
            if base_command_ledger.enabled():
                attempt['settledExecution'] = base_command_ledger.settled_execution(before_state, observation)
            if observation is None:
                # No more XY/yaw moves from a propagated observation. The
                # short FK-only path is reserved for the final vertical descent.
                return fail(controller, plan, 'remeasurement_unavailable')
            row['observations'].append(observation)
            attempt['after'] = deepcopy(observation['error'])
            attempt['residualDecreased'] = observation['error']['residualM'] < previous
            plan['_relative_entrance_servo']['last'] = observation
            row['lastVisibleError'] = deepcopy(observation['error'])
            residual = observation['error']['residualM']
            if residual <= controller.ex.config.interface_precision_m:
                row.update(reason='measured_convergence', converged=True)
                break
            if residual >= previous:
                return fail(controller, plan, 'error_not_decreasing')
        else:
            row.update(reason='four_round_limit', converged=False)
        state = deepcopy(row['rounds'][-1]['correction']['targetState'])
        grip, _ = pose(controller.rt.estimator, state)
        # Keep the last COMMANDED target. Replacing it with the stopped encoder
        # pose would silently erase execution error before the release check.
        landing = deepcopy(state); landing['joints']['lift'] = plan['lift']
        release_grip, release_rotation = pose(controller.rt.estimator, landing)
        centre = release_grip+release_rotation @ observation['heldOffsetTcpM']
        orientation = release_rotation @ observation['heldRotationTcp']
        plan.update(gripSpot=deepcopy(row['rounds'][-1]['correction']['gripTargetM']), spot=centre[:2].tolist(),
                    wristYaw=state['joints']['wrist_yaw'])
        plan['placementTarget']['yawRad'] = math.atan2(orientation[1, 0], orientation[0, 0])
        if observation.get('bodyRegistration'):
            plan['placementTarget']['componentHoleRegistration'] = stationary_entrance_layers.export_registration(controller, observation)
        plan['_relative_entrance_servo'].update(active=True, targetState=deepcopy(state), targetGrip=grip.tolist())
        if not controller.ex._over_support(plan):
            fail(controller, plan, 'corrected_target_outside_admitted_support')
    except (ValueError, KeyError, TypeError, AttributeError, OSError) as exc:
        fail(controller, plan, 'loop_unavailable:'+type(exc).__name__)
        row['failureDetail'] = str(exc)


def propagation(controller, plan):
    """Synchronized FK from the last actual sample, with strict time/motion bounds."""
    state = controller.ex._state(); saved = plan['_relative_entrance_servo']; last = saved['last']
    if last is None:
        return dict(valid=False, reason='no_last_visible_sample', durationS=None, displacementM=None)
    local = last.get('coordinateFrame') == 'stationary_robot_base'
    grip, rotation = pose(controller.rt.estimator, stationary_entrance_layers.local_state(state) if local else state)
    delta = grip-np.asarray(last['gripM']); age = state['stamp']-last['stamp']
    allowance = abs(last['robotState']['joints']['lift']-plan['lift'])+2*controller.ex.config.interface_precision_m
    reason = stationary_entrance_layers.propagation_reason(controller, last) if local else None
    if reason: pass
    elif body_moved(controller.rt.estimator, last['body'], last['stamp']): reason = 'body_may_have_moved'
    elif controller.rt.estimator.held_since != last['graspSince']: reason = 'grasp_changed'
    elif not 0 <= age <= MAX_PROPAGATION_S: reason = 'propagation_time_exceeded'
    elif np.linalg.norm(delta) > allowance or np.linalg.norm(delta[:2]) > 2*controller.ex.config.interface_precision_m:
        reason = 'propagation_displacement_exceeded'
    elif np.linalg.norm(rotation-np.asarray(last['gripRotation'])) > controller.ex.config.interface_precision_m/lever(last, controller.rt.esim.arm.geometry.jaw_opening):
        reason = 'propagation_orientation_changed'
    result = dict(valid=reason is None, reason=reason, durationS=float(age), displacementM=delta.tolist(),
        distanceM=float(np.linalg.norm(delta)), maxDurationS=MAX_PROPAGATION_S, maxDistanceM=allowance,
        fromStamp=last['stamp'], toStamp=state['stamp'], source='last_raw_entrances_and_synchronized_fk',
        propagatedUpperM=predict_holes(controller.rt.estimator, state, last).tolist())
    if stationary_reacquisition.enabled() and local:
        result['holdEvidence'] = deepcopy(last.get('propagationEvidence'))
    return result


def execution_residual(controller, plan):
    ex = controller.ex
    state = ex._state(); goal = np.asarray(plan['gripSpot'])
    delta = ex._gripper_xy()-goal
    if plan['_relative_entrance_servo']['active']:
        target = deepcopy(plan['_relative_entrance_servo']['targetState'])
        target['joints']['lift'] = plan['lift']
        local = (plan['_relative_entrance_servo'].get('last') or {}).get('coordinateFrame') == 'stationary_robot_base'
        actual, _ = pose(controller.rt.estimator, stationary_entrance_layers.local_state(state) if local else state)
        intended, _ = pose(controller.rt.estimator, stationary_entrance_layers.local_state(target) if local else target)
        delta = (actual-intended)[:2]
    z = state['joints']['lift']-plan['lift']
    yaw = wrap(state['joints']['wrist_yaw']-plan.get('wristYaw', state['joints']['wrist_yaw']))
    return dict(xyM=delta.tolist(), liftM=float(z), wristYawRad=float(yaw),
        positionM=float(np.linalg.norm([*delta, z])), stamp=state['stamp'])


def vertical(ex, target, report=None):
    """Keep the configured vertical ramp, then check its tighter endpoint.

    The executor's ramp branch returns at its ordinary settle tolerance. A
    second endpoint check is necessary; disabling the ramp is not a substitute.
    """
    from steward.ego import lift_endpoint_compensation
    tolerance = ex.config.interface_precision_m/2
    if lift_endpoint_compensation.enabled() and lift_endpoint_compensation.available(ex):
        row = {}
        if report is not None: report.setdefault('liftCompensation', []).append(row)
        return lift_endpoint_compensation.endpoint(ex, target, tolerance, row)
    if report is not None and lift_endpoint_compensation.enabled():
        report['liftFeedbackReason'] = 'feedback_interface_unavailable'
    ex._set_joint('lift', target, tol=tolerance)
    ex._set_joint('lift', target, ramp=False, tol=tolerance)


def descend(ex, plan):
    """A precise descent timeout is evidence for the bounded hover retry."""
    from steward.ego.skills import SkillFailed
    try:
        if component_measurement_admission.enabled() and not plan.get('_relative_entrance_servo', {}).get('active', False):
            ex._set_joint('lift', plan['lift'])
        else:
            vertical(ex, plan['lift'], plan['relative_entrance_servo'])
    except SkillFailed as exc:
        plan['relative_entrance_servo']['descentFailure'] = str(exc)
        feedback = plan['relative_entrance_servo'].get('liftCompensation', [])
        if feedback and not feedback[-1]['confirmed']:
            plan['_relative_entrance_servo']['liftFeedbackFailed'] = feedback[-1]['reason']


def retain_hand(controller, observation):
    """Release bookkeeping uses the measured hand transform, not the old grasp."""
    est = controller.rt.estimator
    held = est.tracks.get(est.held_id)
    if held is None or est.held_since != observation['graspSince']:
        return
    registration = stationary_entrance_layers.export_registration(controller, observation)
    with controller.rt._lock:
        held.held_offset = np.asarray(observation['heldOffsetTcpM'])
        rotation = np.asarray(observation['heldRotationTcp'])
        held.held_interface.update(offset=held.held_offset.tolist(), rotation=rotation.tolist(),
            observedAt=observation['stamp'], source='native_inhand_holes_and_synchronized_fk',
            inhandMeasurement=dict(uncertaintyM=observation['uncertaintyM'],
                                   holesTcpM=deepcopy(observation['holesTcpM'])))
        held.held_yaw = observation['heldYawRad']
        record = est.stationary_assemblies.records.get(observation['body'])
        if registration and record:
            # Only this interface's raw observation is refreshed. The body's
            # other interfaces and historical release receipts are untouched.
            record['reference']['componentHoleRegistration'] = deepcopy(registration)
        est._follow_held(); est.version += 1


def finish(controller, plan):
    """Immediately before opening: check encoders, expose/retry at most once."""
    from steward.ego.skills import SkillFailed
    if '_relative_entrance_servo' not in plan: return
    ex = controller.ex; row = plan['relative_entrance_servo']; saved = plan['_relative_entrance_servo']
    if saved.get('legacy') or (component_measurement_admission.enabled() and not saved['active']):
        try: row['preReleaseExecutionResidual'] = execution_residual(controller, plan)
        except (KeyError, TypeError, AttributeError, ValueError):
            row['executionResidualReason'] = 'encoder_geometry_unavailable'
        return
    if saved.get('liftFeedbackFailed'):
        fail(controller, plan, saved['liftFeedbackFailed'])
        try: ex._set_joint('lift', plan['lift'])
        except SkillFailed as exc:
            controller.check(); row['fallbackDescentFailure'] = str(exc)
        try: row['preReleaseExecutionResidual'] = execution_residual(controller, plan)
        except (KeyError, TypeError, AttributeError, ValueError):
            row['executionResidualReason'] = 'encoder_geometry_unavailable'
        return
    tolerance = ex.config.interface_precision_m
    radius = lever(saved['last'], controller.rt.esim.arm.geometry.jaw_opening) if saved['last'] else controller.rt.esim.arm.geometry.jaw_opening
    def reached(error):
        return error['positionM'] <= tolerance and abs(error['wristYawRad'])*radius <= tolerance
    def checked_propagation():
        prop = propagation(controller, plan) if saved['active'] else None
        if stationary_reacquisition.enabled() and prop and not prop['valid']:
            row.setdefault('propagationChecks', []).append(deepcopy(prop))
            fresh = stationary_reacquisition.refresh(controller, plan, saved['last'])
            if fresh is not None:
                # A fresh pair must still satisfy the existing control bound.
                # Stationarity alone never licenses release using an old residual.
                if fresh['error']['residualM'] > tolerance:
                    return dict(valid=False, reason='stationary_reacquired_pair_outside_control_bound')
                prop = propagation(controller, plan)
        return prop
    try:
        error = execution_residual(controller, plan)
        prop = checked_propagation()
        if stationary_reacquisition.enabled():
            error = execution_residual(controller, plan)
        row['executionChecks'] = [error]
        if not reached(error) or (prop and not prop['valid']):
            row['executionRetries'] = 1
            row['retryReason'] = prop['reason'] if prop and not prop['valid'] else 'encoder_residual_exceeded'
            vertical(ex, plan['placementXYCompensationClearLift'], row)
            ex._align_lateral(plan['gripSpot'], tol=tolerance/2)
            ex._set_joint('reach', ex._reach_to(plan['gripSpot']), ramp=False, tol=tolerance/2)
            ex._set_joint('wrist_yaw', plan.get('wristYaw', ex._state()['joints']['wrist_yaw']), ramp=False,
                          tol=tolerance/radius)
            if saved['active']:
                observation, detail = measure(controller, plan, row['camera'], ex._state()['stamp'])
                row['retryObservation'] = detail
                if observation is None:
                    fail(controller, plan, 'retry_pair_unseen')
                else:
                    saved['last'] = observation; row['observations'].append(observation)
                    row['lastVisibleError'] = deepcopy(observation['error'])
                    if observation['error']['residualM'] > tolerance:
                        fail(controller, plan, 'retry_pair_outside_control_bound')
            if component_measurement_admission.enabled() and not saved['active']:
                ex._set_joint('lift', plan['lift'])
            else:
                vertical(ex, plan['lift'], row)
            ex._settle_for_release()
            error = execution_residual(controller, plan); row['executionChecks'].append(error)
        row['preReleaseExecutionResidual'] = error
        row['propagation'] = checked_propagation() if saved['active'] else prop
        row['closedLoop'] = bool(saved['active'] and reached(error) and row['propagation']['valid'])
        if saved['active'] and not row['closedLoop']:
            fail(controller, plan, 'execution_or_short_propagation_unverified')
            ex._set_joint('lift', plan['lift'])
            row['preReleaseExecutionResidual'] = execution_residual(controller, plan)
        elif not reached(error) and not component_measurement_admission.enabled():
            row['reason'] = 'encoder_residual_unconfirmed_continue'
        if row['closedLoop']:
            if saved['last'].get('bodyRegistration'):
                plan['placementTarget']['componentHoleRegistration'] = stationary_entrance_layers.export_registration(controller, saved['last'])
            retain_hand(controller, saved['last'])
    except (SkillFailed, ValueError, KeyError, TypeError, AttributeError, OSError) as exc:
        controller.check()
        feedback = row.get('liftCompensation', [])
        reason = (feedback[-1]['reason'] if feedback and not feedback[-1]['confirmed']
                  else 'precision_execution_unavailable:'+type(exc).__name__)
        fail(controller, plan, reason)
        # An observation or precise servo timeout must not bypass the release.
        try: ex._set_joint('lift', plan['lift'])
        except SkillFailed as fallback:
            controller.check(); row['fallbackDescentFailure'] = str(fallback)
    finally:
        try:
            row['preReleaseExecutionResidual'] = execution_residual(controller, plan)
        except (ValueError, KeyError, TypeError, AttributeError):
            row['executionResidualReason'] = 'encoder_geometry_unavailable'


def released(controller, job, plan):
    if '_relative_entrance_servo' not in plan or plan['_relative_entrance_servo'].get('emitted'): return
    from steward.ego.capability_usage import record
    row = plan['relative_entrance_servo']
    if row.get('restoredPrior'): return
    from steward.ego.geometry_relocalization import provenance
    row.update(provenance(plan.get('placementTarget'), plan.get('releasedAt'), controller.rt.estimator))
    row['placementId'] = row.get('placementId') or row['id']
    if row['geometryUnconfirmed'] and plan.get('placementTarget', {}).get('mateReference', {}).get('geometryRelocalization'):
        row.update(closedLoop=False, converged=False)
    row.update(releaseExecuted=bool(plan.get('releaseExecuted')), releasedAt=plan.get('releasedAt'))
    if component_measurement_admission.enabled() and row.get('fallbackStartedAt') is not None and row.get('releasedAt') is not None:
        row['fallbackDurationS'] = max(0., row['releasedAt']-row['fallbackStartedAt'])
    report = plan.get('precision')
    if report is not None:
        report['relativeEntranceServo'] = row
    if report is not None and not plan['_relative_entrance_servo'].get('legacy'):
        report['fallback'] = None if row['closedLoop'] else row['reason']
        report['beforeResidualM'] = row['observations'][0]['error']['residualM'] if row['observations'] else None
        report['preReleaseResidualM'] = row['lastVisibleError']['residualM'] if row['closedLoop'] else None
        report.setdefault('corrections', []).extend(dict(source='raw_relative_entrance_servo', executed=r['executed'],
            correction=deepcopy(r['correction'])) for r in row['rounds'] if any(
                abs(r['correction'][k]) > 0 for k in ('reachM', 'lateralM', 'wristYawRad')))
    if job.postcondition is None: job.postcondition = {}
    job.postcondition['relativeEntranceServo'] = deepcopy(row)
    stored = controller.rt.estimator.released_interfaces.get(plan['object'])
    if stored is not None: stored['relativeEntranceServo'] = deepcopy(row)
    if not plan['_relative_entrance_servo'].get('emitted'):
        record('relative_entrance_servo', entity=plan['object'], event_id=row['id'], stage='release',
               measured=row['closedLoop'], action=row['releaseExecuted'], reason=row['reason'], evidence=row)
        plan['_relative_entrance_servo']['emitted'] = True
