"""Two independently visible entrance layers in one stationary robot-base frame.

Sensor-clock windows, encoder FK and calibrated mounts only. World poses are
used for search priors and exported bookkeeping, never to join measured points.
"""
from collections import deque
from copy import copy, deepcopy
import math
import uuid
import numpy as np
from steward.ego import capabilities, component_registration as cr, component_measurement_admission, stationary_reacquisition, inhand
from steward.math3d import pose_compose

PAIR_SKEW_S = .1  # sensor synchronization contract, independent of task geometry
FRAME_AGE_S = .25  # native capture freshness contract


def native(value):
    """Receipt writers require ordinary JSON, including rejection evidence."""
    if isinstance(value, np.ndarray): return native(value.tolist())
    if isinstance(value, np.generic): return native(value.item())
    if isinstance(value, dict): return {k:native(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [native(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def enabled():
    return capabilities.get('STEWARD_ASSEMBLY_STATIONARY_ENTRANCE_LAYERS') != '0'


def local_state(state):
    result = deepcopy(state)
    result['odometry'] = dict(state['odometry'], x=0., y=0., yaw=0.)
    # Only the numerical control Jacobian may translate this local origin.
    result['odometry']['y'] = state.get('_stationaryEntranceLayersLateral', 0.)
    return result


def xy_to_base(xy, odometry):
    return (np.asarray(xy)-[odometry['x'], odometry['y']]) @ cr.rotation(odometry['yaw'])


def compact_state(state):
    return {k:deepcopy(state[k]) for k in ('stamp', 'episode', 'joints', 'odometry', 'gripper', 'grippers') if k in state}


class MotionRecord:
    """Observe every executor read/command, including intermediate out-and-back moves."""
    def __init__(self):
        self.states = deque(maxlen=1024)
        self.commands = deque(maxlen=1024)
        self.sequence = self.base_epoch = self.grasp_epoch = self.read_sequence = 0
        self.targets = {}

    def read(self, state):
        row = compact_state(state)
        self.read_sequence += 1
        row['_readSequence'] = self.read_sequence
        if self.states and row['stamp'] == self.states[-1]['stamp'] and (not stationary_reacquisition.enabled() or row.get('episode') == self.states[-1].get('episode')):
            self.states[-1] = row
        else:
            self.states.append(row)
        return state

    def command(self, command):
        self.sequence += 1
        if 'base' in command: self.base_epoch += 1
        if 'gripper' in command: self.grasp_epoch += 1
        self.targets.update(deepcopy(command))
        self.commands.append(dict(sequence=self.sequence,
            stamp=self.states[-1]['stamp'] if self.states else None,
            command=deepcopy(command)))


def monitor(ex):
    record = getattr(ex, '_stationaryEntranceLayersMotion', None)
    if record is None:
        record = ex._stationaryEntranceLayersMotion = MotionRecord()
        record.targets = deepcopy(getattr(ex, '_setpoints', {}))
        read, send = ex._state, ex._command
        def state():
            return record.read(read())
        def command(value):
            # Mark before transport, so partial execution or a failed send
            # cannot leave an old measurement window apparently admissible.
            record.command(value)
            return send(value)
        ex._state, ex._command = state, command
    return record


def motion_reason(states, max_gap, tolerance, est, *, vertical=False):
    if stationary_reacquisition.enabled():
        return stationary_reacquisition.assess(states, max_gap, tolerance, est, vertical=vertical)['reason']
    return legacy_motion_reason(states, max_gap, tolerance, est, vertical=vertical)


def legacy_motion_reason(states, max_gap, tolerance, est, *, vertical=False):
    """Check the entire sensor trace, with noise-aware odometry admission."""
    from steward.ego.relative_entrance_servo import pose, wrap
    if len(states) < 2: return 'continuous_motion_record_unavailable'
    stamps = np.array([s['stamp'] for s in states])
    if not np.isfinite(stamps).all() or np.any(np.diff(stamps) <= 0):
        return 'motion_clock_not_monotonic'
    if np.max(np.diff(stamps)) > max_gap+1e-8: return 'motion_record_gap'
    first = states[0]; g0, r0 = pose(est, local_state(first))
    for state in states:
        if state.get('episode') != first.get('episode'): return 'episode_changed'
        if not state.get('gripper', {}).get('holding'): return 'grasp_not_held'
        g, r = pose(est, local_state(state))
        displacement = g-g0
        if vertical: displacement[2] = 0.
        if np.linalg.norm(displacement) > tolerance/2 or np.linalg.norm(r-r0) > tolerance/max(np.linalg.norm(g0), tolerance):
            return 'actuator_not_holding'
        a, b = first['odometry'], state['odometry']
        for key in ('x', 'y', 'yaw'):
            sigma_key = 'yaw' if key == 'yaw' else 'pos'
            noise = math.hypot(a.get('sigma', {}).get(sigma_key, 0.), b.get('sigma', {}).get(sigma_key, 0.))
            delta = wrap(b[key]-a[key]) if key == 'yaw' else b[key]-a[key]
            if not math.isfinite(delta) or abs(delta) > max(3*noise, np.finfo(float).eps):
                return 'base_motion_evidence'
    return None


def _begin_window_once(controller, after):
    ex, est = controller.ex, controller.rt.estimator
    record = monitor(ex)
    first = ex._state()
    start = max(after, first['stamp'])
    duration = max(ex.config.settle_s, 2*ex.config.tick_s)
    sequence = record.sequence
    first_read = record.read_sequence
    grasp = (est.held_since, est.held_id)
    samples = [compact_state(first)]
    # A bounded settle, no extra view motion or repeated precision attempts.
    for _ in range(math.ceil(duration/ex.config.tick_s)+2):
        ex._wait(ex.config.tick_s)
        state = ex._state()
        if state['stamp'] > samples[-1]['stamp'] or (stationary_reacquisition.enabled() and state['stamp'] < samples[-1]['stamp']): samples.append(compact_state(state))
        if state['stamp']-start >= duration: break
    gap = max(2*ex.config.tick_s, PAIR_SKEW_S)
    evidence = None
    if stationary_reacquisition.enabled():
        # Include reads inside _wait as well; receipts must not thin a trace.
        samples = [deepcopy(s) for s in record.states if s['_readSequence'] >= first_read]
        evidence = stationary_reacquisition.assess(samples, gap, ex.config.interface_precision_m, est)
        reason = evidence['reason']
    else:
        reason = motion_reason(samples, gap, ex.config.interface_precision_m, est)
    if samples[-1]['stamp']-start < duration and (evidence is None or evidence['decision'] != 'reject'):
        reason = 'hold_window_incomplete'
    if sequence != record.sequence: reason = 'command_during_hold_entry'
    if stationary_reacquisition.enabled():
        if samples[0]['stamp'] > start: reason = 'motion_record_truncated'
        if grasp != (est.held_since, est.held_id): reason = 'grasp_changed'
    from steward.ego import base_command_ledger
    base_hold = None
    if base_command_ledger.enabled():
        base_hold = base_command_ledger.snapshot(ex)
        record.targets.pop('base', None)
        if base_hold['valid']:
            record.targets['base'] = deepcopy(base_hold['base'])
    window = dict(id=uuid.uuid4().hex, after=max(start, samples[-1]['stamp']),
        holdEstablished=reason is None, holdMode='position_servo_stable_encoder_window', holdSamples=samples,
        traceStart=record.read_sequence, commandSequence=record.sequence, baseEpoch=record.base_epoch,
        graspEpoch=record.grasp_epoch, graspSince=est.held_since, heldId=est.held_id,
        episode=state.get('episode'), maxGapS=gap, maxSkewS=PAIR_SKEW_S,
        holdTargets=deepcopy(record.targets),
        holdCommands=[deepcopy(row) for row in record.commands if row['sequence'] > sequence], reason=reason)
    if evidence is not None:
        if reason != evidence['reason']:
            evidence.update(decision='reject', reason=reason)
        evidence['commandSequence'] = record.sequence
        window['holdEvidence'] = evidence
        window['durationS'] = duration
    if base_hold is not None:
        window['baseHoldCommand'] = base_hold
    return native(window)



def begin_window(controller, after, *, resample=True, prior=None):
    window = _begin_window_once(controller, after)
    if not stationary_reacquisition.enabled(): return window
    windows = [deepcopy(window)]
    if prior is None and window['holdEvidence']['decision'] == 'suspect' and resample:
        prior = window['holdEvidence']
        window = _begin_window_once(controller, window['after'])
        windows.append(deepcopy(window))
    if prior is not None:
        evidence = window['holdEvidence']
        combined = stationary_reacquisition.join_traces(prior.get('states', []), evidence['states'])
        # The earlier trace may include vertical descent; its own encoder
        # validity was checked at the originating entrance.
        continuity = stationary_reacquisition.assess(combined, window['maxGapS'],
            controller.ex.config.interface_precision_m, controller.rt.estimator, vertical=True)
        window['continuityEvidence'] = continuity
        if continuity['decision'] == 'reject':
            evidence.update(decision='reject', reason=continuity['reason'])
        previous = prior.get('commandSequence')
        if previous is not None and previous != window['commandSequence']:
            evidence.update(decision='reject', reason='command_during_stationary_resample')
        window['priorEvidence'] = deepcopy(prior)
        evidence['resampleCount'] = 1
        window['reason'] = evidence['reason']
        window['holdEstablished'] = evidence['decision'] == 'accept'
    window['holdEvidence']['commandSequence'] = window['commandSequence']
    window['holdEvidence']['finalDecision'] = 'accept' if window['holdEstablished'] else 'reject'
    window['holdWindows'] = windows
    return native(window)


def measure_sample(controller, plan, sample, after, detail=None):
    """Return both independent layer outcomes, including measured single rims."""
    from steward.ego import placement_xy_compensation, relative_entrance_servo
    raw, _ = sample; header = raw['header']; est = controller.rt.estimator
    detail = dict(detail or {}, camera=raw['camera'])
    if not math.isfinite(header['stamp']): return None, dict(detail, reason='invalid_frame_timestamp')
    if header['stamp'] <= max(after, est.held_since): return None, dict(detail, reason='frame_not_after_motion')
    if not header.get('gripper', {}).get('holding') or est.held_id != plan['object']:
        return None, dict(detail, reason='frame_not_from_current_grasp')
    state = local_state(header)
    proxy = copy(est); proxy.joints = state['joints']; proxy.odometry = state['odometry']; proxy.stamp = state['stamp']
    # An archived camera world-pose override is never the ASSEMBLY_STATIONARY_ENTRANCE_LAYERS calibration.
    proxy.__dict__.pop('camera_pose', None)
    camera = raw['camera']; link, sensor = proxy.cameras[camera]
    cam = pose_compose(proxy.link_pose(link, state['joints'], state['odometry']), sensor['mount'])
    local_raw = dict(raw, header=state, _stationary_entrance_layers=True)
    local_plan = deepcopy(plan); odom = header['odometry']
    local_plan['spot'] = xy_to_base(plan['spot'], odom).tolist()
    local_plan['placementTarget']['yawRad'] -= odom['yaw']
    reader = copy(controller); reader.rt = copy(controller.rt); reader.rt.estimator = proxy
    try:
        lower, lower_detail = placement_xy_compensation.observe(reader, local_plan, (local_raw, cam))
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        lower, lower_detail = None, dict(reason='lower_measurement_unavailable:'+type(exc).__name__)
    if lower is not None and relative_entrance_servo.body_moved(est, lower['body'], header['stamp']):
        lower = None; lower_detail['reason'] = 'body_not_stationary'
    upper_detail = {}
    try:
        upper = inhand.measure(proxy, est.tracks[plan['object']], local_raw, upper_detail, yaw_limit=cr.AXIS_LIMIT)
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        upper = None; upper_detail['reason'] = 'upper_measurement_unavailable:'+type(exc).__name__
    detail.update(body=lower_detail, inhand=upper_detail)
    local, _, _ = component_measurement_admission.body_search(local_plan)
    prior = cr.rotation(local_plan['placementTarget']['yawRad']) @ (local[1]-local[0])
    result = dict(coordinateFrame='stationary_robot_base', camera=camera, stamp=header['stamp'],
        robotState=compact_state(header), cameraToBase=deepcopy(cam),
        cameraLink=link, cameraMount=deepcopy(sensor['mount']),
        graspSince=est.held_since, heldId=est.held_id,
        upperObservation=upper, lowerObservation=lower,
        upperDetail=upper_detail, lowerDetail=lower_detail,
        priorAxis=prior.tolist(), windowId=plan.get('_stationary_entrance_layers', {}).get('id'))
    detail.update(upperObservation=deepcopy(upper), lowerObservation=deepcopy(lower),
        upperReason=upper_detail.get('reason'), lowerReason=lower_detail.get('reason'), reason=None)
    return native(result), native(detail)


def joint(controller, plan, upper_frame, lower_frame, window, trace):
    """Join only actual independent pairs; admission precedes all control use."""
    from steward.ego import relative_entrance_servo as pb
    est = controller.rt.estimator
    sources = dict(upper=deepcopy(upper_frame), lower=deepcopy(lower_frame))
    detail = dict(window=deepcopy(window), sources=sources, motionTrace=deepcopy(trace), reason=None)
    def reject(reason):
        detail['reason'] = reason
        return None, native(detail)
    if upper_frame is None or lower_frame is None: return reject('layer_frame_unavailable')
    # Missing image evidence is the primary cause, even if the failed frames aged.
    # Successful pairs still traverse every original time and motion gate below.
    if upper_frame.get('upperObservation') is None or lower_frame.get('lowerObservation') is None:
        detail['missingLayers'] = [name for name, frame in sources.items()
                                  if frame.get(name+'Observation') is None]
        return reject('independent_layer_pair_unavailable')
    frames = [upper_frame, lower_frame]
    if any(f.get('coordinateFrame') != 'stationary_robot_base' for f in frames): return reject('coordinate_frame_mismatch')
    stamps = [f['stamp'] for f in frames]
    if any(not math.isfinite(t) for t in stamps): return reject('invalid_frame_timestamp')
    if any(f.get('windowId') != window['id'] for f in frames): return reject('measurement_window_changed')
    if any(f['graspSince'] != window['graspSince'] or f['heldId'] != window['heldId'] for f in frames) or est.held_since != window['graspSince'] or est.held_id != window['heldId']:
        return reject('grasp_changed')
    if any(f['robotState'].get('episode') != window.get('episode') for f in frames): return reject('episode_changed')
    if min(stamps) <= window['after']: return reject('frame_not_after_motion')
    if max(stamps)-min(stamps) > PAIR_SKEW_S+1e-8: return reject('pair_timestamp_skew')
    if not window.get('holdEstablished'): return reject(window.get('reason') or 'actuator_hold_unconfirmed')
    if not trace or trace[0]['stamp'] > window['after'] or trace[-1]['stamp'] < max(stamps): return reject('motion_record_does_not_cover_pair')
    if trace[-1]['stamp']-min(stamps) > FRAME_AGE_S+1e-8: return reject('stale_pair')
    record = getattr(controller.ex, '_stationaryEntranceLayersMotion', None)
    if record is None: return reject('continuous_motion_record_unavailable')
    detail['controlProof'] = dict(commandSequence=record.sequence, baseEpoch=record.base_epoch,
        graspEpoch=record.grasp_epoch, commands=[deepcopy(row) for row in record.commands
            if row['sequence'] > window['commandSequence']])
    if record.sequence != window['commandSequence']: return reject('command_during_measurement')
    if record.base_epoch != window['baseEpoch']: return reject('base_moved_new_window_required')
    if record.grasp_epoch != window['graspEpoch']: return reject('grasp_changed')
    if stationary_reacquisition.enabled():
        evidence_trace = stationary_reacquisition.join_traces(window['holdSamples'], trace)
        evidence = stationary_reacquisition.assess(evidence_trace, window['maxGapS'], controller.ex.config.interface_precision_m, est)
        evidence['commandSequence'] = record.sequence
        evidence['resampleCount'] = window.get('holdEvidence', {}).get('resampleCount', 0)
        detail['holdEvidence'] = evidence
        reason = evidence['reason']
    else:
        reason = motion_reason(trace, window['maxGapS'], controller.ex.config.interface_precision_m, est)
    if reason: return reject(reason)
    hand, reg = upper_frame['upperObservation'], lower_frame['lowerObservation']
    if hand is None or reg is None: return reject('independent_layer_pair_unavailable')
    if len(hand['rawHoles']) != 2 or len(reg['observedBodyHoles']) != 2: return reject('two_measured_holes_per_layer_required')
    if pb.body_moved(est, reg['body'], min(stamps)): return reject('body_not_stationary')
    held = est.tracks[plan['object']]
    ids = [h['id'] for h in held.held_interface['holes']]
    indices = [ids.index(p['partHole']) for p in plan['placementTarget']['holePairs']]
    upper = [hand['rawHoles'][i] for i in indices]; lower = reg['observedBodyHoles']
    if any(u['point'][2]-b['point'][2] <= u.get('relUncM', u['uncertaintyM'])+b.get('relUncM', b.get('uncertaintyM', 0.)) for u,b in zip(upper,lower)):
        return reject('entrance_layers_not_separated')
    state = local_state(upper_frame['robotState']); grip, rotation = pb.pose(est, state)
    landing = deepcopy(state); landing['joints']['lift'] = plan['lift']
    release_grip, release_rotation = pb.pose(est, landing)
    tcp = np.asarray([hand['holesTcpM'][i] for i in indices])
    ntcp = np.asarray([rotation.T @ h['normal'] for h in upper])
    error = pb.pair_error(tcp @ release_rotation.T+release_grip, [h['point'] for h in lower],
        ntcp @ release_rotation.T, [h['normal'] for h in lower], upper_frame['priorAxis'])
    observation = dict(coordinateFrame='stationary_robot_base', window=deepcopy(window),
        stamp=max(stamps), camera={k:f['camera'] for k,f in sources.items()}, sources=sources,
        robotState=deepcopy(upper_frame['robotState']), upper=deepcopy(upper), lower=deepcopy(lower),
        error=error, body=reg['body'], bodyRegistration=deepcopy(reg), graspSince=est.held_since,
        gripM=grip.tolist(), gripRotation=rotation.tolist(), releaseLift=plan['lift'],
        holesTcpM=tcp.tolist(), normalsTcp=ntcp.tolist(), heldOffsetTcpM=np.asarray(hand['offset']).tolist(),
        heldRotationTcp=np.asarray(hand['rotation']).tolist(), heldYawRad=float(hand['yaw']),
        uncertaintyM=float(hand['uncertaintyM']+reg['uncertaintyM']))
    detail['jointResidual'] = deepcopy(error)
    return native(observation), native(detail)


def configurations(controller, plan, available):
    from steward.ego import component_measurement_admission
    rankings = {layer:component_measurement_admission.camera_ranking(controller, plan, available, layer=layer)
                for layer in ('upper', 'lower')}
    plan['relative_entrance_servo']['cameraRanking'] = rankings
    return rankings


def measure(controller, plan, rankings, after, *, resample=True, prior=None):
    result, detail = _measure_once(controller, plan, rankings, after, resample=resample, prior=prior)
    if not stationary_reacquisition.enabled(): return result, detail
    evidence = detail.get('holdEvidence', {})
    used = detail['window'].get('holdEvidence', {}).get('resampleCount', 0)
    if result is None and evidence.get('decision') == 'suspect' and resample and not used:
        result, final = _measure_once(controller, plan, rankings, controller.ex._state()['stamp'],
                                     resample=False, prior=evidence)
        final['previousMeasurement'] = detail
        return result, native(final)
    return result, detail


def _measure_once(controller, plan, rankings, after, *, resample=True, prior=None):
    from steward.ego import wrist_observation as wo, action_budget as ab, component_measurement_admission
    ex, est = controller.ex, controller.rt.estimator
    window = begin_window(controller, after, resample=resample, prior=prior)
    plan['_stationary_entrance_layers'] = window
    detail = dict(window=window, frames=[], reason=window['reason'])
    if not window['holdEstablished']: return None, native(detail)
    # Capture at most two views. Use actual first-frame results when choosing
    # the second role; never search for one camera that must see both layers.
    frames = []; used = []
    camera = rankings['upper'][0]['camera']
    ex._wait(ex.config.tick_s)
    for index in range(2):
        controller.check()
        wo.context(controller, 'stationary_entrance_layers', plan['spot'], None)
        if capabilities.get('STEWARD_WRIST_DUMP_DIR'):
            controller._wristContext['measureCall'] = component_measurement_admission.replay_context(controller, plan, camera, window['after'])
            controller._wristContext['measureCall']['stationaryEntranceLayersWindow'] = deepcopy(window)
        sample = wo.capture(controller, ab.make(ex, ab.frame_timeout(ex, 4.), 'joint entrance frames'), camera=camera, check_view=False)
        used.append(camera)
        if sample is None:
            frame = None; outcome = dict(reason='fresh_frame_unavailable', capture=deepcopy(getattr(controller, '_wristCapture', {})))
        else:
            frame, outcome = measure_sample(controller, plan, sample, window['after'])
            if capabilities.get('STEWARD_WRIST_DUMP_DIR'):
                raw, cam = sample
                context = deepcopy(raw.get('_archiveContext', controller._wristContext))
                context['measureDetail'] = deepcopy(outcome)
                wo._dump(raw, cam, raw['header'], context)
        detail['frames'].append(dict(camera=camera, detail=outcome))
        if frame: frames.append(frame)
        uppers = [f for f in frames if f['upperObservation'] is not None]
        lowers = [f for f in frames if f['lowerObservation'] is not None]
        if uppers and lowers: break
        missing = 'upper' if lowers and not uppers else 'lower'
        choices = [r['camera'] for r in rankings[missing] if r['camera'] not in used]
        if not choices: break
        camera = choices[0]
    ex._state()
    record = ex._stationaryEntranceLayersMotion
    trace = [s for s in record.states if s['_readSequence'] >= window.get('traceStart', 0)]
    # Preserve the best partial layer even when neither frame yielded a pair.
    def best(layer):
        return max(frames, key=lambda f:(f[layer+'Observation'] is not None,
            f[layer+'Detail'].get('afterOcclusionCount', f[layer+'Detail'].get('holeCount', 0)))) if frames else None
    result, joined = joint(controller, plan, best('upper'), best('lower'), window, trace)
    detail.update(joined)
    return native(result), native(detail)


def propagation_reason(controller, observation):
    if stationary_reacquisition.enabled():
        observation.pop('propagationEvidence', None)
    reason = _propagation_reason(controller, observation)
    if stationary_reacquisition.enabled() and 'propagationEvidence' not in observation:
        record = getattr(controller.ex, '_stationaryEntranceLayersMotion', None)
        observation['propagationEvidence'] = dict(decision='reject', reason=reason,
            states=[deepcopy(s) for s in record.states] if record else [], resampleCount=0,
            commands=[deepcopy(r) for r in record.commands] if record else [])
    return reason


def _propagation_reason(controller, observation):
    record = getattr(controller.ex, '_stationaryEntranceLayersMotion', None)
    window = observation['window']
    if record is None: return 'continuous_motion_record_unavailable'
    if stationary_reacquisition.enabled():
        est = controller.rt.estimator
        if est.held_since != window['graspSince'] or est.held_id != window['heldId']:
            return 'grasp_changed'
        if any(row['sequence'] > window['commandSequence'] and any(k != 'lift' for k in row['command'])
               for row in record.commands):
            # Vertical-only propagation is the sole command exception.
            if record.base_epoch == window['baseEpoch'] and record.grasp_epoch == window['graspEpoch']:
                return 'nonvertical_command_requires_new_pair'
    if record.base_epoch != window['baseEpoch']: return 'base_moved_new_window_required'
    if record.grasp_epoch != window['graspEpoch']: return 'grasp_changed'
    if any(row['sequence'] > window['commandSequence'] and
           any(k in row['command'] for k in ('reach', 'wrist_yaw', 'wrist_roll', 'wrist_pitch'))
           for row in record.commands):
        return 'nonvertical_command_requires_new_pair'
    trace = [s for s in record.states if s['_readSequence'] >= window.get('traceStart', 0)]
    if not trace or trace[0]['stamp'] > observation['stamp']: return 'motion_record_truncated'
    if stationary_reacquisition.enabled():
        trace = stationary_reacquisition.join_traces(window['holdSamples'], trace)
        evidence = stationary_reacquisition.assess(trace, window['maxGapS'], controller.ex.config.interface_precision_m,
                                       controller.rt.estimator, vertical=True)
        evidence.update(commandSequence=record.sequence, commands=[deepcopy(r) for r in record.commands
                        if r['sequence'] > window['commandSequence']])
        observation['propagationEvidence'] = evidence
        return evidence['reason']
    return motion_reason(trace, window['maxGapS'], controller.ex.config.interface_precision_m,
                         controller.rt.estimator, vertical=True)


def registration_world(observation, state):
    """Export once for legacy world bookkeeping; never feed this into ASSEMBLY_STATIONARY_ENTRANCE_LAYERS control."""
    reg = deepcopy(observation.get('bodyRegistration'))
    if not reg or observation.get('coordinateFrame') != 'stationary_robot_base': return reg
    odom = state['odometry']; r = cr.rotation(odom['yaw']); r3 = np.eye(3); r3[:2,:2] = r
    for key in ('position', 'nominalPosition'):
        if key in reg: reg[key] = (r @ reg[key]+[odom['x'], odom['y']]).tolist()
    for key in ('yaw', 'nominalYaw'):
        if key in reg: reg[key] += odom['yaw']
    for candidate in reg.get('yawCandidates', []):
        candidate['yaw'] += odom['yaw']
    if 'plannedDeltaM' in reg:
        reg['plannedDeltaM'] = (np.asarray(reg['plannedDeltaM']) @ r.T).tolist()
    for hole in reg['observedBodyHoles']:
        hole['point'] = (r3 @ hole['point']+[odom['x'], odom['y'], 0.]).tolist()
        hole['normal'] = (r3 @ hole['normal']).tolist()
    reg['baseOdometry'] = deepcopy(odom)
    reg['exportedFrom'] = 'stationary_robot_base'
    return reg


def export_registration(controller, observation):
    state = (controller.ex._state() if observation.get('coordinateFrame') == 'stationary_robot_base' else None)
    reg = registration_world(observation, state)
    record = controller.rt.estimator.stationary_assemblies.records.get(observation.get('body'), {})
    ref = record.get('reference', {})
    prior = ref.get('componentHoleRegistration')
    from steward.ego.geometry_relocalization import eligible
    if reg and ref.get('geometryRelocalization') and eligible(ref, prior):
        if {p['bodyHole'] for p in reg.get('holePairs', [])} == {p['bodyHole'] for p in prior.get('holePairs', [])}:
            reg.update(prePickRelocalized=True, geometryIdentity=deepcopy(prior['geometryIdentity']))
    return reg
