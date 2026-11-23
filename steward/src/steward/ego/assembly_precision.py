"""Close-view interface/endpoint feedback. Inputs are Ego RGB-D and robot FK only.

Geometric replans use ordinary place/contact admission. Final differential servo
uses a bounded fixed-height reachable path. Missing images retain the admitted
action, and never become a zero measured residual.
"""
from steward.ego.phase_timing import timed
from contextlib import contextmanager
from copy import deepcopy
import math
import time
from steward.ego import action_budget as ab, release_conditions
import numpy as np

from steward.ego.fasten_align import measured_frame, outline
from steward.ego.fasten_geometry import directed_tool, transform_feature
from steward.ego.fasten_plan import value
from steward.ego.fasten_release import release_frame, released_interface
from steward.ego.skills import SkillFailed
from steward.math3d import quat_from_matrix


def tolerance(controller):
    return float(getattr(controller.ex.config, 'interface_precision_m', .0035))


def close(view, stamp, after):
    return bool(view and view.get('source') == 'rgbd_segment'
        and after < view.get('stamp', -1) <= stamp and stamp-view['stamp'] <= 2.
        and view.get('rangeM', math.inf) <= .8 and not view.get('truncated', True)
        and view.get('invalidDepthFraction', 1) <= .05)


def view_reason(view, stamp, after):
    if not view or view.get('source') != 'rgbd_segment':
        return 'no_rgbd_segment'
    if not after < view.get('stamp', -1) <= stamp or stamp-view['stamp'] > 2.:
        return 'no_fresh_frame'
    if view.get('rangeM', math.inf) > .8:
        return 'outside_close_range'
    if view.get('truncated', True):
        return 'truncated_view'
    if view.get('invalidDepthFraction', 1) > .05:
        return 'invalid_depth'
    return None


@timed("precision_observation")
def observe(controller):
    """Request one targeted camera frame at this hover, using the existing worker.

    Fixed cameras are selected by calibrated FOV/range. No survey, base motion,
    load rotation or extension of the enclosing primitive/job deadline.
    """
    ex, est = controller.ex, controller.rt.estimator
    since = ex._state()['stamp']
    ids = getattr(controller, '_precision_entities', ())
    scene = controller.rt.ego_state()
    targets = [scene.find(eid).pose.value['p'] for eid in ids if scene.find(eid)]
    with controller.rt._lock:
        request = controller.rt.request_precision_view(targets, since)
        request['entities'] = list(ids)
    controller._precision_acquisition = request
    start = time.monotonic()
    deadline = ab.make(ex, ab.observation(ex) if ab.enabled() else 8., 'existing eight-sensor-second frame window')
    try:
        while not ab.expired(deadline):
            controller.check()
            scene = controller.rt.ego_state()
            reasons = {eid:view_reason(value(scene.find(eid), 'contactView'), scene.stamp, since) for eid in ids}
            request['missing'] = {eid:reason for eid,reason in reasons.items() if reason}
            if ids and not request['missing']:
                request['status'] = 'measured_views'
                break
            if not request.get('cameras') or request.get('completed'):
                break
            ex._wait(.1)
            # Body-time cap also bounds offline IO and a fast sensor stream.
            if ex._state()['stamp']-since >= (ab.observation(ex) if ab.enabled() else 8.):
                break
        request['status'] = request.get('status', 'unknown')
        request['wallS'] = time.monotonic()-start
        if ab.enabled(): request.update(clock='simulation',simS=round(deadline.sim_s,3))
    finally:
        controller.rt.finish_precision_view(request)
    return since


def acquire(controller, ids):
    controller._precision_entities = tuple(dict.fromkeys(eid for eid in ids if eid))
    return observe(controller)


def summary(report):
    def measured(row):
        return {k:deepcopy(v) for k,v in row.items() if k in (
            'status', 'residualM', 'reason', 'source', 'observedAt', 'uncertaintyM', 'perHoleM', 'acquisition')}
    keys = ('kind', 'object', 'body', 'hole', 'toleranceM', 'beforeResidualM',
            'preReleaseResidualM', 'postReleaseResidualM', 'fallback', 'insertionObserved', 'tipDepthM', 'releaseExecuted')
    row = {k:deepcopy(report.get(k)) for k in keys}
    for key in ('placementId', 'restoredPrior', 'processId', 'memoryRestoredAt', 'releasedAfterMemoryRestore',
                'referenceSource', 'referencePrecision', 'geometryUnconfirmed', 'releasedAt', 'alignmentMode', 'alignmentBasis', 'registrationId', 'kinematicResidualAxes',
                'tipOffsetUncertaintyM', 'alignmentUncertaintyM', 'holeRegistrationAttempts', 'holeRegistrationSuccesses',
                'placementXYCompensation', 'relativeEntranceServo', 'tipOffsetAttempts', 'tipOffsetMeasuredCount', 'tipOffsetMeasured', 'registrationAgeS',
                'registrationUncertaintyM', 'registrationSource', 'registrationPixels',
                'kinematicAlignmentExecutions', 'kinematicBeforeResidualM', 'kinematicAfterResidualM',
                'kinematicAxisErrorDeg', 'kinematicTipHeightM', 'preReleaseSnapshots',
                'registrationEvents', 'registrationFailureReason', 'tipOffsetSource', 'tipInferenceFailure',
                'kinematicMeasuredOffsetExecutions', 'kinematicInferredOffsetExecutions',
                'registeredHoles', 'registeredPointM', 'registeredBasePointM',
                'fallbackAction', 'endpointUnavailable', 'seatDelivery', 'wristAlignment', 'wristPlacement', 'landingCompensation', 'tipEntryServos', 'preplaceRelocalization'):
        if key in report:
            row[key] = deepcopy(report[key])
    if report.get('releaseConditions'):
        conditions = release_conditions.summary(report['releaseConditions'])
        if conditions: row['releaseConditions'] = conditions
    if report.get('seatingGeometryDelivery', {}).get('afterXYM') is not None:
        row['seatingGeometryDelivery'] = {k:deepcopy(report['seatingGeometryDelivery'][k]) for k in
            ('afterXYM', 'tipRelativeToEntryM', 'tiltDeg', 'entryObservedAt',
             'entrySource', 'endpointSource', 'observationsAdded', 'unconfirmed', 'reason')
            if k in report['seatingGeometryDelivery']}
    if report.get('releaseHeight', {}).get('policy') == 'observed_plane_height_only':
        from steward.ego.release_plane import summary as height_summary
        row['releaseHeight'] = height_summary(report['releaseHeight'])
        # Make room for the final height without repeating empty acquisition
        # scaffolding. Full keyed events and all measured frames stay intact.
        row['tipEntryServos'] = [s if s.get('frames') or s.get('corrections') else
            {k:([{q:v for q,v in o.items() if q in ('phase','measured','action','reason')}
                  for o in value] if k == 'keyObservations' else value)
             for k,value in s.items() if k in ('id','reason','releaseExecuted','releasedAt',
                 'nearContactUnconfirmed','frames','corrections','keyObservations')}
            for s in row.get('tipEntryServos', [])]
    if 'wristAlignment' in row or 'wristPlacement' in row:
        # Wrist diagnostics replace repeated registration geometry in the model receipt.
        for key in ('registeredHoles','registeredPointM','registeredBasePointM',
                    'registrationEvents','registrationPixels','preReleaseSnapshots'):
            row.pop(key,None)
    row['corrected'] = any(c.get('executed') for c in report.get('corrections', []))
    row['correctionCount'] = sum(bool(c.get('executed')) for c in report.get('corrections', []))
    row['postCorrectionCount'] = report.get('postCorrectionCount', 0)
    row['preRelease'] = measured(report.get('observations', [{}])[-1]) if report.get('observations') else dict(
        status='unknown', reason=report.get('fallback') or 'hover_not_reached', residualM=None)
    row['postRelease'] = measured(report.get('postRelease', dict(status='unknown',
        residualM=None, reason='release_not_observed')))
    def phrase(key, phase):
        val = row[key]
        detail = row['preRelease' if phase == '释放前' else 'postRelease']
        return phase + (f'实测残差 {val*1000:.1f} mm' if val is not None else
                       '未能测量（'+str(detail.get('reason') or row['fallback'] or 'unknown')+'）')
    row['message'] = phrase('preReleaseResidualM', '释放前')+'；'+phrase('postReleaseResidualM', '释放后')+'。'
    if report.get('alignmentMode') in ('registered_hole_fk_v1', 'support_hole_fk_v2'):
        before, after = report.get('kinematicBeforeResidualM'), report.get('kinematicAfterResidualM')
        row['message'] = ('依据 '+str(report.get('alignmentBasis'))+'；下端偏移'
            + ('实测' if report.get('tipOffsetMeasured') else '按抓取几何推算'
               if report.get('tipOffsetSource') == 'grasp_geometry_inferred' else '按落点反馈推算'
               if report.get('tipOffsetSource') == 'landing_bias_estimated' else '未知')+'；运动学残差 '
            + (f'{before*1000:.1f} → {after*1000:.1f} mm' if before is not None and after is not None
               else '未能测量（'+str(report.get('fallback') or 'endpoint_not_reached')+'）')
            + '；不是实测入孔。' + phrase('postReleaseResidualM', '释放后')+'。')
    from steward.ego import prerelease_tip
    measured_tip = prerelease_tip.measurement_summary(report) if prerelease_tip.enabled() else None
    if measured_tip is not None:
        row['preReleaseTipEntry'] = measured_tip
        row['corrected'] |= measured_tip['correctionCount'] > 0
        row['correctionCount'] += measured_tip['correctionCount']
        if measured_tip['correctionCount'] or (measured_tip['latestObservationConfirmed'] and
                measured_tip['lastResidualM'] <= measured_tip['uncertaintyM']):
            row['plannedAlignmentBasis'] = row.get('alignmentBasis')
            row['alignmentBasis'] = 'same_frame_tip_entry_servo'
        # A missing FK residual is not evidence of a failed endpoint move.
        # Keep nominal grasp-offset diagnostics separate from differential vision.
        row['message'] = ('松手前同帧尖端—入口偏差 '
            + f"{measured_tip['firstResidualM']*1000:.1f} → {measured_tip['lastResidualM']*1000:.1f} mm"
            + ('（入口由同帧锚点和上件孔布局推算）' if measured_tip['entryInferredFromLayout'] else '')
            + f"；实测修正 {measured_tip['correctionCount']} 次；"
            + ('末次观察有测量' if measured_tip['latestObservationConfirmed'] else '末次观察未确认')
            + '；不是实测入孔。' + phrase('postReleaseResidualM', '释放后')+'。')
    if report.get('insertionSearch'):
        from steward.ego.insertion_search import summary as search_summary
        row['insertionSearch'] = search_summary(report['insertionSearch'])
        # The native search event retains full local geometry. Do not repeat
        # the superseded registration transcript in the compact public result.
        for key in ('registeredHoles', 'registeredPointM', 'registeredBasePointM',
                    'registrationEvents', 'registrationPixels'):
            row.pop(key, None)
        if 'preReleaseSnapshots' in row:
            snapshots = row['preReleaseSnapshots']
            snapshots['files'] = {camera: {k: v for k, v in item.items()
                if k in ('filename', 'cached')} for camera, item in snapshots.get('files', {}).items()}
        if row['insertionSearch'].get('executed'):
            row['message'] += (' 视觉入孔。' if row['insertionSearch'].get('detected') else ' 搜索未确认。')
    return row


def publish(job):
    reports = getattr(job, '_precision_reports', [])
    if reports:
        job.postcondition['precisionReports'] = [summary(r) for r in reports]
        job.postcondition['precisionSummary'] = ' '.join(r['message'] for r in job.postcondition['precisionReports'])


def pair_measurement(planner, scene, frame, after):
    unknown = dict(status='unknown', residualM=None, source=None,
                   reason='no_close_independent_interface_views')
    if not frame:
        return dict(unknown, reason='no_interface_frame')
    for eid in (frame['entity'], frame['body']):
        reason = view_reason(value(scene.find(eid), 'contactView'), scene.stamp, after)
        if reason:
            return dict(unknown, reason=reason, entity=eid)
    # The planner's body collision box may include released members. Registration
    # must use the body's own stored outline, never that union envelope.
    scene = deepcopy(scene)
    body = scene.find(frame['body'])
    ref = value(body, 'orientation', {}).get('assembly', {}).get('reference')
    if ref:
        body.rigid.links[0].geometry = {'collision':[dict(type='box', size=ref['size'])]}
    measured = measured_frame(planner, scene, frame, after)
    if measured is None:
        return unknown
    f, bp, p, b = measured
    pp = {h['id']: np.asarray(h['point']) for h in f['holes']}
    pairs = f.get('holePairs', [])
    if not pairs or any(h['partHole'] not in pp or h['bodyHole'] not in bp for h in pairs):
        return dict(unknown, reason='missing_hole_correspondence')
    upper = np.array([pp[h['partHole']][:2] for h in pairs])
    lower = np.array([bp[h['bodyHole']][:2] for h in pairs])
    residuals = np.linalg.norm(upper-lower, axis=1)
    return dict(status='estimated', residualM=float(max(residuals)),
        perHoleM={h['bodyHole']:float(e) for h,e in zip(pairs,residuals)},
        translationM=(lower.mean(axis=0)-upper.mean(axis=0)).tolist(),
        partFrameCorrectionM=(np.asarray(p['position'])-frame['position']).tolist(),
        bodyFrameCorrectionM=(np.asarray(b['position'])-body.pose.value['p']).tolist(),
        source='close_rgbd_outlines_registered_holes', observedAt=min(p['observedAt'],b['observedAt']),
        uncertaintyM=p['positionUncertaintyM']+b['positionUncertaintyM'],
        frame=f, partPose=p, bodyPose=b)


def compact(measure):
    return {k:deepcopy(v) for k,v in measure.items() if k not in ('frame','partPose','bodyPose','feature')}


def install_pose(est, eid, pose):
    """Update the registration used by the existing planner, retaining its holes."""
    position, rotation = np.asarray(pose['position']), np.asarray(pose['rotation'])
    yaw = math.atan2(rotation[1,0],rotation[0,0])
    track = est.tracks.get(eid)
    if track is not None and not track.held:
        track.pos = position.copy(); track.yaw = yaw
    record = est.stationary_assemblies.records.get(eid)
    if record is not None:
        record['reference'].update(position=position.tolist(), yaw=yaw, registeredAt=pose['observedAt'])
        record['entity'].pose.value = dict(p=position.tolist(), q=quat_from_matrix(rotation))
        record['entity'].pose.t = pose['observedAt']
        record['registrationEvidence'].update(source=pose['source'], observedAt=pose['observedAt'], tier=0)


def calibrate(controller, eid, *, pose=None, feature=None):
    """Re-estimate the load relative to the pinch point at a stationary image."""
    with controller.rt._lock:
        est = controller.rt.estimator
        track = est.tracks.get(eid)
        if track is None or not track.held or est.held_id != eid:
            return None
        state = controller.ex._state()
        grip = est.gripper_position(est.held_gripper, state['joints'], state['odometry'])
        rotation = est.gripper_rotation(est.held_gripper, state['joints'], state['odometry'])
        old = np.asarray(track.held_offset).copy()
        if feature:
            centre = (np.asarray(feature['tip'])+feature['head'])/2
            track.held_tool_feature = transform_feature(feature, rotation.T, -rotation.T @ grip)
            from steward.ego.grasp_endpoints import inhand_observation
            inhand_observation(track)
        else:
            centre = np.asarray(pose['position'])
            r = np.asarray(pose['rotation'])
            if track.held_interface:
                track.held_interface.update(offset=(rotation.T @ (centre-grip)).tolist(),
                    rotation=(rotation.T @ r).tolist(), observedAt=pose['observedAt'],
                    source='close_rgbd_grasp_offset_plus_fk')
            track.held_yaw = math.atan2(r[1,0],r[0,0])-math.atan2(rotation[1,0],rotation[0,0])
        track.held_offset = rotation.T @ (centre-grip)
        track.pos = centre.copy()
        est.version += 1
        return dict(offsetM=track.held_offset.tolist(), correctionM=(track.held_offset-old).tolist(),
                    source='close_rgbd_relative_to_robot_fk', observedAt=(feature or pose).get('stamp', (pose or {}).get('observedAt')))


@timed("inhand_inspection")
def acquired(controller, job, eid):
    """Screw measurement after grasp, before transport; no nominal endpoint promotion."""
    context = getattr(controller, '_registered_alignment', None)
    if context is not None:
        from steward.ego.registered_alignment import attach
        context['tipOffsetAttempts'] += 1
        attach(controller, context['report'])
    after = max(acquire(controller, [eid]), controller.rt.estimator.held_since)
    scene = controller.rt.ego_state(); ent = scene.find(eid)
    view = value(ent, 'contactView')
    detector = dict(called=False, reason=view_reason(view, scene.stamp, after))
    feature = (directed_tool(view['points'], ent.category.value, view['stamp'], detector)
               if ent and close(view, scene.stamp, after) else None)
    row = dict(status='unknown', reason='no_close_grasp_endpoints')
    if feature:
        # The endpoint must be a fresh stationary sample of this very grasp.
        state = controller.ex._state()
        joints = view.get('sensorJoints')
        odom = view.get('sensorOdometry')
        synchronized = (bool(view.get('nativeResolution')) and joints is not None and odom is not None
            and all(abs(state['joints'].get(k, v)-v) < .001 for k,v in joints.items())
            and all(abs(state['odometry'][k]-odom[k]) < .001 for k in ('x','y','yaw')))
        if not hasattr(controller, '_registered_alignment') or synchronized:
            compensation = calibrate(controller, eid, feature=feature)
            if compensation:
                row = dict(status='estimated', **compensation)
        else:
            row['reason'] = 'endpoint_frame_not_stationary_or_unsynchronized'
    context = getattr(controller, '_registered_alignment', None)
    if context is not None:
        if row['status'] == 'estimated':
            context['tipOffsetMeasuredCount'] += 1
            context['tip'] = deepcopy(controller.rt.estimator.tracks[eid].held_tool_feature)
            context['tipHeldSince'] = controller.rt.estimator.held_since
            context['tipOffsetSource'] = 'measured'
            if context['mode'] == 'support_hole_fk_v2':
                context['tip']['uncertaintyM'] = float((feature or {}).get('uncertaintyM', .002))
        else:
            context['reason'] = row['reason']
            if context['mode'] == 'support_hole_fk_v2':
                from steward.ego.support_registration import grasp_geometry
                inferred = grasp_geometry(controller, eid)
                if inferred:
                    context.update(tip=inferred, tipHeldSince=controller.rt.estimator.held_since,
                                   tipOffsetSource='grasp_geometry_inferred',
                                   reason=context.get('registrationFailureReason'))
                    row.update(status='inferred', source='grasp_geometry_inferred',
                               uncertaintyM=inferred['uncertaintyM'])
        from steward.ego.registered_alignment import attach
        attach(controller, context['report'])
    row['acquisition'] = deepcopy(getattr(controller, '_precision_acquisition', {}))
    from steward.ego.grasp_diagnostics import inspection
    raw = getattr(controller.rt, '_precision_native_frames', {}).get((view or {}).get('camera'))
    inspection(controller.rt, eid, 'head_endpoints',
               dict(attempted=True, measured=bool(feature), outcome=deepcopy(row), feature=feature,
                    detector=detector, view={k:v for k,v in (view or {}).items() if k != 'points'}), raw=raw)
    job.postcondition.setdefault('graspPrecision', []).append(row)
    if context is not None and context['mode'] == 'support_hole_fk_v2':
        publish(job)


def held_frame(controller, plan):
    with controller.rt._lock:
        est = controller.rt.estimator; t = est.tracks.get(plan['object'])
        if t is None or not t.held_interface:
            return None
        state = controller.ex._state()
        # Do not apply the settled Z hypothesis while still hovering.
        target = deepcopy(plan.get('placementTarget', {}))
        target.get('releasedFrame', {}).pop('settledCentreZM', None)
        return release_frame(t.held_interface,
            est.gripper_position(est.held_gripper, state['joints'], state['odometry']),
            est.gripper_rotation(est.held_gripper, state['joints'], state['odometry']), target, state['stamp'])


def screw_measurement(controller, plan, params, after):
    scene = controller.rt.ego_state()
    ent = scene.find(plan['object']); view = value(ent, 'contactView')
    unknown = dict(status='unknown', residualM=None, reason='no_close_tip_or_entry', source=None)
    reason = view_reason(view, scene.stamp, after)
    if reason:
        return dict(unknown, reason=reason, entity=plan['object'])
    feature = directed_tool(view['points'], ent.category.value, view['stamp'])
    frame = released_interface(scene, params['on'], params['seatBody'])
    if not feature or not frame:
        return dict(unknown, reason='no_directed_tip' if not feature else 'no_released_interface_frame')
    # Entry XY follows the separately observed upper component even when it
    # occludes the lower body. Pair error remains a separate measured quantity.
    pv = value(scene.find(frame['entity']), 'contactView')
    pose = (outline(pv, frame['position'], np.asarray(frame['rotation']), frame['size'], after, scene.stamp)
            if close(pv, scene.stamp, after) else None)
    if pose is None:
        return dict(unknown, reason=view_reason(pv, scene.stamp, after) or 'incomplete_interface_outline', entity=frame['entity'])
    f = deepcopy(frame); f.update(position=pose['position'], rotation=pose['rotation'],
        observedAt=pose['observedAt'], outlineObserved=True, observed=False, source=pose['source'])
    for h in f['holes']:
        h['point'] = (np.asarray(pose['position'])+np.asarray(pose['rotation']) @ h['p']).tolist()
    pair = next((p for p in f['holePairs'] if p['bodyHole']==params['seatHole']), None)
    hole = next((h for h in f['holes'] if pair and h['id']==pair['partHole']), None)
    if hole is None:
        return unknown
    from steward.ego.fasten_geometry import surface_patch
    patch = surface_patch(pv['points'], hole['point'][:2])
    if patch is None:
        return dict(unknown, reason='no_local_entry_surface')
    point = np.asarray(patch['point'])
    delta = point-np.asarray(feature['tip'])
    return dict(status='estimated', residualM=float(np.linalg.norm(delta[:2])),
        translationM=delta[:2].tolist(), entryM=point.tolist(), feature=feature, frame=f,
        source='close_rgbd_tip_and_registered_upper_hole', observedAt=min(view['stamp'],pv['stamp']),
        uncertaintyM=None,
        axisErrorDeg=math.degrees(math.acos(np.clip(feature['axis'][2],-1,1))))


@contextmanager
def placement(controller, job, plan, params):
    """Install a scoped callback at hover; replacement uses a fresh full receipt."""
    ex = controller.ex
    previous = getattr(ex, '_precision_hover', None)
    previous_insert = getattr(ex, '_precision_insert', None)
    previous_release = getattr(ex, '_precision_before_release', None)
    previous_after = getattr(ex, '_precision_after_release', None)
    previous_servo = getattr(ex, '_precision_tip_entry', None)
    previous_component_finish = getattr(ex, '_precision_component_finish', None)
    previous_component_native = getattr(ex, '_precision_component_native', None)
    screw = bool(plan.get('seatTarget') or (not plan.get('placementTarget', {}).get('holePairs')
                 and params.get('seatBody') and params.get('seatHole')))
    eligible = screw or bool(plan.get('placementTarget', {}).get('holePairs')) or params.get('alignHoles') is True
    report = plan.setdefault('precision', dict(kind='screw' if screw else 'interface',
        toleranceM=tolerance(controller), beforeResidualM=None, preReleaseResidualM=None,
        postReleaseResidualM=None, observations=[], corrections=[], fallback='hover_not_reached',
        object=plan.get('object'), body=params.get('seatBody', params.get('on')), hole=params.get('seatHole'))) if eligible else None
    if screw and hasattr(controller, '_registered_alignment'):
        pending = controller._registered_alignment.get('report')
        if pending is not None and pending.get('kind') == 'screw_registration':
            pending.update(report)
            report = plan['precision'] = pending
    if eligible:
        if not hasattr(job, '_precision_reports'):
            job._precision_reports = []
        if not any(r is report for r in job._precision_reports):
            job._precision_reports.append(report)
    registered = bool(screw and hasattr(controller, '_registered_alignment'))
    if registered:
        from steward.ego.registered_alignment import attach
        attach(controller, report)
    count = 0

    def hover(current):
        nonlocal count
        from steward.ego import relative_entrance_servo
        if not screw and relative_entrance_servo.applies(current):
            relative_entrance_servo.hover(controller, job, current, report)
            if not current.get('_relative_entrance_servo', {}).get('legacy'):
                return False
        from steward.ego.geometry_relocalization import eligible, current_reference
        target = current.get('placementTarget', {})
        ref = current_reference(target, controller.rt.estimator)
        if not screw and ref.get('geometryRelocalization') and not eligible(ref, target.get('componentHoleRegistration')):
            report.update(fallback='geometry_relocalization_required', geometryUnconfirmed=True, referencePrecision='coarse')
            return False
        from steward.ego.preplace_relocalization import hover as relocalize
        if relocalize(controller, job, current, params, report):
            return True
        from steward.ego import placement_xy_compensation
        if not screw and placement_xy_compensation.applies(current) and current.get('placementTarget', {}).get(
                'componentHoleRegistration', {}).get('observedBodyHoles'):
            report['fallback'] = 'placement_xy_compensation_existing_pair_or_best_estimate'
            return False
        if report.get('preplaceRelocalization', {}).get('fit'):
            # A measured frame supersedes nominal targets. Recursive replacement
            # must not let an older wrist/outline correction undo localization.
            return False
        if registered:
            from steward.ego.fasten_delivery import measure_hover
            from steward.ego.wrist_fasten import hover as wrist_hover
            report['_wristMeasured'] = False
            if not wrist_hover(controller, current, params, report) and not report.get('wristAlignment', {}).get('observedAt'):
                measure_hover(controller, current, report)
            if not report['observations']:
                report['observations'].append(dict(status='unknown', residualM=None,
                    reason='release_visual_not_requested', source=None))
            from steward.ego.wrist_landing import apply_hover
            apply_hover(controller, current, report)
            from steward.ego.registered_alignment import hover as fk_hover
            return fk_hover(controller, job, current, params, report)
        if not screw:
            from steward.ego.wrist_assembly import hover as wrist_assembly_hover
            if wrist_assembly_hover(controller, job, current, params, report):
                return True
        held = controller.rt.estimator.tracks.get(current.get('object'))
        if not screw and held and (held.held_interface or {}).get('inhandMeasurement'):
            report['fallback'] = 'native_inhand_transform_retained'
            return False
        if current.get('placementTarget', {}).get('componentHoleRegistration'):
            # Bare support pixels fixed this component's full target before pick.
            # The old merged-outline prior must not translate it back at hover.
            report['fallback'] = 'component_registered_target_no_hover_observation'
            return False
        after = acquire(controller, [current.get('object'), params.get('on')] +
                        ([] if screw else [report['body']]))
        measurement = (screw_measurement(controller, current, params, after) if screw else
            pair_measurement(controller.rt.esim, controller.rt.ego_state(), held_frame(controller,current), after))
        measurement['acquisition'] = deepcopy(getattr(controller, '_precision_acquisition', {}))
        report['observations'].append(compact(measurement))
        report['fallback'] = None
        report['preReleaseResidualM'] = measurement['residualM']
        if len(report['observations']) == 1:
            report['beforeResidualM'] = measurement['residualM']
        if measurement['status'] != 'estimated':
            report['fallback'] = measurement['reason']
            return False
        if count >= 2:
            if measurement['residualM'] > tolerance(controller):
                report['fallback'] = 'residual_after_corrections'
                raise SkillFailed('close-view residual remains above precision tolerance; load remains held')
            return False
        # Even a centred screw needs a checked upright tip-in-mouth release.
        needs = (measurement['residualM'] > tolerance(controller) or
                 (screw and not current.get('seatEndpoint')) or
                 (not screw and not current.get('precisionLowRelease')))
        if not needs:
            return False
        if screw:
            compensation = calibrate(controller, current['object'], feature=measurement['feature'])
            with controller.rt._lock:
                controller.rt.estimator.released_interfaces[params['on']] = deepcopy(measurement['frame'])
                controller.rt.estimator.version += 1
            revised = dict(params, controlledSeat=True, precisionInsert=True)
        else:
            compensation = calibrate(controller, current['object'], pose=measurement['partPose'])
            with controller.rt._lock:
                install_pose(controller.rt.estimator, measurement['frame']['body'], measurement['bodyPose'])
                controller.rt.estimator.version += 1
            revised = dict(params, closeRelease=True)
        receipt = controller.rt.predict('place', revised)
        row = dict(residualBeforeM=measurement['residualM'], graspCompensation=compensation,
                   receiptId=receipt.get('id'), executed=False)
        report['corrections'].append(row)
        if not receipt.get('withinRiskLimit') or (screw and not receipt.get('plan', {}).get('seatEndpoint')):
            report['fallback'] = 'precision_path_unavailable'
            row['reason'] = receipt.get('issues', [])
            if measurement['residualM'] > tolerance(controller):
                raise SkillFailed('measured precision correction unavailable; load remains held')
            return False
        controller.check()
        replacement = controller.rt._consume_receipt(receipt)
        replacement.update(precision=report, **{k:current[k] for k in ('seatBody','seatHole','seatPart') if k in current})
        if not screw:
            replacement.update(precisionLowRelease=True, refineReleaseXY=True, placementToleranceM=tolerance(controller))
        deadline = getattr(ex, '_primitive_deadline', None)
        ex._begin(job, 'place', receipt)
        if deadline is not None:
            ex._primitive_deadline = ab.limited(deadline, ex._primitive_deadline)
        count += 1
        row['executed'] = True
        try:
            ex._place(job, replacement)
        finally:
            current.update(replacement)  # Includes release even if withdrawal failed.
        return True

    def inserted(current):
        from steward.ego.insertion_search import enabled as search_enabled
        if search_enabled() or release_conditions.enabled():
            report['contactEvidence'] = 'bounded_search_before_release' if search_enabled() else 'retained_endpoint_release_unconfirmed'
            return
        if current.get('_fallbackReleaseEndpoint'):
            report['contactEvidence']='near_contact_unconfirmed'
            return
        if current.get('_observedReleaseEndpoint'):
            report['contactEvidence']='observed_gap_endpoint_executed_insertion_unverified'
            return
        if registered:
            return  # FK is checked immediately before opening; never wait for occluded pixels.
        after = acquire(controller, [current['object'], params['on']])
        measurement = screw_measurement(controller, current, params, after)
        measurement['acquisition'] = deepcopy(getattr(controller, '_precision_acquisition', {}))
        report['observations'].append(compact(measurement))
        report['preReleaseResidualM'] = measurement['residualM']
        report['insertionObserved'] = None
        if measurement['status'] != 'estimated':
            report['fallback'] = measurement['reason']
            return
        depth = measurement['entryM'][2]-measurement['feature']['tip'][2]
        report['tipDepthM'] = float(depth)
        report['insertionObserved'] = bool(depth >= 0 and measurement['residualM'] <= tolerance(controller))
        if not report['insertionObserved']:
            report['fallback'] = 'measured_tip_not_in_mouth'
            raise SkillFailed('measured tip has not entered the mouth; load remains held')

    def before_release(current):
        if screw:
            from steward.ego import wrist_release_geometry as wrg
            from steward.ego.wrist_release_capture import enabled as capture_enabled
            if capture_enabled() and wrist_observation.enabled(controller.rt, 'RELEASE'):
                current['_wristReleasePrior']=wrg.prior(controller.rt.estimator.tracks.get(current.get('object')))
            from steward.ego.fasten_delivery import low_release
            low_release(controller, current, report)
            from steward.ego.insertion_search import run as search_insert
            search_insert(controller, current, report)
            from steward.ego.release_plane import finish as finish_height, enabled as height_enabled
            finish_height(controller, current, report)
            from steward.ego.release_conditions import prepare as prepare_release
            prepare_release(controller, current, report)
            from steward.ego import seating_geometry
            seating_geometry.finish(controller, current, report)
            from steward.ego.release_height import record
            record(controller, current, report)
            from steward.ego.released_object import prepare as prepare_object
            prepare_object(controller, current)
            job.postcondition.setdefault('seatDelivery', {}).update(report.get('seatDelivery', {}))
        from steward.ego.insertion_search import enabled as search_enabled
        if registered and not (screw and (search_enabled() or release_conditions.enabled() or height_enabled() or seating_geometry.enabled())):
            from steward.ego.registered_alignment import before_release as fk_release
            fk_release(controller, current, report)
        elif screw and search_enabled():
            report['releaseGeometryBasis'] = ('visual_insertion_search' if
                current.get('insertionSearch', {}).get('detected') else 'search_best_estimate_unconfirmed')
        try:
            report['preReleaseSnapshots'] = (controller.rt.prerelease_snapshots(cached=True)
                if registered and report.get('alignmentMode') == 'support_hole_fk_v2'
                else controller.rt.prerelease_snapshots())
        except Exception as exc:
            report['preReleaseSnapshots'] = dict(files={}, errors=dict(capture=type(exc).__name__))

    def after_release(current):
        from steward.ego.geometry_relocalization import provenance
        metadata = provenance(current.get('placementTarget'), current.get('releasedAt'), controller.rt.estimator)
        # Component reports are placement receipts. Screw precision summaries
        # are diagnostics; their actual place receipt is estimator.placement_evidence.
        if not screw:
            report.update(metadata, releasedAt=current.get('releasedAt'))
        if job.postcondition is None: job.postcondition = {}
        job.postcondition.update(metadata, releasedAt=current.get('releasedAt'))
        from steward.ego import relative_entrance_servo
        relative_entrance_servo.released(controller, job, current)
        from steward.ego.insertion_search import record_release
        record_release(current, report, job)
        if screw:
            from steward.ego.release_conditions import after_open, outcome
            after_open(controller, current, report)
            from steward.ego.wrist_fasten import capture_release
            try:
                capture_release(controller, current)
            finally:
                outcome(controller, current, report)
                if report.get('releaseConditions'):
                    conditions = release_conditions.summary(report['releaseConditions'])
                    if conditions: job.postcondition['releaseConditions'] = conditions
                    else: job.postcondition.pop('releaseConditions', None)
                if current.get('_holeSeatReport'):
                    job.postcondition['holeSeatObservation'] = deepcopy(current['_holeSeatReport'])
                frame = current.get('_wristReleaseFrame')
                if frame:
                    # Preserve the image receipt even if ordinary withdrawal is cancelled.
                    job.postcondition['wristRelease'] = dict(camera=wrist_observation.CAMERA,
                        status='unknown', reason='release_frame_recorded_unclassified',
                        screw=current.get('object'), hole=current.get('seatHole', params.get('seatHole')),
                        seatedAt=current.get('releasedAt'), releaseFrame=deepcopy(frame))

    from steward.ego import wrist_release_capture, wrist_observation
    after_release.wrist_capture = bool(screw and wrist_release_capture.enabled()
        and wrist_observation.enabled(controller.rt, 'RELEASE'))

    previous_clearance = getattr(ex, '_precision_release_clearance', None)
    previous_descent = getattr(ex, '_precision_descent', None)
    previous_floor = getattr(ex, '_precision_release_floor', None)
    active_entry = []
    if eligible:
        from steward.ego import relative_entrance_servo
        ex._precision_component_finish = lambda current: relative_entrance_servo.finish(controller, current)
        ex._precision_component_native = bool(relative_entrance_servo.configurations(controller)) if relative_entrance_servo.applies(plan) else False
        ex._precision_before_release = before_release
        ex._precision_after_release = after_release
        ex._precision_hover = hover
        ex._precision_insert = inserted if screw else None
        if screw:
            from steward.ego.release_height import guard
            ex._precision_release_floor = lambda current, path: guard(controller, current, path)
            from steward.ego.tip_entry_servo import run as tip_entry_run
            def entry_hover(current):
                active_entry[:] = [current]
                tip_entry_run(controller,current,report)
            ex._precision_tip_entry = entry_hover
            from steward.ego.tip_entry_servo import placement_descent
            def descent(path, **kw):
                from steward.ego.insertion_search import enabled as search_enabled
                if not active_entry:
                    return
                if not search_enabled():
                    return placement_descent(controller, active_entry[0], report, path, **kw)
                try:
                    placement_descent(controller, active_entry[0], report, path, **kw)
                except SkillFailed as exc:
                    controller.check()
                    # Unresolved near-view geometry is a search prior, not a
                    # reason to retain the load. Actual cancellation propagates.
                    active_entry[0]['nearContactUnconfirmed'] = True
                    report['nearContactSearchFallback'] = str(exc)
            ex._precision_descent = descent
        else:
            ex._precision_tip_entry = None
            ex._precision_descent = None
        if registered:
            from steward.ego.registered_alignment import release_clearance
            ex._precision_release_clearance = lambda: release_clearance(controller, report)
    try:
        yield report
    finally:
        ex._precision_component_finish = previous_component_finish
        ex._precision_component_native = previous_component_native
        ex._precision_tip_entry = previous_servo
        ex._precision_descent = previous_descent
        ex._precision_release_floor = previous_floor
        from steward.ego.tip_entry_servo import finish as tip_entry_finish
        tip_entry_finish(plan)
        ex._precision_release_clearance = previous_clearance
        ex._precision_hover = previous
        ex._precision_insert = previous_insert
        ex._precision_before_release = previous_release
        ex._precision_after_release = previous_after
        if report is not None:
            from steward.ego.insertion_search import record_release
            record_release(plan, report, job)
            release_conditions.record_release(plan, report)
            report['releaseExecuted'] = bool(plan.get('releaseExecuted'))
            publish(job)


@timed("post_release_observation")
def post_screw(controller, plan, params):
    report = plan.get('precision')
    if not report or report['kind'] != 'screw' or not plan.get('releaseExecuted'):
        return
    from steward.ego import released_object
    check = (plan.get('_holeSeatReport') or {}).get('check', {})
    if released_object.enabled() and check.get('source') == released_object.SOURCE:
        report['postRelease'] = {k:deepcopy(check[k]) for k in
            ('status','passed','reason','source','observedAt','landingState') if k in check}
        report['postReleaseResidualM'] = check.get('axisDistanceM')
        return  # The bounded cleared views already ran; do not reacquire a third view.
    from steward.ego.release_observation import ENABLED as release_view
    if release_view:
        report['postRelease'] = dict(status='unknown', reason='pending_per_hole_release_observation', residualM=None)
        return
    after = acquire(controller, [plan['object'], params['on']])
    measurement = screw_measurement(controller, plan, params, max(after, plan['releasedAt']))
    measurement['acquisition'] = deepcopy(getattr(controller, '_precision_acquisition', {}))
    report['postRelease'] = compact(measurement)
    report['postReleaseResidualM'] = measurement['residualM']


def post_component(controller, job, plan, params):
    from steward.ego import placement_xy_compensation
    from steward.ego.placement_ready import ENABLED as legacy_readiness
    from steward.ego.assembly_session import current as active_session
    if placement_xy_compensation.applies(plan) and plan.get('releaseExecuted') and (not legacy_readiness or active_session() is not None):
        # This replaces the old post-component acquisition; never add a tour.
        acquire(controller, [plan['object'], params.get('on')])
        placement_xy_compensation.post(controller, job, plan, params)
        return
    from steward.ego.wrist_assembly import post_release as wrist_post
    wrist_post(controller, job, plan, params)
    from steward.ego.assembly_landing import ENABLED as all_paths, post_release as landing_check
    if all_paths:
        landing_check(controller, job, plan, params)
        if plan.get('_landingChecked'):
            report = plan.get('precision')
            if report is not None:
                report['postRelease'] = deepcopy(plan['placementReadiness'])
                report['postReleaseResidualM'] = plan['placementReadiness'].get('residualM')
                report['postCorrectionCount'] = plan['placementReadiness'].get('replays', 0)
            return
    report = plan.get('precision')
    if report is None or report['kind'] != 'interface' or not plan.get('releaseExecuted'):
        return
    after = acquire(controller, [plan['object'], controller.body_id])
    from steward.ego.placement_ready import ENABLED, post_release
    if ENABLED:
        post_release(controller, job, plan, params)
        report['postRelease'] = deepcopy(plan['placementReadiness'])
        report['postRelease']['acquisition'] = deepcopy(getattr(controller, '_precision_acquisition', {}))
        report['postReleaseResidualM'] = plan['placementReadiness']['residualM']
        report['postCorrectionCount'] = plan['placementReadiness']['replays']
        return
    scene = controller.rt.ego_state()
    if plan.get('placementTarget', {}).get('componentHoleRegistration'):
        report['postRelease'] = dict(status='unknown', residualM=None,
            reason='support_holes_covered_after_component_release')
        report['postCorrectionCount'] = 0
        return
    frame = released_interface(scene, plan['object'], controller.body_id)
    measurement = pair_measurement(controller.rt.esim, scene, frame, max(after,plan['releasedAt']))
    measurement['acquisition'] = deepcopy(getattr(controller, '_precision_acquisition', {}))
    report['postRelease'] = compact(measurement)
    report['postReleaseResidualM'] = measurement['residualM']
    report['postCorrectionCount'] = 0
    if measurement['status'] != 'estimated':
        return
    with controller.rt._lock:
        est = controller.rt.estimator
        est.released_interfaces[frame['entity']] = deepcopy(measurement['frame'])
        install_pose(est, frame['body'], measurement['bodyPose'])
        est.version += 1
    if measurement['residualM'] <= tolerance(controller) or getattr(controller, '_precision_regrasp', False):
        return
    if any(a.get('releaseExecuted') and a['part']==frame['entity'] for a in est.fastening_attempts.values()):
        report['postCorrectionReason'] = 'released_fastener_ledger_preserved'
        return
    if np.linalg.norm(measurement['translationM']) > .03:
        report['postCorrectionReason'] = 'outside_local_correction_range'
        return
    # One local regrasp, before screws exist. Never park/remove the component,
    # recurse fasten, clear an interface ledger, or fetch a replacement part.
    report['postCorrectionCount'] = 1
    controller._precision_regrasp = True
    try:
        with controller.rt._lock:
            install_pose(est, frame['entity'], measurement['partPose'])
        controller.step(job, 'pick', dict(target=frame['entity'], stayNear=True))
        corrected = controller.step(job, 'place', dict(params, stayNear=True, closeRelease=True))
        report['postCorrection'] = deepcopy(corrected.get('precision'))
        report['postReleaseResidualM'] = (corrected.get('precision') or {}).get('postReleaseResidualM')
        for key in ('releasedAt','placementTarget'):
            if key in corrected:
                plan[key] = corrected[key]
    except SkillFailed as exc:
        controller.check()
        report['postCorrectionFailure'] = str(exc)
        if est.held_id:
            raise
    finally:
        controller._precision_regrasp = False


def head_check(attempt, check, limit, scene=None):
    """Tight posterior only for this precision attempt; leave accounting untouched."""
    report = attempt.get('precision')
    if report is None or not report.get('observations'):
        return check
    error = check.get('headDistanceXYM')
    entry_measured = False
    if scene is not None and check.get('headM') is not None:
        frame = released_interface(scene, attempt['part'], attempt['body'])
        view = value(scene.find(attempt['part']), 'contactView')
        pose = (outline(view, frame['position'], np.asarray(frame['rotation']), frame['size'],
                        attempt['seatedAt'], scene.stamp)
                if frame and close(view, scene.stamp, attempt['seatedAt']) else None)
        hole = next((h for h in (frame or {}).get('holes', []) if h['id']==attempt.get('partHole')), None)
        if pose and hole:
            entry_measured = True
            entry = np.asarray(pose['position'])+np.asarray(pose['rotation']) @ hole['p']
            error = float(np.linalg.norm(np.asarray(check['headM'])[:2]-entry[:2]))
            report.update(postEntryM=entry.tolist(), postEntrySource=pose['source'],
                          postEntryObservedAt=pose['observedAt'])
    if error is None and check.get('headM') is not None:
        error = float(np.linalg.norm(np.asarray(check['headM'])[:2]-np.asarray(attempt['entry'])[:2]))
    if scene is None:  # pure head-check callers supply an already measured distance
        report['postReleaseResidualM'] = error
    elif entry_measured and close(value(scene.find(attempt.get('screw')), 'contactView'), scene.stamp, attempt['seatedAt']):
        report['postReleaseResidualM'] = error
        report['postRelease'] = dict(status='estimated', residualM=error, source='close_rgbd_head_and_registered_upper_hole', observedAt=check.get('observedAt'))
    report['headObservedAt'] = check.get('observedAt')
    if error is not None and error > limit:
        check = dict(check, passed=False, status='failed', precisionReason='head_not_concentric', toleranceM=limit)
    return check
