"""Current-frame differential alignment and key contact observations.

The saved entry is only a search/correspondence prior. Every error uses ONE
native frame, with explicit relative-layout inference when the target is hidden. No held offset is calibrated.
"""
from copy import deepcopy
import math
import uuid
import numpy as np
from steward.ego import action_budget as ab, capabilities, wrist_observation as wo
from steward.ego.support_holes import components, project


def camera_for(rt):
    """Discover calibrated purpose, not a sensor name or robot model."""
    return next((c for c, (_, s) in rt.estimator.cameras.items()
                 if s.get('purpose') == 'tip_entry_servo'
                 and c in getattr(rt, '_provided_camera_ids', ())), None)


def lower_end(controller, sample, entry, opening):
    """Unique elongated foreground with a visible lower end and curved shaft.

    Gravity resolves the LOWER end, not semantic head/tip polarity. The upper
    end can be obscured by the grip. No catalogue length, colour or entity mask.
    """
    raw, cam = sample
    est = controller.rt.estimator; h = raw['header']
    grip = est.gripper_position(est.held_gripper, h['joints'], h['odometry'])
    region = wo.window(raw, cam, grip, opening)
    depth = wo.local_median(raw['depth'], region)
    raw = dict(raw, depth=np.where(raw['depth'] > 0, depth, 0))
    p, y, x = wo.cloud(raw, cam, region)
    if len(p) < 30: return None
    intr = dict(width=depth.shape[1], height=depth.shape[0], fovy=raw['fovy'])
    focal = intr['height']/(2*math.tan(math.radians(intr['fovy'])/2))
    pixel = float(np.median(raw['depth'][y,x]))/focal
    noise = max(2*pixel, float(entry.get('uncertaintyM', 0.)))
    point = np.asarray(entry['point']); normal = np.asarray(entry.get('normal', [0,0,1]))
    if normal[2] < math.cos(math.pi/6): return None
    # An observed gap separates the lower end from the entry plane. Cutting a
    # touching shaft at this plane is never accepted as a visible endpoint.
    keep = (np.linalg.norm(p-grip,axis=1) < opening) & ((p-point) @ normal > 2*noise)
    bodies = []
    with controller.rt._lock:
        old = est.joints, est.odometry
        try:
            est.joints, est.odometry = h['joints'], h['odometry']
            bodies = list(est.body_boxes())
            for c, half, r in bodies:
                keep &= ~np.all(abs((p-c) @ r) <= np.asarray(half)+noise, axis=1)
        finally:
            est.joints, est.odometry = old
    selected = np.zeros(depth.shape, bool); selected[y[keep],x[keep]] = True
    # Split touching projected surfaces by observed depth, retaining no labels.
    edges = np.zeros_like(selected)
    for axis in (0,1):
        a,b = (slice(1,None),slice(None,-1))
        aa,bb = ((a,slice(None)),(b,slice(None))) if axis==0 else ((slice(None),a),(slice(None),b))
        jump = (abs(depth[aa]-depth[bb]) > 3*noise) & selected[aa] & selected[bb]
        edges[aa] |= jump; edges[bb] |= jump
    selected &= ~edges
    selected &= wo.local_median(selected, region).astype(bool)
    found = []
    for pixels in components(selected):
        if len(pixels) < 30: continue
        mask = np.zeros_like(selected); mask[pixels[:,1],pixels[:,0]] = True
        q = p[mask[y, x]]
        centre = np.mean(q,axis=0)
        _, values, axes = np.linalg.svd(q-centre, full_matrices=False)
        axis = axes[0]*np.sign(axes[0,2])
        if values[0] < 2*values[1] or axis[2] < math.cos(math.pi/6): continue
        lo, hi = np.quantile((q-centre) @ axis, [.01,.99])
        f = wo.centre_axis(q, dict(tip=(centre+lo*axis).tolist(), head=(centre+hi*axis).tolist()))
        if f is None: continue
        tip = np.asarray(f['tip'])
        if (tip-point) @ normal <= 3*noise: continue
        # Do not interpret a robot-mask cut, image border, or ROI cut as an end.
        if any(np.all(abs((tip-c) @ r) <= np.asarray(half)+3*noise) for c,half,r in bodies): continue
        uv,_ = project([tip],cam,intr)
        if np.any(uv[0]<3) or np.any(uv[0]>=np.array([intr['width'],intr['height']])-3): continue
        if np.linalg.norm(tip-grip) >= opening-3*noise: continue
        f.update(uncertaintyM=max(float(f['uncertaintyM']),noise), source='same_frame_lower_shaft_end')
        found.append(f)
    return found[0] if len(found)==1 else None


def observe(controller, sample, target, opening, association_radius=None, registration=None):
    """Measure both ends in one frame; layout inference is explicit in receipts."""
    from steward.ego import entry_layout as el
    if (registration or {}).get('targetKind') == 'head':
        from steward.ego.driver_view import observe as driver_observe
        return driver_observe(controller, sample, target, opening, registration)
    diag = dict(candidates=[])
    region = None
    if registration and registration.get('layout'):
        region = wo.window(sample[0],sample[1],target,opening/2)
        for h in registration['layout']:
            region |= wo.window(sample[0],sample[1],h['point'],opening/2)
    found = wo.holes(sample, target, radius=opening/2, diagnostics=diag, region=region)
    common = float(sample[0]['header'].get('odometry',{}).get('sigma',{}).get('pos',0.))
    found = [dict(h, uncertaintyM=max(float(h.get('planeNoiseM',0.)),
                  float(h.get('uncertaintyM',0.))-common)) for h in found]
    arc_diag = {}; arcs, plane = el.surface_arcs(sample, target, opening, arc_diag)
    for h in (registration or {}).get('layout',[]):
        if np.linalg.norm(np.asarray(h['point'])-target) > opening/4:
            more, _ = el.surface_arcs(sample,h['point'],opening,{})
            arcs.extend(more)
    for h in arcs:
        if not any(np.linalg.norm(np.asarray(h['point'])-q['point']) <
                   max(h['radiusM'], q.get('radiusM',0.)) for q in found): found.append(h)
    detail = dict(holeCandidates=len(found), nativeHoleDiagnostics=diag, arcDiagnostics=arc_diag)
    radius = min(opening/4,association_radius) if association_radius is not None else opening/4
    local = [h for h in found if np.linalg.norm(np.asarray(h['point'])[:2]-np.asarray(target)[:2]) < radius]
    detail['localCandidates'] = len(local)
    reference = plane or (local[0] if local else None)
    feature = None; lower_diag = {}; foreground_data = None
    if reference is not None:
        foreground_data = el.foreground(controller, sample, reference, opening)
        feature = el.short_lower(controller, sample, reference, opening, lower_diag,
                                 foreground_data=foreground_data)
        if feature is None: feature = lower_end(controller, sample, reference, opening)
        from steward.ego import prerelease_tip
        if feature is None and prerelease_tip.enabled():
            feature = prerelease_tip.lower(controller, sample, reference, opening, lower_diag,
                                           foreground_data=foreground_data)
    caps = el.heads(controller, sample, reference, opening, feature,
                    foreground_data=foreground_data) if reference is not None else []
    detail.update(lowerDiagnostics=lower_diag, occupiedHeadCandidates=len(caps), heads=caps)
    correspondence = {}
    if registration and registration.get('layout'):
        entry = el.associate(found, caps, registration, opening, correspondence, plane)
    else:
        if registration and registration.get('targetKind') == 'head':
            local = [h for h in caps if np.linalg.norm(np.asarray(h['point'])[:2]-np.asarray(target)[:2]) < radius]
        entry = dict(local[0], targetFromLayout=False) if len(local) == 1 else None
    detail['correspondence'] = correspondence
    from steward.ego import current_entry
    if (registration and registration.get('layout') and current_entry.enabled()
            and (entry is None or len(found) == 1)):
        fresh_diagnostics = {}
        fresh_entry = current_entry.recover(sample, target, opening, registration, fresh_diagnostics)
        if fresh_entry is not None:
            detail['currentFaceCorrespondence'] = fresh_diagnostics
            # Keep a direct native centre when the current outline agrees
            # within measurement resolution; replace stale/inferred identity.
            agrees = (entry is not None and not entry.get('targetFromLayout')
                and np.linalg.norm(np.asarray(entry['point'])-fresh_entry['point']) <=
                entry.get('uncertaintyM',0.)+fresh_entry.get('uncertaintyM',0.))
            if not agrees: entry = fresh_entry
    from steward.ego import visible_entry
    if registration and registration.get('layout') and visible_entry.enabled():
        visible_diagnostics = {}
        visible = visible_entry.recover(sample, target, opening, registration, visible_diagnostics)
        if visible is not None:
            detail['visibleEntrancePair'] = visible_diagnostics
            agrees = (entry is not None and not entry.get('targetFromLayout')
                and np.linalg.norm(np.asarray(entry['point'])-visible['point']) <=
                entry.get('uncertaintyM',0.)+visible.get('uncertaintyM',0.))
            if not agrees:
                detail['supersededEntry'] = entry
                entry = visible
    if entry is None:
        return dict(bothVisible=False, reason='entry_unseen_or_ambiguous', diagnostics=detail)
    if feature is None:
        entry_foreground = el.foreground(controller, sample, entry, opening)
        feature = el.short_lower(controller, sample, entry, opening, lower_diag,
                                 foreground_data=entry_foreground)
        if feature is None: feature = lower_end(controller, sample, entry, opening)
        from steward.ego import prerelease_tip
        if feature is None and prerelease_tip.enabled():
            feature = prerelease_tip.lower(controller, sample, entry, opening, lower_diag,
                                           foreground_data=entry_foreground)
    inferred = bool(entry.get('targetFromLayout'))
    if feature is None:
        return dict(bothVisible=False, reason='lower_end_unseen_or_ambiguous', entry=entry,
                    targetFromLayout=inferred, diagnostics=detail)
    delta = np.asarray(entry['point'])-feature['tip']
    # Occlusion is positive alignment evidence only when the measured lower
    # section overlaps in XY. A broad head alone must not report alignment.
    overlap = float(np.linalg.norm(delta[:2])) <= (feature.get('shaftRadiusM',0.)+
               entry.get('radiusM',0.)+feature['uncertaintyM']+entry.get('uncertaintyM',0.))
    occluded = el.occluded_by_load(sample, entry, feature) if inferred else False
    return dict(bothVisible=True, reason=None, entry=entry, lowerEnd=feature,
        deltaM=delta.tolist(), residualM=float(np.linalg.norm(delta[:2])),
        uncertaintyM=float(entry.get('uncertaintyM',0.))+feature['uncertaintyM'],
        targetFromLayout=inferred, targetOccludedByLowerEnd=occluded and overlap, targetOccludedByHeldObject=occluded,
        lowerEndFootprintOverlapsEntry=overlap,
        alignmentEvidence='inferred_entry_overlaps_observed_lower_end' if occluded and overlap else None,
        diagnostics=detail, source='same_frame_entry_minus_lower_end', stamp=sample[0]['header']['stamp'])


def emit(row, stage):
    from steward.ego.capability_usage import record
    record('tip_entry_servo', entity=row['object'], event_id=row['id'], stage=stage,
           measured=any(f.get('bothVisible') for f in row['frames']),
           action=any(c.get('executed') for c in row['corrections']), reason=row.get('reason'), evidence=row)


def run(controller, current, report, *, phase="hover", correct=True):
    if capabilities.get('STEWARD_TIP_ENTRY_SERVO') == '0' or (phase == 'hover' and current.get('_tipEntryDone')):
        return
    current['_tipEntryDone'] = True
    row = current.get('tipEntryServo')
    if row is None:
        row = dict(schema='tip_entry_servo.v1', id=uuid.uuid4().hex, object=current.get('object'),
        frames=[], corrections=[], residualAfterM=None, releaseExecuted=False,
        reason='target_unavailable', maxCorrections=2, kind=current.get('entryTargetKind','recess'))
    current['tipEntryServo'] = row
    if phase == 'hover': report.setdefault('tipEntryServos', []).append(row)
    ex, rt = controller.ex, controller.rt
    from steward.ego.key_observation import begin, finish
    checkpoint = begin(row, 'before_descent' if phase == 'hover' else 'near_contact' if phase == 'descent' else 'after_action')
    first_frame, first_correction = len(row['frames']), len(row['corrections'])
    try:
        target = (current.get('seatTarget') or {}).get('point')
        if target is None:
            target = (getattr(controller,'_registered_alignment',{}).get('registration') or {}).get('point')
        if target is None: return
        camera = current.get('_entryCameraOverride') or camera_for(rt); row['camera'] = camera
        if camera is None:
            row['reason']='camera_not_provided'; return
        opening = rt.esim.arm.geometry.jaw_opening
        if not opening or opening <= 0:
            row['reason']='jaw_envelope_unknown'; return
        from steward.ego.fasten_target import spacing
        from steward.ego.entry_layout import prior
        interface = prior(controller, current)
        if phase == 'after_drive': interface['contactPhase']=True
        separation = spacing(interface.get('layout', []))
        association_radius = min(opening/4,separation/2) if separation else opening/4
        row.update(reason=None,associationRadiusM=association_radius, entryRegistration=deepcopy(interface))
        last_stamp = row.setdefault('cameraStamps',{}).get(camera, -math.inf)
        phase_budget = row['maxCorrections'] if correct else 0
        index = 0
        while index <= phase_budget:
            controller.check()
            if rt.estimator.held_id != current.get('object'):
                row['reason']='held_object_changed'; break
            deadline = ab.make(ex,ab.frame_timeout(ex,4.),'one approach RGB-D frame')
            wo.context(controller,'tip_entry_'+phase,target,opening/2)
            controller._wristContext['driverProvenance'] = current.get('driverProvenance')
            controller._wristContext['entryRegistration'] = deepcopy(interface)
            sample = wo.capture(controller,deadline,camera=camera,check_view=False)
            frame = dict(index=len(row['frames']), phase=phase, capture=deepcopy(getattr(controller,'_wristCapture',{})), bothVisible=False)
            row['frames'].append(frame)
            if sample is None:
                row['reason']=frame['capture'].get('reason','frame_unavailable'); break
            stamp = sample[0]['header']['stamp']
            if stamp <= max(last_stamp,rt.estimator.held_since):
                row['reason']='frame_not_new'; break
            last_stamp=stamp; row['lastStamp']=stamp; row['cameraStamps'][camera]=stamp
            frame.update(observe(controller,sample,target,opening,association_radius,interface))
            from steward.ego import seating_geometry
            seating_geometry.remember(controller, current, frame, sample[0]['header'])
            from steward.ego import visible_entry
            if visible_entry.enabled():
                if current.get('_visibleEntrancePlan') and frame.get('targetFromLayout'):
                    frame['rejectedInferredMeasurement'] = {k:frame.get(k) for k in
                        ('entry','lowerEnd','deltaM','residualM','uncertaintyM')}
                    frame.update(bothVisible=False,reason='layout_inference_after_visible_target_unconfirmed')
                    for key in ('entry','deltaM','residualM'):
                        frame.pop(key,None)
                visible_entry.remember(controller,current,frame,target)
            emit(row,'frame')
            if not frame['bothVisible']:
                row['reason']=frame['reason']; break
            frame['lift'] = float(sample[0]['header']['joints']['lift'])
            if current.get('entryTargetKind') == 'head':
                current['_driverMeasurement'] = dict(frame, lift=frame['lift'],
                    gripXY=ex._gripper_xy().tolist())
            if not correct:
                break
            if row['corrections']:
                row['residualAfterM']=frame['residualM']
                row['corrections'][-1]['residualAfterM']=frame['residualM']
            if frame['residualM'] <= frame['uncertaintyM']:
                row['reason']='within_observation_resolution'; break
            if index == phase_budget and not (visible_entry.enabled() and frame.get('entry',{}).get('source') == 'same_frame_visible_entrance_pair'):
                row['reason']='correction_budget_exhausted'; break
            # Two steps cover a local trust radius <= jaw/4 and observed vertical
            # clearance. Each step <= jaw/8 and clearance/2: no dimensions of the
            # carried object, absolute map correction, or judge tolerances.
            clearance = -float(frame['deltaM'][2])-frame['uncertaintyM']
            trust = min(opening/4,clearance)
            if trust <= 0:
                row['reason']='no_observed_clearance'; break
            row.update(trustRadiusM=trust,stepLimitM=trust/2)
            visible_target = (visible_entry.enabled() and frame.get('entry',{}).get('source') == 'same_frame_visible_entrance_pair')
            if visible_target and index == 0:
                phase_budget = visible_entry.correction_budget(frame,opening,row['stepLimitM'])
                if seating_geometry.applies(current):
                    phase_budget = min(phase_budget, row['maxCorrections'])
                row['visibleTargetCorrectionBudget'] = phase_budget
            if index >= phase_budget:
                row['reason']='correction_budget_exhausted'; break
            if visible_target and index >= 2:
                previous = [f for f in row['frames'][:frame['index']] if f.get('bothVisible') and f.get('phase') == phase]
                if previous and frame['residualM'] > previous[-1]['residualM']+frame['uncertaintyM']:
                    row['reason']='correction_progress_unresolved'; break
            delta = np.asarray(frame['deltaM'][:2])
            step = delta*min(1.,row['stepLimitM']/frame['residualM'])
            correction = move(controller, step, frame['uncertaintyM'])
            correction.update(phase=phase,frameIndex=frame['index'],stepLimitM=row['stepLimitM'])
            row['corrections'].append(correction)
            row['residualAfterM']=None  # never reuse an earlier residual after motion
            emit(row,'correction')
            if not correction['executed']:
                row['reason']=correction['reason']; break
            # Keep the execution target consistent so later placement checks do
            # not drive back toward the old nominal TCP. Entry estimate is intact.
            path = current.get('seatEndpoint',current)
            path['gripSpot'] = correction['gripTargetM']
            if 'seatEndpoint' not in current: current['refineReleaseXY']=False
            index += 1
    except (TimeoutError, OSError, ValueError, KeyError, AttributeError, TypeError) as exc:
        row['reason']='servo_unavailable:'+type(exc).__name__
    finally:
        fresh = row['frames'][first_frame:]
        finish(checkpoint, measured=any(f.get('bothVisible') or f.get('visualContact') for f in fresh),
               action='correct_xy' if any(c.get('executed') for c in row['corrections'][first_correction:]) else None,
               reason=row.get('reason'))
        emit(row,phase+'_finished')


def move(controller, delta, uncertainty):
    """Plan a fixed-height, fixed-orientation local translation via the arm API."""
    ex=controller.ex; arm=controller.rt.esim.arm; state=ex._state()
    start=dict(state['joints']); yaw=state['odometry']['yaw']
    forward=np.array([math.cos(yaw),math.sin(yaw)])
    reach=start['reach']+float(np.asarray(delta) @ forward)
    path=[dict(start,reach=float(r)) for r in np.linspace(start['reach'],reach,5)]
    row=dict(deltaM=np.asarray(delta).tolist(),normM=float(np.linalg.norm(delta)),executed=False,
             residualAfterM=None,reason='local_path_unreachable')
    if not arm.reachable_path(path,start_joints=state['joints']): return row
    target=ex._gripper_xy()+delta
    row['gripTargetM']=target.tolist()
    from steward.ego.skills import SkillFailed
    try:
        # No rotational base command: the virtual planar Jacobian columns are
        # forward reach and sideways base translation, both unit length.
        ex._align_lateral(target.tolist(),tol=uncertainty/2)
        ex._set_joint('reach',ex._reach_to(target),tol=uncertainty/2)
        row.update(executed=True,reason=None)
    except SkillFailed as exc:
        controller.check()  # cancellation/deadline still propagates
        row['reason']='local_motion_incomplete:'+str(exc)[:100]
    return row


def finish(current):
    row=current.get('tipEntryServo')
    if row is not None:
        row['releaseExecuted']=bool(current.get('releaseExecuted'))
        if row['releaseExecuted']: row['releasedAt']=current.get('releasedAt')
        emit(row,'release_outcome')


def driver_hover(controller, plan, motion):
    """Use the same bounded observation on the existing tool descent path."""
    if plan.get('_entryServoDone'): return
    rt=controller.rt
    if getattr(rt,'estimator',None) is None: return
    opening=rt.esim.arm.geometry.jaw_opening
    if not opening or controller.ex._state()['joints']['lift']-plan['lift'] > opening/2: return
    plan['_entryServoDone']=True
    current=dict(object=plan['object'],seatTarget=dict(point=plan['head']),
        seatHole=plan.get('seatHole'),body=plan.get('body'),entryTargetKind='head',
        gripSpot=plan['gripSpot'],driverProvenance=dict(targetSource=plan.get('targetSource'),
            targetEvidence=deepcopy(plan.get('targetEvidence')),toolGeometry=deepcopy(plan.get('driverTipGeometry')),
            plannedLift=plan['lift'],goal=plan.get('goal')))
    plan['_entryCurrent']=current
    run(controller,current,motion)
    if 'tipEntryServo' in current: motion['tipEntryServo']=current['tipEntryServo']
    driver_adopt(controller,plan,motion)
    if any(c.get('executed') for c in current.get('tipEntryServo',{}).get('corrections',[])):
        plan['gripSpot']=current['gripSpot']


def driver_adopt(controller, plan, motion):
    """Lift is a unit vertical Jacobian. Use visible tip/head gap for endpoint.

    End one existing contact increment below the observed head, so following
    error can be detected. Never compensate a blocked joint or extend past the
    local half-jaw envelope; retain the existing reachability/contact checks.
    """
    current=plan.get('_entryCurrent',{}); measured=current.pop('_driverMeasurement',None)
    if not measured:return
    from steward.ego.driver_contact import STEP_M
    row=current['tipEntryServo']; opening=controller.rt.esim.arm.geometry.jaw_opening
    original=plan.setdefault('_entryOriginalLift',float(plan['lift']))
    end=float(measured['lift']+measured['deltaM'][2]-STEP_M)
    if abs(end-original)>opening/2:
        row['endpointUpdateReason']='observed_endpoint_outside_local_envelope'
        return False
    state=controller.ex._state(); joints=dict(state['joints'],lift=end)
    if not controller.rt.esim.arm.reachable_path([dict(state['joints']),joints]):
        row['endpointUpdateReason']='observed_endpoint_path_unreachable'; return
    current['seatTarget']['point']=list(measured['entry']['point'])
    plan.update(lift=end,head=list(measured['entry']['point']),
        gripSpot=current['gripSpot'],targetSource='same_frame_raised_head')
    plan['_entryMeasuredTip']=dict(tip=measured['lowerEnd']['tip'],lift=measured['lift'],
        stamp=measured['stamp'],source=measured['lowerEnd']['source'],
        gripXY=measured.get('gripXY'))
    row.setdefault('endpointUpdates',[]).append(dict(frameIndex=measured['index'],lift=end,
        rawLift=measured['lift']+measured['deltaM'][2]-STEP_M,
        head=plan['head'],source=plan['targetSource']))
    emit(row,'endpoint_updated')
    return True


def placement_descent(controller, current, report, path, *, initialize=False, reached=None):
    if initialize:
        current['_nearContactObserved'] = False
        return
    if current.get('_nearContactObserved'):
        return
    current['_nearContactObserved'] = True
    near_endpoint(controller, current, report, path)


def placement_view(controller, current, report, path):
    """One primary observation and at most one changed calibrated viewpoint."""
    primary = camera_for(controller.rt)
    run(controller, current, report, phase='descent', correct=True)
    row = current.get('tipEntryServo', {})
    def visible():
        frame = (row.get('frames') or [{}])[-1]
        return frame.get('bothVisible') and frame.get('stamp') == row.get('lastStamp')
    if visible():
        return
    alternate = next((c for c in getattr(controller.rt, '_provided_camera_ids', ())
        if c != primary and c in controller.rt.estimator.cameras
        and controller.rt.estimator.cameras[c][1].get('intrinsics')), None)
    change = dict(executed=False, reason='no_alternate_view')
    row.setdefault('nearContactViewChanges', []).append(change)
    if alternate is not None:
        current['_entryCameraOverride'] = alternate
        change.update(executed=True, camera=alternate, source='alternate_calibrated_camera')
    else:
        ex = controller.ex
        state = ex._state()
        opening = controller.rt.esim.arm.geometry.jaw_opening
        lift = min(path.get('safeLift', state['joints']['lift']), state['joints']['lift']+opening/8)
        joints = dict(state['joints'], lift=lift)
        if lift <= state['joints']['lift'] or not controller.rt.esim.arm.reachable_path([dict(state['joints']), joints]):
            return
        from steward.ego.fasten_seat import lift_to
        lift_to(ex, lift)
        change.update(executed=True, source='checked_clearance_lift')
    try:
        run(controller, current, report, phase='descent', correct=True)
    finally:
        current.pop('_entryCameraOverride', None)


def unconfirmed_endpoint(controller, current, path, saved, row, checkpoint):
    """Return to the pre-descent corrected target, then use ordinary release."""
    from steward.ego.skills import SkillFailed
    from steward.ego.key_observation import finish
    controller.check()
    if controller.rt.estimator.held_id != current.get('object'):
        raise SkillFailed('held_object_changed')
    from steward.ego import visible_entry
    from steward.ego import seating_geometry
    keep_reached = seating_geometry.applies(current) and bool(current.get('seatingGeometryEntry') or current.get('_seatingGeometryEndpoint'))
    if keep_reached:
        # Keep the reached correction. Final delivery transports the last image
        # endpoint with FK; restoring an old target here would undo that evidence.
        saved = dict(saved, gripSpot=controller.ex._gripper_xy().tolist())
    elif not seating_geometry.applies(current) and visible_entry.enabled() and current.get('_visibleEntrancePlan'):
        saved = visible_entry.finish_estimate(controller,current,saved,row)
    # Near-view corrections may have moved XY before visibility was lost.
    # Restore the saved target through the same checked translation API.
    if not seating_geometry.applies(current) and not (visible_entry.enabled() and current.get('_visibleEntrancePlan')) and any(
            c.get('executed') and c.get('phase') == 'descent' for c in row.get('corrections', [])):
        delta = np.asarray(saved['gripSpot'])-controller.ex._gripper_xy()
        uncertainty = next((f['uncertaintyM'] for f in reversed(row.get('frames', []))
                            if f.get('bothVisible')), controller.rt.esim.arm.geometry.jaw_opening/8)
        motion = move(controller, delta, uncertainty)
        row['fallbackRestore'] = motion
        if not motion['executed']:
            raise SkillFailed('pre_descent_target_restore_failed')
    from steward.ego.release_height import guard
    guard(controller, current, saved)
    path.update(saved)
    actual = current.get('seatEndpoint', current)
    actual.update(saved)
    current['refineReleaseXY'] = False
    current['_fallbackReleaseEndpoint'] = dict(saved, heldSince=controller.rt.estimator.held_since)
    current['nearContactUnconfirmed'] = True
    current.setdefault('seatMotion', {})['nearContactStatus'] = '临近未确认'
    if hasattr(controller, 'warn'):
        controller.warn('near_contact_unconfirmed', detail='临近未确认')
    row['nearContactUnconfirmed'] = True
    if not (visible_entry.enabled() and current.get('_visibleEntrancePlan')):
        row['fallbackTargetSource'] = 'pre_descent_corrected_release_plan'
    finish(checkpoint, measured=False, action='release_unconfirmed', reason='near_contact_unconfirmed')
    emit(row, 'endpoint_unconfirmed_continue') if row else None


def near_endpoint(controller, current, report, path):
    if capabilities.get('STEWARD_TIP_ENTRY_SERVO') == '0':
        return
    from steward.ego.skills import SkillFailed
    from steward.ego.key_observation import finish
    saved = {k:deepcopy(path[k]) for k in ('lift', 'gripSpot') if k in path}
    actual = current.get('seatEndpoint', current)
    if 'gripSpot' in actual: saved['gripSpot'] = deepcopy(actual['gripSpot'])
    placement_view(controller,current,report,path)
    row=current.get('tipEntryServo',{})
    frame=(row.get('frames') or [{}])[-1]
    checkpoint=(row.get('keyObservations') or [None])[-1]
    if frame.get('phase') != 'descent' or not frame.get('bothVisible') or frame.get('stamp') != row.get('lastStamp'):
        return unconfirmed_endpoint(controller,current,path,saved,row,checkpoint)
    # A lateral correction invalidates the frame for endpoint adoption. Read once
    # more at the changed pose; never descend using pre-correction geometry.
    if any(c.get('executed') and c.get('frameIndex')==frame['index'] for c in row.get('corrections',[])):
        run(controller,current,report,phase='descent',correct=False)
        frame=row['frames'][-1]; checkpoint=row['keyObservations'][-1]
        if not frame.get('bothVisible') or frame.get('stamp') != row.get('lastStamp'):
            return unconfirmed_endpoint(controller,current,path,saved,row,checkpoint)
    if frame['residualM'] > frame['uncertaintyM']:
        from steward.ego import visible_entry
        from steward.ego import seating_geometry
        if seating_geometry.applies(current) or visible_entry.enabled() and current.get('_visibleEntrancePlan'):
            return unconfirmed_endpoint(controller,current,path,saved,row,checkpoint)
        finish(checkpoint,action='hold_for_observation',reason='lateral_alignment_unresolved')
        emit(row,'endpoint_unresolved')
        raise SkillFailed('lateral_alignment_unresolved')
    opening=controller.rt.esim.arm.geometry.jaw_opening
    from steward.ego.fasten_delivery import RELEASE_HEIGHT_M
    end=float(frame['lift']+frame['deltaM'][2]+RELEASE_HEIGHT_M)
    state=controller.ex._state()
    original=float(path['lift'])
    if (abs(end-original)>opening/2 or not controller.rt.esim.arm.reachable_path(
            [dict(state['joints']),dict(state['joints'],lift=end)])):
        from steward.ego import visible_entry
        from steward.ego import seating_geometry
        if seating_geometry.applies(current) or visible_entry.enabled() and current.get('_visibleEntrancePlan'):
            return unconfirmed_endpoint(controller,current,path,saved,row,checkpoint)
        finish(checkpoint,action='hold_for_observation',reason='observed_endpoint_unreachable')
        emit(row,'endpoint_unresolved')
        raise SkillFailed('observed_endpoint_unreachable')
    path['lift']=end
    from steward.ego.release_height import guard
    guard(controller, current, path)
    end = path['lift']
    actual=current.get('seatEndpoint',current)
    if 'gripSpot' in actual:path['gripSpot']=actual['gripSpot']
    current['seatTarget']=dict(current.get('seatTarget',{}),**frame['entry'])
    current['_observedReleaseEndpoint']=dict(lift=end, stamp=frame['stamp'],
        uncertaintyM=frame['uncertaintyM'], heldSince=controller.rt.estimator.held_since)
    current['refineReleaseXY']=False
    if 'targetEvidence' in path:
        path['targetEvidence']=dict(path['targetEvidence'],point=list(frame['entry']['point']),
            observedAt=frame['stamp'],targetSource='same_frame_visible_gap')
    row.setdefault('endpointUpdates',[]).append(dict(frameIndex=frame['index'],lift=end,
        previousLift=original,source='same_frame_visible_gap'))
    finish(checkpoint,action='update_endpoint' if end!=original else 'continue',measured=True)
    emit(row,'endpoint_updated')


def ordinary_descent(ex, plan, observe):
    observe(plan,initialize=True)
    observe(plan,reached=ex._state()['joints']['lift'])
    from steward.ego.fasten_seat import lift_to
    lift_to(ex,float(plan['lift']))


def driver_after_contact(controller, plan, motion):
    """A fresh tip/head view controls dwell and continuation, not just a receipt."""
    from steward.ego.skills import SkillFailed
    from steward.ego.key_observation import finish
    current=plan.get('_entryCurrent')
    if current is None:
        raise SkillFailed('tool_tip_on_head_unresolved')
    current_view(controller,current,motion,phase='after_drive')
    row=current.get('tipEntryServo',{})
    frame=(row.get('frames') or [{}])[-1]
    checkpoint=(row.get('keyObservations') or [None])[-1]
    measured=bool((frame.get('bothVisible') or frame.get('visualContact')) and frame.get('stamp')==row.get('lastStamp')
                  and frame.get('phase')=='after_drive')
    if measured and frame.get('visualContact'):
        passed=True
    elif measured:
        delta=np.asarray(frame['deltaM']); uncertainty=frame['uncertaintyM']
        passed=bool(np.linalg.norm(delta[:2]) <= frame['entry'].get('radiusM',0.)+uncertainty
                    and abs(delta[2])<=uncertainty)
    else: passed=None
    motion['postContactCheck']=dict(passed=passed,status='on_head' if passed else
        'off_head' if passed is False else 'unknown',observedAt=frame.get('stamp'))
    finish(checkpoint,measured=measured,action='continue' if passed else 'withdraw_and_report',
           reason=motion['postContactCheck']['status'])
    emit(row,'post_contact_checked')
    if not passed:raise SkillFailed('tool_tip_'+motion['postContactCheck']['status'])


def current_view(controller, current, report, *, phase, correct=False):
    """Try the available calibrated viewpoints without moving on remembered Z."""
    rt=controller.rt
    primary=camera_for(rt)
    cameras=[primary] if primary else []
    cameras += [c for c in getattr(rt, '_provided_camera_ids', ())
                if c != primary and c in rt.estimator.cameras
                and rt.estimator.cameras[c][1].get('intrinsics')]
    try:
        for camera in cameras or [None]:
            current['_entryCameraOverride']=camera
            run(controller,current,report,phase=phase,correct=correct)
            row=current.get('tipEntryServo',{}); frame=(row.get('frames') or [{}])[-1]
            if (frame.get('bothVisible') or frame.get('visualContact')) and frame.get('stamp')==row.get('lastStamp'):
                return
    finally:
        current.pop('_entryCameraOverride',None)
