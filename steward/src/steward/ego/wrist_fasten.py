"""Optional wrist observations inside existing deterministic fastening stages."""
from steward.ego.phase_timing import timed
from copy import deepcopy
import math
import time
from steward.ego import action_budget as ab
import numpy as np
from steward.ego import wrist_observation as wo


TOLERANCE_M=.002
STEP_M=.010
MAX_SAMPLES=3


def held_feature(controller, sample, *, search_radius=None):
    controller._heldFeatureDiagnostic = dict(reason='held_track_unavailable')
    est=controller.rt.estimator; track=est.tracks.get(est.held_id)
    if track is None: return None
    header=sample[0]['header']
    if header['stamp']<=est.held_since:
        controller._heldFeatureDiagnostic['reason'] = 'frame_precedes_grasp'
        return None
    grip=est.gripper_position(est.held_gripper,header['joints'],header['odometry'])
    prior=getattr(track,'held_grasp_geometry',None) or {}
    radius=max(.025,float(prior.get('lengthM',np.max(track.size))))+.015
    radius = min(radius,.10) if search_radius is None else max(radius,search_radius)
    return wo.tool(controller,sample,grip,track.category,radius,held=True,
                   diagnostics=controller._heldFeatureDiagnostic)


def calibrate_sample(controller,feature,sample):
    """Cancel same-frame odometry noise before storing the measured gripper offset."""
    from steward.ego.fasten_geometry import transform_feature
    with controller.rt._lock:
        est=controller.rt.estimator;track=est.tracks.get(est.held_id);header=sample[0]['header']
        if track is None or not track.held or header['stamp']<=est.held_since: return False
        grip=est.gripper_position(est.held_gripper,header['joints'],header['odometry'])
        rotation=est.gripper_rotation(est.held_gripper,header['joints'],header['odometry'])
        track.held_tool_feature=transform_feature(feature,rotation.T,-rotation.T @ grip)
        from steward.ego.grasp_endpoints import inhand_observation
        inhand_observation(track)
        centre=(np.asarray(feature['tip'])+feature['head'])/2
        track.held_offset=rotation.T @ (centre-grip);track.pos=centre
        est.version+=1
        return True


def posture(controller):
    from steward.ego.grasp_diagnostics import inspection
    eid = controller.rt.estimator.held_id
    if not wo.enabled(controller.rt,'POSTURE'):
        inspection(controller.rt, eid, 'wrist_posture', dict(attempted=False, reason='wrist_posture_unavailable'))
        return None
    wo.context(controller,'posture')
    start=time.monotonic(); sample=wo.capture(controller,ab.make(controller.ex,ab.frame_timeout(controller.ex,4.),'single configured wrist frame period'))
    controller._wristPosture=deepcopy(getattr(controller,'_wristCapture',{}))
    feature=held_feature(controller,sample) if sample else None
    result = dict(attempted=True, frameCaptured=sample is not None,
        capture=controller._wristPosture, measured=bool(feature), feature=feature,
        detector=getattr(controller, '_heldFeatureDiagnostic', None) if sample else None)
    if feature is None:
        inspection(controller.rt, eid, 'wrist_posture', result, sample)
        return None
    axis=np.asarray(feature['head'])-feature['tip'];axis/=np.linalg.norm(axis)
    angle=math.degrees(math.acos(float(np.clip(axis[2],-1,1))))
    calibrated = calibrate_sample(controller,feature,sample)
    result['calibrated'] = calibrated
    inspection(controller.rt, eid, 'wrist_posture', result, sample)
    if not calibrated: return None
    track=controller.rt.estimator.tracks[controller.rt.estimator.held_id]
    track.held_grasp_geometry=deepcopy(track.held_tool_feature)
    return dict(status='upright' if angle<=10 else 'contradicted',
        reason='upright' if angle<=10 else 'inverted' if angle>90 else 'tilted',
        axisTiltDeg=round(angle,2),directedAngleDeg=round(angle,2),polarityObserved=True,
        source='wrist_rgbd_endpoint_width',observedAt=feature['stamp'],wallS=round(time.monotonic()-start,3))


def select_hole(found, rows, occupied, requested):
    """Use pair geometry; occupancy identity is supplied by the release ledger."""
    from steward.ego.layout_pose import match
    occupied_ids = [o.get('attempt',{}).get('hole') or min(rows,
        key=lambda r:np.linalg.norm(np.asarray(r['point'])[:2]-np.asarray(o['point'])[:2]))['bodyHole']
        for o in occupied]
    fit, _ = match(rows, found, occupied_ids)
    if fit is None: return None
    direct = next((m for m in fit['matches'] if m['hole']==requested),None)
    if direct is None and len(fit['matches'])==1: direct=fit['matches'][0]
    if direct is None: return None
    return dict(rows[direct['row']], **{k:v for k,v in found[direct['feature']].items() if k not in ('id','bodyHole')},
                layoutFit=fit)


@timed("wrist_hover")
def hover(controller,current,params,report):
    if not wo.enabled(controller.rt,'ALIGN'): return False
    ctx=getattr(controller,'_registered_alignment',{});reg=ctx.get('registration')
    if not reg: return False
    row=report.setdefault('wristAlignment',dict(samples=0,status='unknown',budgetS=12.,corrections=0))
    started=report.setdefault('_wristStart',time.monotonic())
    key, deadline = getattr(controller, '_wrist_budget', (None, None))
    if key != id(report):
        deadline=ab.make(controller.ex,ab.frame_window(controller.ex,MAX_SAMPLES),'configured frame period per wrist sample') if ab.enabled() else started+12.
        controller._wrist_budget=(id(report),deadline)
    if row['samples']>=MAX_SAMPLES or ab.expired(deadline): return False
    row['samples']+=1
    from steward.ego.capability_usage import record
    row['_usageId'] = record('wrist_hole', entity=current.get('object'), stage='capture')
    wo.context(controller,'alignment',reg['point'],.05)
    sample=wo.capture(controller, ab.limited(deadline,ab.make(controller.ex,ab.frame_timeout(controller.ex,4.),'single configured wrist frame period')))
    feature=held_feature(controller,sample) if sample else None
    if not sample:
        row.update(reason=getattr(controller,'_wristCapture',{}).get('reason','frame_unavailable'),wallS=round(time.monotonic()-started,3))
        record('wrist_hole', event_id=row['_usageId'], reason=row['reason']);return False
    rows=reg.get('layout',reg.get('holes',[]))
    if not rows or any('bodyHole' not in r for r in rows):
        record('wrist_hole', event_id=row['_usageId'], reason='layout_unavailable');return False
    from steward.ego.fasten_target import evidence
    found=wo.holes(sample,reg['point'])
    row['detection']=sample[0].get('holeDiagnostics',{})
    selected=select_hole(found,rows,evidence(controller.rt.estimator,reg['frame']),reg['hole'])
    if not selected:
        row.update(reason='empty_hole_unseen_or_ambiguous',wallS=round(time.monotonic()-started,3))
        record('wrist_hole', event_id=row['_usageId'], reason=row['reason']);return False
    est=controller.rt.estimator
    target=np.asarray(selected['point']);stamp=sample[0]['header']['stamp']
    record('wrist_hole', event_id=row['_usageId'], measured=True, stamp=stamp)
    old=reg['hole']
    from steward.ego.wrist_target import enabled as transactional, adoption_enabled, propose
    if adoption_enabled() or transactional():
        candidate = propose(reg, selected, stamp, row)
        row.update(status='hole_only', targetFrame='odometry_map_m', selectedHole=reg['hole'],
                   offsetSource=ctx.get('tipOffsetSource', 'prior_grasp'), offsetPreserved=True)
        if candidate is not None:
            report['_wristCandidate'] = candidate
        if feature and est.held_id == current['object'] and calibrate_sample(controller, feature, sample):
            # A fresh directed offset is evidence about the held object; the mouth
            # is still a proposal until the endpoint receipt is consumed.
            ctx.update(tip=deepcopy(est.tracks[current['object']].held_tool_feature),
                       tipHeldSince=est.held_since, tipOffsetSource='measured')
            ctx['tipOffsetMeasuredCount'] += 1
            measured_error = float(np.linalg.norm(target[:2]-feature['tip'][:2]))
            report.update(_wristMeasured=True, tipOffsetMeasured=True, tipOffsetSource='measured',
                alignmentBasis='registered_hole_fk', preReleaseResidualM=measured_error,
                tipOffsetMeasuredCount=ctx['tipOffsetMeasuredCount'])
            row.update(status='measured', residualM=measured_error, offsetSource='measured',
                       offsetRecalibrated=True)
            report.setdefault('observations', []).append(dict(status='estimated', residualM=measured_error,
                source='same_wrist_frame_tip_and_empty_hole', observedAt=stamp,
                uncertaintyM=selected['uncertaintyM']+feature.get('uncertaintyM', TOLERANCE_M)))
            return True
        return False
    # Empty holes remain observable even when the held tip is hidden in the palm.
    reg.update(hole=selected['bodyHole'],point=target.tolist(),executionPoint=target.tolist(),observedAt=stamp,
        measurementSource='wrist_rgbd_empty_recess',uncertaintyM=selected['uncertaintyM'])
    report['_wristEmptyEntry']=dict(point=target.tolist(),stamp=stamp,
        radiusM=(feature or {}).get('radiusM'))
    row.update(status='hole_only',reason='tip_unseen',selectedHole=reg['hole'],
        targetPointM=target.tolist(),observedAt=stamp,wallS=round(time.monotonic()-started,3))
    params['seatHole']=reg['hole'];current['seatHole']=reg['hole'];report['hole']=reg['hole']
    if not feature:
        return False
    if est.held_id!=current['object'] or not calibrate_sample(controller,feature,sample): return False
    delta=target[:2]-feature['tip'][:2]
    residual=float(np.linalg.norm(delta));correction=delta*min(1.,STEP_M/max(residual,1e-12))
    point=target.copy();point[:2]=np.asarray(feature['tip'])[:2]+correction
    params['seatHole']=reg['hole'];current['seatHole']=reg['hole'];report['hole']=reg['hole']
    ctx.update(tip=deepcopy(est.tracks[current['object']].held_tool_feature),tipHeldSince=est.held_since,
        tipOffsetSource='measured',correctionPoint=point.tolist())
    ctx['tipOffsetMeasuredCount']+=1
    report.update(tipOffsetMeasured=True,tipOffsetSource='measured',alignmentBasis='registered_hole_fk',
        preReleaseResidualM=residual,tipOffsetMeasuredCount=ctx['tipOffsetMeasuredCount'])
    report['_wristMeasured']=True
    report['_wristEmptyEntry']=dict(point=target.tolist(),stamp=stamp,radiusM=feature.get('radiusM'))
    row.update(status='converged' if residual<=TOLERANCE_M else 'measured',reason=None,
        residualM=round(residual,6),correctionM=correction.round(6).tolist(),stepLimitM=STEP_M,
        toleranceM=TOLERANCE_M,selectedHole=reg['hole'],targetPointM=target.tolist(),
        reassignedFrom=old if old!=reg['hole'] else None,
        observedAt=stamp,wallS=round(time.monotonic()-started,3))
    report.setdefault('observations',[]).append(dict(status='estimated',residualM=residual,
        source='same_wrist_frame_tip_and_empty_hole',observedAt=stamp,uncertaintyM=selected['uncertaintyM']+.002))
    return True


def capture_release(controller, plan):
    """Clear the fingers vertically before the ordinary retract hides the entry."""
    if not wo.enabled(controller.rt,'RELEASE'): return
    from steward.ego import wrist_release_capture
    if wrist_release_capture.enabled():
        return wrist_release_capture.capture(controller, plan)
    from steward.ego.skills import SkillFailed
    start=time.monotonic();state=controller.ex._state();a=dict(state['joints'])
    capture_deadline=ab.make(controller.ex,2.+ab.frame_window(controller.ex) if ab.enabled() else 4.,'two-second withdrawal servo plus one configured frame period')
    target=(plan.get('seatTarget') or {}).get('point')
    if target is None: target=(getattr(controller,'_registered_alignment',{}).get('registration') or {}).get('point')
    wo.context(controller,'fastener_release',target,.05)
    plan['_wristReleaseReason']='frame_unavailable'
    lift=.025
    b=dict(a,lift=a['lift']+lift)
    path=[dict(a,lift=float(h)) for h in np.linspace(a['lift'],b['lift'],14)]
    try:
        if lift>1e-6 and controller.rt.esim.arm.reachable_path(path):
            left=ab.left(wo.bounded_deadline(controller,capture_deadline))
            if left<=0:
                plan['_wristReleaseSample']=None
                plan['_wristReleaseReason']='existing_deadline_exhausted';return
            controller.ex._set_joint('lift',b['lift'],timeout=min(2.,left),ramp=False)
        plan['_wristReleaseSample']=wo.capture(controller,capture_deadline)
        plan['_wristReleaseReason']=getattr(controller,'_wristCapture',{}).get('reason','frame_unavailable')
    except SkillFailed:
        controller.check()
        plan['_wristReleaseSample']=None
        plan['_wristReleaseReason']='release_clearance_motion_failed'
    plan['_wristReleaseWallS']=time.monotonic()-start
    if ab.enabled(): plan['_wristReleaseSimS']=capture_deadline.sim_s
    if wrist_release_capture.recording():
        plan['_wristReleasePhase'] = 'after_open_before_withdrawal'
        plan['_wristReleaseFrame'] = wrist_release_capture.record(controller,
            plan.get('_wristReleaseSample'), target, plan.get('releasedAt'), plan['_wristReleasePhase'])


def cap_check(sample,attempt):
    """A newly appeared compact head with an observed upright surface normal."""
    baseline=(attempt.get('precision') or {}).get('_wristEmptyEntry')
    if not baseline or not baseline.get('radiusM'): return None
    raw,cam=sample;entry=np.asarray(attempt['entry']);radius=baseline['radiusM']
    if np.linalg.norm(np.asarray(baseline['point'])-entry)>.004: return None
    region=wo.window(raw,cam,entry,max(.03,4*radius))
    points,_,_=wo.cloud(raw,cam,region)
    radial=np.linalg.norm(points[:,:2]-entry[:2],axis=1)
    ring=points[(radial>2*radius)&(radial<4*radius)]
    core=points[radial<1.5*radius]
    if min(len(ring),len(core))<30: return None
    design=np.c_[ring[:,:2]-entry[:2],np.ones(len(ring))]
    coef=np.linalg.lstsq(design,ring[:,2],rcond=None)[0]
    if np.quantile(abs(design @ coef-ring[:,2]),.95)>.003: return None
    height=core[:,2]-np.c_[core[:,:2]-entry[:2],np.ones(len(core))] @ coef
    cap=core[(height>.0025)&(height<.012)]
    if len(cap)<30: return None
    size=np.quantile(cap[:,:2],.95,axis=0)-np.quantile(cap[:,:2],.05,axis=0)
    if min(size)<radius or max(size)>3*radius or max(size)/min(size)>1.8: return None
    centre=np.median(cap,axis=0);design=np.c_[cap[:,:2]-centre[:2],np.ones(len(cap))]
    fit=np.linalg.lstsq(design,cap[:,2],rcond=None)[0]
    if np.quantile(abs(design @ fit-cap[:,2]),.95)>.003: return None
    tilt=math.degrees(math.atan(np.linalg.norm(fit[:2])))
    error=float(np.linalg.norm(centre[:2]-entry[:2]))
    if tilt>10 or error>.004: return None
    from steward.ego.fasten_completion import CONFIRM
    extra = dict(headMinusSurfaceZM=float(np.median(height[(height>.0025)&(height<.012)])),
        surfaceZM=float(coef[2]),axisCrossingErrorM=error) if CONFIRM else {}
    return dict(**extra,status='confirmed',passed=True,reason='new_wrist_head_and_upright_cap',
        source='wrist_rgbd_cap_normal',observedAt=raw['header']['stamp'],baselineAt=baseline['stamp'],
        freshEndpointObserved=False,axisObserved=True,axisTiltDeg=tilt,headM=centre.tolist(),
        headDistanceXYM=error,uncertaintyM=.002,attachmentObserved=False)


@timed("wrist_release_observation")
def released(controller,job,attempt,plan):
    from steward.ego.hole_seating import apply as apply_hole
    if apply_hole(attempt, plan):
        row = deepcopy(attempt['holeSeatObservation'])
        job.postcondition['holeSeatObservation'] = row
        return row
    active=wo.enabled(controller.rt,'RELEASE')
    if not active and (not wo.release_reporting() or
            wo.CAMERA not in getattr(controller.rt,'_provided_camera_ids',())): return None
    start=time.monotonic()
    row=dict(camera=wo.CAMERA,status='unknown',reason='frame_unavailable',budgetS=4.,
             screw=attempt.get('screw'),hole=attempt.get('hole'),seatedAt=attempt.get('seatedAt'))
    sample=None;feature=None
    if not active:
        row.update(status='skipped',reason='release_not_enabled')
        row.pop('camera');row.pop('budgetS')
    try:
        if active:
            retained_sample = plan is not None and '_wristReleaseSample' in plan
            if retained_sample:
                sample=plan.pop('_wristReleaseSample')
                row['reason']=plan.pop('_wristReleaseReason','frame_unavailable')
            else:
                wo.context(controller,'fastener_release',attempt['entry'],.05)
                sample=wo.capture(controller,ab.make(controller.ex,ab.frame_timeout(controller.ex,4.),'single configured wrist frame period'))
                row['reason']=getattr(controller,'_wristCapture',{}).get('reason','frame_unavailable')
            from steward.ego import wrist_release_capture
            if wrist_release_capture.recording():
                retained_frame = plan.get('_wristReleaseFrame') if retained_sample else None
                phase = plan.get('_wristReleasePhase', 'after_open_before_withdrawal') if retained_sample else 'after_place_return'
                row['releaseFrame'] = deepcopy(retained_frame) or wrist_release_capture.record(
                    controller, sample, attempt['entry'], attempt['seatedAt'],
                    phase)
            if retained_sample and wrist_release_capture.enabled() and plan.get('_wristReleaseView'):
                row['view'] = deepcopy(plan['_wristReleaseView'])
            if sample:
                row['observedAt']=sample[0]['header']['stamp']
                row['reason']='stale_release_frame'
            if sample and sample[0]['header']['stamp']>attempt['seatedAt']:
                row['reason']='no_separate_visible_shaft'
                track=controller.rt.estimator.tracks.get(attempt['screw'])
                if track:
                    feature=wo.tool(controller,sample,np.asarray(attempt['entry']),track.category,.05)
                if not (attempt.get('precision') or {}).get('_wristEmptyEntry',{}).get('radiusM'):
                    row['capReason']='no_measured_pre_release_shaft_radius'
        identity_head=feature.get('head') if feature else None
        if identity_head is not None:
            # An already-accounted head is not evidence for this release identity.
            for other in controller.rt.estimator.fastening_attempts.values():
                if other is attempt or other.get('screw')==attempt['screw']: continue
                head=(other.get('seatCheck') or {}).get('headM')
                if head is not None and np.linalg.norm(np.asarray(head)-identity_head)<.010:
                    feature=None
                    break
        if feature:
            axis=np.asarray(feature['head'])-feature['tip'];axis/=np.linalg.norm(axis)
            tilt=math.degrees(math.acos(float(np.clip(axis[2],-1,1))))
            distance=float(np.linalg.norm(np.asarray(feature['tip'])[:2]-attempt['entry'][:2]))
            height=float(feature['tip'][2]-attempt['entry'][2])
            passed=distance<=.004 and tilt<=10 and abs(height)<=.004
            failed=distance>.008 or tilt>20 or abs(height)>.010
            from steward.ego.fasten_completion import CONFIRM
            extra={}
            if CONFIRM:
                head=np.asarray(feature['head']);tip=np.asarray(feature['tip'])
                head_height=float(head[2]-attempt['entry'][2])
                length=float(np.linalg.norm(head-tip))
                crossing=tip+(attempt['entry'][2]-tip[2])*axis/axis[2] if abs(axis[2])>1e-6 else None
                cross_error=float(np.linalg.norm(crossing[:2]-attempt['entry'][:2])) if crossing is not None else None
                passed=passed and cross_error is not None and cross_error<=.004 and -.004<=head_height<=length+.004
                extra=dict(headMinusSurfaceZM=head_height,surfaceZM=float(attempt['entry'][2]),
                    surfaceSource='registered_entry',axisCrossingErrorM=cross_error)

            check=dict(**extra,status='confirmed' if passed else 'failed' if failed else 'unknown',passed=True if passed else False if failed else None,
                reason='independent_wrist_geometry',source='wrist_rgbd_endpoint_width',observedAt=feature['stamp'],
                freshEndpointObserved=True,headM=feature['head'],tipM=feature['tip'],axisTiltDeg=tilt,
                distanceToHoleM=distance,headDistanceXYM=float(np.linalg.norm(np.asarray(feature['head'])[:2]-attempt['entry'][:2])),
                uncertaintyM=.002)
            attempt['wristSeatCheck']=check
            attempt['seatCheck']=check
            row.update(status=check['status'],reason=check['reason'],check=deepcopy(check))
        if not feature and sample and sample[0]['header']['stamp']>attempt['seatedAt']:
            cap=cap_check(sample,attempt)
            if cap:
                attempt['wristSeatCheck']=cap;attempt['seatCheck']=cap
                row.update(status=cap['status'],reason=cap['reason'],check=deepcopy(cap))
    except (ValueError,KeyError,TypeError,AttributeError,np.linalg.LinAlgError) as exc:
        row.update(status='unknown',reason='invalid_release_geometry',detail=type(exc).__name__)
    finally:
        from steward.ego.wrist_landing import observe as observe_landing
        feedback=observe_landing(controller,job,attempt,sample,feature) if active else None
        if feedback is None and wo.release_reporting():
            feedback=dict(status='skipped',reason='landing_not_enabled' if active else 'release_not_enabled')
            attempt['landingFeedback']=feedback;job.postcondition['landingFeedback']=deepcopy(feedback)
            from steward.ego.wrist_landing import record
            record(controller,attempt,feedback)
        from steward.ego.fasten_completion import CONFIRM
        if CONFIRM:
            row.update(confirmedSeated=row['status']=='confirmed',unknown=row['status']=='unknown')
        row['wallS']=round(time.monotonic()-start+(plan or {}).pop('_wristReleaseWallS',0.),3)
        if not active: row.pop('wallS',None)
        attempt['wristRelease']=deepcopy(row)
        job.postcondition['wristRelease']=deepcopy(row)
    return row if row['status'] not in ('unknown','skipped') else None


def publish_release(parent, child):
    """Keep one record per release when nested motion plans are compacted."""
    if not wo.release_reporting() or parent is child: return
    items=child.get('wristReleaseReports',[])
    if 'wristRelease' in child:
        items=[*items,{key:deepcopy(child[key]) for key in ('wristRelease','landingFeedback') if key in child}]
    if not items: return
    history=parent.setdefault('wristReleaseReports',[])
    for item in items:
        row=item['wristRelease']
        identity=tuple(row.get(k) for k in ('screw','hole','seatedAt'))
        history[:]=[r for r in history if tuple(r['wristRelease'].get(k) for k in
            ('screw','hole','seatedAt'))!=identity]
        history.append(deepcopy(item))
