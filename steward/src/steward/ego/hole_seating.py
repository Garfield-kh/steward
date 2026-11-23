"""Post-release occupancy from native pixels, independent of entity segmentation.

A release reserves its hole even when two cleared views remain inconclusive.
All geometric scales come from the observed disk, mouth and calibrated robot.
"""
from copy import deepcopy
import math
import uuid
import numpy as np
from steward.ego import wrist_observation as wo, entry_layout as el, action_budget as ab

SOURCE = 'post_release_hole_rgbd'


def unknown(reason, **evidence):
    return dict(status='unknown', passed=None, landingState='unclear', source=SOURCE,
                reason=reason, **evidence)


def bearing(sample, cap, opening):
    """Fit support outside the cap; require returns around it, not one edge."""
    from steward.ego.wrist_surface import fit
    raw, cam = sample
    p,y,x = wo.cloud(raw,cam,wo.window(raw,cam,cap['point'],opening))
    r=cap['radiusM']; centre=np.asarray(cap['point'])
    radial=np.linalg.norm(p[:,:2]-centre[:2],axis=1)
    ring=(radial>r+cap['uncertaintyM']) & (radial<3*r) & (p[:,2]<centre[2])
    mask=np.zeros(raw['depth'].shape,bool);mask[y[ring],x[ring]]=True
    plane=fit(raw,cam,mask)
    if plane is None:return None
    q=p[ring];n=np.asarray(plane['normal'])
    good=abs((q-plane['point']) @ n)<=max(plane['residualM'],cap['uncertaintyM'])
    q=q[good]
    angles=np.arctan2(q[:,1]-centre[1],q[:,0]-centre[0])
    if len(q)<48 or len(set(np.floor((angles+math.pi)/(math.pi/2)).astype(int)))<3:return None
    plane['surroundingPoints']=len(q)
    return plane


def measure(controller, sample, target, opening, registration):
    """Three states; a hidden mouth inferred from layout is never an empty hole."""
    from steward.ego.driver_view import measure as disks
    from steward.ego.layout_pose import match
    raw,cam=sample;stamp=raw['header']['stamp'];diag={}
    _,caps=disks(controller,sample,target,opening,diag)
    region=wo.window(raw,cam,target,opening)
    for row in registration.get('layout',[]):
        region |= wo.window(raw,cam,row['point'],opening/2)
    found=wo.holes(sample,target,opening/2,region=region)
    arcs,_=el.surface_arcs(sample,target,opening,{})
    for h in arcs:
        if not any(np.linalg.norm(np.asarray(h['point'])-q['point'])<h['radiusM'] for q in found):found.append(h)
    # Occupancy is the question, so do not force a released hole to match a head.
    rows=registration.get('layout',[])
    # A directly visible recess below a separate raised object disproves seating.
    # Do not let the two projected centres invalidate all layout anchors.
    usable_caps=[h for h in caps if not any(
        np.linalg.norm(np.asarray(h['point'])[:2]-np.asarray(q['point'])[:2]) <
        q['radiusM']+h['uncertaintyM'] and h['point'][2]>q['point'][2] for q in found)]
    features=[dict(h,observedKind=k,kind='recess') for hs,k in ((found,'recess'),(usable_caps,'head')) for h in hs]
    fitted,reason=match(rows,features)
    if fitted is None and found:
        # The independent empty-hole anchors can still identify an exposed hole;
        # unrelated loose objects must not erase that observation.
        features=[dict(h,observedKind='recess',kind='recess') for h in found]
        fitted,reason=match(rows,features)
    evidence=dict(observedAt=stamp,heads=caps,emptyHoles=found,correspondence=fitted,
                  headCount=len(caps),emptyCount=len(found))
    if fitted is None:return unknown(reason,**evidence)
    hit=next((m for m in fitted['matches'] if m['hole']==registration.get('targetHole')),None)
    if hit is None:return unknown('target_not_directly_visible',**evidence)
    feature=features[hit['feature']]
    evidence['targetFeature']=feature
    if feature['observedKind']=='recess':
        return dict(evidence,status='failed',passed=False,source=SOURCE,
                    reason='target_hole_visibly_empty',landingState='empty_hole')
    if registration.get('targetOccupiedBeforeRelease'):
        return dict(evidence,status='failed',passed=False,source=SOURCE,
                    reason='target_already_occupied_before_release',landingState='previously_occupied')
    plane=bearing(sample,feature,opening)
    if plane is None:return unknown('head_visible_support_unresolved',**evidence)
    n=np.asarray(plane['normal']);height=float((np.asarray(feature['point'])-plane['point']) @ n)
    noise=max(feature['uncertaintyM'],plane['residualM'])
    # A low disk supported at the mouth, versus a full shaft or stacked disks.
    # The uncertainty interval must fit wholly inside one measured cap radius.
    limit=feature['radiusM']
    evidence.update(headM=feature['point'],headMinusSurfaceZM=height,surface=plane,
                    heightLimitM=limit,uncertaintyM=noise,headDistanceXYM=fitted['residualM'],
                    targetObserved=True,attachmentObserved=False)
    tilt_span=float(np.linalg.norm(np.cross(n,feature['normal']))*limit)
    evidence['headPlaneMismatchM']=tilt_span
    if tilt_span>2*noise:
        return dict(evidence,status='failed',passed=False,source=SOURCE,
                    reason='head_not_parallel_to_bearing_surface',landingState='tilted_head')
    if height-noise>limit:
        return dict(evidence,status='failed',passed=False,source=SOURCE,
                    reason='head_above_seated_disk_envelope',landingState='raised_or_stacked')
    if height<=noise or height+noise>limit:
        return unknown('head_height_interval_ambiguous',**evidence)
    return dict(evidence,status='confirmed',passed=True,source=SOURCE,
                reason='head_at_target_on_bearing_surface',landingState='seated')


def clear_ray(controller, state, camera, target, radius):
    """Require the target disk's camera rays to miss the current robot boxes."""
    from steward.math3d import pose_compose
    from steward.ego.wrist_release_geometry import robot_boxes
    from steward.ego.support_holes import project
    link,sensor=controller.rt.estimator.cameras[camera]
    cam=pose_compose(controller.rt.estimator.link_pose(link,state['joints'],state['odometry']),sensor['mount'])
    points=np.asarray(target)+np.array([[0,0,0],[radius,0,0],[-radius,0,0],[0,radius,0],[0,-radius,0]])
    uv,z=project(points,cam,sensor['intrinsics']);intr=sensor['intrinsics']
    near,far=sensor.get('depthRangeM',[0.,math.inf])
    if not (np.isfinite(uv).all() and np.all((z>near)&(z<far)) and
            np.all(uv>=2) and np.all(uv<np.array([intr['width'],intr['height']])-2)):return False
    origin=np.asarray(cam['p'])
    for c,half,rotation in robot_boxes(controller,dict(header=state)):
        r=np.asarray(rotation);o=(origin-c) @ r
        for point in points:
            d=(point-origin) @ r
            parallel=abs(d)<np.finfo(float).eps
            if np.any(parallel & (abs(o)>half)):continue
            safe=np.where(parallel,1.,d)
            a=(-np.asarray(half)-o)/safe;b=(np.asarray(half)-o)/safe
            lo=np.where(parallel,-math.inf,np.minimum(a,b));hi=np.where(parallel,math.inf,np.maximum(a,b))
            if max(float(lo.max()),0.)<min(float(hi.min()),1.):return False
    return True


def view_pose(controller, plan, camera, target, opening, *, second=False):
    """Vertical retreat within the admitted exit, then a higher viewing angle.

    Reach and wrist orientation stay fixed until ordinary place withdrawal.
    This is one absolute movement per view, never a downward probe.
    """
    state=controller.ex._state();joints=dict(state['joints'])
    ceiling=max(joints['lift'],plan.get('approachLift',joints['lift']),
                (plan.get('seatEndpoint') or {}).get('safeLift',plan.get('safeLift',joints['lift'])))
    # Search the whole admitted interval. A jaw-width minimum used to reject
    # short exits before checking even one camera ray. At the ceiling a fresh
    # first view is still useful; a retry must actually change the viewpoint.
    span=ceiling-joints['lift']
    preferred=joints['lift']+min(opening/2,span/2)
    heights=(list(np.linspace(preferred,ceiling,9))+
             list(np.linspace(joints['lift'],preferred,5)[1:-1])) if span>0 else []
    if not second:heights.append(joints['lift'])
    for height in heights:
        future=dict(joints,lift=float(height));candidate=dict(state,joints=future)
        # Include the bearing ring, not just the projected head centre.
        if not clear_ray(controller,candidate,camera,target,opening/4):continue
        path=[dict(joints,lift=float(h)) for h in np.linspace(joints['lift'],height,14)]
        if not controller.rt.esim.arm.reachable_path(path):continue
        moved=height>joints['lift']
        if moved:controller.ex._set_joint('lift',float(height))
        actual=controller.ex._state()
        if second and actual['joints']['lift']<=joints['lift']:
            return dict(executed=bool(moved),armClear=False,reason='actual_view_unchanged',
                        beforeLift=joints['lift'],afterLift=actual['joints']['lift'])
        clear=clear_ray(controller,actual,camera,target,opening/4)
        return dict(executed=bool(moved),armClear=clear,reason=None if clear else 'actual_view_obstructed',
                    beforeLift=joints['lift'],afterLift=actual['joints']['lift'],secondView=second,
                    admittedCeiling=ceiling, bearingRingChecked=True)
    return dict(executed=False,reason='no_distinct_clear_view' if second else 'no_reachable_clear_view',
                beforeLift=joints['lift'],admittedCeiling=ceiling)


def capture(controller, plan):
    """At release, withdraw vertically before sampling; retry one higher view."""
    from steward.ego import released_object
    if released_object.enabled() and plan.get('releasedObjectContext'):
        result = released_object.capture(controller, plan)
        if result is not None: return result
    from steward.ego.tip_entry_servo import camera_for
    from steward.ego.skills import SkillFailed
    camera=camera_for(controller.rt)
    if camera is None:return None
    row=dict(schema='hole_seating.v1',id=uuid.uuid4().hex,frames=[],motions=[],camera=camera,
             releasedAt=plan.get('releasedAt'),check=unknown('view_not_acquired'),
             releaseKey=[plan.get('seatBody'),plan.get('seatHole'),plan.get('object'),plan.get('releasedAt')])
    plan['_holeSeatReport']=row
    try:
        target=(plan.get('seatTarget') or {}).get('point')
        if target is None:return row
        opening=controller.rt.esim.arm.geometry.jaw_opening
        reg=el.prior(controller,plan)
        occupied = reg.get('targetHole') in reg.get('occupiedHoles', [])
        from steward.ego import assembly_outcome
        if assembly_outcome.enabled():
            # The just-recorded release is not pre-release occupancy evidence.
            occupied = any(a.get('hole') == reg.get('targetHole')
                and a.get('body') == plan.get('seatBody') and a.get('seatConfirmed')
                and (a.get('seatCheck') or {}).get('observedAt', float('inf')) < plan.get('releasedAt', -1)
                for a in getattr(controller.rt.estimator, 'fastening_attempts', {}).values())
        for f in (plan.get('tipEntryServo') or {}).get('frames', []):
            entry = f.get('entry')
            if not entry:
                continue
            occupied |= any(np.linalg.norm(np.asarray(h['point'])[:2]-np.asarray(entry['point'])[:2]) <
                h['radiusM']+entry.get('radiusM', 0.) for h in f.get('diagnostics', {}).get('heads', []))
        reg['targetOccupiedBeforeRelease'] = bool(occupied)
        row['registration']=deepcopy(reg)
        for index in range(2):
            controller.check()
            motion=view_pose(controller,plan,camera,target,opening,second=bool(index))
            row['motions'].append(motion)
            if not motion.get('armClear'):
                row['viewStopReason']=motion['reason']
                if not row['frames']:row['check']=unknown(motion['reason'])
                break
            after=max(plan.get('releasedAt',-math.inf),controller.ex._state()['stamp'])
            # Wait one configured sensor period after the measured clearance.
            # A cached frame at the end of the lift is not a second observation.
            controller.ex._wait(ab.frame_window(controller.ex))
            wo.context(controller,'hole_seating',target,opening/2)
            controller._wristContext['entryRegistration']=deepcopy(reg)
            sample=wo.capture(controller,ab.make(controller.ex,ab.frame_timeout(controller.ex,4.),
                              'one cleared hole view'),camera=camera,check_view=False)
            f=dict(index=index,capture=deepcopy(getattr(controller,'_wristCapture',{})),requestedAfter=after)
            row['frames'].append(f)
            check=unknown('frame_unavailable')
            if sample is not None:
                stamp=sample[0]['header']['stamp']
                f.update(stamp=stamp,cameraPose={k:np.asarray(v).tolist() for k,v in sample[1].items()})
                if stamp>after:
                    check=measure(controller,sample,target,opening,reg)
                else:check=unknown('frame_precedes_clearance',observedAt=stamp)
            f['check']=check;row['check']=check
            if check['passed'] is not None:break
    except SkillFailed as exc:
        controller.check();row['check']=unknown('clearance_motion_failed',detail=str(exc))
    except (AttributeError,KeyError,TypeError,ValueError,np.linalg.LinAlgError) as exc:
        row['check']=unknown('view_geometry_unavailable',detail=type(exc).__name__)
    finally:
        from steward.ego.capability_usage import record
        record('hole_seating',event_id=row['id'],stage='after_release',
               measured=row['check']['passed'] is not None,action=bool(row['motions']),
               reason=row['check']['reason'],evidence=row)
    return row


def apply(attempt, plan):
    row=(plan or {}).get('_holeSeatReport')
    if not row:return False
    attempt['holeSeatObservation']=deepcopy(row)
    check=deepcopy(row['check'])
    from steward.ego.grasp_binding import enabled as binding_enabled
    attempt.update(seatCheck=check,seatConfirmed=check['passed'] is True and not (binding_enabled() and attempt.get('bindingUnconfirmed') and not attempt.get('transportCommitted')),seatStatus=check['status'],
                   seatDeferred=check['passed'] is None)
    return True


def continuable(attempt):
    check=attempt.get('seatCheck') or {}
    return bool(attempt.get('releaseExecuted') and check.get('passed') is not False
                and check.get('status') != 'failed' and attempt.get('seatStatus') != 'failed'
                and not attempt.get('departureEvidence'))


def deferred(attempt):
    return continuable(attempt) and (attempt.get('seatCheck') or {}).get('passed') is None


def confirmed(attempt):
    from steward.ego.grasp_binding import enabled as binding_enabled
    if binding_enabled() and (attempt or {}).get('bindingUnconfirmed') and not (attempt or {}).get('transportCommitted'): return False
    return bool(attempt and attempt.get('seatConfirmed') and
                (attempt.get('seatCheck') or {}).get('passed') is True and
                continuable(attempt))


def transported(attempt):
    """Motion invalidates current-view freshness, not historical completion."""
    if confirmed(attempt):
        attempt['seatCheck'] = dict(attempt['seatCheck'], retainedObservation=True,
                                    currentPoseConfirmed=False)
        return
    # A move is not evidence that a known failed landing became successful.
    if (attempt.get('seatCheck') or {}).get('passed') is False or attempt.get('departureEvidence'):
        return
    attempt.update(seatConfirmed=False, seatDeferred=True, seatStatus='unknown',
                   seatCheck=dict(status='unknown', passed=None, reason='body_moved_reobserve_landing'))
