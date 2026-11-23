"""Measured empty entrances take precedence over inferred layout projections.

Partial arcs are admitted only as a resolved pair on the same observed face.
Neither a raised head nor a remembered support point supplies their centres.
"""
import numpy as np
from steward.ego import capabilities, current_entry as ce


def enabled():
    return capabilities.get('STEWARD_ASSEMBLY_VISIBLE_ENTRY') != '0'


def recover(sample, target, opening, registration, diagnostics):
    rows = registration.get('layout', [])
    if len(rows) != 2 or registration.get('targetKind', 'recess') != 'recess':
        return None
    identity = next((i for i,r in enumerate(rows) if r.get('bodyHole') == registration.get('targetHole')), None)
    if identity is None: return None
    prior_axis = np.asarray(rows[1]['point'])[:2]-rows[0]['point'][:2]
    choices = []
    for holes, outline, plane in ce.faces(sample,target,opening,registration,diagnostics,visible_pair=True):
        if len(holes) != 2: continue
        a,b = holes
        resolution = a['uncertaintyM']+b['uncertaintyM']
        axis = np.asarray(b['point'])[:2]-a['point'][:2]
        span = np.linalg.norm(axis)
        # Independent curvature fits must resolve two disjoint equal rims.
        # This does not require the old spacing or support yaw to be correct.
        if span <= a['radiusM']+b['radiusM']+resolution: continue
        if abs(a['radiusM']-b['radiusM']) > a['planeNoiseM']+b['planeNoiseM']: continue
        if abs(axis @ prior_axis) <= resolution*np.linalg.norm(prior_axis): continue
        ordered = holes if axis @ prior_axis > 0 else holes[::-1]
        entry = dict(ordered[identity],targetHole=registration['targetHole'],targetFromLayout=False,
                     source='same_frame_visible_entrance_pair')
        choices.append((entry,ordered))
    if not choices:
        diagnostics['reason']='visible_entrance_pair_unresolved';return None
    unique = ce.unique([c[0] for c in choices])
    if len(unique) != 1:
        diagnostics['reason']='multiple_visible_entrance_pairs';return None
    entry, pair = min(choices,key=lambda c:c[0]['uncertaintyM'])
    diagnostics.update(reason=None,visibleEntrances=pair,identityBasis='observed_pair_order',
                       priorUsedOnlyForAxisSign=True)
    return entry


def remember(controller, current, frame, target):
    """Remember a visible target, distinguishing measured and estimated tips."""
    entry = frame.get('entry') or {}
    if entry.get('source') != 'same_frame_visible_entrance_pair': return
    path = current.get('seatEndpoint', current)
    previous = current.get('_visibleEntrancePlan')
    point = np.asarray(entry['point'])
    if frame.get('bothVisible'):
        desired = controller.ex._gripper_xy()+np.asarray(frame['deltaM'][:2])
        basis = 'same_frame_entry_minus_lower_end'
        opening=controller.rt.esim.arm.geometry.jaw_opening
        step_limit=min(opening/8,max(0.,(-float(frame['deltaM'][2])-frame['uncertaintyM'])/2))
    elif previous is not None:
        desired = np.asarray(previous['desiredGripXY'])+point[:2]-np.asarray(previous['point'])[:2]
        basis = previous['basis']
        step_limit=previous.get('stepLimitM')
    elif path.get('gripSpot') is not None:
        desired = np.asarray(path['gripSpot'])+point[:2]-np.asarray(target)[:2]
        basis = 'visible_entry_plus_planned_held_offset_unconfirmed'
        step_limit=None
    else:
        return
    current['_visibleEntrancePlan'] = dict(point=point.tolist(),desiredGripXY=desired.tolist(),
        uncertaintyM=frame.get('uncertaintyM', entry['uncertaintyM']),basis=basis,
        observedAt=frame.get('stamp'),stepLimitM=step_limit)
    current['seatTarget'] = dict(current.get('seatTarget') or {}, **entry)


def correction_budget(frame, opening, step_limit):
    """Enough measured short steps for this residual; noise bounds the work."""
    import math
    uncertainty=max(float(frame['uncertaintyM']),np.finfo(float).eps)
    needed=math.ceil(max(0.,frame['residualM']-uncertainty)/step_limit)+1
    resolution_bound=max(2,math.ceil(opening/uncertainty))
    return min(max(2,needed),resolution_bound)


def finish_estimate(controller, current, saved, row):
    """After the one view change, complete at the best observed target.

    This is explicitly estimated motion when the lower end cannot be seen.
    Every short step is followed by another frame from the SAME selected camera.
    Missing observations do not revert to the superseded support projection.
    """
    plan=current.get('_visibleEntrancePlan')
    if not plan: return saved
    from steward.ego import tip_entry_servo as servo
    import math
    opening=controller.rt.esim.arm.geometry.jaw_opening
    distance=np.linalg.norm(np.asarray(plan['desiredGripXY'])-controller.ex._gripper_xy())
    step_limit=plan.get('stepLimitM')
    step_limit=opening/8 if step_limit is None else step_limit
    count=(min(math.ceil(distance/step_limit)+1,
        max(2,math.ceil(opening/max(plan['uncertaintyM'],np.finfo(float).eps)))) if step_limit>0 else 0)
    records=row.setdefault('visibleTargetFallbackSteps',[])
    camera=row.get('camera')
    try:
        current['_entryCameraOverride']=camera
        for _ in range(count):
            controller.check()
            plan=current['_visibleEntrancePlan']
            delta=np.asarray(plan['desiredGripXY'])-controller.ex._gripper_xy()
            norm=float(np.linalg.norm(delta));uncertainty=plan['uncertaintyM']
            if norm<=uncertainty:break
            latest=plan.get('stepLimitM')
            limit=opening/8 if latest is None else latest
            if limit<=0:break
            step=delta*min(1.,limit/norm)
            motion=servo.move(controller,step,uncertainty)
            motion.update(source=plan['basis'],unconfirmed=True,stepLimitM=limit)
            records.append(motion)
            if not motion['executed']:break
            servo.run(controller,current,{},phase='descent',correct=False)
    finally:
        current.pop('_entryCameraOverride',None)
    # Complete at the best reached correction even if the final frame is lost.
    saved=dict(saved,gripSpot=controller.ex._gripper_xy().tolist())
    row['fallbackTargetSource']='visible_entrance_best_estimate_unconfirmed'
    return saved
