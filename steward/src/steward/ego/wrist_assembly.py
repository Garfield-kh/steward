"""Wrist support-plane/hole refresh at hover; held underside remains an estimate."""
from copy import deepcopy
import math
import time
from steward.ego import action_budget as ab
import numpy as np
from steward.ego import wrist_observation as wo
from steward.ego import component_registration as cr
from steward.ego.assembly_edge_observation import body_patch, visible_rays
from steward.math3d import matrix_from_quat, quat_from_yaw


def measure(controller, sample, reference, current, diagnostics=None):
    raw,cam=sample;est=controller.rt.estimator
    diag=diagnostics if diagnostics is not None else raw.setdefault('wristDiagnostics',{})
    stamp=raw['header']['stamp']
    if stamp<=reference.get('ownReleaseAt',-1): return None
    nominal=reference.get('landingTopM',reference['position'][2]+reference['size'][2]/2)
    target=np.array([*current['spot'],nominal])
    tracked=est.tracks.get(reference.get('entity'))
    pad=3*math.sqrt(float(np.max(tracked.var))) if wo.calibrated() and tracked is not None else 0.
    radius=min(.15,float(np.linalg.norm(reference['size'][:2]))/2)+pad
    region=wo.window(raw,cam,target,radius,diag.setdefault('window',{}))
    diag['beliefSearchPaddingM']=pad
    points,y,x=wo.cloud(raw,cam,region)
    diag.update(windowPixels=int(region.sum()),validPoints=len(points))
    if len(points)<48:
        diag["reason"]="insufficient_window_depth";return None
    boxes=[];held=est.tracks.get(est.held_id);held_box=None
    if held is not None:
        header=raw['header'];g=est.gripper_position(est.held_gripper,header['joints'],header['odometry'])
        r=est.gripper_rotation(est.held_gripper,header['joints'],header['odometry'])
        interface=held.held_interface
        if not interface:
            diag['reason']='missing_held_interface';return None
        margin=.003+2*math.sqrt(float(np.max(held.var)))
        held_box=(g+r @ interface['offset'],np.asarray(interface['size'])/2+margin,
                      r @ np.asarray(interface['rotation']))
        boxes.append(held_box)
    with controller.rt._lock:
        previous=est.joints,est.odometry
        try:
            est.joints,est.odometry=raw['header']['joints'],raw['header']['odometry']
            boxes.extend(est.body_boxes())
        finally:
            est.joints,est.odometry=previous
    keep=visible_rays(points,np.asarray(cam['p']),boxes)
    diag["rayVisiblePoints"]=int(keep.sum())
    local=(points[:,:2]-reference['position'][:2]) @ cr.rotation(reference['yaw'])
    keep &= np.all(abs(local)<=np.asarray(reference['size'][:2])*.55+pad,axis=1)
    keep &= abs(points[:,2]-nominal)<.04
    # An unrelated outer support below the registered body is not the bearing face.
    table=controller.rt.ego_state().find(reference.get('support'))
    if table is not None:
        from steward.ego.planner import entity_box
        keep &= points[:,2]>entity_box(table)[2]+.004
    observed_holes=None
    p=points[keep]
    if len(p)>8192: p=p[np.linspace(0,len(p)-1,8192,dtype=int)]
    diag['beliefFilteredPoints']=int(keep.sum())
    plane_diag=diag.setdefault('plane',{})
    if wo.calibrated():
        from steward.ego.wrist_surface import fit
        selected=np.zeros(region.shape,bool);selected[y[keep],x[keep]]=True
        patch=fit(raw,cam,selected,plane_diag)
    else:
        patch=body_patch(p,reference,plane_diag)
    if patch is None:
        diag['reason']='target_not_in_view_or_occluded' if observed_holes==[] else plane_diag.get('reason');return None
    if observed_holes==[] and plane_diag.get('inlierFraction',0)<.75:
        diag['reason']='target_not_in_view_or_occluded';return None
    # Preserve range-clipped pixels inside the search window. They do not supply
    # fitted points; the contour still needs a complete colour rim and supported depth.
    missing=region & (raw['depth']==0) if wo.calibrated() else np.zeros(region.shape,bool)
    region[:]=False;region[y[keep],x[keep]]=True;region |= missing
    if wo.calibrated():
        point=np.asarray(patch['point']);normal=np.asarray(patch['normal'])
        anchor=np.asarray(reference['position'],float).copy()
        anchor[2]=point[2]-np.dot(anchor[:2]-point[:2],normal[:2])/normal[2]
        patch['point']=anchor.tolist()
    from steward.ego.support_holes import detect_region
    holes=observed_holes if observed_holes is not None else wo.holes(sample,target,diagnostics=diag.setdefault('holes',{}),region=region)
    holes=[h for h in holes if abs(np.dot(np.asarray(h['point'])-patch['point'],patch['normal']))<.003]
    return dict(patch=patch,holes=holes,stamp=stamp,nominal=nominal)


def hover(controller,job,current,params,report):
    """Record each attempted support refresh, including failed/no-frame attempts."""
    previous = report.get('_wristAssemblyDone')
    try:
        return _hover(controller,job,current,params,report)
    finally:
        row = report.get('wristPlacement')
        if not previous and row:
            from steward.ego.capability_usage import record
            record('wrist_surface', entity=current.get('object'), stamp=row.get('observedAt'),
                   measured=row.get('status')=='measured', action=bool(row.get('applied')),
                   reason=None if row.get('applied') else row.get('reason'))


def _hover(controller,job,current,params,report):
    if not wo.enabled(controller.rt,'ASSEMBLY') or report.get('_wristAssemblyDone'): return False
    report['_wristAssemblyDone']=True
    est=controller.rt.estimator
    record=est.stationary_assemblies.records.get(params.get('on'))
    if record is None: return False
    ref=record['reference'];start=time.monotonic()
    deadline=ab.make(controller.ex,ab.frame_timeout(controller.ex,6.,2),'two frame periods for wrist observation and replanning')
    row=report.setdefault('wristPlacement',dict(status='unknown',budgetS=6.,applied=False,
        heldHoleOffsetMeasured=False))
    nominal=ref.get('landingTopM',ref['position'][2]+ref['size'][2]/2)
    wo.context(controller,'placement', [*current['spot'],nominal],
               min(.15,float(np.linalg.norm(ref['size'][:2]))/2),ref)
    sample=wo.capture(controller,ab.make(controller.ex,ab.frame_timeout(controller.ex,4.),'single configured wrist frame period'))
    if sample: sample[0]['wristDiagnostics']=row.setdefault('diagnostics',{})
    row['capture']=dict(getattr(controller,'_wristCapture',{}))
    measured=measure(controller,sample,ref,current) if sample else None
    if measured is None:
        row.update(reason=row.get('diagnostics',{}).get('reason') or row['capture'].get('reason') or 'support_plane_occluded_or_unresolved',wallS=round(time.monotonic()-start,3));return False
    patch=measured['patch'];normal=np.asarray(patch['normal'])
    tilt=math.degrees(math.acos(float(np.clip(normal[2],-1,1))))
    surface=dict(observed=True,source='wrist_native_plane',patch=patch,normal=normal.tolist(),
        tiltDeg=tilt,levelFirst=tilt>5.,thresholdDeg=5.,uncertaintyM=max(patch['residualM'],1e-6),
        nominalHeightM=measured['nominal'],observedMinusAssumedM=patch['point'][2]-measured['nominal'],
        observedAt=measured['stamp'],checkedAt=measured['stamp'],referenceAt=ref['registeredAt'],entity=ref['entity'],
        body=ref['entity'],reason='fresh_wrist_bearing_plane')
    row.update(status='measured',targetPointM=patch['point'],surfaceHeightM=patch['point'][2],tiltDeg=round(tilt,3),
        heightCorrectionM=round(surface['observedMinusAssumedM'],6),holeCount=len(measured['holes']),
        source=surface['source'],observedAt=measured['stamp'],translationM=None)
    spec=getattr(controller,'_component_spec',None);registration=None
    if spec and spec['body']==ref['entity'] and spec['part']==current['object']:
        found=measured['holes']
        common=sample[0]['header']['odometry'].get('sigma',{}).get('pos',0.)
        found=[dict(h,relUncM=max(h['uncertaintyM']-common,0.)) for h in found]
        fit,reason=cr.fit_layout(spec['local'],found,spec['expected'],spec['yaw'],.003)
        row['holeFitReason']=reason
        if fit:
            odom=sample[0]['header']['odometry']
            registration=dict(fit,local=spec['local'],holePairs=spec['pairs'],part=current['object'],body=ref['entity'],
                observedAt=measured['stamp'],referenceAt=ref['registeredAt'],nominalPosition=spec['spot'],nominalYaw=spec['yaw'],
                baseOdometry=deepcopy(odom),basePositionM=((np.asarray(fit['position'])-[odom['x'],odom['y']]) @ cr.rotation(odom['yaw'])).tolist(),
                baseYawRad=fit['yaw']-odom['yaw'],source='wrist_support_holes',frame='base_at_observation')
            registration.pop('points',None);registration.pop('indices',None)
            row['translationM']=(np.asarray(fit['position'])-current['spot']).round(6).tolist()
    if ab.expired(deadline):
        row['reason']='observation_budget';return False
    old_surface=deepcopy(ref.get('placementSurface'));old_reg=deepcopy(ref.get('componentHoleRegistration'))
    with controller.rt._lock:
        ref['placementSurface']=surface
        if registration: ref['componentHoleRegistration']=registration
        est.version+=1
    receipt=controller.rt.predict('place',dict(params,closeRelease=True))
    if not receipt.get('withinRiskLimit'):
        with controller.rt._lock:
            for key,old in [('placementSurface',old_surface),('componentHoleRegistration',old_reg)]:
                if old is None: ref.pop(key,None)
                else: ref[key]=old
            est.version+=1
        row.update(reason='corrected_plan_not_admitted',wallS=round(time.monotonic()-start,3));return False
    replacement=controller.rt._consume_receipt(receipt);replacement['precision']=report
    row.update(applied=True,reason='support_corrected_held_transform_still_inferred',wallS=round(time.monotonic()-start,3))
    report['corrections'].append(dict(source='wrist_support_plane_and_holes',executed=True,receiptId=receipt['id']))
    deadline=getattr(controller.ex,'_primitive_deadline',None)
    controller.ex._begin(job,'place',receipt)
    if deadline is not None: controller.ex._primitive_deadline=ab.limited(deadline,controller.ex._primitive_deadline)
    # Preserve the existing admitted path and independent post-release support check.
    try:
        controller.ex._place(job,replacement)
    finally:
        current.update(replacement)
    return True


def post_release(controller,job,plan,params):
    if not wo.enabled(controller.rt,'ASSEMBLY_RELEASE'): return
    report=plan.get('precision') or {};pre=report.get('wristPlacement')
    if not pre or not pre.get('applied') or not plan.get('releaseExecuted'): return
    wo.context(controller,'component_release',pre.get('targetPointM'),None)
    start=time.monotonic();sample=wo.capture(controller,ab.make(controller.ex,ab.frame_timeout(controller.ex,4.),'single configured wrist frame period'))
    row=dict(status='unknown',passed=None,reason='component_and_support_not_separately_resolved',budgetS=4.)
    if sample:
        row.update(observedAt=sample[0]['header']['stamp'],camera=wo.CAMERA)
        # Require independent component/body segmentation; a plane alone must not
        # be promoted to a landed-component observation. Existing landing checks
        # remain authoritative until a separate outline is available.
        row['reason']='wrist_frame_captured_independent_outline_required'
    row['wallS']=round(time.monotonic()-start,3)
    pre['postRelease']=row
    job.postcondition['wristAssemblyRelease']=deepcopy(row)
