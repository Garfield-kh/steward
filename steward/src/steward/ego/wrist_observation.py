"""On-demand, synchronized robot RGB-D; no scene or simulator-state access."""
from steward.ego import capabilities
import os
import time
from steward.ego import action_budget as ab
import math
import json
import numpy as np
from steward.net import HttpError
from steward.math3d import pose_compose, matrix_from_quat
from steward.ego.perception import decode_rgbd
from steward.ego.support_holes import unproject, project, detect_region, components

CAMERA = 'wrist_camera'


def calibrated():
    return (capabilities.get('STEWARD_WRIST_METRIC') != '0'
            and capabilities.get('STEWARD_WRIST_CAMERA') != '0')



def release_reporting():
    return (capabilities.get('STEWARD_WRIST_RELEASE_REPORT') != '0'
            and capabilities.get('STEWARD_WRIST_CAMERA') != '0')


def enabled(rt, behavior):
    return (capabilities.get('STEWARD_WRIST_CAMERA') != '0'
        and capabilities.get('STEWARD_WRIST_'+behavior) != '0'
        and CAMERA in getattr(rt.estimator, 'cameras', {})
        and CAMERA in getattr(rt, '_provided_camera_ids', ())
        and callable(getattr(getattr(rt, 'link', None), 'rgbd', None)))


def context(controller, phase, target=None, radius=None, reference=None):
    """Record search priors separately from measurements (dump remains opt-in)."""
    row=dict(phase=phase,target=target,radiusM=radius)
    if capabilities.get('STEWARD_WRIST_DUMP_DIR'):
        est=controller.rt.estimator
        track=getattr(est,'tracks',{}).get(getattr(est,'held_id',None))
        row.update(reference={k:reference[k] for k in ('entity','position','size','yaw','landingTopM','ownReleaseAt','support','registeredAt') if k in reference} if reference else None,
            heldInterface=getattr(track,'held_interface',None),
            heldVariance=getattr(track,'var',None),
            toolCategory=getattr(track,'category',None),
            heldGripper=getattr(est,'held_gripper',None),heldSince=getattr(est,'held_since',None))
        if reference:
            tracked=getattr(est,'tracks',{}).get(reference.get('entity'))
            row['referenceVariance']=getattr(tracked,'var',None)
            try:
                support=controller.rt.ego_state().find(reference.get('support'))
                if support is not None:
                    from steward.ego.planner import entity_box
                    row['supportTopM']=entity_box(support)[2]
            except (AttributeError,KeyError,TypeError,ValueError):
                row['supportDumpReason']='unavailable'
    controller._wristContext=row


def view_gate():
    return (capabilities.get('STEWARD_WRIST_VIEW_GATE') != '0'
            and capabilities.get('STEWARD_WRIST_CAMERA') != '0')


def bounded_deadline(controller, deadline):
    from steward.ego.fasten_sequence import remaining
    return ab.limited(deadline, getattr(controller.ex,'_primitive_deadline',math.inf),
               getattr(controller.ex,'_job_deadline',math.inf),time.monotonic()+remaining(math.inf))


def target_in_view(controller, state, sensor, cam, row):
    ctx=getattr(controller,'_wristContext',{}) or {}
    if not ctx: return True  # low-level standalone capture, no task target supplied
    target=ctx.get('target')
    if target is None and ctx.get('phase')=='posture':
        est=controller.rt.estimator;track=est.tracks.get(est.held_id)
        feature=getattr(track,'held_tool_feature',None) or getattr(track,'held_grasp_geometry',None)
        if feature:
            grip=est.gripper_position(est.held_gripper,state['joints'],state['odometry'])
            rotation=est.gripper_rotation(est.held_gripper,state['joints'],state['odometry'])
            target=grip+rotation @ ((np.asarray(feature['tip'])+feature['head'])/2)
    if target is None:
        row.update(reason='target_estimate_unavailable',status='skipped');return False
    intr=sensor['intrinsics'];uv,z=project([target],cam,intr)
    near,far=sensor.get('depthRangeM',[0.,math.inf])
    row.update(targetPixel=uv[0].tolist(),opticalDepthM=float(z[0]),phase=ctx['phase'])
    inside=bool(np.isfinite(uv).all() and near<z[0]<far and
                np.all(uv[0]>=2) and np.all(uv[0]<np.array([intr['width'],intr['height']])-2))
    if not inside: row.update(reason='target_outside_wrist_frustum',status='skipped')
    return inside


def capture(controller, deadline, *, camera=CAMERA, check_view=True):
    """One native frame, with FK evaluated at its own encoder/odometry timestamp."""
    rt=controller.rt; est=rt.estimator
    row=dict(status='unknown',reason='frame_unavailable',camera=camera,requested=False,frameCaptured=False)
    controller._wristCapture=row
    started_wall=time.monotonic(); started_sim=None
    try:
        started_sim=controller.ex._state()['stamp']
        deadline=bounded_deadline(controller,deadline)
        left=ab.left(deadline)
        if left<=0:
            row['reason']='existing_deadline_exhausted';return None
        if check_view and view_gate():
            state=controller.ex._state();link,sensor=est.cameras[camera]
            cam=pose_compose(est.link_pose(link,state['joints'],state['odometry']),sensor['mount'])
            if not target_in_view(controller,state,sensor,cam,row): return None
        row['requested']=True
        raw=decode_rgbd(rt.link.rgbd(camera, timeout=min(2.,left)))
        row['frameCaptured']=True
        row['acquisitionWallS']=time.monotonic()-started_wall
        row['acquisitionSimS']=max(0.,controller.ex._state()['stamp']-started_sim)
        if raw.get('camera') is not None and raw['camera'] != camera:
            row['reason']='camera_mismatch';return None
        header=raw['header']; state=controller.ex._state()
        if not 0 <= state['stamp']-header['stamp'] <= .25:
            row['reason']='frame_timestamp_not_synchronized';return None
        noise=header['odometry'].get('sigma',{})
        bounds={k:max(.001,min(.02,3*math.sqrt(2)*noise.get('yaw' if k=='yaw' else 'pos',0.)))
                for k in ('x','y','yaw')}
        if (any(abs(state['joints'].get(k,v)-v)>.001 for k,v in header['joints'].items()) or
            any(abs(state['odometry'][k]-header['odometry'][k])>bounds[k] for k in bounds)):
            row['reason']='robot_moved_during_capture';return None
        link,sensor=est.cameras[camera]
        cam=pose_compose(est.link_pose(link,header['joints'],header['odometry']),sensor['mount'])
        from steward.ego import component_measurement_admission
        declared = sensor.get('depthRangeM')
        near,far = (declared or [0., math.inf]) if component_measurement_admission.enabled() else sensor.get('depthRangeM',[.07,.5])
        raw['depthRangeM'] = declared if component_measurement_admission.enabled() else [near,far]
        if (getattr(controller, '_wristContext', {}) or {}).get('phase') == 'supply_pregrasp':
            raw['_supply_pregrasp_sensor_depth'] = raw['depth'].copy()
        raw['depth']=np.where(np.isfinite(raw['depth']) & (raw['depth']>0) &
                              (raw['depth']>=near)&(raw['depth']<=far),raw['depth'],0)
        if ab.expired(deadline):
            row['reason']='existing_deadline_exhausted';return None
        context=dict(getattr(controller,'_wristContext',{}) or {},depthRangeM=raw['depthRangeM'],camera=camera,
            acquisitionWallS=row['acquisitionWallS'],acquisitionSimS=row['acquisitionSimS'])
        if capabilities.get('STEWARD_WRIST_DUMP_DIR'):
            try:
                with rt._lock:
                    previous=est.joints,est.odometry
                    try:
                        est.joints,est.odometry=header['joints'],header['odometry']
                        context['bodyBoxes']=list(est.body_boxes())
                        context['gripperPosition']=est.gripper_position(est.held_gripper,header['joints'],header['odometry'])
                        context['gripperRotation']=est.gripper_rotation(est.held_gripper,header['joints'],header['odometry'])
                    finally:
                        est.joints,est.odometry=previous
            except (AttributeError,KeyError,TypeError,ValueError):
                context['kinematicsDumpReason']='unavailable'
        raw['_archiveContext'] = context
        _dump(raw,cam,header,context)
        row.update(status="captured",reason=None,observedAt=header["stamp"])
        if capabilities.get('STEWARD_ASSEMBLY_RELEASED_OBJECT') != '0':
            identity = getattr(est,'held_id',None)
            if identity:
                if getattr(controller,'_releasedObjectSampleIdentity',None) != identity:
                    controller._releasedObjectHeldSamples = []
                controller._releasedObjectSampleIdentity = identity
                controller._releasedObjectHeldSamples = [*getattr(controller,'_releasedObjectHeldSamples',[]), (raw,cam)][-4:]
        return raw,cam
    except (TimeoutError, OSError, ValueError, KeyError, AttributeError, TypeError, HttpError) as exc:
        row['reason']='capture_error:'+type(exc).__name__
        return None
    finally:
        row['wallS']=time.monotonic()-started_wall
        if started_sim is not None:
            try: row['simS']=max(0.,controller.ex._state()['stamp']-started_sim)
            except (AttributeError,KeyError,TypeError): pass


def _dump(raw, cam, header, context=None):
    """Diagnostic recording of the robot's own frames; off unless a directory is given."""
    root=capabilities.get('STEWARD_WRIST_DUMP_DIR')
    if not root:
        return
    try:
        os.makedirs(root,exist_ok=True)
        # Encoding only: the same members/dtypes/metadata remain readable by
        # np.load. Keep historical compression unless explicitly disabled.
        save = np.savez if capabilities.get('STEWARD_WRIST_DUMP_COMPRESS') == '0' else np.savez_compressed
        suffix = ('_'+''.join(c for c in raw['camera'] if c.isalnum() or c in '_-')
                  if (context or {}).get('phase') == 'stationary_entrance_layers' else '')
        if (context or {}).get('phase') == 'supply_pregrasp':
            suffix = '_supply_pregrasp_'+''.join(
                c for c in str((context or {}).get('entity', 'capture'))+'_'+str(raw.get('camera', ''))
                if c.isalnum() or c in '_-')
        archive_path = os.path.join(root,('%d_%.3f'%(os.getpid(),header['stamp']))+suffix+'.npz')
        save(archive_path,
            rgb=raw['rgb'],depth=(raw.get('_supply_pregrasp_sensor_depth', raw['depth'])
                if (context or {}).get('phase') == 'supply_pregrasp' else
                raw['depth'] if (context or {}).get('measureCall') else raw['depth'].astype(np.float32)),p=np.asarray(cam['p'],float),
            q=np.asarray(cam['q'],float),fovy=float(raw['fovy']),stamp=float(header['stamp']),
            context_json=json.dumps(context or {},default=lambda a:np.asarray(a).tolist()),
            header_json=json.dumps(header,default=lambda a:np.asarray(a).tolist()))
        if (context or {}).get('phase') == 'supply_pregrasp':
            return archive_path
    except Exception:
        pass


def window(raw, cam, centre, radius, diagnostics=None):
    """Calibrated image window around a belief; the belief is never a measurement."""
    h,w=raw['depth'].shape; intr=dict(width=w,height=h,fovy=raw['fovy'])
    uv,z=project([centre],cam,intr)
    region=np.zeros((h,w),bool)
    if z[0]<=.02 or not np.isfinite(uv).all():
        return region
    f=h/(2*math.tan(math.radians(raw['fovy'])/2))
    r=int(math.ceil(f*radius/z[0]))
    if not calibrated(): r=min(180,r)
    if diagnostics is not None:
        diagnostics.update(centrePixel=uv[0].tolist(), opticalDepthM=float(z[0]), radiusPixels=r,
                           focalPixels=f, pixelM=float(z[0]/f))
    x,y=np.rint(uv[0]).astype(int)
    if x+r<0 or y+r<0 or x-r>=w or y-r>=h:
        return region
    region[max(0,y-r):min(h,y+r+1),max(0,x-r):min(w,x+r+1)]=True
    return region


def cloud(raw, cam, region):
    y,x=np.where(region & (raw['depth']>0))
    intr=dict(width=raw['depth'].shape[1],height=raw['depth'].shape[0],fovy=raw['fovy'])
    return unproject(np.c_[x,y],raw['depth'][y,x],cam,intr),y,x


def centre_axis(points, feature):
    """Fit shaft cross-section curvature instead of mistaking the near skin for its axis."""
    p=np.asarray(points);axis=np.asarray(feature['head'])-feature['tip'];axis/=np.linalg.norm(axis)
    origin=np.mean(p,axis=0)
    for iteration in range(3):
        basis=np.cross(axis,[1,0,0] if abs(axis[0])<.8 else [0,1,0]);basis/=np.linalg.norm(basis)
        basis=np.c_[basis,np.cross(axis,basis)]
        along=(p-origin) @ axis;uv=(p-origin) @ basis
        lo,hi=np.quantile(along,[.01,.99]);span=hi-lo;centres=[];errors=[];positions=[];radii=[]
        for a,b in ((.08,.24),(.24,.40),(.40,.56),(.56,.72)):
            selected=(along>=lo+a*span)&(along<=lo+b*span);q=uv[selected]
            if len(q)<12: continue
            design=np.c_[2*q,np.ones(len(q))]
            fit=np.linalg.lstsq(design,np.sum(q*q,axis=1),rcond=None)[0]
            radius2=fit[2]+fit[:2] @ fit[:2]
            if radius2<=0: continue
            radius=math.sqrt(radius2);residual=abs(np.linalg.norm(q-fit[:2],axis=1)-radius)
            angles=np.sort(np.arctan2(q[:,1]-fit[1],q[:,0]-fit[0]) % (2*math.pi))
            coverage=2*math.pi-np.max(np.diff(np.r_[angles,angles[0]+2*math.pi]))
            if (coverage<math.pi/3 or np.quantile(residual,.9)>.0015
                    or radius>np.linalg.norm(np.ptp(q,axis=0))): continue
            centres.append(fit[:2]);errors.append(float(np.quantile(residual,.9)));radii.append(radius)
            positions.append(origin+axis*np.median(along[selected])+basis @ fit[:2])
        if len(centres)<3: return None
        centre=np.median(centres,axis=0)
        if iteration<2:
            world=np.asarray(positions);_,_,axes=np.linalg.svd(world-world.mean(axis=0),full_matrices=False)
            axis=axes[0]*np.sign(axes[0] @ axis)
    if np.max(np.linalg.norm(np.asarray(centres)-centre,axis=1))>.002: return None
    result=dict(feature,tip=(origin+lo*axis+basis @ centre).tolist(),
                head=(origin+hi*axis+basis @ centre).tolist(),axis=axis.tolist(),lengthM=float(hi-lo))
    result['uncertaintyM']=max(.002,2*max(errors))
    result['shaftRadiusM']=float(np.median(radii))
    result['shaftRadiusUncertaintyM']=float(max(errors)+np.max(abs(np.asarray(radii)-np.median(radii))))
    result['axisCentreSource']='observed_shaft_curvature'
    return result


def tool(controller, sample, centre, category, radius, *, held=False, diagnostics=None):
    """Fit only a separate local shaft. Refuse ambiguous, cropped or robot pixels."""
    from steward.ego.fasten_geometry import directed_tool
    raw,cam=sample; est=controller.rt.estimator
    diag=diagnostics if diagnostics is not None else {}
    region=window(raw,cam,centre,radius)
    if held:
        # Native depth noise otherwise leaves speckles outside the FK body boxes
        # and joins them to the held surface. Denoise BEFORE both self exclusion
        # and curvature fitting, using only the immediate pixel neighbourhood.
        # Keep the recorded frame and missing returns untouched.
        depth=local_median(raw['depth'],region)
        depth=np.where(raw['depth']>0,depth,0)
        raw=dict(raw,depth=depth)
        diag['depthFilter']='local_median_3x3'
    points,y,x=cloud(raw,cam,region)
    diag["validPoints"]=len(points)
    if not len(points):
        diag["reason"]="no_local_depth"; return None
    keep=np.linalg.norm(points-centre,axis=1)<radius
    diag['proximityPoints']=int(keep.sum())
    # Exclude calibrated visual robot primitives, without the estimator's held-load mask.
    if held:
        with controller.rt._lock:
            old=est.joints,est.odometry
            try:
                est.joints,est.odometry=raw['header']['joints'],raw['header']['odometry']
                for c,half,r in est.body_boxes():
                    keep &= ~np.all(abs((points-c) @ r)<=np.asarray(half)+.001,axis=1)
            finally:
                est.joints,est.odometry=old
    diag['selfExcludedPoints']=diag['proximityPoints']-int(keep.sum())
    selected=np.zeros(region.shape,bool);selected[y[keep],x[keep]]=True
    groups=[]
    if raw.get('mask') is not None:
        groups=[selected & (raw['mask']==i) for i in np.unique(raw['mask'][selected]) if i]
    else:
        # Split foreground from a touching background silhouette at measured depth edges.
        depth=raw['depth'];edges=np.zeros(region.shape,bool)
        jump=(abs(depth[1:]-depth[:-1])>.006)&selected[1:]&selected[:-1]
        edges[1:] |= jump;edges[:-1] |= jump
        jump=(abs(depth[:,1:]-depth[:,:-1])>.006)&selected[:,1:]&selected[:,:-1]
        edges[:,1:] |= jump;edges[:,:-1] |= jump
        selected &= ~edges
        diag['depthEdgeExcludedPoints']=int(edges.sum())
        if held:
            # Require local pixel support for the surviving foreground. Never
            # fill a self-masked/missing pixel: that would bridge robot and load.
            supported=local_median(selected,region).astype(bool)
            diag['unsupportedPixels']=int((selected & ~supported).sum())
            selected &= supported
        # Native depth connected regions; no service latency or semantic label required.
        for pixels in components(selected):
            if len(pixels)<30: continue
            m=np.zeros(region.shape,bool);m[pixels[:,1],pixels[:,0]]=True;groups.append(m)
    found=[]; visible_candidates=0
    if category is None:
        diag.update(groups=len(groups),reason='missing_tool_category');return None
    for mask in groups:
        p,yy,xx=cloud(raw,cam,mask)
        detail = dict(pointCount=len(p), called=False,
                      pixelBounds=[int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())])
        diag.setdefault('endpointCandidates', []).append(detail)
        if len(p)<30 or xx.min()==0 or yy.min()==0 or xx.max()==region.shape[1]-1 or yy.max()==region.shape[0]-1:
            detail['reason'] = 'small_or_border_region'; continue
        if np.max(np.linalg.norm(p-centre,axis=1)) >= radius*.98:
            detail['reason'] = 'region_exceeds_search_radius'; continue
        feature=directed_tool(p,category,raw['header']['stamp'],detail)
        # A separate elongated component with usable endpoints is visible even
        # when their widths cannot establish polarity. This is a shape-based
        # association near the held TCP, never an identity claim.
        if detail.get('reason') in ('observed','endpoint_width_ambiguous'):
            visible_candidates+=1
        if feature:
            feature=centre_axis(p,feature)
            if feature: found.append(feature)
            else: detail['reason']='shaft_curvature_unresolved'
    diag.update(groups=len(groups),visible=visible_candidates==1,visibleCandidates=visible_candidates,
                features=len(found),reason="measured" if len(found)==1 else "no_unique_resolved_shaft")
    return found[0] if len(found)==1 else None


def local_median(array, region):
    """Smallest symmetric neighbourhood, evaluated only around the search ROI."""
    result=array.copy()
    y,x=np.where(region)
    if not len(y): return result
    y0,y1=int(y.min()),int(y.max())+1
    x0,x1=int(x.min()),int(x.max())+1
    padded=np.pad(array,1,mode='edge')[y0:y1+2,x0:x1+2]
    windows=np.lib.stride_tricks.sliding_window_view(padded,(3,3))
    result[y0:y1,x0:x1]=np.median(windows,axis=(-2,-1))
    return result


def holes(sample, target, radius=.05, diagnostics=None, region=None):
    raw,cam=sample
    if diagnostics is None: diagnostics=raw.setdefault('holeDiagnostics',{})
    if region is None: region=window(raw,cam,target,radius,diagnostics)
    return detect_region(raw,cam,region,diagnostics,calibrated=calibrated())
