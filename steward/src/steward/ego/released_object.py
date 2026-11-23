"""Follow the released object in robot RGB-D, independently of mouth visibility.

All lengths are measured at grasp or in the current image. Release FK is an
identity/search prior, never a posterior observation or an attachment claim.
"""
from copy import deepcopy
import math
import uuid
import numpy as np
from steward.ego import capabilities, wrist_observation as wo, action_budget as ab

SOURCE = 'released_object_rgbd'


def enabled():
    return capabilities.get('STEWARD_ASSEMBLY_RELEASED_OBJECT') != '0'


def unit(v):
    v = np.asarray(v, float)
    if v.shape != (3,) or not np.isfinite(v).all() or np.linalg.norm(v) == 0:
        raise ValueError('unresolved_axis')
    return v / np.linalg.norm(v)


def unknown(reason, **kw):
    return dict(status='unknown', passed=None, landingState='unclear', source=SOURCE,
                reason=reason, **kw)


def prepare(controller, plan):
    """Save the held identity and background before open, without extra motion."""
    if not enabled(): return
    try:
        from steward.ego.fasten_geometry import transform_feature
        est = controller.rt.estimator
        track = est.tracks.get(plan.get('object'))
        local = (getattr(track, 'held_tool_feature', None) or
                 getattr(track, 'held_grasp_geometry', None))
        if not local or est.held_id != plan.get('object'): return
        state = controller.ex._state()
        feature = transform_feature(local,
            est.gripper_rotation(est.held_gripper, state['joints'], state['odometry']),
            est.gripper_position(est.held_gripper, state['joints'], state['odometry']))
        from steward.ego import seating_geometry
        feature = seating_geometry.release_geometry(controller, plan, feature)
        length = float(np.linalg.norm(np.asarray(feature['head'])-feature['tip']))
        radius = float(feature.get('radiusM', 0.))
        if min(length, radius) <= 0: return
        target = deepcopy(plan['seatTarget'])
        height = plan.get('releaseHeight', {}).get('after', {})
        if height.get('planePoint') is not None:
            # A supporting plane supplies height; its fit centroid cannot rename
            # the selected entrance or move its lateral identity.
            normal = unit(height['normal'])
            point = np.asarray(target['point'],float)
            if abs(normal[2]) > np.finfo(float).eps:
                point[2] = height['planePoint'][2]-np.dot(
                    point[:2]-np.asarray(height['planePoint'])[:2],normal[:2])/normal[2]
            target.update(point=point.tolist(), normal=normal.tolist())
        ctx = dict(object=plan['object'], body=plan.get('seatBody'), hole=plan.get('seatHole'),
            target=target, feature=feature, lengthM=length, radiusM=radius,
            preparedAt=state['stamp'], baseline=[], emptyBefore=False,
            identityBasis='held_identity_and_release_fk', peers=[])
        for a in getattr(est, 'fastening_attempts', {}).values():
            if a.get('body') == ctx['body'] and a.get('hole') != ctx['hole']:
                c = a.get('seatCheck') or {}
                if a.get('releaseExecuted') and c.get('passed') is not False:
                    ctx['peers'].append(dict(hole=a['hole'], point=c.get('headM', a['entry'])))
        history = ((plan.get('tipEntryServo') or {}).get('frames', [])+
                   (plan.get('insertionSearch') or {}).get('observations', []))
        for f in history:
            entry = f.get('entry') or {}
            if (entry.get('point') is not None and not entry.get('targetFromLayout')
                    and not f.get('targetFromLayout') and entry.get('observedKind') != 'head' and
                    np.linalg.norm(np.asarray(entry['point'])-target['point']) < radius):
                ctx['emptyBefore'] = True
            ctx['baseline'].extend(deepcopy(f.get('diagnostics', {}).get('heads', [])))
        plan['releasedObjectContext'] = ctx
        samples = (getattr(controller,'_releasedObjectHeldSamples',[])
                   if getattr(controller,'_releasedObjectSampleIdentity',None) == ctx['object'] else [])
        samples = [sample for sample in samples if
                   getattr(est,'held_since',-math.inf) <= sample[0]['header']['stamp'] <= state['stamp']]
        # Raw arrays are transient, not copied into the public attempt/context.
        plan['_releasedObjectBaselines'] = list(samples)
        if samples:
            sample = samples[-1]
            candidates, _ = detect(controller, sample, ctx, 4)
            header = sample[0]['header']
            held_then = transform_feature(local,
                est.gripper_rotation(est.held_gripper,header['joints'],header['odometry']),
                est.gripper_position(est.held_gripper,header['joints'],header['odometry']))
            ctx['baseline'].extend(dict(point=c['head'], radiusM=c['radiusM'],
                uncertaintyM=c['uncertaintyM']) for c in candidates
                if np.linalg.norm(np.asarray(c['head'])-held_then['head']) > radius+c['uncertaintyM'])
    except (AttributeError, KeyError, TypeError, ValueError, np.linalg.LinAlgError):
        # Missing context leaves the existing bounded best-estimate release intact.
        return


def shape(points, prior, noise, stamp):
    """Scale-relative endpoint widths or a resolved cap with a measured normal."""
    p = np.asarray(points, float)
    if len(p) < 24 or not np.isfinite(p).all(): return None
    centre = np.median(p, axis=0)
    _, sv, axes = np.linalg.svd(p-centre, full_matrices=False)
    length, radius = prior['lengthM'], prior['radiusM']
    # A complete shaft: estimate head polarity from end widths, not world up.
    along = (p-centre) @ axes[0]
    lo, hi = np.quantile(along, [.01, .99]); span = hi-lo
    if sv[0] > 1.25*sv[1] and .5*length-noise <= span <= 1.6*length+noise:
        radial = p-centre-along[:, None]*axes[0]
        masks = [along <= lo+.22*span, along >= hi-.22*span]
        ends = [radial[m] for m in masks]
        if min(map(len, ends)) >= 6:
            widths = [float(np.quantile(np.linalg.norm(e-np.median(e, axis=0), axis=1), .9)) for e in ends]
            wide = int(widths[1] > widths[0])
            if (.4*radius-noise <= widths[wide] <= 1.7*radius+noise and
                    widths[wide] >= 1.35*max(widths[1-wide],noise) and
                    widths[wide]-widths[1-wide] > noise):
                ends = [centre+t*axes[0]+np.median(e, axis=0) for t,e in zip((lo, hi), ends)]
                head, tip = ends[wide], ends[1-wide]
                return dict(head=head.tolist(), tip=tip.tolist(), axis=unit(head-tip).tolist(),
                    lengthM=float(np.linalg.norm(head-tip)), radiusM=widths[wide],
                    centreM=((head+tip)/2).tolist(), polarityObserved=True, axisObserved=True,
                    kind='endpoints', uncertaintyM=noise, stamp=stamp)
    # Once seated only the cap remains visible. Fit its own plane and curvature.
    if sv[1] <= 0 or sv[0] > 1.8*sv[1]: return None
    normal = axes[2]
    if normal @ unit(prior['target'].get('normal', [0.,0.,1.])) < 0: normal = -normal
    # The component includes the cap sidewall. Refine its exposed end plane
    # instead of requiring the entire cap volume to be an infinitesimal disk.
    face = p
    for _ in range(3):
        height = (p-centre) @ normal
        face = p[height >= np.quantile(height,.6)-noise]
        if len(face) < 24: return None
        centre = np.mean(face,axis=0)
        _,_,basis = np.linalg.svd(face-centre,full_matrices=False)
        n = basis[2]
        if n @ normal < 0: n = -n
        normal = n
    if np.quantile(abs((face-centre) @ normal),.95) > noise: return None
    uv = (face-centre) @ basis[:2].T
    from steward.ego.driver_view import convex_boundary
    from steward.ego.entry_layout import circle
    fit = circle(convex_boundary(uv), noise)
    if fit is None: return None
    offset, measured_radius, _ = fit
    if not .5*radius-noise <= measured_radius <= 1.6*radius+noise: return None
    head = centre+basis[:2].T @ offset
    return dict(head=head.tolist(), axis=normal.tolist(), centreM=head.tolist(),
        radiusM=float(measured_radius), uncertaintyM=noise, stamp=stamp,
        kind='cap', polarityObserved=False, axisObserved=True,
        estimatedTipM=(head-normal*length).tolist(), estimatedLengthM=length)


def detect(controller, sample, ctx, ring):
    """Depth clusters after removing observed support planes and robot pixels."""
    from steward.ego.support_holes import components
    from steward.ego.wrist_surface import fit
    from steward.ego.wrist_release_geometry import robot_boxes
    raw, cam = sample
    extent = ring*ctx['lengthM']+ctx['radiusM']
    target = ctx['target']['point']
    region = wo.window(raw, cam, target, extent)
    p,y,x = wo.cloud(raw, cam, region)
    diag = dict(ring=ring, points=len(p), candidates=[])
    if len(p) < 24: return [], diag
    focal = raw['depth'].shape[0]/(2*math.tan(math.radians(raw['fovy'])/2))
    noise = max(float(np.median(raw['depth'][y,x]))/focal,
                float(ctx['target'].get('planeNoiseM', 0.)))
    keep = np.linalg.norm(p-target, axis=1) <= extent
    for c,half,r in robot_boxes(controller, raw):
        keep &= ~np.all(abs((p-c) @ r) <= np.asarray(half)+noise, axis=1)
    mask = np.zeros(raw['depth'].shape, bool); mask[y[keep],x[keep]] = True
    # The lower support may be below the entry face; do not crop everything
    # below that face, which would erase a fallen object on the nearby support.
    for _ in range(2):
        plane = fit(raw, cam, mask)
        if plane is None: break
        on = abs((p-plane['point']) @ plane['normal']) <= max(noise, plane['residualM'])
        if np.count_nonzero(on & keep) < max(48, np.count_nonzero(keep)/5): break
        patch = p[on & keep]
        spread = np.linalg.svd(patch-patch.mean(0),compute_uv=False)/math.sqrt(len(patch))
        if spread[1] < 2*ctx['radiusM']: break  # A cap is an object, not another support plane.
        keep &= ~on
        mask[:] = False; mask[y[keep],x[keep]] = True
    # Split surfaces which overlap in projection but are depth-disconnected.
    xyz = np.full((*mask.shape,3), np.nan); xyz[y,x] = p
    edges = np.zeros_like(mask)
    for axis in (0,1):
        a,b = (slice(1,None),slice(None,-1))
        aa,bb = ((a,slice(None)),(b,slice(None))) if axis==0 else ((slice(None),a),(slice(None),b))
        jump = (np.linalg.norm(xyz[aa]-xyz[bb], axis=2)>3*noise) & mask[aa] & mask[bb]
        edges[aa] |= jump; edges[bb] |= jump
    joined = mask.copy()
    mask &= ~edges
    result = []
    for pixels in [*components(joined),*components(mask)]:
        if len(pixels) < 24: continue
        q = xyz[pixels[:,1],pixels[:,0]]
        candidate = shape(q, ctx, noise, raw['header']['stamp'])
        if candidate is None: continue
        # A cut at the image/ROI boundary cannot be a measured endpoint.
        border = bool(np.any(pixels.min(0)<2) or
                      np.any(pixels.max(0)>=np.array(mask.shape[::-1])-2) or
                      np.max(np.linalg.norm(q-target,axis=1)) >= extent-noise)
        if border: continue
        candidate['pointCount'] = len(q)
        candidate['_points'] = q
        if any(np.linalg.norm(np.asarray(candidate['head'])-c['head']) <= noise+ctx['radiusM']
               for c in result): continue
        result.append(candidate)
    diag.update(noiseM=noise, candidates=[public(c) for c in result])
    return result, diag


def public(candidate):
    return {k:deepcopy(v) for k,v in candidate.items() if not k.startswith('_')}


def select(candidates, ctx, ring):
    """Release continuity plus entrance ownership; never select a known peg/peer."""
    length, radius = ctx['lengthM'], ctx['radiusM']
    origin = (np.asarray(ctx['feature']['head'])+ctx['feature']['tip'])/2
    ranked = []
    for c in candidates:
        head = np.asarray(c['head']); noise = c['uncertaintyM']
        if any(np.linalg.norm(head-b['point']) <= max(b.get('radiusM',radius), c['radiusM'])+
               max(noise,b.get('uncertaintyM',0.)) for b in ctx.get('baseline', [])): continue
        if any(np.linalg.norm(head-b['point']) <= radius+c['radiusM']+noise
               for b in ctx.get('peers', [])): continue
        distance = float(np.linalg.norm(np.asarray(c['centreM'])-origin))
        if distance > ring*length+radius: continue
        # An isolated plateau needs pre-release evidence of an empty entry.
        # This is an observation rule only, never a prerequisite for opening.
        if c['kind'] == 'cap' and not (c.get('newFootprint') or ctx.get('emptyBefore') and
                np.linalg.norm(head-np.asarray(ctx['target']['point'])) <= 2*radius+noise): continue
        ranked.append((distance,c))
    ranked.sort(key=lambda p:p[0])
    if not ranked: return None, 'released_object_not_found'
    if len(ranked)>1 and ranked[1][0]-ranked[0][0] <= radius+ranked[0][1]['uncertaintyM']:
        return None, 'released_object_identity_ambiguous'
    return ranked[0][1], None


def classify(candidate, ctx):
    """A measured axis crossing the entrance and a low supported head seats it."""
    c = candidate; target = ctx['target']; normal = unit(target.get('normal', [0.,0.,1.]))
    axis = unit(c['axis']); head = np.asarray(c['head']); entry = np.asarray(target['point'])
    noise = max(c['uncertaintyM'], float(target.get('planeNoiseM',0.)))
    dot = float(axis @ normal); height = float((head-entry) @ normal)
    extent = c.get('lengthM', 2*c['radiusM'])
    tilt = float(np.linalg.norm(np.cross(axis,normal))*extent)
    if abs(dot)>np.finfo(float).eps:
        crossing = head-axis*(height/dot)
        lateral = float(np.linalg.norm(crossing-entry))
    else: lateral = math.inf
    tolerance = float(target.get('radiusM') or ctx['radiusM']/2)+noise
    landing = 'seated'
    if dot < 0 and c.get('polarityObserved'): landing = 'inverted_upright'
    elif tilt > max(2*noise,c['radiusM']/2): landing = 'fallen_or_tilted'
    elif lateral > tolerance: landing = 'upright_beside'
    elif height > ctx['radiusM']+2*noise: landing = 'raised_at_entry'
    elif height <= 0: landing = 'below_entry'
    passed = landing == 'seated'
    result = dict(status='confirmed' if passed else 'failed', passed=passed,
        source=SOURCE, reason='released_object_'+landing, landingState=landing,
        observedAt=c['stamp'], headM=c['head'], axisM=c['axis'],
        axisObserved=True, freshEndpointObserved=c['kind']=='endpoints',
        headPolarityObserved=c.get('polarityObserved',False), attachmentObserved=False,
        headMinusSurfaceZM=height, axisDistanceM=lateral if math.isfinite(lateral) else None,
        uncertaintyM=noise, object=ctx['object'], body=ctx['body'], hole=ctx['hole'],
        identityBasis=ctx['identityBasis'], objectPose=public(c))
    if c.get('tip') is not None: result['tipM'] = c['tip']
    if not passed:
        result['recoveryTarget'] = dict(object=ctx['object'], body=ctx['body'], hole=ctx['hole'],
            observedAt=c['stamp'], centreM=c['centreM'], headM=c['head'],
            tipM=c.get('tip',c.get('estimatedTipM')), tipEstimated=c.get('tip') is None,
            axisM=c['axis'], polarityObserved=c.get('polarityObserved',False),
            source=SOURCE, identityBasis=ctx['identityBasis'])
    return result


def new_footprint(candidate, samples):
    """A previously exposed annulus proves novelty even with an occluded mouth."""
    from steward.ego.support_holes import project
    axis = unit(candidate['axis'])
    tangent = unit(np.cross(axis, [1.,0.,0.] if abs(axis[0])<.8 else [0.,1.,0.]))
    other = np.cross(axis,tangent)
    probes = np.asarray(candidate['head'])+candidate['radiusM']*.7*np.array([tangent,-tangent,other,-other])
    for raw,cam in samples:
        h,w = raw['depth'].shape
        uv,z = project(probes,cam,dict(width=w,height=h,fovy=raw['fovy']))
        if not np.isfinite(uv).all(): continue
        pixel = np.rint(uv).astype(int)
        inside = (z>0) & np.all(pixel>=0,axis=1) & (pixel[:,0]<w) & (pixel[:,1]<h)
        good = 0
        for i in np.flatnonzero(inside):
            u,v = pixel[i]; depth = raw['depth'][v,u]
            if np.isfinite(depth) and depth > z[i]+2*candidate['uncertaintyM']: good += 1
        if good >= 3:
            return dict(observedAt=raw['header']['stamp'], exposedProbes=good,
                        source='pre_release_depth_behind_current_cap')
    return None


def measure(controller, sample, ctx, baselines=()):
    searches = []
    for ring in (2,4):
        candidates, diag = detect(controller, sample, ctx, ring)
        for candidate in candidates:
            if candidate['kind'] == 'cap':
                candidate['newFootprint'] = new_footprint(candidate,baselines)
        candidate, reason = select(candidates, ctx, ring)
        searches.append(dict(diag, reason=reason))
        if candidate is not None:
            check = classify(candidate, ctx)
            check['searches'] = searches
            return check, candidate
    return unknown(reason, observedAt=sample[0]['header']['stamp'], searches=searches), None


def view_pose(controller, plan, camera, ctx, second=False):
    """Lift inside the admitted exit, then retract enough to clear the sightline."""
    from steward.ego.hole_seating import clear_ray
    ex = controller.ex; before = deepcopy(ex._state())
    joints = before['joints']; target = ctx['target']['point']
    ceiling = max(joints['lift'], plan.get('approachLift', joints['lift']),
        (plan.get('seatEndpoint') or {}).get('safeLift', plan.get('safeLift', joints['lift'])))
    tilted = abs(joints.get('wrist_pitch',0.)) >= ex.config.joint_tol
    exit_floor = max(ex.config.stow_lift,
        ex.config.tilted_exit_lift if tilted else ex.config.untilted_exit_lift)
    retract_lift = max(ceiling,exit_floor)
    # Upward exit precedes any lateral motion. The release planner owns its
    # envelope; no new downward probe or base movement is introduced.
    preferred = min(ceiling, joints['lift']+ctx['lengthM']+ctx['radiusM'])
    floor_reach = min(joints['reach'], ex.config.retract_reach)
    for lift in dict.fromkeys([*np.linspace(preferred, ceiling, 5),retract_lift]):
        for reach in np.linspace(joints['reach'], floor_reach, 9):
            if reach < joints['reach'] and lift < retract_lift: continue
            future = dict(joints, lift=float(lift), reach=float(reach))
            if second and lift == joints['lift'] and reach == joints['reach']: continue
            if not clear_ray(controller, dict(before,joints=future), camera, target, 2*ctx['radiusM']): continue
            vertical = [dict(joints,lift=float(h)) for h in np.linspace(joints['lift'],lift,10)]
            lateral = [dict(joints,lift=float(lift),reach=float(r)) for r in np.linspace(joints['reach'],reach,10)]
            if not controller.rt.esim.arm.reachable_path(vertical+lateral): continue
            if lift > joints['lift']: ex._set_joint('lift',float(lift))
            # Re-evaluate the actual raised pose before allowing retraction.
            actual = ex._state()
            required = lift if reach < joints['reach'] else preferred
            if actual['joints']['lift'] < required-ex.config.joint_tol:
                return dict(armClear=False, reason='axial_retreat_incomplete', secondView=second)
            if reach < joints['reach']: ex._set_joint('reach',float(reach))
            actual = ex._state()
            clear = clear_ray(controller,actual,camera,target,2*ctx['radiusM'])
            return dict(executed=True, armClear=clear, secondView=second,
                reason=None if clear else 'actual_view_obstructed',
                beforeJoints=deepcopy(joints), afterJoints=deepcopy(actual['joints']),
                policy='admitted_lift_then_retract')
    return dict(executed=False, armClear=False, reason='no_reachable_clear_view', secondView=second)


def locate_for_pick(controller, ctx, candidate, check):
    """Attach independent pixels to the existing identity, never create stock."""
    if check.get('passed') is not False: return
    p = candidate.get('_points')
    if p is None: return
    est = controller.rt.estimator
    with controller.rt._lock:
        track = est.tracks.get(ctx['object'])
        if (track is None or getattr(track,'held',False) or
                getattr(track,'last_seen',-math.inf) > candidate['stamp']): return
        centre = np.asarray(candidate['centreM'])
        feature = dict(public(candidate),source='rgbd_endpoint_width')
        if candidate['kind'] == 'cap':
            feature.update(tip=candidate['estimatedTipM'],lengthM=candidate['estimatedLengthM'],
                           source='released_cap_axis_estimate',polarityObserved=False)
            centre = (np.asarray(candidate['head'])+candidate['estimatedTipM'])/2
            check['recoveryTarget'].update(centreM=centre.tolist(),poseEstimated=True)
        # Track extents are yaw-local, not a world-aligned point-cloud box.
        # Replace the old release envelope as well as its centre so the normal
        # pickup planner actually aims at this independently observed pose.
        axis = unit(candidate['axis'])
        yaw = math.atan2(axis[1],axis[0])
        rotation = np.array([[math.cos(yaw),-math.sin(yaw),0.],
                             [math.sin(yaw),math.cos(yaw),0.],[0.,0.,1.]])
        local_axis = rotation.T @ axis
        radial = 2*candidate['radiusM']*np.sqrt(np.maximum(0.,1-local_axis**2))
        size = np.maximum(np.ptp((p-centre) @ rotation,axis=0),
                          abs(local_axis)*feature['lengthM']+radial)
        track.pos = centre.copy()
        track.last_seen = candidate['stamp']
        track.tool_feature = feature
        track.manipulation_view = dict(source='rgbd_segment', stamp=candidate['stamp'],
            centreM=np.median(p,axis=0).tolist(), visibleSizeM=np.ptp(p,axis=0).tolist(), points=p.copy(),
            truncated=False, invalidDepthFraction=0., identityBasis=ctx['identityBasis'])
        track.size = size.copy()
        track.yaw = yaw
        track.var = np.full(3,max(candidate['uncertaintyM'],candidate['radiusM']/8)**2)
        track.observations = getattr(track,'observations',0)+1
        track.settled = True
        track.voxels = track.span = None
        est.version = getattr(est,'version',0)+1
        check['recoveryTarget']['trackUpdated'] = True


def capture(controller, plan):
    from steward.ego.tip_entry_servo import camera_for
    from steward.ego.skills import SkillFailed
    camera = camera_for(controller.rt)
    ctx = deepcopy(plan.get('releasedObjectContext'))
    if not enabled() or not camera or not ctx: return None
    ctx['releasedAt'] = plan['releasedAt']
    row = dict(schema='released_object.v1', id=uuid.uuid4().hex, camera=camera,
        frames=[], motions=[], releasedAt=plan['releasedAt'], context=ctx,
        check=unknown('view_not_acquired'),
        releaseKey=[ctx['body'],ctx['hole'],ctx['object'],plan['releasedAt']])
    plan['_holeSeatReport'] = row
    try:
        for index in range(2):
            controller.check()
            camera_changed = False
            if index:
                from steward.ego.hole_seating import clear_ray
                est = controller.rt.estimator
                for other in getattr(controller.rt,'_provided_camera_ids',()):
                    if other == camera or other not in est.cameras: continue
                    if clear_ray(controller,controller.ex._state(),other,ctx['target']['point'],2*ctx['radiusM']):
                        camera = other; camera_changed = True; break
            try:
                motion = view_pose(controller,plan,camera,ctx,second=bool(index and not camera_changed))
            except SkillFailed as exc:
                controller.check()
                motion = dict(armClear=False,reason='clearance_motion_failed',detail=str(exc))
            row['motions'].append(motion)
            motion.update(camera=camera,viewIndex=index,cameraChanged=camera_changed)
            if not motion.get('armClear'):
                row['check'] = unknown(motion['reason'])
                continue  # The one alternate view must survive an obstructed first exit.
            after = max(plan['releasedAt'],controller.ex._state()['stamp'])
            controller.ex._wait(ab.frame_window(controller.ex))
            wo.context(controller,'released_object',ctx['target']['point'],4*ctx['lengthM'])
            controller._wristContext['releasedObjectContext'] = deepcopy(ctx)
            sample = wo.capture(controller,ab.make(controller.ex,ab.frame_timeout(controller.ex,4.),
                'one cleared released-object frame'),camera=camera,check_view=False)
            f = dict(index=index, requestedAfter=after,
                capture=deepcopy(getattr(controller,'_wristCapture',{})))
            row['frames'].append(f)
            if sample is None:
                row['check'] = unknown('frame_unavailable'); continue
            f['stamp'] = sample[0]['header']['stamp']
            if f['stamp'] <= after:
                row['check'] = unknown('frame_precedes_clearance',observedAt=f['stamp']); continue
            check,candidate = measure(controller,sample,ctx,plan.get('_releasedObjectBaselines',()))
            row['check'] = check
            if candidate is not None: locate_for_pick(controller,ctx,candidate,check)
            f['check'] = deepcopy(check)
            if check['passed'] is not None: break
    except (AttributeError,KeyError,TypeError,ValueError,np.linalg.LinAlgError) as exc:
        row['check'] = unknown('object_geometry_unavailable',detail=type(exc).__name__)
    finally:
        plan.pop('_releasedObjectBaselines',None)
        from steward.ego.capability_usage import record
        record('released_object',event_id=row['id'],stage='after_release',evidence=row,
            measured=row['check']['passed'] is not None,action=bool(row['motions']),
            reason=row['check']['reason'])
    return row


def current_success(attempt):
    check = attempt.get('seatCheck') or {}
    return bool(enabled() and check.get('source') == SOURCE and check.get('passed') is True
        and (check.get('observedAt') or -1) > max(attempt.get('seatedAt',-1),attempt.get('transportedAt',-1))
        and check.get('currentPoseConfirmed') is not False and not attempt.get('departureEvidence'))


def recovery_evidence(attempt, check, now=None):
    if not enabled() or check.get('source') != SOURCE or check.get('passed') is not False: return None
    target = check.get('recoveryTarget') or {}
    if now is not None and not 0 <= now-target.get('observedAt',-math.inf) <= 20:
        return None  # Existing RGB-D endpoint freshness allowance.
    if (target.get('object') != attempt.get('screw') or target.get('hole') != attempt.get('hole')
            or target.get('body') != attempt.get('body') or not target.get('trackUpdated')
            or (target.get('observedAt') or -1) <= max(attempt.get('seatedAt',-1),attempt.get('transportedAt',-1))):
        return None
    return dict(screw=target['object'], observedAt=target['observedAt'], source=SOURCE,
                recoveryTarget=deepcopy(target), check=deepcopy(check))


def refresh(controller, attempt, report):
    """Passive late review uses the same object; moving an assembly is not identity evidence."""
    ctx = deepcopy((attempt.get('holeSeatObservation') or {}).get('context'))
    if not enabled() or not ctx or current_success(attempt): return None
    ctx.update(object=attempt['screw'],body=attempt['body'],hole=attempt['hole'])
    from steward.ego import session_initial
    from steward.ego.fasten_plan import value
    from steward.ego.fasten_supply import measured_fastener
    cutoff = max(attempt.get('seatedAt',-1),attempt.get('transportedAt',-1))
    moved = attempt.get('transportedAt',-1) >= attempt.get('seatedAt',-1)
    if moved:
        frame = report.get('frame') or {}
        if (frame.get('observedAt') or -1) <= cutoff: return None
        entry = next((h for pair in frame.get('holePairs',[]) if pair['bodyHole']==attempt['hole']
                      for h in frame.get('holes',[]) if h['id']==pair['partHole']),None)
        if entry is None: return None
        ctx['target'].update(point=deepcopy(entry['point']))
        if frame.get('rotation') is not None:
            ctx['target']['normal'] = (np.asarray(frame['rotation']) @ [0.,0.,1.]).tolist()
    scene = controller.rt.ego_state()
    entity = scene.find(attempt['screw'])
    feature = measured_fastener(entity,scene.stamp) if entity is not None else None
    if feature and cutoff < feature['stamp'] <= scene.stamp:
        if (.5*ctx['lengthM'] <= feature['lengthM'] <= 1.6*ctx['lengthM'] and
                .4*ctx['radiusM'] <= feature['radiusM'] <= 1.7*ctx['radiusM']):
            c = dict(feature,kind='endpoints',axisObserved=True,
                centreM=((np.asarray(feature['head'])+feature['tip'])/2).tolist(),
                uncertaintyM=feature.get('uncertaintyM',feature['radiusM']/8))
            ctx['identityBasis'] = 'current_rgbd_original_track'
            c['_points'] = np.asarray((value(entity,'contactView') or {}).get('points',[]))
            check = classify(c,ctx)
            locate_for_pick(controller,ctx,c,check)
            return check
    if moved: return None  # Do not transport the pose of an unconfirmed loose object with the body.
    latest = None
    for sample in session_initial.samples(controller):
        if sample[0]['header']['stamp'] <= cutoff: continue
        check,c = measure(controller,sample,ctx)
        if check.get('passed') is not None:
            locate_for_pick(controller,ctx,c,check)
            if latest is None or check['observedAt'] > latest['observedAt']: latest = check
    return latest
