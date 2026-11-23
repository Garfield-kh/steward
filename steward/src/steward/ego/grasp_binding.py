"""ASSEMBLY_GRASP_BINDING: a sensed load is not yet the requested object.

The lift probe is temporary. Unknown identity commits the legacy best estimate
after at most two views; only positive contradictions veto transport. No truth inputs.
"""
from copy import copy, deepcopy
from types import SimpleNamespace
import math
import numpy as np
from steward.ego import capabilities


def enabled():
    return capabilities.get('STEWARD_ASSEMBLY_GRASP_BINDING') != '0'


def receipt(est, eid=None):
    if not enabled(): return None
    row = getattr(est, 'load_bindings', {}).get(eid or est.held_id)
    return deepcopy(row) if row else None


def begin(est, target, since=None, gripper=None, at=None, body=None):
    """Separate operational best-estimate geometry from the stationary identity."""
    source = est.tracks.get(target)
    snapshot = getattr(est, '_binding_pickup_snapshot', None)
    if source is None and snapshot is not None and snapshot.id == target:
        source = snapshot
    if source is None:
        raise KeyError(f'{target} has no pregrasp geometry')
    serial = getattr(est, '_load_serial', 0)+1
    while f'load:{serial}' in est.tracks:
        serial += 1
    est._load_serial = serial
    load = deepcopy(source)
    load.id = f'load:{serial}'
    load.held = False
    load.carried_offset = None
    load.pinned = False
    load.placement = None
    # mark_held below initializes the operational endpoint frame. It may only
    # invalidate this new load ID, never a requested body or assembly member.
    est.tracks[load.id] = load
    est.load_bindings = getattr(est, 'load_bindings', {})
    row = dict(holdingDetected=True, requestedTarget=target, load=load.id,
        bindingStatus='pending', transportCommitted=False, loadEvidence=[],
        originalPosition=source.pos.tolist(), originalSize=source.size.tolist(),
        originalYaw=float(source.yaw), observedAt=float(source.last_seen),
        pickupState=deepcopy(body) if body else dict(joints=deepcopy(est.joints), odometry=deepcopy(est.odometry)),
        recordedAt=float(est.stamp), pickupPoint=None if at is None else list(at),
        graspAt=max(est.stamp, since or est.stamp),
        uncertaintyM=float(np.sqrt(max(source.var))), category=source.category)
    view = source.manipulation_view or {}
    row['originalPoints'] = np.asarray(view.get('points', [])).tolist()
    row['originalViewComplete'] = bool(view and not view.get('truncated', True)
        and view.get('invalidDepthFraction', 1) <= .05)
    # Known attached members enlarge only the expected load envelope. They do
    # not become carried tracks until the requested identity is corroborated.
    low, high = source.pos-source.size/2, source.pos+source.size/2
    record = est.stationary_assemblies.records.get(target, {})
    for member in record.get('parts', []):
        if member.get('status') == 'invalidated': continue
        other = est.tracks.get(member['entity'])
        position = np.asarray(member.get('position', other.pos if other is not None else source.pos))
        size = np.asarray(member.get('size', other.size if other is not None else source.size))
        low, high = np.minimum(low, position-size/2), np.maximum(high, position+size/2)
    row['expectedLoadSize'] = (high-low).tolist()
    support = est.support_of(source)
    row['supportTopM'] = float(support.top()) if support is not None else None
    est.load_bindings[load.id] = row
    est._binding_source_snapshot = deepcopy(source)
    # A load proxy must never acquire children by geometric containment.
    est.mark_held(load.id, since, gripper, at, body)
    row['pickupPoint'] = list(load.grasp_association['gripperAtGrasp'])
    load.grasp_association.update(requested=target, bindingStatus='pending', transportCommitted=False)
    return load.id


def pending(est, eid=None):
    row = receipt(est, eid)
    return bool(row and not row['transportCommitted'])


def decide(evidence):
    """Only an isolated, sufficiently sampled oversized load can veto.

    Origin depth matches are retained as diagnostics: a thin object's exposed
    support or another surface is not a re-identification of that object.
    """
    if (evidence.get('connectedExtentIncompatible') is True
            and evidence.get('completeLoad') is True
            and evidence.get('followsHand') is True
            and evidence.get('connectedPoints', 0) >= 20):
        return 'contradicted'
    if (evidence.get('completeLoad') is True and evidence.get('geometryCompatible') is True
            and evidence.get('followsHand') is True and evidence.get('originalStillPresent') is False):
        return 'verified'
    return 'tentative'


def resolve(est, eid, evidence, *, defer_unknown=False):
    row = est.load_bindings[eid]
    row['loadEvidence'].append(deepcopy(evidence))
    row['bindingStatus'] = decide(evidence)
    if row['bindingStatus'] == 'contradicted' or (defer_unknown and row['bindingStatus'] == 'tentative'):
        return eid
    return commit(est, eid)


def commit(est, eid):
    """Verified and unknown both execute the same legacy transport bookkeeping."""
    row = est.load_bindings[eid]
    if row['transportCommitted'] or row['bindingStatus'] == 'contradicted':
        return eid
    target = row['requestedTarget']
    if target not in est.tracks:
        # The robot's own pickup snapshot keeps identity when perception loses
        # the target during docking/lift. Never substitute an oracle identity.
        snapshot = getattr(est, '_binding_source_snapshot', None)
        if snapshot is None or snapshot.id != target:
            snapshot = deepcopy(est.tracks[eid])
            snapshot.id = target
        est.tracks[target] = deepcopy(snapshot)
    grip = est.held_gripper
    est.tracks[target].pos = np.asarray(row['originalPosition'], float)
    est.tracks[target].yaw = row['originalYaw']
    # Remove the proxy BEFORE mark_held's containment scan; it is not a member.
    est.tracks.pop(eid, None)
    est.load_bindings.pop(target, None)  # a previous grasp must not suppress this commit
    stamp = est.stamp
    try:
        # mark_held records grasp time in registration, member and endpoint
        # history. Delayed inspection must not move those events to view time.
        est.stamp = row['recordedAt']
        est.mark_held(target, row['graspAt'], grip, row['pickupPoint'], row['pickupState'])
    finally:
        est.stamp = stamp
    row.update(transportCommitted=True, load=target,
               bindingUnconfirmed=row['bindingStatus'] != 'verified')
    est.load_bindings[target] = row
    est.load_bindings.pop(eid, None)
    est.tracks[target].grasp_association.update(identityVerified=not row['bindingUnconfirmed'],
        bindingStatus=row['bindingStatus'], bindingUnconfirmed=row['bindingUnconfirmed'],
        transportCommitted=True)
    est._follow_held()
    return target


def public(row):
    return {k:deepcopy(row[k]) for k in ('holdingDetected','requestedTarget','loadEvidence',
        'bindingStatus','transportCommitted','bindingUnconfirmed') if k in row}


def release(est, eid, spot, body=None):
    """Record the physical event; committed estimates use ordinary release_at."""
    row = receipt(est, eid)
    if not row: return False
    new = not row.get('releaseRecorded', False)
    if new:
        est._load_release_serial = getattr(est, '_load_release_serial', 0)+1
    row.update(newReleaseExecuted=new, releaseRecorded=True, releasedAt=float((body or {}).get('stamp', est.stamp)))
    est.load_bindings[eid] = row
    est.last_load_release = dict(public(row), load=eid, releasedAt=row['releasedAt'],
        newReleaseExecuted=new, specifiedTargetReleased=bool(new and row['transportCommitted']))
    if row['transportCommitted']: return False
    load = est.tracks.get(eid)
    if load is not None:
        if body and body.get('joints') is not None and body.get('odometry') is not None and load.held_offset is not None:
            load.pos = est.gripper_position(est.held_gripper,body['joints'],body['odometry'])+est.gripper_rotation(
                est.held_gripper,body['joints'],body['odometry']) @ load.held_offset
        load.pos[:len(spot)] = spot
        load.placement = dict(releasedAt=row['releasedAt'], expectedXYM=list(spot[:2]),
            observation=None, bindingUnconfirmed=True, source='unbound_load_release_fk')
    est._unmark_held(load)
    return True


def annotate_release(est, result, eid, *, new=True):
    row = receipt(est, eid)
    if not row: return result
    stamp=result.get('seatedAt', result.get('releasedAt'))
    history=result.get('releaseHistory', [])
    result.update(public(row), newReleaseExecuted=bool(new),
        historicalReleaseExecuted=bool(any(not new or h.get('stamp') != stamp for h in history)),
        specifiedTargetReleased=bool(new and row['transportCommitted']))
    if not row['transportCommitted']:
        # A physical load event is retained, but is not a named-material release.
        history = result.get('releaseHistory', [])
        unknown = [h for h in history if h.get('screw') in (eid, row['load'])]
        if unknown:
            result['loadReleaseHistory'] = deepcopy(unknown)
            result['releaseHistory'] = [h for h in history if h not in unknown]
        result['historicalReleaseExecuted'] = bool(result.get('releaseHistory'))
        result.update(releaseExecuted=False, verified=False, seatConfirmed=False,
            status='load_released_identity_unconfirmed', bindingUnconfirmed=True)
    return result


def measure(est, row, sample):
    """Whole-image depth connectivity, with robot geometry removed, no instance mask.

    Grow from the pinch through ALL connected depth pixels, including a large
    object attached to a narrow tip. Bounds are catalogue/observation derived.
    A border, missing depth or ambiguous component leaves completeness unknown.
    """
    from steward.ego.support_holes import unproject
    from steward.ego.classify import SHAPES
    raw, cam = sample
    depth = raw['depth']; height, width = depth.shape
    header = raw['header']; stamp = float(header['stamp'])
    out = dict(source='native_full_depth_connected_load', observedAt=stamp,
        completeLoad=False, geometryCompatible=None, followsHand=None, originalStillPresent=None)
    if stamp <= row['graspAt']:
        return dict(out, reason='frame_before_lift')
    intr = dict(width=width, height=height, fovy=raw['fovy'])
    yy, xx = np.nonzero(np.isfinite(depth) & (depth > 0))
    if not len(xx): return dict(out, reason='depth_unavailable')
    points = unproject(np.c_[xx,yy], depth[yy,xx], cam, intr)
    focal = height/(2*math.tan(math.radians(raw['fovy'])/2))
    pixel = depth[yy,xx]/focal
    proxy = copy(est); proxy.joints=header['joints']; proxy.odometry=header['odometry']
    valid = np.ones(len(points), bool)
    for centre, half, rotation in proxy.body_boxes():
        local = (points-centre) @ rotation
        valid &= ~np.all(abs(local) <= np.asarray(half)+pixel[:,None], axis=1)
    # Depth-connected pixels; adjacency distance is derived from optical pixel
    # footprint. No target-sized crop can hide the back of an attached object.
    grid = np.full(depth.shape, -1, dtype=int); grid[yy[valid],xx[valid]]=np.flatnonzero(valid)
    grip = est.gripper_position(est.held_gripper, header['joints'], header['odometry'])
    load = est.tracks[est.held_id]
    rotation = est.gripper_rotation(est.held_gripper, header['joints'], header['odometry'])
    predicted = grip+rotation @ load.held_offset
    size = np.asarray(row.get('expectedLoadSize', row['originalSize']))
    sigma = row['uncertaintyM']
    nominal = SHAPES.get(row['category'], {})
    dims = np.asarray(nominal.get('dims', size), float)
    scale = max(nominal.get('scale', (1.,1.)))
    bound = max(np.linalg.norm(size), np.linalg.norm(dims)*scale)+nominal.get('extent_slack', 0.)+2*sigma
    # The grasp frame is a search hypothesis, not evidence of movement.
    distance = np.linalg.norm(points-predicted,axis=1)
    # The observed support is stationary. Its returns cannot serve as a load
    # seed when the hand has risen clear of it.
    support_top = row.get('supportTopM')
    seed_valid = valid.copy()
    if support_top is not None and predicted[2]-size[2]/2 > support_top+2*sigma:
        seed_valid &= points[:,2] > support_top+2*pixel
    candidates = np.flatnonzero(seed_valid & (distance <= np.linalg.norm(size)/2+2*sigma+2*pixel))
    if not len(candidates): return dict(out, reason='no_connected_load_at_hand')
    seed = int(candidates[np.argmin(distance[candidates])]); stack=[seed]; seen={seed}
    while stack:
        i=stack.pop(); y,x=int(yy[i]),int(xx[i])
        for dy,dx in ((-1,0),(1,0),(0,-1),(0,1)):
            y2,x2=y+dy,x+dx
            if not (0<=y2<height and 0<=x2<width): continue
            j=int(grid[y2,x2])
            if j<0 or j in seen: continue
            if np.linalg.norm(points[i]-points[j]) <= 4*max(pixel[i],pixel[j]):
                seen.add(j); stack.append(j)
    ids=np.fromiter(seen,int); cloud=points[ids]
    extent=np.ptp(cloud,axis=0); diameter=float(np.linalg.norm(extent))
    # A diagonal of an axis-aligned box may overbound an arbitrary rotation.
    # Use its longest axis for a conservative, rotation-independent veto.
    incompatible=bool(max(extent)>bound)
    border=bool(np.any((xx[ids]==0)|(xx[ids]==width-1)|(yy[ids]==0)|(yy[ids]==height-1)))
    missing=False
    for dy,dx in ((-1,0),(1,0),(0,-1),(0,1)):
        y2=np.clip(yy[ids]+dy,0,height-1);x2=np.clip(xx[ids]+dx,0,width-1)
        adjacent = depth[y2,x2]
        missing |= bool(np.any(~np.isfinite(adjacent) | (adjacent<=0)))
    moved=float(np.linalg.norm(predicted-np.asarray(row['originalPosition'])))
    uncertainty=2*sigma+4*float(np.median(pixel[ids]))
    # A segmentation fragment terminated by the robot mask is not a complete
    # load, even when the visible endpoint happens to fit the catalogue.
    occluded=False
    for dy,dx in ((-1,0),(1,0),(0,-1),(0,1)):
        y2=np.clip(yy[ids]+dy,0,height-1);x2=np.clip(xx[ids]+dx,0,width-1)
        occluded |= bool(np.any((depth[y2,x2]>0)&(grid[y2,x2]<0)))
    complete=not border and not missing and not occluded and len(ids)>=20
    # Compare independent observed surface spans to the pregrasp outline; an
    # isolated endpoint cannot satisfy the full major/minor dimensions.
    # Even a one-pixel fragment needs three axes (reduced SVD has only one
    # row and used to raise a broadcasting ValueError below).
    axes=np.linalg.svd(cloud-cloud.mean(axis=0),full_matrices=len(cloud)<3)[2]
    spans=np.sort(np.ptp(cloud @ axes.T,axis=0))
    expected=np.sort(size)
    compatible=bool(np.all(abs(spans[1:]-expected[1:])<=uncertainty+expected[1:]*.35))
    original=np.asarray(row.get('originalPoints', []),float)
    present=None
    if row.get('originalViewComplete') and original.ndim==2 and original.shape[1:]==(3,) and len(original)>=20 and moved>2*uncertainty:
        from steward.ego.support_holes import project
        uv,z=project(original,cam,intr); finite=np.isfinite(uv).all(axis=1)&(z>0)
        ix=np.rint(np.nan_to_num(uv[:,0])).astype(int); iy=np.rint(np.nan_to_num(uv[:,1])).astype(int)
        inside=finite&(ix>=0)&(ix<width)&(iy>=0)&(iy<height)
        if np.mean(inside)>.8:
            observed=depth[iy[inside],ix[inside]]; prior=z[inside]
            known=observed>0
            if np.mean(known)>.8:
                # Raw cloud correspondence uses optical resolution, not the
                # much wider track association uncertainty (which could count
                # the exposed table as the original thin object).
                depth_tolerance=4*float(np.median(pixel[ids]))
                match=np.mean(abs(observed[known]-prior[known])<=depth_tolerance)
                behind=np.mean(observed[known]>prior[known]+depth_tolerance)
                if match>.8: present=True
                elif behind>.8: present=False
    out.update(completeLoad=complete, connectedExtentIncompatible=incompatible,
        geometryCompatible=compatible, followsHand=bool(moved>2*uncertainty and
            np.linalg.norm(cloud.mean(axis=0)-predicted)<=max(np.linalg.norm(size),diameter)/2+uncertainty),
        originalStillPresent=present, connectedPoints=len(ids), measuredExtentM=extent.tolist(),
        extentBoundM=float(bound), originalToPredictedM=moved, border=border, missingDepth=missing,
        occludedByRobot=occluded, reason='whole_connected_load_measured')
    return out


def after_lift(rt, eid, job, plan, pickup):
    from steward.ego import action_budget as ab, wrist_observation as wo
    row=receipt(rt.estimator,eid)
    if not row: return eid
    ctl=SimpleNamespace(rt=rt,ex=rt.executor,_wristContext=dict(phase='load_binding'))
    cameras=[c for c in ('hand_camera','wrist_camera',*rt.estimator.cameras)
             if c in rt.estimator.cameras and c in getattr(rt,'_provided_camera_ids',())]
    cameras=list(dict.fromkeys(cameras))[:2]  # one initial view, at most one alternate view
    evidence=dict(source='native_full_depth_connected_load',reason='camera_unavailable')
    hand_sample=None
    for camera in cameras:
        phase='capture'
        try:
            sample=wo.capture(ctl,ab.make(ctl.ex,ab.frame_timeout(ctl.ex,4.),'load identity'),
                              camera=camera,check_view=False)
            if camera == 'hand_camera' and sample is not None: hand_sample=sample
            phase='measure'
            evidence=measure(rt.estimator,row,sample) if sample else dict(
                source='native_full_depth_connected_load',reason='frame_unavailable',camera=camera)
        except (TimeoutError,OSError,ValueError,KeyError,AttributeError,TypeError) as exc:
            evidence=dict(source='native_full_depth_connected_load',reason=type(exc).__name__, detail=str(exc), phase=phase, camera=camera)
        with rt._lock:
            if rt.estimator.held_id != eid:
                job.postcondition = job.postcondition or {}
                job.postcondition.update(public(row), bindingStatus='tentative',
                    loadEvidence=[dict(reason='grasp_changed_during_capture')])
                return eid
            eid=resolve(rt.estimator,eid,evidence,defer_unknown=True)
            row=receipt(rt.estimator,eid)
        if row['bindingStatus'] in ('verified','contradicted'): break
    if not cameras:
        with rt._lock:
            eid=resolve(rt.estimator,eid,evidence);row=receipt(rt.estimator,eid)
    if row['bindingStatus'] == 'tentative':
        with rt._lock:
            eid=commit(rt.estimator,eid);row=receipt(rt.estimator,eid)
    job.postcondition = job.postcondition or {}
    job.postcondition.update(public(row))
    if row['bindingStatus']=='contradicted':
        # Only lift moved since close. Reverse that same path before any carry,
        # then open at the actual probe pose (not the nominal target centre).
        ex=rt.executor
        ex._set_joint('lift',pickup['joints']['lift'])
        ex._command({'gripper':'open'})
        ex._released(eid,row['originalPosition'][:2])
        rt.estimator.rejected_load_candidates=getattr(rt.estimator,'rejected_load_candidates',set())|{row['requestedTarget']}
        rejected=getattr(rt.estimator,'rejected_grasp_points',{})
        rejected.setdefault(row['requestedTarget'],[]).append(row['pickupPoint'][:2])
        rt.estimator.rejected_grasp_points=rejected
        job.postcondition.update(returnedToPickup=True,candidateConsumed=True,newReleaseExecuted=True,
                                 specifiedTargetReleased=False)
        ex._set_joint('lift',plan['carryLift'])
        from steward.ego.skills import SkillFailed
        raise SkillFailed('grasp mismatch: independent load geometry contradicted requested target')
    if hand_sample is not None:
        rt._binding_hand_sample=dict(eid=eid,since=rt.estimator.held_since,sample=hand_sample)
    return eid
