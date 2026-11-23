"""Bounded, observation-reusing edge placement from robot-owned geometry."""
from copy import deepcopy
import math
import numpy as np
from steward.ego import capabilities
from steward.math3d import matrix_from_quat, quat_from_yaw


def enabled():
    return capabilities.get('STEWARD_ASSEMBLY_EDGE_RESULT') != '0'


def final_contact(shape):
    return enabled() and bool((shape or {}).get('finalContactOnly'))


def cached_view(controller, record, after=-math.inf, *, face=False):
    from steward.ego.assembly_edge_observation import native_view
    previous = getattr(controller, '_edge_reuse_frames', False)
    controller._edge_reuse_frames = True
    try:
        return native_view(controller, record, after, face=face)
    finally:
        controller._edge_reuse_frames = previous


def measure(controller, reference, view, after):
    from steward.ego.assembly_edge import observe_level
    previous = getattr(controller, '_edge_reuse_frames', False)
    controller._edge_reuse_frames = True
    try:
        return observe_level(controller, reference, view, after)
    finally:
        controller._edge_reuse_frames = previous


def preflight(controller, entity, view):
    from steward.ego.assembly_edge import preflip_view
    previous = getattr(controller, '_edge_reuse_frames', False)
    controller._edge_reuse_frames = True
    try:
        return preflip_view(controller, entity, view)
    finally:
        controller._edge_reuse_frames = previous


def pre_view(controller, entity, view):
    from steward.ego.assembly_edge_observation import usable, merged_view
    controller._edge_observation_diag = {}
    record = controller.rt.estimator.stationary_assemblies.records.get(entity)
    after = (record or {}).get('reference', {}).get('ownReleaseAt', -math.inf)
    now = controller.ex._state()['stamp']
    if not usable(view, now, after):
        view = (merged_view(controller, record, after) or cached_view(controller, record, after)) if record else None
    return view, dict(attempts=0, reason='fresh_local_geometry' if view else 'cached_geometry_unavailable',
                     source=(view or {}).get('source'), reusedExistingFrame=True)


def prepare(record, turn_over, view, shape, diagnostics):
    """Prefer visible raised points, then registration, finally an explicit end prior."""
    from steward.ego import assembly_edge as edge
    if shape is None:
        flat = edge.geometry(record, turn_over, body_face=True, stable_bearing=True, allow_flat=True)
        if flat is None:
            diagnostics['priorReason'] = 'no_registered_extent'
            return None
        ref = record['reference']; origin = np.asarray(ref['position'])
        rot = matrix_from_quat(quat_from_yaw(ref['yaw']))
        direction = None; source = 'registered_extent_end_prior'
        points = np.asarray((view or {}).get('points', []))
        if turn_over and points.ndim == 2 and points.shape[1] == 3 and len(points) >= 12:
            local = (points-origin) @ rot
            high = local[local[:, 2] > np.quantile(local[:, 2], .65)]
            if len(high):
                direction = np.mean(high[:, :2], axis=0)
                source = 'observed_raised_end_estimate'
        if direction is None or np.linalg.norm(direction) < .05*max(ref['size'][:2]):
            offsets = [(np.asarray(p['position'])-origin) @ rot for p in record.get('parts', [])]
            direction = max(offsets, key=lambda d:np.linalg.norm(d[:2]))[:2] if offsets else np.zeros(2)
            source = 'registered_attachment_end_prior'
        if np.linalg.norm(direction) < .05*max(ref['size'][:2]):
            direction = np.eye(2)[int(np.argmax(ref['size'][:2]))]
            source = 'registered_extent_end_prior'
        axis = int(np.argmax(abs(direction))); end = ('+' if direction[axis] >= 0 else '-')+'xy'[axis]
        from steward.ego.assembly_edge_hint import apply
        shape, report = apply(record, turn_over, None, dict(end=end, uncertain=False))
        if shape:
            shape.update(source=source, inferred=True)
            diagnostics.update(priorReason=source, endPrior=end, geometryConfirmed=False)
    if shape:
        shape['finalContactOnly'] = True
        diagnostics['sweepSeparatedFromContact'] = True
    return shape


def rotated(shape, angle):
    result = deepcopy(shape); rot = matrix_from_quat(quat_from_yaw(angle))
    for key in ('protrusions','envelope','core','bearingPolygon'):
        if key in shape: result[key] = (np.asarray(shape[key]) @ rot.T).tolist()
    result['centreOfMass'] = (rot @ shape['centreOfMass']).tolist()
    return result


def validate(record, table, offset, sigma, yaw):
    """Check the requested contact continuously, not equality with a regenerated grid."""
    from steward.ego import assembly_edge as edge
    if offset is None: return None
    offset = np.asarray(offset, float)
    if offset.shape != (2,) or not np.isfinite(offset).all(): return None
    shape = record['edgeGeometry']; angle = (yaw if yaw is not None else record['reference']['yaw'])-record['reference']['yaw']
    shape = rotated(shape, angle)
    box, _, top = edge.entity_box(table)
    r = matrix_from_quat(quat_from_yaw(box.yaw))[:2,:2]
    half = np.array([box.hx,box.hy]); com = r.T @ np.asarray(shape['centreOfMass'])[:2]
    margin = shape['comUncertaintyM']+2*sigma
    clearance = float(np.min(half-abs(offset+com)))
    if clearance < margin or not edge.supports(shape['bearingPolygon'], shape['centreOfMass'], shape['comUncertaintyM']): return None
    pts = np.asarray(shape['protrusions'])[:,:2] @ r+offset
    env = np.asarray(shape['envelope'])[:,:2] @ r+offset
    gap = max(.008,2*sigma)+record.get('edgeClearanceExtraM',0.)
    for axis in range(2):
        for sign in (-1,1):
            tangent = 1-axis
            if (np.min(sign*pts[:,axis]) < half[axis]+gap-1e-8 or
                np.min(sign*env[:,axis]) < -half[axis]+gap-1e-8 or
                np.max(abs(env[:,tangent])) > half[tangent]-gap+1e-8): continue
            return dict(offset=offset.tolist(),spot=(np.array([box.cx,box.cy])+r @ offset).tolist(),
                edgeAxis=axis,edgeSign=sign,overhangM=float(np.max(sign*env[:,axis])-half[axis]),
                comMarginM=clearance,requiredComMarginM=margin,protrusionClearanceM=gap,
                bodyCentreZM=top-shape['bearingZM'],bearingZM=shape['bearingZM'],
                bodyGraspZM=top+(shape['upperBodyZM']-shape['bearingZM'])/2,
                comOffsetWorldM=shape['centreOfMass'][:2],massSource=shape['massSource'],
                bearingRole=shape.get('bearingRole','body_face'),estimatedTiltDeg=0.,observed=False,
                yaw=record['reference']['yaw']+angle,yawDeltaRad=angle,validation='continuous_contact_geometry')
    return None


def adjust(controller, job, plan):
    """Two release-feedback attempts at most; every attempt uses the normal release look."""
    from steward.ego import assembly_edge as edge
    from steward.ego.skills import SkillFailed
    post = job.postcondition; original = deepcopy(post['edgePlacement'])
    initial = original.get('estimatedTiltDeg'); history=[]; avoided=[]; attempts=0
    records=controller.rt.estimator.stationary_assemblies.records
    start=np.asarray(records[plan['body']]['reference']['position'][:2]).copy()
    best=None
    for attempt in range(2):
        current=post['edgePlacement']; record=records[plan['body']]
        angle=current.get('estimatedTiltDeg')
        if current.get('level') is True: break
        if angle is not None and (best is None or angle < best[0]):
            best=(angle,deepcopy(record),deepcopy(current))
        now = controller.ex._state()['stamp']
        stamp = current.get('observedAt', -math.inf)
        fresh = record['reference'].get('ownReleaseAt', -math.inf) < stamp <= now and now-stamp <= 20
        shape=edge.tilt_geometry(record,current) if fresh else None
        if shape:
            # Replace the old inferred end; geometry() retains registered low members.
            shape['finalContactOnly']=True; shape['inferred']=True
            record['edgeGeometry']=shape
        elif not record.get('edgeGeometry'):
            record['edgeGeometry']=prepare(record,False,None,None,{})
        spare = max(0., current.get('comMarginM', 0.)-current.get('requiredComMarginM', 0.))
        record['edgeClearanceExtraM'] = .5*spare if current.get('used') and fresh else 0.
        table=controller.rt.ego_state().find(record['reference']['support'])
        options=edge.candidates(record,table,controller.rt.esim._sigma_xy(table)) if table else []
        if not options:
            current.update(fallback=True,fallbackResult='full_support_pose_search',unconfirmed=True)
            # Retain COM protection: ordinary alternatives require the entire load footprint.
            if not current.get('observed') or angle is None:
                break
        if current.get('used') and 'offset' in current:
            avoided.append(dict(on=record['reference']['support'],offset=current['offset'],yaw=current['yaw']))
        child={};job.postcondition=child;failure=None;attempts+=1
        try:
            controller.flip(job,dict(plan,turnOver=False,_edgeAdjusted=True,_edgeAvoid=avoided,_edgeLevelSearch=not options))
        except SkillFailed as exc:
            controller.check()
            failure=str(exc)
            if controller.rt.estimator.held_id: raise
            if not child.get('releaseExecuted'):
                current.update(fallbackResult='adjustment_failed',adjustmentFailure=failure,unconfirmed=True)
                break
        finally:
            job.postcondition=post
        result=child['edgePlacement']
        if failure:
            result['adjustmentFailure']=failure
        history.append(dict(used=result['used'],tiltDeg=result.get('estimatedTiltDeg'),
            translationM=float(np.linalg.norm(np.asarray(records[plan['body']]['reference']['position'][:2])-start)),
            source=result.get('source')))
        post['edgePlacement']=dict(result,adjustment='one_body_reposition' if len(history)==1 else 'bounded_body_reposition',
            fallback=True,fallbackResult='used' if result['used'] else 'not_admitted')
        post['reference']=child['reference'];post.setdefault('stages',[]).extend(child.get('stages',[]))
        # Missing pixels do not cause repeated blind moves. Finish this best estimate unconfirmed.
        if failure or result.get('estimatedTiltDeg') is None or result.get('level') is True: break
        if (not result['used'] and options) or not result.get('patch'): break
    final=post['edgePlacement']
    # A worse trial is not the selected result. Restore the best measured safe
    # edge pose once; it must pass current geometry, collision and reach admission.
    restored = False
    if (best and final.get('estimatedTiltDeg') is not None
            and final['estimatedTiltDeg'] > best[0] and history):
        record = records[plan['body']]
        old_shape = deepcopy(record.get('edgeGeometry'))
        record['edgeGeometry'] = (rotated(best[1]['edgeGeometry'],
            record['reference']['yaw']-best[1]['reference']['yaw']) if best[1].get('edgeGeometry') else None)
        if record['edgeGeometry']:
            record['edgeGeometry']['turnOver'] = False
        record['edgeClearanceExtraM'] = best[1].get('edgeClearanceExtraM', 0.)
        from steward.ego.fasten_simple import offset_on
        table = controller.rt.ego_state().find(best[1]['reference']['support'])
        target = dict(on=best[1]['reference']['support'],
            offset=offset_on(table, best[1]['reference']['position'][:2]) if table else None,
            yaw=best[1]['reference']['yaw'], assemblyEdge=bool(best[2].get('used')))
        safe = ((edge.validate(record, table, target['offset'], controller.rt.esim._sigma_xy(table), target['yaw'])
            if target['assemblyEdge'] else True) if table else None)
        if safe:
            child={};job.postcondition=child;attempts+=1
            try:
                controller.flip(job,dict(plan,turnOver=False,_edgeAdjusted=True,_edgeRestore=target))
                final=post['edgePlacement']=dict(child['edgePlacement'],adjustment='restore_best_measured_pose',
                    fallback=True,fallbackResult='used' if child['edgePlacement']['used'] else 'not_admitted')
                post['reference']=child['reference'];post['stages'].extend(child.get('stages',[]))
                restored = bool(child.get('releaseExecuted') and np.allclose(
                    child['reference']['position'][:2], best[1]['reference']['position'][:2], atol=1e-6) and
                    abs((child['reference']['yaw']-target['yaw']+math.pi) % (2*math.pi)-math.pi)<1e-6)
            except SkillFailed as exc:
                controller.check()
                if controller.rt.estimator.held_id: raise
                if child.get('releaseExecuted'):
                    final=post['edgePlacement']=dict(child['edgePlacement'])
                    post['reference']=child['reference'];post['stages'].extend(child.get('stages',[]))
                final['restoreFailure']=str(exc)
            finally:
                job.postcondition=post
        else:
            record['edgeGeometry']=old_shape
            final['restoreFailure']='best_pose_no_longer_supported'
    final.update(bestMeasuredTiltDeg=min([h['tiltDeg'] for h in history if h['tiltDeg'] is not None]+([initial] if initial is not None else [])
        +([final['estimatedTiltDeg']] if final.get('estimatedTiltDeg') is not None else []), default=None),
        restoredBestPose=restored, tiltBefore=initial,tiltAfter=final.get('estimatedTiltDeg'),
        finalTiltDeg=final.get('estimatedTiltDeg'),adjustments=history,
        adjustmentTranslationM=float(np.linalg.norm(np.asarray(records[plan['body']]['reference']['position'][:2])-start)),
        unconfirmed=not final.get('observed',False))
    post['levelAlignment']['adjustmentAttempts']=attempts
    post['edgePlacementBeforeAdjustment']={k:original.get(k) for k in ('used','source','estimatedTiltDeg','reason')}
