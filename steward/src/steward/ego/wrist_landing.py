"""Release -> observe miss -> estimate offset; robot RGB-D/FK evidence only.

No extra pick, camera tour, or retry: reuse the existing one-screw replay and its
release frames. Bias is an estimate, never an observed held endpoint/attachment.
"""
from copy import deepcopy
import math
import time
from steward.ego import action_budget as ab
import numpy as np
from steward.ego import wrist_observation as wo
from steward.ego.fasten_target import spacing

BUDGET_S = 6.  # total feedback computation across first release and one replay
MAX_UNCERTAINTY_M = .006
MIN_BIAS_M = .004
MAX_ANGLE_DEG = 15.
MAX_NORM_DIFF_M = .003
MAX_RELATIVE_DIFF = .25


def enabled(controller):
    return wo.enabled(controller.rt, 'LANDING')


def bounded(vector, bound):
    v=np.asarray(vector,float)
    return v*min(1.,bound/max(float(np.linalg.norm(v)),1e-12))


def scope(attempt):
    frame=attempt.get('componentFrame') or {}
    return (attempt.get('body'),attempt.get('part'),frame.get('releasedAt'),frame.get('alignmentMovedAt'))


def common_bias(first, second):
    """15 degrees, <=3 mm AND <=25% norm difference; use shared smaller magnitude."""
    a,b=np.asarray(first,float),np.asarray(second,float)
    na,nb=np.linalg.norm(a),np.linalg.norm(b)
    if not np.isfinite([*a,*b]).all() or min(na,nb)<MIN_BIAS_M: return None
    if abs(na-nb)>min(MAX_NORM_DIFF_M,MAX_RELATIVE_DIFF*min(na,nb)): return None
    if float(a @ b/(na*nb))<math.cos(math.radians(MAX_ANGLE_DEG)): return None
    direction=a/na+b/nb
    return direction/np.linalg.norm(direction)*min(na,nb)


def record(controller, attempt, row):
    """Consecutive releases (including replay) only; unknown breaks the streak."""
    est=controller.rt.estimator
    history=getattr(est,'_wrist_landing_history',[])
    key=(attempt['screw'],attempt['seatedAt'])
    if any(h['release']==key for h in history): return
    item=dict(release=key,scope=scope(attempt),stamp=attempt['seatedAt'],
              bias=row.get('biasEstimateXYM') if row.get('status')=='measured' else None,
              error=row.get('errorXYM') if row.get('status')=='measured' else None)
    est._wrist_landing_history=(history+[item])[-2:]


def observe(controller, job, attempt, sample, feature):
    if not enabled(controller): return None
    start=time.monotonic();used=attempt.get('_landingSimS' if ab.enabled() else '_landingWallS',0.)
    seconds=ab.frame_timeout(controller.ex,BUDGET_S,2)
    deadline=ab.make(controller.ex,max(0.,seconds-used),'remaining two-frame feedback window')
    row=dict(status='unknown',reason='no_fresh_visible_miss',budgetS=seconds,
             replay=bool(attempt.get('seatRetryCount')),source='same_wrist_frame_landing_and_empty_hole')
    attempt['landingFeedback']=row
    previous=(attempt.get('precision') or {}).get('landingCompensation') or {}
    if previous:
        row['compensation']=deepcopy(previous)
    try:
        if used>=seconds:
            row['reason']='feedback_wall_budget';return row
        if not sample:
            row['reason']='release_frame_unavailable';return row
        if not feature:
            row['reason']='released_shaft_unresolved';return row
        if feature.get('classification')=='lying':
            row.update(reason='lying_position_only_no_directed_landing_axis',landingPointM=feature.get('tip'));return row
        if attempt.get('seatCheck',{}).get('passed') is not False:
            row['reason']='no_independently_observed_miss';return row
        stamp=sample[0]['header']['stamp']
        now=controller.ex._state()['stamp']
        if not max(attempt['seatedAt'],attempt.get('transportedAt',0))<stamp<=now:
            row['reason']='release_frame_not_fresh';return row
        if feature.get('stamp')!=stamp or now-stamp>2.:
            row['reason']='shaft_frame_not_synchronized';return row
        ctx=getattr(controller,'_registered_alignment',{})
        reg=ctx.get('registration')
        if not reg or reg.get('entity')!=attempt['part'] or reg.get('body')!=attempt['body']:
            row['reason']='no_matching_registration';return row
        from steward.ego.registered_alignment import invalid_reason
        if invalid_reason(controller,reg):
            row['reason']='target_moved';return row
        rows=reg.get('layout',reg.get('holes',[]));distance=spacing(rows)
        if distance is None:
            row['reason']='unknown_hole_spacing';return row
        from steward.ego.wrist_fasten import select_hole
        from steward.ego.fasten_target import evidence
        occupied=[o for o in evidence(controller.rt.estimator,reg['frame']) if o.get('attempt') is not attempt
                  and o.get('attempt',{}).get('screw')!=attempt['screw']]
        found=wo.holes(sample,attempt['entry'])
        selected=select_hole(found,rows,occupied,attempt['hole'])
        if selected is None or selected['bodyHole']!=attempt['hole']:
            row['reason']='target_empty_hole_unseen_or_ambiguous';return row
        uncertainty=float(feature.get('uncertaintyM',.002))+float(selected.get('uncertaintyM',.004))
        if not math.isfinite(uncertainty) or uncertainty>MAX_UNCERTAINTY_M:
            row['reason']='uncertain_landing';return row
        tip=np.asarray(feature['tip'],float);hole=np.asarray(selected['point'],float)
        delta=tip[:2]-hole[:2]
        if not np.isfinite([*tip,*hole]).all() or np.linalg.norm(delta)>.05:
            row['reason']='landing_outside_local_identity_window';return row
        prior=np.asarray(previous.get('biasXYM',[0.,0.])) if previous.get('applied') else np.zeros(2)
        estimate=bounded(delta+prior,distance/2)
        if ab.expired(deadline):
            row['reason']='feedback_wall_budget';return row
        row.update(status='measured',reason=None,observedAt=stamp,screw=attempt['screw'],
            landingPointM=tip.tolist(),targetPointM=hole.tolist(),errorXYM=delta.tolist(),
            biasEstimateXYM=estimate.tolist(),correctionXYM=(-estimate).tolist(),
            boundM=distance/2,holeSpacingM=distance,uncertaintyM=uncertainty,
            axisTiltDeg=attempt['seatCheck'].get('axisTiltDeg'),
            estimateKind='release_error_proxy_for_grasp_bias',attachmentObserved=False)
        row['pickupPoseUpdated']=update_pickup(controller,attempt,feature,uncertainty)
        if previous.get('applied'):
            row['residualAfterReplayM']=float(np.linalg.norm(delta))
    except (ValueError, KeyError, TypeError, np.linalg.LinAlgError):
        row.update(status='unknown',reason='invalid_rgbd_measurement')
        return row
    finally:
        attempt['_landingWallS']=attempt.get('_landingWallS',0.)+time.monotonic()-start
        if ab.enabled():
            attempt['_landingSimS']=used+deadline.sim_s
            row.update(ab.report(deadline))
        row['wallS']=round(attempt['_landingWallS'],4)
        record(controller,attempt,row)
        job.postcondition['landingFeedback']=deepcopy(row)
    return row



def update_pickup(controller, attempt, feature, uncertainty):
    """Keep the same tracked identity, but pick its observed post-release pose."""
    from steward.math3d import matrix_from_quat, quat_from_yaw
    est=controller.rt.estimator
    with controller.rt._lock:
        track=est.tracks.get(attempt['screw'])
        if track is None or track.held or track.last_seen>feature['stamp']: return False
        tip,head=np.asarray(feature['tip']),np.asarray(feature['head'])
        delta=head-tip;length=float(np.linalg.norm(delta))
        if length<=0 or feature.get('radiusM',0)<=0: return False
        yaw=math.atan2(delta[1],delta[0]) if np.linalg.norm(delta[:2])>.003 else track.yaw
        local=matrix_from_quat(quat_from_yaw(yaw)).T @ delta
        radial=2*feature['radiusM']*np.sqrt(np.maximum(0,1-(local/length)**2))
        track.pos=(tip+head)/2;track.yaw=yaw
        track.size=np.maximum(.002,abs(local)+radial)
        track.var=np.full(3,max(.002,uncertainty)**2)
        track.last_seen=feature['stamp'];track.observations+=1;track.settled=True
        track.voxels=track.span=None
        track.tool_feature=deepcopy(feature)
        est.version+=1
    return True


def prepare(controller, plan, *, replay=False):
    """Bind a candidate to the actually acquired identity and interface epoch."""
    controller._wrist_landing_pending=None
    if not enabled(controller): return
    est=controller.rt.estimator;reg=getattr(controller,'_registered_alignment',{}).get('registration')
    if (not reg or reg.get('invalidated') or reg.get('entity')!=plan.get('part')
            or reg.get('body')!=plan.get('body')): return
    candidate=None;source=None
    if replay:
        row=plan.get('landingFeedback') or {}
        if (row.get('status')=='measured' and row.get('observedAt',-1)>plan['seatedAt']
                and est.held_id in {plan['screw'],*plan.get('screwAliases',[])}
                and scope(plan)==(reg['body'],reg['entity'],reg['frame'].get('releasedAt'),reg['frame'].get('alignmentMovedAt'))):
            candidate=row['biasEstimateXYM'];source='same_fastener_landing'
    else:
        history=getattr(est,'_wrist_landing_history',[])
        current_scope=(reg['body'],reg['entity'],reg['frame'].get('releasedAt'),reg['frame'].get('alignmentMovedAt'))
        if (len(history)==2 and all(h['scope']==current_scope and h['bias'] is not None for h in history)
                and est.held_id not in {h['release'][0] for h in history}):
            if common_bias(history[0]['error'],history[1]['error']) is not None:
                candidate=common_bias(history[0]['bias'],history[1]['bias']);source='two_consecutive_landings'
    if candidate is None: return
    distance=spacing(reg.get('layout',reg.get('holes',[])))
    if not distance: return
    controller._wrist_landing_pending=dict(biasXYM=bounded(candidate,distance/2).tolist(),
        source=source,object=est.held_id,heldSince=est.held_since,scope=(reg['body'],reg['entity'],
        reg['frame'].get('releasedAt'),reg['frame'].get('alignmentMovedAt')),boundM=distance/2,
        budgetS=BUDGET_S,consistencyAngleDeg=MAX_ANGLE_DEG,
        consistencyNormDifferenceM=MAX_NORM_DIFF_M,consistencyRelativeDifference=MAX_RELATIVE_DIFF)


def apply_hover(controller, current, report):
    """Translate the inferred grasp endpoints; ordinary endpoint planning consumes it.

    A fresh direct tip measurement wins. Do not shift the measured target hole or
    label this inferred offset as an observed held tip.
    """
    if not enabled(controller): return False
    ctx=getattr(controller,'_registered_alignment',{})
    if report.get('landingCompensation'):
        if report.get('_wristMeasured') or report.get('seatDelivery',{}).get('measurement')=='measured_tip':
            report['landingCompensation'].update(status='superseded',reason='fresh_held_tip_wins',applied=False)
        return False
    pending=getattr(controller,'_wrist_landing_pending',None)
    if not pending: return False
    est=controller.rt.estimator;ctx=getattr(controller,'_registered_alignment',{});reg=ctx.get('registration')
    from steward.ego.registered_alignment import invalid_reason
    if (not reg or invalid_reason(controller,reg)
            or pending['scope']!=(reg['body'],reg['entity'],reg['frame'].get('releasedAt'),reg['frame'].get('alignmentMovedAt'))
            or est.held_id!=pending['object']
            or current['object']!=est.held_id or est.held_since!=pending['heldSince']
            or ctx.get('tipHeldSince')!=est.held_since or not ctx.get('tip')): return False
    if report.get('_wristMeasured') or report.get('seatDelivery',{}).get('measurement')=='measured_tip':
        report['landingCompensation']=dict(status='skipped',reason='fresh_held_tip_wins',applied=False)
        return False
    distance=spacing(reg.get('layout',reg.get('holes',[])))
    if not distance: return False
    bias=bounded(pending['biasXYM'],distance/2)
    state=controller.ex._state()
    rotation=est.gripper_rotation(est.held_gripper,state['joints'],state['odometry'])
    local=rotation.T @ np.r_[bias,0.]
    feature=deepcopy(ctx['tip'])
    for key in ('tip','head'):
        feature[key]=(np.asarray(feature[key])+local).tolist()
    feature.update(source='landing_bias_estimated',uncertaintyM=max(.004,feature.get('uncertaintyM',0.)))
    with controller.rt._lock:
        est.tracks[est.held_id].held_tool_feature=deepcopy(feature)
        ctx.update(tip=feature,tipOffsetSource='landing_bias_estimated',
                   correctionPoint=deepcopy(reg.get('executionPoint',reg['point'])))
        est.version+=1
    report.update(tipOffsetSource='landing_bias_estimated',tipOffsetMeasured=False)
    report['landingCompensation']={k:deepcopy(v) for k,v in pending.items() if k not in ('scope','object','heldSince')}
    report['landingCompensation'].update(status='estimated',applied=False,biasXYM=bias.tolist(),
        correctionXYM=(-bias).tolist(),boundM=distance/2,heldOffsetEstimateM=local.tolist(),
        targetPointM=deepcopy(reg['point']))
    return True
