"""Supply identity, grip belief and extent sanity from robot evidence only."""
from steward.ego import capabilities
from copy import deepcopy
from functools import wraps
import math
import numpy as np


def enabled():
    return capabilities.get('STEWARD_ASSEMBLY_SUPPLY_RELIABILITY') != '0'


def holding_enabled():
    return capabilities.get('STEWARD_ASSEMBLY_SUPPLY_RELIABILITY_HOLDING') != '0'


def size_enabled():
    return capabilities.get('STEWARD_ASSEMBLY_SUPPLY_RELIABILITY_SIZE') != '0'


def role_enabled():
    return capabilities.get('STEWARD_ASSEMBLY_SUPPLY_RELIABILITY_ROLE') != '0'


def cross_support_enabled():
    return False  # Retired category-only cross-surface supply; sessions use bound ownership.


def size_issue(category,size):
    from steward.ego.classify import SHAPES
    a=np.asarray(size,float)
    if a.shape!=(3,) or not np.isfinite(a).all() or (np.any(a<0) or max(a)<=0):
        return dict(reason='invalid_extent',remeasure=True)
    shape=SHAPES.get(category)
    if not shape:return None
    diagonal=float(np.linalg.norm(shape['dims'])*max(shape.get('scale',(1.,1.))))
    ratio=float(max(a)/diagonal) if diagonal>0 else math.inf
    if ratio>10:
        return dict(reason='extent_exceeds_catalogue_order',remeasure=True,extentRatio=round(ratio,3))
    return None


def reject(est,track,size,stage,category=None):
    if not size_enabled():return False
    issue=size_issue(category or track.category,size)
    from steward.ego.capability_usage import record
    record('size_guard', entity=track.id if track else None, stamp=est.stamp, stage=stage,
           measured=True, action=bool(issue), reason=issue['reason'] if issue else None)
    if not issue:return False
    row=dict(issue,entity=track.id if track else None,stage=stage,observedAt=est.stamp)
    if track is not None:track.geometry_check=row
    est.geometry_rejections=(getattr(est,'geometry_rejections',[])+[row])[-8:]
    return True


def guard_fusion(fn):
    """Check the raw frame before mutation and the final union before accepting it.

    Only small geometry metadata is copied. Rejected union clouds are discarded
    for remeasurement; no large point-cloud copy in the normal path.
    """
    @wraps(fn)
    def guarded(est,track,m,*args,**kwargs):
        if not size_enabled():return fn(est,track,m,*args,**kwargs)
        category=track.category
        if reject(est,track,m.get('size',track.size),'raw_frame'):return
        keys=('pos','size','yaw','var','yaw_var','category_counts','observations','last_seen')
        saved={k:deepcopy(getattr(track,k)) for k in keys}
        result=fn(est,track,m,*args,**kwargs)
        if reject(est,track,track.size,'fused_extent',category):
            for k,v in saved.items():setattr(track,k,v)
            track.voxels=track.span=None
            track.recent.clear()
        elif track.last_seen==est.stamp and track.observations>saved['observations'] and getattr(track,'geometry_check',None):
            track.geometry_check=dict(reason='remeasured',remeasure=False,observedAt=est.stamp)
        return result
    return guarded


def grip_evidence(gripper):
    """Positive tactile holding wins over a missing image, never over an open hand.

    Optional force/aperture signals need their sensor-supplied zero and error
    bounds. Without those, closed fingers alone cannot establish a held load.
    """
    if gripper.get('feedbackMissing'):return None,'gripper_feedback_missing'
    if gripper.get('state')=='open':return False,'gripper_open'
    if gripper.get('holding') is True:return True,'holding_feedback'
    keys=('forceN','forceNoiseN','apertureM','closedApertureM','apertureUncertaintyM')
    if all(k in gripper for k in keys):
        values=[gripper[k] for k in keys]
        if all(isinstance(v,(int,float)) and math.isfinite(v) for v in values):
            f,noise,width,zero,error=values
            if noise>=0 and error>=0 and f>noise and width>zero+error:
                return True,'force_and_aperture_above_sensor_bounds'
    if gripper.get('holding') is False:return False,'empty_grip_feedback'
    return None,'gripper_feedback_unknown'


def retain(est,track,reason):
    if not holding_enabled() or track is None:return False
    held,source=grip_evidence(est.gripper)
    row=dict(entity=track.id,status='held' if held is True else 'unknown' if held is None else 'empty',
        source=source,trigger=reason,observedAt=est.stamp,visualConfirmation=False)
    est.holding_belief=row
    from steward.ego.capability_usage import record
    record('holding_belief', entity=track.id, stamp=est.stamp, stage=reason,
           measured=held is not None, action=held is not False,
           reason=source if held is not True else None)
    return held is not False


def preflight(controller,job,target):
    """Verify category and observed extents before invoking any grasp motion."""
    if not (role_enabled() or size_enabled()):return
    from steward.ego.priors import prior
    from steward.ego.fasten_identity import check
    from steward.ego.fasten_plan import value
    from steward.ego.skills import SkillFailed
    scene=controller.rt.ego_state();entity=scene.find(target)
    live=getattr(controller.rt,'perception_mode',None) is not None
    if live and entity is not None and (entity.pose.t is None or not 0<=scene.stamp-entity.pose.t<=20):
        from steward.ego.assembly_precision import acquire
        acquire(controller,[target])
        scene=controller.rt.ego_state();entity=scene.find(target)
    reason=None;source=None;stamp=None
    if entity is None:reason='target_absent'
    elif role_enabled() and prior(entity.category.value).get('assemblyRole')!='fastener':reason='target_not_a_fastener'
    else:
        est=controller.rt.estimator
        from steward.ego.fasten_accounting import departure
        attempts=[a for a in est.fastening_attempts.values()
            if not (target in (a.get('screw'),*a.get('screwAliases',[])) and departure(scene,a))]
        result=check(entity,scene.stamp,attempts,
            list(getattr(est,'released_interfaces',{}).values()),members=getattr(controller,'_supply_members',()))
        if role_enabled() and result['status']=='contradicted':reason=result['reason']
        track=est.tracks.get(entity.id)
        if size_enabled() and getattr(track,'geometry_check',{}).get('remeasure'):reason='extent_pending_remeasurement'
        view=value(entity,'contactView')
        measured=bool(view and view.get('source')=='rgbd_segment' and not view.get('truncated',True)
            and view.get('invalidDepthFraction',1)<=.05 and 0<=scene.stamp-view.get('stamp',-1)<=20)
        if measured:
            size=view.get('visibleSizeM',[]);source='head_rgbd_extent';stamp=view['stamp']
        elif track is not None and track.observations>0 and (not live or 0<=scene.stamp-track.last_seen<=20):
            # A stable object's last measured dimensions remain identity evidence.
            # This does not claim a fresh camera frame or a measured grasp.
            size=track.size;source='remembered_head_extent';stamp=track.last_seen
        else:size=[];reason=reason or ('no_observed_extent' if size_enabled() else None)
        if reason is None and size_enabled():
            from steward.ego.classify import SHAPES
            shape=SHAPES.get(entity.category.value)
            a=np.asarray(size,float)
            if a.shape!=(3,) or not np.isfinite(a).all() or min(a)<=0 or shape is None:
                reason='no_observed_extent'
            else:
                bound=float(np.linalg.norm(shape['dims'])*max(shape.get('scale',(1.,1.)))+shape.get('extent_slack',0.))
                # The existing two-sigma perception uncertainty allowance is kept.
                if track is not None:bound+=2*math.sqrt(max(0.,float(max(track.var))))
                if max(a)>bound:reason='measured_extent_exceeds_category_prior'
    unknown = capabilities.get('STEWARD_ASSEMBLY_SUPPLY_UNKNOWN_EXTENT') != '0' and reason == 'no_observed_extent'
    diagnostic_reason = reason
    if unknown:
        reason = None  # absence of measurement is not contradictory identity evidence
    row=dict(entity=target,status='unknown' if unknown else 'rejected' if reason else 'checked',
        reason=diagnostic_reason,source=source,
        observedAt=stamp,fresh=bool(stamp is not None and 0<=scene.stamp-stamp<=20))
    job.postcondition.setdefault('pregraspChecks',[]).append({k:v for k,v in row.items() if v is not None})
    if reason:
        controller._missed_screws=getattr(controller,'_missed_screws',set())|{target}
        raise SkillFailed('grasp mismatch: pregrasp_'+reason)


def publish(controller,job):
    if not (holding_enabled() or size_enabled() or role_enabled()):return
    est=controller.rt.estimator
    rows=[r for r in getattr(est,'geometry_rejections',[]) if r['observedAt']>=job.started]
    if rows:job.postcondition['geometryRejections']=deepcopy(rows)
    belief=getattr(est,'holding_belief',None)
    if belief and belief['observedAt']>=job.started:job.postcondition['holdingBelief']=deepcopy(belief)
