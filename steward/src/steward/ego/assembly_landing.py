"""Independent post-release support checks for every assembly path."""
from steward.ego import capabilities
from copy import deepcopy
import math
import numpy as np
from steward.ego.fasten_plan import value
from steward.ego.planner import Box2D, rectangle_separation, entity_box

ENABLED = capabilities.get('STEWARD_ASSEMBLY_COMPONENT_LANDING_CHECK') != '0'


def same_destination(mate, support):
    return mate is support or bool(ENABLED and mate is not None and support is not None and mate.id == support.id)


def fresh(view, after, stamp):
    return bool(view and view.get('source') == 'rgbd_segment' and
        after < view.get('stamp', -1) <= stamp and stamp-view['stamp'] <= 20 and
        not view.get('truncated', True) and view.get('invalidDepthFraction',1) <= .05)


def assess(controller, plan, frame):
    result = dict(status='unknown', passed=None, reason='no_independent_support_evidence')
    if not ENABLED or not frame:
        return result
    scene=controller.rt.ego_state();est=controller.rt.estimator
    part=scene.find(frame['entity']);body=scene.find(frame['body'])
    pv=value(part,'contactView')
    if not fresh(pv,plan['releasedAt'],scene.stamp) or body is None:
        return result
    size=np.asarray(pv.get('visibleSizeM',[]),float)
    expected=np.asarray(frame['size'],float)
    if size.shape!=(3,) or not np.allclose(np.sort(size[:2]),np.sort(expected[:2]),atol=.015):
        return dict(result,reason='released_component_mask_partial_or_merged')
    track=est.tracks.get(body.id)
    bv=(getattr(track,'manipulation_view',None) or getattr(track,'assembly_observation',None)
        or value(body,'contactView'))
    ref=(getattr(est,'stationary_assemblies',None).records.get(body.id,{}) or {}).get('reference',{})
    if not fresh(bv,ref.get('ownReleaseAt',plan['releasedAt']-20),scene.stamp):
        return result
    bsize=np.asarray(bv.get('visibleSizeM',[]),float)
    reference=np.asarray(ref.get('size',bsize),float)
    if bsize.shape!=(3,) or reference.shape!=(3,) or not np.allclose(
            np.sort(bsize[:2]),np.sort(reference[:2]),atol=.015):
        return dict(result,reason='body_mask_partial_or_merged')
    pp=np.asarray(pv['centreM']);bp=np.asarray(bv['centreM'])
    upper=Box2D(part.id,*pp[:2],pv.get('boxYawRad',0.),*size[:2]/2)
    lower=Box2D(body.id,*bp[:2],bv.get('boxYawRad',0.),*bsize[:2]/2)
    separation=rectangle_separation(upper,lower)
    # Existing assembly placement tolerance plus independent outline error.
    margin=float(plan.get('placementToleranceM',.02))+float(pv.get('uncertaintyM',.015))+float(bv.get('uncertaintyM',.015))
    result.update(source='independent_component_and_body_views', component=part.id, body=body.id,
        observedAt=pv['stamp'], bodyObservedAt=bv['stamp'], footprintSeparationM=separation,toleranceM=margin)
    if separation>margin:
        return dict(result,status='fail',passed=False,reason='released_component_outside_body')
    table=scene.find(ref.get('support',''))
    if table is None or table.id==body.id:
        return result
    if table.pose.src not in ('observed','fused') or table.pose.t is None or not 0<=scene.stamp-table.pose.t<=20:
        return result
    tb,_,table_top=entity_box(table)
    points=np.asarray(pv.get('points',[]),float);body_points=np.asarray(bv.get('points',[]),float)
    if (points.ndim!=2 or points.shape[1]!=3 or len(points)<30 or not np.isfinite(points).all()
            or body_points.ndim!=2 or body_points.shape[1]!=3 or len(body_points)<30 or not np.isfinite(body_points).all()):
        return result
    bottom=float(np.quantile(points[:,2],.01));part_top=float(np.quantile(points[:,2],.99))
    body_top=float(np.quantile(body_points[:,2],.99))
    # A component touching the outer support below the body's top cannot be
    # resting on that body's upper face, even if their XY projections overlap.
    height_margin=.012+float(pv.get('uncertaintyM',0.))+float(bv.get('uncertaintyM',0.))
    result.update(outerSupport=table.id,componentBottomM=bottom,componentTopM=part_top,
                  outerSupportTopM=table_top,bodyTopM=body_top,heightToleranceM=height_margin)
    if (tb.contains(*pp[:2]) and abs(bottom-table_top)<=height_margin and
            part_top<body_top-height_margin):
        return dict(result,status='fail',passed=False,reason='component_on_outer_support_not_body')
    return result


def post_release(controller, job, plan, params):
    """Readiness runs without a precision report, under a short capture budget."""
    if (not ENABLED or plan.get('_landingChecked') or not plan.get('releaseExecuted')
            or params.get('seatBody') or params.get('flip') or params.get('assemblyBody')):
        return
    target=plan.get('placementTarget') or {}
    if not target.get('mateReference'):
        return
    from steward.ego.fasten_sequence import budget, DriveExpired
    from steward.ego.assembly_precision import acquire
    from steward.ego.placement_ready import post_release as check_release
    import time
    from steward.ego import action_budget as ab
    started=time.monotonic();capture=dict(budgetS=8.,timedOut=False)
    deadline=ab.make(controller.ex,ab.observation(controller.ex) if ab.enabled() else 8.,'one existing registration frame window')
    try:
        with budget(deadline):
            acquire(controller,[plan['object'],controller.body_id])
    except DriveExpired:
        capture['timedOut']=True
    capture['wallS']=time.monotonic()-started
    capture.update(ab.report(deadline, compact=True))
    # Mark before readiness, whose bounded replay may call this hook again.
    plan['_landingChecked']=True
    try:
        check_release(controller,job,plan,params)
    finally:
        report=plan.get('placementReadiness') or job.postcondition.get('placementReadiness')
        if report:
            report['acquisitionBudget']=capture
            report['acquisition']=deepcopy(getattr(controller,'_precision_acquisition',{}))
            if report.get('passed') is False:
                saved=controller.rt.estimator.released_interfaces.get(plan['object'])
                if saved is not None:
                    saved['placementFailure']={k:deepcopy(report[k]) for k in
                        ('status','passed','reason','destinationCheck') if k in report}
