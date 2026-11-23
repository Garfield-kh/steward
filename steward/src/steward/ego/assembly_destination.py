"""Keep mating releases on their intended support using robot geometry only."""
from steward.ego import capabilities
import math
import numpy as np
from steward.ego.fasten_plan import value

ENABLED = capabilities.get('STEWARD_ASSEMBLY_DESTINATION_CHECK') != '0'


def assess(plan, scene=None, expected=None):
    target = plan.get('placementTarget') or {}
    ref = target.get('mateReference') or {}
    result = dict(status='unknown', passed=None, reason='no_mating_destination_geometry')
    if not ENABLED or plan.get('seatTarget') or not ref or not target.get('planar'):
        return result
    body = expected or ref['entity']
    result.update(body=body, support=plan.get('support'), source='ego_planned_geometry')
    if ref['entity'] != body or plan.get('support') not in (None, body):
        return dict(result, status='failed', passed=False, reason='mating_support_substituted')
    size = target.get('sizeM'); spot = plan.get('spot')
    tolerance = plan.get('placementToleranceM')
    if not size or spot is None or tolerance is None:
        return result
    uncertainty = 0.
    if scene:
        for eid in (body, plan.get('object')):
            entity = scene.find(eid)
            cov = (entity.pose.unc or {}).get('cov', []) if entity else []
            if len(cov) == 36:
                uncertainty += 2*math.sqrt(max(0., cov[0], cov[7], cov[14]))
    margin = float(tolerance)+uncertainty
    # Separating axes allow overhanging parts; require some actual overlap of
    # the bare support and part, not just membership in an expanded assembly box.
    def rotation(yaw):
        c,s=math.cos(yaw),math.sin(yaw)
        return np.array([[c,-s],[s,c]])
    rb, rp = rotation(ref.get('yaw', 0.)), rotation(target.get('yawRad', 0.))
    delta = np.asarray(spot)[:2]-np.asarray(ref['position'])[:2]
    gap = max(float(abs(axis @ delta)-abs(axis @ rb) @ (np.asarray(ref['size'])[:2]/2)
                    -abs(axis @ rp) @ (np.asarray(size)[:2]/2)) for axis in (*rb.T,*rp.T))
    result.update(footprintSeparationM=gap, uncertaintyM=uncertainty, toleranceM=float(tolerance))
    if gap > margin:
        return dict(result,status='failed',passed=False,reason='component_target_outside_body_footprint')
    surface = plan.get('placementSurface') or target.get('placementSurface')
    settled = target.get('releasedFrame', {}).get('settledCentreZM')
    if surface and surface.get('observed') and surface.get('patch') and settled is not None:
        from steward.ego.place_surface import valid, height_at
        if scene is None or valid(surface, ref, scene.stamp):
            height = height_at(surface['patch'], spot)
            bottom = float(settled)-float(size[2])/2
            vertical_margin = margin+surface.get('uncertaintyM', 0.)
            result.update(surfaceHeightM=height, settledBottomM=bottom)
            if bottom < height-vertical_margin:
                return dict(result,status='failed',passed=False,reason='component_target_below_observed_body_surface')
    return dict(result, reason='no_destination_contradiction')


def require(controller, job, plan, params):
    if not ENABLED or params.get('seatBody') or params.get('assemblyBody') or params.get('flip'):
        return
    if not (plan.get('placementTarget') or {}).get('mateReference'):
        return
    check = assess(plan, controller.rt.ego_state(), getattr(controller, 'body_id', None))
    plan['supportTargetCheck'] = check
    job.postcondition['supportTargetCheck'] = check
    if check['passed'] is False:
        from steward.ego.skills import SkillFailed
        raise SkillFailed('assembly destination rejected: '+check['reason'])
    from steward.ego.postflip_destination import require as require_postflip
    require_postflip(controller.rt, plan, job)


def validate(rt, plan):
    if not ENABLED or plan.get('seatTarget') or not (plan.get('placementTarget') or {}).get('mateReference'):
        return
    ref = plan['placementTarget']['mateReference']
    # Explicit ordinary placement on a separate surface with alignTo is an
    # existing public operation, not an assemble_onto destination substitution.
    if plan.get('support') not in (None, ref['entity']) and 'supportTargetCheck' not in plan:
        return
    check = assess(plan, rt.ego_state())
    record = rt.estimator.stationary_assemblies.records.get(ref['entity'], {})
    moved = record.get('displacementEvidence') or {}
    since = max(ref.get('registeredAt') or 0., ref.get('ownReleaseAt') or 0.)
    # This evidence is already independently gated by stationary_assembly;
    # old pre-flip invalidation and absent measurements remain non-blocking.
    from steward.ego.stationary_assembly import FRESH_VIEW_S
    last, count = record.get('motionSamples', {}).get('body', (-math.inf, 0))
    post_release_displacement = count >= 2 and last > since and 0 <= rt.estimator.stamp-last <= FRESH_VIEW_S
    if (record.get('hardMotion') and moved.get('stamp', -math.inf) > since) or post_release_displacement:
        check.update(status='failed', passed=False, reason='body_moved_after_destination_registration',
                     motionAt=last if post_release_displacement else moved['stamp'])
    plan['supportTargetCheck'] = check
    if check['passed'] is False:
        from steward.ego.skills import SkillFailed
        raise SkillFailed('assembly destination rejected: '+check['reason'])
    from steward.ego.postflip_destination import require as require_postflip
    require_postflip(rt, plan)


def refresh_contradiction(controller, plan, fresh):
    """A refused optional replan alone says nothing about the old destination.

    Retain an admitted fallback only when fresh geometry has not disproved it.
    """
    if not ENABLED:
        return None
    check = fresh.get('plan', {}).get('supportTargetCheck')
    if check and check.get('passed') is False:
        return check
    scene = controller.rt.ego_state()
    check = assess(plan, scene, getattr(controller, 'body_id', None))
    if check['passed'] is False:
        return check
    from steward.ego.fasten_geometry import surface_patch
    body = scene.find(getattr(controller, 'body_id', None))
    view = value(body, 'contactView')
    if (not view or view.get('source') != 'rgbd_segment' or view.get('truncated', True)
            or view.get('invalidDepthFraction', 1) > .05
            or not 0 <= scene.stamp-view.get('stamp', -1) <= 20):
        return None
    reference = plan.get('placementTarget', {}).get('mateReference', {})
    if view['stamp'] <= max(reference.get('registeredAt') or 0., reference.get('ownReleaseAt') or 0.):
        return None
    patch = surface_patch(view.get('points', []), plan['spot'], max_tilt_deg=25)
    old = plan.get('lowerSeatPlacement') or {}
    height = old.get('releaseTopM', old.get('nominalTopM'))
    tolerance = plan.get('placementToleranceM')
    if patch is None or height is None or tolerance is None:
        return None
    if patch['point'][2]-height > tolerance+patch['residualM']:
        return dict(status='failed', passed=False, reason='refused_refresh_above_admitted_target',
            body=controller.body_id, plannedSurfaceM=height, observedSurfaceM=patch['point'][2],
            observedAt=view['stamp'], toleranceM=tolerance, issues=fresh.get('issues', []))
    return None
