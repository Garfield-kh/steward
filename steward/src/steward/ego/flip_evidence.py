"""Face estimates from robot layout, grasp evidence and measured wrist motion.

Missing independent vision is not a veto. Contradictions are; a requested ID
and an executed half-turn command alone are never measured body motion.
"""
from copy import deepcopy
import math
import numpy as np


def body_grasp(est, record, samples=(), since=-math.inf, lift=None):
    ref = record['reference']
    held = est.tracks.get(est.held_id)
    assoc = getattr(held, 'grasp_association', None) or {}
    aliases = {ref['entity'], *record.get('aliases', [])}
    report = dict(status='estimated', passed=None, requested=ref['entity'],
        held=est.held_id, source='registered_grasp_and_holding_sensor',
        reason='body_grasp_not_contradicted', identityVerified=False,
        association={k:deepcopy(assoc[k]) for k in ('requested', 'held', 'category', 'observedAt',
            'gripperAtGrasp', 'gripperRotationAtGrasp') if k in assoc})
    reason = None
    from steward.ego.grasp_binding import receipt
    binding = receipt(est)
    tentative = bool(binding and not binding['transportCommitted'] and binding['requestedTarget'] in aliases)
    if est.held_id not in aliases and not tentative:
        reason = 'held_identity_differs_from_body'
    elif held is not None and held.category not in (ref['category'], ref.get('family')):
        reason = 'held_category_differs_from_body'
    elif (lift or {}).get('passed') is False:
        reason = 'body_observed_stationary_during_lift'
    # The frozen body frame is smaller than the carried assembly envelope. A
    # close well above/beside it can catch an upper component or protruding pin.
    # This uses the actual encoder TCP at close, not the planned grasp height.
    if not reason and assoc.get('gripperAtGrasp') is not None:
        from steward.math3d import matrix_from_quat, quat_from_yaw
        local = matrix_from_quat(quat_from_yaw(ref['yaw'])).T @ (
            np.asarray(assoc['gripperAtGrasp'])-ref['position'])
        excess = np.abs(local)-np.asarray(ref['size'])/2
        report['closeOutsideBodyM'] = float(max(excess))
        # Reported only: on the server every flip of a correctly grasped body tripped this
        # (TCP at close sits above the frozen body frame), so it must not veto the flip.
    # Raw views remain independent even when the track is pinned/carried by FK.
    # Two fresh matching body outlines at pickup refute transport without needing
    # a separate support mask or a pre-lift separate body mask.
    stayed = []
    for v in samples:
        if (not v or v.get('source') != 'rgbd_segment' or v.get('stamp', -math.inf) <= since
                or not 0 <= est.stamp-v['stamp'] <= 20 or v.get('truncated', True)
                or v.get('invalidDepthFraction', 1) > .05):
            continue
        if (np.allclose(np.sort(v['visibleSizeM']), np.sort(ref['size']), rtol=.2, atol=.015)
                and np.linalg.norm(np.asarray(v['centreM'])-ref['position']) < .025):
            stayed.append(v['stamp'])
    if len(set(stayed)) >= 2:
        reason = 'body_observed_at_pickup_after_lift'
    if reason:
        report.update(status='contradicted', passed=False, reason=reason)
    elif (lift or {}).get('passed') is True:
        report.update(status='observed', passed=True, identityVerified=True,
                      source='rgbd_body_lift', reason='body_lift_observed')
    return report


def reject_mismatch(job, grasp, lift=None):
    if grasp.get('passed') is not False:
        return
    from steward.ego.preconditions import finish_flip
    from steward.ego.skills import SkillFailed
    check = dict(passed=False, status='fail', reason=grasp['reason'])
    job.postcondition['orientationCheck'] = check
    try:
        finish_flip(job, check, lift or dict(passed=None))
    except SkillFailed as exc:
        raise SkillFailed('flip_body_mismatch: '+grasp['reason']+'；'+str(exc)) from exc


def resolve_face(ref, rotation, grasp, check, since):
    """Expose catalogue hole IDs; keep estimates separate from visual checks.

    rotation is the *measured* grasp-to-release delta, before any nominal
    half-turn/settling fallback. Its absence cannot be filled by a command.
    """
    result = dict(status='unknown', source=None, exposedHoles=[], exposedInterfaces=[],
                  observed=False, since=since, reason='insufficient_face_evidence')
    if grasp.get('passed') is False or check.get('passed') is False:
        return dict(result, status='contradicted', reason=grasp.get('reason')
                    if grasp.get('passed') is False else check.get('reason'))
    if rotation is None:
        return result
    previous = ref.get('faceResolution', {})
    before = np.asarray(previous.get('rotationM', ref.get('rotationM', np.eye(3))), float)
    delta = np.asarray(rotation, float)
    if delta.shape != (3, 3) or not np.isfinite(delta).all():
        return result
    current = delta @ before
    up = float(current[2, 2])
    holes = ref.get('holes', [])
    levels = [float(h['p'][2]) for h in holes]
    if not levels or max(levels)-min(levels) <= .01 or abs(up) < .8:
        return result
    # Do not accumulate an unresolved earlier motion as if its face were known.
    if (ref.get('faceCheck', {}).get('passed', True) is None
            and previous.get('status') not in ('estimated', 'observed')):
        return dict(result, reason='previous_face_unresolved')
    mid = (max(levels)+min(levels))/2
    exposed = [h for h in holes if (float(h['p'][2])-mid)*up > .005]
    observed = check.get('passed') is True
    return dict(result, status='observed' if observed else 'estimated',
        source='visual_face_and_layout' if observed else 'wrist_encoders_and_layout',
        reason='exposed_interface_from_body_frame', observed=observed,
        exposedHoles=[h['id'] for h in exposed],
        exposedInterfaces=sorted({h.get('group', h['id']) for h in exposed}),
        rotationM=current.tolist(), bodyUpZEstimate=up,
        graspEvidence=grasp['status'], rigidGraspAssumed=not grasp.get('identityVerified', False))
