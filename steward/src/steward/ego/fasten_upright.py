"""Check grasp-time geometry transported by current gripper encoders."""
from steward.ego import capabilities
import math
import numpy as np
from steward.ego.fasten_geometry import transform_feature
from steward.ego.skills import SkillFailed

ENABLED = capabilities.get('STEWARD_ASSEMBLY_FASTENER_UPRIGHT') != '0'
MAX_TILT_DEG = 10.
MAX_REGRASPS = 2


def assess(controller):
    from steward.ego.wrist_fasten import posture
    observed = posture(controller)
    if observed is not None:
        return observed
    est = controller.rt.estimator
    track = est.tracks.get(est.held_id)
    feature = getattr(track, 'held_grasp_geometry', None)
    if feature is None:
        return dict(status='unknown', reason='no_grasp_geometry')
    state = controller.ex._state()
    transformed = transform_feature(feature,
        est.gripper_rotation(est.held_gripper, state['joints'], state['odometry']),
        est.gripper_position(est.held_gripper, state['joints'], state['odometry']))
    direction = np.asarray(transformed['head'])-transformed['tip']
    length = np.linalg.norm(direction)
    if not np.isfinite(direction).all() or length <= 0:
        return dict(status='unknown', reason='invalid_grasp_axis')
    angle = math.degrees(math.acos(float(np.clip(direction[2]/length, -1., 1.))))
    polarity = bool(feature.get('polarityObserved'))
    tilt = angle if polarity else min(angle, 180.-angle)
    bad = tilt > MAX_TILT_DEG
    return dict(status='contradicted' if bad else ('upright' if polarity else 'unknown'),
        reason=('inverted' if polarity and angle > 90 else 'tilted') if bad else
               ('axis_upright_polarity_unknown' if not polarity else 'upright'),
        axisTiltDeg=round(tilt, 2), directedAngleDeg=round(angle, 2) if polarity else None,
        polarityObserved=polarity, source='grasp_observation_plus_gripper_fk')


def ensure(controller, job, support):
    if not ENABLED:
        return
    checks = []
    for retry in range(MAX_REGRASPS+1):
        controller.check()
        check = assess(controller); checks.append(check)
        row = dict(check, regraspCount=retry)
        if getattr(controller,'_wristPosture',None):
            row['wristCapture']=dict(controller._wristPosture)
        job.postcondition.setdefault('seatDelivery', {})['posture'] = row
        ctx = getattr(controller, '_registered_alignment', None)
        if ctx:
            ctx['report'].setdefault('seatDelivery', {})['posture'] = row
        if check['status'] != 'contradicted':
            return
        if retry == MAX_REGRASPS:
            controller.park(job, support)
            raise SkillFailed('held fastener orientation still contradicted after two regrasp attempts')
        try:
            controller.regrasp_upright(job, dict(target=controller.rt.estimator.held_id,
                support=support, fastenerPosture=True))
        except SkillFailed:
            controller.check()
            # A failed stand plan still allows a local ordinary park and top grasp.
            eid = controller.rt.estimator.held_id
            if not eid or not controller.park(job, support):
                raise
            controller.step(job, 'pick', dict(target=eid, topGrasp=True))
