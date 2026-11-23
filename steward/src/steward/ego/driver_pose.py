"""Check each acquired and oriented driver using current robot evidence."""
from steward.ego import capabilities
import math
import numpy as np
from steward.ego.skills import SkillFailed


def enabled():
    return capabilities.get('STEWARD_ASSEMBLY_DRIVER_POSE') != '0'


# Reuse the existing driver's geometric orientation admission, not a judge limit.
MAX_ANGLE_DEG = 12.


class PoseMismatch(SkillFailed):
    pass


def check(ctl, job, stage, *, since=None, require=False):
    if not enabled():
        return
    from steward.ego.fasten_plan import tool_estimate
    from steward.ego.fasten_geometry import directed_tool
    est = ctl.rt.estimator
    state = ctl.ex._state()
    scene = ctl.rt.ego_state()
    entity = scene.find(est.held_id) if est.held_id else None
    feature = tool_estimate(entity, scene.stamp) if entity else None
    # A newly measured in-hand head view wins over transported grasp geometry.
    # An earlier image cannot describe the wrist after it has turned.
    raw = ctl.sample(est.held_id) if entity else None
    after = max(est.held_since, since if since is not None else est.held_since)
    if (raw and raw.get('source') == 'rgbd_segment' and raw.get('stamp', -1) >= after
            and 0 <= state['stamp']-raw['stamp'] <= 20 and not raw.get('truncated', True)
            and raw.get('invalidDepthFraction', 1) <= .05):
        observed = directed_tool(raw.get('points', []), entity.category.value, raw['stamp'])
        if observed:
            feature = observed
    source = (feature or {}).get('source', 'unavailable')
    axis = np.asarray((feature or {}).get('axis', []), float)
    stale = (source == 'grasp_relative_plus_encoders' and
             any(w.get('reason') == 'stale_grasp_endpoint' for w in (feature or {}).get('warnings', [])))
    fresh_axis = (source != 'rgbd_endpoint_width' or
                  after <= (feature or {}).get('stamp', -1) <= state['stamp'] and
                  state['stamp']-(feature or {}).get('stamp', -1) <= 20)
    usable = (not stale and fresh_axis and source in ('grasp_relative_plus_encoders', 'rgbd_endpoint_width')
              and axis.shape == (3,) and np.isfinite(axis).all() and np.linalg.norm(axis)>0)
    angle = math.degrees(math.acos(float(np.clip(-axis[2]/np.linalg.norm(axis), -1, 1)))) if usable else None
    passed = bool(angle is not None and angle <= MAX_ANGLE_DEG)
    row = dict(stage=stage, entity=est.held_id, angleDeg=angle, source=source,
        observedAt=(feature or {}).get('stamp'), sampledAt=state['stamp'],
        graspedAt=est.held_since, limitDeg=MAX_ANGLE_DEG,
        status=('usable' if passed else 'tilted' if angle is not None else 'unknown'),
        required=require, independentObservation=source == 'rgbd_endpoint_width')
    if stale:
        row['reason'] = 'stale_grasp_axis'
    from steward.ego.driver_batch import compact
    job.postcondition.setdefault('driverPoseChecks', []).append(compact(row))
    if require and not passed:
        raise PoseMismatch('driver in-hand axis '+row['status']+'; regrasp before pressing')
    return row
