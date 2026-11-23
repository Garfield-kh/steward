"""Lower-interface seat recovery from robot observations and executed releases."""
from copy import deepcopy

from steward.ego.fasten_plan import value
from steward.ego.fasten_geometry import surface_patch
from steward.ego.skills import SkillFailed


def entry_surface(scene, part, target):
    """Retain hole XY/correspondence; update Z after the component has settled.

    An inverted assembly may rest on its attached parts with a slight slope.
    A local horizontal/sloping patch is useful even when its overall mask merges;
    neither its centre nor its bounding-box top relocates a hole.
    """
    view = value(scene.find(part), 'contactView')
    if (not view or view.get('source') != 'rgbd_segment' or view.get('truncated', True)
            or view.get('invalidDepthFraction', 1) > .05
            or not target.get('releasedAt', -1) < view.get('stamp', -1) <= scene.stamp
            or scene.stamp-view['stamp'] > 20):
        return target
    patch = surface_patch(view.get('points', []), target['point'][:2], max_tilt_deg=25)
    if patch is None or abs(patch['point'][2]-target['point'][2]) > .02:
        return target
    result = deepcopy(target)
    result.update(point=patch['point'], targetSource='interface_xy_plus_settled_surface_z',
                  surfaceEvidence=dict(patch, stamp=view['stamp'], source='rgbd_local_entry_surface',
                                       previousEntryM=list(target['point'])))
    return result


def retry_unreleased(controller, job, requests, support):
    """Spend the existing shared replay budget on empty action slots first.

    A failed pick has no release attempt, so the ordinary posterior loop cannot
    revisit it. Never re-place the component or replace a released/confirmed pin
    here, and never buy a fresh budget on recursion.
    """
    recovered = set()
    for hole, params in requests.items():
        controller.check()
        key = (params['body'], hole)
        a = controller.rt.estimator.fastening_attempts.get(key, {})
        if a.get('releaseExecuted') or not controller._seat_retry_budget[0]:
            continue
        if controller.rt.estimator.held_id:
            break
        controller._seat_retry_budget[0] -= 1
        controller._seat_retry_counts[key] = controller._seat_retry_counts.get(key, 0)+1
        retry = dict(params)
        retry.pop('stagedScrew', None)  # The failed source must be reselected.
        row = dict(hole=hole, executed=False)
        job.postcondition.setdefault('unreleasedHoleRetries', []).append(row)
        try:
            controller.step(job, 'seat_screw', retry)
            a = controller.rt.estimator.fastening_attempts.get(key, {})
            row['executed'] = bool(a.get('releaseExecuted'))
            if row['executed']:
                recovered.add(hole)
        except SkillFailed as exc:
            controller.check()
            row['failure'] = str(exc)
            if controller.rt.estimator.held_id:
                controller.park(job, support)
    return recovered
