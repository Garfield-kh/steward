"""Release readiness from existing RGB-D evidence; no observation tour or model call."""
from steward.ego import capabilities
from copy import deepcopy
import time
import numpy as np

from steward.ego.fasten_plan import value
from steward.ego.fasten_release import released_interface
from steward.ego.skills import SkillFailed

ENABLED = capabilities.get('STEWARD_PLACEMENT_READY') != '0'
REPLAY_ENABLED = capabilities.get('STEWARD_PLACEMENT_REPLAY') != '0'
# One is a hard ceiling, including recursive place callbacks.
MAX_REPLAYS = min(1, max(0, int(capabilities.get('STEWARD_PLACEMENT_REPLAYS'))))


def assess(controller, plan):
    from steward.ego.assembly_session import current
    if current() is not None:
        return dict(status='unknown', passed=None, residualM=None, reason='legacy_environment_criterion_not_used'), None
    from steward.scene.fasteners import FastenConfig
    from steward.ego.assembly_precision import pair_measurement, compact
    from steward.ego.fasten_align import outline
    from steward.ego.assembly_motion import placement_check
    from steward.ego.component_registration import detect_native, rotation
    cfg = FastenConfig.from_env()
    rt = controller.rt; est = rt.estimator; scene = rt.ego_state()
    after = plan['releasedAt']; now = controller.ex._state()['stamp']
    frame = released_interface(scene, plan['object'], controller.body_id)
    result = dict(status='unknown', passed=None, reason='no_independent_release_evidence',
        radialToleranceM=cfg.radial, residualM=None, stable=None, uncoveredHoles=[],
        criterion='all paired holes within radial tolerance and component supported/stationary')
    if not frame:
        return result, None
    part = scene.find(frame['entity']); view = value(part, 'contactView')
    pose = outline(view, frame['position'], np.asarray(frame['rotation']), frame['size'], after, now)
    measurement = pair_measurement(rt.esim, scene, frame, after)
    result['alignment'] = compact(measurement)
    residual = measurement.get('residualM')
    uncertainty = measurement.get('uncertaintyM')
    # A separately observed upper outline can be paired with the native bare
    # support registration even when the lower mask merges after release.
    reg = plan.get('placementTarget', {}).get('componentHoleRegistration')
    if reg and any(e['entity'] == controller.body_id and e['stamp'] > reg['observedAt']
                   for e in est.stationary_assemblies.motion_events):
        reg = None
    if pose and reg:
        lower = np.asarray(reg['local']) @ rotation(reg['yaw']).T+reg['position']
        own = {h['id']:h for h in frame['holes']}
        pairs = reg['holePairs']
        if all(p['partHole'] in own for p in pairs):
            upper = np.array([np.asarray(pose['position'])+np.asarray(pose['rotation']) @
                              own[p['partHole']]['p'] for p in pairs])[:, :2]
            residual = float(max(np.linalg.norm(upper-lower, axis=1)))
            uncertainty = pose['positionUncertaintyM']+reg.get('worldUncM', reg['uncertaintyM'])
            result['alignment'] = dict(source='rgbd_outline_and_native_registered_holes',
                observedAt=pose['observedAt'], perHoleM={p['bodyHole']:float(d) for p,d in
                zip(pairs, np.linalg.norm(upper-lower, axis=1))})
            # Visibility alone is ambiguous (a concentric through-hole is visible
            # too). Require the measured lower rim to lie outside the independently
            # observed upper footprint, including both uncertainty bounds.
            raws = {**getattr(rt, '_precision_native_frames', {}),
                    **getattr(rt, '_latest_native_frames', {})}
            raw = raws.get(est.primary_camera)
            if raw and after < raw['header']['stamp'] <= now and now-raw['header']['stamp'] <= 2:
                spec = getattr(controller, '_component_spec', None)
                if spec:
                    from steward.ego.inhand import camera_pose
                    cam = camera_pose(est, raw['header'], raw['camera'])
                    found = detect_native(raw, cam, lower, spec['frame']['position'][2])
                    result['nativeReason'] = found['reason']
                    for pair, point in zip(pairs, lower):
                        matches = [h for h in found['holes'] if
                            np.linalg.norm(np.asarray(h['point'])[:2]-point) <= h['uncertaintyM']+reg['worldUncM']]
                        if len(matches) != 1:
                            continue
                        h = matches[0]
                        local = np.asarray(pose['rotation']).T @ (np.asarray(h['point'])-pose['position'])
                        margin = h['uncertaintyM']+pose['positionUncertaintyM']
                        if np.any(abs(local[:2]) > np.asarray(frame['size'])[:2]/2+margin):
                            result['uncoveredHoles'].append(pair['bodyHole'])
    result.update(residualM=residual, uncertaintyM=uncertainty)
    expected = dict(entity=frame['entity'], position=frame['position'], size=frame['size'])
    source = dict(category=frame['category'], position=frame['position'])
    with rt._lock:
        landing = placement_check(est.tracks.values(), source, expected, after, now)
        track = est.tracks.get(frame['entity'])
        placement = getattr(track, 'placement', None) or {}
        samples = [placement.get('previousObservation'), placement.get('observation')]
    result['supportCheck'] = landing
    # Use the two already retained post-release images, never wait for another.
    good = [v for v in samples if v and after < v.get('stamp', -1) <= now
            and now-v['stamp'] <= 20 and not v.get('truncated', True)
            and v.get('invalidDepthFraction', 1) <= .05
            and np.allclose(np.sort(v['visibleSizeM'][:2]), np.sort(frame['size'][:2]), atol=.015)]
    if len(good) == 2 and good[1]['stamp'] > good[0]['stamp']:
        distance = float(np.linalg.norm(np.asarray(good[1]['centreM'])-good[0]['centreM']))
        # Readiness is at fastening scale, not millimetric outline precision.
        result.update(stabilityDisplacementM=distance, stabilityLimitM=cfg.radial,
                      stable=distance <= cfg.radial)
    from steward.ego.assembly_landing import assess as support_assess
    support_landing = support_assess(controller, plan, frame)
    result['destinationCheck'] = support_landing
    if support_landing['passed'] is False:
        result.update(status='fail', passed=False, reason=support_landing['reason'])
    elif landing['passed'] is False:
        result.update(status='fail', passed=False, reason='component_not_supported_at_release_target')
    elif result['uncoveredHoles']:
        result.update(status='fail', passed=False, reason='target_holes_visible_outside_component')
    elif residual is not None and uncertainty is not None and residual-uncertainty > cfg.radial:
        result.update(status='fail', passed=False, reason='paired_holes_outside_radial_tolerance')
    elif result['stable'] is False:
        result.update(status='fail', passed=False, reason='component_moving_after_release')
    elif residual is not None and uncertainty is not None and residual+uncertainty <= cfg.radial and result['stable'] is True and landing['passed'] is True:
        result.update(status='ok', passed=True, reason='paired_holes_in_tolerance_and_supported_stable')
    elif residual is not None:
        result['reason'] = 'alignment_or_stability_unresolved'
    return result, pose


def post_release(controller, job, plan, params):
    """At most one ordinary, freshly admitted pick/place in this assembly job."""
    started = time.monotonic()
    try:
        report, pose = assess(controller, plan)
    except (KeyError, ValueError, TypeError, AttributeError) as exc:
        # Optional observations are sometimes partial while the estimator updates.
        report, pose = dict(status='unknown', passed=None, residualM=None,
            reason='release_evidence_unavailable', detail=type(exc).__name__), None
    count = getattr(controller, '_placement_replays', 0)
    report.update(replays=count, maxReplays=MAX_REPLAYS if REPLAY_ENABLED else 0)
    job.postcondition['placementReadiness'] = report
    plan['placementReadiness'] = report
    if report['passed'] is not False:
        report['wallS'] = time.monotonic()-started
        return
    est = controller.rt.estimator
    occupied = any(a.get('releaseExecuted') and a.get('part') == plan['object']
                   for a in est.fastening_attempts.values())
    if not REPLAY_ENABLED or count >= MAX_REPLAYS or occupied:
        report['wallS'] = time.monotonic()-started
        raise SkillFailed('placement readiness failed: '+report['reason'])
    source = report['supportCheck'].get('retrySource')
    if pose is None and source is None:
        report['replayReason'] = 'no_independent_regrasp_pose'
        report['wallS'] = time.monotonic()-started
        raise SkillFailed('placement readiness failed: '+report['reason']+'; no observed regrasp pose')
    controller._placement_replays = count+1
    history = job.postcondition.setdefault('placementReadinessHistory', [])
    history.append(deepcopy(report))
    from steward.ego.assembly_precision import install_pose
    with controller.rt._lock:
        if pose:
            install_pose(est, plan['object'], pose)
        elif source['entity'] in est.tracks:
            est.tracks[source['entity']].pos = np.asarray(source['position'])
    target = plan['object'] if pose else source['entity']
    try:
        controller.step(job, 'pick', dict(target=target, stayNear=True))
        corrected = controller.step(job, 'place', dict(params, stayNear=True))
        plan.update({k:corrected[k] for k in ('releasedAt', 'placementTarget', 'placementReadiness') if k in corrected})
    except SkillFailed as exc:
        job.postcondition.setdefault('placementReadiness', report).update(replayFailure=str(exc), passed=False, status='fail')
        raise
    finally:
        job.postcondition['placementReadinessHistory'] = history
        job.postcondition.setdefault('placementReadiness', report)['wallS'] = time.monotonic()-started
