"""One bounded head-camera observation before leaving a released fastener."""
from steward.ego import capabilities
from copy import deepcopy
import math
import time
from steward.ego import action_budget as ab
import numpy as np

ENABLED = capabilities.get('STEWARD_ASSEMBLY_RELEASE_OBSERVATION') != '0'
BUDGET_S = 25.


def observe(controller, job, attempt, release_plan=None):
    from steward.ego.fasten_sequence import budget, DriveExpired
    from steward.ego.skills import SkillFailed
    from steward.ego.registered_alignment import arm_clear
    from steward.ego.fasten_plan import value
    from steward.ego.wrist_fasten import released
    wrist = released(controller, job, attempt, release_plan)
    if wrist is not None:
        attempt['releaseObservation'] = wrist
        job.postcondition['releaseObservation'] = deepcopy(wrist)
        controller.seat_posterior(attempt)
        return wrist
    start = time.monotonic()
    seconds=ab.local_motion_estimate(controller) if ab.enabled() else BUDGET_S
    deadline = ab.make(controller.ex, seconds, 'local joint ranges/speeds plus servo settling and frame acquisition')
    report = dict(budgetS=seconds, hole=attempt['hole'], reason='no_fresh_head_frame',
                  status='unknown', wallS=0., camera=controller.rt.estimator.primary_camera)
    attempt['releaseObservation'] = report
    request = None
    try:
        with budget(deadline):
            controller.wait_view(attempt['screw'], attempt['seatedAt'], timeout=0)
            controller.seat_posterior(attempt)
            est = controller.rt.estimator; camera = est.primary_camera
            # The normal place exit already withdraws. Retry only its admitted
            # withdrawal when it left the ray obstructed, never invent a pose.
            clear = arm_clear(est, attempt['entry'], camera)
            if not clear and release_plan:
                controller.ex._withdraw(release_plan)
                clear = arm_clear(est, attempt['entry'], camera)
            report['armClear'] = bool(clear)
            if not clear:
                report['reason'] = 'gripper_occludes_hole'
            else:
                after = max(attempt['seatedAt'], controller.ex._state()['stamp'])
                request = controller.rt.request_precision_view([attempt['entry']], after)
                request['cameras'] = [camera] if camera in request.get('cameras', []) else []
                report['requestedAfter'] = after
                if not request['cameras']:
                    report['reason'] = 'hole_outside_head_camera_close_fov'
                else:
                    sample_deadline = ab.make(controller.ex, ab.observation(controller.ex) if ab.enabled() else 8., 'existing eight-sensor-second frame window')
                    while not ab.expired(sample_deadline):
                        controller.check()
                        controller.seat_posterior(attempt)
                        if attempt.get('seatCheck', {}).get('observedAt', -1) is not None and (
                                attempt.get('seatCheck', {}).get('observedAt', -1) > after):
                            break
                        if request.get('completed') or controller.ex._state()['stamp']-after >= (ab.observation(controller.ex) if ab.enabled() else 8.):
                            break
                        controller.ex._wait(.1)
                    report['reason'] = 'head_frame_processed' if request.get('completed') else 'head_frame_wait_finished'
                report['acquisition'] = {k:deepcopy(request[k]) for k in
                    ('cameras','captured','completed','reason','nativeResolutions') if k in request}
            controller.seat_posterior(attempt)
    except DriveExpired:
        report.update(reason='release_observation_sim_timeout_or_deadlock' if ab.enabled() else 'release_observation_wall_timeout', timedOut=True)
    except SkillFailed as exc:
        controller.check()
        report.update(reason='release_observation_motion_unavailable', detail=str(exc))
    except (KeyError, ValueError, AttributeError, TypeError) as exc:
        report.update(reason='release_observation_geometry_unavailable', detail=type(exc).__name__)
    finally:
        if request is not None:
            controller.rt.finish_precision_view(request)
        report['wallS'] = time.monotonic()-start
        report.update(ab.report(deadline, compact=True))
    check = attempt.get('seatCheck', {})
    report.update(status=check.get('status','unknown'), check=deepcopy(check))
    precision = attempt.get('precision')
    if precision is not None:
        precision['postRelease'] = dict(status=check.get('status','unknown'),
            reason=check.get('reason', report['reason']), observedAt=check.get('observedAt'),
            source=check.get('source'), residualM=check.get('headDistanceXYM', check.get('distanceToHoleM')))
        precision['postReleaseResidualM'] = precision['postRelease']['residualM']
    attempt['releaseObservation'] = report
    job.postcondition['releaseObservation'] = deepcopy(report)
    return report


def axis_check(entity, entry, after, stamp):
    """A polarity-ambiguous full shaft can disprove seating, never confirm it."""
    from steward.ego.fasten_plan import value
    from steward.ego.priors import prior
    from steward.ego.classify import SHAPES
    if entity is None or prior(entity.category.value).get('assemblyRole') != 'fastener':
        return None
    view = value(entity, 'contactView')
    if (not view or view.get('source') != 'rgbd_segment' or view.get('truncated', True)
            or view.get('invalidDepthFraction', 1) > .05
            or not after < view.get('stamp', -1) <= stamp or stamp-view['stamp'] > 20):
        return None
    points = np.asarray(view.get('points', []), float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 30 or not np.isfinite(points).all():
        return None
    centre = np.median(points, axis=0)
    _, values, axes = np.linalg.svd(points-centre, full_matrices=False)
    if values[0] < 1.25*values[1] or values[1] > 4*values[2]:
        return None
    axis = axes[0]; along = (points-centre) @ axis
    lo, hi = np.quantile(along, [.01,.99])
    nominal = max(SHAPES[entity.category.value]['dims'])
    if not .55 <= (hi-lo)/nominal <= 1.6:
        return None
    ends = centre + np.array([lo,hi])[:,None]*axis
    vector = ends[1,:2]-ends[0,:2]; q = np.asarray(entry)[:2]-ends[0,:2]
    t = float(np.clip(q @ vector/max(float(vector @ vector),1e-12),0,1))
    distance = float(np.linalg.norm(ends[0,:2]+t*vector-np.asarray(entry)[:2]))
    tilt = math.degrees(math.acos(float(np.clip(abs(axis[2]),0,1))))
    if tilt <= 20 and distance <= .012:
        return None  # no polarity / head height: unknown, not confirmation
    return dict(status='failed', passed=False, reason='independent_screw_axis',
        source='rgbd_axial_shape', observedAt=view['stamp'], freshEndpointObserved=False,
        axisTiltDeg=tilt, distanceToHoleM=distance, axisM=axis.tolist(),
        endpointsM=ends.tolist(), headPolarityObserved=False)
