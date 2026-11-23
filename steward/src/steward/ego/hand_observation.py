"""One optional native hand image immediately after a successful lift.

Only robot RGB-D, calibrated hardware and same-frame encoders enter this path.
Unresolved polarity leaves the existing nominal/previous estimate untouched.
"""
from copy import deepcopy
from types import SimpleNamespace

from steward.ego import action_budget as ab
from steward.ego import wrist_observation as wo
from steward.ego.grasp_diagnostics import inspection
from steward.ego.priors import prior
from steward.ego.wrist_fasten import held_feature, calibrate_sample

CAMERA = 'hand_camera'


def after_lift(rt, eid, *, requested=False):
    est = rt.estimator
    cached = rt.__dict__.pop('_binding_hand_sample', None)
    with rt._lock:
        track = est.tracks.get(eid)
        if requested and track is None:
            return dict(schema='hand_observation.v1', entity=eid, attempted=False,
                        calibrated=False, polarityObserved=False, reason='held_track_unavailable')
        if track is None or (not requested and prior(track.category).get('assemblyRole') != 'fastener'):
            return None
        evidence = (getattr(track, 'grasp_association', None) or {}).get('endpointEvidence', {})
        since = est.held_since
        grasp_id = evidence.get('graspId')
    result = dict(schema='hand_observation.v1', graspId=grasp_id, entity=eid,
        camera=CAMERA, attempted=False, frameRequested=False, frameCaptured=False,
        visible=False, polarityObserved=False, calibrated=False, reason='camera_not_provided',
        identityVerified=False, source='native_rgbd_endpoint_width_and_same_frame_fk')
    sample = None
    ctl = SimpleNamespace(rt=rt, ex=rt.executor, _wristContext=dict(phase='postgrasp_hand'))
    try:
        if (CAMERA not in est.cameras or CAMERA not in getattr(rt, '_provided_camera_ids', ())
                or not callable(getattr(getattr(rt, 'link', None), 'rgbd', None))):
            return result
        result['attempted'] = True
        deadline = ab.make(ctl.ex, ab.frame_timeout(ctl.ex, 4.), 'one post-lift hand frame')
        # Search around the encoder TCP; unknown nominal polarity cannot veto a frame.
        from steward.ego.grasp_binding import enabled as binding_enabled
        reuse = binding_enabled() and cached and cached['eid'] == eid and cached['since'] == since
        sample = cached['sample'] if reuse else wo.capture(ctl, deadline, camera=CAMERA, check_view=False)
        capture = (dict(requested=False, frameCaptured=True, reason='reused_load_binding_frame') if reuse
                   else deepcopy(getattr(ctl, '_wristCapture', {})))
        result.update(capture=capture, frameRequested=capture.get('requested', False),
                      frameCaptured=capture.get('frameCaptured', False), reason=capture.get('reason'))
        if sample is None:
            return result
        with rt._lock:
            if est.held_id != eid or est.held_since != since or not track.held:
                result['reason'] = 'grasp_changed_during_capture'
                return result
            # The grip may be off-centre. Search the robot's own opening envelope
            # instead of truncating a visible shaft to its nominal length at TCP.
            feature = held_feature(ctl, sample, search_radius=rt.esim.arm.geometry.jaw_opening)
            diagnostic = deepcopy(getattr(ctl, '_heldFeatureDiagnostic', {}))
            if feature is not None and not diagnostic.get('visible', False):
                feature = None
                diagnostic['reason'] = 'multiple_visible_shafts'
            result.update(detector=diagnostic, visible=diagnostic.get('visible', False),
                          polarityObserved=bool(feature), reason=diagnostic.get('reason'))
            if feature is not None:
                result['calibrated'] = calibrate_sample(ctl, feature, sample)
                result['reason'] = None if result['calibrated'] else 'calibration_not_admitted'
                result['feature'] = deepcopy(feature)
                result['offsetEvidence'] = deepcopy((track.held_tool_feature or {}).get('offsetEvidence'))
        return result
    except (TimeoutError, OSError, ValueError, KeyError, AttributeError, TypeError) as exc:
        result['reason'] = 'hand_observation_error:'+type(exc).__name__
        return result
    finally:
        # Telemetry failure cannot cancel a successful grasp or its calibration.
        inspection(rt, eid, 'postgrasp_hand', result, sample)
