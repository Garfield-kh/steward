"""Optional release imaging, independent of seating classification or admission."""
from steward.ego import capabilities
from copy import deepcopy
import os
from pathlib import Path
import tempfile
import time
import uuid

import numpy as np

from steward.ego import action_budget as ab, wrist_observation as wo
from steward.math3d import pose_compose


def enabled():
    return capabilities.get('STEWARD_ASSEMBLY_WRIST_RELEASE_CAPTURE') != '0'


def recording():
    return capabilities.get('STEWARD_ASSEMBLY_WRIST_RELEASE_RECORDING') != '0'


def record(controller, sample, target, released_at, phase):
    """Persist exactly the synchronized RGB supplied to the release observer."""
    row = dict(phase=phase, releasedAt=released_at, captured=False,
               targetInFrame=None, rgbPath=None, storageError=None)
    if sample is None:
        row['capture'] = deepcopy(getattr(controller, '_wristCapture', {}))
        return row
    raw, cam = sample
    header = raw['header']
    row.update(captured=True, observedAt=header['stamp'], wallTime=time.time(),
        postRelease=bool(released_at is not None and header['stamp'] > released_at),
        afterReleaseS=header['stamp']-released_at if released_at is not None else None,
        joints=deepcopy(header.get('joints')), odometry=deepcopy(header.get('odometry')),
        cameraPose=deepcopy(cam), targetM=deepcopy(target), visibility='projection_only')
    try:
        if target is not None:
            sensor = controller.rt.estimator.cameras[wo.CAMERA][1]
            # Use the frame's pose, not the pose at capture request time.
            from steward.ego.support_holes import project
            uv, depth = project([target], cam, sensor['intrinsics'])
            near, far = sensor.get('depthRangeM', [0., float('inf')])
            size = np.array([sensor['intrinsics']['width'], sensor['intrinsics']['height']])
            row.update(targetPixel=uv[0].tolist(), opticalDepthM=float(depth[0]),
                targetInFrame=bool(np.isfinite(uv).all() and near < depth[0] < far and
                                   np.all(uv[0] >= 2) and np.all(uv[0] < size-2)))
    except (ValueError, KeyError, AttributeError, TypeError) as exc:
        row['projectionError'] = type(exc).__name__
    try:
        from PIL import Image
        root = (capabilities.get('STEWARD_WRIST_RELEASE_RGB_DIR') or
                capabilities.get('STEWARD_WRIST_DUMP_DIR') or
                os.environ.get('STEWARD_RECEIPT_LOG_DIR') or getattr(controller.rt, 'out_dir', None) or
                Path(tempfile.gettempdir()) / ('steward-wrist-release-'+str(os.getpid())))
        root = Path(root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        path = root / ('wrist-release-'+uuid.uuid4().hex+'.png')
        Image.fromarray(np.asarray(raw['rgb'], dtype=np.uint8)).save(path)
        row['rgbPath'] = str(path)
    except (OSError, ValueError, KeyError, AttributeError, TypeError) as exc:
        row['storageError'] = type(exc).__name__+': '+str(exc)
    return row


def capture(controller, plan):
    """Keep release orientation/reach; only use the admitted vertical withdrawal."""
    from steward.ego.skills import SkillFailed
    from steward.ego import wrist_release_geometry as geometry
    from steward.ego.wrist_release_view import choose
    if not wo.enabled(controller.rt, 'RELEASE'):
        return
    from steward.ego.hole_seating import capture as capture_hole
    from steward.ego.tip_entry_servo import camera_for
    if getattr(controller.rt.estimator, 'cameras', None) and camera_for(controller.rt):
        capture_hole(controller, plan)
        return
    started = time.monotonic()
    state = controller.ex._state()
    joints = dict(state['joints'])
    capture_budget = ab.make(controller.ex,
        2.+ab.frame_window(controller.ex) if ab.enabled() else 4.,
        'release withdrawal servo and one wrist frame')
    deadline = wo.bounded_deadline(controller, capture_budget)
    target = (plan.get('seatTarget') or {}).get('point')
    wo.context(controller, 'fastener_release', target)
    view = dict(status='unknown', reason='target_estimate_unavailable', executedLiftM=0.)
    plan['_wristReleaseSample'] = None
    controller._wristCapture = dict(status='unknown', reason='release_view_not_captured')
    try:
        if target is not None:
            est = controller.rt.estimator
            link, sensor = est.cameras[wo.CAMERA]
            cam = pose_compose(est.link_pose(link, joints, state['odometry']), sensor['mount'])
            known = plan.get('_wristReleasePrior') or geometry.prior(est.tracks.get(plan.get('object')))
            view = choose(cam, sensor, target, geometry.robot_boxes(controller, dict(header=state)), known)
            view.update(executedLiftM=0., wrist_release_capture=True)
            lift = view.get('liftM', 0.)
            # Never extend the original exit's vertical envelope for a photograph.
            ceiling = (plan.get('seatEndpoint') or {}).get('safeLift', plan.get('safeLift', joints['lift']))
            if joints['lift']+lift > max(joints['lift'], ceiling):
                view.update(status='unknown', reason='view_outside_admitted_withdrawal')
                lift = 0.
            path = [dict(joints, lift=float(h)) for h in
                    np.linspace(joints['lift'], joints['lift']+lift, 14)]
            if lift > 0 and controller.rt.esim.arm.reachable_path(path):
                left = ab.left(deadline)
                if left > 0:
                    controller.ex._set_joint('lift', joints['lift']+lift, timeout=min(2., left), ramp=False)
                    view['executedLiftM'] = float(controller.ex._state()['joints']['lift']-joints['lift'])
            elif lift > 0:
                view.update(status='unknown', reason='view_path_unreachable')
        controller._wristContext['releaseView'] = deepcopy(view)
        plan['_wristReleaseSample'] = wo.capture(controller, deadline)
        plan['_wristReleaseReason'] = getattr(controller, '_wristCapture', {}).get('reason', 'frame_unavailable')
    except SkillFailed:
        controller.check()  # cancellation/deadline still propagate
        plan['_wristReleaseReason'] = 'release_clearance_motion_failed'
    except (ValueError, KeyError, AttributeError, TypeError):
        plan['_wristReleaseReason'] = 'release_view_geometry_unavailable'
    finally:
        plan['_wristReleaseView'] = view
        plan['_wristReleaseWallS'] = time.monotonic()-started
        if ab.enabled():
            plan['_wristReleaseSimS'] = capture_budget.sim_s
        plan['_wristReleasePhase'] = 'after_open_before_withdrawal'
        if recording():
            plan['_wristReleaseFrame'] = record(controller, plan['_wristReleaseSample'], target,
                plan.get('releasedAt'), plan['_wristReleasePhase'])
