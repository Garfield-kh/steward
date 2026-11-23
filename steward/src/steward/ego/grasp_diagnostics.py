"""Passive grasp evidence. Files are never read by planning or control.

One durable close event per fastener grasp; later images/inspections append
updates with the same graspId, so a failed lift cannot erase the denominator.
"""
from copy import deepcopy
from functools import wraps
import json
import math
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np
from PIL import Image

_lock = threading.Lock()


def passive(fn):
    @wraps(fn)
    def call(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            print(f'grasp_diagnostics {fn.__name__}: {type(exc).__name__}: {exc}', file=sys.stderr)
    return call


def root(rt):
    directory = os.environ.get('STEWARD_RECEIPT_LOG_DIR') or getattr(rt, 'out_dir', None)
    return Path(directory) if directory else None


def clean(value):
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, dict): return {k:clean(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def evidence(track):
    return (getattr(track, 'grasp_association', None) or {}).get('endpointEvidence', {})


def emit(rt, track, stage, **fields):
    directory = root(rt)
    if directory is None or not evidence(track).get('graspId'): return
    row = dict(schema='fastener_grasp.v1', event='fastener_grasp', stage=stage,
               graspId=evidence(track)['graspId'], category=track.category,
               entity=track.id, endpointEvidence=deepcopy(evidence(track)), **fields)
    directory.mkdir(parents=True, exist_ok=True)
    with _lock, (directory / f'fastener-grasps-{os.getpid()}.jsonl').open('a') as stream:
        stream.write(json.dumps(clean(row), allow_nan=False)+'\n')
    return row


def camera_pose(est, raw):
    from steward.math3d import pose_compose
    link, sensor = est.cameras[raw['camera']]
    h = raw['header']
    return pose_compose(est.link_pose(link, h['joints'], h['odometry']), sensor['mount'])


@passive
def cache_views(rt, processed, native):
    """Keep the exact associated frame crop, not a later/latest camera frame."""
    if root(rt) is None: return
    from steward.ego.support_holes import project
    stamp = processed['header']['stamp']
    cam = camera_pose(rt.estimator, native)
    h, w = native['depth'].shape
    for track in rt.estimator.tracks.values():
        view = track.manipulation_view
        if (track.category != 'toy_screw' or track.held or not view or view['stamp'] != stamp
                or view.get('camera') != native['camera']):
            continue
        points = np.asarray(view['points'])
        uv, z = project(points, cam, dict(width=w, height=h, fovy=native['fovy']))
        valid = (z > 0) & np.isfinite(uv).all(axis=1)
        if not valid.any(): continue
        x0, y0 = np.maximum(0, np.floor(uv[valid].min(axis=0)).astype(int)-8)
        x1, y1 = np.minimum([w, h], np.ceil(uv[valid].max(axis=0)).astype(int)+9)
        if x1 <= x0 or y1 <= y0: continue
        view['diagnosticFrame'] = dict(camera=native['camera'], stamp=stamp,
            nativeResolution=[w,h], detectorResolution=list(processed['depth'].shape[::-1]),
            cropXYXY=[int(x0),int(y0),int(x1),int(y1)],
            projectedExtentPixels=np.ptp(uv[valid],axis=0).tolist())
        mask = processed.get('mask')
        if mask is not None:
            mh, mw = mask.shape
            mx0, my0 = int(x0*mw/w), int(y0*mh/h)
            mx1, my1 = int(math.ceil(x1*mw/w)), int(math.ceil(y1*mh/h))
            mask = mask[my0:my1, mx0:mx1].copy()
            view['diagnosticFrame']['maskCropXYXY'] = [mx0,my0,mx1,my1]
        track.grasp_frame = dict(rgb=native['rgb'][y0:y1,x0:x1].copy(),
            depth=native['depth'][y0:y1,x0:x1].copy(),
            mask=mask,
            points=points.copy(), p=np.asarray(cam['p']), q=np.asarray(cam['q']),
            fovy=native['fovy'], header=deepcopy(native['header']), metadata=deepcopy(view['diagnosticFrame']))


def save(rt, track, stage, raw, cam=None):
    directory = root(rt)
    if directory is None: return dict(reason='output_directory_unavailable')
    if raw is None: return dict(reason='associated_frame_unavailable')
    directory = directory / 'grasp-frames'
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{evidence(track)['graspId']}-{stage}-{raw['header']['stamp']:.6f}"
    path = directory / (stem+'.npz')
    arrays = {k:v for k,v in raw.items() if k in ('rgb','depth','mask','points','p','q','fovy') and v is not None}
    if cam: arrays.update(p=np.asarray(cam['p']),q=np.asarray(cam['q']))
    np.savez_compressed(path, **arrays, header_json=json.dumps(clean(raw['header'])),
                        metadata_json=json.dumps(clean(raw.get('metadata', {}))))
    png = directory / (stem+'.png')
    Image.fromarray(raw['rgb']).save(png)
    return dict(npz=str(path.resolve()), rgb=str(png.resolve()), stamp=raw['header']['stamp'],
                metadata=raw.get('metadata', {}), reason=None)


@passive
def closed(rt, track, body):
    if track.category != 'toy_screw': return
    body = body or {}
    gripper = (body.get('grippers') or {}).get(rt.estimator.held_gripper, body.get('gripper', {}))
    # Sensor schemas may only report state/holding. Never convert those to a width.
    width = next((gripper[k] for k in ('openingM','widthM','gapM') if k in gripper), None)
    frame = getattr(track, 'grasp_frame', None)
    view = track.manipulation_view or {}
    if frame and frame['header']['stamp'] != view.get('stamp'): frame = None
    row = dict(pregrasp=deepcopy(getattr(track, 'grasp_direction_diagnostic', {})),
        association=deepcopy(track.grasp_association), boxSizeM=track.size.tolist(), boxYawRad=track.yaw,
        selectedGrasp=deepcopy(getattr(rt.executor, '_active_grasp_diagnostic', None)),
        closeSensor=body, closedOpeningM=width,
        openingReason=None if width is not None else 'sensor_does_not_report_metric_opening',
        gripperRotation=rt.estimator.gripper_rotation(rt.estimator.held_gripper,
            body.get('joints'), body.get('odometry')).tolist(),
        pregraspFrame=dict(reason='not_saved_yet'),
        inhandObservation=dict(attempted=False, reason='not_run_at_close'))
    from steward.math3d import matrix_from_quat, quat_from_yaw
    box_rotation = matrix_from_quat(quat_from_yaw(track.yaw))
    row['gripperInEstimatedBoxM'] = (box_rotation.T @ (
        np.asarray(track.grasp_association['gripperAtGrasp'])-track.pos)).tolist()
    # Write before image I/O: disk/image failures must not lose the grasp count.
    emit(rt, track, 'close', **row)
    saved = save(rt, track, 'pre', frame)
    emit(rt, track, 'pregrasp_frame', pregraspFrame=saved)


@passive
def after_lift(rt, eid):
    """Save head/wrist frames without moving or feeding them to the estimator."""
    from steward.ego.perception import decode_rgbd
    track = rt.estimator.tracks.get(eid)
    if track is None or track.category != 'toy_screw' or root(rt) is None: return
    deadline = min(time.monotonic()+2., getattr(rt.executor, '_primitive_deadline', math.inf),
                   getattr(rt.executor, '_job_deadline', math.inf))
    frames = []
    for camera in dict.fromkeys([rt.estimator.primary_camera, 'wrist_camera']):
        row = dict(camera=camera)
        try:
            if camera not in getattr(rt, '_provided_camera_ids', ()):
                row['reason'] = 'camera_not_provided'
            elif time.monotonic() >= deadline:
                row['reason'] = 'existing_deadline_exhausted'
            else:
                raw = decode_rgbd(rt.link.rgbd(camera, timeout=max(.001,deadline-time.monotonic())))
                row.update(save(rt, track, 'post-'+camera, raw, camera_pose(rt.estimator, raw)))
                row['afterClose'] = raw['header']['stamp'] > evidence(track)['graspedAt']
        except Exception as exc:
            row['reason'] = type(exc).__name__
        frames.append(row)
    emit(rt, track, 'postgrasp_frames', frames=frames,
         inhandObservation=dict(attempted=False, reason='raw_capture_only_not_direction_measurement'))


@passive
def inspection(rt, eid, method, result, sample=None, raw=None):
    track = rt.estimator.tracks.get(eid)
    if track is None or track.category != 'toy_screw': return
    frame = None
    if sample is None and raw is not None:
        sample = raw, camera_pose(rt.estimator, raw)
    if sample is not None:
        frame = save(rt, track, method, sample[0], sample[1])
    emit(rt, track, 'inhand_observation', method=method, result=result, frame=frame,
         currentOffsetEvidence=deepcopy((track.held_tool_feature or {}).get('offsetEvidence')))
