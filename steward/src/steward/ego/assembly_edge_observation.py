"""Robot-local RGB-D for an assembly whose instance labels may have merged.

These samples are geometry only, never independent body-motion evidence.
"""
import math
import numpy as np
from steward.ego.planner import entity_box
from steward.math3d import matrix_from_quat, quat_from_yaw

SOURCES = ('rgbd_segment', 'rgbd_assembly', 'rgbd_local')


def usable(view, now, after=-math.inf):
    if not view or view.get('source') not in SOURCES:
        return False
    points = np.asarray(view.get('points', []))
    return bool(after < view.get('stamp', -1) <= now and now-view['stamp'] <= 20
        and not view.get('truncated', True) and view.get('invalidDepthFraction', 1) <= .05
        and points.ndim == 2 and points.shape[1] == 3 and len(points) >= 12
        and np.isfinite(points).all())


def body_patch(points, reference, diagnostics=None):
    """Robust broad body face fit; local steps/occluders cannot define its slope."""
    diag = diagnostics if diagnostics is not None else {}
    def reject(reason):
        diag['reason'] = reason
        return None
    p = np.asarray(points, float)
    diag['inputPoints'] = len(p) if p.ndim else 0
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        return reject('invalid_points')
    origin = np.asarray(reference['position'])
    r = matrix_from_quat(quat_from_yaw(reference['yaw']))[:2, :2]
    local = (p[:, :2]-origin[:2]) @ r
    half = np.asarray(reference['size'][:2])*.35
    keep = np.all(abs(local) <= half, axis=1)
    p, local = p[keep], local[keep]
    diag['bodyPoints'] = len(p)
    if len(p) < 12:
        return reject('insufficient_body_points')
    coverage = np.ptp(local, axis=0)/half
    diag['coverage'] = np.round(coverage, 3).tolist()
    if np.any(coverage < 1):
        return reject('insufficient_face_coverage')
    design = np.c_[p[:, :2]-origin[:2], np.ones(len(p))]
    if np.linalg.matrix_rank(design) < 3:
        return reject('degenerate_face')
    fit = np.linalg.lstsq(design, p[:, 2], rcond=None)[0]
    residual = float(np.quantile(abs(design @ fit-p[:, 2]), .95))
    diag['residualM'] = round(residual, 6)
    fits = [(fit, np.ones(len(p), dtype=bool))] if residual <= .0015 else []
    # Deterministic local hypotheses, scored on the whole face. This tolerates
    # a minority of raised attachments without fitting a diagonal through steps.
    if not fits:
        for axis in range(2):
            for lo, hi in ((-1., -.2), (-.4, .4), (.2, 1.)):
                seed = (local[:, axis] >= lo*half[axis]) & (local[:, axis] <= hi*half[axis])
                if seed.sum() < 12 or np.linalg.matrix_rank(design[seed]) < 3:
                    continue
                candidate = np.linalg.lstsq(design[seed], p[seed, 2], rcond=None)[0]
                inliers = abs(design @ candidate-p[:, 2]) <= .0015
                for _ in range(2):
                    if inliers.sum() < max(12, .60*len(p)):
                        break
                    candidate = np.linalg.lstsq(design[inliers], p[inliers, 2], rcond=None)[0]
                    inliers = abs(design @ candidate-p[:, 2]) <= .0015
                if (inliers.sum() >= max(12, .60*len(p))
                        and np.all(np.ptp(local[inliers], axis=0) >= half)):
                    fits.append((candidate, inliers))
    if not fits:
        return reject('plane_residual_exceeded')
    fit, inliers = max(fits, key=lambda row:int(row[1].sum()))
    residual = float(np.quantile(abs(design[inliers] @ fit-p[inliers, 2]), .95))
    normal = np.r_[-fit[:2], 1.]; normal /= np.linalg.norm(normal)
    diag.update(inliers=int(inliers.sum()), residualM=round(residual, 6),
                tiltDeg=round(math.degrees(math.acos(normal[2])), 3))
    if normal[2] < math.cos(math.radians(35)):
        return reject('face_tilt_exceeded')
    if residual > .0015:
        return reject('plane_residual_exceeded')
    diag['reason'] = 'body_face_fit'
    return dict(point=[*origin[:2].tolist(), float(fit[2])], normal=normal.tolist(),
                residualM=residual, points=int(inliers.sum()))


def diagnostic(controller, key):
    report = getattr(controller, '_edge_observation_diag', None)
    if report is None:
        report = {}; controller._edge_observation_diag = report
    report[key] = {}
    return report[key]


def merged_view(controller, record, after=-math.inf):
    """Require coverage of the registered body, not an instance-name match."""
    from steward.ego.assembly_edge import corners
    ref = record['reference']; origin = np.asarray(ref['position'])
    after = max(after, ref.get('ownReleaseAt', -math.inf))
    r = matrix_from_quat(quat_from_yaw(ref['yaw']))
    diag = diagnostic(controller, 'merged')
    diag.update(tracks=0, usable=0, covered=0, heightMatched=0, bounded=0)
    candidates = []
    for track in controller.rt.estimator.tracks.values():
        diag['tracks'] += 1
        v = track.manipulation_view
        if getattr(track, 'held', False) or not usable(v, controller.ex._state()['stamp'], after):
            continue
        diag['usable'] += 1
        points = np.asarray(v['points'])
        local = (points-origin) @ r
        near = local[np.all(abs(local[:, :2]) <= np.asarray(ref['size'][:2])*.6, axis=1)]
        if len(near) < 12 or np.any(np.ptp(near[:, :2], axis=0) < np.asarray(ref['size'][:2])*.55):
            continue
        diag['covered'] += 1
        # Exclude a nearby table or unrelated overhead mask by registered height.
        face_z = origin[2]+ref['size'][2]/2
        if abs(np.median(near[:, 2])+origin[2]-face_z) > ref['size'][2]:
            continue
        diag['heightMatched'] += 1
        # The complete observed outline must fit the local assembly neighbourhood.
        envelope = np.vstack([corners(p, origin, np.eye(3)) for p in [ref, *record.get('parts', [])]])
        pad = .15*np.linalg.norm(ref['size'])
        if np.any(points-origin < envelope.min(axis=0)-pad) or np.any(points-origin > envelope.max(axis=0)+pad):
            continue
        diag['bounded'] += 1
        candidates.append(dict(v, source='rgbd_assembly'))
    return min(candidates, key=lambda v:len(v['points'])) if candidates else None


def visible_rays(points, camera, boxes):
    """Reject each depth ray crossing the robot, rather than the whole face."""
    visible = np.ones(len(points), dtype=bool)
    for centre, half, rotation in boxes:
        start = (np.asarray(camera)-centre) @ rotation
        rays = (points-camera) @ rotation
        half = np.asarray(half)+.003
        parallel = abs(rays) < 1e-12
        safe = np.where(parallel, 1., rays)
        t0, t1 = (-half-start)/safe, (half-start)/safe
        lower = np.where(parallel, -np.inf, np.minimum(t0, t1))
        upper = np.where(parallel, np.inf, np.maximum(t0, t1))
        outside = np.any(parallel & (abs(start) > half), axis=1)
        entry = np.maximum(lower.max(axis=1), .02)
        leave = np.minimum(upper.min(axis=1), 1.005)
        visible &= outside | (entry > leave)
    return visible


def native_view(controller, record, after=-math.inf, *, face=False, raw=None, exclude_boxes=(), partial_face=False):
    """Native registered ROI; full-outline gates apply only to preflip geometry."""
    from steward.ego import support_registration as registration, support_holes as holes
    from steward.ego.registered_alignment import arm_clear
    from steward.ego.assembly_edge import corners
    rt = controller.rt; est = rt.estimator; ref = record['reference']
    diag = diagnostic(controller, 'native')
    def reject(reason):
        diag['reason'] = reason
        return None
    if raw is None and getattr(rt, 'segmentation', None) is None and not getattr(rt, '_precision_native_frames', None) and not (getattr(controller, '_edge_reuse_frames', False) and getattr(rt, '_latest_native_frames', None)):
        return reject('no_native_stream')
    origin = np.asarray(ref['position']); target = origin.copy()
    target[2] = ref.get('landingTopM', origin[2]+ref['size'][2]/2)
    clear = arm_clear(est, target, est.primary_camera)
    diag['centreRayBlocked'] = not clear
    if not clear and not face:
        return reject('arm_occlusion')
    if raw is None and getattr(controller, '_edge_reuse_frames', False):
        frames = [getattr(rt, name, {}).get(est.primary_camera) for name in
                  ('_latest_native_frames', '_precision_native_frames')]
        frames = [f for f in frames if f and after < f['header']['stamp'] <= controller.ex._state()['stamp']]
        if not frames:
            return reject('no_cached_native_frame')
        raw = max(frames, key=lambda f:f['header']['stamp'])
        diag['reusedExistingFrame'] = True
    if raw is None:
        raw = registration.acquire(controller, dict(entity=ref['entity'], body=ref['entity'], position=target.tolist()))
    now = controller.ex._state()['stamp']
    if raw is None:
        return reject('no_native_frame')
    diag['frameAgeS'] = round(now-raw['header']['stamp'], 3)
    if not after < raw['header']['stamp'] <= now or now-raw['header']['stamp'] > 20:
        return reject('frame_not_fresh_after_release')
    cam = est.camera_pose(raw['header']['odometry'], raw['camera'])
    depth = raw['depth']; intr = dict(width=depth.shape[1], height=depth.shape[0], fovy=raw['fovy'])
    items = [ref] if face else [ref, *record.get('parts', [])]
    envelope = np.vstack([corners(p, origin, np.eye(3)) for p in items])+origin
    pad = .15*np.linalg.norm(ref['size'])
    lo, hi = envelope.min(axis=0)-pad, envelope.max(axis=0)+pad
    if face:
        # A tilted face can rise well beyond the nominal flat body's height.
        rise = .5*np.linalg.norm(ref['size'][:2])*math.sin(math.radians(35))
        lo[2] = min(lo[2], target[2]-rise-ref['size'][2])
        hi[2] = max(hi[2], target[2]+rise+ref['size'][2])
    from itertools import product
    bounds = np.array([np.where(s, hi, lo) for s in product((False, True), repeat=3)])
    uv, forward = holes.project(bounds, cam, intr)
    if not np.isfinite(uv).all() or np.min(forward) <= .05:
        return reject('roi_behind_camera')
    clipped = bool(np.any(uv.min(axis=0) < 1) or np.any(uv.max(axis=0) >= [depth.shape[1]-1, depth.shape[0]-1]))
    diag['roiClipped'] = clipped
    if clipped and not face:
        return reject('roi_out_of_frame')
    low = np.maximum(0, np.floor(uv.min(axis=0))).astype(int)
    high = np.minimum([depth.shape[1]-1, depth.shape[0]-1], np.ceil(uv.max(axis=0))).astype(int)
    if np.any(low >= high):
        return reject('roi_out_of_frame')
    yy, xx = np.mgrid[low[1]:high[1]+1, low[0]:high[0]+1]
    xx, yy = xx.ravel(), yy.ravel(); z = depth[yy, xx]
    valid = np.isfinite(z) & (z > .05)
    diag.update(roiPixels=len(z), validPixels=int(valid.sum()), invalidFraction=round(float(1-valid.mean()), 4))
    if valid.sum() < 12 or (not face and valid.mean() < .95):
        return reject('invalid_depth')
    points = holes.unproject(np.c_[xx[valid], yy[valid]], z[valid], cam, intr)
    table = rt.ego_state().find(ref['support'])
    if table is None:
        return reject('no_support_height')
    above = points[:, 2] > entity_box(table)[2]+(.025*min(ref['size']) if partial_face else .004)
    diag['aboveSupportPoints'] = int(above.sum())
    if not face:
        border = ((xx[valid] <= low[0]+1) | (xx[valid] >= high[0]-1)
                  | (yy[valid] <= low[1]+1) | (yy[valid] >= high[1]-1))
        if np.any(above & border):
            return reject('outline_touches_roi')
        in_xy = np.all((points[:, :2] >= lo[:2]) & (points[:, :2] <= hi[:2]), axis=1)
        if np.any(above & in_xy & (points[:, 2] >= hi[2])):
            return reject('outline_exceeds_height')
    keep = np.all((points >= lo) & (points <= hi), axis=1) & above
    diag['bodyVolumePoints'] = int(keep.sum())
    points = points[keep]
    if face:
        visible = visible_rays(points, np.asarray(cam['p']), [*est.body_boxes(), *exclude_boxes])
        diag['occludedPoints'] = int((~visible).sum())
        points = points[visible]
    diag['localPoints'] = len(points)
    if len(points) < (6 if partial_face else 12):
        return reject('no_visible_body_points')
    view = dict(source='rgbd_local', points=points, stamp=raw['header']['stamp'],
                truncated=False, invalidDepthFraction=float(1-valid.mean()))
    local = (points-origin) @ matrix_from_quat(quat_from_yaw(ref['yaw']))
    body = local[np.all(abs(local[:, :2]) <= np.asarray(ref['size'][:2])*.6, axis=1)]
    diag['bodyPoints'] = len(body)
    if len(body) < (6 if partial_face else 12):
        return reject('insufficient_body_points')
    coverage = np.ptp(body[:, :2], axis=0)/np.asarray(ref['size'][:2])
    diag['coverage'] = np.round(coverage, 3).tolist()
    if not partial_face and np.any(coverage < .55):
        return reject('insufficient_face_coverage')
    # Face-only data is consumed by the broad fit below, never as a complete
    # envelope or independent motion evidence. Its missing pixels stay reported.
    diag['reason'] = 'local_face_points' if face else 'local_geometry'
    return view
