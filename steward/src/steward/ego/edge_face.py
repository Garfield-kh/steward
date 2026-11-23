"""Fit the exposed body face in already captured depth, without new looks."""
import math
import numpy as np
from steward.math3d import matrix_from_quat, quat_from_yaw


def fit(points, reference, diagnostics=None, parts=(), support_top=None):
    from steward.ego import edge_retention
    strict = edge_retention.enabled()
    diag = diagnostics if diagnostics is not None else {}
    p = np.asarray(points, float)
    if p.ndim != 2 or p.shape[1] != 3:
        diag['reason'] = 'invalid_points'; return None
    p = p[np.isfinite(p).all(axis=1)]
    origin = np.asarray(reference['position']); size = np.asarray(reference['size'])
    rotation = matrix_from_quat(quat_from_yaw(reference['yaw']))[:2, :2]
    local = (p[:, :2]-origin[:2]) @ rotation
    keep = np.all(abs(local) <= size[:2]*.49, axis=1)
    target_z = reference.get('landingTopM', origin[2]+size[2]/2)
    height_noise = .025*min(size)
    if strict and support_top is not None:
        keep &= p[:,2] > support_top+max(3*height_noise,.1*size[2])
    if strict:
        diag.update(targetFaceZM=float(target_z), supportTopM=support_top)
        keep &= abs(p[:,2]-target_z) <= size[2]/2+max(size[:2])/2
    # Known attached parts are masked by their own oriented footprints, not by
    # identity labels (which may merge with the body after release).
    for part in parts:
        if (part['position'][2]+part['size'][2]/2 < target_z-3*height_noise if strict
                else part['position'][2] <= origin[2]):
            continue
        pr = matrix_from_quat(quat_from_yaw(part.get('yaw', 0.)))[:2, :2]
        q = (p[:, :2]-np.asarray(part['position'])[:2]) @ pr
        keep &= ~np.all(abs(q) <= np.asarray(part['size'][:2])/2, axis=1)
    p, local = p[keep], local[keep]
    diag['bodyPoints'] = len(p)
    if len(p) < 6:
        diag['reason'] = 'insufficient_face_pixels'; return None
    # Deterministic spatial sampling bounds the cost at native resolution and
    # prevents the densely imaged near end from dominating the entire fit.
    cells = np.floor((local/size[:2]+.5)*24).astype(int)
    _, inverse = np.unique(cells, axis=0, return_inverse=True)
    samples = []
    for k in range(inverse.max()+1):
        q = p[inverse == k]
        if len(q): samples.append(np.median(q, axis=0))
    q = np.asarray(samples); xy = (q[:, :2]-origin[:2]) @ rotation
    if len(q) < 6:
        diag['reason'] = 'insufficient_spatial_samples'; return None
    design = np.c_[q[:, :2]-origin[:2], np.ones(len(q))]
    # Scale tolerance from registered thickness; not an evaluator angle gate.
    noise = max(np.finfo(float).eps, .025*min(size))
    hypotheses = []
    for i in range(len(q)):
        distances = np.linalg.norm((xy-xy[i])/size[:2], axis=1)
        neighbours = np.argsort(distances)[:min(12, len(q))]
        if np.linalg.matrix_rank(design[neighbours]) < 3: continue
        coef = np.linalg.lstsq(design[neighbours], q[neighbours, 2], rcond=None)[0]
        residual = abs(design @ coef-q[:, 2])
        inliers = residual <= noise
        if inliers.sum() < 6: continue
        for _ in range(2):
            coef = np.linalg.lstsq(design[inliers], q[inliers, 2], rcond=None)[0]
            inliers = abs(design @ coef-q[:, 2]) <= noise
            if inliers.sum() < 6: break
        if inliers.sum() < 6 or np.linalg.norm(coef[:2]) > 1: continue
        if strict and abs(coef[2]-target_z) > size[2]/2+3*noise:
            diag['rejectedPlaneIdentity'] = 'height_outside_body_face'; continue
        coverage = np.ptp(xy[inliers], axis=0)/size[:2]
        if max(coverage) < .4 or min(coverage) < .15: continue
        # Broad support across both axes beats a small horizontal attachment.
        score = float(np.prod(coverage))*int(inliers.sum())
        hypotheses.append((score, coef, inliers))
    if hypotheses:
        _, coef, inliers = max(hypotheses, key=lambda h:h[0])
        error = float(np.quantile(abs(design[inliers] @ coef-q[inliers, 2]), .90))
        normal = np.r_[-coef[:2], 1.]; normal /= np.linalg.norm(normal)
        diag.update(reason='robust_spatial_body_plane', inliers=int(inliers.sum()), residualM=error)
        return dict(point=[*origin[:2], float(coef[2])], normal=normal.tolist(),
            residualM=error, points=int(inliers.sum()), noiseM=noise, method=diag['reason'])
    # If a full plane has too few inliers, compare robust heights at the two
    # ends along EACH visible axis. This still measures depth, never commanded upZ.
    if strict and len(q)<12:
        diag['reason']='insufficient_face_identity_samples'; return None
    slope = np.zeros(2); measured = 0
    for axis in range(2):
        lo, hi = np.quantile(xy[:, axis], [.20, .80])
        left, right = xy[:, axis] <= lo, xy[:, axis] >= hi
        distance = float(np.median(xy[right, axis])-np.median(xy[left, axis]))
        if distance <= size[axis]*.25: continue
        slope[axis] = (np.median(q[right, 2])-np.median(q[left, 2]))/distance
        measured += 1
    if not measured:
        diag['reason'] = 'no_visible_end_separation'; return None
    world_slope = rotation @ slope
    normal = np.r_[-world_slope, 1.]; normal /= np.linalg.norm(normal)
    height = float(np.median(q[:, 2]-(q[:, :2]-origin[:2]) @ world_slope))
    if strict and (measured<2 or abs(height-target_z)>size[2]/2+3*noise
            or np.quantile(abs(q[:,2]-height-(q[:,:2]-origin[:2]) @ world_slope),.90)>3*noise):
        diag['reason']='end_heights_not_a_body_plane'; return None
    diag.update(reason='end_height_difference', measuredAxes=measured)
    return dict(point=[*origin[:2], height], normal=normal.tolist(), residualM=noise,
        points=len(q), noiseM=noise, method=diag['reason'], uncertain=True)


def measure(controller, reference, view, after):
    from steward.ego import edge_retention
    from steward.ego.assembly_edge_observation import native_view, merged_view
    from steward.ego.assembly_edge import entity_box
    rt = controller.rt; now = controller.ex._state()['stamp']; diagnostics = {}
    record = getattr(controller, '_edge_measure_record', None) or getattr(rt.estimator.stationary_assemblies, 'records', {}).get(reference['entity'], {})
    # Use the new reference, while carrying only the member geometry needed for
    # masking. The old reference may still be installed during flip rebasing.
    local_record = dict(record, reference=reference)
    candidates = [view]
    candidates.append(merged_view(controller, local_record, after))
    frames = {}
    for name in ('_latest_native_frames', '_precision_native_frames'):
        for camera, raw in getattr(rt, name, {}).items():
            if raw and after < raw['header']['stamp'] <= now and now-raw['header']['stamp'] <= 20:
                if raw['header']['stamp'] >= frames.get(camera, {}).get('header', {}).get('stamp', -1):
                    frames[camera] = raw
    # Multiple cameras here are already captured samples, never new observations.
    for camera, raw in frames.items():
        sample = native_view(controller, local_record, after, face=True, raw=raw, partial_face=True)
        diagnostics[camera] = dict(getattr(controller, '_edge_observation_diag', {}).get('native', {}))
        if sample is not None: candidates.append(sample)
    if not frames:
        from steward.ego.edge_result import cached_view
        candidates.append(cached_view(controller, local_record, after, face=True))
    surface = rt.ego_state().find(reference['support']); results = []
    for index, sample in enumerate(candidates):
        now = controller.ex._state()['stamp']
        if sample is None or sample.get('source') not in ('rgbd_segment', 'rgbd_assembly', 'rgbd_local') or not after < sample.get('stamp', -1) <= now or now-sample['stamp'] > 20:
            continue
        points = np.asarray(sample.get('points', []))
        if points.ndim != 2 or points.shape[1] != 3: continue
        if surface is not None:
            points = points[points[:, 2] > entity_box(surface)[2]+.025*min(reference['size'])]
        diag = diagnostics.setdefault('fit'+str(index), {})
        contact_ref = local_record.get('contactReference', {})
        if edge_retention.enabled() and contact_ref and contact_ref.get('transportedAt', -math.inf) < reference.get('ownReleaseAt', -math.inf):
            diag['reason']='stale_body_face_reference'; continue
        if edge_retention.enabled() and contact_ref:
            rot = matrix_from_quat(quat_from_yaw(reference['yaw']))[:2,:2]
            displacement = (np.asarray(contact_ref['position'])[:2]-reference['position'][:2]) @ rot
            if np.any(abs(displacement) > (np.asarray(reference['size'][:2])+max(contact_ref['size'][:2]))/2):
                diag['reason']='body_face_outside_current_envelope'; continue
        face_ref = dict(reference, **contact_ref)
        if edge_retention.enabled() and contact_ref:
            face_ref['landingTopM'] = face_ref['position'][2]+face_ref['size'][2]/2
        patch = fit(points, face_ref, diag, local_record.get('parts', []),
            entity_box(surface)[2] if edge_retention.enabled() and surface is not None else None)
        if patch is None: continue
        target_z = reference.get('landingTopM', reference['position'][2]+reference['size'][2]/2)
        if abs(patch['point'][2]-target_z) > reference['size'][2]+max(reference['size'][:2])*.5:
            diag['reason'] = 'face_height_inconsistent'; continue
        tilt = math.degrees(math.acos(np.clip(patch['normal'][2], -1, 1)))
        span = max(reference['size'][:2])*.5
        flat_angle = math.degrees(math.atan2(max(3*patch['residualM'], 3*patch['noiseM']), span))
        results.append(dict(estimatedTiltDeg=tilt, tiltAfter=tilt, observed=True,
            level=(None if edge_retention.enabled() and patch.get('uncertain') else tilt <= flat_angle), levelToleranceDeg=flat_angle,
            reason=patch['method'], observedAt=sample['stamp'], faceSource=sample['source'],
            patch=patch, observationDiagnostics=diagnostics, extraObservations=0,
            unconfirmed=bool(patch.get('uncertain'))))
    if results:
        return min(results, key=lambda r:(r['patch'].get('uncertain', False), -r['patch']['points']))
    return dict(estimatedTiltDeg=None, tiltAfter=None, observed=False, level=None,
        reason='no_visible_face_in_existing_frames', observationDiagnostics=diagnostics,
        extraObservations=0, unconfirmed=True)
