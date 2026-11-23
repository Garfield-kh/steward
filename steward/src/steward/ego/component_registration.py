"""Pre-pick and pre-place component registration (Ego RGB-D only).

The measured rigid layout is accepted independently of displacement from memory.
The saved component layout supplies scale; observations supply translation/yaw.
"""
from steward.ego import action_budget as ab, capabilities
from copy import deepcopy
from itertools import permutations, combinations
import math
import numpy as np

from steward.math3d import matrix_from_quat, quat_from_yaw

MAX_AGE_S = 90.
MAX_NATIVE_FRAMES = 5
AXIS_LIMIT = math.radians(10)
# The existing request_precision_view/assembly_precision close-view envelope.
# Resolution and plane-fit tests below may require a still closer viewpoint.
# No separate range cap: a robot standing beside a support cannot bring its camera within a
# fixed short distance of a feature lying well inside the support, and the pixel-footprint
# budget below already bounds how far and how oblique an admitted view may be.
CLOSE_RANGE_M = math.inf


def rotation(yaw):
    return matrix_from_quat(quat_from_yaw(yaw))[:2, :2]


def fit_layout(local, detected, expected, preferred_yaw, tolerance=.0035):
    """Full one-to-one proper rigid fit, without scale or nearest-hole guesses.

    Different equally plausible subsets are rejected. Symmetries of the SAME
    subset retain the nominal heading and are explicitly reported as ambiguous.
    """
    local = np.asarray(local, float)
    expected = np.asarray(expected, float)
    free = [h for h in detected if h.get('available', True)]
    if (local.ndim != 2 or local.shape[1] != 2 or len(local) < 1
            or not np.isfinite(local).all() or expected.shape != local.shape):
        return None, 'insufficient_layout'
    if len(free) < len(local):
        return None, 'not_enough_visible_holes'
    if math.perm(len(free), len(local)) > 4096:
        return None, 'layout_search_limit'
    if any(np.linalg.norm(a-b) < 1e-6 for i,a in enumerate(local) for b in local[i+1:]):
        return None, 'degenerate_layout'
    seen = np.array([h['point'][:2] for h in free], float)
    if not np.isfinite(seen).all() or not np.isfinite(expected).all():
        return None, 'invalid_geometry'
    centred = local-local.mean(axis=0)
    choices = []
    for indices in permutations(range(len(free)), len(local)):
        points = seen[list(indices)]
        selected = [free[i] for i in indices]
        if all('normal' in h for h in selected):
            anchor = np.asarray(selected[0]['point'])
            normal = np.asarray(selected[0]['normal'])
            if any(np.dot(normal, h['normal']) < math.cos(math.radians(10)) or
                   abs(np.dot(np.asarray(h['point'])-anchor, normal)) >
                   tolerance+float(h.get('relUncM', h.get('uncertaintyM', 0.))) for h in selected):
                continue
        r = rotation(preferred_yaw)
        if len(local) > 1:
            u, _, vt = np.linalg.svd((points-points.mean(axis=0)).T @ centred)
            r = u @ vt
            if np.linalg.det(r) < 0:
                u[:, -1] *= -1
                r = u @ vt
        offset = points.mean(axis=0)-r @ local.mean(axis=0)
        residual = float(np.max(np.linalg.norm(local @ r.T+offset-points, axis=1)))
        noise = max(float(h.get('relUncM', h.get('uncertaintyM', 0.))) for h in selected)
        span = float(np.max(np.linalg.norm(points[:,None]-points[None,:], axis=2)))
        # Do not claim a useful heading when pixel uncertainty spans the layout.
        axis_uncertainty = math.degrees(math.atan2(2*noise, span)) if len(local) > 1 else None
        # Two centres can differ by the sum of their measurement bounds. The
        # heading bound follows atan(2*u/span); odometry translation is common
        # to this entire raster and must not enter either test.
        residual_limit = min(tolerance, 2*noise+1e-6)
        if noise > tolerance or residual > residual_limit:
            continue
        mode = 'xy_yaw'
        yaw_reason = 'layout_axis_resolved'
        if axis_uncertainty is None or axis_uncertainty > math.degrees(AXIS_LIMIT):
            # Match the complete shape first, then retain the planned heading.
            # The centroid determines translation independently of heading.
            mode = 'xy'
            yaw_reason = 'single_hole' if axis_uncertainty is None else 'layout_axis_uncertain'
            r = rotation(preferred_yaw)
            offset = points.mean(axis=0)-r @ local.mean(axis=0)
            nominal_residual = float(np.max(np.linalg.norm(local @ r.T+offset-points, axis=1)))
            if nominal_residual > tolerance+2*noise:
                continue
        choices.append(dict(position=offset.tolist(), yaw=math.atan2(r[1,0], r[0,0]),
            residualM=residual, indices=indices, uncertaintyM=noise, axisUncertaintyDeg=axis_uncertainty,
            worldUncM=max(float(h.get('uncertaintyM', 0.)) for h in selected),
            residualLimitM=residual_limit, mode=mode, yawReason=yaw_reason,
            points=[deepcopy(free[i]) for i in indices]))
    if not choices:
        return None, 'layout_mismatch_or_uncertain'
    if len({tuple(sorted(c['indices'])) for c in choices}) != 1:
        return None, 'ambiguous_hole_subset'
    from steward.ego import placement_xy_compensation
    if placement_xy_compensation.enabled() and len(local) == 2:
        for c in choices:
            if c['mode'] == 'xy':
                c['residualSumM'] = sum(np.linalg.norm(local @ rotation(c['yaw']).T+c['position']-
                    np.array([h['point'][:2] for h in c['points']]), axis=1))
                continue
            fit = placement_xy_compensation.solve(local, [h['point'] for h in c['points']], preferred_yaw)
            c.update(fit)
            c['residualM'] = max(c['plannedResidualM'])
        lowest = min(c['residualSumM'] for c in choices)
        roundoff = np.finfo(float).eps*max(1., float(np.linalg.norm(local)), float(np.linalg.norm(seen)))*16
        choices = [c for c in choices if c['residualSumM'] <= lowest+roundoff]
    chosen = min(choices, key=lambda c:abs((c['yaw']-preferred_yaw+math.pi) % (2*math.pi)-math.pi))
    chosen['symmetricYaw'] = len(choices) > 1
    return chosen, None


def detect_native(raw, cam, expected, height, measurement_context=None):
    """Search a calibrated native window without any instance mask.

    The prior bounds only the search. Every output centre and local plane is
    measured by the existing support-hole contour detector, at native scale.
    """
    from steward.ego import support_holes as holes, component_measurement_admission
    component_measurement_enabled = component_measurement_admission.enabled()
    depth, rgb = raw['depth'], raw['rgb']
    result = dict(holes=[], framing=None, reason='invalid_native_frame', holeRejections={})
    if depth.ndim != 2 or rgb.shape != (*depth.shape, 3):
        return result
    expected = np.asarray(expected, float)
    pad = holes.MAX_LAYOUT_SHIFT
    low, high = expected.min(axis=0)-pad, expected.max(axis=0)+pad
    # Project the already permitted height band, not only its nominal plane.
    # At oblique incidence a lifted face otherwise shifts outside the image ROI.
    corners = [[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1])
               for z in (height-.03, height+.03)]
    intr = dict(width=depth.shape[1], height=depth.shape[0], fovy=raw['fovy'])
    uv, forward = holes.project(corners, cam, intr)
    if not np.isfinite(uv).all() or np.any(forward <= .05):
        result['reason'] = 'search_region_not_visible'
        return result
    result['framing'] = holes.framing(uv, depth.shape)
    if not component_measurement_enabled and (result['framing']['border'] or not result['framing']['centred']):
        result['reason'] = 'view_adjustment_failed'
        return result
    lo = np.floor(uv.min(axis=0)).astype(int)
    hi = np.ceil(uv.max(axis=0)).astype(int)
    if component_measurement_enabled:
        requested = [lo.tolist(), hi.tolist()]
        lo = np.maximum(lo, 0)
        hi = np.minimum(hi, np.array(depth.shape[::-1])-1)
        result['searchWindow'] = dict(expected=expected.tolist(), heightM=float(height),
            requested=requested, clipped=[lo.tolist(), hi.tolist()])
        if np.any(hi < lo):
            result['reason'] = 'search_region_not_visible'
            return result
    region = np.zeros(depth.shape, bool)
    region[lo[1]:hi[1]+1, lo[0]:hi[0]+1] = True
    request = measurement_context if capabilities.get('STEWARD_ASSEMBLY_PARTIAL_RIM_REGISTRATION') != '0' and component_measurement_enabled else None
    context = request if request and request.get('admitAppearance', True) else None
    cached = request.get('priorDetection', {}).get('_candidateExtraction') if request else None
    if cached and cached['rawIdentity'] == id(raw) and cached['searchWindow'] == result.get('searchWindow'):
        # This cache exists only within one inhand.measure call on the SAME raw
        # raster. It avoids repeated segmentation, never acquisition/freshness checks.
        detected = deepcopy(cached['accepted']+cached['deferred'])
        result['holeRejections'] = deepcopy(cached['rejections'])
        result['holeRejections'].pop('unresolved_neutral_contrast', None)
        for h in detected:
            h.update(evidenceKind='complete_contour', candidateSource='closed_contour_and_local_depth_plane',
                     identityStatus='requires_interface_layout')
    else:
        kwargs = dict(calibrated=True) if component_measurement_enabled else {}
        deferred = []
        if context is not None: kwargs['measurement_context'] = context
        if request and not context: kwargs['candidate_sink'] = deferred
        detected = holes.detect_region(raw, cam, region, result['holeRejections'], **kwargs)
        if request and not context:
            result['_candidateExtraction'] = dict(rawIdentity=id(raw), searchWindow=result.get('searchWindow'),
                accepted=deepcopy(detected), deferred=deferred, rejections=deepcopy(result['holeRejections']))
    if context is not None and context.get('kind') == 'existing_inhand_interface':
        from steward.ego.support_hole_edges import extract
        anchors = [h for h in detected if abs(h['point'][2]-height) <= .03
                   and h['normal'][2] >= math.cos(math.radians(25))
                   and np.all(np.asarray(h['point'][:2]) >= low)
                   and np.all(np.asarray(h['point'][:2]) <= high)]
        # Prefer a complete consistent image pair. An unrelated second contour
        # must not suppress the partial-rim search for the actual interface.
        from steward.ego.component_measurement import centre_uncertainty
        for anchor in anchors:
            anchor['relUncM'] = centre_uncertainty(anchor, raw, cam)['relativeM']
        complete, _ = fit_layout(expected-expected.mean(0), anchors, expected, 0.,
                                 tolerance=context.get('precisionM', .002))
        if complete is None:
            detected += extract(raw, cam, region, anchors, result['holeRejections'])
    if component_measurement_enabled:
        result['afterDetectionCount'] = len(detected)
    for h in detected:
        # detect_region adds the header's position sigma linearly. Preserve its
        # world uncertainty for temporal comparisons; remove ONLY this known
        # common translation for a fit within one raster (rounding allowance).
        common = float(raw['header'].get('odometry', {}).get('sigma', {}).get('pos', 0.))
        from steward.ego.component_measurement import centre_uncertainty
        estimate = (h['centreUncertainty'] if h.get('evidenceKind') == 'partial_hole_edge'
                    else centre_uncertainty(h, raw, cam))
        h['centreUncertainty'] = estimate
        h['relUncM'] = estimate['relativeM']
        # Keep the detector's conservative world bound for temporal settling.
        h['uncertaintyM'] = max(h['uncertaintyM'], common+h['relUncM'])
        point = np.asarray(h['point'])
        # No table/side-face substitution: retain only upward planes near the
        # registered exposed face, and keep all plausible subsets for ambiguity checks.
        if (abs(point[2]-height) > .03 or h['normal'][2] < math.cos(math.radians(25))
                or np.any(point[:2] < low) or np.any(point[:2] > high)):
            counts = result['holeRejections']
            counts['outside_support_face'] = counts.get('outside_support_face', 0)+1
            continue
        if component_measurement_enabled:
            h['nativeEvidence'] = dict(stamp=raw['header']['stamp'], camera=raw.get('camera'),
                contourPlaneAccepted=True, source=h.get('candidateSource', 'native_contour_and_raw_rim_depth'))
        result['holes'].append(h)
    if component_measurement_enabled:
        result['afterFaceCount'] = len(result['holes'])
    result['reason'] = None if result['holes'] else 'no_holes_in_region'
    if component_measurement_enabled and not result['holes'] and result['holeRejections'].get('invalid_rim_depth'):
        result['depthReason'] = 'missing_or_invalid_rim_depth'
    return result


def correction(fit, spot, yaw):
    shift = np.asarray(fit['position'])-spot
    angle = 0. if fit.get('mode') == 'xy' else (fit['yaw']-yaw+math.pi) % (2*math.pi)-math.pi
    bounded = np.isfinite(shift).all() and np.isfinite(angle)
    return bool(bounded), dict(translationM=shift.tolist(), yawDeg=math.degrees(angle),
        translationLimitM=None, yawLimitDeg=None, basis='rigid_layout_consistency')


def fallback(report, reason):
    # One operation may try several frames; count its fallback once.
    report.update(fallbacks=1, fallbackReasons={reason:1}, reason=reason, applied=False)


def view_quality(controller, frame, local, tolerance, odometry=None, intr=None):
    """Coarse view admission; measured contour/plane precision is checked later.

    A quarter pixel is the retained systematic raster bias. Do not reject a
    whole view merely because heading may be unresolved for a short layout.
    """
    from steward.ego import support_registration as sr
    est = controller.rt.estimator
    geometry = sr.view_geometry(controller, frame, odometry)
    cam = est.camera_pose(odometry, est.primary_camera)
    intr = intr or frame.get('nativeIntrinsics') or est.camera_intrinsics(est.primary_camera)
    ray = np.asarray(frame['position'])-cam['p']
    distance = float(np.linalg.norm(ray))
    cosine = abs(float(ray[2]))/max(distance, 1e-12)
    focal = intr['height']/(2*math.tan(math.radians(intr['fovy'])/2))
    pixel = distance/(focal*max(cosine, 1e-12))
    local = np.asarray(local)
    budget = tolerance/.25
    return dict(usable=bool(geometry['usable'] and distance <= CLOSE_RANGE_M and pixel <= budget),
        rangeM=round(distance, 5), incDeg=round(math.degrees(math.acos(min(1., cosine))), 2),
        pxM=round(pixel, 7), pxLimitM=round(budget, 7), resolutionModel='quarter_pixel_bias_then_measured_fit')



def adjust_view(controller, job, frame, event, local, tolerance):
    """Use the existing support viewpoint search and navigator, with an extra
    component-only resolution admission. No global patch or screw-path change.
    """
    report = job.postcondition.get('componentHoleAlignment')
    if report is not None:
        report['stage'] = 'view_search'
    from steward.ego import support_registration as sr
    class RuntimeView:
        def __getattr__(self, name):
            return getattr(controller.rt, name)

        def predict(self, skill, params):
            if skill == 'navigate' and not view_quality(controller, frame, local, tolerance, params)['usable']:
                return dict(withinRiskLimit=False, feasible=False, issues=['component_pixel_resolution'])
            return controller.rt.predict(skill, params)

    class ControllerView:
        rt = RuntimeView()

        def __getattr__(self, name):
            return getattr(controller, name)

    from steward.ego import placement_xy_compensation
    if placement_xy_compensation.enabled() or (getattr(controller, '_component_spec', None) or {}).get('recoveryReference'):
        if report is not None and report.get('placementXYCompensationViewChanges', 0): return
        if report is not None: report['placementXYCompensationViewChanges'] = 1
    sr.adjust_view(ControllerView(), job, frame, event)


def entry_clear(est, point, camera):
    """Existing robot-ray veto plus the carried component's inferred envelope.
    Its own holes must never be mistaken for the exposed support's holes.
    """
    from steward.ego.registered_alignment import arm_clear
    if not arm_clear(est, point, camera):
        return False
    held = est.tracks.get(est.held_id)
    if held is None:
        return True
    pose = est.held_pose().get(held.id)
    if pose is None:
        return False
    centre, yaw = pose
    interface = held.held_interface
    r = (est.gripper_rotation(est.held_gripper) @ np.asarray(interface['rotation']) if interface else
         matrix_from_quat(quat_from_yaw(yaw)))
    origin = np.asarray(est.camera_pose(camera=camera)['p'])
    start, end = (np.asarray([origin, point])-centre) @ r
    delta = end-start
    half = held.size/2+math.sqrt(float(np.max(held.var)))
    near, far = 0., 1.
    for axis in range(3):
        if abs(delta[axis]) < 1e-12:
            if abs(start[axis]) > half[axis]:
                return True
        else:
            a, b = sorted(((-half[axis]-start[axis])/delta[axis], (half[axis]-start[axis])/delta[axis]))
            near, far = max(near, a), min(far, b)
            if near > far:
                return True
    return False


def observe(controller, job, spec, report):
    """Same native detector, framing/search and consecutive-frame settle policy
    as support registration. Fit within each frame in the acquisition base.
    """
    from steward.ego import support_registration as sr
    rt = controller.rt; est = rt.estimator
    local, expected = np.asarray(spec['local']), np.asarray(spec['expected'])
    spot, yaw, frame = np.asarray(spec['spot']), spec['yaw'], spec['frame']
    tolerance = float(controller.ex.config.interface_precision_m)
    event = dict(viewCandidates=[])
    try:
        for _ in range(sr.VIEW_ATTEMPTS):
            quality = view_quality(controller, frame, local, tolerance)
            if quality['usable']:
                break
            report['stage'] = 'view_search'
            adjust_view(controller, job, frame, event, local, tolerance)
        quality = view_quality(controller, frame, local, tolerance)
        report.update({k:v for k,v in quality.items() if k != 'usable'})
        if not quality['usable']:
            fallback(report, 'view_adjustment_failed'); return None
        samples = []; view_retried = False
        for _ in range(MAX_NATIVE_FRAMES):
            controller.check()
            report['stage'] = 'native_acquisition'
            raw = sr.acquire(controller, frame)
            if raw is None:
                if spec.get('recoveryReference') and not view_retried:
                    view_retried = True
                    adjust_view(controller, job, frame, event, local, tolerance)
                    continue
                fallback(report, 'no_registration_frame'); return None
            controller._component_native_frame = raw
            odom = raw['header']['odometry']
            cam = est.camera_pose(odom, raw['camera'])
            frame['nativeIntrinsics'] = dict(height=raw['depth'].shape[0], fovy=raw['fovy'])
            quality = view_quality(controller, frame, local, tolerance, odom,
                frame['nativeIntrinsics'])
            report.update({k:v for k,v in quality.items() if k != 'usable'})
            report['stage'] = 'native_detection'
            found = (detect_native(raw, cam, expected, frame['position'][2]) if quality['usable'] else
                     dict(holes=[], reason='view_adjustment_failed', holeRejections={}))
            report['holeCount'] = len(found['holes'])
            report['holeRejections'] = found.get('holeRejections', {})
            report['detectionReason'] = found['reason']
            if not found['holes'] and not report['holeRejections']:
                report['holeRejections'] = {'no_closed_recess_candidate': 1}
            report['stage'] = 'layout_fit'
            report['frames'].append(dict(stamp=raw['header']['stamp'], holeCount=len(found['holes']),
                reason=found['reason']))
            # Keep the receipt bounded even after a pre-place refresh.
            report['frames'] = report['frames'][-MAX_NATIVE_FRAMES:]
            occupied = [a['entry'] for a in est.fastening_attempts.values()
                        if a.get('body') == spec['body'] and a.get('releaseExecuted') and a.get('entry')]
            for h in found['holes']:
                h['available'] = h.get('available', True) and not any(
                    np.linalg.norm(np.asarray(h['point'])-p) <= .006 for p in occupied)
            base = np.array([odom['x'], odom['y']])
            r = rotation(odom['yaw'])
            base_holes = deepcopy(found['holes'])
            for h in base_holes:
                h['point'][:2] = ((np.asarray(h['point'][:2])-base) @ r).tolist()
                h['normal'][:2] = (np.asarray(h['normal'][:2]) @ r).tolist()
            recovery = spec.get('recoveryReference')
            fitted, reason = (None, found['reason']) if found['reason'] else fit_layout(
                local, base_holes, (expected-base) @ r, yaw-odom['yaw'], tolerance)
            if recovery and not found['reason'] and reason != 'layout_search_limit':
                from steward.ego.geometry_relocalization import identity, eligible
                track = est.tracks.get(spec['body'])
                outline = (track.manipulation_view or track.assembly_observation) if track else None
                if track and track.category not in (recovery['category'], recovery.get('family')):
                    outline = None
                # Evaluate structure BEFORE rejecting equal-distance subsets.
                # The detector and rigid-fit tolerances remain unchanged.
                choices = []
                subsets = ([base_holes] if len(local) != 2 else combinations(base_holes, 2))
                for subset in subsets:
                    candidate, _ = fit_layout(local, subset, (expected-base) @ r, yaw-odom['yaw'], tolerance)
                    if candidate is None: continue
                    world_points = deepcopy(candidate['points'])
                    for h in world_points:
                        h['point'][:2] = (r @ h['point'][:2]+base).tolist()
                        h['normal'][:2] = (r @ h['normal'][:2]).tolist()
                    evidence = identity(recovery, spec['pairs'], world_points, found['holes'],
                                        outline, raw['header']['stamp'], tolerance)
                    report['geometryIdentity'] = evidence
                    if evidence['confirmed']:
                        candidate['geometryIdentity'] = evidence
                        prior = recovery.get('componentHoleRegistration') or {}
                        candidate['prePickRelocalized'] = bool(spec.get('prePick') or (
                            eligible(recovery, prior) and prior.get('part') == est.held_id and
                            prior.get('holePairs') == spec['pairs']))
                        choices.append(candidate)
                fitted = choices[0] if len(choices) == 1 else None
                reason = ('ambiguous_interface_geometry' if len(choices) > 1 else
                    report.get('geometryIdentity', {}).get('reason', reason))
            if fitted is None:
                report['frames'][-1]['reason'] = reason
                if not view_retried and reason in ('view_adjustment_failed', 'search_region_not_visible',
                        'no_holes_in_region', 'not_enough_visible_holes', 'layout_mismatch_or_uncertain',
                        'ambiguous_hole_subset', 'ambiguous_interface_geometry',
                        'independent_structure_unavailable', 'support_plane_unresolved'):
                    view_retried = True; samples.clear()
                    adjust_view(controller, job, frame, event, local, tolerance)
                    continue
                fallback(report, reason); return None
            report['centreUncertainty'] = [h.get('centreUncertainty') for h in fitted['points']]
            fitted['basePositionM'] = fitted['position']
            fitted['baseYawRad'] = fitted['yaw']
            fitted['position'] = (r @ fitted['basePositionM']+base).tolist()
            fitted['yaw'] += odom['yaw']
            for h in fitted['points']:
                h['point'][:2] = (r @ h['point'][:2]+base).tolist()
                h['normal'][:2] = (r @ h['normal'][:2]).tolist()
            anchor = np.mean([h['point'] for h in fitted['points']], axis=0)
            normal = np.mean([h['normal'] for h in fitted['points']], axis=0)
            ray = np.asarray(cam['p'])-anchor
            cosine = abs(float(ray @ normal))/(np.linalg.norm(ray)*np.linalg.norm(normal))
            report.update(rangeM=round(float(np.linalg.norm(ray)), 5),
                incDeg=round(math.degrees(math.acos(min(1., cosine))), 2),
                planeM=max(h['planeNoiseM'] for h in fitted['points']))
            # Header yaw error is also common within a frame, but not across stations.
            lever = max(float(np.linalg.norm(np.asarray(h['point'][:2])-base)) for h in fitted['points'])
            fitted['worldUncM'] += lever*float(odom.get('sigma', {}).get('yaw', 0.))
            bounded, delta = correction(fitted, spot, yaw)
            report.update(correction=delta, fitResidualM=fitted['residualM'],
                fitLimitM=fitted['residualLimitM'], relUncM=fitted['uncertaintyM'])
            if not bounded:
                fallback(report, 'correction_limit'); return None
            report['stage'] = 'entry_ray_check'
            if not all(entry_clear(est, h['point'], raw['camera']) for h in fitted['points']):
                fallback(report, 'robot_occludes_entry_ray'); return None
            report['stage'] = 'settle_frames'
            report['settleSamples'] = len(samples)+1
            samples.append((raw['header']['stamp'], fitted, deepcopy(odom)))
            if len(samples) >= 2:
                first, last = samples[-2:]
                shifts = np.linalg.norm(np.array([h['point'] for h in first[1]['points']])-
                                        np.array([h['point'] for h in last[1]['points']]), axis=1)
                # This is the existing support-registration settle policy. Its
                # world-frame uncertainty deliberately INCLUDES common odometry.
                limit = float(np.clip(first[1]['worldUncM']+last[1]['worldUncM'], .002, sr.SETTLE_SHIFT_MAX))
                if last[0] > first[0] and max(shifts) <= limit:
                    break
            controller.ex._wait(.25)
        else:
            fallback(report, 'support_not_settled'); return None
        saved_odom = {k:deepcopy(last[2][k]) for k in ('x', 'y', 'yaw', 'sigma') if k in last[2]}
        reg = dict(last[1], local=local.tolist(), holePairs=spec['pairs'], observedAt=last[0],
            part=spec['part'], body=spec['body'], referenceAt=spec['referenceAt'],
            source='native_rgbd_support_holes_rigid_fit', correction=delta,
            nominalPosition=spot.tolist(), nominalYaw=float(yaw), baseOdometry=saved_odom,
            frame='base_at_observation')
        # Raster contours stay in the local computation; do not repeat them in
        # every nested plan/reference copy in the skill receipt.
        from steward.ego import placement_xy_compensation
        if placement_xy_compensation.enabled(): placement_xy_compensation.retain(reg, reg['points'])
        reg.pop('points'); reg.pop('indices')
        report.update(registrationSuccesses=1, fallbacks=0, fallbackReasons={},
            reason='registered_pending_place', symmetricYaw=reg['symmetricYaw'],
            axisUncertaintyDeg=reg['axisUncertaintyDeg'], mode=reg['mode'],
            frame=reg['frame'], source=reg['source'], stage='awaiting_place', yawReason=reg['yawReason'])
        return reg
    finally:
        if report.get('reason') == 'registration_pending':
            report['pendingAt'] = report.get('stage', 'setup')
        candidates = event.get('viewCandidates', [])
        reasons = {}
        for candidate in candidates:
            for reason in candidate.get('rejections', []):
                reasons[reason] = reasons.get(reason, 0)+1
        report['view'] = dict(candidateCount=len(candidates), rejectionCounts=reasons,
            failureReason=event.get('viewSearchFailure'))


def prepare(controller, job, plan, part):
    """Observe the bare support before acquiring the component; bounded fallback."""
    from steward.ego.registered_alignment import arm_clear
    from steward.ego.fasten_release import placement_pairs
    from steward.ego.skills import SkillFailed
    rt = controller.rt; est = rt.estimator
    controller._component_spec = None
    report = dict(attempts=1, successes=0, registrationSuccesses=0, fallbacks=0,
        fallbackReasons={}, measuredYawErrorDeg=None, axisErrorDeg=None,
        orientationReason='release_not_observed', orientationVerifiedSuccesses=0,
        yawErrorPeriodDeg=180, thresholdDeg=5., applied=False, frames=[],
        reason='registration_pending', stage='setup', detector='native_contour_plane', rangeM=None,
        incDeg=None, holeCount=0, fitResidualM=None, frame=None, correction=None)
    job.postcondition['componentHoleAlignment'] = report
    with rt._lock:
        record = est.stationary_assemblies.records[plan['body']]
        if record['reference'].get('geometryRelocalization'):
            from steward.ego.geometry_relocalization import pending
            pending(record['reference'], est.stamp, 'prepick_current_interface')
    if est.held_id:
        fallback(report, 'hand_not_empty'); return None
    if getattr(rt, 'segmentation', None) is None:
        fallback(report, 'native_acquisition_unavailable'); return None
    scene = rt.ego_state(); body = scene.find(plan['body']); item = scene.find(part.id)
    params = dict(on=plan['body'], alignHoles=True,
                  **{k:plan[k] for k in ('hole','heldHole') if k in plan})
    try:
        _, offset, yaw, _ = rt.esim._place_hole_align(scene, item, body, params)
        spot = np.asarray(body.pose.value['p'])[:2]+rotation(record['reference']['yaw']) @ offset
        pairs = placement_pairs(rt.esim, item, body, spot, yaw, params)
        own = {h['id']:h for h in rt.esim._screw_holes(item)}
        local = np.array([own[p['partHole']]['p'][:2] for p in pairs])
        report['layoutCount'] = len(local)
        if not len(local):
            fallback(report, 'insufficient_layout'); return None
        expected = local @ rotation(yaw).T+spot
        ref = deepcopy(record['reference'])
        target = [*expected.mean(axis=0), ref.get('landingTopM', ref['position'][2]+ref['size'][2]/2)]
        frame = dict(entity=body.id, body=body.id, position=target, size=ref['size'],
                     rotation=matrix_from_quat(quat_from_yaw(ref['yaw'])).tolist())
        from steward.ego.support_holes import MAX_LAYOUT_SHIFT
        # Frame the complete bounded interface search, not an elongated/merged
        # body envelope that can force the camera back into an unusable far view.
        frame['viewBoundsM'] = [
            [*(expected.min(axis=0)-MAX_LAYOUT_SHIFT), target[2]],
            [*(expected.max(axis=0)+MAX_LAYOUT_SHIFT), target[2]]]
        spec = dict(local=local.tolist(), expected=expected.tolist(), spot=spot.tolist(), yaw=yaw,
            pairs=pairs, frame=frame, part=part.id, body=body.id, referenceAt=ref['registeredAt'],
            recoveryReference=ref if ref.get('geometryRelocalization') else None, prePick=True)
        controller._component_spec = spec
        if not arm_clear(est, target, est.primary_camera):
            controller.ex._withdraw(dict(object=body.id)); controller.check()
        return observe(controller, job, spec, report)
    except (SkillFailed, ValueError, KeyError, TypeError, AttributeError) as exc:
        controller.check()
        fallback(report, 'registration_unavailable')
        report['detail'] = type(exc).__name__
        return None
    finally:
        if not report['registrationSuccesses'] and not report['fallbacks']:
            fallback(report, 'registration_interrupted')


def publish_parent(root, report):
    """Count child attempts once, including the parent's existing retries."""
    previous = root.get('componentHoleAlignment', {})
    combined = deepcopy(report)
    for key in ('attempts', 'successes', 'registrationSuccesses', 'fallbacks', 'orientationVerifiedSuccesses'):
        combined[key] = previous.get(key, 0)+report.get(key, 0)
    reasons = dict(previous.get('fallbackReasons', {}))
    for reason, count in report.get('fallbackReasons', {}).items():
        reasons[reason] = reasons.get(reason, 0)+count
    combined.update(fallbackReasons=reasons, measurementScope='last_placement')
    root['componentHoleAlignment'] = combined


def invalid_reason(est, reg):
    record = est.stationary_assemblies.records.get(reg['body'])
    from steward.ego.geometry_relocalization import eligible
    if record and not eligible(record['reference'], reg):
        return 'geometry_relocalization_required'
    moved = any(e['entity'] == reg['body'] and e['stamp'] > reg['observedAt']
                for e in est.stationary_assemblies.motion_events)
    if (not record or record.get('hardMotion') or moved
            or record.get('displacementEvidence', {}).get('stamp', -math.inf) > reg['observedAt']
            or record['reference']['registeredAt'] != reg['referenceAt']
            or not 0 <= est.stamp-reg['observedAt'] <= MAX_AGE_S):
        return 'registration_invalidated'
    return None


def grasp_evidence(est, held, reg):
    """Report this grasp's native hole measurement when one was installed.
    Otherwise retain the inferred transform and propagate its track variance.
    """
    radius = float(np.max(np.linalg.norm(reg['local'], axis=1)))
    uncertainty = math.sqrt(float(np.max(held.var)))+radius*math.sqrt(float(held.yaw_var))
    interface = held.held_interface or {}
    measured = interface.get('inhandMeasurement')
    return dict(source=interface.get('source', 'ego_grasp_transform_inferred'),
        measured=bool(measured), uncertaintyM=round(measured['uncertaintyM'] if measured else uncertainty, 6),
        offsetM=held.held_offset.tolist() if held.held_offset is not None else None)


def refresh(controller, job, plan, params):
    """Return to the admitted dock, then register again with the load held.

    Replanning uses the latest acquisition base; no pre-pick world target is
    carried across the trip when this observation fails. A failed observation
    retains ordinary placement, with its lack of correction made explicit.
    """
    from steward.ego.skills import SkillFailed
    from steward.ego import placement_xy_compensation
    if placement_xy_compensation.enabled() and getattr(controller, '_placement_xy_compensation_replay', False): return plan
    spec = getattr(controller, '_component_spec', None)
    if not spec or params.get('on') != spec['body']:
        return plan
    rt = controller.rt; est = rt.estimator
    from steward.ego.component_report import missing, ordinary_plan
    if missing(controller,job,'refresh'):
        return ordinary_plan(controller,job,params)
    report = job.postcondition['componentHoleAlignment']
    # This motion consumes the already admitted place approach. Never bypass
    # docking/collision checks to gain a view while carrying a component.
    controller.ex._dock(job, plan)
    controller.check()
    with rt._lock:
        record = est.stationary_assemblies.records.get(spec['body'])
        old = deepcopy(record['reference'].get('componentHoleRegistration')) if record else None
        est.version += 1
    report['refresh'] = 'attempted'
    report['prePick'] = {k:report.get(k) for k in ('reason', 'rangeM', 'incDeg', 'holeCount')}
    if old and invalid_reason(est, old):
        report['invalidated'] = 'body_motion_or_age'
    fresh = None
    try:
        # Reference changes invalidate the old fit. New pixels still have to
        # fit the same requested local interface within the original bound.
        if record:
            spec = dict(spec, referenceAt=record['reference']['registeredAt'], prePick=False,
                recoveryReference=deepcopy(record['reference']) if record['reference'].get('geometryRelocalization') else None)
        fresh = observe(controller, job, spec, report)
        if fresh:
            if old:
                shift = float(np.linalg.norm(np.asarray(fresh['position'])-old['position']))
                report['refreshShiftM'] = round(shift, 6)
                if shift > old['worldUncM']+fresh['worldUncM']:
                    report['invalidated'] = 'observed_support_shift'
            held = est.tracks.get(est.held_id)
            if held:
                install(controller, job, fresh, held)
            installed = record and record['reference'].get('componentHoleRegistration')
            if not installed or installed.get('observedAt') != fresh['observedAt']:
                fresh = None
            report['refresh'] = 'registered' if fresh else 'invalidated'
        else:
            report['refresh'] = report['reason']
    except (SkillFailed, ValueError, KeyError, TypeError, AttributeError) as exc:
        controller.check()
        fresh = None
        fallback(report, 'refresh_unavailable')
        report.update(refresh='unavailable', detail=type(exc).__name__)
    if not fresh and record and old is not None:
        # Retain the failed-refresh prior for recovery, but do not plan a fresh
        # precision placement from a pre-trip target.
        with rt._lock:
            record['reference']['componentHoleRegistration'] = dict(old, historicalPrior=True)
            est.version += 1
    receipt = rt.predict('place', params)
    if fresh and not receipt.get('withinRiskLimit'):
        with rt._lock:
            if old is not None:
                record['reference']['componentHoleRegistration'] = dict(old, historicalPrior=True)
            else:
                record['reference'].pop('componentHoleRegistration', None)
            est.version += 1
        fallback(report, 'registered_place_not_admitted')
        receipt = rt.predict('place', params)
    if not receipt.get('withinRiskLimit'):
        raise SkillFailed('place after component observation: ' + '; '.join(receipt.get('issues', [])))
    replacement = rt._consume_receipt(receipt)
    deadline = getattr(controller.ex, '_primitive_deadline', None)
    controller.ex._begin(job, 'place', receipt)
    if deadline is not None:
        controller.ex._primitive_deadline = ab.limited(deadline, controller.ex._primitive_deadline)
    return replacement


def validate_target(rt, plan):
    """Last motion check before release; a newly moved body cannot use an old
    corrected plan. Keep the load if it needs another registration.
    """
    from steward.ego.skills import SkillFailed
    reg = plan.get('placementTarget', {}).get('componentHoleRegistration')
    if not reg:
        return
    if invalid_reason(rt.estimator, reg):
        record = rt.estimator.stationary_assemblies.records.get(reg['body'])
        if record and not record['reference'].get('geometryRelocalization'):
            record['reference'].pop('componentHoleRegistration', None)
        plan['componentInvalid'] = 'registration_invalidated'
        if record and record['reference'].get('geometryRelocalization'):
            from steward.ego.geometry_relocalization import pending
            pending(record['reference'], rt.estimator.stamp, 'registration_invalidated')
            reg['historicalPrior'] = True
            plan.setdefault('placement_xy_compensation', {}).update(confirmed=False, reason='geometry_pending_best_estimate_continue')
            return
        from steward.ego import placement_xy_compensation
        positive_motion = record and (record.get('hardMotion') or
            record.get('displacementEvidence', {}).get('stamp', -math.inf) > reg['observedAt'])
        positive_motion |= any(e['entity'] == reg['body'] and e['stamp'] > reg['observedAt']
            for e in rt.estimator.stationary_assemblies.motion_events)
        if placement_xy_compensation.enabled() and not positive_motion:
            plan.setdefault('placement_xy_compensation', {}).update(confirmed=False, reason='stale_target_best_estimate_continue')
            return
        raise SkillFailed('component support moved or registration expired; re-register before release')
    # Only the short move after the latest observation remains. This is an
    # uncertainty estimate, not measured odometry drift or a precision claim.
    current = rt.executor._state()['odometry']
    before = reg['baseOdometry']
    radius = float(np.linalg.norm(reg['basePositionM']))
    sigmas = [o.get('sigma') for o in (before, current)]
    reg['driftUncM'] = (sum(float(s.get('pos', 0.))+radius*float(s.get('yaw', 0.)) for s in sigmas)
                        if all(s is not None for s in sigmas) else None)
    reg['postViewTravelM'] = float(np.linalg.norm([current['x']-before['x'], current['y']-before['y']]))


def install(controller, job, reg, held):
    """Bind a registration to this grasp; never persist it as a new body pose."""
    if reg is None:
        return
    from steward.ego.component_report import missing
    if missing(controller,job,'install'):
        return
    est = controller.rt.estimator; report = job.postcondition['componentHoleAlignment']
    if not correction(reg, reg['nominalPosition'], reg['nominalYaw'])[0]:
        fallback(report, 'correction_limit'); return
    with controller.rt._lock:
        record = est.stationary_assemblies.records.get(reg['body'])
        if invalid_reason(est, reg):
            if record and not record['reference'].get('geometryRelocalization'):
                record['reference'].pop('componentHoleRegistration', None)
            fallback(report, 'registration_invalidated'); return
        assoc = held.grasp_association
        if not assoc or assoc['requested'] != reg['part']:
            fallback(report, 'grasp_identity_changed'); return
        reg['part'] = held.id
        report['grasp'] = grasp_evidence(est, held, reg)
        record['reference']['componentHoleRegistration'] = deepcopy(reg)
        if reg.get('prePickRelocalized'):
            record['reference']['geometryRelocalization'].update(status='registered',
                observedAt=reg['observedAt'], holePairs=deepcopy(reg['holePairs']))
            record['reference']['landingTopM'] = reg['geometryIdentity']['planePoint'][2]
            record['reference']['landingPlane'] = deepcopy(reg['geometryIdentity'])
        est.version += 1


def planning_registration(es, held, support):
    info = getattr(support, 'state_vars', {}).get('orientation')
    orientation = info.value if info else {}
    if orientation.get('stationaryAssembly', {}).get('hardMotion'):
        return None
    ref = orientation.get('assembly', {}).get('reference', {})
    reg = ref.get('componentHoleRegistration')
    from steward.ego.geometry_relocalization import eligible
    if not eligible(ref, reg): return None
    if (not reg or reg['part'] != held.id or reg['body'] != support.id
            or not 0 <= es.stamp-reg['observedAt'] <= MAX_AGE_S
            or ref['registeredAt'] != reg['referenceAt']):
        return None
    return reg


def planner_fit(planner, es, held, support, params):
    reg = planning_registration(es, held, support)
    if not reg or params.get('alignHoles') is False or params.get('flip') or params.get('roll'):
        return None
    if not correction(reg, reg['nominalPosition'], reg['nominalYaw'])[0]:
        return None
    own = {h['id']:h for h in planner._screw_holes(held)}
    if not all(p['partHole'] in own for p in reg['holePairs']):
        return None
    local = [own[p['partHole']]['p'][:2] for p in reg['holePairs']]
    if not np.allclose(local, reg['local'], atol=1e-5):
        return None
    # Selectors must still name this interface; a registration cannot leak into
    # a later independent request on another face/group.
    bh = {h['id'] for h in planner._select_holes(planner._screw_holes(support), params.get('hole'), support.id)}
    ph = {h['id'] for h in planner._select_holes(planner._screw_holes(held), params.get('heldHole'), held.id)}
    if not all(p['partHole'] in ph and p['bodyHole'] in bh for p in reg['holePairs']):
        return None
    from steward.ego.fasten_simple import offset_on
    # The planner's public API takes odometry-map coordinates. Materialize
    # them from the most recent acquisition base only at this boundary.
    odom = reg['baseOdometry']
    position = rotation(odom['yaw']) @ reg['basePositionM']+[odom['x'], odom['y']]
    if not correction(dict(reg, position=position), reg['nominalPosition'], reg['nominalYaw'])[0]:
        return None
    return support, offset_on(support, position), reg['baseYawRad']+odom['yaw'], [
        '[COMPONENT_REGISTRATION_COMPONENT_HOLES] recent base-frame fit; single-hole yaw remains nominal']


def post_orientation(controller, job, plan):
    """Independent part outline measures axis error; FK is not an observation."""
    report = job.postcondition.get('componentHoleAlignment')
    if report is None:
        return
    target = plan.get('placementTarget', {})
    report.update(measuredYawErrorDeg=None, axisErrorDeg=None, orientationVerifiedSuccesses=0)
    reg = target.get('componentHoleRegistration')
    released = bool(plan.get('releaseExecuted') or 'releasedAt' in plan)
    report.update(targetYawRad=target.get('yawRad'), targetPositionM=plan.get('spot'),
                  applied=bool(reg and released and not plan.get('componentInvalid')))
    if reg:
        report.update(correction=reg['correction'] if 'correction' in reg else report.get('correction'),
                      driftUncM=reg.get('driftUncM'), postViewTravelM=reg.get('postViewTravelM'))
    if plan.get('componentInvalid'):
        fallback(report, plan['componentInvalid'])
    report['successes'] = int(report['applied'] and bool(plan.get('releaseExecuted') or 'releasedAt' in plan))
    if report['applied']:
        report['reason'] = 'registered_target_applied'
        report['appliedCorrection'] = report.get('correction')
    else:
        report['appliedCorrection'] = None
    if not report['applied'] and not report['fallbacks']:
        fallback(report, 'registered_plan_unavailable')
    from steward.ego.fasten_plan import value
    view = value(controller.rt.ego_state().find(plan['object']), 'contactView')
    after = plan.get('releasedAt', math.inf)
    now = controller.ex._state()['stamp']
    if (not view or view.get('source') != 'rgbd_segment' or view.get('truncated', True)
            or view.get('invalidDepthFraction', 1) > .05
            or not after < view.get('stamp', -1) <= now or now-view['stamp'] > 20):
        report['orientationReason'] = 'no_fresh_separate_part'; return
    size = np.asarray(view.get('visibleSizeM', []))
    wanted = np.asarray(target.get('sizeM', []))
    if (size.shape != (3,) or wanted.shape != (3,) or min(size[:2]) <= 0
            or max(size[:2])/min(size[:2]) < 1.2
            or not np.allclose(np.sort(size[:2]), np.sort(wanted[:2]), atol=.015)):
        report['orientationReason'] = 'ambiguous_or_merged_outline'; return
    if 'boxYawRad' not in view or target.get('yawRad') is None:
        report['orientationReason'] = 'no_observed_axis'; return
    centre = np.asarray(view.get('centreM', []))
    if centre.shape != (3,) or np.linalg.norm(centre[:2]-plan['spot']) > .03:
        report['orientationReason'] = 'part_not_at_target'; return
    observed_axis = view['boxYawRad']+(math.pi/2 if size[1] > size[0] else 0)
    target_axis = target['yawRad']+(math.pi/2 if wanted[1] > wanted[0] else 0)
    error = math.degrees(abs((observed_axis-target_axis+math.pi/2) % math.pi-math.pi/2))
    report.update(axisErrorDeg=error, measuredYawErrorDeg=error, yawErrorPeriodDeg=180,
        orientationVerifiedSuccesses=int(error <= 5.),
        orientationReason='fresh_rgbd_outline_axis', orientationObservedAt=view['stamp'])
