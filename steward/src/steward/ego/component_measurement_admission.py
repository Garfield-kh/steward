"""Native component measurement admission, provenance and replay inputs."""
from copy import copy, deepcopy
import math
import numpy as np
from steward.ego import capabilities


def enabled():
    return capabilities.get('STEWARD_ASSEMBLY_COMPONENT_MEASUREMENT_ADMISSION') != '0'


def body_search(plan):
    from steward.ego.component_registration import rotation
    target = plan['placementTarget']; ref = target['mateReference']
    own = {h['id']: h for h in target.get('releasedFrame', {}).get('holes', [])}
    local = np.asarray([own[p['partHole']]['p'][:2] for p in target['holePairs']])
    expected = local @ rotation(target['yawRad']).T+plan['spot']
    height = ref.get('landingTopM', ref['position'][2]+ref['size'][2]/2)
    return local, expected, height


def visibility(est, hole, raw):
    """Current accepted contour/plane beats a stale held-box estimate only.

    Robot rays remain a veto. The evidence token is issued only after native
    rim completeness, depth coverage, plane quality and target-face checks.
    """
    from steward.ego import component_registration as cr
    from steward.ego.registered_alignment import arm_clear
    if cr.entry_clear(est, hole['point'], raw['camera']):
        return True, 'ray_clear'
    if not arm_clear(est, hole['point'], raw['camera']):
        return False, 'robot_ray_occluded'
    held = est.tracks.get(est.held_id)
    if held is None or est.held_pose().get(held.id) is None:
        return False, 'held_pose_unavailable'
    evidence = hole.get('nativeEvidence', {})
    current = (evidence.get('stamp') == raw['header']['stamp']
               and evidence.get('camera') == raw['camera']
               and evidence.get('contourPlaneAccepted') is True)
    if current:
        return True, 'current_plane_overrides_estimated_held_box'
    return False, 'estimated_held_box_without_current_plane'


def camera_ranking(controller, plan, available, *, layer=None):
    from steward.ego import inhand, support_holes as sh, component_registration as cr
    est = controller.rt.estimator; state = controller.ex._state()
    held = est.tracks[plan['object']]
    _, expected, height = body_search(plan)
    lower = np.c_[expected, np.full(len(expected), height)]
    centre, rotation, _, _ = inhand.load_pose(est, held, state)
    local = inhand.layout(held).copy(); local[:, 2] = held.size[2]/2
    upper = local @ rotation.T+centre
    rows = []
    for camera in available:
        sensor = est.cameras[camera][1]; intr = sensor['intrinsics']
        cam = inhand.camera_pose(est, state, camera)
        near, far = sensor.get('depthRangeM', [0., math.inf])
        fractions = []; depth_fractions = []; resolutions = []
        recent = [s for s in getattr(controller, '_releasedObjectHeldSamples', [])
                  if s[0].get('camera') == camera and
                  0 <= state['stamp']-s[0]['header']['stamp'] <= .25 and
                  s[0]['header']['stamp'] > est.held_since]
        for points in (lower, upper):
            uv, z = sh.project(points, cam, intr)
            inside = (np.isfinite(uv).all(axis=1) & (z > 0) & (uv >= 0).all(axis=1)
                      & (uv < [intr['width'], intr['height']]).all(axis=1))
            fractions.append(float(inside.mean()))
            valid = inside & (z >= near) & (z <= far)
            depth_fractions.append(float(valid.mean()))
            if recent:
                raw, raw_cam = recent[-1]
                ri = dict(width=raw['depth'].shape[1], height=raw['depth'].shape[0], fovy=raw['fovy'])
                puv, _ = sh.project(points, raw_cam, ri)
                values = []
                for pixel in puv:
                    if not np.isfinite(pixel).all(): values.append(0.); continue
                    x, y = np.rint(pixel).astype(int)
                    patch = raw['depth'][max(0,y-1):min(ri['height'],y+2), max(0,x-1):min(ri['width'],x+2)]
                    values.append(float((np.isfinite(patch) & (patch > 0)).mean()) if patch.size else 0.)
                depth_fractions[-1] *= min(values)
            ray = np.asarray(cam['p'])-points.mean(0)
            cosine = abs(ray[2])/max(np.linalg.norm(ray), np.finfo(float).eps)
            focal = intr['height']/(2*math.tan(math.radians(intr['fovy'])/2))
            resolutions.append(float(focal*cosine/max(np.linalg.norm(ray), np.finfo(float).eps)))
        if layer is not None:
            index = 1 if layer == 'upper' else 0
            points = upper if layer == 'upper' else lower
            proxy = copy(est); proxy.joints = state['joints']; proxy.odometry = state['odometry']
            visible = [inhand.ray_clear(est, camera, [p], state) if layer == 'upper' else
                       cr.entry_clear(proxy, p, camera) for p in points]
            # Score the existing detector search band, not just two predicted
            # centre pixels. A barely visible pair is a poorer first view.
            pad = sh.MAX_LAYOUT_SHIFT
            corners = [[x,y,z] for x in (points[:,0].min()-pad, points[:,0].max()+pad)
                       for y in (points[:,1].min()-pad, points[:,1].max()+pad)
                       for z in (points[:,2].min(), points[:,2].max())]
            search_uv, search_z = sh.project(corners, cam, intr)
            search_coverage = 0.
            if np.isfinite(search_uv).all() and np.all(search_z > 0):
                lo, hi = search_uv.min(0), search_uv.max(0)
                clipped = np.maximum(0., np.minimum(hi, [intr['width'],intr['height']])-np.maximum(lo, 0.))
                search_coverage = float(np.prod(clipped)/max(np.prod(hi-lo), np.finfo(float).eps))
            rows.append(dict(camera=camera, targetCoverage=fractions[index], validDepth=depth_fractions[index],
                searchCoverage=search_coverage,
                pixelsPerM=resolutions[index] if cam['p'][2] > points[:,2].mean() else 0.,
                predictedVisible=float(np.mean(visible)), layer=layer,
                depthBasis='recent_raw' if recent else 'declared_range_and_projection'))
            continue
        rows.append(dict(camera=camera, targetCoverage=min(fractions), validDepth=min(depth_fractions),
                         pixelsPerM=min(resolutions), depthBasis='recent_raw' if recent else 'declared_range_and_projection'))
    return sorted(rows, key=lambda r:(-r['targetCoverage'], -r['validDepth'], -r.get('predictedVisible', 0.), -r['pixelsPerM']*r.get('searchCoverage', 1.)))


def replay_context(controller, plan, camera, after):
    """All non-raster inputs to measure_sample; only robot-owned state/model."""
    est = controller.rt.estimator; held = est.tracks.get(plan['object'])
    return dict(schema='component_measure.component_measurement_admission.v1', camera=camera,
        after=after if math.isfinite(after) else None,
        plan={k:deepcopy(plan[k]) for k in ('object','spot','gripSpot','lift','wristYaw','placementXYCompensationClearLift','placementTarget') if k in plan},
        interfacePrecisionM=controller.ex.config.interface_precision_m,
        robotModel=deepcopy(getattr(est, 'robot_model', None)),
        capabilities={k:capabilities.get(k) for k in ('STEWARD_ASSEMBLY_COMPONENT_MEASUREMENT_ADMISSION','STEWARD_ASSEMBLY_PLACEMENT_XY_COMPENSATION','STEWARD_ASSEMBLY_STATIONARY_ENTRANCE_LAYERS')},
        held={k:deepcopy(getattr(held, k, None)) for k in
              ('id','size','var','held_offset','held_yaw','held_interface')},
        heldId=est.held_id, heldGripper=est.held_gripper, heldSince=est.held_since,
        stationaryRecords={body:{k:deepcopy(record[k]) for k in ('hardMotion','displacementEvidence') if k in record}
            | {'present': True} for body,record in est.stationary_assemblies.records.items()},
        motionEvents=deepcopy(list(est.stationary_assemblies.motion_events)))
