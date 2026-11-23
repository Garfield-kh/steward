"""Final lower-end height only: robot RGB-D, grasp-local geometry and actual FK.

No orientation, lateral motion, release dwell or withdrawal policy lives here.
Missing observations use the best estimate and never gate the ordinary open.
"""
from copy import deepcopy
import math
import time
import numpy as np
from steward.ego import capabilities
from steward.ego.fasten_geometry import transform_feature, surface_patch


def enabled():
    return capabilities.get('STEWARD_ASSEMBLY_RELEASE_PLANE') != '0'


def finite_point(value):
    return value is not None and np.shape(value) == (3,) and np.isfinite(value).all()


def positive(*values):
    return [float(v) for v in values if v is not None and np.isfinite(v) and v > 0]


def height_noise(sample, plane, tip):
    """Axial sampling uncertainty, not the lateral association/search envelope.

    A pixel's vertical footprint is a uniform quantization bin (RMS / sqrt(12)).
    Plane height uses the local fit's standard error. This is a precision
    estimate, not a guarantee against segmentation bias or grasp slip.
    """
    from steward.ego import wrist_observation as wo
    from steward.ego.support_holes import project, unproject
    raw, cam = sample
    radius = plane.get('radiusM')
    if not radius:
        return None
    region = wo.window(raw, cam, plane['point'], radius*2)
    points, _, _ = wo.cloud(raw, cam, region)
    normal = np.asarray(plane.get('normal', [0., 0., 1.]), float)
    if not len(points) or abs(normal[2]) <= np.finfo(float).eps:
        return None
    residual = (points-plane['point']) @ normal / normal[2]
    band = max(positive(plane.get('planeNoiseM'), plane.get('residualM')) or
               [float(np.std(residual))])
    residual = residual[abs(residual) <= 3*band]
    if not len(residual):
        return None
    # Estimate precision of the already fitted local surface, excluding outliers.
    plane_error = float(np.std(residual)/math.sqrt(len(residual)))
    h, w = raw['depth'].shape
    intr = dict(width=w, height=h, fovy=raw['fovy'])
    uv, depth = project([tip], cam, intr)
    rays = unproject(np.vstack((uv[0], uv[0]+[1., 0.], uv[0]+[0., 1.])),
                     np.full(3, depth[0]), cam, intr)
    pitch = float(np.linalg.norm(rays[1:, 2]-rays[0, 2]))
    rounding = abs(float(np.spacing(np.float32(depth[0]))))
    return dict(gapM=max(math.hypot(plane_error, pitch/math.sqrt(12)), rounding),
                planeStandardErrorM=plane_error, verticalPixelPitchM=pitch,
                planeSamples=int(len(residual)), source='axial_pixel_rms_and_local_plane_standard_error')


class Geometry:
    def __init__(self, controller, current):
        self.controller, self.current = controller, current
        self.ex, self.rt = controller.ex, controller.rt
        self.est = self.rt.estimator
        self.target = deepcopy(current.get('seatTarget') or
                               current.get('seatEndpoint', {}).get('targetEvidence') or {})
        self.plane = deepcopy(self.target)
        self.plane_source = 'best_estimated_upper_surface:' + self.target.get(
            'source', self.target.get('targetSource', 'admitted_target'))
        self.plane_direct = False
        self.local = None
        self.endpoint_verified = False
        self.endpoint_source = 'unavailable'
        self.noise = None
        track = self.est.tracks.get(current.get('object'))
        if self.est.held_id != current.get('object'):
            raise ValueError('held_identity_changed')
        candidates = [getattr(track, k, None) for k in ('held_tool_feature', 'held_grasp_geometry')]
        def verified(f):
            ev = (f or {}).get('offsetEvidence') or {}
            return ev.get('basis') == 'inhand_rgbd_plus_same_frame_fk' or (
                (f or {}).get('source') == 'rgbd_endpoint_width' and
                (f or {}).get('stamp', -math.inf) > self.est.held_since)
        self.local = deepcopy(next((f for f in candidates if verified(f)),
                                   next((f for f in candidates if f), None)))
        self.endpoint_verified = verified(self.local)
        if self.local:
            self.endpoint_source = ('verified_inhand_geometry_plus_actual_fk' if self.endpoint_verified
                                    else 'best_grasp_estimate_plus_actual_fk')
        # A recent direct rim is a better height reference than max(old, new).
        history = (current.get('tipEntryServo', {}).get('frames', []) +
                   current.get('insertionSearch', {}).get('observations', []))
        for frame in sorted(history, key=lambda f:f.get('stamp') or -math.inf, reverse=True):
            entry = frame.get('entry') or {}
            if (finite_point(entry.get('point')) and not entry.get('targetFromLayout')
                    and not frame.get('targetFromLayout') and
                    self.est.held_since < frame.get('stamp', -math.inf) <= self.ex._state()['stamp']):
                self.plane = dict(deepcopy(entry), observedAt=frame['stamp'])
                self.plane_source = 'recent_observed_entry:' + entry.get('source', 'rgbd')
                self.plane_direct = True
                break
        try:
            self.upper_surface()
        except (AttributeError, KeyError, TypeError, ValueError):
            # A missing component view must not discard usable entry/grasp data.
            pass

    def upper_surface(self):
        """Only the identified upper component; never an unlabelled surrounding ROI."""
        from steward.ego.fasten_plan import value
        part = self.current.get('seatPart') or self.target.get('entity')
        if not part or not finite_point(self.target.get('point')):
            return
        scene = self.rt.ego_state()
        view = value(scene.find(part), 'contactView')
        if (not view or view.get('source') != 'rgbd_segment' or view.get('truncated', True)
                or view.get('stamp', -math.inf) > self.ex._state()['stamp']):
            return
        stamp = view['stamp']
        plane_stamp = self.plane.get('observedAt', self.plane.get('stamp', -math.inf))
        if self.plane_direct and stamp <= (plane_stamp or -math.inf):
            return
        patch = surface_patch(view.get('points', []), self.target['point'][:2])
        if patch:
            self.plane = dict(patch, observedAt=stamp, radiusM=self.target.get('radiusM'))
            self.plane_source = 'observed_upper_component_surface'
            self.plane_direct = False
            self.noise = dict(gapM=max(patch['residualM']/math.sqrt(patch['points']),
                                      abs(float(np.spacing(np.float32(patch['point'][2]))))),
                              source='upper_component_plane_standard_error')

    def adopt(self, sample, measurement):
        header = sample[0]['header']
        if not self.est.held_since < header['stamp'] <= self.ex._state()['stamp']:
            return
        entry = measurement.get('entry') or {}
        if (finite_point(entry.get('point')) and not entry.get('targetFromLayout')
                and not measurement.get('targetFromLayout')):
            self.plane = dict(deepcopy(entry), observedAt=header['stamp'])
            self.plane_source = 'same_frame_entry:' + entry.get('source', 'rgbd')
            self.plane_direct = True
        feature = measurement.get('lowerEnd')
        if feature and finite_point(feature.get('tip')) and finite_point(feature.get('head')):
            rotation = np.asarray(self.est.gripper_rotation(self.est.held_gripper,
                header['joints'], header['odometry']))
            grip = np.asarray(self.est.gripper_position(self.est.held_gripper,
                header['joints'], header['odometry']))
            # Localize at the IMAGE encoders, then transport at the reached pose.
            # No held-track cache or semantic polarity is overwritten.
            self.local = transform_feature(feature, rotation.T, -rotation.T @ grip)
            self.endpoint_verified = True
            self.endpoint_source = 'same_frame_lower_end_plus_actual_fk'
        g = self.read()
        if g and self.plane_direct and self.plane.get('observedAt') == header['stamp']:
            self.noise = height_noise(sample, self.plane, g['tip']) or self.noise

    def read(self):
        if not self.local or not finite_point(self.plane.get('point')):
            return None
        state = self.ex._state()
        f = transform_feature(self.local,
            self.est.gripper_rotation(self.est.held_gripper, state['joints'], state['odometry']),
            self.est.gripper_position(self.est.held_gripper, state['joints'], state['odometry']))
        # The physical lower end, independent of head/tip polarity; no wrist correction.
        tip = min((f['tip'], f['head']), key=lambda p:p[2])
        normal = np.asarray(self.plane.get('normal', [0., 0., 1.]), float)
        if not finite_point(normal) or abs(normal[2]) <= np.finfo(float).eps:
            return None
        entry = float(self.plane['point'][2] - np.dot(
            np.asarray(tip[:2])-self.plane['point'][:2], normal[:2])/normal[2])
        if not np.isfinite(entry) or not finite_point(tip):
            return None
        # If only old observations remain, their plane noise is the best precision
        # estimate; absent noise stays at sensor numeric resolution, unconfirmed.
        fallback = max(positive(self.plane.get('planeNoiseM'), self.plane.get('residualM')) or
                       [abs(float(np.spacing(np.float32(entry))))])
        gap = max((self.noise or {}).get('gapM', fallback), np.finfo(float).eps)
        return dict(tip=list(tip), entryM=entry, tipM=float(tip[2]), gapM=float(tip[2]-entry),
                    planePoint=deepcopy(self.plane['point']), normal=normal.tolist(),
                    clearanceM=gap, lift=float(state['joints']['lift']), observedAt=state['stamp'],
                    planeObservedAt=self.plane.get('observedAt', self.plane.get('stamp')),
                    planeSource=self.plane_source, endpointSource=self.endpoint_source,
                    endpointVerified=self.endpoint_verified, planeDirect=self.plane_direct,
                    noise=deepcopy(self.noise or dict(source='best_available_plane_precision', gapM=gap)))


def observe(controller, current, geometry, row):
    """One current frame and at most one shared alternate camera, with no motion."""
    from steward.ego import wrist_observation as wo, tip_entry_servo as servo
    from steward.ego.entry_layout import prior
    camera = servo.camera_for(controller.rt)
    if camera is None:
        return
    cameras = [camera]
    used = bool(current.get('insertionSearch', {}).get('viewChanges') or
                current.get('tipEntryServo', {}).get('nearContactViewChanges'))
    if not used:
        alternate = next((c for c in getattr(controller.rt, '_provided_camera_ids', ())
            if c != camera and c in controller.rt.estimator.cameras), None)
        if alternate: cameras.append(alternate)
    opening = controller.rt.esim.arm.geometry.jaw_opening
    for index, camera in enumerate(cameras):
        if index: row['viewChanges'] = 1
        wo.context(controller, 'release_height', geometry.target['point'], opening/2)
        sample = wo.capture(controller, time.monotonic()+controller.ex.config.settle_s*4,
                            camera=camera, check_view=False)
        if sample is None: continue
        measurement = servo.observe(controller, sample, geometry.target['point'], opening,
                                    registration=prior(controller, current))
        geometry.adopt(sample, measurement)
        if measurement.get('bothVisible') and not measurement.get('targetFromLayout'):
            break


def correct(controller, geometry, row):
    """Finite feedback on REACHED poses, including timed-out partial moves."""
    from steward.ego.fasten_seat import lift_to
    from steward.ego.skills import SkillFailed
    g = geometry.read()
    if g is None:
        row['reason'] = 'no_endpoint_best_estimate_ordinary_release'
        return
    row['before'] = deepcopy(g)
    for _ in range(3):
        floor = float(np.nextafter(g['lift']+g['entryM']-g['tipM'], math.inf))
        target = floor+g['clearanceM']
        row.update(commandedTipM=g['entryM']+g['clearanceM'],
                   commandedTipRelativeToEntryM=g['clearanceM'])
        tolerance = max(g['clearanceM']/4, np.finfo(float).eps*max(1., abs(target)))
        if g['gapM'] >= 0 and abs(g['gapM']-g['clearanceM']) <= tolerance:
            break
        state = controller.ex._state()
        motion = dict(kind='height', targetLift=target, commandedTipM=row['commandedTipM'],
                      commandedTipRelativeToEntryM=g['clearanceM'], completed=False,
                      executed=False, commandAttempted=False)
        row['motions'].append(motion)
        if not controller.rt.esim.arm.reachable_path(
                [dict(state['joints']), dict(state['joints'], lift=target)], start_joints=state['joints']):
            row['reason'] = 'height_path_unreachable_best_estimate_release'
            break
        try:
            motion['commandAttempted'] = True
            lift_to(controller.ex, target, ramp=False, tolerance=tolerance, floor=floor)
            motion.update(completed=True, executed=True)
        except SkillFailed as exc:
            controller.check()
            motion['reason'] = str(exc)[:120]
        # Never replace encoder/FK state by target, even when the call returned.
        reached = geometry.read()
        if reached is None:
            row['reason'] = 'actual_endpoint_unavailable'
            g = None
            break
        g = reached
        motion.update(actualLift=g['lift'], verifiedTipM=g['tipM'],
                      verifiedTipRelativeToEntryM=g['gapM'],
                      actuatorSetpointLift=getattr(controller.ex, '_setpoints', {}).get('lift'))
    if g is not None:
        # One final read also covers the already-at-height path.
        g = geometry.read()
        row.update(after=deepcopy(g), verifiedTipM=g['tipM'],
                   verifiedTipRelativeToEntryM=g['gapM'], tipRelativeToEntryM=g['gapM'],
                   entryM=g['entryM'], planeSource=g['planeSource'],
                   endpointSource=g['endpointSource'], observedAt=g['observedAt'],
                   planeObservedAt=g['planeObservedAt'], clearanceM=g['clearanceM'])
        row['heightReached'] = bool(g['gapM'] >= 0 and abs(g['gapM']-g['clearanceM']) <= tolerance)
        # FK checks arrival, not independent visual contact/insertion.
        row['unconfirmed'] = True


def finish(controller, current, report):
    if not enabled(): return
    row = dict(policy='observed_plane_height_only', status='estimated', unconfirmed=True,
               viewChanges=0, motions=[], commandedTipM=None, verifiedTipM=None,
               commandedTipRelativeToEntryM=None, verifiedTipRelativeToEntryM=None,
               planeSource='best_estimated_upper_surface', heightReached=False)
    current['releaseHeight'] = row
    try:
        geometry = Geometry(controller, current)
        try:
            observe(controller, current, geometry, row)
        except (AttributeError, KeyError, TypeError, ValueError, TimeoutError, OSError) as exc:
            row['observationReason'] = type(exc).__name__
        correct(controller, geometry, row)
    except (AttributeError, KeyError, TypeError, ValueError, TimeoutError, OSError) as exc:
        row.update(status='unknown', reason='best_estimate_release:'+type(exc).__name__)
    current['nearContactUnconfirmed'] = True
    report['releaseHeight'] = deepcopy(row)
    from steward.ego.capability_usage import record
    record('release_height', entity=current.get('object'), stage='before_open',
           action=bool(row['motions']), measured=False, reason=row.get('reason'), evidence=row)


def summary(row):
    # Full timestamps, endpoint provenance and actuator setpoints stay in the
    # keyed event; export the requested/actual heights and plane once per release.
    keep = ('commandedTipM', 'verifiedTipM', 'entryM',
            'commandedTipRelativeToEntryM', 'verifiedTipRelativeToEntryM',
            'planeSource', 'heightReached', 'unconfirmed')
    return {k:round(v, 7) if isinstance(v, float) else deepcopy(v)
            for k, v in row.items() if k in keep}
