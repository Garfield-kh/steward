"""Bounded held-object insertion search; unknown always finishes ordinary release.

Dimensions come from robot observations and its jaw envelope. Native RGB-D is
used because the current held-object actuator does not transmit contact forces.
The one insertion_search switch wraps all changed execution behaviour.
"""
from copy import deepcopy
import math
import time
import uuid
import numpy as np
from steward.ego import capabilities, action_budget as ab


def enabled():
    return capabilities.get('STEWARD_ASSEMBLY_INSERTION_SEARCH') != '0'


def positive(*values):
    return [float(v) for v in values if v is not None and np.isfinite(v) and v > 0]


def limits(opening, target, feature, uncertainty, visible, config):
    """Scale a search from the aperture, carried shape and uncertainty envelope."""
    aperture = positive(target.get('radiusM'))
    measured_aperture = bool(aperture)
    shaft = positive(feature.get('shaftRadiusM'), feature.get('radiusM'))
    length = positive(feature.get('lengthM'))
    aperture = aperture[0] if aperture else (2*shaft[0] if shaft else opening/12)
    shaft = shaft[0] if shaft else aperture/2
    length = length[0] if length else opening/2
    noise = max(positive(target.get('planeNoiseM'), feature.get('uncertaintyM')) or [aperture/8])
    radius = min(opening/2, max(2*aperture, uncertainty or 0.) if visible else opening/2)
    # Half-diameter raster, finer when the available clearance can be resolved.
    step = min(aperture, max(aperture-shaft, noise))
    depth = min(aperture, length/4)
    hover = max(2*aperture, 6*noise)
    dwell = max(config.tick_s*2, config.settle_s)
    return dict(radiusM=radius, stepM=step, apertureRadiusM=aperture,
                shaftRadiusM=shaft, depthM=depth, hoverM=hover, noiseM=noise,
                dwellS=dwell, maxPoints=257, budgetSimS=120., budgetWallS=120.,
                scaleSource='observed_aperture_and_held_shape' if measured_aperture and visible and
                    feature.get('tip') is not None and feature.get('head') is not None else
                    'best_shape_estimate_and_robot_jaw_envelope')


def spiral(radius, step):
    """Concentric centre-out samples; arc and radial spacing never exceed step."""
    yield np.zeros(2)
    rings = max(1, math.ceil(radius/step))
    for ring in range(1, rings+1):
        r = radius*ring/rings
        count = max(6, math.ceil(2*math.pi*r/step))
        for i in range(count):
            angle = 2*math.pi*i/count
            yield r*np.array([math.cos(angle), math.sin(angle)])


def search(io, spec):
    """Execute pecks, keeping the jaws closed. IO never opens the gripper.

    Every lateral move is preceded by unloading vertically. Successful visual
    detection leaves the load at depth; all other outcomes restore the best
    estimate's ordinary release pose. The caller then opens and postchecks.
    """
    row = dict(schema='insertion_search.v1', id=uuid.uuid4().hex, executed=False,
               detected=False, status='unconfirmed', observations=[], probes=[],
               pressureReductions=0, viewChanges=0, proprioDetectionAvailable=False,
               detectionMode='native_rgbd', limits=deepcopy(spec), reason='search_exhausted')
    centre = np.asarray(io.centre, float).copy()
    best = centre.copy()
    original_lift = io.release_lift
    release_gap = original_lift-io.surface_lift
    depth = spec['depthM']
    rings = max(1, math.ceil(spec['radiusM']/spec['stepM']))
    row['plannedPointCount'] = 1+sum(max(6, math.ceil(2*math.pi*(spec['radiusM']*r/rings)/spec['stepM']))
                                    for r in range(1, rings+1))
    start = io.stamp(); wall = time.monotonic()
    row['startedAt'] = start
    deadline = wall+spec['budgetWallS']
    def expired():
        return io.stamp()-start >= spec['budgetSimS'] or time.monotonic() >= deadline
    def bounded(xy):
        return np.linalg.norm(np.asarray(xy)-centre) <= spec['radiusM']+np.finfo(float).eps*16
    try:
        for index, offset in enumerate(spiral(spec['radiusM'], spec['stepM'])):
            if index >= spec['maxPoints'] or expired():
                row['reason'] = 'search_budget_exhausted'
                break
            io.check()
            candidate = centre+offset
            hover = io.surface_lift+spec['hoverM']
            if not io.lift(hover):
                row['reason'] = 'clearance_motion_incomplete'
                break
            if not io.move(candidate, spec['stepM']/4):
                row['reason'] = 'lateral_motion_incomplete'
                continue
            row['executed'] = True
            if np.linalg.norm(io.xy()-centre) > spec['radiusM']+spec['noiseM']:
                row['reason'] = 'search_envelope_tracking_error'
                continue
            before = io.observe('search')
            row['observations'].append(io.compact(before))
            if not before.get('entryVisible') and not row['viewChanges'] and not io.view_already_changed:
                row['viewChanges'] += 1
                before = io.change_view()
                row['observations'].append(io.compact(before))
            shift = io.support_shift(before)
            if shift is not None:
                depth /= 2
                row['pressureReductions'] += 1
                row.setdefault('supportMotion', []).append(shift)
            probe_reference = before
            correction = before.get('correctionM')
            if correction is not None:
                revised = io.xy()+np.asarray(correction)
                if bounded(revised):
                    best = revised
                    if np.linalg.norm(revised-io.xy()) > spec['stepM']/4:
                        if not io.move(revised, spec['stepM']/4):
                            continue
                        aligned = io.observe('aligned')
                        row['observations'].append(io.compact(aligned))
                        # The shaft can now occlude the very aperture just seen.
                        # Keep that immediate pre-alignment image as the empty
                        # reference; it is never itself an insertion verdict.
                        if aligned.get('entryVisible'):
                            before = aligned
                            if aligned.get('measurement', {}).get('lowerEnd'):
                                probe_reference = aligned
            # A fresh local plane and held lower end can update contact height,
            # but missing pixels keep the same bounded best estimate.
            surface = before.get('surfaceLift', probe_reference.get('surfaceLift'))
            if surface is not None and abs(surface-io.surface_lift) <= spec['hoverM']+spec['depthM']:
                io.surface_lift = surface
            if expired():
                row['reason'] = 'search_budget_exhausted'
                break
            # Never infer falling from commanded/virtual joint motion.
            probe = dict(index=index, gripXYM=io.xy().tolist(), depthLimitM=depth,
                         liftBefore=io.lift_position(), inserted=False)
            row['probes'].append(probe)
            probe['executed'] = io.lift(io.surface_lift-depth)
            io.dwell(spec['dwellS'])
            probe['liftAfter'] = io.lift_position()
            after = io.observe('probe')
            row['observations'].append(io.compact(after))
            shifted = io.support_shift(after)
            if shifted is not None:
                depth /= 2
                row['pressureReductions'] += 1
                row.setdefault('supportMotion', []).append(shifted)
                io.lift(io.surface_lift+spec['hoverM'])
                continue
            evidence = io.detect(probe_reference, after, depth)
            probe['evidence'] = evidence
            if evidence.get('detected'):
                row.update(detected=True, status='visually_inserted', reason=evidence['reason'])
                probe['inserted'] = True
                break
    except (TimeoutError, OSError, KeyError, ValueError, TypeError, AttributeError) as exc:
        row['reason'] = 'search_observation_unavailable:'+type(exc).__name__
    finally:
        # Even acquisition/motion errors must finish at a release pose. Only
        # cancellation/actual hardware faults escape to the executor's safety.
        if not row['detected']:
            restored = lowered = False
            try:
                raised = io.lift(io.surface_lift+spec['hoverM'])
                restored = raised and io.move(best, spec['stepM']/4)
                release_lift = max(original_lift, io.surface_lift+release_gap)
                lowered = io.lift(release_lift) if restored else False
            except (TimeoutError, OSError, KeyError, ValueError, TypeError, AttributeError) as exc:
                row['restoreReason'] = 'restore_state_unavailable:'+type(exc).__name__
            row['fallback'] = dict(policy='best_estimate_then_ordinary_release',
                                   targetXYM=best.tolist(), restored=bool(restored and lowered),
                                   releaseAtReachedPose=not (restored and lowered))
        try:
            row['finishedAt'] = io.stamp()
        except (KeyError, ValueError, TypeError, AttributeError):
            row['finishedAt'] = None
        row['elapsedSimS'] = (max(0., row['finishedAt']-start) if row['finishedAt'] is not None else None)
        row['elapsedWallS'] = time.monotonic()-wall
        row['visitedRadiusM'] = max((float(np.linalg.norm(np.asarray(p['gripXYM'])-centre))
                                    for p in row['probes']), default=0.)
        row['coverageComplete'] = (len(row['probes']) == row['plannedPointCount'] and
                                   all(p['executed'] for p in row['probes']))
        row['releaseExecuted'] = False
        io.result(row)
    return row


class RobotIO:
    """Adapter over the existing robot actuator, calibrated camera and beliefs."""
    def __init__(self, controller, current, spec):
        self.controller, self.current, self.spec = controller, current, spec
        self.ex, self.rt = controller.ex, controller.rt
        self.path = current.get('seatEndpoint', current)
        self.centre = np.asarray(self.path.get('gripSpot', self.path.get('spot')), float)
        self.release_spot = deepcopy(current.get('spot'))
        self.release_lift = float(self.ex._state()['joints']['lift'])
        # Missing proprioceptive geometry means ordinary release, not a partly
        # executed search whose cleanup would need that same unavailable state.
        if not np.isfinite(self.xy()).all():
            raise ValueError('gripper_position_unavailable')
        from steward.ego.release_height import geometry
        try:
            g = geometry(controller, current)
        except (AttributeError, KeyError, TypeError, ValueError):
            g = None
        self.surface_lift = (g['lift']+g['entryM']-g['tipM'] if g else
                             float(self.path['lift']))
        self.target = deepcopy(current.get('seatTarget') or self.path.get('targetEvidence') or {})
        self.tip_offset_xy = np.asarray(self.target['point'], float)[:2]-self.centre
        from steward.ego.tip_entry_servo import camera_for
        self.camera = camera_for(self.rt)
        self.last_stamp = -math.inf
        self.previous_support = None
        self.previous_plane = None
        self.jaw_floor_lift = None
        self.jaw_clamps = 0
        self.view_already_changed = bool(current.get('tipEntryServo', {}).get('nearContactViewChanges'))
        self._deadline = time.monotonic()+spec['budgetWallS']

    def stamp(self): return float(self.ex._state()['stamp'])
    def xy(self): return np.asarray(self.ex._gripper_xy(), float)
    def lift_position(self): return float(self.ex._state()['joints']['lift'])
    def check(self): self.controller.check()
    def dwell(self, seconds):
        start = self.stamp(); wall = time.monotonic()
        for _ in range(max(1, math.ceil(seconds/self.ex.config.tick_s)*4)):
            if self.stamp()-start >= seconds or time.monotonic()-wall >= seconds*4:
                break
            self.ex._tick()

    def lift(self, target):
        from steward.ego.skills import SkillFailed
        state = self.ex._state()
        if self.jaw_floor_lift is not None and target < self.lift_position() and target < self.jaw_floor_lift:
            target = self.jaw_floor_lift
            self.jaw_clamps += 1
        # No lift_to offset compensation: a contact miss never increases pressure.
        path = [dict(state['joints']), dict(state['joints'], lift=float(target))]
        if not self.rt.esim.arm.reachable_path(path, start_joints=state['joints']):
            return False
        started = self.stamp(); wall = time.monotonic()
        timeout = max(self.spec['dwellS']*2, self.ex.config.tick_s*4)
        try:
            self.ex._command({'lift':float(target)})
            if hasattr(self.ex, '_setpoints'): self.ex._setpoints['lift'] = float(target)
            for _ in range(max(1, math.ceil(timeout/self.ex.config.tick_s)*4)):
                if self.stamp()-started > timeout or time.monotonic()-wall > timeout*4:
                    break
                self.ex._tick()
                if abs(self.lift_position()-target) <= self.spec['noiseM']/2:
                    return True
        except SkillFailed:
            self.check()
        # An upward clearance may settle short under load. Never compensate
        # downward contact commands, and never sweep below clearance.
        return bool(target > self.surface_lift and
                    self.lift_position() > self.surface_lift+self.spec['noiseM'] and
                    abs(self.lift_position()-target) <= self.spec['hoverM']/2)

    def move(self, target, tolerance):
        from steward.ego.skills import SkillFailed
        state = self.ex._state(); odom = state['odometry']
        yaw = odom['yaw']; forward = np.array([math.cos(yaw), math.sin(yaw)])
        side = np.array([-forward[1], forward[0]])
        delta = np.asarray(target)-self.xy()
        reach = float(state['joints']['reach']+delta @ forward)
        lateral = float(delta @ side)
        base = np.array([odom['x'], odom['y']])+lateral*side
        stages = [dict(state['joints'], reach=float(q)) for q in
                  np.linspace(state['joints']['reach'], reach, 5)]
        if not self.rt.esim.arm.reachable_path(stages, start_joints=state['joints']):
            return False
        started = self.stamp(); wall = time.monotonic()
        timeout = max(self.spec['dwellS']*2, self.ex.config.tick_s*4)
        try:
            # Both Jacobian columns are translations: no yaw change, no loaded
            # lateral drag. Poll with a local deadline instead of a navigation
            # primitive which may consume the entire enclosing seat budget.
            self.ex._command({'base':[float(base[0]), float(base[1]), yaw], 'reach':reach})
            if hasattr(self.ex, '_setpoints'): self.ex._setpoints['reach'] = reach
            for _ in range(max(1, math.ceil(timeout/self.ex.config.tick_s)*4)):
                if self.stamp()-started > timeout or time.monotonic()-wall > timeout*4:
                    break
                self.ex._tick()
                if (self.stamp()-started >= self.ex.config.tick_s*2 and
                        np.linalg.norm(self.xy()-target) <= max(tolerance, self.spec['noiseM'])):
                    return True
            return False
        except SkillFailed:
            self.check()
            return False

    def observe(self, phase):
        from steward.ego import wrist_observation as wo, tip_entry_servo as servo
        from steward.ego.entry_layout import prior
        result = dict(entryVisible=False, reason='frame_unavailable', phase=phase)
        if self.camera is None or time.monotonic() >= self._deadline:
            return result
        # Move the local aperture window with the probe. Keeping it at the
        # original visual centre would never inspect the outer search rings.
        focus = np.r_[self.xy()+self.tip_offset_xy, self.target['point'][2]]
        wo.context(self.controller, 'insertion_'+phase, focus.tolist(), self.spec['radiusM'])
        sample = wo.capture(self.controller, self._deadline, camera=self.camera, check_view=False)
        if sample is None:
            return result
        stamp = sample[0]['header']['stamp']
        if stamp <= self.last_stamp:
            return dict(result, reason='frame_not_new')
        self.last_stamp = stamp
        try:
            opening = self.rt.esim.arm.geometry.jaw_opening
            measured = servo.observe(self.controller, sample, focus.tolist(), opening,
                                     registration=prior(self.controller, self.current))
            entry = measured.get('entry') or {}
            result.update(sample=sample, stamp=stamp, measurement=measured,
                          reason=measured.get('reason'), entryVisible=bool(entry and not entry.get('targetFromLayout')))
            if result['entryVisible']:
                result['entry'] = entry
                self.target = dict(self.target, **entry)
                if measured.get('bothVisible'):
                    self.tip_offset_xy = np.asarray(measured['lowerEnd']['tip'])[:2]-self.xy()
                    result.update(correctionM=measured['deltaM'][:2],
                                  surfaceLift=self.lift_position()+measured['deltaM'][2])
                else:
                    # The load offset is still the admitted estimate. This only
                    # updates the search centre, never the insertion verdict.
                    result['correctionM'] = (np.asarray(entry['point'])[:2]-focus[:2]).tolist()
                self.current['seatTarget'] = deepcopy(self.target)
            from steward.ego.insertion_vision import surface
            plane_prior = dict(self.target, radiusM=self.target.get('radiusM') or self.spec['apertureRadiusM'])
            result['supportPlane'] = surface(sample, plane_prior)
            self.jaw_clearance(sample, plane_prior)
            return result
        except (KeyError, ValueError, TypeError, AttributeError) as exc:
            return dict(result, reason='measurement_unavailable:'+type(exc).__name__)

    def change_view(self):
        alternate = next((c for c in getattr(self.rt, '_provided_camera_ids', ())
                          if c != self.camera and c in self.rt.estimator.cameras
                          and self.rt.estimator.cameras[c][1].get('intrinsics')), None)
        if alternate is not None:
            self.camera = alternate
            self.last_stamp = -math.inf
        return self.observe('alternate') if alternate else dict(entryVisible=False, reason='no_alternate_view')

    def jaw_clearance(self, sample, entry):
        """Bound descent by calibrated robot geometry above the observed face."""
        from steward.ego.wrist_release_geometry import robot_boxes
        try:
            boxes = robot_boxes(self.controller, sample[0])
            point = np.asarray(entry['point']); normal = np.asarray(entry.get('normal', [0,0,1]))
            clearances = []
            for centre, half, rotation in boxes:
                extent = abs(rotation) @ half
                planar_gap = np.maximum(abs(np.asarray(centre)[:2]-self.xy())-extent[:2], 0.)
                if np.linalg.norm(planar_gap) <= self.spec['radiusM']:
                    clearances.append(float((centre-point) @ normal - abs(normal @ rotation) @ half))
            if clearances:
                self.jaw_floor_lift = self.lift_position()-(min(clearances)-self.spec['noiseM'])/normal[2]
        except (AttributeError, KeyError, TypeError, ValueError):
            # Missing optional geometry does not create a new release gate.
            pass

    def support_shift(self, observation):
        entry = observation.get('entry')
        plane = observation.get('supportPlane')
        deltas = []
        odom = self.ex._state().get('odometry', {}).get('sigma', {}).get('pos', 0.)
        bound = 3*max(self.spec['noiseM'], math.sqrt(2)*odom)
        if entry:
            previous = self.previous_support
            self.previous_support = deepcopy(entry)
            if previous:
                bound = max(bound, 3*float(entry.get('uncertaintyM', 0.)),
                            3*float(previous.get('uncertaintyM', 0.)))
                deltas.append(np.asarray(entry['point'])-np.asarray(previous['point']))
        if plane:
            previous = self.previous_plane
            self.previous_plane = deepcopy(plane)
            if previous:
                bound = max(bound, 3*plane['uncertaintyM'], 3*previous['uncertaintyM'])
                deltas.append(np.array([0., 0., plane['point'][2]-previous['point'][2]]))
                tilt = np.linalg.norm(np.asarray(plane['normal'])-previous['normal'])*self.spec['radiusM']
                if tilt > bound:
                    deltas.append(np.array([tilt, 0., 0.]))
        if not deltas:
            return None
        delta = max(deltas, key=np.linalg.norm)
        if np.linalg.norm(delta) <= bound:
            return None
        self.surface_lift += float(delta[2])
        return dict(displacementM=delta.tolist(), uncertaintyBoundM=bound,
                    source='successive_visible_surface_motion', observedAt=observation.get('stamp'))

    def detect(self, before, after, depth):
        from steward.ego.insertion_vision import insertion
        entry = before.get('entry')
        if not entry or entry.get('radiusM', 0.) <= 0:
            return dict(detected=False, reason='no_observed_mouth_before_probe')
        # Same-frame lower end is required for shaft width; nominal dimensions
        # cannot turn recess walls into a claimed inserted shaft.
        feature = before.get('measurement', {}).get('lowerEnd') or {}
        shaft = feature.get('shaftRadiusM')
        if not shaft:
            return dict(detected=False, reason='shaft_width_unobserved')
        return insertion(before.get('sample'), after.get('sample'), entry, shaft, depth,
                         current_entry=after.get('entry'))

    @staticmethod
    def compact(observation):
        return {k:deepcopy(v) for k, v in observation.items() if k not in ('sample', 'measurement')}

    def result(self, row):
        row['jawClearanceClamps'] = self.jaw_clamps
        self.current['insertionSearch'] = row
        self.current['nearContactUnconfirmed'] = not row['detected']
        if not row['detected'] and self.current.get('tipEntryServo'):
            self.current['tipEntryServo']['nearContactUnconfirmed'] = True
        self.current.setdefault('seatMotion', {}).update(
            mode='bounded_visual_insertion_search', contactObserved=False,
            insertionObserved=row['detected'], pressureControl='bounded_position_no_contact_feedback')
        try:
            actual = self.xy()
            self.path['gripSpot'] = actual.tolist()
            self.path['lift'] = self.lift_position()
            if self.release_spot is not None:
                self.current['spot'] = (np.asarray(self.release_spot)+actual-self.centre).tolist()
        except (KeyError, ValueError, TypeError, AttributeError):
            row['releasePoseUnconfirmed'] = True
        from steward.ego.capability_usage import record
        record('insertion_search', event_id=row['id'], entity=self.current.get('object'),
               stage='finished', action=row['executed'], measured=row['detected'],
               reason=row['reason'], evidence=row)


def run(controller, current, report):
    if not enabled() or current.get('insertionSearch'):
        return current.get('insertionSearch')
    path = current.get('seatEndpoint', current)
    target = current.get('seatTarget') or path.get('targetEvidence') or {}
    if 'point' not in target:
        current['nearContactUnconfirmed'] = True
        current['insertionSearch'] = dict(executed=False, detected=False, reason='target_geometry_unavailable',
                                       status='unconfirmed', fallback='ordinary_release')
        report['insertionSearch'] = deepcopy(current['insertionSearch'])
        return report['insertionSearch']
    try:
        opening = controller.rt.esim.arm.geometry.jaw_opening
        feature = dict(lengthM=path.get('toolLengthM'), radiusM=path.get('toolRadiusM'))
        track = getattr(controller.rt.estimator, 'tracks', {}).get(current.get('object'))
        held = getattr(track, 'held_tool_feature', None) or getattr(track, 'held_grasp_geometry', None) or {}
        feature.update({k:held[k] for k in ('lengthM', 'radiusM', 'uncertaintyM') if k in held})
        frames = current.get('tipEntryServo', {}).get('frames') or []
        frame = next((f for f in reversed(frames) if f.get('bothVisible')), {})
        feature.update(frame.get('lowerEnd') or {})
        if feature.get('tip') is not None and feature.get('head') is not None:
            feature['lengthM'] = float(np.linalg.norm(np.asarray(feature['head'])-feature['tip']))
        uncertainty = max(positive(target.get('uncertaintyM'), frame.get('uncertaintyM'),
                                   report.get('alignmentUncertaintyM')) or [opening/2])
        spec = limits(opening, target, feature, uncertainty,
                      bool(frame and not frame.get('targetFromLayout')), controller.ex.config)
        # Reserve cleanup before entering a bounded search. Reading the public
        # session context does not change session admission or interface order.
        from steward.ego.assembly_session import current as active_session
        session = active_session()
        if session is not None:
            remaining = max(0., session.deadline.seconds-session.deadline.sim_s)
            spec['cleanupReserveS'] = 12*spec['dwellS']
            spec['budgetSimS'] = min(spec['budgetSimS'], max(0., remaining/3-spec['cleanupReserveS']))
        io = RobotIO(controller, current, spec)
        if io.camera is None:
            # No native imager and no contact transducer: one bounded best-
            # estimate peck, then release. Blind repeated pecks add no evidence.
            spec['maxPoints'] = 1
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        row = dict(executed=False, detected=False, status='unconfirmed',
                   reason='search_geometry_unavailable:'+type(exc).__name__, fallback='ordinary_release')
        current.update(insertionSearch=row, nearContactUnconfirmed=True)
        report['insertionSearch'] = row
        return row
    row = search(io, spec)
    report['insertionSearch'] = deepcopy(row)
    report['insertionObserved'] = True if row['detected'] else None
    report['contactEvidence'] = 'visual_shaft_crosses_mouth' if row['detected'] else 'search_unconfirmed'
    return row


def summary(row):
    """Small, idempotent public receipt; per-point evidence stays in the log."""
    if not row.get('executed'):
        # No probe occurred; the enclosing placement already records release.
        # Keep its reason without duplicating empty search counters and flags.
        return {k:deepcopy(row[k]) for k in ('id', 'executed', 'reason') if k in row}
    keys = ('schema', 'id', 'executed', 'detected', 'status', 'reason', 'detectionMode',
            'proprioDetectionAvailable', 'viewChanges', 'pressureReductions', 'jawClearanceClamps',
            'releaseExecuted', 'releasedAt', 'radiusM', 'stepM', 'depthLimitM',
            'elapsedSimS', 'fallbackRestored', 'releaseAtReachedPose', 'coverageComplete', 'visitedRadiusM')
    result = {k:deepcopy(row[k]) for k in keys if k in row}
    for key in ('releasedAt', 'elapsedSimS'):
        if isinstance(result.get(key), float): result[key] = round(result[key], 3)
    if row.get('limits'):
        result.update(radiusM=row['limits']['radiusM'], stepM=row['limits']['stepM'],
                      depthLimitM=row['limits']['depthM'])
    if isinstance(row.get('fallback'), dict):
        result.update(fallbackRestored=row['fallback'].get('restored', False),
                      releaseAtReachedPose=row['fallback'].get('releaseAtReachedPose', False))
    result.update(probeCount=row.get('probeCount', len(row.get('probes', []))),
                  observationCount=row.get('observationCount', len(row.get('observations', []))))
    return result


def record_release(current, report, job):
    """Preserve a completed open even if post-release acquisition/withdrawal fails."""
    row = current.get('insertionSearch')
    if row is None:
        return
    row['releaseExecuted'] = bool(current.get('releaseExecuted'))
    if row['releaseExecuted']:
        row['releasedAt'] = current.get('releasedAt')
        if row.get('id') and not row.get('_releaseEventRecorded'):
            from steward.ego.capability_usage import record
            record('insertion_search', event_id=row['id'], entity=current.get('object'),
                   stage='released', stamp=row['releasedAt'], action=row.get('executed', False),
                   measured=row.get('detected', False), reason=row.get('reason'), evidence=summary(row))
            row['_releaseEventRecorded'] = True
    report['insertionSearch'] = deepcopy(row)
    job.postcondition['insertionSearch'] = summary(row)
