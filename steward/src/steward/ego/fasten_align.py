"""Post-release planar registration and bounded pin-pivot pushing, robot evidence only."""
from copy import deepcopy
import math
import numpy as np

from steward.math3d import matrix_from_quat, quat_from_yaw, quat_from_matrix
from steward.ego.fasten_plan import value, interface_holes, seat_target, support_anchor
from steward.ego.fasten_release import released_interface
from steward.ego.planner import entity_box
from steward.ego.esim import PredictionError

TOLERANCE_M = .012  # Same robot-side visual landing scale as seat_check.


def rz(angle):
    return matrix_from_quat(quat_from_yaw(angle))


def outline(view, position, rotation, size, after, stamp):
    """Raw outline bearing, with half-turn continuity and explicit error floor.

    A complete, elongated separate mask is needed. A template plus a measured
    outline estimates hidden holes; it never observes a mechanical connection.
    """
    if (not view or view.get('source') != 'rgbd_segment'
            or not after < view.get('stamp', -1) <= stamp
            or stamp-view['stamp'] > 20 or view.get('truncated', True)
            or view.get('invalidDepthFraction', 1) > .05):
        return None
    pts = np.asarray(view.get('points', []), float)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 48 or not np.isfinite(pts).all():
        return None
    # Scoped independent samples carry the pre-association perceived shape.
    # A collision envelope includes clearance and is not the object's outline.
    size = np.asarray(size, float).copy()
    if 'referenceSizeM' in view:
        reference = np.asarray(view['referenceSizeM'], float)
        # Box fitting can exchange X/Y while the registered hole frame keeps its
        # axes. Preserve that frame's long-axis convention, not the track's.
        axis = int(np.argmax(size[:2]))
        size[axis], size[1-axis] = max(reference[:2]), min(reference[:2])
        size[2] = reference[2]
    axis = int(np.argmax(size[:2])); other = 1-axis
    if size[axis] < 1.4*size[other] or size[other] <= 0:
        return None
    # The raw mask's oriented box can exchange its X/Y axes; follow its long axis.
    raw_size = np.asarray(view.get('visibleSizeM', []), float)
    if raw_size.shape != (3,) or not np.isfinite(raw_size).all() or 'boxYawRad' not in view:
        return None
    bearing = float(view['boxYawRad']) + (math.pi/2 if raw_size[1] > raw_size[0] else 0)
    expected = math.atan2(rotation[1, axis], rotation[0, axis])
    delta = (bearing-expected+math.pi/2) % math.pi-math.pi/2
    if not math.isfinite(delta) or abs(delta) > math.pi/4 or abs(rotation[2, 2]) < .9:
        return None
    r = rz(delta) @ rotation
    local = pts @ r
    low, high = np.quantile(local, [.01, .99], axis=0)
    extent = high-low
    error = abs(extent[:2]-size[:2])
    if np.any(error > np.maximum(.006, .15*size[:2])):
        return None  # partial/merged masks cannot translate the hole frame
    centre = r @ ((low+high)/2)
    if extent[2] > size[2]+.012 or abs(centre[2]-position[2]) > max(.02, size[2]/4):
        return None
    # Require both ends and both sides, not a diagonal line or tiny cap.
    q = (local[:, :2]-(low[:2]+high[:2])/2)/np.maximum(extent[:2]/2, .001)
    if any(np.sum(q[:, i]*sign > .7) < 6 for i in (0, 1) for sign in (-1, 1)):
        return None
    sigma = .003+float(max(error))/2
    angular = max(math.radians(2), math.atan2(sigma, size[axis]/2))
    if sigma > TOLERANCE_M or angular > math.radians(12):
        return None
    # XY is measurable from the outline; template-root Z remains the old estimate.
    p = np.asarray(position, float).copy(); p[:2] = centre[:2]
    return dict(position=p.tolist(), rotation=r.tolist(), observedAt=view['stamp'],
                positionUncertaintyM=sigma, angleUncertaintyRad=angular,
                source='rgbd_outline_plus_hole_template')


def measured_frame(planner, scene, frame, after):
    part, body = scene.find(frame['entity']), scene.find(frame['body'])
    if part is None or body is None or part.id == body.id:
        return None
    bsize = [2*entity_box(body)[0].hx, 2*entity_box(body)[0].hy,
             entity_box(body)[2]-entity_box(body)[1]]
    p = outline(value(part, 'contactView'), frame['position'], np.asarray(frame['rotation']),
                frame['size'], after, scene.stamp)
    b = outline(value(body, 'contactView'), body.pose.value['p'],
                matrix_from_quat(body.pose.value['q']), bsize, after, scene.stamp)
    if not p or not b or abs(p['observedAt']-b['observedAt']) > 2:
        return None
    # A mask copied onto both ids is not two independent objects.
    pv, bv = value(part, 'contactView'), value(body, 'contactView')
    if np.array_equal(np.asarray(pv['points']), np.asarray(bv['points'])):
        return None
    result = deepcopy(frame)
    result.update(position=p['position'], rotation=p['rotation'], observedAt=p['observedAt'],
                  source=p['source'], observed=False, outlineObserved=True)
    r = np.asarray(p['rotation']); origin = np.asarray(p['position'])
    for h in result['holes']:
        h['point'] = (origin+r @ h['p']).tolist()
    br = np.asarray(b['rotation']); bp = np.asarray(b['position'])
    body_points = {h['id']:(bp+br @ h['p']).tolist() for h in interface_holes(planner, body)}
    result['bodyHolePoints'] = body_points
    return result, body_points, p, b


def assess(planner, scene, part, body, pivot_hole, after):
    out = dict(status='unknown', angleDeg=None, residualM=None, remainingResidualsM={},
               toleranceM=TOLERANCE_M, reason='no_fresh_independent_outlines',
               throughBothLayers=None)
    frame = released_interface(scene, part, body)
    measured = measured_frame(planner, scene, frame, after) if frame else None
    if measured is None:
        return out
    frame, body_points, p, b = measured
    points = {h['id']:np.asarray(h['point']) for h in frame['holes']}
    pairs = {h['bodyHole']:points[h['partHole']] for h in frame['holePairs']
             if h['partHole'] in points and h['bodyHole'] in body_points}
    if pivot_hole not in pairs or len(pairs) < 2:
        out['reason'] = 'missing_hole_correspondence'; return out
    evidence = value(scene.find(body), 'fasteningEvidence', [])
    a = next((a for a in evidence if a['hole'] == pivot_hole and a['part'] == part), {})
    check = a.get('seatCheck', {})
    head = (check.get('headM') if check.get('passed') is True
            and after < check.get('observedAt', -1) <= scene.stamp
            and scene.stamp-check['observedAt'] <= 20 else None)
    # Unknown/contradicted first screw is not silently promoted into a pin.
    pivot = np.asarray(head if head is not None else pairs[pivot_hole])[:2]
    remaining = [h for h in pairs if h != pivot_hole]
    u = np.array([pairs[h][:2]-pivot for h in remaining])
    v = np.array([np.asarray(body_points[h])[:2]-pivot for h in remaining])
    dot = float(np.sum(u*v)); cross = float(np.sum(u[:, 0]*v[:, 1]-u[:, 1]*v[:, 0]))
    angle = math.atan2(cross, dot)
    residuals = {h:float(np.linalg.norm(pairs[h][:2]-np.asarray(body_points[h])[:2])) for h in remaining}
    predicted = np.linalg.norm(u @ rz(angle)[:2, :2].T-v, axis=1)
    uncertainty = p['positionUncertaintyM']+b['positionUncertaintyM']
    upper = np.array([v[:2] for v in pairs.values()]); lower = np.array([body_points[h][:2] for h in pairs])
    upper -= upper.mean(axis=0); lower -= lower.mean(axis=0)
    yaw_error = -math.atan2(float(np.sum(upper[:,0]*lower[:,1]-upper[:,1]*lower[:,0])),
                            float(np.sum(upper*lower)))
    out.update(status='estimated', angleDeg=math.degrees(angle), residualM=max(residuals.values()),
        remainingResidualsM=residuals, predictedResidualM=float(max(predicted)),
        relativeYawErrorDeg=math.degrees(yaw_error),
        positionUncertaintyM=uncertainty,
        angleUncertaintyDeg=math.degrees(p['angleUncertaintyRad']+b['angleUncertaintyRad']),
        reason='independent_outlines_template_holes', frame=frame, bodyPose=b,
        pivotM=pivot.tolist(), pivotObserved=head is not None,
        pivotResidualM=float(np.linalg.norm(pairs[pivot_hole][:2]-np.asarray(body_points[pivot_hole])[:2])),
        observedAt=min(p['observedAt'], b['observedAt']))
    out['needsCorrection'] = out['residualM'] > TOLERANCE_M
    return out


def hole_order(planner, scene, part, body, holes):
    """Smallest current pair error first; robot travel breaks equal-error ties."""
    robot = planner._robot_xy_yaw(scene)
    frame = released_interface(scene, part, body)
    try:
        measured = measured_frame(planner, scene, frame, frame['releasedAt']) if frame else None
    except (KeyError, TypeError, ValueError):
        measured = None
    observed = {}
    if measured:
        f, bp, _, _ = measured
        hp = {h['id']:h['point'] for h in f['holes']}
        observed = {p['bodyHole']:(hp[p['partHole']], bp[p['bodyHole']]) for p in f['holePairs']
                    if p['partHole'] in hp and p['bodyHole'] in bp}
    ranks = []
    for i, hole in enumerate(holes):
        try:
            t = seat_target(planner, scene, dict(part=part, body=body, hole=hole['id']))
            point = t['point']; error = (t.get('pairAlignment') or {}).get('distanceXYM')
            if hole['id'] in observed:
                point, base = observed[hole['id']]
                error = float(np.linalg.norm(np.asarray(point)[:2]-np.asarray(base)[:2]))
            a = next((a for a in value(scene.find(body), 'fasteningEvidence', [])
                      if a['hole'] == hole['id'] and a['part'] == part and a.get('seatConfirmed')), None)
            ranks.append((0 if a else 1, round(error, 6) if error is not None else math.inf,
                          float(np.linalg.norm(np.asarray(point)[:2]-robot[:2])), i, hole))
        except (PredictionError, KeyError, ValueError):
            ranks.append((1, math.inf, math.inf, i, hole))
    return ([r[-1] for r in sorted(ranks, key=lambda r:r[:-1])],
            'rgbd_outline_plus_hole_template' if measured else 'registered_frame_estimate',
            measured[0] if measured else None)


def push_paths(assessment, radius=.02):
    """Two far-end side contacts, no lifting or translation of the pin."""
    f = assessment['frame']; r = np.asarray(f['rotation']); centre = np.asarray(f['position'])
    size = np.asarray(f['size']); axis = int(np.argmax(size[:2])); side = 1-axis
    pivot = np.asarray(assessment['pivotM']); angle = math.radians(assessment['angleDeg'])
    result = []
    for sign in (-1, 1):
        end = centre+r[:, axis]*size[axis]*.45*sign
        lever = end[:2]-pivot
        if np.linalg.norm(lever) < .025:
            continue
        tangent = np.sign(angle)*np.array([-lever[1], lever[0]])/np.linalg.norm(lever)
        normal = r[:2, side]*np.sign(tangent @ r[:2, side])
        if tangent @ normal < .7:
            continue
        start = end.copy(); start[:2] -= normal*(size[side]/2+radius)
        sweep = math.copysign(min(abs(angle), math.radians(12), .012/np.linalg.norm(start[:2]-pivot)), angle)
        n = max(2, math.ceil(abs(sweep)*np.linalg.norm(start[:2]-pivot)/.001))
        points = [np.r_[pivot+rz(a)[:2, :2] @ (start[:2]-pivot), start[2]]
                  for a in np.linspace(0., sweep, n+1)]
        result.append((points, sweep))
    return sorted(result, key=lambda row:-np.linalg.norm(row[0][0][:2]-pivot))


def push_plan(planner, scene, params):
    """Fresh internal receipt; contact motion reuses bounded approach/arm primitives."""
    if planner._held(scene) is not None:
        raise PredictionError('pivot push needs an empty gripper')
    a = assess(planner, scene, params['part'], params['body'], params['hole'], params['after'])
    if a['status'] != 'estimated' or not a['pivotObserved'] or not a['needsCorrection']:
        raise PredictionError('pivot or alignment unavailable for a pin-constrained push')
    if a['positionUncertaintyM'] > TOLERANCE_M or a['predictedResidualM'] >= a['residualM']-.001:
        raise PredictionError('rotation cannot resolve measured mismatch within observation precision')
    others = [x for x in value(scene.find(params['body']), 'fasteningEvidence', [])
              if x['part'] == params['part'] and x['hole'] != params['hole'] and x.get('releaseExecuted')]
    if others:
        raise PredictionError('another released screw may constrain rotation; continue remaining holes')
    body = deepcopy(scene.entity(params['body']))
    support = support_anchor(planner, scene, body, [])
    body.pose.value = dict(p=a['bodyPose']['position'], q=quat_from_matrix(a['bodyPose']['rotation']))
    current = {'wrist_yaw':0., 'wrist_roll':0., 'wrist_pitch':0., **planner._arm_joints(scene)}
    radius = .02
    clearance = radius  # closed jaws; opening is deferred until clear of the part
    safe = max(current['lift'], entity_box(body)[2]+.08, a['frame']['position'][2]+.08)
    if safe > planner.lift_max:
        raise PredictionError('pivot push lacks lift clearance')
    errors = []
    for points, angle in push_paths(a, radius):
        try:
            # All paths, including descent/withdrawal, exclude body and every screw.
            for point in points:
                if np.linalg.norm(point[:2]-a['pivotM']) < clearance+.015:
                    raise PredictionError('pivot screw inside gripper sweep')
                box, _, top = entity_box(support)
                local = rz(-box.yaw)[:2, :2] @ (point[:2]-[box.cx, box.cy])
                if abs(local[0])+clearance > box.hx or abs(local[1])+clearance > box.hy or point[2]-clearance < top:
                    raise PredictionError('pivot contact too near support')
                for ent in scene.entities:
                    if ent.id in (params['part'], support.id, scene.self_id):
                        continue
                    box, low, high = entity_box(body if ent.id == body.id else ent)
                    xy = rz(-box.yaw)[:2, :2] @ (point[:2]-[box.cx, box.cy])
                    if abs(xy[0]) < box.hx+clearance+.002 and abs(xy[1]) < box.hy+clearance+.002 and low < safe and high > point[2]-clearance:
                        raise PredictionError('pivot corridor intersects '+ent.id)
            start = points[0]
            def path(heading, reach, near, _pitch):
                forward = np.array([math.cos(heading), math.sin(heading)])
                changes = [dict(lift=safe), dict(reach=near), dict(wrist_yaw=0.),
                           dict(wrist_roll=0.), dict(wrist_pitch=0.), dict(reach=reach), dict(lift=start[2])]
                changes += [dict(reach=reach+float((p[:2]-start[:2]) @ forward)) for p in points]
                changes += [dict(lift=safe), dict(reach=near)]
                targets = [current]
                for change in changes:
                    prev = targets[-1]; end = {**prev, **change}
                    n = max(1, *(math.ceil(abs(end[k]-prev[k])/(.005 if k in ('reach','lift') else .05)) for k in change))
                    targets.extend({k:prev[k]+(end[k]-prev[k])*i/n for k in prev} for i in range(1,n+1))
                return planner.arm.reachable_path(targets, start_joints={j.id:j.q.value[0] for j in planner._self(scene).rigid.joints if j.q.value})
            approach = planner._approach(scene, support, start[:2], float(start[2]), set(),
                                         lifts=[safe, float(start[2])], approach_path=path)
            heading = approach['yaw']; left = np.array([-math.sin(heading), math.cos(heading)])
            obstacles = planner._floor_obstacles(scene, set()); carry = planner._carry(scene)
            for point in points:
                dock = np.asarray(approach['dock'])+left*float((point[:2]-start[:2]) @ left)
                if planner._footprint_blocked(obstacles, dock, heading, carry):
                    raise PredictionError('pivot lateral base path obstructed')
            plan = dict(approach, object=params['part'], body=params['body'], support=support.id,
                        gripSpot=start[:2].tolist(), lift=float(start[2]), safeLift=safe,
                        points=[p.tolist() for p in points], angleRad=angle, evidence=a,
                        wristYaw=0., wristRoll=0., wristPitch=0.)
            return planner._receipt(scene, '_pivot_align', params, .03,
                ['Planar empty-jaw push; alignment requires an independent post-action observation.'], plan)
        except PredictionError as exc:
            errors.append(str(exc))
    raise PredictionError('no exposed-end pivot push: '+'; '.join(errors))


def save_measurement(controller, assessment):
    if assessment.get('status') != 'estimated':
        return
    f = deepcopy(assessment['frame'])
    with controller.rt._lock:
        est = controller.rt.estimator
        old = est.released_interfaces.get(f['entity'])
        if old is None or old['releasedAt'] != f['releasedAt']:
            return
        # Keep release FK intact as history; subsequent targets use this posterior.
        f['releaseSnapshot'] = deepcopy(old.get('releaseSnapshot') or old)
        f['alignmentEvidence'] = {k:deepcopy(v) for k,v in assessment.items() if k not in ('frame','bodyPose')}
        est.released_interfaces[f['entity']] = f
        est.version += 1


def close_view(controller, job, attempt):
    """One near view, without a second screw wait or an additional camera tour."""
    scene = controller.rt.ego_state()
    support = support_anchor(controller.rt.esim, scene, scene.entity(attempt['body']), [])
    box, _, _ = entity_box(support); o = controller.ex._state()['odometry']
    old_close = getattr(controller, '_close_view', False)
    old_target = getattr(controller, 'observation_target', None)
    controller._close_view = True
    controller.observation_target = list(attempt['entry'])
    try:
        try:
            since = controller.view(job, support, math.atan2(o['y']-box.cy,o['x']-box.cx), 0.)
        except Exception as exc:
            from steward.ego.skills import SkillFailed
            controller.check()
            if not ((getattr(controller, '_seat_only', False) and getattr(controller, '_lower_interface', False))
                    or getattr(controller, '_driver_alignment', False)) or not isinstance(exc, SkillFailed):
                raise
            controller.warn('driver_near_view_unreachable_use_current_view' if getattr(controller, '_driver_alignment', False)
                            else 'seat_near_view_unreachable_use_current_view', detail=str(exc))
            since = controller.ex._state()['stamp']
        controller.look(controller.ex._state()['stamp'] if since is None else since, cycles=1)
    finally:
        controller._close_view = old_close
        if old_target is None:
            del controller.observation_target
        else:
            controller.observation_target = old_target


def align_after_first(controller, job, attempt):
    """Advisory posterior/action/posterior, at most two pushes for this invocation."""
    from steward.ego.skills import SkillStopped
    row = dict(pivotHole=attempt['hole'], pivotScrew=attempt['screw'], status='unknown',
               angleDeg=None, corrected=False, residualAfterM=None, corrections=[], observations=[],
               budget=2, throughBothLayers=None)
    job.postcondition['pinAlignment'] = row
    after = attempt['seatedAt']
    for index in range(3):
        try:
            close_view(controller, job, attempt)
            controller.seat_posterior(attempt)
            a = assess(controller.rt.esim, controller.rt.ego_state(), attempt['part'],
                       attempt['body'], attempt['hole'], after)
        except SkillStopped:
            raise
        except Exception as exc:
            controller.check()
            a = dict(status='unknown', angleDeg=None, residualM=None, reason=str(exc))
        row['observations'].append({k:deepcopy(v) for k,v in a.items() if k not in ('frame','bodyPose')})
        row.update(status=a['status'], residualAfterM=a['residualM'],
                   withinToleranceEstimate=(a['residualM'] <= TOLERANCE_M if a['residualM'] is not None else None))
        if index == 0:
            row['angleDeg'] = a['angleDeg']
            row['relativeYawErrorDeg'] = a.get('relativeYawErrorDeg')
        save_measurement(controller, a)
        if a['status'] != 'estimated' or not a.get('needsCorrection') or index == 2:
            break
        if not a.get('pivotObserved'):
            row['reason'] = 'first_screw_location_unknown'; break
        # Budget is shared across a drop-recovery recursion; it is separate from
        # the pre-existing two screw replays and never changes their allowance.
        budget = getattr(controller, '_pivot_budget', None)
        if budget is None:
            budget = controller._pivot_budget = [2]
        if not budget[0]:
            row['reason'] = 'pivot_budget_exhausted'; break
        budget[0] -= 1
        action = dict(executed=False)
        row['corrections'].append(action)
        try:
            plan = controller.step(job, '_pivot_align', dict(part=attempt['part'], body=attempt['body'],
                                   hole=attempt['hole'], after=after))
            action.update(executed=True, commandedAngleDeg=math.degrees(plan['angleRad']))
            row['corrected'] = True
        except SkillStopped:
            raise
        except Exception as exc:
            controller.check()
            action['failure'] = str(exc)
            controller.warn('pin_alignment_failed_continue', detail=str(exc))
        # A failed motion may have partly moved the part: always observe again.
        after = controller.ex._state()['stamp']
    row['attemptsUsed'] = len(row['corrections'])
    row['corrected'] = bool(row['corrected'] or row.get('pushExecuted'))


def execute_push(controller, job, plan):
    """Small sampled arc; empty closed jaws push, never grip or lift the part."""
    ex = controller.ex
    ex._dock(job, plan, retract=False)
    controller.check()
    ex._set_joint('lift', plan['safeLift'])
    ex._set_joint('reach', plan['retractReach'])
    for key in ('wrist_yaw','wrist_roll','wrist_pitch'):
        if key in ex._state()['joints']:
            ex._set_joint(key, 0., tol=.02)
    if controller.rt.estimator.held_id or ex._state().get('gripper', {}).get('holding'):
        from steward.ego.skills import SkillFailed
        raise SkillFailed('pivot push gripper is occupied')
    ex._command({'gripper':'close'})  # in clear space, never around the upper part
    ex._align_lateral(plan['gripSpot'], tol=.001)
    ex._set_joint('reach', ex._reach_to(plan['gripSpot']))
    try:
        with controller.rt._lock:
            f = controller.rt.estimator.released_interfaces.get(plan['object'])
            if f is not None:
                f.update(source='pre_push_pose_unverified_after_motion', observed=False, outlineObserved=False,
                         alignmentMovedAt=ex._state()['stamp'])
                controller.rt.estimator.version += 1
        ex._set_joint('lift', plan['lift'])
        for point in plan['points'][1:]:
            controller.check()
            ex._align_lateral(point[:2], tol=.001)
            ex._set_joint('reach', ex._reach_to(point[:2]), ramp=True, tol=.001)
            ex._wait(.08)
            # Encoder motion is only a target hypothesis until the next RGB-D view.
            from steward.ego.skills import SkillFailed
            actual = ex._gripper_xy()
            if np.linalg.norm(actual-np.asarray(point[:2])) > .003:
                raise SkillFailed('pivot push encoder residual exceeds 3 mm')
            start = np.asarray(plan['points'][0])[:2]-plan['evidence']['pivotM']
            end = actual-np.asarray(plan['evidence']['pivotM'])
            angle = math.atan2(start[0]*end[1]-start[1]*end[0], float(start @ end))
            f = deepcopy(plan['evidence']['frame']); r = rz(angle)
            pivot = np.r_[plan['evidence']['pivotM'], f['position'][2]]
            f['position'] = (pivot+r @ (np.asarray(f['position'])-pivot)).tolist()
            f['rotation'] = (r @ np.asarray(f['rotation'])).tolist()
            for h in f['holes']:
                h['point'] = (pivot+r @ (np.asarray(h['point'])-pivot)).tolist()
            f.update(source='pivot_push_encoder_estimate', observed=False, outlineObserved=False,
                     alignmentMovedAt=ex._state()['stamp'])
            with controller.rt._lock:
                old = controller.rt.estimator.released_interfaces.get(f['entity'])
                if old is not None and old['releasedAt'] == f['releasedAt']:
                    f['releaseSnapshot'] = deepcopy(old.get('releaseSnapshot') or old)
                    controller.rt.estimator.released_interfaces[f['entity']] = f
                    controller.rt.estimator.version += 1
            job.postcondition.setdefault('pinAlignment', {})['pushExecuted'] = True
        job.postcondition.setdefault('pinAlignment', {})['pushExecuted'] = True
    finally:
        controller.check()
        # Never lift a sensed accidental grasp. Normally the jaws remain closed
        # through withdrawal, avoiding an opening sweep against the body.
        if controller.rt.estimator.held_id or ex._state().get('gripper', {}).get('holding'):
            ex._command({'gripper':'open'})
        ex._set_joint('lift', plan['safeLift'])
        ex._withdraw(plan)
        ex._command({'gripper':'open'})
