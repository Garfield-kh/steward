"""Powered-tool recovery and per-hole evidence from perception and robot FK only."""
from copy import deepcopy
import math
import numpy as np

from steward.math3d import matrix_from_quat, quat_from_yaw


def grasp_wrist(planner, es, target, pitch):
    """Decouple docking heading from shaft heading using the available wrist yaw."""
    from steward.ego.slender_tool import axial_feature
    from steward.ego.esim import PredictionError
    feature = axial_feature(target, es.stamp)
    if not feature:
        raise PredictionError('driver grasp requires elongated geometry')
    g = planner.arm.geometry
    axis = np.asarray(feature['axis'])
    future = np.array([planner.arm.tool_rotation(0., r, p)[2]
        for p in np.linspace(0, g.wrist_pitch_max, 19)
        for r in np.linspace(-g.wrist_roll_max, g.wrist_roll_max, 73)])
    cache = {}
    def choose(heading):
        if heading not in cache:
            options = []
            for yaw in np.linspace(-g.wrist_yaw_max, g.wrist_yaw_max, 73):
                rotation = matrix_from_quat(quat_from_yaw(heading)) @ planner.arm.tool_rotation(yaw, 0., pitch)
                error = math.acos(float(np.clip(-np.min(future @ (rotation.T @ axis)), -1, 1)))
                if error <= math.radians(6):
                    options.append((abs(yaw), error, float(yaw)))
            cache[heading] = min(options)[2] if options else None
        return cache[heading]
    def wrist(heading):
        return (choose(heading) or 0.), 0.
    def issue(heading):
        return None if choose(heading) is not None else 'no wrist configuration can point this grasp down'
    return wrist, issue


def head_target(attempt, stamp):
    """Use a local post-release cap, else the screw's recorded release geometry.

    Component release FK already anchors entry. Never aim at a current union OBB.
    Estimates remain executable, but cannot serve as measured drive evidence.
    """
    check = attempt.get('seatCheck', {})
    if (check.get('passed') is True and check.get('headM') is not None
            and check.get('source') in ('rgbd_endpoint_width', 'rgbd_screw_head', 'rgbd_local_head_change', 'post_release_hole_rgbd')
            and check.get('observedAt', -1) > attempt['seatedAt']
            and 0 <= stamp-check['observedAt'] <= 20):
        return dict(head=list(check['headM']), stamp=check['observedAt'], source=check['source'], observed=True)
    released = attempt.get('releaseToolFeature')
    if released:
        head, tip = np.asarray(released['head']), np.asarray(released['tip'])
        # Z-only gravity settling; keep the executed screw XY and label the assumption.
        head = head + [0, 0, attempt['entry'][2]-tip[2]]
        return dict(head=head.tolist(), stamp=attempt['seatedAt'],
                    source='screw_release_fk_assumed_settled', observed=False)
    return dict(head=list(attempt['headBeforeRelease']), stamp=attempt['seatedAt'],
                source=attempt.get('headBeforeSource', 'release_estimate'), observed=False)


def drive_check(before, after, entry, started):
    """A command or an unchanged/occluded head cannot prove mechanical fastening."""
    observed = ('rgbd_endpoint_width', 'rgbd_screw_head', 'rgbd_local_head_change')
    out = dict(status='unknown', reason='no_independent_head_motion', before=before, after=after,
               mechanicalConnection=None)
    if not before or not after or before.get('source') not in observed or after.get('source') not in observed:
        return out
    if after['stamp'] <= max(started, before['stamp']):
        return out
    a, b, e = map(np.asarray, (before['head'], after['head'], entry))
    sink = float(a[2]-b[2]); lateral = float(np.linalg.norm(a[:2]-b[:2]))
    out.update(sinkM=sink, lateralM=lateral)
    if .006 <= sink <= .030 and lateral <= .008 and np.linalg.norm(b[:2]-e[:2]) <= .012 and -.003 <= b[2]-e[2] <= .012:
        out.update(status='driven', reason='independent_head_sink_after_drive')
    elif np.linalg.norm(b[:2]-e[:2]) > .012 or sink < -.006:
        out.update(status='not_driven', reason='head_displaced_after_drive')
    # An unchanged head could already be tight: retain unknown, never fabricate failure.
    return out


def progress(selected, attempts, part, body):
    rows = {a['hole']:a for a in attempts}
    holes = [dict(hole=h, screw=rows.get(h, {}).get('screw'),
                  status=rows.get(h, {}).get('driveStatus', 'not_driven'),
                  stage=rows.get(h, {}).get('driveStage', 'not_started'),
                  strokeExecuted=bool(rows.get(h, {}).get('driveStrokeExecuted')),
                  evidence=deepcopy(rows.get(h, {}).get('driveCheck')),
                  motion={k:deepcopy(v) for k,v in rows.get(h, {}).get('driverMotion', {}).items()
                          if k in ('stage', 'failureStage', 'contact', 'failureSample', 'heldForS')})
             for h in sorted(selected)]
    from steward.ego import driver_batch
    if driver_batch.enabled(): holes = driver_batch.compact(holes)
    remaining = [h['hole'] for h in holes if h['status'] != 'driven']
    return dict(driveProgress=holes, drivenScrews=len(holes)-len(remaining),
                remainingDriveHoles=remaining,
                driveResumeSuggestion=(dict(skill='fasten_onto', part=part, body=body, drive=True)
                                       if remaining else None))


def support_slots(planner, scene, table, radius):
    """World XY candidates from a perceived support's own local frame."""
    from steward.ego.assembly_motion import front_spots
    from steward.ego.planner import entity_box
    box, _, _ = entity_box(table)
    inset = np.array([box.hx, box.hy])-radius-.04
    if min(inset) <= 0:
        return []
    local = front_spots(planner, scene, table, radius)
    local += [(np.array([u,v])*inset).tolist() for u in np.linspace(-1,1,5)
              for v in np.linspace(-1,1,5)]
    r = np.array([[math.cos(box.yaw),-math.sin(box.yaw)],
                  [math.sin(box.yaw), math.cos(box.yaw)]])
    return [(np.array([box.cx,box.cy])+r @ xy).tolist() for xy in local]


def settle_joint(ctl, job, name, target, *, tol):
    """Driver-only steady-error recovery; never relax the requested encoder tolerance.

    Correct only after a timed-out absolute move and two stationary readings.
    No correction for an in-flight command, a large obstruction, or cancellation.
    The bounded command offset is measured afresh, not a tool/load constant.
    """
    from steward.ego.skills import SkillFailed
    ex = ctl.ex
    try:
        ex._set_joint(name, target, ramp=False, tol=tol)
        return
    except SkillFailed as exc:
        ctl.check()
        if not str(exc).startswith(f'joint {name} did not reach '):
            raise
        failure = exc
    sent = target
    for _ in range(2):
        ctl.check()
        q = ex._state()['joints'][name]
        ex._wait(.2)
        ctl.check()
        now = ex._state()['joints'][name]
        error = target-now
        if abs(error) < tol:
            return
        if abs(now-q) > tol/2 or abs(error) > .012:
            break
        command = sent+error
        if abs(command-target) > .012:
            break
        joints = dict(ex._state()['joints']); joints[name] = command
        arm = ctl.rt.esim.arm
        if not arm.reachable_path([dict(ex._state()['joints']), joints]):
            break
        job.postcondition.setdefault('driverServoCorrections', []).append(dict(
            joint=name, target=target, measured=now, command=command,
            source='settled_encoder_residual'))
        # The command can retain a load-dependent offset. Verify against TARGET
        # below, not against the compensated setpoint. Do not mutate shared servo.
        ex._set_joint(name, command, ramp=False, tol=.013)
        ex._wait(max(.2, ex.config.settle_s))
        ctl.check()
        sent = command
        q = ex._state()['joints'][name]
        ex._wait(.2)
        now = ex._state()['joints'][name]
        if abs(now-target) < tol and abs(now-q) < tol/2:
            return
    raise failure


def motion_sample(ctl, plan):
    """Compact FK evidence; none of this is an independent camera/contact sensor."""
    from steward.ego.fasten_plan import tool_estimate
    scene = ctl.rt.ego_state()
    held = scene.find(ctl.rt.estimator.held_id) if ctl.rt.estimator.held_id else None
    feature = tool_estimate(held, scene.stamp) if held else None
    state = ctl.ex._state()
    out = dict(stamp=state['stamp'], held=ctl.rt.estimator.held_id,
               source='grasp_relative_plus_encoders', forceObserved=False)
    if feature:
        tip = np.asarray(feature['tip'])
        head = np.asarray(plan.get('head', np.asarray(plan['goal'])-[0,0,.002]))
        delta = tip-head
        out.update(tipM=tip.tolist(), headM=head.tolist(), toolFeatureSource=feature.get('source'),
                   planarErrorM=float(np.linalg.norm(delta[:2])), heightAboveHeadM=float(delta[2]),
                   distanceToHeadM=float(np.linalg.norm(delta)),
                   axisErrorDeg=math.degrees(math.acos(float(np.clip(-np.asarray(feature['axis'])[2], -1, 1)))),
                   targetSource=plan.get('targetSource'))
    measured=plan.get('_entryMeasuredTip')
    if measured:
        tip=np.asarray(measured['tip'])+[0,0,state['joints']['lift']-measured['lift']]
        if measured.get('gripXY') is not None:
            tip[:2]+=ctl.ex._gripper_xy()-np.asarray(measured['gripXY'])
        delta=tip-np.asarray(plan['head'])
        out.update(tipM=tip.tolist(),headM=plan['head'],heightAboveHeadM=float(delta[2]),
            planarErrorM=float(np.linalg.norm(delta[:2])),distanceToHeadM=float(np.linalg.norm(delta)),
            toolFeatureSource='same_frame_tip_plus_translation_encoders',observedAt=measured['stamp'])
    return out


def remember_release(est, track, feature):
    """Keep a separate release-FK hypothesis even if segmentation later merges it."""
    if track.category != 'electric_screwdriver' or not feature:
        return
    saved = deepcopy(track)
    saved.pos = (np.asarray(feature['tip'])+feature['head'])/2
    saved.held = False
    saved.held_offset = saved.held_tool_feature = None
    saved.tool_feature = deepcopy(feature)
    saved.manipulation_view = saved.voxels = None
    saved.interface_views.clear()
    saved.interface_best_near = None
    axis = np.asarray(feature['axis'])
    saved.yaw = math.atan2(axis[1], axis[0])
    saved.last_seen = track.last_seen  # release encoders are not a new image
    origin = est.fastener_origins.setdefault(track.id, {})
    origin['driverRelease'] = dict(track=saved, feature=deepcopy(feature),
                                   stamp=feature['stamp'], source='release_fk')


def remembered_location(ctl):
    rows = [(eid, origin, origin['driverRelease'])
            for eid, origin in ctl.rt.estimator.fastener_origins.items()
            if 'driverRelease' in origin]
    return max(rows, key=lambda row:row[2]['stamp'], default=None)


def search_driver(ctl, job, support):
    """Visit the last release table and look once before choosing a grasp target."""
    from steward.ego.planner import entity_box
    from steward.ego.skills import SkillFailed
    scene = ctl.rt.ego_state()
    ctl._driver_target = None
    memory = remembered_location(ctl)
    table = scene.entity(support)
    target = None
    if memory:
        eid, origin, row = memory
        target = row['track'].pos.tolist()
        if scene.find(origin.get('support')):
            table = scene.entity(origin['support'])
        else:
            tables = [e for e in scene.entities if ctl.rt.esim._affords(e, 'support-surface')]
            if tables:
                table = min(tables, key=lambda e:np.linalg.norm(np.asarray(e.pose.value['p'])[:2]-target[:2]))
    else:
        # A currently perceived tool can be on another table than the assembly.
        candidates = [e for e in scene.entities if e.category.value == 'electric_screwdriver']
        if candidates:
            driver = min(candidates, key=lambda e:np.linalg.norm(np.asarray(e.pose.value['p'])[:2]-table.pose.value['p'][:2]))
            target = list(driver.pose.value['p'])
            table = ctl.rt.esim._anchor(scene, driver)
    old = getattr(ctl, 'observation_target', None)
    if target is not None:
        ctl.observation_target = target
    box, _, _ = entity_box(table); o = ctl.ex._state()['odometry']
    try:
        try:
            since = ctl.view(job, table, math.atan2(o['y']-box.cy, o['x']-box.cx), 0.)
        except SkillFailed as exc:
            ctl.check()
            ctl.warn('driver_table_view_unreachable', detail=str(exc))
            since = ctl.ex._state()['stamp']
        ctl.look(ctl.ex._state()['stamp'] if since is None else since, cycles=1)
    finally:
        if old is None:
            ctl.__dict__.pop('observation_target', None)
        else:
            ctl.observation_target = old
    if memory:
        restore_release(ctl, *memory)
    return table.id


def restore_release(ctl, eid, origin, row):
    """Prefer a separate new view; a union OBB cannot replace the remembered tool."""
    from steward.ego.planner import entity_box
    est = ctl.rt.estimator
    scene = ctl.rt.ego_state()
    saved = row['track']
    with ctl.rt._lock:
        if est.held_id or row.get('missedAt', -1) >= row['stamp']:
            return
        for ent in scene.entities:
            if ent.category.value != 'electric_screwdriver':
                continue
            sizes = sorted(est.tracks[ent.id].size) if ent.id in est.tracks else []
            expected = sorted(saved.size)
            t = est.tracks.get(ent.id)
            native_release = (t and t.id == eid
                              and (t.placement or {}).get('releasedAt') == row['stamp']
                              and t.placement.get('observation') is None
                              and np.linalg.norm(t.pos[:2]-t.placement['expectedXYM']) <= .01)
            fresh_nearby = (t and t.last_seen > row['stamp']
                            and np.linalg.norm(t.pos[:2]-saved.pos[:2]) < .08)
            if ((native_release or fresh_nearby)
                    and np.all(np.asarray(sizes) <= np.asarray(expected)*1.6+.01)):
                ctl._driver_target = ent.id
                alias = row.get('alias')
                if alias and alias != ent.id:
                    est.tracks.pop(alias, None)
                return
        # Preserve a merged track as an obstacle. Use a separate, explicitly
        # estimated alias with the release geometry; never its enlarged box.
        alias = eid+':driver-release'
        t = deepcopy(saved); t.id = alias
        support = scene.find(origin.get('support'))
        if support:
            top = entity_box(support)[2]
            dz = top+float(t.size[2])/2-t.pos[2]
            t.pos[2] += dz
            for key in ('tip', 'head'):
                t.tool_feature[key][2] += dz
            t.tool_feature['source'] = 'release_fk_assumed_settled'
        est.tracks[alias] = t
        est.version += 1
        row['alias'] = alias
        ctl._driver_target = alias
        ctl.warn('driver_release_target_restored', entity=alias, releasedAt=row['stamp'],
                 source=t.tool_feature['source'])


def observe_head(ctl, job, attempt):
    """Take a pre-drive view; fresh head evidence may correct a nominal target."""
    from steward.ego.fasten_align import close_view
    from steward.ego.fasten_geometry import screw_head
    from steward.ego.skills import SkillFailed
    try:
        close_view(ctl, job, attempt)
    except SkillFailed as exc:
        ctl.check()
        ctl.warn('driver_head_view_unreachable', detail=str(exc))
        ctl.look(ctl.ex._state()['stamp'], cycles=1)
    ctl.seat_posterior(attempt)
    target = head_target(attempt, ctl.rt.estimator.stamp)
    view = ctl.sample(attempt['screw'])
    measured = screw_head(view, dict(stamp=attempt['seatedAt']))
    # Associate a separate head locally, not a remote same-ID/merged envelope.
    if (measured and 0 <= ctl.rt.estimator.stamp-measured['stamp'] <= 20
            and np.linalg.norm(np.asarray(measured['head'])[:2]-np.asarray(attempt['entry'])[:2]) <= .025):
        target = dict(measured, observed=True)
    attempt['driveHeadObservation'] = deepcopy(target)
    return target


def correct_tip_xy(ctl, job, plan, motion):
    """Two bounded FK corrections at clearance height, before any descent."""
    from steward.ego.fasten_plan import tool_estimate
    for _ in range(2):
        ctl.check()
        scene = ctl.rt.ego_state()
        feature = tool_estimate(scene.find(ctl.rt.estimator.held_id), scene.stamp)
        if not feature or 'grip' not in feature:
            break
        error = np.asarray(plan['goal'])[:2]-np.asarray(feature['tip'])[:2]
        if np.linalg.norm(error) <= .002:
            break
        if np.linalg.norm(error) > .012:
            ctl.warn('driver_tip_residual_large', errorM=error.tolist())
            break
        grip = np.asarray(feature['grip'])[:2]+error
        ctl.ex._align_lateral(grip.tolist(), tol=.001)
        settle_joint(ctl, job, 'reach', ctl.ex._reach_to(grip), tol=.001)
        motion.setdefault('xyCorrections', []).append(dict(errorM=error.tolist(),
            sample=motion_sample(ctl, plan), source='grasp_relative_plus_encoders'))
