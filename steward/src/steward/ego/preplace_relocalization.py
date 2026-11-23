"""One fresh planar localization at each placement hover, using robot RGB-D only."""
from copy import deepcopy
import math
import uuid
import numpy as np
from steward.ego import action_budget as ab, wrist_observation as wo
from steward.ego.layout_pose import match, transform, rotation


def observe(sample, rows, occupied, opening):
    from steward.ego.entry_layout import surface_arcs
    raw, cam = sample
    diagnostics = {}
    # Search the raster, not a ball around a stale map point. Detector geometry
    # and the registered inter-hole distances reject unrelated features.
    found = wo.holes(sample, None, diagnostics=diagnostics,
                     region=np.ones(raw['depth'].shape, bool))
    arcs, _ = surface_arcs(sample, rows[0]['point'], opening, {}, full_frame=True)
    for prior_row in rows:
        for diameter in (opening, 2*opening):
            local, _ = surface_arcs(sample,prior_row['point'],diameter,{})
            arcs.extend(local)
    for h in arcs:
        if not any(np.linalg.norm(np.asarray(h['point'])-q['point']) <
                   max(h['radiusM'], q.get('radiusM',0.)) for q in found):
            found.append(h)
    common = float(raw['header'].get('odometry',{}).get('sigma',{}).get('pos',0.))
    features = [dict(h, relUncM=max(float(h.get('planeNoiseM',0.)),
                     float(h.get('uncertaintyM',0.))-(0. if h.get('source')=='same_frame_support_boundary_arc' else common))) for h in found]
    fit, reason = match(rows, features, occupied)
    return fit, dict(reason=reason, detectedHoles=len(found), diagnostics=diagnostics)


def layout(controller, current, params):
    est = controller.rt.estimator
    body = params.get('seatBody') or params.get('on')
    ctx = getattr(controller, '_registered_alignment', {}) or {}
    reg = ctx.get('registration') or {}
    if params.get('seatBody'):
        from steward.ego.entry_layout import prior
        prior_row = prior(controller, dict(current, body=body, seatHole=params.get('seatHole')))
        rows = prior_row['layout']
        return body, rows, prior_row['occupiedHoles'], reg
    target = current.get('placementTarget') or {}
    record = est.stationary_assemblies.records.get(body, {})
    ref = record.get('reference') or target.get('mateReference') or {}
    pairs = target.get('holePairs') or []
    wanted = {p['bodyHole'] for p in pairs}
    rows = []
    component = target.get('componentHoleRegistration') or ref.get('componentHoleRegistration')
    if component and component.get('part') == current.get('object'):
        for local, pair in zip(component['local'], component['holePairs']):
            xy = rotation(component['yaw']) @ local + component['position']
            rows.append(dict(bodyHole=pair['bodyHole'],point=[*xy,ref['position'][2]]))
        return body, rows, [], {}
    for h in ref.get('holes', []):
        if h['id'] not in wanted: continue
        point = np.asarray(ref['position'], float).copy()
        point[:2] += rotation(ref['yaw']) @ np.asarray(h['p'][:2])
        point[2] += h['p'][2]
        rows.append(dict(bodyHole=h['id'], point=point.tolist()))
    return body, rows, [], {}


def move_frame(frame, fit, stamp):
    """Only map-frame geometry is changed; local hole/grasp coordinates stay local."""
    for key in ('position', 'centreM'):
        if key in frame: frame[key] = transform(frame[key], fit)
    for key in ('yaw', 'yawRad'):
        if key in frame: frame[key] += fit['yawRad']
    if 'rotation' in frame:
        frame['rotation'] = np.array([rotate_vector(v,fit) for v in np.asarray(frame['rotation']).T]).T.tolist()
    for h in frame.get('holes', []):
        if 'point' in h: h['point'] = transform(h['point'], fit)
        if 'normal' in h: h['normal'] = rotate_vector(h['normal'], fit)
    for key, p in frame.get('bodyHolePoints', {}).items():
        frame['bodyHolePoints'][key] = transform(p, fit)
    for target in frame.get('registeredSeatTargets', {}).values():
        for key in ('point', 'upperPoint'):
            if key in target: target[key] = transform(target[key], fit)
        target['observedAt'] = stamp
    frame['relocalizedAt'] = stamp


def rotate_vector(vector, fit):
    v = np.asarray(vector, float).copy()
    v[:2] = rotation(fit['yawRad']) @ v[:2]
    return v.tolist()


def move_registration(reg, fit, stamp, odom):
    for key in ('point', 'executionPoint'):
        if key in reg: reg[key] = transform(reg[key], fit)
    for h in reg.get('layout', []):
        h['point'] = transform(h['point'], fit)
    move_frame(reg, fit, stamp)
    if reg.get('frame'): move_frame(reg['frame'], fit, stamp)
    reg.setdefault('originalRegistrationSource',reg.get('source'))
    reg.update(observedAt=stamp, invalidated=None, measurementSource='preplace_layout_rgbd',
               source='native_rgbd_support_holes',
               baseOdometry=deepcopy(odom))
    if 'point' in reg:
        p = np.asarray(reg['point'])-[odom['x'],odom['y'],0.]
        p[:2] = rotation(odom['yaw']).T @ p[:2]
        reg['basePointM'] = p.tolist()


def install(controller, body, fit, sample):
    """Persist the new body frame and every live interface cache atomically.

    Historical fastening receipts are observations at their original time and
    deliberately stay untouched. Local held transforms also stay untouched.
    """
    from steward.math3d import quat_from_yaw
    est = controller.rt.estimator; header = sample[0]['header']
    stamp, odom = header['stamp'], header['odometry']
    records = est.stationary_assemblies.records
    record = records.get(body)
    if record:
        ref = record['reference']; move_frame(ref, fit, stamp)
        ref.update(registeredAt=stamp, localizationSource='preplace_layout_rgbd')
        if 'rotationM' in ref:
            ref['rotationM'] = np.array([rotate_vector(v,fit) for v in np.asarray(ref['rotationM']).T]).T.tolist()
        surface = ref.get('placementSurface')
        if surface and surface.get('patch'):
            surface['patch']['point'] = transform(surface['patch']['point'],fit)
            surface['patch']['normal'] = rotate_vector(surface['patch']['normal'],fit)
            surface['referenceAt'] = stamp
        component = ref.get('componentHoleRegistration')
        if component:
            for key in ('position','nominalPosition'):
                component[key] = transform(component[key],fit)
            for key in ('yaw','nominalYaw'): component[key] += fit['yawRad']
            component.update(referenceAt=stamp, observedAt=stamp, baseOdometry=deepcopy(odom),
                basePositionM=(rotation(odom['yaw']).T @ (np.asarray(component['position'])-[odom['x'],odom['y']])).tolist(),
                baseYawRad=component['yaw']-odom['yaw'])
        entity = record['entity']
        entity.pose.value = dict(p=deepcopy(ref['position']),q=quat_from_yaw(ref['yaw']))
        entity.pose.t = stamp
        record.update(hardMotion=False, motionSamples={})
        for part in record.get('parts',[]): move_frame(part,fit,stamp)
    for memory in getattr(getattr(est,'assembly_memory',None),'records',[]):
        if memory.get('reference',{}).get('entity') == body:
            move_frame(memory['reference'],fit,stamp)
            for part in memory.get('parts',[]): move_frame(part,fit,stamp)
    for frame in getattr(est,'released_interfaces',{}).values():
        if frame.get('body') == body: move_frame(frame,fit,stamp)
    ctx = getattr(controller,'_registered_alignment',{}) or {}
    registrations = list(getattr(est,'_seat_registrations',{}).values())
    if ctx.get('registration'): registrations.append(ctx['registration'])
    watch = getattr(controller.rt,'_support_hole_watch',None)
    if watch: registrations.append(watch)
    seen = set()
    for reg in registrations:
        if reg.get('body') == body and id(reg) not in seen:
            seen.add(id(reg)); move_registration(reg,fit,stamp,odom)
            # The observed upper opening supersedes the old upper/lower midpoint.
            if 'point' in reg: reg['executionPoint'] = deepcopy(reg['point'])
            frame = getattr(est,'released_interfaces',{}).get(reg.get('entity'),{})
            for h in reg.get('layout',[]):
                target = frame.get('registeredSeatTargets',{}).get(h['bodyHole'])
                if target is not None:
                    target.update(point=deepcopy(h['point']),upperPoint=deepcopy(h['point']),
                                  observedAt=stamp,targetSource='preplace_layout_rgbd')
    ctx.pop('correctionPoint',None)
    est.version += 1


def emit(row):
    from steward.ego.capability_usage import record
    record('preplace_relocalization',entity=row['object'],event_id=row['id'],
           measured=bool(row.get('fit')),action=row['adopted'],reason=row.get('reason'),evidence=row)


def hover(controller, job, current, params, report):
    from steward.ego import placement_xy_compensation
    component = placement_xy_compensation.applies(current)
    if component and getattr(controller, '_placement_xy_compensation_replay', False): return False
    if report.get('preplaceRelocalization'): return False
    row = dict(schema='preplace_relocalization.v1',id=uuid.uuid4().hex,object=current.get('object'),
               frames=[],adopted=False,reason='layout_unavailable')
    report['preplaceRelocalization'] = row
    rt, ex = controller.rt, controller.ex
    try:
        body, rows, occupied, reg = layout(controller,current,params)
        row['body'] = body
        if not rows: return False
        from steward.ego.tip_entry_servo import camera_for
        cameras = [camera_for(rt)]
        if wo.CAMERA in getattr(rt,'_provided_camera_ids',()) and wo.CAMERA in rt.estimator.cameras:
            cameras.append(wo.CAMERA)
        cameras = list(dict.fromkeys(c for c in cameras if c))
        row['reason'] = 'camera_unavailable'
        candidates = []
        for camera in cameras:
            wo.context(controller,'preplace_relocalization',rows[0]['point'],None)
            controller._wristContext['entryRegistration'] = dict(layout=deepcopy(rows),occupiedHoles=occupied)
            sample = wo.capture(controller,ab.make(ex,ab.frame_timeout(ex,4.),'preplace RGB-D frame'),
                                camera=camera,check_view=False)
            frame = dict(camera=camera,capture=deepcopy(getattr(controller,'_wristCapture',{})))
            row['frames'].append(frame)
            if sample is None:
                frame['reason'] = frame['capture'].get('reason','frame_unavailable'); continue
            if sample[0]['header']['stamp'] <= getattr(rt.estimator,'held_since',-math.inf):
                frame['reason'] = 'frame_precedes_grasp'; continue
            # Non-raster geometry providers retain their existing localization
            # contract; only native pixels can create a ASSEMBLY_PLACEMENT_XY_COMPENSATION observed entrance.
            component = component and 'depth' in sample[0]
            fit, detail = (placement_xy_compensation.observe(controller, current, sample) if component else
                observe(sample,rows,occupied,rt.esim.arm.geometry.jaw_opening))
            frame.update(detail,fit=fit)
            if fit:
                candidates.append((fit,sample,camera))
                if component: break  # one successful frame; no second camera confirmation
        if not candidates:
            row['reason'] = row['frames'][-1].get('reason') if row['frames'] else row['reason']
            return False
        # Prefer the most constrained fit; equal support keeps the entry view first.
        fit, sample, camera = max(candidates,key=lambda c:len(c[0].get('holePairs', [])) if component else c[0]['visibleHoles'])
        row['crossViewChecks'] = []
        chosen = {m['hole']:m['point'] for m in fit.get('matches',[])}
        for other, other_sample, other_camera in candidates:
            if other_camera == camera: continue
            for m in other.get('matches',[]):
                if m['hole'] not in chosen: continue
                error = float(np.linalg.norm(np.asarray(m['point'])[:2]-np.asarray(chosen[m['hole']])[:2]))
                common = sum(float(s[0]['header'].get('odometry',{}).get('sigma',{}).get('pos',0.))
                             for s in (sample,other_sample))
                limit = fit['uncertaintyM']+other['uncertaintyM']+common
                row['crossViewChecks'].append(dict(camera=other_camera,hole=m['hole'],residualM=error,uncertaintyM=limit))
                if error > limit:
                    row['reason'] = 'camera_correspondence_inconsistent'
                    return False
        if component:
            # Only the selected face's observed pair changes. Do not turn or
            # translate the entire body/other interface from symmetric holes.
            with rt._lock:
                record = rt.estimator.stationary_assemblies.records.get(body)
                if record is None:
                    row['reason'] = 'body_registration_missing'; return False
                record['reference']['componentHoleRegistration'] = deepcopy(fit)
                rt.estimator.version += 1
            receipt = rt.predict('place', dict(params, closeRelease=True))
            if not receipt.get('withinRiskLimit'):
                row['reason'] = 'corrected_plan_not_admitted_best_estimate_continue'
                return False
            replacement = rt._consume_receipt(receipt); replacement['precision'] = report
            row.update(fit=fit, adopted=True, reason=None, observedAt=fit['observedAt'])
            deadline = getattr(ex, '_primitive_deadline', None)
            ex._begin(job, 'place', receipt)
            if deadline is not None: ex._primitive_deadline = ab.limited(deadline, ex._primitive_deadline)
            try: ex._place(job, replacement)
            finally: current.update(replacement)
            return True
        reference = rt.estimator.stationary_assemblies.records.get(body,{}).get('reference',{})
        anchor = np.asarray(reference.get('position',np.mean([r['point'] for r in rows],axis=0)))
        correction = np.asarray(transform(anchor,fit))-anchor
        row.update(fit=fit,camera=camera,correctionAnchorM=anchor.tolist(),translationM=correction.tolist(),
                   translationMm=float(np.linalg.norm(correction[:2])*1000),
                   yawDeg=math.degrees(fit['yawRad']),observedAt=sample[0]['header']['stamp'])
        # Small hover changes use the existing same-frame differential servo,
        # whose admission scales with observed clearance and step size. Do not
        # mutate the whole assembly then submit a second transport plan.
        from steward.ego import current_entry
        local = (current_entry.edge_refinement(controller, params, rows, fit, rt.esim.arm.geometry.jaw_opening)
                 if params.get('seatBody') and current_entry.enabled() else None)
        if local is not None:
            row.update(reason='local_refinement_via_tip_entry_servo',localRefinement=local)
            return False
        # Plan from the new belief. An unavailable motion receipt is still an
        # existing planner refusal, not evidence that a large observation is false.
        with rt._lock:
            row['beliefUpdateStarted'] = True
            install(controller,body,fit,sample)
            row['beliefUpdated'] = True
        revised = dict(params,closeRelease=True)
        if params.get('seatBody'):
            entry = {k:deepcopy(reg[k]) for k in ('entity','body','hole','point','observedAt','source')} if reg else dict(
                entity=params['on'],body=body,hole=params['seatHole'],observedAt=sample[0]['header']['stamp'],
                source='preplace_layout_rgbd',point=transform(next(r['point'] for r in rows
                    if r['bodyHole']==params['seatHole']),fit))
            revised.update(controlledSeat=True,precisionInsert=True,registeredEntry=dict(entry,
                persistent=bool(reg.get('persistent')),measurementSource='preplace_layout_rgbd'))
        receipt = rt.predict('place',revised)
        row['receiptId'] = receipt.get('id')
        if not receipt.get('withinRiskLimit'):
            row.update(reason='corrected_plan_not_admitted',issues=receipt.get('issues',[]))
            from steward.ego.skills import SkillFailed
            raise SkillFailed('observed placement target path not admitted: '+str(row['issues']))
        controller.check()
        replacement = rt._consume_receipt(receipt)
        replacement.update(precision=report,**{k:current[k] for k in ('seatBody','seatHole','seatPart') if k in current})
        row.update(adopted=True,reason=None)
        report.setdefault('corrections',[]).append(dict(source='preplace_layout_rgbd',executed=True,
                                                       receiptId=receipt.get('id')))
        deadline = getattr(ex,'_primitive_deadline',None)
        ex._begin(job,'place',receipt)
        if deadline is not None: ex._primitive_deadline = ab.limited(deadline,ex._primitive_deadline)
        emit(row)
        try: ex._place(job,replacement)
        finally: current.update(replacement)
        return True
    except (TimeoutError,OSError,ValueError,KeyError,AttributeError,TypeError) as exc:
        row['reason'] = 'relocalization_unavailable:'+type(exc).__name__
        if row['adopted'] or row.get('beliefUpdateStarted'): raise
        return False
    finally:
        emit(row)
