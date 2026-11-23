"""Reconstruct pre-existing work from current robot evidence, never world state.

Occupancy is separate from the execution ledger. Missing views do not gate work;
uncertain existing structures are preserved with explicit evidence provenance.
A later direct empty-hole observation revokes an occupancy estimate.
"""
from copy import deepcopy
import numpy as np
from steward.ego.fasten_plan import value, interface_holes
from steward.math3d import matrix_from_quat
from steward.ego.esim import PredictionError
from steward.ego.skills import SkillFailed
from steward.net import HttpError


def component(controller, item):
    from steward.ego import session_observation as current
    if current.enabled():
        measured = current.component(controller, item, list(samples(controller)), interface_holes)
        if measured is not None:
            return measured
    from steward.ego.planner import entity_box, rectangle_separation, Box2D
    from steward.ego.assembly_precision import tolerance
    scene = controller.rt.ego_state()
    entities = [scene.find(item[k]) for k in ('part', 'body')]
    layers = []
    for ent in entities:
        if ent is None:
            return dict(onBody=None, source='unavailable', reason='member_unseen'), None
        view = value(ent, 'contactView') or {}
        if view.get('source') == 'rgbd_segment' and 0 <= scene.stamp-view.get('stamp', -1) <= 2:
            p = np.asarray(view['centreM']); size = np.asarray(view['visibleSizeM'])
            box = Box2D(ent.id, *p[:2], view.get('boxYawRad', 0), *size[:2]/2)
            low, high = p[2]-size[2]/2, p[2]+size[2]/2
            source = 'current_rgbd_layers'
            stamp = view['stamp']
        elif ent.pose.src in ('observed', 'fused') and ent.pose.t is not None and 0 <= scene.stamp-ent.pose.t <= 20:
            box, low, high = entity_box(ent)
            size = np.array([2*box.hx, 2*box.hy, high-low])
            p = np.asarray(ent.pose.value['p']); stamp = ent.pose.t
            source = 'current_perceived_boxes'
        else:
            return dict(onBody=None, source='unavailable', reason='no_current_layer_pose'), None
        layers.append((box, low, high, p, size, stamp, source))
    upper, lower = layers
    margin = tolerance(controller)
    overlap = rectangle_separation(upper[0], lower[0]) <= margin
    on_body = bool(overlap and (
        upper[3][2] > lower[3][2] and abs(upper[1]-lower[2]) <= margin or
        upper[3][2] < lower[3][2] and abs(upper[2]-lower[1]) <= margin))
    report = dict(onBody=on_body, source=upper[6], observedAt=min(upper[5], lower[5]),
                  reason='stacked_layers' if on_body else 'separate_layers', confirmed=False)
    if not on_body:
        return report, None
    part, body = entities
    rotation = matrix_from_quat(part.pose.value['q'])
    from steward.ego.fasten_release import grasp_frame
    geometry = grasp_frame(part, upper[3], np.eye(3)) if getattr(part, 'rigid', None) else None
    # Use the same catalogue-root-to-centre conversion as ordinary placement.
    catalogue = geometry['holes'] if geometry else interface_holes(controller.rt.esim, part)
    holes = [deepcopy(h) for h in catalogue if h['id'] in item['heldHoles']]
    for h in holes:
        h['point'] = (upper[3]+rotation @ h['p']).tolist()
    frame = dict(entity=part.id, body=body.id, category=part.category.value,
        position=upper[3].tolist(), size=upper[4].tolist(), rotation=rotation.tolist(), holes=holes,
        holePairs=[dict(partHole=p, bodyHole=b) for p,b in zip(item['heldHoles'], item['holes'])],
        observedAt=report['observedAt'], releasedAt=report['observedAt'],
        source='session_initial_perceived_stack', heightSource='current_perceived_stack',
        observed=False, preExisting=True)
    return report, frame


def samples(controller):
    """Use cached current native frames; no waiting or navigation at entry."""
    from steward.ego.inhand import camera_pose
    rt = controller.rt; now = rt.ego_state().stamp
    from steward.ego import session_observation as current
    if not current.enabled():
        frames = {**getattr(rt, '_precision_native_frames', {}), **getattr(rt, '_latest_native_frames', {})}
        for camera, raw in frames.items():
            if 0 <= now-raw['header']['stamp'] <= 2:
                yield raw, camera_pose(rt.estimator, raw['header'], camera)
        return
    frames = {}
    for cache in (getattr(rt, '_precision_native_frames', {}), getattr(rt, '_latest_native_frames', {})):
        for camera, raw in cache.items():
            if 'header' in raw and raw['header'].get('stamp', -1) >= frames.get(camera, {}).get('header', {}).get('stamp', -1):
                frames[camera] = raw
    for camera, raw in sorted(frames.items(), key=lambda pair:pair[1]['header']['stamp']):
        if 0 <= now-raw['header']['stamp'] <= 2:
            try:
                pose = camera_pose(rt.estimator, raw['header'], camera)
            except (AttributeError, KeyError, TypeError, ValueError):
                continue  # An unavailable camera must not discard another current view.
            yield raw, pose


def observe(controller, item):
    """The hole_seating detector identifies heads/empty mouths without a prior release."""
    from steward.ego.hole_seating import measure
    report = dict(component=dict(onBody=None, source='unavailable'), frame=None,
                  holes={h:dict(occupied=None, confirmed=False, source='unavailable') for h in item['holes']})
    try:
        from steward.ego import session_faces
        context = getattr(controller, '_session_face_context', {}) if session_faces.enabled() else {}
        if session_faces.enabled() and item.get('key') is not None and item['key'] not in context:
            from steward.ego.assembly_session import current as active_session
            session = active_session()
            peers = ([dict(i, part=session.canonical(i['part']), body=session.canonical(i['body']))
                      for i in session.state.get('goal', {}).get('interfaces', [])]
                     if session is not None else [item])
            context = session_faces.prepare(controller, peers, component, list(samples(controller)))
        if item.get('key') in context:
            report.update(deepcopy(context[item['key']]))
        else:
            report['component'], report['frame'] = component(controller, item)
        from steward.ego import session_observation as current
        if current.enabled():
            report['observationPolicy'] = 'current_rgbd'
            # Never identify an opposite face through the body, or mistake a
            # nearby loose head for completion of a visibly separate component.
            if (report['component'].get('faceExposed') is False or
                    report['component'].get('onBody') is False):
                return report
        # An unresolved or hidden catalogue face cannot borrow visible entries.
        # This is evidence filtering, not a prerequisite for executing work.
        if session_faces.enabled() and report.get('holeMapping', {}).get('faceExposed') is not True and report.get('holeMapping'):
            return report
        frame = report['frame']
        if frame is not None:
            layout = [dict(h, bodyHole=p['bodyHole']) for p in frame['holePairs']
                      for h in frame['holes'] if h['id'] == p['partHole']]
        else:
            # An upper mask can be merged or bound to a loose duplicate. Inspect
            # the selected body interface even without a separate upper layer.
            from steward.ego.assembly_status import interface_access
            body = controller.rt.ego_state().find(item['body'])
            if body is None or interface_access(controller.rt.esim, body, set(item['holes']))['status'] == 'inaccessible':
                return report
            rotation = matrix_from_quat(body.pose.value['q'])
            if session_faces.enabled() and report.get('face', {}).get('bodyUpZEstimate') is not None:
                if rotation[2, 2]*report['face']['bodyUpZEstimate'] < 0:
                    rotation = rotation @ np.diag([1., -1., -1.])
            layout = [dict(h, bodyHole=h['id'], point=(np.asarray(body.pose.value['p'])+rotation @ h['p']).tolist())
                      for h in interface_holes(controller.rt.esim, body) if h['id'] in item['holes']]
        opening = controller.rt.esim.arm.geometry.jaw_opening
        for sample in samples(controller):
            for row in layout:
                check = measure(controller, sample, row['point'], opening,
                                dict(layout=layout, targetHole=row['bodyHole']))
                feature = check.get('targetFeature') or {}
                occupied = (False if check.get('landingState') == 'empty_hole' else
                            True if feature.get('observedKind') == 'head' else None)
                if occupied is not None:
                    report['holes'][row['bodyHole']] = dict(occupied=occupied,
                        confirmed=check.get('passed') is True or current.enabled() and occupied is False,
                        source='current_hole_rgbd', check=check,
                        **(dict(reason=check.get('reason', 'direct_entry_observation')) if current.enabled() else {}))
                    if report.get('holeMapping'):
                        mapped = report['holeMapping']['acceptedVisibleHoles']
                        if row['bodyHole'] not in mapped:
                            mapped.append(row['bodyHole'])
                        report['holeMapping']['observedEntries'].append(dict(
                            hole=row['bodyHole'], observedKind=feature.get('observedKind'),
                            observedAt=check.get('observedAt'), pixel=deepcopy(feature.get('pixel')),
                            point=deepcopy(feature.get('point'))))
    except (AttributeError, KeyError, TypeError, ValueError, np.linalg.LinAlgError, PredictionError, SkillFailed) as exc:
        report['reason'] = 'observation_unavailable_'+type(exc).__name__
    return report


def inventory(session):
    """Count currently observed bound-kit stock, not missing-track history."""
    from steward.ego.planner import entity_box
    from steward.ego.assembly_precision import tolerance
    scene = session.rt.ego_state(); available = []; classified = []; excluded = []
    for eid, binding in session.state['goal']['materials'].items():
        ent = scene.find(session.canonical(eid))
        if ent is None or value(ent, 'held', False):
            continue
        pose = ent.pose
        if pose.src not in ('observed', 'fused') or pose.t is None or not 0 <= scene.stamp-pose.t <= 20:
            continue
        source = scene.find(session.canonical(binding['source']))
        try:
            _, bottom, _ = entity_box(ent)
            _, _, top = entity_box(source)
            # Docking anchors recurse through assemblies to the table; they do
            # not prove that a visible head is loose stock on that table.
            loose = abs(bottom-top) <= tolerance(session.controller)
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
        classified.append(ent.id)
        (available if loose else excluded).append(ent.id)
    return dict(remaining=len(available) if classified and len(classified) == len(session.state['goal']['materials']) else None,
                entities=available, excludedAboveSupply=excluded, source='current_bound_kit_stock')


def reconcile(reports, stock, released):
    """Allocate no ambiguous partial deficit to arbitrary hole identities."""
    from steward.ego import session_faces
    known = sum(h.get('occupied') is True and (key,hole) not in released
                for key,r in reports.items() for hole,h in r['holes'].items())
    total = sum(len(r['holes']) for r in reports.values() if not session_faces.protected(r))
    candidates = [(key, hole) for key,r in reports.items() if r['component'].get('onBody') is True and not session_faces.protected(r)
                  for hole,h in r['holes'].items() if h.get('occupied') is None and (key,hole) not in released]
    deficit = total-len(released)-known-stock['remaining'] if stock['remaining'] is not None else 0
    ownership_resolved = not session_faces.enabled() or all(
        not r.get('face', {}).get('resolvedBeforeHoles') or
        r['face'].get('exposed') is True or session_faces.protected(r) for r in reports.values())
    # An unexplained hidden face cannot spend the stock deficit on the visible
    # interface. That would merely recreate the cross-face attribution error.
    from steward.ego import first_interface
    if candidates and deficit == len(candidates) and ownership_resolved and not first_interface.enabled():
        for key,hole in candidates:
            reports[key]['holes'][hole] = dict(occupied=True, confirmed=False,
                source='stacked_component_and_stock_deficit', reason='best_estimate_unconfirmed')
    for key, report in reports.items():
        rows = report['holes']
        occupied = sum(h.get('occupied') is True or (key,hole) in released for hole,h in rows.items())
        from steward.ego import session_observation as current
        in_place = current.enabled() and current.actionable_stack(report)
        report['status'] = ('existing_unconfirmed' if session_faces.protected(report) else
                            'completed' if occupied == len(rows) else 'partial' if occupied else
                            'ready_to_fasten' if in_place else
                            'unstarted' if report['component'].get('onBody') is False or
                            all(h.get('occupied') is False for h in rows.values()) else 'unknown')
        report['sources'] = sorted({report['component']['source'], *(h['source'] for h in rows.values()),
                                   *(['existing_release_ledger'] if any((key,h) in released for h in rows) else [])})
        report['default'] = ('preserve_existing_structure_unconfirmed' if session_faces.protected(report)
                             else 'unknown_holes_pending_unconfirmed')
    return reports


def initialize(session):
    from steward.ego import session_faces
    items = session.state.get('goal', {}).get('interfaces', [])
    reports = observe_interfaces(session, items)
    stock = inventory(session)
    released = {(i['key'], h) for i in items for h,a in zip(i['holes'],session.rows(i)) if a.get('releaseExecuted')}
    # ASSEMBLY_CURRENT_ENTRY uses a stock deficit to request a view. The current policy also
    # inspects unresolved stacks when the bound inventory cannot be counted.
    pending = sum((i['key'], h) not in released and reports[i['key']]['holes'][h].get('occupied') is not True
                  for i in items for h in i['holes'])
    unknown = [i for i in items if any(h.get('occupied') is None for h in reports[i['key']]['holes'].values())]
    native = getattr(session.rt, '_latest_native_frames', {}) or getattr(session.rt, '_precision_native_frames', {})
    from steward.ego import session_observation as current
    need_view = stock['remaining'] is not None and stock['remaining'] < pending
    if current.enabled():
        unknown = [i for i in items if not session_faces.hidden_stack(reports[i['key']]) and (
                   reports[i['key']]['component'].get('onBody') is None or
                   any(h.get('occupied') is None for h in reports[i['key']]['holes'].values()))]
        need_view |= any(not session_faces.hidden_stack(reports[i['key']]) and (
            reports[i['key']]['component'].get('onBody') is None or
            (reports[i['key']]['component'].get('onBody') is True and
             any(h.get('occupied') is None for h in reports[i['key']]['holes'].values()))) for i in items)
    sensor = native or (current.enabled() and getattr(session.rt, '_provided_camera_ids', ()))
    if (sensor and unknown and need_view and not session.rt.estimator.held_id and
            (not session_faces.enabled() or not session.state.get('initialViewChanges'))):
        item = unknown[0]
        priors = {eid:deepcopy(track) for eid,track in getattr(session.rt.estimator, 'tracks', {}).items()
                  if current.enabled() and eid in {session.canonical(i[k]) for i in items for k in ('part','body')}}
        try:
            session.action('_assembly_observe', dict(body=session.canonical(item['body']),
                part=session.canonical(item['part']), changeView=True))
        except (SkillFailed, AttributeError, KeyError, TypeError, ValueError) as exc:
            session.emit('initial_view_unconfirmed', reason=str(exc))
        except HttpError as exc:
            if not current.enabled(): raise
            session.emit('initial_view_unconfirmed', reason=str(exc))
        if current.enabled():
            session.state['initialViewChanges'] = 1
            # An observation-only trip does not prove a missing member moved.
            # Restore its last source as a prior, retaining its old timestamp.
            with session.rt._lock:
                for eid,track in priors.items():
                    if eid not in session.rt.estimator.tracks:
                        session.rt.estimator.tracks[eid] = track
        reports = observe_interfaces(session, items)
        refreshed = inventory(session)
        if refreshed['remaining'] is not None:
            stock = refreshed
        else:
            stock = dict(stock, retainedAcrossObservationOnly=True)
    if current.enabled():
        for item in items:
            report = reports[item['key']]
            if sensor:
                try:
                    current.estimate(session.controller, dict(item, part=session.canonical(item['part']),
                        body=session.canonical(item['body'])), report)
                except (AttributeError, KeyError, TypeError, ValueError):
                    report['estimateReason'] = 'bound_member_geometry_unavailable'
        try:
            current.resolve_faces(session.controller, [dict(i, body=session.canonical(i['body'])) for i in items], reports)
        except (AttributeError, KeyError, TypeError, ValueError):
            pass  # No independent face vote: preserve each local estimate.
        for report in reports.values():
            current.finish(report)
    session.state['initialInterfaces'] = reconcile(reports, stock, released)
    if current.enabled():
        session.state['initialInterfaceSnapshot'] = current.summary(session, snapshot=False)
    for item in items:
        row = reports[item['key']]
        session.emit('interface_initial_state', interface=item['key'], body=session.canonical(item['body']),
                     part=session.canonical(item['part']),
                     status=row['status'], sources=row['sources'], component=row['component'],
                     holes={h:{k:e[k] for k in ('occupied', 'confirmed', 'source', 'reason') if k in e}
                            for h,e in row['holes'].items()},
                     inventory=stock, default=row['default'],
                     **({k:deepcopy(row[k]) for k in ('face', 'holeMapping') if k in row}
                        if session_faces.enabled() else {}))


def observe_interfaces(session, items):
    from steward.ego import session_faces
    from steward.ego import session_inventory
    context = session_inventory.prepare(session, items)
    canonical = [dict(i, part=session.canonical(i['part']), body=session.canonical(i['body'])) for i in items]
    previous = getattr(session.controller, '_session_inventory_context', None)
    try:
        session.controller._session_inventory_context = context
        return session_faces.observe_all(session.controller, canonical, observe, component, samples)
    finally:
        if previous is None:
            session.controller.__dict__.pop('_session_inventory_context', None)
        else:
            session.controller._session_inventory_context = previous


def refresh(session):
    """Re-entry consumes current evidence without spending another view change.

    Preserve unknown occupancy estimates until contradicted. Direct empty
    evidence can revoke old initial occupancy; the release ledger stays factual.
    """
    from steward.ego import session_observation as current
    from steward.ego import session_faces
    if not current.enabled(): return
    items = session.state.get('goal', {}).get('interfaces', [])
    from steward.ego import axial_release
    if axial_release.enabled():
        items = [i for i in items if not all(axial_release.checked(a) for a in session.rows(i))]
        if not items: return
    def observe_all():
        return observe_interfaces(session, items)
    reports = observe_all()
    unresolved = [i for i in items if not session_faces.hidden_stack(reports[i['key']]) and
        reports[i['key']].get('observationPolicy') == 'current_rgbd' and (
        reports[i['key']]['component'].get('onBody') is None or
        reports[i['key']]['component'].get('faceExposed') is not False and
        any(h.get('occupied') is None for h in reports[i['key']]['holes'].values()))]
    sensor = (getattr(session.rt, '_latest_native_frames', {}) or
              getattr(session.rt, '_precision_native_frames', {}) or getattr(session.rt, '_provided_camera_ids', ()))
    if unresolved and sensor and not session.rt.estimator.held_id:
        item = unresolved[0]
        params = dict(body=session.canonical(item['body']), part=session.canonical(item['part']))
        if not session.state.get('initialViewChanges'):
            params['changeView'] = True
            session.state['initialViewChanges'] = 1
        try:
            session.action('_assembly_observe', params)
        except (SkillFailed, AttributeError, KeyError, TypeError, ValueError, HttpError) as exc:
            session.emit('initial_view_unconfirmed', reason=str(exc))
        reports = observe_all()
    try:
        current.resolve_faces(session.controller, [dict(i,body=session.canonical(i['body'])) for i in items], reports)
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    for item in items:
        before = session.state.get('initialInterfaces', {}).get(item['key'])
        if before is None: continue
        now = reports[item['key']]
        if sensor:
            try:
                current.estimate(session.controller, dict(item,part=session.canonical(item['part']),
                    body=session.canonical(item['body'])), now)
            except (AttributeError, KeyError, TypeError, ValueError):
                now['estimateReason'] = 'bound_member_geometry_unavailable'
        session_faces.retain_face(session, item, before, now)
        current.finish(now)
        if now['component'].get('onBody') is not None:
            before['component'] = now['component']
            before['frame'] = now['frame']
            before.pop('face', None)
            if now.get('face'):before['face'] = now['face']
            if session_faces.enabled():
                before.pop('preserveExisting', None)
                if now.get('preserveExisting'):before['preserveExisting'] = True
                if now.get('holeMapping'):before['holeMapping'] = now['holeMapping']
            if now['component'].get('onBody') is False and now['component'].get('source') == 'current_rgbd_stack':
                for evidence in before['holes'].values():
                    evidence.update(occupied=None, confirmed=False, source='current_rgbd_stack',
                                    reason='component_observed_separate')
                now['support'] = dict(measured=True, matches=False, source='current_rgbd_stack',
                    observedAt=now['component'].get('observedAt'), reason='component_observed_separate')
        for h, evidence in now['holes'].items():
            if evidence.get('occupied') is not None:
                before['holes'][h] = evidence
        from steward.ego import assembly_outcome
        if assembly_outcome.enabled():
            assembly_outcome.apply(session, item, now, 'reentry_current_observation')
        before['reviewed'] = False
        session.emit('interface_reentry_state', interface=item['key'], component=before['component'],
                     holes={h:{k:e[k] for k in ('occupied','confirmed','source','reason') if k in e}
                            for h,e in before['holes'].items()})
    released = {(i['key'],h) for i in items for h,a in zip(i['holes'],session.rows(i)) if a.get('releaseExecuted')}
    reconcile(session.state.get('initialInterfaces', {}), inventory(session), released)
    session.state['reentryInterfaceSnapshot'] = current.summary(session, snapshot=False)


def occupied(body, hole):
    from steward.ego.assembly_session import current
    session = current()
    return bool(session and any(session.canonical(i['body']) == session.canonical(body) and
        session.state.get('initialInterfaces', {}).get(i['key'], {}).get('holes', {}).get(hole, {}).get('occupied')
        for i in session.state.get('goal', {}).get('interfaces', [])))


def protects_stock(session, entity):
    """A currently identified occupied head is not loose material for another hole."""
    from steward.ego.planner import entity_box
    from steward.ego.assembly_precision import tolerance
    try:
        box, low, high = entity_box(entity)
        extent = np.linalg.norm([box.hx, box.hy, (high-low)/2])
        for initial in session.state.get('initialInterfaces', {}).values():
            for hole in initial.get('holes', {}).values():
                feature = (hole.get('check') or {}).get('targetFeature') or {}
                if hole.get('occupied') and feature.get('observedKind') == 'head':
                    radius = feature['radiusM']+feature.get('uncertaintyM', 0)+extent+tolerance(session.controller)
                    if np.linalg.norm(np.asarray(entity.pose.value['p'])-feature['point']) <= radius:
                        return True
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    return False


def frame(part, body):
    from steward.ego.assembly_session import current
    session = current()
    if session:
        for item in session.state.get('goal', {}).get('interfaces', []):
            if session.canonical(item['part']) == part and session.canonical(item['body']) == body:
                initial = session.state.get('initialInterfaces', {}).get(item['key'], {})
                from steward.ego import session_observation as current
                if (any(h.get('occupied') for h in initial.get('holes', {}).values()) or
                        current.enabled() and current.actionable_stack(initial)):
                    cutoff = session.state.get('initialFaceInvalidatedAt', {}).get(body) if current.enabled() else None
                    if cutoff is not None and initial.get('frame') and initial['frame'].get('observedAt', -1) <= cutoff:
                        try:
                            measured, fresh = component(session.controller, dict(item, part=part, body=body))
                        except (AttributeError, KeyError, TypeError, ValueError, np.linalg.LinAlgError):
                            measured, fresh = {}, None
                        if fresh and measured.get('observedAt', -1) > cutoff:
                            initial.update(component=measured, frame=fresh)
                            initial.pop('face', None)
                            current.finish(initial)
                            return fresh
                        return current.transported_frame(session.controller, initial['frame'])
                    return initial.get('frame')
    return None


def review(session):
    """One late observation and one repair pass; unknown preserves the estimate."""
    for item in session.state.get('goal', {}).get('interfaces', []):
        from steward.ego import axial_release
        if axial_release.enabled() and all(axial_release.checked(a) for a in session.rows(item)):
            continue
        initial = session.state.get('initialInterfaces', {}).get(item['key'], {})
        reserved = [h for h,r in initial.get('holes', {}).items() if r.get('occupied')]
        from steward.ego import session_observation as current
        preexisting = current.enabled() and current.actionable_stack(initial)
        if (not reserved and not preexisting) or initial.get('reviewed'):
            continue
        initial['reviewed'] = True
        session.state['interface'] = item['key']
        try:
            session.action('_assembly_observe', dict(body=session.canonical(item['body']),
                part=session.canonical(item['part'])))
        except SkillFailed as exc:
            session.emit('initial_review_unconfirmed', reason=str(exc))
        except (HttpError, AttributeError, KeyError, TypeError, ValueError) as exc:
            if not current.enabled(): raise
            session.emit('initial_review_unconfirmed', reason=str(exc))
        from steward.ego import session_faces
        now = (observe_interfaces(session, session.state['goal']['interfaces'])[item['key']]
               if session_faces.enabled() else observe(session.controller, dict(item,
                   part=session.canonical(item['part']), body=session.canonical(item['body']))))
        for hole in reserved:
            evidence = now['holes'][hole]
            if evidence['occupied'] is not None:
                initial['holes'][hole] = evidence
        session.emit('initial_review', body=session.canonical(item['body']),
            holes={h:{k:e[k] for k in ('occupied', 'confirmed', 'source', 'reason') if k in e}
                   for h,e in initial['holes'].items()})
        if not session.complete(item):
            session.execute_interface(item)
