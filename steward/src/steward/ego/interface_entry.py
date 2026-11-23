"""Current face/support evidence before fastening; missing vision never blocks.

Catalogue geometry supplies correspondence only. Raw independent RGB-D supplies
face signatures, footprints and bearing heights; release FK is not an observation.
"""
from copy import deepcopy
import uuid
import numpy as np
from steward.ego import assembly_precision as ap
from steward.ego.fasten_geometry import vertical_signature, surface_patch
from steward.ego.fasten_plan import value, interface_holes
from steward.ego.fasten_release import released_interface
from steward.ego.skills import SkillFailed
from steward.math3d import matrix_from_quat


def unknown(reason):
    return dict(measured=False, matches=None, reason=reason)


def face(controller, body, holes, after):
    from steward.ego import session_observation
    observed = session_observation.face(controller, body, holes)
    if observed is not None:
        return observed
    scene = controller.rt.ego_state(); ent = scene.find(body)
    view = value(ent, 'contactView')
    if not ap.close(view, scene.stamp, after):
        return unknown('no_current_body_view')
    record = controller.rt.estimator.stationary_assemblies.records.get(body, {})
    ref = record.get('reference', {})
    # Compare with a registered reference image, never with commanded upZ.
    before = ref.get('faceBefore') or value(record.get('entity'), 'contactView')
    a, b = vertical_signature(before), vertical_signature(view)
    anchor = ref.get('entryFaceReference') if not ref.get('faceBefore') else None
    if anchor:
        a = anchor['signature']
    if not a or not b:
        return unknown('face_signature_unresolved')
    rotation = np.asarray(anchor['rotation'] if anchor else ref.get('rotationBefore', ref.get('rotationM',
        matrix_from_quat(record['entity'].pose.value['q']))))
    reference_stamp = anchor['observedAt'] if anchor else before['stamp']
    if not ref.get('faceBefore') and not anchor:
        ref['entryFaceReference'] = dict(signature=a, rotation=rotation.tolist(), observedAt=reference_stamp)
    catalogue = ref.get('holes') or interface_holes(controller.rt.esim, ent)
    levels = [float(h['p'][2]) for h in catalogue]
    selected = [h for h in catalogue if h['id'] in holes]
    if not selected or not levels or max(levels) == min(levels):
        return unknown('no_opposed_face_correspondence')
    up = rotation[2, 2] * (1 if a == b else -1)
    mid = (max(levels)+min(levels))/2
    facing = [(h['p'][2]-mid)*up for h in selected]
    if not all(x > 0 for x in facing) and not all(x < 0 for x in facing):
        return unknown('selected_face_ambiguous')
    return dict(measured=True, matches=all(x > 0 for x in facing),
        source='current_rgbd_signature_registered_face', observedAt=view['stamp'],
        referenceObservedAt=reference_stamp, signature=b, referenceSignature=a,
        bodyUpZEstimate=float(up), reason='current_face_compared_to_registration')


def install_face(controller, body, evidence):
    if not evidence.get('measured'): return
    from steward.math3d import quat_from_matrix
    est = controller.rt.estimator
    with controller.rt._lock:
        record = est.stationary_assemblies.records.get(body)
        if not record: return
        ref = record['reference']
        r = np.asarray(ref.get('rotationM', matrix_from_quat(record['entity'].pose.value['q'])))
        if r[2, 2]*evidence['bodyUpZEstimate'] < 0:
            r = r @ np.diag([1., -1., -1.])
        ref['rotationM'] = r.tolist()
        record['entity'].pose.value['q'] = quat_from_matrix(r)
        catalogue = ref.get('holes', [])
        if catalogue:
            mid = (max(h['p'][2] for h in catalogue)+min(h['p'][2] for h in catalogue))/2
            ref['faceResolution'] = dict(status='observed', observed=True,
                source=evidence['source'], since=evidence['observedAt'],
                exposedHoles=[h['id'] for h in catalogue if (h['p'][2]-mid)*r[2, 2] > 0])
        est.version += 1


def bearing_plane(points, xy, tolerance):
    """Fit a broad visible horizontal band, allowing an occluded centre."""
    direct = surface_patch(points, xy)
    if direct: return direct
    candidates = []
    for z in np.quantile(points[:, 2], np.linspace(0, 1, 9)):
        band = points[abs(points[:, 2]-z) <= tolerance]
        if len(band) < 30: continue
        design = np.c_[band[:, :2]-xy, np.ones(len(band))]
        if np.linalg.matrix_rank(design) < 3: continue
        fit = np.linalg.lstsq(design, band[:, 2], rcond=None)[0]
        error = float(np.quantile(abs(design @ fit-band[:, 2]), .95))
        if error > tolerance or np.linalg.norm(fit[:2]) > tolerance/max(np.ptp(band[:, :2], axis=0)):
            continue
        candidates.append(dict(point=[*xy, float(fit[2])], residualM=error, points=len(band)))
    return max(candidates, key=lambda p:p['points']) if candidates else None


def support(controller, part, body, after):
    scene = controller.rt.ego_state(); frame = released_interface(scene, part, body)
    if not frame: return unknown('no_released_interface')
    after = max(after, frame['releasedAt'], frame.get('transportedAt', frame['releasedAt']))
    pv, bv = [value(scene.find(e), 'contactView') for e in (part, body)]
    if not all(ap.close(v, scene.stamp, after) for v in (pv, bv)):
        return unknown('no_current_independent_layers')
    if pv['stamp'] != bv['stamp'] or np.array_equal(pv['points'], bv['points']):
        return unknown('layers_not_independent_same_frame')
    points, base = [np.asarray(v['points'], float) for v in (pv, bv)]
    if any(p.ndim != 2 or p.shape[1] != 3 or len(p) < 30 or not np.isfinite(p).all() for p in (points, base)):
        return unknown('insufficient_layer_points')
    from steward.ego.planner import Box2D, rectangle_separation
    ref = controller.rt.estimator.stationary_assemblies.records.get(body, {}).get('reference', {})
    for v, expected in ((pv, frame['size']), (bv, ref.get('size', bv.get('referenceSizeM')))):
        size = np.asarray(v.get('visibleSizeM', []))
        if expected is None or size.shape != (3,) or np.any(
                abs(np.sort(size[:2])-np.sort(expected[:2])) > ap.tolerance(controller)*2):
            return unknown('partial_or_merged_layer_outline')
    margin = ap.tolerance(controller)+sum(float(v.get('uncertaintyM', 0)) for v in (pv, bv))
    boxes = [Box2D(e, *v['centreM'][:2], v.get('boxYawRad', 0), *np.asarray(v['visibleSizeM'])[:2]/2)
             for e,v in ((part,pv),(body,bv))]
    separation = float(rectangle_separation(*boxes))
    row = dict(measured=True, matches=False, observedAt=pv['stamp'],
        source='same_frame_independent_layers', footprintSeparationM=separation,
        uncertaintyM=margin, reason='component_outside_body')
    if separation > margin: return row
    # A local bearing plane, not the global top of protrusions. Upper thickness
    # follows its registered hole depth; it is not a measured mechanical fit.
    xy = np.asarray(pv['centreM'])[:2]
    upper = bearing_plane(points, xy, ap.tolerance(controller))
    lower = bearing_plane(base, xy, ap.tolerance(controller))
    rotation = np.asarray(frame['rotation'])
    half_height = abs(rotation[2]) @ (np.asarray(frame['size'])/2)
    depths = [float((rotation @ h['p'])[2]+half_height) for h in frame['holes']]
    if not upper or not lower or not depths:
        return dict(unknown('bearing_planes_unresolved'), observedAt=pv['stamp'],
                    footprintSeparationM=separation)
    gap = float(upper['point'][2]-lower['point'][2]-max(depths))
    row.update(matches=abs(gap) <= margin, bearingGapM=gap,
        thicknessSource='registered_hole_to_bearing_extent',
        reason='bearing_stack_consistent' if abs(gap) <= margin else 'component_not_on_body_face')
    return row


def publish(controller, job, report):
    if getattr(getattr(controller, 'rt', None), '_memory_checkpoint', None) is not None:
        from steward.ego.memory_checkpoint import compact_entry_report
        bounded = compact_entry_report(report)
        if bounded is not report:
            report.update(bounded)
    totals = report.get('observationArchive', {})
    compact = {k:deepcopy(v) for k,v in report.items() if k in
        ('schema', 'id', 'body', 'holes', 'corrections', 'reason')}
    compact['observationCount'] = totals.get('count', 0) + len(report['observations'])
    compact['observationCounts'] = {kind:dict(
        measured=totals.get(kind, {}).get('measured', 0) + sum(bool(o[kind]['measured']) for o in report['observations']),
        mismatch=totals.get(kind, {}).get('mismatch', 0) + sum(o[kind]['matches'] is False for o in report['observations']))
        for kind in ('orientation', 'support')}
    compact['observationCounts'] = {k:v for k,v in compact['observationCounts'].items()
                                    if v['measured'] or v['mismatch']}
    compact['corrections'] = {k:v for k,v in compact['corrections'].items() if v}
    # The schema defaults absent counters to zero; missing coverage is a
    # missing report, not an empty counter map. Keep the explicit unknown reason.
    for field in ('observationCounts', 'corrections'):
        if not compact[field]: compact.pop(field)
    records = job.postcondition.setdefault('interfaceEntryChecks', [])
    records[:] = [r for r in records if r['id'] != report['id']]
    records.append(compact)
    from steward.ego.assembly_session import current
    session = current()
    if session:
        root = job.postcondition
        try: session.emit('interface_entry_check', entryCheck=deepcopy(report))
        finally: job.postcondition = root


def check(controller, job, part, body, holes, *, placed=False):
    """Use current evidence without waiting; one correction of each kind."""
    controller.check()
    est = controller.rt.estimator
    ids = [h['id'] if isinstance(h, dict) else h for h in holes]
    report = dict(schema='interface_entry.v1', id=uuid.uuid4().hex, part=part, body=body,
        holes=ids, phase='before_fasteners' if placed else 'enter', observations=[],
        viewChanges=0, corrections=dict(orientation=0, support=0), confirmed=False)
    budgets = getattr(est, '_interface_entry_budgets', {})
    est._interface_entry_budgets = budgets
    records = getattr(getattr(est, 'stationary_assemblies', None), 'records', {})
    aliases = records.get(body, {}).get('aliases', [])
    key = next((k for k in budgets if k[1] == tuple(sorted(ids)) and k[0] in [body, *aliases]),
               (body, tuple(sorted(ids))))
    budget = budgets.setdefault(key, set())
    reports = getattr(est, '_interface_entry_reports', {})
    est._interface_entry_reports = reports
    previous = reports.get(key)
    if previous:
        report.update(id=previous['id'], observations=deepcopy(previous['observations']),
                      corrections=deepcopy(previous['corrections']), viewChanges=previous['viewChanges'])
    if previous and previous.get('observationArchive'):
        report['observationArchive'] = deepcopy(previous['observationArchive'])
    reports[key] = report
    def observe(*, refresh=False):
        try:
            sources = {eid:deepcopy(est.tracks[eid]) for eid in (body, part) if eid in est.tracks}
            after = ap.acquire(controller, [body, part]) if refresh else None
            # A scoped observation can retire an unmatched ordinary track. Keep
            # its last source as an unconfirmed prior, never a fresh measurement.
            with controller.rt._lock:
                for eid, track in sources.items():
                    if eid not in est.tracks or est.tracks[eid].category != track.category:
                        est.tracks[eid] = track
            if after is None: after = controller.rt.ego_state().stamp-2
            orientation = face(controller, body, ids, after)
            install_face(controller, body, orientation)
            return dict(orientation=orientation,
                support=support(controller, part, body, after) if placed else unknown('component_not_placed_yet'))
        except (AttributeError, KeyError, TypeError, ValueError, SkillFailed) as exc:
            controller.check()
            return {k:unknown('observation_unavailable_'+type(exc).__name__) for k in ('orientation','support')}
    row = observe(); report['observations'].append(row)
    required = ('orientation', 'support') if placed else ('orientation',)
    occupied = any(a.get('releaseExecuted') and a.get('body') == body and a.get('hole') in ids
                   for a in est.fastening_attempts.values())
    from steward.ego.session_initial import occupied as initially_occupied
    occupied |= any(initially_occupied(body, h) for h in ids)
    # Missing face evidence uses the best registered estimate, explicitly unconfirmed.
    from steward.ego.assembly_status import interface_access
    try:
        access = interface_access(controller.rt.esim, controller.rt.ego_state().find(body), set(ids))
    except (AttributeError, KeyError, TypeError, ValueError):
        access = {}
    turn = row['orientation']['matches'] is False or (
        not row['orientation']['measured'] and access.get('status') == 'inaccessible')
    from steward.ego import edge_retention
    retained = edge_retention.retained_record(records, body) if edge_retention.enabled() else None
    replay = placed and row['support']['matches'] is False
    from steward.ego.session_observation import preserve_component
    if preserve_component(body, ids, part):
        # A visible in-place upper layer is a continuation even with zero heads.
        # Only an observed opposite face permits a turn; late outcome repair
        # handles fresh displacement after completing the pending holes.
        turn = access.get('status') == 'inaccessible'
        replay = False
    if retained:
        turn = False
        report.update(bodyRetained=True, faceUnconfirmed=row['orientation']['matches'] is not True)
    from steward.ego import working_face
    scene = controller.rt.ego_state()
    current_face = (working_face.prepare(controller, body, ids) if not placed or
        working_face.edge_context(controller.rt.esim, scene, scene.find(body)) else None)
    if current_face is not None and (current_face['faceObserved'] or not row['orientation']['measured']):
        turn = current_face['status'] == 'inaccessible' and current_face['faceObserved']
        report.update(currentFace=current_face, bodyRetained=not turn,
                      faceUnconfirmed=not current_face['faceObserved'])
    if not occupied and not est.held_id:
        if turn and 'orientation' not in budget:
            budget.add('orientation')
            publish(controller, job, report)
            from steward.ego.assembly_session import current
            session = current()
            result = {}
            try:
                flip_params = dict(body=body)
                if working_face.enabled():
                    flip_params['hole'] = ids
                if session:
                    interface_key = session.state.get('interface')
                    if interface_key not in session.state['flipped']:
                        session.state['flipped'].append(interface_key)
                    result = session.action('flip_assembly', flip_params)
                else:
                    result = controller.step(job, 'flip_assembly', flip_params)
                    executed = (job.postcondition.get('stages') or [{}])[-1].get('result') or {}
                    result = dict(result, **executed)
            except SkillFailed as exc:
                controller.check()
                result = (session.state.get('lastResult') or {}) if session else (                    job.postcondition.get('stages') or [{}])[-1].get('result', {})
                report.setdefault('actionFailures', []).append(str(exc))
                if est.held_id or not result.get('releaseExecuted'):
                    publish(controller, job, report)
                    raise
            report['corrections']['orientation'] = int(bool(result.get('releaseExecuted')))
            for attr in ('_registered_alignment', '_component_spec', '_component_native_frame'):
                if hasattr(controller, attr): delattr(controller, attr)
            row = observe(refresh=True); report['observations'].append(row)
            replay = placed  # A loose upper layer cannot inherit a new face by assumption.
        if replay and 'support' not in budget:
            budget.add('support')
            frame = released_interface(controller.rt.ego_state(), part, body)
            # Install the fresh independent component centre for ordinary pickup.
            scene = controller.rt.ego_state(); view = value(scene.find(part), 'contactView')
            if ap.close(view, scene.stamp, (frame or {}).get('releasedAt', -1)):
                with controller.rt._lock:
                    track = est.tracks.get(part)
                    if track is not None: track.pos = np.asarray(view['centreM'])
            with controller.rt._lock:
                record = est.stationary_assemblies.records.get(body)
                if record: record['reference'].pop('componentHoleRegistration', None)
                getattr(est, '_seat_registrations', {}).pop(part, None)
                controller._registered_alignment = {}
            released_before = (frame or {}).get('releasedAt', float('-inf'))
            publish(controller, job, report)
            try:
                controller.step(job, 'pick', dict(target=part, stayNear=True))
                result = controller.step(job, 'place', dict(on=body, alignHoles=True, hole=ids,
                    heldHole=[p['partHole'] for p in (frame or {}).get('holePairs', [])],
                    stayNear=True, closeRelease=True))
            except SkillFailed as exc:
                controller.check()
                report.setdefault('actionFailures', []).append(str(exc))
                latest = est.released_interfaces.get(part, {})
                if est.held_id or latest.get('releasedAt', released_before) <= released_before:
                    publish(controller, job, report)
                    raise
                result = dict(releaseExecuted=True)
            report['corrections']['support'] = int(bool(result.get('releaseExecuted') or 'releasedAt' in result))
            row = observe(refresh=True); report['observations'].append(row)
    report['confirmed'] = all(row[k]['matches'] is True for k in required)
    report['reason'] = 'current_state_confirmed' if report['confirmed'] else 'unconfirmed_continue'
    publish(controller, job, report)
    return report
