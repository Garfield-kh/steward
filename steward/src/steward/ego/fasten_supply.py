"""Loose fastener association from robot geometry and release history only."""
import numpy as np
from steward.ego.fasten_plan import value, tool_estimate
from steward.ego.fasten_geometry import directed_tool


def measured_fastener(entity, stamp):
    view = value(entity, 'contactView')
    if (not view or view.get('source') != 'rgbd_segment' or view.get('truncated', True)
            or view.get('invalidDepthFraction', 1) > .05
            or not 0 <= stamp-view.get('stamp', -1) <= 20):
        return None
    return directed_tool(view.get('points', []), 'toy_screw', view['stamp'])


def supply_candidate(entity, scene, attempts, frames, members=(), diagnostics=None):
    """Protect installed regions even after a mask changes category or identity.

    A previously attempted loose screw is reusable only with fresh independent
    endpoints away from every installed region. A changed class needs the same
    shape evidence; a remembered nominal pose cannot resurrect a used screw.
    """
    def reject(reason):
        if diagnostics is not None: diagnostics["reason"] = reason
        return None
    from steward.ego.fasten_identity import check, GENTLE
    if GENTLE and not supply_shape(entity, scene.stamp):
        return reject("shape_contradicted")
    identity = check(entity, scene.stamp, attempts, frames, members=members)
    if identity["status"] == "contradicted":
        return reject(identity["reason"])
    p = np.asarray(entity.pose.value['p'])
    measured = measured_fastener(entity, scene.stamp)
    for frame in frames:
        if entity.id in (frame['body'], frame['entity']):
            return reject('installed_interface_identity')
        q = np.asarray(frame['rotation']).T @ (p-np.asarray(frame['position']))
        if np.all(abs(q) <= np.asarray(frame['size'])/2+.012):
            return reject('inside_installed_interface_volume')
    for a in attempts:
        ids = {a['screw'], *a.get('screwAliases', []),
               *(h['screw'] for h in a.get('seatHistory', []))}
        if np.linalg.norm(p-np.asarray(a['entry'])) < .025:
            return reject('near_released_entry_in_3d')
        if entity.id in ids and (a.get('seatConfirmed') or a.get('verified') or measured is None):
            return reject('used_identity_without_departure_geometry')
    if entity.category.value == 'toy_screw':
        return measured or tool_estimate(entity, scene.stamp)
    return measured


def hole_progress(selected, attempts, part, body):
    rows = []
    for hole in sorted(selected):
        a = next((a for a in attempts if a['hole'] == hole), {})
        status = ('confirmed' if a.get('seatConfirmed') else
                  'unattempted' if not a.get('releaseExecuted') else
                  'failed' if a.get('seatCheck', {}).get('passed') is False else 'unknown')
        rows.append(dict(hole=hole, status=status, screw=a.get('screw'),
                         released=bool(a.get('releaseExecuted')),
                         check=a.get('seatCheck'), pairAlignment=a.get('pairAlignment')))
    unconfirmed = [r['hole'] for r in rows if r['status'] != 'confirmed']
    pending = [r['hole'] for r in rows if not r['released']]
    confirmed = sum(r['status'] == 'confirmed' for r in rows)
    result = dict(holeProgress=rows, remainingHoles=unconfirmed,
        unattemptedHoles=pending, occupiedHoles=[r['hole'] for r in rows if r['released']],
        failedHoles=[r['hole'] for r in rows if r['status'] == 'failed'],
        resumeSuggestion=dict(skill='fasten_onto', part=part, body=body,
                              remainingHoles=pending) if pending else None,
        guidance=('尚未释放孔：'+', '.join(pending)+'；仅续做这些孔，保留已放上件。'
                  if pending else f'各孔均已放置一颗（确认 {confirmed}/{len(rows)}，其余未能目视确认）；建议进入下一步。'))

    from steward.ego.fasten_landing import ENABLED, advice
    if ENABLED:
        result.update(advice(rows, attempts, part, body))
    from steward.ego.fasten_completion import REPORT, progress
    if REPORT:
        result.update(progress(rows, part, body))
    from steward.ego.hole_seating import continuable
    selected_attempts = [next((a for a in attempts if a['hole'] == h), {}) for h in selected]
    if selected_attempts and all(continuable(a) for a in selected_attempts):
        result.update(canContinue=True, interfaceStatus='released_unconfirmed' if unconfirmed else 'seated',
                      unconfirmedHoles=unconfirmed)
    return result


def supply_shape(entity, stamp):
    """Class plus shape prior; missing/partial geometry remains unknown-allowed.

    Reject an independently measured sheet, not inflated/merged collision boxes.
    Endpoint fits already enforce the catalogue axial length interval.
    """
    from steward.ego.priors import prior
    if prior(entity.category.value).get('assemblyRole') != 'fastener':
        return False
    view = value(entity, 'contactView')
    if (not view or view.get('source') != 'rgbd_segment' or view.get('truncated', True)
            or view.get('invalidDepthFraction', 1) > .05
            or not 0 <= stamp-view.get('stamp', -1) <= 20):
        return True
    points = np.asarray(view.get('points', []), float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 30 or not np.isfinite(points).all():
        return True
    _, values, axes = np.linalg.svd(points-np.median(points, axis=0), full_matrices=False)
    # A head-on view may be a flat disk: two similar planar axes are unknown.
    # A broad elongated sheet has a distinct long axis AND much less thickness.
    sheet = values[0] > 1.25*values[1] and values[1] > 4*values[2]
    if sheet:
        return False
    if values[0] >= 1.25*values[1]:
        from steward.ego.classify import SHAPES
        along = (points-np.median(points, axis=0)) @ axes[0]
        length = float(np.diff(np.quantile(along,[.01,.99]))[0])
        nominal = max(SHAPES[entity.category.value]['dims'])
        return .55 <= length/nominal <= 1.6
    # No resolved shaft in a compact view: unknown, not a size veto.
    return True


def selection_candidate(controller, entity, scene, support, detail):
    """One supply predicate, with a reason for every rejected track."""
    from steward.ego.assembly_session import current
    from steward.ego.fasten_accounting import belongs
    def reject(reason):
        detail.update(eligible=False, reason=reason)
        return None
    if value(entity, 'held', False) or entity.id == controller.rt.estimator.held_id:
        return reject('currently_held')
    session = current()
    if session is not None:
        source = session.stock_source(entity, scene)
        if source is None:
            return reject('outside_bound_source_or_unobserved_stock')
        detail['sourceSupport'] = source
    elif not belongs(controller, entity, support):
        return reject('outside_original_stock_support')
    frames = list(getattr(controller.rt.estimator, 'released_interfaces', {}).values())
    if session is not None:
        frames += [r['frame'] for r in session.state.get('initialInterfaces', {}).values()
                   if r.get('frame') and any(h.get('occupied') for h in r['holes'].values())]
    feature = supply_candidate(entity, scene, list(controller.rt.estimator.fastening_attempts.values()),
        frames,
        members=getattr(controller, '_supply_members', ()), diagnostics=detail)
    if feature is None:
        return reject(detail.get('reason', 'no_fastener_geometry'))
    track = controller.rt.estimator.tracks.get(entity.id)
    seen = track.last_seen if track is not None else (entity.pose.t or -1)
    missed = getattr(controller, '_screw_misses', {}).get(entity.id)
    if missed is not None and seen <= missed:
        return reject('miss_without_new_observation')
    for field in ('_retired_screws', '_missed_screws', '_reserved_screws'):
        if entity.id in getattr(controller, field, ()):
            return reject(field.lstrip('_'))
    if any(np.linalg.norm(np.asarray(entity.pose.value['p'])[:2]-xy) < .025
           for xy in getattr(controller, '_reserved_positions', [])):
        return reject('reserved_source_position')
    detail.update(eligible=True, reason='available_loose_stock')
    return feature
