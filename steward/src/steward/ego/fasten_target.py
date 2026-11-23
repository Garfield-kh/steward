"""Persistent interface registration and occupancy from robot observations/FK."""
from steward.ego import capabilities
from copy import deepcopy
import numpy as np

ENABLED = capabilities.get('STEWARD_ASSEMBLY_FASTEN_TARGET') != '0'
RADIAL_TOLERANCE_M = .020


def spacing(rows):
    points = [np.asarray(h['point'])[:2] for h in rows]
    return min((float(np.linalg.norm(a-b)) for i,a in enumerate(points)
                for b in points[i+1:] if np.linalg.norm(a-b) > 0), default=None)


def layout(reg):
    """Keep every measured entry, with the accepted layout's correspondence."""
    frame = reg['frame']
    from steward.ego import axial_release
    if axial_release.enabled():
        pairs = frame.get('holePairs', [])
        holes = {h['id']:h for h in frame.get('holes', [])}
        pair = next((p for p in pairs if p['bodyHole'] == reg['hole']), None)
        if pair is None or pair['partHole'] not in holes or any(p['partHole'] not in holes for p in pairs):
            # A missing semantic mapping does not invalidate the observed target.
            return [dict(bodyHole=reg['hole'], point=list(reg['point']),
                         measured=False, uncertaintyM=reg.get('uncertaintyM'),
                         reason='partial_mapping_best_estimate')]
    pair = next(p for p in frame['holePairs'] if p['bodyHole'] == reg['hole'])
    anchor = next(h for h in frame['holes'] if h['id'] == pair['partHole'])
    shift = np.asarray(reg['point'])-anchor['point']
    rows = []; used = set()
    limit = spacing(frame['holes'])
    for pair in frame['holePairs']:
        prior = next(h for h in frame['holes'] if h['id'] == pair['partHole'])
        predicted = np.asarray(prior['point'])+shift
        choices = sorted((float(np.linalg.norm(np.asarray(h['point'])-predicted)), i)
                         for i,h in enumerate(reg['holes']) if i not in used)
        if not choices or (limit and choices[0][0] >= limit/2):
            # An unseen entry remains a translated hypothesis, never a measured hole.
            rows.append(dict(prior, point=predicted.tolist(), bodyHole=pair['bodyHole'],
                             measured=False, uncertaintyM=reg['uncertaintyM']))
            continue
        _, i = choices[0]; used.add(i)
        rows.append(dict(reg['holes'][i], id=prior['id'], bodyHole=pair['bodyHole'], measured=True))
    return rows


def reuse(controller, plan):
    if not ENABLED:
        return None
    from steward.ego.support_registration import invalid_reason
    est = controller.rt.estimator
    reg = getattr(est, '_seat_registrations', {}).get(plan['part'])
    if not reg or reg['body'] != plan['body'] or invalid_reason(controller, reg):
        frame = est.released_interfaces.get(plan['part'], {})
        frame.pop('registeredSeatTargets', None)
        return None
    return deepcopy(reg)


def evidence(est, frame):
    """Independent post-release heads win; otherwise use the actual release grip."""
    from steward.ego.classify import SHAPES
    result = []
    for key,a in list(est.fastening_attempts.items()):
        if not a.get('releaseExecuted') or a.get('part') != frame['entity'] or a['body'] != frame['body']:
            continue
        departed = a.get('departureEvidence') or {}
        if departed.get('screw') == a['screw'] and departed.get('observedAt', -1) > a.get('seatedAt', 0):
            continue
        old = a.get('componentFrame')
        if old and old.get('releasedAt') != frame['releasedAt']:
            continue
        check = a.get('seatCheck') or {}
        released = getattr(est, 'fastener_releases', {}).get(a['screw'], a.get('releaseToolFeature') or {})
        head = check.get('headM') if (check.get('observedAt') or -1) > a.get('seatedAt', 0) else None
        source = 'measured_head'
        if head is None:
            head = released.get('gripM') or a.get('releaseGripM')
            source = 'release_gripper_fk'
        if head is None:
            continue
        radius = released.get('radiusM')
        track = est.tracks.get(a['screw'])
        if radius is None and track is not None:
            dims = SHAPES.get(track.category, {}).get('dims')
            radius = sorted(dims)[1]/2 if dims else None
        result.append(dict(key=key, attempt=a, point=list(head), source=source,
                           radiusM=radius or 0., uncertaintyM=check.get('uncertaintyM', released.get('uncertaintyM', 0.)) or 0.))
    return result


def select(rows, occupied, requested):
    excluded = []
    for h in rows:
        if any(np.linalg.norm(np.asarray(h['point'])[:2]-np.asarray(o['point'])[:2]) <
               o['radiusM']+h.get('radiusM', 0.)+o['uncertaintyM']+h.get('uncertaintyM', 0.) for o in occupied):
            excluded.append(h['bodyHole'])
    free = [h for h in rows if h['bodyHole'] not in excluded]
    if requested in excluded and occupied:
        selected = max(free or rows, key=lambda h:min(np.linalg.norm(np.asarray(h['point'])[:2]-np.asarray(o['point'])[:2]) for o in occupied))
    else:
        selected = next((h for h in rows if h['bodyHole'] == requested), None)
    return selected, excluded


def apply(controller, job, plan, ctx):
    if not ENABLED:
        return
    est = controller.rt.estimator
    from steward.ego import seating_geometry
    # The bounded delivery policy owns the live registered controller. Keep the
    # legacy standalone layout-association API available to its other callers.
    keep_identity = seating_geometry.enabled() and ctx.get('mode') == 'support_hole_fk_v2'
    frame = est.released_interfaces.get(plan['part'])
    if not frame:
        return
    reg = ctx.get('registration')
    source = ctx.get('targetSource', 'fresh_registration' if reg else 'prior')
    rows = deepcopy(reg.get('layout') or layout(reg)) if reg else [dict(h, bodyHole=p['bodyHole'])
        for p in frame['holePairs'] for h in frame['holes'] if h['id'] == p['partHole']]
    # Reuse the first layout; a current visible empty recess may refine one entry
    # locally, without fitting a new global translation through an occluding head.
    current_empty = []
    if reg and source == 'reused_registration':
        from steward.ego.support_holes import detect
        raw = getattr(controller.rt, '_precision_native_frames', {}).get(est.primary_camera)
        now = controller.ex._state()['stamp']
        if raw is not None and 0 <= now-raw['header']['stamp'] <= 2.:
            cam = est.camera_pose(raw['header']['odometry'], raw['camera'])
            found = detect(raw, cam, selected_point(rows, plan['hole']))
            from steward.ego.layout_pose import match
            fit, _ = match(rows, found['holes'])
            if fit:
                current_empty.extend((m['hole'],found['holes'][m['feature']]) for m in fit['matches'])
    occupied = evidence(est, frame)
    # Correct ledger ids as a batch, preserving every release on ambiguous collisions.
    changes = []
    for o in occupied:
        nearest = min(rows, key=lambda h:np.linalg.norm(np.asarray(h['point'])[:2]-np.asarray(o['point'])[:2]), default=None)
        if nearest:
            distance = np.linalg.norm(np.asarray(nearest['point'])[:2]-np.asarray(o['point'])[:2])
            limit = spacing(rows)
            if limit and distance < limit/2:
                changes.append((o, (frame['body'], nearest['bodyHole']), nearest['id']))
    destinations = [k for _,k,_ in changes]
    old_keys = {o['key'] for o,_,_ in changes}
    if not keep_identity and len(set(destinations)) == len(destinations) and all(k in old_keys or k not in est.fastening_attempts for k in destinations):
        for o,_,_ in changes: est.fastening_attempts.pop(o['key'], None)
        for o,k,part_hole in changes:
            o['attempt'].update(hole=k[1], partHole=part_hole)
            est.fastening_attempts[k] = o['attempt']
    selected, excluded = select(rows, occupied, plan['hole'])
    if keep_identity:
        selected = next((h for h in rows if h['bodyHole'] == plan['hole']), None)
    if selected is None:
        return
    # Prefer a visibly empty candidate among geometrically free entries.
    visible = [pair for pair in current_empty if pair[0] not in excluded]
    if keep_identity:
        visible = [pair for pair in current_empty if pair[0] == plan['hole']]
        if visible and raw['header']['stamp'] < max(
                selected.get('observedAt') or reg['observedAt'],
                frame.get('releasedAt') or 0, frame.get('alignmentMovedAt') or 0):
            visible = []
    if visible:
        chosen = max(visible, key=lambda pair:min((np.linalg.norm(np.asarray(pair[1]['point'])[:2]-np.asarray(o['point'])[:2])
                     for o in occupied), default=0.)) if occupied else next((pair for pair in visible if pair[0] == plan['hole']), visible[0])
        selected = next(h for h in rows if h['bodyHole'] == chosen[0])
    else:
        chosen = None
    if keep_identity and chosen:
        old_point = list(selected['point'])
        stamp = raw['header']['stamp']
        for h in rows: h.setdefault('observedAt', reg['observedAt'])
        selected.update(point=list(chosen[1]['point']), observedAt=stamp,
                        targetSource='cached_frame_upper_entry', measured=True)
        if reg.get('holes'):
            nearest = min(reg['holes'], key=lambda h:np.linalg.norm(np.asarray(h['point'])-old_point))
            separation = spacing(rows)
            if separation and np.linalg.norm(np.asarray(nearest['point'])-old_point) < separation/2:
                nearest.update(point=list(selected['point']), observedAt=stamp)
    old_hole = plan['hole']; plan['hole'] = selected['bodyHole']
    targets = {}
    for h in rows:
        upper = chosen[1]['point'] if chosen and h['bodyHole'] == chosen[0] else h['point']
        point = list(upper); lower = frame.get('bodyHolePoints', {}).get(h['bodyHole']) if reg else None
        separation = None
        if lower is None and reg:
            from steward.ego.fasten_plan import hole_point
            from steward.ego.esim import PredictionError
            body = controller.rt.ego_state().find(frame['body'])
            if body is not None:
                try:
                    lower = hole_point(controller.rt.esim, body, h['bodyHole'])
                except PredictionError:
                    pass
        if lower is not None:
            separation = float(np.linalg.norm(np.asarray(point)[:2]-np.asarray(lower)[:2]))
            if not keep_identity:
                point[:2] = ((np.asarray(point)[:2]+np.asarray(lower)[:2])/2).tolist()
        targets[h['bodyHole']] = dict(point=point, upperPoint=list(upper if keep_identity else h['point']),
            targetSource=h.get('targetSource', source) if keep_identity else source,
            pairDistanceM=separation, observedAt=(h.get('observedAt') if keep_identity and h.get('observedAt') is not None
                else reg['observedAt'] if reg else frame['observedAt']))
    frame['registeredSeatTargets'] = targets
    if reg:
        reg.update(layout=rows, persistent=True, hole=plan['hole'], point=list(selected['point']),
                   executionPoint=targets[plan['hole']]['point'])
        if keep_identity:
            reg['observedAt'] = targets[plan['hole']]['observedAt']
        # Cache the upper layout separately from the midpoint used for execution.
        reg['frame'] = deepcopy({k:v for k,v in frame.items() if k != 'registeredSeatTargets'})
        if not hasattr(est, '_seat_registrations'): est._seat_registrations = {}
        est._seat_registrations[plan['part']] = reg
        controller.rt._support_hole_watch = reg
    diag = dict(targetSource=source, excludedHoles=excluded, selectedHole=plan['hole'],
                reassignedFrom=old_hole if old_hole != plan['hole'] else None,
                currentEmptyHole=bool(chosen),
                pairDistanceM=targets[plan['hole']]['pairDistanceM'])
    if diag['pairDistanceM'] is not None and diag['pairDistanceM'] > RADIAL_TOLERANCE_M:
        diag['warning'] = 'upper_component_misaligned'
    ctx['report']['seatDelivery'] = diag
    job.postcondition['seatDelivery'] = diag
    est.version += 1


def selected_point(rows, hole):
    return next((h['point'] for h in rows if h['bodyHole'] == hole), rows[0]['point'])
