"""Grasp/release interface frames. Encoder predictions are never observations."""

from steward.assembly_profile import enabled as _assembly_enabled

from copy import deepcopy
import math
import numpy as np

from steward.math3d import matrix_from_quat, quat_from_matrix


def grasp_frame(entity, grip, rotation):
    from steward.ego.classify import SHAPES
    from steward.ego.planner import entity_box
    holes = [deepcopy(a.value['screw-hole']) for a in entity.rigid.links[0].affordances
             if isinstance(a.value, dict) and 'screw-hole' in a.value
             and a.value['screw-hole'].get('frame') == 'entity']
    if not holes:
        return None
    box, low, high = entity_box(entity)
    size = [2*box.hx, 2*box.hy, high-low]
    height = SHAPES.get(entity.category.value, {}).get('dims', size)[2]
    # Outline registration only centres XY. Catalogue Z is measured from its base.
    for h in holes:
        if h.get('source') != 'observed':
            h['p'][2] -= height/2
    r = matrix_from_quat(entity.pose.value['q'])
    return dict(entity=entity.id, category=entity.category.value, size=size, holes=holes,
        offset=(rotation.T @ (np.asarray(entity.pose.value['p'])-grip)).tolist(),
        rotation=(rotation.T @ r).tolist(), observedAt=entity.pose.t,
        source='grasp_geometry_plus_release_fk')


def release_frame(grasp, grip, rotation, target, stamp):
    """Propagate the acquired component, not the command spot or a post-release mask."""
    if not grasp or not (target or {}).get('mateReference'):
        return None
    p = np.asarray(grip)+rotation @ grasp['offset']
    r = rotation @ np.asarray(grasp['rotation'])
    release_p = p.copy()
    # Gravity settling is an explicit Z-only hypothesis. XY and orientation come
    # from FK; the postcheck must still reject a fallen or tilted component.
    settled = target.get('releasedFrame', {}).get('settledCentreZM')
    if settled is not None:
        p[2] = settled
    if _assembly_enabled():
        result = dict(entity=grasp['entity'], body=target['mateReference']['entity'],
            category=grasp['category'], size=deepcopy(grasp['size']), position=p.tolist(),
            rotation=r.tolist(), releasePosition=release_p.tolist(),
            releaseGrip=list(map(float, grip)), releaseGripRotation=rotation.tolist(),
            holes=[dict(h, point=(p+r @ h['p']).tolist()) for h in grasp['holes']],
            holePairs=deepcopy(target.get('holePairs', [])), releasedAt=float(stamp),
            observedAt=grasp['observedAt'], source=grasp['source'],
            heightSource='assumed_settled_on_support' if settled is not None else 'release_fk',
            observed=False)
        from steward.ego.geometry_relocalization import provenance
        result.update(provenance(target, float(stamp)))
        if target.get('placementSurface'):
            result['placementSurface'] = deepcopy(target['placementSurface'])
            result['heightSource'] = 'settling_estimate_from_contact_surface'
        return result
    else:
        return dict(entity=grasp['entity'], body=target['mateReference']['entity'],
            category=grasp['category'], size=deepcopy(grasp['size']), position=p.tolist(),
            rotation=r.tolist(), releasePosition=release_p.tolist(),
            releaseGrip=list(map(float, grip)), releaseGripRotation=rotation.tolist(),
            holes=[dict(h, point=(p+r @ h['p']).tolist()) for h in grasp['holes']],
            holePairs=deepcopy(target.get('holePairs', [])), releasedAt=float(stamp),
            observedAt=grasp['observedAt'], source=grasp['source'],
            heightSource='assumed_settled_on_support' if settled is not None else 'release_fk',
            observed=False)


def placement_pairs(planner, held, mate, spot, yaw, params):
    """Keep the correspondence of the actual placement fit, including its half turn."""
    ph = planner._select_holes(planner._screw_holes(held), params.get('heldHole'), held.id)
    bh = planner._select_holes(planner._screw_holes(mate), params.get('hole'), mate.id)
    fits = planner._hole_fit_candidates(held, ph, mate, bh,
        use_catalog=params.get('hole') is None and params.get('heldHole') is None)
    r = matrix_from_quat(mate.pose.value['q'])
    def error(fit):
        offset, angle, _ = fit
        xy = np.asarray(mate.pose.value['p'])[:2]+r[:2,:2] @ offset
        return np.linalg.norm(xy-spot)+abs((angle-yaw+math.pi) % (2*math.pi)-math.pi)
    return [dict(partHole=a, bodyHole=b) for a, b in min(fits, key=error)[2]] if fits else []


if _assembly_enabled():
    def released_interface(es, part, body, *, allow_alias=False):
        """The body also carries the snapshot when segmentation has lost the part id."""
        from steward.ego.fasten_plan import value
        frames = [f for e in es.entities for f in value(e, 'releasedInterfaces', [])
                  if f['entity'] == part and f['body'] == body]
        if not frames and allow_alias:
            candidate = es.find(part)
            if candidate is not None:
                from steward.ego.planner import entity_box
                box, bottom, top = entity_box(candidate)
                size = np.array([2*box.hx, 2*box.hy, top-bottom])
                # A renamed mask at an already released component is a continuation,
                # never a reason to pick that component again.
                aliases = {f['entity']:f for e in es.entities for f in value(e, 'releasedInterfaces', [])
                    if f['body'] == body and (candidate.category.value == f['category'])
                    and np.linalg.norm(np.asarray(candidate.pose.value['p'])-f['position']) < .04
                    and np.allclose(np.sort(size), np.sort(f['size']), atol=.025)}
                if len(aliases) == 1:
                    frames = list(aliases.values())
        if frames:
            return max(frames, key=lambda f:f['releasedAt'])
        from steward.ego.session_initial import frame
        return frame(part, body)
else:
    def released_interface(es, part, body):
        """The body also carries the snapshot when segmentation has lost the part id."""
        from steward.ego.fasten_plan import value
        frames = [f for e in es.entities for f in value(e, 'releasedInterfaces', [])
                  if f['entity'] == part and f['body'] == body]
        return max(frames, key=lambda f:f['releasedAt']) if frames else None


def component_support(es, frame):
    """A planning proxy from acquisition geometry; never a new measured entity."""
    from steward.scene.schema import Est
    ent = deepcopy(es.find(frame['entity']) or es.entity(frame['body']))
    ent.id = frame['entity']
    ent.category = Est(value=frame['category'], src='inferred')
    ent.pose.value = dict(p=frame['position'], q=quat_from_matrix(frame['rotation']))
    ent.rigid.links[0].geometry = {'collision':[dict(type='box', size=frame['size'])]}
    ent.rigid.links[0].affordances = [Est(value={'screw-hole':h}, src='inferred') for h in frame['holes']]
    ent.state_vars = {}
    return ent


def loose_components(es, body):
    from steward.ego.fasten_plan import value
    ent = es.entity(body)
    attempts = value(ent, 'fasteningEvidence', [])
    info = value(ent, 'orientation', {})
    ledger = info.get('stationaryAssembly', {})
    parts = {p['entity']:deepcopy(p) for p in ledger.get('parts', [])}
    parts.update({f['entity']:deepcopy(f) for f in value(ent, 'releasedInterfaces', [])
                  if f['body'] == body})
    return [p for eid, p in parts.items() if not any(
        a['part'] == eid and a['body'] == body and a.get('seatConfirmed')
        for a in attempts)]


def body_grip_points(reference, parts):
    """Body-only pinches, excluding the volumes of loose components."""
    from steward.math3d import quat_from_yaw
    p = np.asarray(reference['position'], float).copy()
    p[2] = reference.get('graspHeightM', p[2])
    r = matrix_from_quat(quat_from_yaw(reference['yaw']))
    size = np.asarray(reference['size'])
    axis = int(np.argmax(size[:2]))
    result = []
    for fraction in (.25, -.25, .4, -.4, 0.):
        local = np.zeros(3); local[axis] = size[axis]*fraction
        point = p+r @ local
        overlaps = False
        for part in parts:
            pr = np.asarray(part.get('rotation', matrix_from_quat(quat_from_yaw(part.get('yaw', 0.)))))
            delta = pr.T @ (point-np.asarray(part['position']))
            if np.all(abs(delta) <= np.asarray(part['size'])/2+.006):
                overlaps = True
                break
        if not overlaps:
            result.append((point[:2].tolist(), (point-p)[:2].tolist()))
    return result


def seat_check(entity, frame, entry, after, stamp):
    """Cheap independent RGB-D posterior, including a shaft-hidden head observation."""
    from steward.ego.fasten_plan import value
    from steward.ego.fasten_geometry import screw_head
    feature = value(entity, 'toolFeature')
    if _assembly_enabled():
        from steward.ego.fasten_supply import measured_fastener
        from steward.ego.release_observation import ENABLED
        if ENABLED and entity is not None:
            feature = measured_fastener(entity, stamp) or feature
    view = value(entity, 'contactView')
    valid = lambda t: t > after and 0 <= stamp-t <= 20
    endpoint = bool(feature and feature.get('source') == 'rgbd_endpoint_width'
                    and valid(feature.get('stamp', -1)))
    head = (dict(head=feature['head'], stamp=feature['stamp'], source=feature['source'])
            if endpoint else screw_head(view, dict(stamp=after)) if view and valid(view['stamp']) else None)
    out = dict(status='unknown', passed=None, reason='no_fresh_separate_screw',
               observedAt=None, freshEndpointObserved=endpoint)
    if _assembly_enabled():
        if head is None:
            if ENABLED:
                from steward.ego.release_observation import axis_check
                axis = axis_check(entity, entry, after, stamp)
                if axis:
                    return axis
            points = np.asarray((view or {}).get('points', []), float)
            out['visibility'] = dict(entityPresent=entity is not None,
                endpointSource=(feature or {}).get('source'), viewSource=(view or {}).get('source'),
                viewAgeS=stamp-view['stamp'] if view else None,
                postRelease=bool(view and view['stamp'] > after),
                truncated=(view or {}).get('truncated'), invalidDepthFraction=(view or {}).get('invalidDepthFraction'),
                pointCount=len(points),
                rawExtentM=np.ptp(points, axis=0).tolist() if points.ndim == 2 and len(points) else None)
            return out
    else:
        if head is None:
            return out
    e, h = np.asarray(entry), np.asarray(head['head'])
    xy = float(np.linalg.norm(h[:2]-e[:2])); dz = float(h[2]-e[2])
    inside = None
    if frame:
        q = np.asarray(frame['rotation']).T @ (h-np.asarray(frame['position']))
        inside = bool(np.all(abs(q[:2]) <= np.asarray(frame['size'])[:2]/2+.006))
    if endpoint:
        tip = np.asarray(feature['tip'])
        length = feature.get('lengthM', float(np.linalg.norm(h-tip)))
        aligned = (np.linalg.norm(tip[:2]-e[:2]) <= .012 and abs(tip[2]-e[2]) <= .012
                   and feature['axis'][2] >= math.cos(math.radians(20)))
        height_ok = -.004 <= dz <= length+.006
    else:
        aligned = True
        height_ok = -.004 <= dz <= .012
    passed = bool(xy <= .012 and height_ok and aligned and inside is not False)
    if _assembly_enabled():
        landing = ('seated' if passed else 'outside_hole' if inside is False or xy > .012 else
                   'tilted' if endpoint and feature['axis'][2] < math.cos(math.radians(20)) else
                   'tip_outside_entry' if not aligned else 'height_mismatch')
        return dict(status='confirmed' if passed else 'failed', passed=passed, landingState=landing,
            reason='independent_screw_geometry', observedAt=head['stamp'], source=head['source'],
            headM=h.tolist(), headMinusEntryZM=dz, headDistanceXYM=xy,
            insideComponent=inside, freshEndpointObserved=endpoint)
    else:
        return dict(status='confirmed' if passed else 'failed', passed=passed,
            reason='independent_screw_geometry', observedAt=head['stamp'], source=head['source'],
            headM=h.tolist(), headMinusEntryZM=dz, headDistanceXYM=xy,
            insideComponent=inside, freshEndpointObserved=endpoint)
if _assembly_enabled():
    def local_head_patch(view, entry, stamp):
        """Measure a compact cap above a surrounding raw RGB-D plane, never a box.

        Return coverage even when no cap is present, so a visible empty pre-release
        patch can distinguish a newly appeared head from an existing peg or ridge.
        """
        if (not view or view.get('source') != 'rgbd_segment' or view.get('truncated', True)
                or view.get('invalidDepthFraction', 1) > .05
                or not 0 <= stamp-view.get('stamp', -1) <= 20):
            return None
        points = np.asarray(view.get('points', []), float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            return None
        p = points-np.asarray(entry)
        radius = np.linalg.norm(p[:, :2], axis=1)
        ring = p[(radius >= .016) & (radius <= .028)]
        core = p[radius <= .013]
        if len(ring) < 24 or len(core) < 20:
            return None
        quadrants = {(x >= 0, y >= 0) for x, y in ring[:, :2]}
        if len(quadrants) < 3:
            return None
        design = np.c_[ring[:, :2], np.ones(len(ring))]
        coef, _, rank, _ = np.linalg.lstsq(design, ring[:, 2], rcond=None)
        residual = ring[:, 2]-design @ coef
        if rank < 3 or np.linalg.norm(coef[:2]) > .35 or np.max(abs(residual)) > .0015:
            return None
        heights = core[:, 2]-np.c_[core[:, :2], np.ones(len(core))] @ coef
        out = dict(stamp=view['stamp'], planeZ=float(entry[2]+coef[2]),
                   empty=bool(np.max(abs(heights)) <= .0015), head=None)
        cap = core[heights >= .0025]
        raised = heights[heights >= .0025]
        if len(cap) < 20 or np.max(raised) > .012 or np.ptp(raised) > .002:
            return out
        size = np.quantile(cap[:, :2], .98, axis=0)-np.quantile(cap[:, :2], .02, axis=0)
        if min(size) < .007 or max(size) > .025 or max(size)/min(size) > 1.8:
            return out
        head = np.median(cap, axis=0)+entry
        if np.linalg.norm(head[:2]-np.asarray(entry)[:2]) > .012:
            return out
        out['head'] = head.tolist()
        return out


    def local_head_baseline(scene, entry):
        from steward.ego.fasten_plan import value
        patches = [local_head_patch(value(e, 'contactView'), entry, scene.stamp) for e in scene.entities]
        patches = [p for p in patches if p is not None]
        # Conflicting masks or an existing protrusion cannot establish an empty hole.
        if not patches or any(not p['empty'] for p in patches):
            return None
        return max(patches, key=lambda p:p['stamp'])


    def local_head_check(scene, attempt):
        """Confirm visible landing via a new cap in raw support pixels after release.

        This can survive a merged segmentation id. It does not prove attachment and
        cannot operate without an independently visible empty baseline at this hole.
        """
        from steward.ego.fasten_plan import value
        before = attempt.get('headPatchBefore')
        if not before or not before['empty'] or before['stamp'] > attempt['seatedAt']:
            return None
        patches = [local_head_patch(value(e, 'contactView'), attempt['entry'], scene.stamp)
                   for e in scene.entities]
        patches = [p for p in patches if p and p['stamp'] > attempt['seatedAt']]
        if not patches or any(p['empty'] for p in patches):
            return None
        heads = [p for p in patches if p['head'] is not None
                 and abs(p['planeZ']-before['planeZ']) <= .003]
        if not heads or any(np.linalg.norm(np.asarray(p['head'])-heads[0]['head']) > .003 for p in heads):
            return None
        p = max(heads, key=lambda p:p['stamp'])
        head = np.asarray(p['head'])
        if not -.004 <= head[2]-attempt['entry'][2] <= .012:
            return None
        frame = attempt.get('componentFrame')
        if frame:
            q = np.asarray(frame['rotation']).T @ (head-np.asarray(frame['position']))
            if np.any(abs(q[:2]) > np.asarray(frame['size'])[:2]/2+.006):
                return None
        return dict(status='confirmed', passed=True, reason='new_local_head_after_release',
                    source='rgbd_local_head_change', observedAt=p['stamp'], headM=p['head'],
                    baselineAt=before['stamp'], attachmentObserved=False)
