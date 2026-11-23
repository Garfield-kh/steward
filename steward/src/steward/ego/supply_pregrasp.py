"""Loose-stock ranking and local, mask-independent pregrasp evidence.

Catalogue bounds are hypotheses, never measured extents. Only native depth can
contradict one. This module neither merges entities nor changes load accounting.
"""
from copy import deepcopy
import math
import time
import numpy as np
from steward.ego import capabilities
from steward.ego.fasten_plan import value
from steward.math3d import matrix_from_quat, pose_compose


def enabled():
    return capabilities.get('STEWARD_ASSEMBLY_SUPPLY_PREGRASP') != '0'


def bounds(category):
    from steward.ego.classify import SHAPES
    shape = SHAPES.get(category)
    if not shape:
        return None
    dims = np.asarray(shape['dims'], float)
    scales = shape.get('scale', (1., 1.))
    return dict(seedRadiusM=float(np.linalg.norm(dims)/2),
                minHeightM=float(min(dims)*min(scales)*.5),
                maxSpanM=float(np.linalg.norm(dims)*max(scales)+shape.get('extent_slack', 0.)))


def remembered(est, entity):
    """Keep contradictions across calls and local track renumbering; never erase on unknown."""
    ledger = getattr(est, 'supply_pregrasp_evidence', {})
    if entity.id in ledger:
        old = ledger[entity.id]
        if old.get('status') == 'contradicted' or (
                0 <= est.stamp-old.get('observedAt', -math.inf) <= 20
                and np.linalg.norm(np.asarray(entity.pose.value['p'])-old['position']) <= old.get('associationRadiusM', 0.)):
            return old
    p = np.asarray(entity.pose.value['p'])
    for row in ledger.values():
        if (row.get('status') == 'contradicted' and row.get('category') == entity.category.value
                and np.linalg.norm(p-row['position']) <= row.get('associationRadiusM', 0.)):
            return row
    return {}


def rank_records(records, known, xy):
    """Pure ordering shared by live selection and event replay. Containment is advisory."""
    rows = []
    for record in records:
        row = deepcopy(record)
        p = np.asarray(row['position'], float)
        own = np.asarray(row.get('sizeM', [0, 0, 0]), float)
        parents = []
        for other in known:
            if other['id'] == row['id'] or other.get('supportSurface'):
                continue
            size = np.asarray(other.get('sizeM', [0, 0, 0]), float)
            if max(size) <= 1.5*max(own) or np.prod(size) <= 2*np.prod(own):
                continue
            rot = matrix_from_quat(other.get('q', [1., 0., 0., 0.]))
            local = (p-np.asarray(other['position'])) @ rot
            if np.all(abs(local) <= size/2):
                parents.append(other['id'])
        evidence = row.get('evidence') or {}
        row.update(enclosingEntities=parents, distanceM=float(np.linalg.norm(p[:2]-xy)),
                   status=evidence.get('status', 'unknown'),
                   independentProtrusion=bool(evidence.get('independentProtrusion')))
        # Geometry first, then envelope suspicion, independent segmentation and
        # query evidence. Distance breaks evidence ties, not vice versa.
        row['rankKey'] = [row['status'] == 'contradicted',
                          not row['independentProtrusion'], bool(parents),
                          -float(row.get('segmentScore') or 0.),
                          -float(row.get('queryScore') or 0.),
                          bool(row.get('supportMismatch')), row['distanceM'], row['id']]
        rows.append(row)
    return sorted(rows, key=lambda r: r['rankKey'])


def rank(controller, candidates, scene, support):
    from steward.ego.planner import entity_box
    from steward.ego.priors import prior
    from steward.ego import reflection
    est = controller.rt.estimator
    records, known = [], []
    for entity in scene.entities:
        try:
            box, lo, hi = entity_box(entity)
            known.append(dict(id=entity.id, position=entity.pose.value['p'], q=entity.pose.value['q'],
                              sizeM=[2*box.hx, 2*box.hy, hi-lo],
                              supportSurface='support-surface' in prior(entity.category.value).get('affordances', ())))
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
    by_id = {r['id']: r for r in known}
    for entity in candidates:
        track = est.tracks.get(entity.id)
        category = entity.category.value
        score = (getattr(track, 'semantic_counts', {}).get(category, 0.) /
                 max(1, getattr(track, 'semantic_views', 0)))
        obj = dict(by_id.get(entity.id, {}), id=entity.id, category=category, role=category,
                   position=entity.pose.value['p'])
        if track is not None:
            obj['extentM'] = np.asarray(track.size).tolist()  # query evidence only
        counts = getattr(track, 'category_counts', {})
        total = sum(counts.values())
        evidence = reflection.Evidence({k:v/total for k,v in counts.items()} if total else {},
                                       {category: score}, getattr(track, 'label', None))
        # A nominal catalogue-only query is not positive observation evidence.
        query = None
        if getattr(track, 'semantic_views', 0):
            query, _ = reflection.match(obj, evidence, category, None, reflection.ReflectionConfig())
        records.append(dict(obj, segmentScore=score, queryScore=query,
                            supportMismatch=controller.rt.esim._anchor(scene, entity).id != support,
                            evidence=deepcopy(remembered(est, entity))))
    o = controller.ex._state()['odometry']
    return rank_records(records, known, np.array([o['x'], o['y']]))


def _plane(points, xy, tolerance):
    """External annulus only; robust gravity-facing plane with surrounding coverage."""
    if len(points) < 40:
        return None
    sample = points[np.linspace(0, len(points)-1, min(len(points), 1600)).astype(int)]
    rng = np.random.default_rng(102)
    best = None
    for _ in range(48):
        p = sample[rng.choice(len(sample), 3, replace=False)]
        design = np.c_[p[:, :2]-xy, np.ones(3)]
        if abs(np.linalg.det(design)) < np.finfo(float).eps:
            continue
        fit = np.linalg.solve(design, p[:, 2])
        if np.linalg.norm(fit[:2]) > math.tan(math.radians(35)):
            continue
        residual = abs(sample[:, 2]-np.c_[sample[:, :2]-xy, np.ones(len(sample))] @ fit)
        inside = residual <= tolerance
        if best is None or inside.sum() > best[0]:
            best = (int(inside.sum()), fit)
    if best is None:
        return None
    design = np.c_[points[:, :2]-xy, np.ones(len(points))]
    fit = best[1]
    for _ in range(3):
        inside = abs(points[:, 2]-design @ fit) <= tolerance
        if inside.sum() < 40:
            return None
        fit = np.linalg.lstsq(design[inside], points[inside, 2], rcond=None)[0]
    residual = abs(points[inside, 2]-design[inside] @ fit)
    quadrants = (points[inside, 0] > xy[0]).astype(int)+2*(points[inside, 1] > xy[1])
    coverage = np.bincount(quadrants, minlength=4)
    error = float(np.quantile(residual, .95))
    reliable = bool(inside.mean() >= .55 and np.all(coverage >= 5)
                    and np.linalg.norm(fit[:2]) <= math.tan(math.radians(35)))
    return dict(coefficients=fit.tolist(), originXY=np.asarray(xy).tolist(), errorM=error,
                inlierFraction=float(inside.mean()), quadrantCounts=coverage.tolist(),
                pointCount=len(points), reliable=reliable)


def measure(raw, cam, centre, limits, *, expansion=1., excluded_boxes=()):
    """Two independent contradictions from depth, never sizeM/extentM or semantic masks.

    All metric tolerances derive from catalogue scale, measured noise and pixel
    footprint. A smooth elevated patch may prove a lower span without a silhouette.
    """
    from steward.ego import wrist_observation as wo
    from steward.ego.support_holes import components, unproject
    started = time.monotonic()
    row = dict(status='unknown', reasons=[], independentProtrusion=False,
               observedAt=float(raw['header']['stamp']), camera=raw.get('camera'),
               catalogue=limits, expansion=expansion, source='native_rgbd_cross_mask')
    if not limits:
        return dict(row, reason='catalogue_unavailable'), {}
    centre = np.asarray(centre, float)
    depth = np.asarray(raw['depth'], float)
    radius = limits['seedRadiusM']+limits.get('positionUncertaintyM', 0.)
    diag = {}
    region = wo.window(raw, cam, centre, radius*4*expansion, diag)
    yy, xx = np.where(region)
    if not len(yy):
        return dict(row, reason='seed_outside_frame'), {}
    y0, y1, x0, x1 = int(yy.min()), int(yy.max()+1), int(xx.min()), int(xx.max()+1)
    ys, xs = np.mgrid[y0:y1, x0:x1]
    d = depth[y0:y1, x0:x1]
    valid = np.isfinite(d) & (d > 0)
    depth_range = raw.get('depthRangeM')
    if depth_range:
        valid &= (d >= depth_range[0]) & (d <= depth_range[1])
    intr = dict(width=depth.shape[1], height=depth.shape[0], fovy=raw['fovy'])
    points = unproject(np.c_[xs.ravel(), ys.ravel()], np.where(valid, d, 0).ravel(), cam, intr).reshape(*d.shape, 3)
    for c, half, rotation in excluded_boxes:
        valid &= ~np.all(abs((points-np.asarray(c)) @ np.asarray(rotation)) <= np.asarray(half), axis=-1)
    pixel = float(diag['pixelM'])
    # Raw high-frequency residual estimates noise; no gap filling or averaging
    # across a depth edge is permitted in the connectivity graph.
    differences = []
    for axis in (0, 1):
        diff = abs(np.diff(d, axis=axis))
        pair = (valid[1:] & valid[:-1]) if axis == 0 else (valid[:, 1:] & valid[:, :-1])
        differences.extend(diff[pair].ravel()[::max(1, int(pair.sum())//4000)].tolist())
    if not differences:
        return dict(row, reason='depth_noise_unresolved', wallS=time.monotonic()-started), {}
    noise = float(np.median(differences)/.954)
    error = 3*noise+2*pixel
    uv = np.asarray(diag['centrePixel'])
    projected_r = radius/pixel
    seed_image = ((xs-uv[0])**2+(ys-uv[1])**2 <= projected_r**2)
    radial = np.linalg.norm(points[..., :2]-centre[:2], axis=-1)
    seed = seed_image & valid & (radial <= radius*1.25)
    annulus = valid & (radial > radius*1.5) & (radial < radius*3.5*expansion)
    plane = _plane(points[annulus], centre[:2], max(error, limits['minHeightM']*.15))
    row.update(pixelM=pixel, noiseSigmaM=noise, seedPoints=int(seed.sum()),
               seedValidFraction=float((seed_image & valid).sum()/max(1, seed_image.sum())), supportPlane=plane)
    artifacts = dict(crop=[x0, y0, x1, y1], seedPixels=np.c_[xs[seed], ys[seed]].tolist(),
                     supportPixels=np.c_[xs[annulus], ys[annulus]].tolist(), connectedPixels=[])
    rgb_seed = np.asarray(raw['rgb'][y0:y1, x0:x1])[seed]
    saturated = float(np.mean(np.min(rgb_seed, axis=-1) >= 250)) if len(rgb_seed) else 0.
    row['saturatedSeedFraction'] = saturated
    if saturated > .2:
        return dict(row, reason='photometric_saturation_depth_unconfirmed', wallS=time.monotonic()-started), artifacts
    if plane is None or not plane['reliable'] or seed.sum() < 20:
        return dict(row, reason='external_support_or_seed_unresolved', wallS=time.monotonic()-started), artifacts
    a, b, c = plane['coefficients']
    height = (points[..., 2]-(a*(points[..., 0]-centre[0])+b*(points[..., 1]-centre[1])+c))/math.sqrt(1+a*a+b*b)
    error += plane['errorM']
    measured = float(np.quantile(height[seed], .99))
    upper = measured+error
    row.update(heightM=measured, heightErrorUpperM=error, heightUpperM=upper)
    complete_seed = (row['seedValidFraction'] >= .98 and projected_r >= 5 and
                     uv[0]-projected_r > 0 and uv[1]-projected_r > 0 and
                     uv[0]+projected_r < depth.shape[1]-1 and uv[1]+projected_r < depth.shape[0]-1)
    # Unresolved top-only views do not have a reliable external annulus.
    if (complete_seed and abs(measured) <= error and error < limits['minHeightM']/2
            and upper < limits['minHeightM']):
        row['reasons'].append('measured_coplanar')
    elevated = valid & (height > error*2) & (height > limits['minHeightM']*.5)
    row['independentProtrusion'] = bool((seed & elevated).sum() >= 20)
    foreground = seed & elevated
    if foreground.sum() >= 20 and error < limits['minHeightM']:
        # Keep a tangent-consistent sheet. A contact with the support cannot
        # enter; a depth crease/jump cannot connect two bodies via a single pixel.
        local = points[foreground]
        origin = np.median(local, axis=0)
        _, _, axes = np.linalg.svd(local-origin, full_matrices=False)
        normal = axes[-1]
        tangent_error = float(np.quantile(abs((local-origin) @ normal), .9))
        sheet = elevated & (abs((points-origin) @ normal) <= error*2)
        edges = np.zeros(d.shape, bool)
        for axis in (0, 1):
            delta = np.linalg.norm(np.diff(points, axis=axis), axis=-1)
            jump = delta > error*2+pixel*2
            if axis == 0:
                edges[1:] |= jump; edges[:-1] |= jump
            else:
                edges[:, 1:] |= jump; edges[:, :-1] |= jump
        # A visible material boundary can separate touching coplanar objects.
        # This only removes graph edges; it never fills a gap or proves identity.
        rgb = np.asarray(raw['rgb'][y0:y1, x0:x1], float)/255.
        for axis in (0, 1):
            contrast = np.max(abs(np.diff(rgb, axis=axis)), axis=-1)
            cut = contrast > max(.12, 6*float(np.median(contrast)))
            if axis == 0:
                edges[1:] |= cut; edges[:-1] |= cut
            else:
                edges[:, 1:] |= cut; edges[:, :-1] |= cut
        sheet &= ~edges
        padded = np.pad(sheet, 1)
        neighbours = np.lib.stride_tricks.sliding_window_view(padded, (3, 3)).sum(axis=(-2, -1))
        sheet &= neighbours >= 6
        groups = components(sheet)
        groups = [g for g in groups if len(g) >= 20 and np.any(foreground[g[:, 1], g[:, 0]])]
        if groups and tangent_error <= error:
            group = max(groups, key=lambda g: int(foreground[g[:, 1], g[:, 0]].sum()))
            cloud = points[group[:, 1], group[:, 0]]
            _, _, axes = np.linalg.svd(cloud-cloud.mean(axis=0), full_matrices=False)
            spans = np.diff(np.quantile(cloud @ axes.T, [.01, .99], axis=0), axis=0)[0]
            lower = max(0., float(max(spans))-2*error)
            row.update(connectedPoints=len(cloud), visibleSpanLowerM=lower,
                       tangentErrorM=tangent_error, supportExcluded=True,
                       continuity='elevated_tangent_patch_without_depth_or_colour_seam')
            artifacts['connectedPixels'] = (group+[x0, y0]).tolist()
            if lower > limits['maxSpanM']:
                row['reasons'].append('visible_surface_oversize')
    row.update(status='contradicted' if row['reasons'] else 'supported' if row['independentProtrusion'] else 'unknown',
               reason='measured' if row['reasons'] or row['independentProtrusion'] else 'insufficient_positive_geometry',
               wallS=time.monotonic()-started)
    return row, artifacts


def samples(controller, entity):
    """Reuse native near frames and their own encoder pose, never current-pose reprojection."""
    est = controller.rt.estimator
    frames = getattr(controller.rt, '_latest_native_frames', {})
    result = []
    from steward.ego.support_holes import project
    for camera, raw in list(frames.items()):
        try:
            header = raw['header']
            if not 0 <= est.stamp-header['stamp'] <= 2.:
                continue
            link, sensor = est.cameras[camera]
            cam = pose_compose(est.link_pose(link, header['joints'], header['odometry']), sensor['mount'])
            h, w = raw['depth'].shape
            uv, z = project([entity.pose.value['p']], cam, dict(width=w, height=h, fovy=raw['fovy']))
            if z[0] <= 0 or not np.isfinite(uv).all() or not (0 <= uv[0, 0] < w and 0 <= uv[0, 1] < h):
                continue
            pixel = z[0]*2*math.tan(math.radians(raw['fovy'])/2)/h
            result.append((pixel, dict(raw, depthRangeM=sensor.get('depthRangeM')), cam))
        except (KeyError, AttributeError, ValueError, TypeError):
            continue
    return [(raw, cam) for _, raw, cam in sorted(result, key=lambda r:r[0])]


def validate(controller, entity, sample, *, expansion=1.):
    est = controller.rt.estimator
    row = dict(status='unknown', reason='native_frame_unavailable', reasons=[])
    artifacts = {}
    if sample is not None:
        raw, cam = sample
        try:
            # Kinematic robot exclusion uses the frame's encoders.
            from copy import copy
            proxy = copy(est)
            proxy.joints, proxy.odometry = raw['header']['joints'], raw['header']['odometry']
            boxes = list(proxy.body_boxes())
            limits = bounds(entity.category.value)
            cov = (getattr(entity.pose, 'unc', None) or {}).get('cov', [])
            if limits and len(cov) == 36:
                limits = dict(limits, positionUncertaintyM=2*math.sqrt(max(0., cov[0], cov[7])))
            row, artifacts = measure(raw, cam, entity.pose.value['p'], limits,
                                     expansion=expansion, excluded_boxes=boxes)
        except (ValueError, KeyError, AttributeError, TypeError, np.linalg.LinAlgError) as exc:
            row.update(reason='measurement_unavailable', detail=type(exc).__name__+': '+str(exc))
    old = remembered(est, entity)
    row.update(entity=entity.id, category=entity.category.value, position=list(entity.pose.value['p']),
               associationRadiusM=(bounds(entity.category.value) or {}).get('seedRadiusM', 0.)*.5)
    if old.get('status') == 'contradicted':
        row = dict(row, status='contradicted', retainedContradiction=deepcopy(old.get('retainedContradiction') or old))
    est.supply_pregrasp_evidence = getattr(est, 'supply_pregrasp_evidence', {})
    est.supply_pregrasp_evidence[entity.id] = deepcopy(row)
    return row, artifacts


def select(controller, candidates, scene, support, selection):
    """Rank at selection; local measurement waits for the normal pre-close approach."""
    rows = rank(controller, candidates, scene, support)
    selection.update(supply_pregrasp=True, rankingBefore=[r['id'] for r in sorted(rows, key=lambda r:(r['supportMismatch'], r['distanceM'], r['id']))],
                     rankingAfter=[r['id'] for r in rows])
    selection['candidateEvidence'] = [compact(r) for r in rows]
    controller._supply_pregrasp_selection = deepcopy(selection)
    controller._supply_pregrasp_selection['candidateEvidence'] = deepcopy(rows)
    controller._supply_pregrasp_candidates = [e.id for e in candidates]
    controller._supply_pregrasp_pending_target = rows[0]['id']
    return next(e for e in candidates if e.id == rows[0]['id'])


def compact(row):
    """Keep each candidate's decision evidence, not duplicate pose/box/ledger payloads."""
    result = dict(entity=row['id'], status=row['status'])
    for key in ('segmentScore', 'queryScore', 'enclosingEntities', 'independentProtrusion'):
        if row.get(key):
            result[key] = row[key]
    evidence = row.get('evidence') or {}
    for key in ('reasons', 'heightM', 'heightErrorUpperM', 'heightUpperM', 'visibleSpanLowerM', 'observedAt'):
        if key in evidence:
            result[key] = evidence[key]
    if evidence.get('retainedContradiction'):
        result['retainedReasons'] = evidence['retainedContradiction'].get('reasons', [])
    return result


def pregrasp(controller, job, entity, support):
    """One check after normal approach; one local expansion if all contradicted.

    Returns a replacement ID or None. Unknown always proceeds. No exit/hold/input
    path is introduced; normal release/result observation and remedies stay active.
    """
    available = samples(controller, entity)
    sample = available[0] if available else None
    if sample is None and getattr(controller.rt, 'perception_mode', None) is not None:
        from steward.ego import wrist_observation as wo
        old = getattr(controller, '_wristContext', None)
        controller._wristContext = dict(phase='supply_pregrasp', target=entity.pose.value['p'])
        try:
            sample = wo.capture(controller, time.monotonic()+2.)
        finally:
            controller._wristContext = old
    evidence, artifacts = validate(controller, entity, sample)
    scene = controller.rt.ego_state()
    candidates = [e for eid in getattr(controller, '_supply_pregrasp_candidates', [entity.id]) if (e := scene.find(eid)) is not None]
    if not any(e.id == entity.id for e in candidates):
        candidates.append(entity)
    rows = rank(controller, candidates, scene, support)
    all_bad = all(r['status'] == 'contradicted' for r in rows)
    if all_bad:
        if sample is not None:
            evidence, artifacts = validate(controller, entity, sample, expansion=2.)
            evidence['boundedFallback'] = 'expanded_local_window_once_then_best_estimate'
            rows = rank(controller, candidates, scene, support)
        else:
            evidence['boundedFallback'] = 'local_expansion_unavailable_then_best_estimate'
    replacement = rows[0]['id'] if evidence['status'] == 'contradicted' and not all_bad and rows[0]['id'] != entity.id else None
    result = dict(entity=entity.id, evidence=evidence, candidates=[compact(r) for r in rows],
                  rankingBefore=(getattr(controller, '_supply_pregrasp_selection', {}) or {}).get('rankingAfter', [entity.id]),
                  rankingAfter=[r['id'] for r in rows], replacement=replacement,
                  unconfirmed=evidence['status'] != 'supported', allContradicted=all_bad,
                  action='rerank_once' if replacement else 'execute_best_estimate', resultObservation='existing_post_release_path')
    if sample is not None:
        from steward.ego.wrist_observation import _dump
        raw, cam = sample
        context = dict(phase='supply_pregrasp', measureCall=True, **result, expansion=artifacts,
                       selection=deepcopy(getattr(controller, '_supply_pregrasp_selection', {})), rankedCandidates=rows)
        result['frameArchive'] = _dump(raw, cam, raw['header'], context)
    result['archiveConfigured'] = bool(capabilities.get('STEWARD_WRIST_DUMP_DIR'))
    job.postcondition.setdefault('supplyPregrasp', []).append(deepcopy(result))
    if result['unconfirmed']:
        job.postcondition['supplyUnconfirmed'] = True
    from steward.ego.assembly_session import current
    if current() is not None:
        current().emit('supply_pregrasp', **deepcopy(result))
    return replacement


class _Rerank(Exception):
    """Internal pre-close control transfer; never a skill failure or user input request."""


def pick(controller, job, plan, params):
    """Check once before closure, then at most one replacement grasp plan."""
    ex = controller.ex
    previous = getattr(ex, '_supply_pregrasp_before_pick', None)
    def check(current_plan):
        scene = controller.rt.ego_state()
        entity = scene.find(current_plan['target'])
        if entity is None:
            job.postcondition.setdefault('supplyPregrasp', []).append(dict(
                entity=current_plan['target'], unconfirmed=True, reason='target_track_unavailable',
                action='execute_best_estimate'))
            job.postcondition['supplyUnconfirmed'] = True
            return
        if pregrasp(controller, job, entity, current_plan['support']):
            raise _Rerank()
    ex._supply_pregrasp_before_pick = check
    try:
        try:
            ex._pick(job, plan)
            return
        except _Rerank:
            # No jaw close or load yet. Withdraw before any replacement base
            # motion. The second pick cannot restart this check.
            ex._supply_pregrasp_before_pick = None
            ex._withdraw(plan)
            scene = controller.rt.ego_state()
            original = scene.find(plan['target'])
            from steward.ego.skills import SkillFailed
            from steward.ego import supply_reliability as sr
            try:
                alternative = controller.choose(original.category.value, plan['support'], free_screw=True) if original else None
                if alternative is not None and (sr.role_enabled() or sr.size_enabled()):
                    sr.preflight(controller, job, alternative.id)
                receipt = (controller.rt.predict('pick', dict(params, target=alternative.id))
                           if alternative is not None else None)
            except SkillFailed as exc:
                controller.check()
                receipt = None
                job.postcondition['supplyPregrasp'][-1]['replacementUnavailable'] = str(exc)
            if receipt and receipt.get('withinRiskLimit'):
                new_plan = controller.rt._consume_receipt(receipt)
                if sr.role_enabled() or sr.size_enabled():
                    new_plan['protectGraspFrame'] = True
                from steward.ego.assembly_session import current
                if current() is not None:
                    current().require_pick(new_plan['target'])
                controller.ex._begin(job, 'pick', receipt)
                plan.clear(); plan.update(new_plan)
                origins = controller.rt.estimator.fastener_origins
                origins[alternative.id] = dict(origins.get(alternative.id, {}), support=plan['support'],
                    kitSupport=origins.get(alternative.id, {}).get('kitSupport', plan['support']),
                    xy=list(alternative.pose.value['p'][:2]))
                job.postcondition['supplyPregrasp'][-1].update(
                    executedTarget=alternative.id, replacementReceipt=deepcopy(receipt), replacementUnconfirmed=True)
            else:
                job.postcondition['supplyPregrasp'][-1]['fallbackReason'] = 'replacement_plan_unavailable'
            job.postcondition['supplyUnconfirmed'] = True
            ex._pick(job, plan)
    finally:
        controller.__dict__.pop('_supply_pregrasp_pending_target', None)
        if previous is None:
            ex.__dict__.pop('_supply_pregrasp_before_pick', None)
        else:
            ex._supply_pregrasp_before_pick = previous
