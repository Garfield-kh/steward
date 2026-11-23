"""Native RGB-D recesses inside a support region, independent of track identity.

The saved interface is only a search/correspondence prior. Measured centres come
from closed image contours intersected with their surrounding depth plane.
"""
import math
import numpy as np
from PIL import Image, ImageFilter
from steward.math3d import matrix_from_quat
from steward.ego.perception import PerceptionConfig

MIN_HOLE_PIXELS = 8
MAX_PLANE_ERROR_M = .003


# The released part's estimated pose bounds how far its saved hole layout can be from the holes seen.
MAX_LAYOUT_SHIFT = .05

def project(points, cam, intr):
    local = (np.asarray(points)-cam['p']) @ matrix_from_quat(cam['q'])
    focal = intr['height']/(2*math.tan(math.radians(intr['fovy'])/2))
    uv = np.array([intr['width']/2, intr['height']/2])-focal*local[..., 1:]/local[..., :1]
    return uv, local[..., 0]


def unproject(uv, depth, cam, intr):
    focal = intr['height']/(2*math.tan(math.radians(intr['fovy'])/2))
    xy = (np.asarray(uv)-[intr['width']/2, intr['height']/2])/focal
    local = np.c_[depth, -xy*np.asarray(depth)[:, None]]
    return local @ matrix_from_quat(cam['q']).T+cam['p']


def framing(uv, shape):
    height, width = shape
    uv = np.asarray(uv)
    lo, hi = uv.min(axis=0), uv.max(axis=0)
    centre = (lo+hi)/2 / [width, height]
    edge = float(min(*lo, width-1-hi[0], height-1-hi[1]))
    border = bool(edge < .05*min(width, height))
    return dict(border=border, centred=bool(np.all((centre >= .25) & (centre <= .75))),
                edgePixels=round(edge,3), centreFraction=centre.round(6).tolist())


def components(binary):
    """Connected components without an additional image-processing dependency."""
    # Flood runs, rather than individual pixels of potentially large shadows.
    runs = []
    parents = []
    previous = []
    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]; i = parents[i]
        return i
    for y, row in enumerate(binary):
        changes = np.diff(np.r_[False, row, False].astype(np.int8))
        current = []
        for a, b in zip(np.where(changes == 1)[0], np.where(changes == -1)[0]):
            i = len(runs); runs.append((y, int(a), int(b))); parents.append(i)
            for j in previous:
                _, c, d = runs[j]
                if c <= b and d >= a:
                    parents[root(j)] = root(i)
            current.append(i)
        previous = current
    groups = {}
    for i, run in enumerate(runs):
        groups.setdefault(root(i), []).append(run)
    for group in groups.values():
        yield np.concatenate([np.c_[np.arange(a, b), np.full(b-a, y)] for y, a, b in group])


def detect(raw, cam, target):
    """Return visible empty recesses, with no semantic or single-part size test.

    Use local depth precision and native pixel size, without a near-range gate. A hole
    pixel may be absent from the mask; its complete surrounding ring must belong
    to the selected region. No depth reading is fabricated for a missing core.
    """
    rgb, depth, mask = raw['rgb'], raw['depth'], raw.get('mask')
    shape = depth.shape
    intr = dict(width=shape[1], height=shape[0], fovy=raw['fovy'])
    result = dict(holes=[], framing=None, mask=None, reason='no_assembly_mask')
    if mask is None or mask.shape != shape or rgb.shape[:2] != shape:
        return result
    uv, forward = project([target], cam, intr)
    if forward[0] <= 0 or not np.isfinite(uv).all():
        return result
    candidates = []
    for ident in np.unique(mask):
        if ident == 0:
            continue
        yy, xx = np.where(mask == ident)
        if not (xx.min() <= uv[0,0] <= xx.max() and yy.min() <= uv[0,1] <= yy.max()):
            continue
        d = depth[yy, xx]
        valid = np.isfinite(d) & (d > .05) & (d < PerceptionConfig().max_range)
        if valid.sum() < 48 or 1-valid.mean() > .05:
            continue
        indices = np.linspace(0, valid.sum()-1, min(8192, valid.sum()), dtype=int)
        points = unproject(np.c_[xx[valid][indices], yy[valid][indices]], d[valid][indices], cam, intr)
        if not np.isfinite(points).all():
            continue
        distance = float(np.min(np.linalg.norm(points-np.asarray(target), axis=1)))
        if distance <= .03:
            candidates.append((distance, int(ident), xx, yy, points))
    if not candidates:
        return result
    candidates.sort(key=lambda c: c[0])
    if len(candidates) > 1 and candidates[1][0]-candidates[0][0] < .003:
        result['reason'] = 'ambiguous_assembly_mask'
        return result
    _, ident, xx, yy, points = candidates[0]
    result.update(mask=ident, framing=framing(np.c_[xx, yy], shape),
                  regionBoundsM=[points.min(axis=0).tolist(), points.max(axis=0).tolist()])
    if result['framing']['border'] or not result['framing']['centred']:
        result['reason'] = 'view_adjustment_failed'
        return result
    result['holeRejections'] = {}
    result['holes'] = detect_region(raw, cam, mask == ident, result['holeRejections'])
    result['reason'] = None if result['holes'] else 'no_holes_in_mask'
    return result


def detect_region(raw, cam, region, diagnostics=None, *, calibrated=False, measurement_context=None, candidate_sink=None):
    """Closed native colour contours, located on a robust surrounding plane.

    Also usable on a native window embedded in its calibrated full raster. The
    caller owns region selection and whole-frame eligibility. Missing/quantized
    core depth is not a veto; a resolved protrusion is. No semantic colour prior.
    """
    diagnostics = diagnostics if diagnostics is not None else {}
    def reject(reason):
        diagnostics[reason] = diagnostics.get(reason, 0)+1
    rgb, depth = raw['rgb'], raw['depth']
    shape = depth.shape
    intr = dict(width=shape[1], height=shape[0], fovy=raw['fovy'])
    yy, xx = np.where(region)
    if not len(xx):
        return []
    x0, y0, x1, y1 = xx.min(), yy.min(), xx.max()+1, yy.max()+1
    region = region[y0:y1,x0:x1]
    gray = rgb[y0:y1,x0:x1].astype(float).mean(axis=2)
    radius = max(4, int(min(region.shape)*.06))
    image = Image.fromarray(gray.astype(np.uint8))
    # Separable square morphology avoids the quadratic PIL square-kernel cost.
    def extrema(array, maximum):
        padded = np.pad(array, ((0,0),(radius,radius)), mode='edge')
        windows = np.lib.stride_tricks.sliding_window_view(padded, 2*radius+1, axis=1)
        horizontal = windows.max(-1) if maximum else windows.min(-1)
        padded = np.pad(horizontal, ((radius,radius),(0,0)), mode='edge')
        windows = np.lib.stride_tricks.sliding_window_view(padded, 2*radius+1, axis=0)
        return windows.max(-1) if maximum else windows.min(-1)
    closing = extrema(extrema(np.asarray(image), True), False)
    dark = gray < closing*.65
    if calibrated:
        # Illumination scales channels together. Separate reflectance boundaries
        # before components, so a shadow cannot join a recess to a part edge.
        colour = rgb[y0:y1,x0:x1].astype(float)
        saturation = np.ptp(colour, axis=2)/np.maximum(colour.max(2), 1)
        surround_saturation = extrema(saturation, True)
        # A shadow on a chromatic surface preserves saturation. Neutral recesses
        # differ; neutral surfaces retain the original brightness detector.
        dark &= (surround_saturation <= .15) | (saturation < .5*surround_saturation)
        dark = np.asarray(Image.fromarray(dark.astype(np.uint8)).filter(
            ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))) > 0
    output = []
    trace = diagnostics.get('candidates')
    for pixels in components(dark):
        lo, hi = pixels.min(axis=0), pixels.max(axis=0)
        spans = hi-lo+1
        detail = dict(pixel=(pixels.mean(axis=0)+[x0,y0]).tolist(), spanPixels=spans.tolist(), pixels=len(pixels), fill=float(len(pixels)/np.prod(spans)))
        if trace is not None: trace.append(detail)
        def veto(reason):
            detail['reason'] = reason
            # Preserve the legacy head-camera receipt when no detailed trace is requested.
            if calibrated or trace is not None or reason in (
                    'incomplete_mask_rim','dark_or_occluded_rim','invalid_rim_depth',
                    'plane_depth_noise','protrusion','insufficient_pixels'):
                reject(reason)
        if (len(pixels) < 6 or len(pixels) > .05*region.sum() or min(spans) < 2
                or (not calibrated and max(spans)/min(spans) > 4) or len(pixels)/np.prod(spans) < .6):
            veto('contour_size_aspect_fill'); continue
        # A closed, convex, near-elliptic contour excludes crescents, stripes,
        # shadow junctions and components connected to the image boundary.
        centred = pixels-pixels.mean(axis=0)
        covariance = centred.T @ centred/len(pixels)
        radial = np.einsum('ij,jk,ik->i', centred, np.linalg.inv(covariance), centred)
        detail['radial98'] = float(np.quantile(radial, .98))
        if np.quantile(radial, .98) > 4.8:
            veto('non_elliptic_contour'); continue
        pad = max(2 if calibrated else 3, int(math.ceil(min(spans)*(.25 if calibrated else .5))))
        a, b = lo-pad, hi+pad+1
        if min(a) < 0 or b[0] > region.shape[1] or b[1] > region.shape[0]:
            veto('rim_outside_image'); continue
        core = np.zeros((b[1]-a[1], b[0]-a[0]), dtype=np.uint8)
        core[pixels[:,1]-a[1], pixels[:,0]-a[0]] = 1
        dilation = np.asarray(Image.fromarray(core).filter(ImageFilter.MaxFilter(2*pad+1))) > 0
        ring = dilation & ~core.astype(bool)
        local_region = region[a[1]:b[1], a[0]:b[0]]
        if not ring.any() or local_region[ring].mean() < .95:
            veto('incomplete_mask_rim')
            continue
        if not calibrated and np.quantile(gray[a[1]:b[1], a[0]:b[0]][ring], .1) < np.median(gray[pixels[:,1],pixels[:,0]])/.65:
            veto('dark_or_occluded_rim')
            continue
        ry, rx = np.where(ring)
        rim_uv = np.c_[rx+a[0]+x0, ry+a[1]+y0]
        rim_depth = depth[rim_uv[:,1], rim_uv[:,0]]
        valid = np.isfinite(rim_depth) & (rim_depth > (0 if calibrated else .05)) & (rim_depth < PerceptionConfig().max_range)
        detail.update(rimValidFraction=float(valid.mean()), rimValidPoints=int(valid.sum()))
        rim_coverage = (np.ptp(rim_uv[valid], axis=0)/np.maximum(np.ptp(rim_uv, axis=0),1)
                        if valid.any() else np.zeros(2))
        detail['rimCoverage'] = rim_coverage.tolist()
        if valid.mean() < (.8 if calibrated else .95) or valid.sum() < 24 or (calibrated and min(rim_coverage)<.7):
            veto('invalid_rim_depth')
            continue
        # Inverse optical depth is linear on a plane. Unlike orthogonal PCA on
        # a tiny noisy ring, this fit does not tilt toward the noise axis.
        uv = rim_uv[valid].astype(float)
        origin = pixels.mean(axis=0)+[x0,y0]
        design = np.c_[uv-origin, np.ones(len(uv))]
        inverse = 1/rim_depth[valid]
        keep = np.ones(len(uv),bool)
        for _ in range(4):
            coef = np.linalg.lstsq(design[keep], inverse[keep], rcond=None)[0]
            residual = 1/(design @ coef)-rim_depth[valid]
            median = np.median(residual[keep])
            scale = max(.001, 1.4826*np.median(abs(residual[keep]-median)))
            keep = abs(residual-median) <= 3*scale
        detail.update(planeInlierFraction=float(keep.mean()), rimValidFraction=float(valid.mean()), rimValidPoints=int(valid.sum()))
        if keep.mean() < .8:
            veto('plane_inliers'); continue
        plane_uv = np.array([origin, origin+[1,0], origin+[0,1]])
        plane_d = 1/(np.c_[plane_uv-origin,np.ones(3)] @ coef)
        if not np.all((plane_d > (0 if calibrated else .05)) & (plane_d < PerceptionConfig().max_range)):
            veto('plane_range'); continue
        plane = unproject(plane_uv, plane_d, cam, intr)
        point = plane[0]
        normal = np.cross(plane[1]-point, plane[2]-point)
        normal /= np.linalg.norm(normal)
        if normal @ (np.asarray(cam['p'])-point) < 0: normal = -normal
        rim = unproject(uv, rim_depth[valid], cam, intr)
        errors = (rim-point) @ normal
        noise = float(np.sqrt(np.mean(errors[keep]**2)))
        detail.update(normal=normal.tolist(), planeNoiseM=noise, fitScaleM=float(scale),
                      rimPadPixels=pad, rayPlaneCosine=abs(float((plane[0]-cam['p']) @ normal))/float(np.linalg.norm(plane[0]-cam['p'])))
        if noise > MAX_PLANE_ERROR_M:
            veto('plane_depth_noise')
            continue
        ray = unproject([origin], np.ones(1), cam, intr)[0]-cam['p']
        denom = float(ray @ normal)
        if abs(denom) < (.06 if calibrated else .15):
            veto('grazing_ray'); continue
        if calibrated:
            # Test roundness in the measured plane, never in oblique image axes.
            contour_d = 1/(np.c_[pixels+[x0,y0]-origin, np.ones(len(pixels))] @ coef)
            footprint = unproject(pixels+[x0,y0], contour_d, cam, intr)
            _, singular, _ = np.linalg.svd(footprint-footprint.mean(0), full_matrices=False)
            aspect = float(singular[0]/max(singular[1], 1e-12))
            detail.update(planeAspect=aspect, diameterM=float(4*singular[0]/math.sqrt(len(pixels))))
            if aspect > 1.8:
                veto('non_circular_on_plane'); continue
            if normal[2] < math.cos(math.radians(35)):
                veto('non_support_normal'); continue
        core_uv = pixels+[x0,y0]
        core_depth = depth[core_uv[:,1],core_uv[:,0]]
        valid_core = np.isfinite(core_depth) & (core_depth > (0 if calibrated else .05)) & (core_depth < PerceptionConfig().max_range)
        inside = unproject(core_uv[valid_core], core_depth[valid_core], cam, intr)
        recess = -(inside-point) @ normal
        # Resolved caps reject even when another portion reads through an edge.
        tolerance = max(.0015, 3*noise)
        detail.update(recessToleranceM=tolerance, coreValidFraction=float(valid_core.mean()),
                      recessMedianM=float(np.median(recess)) if len(recess) else None)
        if len(recess) and (np.quantile(recess,.2) < -tolerance or np.median(recess) < -.0015):
            veto('protrusion')
            continue
        resolved = bool(valid_core.mean() >= .95 and len(recess) and np.quantile(recess,.2) >= tolerance)
        if not resolved and (min(spans) < (3 if calibrated else MIN_HOLE_PIXELS) or len(pixels) < 24):
            veto('insufficient_pixels')
            continue
        # With effectively exact depth a perfectly coplanar dark marking is
        # resolved as flat. Quantized/noisy observations cannot make that claim.
        if noise < 1e-7 and len(recess) and np.max(abs(recess)) < 1e-7:
            veto('flat_marking'); continue
        # A cast shadow approximately scales all RGB channels alike. On a
        # chromatic surface require a chromaticity change unless depth resolves
        # a recess. Neutral surfaces rely on contour + complete planar rim.
        ring_rgb = np.median(rgb[rim_uv[:,1],rim_uv[:,0]],axis=0).astype(float)
        core_rgb = np.median(rgb[core_uv[:,1],core_uv[:,0]],axis=0).astype(float)
        colour_delta = np.linalg.norm(ring_rgb/max(ring_rgb.sum(),1)-core_rgb/max(core_rgb.sum(),1))
        chromatic = (ring_rgb.max()-ring_rgb.min())/max(ring_rgb.max(),1)
        deferred = False
        if calibrated and chromatic <= .15 and not resolved:
            if not measurement_context or measurement_context.get('kind') != 'existing_inhand_interface':
                veto('unresolved_neutral_contrast')
                if candidate_sink is None: continue
                deferred = True
            # This is an appearance candidate, not an independently identified hole.
            # Only the caller's complete measured layout may admit its identity.
            detail['identityStatus'] = 'requires_interface_layout'
        if chromatic > .15 and colour_delta < .08 and (not len(recess) or np.median(recess) < tolerance):
            veto('shadow_chromaticity'); continue
        detail['reason'] = 'unresolved_neutral_contrast' if deferred else 'accepted'
        focal = shape[0]/(2*math.tan(math.radians(raw['fovy'])/2))
        position_sigma = float(raw['header'].get('odometry', {}).get('sigma', {}).get('pos', 0.))
        sigma = float((noise+float(plane_d[0])/focal)/abs(denom)+position_sigma)
        detail.update(pixelM=float(plane_d[0])/focal, uncertaintyM=sigma)
        candidate = dict(point=point.round(6).tolist(), uncertaintyM=round(sigma,7),
            pixel=origin.round(3).tolist(), spanPixels=spans.tolist(), normal=normal.round(6).tolist(),
            radiusM=detail['diameterM']/2 if calibrated else float(max(np.linalg.norm(plane[1]-point)*spans[0],
                              np.linalg.norm(plane[2]-point)*spans[1])/2),
            recessM=round(float(np.median(recess)),7) if len(recess) else None,
            coreValidFraction=round(float(valid_core.mean()),4), planeNoiseM=round(noise,7),
            depthEvidence='resolved_recess' if resolved else 'protrusion_veto_only', available=True,
            **(dict(evidenceKind='complete_contour', candidateSource='closed_contour_and_local_depth_plane',
                    identityStatus='requires_interface_layout') if measurement_context else {}))
        if deferred:
            candidate_sink.append(candidate)
        else:
            output.append(candidate)
    return output


def correspond(holes, frame, hole_id, occupied=(), diagnostics=None):
    """Match the saved hole layout to the detected holes as one rigid shift.

    The saved interface comes from the released part's estimated pose and can be off by
    about one hole spacing, so the nearest detected hole to one saved point is not
    evidence. Every pairing (saved hole, detected hole) proposes a shift of the whole
    layout; the shift that explains the most saved holes wins. The target is assigned
    only when all best shifts agree on the same free detected hole.
    """
    pair = next((p for p in frame['holePairs'] if p['bodyHole'] == hole_id), None)
    reference = next((h for h in frame['holes'] if pair and h['id'] == pair['partHole']), None)
    if reference is None:
        return None, None
    target = np.asarray(reference['point'], float)
    occupied = [np.asarray(p, float) for p in occupied]
    free = [bool(h.get('available')) and not any(np.linalg.norm(np.asarray(h['point'])-p) <= .006
            for p in occupied) for h in holes]
    if not any(free):
        return None, None
    saved = [np.asarray(h['point'], float) for h in frame['holes']]
    k = next(i for i, h in enumerate(frame['holes']) if h['id'] == reference['id'])
    seen = [np.asarray(h['point'], float) for h in holes]
    nearest = min(float(np.linalg.norm(seen[i]-target)) for i in range(len(holes)) if free[i])
    hypotheses = []
    for a, origin in enumerate(saved):
        for j, point in enumerate(seen):
            shift = point-origin
            used, assigned = set(), {}
            for b, other in enumerate(saved):
                moved = other+shift
                order = sorted((float(np.linalg.norm(q-moved)), i) for i, q in enumerate(seen) if i not in used)
                if order and order[0][0] <= .008+holes[order[0][1]]['uncertaintyM']:
                    used.add(order[0][1]); assigned[b] = order[0][1]
                elif any(np.linalg.norm(p-moved) <= .012 for p in occupied):
                    assigned[b] = None
            hypotheses.append((len(assigned), float(np.linalg.norm(shift)), assigned.get(k, -1)))
    def unresolved():
        # Recorded only for an unresolved match, so that a successful receipt keeps its size.
        if diagnostics is not None:
            diagnostics['correspondence'] = dict(
                savedPointsM=[p.round(4).tolist() for p in saved], targetIndex=int(k),
                detectedPointsM=[p.round(4).tolist() for p in seen], detectedFree=free,
                occupiedPointsM=[p.round(4).tolist() for p in occupied],
                hypotheses=[[int(n), round(d, 4), (-1 if c is None else int(c))]
                            for n, d, c in sorted(hypotheses, key=lambda h: (-h[0], h[1]))[:6]])
        return None, nearest
    if not hypotheses:
        return unresolved()
    best = max(h[0] for h in hypotheses)
    choices = {c for n, _, c in hypotheses if n == best and c is not None and c >= 0 and free[c]}
    if len(choices) != 1:
        return unresolved()
    i = choices.pop()
    return i, float(np.linalg.norm(seen[i]-target))
