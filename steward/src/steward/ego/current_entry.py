"""Current visible face and entrance correspondence for pre-release fallback.

Memory supplies identities and relative layout, never a veto on a measured
entrance. A face is fitted from image rims, not from the largest support below
it. No scene labels, object dimensions or simulator state are read here.
"""
import math
import numpy as np
from steward.ego import capabilities, wrist_observation as wo
from steward.ego.support_holes import components, unproject
from steward.ego.wrist_surface import fit
from steward.ego.entry_layout import circle


def enabled():
    return capabilities.get('STEWARD_ASSEMBLY_CURRENT_ENTRY') != '0'


def split_level(values):
    """Otsu split of observed dark returns separates recesses from cast shadows."""
    hist, edges = np.histogram(values, bins=64)
    centres = (edges[:-1]+edges[1:])/2
    weight = np.cumsum(hist); moment = np.cumsum(hist*centres)
    valid = (weight > 0) & (weight < weight[-1])
    scores = np.zeros(len(hist))
    scores[valid] = (moment[-1]*weight[valid]-moment[valid]*weight[-1])**2/(weight[valid]*(weight[-1]-weight[valid]))
    return float(edges[int(np.argmax(scores))+1])


def unique(features):
    """Merge repeat fits only at the same plane and within sensor resolution."""
    result = []
    for h in sorted(features, key=lambda h:h['uncertaintyM']):
        if any(np.linalg.norm(np.asarray(h['point'])-q['point']) <=
               max(h['uncertaintyM'], q['uncertaintyM']) for q in result):
            continue
        result.append(h)
    return result


def faces(sample, target, opening, registration, diagnostics, *, visible_pair=False):
    raw, cam = sample; depth = raw['depth']; height, width = depth.shape
    region = wo.window(raw, cam, target, opening)
    for r in registration['layout']:
        region |= wo.window(raw, cam, r['point'], opening)
    # A separate local search prevents the union window's morphology scale
    # from joining an entrance to a neighbouring shadow.
    trace = dict(candidates=[])
    wo.holes(sample, target, opening/2, diagnostics=trace)
    diagnostics['localDetector'] = trace
    yy, xx = np.indices(depth.shape)
    p, y, x = wo.cloud(raw, cam, region)
    if len(p) < 48: return []
    focal = height/(2*math.tan(math.radians(raw['fovy'])/2))
    pixel = float(np.median(depth[y,x]))/focal
    rgb = raw['rgb'].astype(float); gray = rgb.mean(2)
    chroma = rgb/np.maximum(rgb.sum(2,keepdims=True), 1.)
    intr = dict(width=width, height=height, fovy=raw['fovy'])
    output = []; diagnostics['faces'] = []
    # Legacy face seeds need a complete rim. Pair recovery also searches around
    # merged dark contours; the independent plane and two arcs must still fit.
    seeds = [r for r in trace['candidates'] if r.get('reason') in
             ('accepted', 'unresolved_neutral_contrast')
             or (visible_pair and min(r.get('spanPixels', [0])) >= 4)]
    for seed in seeds:
        centre = np.asarray(seed['pixel']); span = np.asarray(seed['spanPixels'])
        rect = (abs(xx-centre[0]) < span[0]) & (abs(yy-centre[1]) < span[1])
        ring = rect & ((abs(xx-centre[0]) > span[0]/2) | (abs(yy-centre[1]) > span[1]/2)) & region
        if visible_pair and ring.any():
            ring &= gray >= np.quantile(gray[ring], .6)
        plane = fit(raw, cam, ring)
        if plane is None: continue
        point = np.asarray(plane['point']); normal = np.asarray(plane['normal'])
        noise = max(pixel, plane['residualM'])
        distance = (p-point) @ normal
        on = np.zeros(depth.shape,bool); on[y,x] = abs(distance) < 2*noise
        if (ring & on).sum() < 24: continue
        reflectance = np.median(rgb[ring & on],axis=0)
        face = on & (np.linalg.norm(chroma-reflectance/max(reflectance.sum(),1),axis=2) < .04)
        bright = face & (gray > reflectance.mean()*.65)
        if visible_pair:
            shaded = gray[rect & region & (gray < reflectance.mean()*.65)]
            if len(shaded) >= 24:
                # A cast shadow remains a measured coplanar rim. The same
                # frame's intensity split separates it from the cavity core.
                bright = face & (gray > split_level(shaded))
        groups = [g for g in components(bright) if len(g) >= 48]
        if not groups: continue
        # Select the continuous rim-bearing surface, not a larger lower plane.
        pixels = max(groups, key=lambda g:int(ring[g[:,1],g[:,0]].sum()))
        surface = np.zeros(depth.shape,bool); surface[pixels[:,1],pixels[:,0]] = True
        surface_points = p[surface[y,x]]
        if len(surface_points) < 48: continue
        centre3 = surface_points.mean(0)
        _, singular, axes = np.linalg.svd(surface_points[:,:2]-centre3[:2],full_matrices=False)
        if singular[1] <= 3*pixel: continue
        axis = axes[0]
        projected = (surface_points[:,:2]-centre3[:2]) @ axes.T
        lo, hi = np.quantile(projected,[.01,.99],axis=0)
        outline_centre = centre3[:2]+((lo+hi)/2) @ axes
        extent = hi-lo
        outline = dict(centre=outline_centre.tolist(),axis=axis.tolist(),extent=extent.tolist(),
                       uncertaintyM=2*noise,elongation=float(extent[0]/max(extent[1],pixel)))
        adjacent = np.zeros(depth.shape,bool)
        for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
            adjacent |= np.roll(surface,(dy,dx),(0,1))
        # Keep the support noise band for topology, but exclude raised load
        # pixels from every curve. Missing core depth may still be a recess.
        not_raised = region.copy(); not_raised[y,x] = distance <= 3*noise
        dark = not_raised & (gray < reflectance.mean()*.65)
        values = gray[dark & rect]
        if len(values) < 24: continue
        level = split_level(values)
        dark &= gray <= level
        row = dict(plane=plane,outline=outline,splitIntensity=level,entrances=[])
        diagnostics['faces'].append(row)
        holes = []
        for candidate in components(dark):
            if len(candidate) < 24: continue
            boundary = candidate[adjacent[candidate[:,1],candidate[:,0]]]
            if len(boundary) < 12: continue
            rays = unproject(boundary,np.ones(len(boundary)),cam,intr)-cam['p']
            denom = rays @ normal
            if np.any(abs(denom) < .06): continue
            world = np.asarray(cam['p'])+rays*(((point-cam['p']) @ normal)/denom)[:,None]
            basis = np.cross(normal,[1.,0,0]); basis /= np.linalg.norm(basis)
            basis = np.c_[basis,np.cross(normal,basis)]
            uv = (world-point) @ basis
            fitted = circle(uv,2*pixel)
            if fitted is None: continue
            centre2, radius, error = fitted
            angles = np.sort(np.arctan2(*(uv-centre2)[:,::-1].T) % (2*math.pi))
            coverage = 2*math.pi-np.diff(np.r_[angles,angles[0]+2*math.pi]).max()
            if coverage < (math.pi/2 if visible_pair else math.pi) or radius < 2*pixel or radius > opening/4: continue
            centre_h = point+basis @ centre2
            local = (centre_h[:2]-outline_centre) @ axes.T
            if not visible_pair and np.any(abs(local)+radius > extent/2+2*noise): continue
            core_mask = np.zeros(depth.shape,bool); core_mask[candidate[:,1],candidate[:,0]]=True
            core = p[core_mask[y,x]]
            if len(core):
                gaps = (core-point) @ normal
                if np.quantile(gaps,.8) > 3*noise: continue
                if plane['residualM'] < 1e-7 and np.max(abs(gaps)) < 1e-7: continue
            hole = dict(point=centre_h.tolist(),normal=normal.tolist(),radiusM=float(radius),
                        uncertaintyM=noise+2*pixel+error,planeNoiseM=noise,
                        source='same_frame_face_boundary_arc',available=True)
            if visible_pair:
                hole.update(arcCoverageRadians=float(coverage),
                    uncertaintyM=(noise+2*pixel+error)/math.sin(min(coverage, math.pi)/2)**2)
            holes.append(hole)
            row['entrances'].append(dict(pixel=candidate.mean(0).tolist(),boundaryPixels=len(boundary),
                coverageRadians=float(coverage),fitErrorM=error,**hole))
        holes = unique(holes)
        if holes: output.append((holes, outline, plane))
    return output


def correspond(rows, holes, outline, identity, diagnostics):
    """Use current face axis for single-entrance identity and current pair order.

    The prior chooses only the sign of an undirected outline axis. Every direct
    point is a measured entrance, even if the old spacing is inconsistent.
    """
    if len(rows) != 2 or not holes or len(holes) > 2: return None
    p = np.array([r['point'][:2] for r in rows]); span = np.linalg.norm(p[1]-p[0])
    if span <= outline['uncertaintyM']: return None
    axis = np.asarray(outline['axis'])
    if axis @ (p[1]-p[0]) < 0: axis = -axis
    if outline['elongation'] < 2: return None
    q = np.array([h['point'][:2] for h in holes])
    order = np.argsort((q-outline['centre']) @ axis)
    if len(holes) == 2:
        # Cross-axis alignment is checked against the current face, not old yaw.
        across = abs(np.cross(q[1]-q[0],axis))
        limit = sum(h['uncertaintyM'] for h in holes)+outline['uncertaintyM']
        if across > limit: return None
        pairs = [(i,int(j)) for i,j in enumerate(order)]
        diagnostics.update(observedSpacingM=float(np.linalg.norm(q[1]-q[0])),
            priorSpacingM=float(span),crossAxisM=float(across),crossAxisLimitM=limit)
    else:
        position = float((q[0]-outline['centre']) @ axis)
        if abs(position) <= holes[0]['uncertaintyM']+outline['uncertaintyM']:
            return None
        pairs = [(int(position > 0),0)]
    match = next((j for i,j in pairs if rows[i]['bodyHole']==identity),None)
    diagnostics.update(matches=[dict(hole=rows[i]['bodyHole'],feature=j,point=holes[j]['point']) for i,j in pairs],
        identityBasis='current_face_axis_and_visible_entrances',priorUsedOnlyForAxisSign=True)
    if match is not None:
        return dict(holes[match],targetFromLayout=False,targetHole=identity)
    # With one opening visible, use its anchor and the CURRENT outline heading.
    i,j = pairs[0]; target_i = next((k for k,r in enumerate(rows) if r['bodyHole']==identity),None)
    if target_i is None: return None
    point = np.asarray(holes[j]['point']).copy();point[:2] += (target_i-i)*span*axis
    diagnostics['inferredSpacingFromPrior'] = True
    return dict(holes[j],point=point.tolist(),targetFromLayout=True,targetHole=identity,
        uncertaintyM=holes[j]['uncertaintyM']+outline['uncertaintyM'],
        source='same_frame_entrance_and_face_axis_plus_layout_spacing')


def recover(sample, target, opening, registration, diagnostics):
    diagnostics['reason'] = 'current_face_entrances_unresolved'
    choices = []
    for holes, outline, plane in faces(sample,target,opening,registration,diagnostics):
        detail = {}
        entry = correspond(registration['layout'],holes,outline,registration.get('targetHole'),detail)
        # Only a directly visible empty entrance can contradict occupancy.
        # An unseen target inferred from its sibling cannot clear that memory.
        if (entry is not None and entry.get('targetFromLayout')
                and registration.get('targetHole') in registration.get('occupiedHoles', [])):
            continue
        if entry is not None: choices.append((entry,detail))
    if not choices: return None
    entries = unique([c[0] for c in choices])
    if len(entries) != 1:
        diagnostics['reason'] = 'multiple_current_faces'; return None
    entry, detail = min(choices,key=lambda c:c[0]['uncertaintyM'])
    diagnostics.update(detail,reason=None)
    return entry


def local_refinement(rows, fitted, opening):
    """A small hover update belongs to the existing differential servo.

    This is a routing decision, not motion permission. The servo still measures
    tip/entry clearance and checks its bounded path before executing each step.
    """
    from steward.ego.layout_pose import transform
    if not rows or not opening or opening <= 0: return None
    shifts = [float(np.linalg.norm(np.asarray(transform(r['point'],fitted))[:2]-r['point'][:2]))
              for r in rows]
    limit = opening/8
    if max(shifts) > limit: return None
    return dict(maxTargetShiftM=max(shifts),localStepLimitM=limit,
                source='current_frame_layout',motionAdmission='tip_entry_clearance_and_reachable_path')


def edge_refinement(controller, params, rows, fitted, opening):
    """Route only a local update clipped by the perceived support footprint.

    This reads the same belief box as the planner, not simulator geometry.
    Other planning failures retain their original handling.
    """
    local = local_refinement(rows, fitted, opening)
    if local is None: return None
    ego_state = getattr(controller.rt, 'ego_state', None)
    if not callable(ego_state): return None
    support = ego_state().find(params.get('on'))
    if support is None: return None
    wanted = next((r for r in rows if r.get('bodyHole')==params.get('seatHole')),None)
    if wanted is None: return None
    from steward.ego.planner import entity_box
    from steward.ego.layout_pose import transform, rotation
    box = entity_box(support)[0]
    xy = np.asarray(transform(wanted['point'], fitted))[:2]
    offset = rotation(box.yaw).T @ (xy-[box.cx,box.cy])
    margin = float(np.min(np.array([box.hx,box.hy])-abs(offset)))
    if margin >= 0: return None
    return dict(local, perceivedSupportMarginM=margin, transportEdgeKeepM=0.)
