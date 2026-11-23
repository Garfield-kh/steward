"""Same-frame interface correspondence; registered geometry is only a prior.

A translated layout needs an observed anchor. No anchor means no target, and
unresolved permutations stay unknown. Raised heads are separate from recesses.
"""
from copy import deepcopy
import math
import numpy as np
from steward.ego import wrist_observation as wo
from steward.ego.support_holes import components
from steward.ego.fasten_target import spacing


def prior(controller, current):
    reg = (getattr(controller, '_registered_alignment', {}) or {}).get('registration') or {}
    if current.get('body') and reg.get('body') != current['body']:
        candidates = [r for r in getattr(controller.rt.estimator, '_seat_registrations', {}).values()
                      if r.get('body') == current['body']]
        reg = candidates[0] if len(candidates) == 1 else {}
    frame = reg.get('frame') or (current.get('seatTarget') or {}).get('componentFrame') or {}
    rows = deepcopy(reg.get('layout') or [])
    if not rows:
        rows = [dict(h, bodyHole=p['bodyHole']) for p in frame.get('holePairs', [])
                for h in frame.get('holes', []) if h['id'] == p['partHole']]
    identity = current.get('seatHole') or current.get('hole') or reg.get('hole')
    occupied = []
    for a in getattr(controller.rt.estimator, 'fastening_attempts', {}).values():
        if (a.get('releaseExecuted') and a.get('part') == frame.get('entity')
                and a.get('body') == frame.get('body') and not a.get('departureEvidence')):
            occupied.append(a.get('hole'))
    if current.get('entryTargetKind') == 'head' and identity is not None and identity not in occupied:
        occupied.append(identity)
    return dict(layout=rows, targetHole=identity, occupiedHoles=occupied,
                targetKind=current.get('entryTargetKind', 'recess'))


def foreground(controller, sample, plane, opening):
    """Connected above-surface returns, preserving the head/shaft junction."""
    raw, cam = sample; est = controller.rt.estimator; h = raw['header']
    grip = est.gripper_position(est.held_gripper, h['joints'], h['odometry'])
    region = wo.window(raw, cam, grip, opening)
    depth = wo.local_median(raw['depth'], region)
    raw = dict(raw, depth=np.where(raw['depth'] > 0, depth, 0))
    p, y, x = wo.cloud(raw, cam, region)
    if not len(p): return [], None
    focal = depth.shape[0]/(2*math.tan(math.radians(raw['fovy'])/2))
    noise = max(2*float(np.median(depth[y,x]))/focal, float(plane.get('planeNoiseM', 0.)))
    point = np.asarray(plane['point']); normal = np.asarray(plane.get('normal', [0,0,1]))
    keep = (np.linalg.norm(p-grip, axis=1) < opening) & ((p-point) @ normal > 2*noise)
    with controller.rt._lock:
        old = est.joints, est.odometry
        try:
            est.joints, est.odometry = h['joints'], h['odometry']
            bodies = list(est.body_boxes())
        finally:
            est.joints, est.odometry = old
    for c, half, r in bodies:
        keep &= ~np.all(abs((p-c) @ r) <= np.asarray(half)+noise, axis=1)
    selected = np.zeros(depth.shape, bool); selected[y[keep], x[keep]] = True
    selected &= wo.local_median(selected, region).astype(bool)
    groups = []
    for pixels in components(selected):
        if len(pixels) < 30: continue
        mask = np.zeros_like(selected); mask[pixels[:,1], pixels[:,0]] = True
        # cloud() and this subset both retain valid pixels in row-major order.
        q = p[mask[y, x]]
        groups.append(dict(points=q, pixels=pixels, mask=mask))
    return groups, dict(noise=noise, grip=grip, bodies=bodies, raw=raw, region=region)


def circle(points, noise):
    """Observed arc curvature; reject straight edges and unresolved fits."""
    origin = points.mean(0); q = points-origin
    fit = np.linalg.lstsq(np.c_[2*q, np.ones(len(q))], np.sum(q*q,axis=1), rcond=None)[0]
    radius2 = fit[2]+fit[:2] @ fit[:2]
    if radius2 <= 0: return None
    radius = math.sqrt(radius2); errors = abs(np.linalg.norm(q-fit[:2],axis=1)-radius)
    angles = np.sort(np.arctan2(q[:,1]-fit[1], q[:,0]-fit[0]) % (2*math.pi))
    coverage = 2*math.pi-np.max(np.diff(np.r_[angles, angles[0]+2*math.pi]))
    if coverage < math.pi/3 or np.quantile(errors,.9) > noise or radius > np.linalg.norm(np.ptp(q,axis=0)):
        return None
    return origin+fit[:2], radius, float(np.quantile(errors,.9))


def short_lower(controller, sample, plane, opening, diagnostics, *, foreground_data=None, grip_radius=None):
    """A short shaft below a broad head need not dominate whole-object PCA."""
    groups, info = foreground(controller, sample, plane, opening) if foreground_data is None else foreground_data
    diagnostics['components'] = []; found = []
    grip_radius = opening/2 if grip_radius is None else min(opening, grip_radius)
    if info is None: return None
    noise = info['noise']; grip = info['grip']; normal = np.asarray(plane.get('normal', [0,0,1]))
    for group in groups:
        q = group['points']; lo, hi = np.quantile(q[:,2], [.02,.98])
        row = dict(points=len(q), verticalSpanM=float(hi-lo), reason='unresolved_lower_arc')
        diagnostics['components'].append(row)
        if np.min(np.linalg.norm(q-grip, axis=1)) >= grip_radius:
            row['reason'] = 'outside_grip_neighbourhood'; continue
        if hi-lo < 6*noise: continue
        fits = []
        for a,b in ((.05,.28),(.28,.51)):
            band = q[(q[:,2] >= lo+a*(hi-lo)) & (q[:,2] <= lo+b*(hi-lo))]
            if len(band) < 12: break
            f = circle(band[:,:2], noise)
            if f is None: break
            fits.append((*f, float(np.median(band[:,2]))))
        if len(fits) != 2: continue
        direction = (fits[1][0]-fits[0][0])/(fits[1][3]-fits[0][3])
        if np.linalg.norm(direction) > math.tan(math.pi/6): continue
        if abs(fits[1][1]-fits[0][1]) > noise: continue
        xy = fits[0][0]+direction*(lo-fits[0][3]); tip = np.r_[xy,lo]
        bottom = q[q[:,2] <= lo+2*noise]
        if len(bottom) and np.quantile(np.linalg.norm(bottom[:,:2]-xy,axis=1),.9) > max(f[1] for f in fits)+3*noise:
            row['reason'] = 'lower_end_hidden_by_broad_surface'; continue
        clearance = float((tip-plane['point']) @ normal)
        if clearance <= 3*noise:
            row['reason'] = 'surface_connected_or_occupied'; continue
        if any(np.all(abs((tip-c) @ r) <= np.asarray(half)+3*noise) for c,half,r in info['bodies']):
            row['reason'] = 'robot_cut'; continue
        pixels = group['pixels']; width=info['raw']['depth'].shape[1]; height=info['raw']['depth'].shape[0]
        if np.any(pixels.min(0) < 2) or np.any(pixels.max(0) >= [width-2,height-2]): continue
        if np.max(np.linalg.norm(q-grip,axis=1)) >= opening-2*noise: continue
        row['reason'] = 'measured'
        found.append(dict(tip=tip.tolist(), head=np.r_[xy+direction*(hi-lo),hi].tolist(),
            axis=np.r_[direction,1.].tolist(), shaftRadiusM=float(np.mean([f[1] for f in fits])),
            headRadiusM=float(np.quantile(np.linalg.norm(q[:,:2]-xy,axis=1),.98)),
            uncertaintyM=2*noise+max(f[2] for f in fits), source='same_frame_short_lower_arc'))
    diagnostics['resolved'] = len(found)
    return found[0] if len(found) == 1 else None


def associate(found, caps, registration, opening, diagnostics, plane=None):
    """Correspond by rigid layout; prior pose only resolves identity/symmetry."""
    from steward.ego.layout_pose import match, transform
    rows = registration.get('layout') or []; identity = registration.get('targetHole')
    wanted = next((r for r in rows if r.get('bodyHole') == identity), None)
    if wanted is None: return None
    occupied = registration.get('occupiedHoles', [])
    features = [dict(h,kind=k) for group,k in ((found,'recess'),(caps,'head')) for h in group]
    fitted, reason = match(rows,features,occupied)
    diagnostics['reason'] = reason
    if fitted is None: return None
    matches = [(rows[m['row']],features[m['feature']],m['kind']) for m in fitted['matches']]
    diagnostics.update(matches=fitted['matches'],layoutFit=fitted)
    uncertainty = fitted['uncertaintyM']
    kind = registration.get('targetKind', 'recess')
    direct = next((h for r,h,k in matches if r.get('bodyHole') == identity and k == kind), None)
    if direct is not None: return dict(direct, targetFromLayout=False, targetHole=identity)
    # A hidden raised head needs its observed height, not the underlying plane.
    if kind == 'head' or identity in occupied: return None
    anchor, observed, anchor_kind = matches[0]
    point = np.asarray(transform(wanted['point'],fitted))
    # Raised heads supply XY only; an empty recess supplies its measured plane.
    if anchor_kind == 'recess':
        point[2] += observed['point'][2]-anchor['point'][2]
    elif plane is not None:
        n=np.asarray(plane['normal']); origin=np.asarray(plane['point'])
        point[2]=origin[2]-((point-origin)[:2] @ n[:2])/n[2]
        uncertainty += plane.get('uncertaintyM',0.)
    return dict(point=point.tolist(), normal=observed.get('normal',[0,0,1]),
        uncertaintyM=uncertainty+wanted.get('uncertaintyM',0.),
        planeNoiseM=observed.get('planeNoiseM',0.), radiusM=wanted.get('radiusM',0.),
        targetFromLayout=True, targetHole=identity, anchors=diagnostics['matches'],
        source='same_frame_anchor_plus_registered_layout')


def surface_arcs(sample, target, opening, diagnostics, *, full_frame=False):
    """Recover visible recess arcs on a larger measured support plane.

    Only edges against observed support pixels enter the circle fit. The edge
    against a carried occluder is omitted, so a crescent is not fitted as a disk.
    """
    from steward.ego.wrist_surface import fit
    from steward.ego.support_holes import unproject
    raw, cam = sample
    region = np.ones(raw['depth'].shape,bool) if full_frame else wo.window(raw, cam, target, opening/2)
    plane = fit(raw, cam, region)
    if plane is None: return [], None
    normal = np.asarray(plane['normal']); point = np.asarray(plane['point'])
    if not full_frame and abs((np.asarray(target)-point) @ normal) > opening/4: return [], None
    p,y,x = wo.cloud(raw,cam,region)
    height,width = raw['depth'].shape
    intr = dict(width=width,height=height,fovy=raw['fovy'])
    focal=height/(2*math.tan(math.radians(raw['fovy'])/2))
    pixel=float(np.median(raw['depth'][y,x]))/focal
    noise=max(pixel, plane['residualM']/2)
    distance=(p-point) @ normal
    support=np.zeros(region.shape,bool);support[y[abs(distance)<2*noise],x[abs(distance)<2*noise]]=True
    rgb=raw['rgb'].astype(float); chroma=rgb/np.maximum(rgb.sum(2,keepdims=True),1.)
    colour=np.median(chroma[support],axis=0)
    contrast=np.linalg.norm(chroma-colour,axis=2)
    # Same generic reflectance change used by the native recess detector;
    # no colour identity or task-specific appearance is supplied.
    dark=(contrast>.08)&(rgb.mean(2)<np.median(rgb[support].mean(1))*.8)&region
    foreground_mask=np.zeros(region.shape,bool);foreground_mask[y[distance>3*noise],x[distance>3*noise]]=True
    dark &= ~foreground_mask
    surface=support & (contrast<.04)
    adjacent=np.zeros_like(region)
    for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
        adjacent |= np.roll(surface,(dy,dx),(0,1))
    basis=np.cross(normal,[1.,0,0]);basis/=np.linalg.norm(basis);basis=np.c_[basis,np.cross(normal,basis)]
    found=[];diagnostics['arcs']=[]
    for pixels in components(dark):
        if len(pixels)<24: continue
        boundary=pixels[adjacent[pixels[:,1],pixels[:,0]]]
        if len(boundary)<12: continue
        rays=unproject(boundary,np.ones(len(boundary)),cam,intr)-cam['p']
        denom=rays @ normal
        if np.any(abs(denom)<.06): continue
        world=np.asarray(cam['p'])+rays*((point-np.asarray(cam['p']))@normal/denom)[:,None]
        uv=(world-point) @ basis
        fitted=circle(uv,2*pixel)
        if fitted is None: continue
        centre,radius,error=fitted
        angles=np.sort(np.arctan2(*(uv-centre)[:,::-1].T) % (2*math.pi))
        coverage=2*math.pi-np.max(np.diff(np.r_[angles,angles[0]+2*math.pi]))
        if coverage<math.pi or radius<2*pixel or radius>opening/4: continue
        centre3=point+basis @ centre
        # Flat graphics with exact depth remain distinguishable from cavities.
        cy,cx=pixels[:,1],pixels[:,0]
        if plane['residualM'] < 1e-7:
            mask=np.zeros_like(region);mask[cy,cx]=True
            core=p[mask[y,x]]
            if len(core) and np.max(abs((core-point)@normal))<1e-7: continue
        row=dict(point=centre3.tolist(),normal=normal.tolist(),radiusM=radius,
            uncertaintyM=noise+2*pixel+error,planeNoiseM=noise,
            source='same_frame_support_boundary_arc',available=True)
        found.append(row)
        diagnostics['arcs'].append(dict(boundaryPixels=len(boundary),coverageRadians=float(coverage),
            radiusM=radius,fitErrorM=error))
    plane.update(planeNoiseM=noise,uncertaintyM=noise+2*pixel)
    return found,plane


def heads(controller, sample, plane, opening, held_feature=None, *, foreground_data=None):
    """Round raised plateaus, split by height even when silhouettes touch."""
    groups,info=foreground(controller,sample,plane,opening) if foreground_data is None else foreground_data
    if info is None:return []
    noise=info['noise'];found=[]
    # A head may touch the carried silhouette in the image. Search horizontal
    # plateaus rather than declaring the whole connected component occupied.
    for g in groups:
        p=g['points'];lo,hi=np.quantile(p[:,2],[.02,.98])
        candidates=[]
        yy,xx=np.where(g['mask'] & (info['raw']['depth']>0))
        if held_feature: hi=min(hi,held_feature['tip'][2]-held_feature['uncertaintyM'])
        for z in np.arange(lo,hi+noise,noise):
            band=abs(p[:,2]-z)<=noise
            mask=np.zeros(info['raw']['depth'].shape,bool)
            mask[yy[band],xx[band]]=True
            for component in components(mask):
                if len(component)<24:continue
                local=np.zeros_like(mask);local[component[:,1],component[:,0]]=True
                cap=p[local[yy,xx]]
                centre=np.median(cap,axis=0);xy=cap[:,:2]-centre[:2]
                _,sv,_=np.linalg.svd(xy,full_matrices=False)
                radial=np.linalg.norm(xy,axis=1);radius=float(np.quantile(radial,.95))
                if sv[0]>1.8*sv[1] or radius<2*noise or np.mean(radial<radius/2)<.1:continue
                # The underlying surface must still be below this plateau.
                if float((centre-plane['point'])@np.asarray(plane['normal']))<=2*noise:continue
                if held_feature:
                    tip=np.asarray(held_feature['tip']);head=np.asarray(held_feature['head'])
                    if centre[2]>=tip[2]-held_feature['uncertaintyM'] and np.linalg.norm(centre[:2]-head[:2])<radius+held_feature.get('headRadiusM',0.)+3*noise:continue
                # Require an actual horizontal plateau, not a thin height slice
                # through a curved side. The plane must explain its local pixels.
                design=np.c_[xy,np.ones(len(xy))]
                fit=np.linalg.lstsq(design,cap[:,2]-centre[2],rcond=None)[0]
                if np.linalg.norm(fit[:2])>math.tan(math.pi/6):continue
                candidates.append((len(cap),dict(point=centre.tolist(),normal=plane['normal'],radiusM=radius,
                    uncertaintyM=3*noise,planeNoiseM=noise,source='same_frame_occupied_head')))
        for _,cap in sorted(candidates,key=lambda x:-x[0]):
            if not any(np.linalg.norm(np.asarray(cap['point'])-h['point'])<cap['radiusM']+h['radiusM'] for h in found):found.append(cap)
    return found


def occluded_by_load(sample, entry, feature):
    """The target ray must actually hit the measured carried surface."""
    from steward.ego.support_holes import project, unproject
    raw,cam=sample; height,width=raw['depth'].shape
    intr=dict(width=width,height=height,fovy=raw['fovy'])
    uv,z=project([entry['point']],cam,intr)
    x,y=np.rint(uv[0]).astype(int)
    if not (0<=x<width and 0<=y<height):return False
    depth=float(raw['depth'][y,x]);noise=feature['uncertaintyM']+entry.get('uncertaintyM',0.)
    if depth<=0 or z[0]-depth<=noise:return False
    p=unproject([[x,y]],[depth],cam,intr)[0]
    tip=np.asarray(feature['tip']);head=np.asarray(feature['head']);axis=head-tip
    t=float((p-tip)@axis/max(axis@axis,1e-12))
    radius=feature.get('headRadiusM',feature.get('shaftRadiusM',0.))
    return bool(-.1<=t<=1.1 and np.linalg.norm(p-(tip+np.clip(t,0,1)*axis))<=radius+noise)
