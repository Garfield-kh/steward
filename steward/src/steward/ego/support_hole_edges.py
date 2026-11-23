"""Image-supported partial rims, conditional on an existing in-hand interface.

An independently measured complete rim supplies a plane/radius search scale.
Neither catalogue spacing nor a predicted second centre creates edge evidence.
Candidates remain unclassified until the caller admits a unique rigid layout.
"""
import math
import numpy as np
from PIL import Image, ImageFilter
from steward.ego import support_holes as sh
from steward.ego.perception import PerceptionConfig


def edge_pixels(rgb, region, threshold_scale=1.):
    yy,xx = np.where(region)
    if not len(xx): return np.empty((0,2),int),np.empty((0,2))
    lo = np.maximum([xx.min()-2,yy.min()-2],0)
    hi = np.minimum([xx.max()+3,yy.max()+3],region.shape[::-1])
    rgb = rgb[lo[1]:hi[1],lo[0]:hi[0]]
    region = region[lo[1]:hi[1],lo[0]:hi[0]]
    gray = rgb.mean(2).astype('uint8').astype(float)
    padded = np.pad(gray, 1, mode='edge')
    gx = (padded[:-2,2:]+2*padded[1:-1,2:]+padded[2:,2:]
          -padded[:-2,:-2]-2*padded[1:-1,:-2]-padded[2:,:-2])
    gy = (padded[2:,:-2]+2*padded[2:,1:-1]+padded[2:,2:]
          -padded[:-2,:-2]-2*padded[:-2,1:-1]-padded[:-2,2:])
    magnitude = abs(gx)+abs(gy)
    # Four-direction nonmaximum suppression followed by local hysteresis.
    direction = np.rint(np.arctan2(gy,gx)/(math.pi/4)).astype(int) % 4
    keep = np.zeros(gray.shape,bool)
    for k,(dy,dx) in enumerate(((0,1),(1,1),(1,0),(1,-1))):
        before = np.roll(magnitude,(dy,dx),(0,1))
        after = np.roll(magnitude,(-dy,-dx),(0,1))
        keep |= (direction==k)&(magnitude>before)&(magnitude>=after)
    keep &= region & (magnitude>30*threshold_scale)
    keep[[0,-1],:] = False; keep[:,[0,-1]] = False
    strong = keep & (magnitude>80*threshold_scale)
    # Flood only edge runs; repeated full-raster dilation would make long
    # connected shadows expensive and could consume the capture age budget.
    connected = np.zeros_like(keep)
    for pixels in sh.components(keep):
        x,y = pixels.T
        if strong[y,x].any(): connected[y,x] = True
    strong = connected
    y,x = np.where(strong)
    return np.c_[x,y]+lo, np.c_[gx[y,x],gy[y,x]]


def circle(points):
    origin = points.mean(0)
    p = points-origin
    a = np.c_[2*p, np.ones(len(p))]
    if np.linalg.matrix_rank(a) < 3: return None
    coef = np.linalg.lstsq(a, (p*p).sum(1), rcond=None)[0]
    radius2 = coef[2]+coef[:2] @ coef[:2]
    if radius2 <= 0: return None
    return origin+coef[:2], math.sqrt(radius2)


def arc_support(points, gradients, centre, radius, tolerance):
    vectors = points-centre
    length = np.linalg.norm(vectors, axis=1)
    keep = (abs(length-radius) <= tolerance) & ((gradients*vectors).sum(1)/np.maximum(length,1e-12) > .7)
    return keep


def arc_angles(points, centre):
    angles = np.sort(np.mod(np.arctan2(*(points-centre)[:,::-1].T), math.tau))
    gaps = np.diff(np.r_[angles, angles[0]+math.tau])
    gap = int(np.argmax(gaps)); start = angles[(gap+1) % len(angles)]
    coverage = math.tau-gaps[gap]
    return float(start), float(start+coverage), float(coverage)


def visible_segments(points, centre, radius, pixel_m):
    start, end, _ = arc_angles(points, centre)
    angles = np.mod(np.arctan2(*(points-centre)[:,::-1].T)-start, math.tau)+start
    # Floating-point wrap at the first angle must not create a spurious turn.
    angles[angles > end+1e-8] -= math.tau
    angles.sort()
    breaks = np.flatnonzero(np.diff(angles) > 3*pixel_m/radius)+1
    return [[float(group[0]),float(group[-1])] for group in np.split(angles,breaks) if len(group)>1]


def uncertainty(points, centre, radius, pixel_m, noise, cosine):
    """Separate centre/radius conditioning and correlated edge/plane errors.

    Delete angular sectors to expose partial-arc extrapolation instability.
    The residual alone is never reported as centre accuracy.
    """
    radial = points-centre
    length = np.linalg.norm(radial, axis=1)
    design = np.c_[radial/np.maximum(length[:,None], 1e-12), np.ones(len(points))]
    covariance = np.linalg.pinv(design.T @ design)
    residual = float(np.sqrt(np.mean((length-radius)**2)))
    random = 2*max(residual, pixel_m/math.sqrt(12))*math.sqrt(float(np.linalg.eigvalsh(covariance[:2,:2])[-1]))
    angle = np.mod(np.arctan2(radial[:,1],radial[:,0]), math.tau)
    changes = []
    for sector in range(8):
        keep = (angle < sector*math.tau/8) | (angle >= (sector+1)*math.tau/8)
        if keep.sum() < 12: continue
        fit = circle(points[keep])
        if fit is not None: changes.append(float(np.linalg.norm(fit[0]-centre)))
    stability = max(changes, default=math.inf)
    plane = noise*.5/max(cosine, 1e-12)
    relative = .5*pixel_m+max(random,stability)+plane
    return dict(method='partial_arc_conditioning_sector_jackknife', relativeM=relative,
        fitRmsM=residual, centreRandomBoundM=random, sectorJackknifeM=stability,
        rasterBiasM=.5*pixel_m, planePredictionM=plane,
        conditionNumber=float(np.linalg.cond(design)), independentCentreEstimate=True)


def extract(raw, cam, region, anchors, diagnostics, *, threshold_scale=1.):
    """Deterministic edge proposals on locally verified measured planes."""
    if not anchors: return []
    intr = dict(width=raw['depth'].shape[1], height=raw['depth'].shape[0], fovy=raw['fovy'])
    uv_all, gradient_all = edge_pixels(raw['rgb'], region, threshold_scale)
    if len(uv_all) < 12: return []
    d = raw['depth'][uv_all[:,1],uv_all[:,0]]
    valid = np.isfinite(d) & (d > 0) & (d < PerceptionConfig().max_range)
    uv_all, gradient_all, d = uv_all[valid], gradient_all[valid], d[valid]
    actual = sh.unproject(uv_all,d,cam,intr)
    output = []
    def reject(reason): diagnostics[reason] = diagnostics.get(reason,0)+1
    for anchor in anchors:
        normal = np.asarray(anchor['normal']); origin = np.asarray(anchor['point'])
        radius0 = anchor['radiusM']; noise = anchor['planeNoiseM']
        tolerance_depth = max(.0015, 3*noise)
        selected = (abs((actual-origin) @ normal) <= tolerance_depth) & (np.linalg.norm(actual[:,:2]-origin[:2],axis=1) <= sh.MAX_LAYOUT_SHIFT)
        uv, gradient = uv_all[selected], gradient_all[selected]
        if len(uv) < 12: continue
        rays = sh.unproject(uv,np.ones(len(uv)),cam,intr)-cam['p']
        denominator = rays @ normal
        if np.any(abs(denominator) < .06): continue
        depth = ((origin-cam['p']) @ normal)/denominator
        points = np.asarray(cam['p'])+depth[:,None]*rays
        e1 = np.cross(normal,[0,1,0]); e1 /= np.linalg.norm(e1)
        basis = np.array([e1,np.cross(normal,e1)])
        p = (points-origin) @ basis.T
        jac = np.stack([(sh.project(points+v*1e-5,cam,intr)[0]-uv)/1e-5 for v in basis],axis=2)
        gradient = np.einsum('ni,nij->nj',gradient,jac)
        gradient /= np.maximum(np.linalg.norm(gradient,axis=1)[:,None],1e-12)
        distance = float(np.linalg.norm(origin-cam['p']))
        cosine = abs(float((origin-cam['p']) @ normal))/distance
        focal = intr['height']/(2*math.tan(math.radians(intr['fovy'])/2))
        pixel_m = distance/focal
        tolerance = pixel_m*.9
        # Low-discrepancy triples cover the observed edges without random seeds.
        index = np.arange(1,4097)[:,None]
        samples = np.floor(np.mod(index*np.sqrt([2.,3.,5.]),1)*len(p)).astype(int)
        proposals = []
        # Gradient-centre votes retain small arcs even when large unrelated
        # contours dominate the edge population. The scale is image-measured.
        radii = np.arange(.6*radius0, 1.5*radius0+tolerance, tolerance)
        votes = (p[:,None,:]-radii[None,:,None]*gradient[:,None,:]).reshape(-1,2)
        vote_radii = np.tile(radii,len(p))
        cells, inverse, counts = np.unique(np.floor(votes/(2*tolerance)).astype(int),
                                         axis=0, return_inverse=True, return_counts=True)
        for cell in np.argsort(counts)[-128:][::-1]:
            if counts[cell] < 6: break
            members = inverse==cell
            centre = np.mean(votes[members],axis=0)
            radius = float(np.median(vote_radii[members]))
            keep = arc_support(p,gradient,centre,radius,tolerance*1.5)
            if keep.sum() < 12: continue
            fit = circle(p[keep])
            if fit is None: continue
            centre,radius = fit
            keep = arc_support(p,gradient,centre,radius,tolerance)
            if keep.sum() >= 12 and arc_angles(p[keep],centre)[2] >= math.pi:
                proposals.append((int(keep.sum()),centre,radius,keep))
        triples = p[samples]
        a = 2*(triples[:,1:]-triples[:,:1])
        rhs = (triples[:,1:]**2).sum(2)-(triples[:,:1]**2).sum(2)
        det = a[:,0,0]*a[:,1,1]-a[:,0,1]*a[:,1,0]
        valid = abs(det) >= pixel_m**2*.01
        a,rhs,det,triples = a[valid],rhs[valid],det[valid],triples[valid]
        centres = np.c_[rhs[:,0]*a[:,1,1]-a[:,0,1]*rhs[:,1],
                        a[:,0,0]*rhs[:,1]-rhs[:,0]*a[:,1,0]]/det[:,None]
        radii = np.linalg.norm(centres-triples[:,0],axis=1)
        valid = (radii>.6*radius0)&(radii<1.5*radius0)
        for centre,radius in zip(centres[valid],radii[valid]):
            keep = arc_support(p,gradient,centre,radius,tolerance)
            if keep.sum() < 12: continue
            if arc_angles(p[keep],centre)[2] < math.pi: continue
            proposals.append((int(keep.sum()),centre,radius,keep))
        chosen = []
        for count, centre, radius, keep in sorted(proposals,key=lambda row:-row[0]):
            if any(np.linalg.norm(centre-c) < radius0 for c in chosen): continue
            for _ in range(3):
                fit = circle(p[keep])
                if fit is None: break
                centre,radius = fit
                keep = arc_support(p,gradient,centre,radius,tolerance)
                if keep.sum() < 12: break
            if fit is None or keep.sum() < 12: continue
            start,end,coverage = arc_angles(p[keep],centre)
            if coverage < math.pi or not .6*radius0 < radius < 1.5*radius0: continue
            chosen.append(centre)
            point = origin+centre @ basis
            if any(np.linalg.norm(point-np.asarray(h['point'])) < radius0 for h in anchors+output): continue
            # Verify local outward rim depth at this arc, not merely at the anchor.
            radial = (p[keep]-centre)/np.linalg.norm(p[keep]-centre,axis=1)[:,None]
            outward = points[keep]+(radial @ basis)*(2*pixel_m)
            rim_uv = np.rint(sh.project(outward,cam,intr)[0]).astype(int)
            inside = (rim_uv[:,0]>=0)&(rim_uv[:,0]<intr['width'])&(rim_uv[:,1]>=0)&(rim_uv[:,1]<intr['height'])
            if not inside.all(): reject('arc_rim_outside_image'); continue
            rim_d = raw['depth'][rim_uv[:,1],rim_uv[:,0]]
            valid = np.isfinite(rim_d)&(rim_d>0)&(rim_d<PerceptionConfig().max_range)&region[rim_uv[:,1],rim_uv[:,0]]
            if valid.mean()<.8: reject('arc_invalid_rim_depth'); continue
            rim = sh.unproject(rim_uv[valid],rim_d[valid],cam,intr)
            error = (rim-origin) @ normal
            if np.mean(abs(error)<=tolerance_depth)<.8: reject('arc_wrong_plane_or_protrusion'); continue
            # Shadows on chromatic surfaces preserve channel ratios.
            inward_uv = np.rint(sh.project(points[keep]-(radial @ basis)*(2*pixel_m),cam,intr)[0]).astype(int)
            inward_uv[:,0] = np.clip(inward_uv[:,0],0,intr['width']-1)
            inward_uv[:,1] = np.clip(inward_uv[:,1],0,intr['height']-1)
            outer_rgb = np.median(raw['rgb'][rim_uv[:,1],rim_uv[:,0]],axis=0).astype(float)
            inner_rgb = np.median(raw['rgb'][inward_uv[:,1],inward_uv[:,0]],axis=0).astype(float)
            chromatic = np.ptp(outer_rgb)/max(outer_rgb.max(),1)
            delta = np.linalg.norm(outer_rgb/max(outer_rgb.sum(),1)-inner_rgb/max(inner_rgb.sum(),1))
            if inner_rgb.mean() >= outer_rgb.mean()*.8 or (chromatic>.15 and delta<.08):
                reject('arc_shadow_or_wrong_gradient'); continue
            # Valid core pixels must not demonstrate a raised cap.
            core_d = raw['depth'][inward_uv[:,1],inward_uv[:,0]]
            core_valid = np.isfinite(core_d)&(core_d>0)&(core_d<PerceptionConfig().max_range)
            if core_valid.any():
                core = sh.unproject(inward_uv[core_valid],core_d[core_valid],cam,intr)
                protrusion = (core-origin) @ normal
                if np.quantile(protrusion,.8)>tolerance_depth or np.median(protrusion)>.0015:
                    reject('arc_protrusion'); continue
                if noise<1e-7 and np.max(abs(protrusion))<1e-7:
                    reject('arc_flat_marking'); continue
            estimate = uncertainty(p[keep],centre,radius,pixel_m/max(cosine,1e-12),noise,cosine)
            pixel = sh.project([point],cam,intr)[0][0]
            support_uv = uv[keep]
            common = float(raw['header'].get('odometry',{}).get('sigma',{}).get('pos',0.))
            output.append(dict(point=point.tolist(),pixel=pixel.tolist(),normal=normal.tolist(),
                spanPixels=(np.ptp(support_uv,axis=0)+1).tolist(), radiusM=radius,
                uncertaintyM=common+estimate['relativeM'], centreUncertainty=estimate,
                planeNoiseM=noise, available=True, evidenceKind='partial_hole_edge',
                candidateSource='local_image_edges_on_measured_plane',
                identityStatus='requires_interface_layout', depthEvidence='protrusion_veto_only',
                visibleArcRangeRad=[start,end], visibleArcDegrees=math.degrees(coverage),
                visibleArcSegmentsRad=visible_segments(p[keep],centre,radius,pixel_m),
                measuredPlaneAnchor=dict(point=anchor['point'],normal=anchor['normal'],
                                         pixel=anchor['pixel']),
                edgeSupportCount=int(keep.sum()), edgePixels=support_uv.tolist(),
                localRimSupportCount=int(valid.sum()), localRimInlierFraction=float(np.mean(abs(error)<=tolerance_depth)),
                candidateAmbiguity=dict(identity='pending_layout', samePlaneProposals=len(proposals))))
    return output
