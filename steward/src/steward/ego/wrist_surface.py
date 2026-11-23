"""Local wrist bearing face: calibrated rays, observed extent, gravity in world frame."""
import math
import numpy as np
from steward.ego.support_holes import unproject


def fit(raw, cam, region, diagnostics=None):
    diag = diagnostics if diagnostics is not None else {}
    depth = raw['depth']; h, w = depth.shape
    valid = region & np.isfinite(depth) & (depth > 0)
    y, x = np.where(valid)
    diag['validPoints'] = len(x)
    if len(x) < 48:
        diag['reason'] = 'insufficient_depth'; return None
    intr = dict(width=w, height=h, fovy=raw['fovy'])
    # Fixed computation bound, independent of native resolution.
    index = np.linspace(0, len(x)-1, min(len(x), 4096), dtype=int)
    uv = np.c_[x[index], y[index]]; d = depth[y[index], x[index]]
    p = unproject(uv, d, cam, intr)
    focal = h/(2*math.tan(math.radians(raw['fovy'])/2))
    pixel = float(np.median(d)/focal)
    band = max(2*pixel, .0015)  # existing depth precision floor, in metres
    diag.update(pixelM=pixel, bandM=band, rejections={})
    design = np.c_[uv-uv.mean(0), np.ones(len(uv))]
    candidates = []
    # Whole ROI plus overlapping image tiles; no random seed or part template.
    masks = [np.ones(len(uv), bool)]
    for u in np.linspace(x.min(), x.max(), 5):
        for v in np.linspace(y.min(), y.max(), 5):
            masks.append((abs(uv[:,0]-u)<=max(4,np.ptp(x)/4)) &
                         (abs(uv[:,1]-v)<=max(4,np.ptp(y)/4)))
    for mask in masks:
        if mask.sum() < 24: continue
        coef = np.linalg.lstsq(design[mask], 1/d[mask], rcond=None)[0]
        # Robust optical fit; evaluate orthogonal residual in world afterwards.
        for _ in range(3):
            pred = design @ coef
            good = (pred>0) & (abs(1/np.maximum(pred,1e-12)-d)<band)
            if good.sum() < 24: break
            coef = np.linalg.lstsq(design[good], 1/d[good], rcond=None)[0]
        if good.sum() < 48: continue
        centre = uv[good].mean(0)
        probes = np.array([centre,centre+[1,0],centre+[0,1]])
        zd = 1/(np.c_[probes-uv.mean(0),np.ones(3)] @ coef)
        if not np.all(np.isfinite(zd) & (zd>0)): continue
        plane = unproject(probes, zd, cam, intr)
        normal = np.cross(plane[1]-plane[0],plane[2]-plane[0]);normal/=np.linalg.norm(normal)
        if normal[2]<0: normal=-normal
        error = abs((p-plane[0]) @ normal)
        keep = error<band
        row = dict(inlierFraction=float(keep.mean()), normal=normal.tolist())
        if 'hypotheses' in diag: diag['hypotheses'].append(row)
        if normal[2]<math.cos(math.radians(35)):
            row['reason']='face_tilt_exceeded'
            diag['rejections'][row['reason']]=diag['rejections'].get(row['reason'],0)+1;continue
        if keep.sum()<48 or keep.mean()<.15:
            row['reason']='insufficient_plane_inliers'
            diag['rejections'][row['reason']]=diag['rejections'].get(row['reason'],0)+1;continue
        spread = np.linalg.svd(p[keep]-p[keep].mean(0),compute_uv=False)/math.sqrt(keep.sum())
        if spread[1] < 3*pixel:
            row['reason']='unresolved_plane_extent'
            diag['rejections'][row['reason']]=diag['rejections'].get(row['reason'],0)+1;continue
        row['reason']='accepted'
        candidates.append((int(keep.sum()),dict(point=plane[0].tolist(),normal=normal.tolist(),
            residualM=float(np.quantile(error[keep],.95)),points=int(keep.sum()))))
    if not candidates:
        diag['reason']='no_local_bearing_plane';return None
    _, patch = max(candidates,key=lambda item:item[0])
    diag.update(reason='local_face_fit',inlierFraction=patch['points']/len(p),normal=patch['normal'])
    return patch
