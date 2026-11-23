"""Uncertainty of a contour centroid intersected with a fitted depth plane.

Component-only interpretation of native detector outputs; screw detection is unchanged.
"""
import math
import numpy as np
from steward.ego import support_holes as holes


def centre_uncertainty(hole, raw, cam):
    origin = np.asarray(hole['pixel'])
    spans = np.asarray(hole['spanPixels'])
    pad = max(3, int(math.ceil(min(spans)*.5)))
    low = np.maximum(0, np.floor(origin-spans/2-pad).astype(int))
    high = np.minimum(raw['depth'].shape[::-1], np.ceil(origin+spans/2+pad+1).astype(int))
    gray = raw['rgb'][low[1]:high[1], low[0]:high[0]].mean(axis=2)
    groups = holes.components(gray < np.median(gray)*.65)
    groups = [g for g in groups if len(g) >= 6]
    if not groups:
        return dict(method='detector_bound_fallback', relativeM=float(hole['uncertaintyM']))
    core = min(groups, key=lambda g:np.linalg.norm(g.mean(axis=0)+low-origin))
    centre = core.mean(axis=0)
    mask = np.zeros(gray.shape, bool); mask[core[:, 1], core[:, 0]] = True
    interior = mask.copy()
    for axis in (0, 1):
        interior &= np.roll(mask, 1, axis) & np.roll(mask, -1, axis)
    by, bx = np.where(mask & ~interior)
    boundary = np.c_[bx, by]-centre
    # A boundary displacement changes centroid by (edge-centre)*dA/area.
    # Pixel edge quantisation has variance 1/12; retain a quarter-pixel
    # correlated raster/threshold bias, which does NOT shrink with sample count.
    covariance = boundary.T @ boundary/(12*len(core)**2)
    random_px = math.sqrt(float(np.linalg.eigvalsh(covariance)[-1]))
    centre_px = .25+2*random_px
    yy, xx = np.where(~mask)
    depth = raw['depth'][yy+low[1], xx+low[0]]
    valid = np.isfinite(depth) & (depth > .05)
    design = np.c_[xx[valid]+low[0]-origin[0], yy[valid]+low[1]-origin[1], np.ones(valid.sum())]
    # Prediction at the centre of a local plane. Cap effective independent
    # depth samples at 16 to account for spatial correlation in native depth.
    leverage = float(np.linalg.pinv(design.T @ design)[2, 2]) if len(design) >= 3 else 1.
    leverage = max(leverage, 1/16)
    plane_m = 2*hole['planeNoiseM']*math.sqrt(leverage)
    normal = np.asarray(hole['normal']); ray = np.asarray(hole['point'])-cam['p']
    distance = np.linalg.norm(ray)
    cosine = abs(float(ray @ normal))/distance
    focal = raw['depth'].shape[0]/(2*math.tan(math.radians(raw['fovy'])/2))
    pixel_m = distance/(focal*max(cosine, 1e-12))
    relative = pixel_m*centre_px+plane_m/max(cosine, 1e-12)+1e-6
    return dict(method='contour_centroid_plane_prediction', contourPixels=len(core),
        boundaryPixels=len(boundary), rimSamples=int(valid.sum()), effectivePlaneSamples=round(1/leverage, 2),
        centroidSigmaPx=round(random_px, 5), rasterBiasPx=.25, centroidErrorPx=round(centre_px, 5),
        pixelFootprintM=round(float(pixel_m), 7), planePredictionM=round(plane_m, 7),
        relativeM=float(relative))
