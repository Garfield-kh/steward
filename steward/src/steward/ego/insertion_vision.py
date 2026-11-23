"""Local RGB-D evidence for a held shaft crossing an observed opening plane.

No FK depth can confirm insertion: carried objects need not transmit contact to
an actuator. A newly visible, continuous curved shaft below a previously empty
mouth is independent visual evidence. Occlusion is unknown, never success.
"""
import math
import numpy as np
from steward.ego import wrist_observation as wo
from steward.ego.entry_layout import circle


def cloud(sample, entry):
    raw, cam = sample
    radius = float(entry['radiusM'])
    region = wo.window(raw, cam, entry['point'], 3*radius)
    points, y, x = wo.cloud(raw, cam, region)
    if not len(points):
        return None
    focal = raw['depth'].shape[0]/(2*math.tan(math.radians(raw['fovy'])/2))
    noise = max(float(np.median(raw['depth'][y, x]))/focal,
                float(entry.get('planeNoiseM', 0.)), np.finfo(float).eps)
    normal = np.asarray(entry.get('normal', [0., 0., 1.]), float)
    normal /= np.linalg.norm(normal)
    # A surface-tangent basis; no task/world axis position is encoded.
    tangent = np.cross(normal, np.eye(3)[np.argmin(abs(normal))])
    tangent /= np.linalg.norm(tangent)
    basis = np.c_[tangent, np.cross(normal, tangent)]
    delta = points-np.asarray(entry['point'])
    return delta @ basis, delta @ normal, noise


def crossing(sample, entry, shaft_radius, max_depth):
    """Require curved sections on BOTH sides of the mouth, clear of its wall.

    Fitting each cross-section avoids mistaking the hole wall, a flat graphic or
    the floor of a recess for a shaft. Resolution scales come from this frame.
    """
    data = cloud(sample, entry)
    if data is None:
        return dict(detected=False, reason='mouth_pixels_unavailable')
    xy, height, noise = data
    radius = float(entry['radiusM'])
    if min(shaft_radius, radius-shaft_radius, max_depth) <= noise:
        return dict(detected=False, reason='shaft_or_depth_below_pixel_resolution', noiseM=noise)
    fits = []
    for low, high in ((noise, max_depth), (-max_depth, -noise)):
        keep = ((height >= low) & (height <= high)
                & (np.linalg.norm(xy, axis=1) < radius-noise))
        if np.count_nonzero(keep) < 12:
            return dict(detected=False, reason='shaft_section_occluded', noiseM=noise)
        fit = circle(xy[keep], noise)
        if fit is None:
            return dict(detected=False, reason='shaft_curvature_unresolved', noiseM=noise)
        centre, measured_radius, error = fit
        if (abs(measured_radius-shaft_radius) > noise
                or np.linalg.norm(centre)+measured_radius+noise >= radius):
            return dict(detected=False, reason='wall_or_off_mouth_surface', noiseM=noise)
        fits.append((centre, measured_radius, error))
    if np.linalg.norm(fits[0][0]-fits[1][0]) > 2*noise:
        return dict(detected=False, reason='shaft_sections_not_continuous', noiseM=noise)
    return dict(detected=True, reason='curved_shaft_crosses_visible_mouth', noiseM=noise,
                observedDepthM=float(-np.quantile(height[(height < -noise) &
                    (np.linalg.norm(xy, axis=1) < radius-noise)], .1)),
                source='native_rgbd_shaft_across_entry_plane')


def surface(sample, entry):
    """Reobserve the surrounding plane even when the load hides the mouth."""
    from steward.ego.wrist_surface import fit
    raw, cam = sample
    radius = float(entry.get('radiusM', 0.))
    if radius <= 0:
        return None
    region = (wo.window(raw, cam, entry['point'], 3*radius) &
              ~wo.window(raw, cam, entry['point'], radius*1.5))
    plane = fit(raw, cam, region)
    if plane is None:
        return None
    normal = np.asarray(plane['normal'])
    point = np.asarray(entry['point'], float).copy()
    point[2] -= float((point-np.asarray(plane['point'])) @ normal)/normal[2]
    data = cloud(sample, entry)
    noise = max(plane['residualM'], data[2] if data else 0.)
    return dict(point=point.tolist(), normal=normal.tolist(), radiusM=radius,
                planeNoiseM=noise, uncertaintyM=noise,
                source='same_frame_surrounding_plane')


def insertion(before, after, entry, shaft_radius, max_depth, *, current_entry=None):
    if before is None or after is None:
        return dict(detected=False, reason='probe_frame_unavailable')
    if after[0]['header']['stamp'] <= before[0]['header']['stamp']:
        return dict(detected=False, reason='probe_frame_not_new')
    # Prefer a rim measured in the AFTER frame: its common odometry error then
    # cancels against that frame's shaft. A layout-only centre is not admissible.
    if current_entry is not None and current_entry.get('targetFromLayout'):
        current_entry = None
    previous, current = surface(before, entry), surface(after, current_entry or entry)
    if previous is None or current is None:
        return dict(detected=False, reason='surrounding_plane_unobserved')
    shift = abs(current['point'][2]-previous['point'][2])
    tilt = np.linalg.norm(np.asarray(current['normal'])-previous['normal'])*entry['radiusM']
    if max(shift, tilt) > 3*(previous['planeNoiseM']+current['planeNoiseM']):
        return dict(detected=False, reason='support_moved_during_probe')
    data = cloud(before, previous)
    xy, heights, noise = data
    empty = (np.linalg.norm(xy, axis=1) < previous['radiusM']-noise) & (heights < -noise)
    if np.count_nonzero(empty) < 12:
        return dict(detected=False, reason='preprobe_empty_mouth_unobserved')
    old = crossing(before, previous, shaft_radius, max_depth)
    new = crossing(after, current, shaft_radius, max_depth)
    if old['detected']:
        return dict(detected=False, reason='shaft_or_occupancy_already_present')
    return new
