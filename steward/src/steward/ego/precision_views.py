"""Independent samples for a scoped precision request, never identity/pose fusion."""
import numpy as np


def match_views(tracks, measurements, targets, stamp):
    """Accept only mutually unique, complete masks near an existing prediction.

    All tracks compete, including non-target neighbours. A mask cannot be shared
    by two targets or attached solely because it is near the gripper. Missing or
    ambiguous evidence stays missing; no nominal geometry becomes sensor data.
    """
    candidates = {}
    for i, m in enumerate(measurements):
        seg = m['seg']
        points = np.asarray(seg.points)
        if (seg.border or seg.invalid_frac > .05 or seg.range > .8 or len(points) < 48
                or not np.isfinite(points).all() or not seg.semantics):
            continue
        scores = seg.semantics
        total = sum(scores.values())
        if total <= 0:
            continue
        for eid, track in tracks.items():
            reference = track.semantic_counts or track.category_counts
            reference_total = sum(reference.values())
            overlap = sum(min(score/total, reference.get(label, 0.)/reference_total)
                          for label, score in scores.items()) if reference_total > 0 else 0.
            if overlap < .7:
                continue
            # The same local correction budget used by the precision controller.
            if np.linalg.norm(np.asarray(m['centre'])-track.pos) > .03:
                continue
            expected, measured = np.sort(track.size), np.sort(m['size'])
            if np.any(abs(measured-expected) > np.maximum(.006, .2*expected)):
                continue
            candidates.setdefault(eid, []).append(i)
    owners = {}
    for eid, indices in candidates.items():
        for i in indices:
            owners.setdefault(i, []).append(eid)
    out = {}
    for eid in targets:
        indices = candidates.get(eid, [])
        if len(indices) != 1 or len(owners[indices[0]]) != 1:
            continue
        m = measurements[indices[0]]; seg = m['seg']
        points = seg.points[np.linspace(0, len(seg.points)-1, min(2048, len(seg.points)), dtype=int)]
        out[eid] = dict(stamp=float(stamp), source='rgbd_segment', points=points.copy(),
            centreM=np.asarray(m['centre']).tolist(), visibleSizeM=np.asarray(m['size']).tolist(),
            referenceSizeM=np.asarray(tracks[eid].size).tolist(),
            boxYawRad=float(m['yaw']), rangeM=float(seg.range), truncated=bool(seg.border),
            invalidDepthFraction=float(seg.invalid_frac), association='precision_independent_mask')
    return out
