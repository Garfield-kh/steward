"""Resolve a visible shaft separated in depth from a touching image silhouette.

Only pre-release observations call this fallback. A depth discontinuity separates
surfaces; it does not establish an endpoint. The ordinary curved-shaft, visible
clearance, robot-cut and image/ROI-cut checks still validate each candidate.
"""
import numpy as np
from steward.ego import capabilities
from steward.ego.support_holes import components


def enabled():
    return capabilities.get('STEWARD_ASSEMBLY_PRERELEASE_TIP') != '0'


def separated(groups, info):
    """Split touching projected surfaces at measured metric depth jumps.

    Remove both sides of a discontinuity before image connectivity, including
    diagonal contacts. The distance scale is the frame's depth/pixel noise.
    No semantic mask, nominal object extent or scene geometry is consulted.
    """
    if info is None:
        return groups, info
    result = []
    for group in groups:
        points, mask = group['points'], group['mask']
        y, x = np.where(mask & (info['raw']['depth'] > 0))
        point_index = np.full(mask.shape, -1, dtype=np.int32)
        point_index[y, x] = np.arange(len(points))
        cut = np.zeros_like(mask)
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
            ny, nx = y+dy, x+dx
            valid = (ny >= 0) & (ny < mask.shape[0]) & (nx >= 0) & (nx < mask.shape[1])
            indices = np.flatnonzero(valid)
            indices = indices[mask[ny[indices], nx[indices]]]
            jumps = indices[np.linalg.norm(points[point_index[ny[indices], nx[indices]]]-points[indices], axis=1) > 4*info['noise']]
            cut[y[jumps], x[jumps]] = True
            cut[ny[jumps], nx[jumps]] = True
        for pixels in components(mask & ~cut):
            if len(pixels) < 30:
                continue
            local = np.zeros_like(mask)
            local[pixels[:, 1], pixels[:, 0]] = True
            result.append(dict(points=points[local[y, x]], mask=local, pixels=pixels))
    return result, info


def lower(controller, sample, plane, opening, diagnostics, *, foreground_data=None):
    """Fallback only: unique floating curved shaft in the existing jaw envelope."""
    from steward.ego import entry_layout as el
    data = foreground_data if foreground_data is not None else el.foreground(controller, sample, plane, opening)
    groups, info = separated(*data)
    trace = diagnostics.setdefault('depthSeparatedLower', {})
    trace.update(originalComponents=len(data[0]), separatedComponents=len(groups),
                 source='same_frame_depth_separated_lower_arc')
    # The existing foreground already bounds the entire component by the jaw
    # envelope. Its midpoint is not the pinch point: an eccentric grasp can be
    # outside half that envelope. Floating clearance and a unique curved shaft,
    # not proximity to its centre, distinguish it from an installed head.
    feature = el.short_lower(controller, sample, plane, opening, trace,
                            foreground_data=(groups, info), grip_radius=opening)
    if feature is not None:
        feature['source'] = 'same_frame_depth_separated_lower_arc'
    return feature


def measurement_summary(report):
    """Expose existing differential observations, separate from the FK report.

    A measurement earlier in the approach is not a final release confirmation.
    Corrections and a later lost view remain explicit; never invent a zero error.
    """
    frames = []
    corrections = []
    for servo in report.get('tipEntryServos', []):
        before = [f for f in servo.get('frames', []) if f.get('phase') in ('hover', 'descent', 'reacquire')]
        frames.extend((servo.get('id'), f) for f in before)
        corrections.extend((servo.get('id'), c) for c in servo.get('corrections', []) if c.get('executed'))
    measured = [(sid, f) for sid, f in frames if f.get('bothVisible') and f.get('deltaM') is not None]
    if not measured:
        return None
    sid, last = measured[-1]
    newer_than_motion = all(s != sid or c.get('frameIndex', -1) < last.get('index', -1) for s, c in corrections)
    latest_confirmed = frames[-1][0] == sid and frames[-1][1] is last and newer_than_motion
    return dict(source='same_frame_entry_minus_lower_end', samples=len(measured),
        firstResidualM=measured[0][1]['residualM'], lastResidualM=last['residualM'],
        lastObservedAt=last.get('stamp'), uncertaintyM=last.get('uncertaintyM'),
        entryInferredFromLayout=bool(last.get('targetFromLayout')),
        correctionCount=len(corrections), latestObservationConfirmed=latest_confirmed,
        insertionConfirmed=False)
