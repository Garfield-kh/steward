"""Keep a registered grasp reference alive through pre-grasp observation.

This is an execution hypothesis, not evidence of identity, attachment or face.
Raw RGB-D samples still enter the estimator's independent observation channel.
"""
from contextlib import contextmanager

import numpy as np

from steward.ego import capabilities


def local_pinches(ref, view, opening, now, parts):
    """Measure full transverse sections, never shrink the motion envelope.

    A complete local cloud can include an attached wing: its global short side
    is not the width at every pinch. No catalogue width or cropped cross-axis
    points may stand in for a measured section. This remains a grasp hypothesis.
    """
    from steward.ego.assembly_edge_observation import usable
    from steward.ego.fasten_release import body_grip_points
    from steward.math3d import matrix_from_quat, quat_from_yaw
    report = dict(source='pregrasp_rgbd_sections', observed=False,
        identityVerified=False, envelopeSizeM=list(ref['size']), openingM=opening,
        referencePositionM=list(ref['position']), referenceYawRad=ref['yaw'],
        candidates=[], sections=[], reason='envelope_fits_opening')
    if opening is None or min(ref['size'][:2]) < opening:
        return report
    report['reason'] = 'no_fresh_complete_local_cloud'
    if not usable(view, now, ref.get('ownReleaseAt', -float('inf'))):
        return report
    origin = np.asarray(ref['position'], float)
    rotation = matrix_from_quat(quat_from_yaw(ref['yaw']))
    points = (np.asarray(view['points'], float)-origin) @ rotation
    axis = int(np.argmax(ref['size'][:2])); cross = 1-axis
    half_window = max(.02, opening/4)
    height = ref.get('graspHeightM', origin[2])-origin[2]
    report.update(observedAt=view['stamp'], cloudSource=view['source'],
        pointCount=len(points), axis=axis, halfWindowM=half_window,
        graspHeightM=float(height+origin[2]), reason='no_grippable_local_section')
    for xy, _ in body_grip_points(ref, parts):
        centre = (np.r_[xy, origin[2]]-origin) @ rotation
        strip = points[abs(points[:, axis]-centre[axis]) <= half_window]
        row = dict(alongM=float(centre[axis]), points=len(strip))
        report['sections'].append(row)
        if len(strip) < 24 or np.ptp(strip[:, axis]) < 1.5*half_window:
            row['reason'] = 'insufficient_longitudinal_coverage'; continue
        lo, hi = strip.min(axis=0), strip.max(axis=0)
        width = float(hi[cross]-lo[cross])
        row.update(widthM=width, lowM=lo.tolist(), highM=hi.tolist())
        # Visible surface must actually reach the planned close height. A top
        # plane alone does not locate a graspable cross-section below itself.
        if lo[2] > height+.003 or hi[2] < height-.003:
            row['reason'] = 'close_height_unobserved'; continue
        if width < .008 or width+.006 >= opening:
            row['reason'] = 'section_width_outside_opening'; continue
        centre[cross] = (lo[cross]+hi[cross])/2
        # Require dense coverage across the section, not two unrelated patches.
        bins = np.floor((strip[:, cross]-lo[cross])/.004).astype(int)
        if len(np.unique(bins)) < int(width/.004):
            row['reason'] = 'disconnected_section'; continue
        if abs(centre[cross]) > ref['size'][cross]/2:
            row['reason'] = 'section_outside_registered_body'; continue
        point = origin+rotation @ centre
        # Recheck loose-component exclusion at the measured transverse centre.
        probe = dict(ref, position=[*point[:2], height+origin[2]], size=[0., 0., ref['size'][2]])
        if not body_grip_points(probe, parts):
            row['reason'] = 'loose_component_overlap'; continue
        row['reason'] = 'measured_section_fits'
        report['candidates'].append(dict(xy=point[:2].tolist(),
            offset=(point-origin)[:2].tolist(), widthM=width,
            observedAt=view['stamp'], source=view['source']))
    if report['candidates']:
        report['reason'] = 'local_sections_available'
    return report


@contextmanager
def reference(rt, record, job):
    from steward.ego.assembly_motion import restore_track
    with rt._lock:
        est = rt.estimator
        track = restore_track(est, record)
        ref = record['reference']
        track.pos = np.array(ref['position'])
        track.size = np.array(ref['size'])
        track.yaw = ref.get('yaw', track.yaw)
        active = capabilities.get('STEWARD_ASSEMBLY_FLIP_GRASP') != '0'
        was_pinned, old_age = track.pinned, track.pin_observed_age
        if active:
            track.pinned = True
            track.pin_observed_age = max(0., est.stamp-track.last_seen)
            track.flip_grasp_protected = True
        diagnostic = dict(entity=track.id, protected=active,
            source='registered_pregrasp_reference', observedAt=track.last_seen,
            sizeAtStartM=track.size.tolist(),
            identityVerified=False)
        job.postcondition['graspReference'] = diagnostic
    released = False

    def release(reason='scope_exit'):
        nonlocal released
        if released:
            return
        released = True
        with rt._lock:
            if active:
                track.pinned, track.pin_observed_age = was_pinned, old_age
                track.flip_grasp_protected = False
            ref.pop('localGraspSections', None)
            current = rt.estimator.stationary_assemblies.records.get(ref['entity'])
            if current:
                current['reference'].pop('localGraspSections', None)
            diagnostic['endedAt'] = rt.estimator.stamp
            diagnostic['endReason'] = reason
            diagnostic['sizeAtEndM'] = track.size.tolist()
            diagnostic['heldEntity'] = rt.estimator.held_id

    try:
        yield release
    finally:
        release()
