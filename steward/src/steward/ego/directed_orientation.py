"""Directed end evidence and bounded release feedback, using robot frames only."""
import math
import numpy as np
from steward.ego import capabilities
from steward.math3d import matrix_from_quat, quat_from_yaw


def enabled():
    return capabilities.get('STEWARD_ASSEMBLY_DIRECTED_ORIENTATION') != '0'


def end_evidence(record, view):
    """Compare raised residuals on both ends of the existing broad-face cloud.

    Remove face slope before looking for a raised end. A clipped/flat cloud is
    explicitly a best estimate; it never causes an extra observation or a wait.
    """
    ref = record.get('contactReference', record['reference'])
    basis = matrix_from_quat(quat_from_yaw(ref['yaw']))
    axis = int(np.argmax(ref['size'][:2]))
    p = np.asarray((view or {}).get('points', []), float)
    result = dict(source='registered_end_best_estimate', confidence='unconfirmed',
                  directionWorld=basis[:, axis].tolist(), extraObservations=0)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 12 or not np.isfinite(p).all():
        return result
    local = (p-np.asarray(ref['position'])) @ basis
    from steward.ego.assembly_edge_observation import body_patch
    patch = body_patch(p, ref, {})
    residual = p[:, 2].copy()
    if patch:
        normal = np.asarray(patch['normal'])
        residual = (p-np.asarray(patch['point'])) @ normal
    else:
        # No stable plane: retain the higher end as an explicit uncertain prior.
        residual -= np.median(residual)
    along = local[:, axis]
    low, high = np.quantile(along, [.25, .75])
    left, right = residual[along <= low], residual[along >= high]
    scores = [float(np.quantile(x, .85)) for x in (left, right)]
    contrast = scores[1]-scores[0]
    noise = max(float((patch or {}).get('noiseM', (patch or {}).get('residualM', 0.))),
                float(np.median(abs(residual-np.median(residual)))), np.finfo(float).eps)
    direction = basis[:, axis]*(1. if contrast >= 0 else -1.)
    return dict(result, source='cached_rgbd_face_residual' if patch else 'cached_rgbd_end_height_best_estimate',
        confidence='observed' if patch and abs(contrast) > 3*noise
            and low < -ref['size'][axis]/4 and high > ref['size'][axis]/4
            and not (view or {}).get('truncated', False) else 'unconfirmed',
        directionWorld=direction.tolist(), endContrastM=contrast, residualNoiseM=noise)


def bind_grasp(record, rotation):
    # Bind to the registered directed frame, not a newly fitted symmetric box.
    basis = matrix_from_quat(quat_from_yaw(record['reference']['yaw']))
    record['reference']['directedGraspBasis'] = (np.asarray(rotation).T @ basis).tolist()


def wrist_target(planner, scene, held, yaw, dock_yaw, flip):
    orientation = held.state_vars.get('orientation')
    record = ((orientation.value or {}).get('stationaryAssembly') or {}) if orientation else {}
    basis = record.get('reference', {}).get('directedGraspBasis')
    if basis is None or not record.get('edgeGeometry'):
        return None
    from steward.ego.assembly_motion import half_turn_roll
    joints = planner._arm_joints(scene)
    roll = joints.get('wrist_roll', 0.)
    if flip: roll = half_turn_roll(roll, planner.arm.geometry.wrist_roll_max)
    transformed = planner.arm.tool_rotation(0., roll, joints.get('wrist_pitch', 0.)) @ np.asarray(basis)[:, 0]
    return (yaw-dock_yaw-math.atan2(transformed[1], transformed[0])+math.pi)%(2*math.pi)-math.pi


def classify(report):
    if report.get('measuredLevel') is True or report.get('level') is True: return 'level'
    if not report.get('observed') or report.get('estimatedTiltDeg') is None: return 'unconfirmed'
    n = np.asarray((report.get('patch') or {}).get('normal', []), float)
    edge = np.asarray(report.get('edgeNormalWorld', []), float)
    if n.shape != (3,) or edge.shape != (2,) or not np.isfinite(n).all(): return 'unconfirmed'
    high = -n[:2]; projection = float(high @ edge)
    # Only a truly transverse/zero slope leaves the end unresolved. A strong
    # transverse component must not hide a measured inward rise.
    if abs(projection) <= np.finfo(float).eps*max(1., float(np.linalg.norm(high))): return 'unconfirmed'
    return 'reversed' if projection < 0 else 'insufficient_overhang'


def correction(record, report):
    from steward.ego import edge_retention
    decision = classify(report)
    report['orientationDecision'] = decision
    if decision == 'level':
        report['correctionStopped'] = 'measured_level_keep_position'
        return False
    if decision == 'reversed':
        if record.get('edgeHalfTurnAttempted'): return False
        anchor = record.get('edgeAnchor')
        if not anchor or not report.get('used'): return False
        # The plane identifies the actual raised end. Mirror only the uncertain
        # end hypothesis; the body and registered members keep their geometry.
        shape = record['edgeGeometry']; normal = np.asarray(anchor['edgeNormalWorld'])
        points = np.asarray(shape['protrusions'])
        points[:, :2] -= 2*np.outer(points[:, :2] @ normal, normal)
        shape['protrusions'] = points.tolist()
        shape['outwardAxis'] = [-normal[0], -normal[1], 0.]
        shape['geometryConfirmed'] = False
        record['reference']['edgeGraspOutward'] = list(shape['outwardAxis'])
        record['edgeCorrectionTarget'] = dict(support=record['reference']['support'],
            spot=list(record['reference']['position'][:2]), edgeAxis=anchor['edgeAxis'],
            edgeSign=anchor['edgeSign'], yawDeltaRad=math.pi)
        from steward.ego import placement_body_frame
        if placement_body_frame.enabled():
            record['edgeCorrectionTarget'].update(spot=list(anchor['spot']), yaw=anchor['yaw']+math.pi)
        record['edgeCorrectionStepM'] = 0.
        record['edgeCorrectionDirection'] = normal.tolist()
        record['edgeCorrectionAction'] = 'rotate_vertical_half_turn'
        return True
    if decision == 'insufficient_overhang':
        ok = edge_retention.correction(record, report)
        if ok: record['edgeCorrectionAction'] = 'outward_step'
        return ok
    report.update(unconfirmed=decision == 'unconfirmed', correctionStopped='direction_unconfirmed_keep_completed_release')
    return False


def plan_report(planner, scene, held, plan, flip):
    orientation = held.state_vars.get('orientation')
    record = ((orientation.value or {}).get('stationaryAssembly') or {}) if orientation else {}
    basis = record.get('reference', {}).get('directedGraspBasis')
    shape = record.get('edgeGeometry') or {}
    edge = plan.get('edgePlacement')
    if basis is None or not edge or not shape.get('outwardAxis') or 'edgeNormalWorld' not in edge: return
    basis = np.asarray(basis); ref = record['reference']
    r = matrix_from_quat(quat_from_yaw(ref['yaw']))
    canonical = r @ np.diag([1.,-1.,-1.]) @ r.T if flip else np.eye(3)
    local_end = r.T @ canonical.T @ np.asarray(shape['outwardAxis'])
    joints = planner._arm_joints(scene)
    release = matrix_from_quat(quat_from_yaw(plan['yaw'])) @ planner.arm.tool_rotation(
        plan.get('wristYaw', joints.get('wrist_yaw',0.)),
        plan.get('wristRoll', joints.get('wrist_roll',0.)),
        plan.get('wristPitch', joints.get('wrist_pitch',0.)))
    end = release @ basis @ local_end
    edge['orientationKinematics'] = dict(source='registered_directed_frame_and_grasp_encoders',
        graspRollAxisBody=(basis.T @ [1.,0.,0.]).tolist(),
        wristYawRad=plan.get('wristYaw'), wristRollRad=plan.get('wristRoll'),
        predictedProtrusionDirectionWorld=end.tolist(),
        outwardDot=float(end[:2] @ np.asarray(edge['edgeNormalWorld'])),
        verified=False)

    from steward.ego import placement_body_frame
    if placement_body_frame.enabled():
        long_end = placement_body_frame.long_axis(record)
        actual = release @ basis @ (r.T @ canonical.T @ long_end)
        normal = np.asarray(edge['edgeNormalWorld'])
        angle = math.degrees(math.atan2(abs(float(actual[:2] @ normal)),
            abs(float(actual[0]*normal[1]-actual[1]*normal[0]))))
        edge['orientationKinematics'].update(predictedLongAxisWorld=actual.tolist(),
            longAxisEdgeAngleDeg=angle)
