"""Preserve released assemblies and spend placement uncertainty inward only.

All inputs are robot registrations, release history and encoder estimates.
"""
from copy import deepcopy
import numpy as np
from steward.ego import capabilities


def enabled():
    return capabilities.get('STEWARD_ASSEMBLY_PLACEMENT_SUPPORT_MARGIN') != '0'


def retained(records, target):
    """A grasp on a released member can transport its entire parent assembly."""
    if not enabled() or not target:
        return None
    for key, record in records.items():
        if not record.get('edgeRetained'):
            continue
        ids = {key, record['reference']['entity'], *record.get('aliases', [])}
        ids.update(p['entity'] for p in record.get('parts', []) if p.get('status') != 'invalidated')
        if target in ids:
            return record
    return None


def scene_records(scene):
    result = {}
    if not enabled():
        return result
    for ent in scene.entities:
        orientation = ent.state_vars.get('orientation')
        record = (orientation.value or {}).get('stationaryAssembly', {}) if orientation else {}
        if record.get('edgeRetained') and record.get('reference', {}).get('entity'):
            result[record['reference']['entity']] = record
    return result


def guidance(record):
    ref = record['reference']
    resolution = ref.get('faceResolution', {})
    holes = resolution.get('exposedHoles', [])
    action = dict(skill='fasten_onto', body=ref['entity'],
        reason='continue_remaining_interfaces_in_place', holes=list(holes))
    return dict(bodyRetained=True, reason='edge_body_retained_assemble_in_place',
        nextAction=action, canContinue=True,
        guidance='已就位，无需移动。下一步是在原位装配朝上接口，补齐尚未释放的紧固件；已放置成员也不要再抓取。')


def guard(records, skill, params):
    target = (params.get('target') if skill in ('pick', 'regrasp_upright') else
              params.get('body') if skill == 'flip_assembly' else
              params.get('part') if skill in ('assemble_onto', 'fasten_onto') else None)
    record = retained(records, target)
    if record and skill in ('assemble_onto', 'fasten_onto'):
        # Resume fastening an already released member in place. Any attempted
        # pick inside that skill still passes through the execution guard.
        if params.get('body') in (record['reference']['entity'], *record.get('aliases', [])):
            return None
    return guidance(record) if record else None


def inward_line(inner, com, uncertainty):
    """Bias the supported midpoint inward by the estimated uncertainty.

    When the window cannot cover uncertainty, incomplete overhang is preferable
    to moving the centre outside support. The post-release slope can correct it.
    """
    window = inner-com
    middle = (inner+com)/2
    if window <= 0:
        return middle
    return max(middle+min(uncertainty,window/4),
               max(0.,com)+uncertainty+window/4)


def inward_release(executor, plan, job):
    edge = plan.get('edgePlacement')
    if not enabled() or not edge or not edge.get('inwardOnly'):
        return
    normal = np.asarray(edge['edgeNormalWorld'], float)
    grip = np.asarray(plan.get('gripSpot', plan['spot']), float)
    before = float((executor._gripper_xy()-grip) @ normal)
    report = dict(source='robot_encoders', outwardBeforeM=before, corrected=False)
    from steward.ego import placement_body_frame
    exact = placement_body_frame.enabled() and edge.get('correctionTarget')
    if (abs(before) > executor.config.release_joint_tol if exact else before > 0):
        # One measured compensation; reserve another actuator tolerance inward.
        target = grip if exact else grip-normal*(before+executor.config.release_joint_tol)
        try:
            executor._align_lateral(target.tolist(), tol=executor.config.release_joint_tol)
            executor._set_joint('reach', executor._reach_to(target.tolist()), ramp=False)
            executor._settle_for_release()
            report['corrected'] = True
        except Exception as exc:
            from steward.ego.skills import SkillFailed
            if not isinstance(exc, SkillFailed):
                raise
            report['failure'] = str(exc)
    report['outwardAfterM'] = float((executor._gripper_xy()-grip) @ normal)
    report['unconfirmed'] = report['outwardAfterM'] > 0
    plan['inwardReleaseCheck'] = report
    job.postcondition['inwardReleaseCheck'] = deepcopy(report)


def correction_ready(controller, record):
    """Check the return path on a copied predicted grasp scene before taking it.

    No sensor read and no estimator mutation: the real post-grasp receipt is
    still freshly planned. A rejected correction leaves the released body alone.
    """
    from steward.scene.schema import Relation
    from steward.math3d import quat_from_yaw
    from steward.ego.edge_retention import candidates
    scene = deepcopy(controller.rt.ego_state())
    planner = controller.rt.esim
    ref = record['reference']; body = scene.find(ref['entity'])
    table = scene.find(ref['support'])
    if body is None or table is None:
        return False, 'registered_correction_scene_unavailable'
    from steward.ego import directed_orientation
    if directed_orientation.enabled():
        body.state_vars['orientation'].value['stationaryAssembly'] = deepcopy(record)
    pick = planner.predict(scene, 'pick', dict(target=body.id, assemblyGrip=True))
    if not pick.get('feasible') or pick['risk'] > controller.rt.config.max_pick_risk:
        return False, 'correction_pick_unreachable'
    grasp = pick['plan']; robot = scene.entity(scene.self_id)
    robot.pose.value = dict(p=[*grasp['dock'], robot.pose.value['p'][2]], q=quat_from_yaw(grasp['yaw']))
    for joint in robot.rigid.joints:
        if joint.id in ('lift', 'reach'):
            joint.q.value = [grasp[joint.id]]
        elif joint.id == 'wrist_yaw':
            joint.q.value = [grasp.get('wristYaw', 0.)]
        elif joint.id == 'wrist_roll':
            joint.q.value = [0.]
    # The scene overlay may still contain the previous release record.
    predicted = deepcopy(record)
    from steward.ego import directed_orientation
    if directed_orientation.enabled():
        from steward.math3d import matrix_from_quat
        rotation = matrix_from_quat(quat_from_yaw(grasp['yaw'])) @ planner.arm.tool_rotation(
            grasp.get('wristYaw', 0.), 0., grasp.get('wristPitch', 0.))
        directed_orientation.bind_grasp(predicted, rotation)
    body.state_vars['orientation'].value['stationaryAssembly'] = predicted
    scene.relations = [r for r in scene.relations if r.a.get('entity') != body.id]
    scene.relations.append(Relation(type='attached', a={'entity':body.id}, b={'entity':scene.self_id},
        confidence=1., derived=True))
    rows = candidates(record, table, planner._sigma_xy(table))
    from steward.ego import placement_body_frame
    failure = 'no_outward_supported_target'
    checks=[]
    if placement_body_frame.enabled(): record['edgeCorrectionPreflight']=checks
    for c in (rows if placement_body_frame.enabled() else rows[:1]):
        if placement_body_frame.enabled() and c.get('correctionRecovery'): continue
        receipt = planner.predict(scene, 'place', dict(on=table.id, alignHoles=False,
            assemblyBody=body.id, assemblyEdge=True, offset=c['offset'], yaw=c['yaw'],
            assemblyEdgeSide=[c['edgeAxis'],c['edgeSign']]))
        if placement_body_frame.enabled():
            checks.append(dict(spot=c['spot'], yaw=c['yaw'], edgeAxis=c['edgeAxis'],
                edgeSign=c['edgeSign'], feasible=receipt.get('feasible',False),
                issues=receipt.get('issues',[])))
        if receipt.get('feasible') and receipt['risk'] <= controller.rt.config.max_risk:
            return True, 'predicted_grasp_and_release_path'
        failure = '; '.join(receipt.get('issues', []))
    return False, failure
