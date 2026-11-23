"""ASSEMBLY_RELEASE_HEIGHT: one-sided assembly release geometry, in the execution reference frame.

Inputs are robot registrations and encoder kinematics. No contact-stop inference:
carried geometry does not collide with the environment until the hand opens.
"""
from copy import deepcopy
import math
import numpy as np
from steward.ego import capabilities
from steward.math3d import matrix_from_quat, quat_from_yaw


def enabled():
    return capabilities.get('STEWARD_ASSEMBLY_RELEASE_HEIGHT') != '0'


def sigma_z(entity):
    cov = (entity.pose.unc or {}).get('cov')
    return math.sqrt(max(0., cov[14])) if cov and len(cov) > 14 else 0.


def estimate(record, top, turn_over, *, edge=False, surface_uncertainty=0., body_uncertainty=0., gap=.002):
    from steward.ego.assembly_edge import corners
    ref = record['reference']; origin = np.asarray(ref['position'], float)
    basis = matrix_from_quat(quat_from_yaw(ref['yaw']))
    delta = basis @ np.diag([1., -1., -1.]) @ basis.T if turn_over else np.eye(3)
    core = corners(ref, origin, delta)
    # A recovered local face may refine a model, but must never erase the
    # measured body envelope and put its execution reference below the table.
    body_low = float(core[:, 2].min()); body_high = float(core[:, 2].max())
    contact = record.get('contactReference')
    disagreement = 0.
    if contact:
        pts = corners(contact, origin, delta)
        disagreement = abs(float(pts[:, 2].min())-body_low)
        body_low = min(body_low, float(pts[:, 2].min()))
        body_high = max(body_high, float(pts[:, 2].max()))
    depth = -body_low; rows = []
    for part in record.get('parts', []):
        pts = corners(part, origin, delta)
        low, high = float(pts[:, 2].min()), float(pts[:, 2].max())
        broad = np.prod(part['size'][:2]) >= .15*np.prod(ref['size'][:2])
        below = float((delta @ (np.asarray(part['position'])-origin))[2]) < 0
        # At an edge, narrow end projections are deliberately outside support.
        # A broad installed member underneath contributes its entire thickness,
        # even if independently estimated envelopes overlap before the turn.
        if edge and not broad:
            continue
        stack = high-low if broad and below else 0.
        candidate = max(-low, -body_low+stack)
        depth = max(depth, candidate)
        rows.append(dict(source=part.get('source', 'registered_member'),
            thicknessM=high-low, below=below, stackContributionM=stack,
            referenceToContactM=candidate))
    shape = record.get('edgeGeometry') or {}
    if edge and shape.get('bearingZM') is not None:
        depth = max(depth, -float(shape['bearingZM']))
    # Ordinary placement has no overhang: every known low point must clear.
    if not edge and shape.get('envelope') and bool(shape.get('turnOver')) == bool(turn_over):
        depth = max(depth, -float(np.min(np.asarray(shape['envelope'])[:, 2])))
    scale = float(min(ref['size']))
    uncertain = bool(ref.get('registration', {}).get('lowConfidence') or
        shape.get('inferred') or (record.get('bearingRecovery') or {}).get('members') == 0)
    # Missing/ambiguous geometry spends an upward allowance, never subtracts a
    # best-fit residual. A bound is an estimate, not a claim about hidden truth.
    model_uncertainty = max(float(ref.get('heightUncertaintyM', 0.)),
        float((contact or {}).get('heightUncertaintyM', 0.)),
        scale/2 if uncertain else 0.)
    uncertainty = max(0., surface_uncertainty)+max(0., body_uncertainty)+model_uncertainty
    uncertainty = min(uncertainty, max(scale/2, .04))  # bounded free fall: a long drop bounces the load off a narrow support
    nominal = float(top+depth)
    return dict(source='registered_post_turn_support_stack', observed=False,
        unconfirmed=True, supportTopM=float(top), bodyContactOffsetM=body_low,
        bodyReferenceToFaceM=-body_low, installedStackM=max(0., depth+body_low),
        referenceToContactM=depth, members=rows, localFaceDisagreementM=disagreement,
        uncertaintyUpM=uncertainty, releaseGapM=gap,
        settledBodyZM=nominal, targetBodyZM=nominal+uncertainty+gap,
        bodyGraspZM=nominal+(body_low+body_high)/2,
        bodyTopZM=nominal+body_high, turnOver=bool(turn_over))


def grasp_lift(planner, scene, frame, r1, target):
    """Convert a body-reference target to lift using the actual grasp TCP.

    Slide-arm lift is a joint coordinate, not the world TCP height. Other arms
    expose a virtual TCP lift by contract. The constant base/mount displacement
    must not be rotated along with the object-to-TCP vector.
    """
    from steward.ego.arm import Arm
    current = planner._arm_joints(scene)
    moved = r1 @ np.asarray(frame['offsetM'])
    tcp_shift = 0.
    if type(planner.arm) is Arm:
        local_tcp = np.asarray(frame['tcpOffsetM'])
        mount = float(frame['linkPositionM'][2])-current['lift']
        tcp_shift = mount+float((r1 @ local_tcp)[2])
    return float(target-moved[2]-tcp_shift), dict(
        gripOffsetAfterTurnM=moved.tolist(), tcpMinusLiftM=tcp_shift,
        sourceKinematics='registered_grasp_tcp_and_encoder_kinematics')


def rebase(new, old, delta, report):
    """Use the same settled reference for subsequent grasps and corrections."""
    ref = new['reference']; ref['position'][2] = report['settledBodyZM']
    ref.update(landingTopM=report['bodyTopZM'], graspHeightM=report['bodyGraspZM'])
    new['releaseHeight'] = deepcopy(report)
    new['entity'].pose.value['p'] = list(ref['position'])
    for previous, part in zip(old.get('parts', []), new.get('parts', [])):
        part['position'] = (np.asarray(ref['position'])+delta @
            (np.asarray(previous['position'])-old['reference']['position'])).tolist()
        prior = np.asarray(previous.get('rotation', previous.get('rotationM',
            matrix_from_quat(quat_from_yaw(previous.get('yaw', 0.))))))
        part['rotation'] = (delta @ prior).tolist()
    if old.get('contactReference'):
        previous=old['contactReference']; contact=new['contactReference']
        contact['position']=(np.asarray(ref['position'])+delta @
            (np.asarray(previous['position'])-old['reference']['position'])).tolist()
        prior=np.asarray(previous.get('rotationM', matrix_from_quat(quat_from_yaw(previous['yaw']))))
        contact['rotationM']=(delta @ prior).tolist()


def enforce_floor(runtime, plan):
    """One encoder-only upward correction before opening; no new camera frame."""
    if not enabled() or not plan.get('assemblyReleaseHeight'):
        return
    report=plan['assemblyReleaseHeight']
    try:
        body=runtime.link.sensors()
    except (AttributeError, OSError, RuntimeError):
        body=dict(joints=runtime.estimator.joints, odometry=runtime.estimator.odometry)
        report['executionUnconfirmed']=True
    with runtime._lock:
        position=runtime.estimator.held_position(body.get('joints'), body.get('odometry'))
    if position is None:
        report['executionUnconfirmed']=True
        return  # The conservative plan remains executable when the estimate is absent.
    deficit=max(0., report['targetBodyZM']-float(position[2]))
    report['encoderBodyZM']=float(position[2]); report['upwardCorrectionM']=deficit
    if deficit:
        ex=runtime.executor
        current=float(ex._state()['joints']['lift'])
        lift=current+deficit+ex.config.release_joint_tol
        from steward.ego import placement_support_margin
        if placement_support_margin.enabled():
            # Called after release settling: use the attained static error, not
            # a robot-specific gravity offset or a moving-target residual.
            setpoint=float(getattr(ex,'_setpoints',{}).get('lift',current))
            sag=max(0.,setpoint-current)
            lift += sag
            report['encoderSagCompensationM']=sag
        try:
            ex._set_joint('lift', lift, tol=(ex.config.joint_tol if placement_support_margin.enabled() and hasattr(ex.config,'joint_tol') else ex.config.release_joint_tol))
            ex._settle_for_release()
            plan['lift']=lift
        except Exception as exc:  # a small unreached correction must never keep the load in the hand
            report['upwardCorrectionFailed']=str(exc)[:120]; report['executionUnconfirmed']=True


def compact_stages(post):
    """Keep each release equation in the summary; primitive receipts stay full."""
    fields=('supportTopM','bodyReferenceToFaceM','installedStackM','uncertaintyUpM',
        'releaseGapM','settledBodyZM','targetBodyZM','turnOver','tcpMinusLiftM',
        'gripOffsetAfterTurnM','encoderBodyZM','upwardCorrectionM','executionUnconfirmed')
    from steward.ego import placement_support_margin
    if placement_support_margin.enabled():
        fields += ('upwardCorrectionFailed','encoderSagCompensationM')
    def rounded(value):
        if isinstance(value, float): return round(value, 6)
        if isinstance(value, list): return [rounded(v) for v in value]
        if isinstance(value, dict): return {k:rounded(v) for k,v in value.items()}
        return value
    for stage in post.get('stages', []):
        stage.pop('wallStart', None); stage.pop('predictWallS', None); stage.pop('wallS', None)
        receipt=stage.get('receipt', {})
        if not isinstance(receipt, dict): continue
        plan=receipt.get('plan', {})
        if plan.get('assemblyReleaseHeight'):
            plan['assemblyReleaseHeight']={k:v for k,v in plan['assemblyReleaseHeight'].items() if k in fields}
            # All of these are duplicate vertical-motion diagnostics; lift and
            # assemblyVerticalRelease retain the actual command and withdrawal.
            for key in ('edgeReleaseHeight','edgeLowRelease','edgeDescentM','edgeVerticalClearanceM'):
                plan.pop(key, None)
        if plan: receipt['plan']=rounded(plan)
