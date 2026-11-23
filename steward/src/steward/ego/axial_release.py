"""Continuous shaft alignment and one result per physical release.

All lengths/poses come from the current robot estimate or arm model. Geometry
helpers acquire no observations. The result policy chooses one existing release
observation path and never reads simulation object state.
"""
from copy import deepcopy
import math
import numpy as np
from steward.ego import capabilities


def enabled():
    return capabilities.get('STEWARD_ASSEMBLY_AXIAL_RELEASE') != '0'


def refine_solutions(arm, solutions, axis, normal, tip, goal, current):
    """Remove grid quantization before admission, using the actual arm rotation.

    The original seeds cover the wrist's range. A bounded least-squares solve
    refines them; impossible axes retain their measured residual, never pretend
    that a wrist without pitch has another degree of freedom.
    """
    if not enabled() or not solutions: return solutions
    axis = np.asarray(axis, float); axis /= np.linalg.norm(axis)
    normal = np.asarray(normal, float); normal /= np.linalg.norm(normal)
    g = arm.geometry
    bounds = np.array([[-math.pi, math.pi], [-g.wrist_roll_max, g.wrist_roll_max],
                       [0., g.wrist_pitch_max]])
    def rotation(q):
        a, roll, pitch = q
        c,s = math.cos(a),math.sin(a)
        return np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]]) @ arm.tool_rotation(0.,roll,pitch)
    refined=[]
    for seed in sorted(solutions,key=lambda x:x[:2])[:24]:
        q=np.array(seed[2:5],float)
        for _ in range(12):
            residual=rotation(q) @ axis-normal
            h=math.sqrt(np.finfo(float).eps)
            jac=np.column_stack([(rotation(q+np.eye(3)[i]*h) @ axis-normal-residual)/h
                                 if bounds[i,1]>bounds[i,0] else np.zeros(3) for i in range(3)])
            delta=np.linalg.lstsq(jac,-residual,rcond=math.sqrt(np.finfo(float).eps))[0]
            candidate=np.clip(q+delta,bounds[:,0],bounds[:,1])
            if np.linalg.norm(rotation(candidate) @ axis-normal)>=np.linalg.norm(residual): break
            q=candidate
        r=rotation(q);error=math.acos(float(np.clip((r @ axis) @ normal,-1.,1.)))
        turn=abs(q[1]-current.get('wrist_roll',0.))+abs(q[2]-current.get('wrist_pitch',0.))
        refined.append((error,turn,*map(float,q),np.asarray(goal)-r @ tip))
    return refined


def pinch(target, stamp):
    """Use an existing directed endpoint observation to pinch the head-side shaft."""
    if not enabled(): return None
    from steward.ego.fasten_plan import value
    from steward.ego.grasp_endpoints import observed
    feature=value(target,'toolFeature')
    if not observed(feature) or not 0 <= stamp-feature['stamp'] <= 20:
        return None
    head,tip=np.asarray(feature['head']),np.asarray(feature['tip'])
    # Midway between the shaft centre and the head stays inside the observed
    # object, leaving more shaft below the fingers during final lowering.
    point=(3*head+tip)/4
    return dict(point=point.tolist(),axis=deepcopy(feature['axis']),
                source=feature.get('source'),observedAt=feature['stamp'])


def checked(attempt):
    return bool(enabled() and attempt.get('axialReleaseChecked') is not None and
                attempt['axialReleaseChecked'] == attempt.get('seatedAt'))


def handled(attempt):
    return bool(checked(attempt) and attempt.get('axialReleaseHandled') is not None and
                attempt['axialReleaseHandled'] == attempt.get('seatedAt'))


def grasp_heading(arm, axis):
    """Put the observed inclination in the model's controllable roll plane.

    A roll-only wrist cannot remove a shaft component along its roll axis.
    Choose the empty-hand heading before closing so that component is absent.
    A tool whose roll is axial has no such plane; retain its existing grasp.
    """
    axis = np.asarray(axis, float)
    h = math.sqrt(np.finfo(float).eps)
    neutral = arm.tool_rotation(0., 0., 0.)
    tangent = ((arm.tool_rotation(0., h, 0.)-arm.tool_rotation(0., -h, 0.)) /
               (2*h)) @ neutral.T @ np.array([0., 0., 1.])
    if np.linalg.norm(tangent[:2]) <= h or np.linalg.norm(axis[:2]) <= h:
        return None
    return math.atan2(axis[1], axis[0])-math.atan2(tangent[1], tangent[0])


def clearance_order(arm, solutions, points, shapes, safe, diagnostics):
    """Break equally upright pose ties using the observed neighbour geometry.

    Sweep the calibrated open-hand boxes over the already planned vertical exit.
    This is a ranking of available actions, not a new visibility/collision gate.
    """
    from steward.math3d import matrix_from_quat
    points = np.asarray(points, float)
    boxes = [s for s in shapes if s.get('shape', s.get('type')) == 'box' and len(s.get('size', [])) == 3]
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not boxes:
        return sorted(solutions, key=lambda x:x[:2])
    ranked=[]
    for row in solutions:
        error,turn,alpha,roll,pitch,endpoint=row
        c,s=math.cos(alpha),math.sin(alpha)
        rotation=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]]) @ arm.tool_rotation(0.,roll,pitch)
        hit=np.zeros(len(points),bool);clearance=math.inf
        for shape in boxes:
            pose=shape.get('pose') or {}
            r=rotation @ matrix_from_quat(pose.get('q',[1.,0.,0.,0.]))
            travel=np.array([0.,0.,max(0.,safe-endpoint[2])])
            centre=endpoint+rotation @ np.asarray(pose.get('p',[0.,0.,0.]))+travel/2
            half=np.asarray(shape['size'])/2+abs(r.T@travel)/2
            distance=abs((points-centre)@r)-half
            hit |= np.all(distance<=0,axis=1)
            clearance=min(clearance,float(np.linalg.norm(np.maximum(distance,0),axis=1).min()))
        report=dict(intersectingPoints=int(hit.sum()),observedPoints=len(points),clearanceM=clearance)
        diagnostics[id(row)]=report
        # Numerical solver noise must not erase the yaw freedom of an upright shaft.
        ranked.append(((round(error*error,12),int(hit.sum()),-clearance,turn),row))
    return [row for _,row in sorted(ranked,key=lambda pair:pair[0])]


def order_for_entry(planner, scene, params, solutions, safe, diagnostics):
    ordered=sorted(solutions,key=lambda x:x[:2])[:24]
    if not enabled(): return ordered
    from steward.ego.fasten_plan import value
    part=scene.find(params.get('part'))
    view=value(part,'contactView') or {}
    if view.get('source') != 'rgbd_segment' or not 0 <= scene.stamp-view.get('stamp',-math.inf) <= 20:
        return ordered
    robot=planner._self(scene)
    link=next((l for l in robot.rigid.links if l.id==planner.arm.geometry.gripper),None)
    if link is None: return ordered
    geometry=link.geometry or {}
    result=clearance_order(planner.arm,ordered,view.get('points',[]),
        geometry.get('collision') or geometry.get('visual') or [],safe,diagnostics)
    for report in diagnostics.values():
        report.update(source='existing_component_rgbd_and_robot_geometry',observedAt=view['stamp'])
    return result


def record(session, attempt, interface=None):
    """Only the native post-release result is consumed; reentry cannot renew it."""
    if not enabled() or checked(attempt) or not attempt.get('releaseExecuted'): return False
    from steward.ego import assembly_outcome
    attempt['axialReleaseChecked']=attempt.get('seatedAt')
    attempt['axialReleaseCheck']=deepcopy(attempt.get('seatCheck') or {})
    if session is not None and hasattr(session, 'emit'):
        session.emit('outcome_checked',interface=interface or session.state.get('interface'),
            body=attempt.get('body'),hole=attempt.get('hole'),screw=attempt.get('screw'),
            phase='after_release',outcome=assembly_outcome.state(attempt),
            seatedAt=attempt.get('seatedAt'),check=deepcopy(attempt.get('seatCheck') or {}))
    return True


def lower(ex, target):
    """Smooth the existing last lift segment without adding dwell or samples."""
    from steward.ego.fasten_seat import lift_to
    if not hasattr(ex, '_ramp_joints'):
        return lift_to(ex, target)
    previous=getattr(ex,'_gentle',False)
    ex._gentle=True
    try:
        # Keep the ordinary arm speed; only its interpolation profile changes.
        speed=ex.config.arm_speed
        seconds=abs(ex._state()['joints']['lift']-target)/speed if speed else 0.
        if seconds>ex.config.tick_s:
            ex._ramp_joints({'lift':target+getattr(ex,'_lift_settled_offset',0.)},seconds)
        lift_to(ex,target,ramp=False)
    finally:
        ex._gentle=previous


def result_view(controller, job, attempt, placement=None):
    """Use the place-owned view, or the existing single release view, never both."""
    if checked(attempt): return
    direct = attempt.get('holeSeatObservation')
    if direct and direct.get('releasedAt') == attempt.get('seatedAt'):
        return
    from steward.ego import wrist_observation
    if wrist_observation.release_reporting():
        from steward.ego.wrist_fasten import released
        if released(controller, job, attempt, placement) is not None:
            return
    controller.wait_view(attempt['screw'], attempt['seatedAt'])
