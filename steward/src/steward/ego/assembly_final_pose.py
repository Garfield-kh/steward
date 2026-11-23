"""Caller-selected final face, using the same robot evidence as interface access."""
from copy import deepcopy
from types import SimpleNamespace

from steward.ego import action_budget as ab
from steward.ego.skills import SkillFailed


def estimate(rt, samples=()):
    """Additional simulation allowance; never reserve it by skipping goal work."""
    if samples:
        seconds = max(s['simS'] for s in samples)
        return dict(simS=round(seconds*1.25, 3), method='session_release_duration_with_reserve',
                    samples=deepcopy(list(samples)), reserveFraction=.25)
    seconds, detail = ab.drive_estimate(SimpleNamespace(rt=rt, ex=rt.executor), holes=0)
    return dict(simS=round(seconds, 3), method='pick_place_motion_envelope',
                detail=detail, includes='pick, transport, controlled turn, place, observation')


def goal_estimate(rt, goal):
    parts = {i['part'] for i in goal['interfaces']}
    roots = {i['body'] for i in goal['interfaces']} - parts
    per_body = estimate(rt)
    return dict(simS=round(len(roots)*per_body['simS'], 3), additional=True,
        maximumRestorations=len(roots), perRestoration=per_body,
        method='one_final_restore_per_root_if_needed')


def face(session, body_id):
    """Do not infer a face from the success flag or a nominal turn command."""
    from steward.ego.assembly_status import interface_access
    from steward.ego.fasten_plan import interface_holes
    scene = session.rt.ego_state()
    body = scene.find(session.canonical(body_id))
    if body is None:
        return dict(status='unknown', holes=[], reason='bound_body_unavailable')
    holes = interface_holes(session.rt.esim, body)
    checks = {}
    for hole in holes:
        access = interface_access(session.rt.esim, body, {hole['id']})
        checks[hole['id']] = {k:deepcopy(access[k]) for k in (
            'status', 'reason', 'source', 'faceResolution', 'bodyUpZEstimate') if k in access}
    if not checks or any(c['status'] == 'unknown' for c in checks.values()):
        return dict(status='unknown', holes=[], reason='final_pose_face_unresolved', evidence=checks)
    exposed = sorted(h for h, c in checks.items() if c['status'] == 'estimated_accessible')
    if not exposed:
        return dict(status='unknown', holes=[], reason='final_pose_no_upward_interface', evidence=checks)
    return dict(status='resolved', holes=exposed, evidence=checks)


def initialize(session):
    state = session.state
    if 'finalPose' in state:
        if any(t['initial']['status'] != 'resolved' for t in state['finalPose']['targets']):
            raise SkillFailed('final_pose_initial_face_unresolved')
        return
    policy = state['goal'].get('finalPose', 'as_is')
    report = state['finalPose'] = dict(policy=policy, achieved=policy == 'as_is', targets=[])
    if policy == 'as_is':
        return
    # A dependent body's orientation belongs to the connected root assembly.
    parts = {i['part'] for i in state['goal']['interfaces']}
    bodies = sorted({i['body'] for i in state['goal']['interfaces']} - parts)
    for body in bodies:
        initial = face(session, body)
        report['targets'].append(dict(body=body, initial=initial, required=False,
            attempted=False, executed=False, faceResolved=False, achieved=False))
    session.emit('final_pose_initial', finalPose=deepcopy(report), budgetEstimate=goal_estimate(session.rt, state['goal']))
    if any(t['initial']['status'] != 'resolved' for t in report['targets']):
        report['reason'] = 'final_pose_initial_face_unresolved'
        raise SkillFailed(report['reason'])


def restore(session):
    from steward.ego.assembly_session import check_budget
    report = session.state['finalPose']
    if report['policy'] == 'as_is':
        return True
    # A failed initial snapshot can never be replaced by a later face on resume.
    if any(t['initial']['status'] != 'resolved' for t in report['targets']):
        raise SkillFailed('final_pose_initial_face_unresolved')
    report['achieved'] = False
    report.pop('reason', None)
    session.state['phase'] = 'final_pose'
    try:
        for target in report['targets']:
            check_budget()
            current = face(session, target['body'])
            if current['status'] == 'unknown':
                session.action('_assembly_observe', dict(body=session.canonical(target['body']), changeView=True))
                current = face(session, target['body'])
            target.update(current=current, achieved=False)
            if current['status'] == 'unknown':
                raise SkillFailed('final_pose_face_unresolved')
            if current['holes'] == target['initial']['holes']:
                target['achieved'] = True
                target['faceResolved'] = bool(target['executed'])
                continue
            target['required'] = True
            allowance = estimate(session.rt, [s for s in session.state.get('flipDurations', [])
                                              if session.canonical(s['body']) == session.canonical(target['body'])])
            target['budgetEstimate'] = allowance
            session.emit('final_pose_required', finalPose=deepcopy(report))
            if target['attempted']:
                raise SkillFailed('final_pose_restore_already_attempted')
            remaining = session.deadline.seconds-session.deadline.sim_s
            if remaining < allowance['simS']:
                target['remainingSimS'] = max(0., remaining)
                raise SkillFailed('final_pose_insufficient_budget')
            target['attempted'] = True
            try:
                session.action('flip_assembly', dict(body=session.canonical(target['body'])))
            finally:
                result = session.state.get('lastResult') or {}
                target['executed'] = bool(result.get('releaseExecuted'))
                target['result'] = {k:deepcopy(result[k]) for k in (
                    'releaseExecuted', 'faceResolution', 'orientationCheck', 'bodyGraspCheck',
                    'finalPoseStability', 'flipRotationEvents') if k in result}
                target['current'] = face(session, target['body'])
                target['faceResolved'] = target['executed'] and target['current']['status'] == 'resolved'
                target['achieved'] = bool(target['faceResolved'] and
                    target['current']['holes'] == target['initial']['holes'] and not session.rt.estimator.held_id)
                session.emit('final_pose_result', finalPose=deepcopy(report))
            if not target['achieved']:
                report['reason'] = 'final_pose_not_reached' if target['faceResolved'] else 'final_pose_unverified'
                return False
        report['achieved'] = all(t['achieved'] for t in report['targets']) and not session.rt.estimator.held_id
        return report['achieved']
    except Exception as exc:
        report.update(achieved=False, reason=str(exc))
        raise
    finally:
        session.state.pop('phase', None)
        session.emit('final_pose_end', finalPose=deepcopy(report))


def active_restore():
    from steward.ego.assembly_session import current
    session = current()
    return session is not None and session.state.get('phase') == 'final_pose'


def landing(record, turn_over):
    """Lowest registered contact hull, with the existing uncertain COM model.

    All co-height contacts contribute, including members installed last. Boxes
    are registered envelopes, so this is an estimate, not measured stability.
    """
    import numpy as np
    from steward.ego.assembly_edge import corners, supports, contact_margin
    from steward.math3d import matrix_from_quat, quat_from_yaw
    ref = record['reference']
    items = [ref, *record.get('parts', [])]
    report = dict(passed=None, source='registered_envelopes', observed=False,
                  reason='final_pose_support_geometry_missing', contacts=[])
    if any(not all(k in item for k in ('position', 'size', 'entity')) for item in items):
        return report
    origin = np.asarray(ref['position'])
    rotation = matrix_from_quat(quat_from_yaw(ref.get('yaw', 0.)))
    delta = rotation @ np.diag([1., -1., -1.]) @ rotation.T if turn_over else np.eye(3)
    points = [corners(item, origin, delta) for item in items]
    if not all(np.isfinite(p).all() for p in points):
        return report
    lowest = min(p[:, 2].min() for p in points)
    faces = [p[np.isclose(p[:, 2], lowest, atol=1e-6, rtol=0.)] for p in points]
    contact = np.vstack(faces)
    weights = [max(float(np.prod(item['size'])), 1e-9) for item in items]
    com = np.average([delta @ (np.asarray(item['position'])-origin) for item in items], axis=0, weights=weights)
    margin = contact_margin(contact, record)
    stable = supports(contact, com, margin)
    return dict(report, passed=stable, reason='registered_contact_supports_com' if stable else
                'final_pose_unstable_contact', contacts=[item['entity'] for item, f in zip(items, faces) if len(f)],
                massSource='uniform_volume_estimate_with_density_uncertainty',
                # A degenerate contact has no finite margin; keep strict JSON.
                comMarginM=float(margin) if np.isfinite(margin) else None)
