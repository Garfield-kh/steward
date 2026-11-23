"""Conservative skill contracts over robot category knowledge, tracks and releases.

No missing observation is a negative observation. Counters count denied receipts
or release checks, not objects, episodes or physical task failures.
"""
from copy import deepcopy
import math

import numpy as np

from steward.ego.priors import prior


KINDS = ('role', 'component', 'face', 'flip')
SKILLS = {'place', 'assemble_onto', 'fasten_onto', 'seat_screw', '_fastener_contact'}


def result(status='unknown', reason='evidence_missing', message='', **evidence):
    return dict(status=status, reasonCode=reason, blockedCount=int(status == 'blocked'),
                unconfirmedCount=int(status == 'unknown'), **({'message': message} if message else {}), **evidence)


def empty_report():
    return {k: result('not_applicable', 'not_applicable') for k in KINDS}


def role(track):
    return prior(track.category).get('assemblyRole') if track is not None else None


def fresh_view(track, stamp, since=-math.inf):
    """Only independent, complete, recent RGB-D; carried/fused poses cannot veto."""
    if track is None:
        return None
    for v in (getattr(track, 'manipulation_view', None), getattr(track, 'assembly_observation', None)):
        if (v and v.get('source') == 'rgbd_segment' and since < v.get('stamp', -math.inf) <= stamp
                and stamp-v['stamp'] <= 20 and not v.get('truncated', True)
                and v.get('invalidDepthFraction', 1) <= .05
                and all(k in v for k in ('centreM', 'visibleSizeM', 'boxYawRad'))
                and np.isfinite([*v['centreM'], *v['visibleSizeM'], v['boxYawRad']]).all()
                and min(v['visibleSizeM']) > 0):
            return v
    return None


def _margin(*tracks):
    # Measurement uncertainty, never a particular workpiece's dimensions.
    return 3*sum(float(getattr(t, 'sigma', 0.)) for t in tracks if t is not None)


def component_check(rt, scene, part_id, body_id, hole, point=None):
    """Release history supports continuity; fresh contradictory geometry wins."""
    est = rt.estimator
    part, body = est.tracks.get(part_id), est.tracks.get(body_id)
    frame = getattr(est, 'released_interfaces', {}).get(part_id, {})
    if frame.get('body') != body_id:
        frame = {}
    if hole and scene.find(body_id):
        from steward.ego.fasten_plan import hole_point
        try:
            point = hole_point(rt.esim, scene.find(body_id), hole)
        except (ValueError, KeyError, TypeError):
            pass
    if part_id is None or point is None:
        return result(reason='component_or_entry_unresolved', message='上件身份或目标位置不确定，放行。')
    part_release = (getattr(part, 'placement', None) or {}).get('releasedAt', -math.inf)
    body_release = (getattr(body, 'placement', None) or {}).get('releasedAt', -math.inf)
    pv = fresh_view(part, est.stamp, max(frame.get('releasedAt', -math.inf), part_release))
    bv = fresh_view(body, est.stamp, body_release)
    if pv and bv and not getattr(part, 'held', False) and not getattr(body, 'held', False):
        p = np.asarray(point)
        margin = _margin(part, body)
        bc, bs = math.cos(bv['boxYawRad']), math.sin(bv['boxYawRad'])
        body_local = np.array([[bc, bs], [-bs, bc]]) @ (p[:2]-bv['centreM'][:2])
        if np.any(abs(body_local) > np.asarray(bv['visibleSizeM'][:2])/2+margin):
            return result(reason='entry_reference_uncertain', message='目标入口与新鲜承载件观测不一致，放行并保留不确定。')
        delta = p[:2]-pv['centreM'][:2]
        c, s = math.cos(pv['boxYawRad']), math.sin(pv['boxYawRad'])
        local = np.array([[c, s], [-s, c]]) @ delta
        outside = bool(np.any(abs(local) > np.asarray(pv['visibleSizeM'][:2])/2+margin))
        # A separately observed component entirely below the carrier cannot cover
        # its upward entry. Overlapping/partial masks remain ambiguous.
        below = pv['centreM'][2]+pv['visibleSizeM'][2]/2+margin < bv['centreM'][2]-bv['visibleSizeM'][2]/2
        if outside or below:
            return result('blocked', 'component_not_at_entry',
                f'先将上件 {part_id} 放到承载件 {body_id} 的目标入口附近，再放紧固件。',
                part=part_id, body=body_id, observedAt=pv['stamp'], outsideFootprint=outside, belowCarrier=below)
        return result('satisfied', 'component_observed_at_entry', part=part_id, body=body_id)
    pairs = frame.get('holePairs', [])
    if frame and (hole is None or any(p['bodyHole'] == hole for p in pairs)):
        return result('estimated', 'component_release_recorded',
                      '沿用上件释放账本；未独立确认当前贴合。', part=part_id, body=body_id)
    return result(reason='component_observation_missing', message='未确认上件已到位；证据不足，放行。',
                  part=part_id, body=body_id)


def face_check(rt, scene, part_id, body_id, params):
    """Use plan-selected interfaces and a measured face, never default identity R.

    A catalogue bearing slab below a protrusion gives a semantic local +Z anchor.
    Without that shape property or a resolved full-shape signature, up is unknown.
    """
    from steward.ego.fasten_plan import interface_holes, mating_holes
    from steward.ego.fasten_geometry import vertical_signature
    from steward.ego.classify import SHAPES
    body = scene.find(body_id)
    track = rt.estimator.tracks.get(body_id)
    if body is None or track is None:
        return result(reason='face_observation_missing', message='当前哪一面朝上不确定，放行。')
    requirements = [r for r in (getattr(rt, 'plan', None) or {}).get('assemblyRequirements', [])
                    if r['part'] == part_id and r['body'] == body_id]
    if len({r['hole'] for r in requirements}) > 1:
        return result(reason='plan_face_conflict', message='计划给出的装配面互相冲突，放行并保留不确定。')
    selector = requirements[0]['hole'] if requirements else params.get('hole')
    try:
        holes = interface_holes(rt.esim, body)
        if selector is not None:
            selected = rt.esim._select_holes(holes, selector, body_id)
        elif scene.find(part_id):
            selected = mating_holes(rt.esim, scene, dict(params, part=part_id, body=body_id))
        else:
            selected = []
    except (ValueError, KeyError, TypeError):
        selected = []
        holes = []
    levels = [h['p'][2] for h in holes]
    chosen = [h['p'][2] for h in selected]
    if not chosen or not levels or max(levels) <= min(levels):
        return result(reason='required_face_unresolved', message='计划未确定相对的装配面，放行。')
    shape = SHAPES.get(track.category, {})
    dims, bearing = shape.get('dims'), shape.get('bearing_height')
    v = fresh_view(track, rt.estimator.stamp)
    # A carried raw view or a pre-motion face must not stop assembly at the new pose.
    since = max(getattr(track, 'interface_motion_at', -math.inf),
                (getattr(rt.estimator, 'stationary_assemblies', None).records.get(body_id, {})
                 .get('reference', {}).get('ownReleaseAt', -math.inf)))
    if v is None or v['stamp'] <= since or getattr(track, 'held', False):
        return result(reason='face_observation_missing', message='当前哪一面朝上不确定，放行。')
    signature = vertical_signature(v) if 'points' in v else None
    if not dims or bearing is None or not 0 < bearing < dims[2] or signature is None:
        return result(reason='face_semantic_anchor_missing', message='可见形状尚不能对应计划中的面，放行。')
    # Reject merged, tipped or dimensionally inconsistent silhouettes as evidence.
    observed = np.sort(v['visibleSizeM'])
    expected = np.sort(dims)
    slack = _margin(track) + float(shape.get('extent_slack', 0.))
    if np.any(abs(observed-expected) > slack) or abs(v['visibleSizeM'][2]-dims[2]) > slack:
        return result(reason='face_shape_ambiguous', message='面的观测有歧义，放行。')
    up = 1 if signature == 'detail_above' else -1
    mid = (min(levels)+max(levels))/2
    facing = [(z-mid)*up for z in chosen]
    if all(z < 0 for z in facing):
        return result('blocked', 'opposite_face_up', '计划的装配面当前朝下；需要先翻面。',
                      body=body_id, observedAt=v['stamp'], faceSignature=signature)
    if all(z > 0 for z in facing):
        return result('satisfied', 'required_face_up', body=body_id, observedAt=v['stamp'])
    return result(reason='required_face_ambiguous', message='所选接口跨面，放行并保留不确定。')


def evaluate(rt, skill, params, plan=None):
    report = empty_report()
    if skill not in SKILLS:
        return report
    plan = plan or {}
    est, scene = rt.estimator, rt.ego_state()
    held = est.held_id
    operand = params.get('part') if skill in ('assemble_onto', 'fasten_onto') else held
    body_id = params.get('seatBody') or params.get('body') or params.get('on')
    body = scene.find(body_id) if body_id else None
    from steward.ego.fasten_plan import value
    orientation = value(body, 'orientation', {})
    assembly = bool(body and (rt.esim._screw_holes(body) or orientation.get('stationaryAssembly')))
    assembly = assembly or skill in ('assemble_onto', 'fasten_onto', 'seat_screw') or bool(params.get('seatBody'))
    operation_role = role(est.tracks.get(operand))
    seating = skill == 'seat_screw' or skill == '_fastener_contact' and params.get('operation') == 'seat'
    if assembly and (skill != '_fastener_contact' or seating):
        operated = [operand] + ([params.get('part'), params.get('screw')] if seating else [])
        tools = [eid for eid in operated if role(est.tracks.get(eid)) == 'tool']
        if tools:
            report['role'] = result('blocked', 'tool_is_not_component',
                '该物体登记为工具，不能作为上件或紧固件装配；请握持它操作连接，使用后放回普通支撑面。', object=tools[0])
        else:
            report['role'] = result('satisfied', 'operand_role_compatible') if operation_role else result(
                reason='operand_role_unknown', message='物体角色不确定，放行。')
    is_fastener = seating or skill == 'place' and (operation_role == 'fastener' or params.get('seatBody'))
    if assembly and is_fastener:
        part_id = params.get('part') if seating else params.get('on') if params.get('seatBody') else None
        hole = params.get('seatHole', params.get('hole'))
        point = plan.get('seatTarget', {}).get('point')
        if point is None and plan.get('spot') is not None:
            point = [*plan['spot'][:2], 0.]
        if part_id is None:
            requirements = [r for r in (getattr(rt, 'plan', None) or {}).get('assemblyRequirements', [])
                            if r['body'] == body_id and (hole is None or r['hole'] == hole)]
            if len({r['part'] for r in requirements}) == 1:
                part_id = requirements[0]['part']
        if part_id is None:
            frames = [f for f in getattr(est, 'released_interfaces', {}).values() if f['body'] == body_id
                      and (hole is None or any(p['bodyHole'] == hole for p in f.get('holePairs', [])))]
            if len(frames) == 1:
                part_id = frames[0]['entity']
            elif body is not None:
                # Plan/catalogue correspondence, never proximity-only identity.
                from steward.ego.fasten_plan import interface_holes
                candidates = []
                try:
                    targets = rt.esim._select_holes(interface_holes(rt.esim, body), hole, body_id)
                except (ValueError, KeyError, TypeError):
                    targets = []
                for ent in scene.entities:
                    if ent.id == body_id or ent.id == held or not rt.esim._screw_holes(ent):
                        continue
                    try:
                        peers = interface_holes(rt.esim, ent)
                        if any(h.get('group') in p.get('mates', ()) and p.get('group') in h.get('mates', ())
                               for h in targets for p in peers):
                            candidates.append(ent.id)
                    except (ValueError, KeyError, TypeError):
                        pass
                if len(candidates) == 1:
                    part_id = candidates[0]
        report['component'] = component_check(rt, scene, part_id, body_id, hole, point)
    elif assembly and skill in ('assemble_onto', 'fasten_onto', 'place') and operation_role != 'tool':
        report['face'] = face_check(rt, scene, operand, body_id, params)
    return report


def blocked(report):
    return [r['message'] for r in report.values() if r['status'] == 'blocked']


def record_job(job, report, event):
    events = dict(getattr(job, '_precondition_events', {}))
    events[event] = deepcopy(report)
    job._precondition_events = events
    publish(job)


def publish(job):
    events = getattr(job, '_precondition_events', {})
    if not events:
        return
    combined = {k: dict(blockedCount=0, reasonCode=None) for k in KINDS}
    for kind in KINDS:
        rows = [r[kind] for r in events.values() if kind in r and r[kind]['status'] != 'not_applicable']
        if not rows:
            continue
        last = next((r for r in reversed(rows) if r['status'] == 'blocked'), rows[-1])
        combined[kind] = dict(blockedCount=sum(r['blockedCount'] for r in rows), reasonCode=last['reasonCode'])
        unknown = sum(r['unconfirmedCount'] for r in rows)
        if unknown or kind == 'flip':
            combined[kind]['unconfirmedCount'] = unknown
        reasons = list(dict.fromkeys(r['reasonCode'] for r in rows if r['status'] == 'blocked'))
        if len(reasons) > 1:
            combined[kind]['blockedReasonCodes'] = reasons
    if job.postcondition is None:
        job.postcondition = {}
    job.postcondition['preconditions'] = combined


def lift_check(before, samples, support_view, since, stamp, uncertainty):
    """Measured bottom above measured support top; gripper closure/FK is not lift."""
    def valid(v, cutoff):
        return bool(v and v.get('source') == 'rgbd_segment' and cutoff < v.get('stamp', -math.inf) <= stamp
                    and stamp-v['stamp'] <= 20 and not v.get('truncated', True)
                    and v.get('invalidDepthFraction', 1) <= .05
                    and all(k in v for k in ('centreM', 'visibleSizeM')))
    if (not valid(before, -math.inf) or not valid(support_view, -math.inf)
            or before['stamp'] > since or support_view['stamp'] > since):
        return dict(passed=None, reason='lift_baseline_missing')
    top = support_view['centreM'][2]+support_view['visibleSizeM'][2]/2
    margin = max(uncertainty, np.finfo(float).eps)
    seen = [v for v in samples if valid(v, since)]
    # A changed identity/merged region must not count as the lifted body.
    seen = [v for v in seen if np.allclose(np.sort(v['visibleSizeM']),
            np.sort(before['visibleSizeM']), rtol=.2, atol=margin)]
    baseline_bottom = before['centreM'][2]-before['visibleSizeM'][2]/2
    if any(v['centreM'][2]-v['visibleSizeM'][2]/2 > max(top, baseline_bottom)+margin
           and v['centreM'][2] > before['centreM'][2]+margin for v in seen):
        return dict(passed=True, reason='body_observed_clear_of_support')
    if len(seen) >= 2 and len({v['stamp'] for v in seen}) >= 2 and all(
            np.linalg.norm(np.asarray(v['centreM'])-before['centreM']) <= margin for v in seen):
        return dict(passed=False, reason='body_observed_stationary_during_lift')
    return dict(passed=None, reason='lift_not_observed')


def finish_flip(job, orientation, lift, *, turn_over=True):
    """Keep command completion separate from verified physical outcome."""
    from steward.ego.skills import SkillFailed
    face = orientation.get('passed') if turn_over else True
    passed = False if face is False or lift['passed'] is False else True if face is True and lift['passed'] is True else None
    code = ('flip_confirmed' if passed is True else 'flip_not_observed' if passed is False else 'flip_unconfirmed')
    label = '已确认离面并换面。' if turn_over and passed is True else '已确认离面。' if passed is True else (
        '翻面失败：观测到未离面或朝上的面未改变。' if passed is False else '翻面未确认：未能同时核实离面和换面。')
    report = empty_report()
    report['flip'] = result('blocked' if passed is False else 'satisfied' if passed is True else 'unknown', code,
                           label, liftCheck=lift, orientationCheck=orientation)
    record_job(job, report, 'flip_result')
    job.postcondition.update(liftCheck=lift, verified=passed is True, passed=passed,
        # Legacy internal action ledger: the release sequence ran. Public Job
        # serialization names this actuatorSequenceExecuted, never flip success.
        motionAttempted=True, executed=bool(job.postcondition.get('releaseExecuted')),
        flipped=orientation.get('passed') if turn_over else False,
        actionStatus='failed' if passed is False else 'confirmed' if passed is True else 'unconfirmed',
        confirmation='failed' if passed is False else 'confirmed' if passed is True else 'unconfirmed',
        canContinue=passed is not False, guidance=label)
    job.message = label
    if passed is False:
        raise SkillFailed(label)
