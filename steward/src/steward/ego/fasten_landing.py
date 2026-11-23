"""Bounded same-fastener replay and truthful post-release advice."""
from steward.ego import capabilities
from steward.ego.fasten_plan import value

ENABLED = capabilities.get('STEWARD_ASSEMBLY_LANDING_CHECK') != '0'


def replay_evidence(scene, attempt, check):
    from steward.ego.released_object import recovery_evidence
    now = getattr(scene, 'stamp', None)
    if check.get('source') == 'released_object_rgbd':
        # Native synchronized frames can be newer than the slow scene update.
        # The self pose carries the live sensor clock; do not reject those
        # independent pixels merely because semantic perception is behind.
        try:
            live = scene.entity(scene.self_id).pose.t
            if live is not None: now = max(now, live) if now is not None else live
        except (AttributeError, KeyError):
            pass
    located = recovery_evidence(attempt, check, now)
    if located is not None:
        return located
    from steward.ego.fasten_accounting import departure
    evidence = departure(scene, attempt)
    if evidence:
        return evidence
    if check.get('source') == 'post_release_hole_rgbd' and check.get('passed') is False:
        from steward.ego.fasten_supply import measured_fastener
        entity = scene.find(attempt['screw'])
        feature = measured_fastener(entity, scene.stamp) if entity is not None else None
        since = max(attempt['seatedAt'], attempt.get('transportedAt', 0))
        # The hole image proves the miss; a fresh, separately tracked original
        # proves which object can be picked. Never bind the closest loose object.
        if (feature and not value(entity, 'held', False) and
                since < feature.get('stamp', -1) <= scene.stamp):
            return dict(screw=entity.id, observedAt=feature['stamp'],
                        source='post_release_same_identity_pose', check=check)
        return None
    if not ENABLED:
        return None
    wrist = attempt.get('wristSeatCheck') or {}
    if (wrist.get('passed') is False and check.get('source') == 'wrist_rgbd_endpoint_width'
            and wrist.get('observedAt', -1) > max(attempt['seatedAt'], attempt.get('transportedAt', 0))):
        return dict(screw=attempt['screw'], observedAt=wrist['observedAt'],
                    source='independent_same_screw_not_seated', check=wrist)
    feedback=attempt.get('landingFeedback') or {}
    from steward.ego.fasten_completion import RETRY
    if RETRY and check.get('passed') is None and check.get('source') in ('rgbd_endpoint_width','wrist_rgbd_endpoint_width'):
        from steward.ego.fasten_supply import measured_fastener
        import numpy as np
        # Reacquire only this separately resolved original identity. Other supply
        # remains forbidden while occupancy is unknown.
        entity=scene.find(attempt['screw'])
        feature=measured_fastener(entity,scene.stamp) if entity is not None else None
        if (feature and feature.get('stamp',-1)>max(attempt['seatedAt'],attempt.get('transportedAt',0))
                and np.linalg.norm(np.asarray(feature['head'])-attempt['entry'])<=feature['lengthM']+.025):
            return dict(screw=attempt['screw'],observedAt=feature['stamp'],
                source='independent_original_identity_reacquisition')
    # A lying screw can intersect the capture disk and never count as departed.
    # Fresh, independent endpoints of THIS identity can still disprove seating.
    if check.get('passed') is not False or check.get('reason') not in ('independent_screw_geometry', 'independent_screw_axis'):
        return None
    for eid in dict.fromkeys([attempt['screw'], *attempt.get('screwAliases', [])]):
        entity = scene.find(eid)
        if entity is None or value(entity, 'held', False):
            continue
        from steward.ego.fasten_supply import measured_fastener
        from steward.ego.fasten_release import seat_check
        if check.get('reason') == 'independent_screw_axis':
            from steward.ego.release_observation import axis_check
            measured = axis_check(entity, attempt['entry'],
                max(attempt['seatedAt'], attempt.get('transportedAt', 0)), scene.stamp)
            if measured:
                return dict(screw=eid, observedAt=measured['observedAt'],
                    source='independent_same_screw_not_seated', check=measured)
            continue
        feature = measured_fastener(entity, scene.stamp) or value(entity, 'toolFeature')
        since = max(attempt['seatedAt'], attempt.get('transportedAt', 0))
        if (not feature or feature.get('source') != 'rgbd_endpoint_width'
                or not since < feature.get('stamp', -1) <= scene.stamp
                or scene.stamp-feature['stamp'] > 20):
            continue
        measured = seat_check(entity, attempt.get('componentFrame'), attempt['entry'], since, scene.stamp)
        if measured.get('passed') is False and measured.get('freshEndpointObserved'):
            return dict(screw=eid, observedAt=feature['stamp'],
                        source='independent_same_screw_not_seated', check=measured)
    return None


def advice(rows, attempts, part, body):
    by_hole = {a['hole']:a for a in attempts}
    failed = [r['hole'] for r in rows if r['status'] == 'failed']
    retryable = [h for h in failed if by_hole[h].get('seatRetryCount', 0) < 1]
    pending = [r['hole'] for r in rows if r['status'] == 'unattempted']
    for row in rows:
        row['replaysUsed'] = by_hole.get(row['hole'], {}).get('seatRetryCount', 0)
        row['replayEligible'] = row['hole'] in retryable
    confirmed = sum(r['status'] == 'confirmed' for r in rows)
    prefix = f'接口 {part} → {body}：确认 {confirmed}/{len(rows)}；'
    if failed:
        line = prefix+'明确未入座孔 '+', '.join(failed)+'。'
        line += ('可补一次（仅拾回同一颗）：'+', '.join(retryable)+'。') if retryable else '同颗重放预算已用完；观察失败原因。'
    elif pending:
        line = prefix+'尚未释放孔 '+', '.join(pending)+'；仅续做未释放孔，保留已放上件。'
    elif confirmed == len(rows):
        line = prefix+'各孔均已放置一颗，均有观测确认入座；可继续后续步骤。'
    else:
        line = prefix+'各孔均已放置一颗（其余未能目视确认）；未确认的不补，保留不确定；建议进入下一步并核对。'
    return dict(retryableFailedHoles=retryable, exhaustedFailedHoles=[h for h in failed if h not in retryable],
        resumeSuggestion=dict(skill='fasten_onto', part=part, body=body,
            remainingHoles=pending+retryable) if pending or retryable else None,
        guidance=line, interfaceSummary=line)


def flip_advice(frames, attempts, body):
    from steward.ego.fasten_seat import flip_advice as original
    attempts = list(attempts)
    result = original(frames, attempts, body)
    failed = [a for a in attempts if a['body'] == body and a.get('seatCheck', {}).get('passed') is False]
    if ENABLED and failed:
        result.update(failedHoles=[a['hole'] for a in failed],
            retryableFailedHoles=[a['hole'] for a in failed if a.get('seatRetryCount', 0) < 1],
            message='仍有明确未入座孔；可对原紧固件补一次，预算用完则观察失败原因；未确认的不补。')
    return result
