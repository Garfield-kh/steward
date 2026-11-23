"""Prediction-only sequencing advice from the robot's per-hole release ledger."""
from steward.ego import capabilities
from copy import deepcopy

from steward.ego.fasten_plan import value

DRIVE_ENABLED = capabilities.get('STEWARD_DRIVE_PRECONDITION') != '0'
FLIP_ENABLED = capabilities.get('STEWARD_FLIP_PRECONDITION') != '0'


def attempts(scene):
    return {(a['body'], a['hole']):a for e in scene.entities
            for a in value(e, 'fasteningEvidence', [])}


def drive_check(scene, params):
    rows = [a for a in attempts(scene).values() if all(
        not params.get(k) or a.get(k) == params[k] for k in ('body','hole','screw'))]
    result = dict(operation='drive', body=params.get('body'), hole=params.get('hole'),
                  status='unknown', reason='ambiguous_drive_target')
    if not rows or len(rows) == 1 and (not rows[0].get('releaseExecuted') or
                                     rows[0].get('seatCheck', {}).get('passed') is False):
        result.update(status='blocked', reason='seat_fastener_before_drive')
        if rows:
            result.update(body=rows[0]['body'], hole=rows[0]['hole'], part=rows[0].get('part'))
        return result
    if len(rows) != 1:
        return result
    a = rows[0]
    result.update(body=a['body'], hole=a['hole'], screw=a['screw'], part=a.get('part'))
    # An executed release whose head is hidden is unknown, not an empty hole.
    if not (a.get('seatConfirmed') or a.get('seatCheck', {}).get('passed') is True):
        result['reason'] = 'released_fastener_seating_unknown'
        return result
    from steward.ego.fasten_driver import head_target
    try:
        target = head_target(a, scene.stamp)
    except (KeyError, TypeError, ValueError):
        result['reason'] = 'seated_head_geometry_unavailable'
        return result
    result.update(status='satisfied' if target['observed'] else 'unknown',
                  reason='observed_seated_head' if target['observed'] else 'seated_head_not_fresh',
                  headTarget=target)
    return result


def apply(planner, scene, skill, params, receipt):
    """Only annotate/refuse prediction receipts; never move, observe or alter a ledger."""
    rows = []
    drive = skill in ('drive_screw','_simple_driver') or (
        skill == '_fastener_contact' and params.get('operation') == 'drive')
    if DRIVE_ENABLED and drive:
        row = drive_check(scene, params)
        rows.append(row)
        if row['status'] == 'blocked':
            receipt['feasible'] = False
            receipt['issues'].append('Drive requires a seated fastener in this exact hole; seat it first.')
            receipt['nextAction'] = {k:v for k,v in dict(skill='seat_screw',
                body=row.get('body'), hole=row.get('hole'), part=row.get('part', params.get('part'))).items() if v is not None}
        elif row['status'] == 'unknown':
            receipt.setdefault('warnings', []).append(dict(reason=row['reason'], advisory=True,
                body=row.get('body'), hole=row.get('hole')))
            receipt['nextAction'] = dict(skill='observe', reason='check_released_fastener_head_once',
                                         body=row.get('body'), hole=row.get('hole'))
        if row.get('headTarget'):
            receipt['driveTarget'] = deepcopy(row['headTarget'])
    if DRIVE_ENABLED and skill == 'fasten_onto' and receipt.get('plan', {}).get('drive'):
        # This compound plan seats before driving. Do not reject its seating stage
        # merely because it has not executed yet; each drive gets its own receipt.
        for hole in receipt['plan'].get('selectedHoleIds', []):
            row = drive_check(scene, dict(body=params.get('body'), hole=hole))
            if row['status'] == 'blocked':
                row.update(status='deferred', reason='seat_stage_must_precede_drive')
            rows.append(row)
        if any(r['status'] != 'satisfied' for r in rows):
            receipt.setdefault('warnings', []).append(dict(reason='drive_requires_per_hole_seating',
                advisory=True, holes=[r['hole'] for r in rows if r['status'] != 'satisfied']))
    if FLIP_ENABLED and skill == 'flip_assembly' and params.get('turnOver', True):
        from steward.ego.assembly_status import interface_access
        body = scene.find(params.get('body'))
        if body:
            ledger = attempts(scene)
            for frame in value(body, 'releasedInterfaces', []):
                if frame['body'] != body.id:
                    continue
                holes = {p['bodyHole'] for p in frame.get('holePairs', [])}
                pending = [h for h in sorted(holes) if not (
                    ledger.get((body.id,h), {}).get('seatConfirmed') or
                    ledger.get((body.id,h), {}).get('seatCheck', {}).get('passed') is True)]
                if not pending:
                    continue
                access = interface_access(planner, body, holes)
                if access['status'] == 'inaccessible':
                    continue
                row = dict(operation='flip', status='warning' if access['status'] == 'estimated_accessible' else 'unknown',
                    reason='complete_upward_interface_before_flip', body=body.id, part=frame['entity'],
                    remainingHoles=pending, face=access, advisory=True)
                rows.append(row)
                receipt.setdefault('warnings', []).append(deepcopy(row))
                receipt.setdefault('nextAction', dict(skill='fasten_onto', body=body.id,
                    part=frame['entity'], drive=False, reason='complete_remaining_seats_before_flip'))
    if rows:
        receipt['procedurePreconditions'] = rows
    return receipt
