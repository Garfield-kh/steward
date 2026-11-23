"""Finish the selected interface's release pass before acquiring the driver."""
from steward.ego import capabilities


def enabled():
    return capabilities.get('STEWARD_ASSEMBLY_INTERFACE_SEQUENCE') != '0'


def ready(controller, job, plan, holes, failures):
    if not enabled():
        return True
    from steward.ego.fasten_sequence import eligibility
    checks=[]
    for hole in holes:
        a=controller.rt.estimator.fastening_attempts.get((plan['body'],hole['id']))
        reason=eligibility(a)
        if a and a.get('part') != plan.get('part'):
            reason='different_interface'
        checks.append(dict(hole=hole['id'], released=bool(a and a.get('releaseExecuted')),
            reason=reason, confirmation=(a or {}).get('seatCheck',{}).get('status','unknown')))
    missing=[r['hole'] for r in checks if r['reason']]
    row=job.postcondition['interfaceDriveOrder']=dict(status='deferred' if missing else 'ready',
        selectedHoles=[r['hole'] for r in checks], pendingHoles=missing, policy='release_all_then_drive',
        unknownConfirmationBlocks=False)
    if missing:
        row['issues']=[dict(hole=r['hole'],reason=r['reason']) for r in checks if r['reason']]
        row['nextAction']=dict(skill='fasten_onto',part=plan.get('part'),body=plan['body'],drive=True,
            reason='continue_unreleased_or_observably_failed_holes')
        failures.append(dict(hole=None,skill='drive_screw',failure='interface releases incomplete; drive deferred'))
    return not missing
