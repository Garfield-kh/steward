"""Missing diagnostic state cannot stand in for a valid component registration."""
from steward.ego import capabilities
from copy import deepcopy


def enabled():
    return capabilities.get('STEWARD_ASSEMBLY_COMPONENT_REPORT') != '0'


def preserve(root,job):
    if (enabled() and root is not job.postcondition and root.get('componentHoleAlignment')
            and not job.postcondition.get('componentHoleAlignment')):
        job.postcondition['componentHoleAlignment']=deepcopy(root['componentHoleAlignment'])


def missing(controller,job,stage):
    if not enabled() or isinstance(job.postcondition.get('componentHoleAlignment'),dict):
        return False
    from steward.ego.component_registration import fallback
    row=dict(attempts=0,successes=0,registrationSuccesses=0,applied=False,
        status='unknown',stage=stage,reportRecovered=True,frames=[])
    fallback(row,'component_alignment_report_missing')
    job.postcondition['componentHoleAlignment']=row
    controller._component_spec=None
    with controller.rt._lock:
        record=controller.rt.estimator.stationary_assemblies.records.get(getattr(controller,'body_id',None))
        if record:
            ref = record['reference']
            if ref.get('geometryRelocalization') and ref.get('componentHoleRegistration'):
                ref['componentHoleRegistration']['historicalPrior'] = True
            else:
                ref.pop('componentHoleRegistration',None)
            controller.rt.estimator.version+=1
    return True


def ordinary_plan(controller,job,params):
    from steward.ego.skills import SkillFailed
    from steward.ego import action_budget as ab
    receipt=controller.rt.predict('place',params)
    if not receipt.get('withinRiskLimit'):
        raise SkillFailed('unregistered placement not admitted after missing alignment report: '+
                          '; '.join(receipt.get('issues',[])))
    plan=controller.rt._consume_receipt(receipt)
    deadline=getattr(controller.ex,'_primitive_deadline',None)
    controller.ex._begin(job,'place',receipt)
    if deadline is not None:
        controller.ex._primitive_deadline=ab.limited(deadline,controller.ex._primitive_deadline)
    return plan
