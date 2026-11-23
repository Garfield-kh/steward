"""One checked tool acquisition/parking per simple-drive batch; measured phases."""
from steward.ego import capabilities
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
import time
import uuid


def enabled():
    return capabilities.get('STEWARD_ASSEMBLY_DRIVER_BATCH') != '0'


@contextmanager
def phase(controller, job, name):
    if not enabled():
        yield
        return
    from steward.ego.action_budget import clock
    wall=time.monotonic();stamp=clock(controller.ex).stamp
    report=job.postcondition
    try:
        yield
    finally:
        row=report.setdefault('driverPhases',{}).setdefault(name,dict(count=0,simS=0.,wallS=0.))
        row.update(count=row['count']+1,
            simS=round(row['simS']+max(0.,clock(controller.ex).stamp-stamp),3),
            wallS=round(row['wallS']+time.monotonic()-wall,3))


def timed(name):
    def decorate(fn):
        @wraps(fn)
        def run(controller,job,*args,**kwargs):
            with phase(controller,job,name):return fn(controller,job,*args,**kwargs)
        return run
    return decorate


class Stages:
    """Disjoint stroke stages, including the failed stage and withdrawal."""
    def __init__(self,controller,job):
        self.controller,self.job=controller,job
        self.scope=None

    def mark(self,name=None):
        if self.scope is not None:
            self.scope.__exit__(None,None,None)
            self.scope=None
        if name is not None:
            self.scope=phase(self.controller,self.job,name)
            self.scope.__enter__()


def publish(root,child,hole):
    if enabled() and root is not child and child.get('driverPhases'):
        root.setdefault('driverPhaseTiming',[]).append(dict(hole=hole,
            columns=['simS','wallS','count'], phases={name:[row['simS'],row['wallS'],row['count']]
                for name,row in child['driverPhases'].items()}))


def run(controller,job,plan,candidates,failures):
    from steward.ego.fasten_sequence import bounded,require_attempt
    from steward.ego.skills import SkillFailed
    wall_start=time.monotonic()
    failures_before = len(failures)
    state=dict(tool=None,clear=True,acquisitions=0,retainedHoles=0)
    previous=getattr(controller,'_driver_batch',None)
    controller._driver_batch=state
    summary=job.postcondition['driverBatch']=dict(id=uuid.uuid4().hex, actionAttempts=0, holes=len(candidates),acquisitions=0,
        parkingAttempts=0,retainedHoles=0,completed=False)
    def operation():
        try:
            for a in candidates:
                require_attempt(controller,job,dict(plan,screw=a['screw'],hole=a['hole']),a)
                try:
                    summary['actionAttempts'] += 1
                    controller.step(job,'drive_screw',dict(screw=a['screw'],body=plan['body'],
                        hole=a['hole'],method='simple',reuseHeldTool=bool(state['tool']),correctOnce=bool(plan.get('correctOnce',True))))
                except SkillFailed as exc:
                    failures.append(dict(hole=a['hole'],skill='drive_screw',failure=str(exc)))
                    break
                controller.check()
                if not state['clear']:break
        finally:
            # Contact/withdrawal failure never authorizes sideways parking.
            controller.check()
            if controller.rt.estimator.held_id and state['clear']:
                summary['parkingAttempts']+=1
                if not controller.park_driver(job,plan['support']):
                    raise SkillFailed('driver batch parking incomplete; tool remains held')
                for a in candidates:
                    if a.get('driverRetained'):
                        a.update(driverRetained=False,driverParked=True)
    try:
        bounded(controller,job,operation,plan['support'],hole_count=len(candidates))
        summary['completed']=not failures and not controller.rt.estimator.held_id
    except SkillFailed as exc:
        failures.append(dict(hole=None,skill='drive_screw',failure=str(exc)))
    finally:
        summary['stopReasons'] = [str(f['failure']) for f in failures[failures_before:] if f.get('failure')]
        summary.update(acquisitions=state['acquisitions'],retainedHoles=state['retainedHoles'],
            held=controller.rt.estimator.held_id)
        job.postcondition.get('driveGate',{})['wallS']=round(time.monotonic()-wall_start,3)
        controller._driver_batch=previous


def compact(value):
    """Public drive diagnostics: six decimals, never fed back into motion checks."""
    if isinstance(value,dict):return {k:compact(v) for k,v in value.items()}
    if isinstance(value,list):return [compact(v) for v in value]
    if isinstance(value,float):return round(value,6)
    return value
