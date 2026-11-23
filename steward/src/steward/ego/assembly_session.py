"""Goal-scoped assembly ownership above the existing interface controller.

Only Ego relations, explicit user bindings and the existing release ledger are
inputs. A session is a single executor job; children consume fresh receipts in
that job, never start competing workers. Cross-process persistence is owned by
memory_checkpoint; its restored estimates are priors, not current observations.
"""
from steward.ego import capabilities
from steward.ego.decision_observation import controller_action
from contextvars import ContextVar
from copy import deepcopy
from functools import wraps
import json
import math
import os
from pathlib import Path
import time
import uuid

from steward.ego import action_budget as ab
from steward.ego.skills import Job, SkillFailed, SkillStopped

_active = ContextVar('assembly_session', default=None)


def origin_log_view(origins):
    """Release recovery keeps live Track objects; diagnostic logs keep evidence."""
    rows=deepcopy(origins)
    for origin in rows.values():
        release=origin.get('driverRelease')
        if release and 'track' in release:
            track=release.pop('track')
            release['trackSummary']=dict(id=track.id,category=track.category,
                position=track.pos.tolist(),lastSeen=track.last_seen)
    return rows


def current():
    return _active.get()


def enabled():
    return capabilities.get('STEWARD_ASSEMBLY_SESSION') != '0'


class SessionExpired(TimeoutError):
    pass


def check_budget():
    session = current()
    if session is not None:
        if session.rt.executor._stop.is_set():
            raise SkillStopped()
        if session.state.get('identityConflict'):
            raise SkillFailed(session.state['identityConflict'])
        if session.rt._generation != session.state['generation']:
            raise SkillFailed('sensor_session_changed')
        if ab.expired(session.deadline):
            raise SessionExpired('session_budget_exhausted')
        now = time.monotonic()
        if now-session.state.get('heartbeatWall', now) >= 10.:
            session.state['heartbeatWall'] = now
            session.emit('heartbeat')


def sessions(rt):
    if not hasattr(rt.estimator, 'assembly_sessions'):
        rt.estimator.assembly_sessions = {}
    return rt.estimator.assembly_sessions


def compile_goal(rt, scene, params):
    from steward.ego.esim import PredictionError
    from steward.ego.fasten_plan import interface_holes
    planner = rt.esim
    budget = params.get('budgetSimS')
    if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not math.isfinite(budget) or budget <= 0:
        raise PredictionError('budgetSimS must be finite and positive')
    final_pose = params.get('finalPose', 'as_is')
    if final_pose not in ('initial_up', 'as_is'):
        raise PredictionError('finalPose must be initial_up or as_is')
    requested = params.get('interfaces')
    if not isinstance(requested, list) or not requested:
        raise PredictionError('interfaces must explicitly select the user goal')
    from steward.ego import session_inventory
    requested, inventoried = session_inventory.compile_requests(rt, scene, requested)
    interfaces, occupied, parts = [], set(), set()
    for index, request in enumerate(requested):
        if not isinstance(request, dict):
            raise PredictionError('each interface needs part, body and complete hole selectors')
        part, body = scene.find(request.get('part')), scene.find(request.get('body'))
        observed = next((r for (b, hs), r in inventoried.items() if b == request.get('body') and
                         set(hs) == set(request.get('hole') or [])), None)
        if observed is not None:
            part = session_inventory.member(scene, request['part'], observed)
        if part is None or body is None or part.id == body.id or part.id in parts:
            raise PredictionError('goal identity absent, duplicated or ambiguous')
        ph, bh = interface_holes(planner, part), interface_holes(planner, body)
        selected = planner._select_holes(bh, request.get('hole'), body.id)
        held = planner._select_holes(ph, request.get('heldHole'), part.id)
        pairs, _ = planner._hole_group_pairs(held, selected)
        if len(pairs) != 1 or not selected or len(pairs[0][0]) != len(pairs[0][1]):
            raise PredictionError('select one uniquely compatible complete interface')
        held, selected = pairs[0]
        for all_holes, chosen in ((ph, held), (bh, selected)):
            groups = {h.get('group') for h in chosen if h.get('group') is not None}
            if groups and {h['id'] for h in all_holes if h.get('group') in groups} != {h['id'] for h in chosen}:
                raise PredictionError('partial interface selection is not permitted')
        keys = {(body.id, h['id']) for h in selected}
        if occupied & keys:
            raise PredictionError('goal interfaces overlap')
        occupied |= keys
        parts.add(part.id)
        interfaces.append(dict(key=str(index), part=part.id, body=body.id,
            holes=[h['id'] for h in selected], heldHoles=[h['id'] for h in held],
            dependencies=[]))
        if observed is not None:
            interfaces[-1]['inventoryReport'] = deepcopy(observed)
    for item in interfaces:
        item['dependencies'] = [other['key'] for other in interfaces if other['part'] == item['body']]
    done = set()
    while len(done) < len(interfaces):
        ready = {i['key'] for i in interfaces if set(i['dependencies']) <= done} - done
        if not ready:
            raise PredictionError('cyclic support dependencies')
        done |= ready
    supplies = params.get('materials')
    if not isinstance(supplies, list):
        raise PredictionError('materials needs explicit entity/source bindings')
    materials = {}
    from steward.ego.priors import prior
    for binding in supplies:
        if not isinstance(binding, dict):
            raise PredictionError('material needs entity and source')
        entity, source = scene.find(binding.get('entity')), scene.find(binding.get('source'))
        if (entity is None or source is None or entity.id in materials or
                prior(entity.category.value).get('assemblyRole') != 'fastener' or
                not planner._affords(source, 'support-surface') or planner._anchor(scene, entity).id != source.id):
            raise PredictionError('material identity or observed source unresolved')
        origin = rt.estimator.fastener_origins.get(entity.id, {})
        if origin.get('kitSupport', source.id) != source.id:
            raise PredictionError('material source conflicts with recorded origin')
        materials[entity.id] = dict(source=source.id, boundAt=scene.stamp)
    tools = deepcopy(params.get('tools', {'mode': 'none'}))
    if not isinstance(tools, dict) or tools.get('mode') not in ('none', 'one', 'all'):
        raise PredictionError('tools.mode must be none, one or all')
    if tools['mode'] != 'none':
        entity = scene.find(tools.get('entity'))
        if entity is None or not planner._affords(entity, 'screwdriver'):
            raise PredictionError('bind an observed tool entity for the requested tool action')
    return dict(interfaces=interfaces, materials=materials, tools=tools, budgetSimS=float(budget), finalPose=final_pose)


def predict(rt, scene, params):
    from steward.ego.esim import PredictionError
    plan, issues = {}, []
    try:
        if not enabled():
            raise PredictionError('STEWARD_ASSEMBLY_SESSION is disabled')
        if params.get('session'):
            state = sessions(rt).get(params['session'])
            if state is None or state['generation'] != rt._generation:
                raise PredictionError('session unavailable in this sensor runtime')
            if set(params) != {'session'}:
                raise PredictionError('resume takes only session; goal and budget cannot be reset')
            plan = dict(resumeSession=params['session'])
        else:
            from steward.ego.memory_checkpoint import owner as memory_owner, progress
            resumed = memory_owner(rt, params)
            if resumed is not None:
                receipt = rt.esim._receipt(scene, "complete_assembly", params, 0., [],
                    dict(resumeSession=resumed), feasible=True)
                receipt["progressSummary"] = progress(rt, sessions(rt)[resumed])
                return receipt
            from steward.ego import first_interface
            owner = first_interface.resume_owner(rt, params.get('interfaces', []))
            if owner is not None:
                return rt.esim._receipt(scene, 'complete_assembly', params, 0., [],
                    dict(resumeSession=owner), feasible=True)
            from steward.ego import session_inventory
            owners = [session_inventory.guard(rt, 'fasten_onto', request)
                      for request in params.get('interfaces', []) if isinstance(request, dict)]
            owners = {owner['session'] for owner in owners if owner is not None}
            if len(owners) == 1:
                # Rephrasing an inventoried goal continues its budget/ledger;
                # it must not create a competing owner for the same stack.
                return rt.esim._receipt(scene, 'complete_assembly', params, 0., [],
                    dict(resumeSession=next(iter(owners))), feasible=True)
            plan = dict(sessionGoal=compile_goal(rt, scene, params))
            if plan['sessionGoal']['finalPose'] == 'initial_up':
                from steward.ego.assembly_final_pose import goal_estimate
                plan['finalPoseBudgetEstimate'] = goal_estimate(rt, plan['sessionGoal'])
    except (PredictionError, KeyError, TypeError, ValueError) as exc:
        issues.append(str(exc))
    return rt.esim._receipt(scene, 'complete_assembly', params, 0., issues, plan, feasible=not issues)


def ledger_locked(method):
    @wraps(method)
    def locked(self, *args, **kwargs):
        with self.rt._lock:
            return method(self, *args, **kwargs)
    return locked


class Session:
    def __init__(self, rt):
        self.rt = rt

    def canonical(self, entity):
        aliases = self.state['aliases']
        seen = set()
        while entity in aliases and entity not in seen:
            seen.add(entity)
            entity = aliases[entity]
        return entity

    def target(self, entity):
        from steward.ego.session_identity import enabled, target
        return target(self.state, entity) if enabled() else self.canonical(entity)

    @ledger_locked
    def owns(self, entity, *, tool=False):
        entity = self.canonical(entity)
        if tool:
            return entity == self.canonical(self.state['goal']['tools'].get('entity'))
        bound = {self.canonical(e) for e in self.state['goal']['materials']}
        bound.update(self.canonical(e) for e in self.state.get('supplyEntities', {}))
        if entity in bound:
            return True
        # seat_posterior can independently reassociate the very released object
        # without renaming an estimator track. Its exact ledger alias is also
        # provenance; category/position similarity by itself is not.
        return any(self.canonical(a.get('screw')) == entity and a.get('releaseExecuted')
            and bound.intersection(self.canonical(e) for e in a.get('screwAliases', []))
            for a in self.rt.estimator.fastening_attempts.values())

    def stock_source(self, entity, scene):
        """Fresh loose stock belongs to the bound bearing surface, not a face.

        This does not assert identity with a lost track. Geometry/occupancy checks
        in choose must pass before adopt_supply grants pick ownership.
        """
        members = {self.canonical(i[k]) for i in self.state['goal'].get('interfaces', [])
                   for k in ('body', 'part')}
        if self.canonical(entity.id) in members:
            return None
        source = self.canonical(self.rt.esim._anchor(scene, entity).id)
        sources = {self.canonical(v['source']) for v in self.state['goal']['materials'].values()}
        if source not in sources:
            return None
        from steward.ego.session_initial import protects_stock
        if protects_stock(self, entity):
            return None
        if self.owns(entity.id):
            return source
        from steward.ego.fasten_plan import value
        from steward.ego.priors import prior
        if (prior(entity.category.value).get('assemblyRole') == 'fastener'
                and entity.pose.src in ('observed', 'fused') and entity.pose.t is not None
                and 0 <= scene.stamp-entity.pose.t <= 20 and not value(entity, 'held', False)):
            return source
        return None

    def adopt_supply(self, entity, source):
        if not self.owns(entity.id):
            self.state.setdefault('supplyEntities', {})[entity.id] = dict(source=source)
            self.emit('supply_reacquired', entity=entity.id, source=source,
                      reason='fresh_loose_stock_on_bound_support')

    def require_pick(self, entity):
        entity = self.canonical(entity)
        members = {self.target(i[k]) for i in self.state['goal']['interfaces'] for k in ('part', 'body')}
        from steward.ego.session_identity import pick_started
        pick_started(self, entity)
        if entity not in members and not self.owns(entity) and not self.owns(entity, tool=True):
            raise SkillFailed('pick_outside_bound_goal')
        self.emit('pick_authorized', entity=entity)

    def source_support(self, fallback):
        # The body's new support does not change the explicitly bound stock.
        sources = [self.canonical(v['source']) for v in self.state['goal']['materials'].values()]
        scene = self.rt.ego_state()
        return next((s for s in sources if scene.find(s) is not None), sources[0] if sources else fallback)

    @ledger_locked
    def rows(self, item):
        bodies = {self.canonical(item['body']), self.target(item['body'])}
        attempts = self.rt.estimator.fastening_attempts
        # A receipt already in flight can record its old IDs after an estimator
        # rename. Count that irreversible event immediately, before normalization.
        result = []
        for hole in item['holes']:
            matches = [a for a in attempts.values() if self.canonical(a.get('body')) in bodies and a.get('hole') == hole]
            # Tentative body binding keeps historical rows at their logical ID;
            # a new physical release must take precedence over that history.
            result.append(max(matches, key=lambda a: a.get('seatedAt', -math.inf), default={})
                          if len(bodies) > 1 else next(iter(matches), {}))
        return result

    @ledger_locked
    def normalize_aliases(self):
        est = self.rt.estimator
        def fields(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    if key in ('entity', 'body', 'part', 'screw', 'support', 'kitSupport') and isinstance(child, str):
                        value[key] = self.canonical(child)
                    elif key != 'releaseHistory':
                        fields(child)
            elif isinstance(value, list):
                for child in value:
                    fields(child)
        fields(self.state.get('initialInterfaces', {}))
        for key, attempt in list(est.fastening_attempts.items()):
            new_key = (self.canonical(attempt['body']), attempt['hole'])
            if key != new_key:
                if new_key in est.fastening_attempts:
                    from steward.ego import first_interface
                    other = est.fastening_attempts[new_key]
                    if first_interface.same_release(attempt, other, self.canonical):
                        est.fastening_attempts[new_key] = first_interface.merge_release(attempt, other)
                        est.fastening_attempts.pop(key)
                        fields(est.fastening_attempts[new_key])
                        continue
                    self.state['identityConflict'] = 'rename_collides_with_occupied_hole'
                    continue
                est.fastening_attempts[new_key] = est.fastening_attempts.pop(key)
            fields(attempt)
        for mapping in (est.fastener_origins, getattr(est, 'released_interfaces', {}),
                        getattr(getattr(est, 'stationary_assemblies', None), 'records', {})):
            for key, value in list(mapping.items()):
                canonical = self.canonical(key)
                if canonical != key and canonical not in mapping:
                    mapping[canonical] = mapping.pop(key)
                fields(value)

    def complete(self, item):
        """First pass accepts prior occupancy or a release, without a seating gate."""
        from steward.ego.working_face import preserve_existing
        if preserve_existing(self, item):
            return True  # Current underbody member is preserved, occupancy stays unconfirmed.
        from steward.ego import first_interface
        if first_interface.applies(self, item):
            return first_interface.ready(self, item)
        rows = self.rows(item)
        from steward.ego import assembly_outcome, axial_release
        if axial_release.enabled() and rows and all(axial_release.handled(a) for a in rows):
            return True  # Action pass finished; contradictory outcomes remain in the summary.
        if axial_release.enabled() and any(axial_release.checked(a) and not axial_release.handled(a) for a in rows):
            return False
        if assembly_outcome.enabled() and (
                any(assembly_outcome.empty(a) for a in rows) or
                self.state.get('outcomeComponents', {}).get(item['key'], {}).get('matches') is False):
            return False
        if all(a.get('releaseExecuted') for a in rows):
            return True
        from steward.ego import session_faces
        initial_row = self.state.get('initialInterfaces', {}).get(item['key'], {})
        if session_faces.protected(initial_row):
            return True
        initial = initial_row.get('holes', {})
        return all(a.get('releaseExecuted') or initial.get(h, {}).get('occupied')
                   for h,a in zip(item['holes'], rows))

    @ledger_locked
    def checkpoint(self):
        est = self.rt.estimator
        bodies = {self.canonical(i['body']) for i in self.state['goal']['interfaces']}
        return dict(goal=deepcopy(self.state['goal']), aliases=deepcopy(self.state['aliases']),
            attempts=deepcopy([a for a in est.fastening_attempts.values() if self.canonical(a.get('body')) in bodies]),
            releasedInterfaces=deepcopy([f for f in getattr(est, 'released_interfaces', {}).values() if self.canonical(f.get('body')) in bodies]),
            origins=origin_log_view({e:v for e,v in est.fastener_origins.items() if self.owns(e) or self.owns(e, tool=True)}),
            contradictions=deepcopy(self.state['contradictions']), retries=deepcopy(self.state['retries']), interfaceCalls=deepcopy(self.state['interfaceCalls']), budget=self.deadline.report(),
            members=deepcopy(self.state.get('members', {})),
            initialInterfaces=deepcopy(self.state.get('initialInterfaces', {})),
            reviewedHoles=list(self.state.get('reviewedHoles', [])),
            finalPose=deepcopy(self.state.get('finalPose', {})))

    def summary(self, decision, reason=None, action=None):
        from steward.ego.hole_seating import confirmed
        rows = [a for i in self.state['goal']['interfaces'] for a in self.rows(i)]
        budget = self.deadline.report()
        result = dict(decision=decision, held=self.rt.estimator.held_id, session=self.state['id'], interface=self.state.get('interface'),
            initiallyOccupied=sum(bool(h.get('occupied')) for r in self.state.get('initialInterfaces', {}).values()
                                  for h in r['holes'].values()),
            initialUnconfirmedHoles=[dict(interface=k, hole=h) for k,r in self.state.get('initialInterfaces', {}).items()
                for h,e in r['holes'].items() if e.get('occupied') and not e.get('confirmed')],
            released=sum(bool(a.get('releaseExecuted')) for a in rows),
            confirmed=sum(bool(a.get('seatConfirmed')) for a in rows), required=len(rows),
            unconfirmedHoles=[dict(body=a.get('body'),hole=a.get('hole'),screw=a.get('screw'))
                for a in rows if a.get('releaseExecuted') and not confirmed(a)],
            releasedUnconfirmedInterfaces=deepcopy(self.state.get('releasedUnconfirmedInterfaces', [])),
            deferredToolHoles=deepcopy(self.state.get('deferredToolHoles', [])),
            nextInternalAction=action, reason=reason,
            finalPose=deepcopy(self.state.get('finalPose', dict(policy=self.state['goal'].get('finalPose', 'as_is'),
                achieved=self.state['goal'].get('finalPose', 'as_is') == 'as_is'))),
            remainingSimS=max(0., budget['budgetSimS']-budget['simS']),
            budget=ab.report(self.deadline, compact=True),
            **({'logError':self.state['logError']} if self.state.get('logError') else {}))

        from steward.ego import assembly_outcome
        if assembly_outcome.enabled():
            result.update(assembly_outcome.summary(self))
            result['released'] = sum(bool(a.get('releaseExecuted') or a.get('releaseHistory')) for a in rows)
            result['confirmed'] = sum(h['state'] == 'confirmed_occupied' for h in result['holeOutcomes'])
            result['taskComplete'] = decision == 'complete_observed' and not result['repairPending']
            if result['repairPending'] and decision not in ('running', 'recovering', 'cancelled', 'budget_exhausted'):
                result['decision'] = 'continue_assembly'
        from steward.ego import session_observation
        if session_observation.enabled():
            result['initialInterfaces'] = session_observation.summary(self)
            if 'reentryInterfaceSnapshot' in self.state:
                result['reentryInterfaces'] = deepcopy(self.state['reentryInterfaceSnapshot'])
            result['initialViewChanges'] = self.state.get('initialViewChanges', 0)
        from steward.ego import session_inventory
        message = session_inventory.narrative(self)
        if message:
            result['inventoryMessage'] = message
            result['stackInventory'] = deepcopy(self.state['stackInventory'])
        if self.state.get('workingFacePreservedInterfaces'):
            result['preservedUnderbodyInterfaces'] = deepcopy(self.state['workingFacePreservedInterfaces'])
        if self.state.get("memoryResumed"):
            from steward.ego.memory_checkpoint import progress
            result["progressSummary"] = progress(self.rt, self.state)
            memory = getattr(self.rt, "_memory_checkpoint", None)
            if memory is not None:
                result["memoryResume"] = deepcopy(memory.restore_report)
        from steward.ego import first_interface
        result = first_interface.summary(self, result)
        from steward.ego.session_identity import public as identity_public
        result.update(identity_public(self.rt.estimator, self.state))
        if self.state.get('memoryResumed') and result.get('identityNextStep'):
            result['nextInternalAction'] = result['identityNextStep']['action']
            result['guidance'] = ('轨迹当前不存在不等于零件不存在。按 identityNextStep 续接原会话的剩余接口；'
                '旧失败只作历史，当前看不见已装件不是拆卸依据。最多一次定向换视角后按最好估计执行，'
                '未抓到就消耗该候选；没有可操作对象则取账本预测区域的局部观测。未确认不等于成功，动作后检查并补救。')
        return result

    def sync_releases(self):
        for item in self.state['goal']['interfaces']:
            for hole, attempt in zip(item['holes'], self.rows(item)):
                epoch = len(attempt.get('releaseHistory', [attempt])) if attempt.get('releaseExecuted') else 0
                key = item['key'] + '/' + hole
                if epoch > self.state['releaseEpochs'].get(key, 0):
                    self.state['releaseEpochs'][key] = epoch
                    if attempt.get('restoredPrior') or (self.state.get('memoryRestoredAt') is not None and
                            attempt.get('seatedAt', -math.inf) <= self.state['memoryRestoredAt']):
                        continue
                    self.emit('release', interface=item['key'], body=attempt['body'], hole=hole,
                              screw=attempt['screw'], epoch=epoch)

    def emit(self, event, **data):
        self.state['sequence'] += 1
        row = dict(schema='assembly_session.v1', session=self.state['id'], job=self.job.id,
            sequence=self.state['sequence'], event=event, simTime=self.deadline.clock.stamp,
            wallTime=time.time(), interface=self.state.get('interface'),
            decision=self.state.get('decision', 'running'), remainingSimS=max(0., self.deadline.seconds-self.deadline.sim_s))
        row.update(data)
        if self.state['sequence'] == 1:
            row['capabilityConfiguration'] = getattr(self.rt, 'capability_configuration', None) or capabilities.snapshot()
        if event in ('release', 'action_end', 'exit'):
            row['checkpoint'] = self.checkpoint()
        directory = os.environ.get('STEWARD_RECEIPT_LOG_DIR')
        if directory:
            from steward.ego.public import public_json
            try:
                path = Path(directory)
                path.mkdir(parents=True, exist_ok=True)
                with (path / f'assembly-sessions-{os.getpid()}.jsonl').open('a', encoding='utf-8') as stream:
                    from steward.ego.geometry_relocalization import event_view
                    stream.write(json.dumps(public_json(event_view(row)), ensure_ascii=False, allow_nan=False)+'\n')
            except (OSError, ValueError, TypeError) as exc:
                self.state['logError'] = str(exc)
        self.job.postcondition = self.summary(self.state.get('decision', 'running'), action=self.state.get('action'))
        if event in ("release", "action_end", "exit", "interface_released",
                     "interface_initially_completed", "interface_preserved", "final_pose_result"):
            from steward.ego.memory_checkpoint import event as save_memory
            save_memory(self.rt, event + (":" + data["skill"] if data.get("skill") else ""))

    @controller_action
    def action(self, skill, params):
        check_budget()
        from steward.ego import session_identity
        if session_identity.enabled() and self.state.get('memoryResumed'):
            params = dict(params)
            for field in ('body', 'part'):
                if params.get(field): params[field] = self.target(params[field])
            if skill == '_assembly_observe' and params.get('changeView'):
                if self.state.get('identityViewAttempted'):
                    params.pop('changeView')
                else:
                    self.state['identityViewAttempted'] = True
        if skill == 'flip_assembly':
            from steward.ego import first_interface
            first_interface.guard_turn(self, params['body'])
        self.state['action'] = skill
        self.state['lastResult'] = None
        receipt = self.rt.predict(skill, params)
        self.emit('predict', skill=skill, params=params, receipt=deepcopy(receipt))
        if not receipt.get('withinRiskLimit'):
            raise SkillFailed('prediction_refused: '+'; '.join(receipt.get('issues', [])))
        plan = self.rt._consume_receipt(receipt)
        check_budget()
        child = Job(f"{self.job.id}/{self.state['sequence']}", skill, receipt['id'],
                    started=self.deadline.clock.stamp, postcondition={})
        self.emit('action_start', skill=skill, receipt=receipt['id'], child=child.id)
        try:
            self.rt.executor._begin(child, skill, receipt)
            check_budget()
            if skill in ('flip_assembly', 'drive_screw'):
                self.state['disturbedAt'] = self.deadline.clock.stamp
            if skill == '_assembly_observe':
                from steward.ego.assembly_precision import acquire
                self.controller._observation_job = child
                if params.get('changeView'):
                    self.change_face_view(child, params['body'])
                acquire(self.controller, [params['body'], params.get('part')])
                child.postcondition['acquisition'] = deepcopy(getattr(self.controller, '_precision_acquisition', {}))
            else:
                self.controller.run(child, plan)
            child.state = 'succeeded'
        except Exception as exc:
            child.state = 'stopped' if isinstance(exc, SkillStopped) else 'failed'
            child.message = str(exc)
            raise
        finally:
            from steward.ego.session_identity import action_finished
            action_finished(self)
            self.normalize_aliases()
            self.sync_releases()
            child.finished = self.deadline.clock.stamp
            self.state['lastResult'] = deepcopy(child.postcondition)
            if skill == 'flip_assembly' and child.postcondition.get('releaseExecuted'):
                if getattr(self.rt, '_memory_checkpoint', None) is not None:
                    self.rt._memory_checkpoint.pending_faces.pop(self.canonical(params['body']), None)
                    self.state.setdefault('flipHistory', []).append(dict(body=params['body'],
                        completedAt=child.finished, intentional=child.postcondition.get('commandedTurnOver', True),
                        observedFlipped=child.postcondition.get('flipped'),
                        confirmation=child.postcondition.get('confirmation'),
                        faceResolution=deepcopy(child.postcondition.get('faceResolution')),
                        orientationCheck=deepcopy(child.postcondition.get('orientationCheck'))))
                    from steward.ego.memory_checkpoint import recent_orientations
                    self.state['flipHistory'] = recent_orientations(self.state['flipHistory'])
                self.state.setdefault('flipDurations', []).append(dict(
                    body=params['body'], simS=max(0., child.finished-child.started)))
                if getattr(self.rt, '_memory_checkpoint', None) is not None:
                    from steward.ego.memory_checkpoint import longest_flips
                    self.state['flipDurations'] = longest_flips(self.state['flipDurations'])
                from steward.ego import session_observation
                if session_observation.enabled():
                    self.state.setdefault('initialFaceInvalidatedAt', {})[self.canonical(params['body'])] = child.finished
            self.emit('action_end', skill=skill, child=child.to_dict())
        check_budget()
        return child.postcondition

    def change_face_view(self, job, body_id):
        """One checked trip to a different side of the perceived support."""
        scene = self.rt.ego_state()
        body = scene.find(body_id)
        support = self.rt.esim._anchor(scene, body)
        from steward.ego.planner import entity_box
        box, _, _ = entity_box(support)
        odom = self.rt.executor._state()['odometry']
        origin = math.atan2(odom['y']-box.cy, odom['x']-box.cx)
        self.controller.observation_target = body.pose.value['p']
        # Existing view() scores calibrated cameras and collision-checks every
        # navigation receipt. Alternate sides across resumes; never re-flip.
        count = self.state.setdefault('faceViewAttempts', {}).get(body_id, 0)
        self.state['faceViewAttempts'][body_id] = count+1
        was_close = getattr(self.controller, '_close_view', False)
        self.controller._close_view = False
        try:
            self.controller.view(job, support, origin, math.pi/3 if count % 2 == 0 else -math.pi/3)
        finally:
            self.controller._close_view = was_close

    def access(self, item):
        from steward.ego.assembly_status import interface_access
        scene = self.rt.ego_state()
        body = scene.find(self.target(item['body']))
        if body is None:
            raise SkillFailed('bound_body_unavailable')
        return interface_access(self.rt.esim, body, set(item['holes']), scene)

    def fingerprint(self, item):
        # Time/version alone is not a changed recovery plan or changed evidence.
        scene = self.rt.ego_state()
        ids = [self.target(item['body']), self.target(item['part']), *self.state['goal']['materials']]
        evidence = [(self.canonical(e), scene.find(self.canonical(e))) for e in ids]
        tracks = getattr(self.rt.estimator, 'tracks', {})
        geometry = [(e, getattr(tracks.get(e), 'size', None), getattr(tracks.get(e), 'category', None),
                     getattr(tracks.get(e), 'geometry_check', {}).get('reason')) for e, _ in evidence]
        def content(row):
            return {k:row[k] for k in ('passed', 'status', 'reason', 'headM', 'tipM',
                'screw', 'distanceXYM', 'source') if k in row}
        return repr((geometry, [(eid, ent.pose.value if ent else None) for eid, ent in evidence],
            [(a.get('screw'), bool(a.get('releaseExecuted')), content(a.get('seatCheck') or {}),
              content(a.get('departureEvidence') or {})) for a in self.rows(item)]))

    def prepare_face(self, item, *, tools=False):
        from steward.ego import edge_retention
        records = getattr(getattr(self.rt.estimator, 'stationary_assemblies', None), 'records', {})
        retained = edge_retention.retained_record(records, self.canonical(item['body'])) if edge_retention.enabled() else None
        if tools and retained:
            return  # Assemble at the completed release; no repeat turn or look.
        if tools:
            access = self.access(item)
            if access['status'] == 'inaccessible' and (
                    self.state.get('reviewingReleases') or item['key'] not in self.state['flipped']):
                self.state['flipped'].append(item['key'])
                from steward.ego.working_face import enabled as observation_first
                params = dict(body=self.target(item['body']))
                if observation_first():
                    params['hole'] = item['holes']
                self.action('flip_assembly', params)
                self.action('_assembly_observe', dict(body=self.target(item['body']),
                    part=self.target(item['part'])))
            return
        from steward.ego.interface_entry import check
        check(self.controller, self.job, self.target(item['part']),
              self.target(item['body']), item['holes'])

    def execute_interface(self, item):
        from steward.ego import session_identity
        if not session_identity.enabled() or not self.state.get('memoryResumed'):
            return self._execute_interface(item)
        session_identity.repeated(self, item)
        before = self.fingerprint(item)
        try:
            return self._execute_interface(item)
        except SkillFailed as exc:
            check_budget()
            session_identity.failure(self, item, before, str(exc))
            self.state['identityPending'] = True
            self.state.setdefault('unconfirmedInterfaces', {})[item['key']] = str(exc)
            from steward.ego.assembly_outcome import park
            park(self)
            self.emit('identity_continuation_pending', interface=item['key'], reason=str(exc))

    def _execute_interface(self, item):
        key = item['key']
        self.state['interface'] = key
        self.state.pop('pickTargetHole', None)
        from steward.ego import axial_release
        from steward.ego import assembly_outcome
        from steward.ego import first_interface
        if axial_release.enabled():
            if any(axial_release.checked(a) and not axial_release.handled(a) for a in self.rows(item)):
                assembly_outcome.after_interface(self, item)
            if all(axial_release.handled(a) for a in self.rows(item)):
                return
        if assembly_outcome.enabled() and not first_interface.applies(self, item) and (any(assembly_outcome.empty(a) for a in self.rows(item)) or
                self.state.get('outcomeComponents', {}).get(key, {}).get('matches') is False):
            try:
                assembly_outcome.apply(self, item, assembly_outcome.observe(self, item), 'resume')
                assembly_outcome.restore_component(self, item)
                assembly_outcome.repair_empty(self, item)
            except SkillFailed as exc:
                check_budget()
                assembly_outcome.park(self)
                self.emit('outcome_repair_pending', reason=str(exc))
            if (any(assembly_outcome.empty(a) for a in self.rows(item)) or
                    self.state.get('outcomeComponents', {}).get(key, {}).get('matches') is False):
                return
        if self.complete(item):
            return
        from steward.ego import session_identity
        if not session_identity.prepare_item(self, item): return
        self.prepare_face(item)
        self.emit('interface_entered', access=self.access(item))
        while not self.complete(item):
            check_budget()
            # Every explicit resume gets one fresh action result. Only the
            # automatic retry below spends the persistent one-retry allowance.
            self.state['interfaceCalls'][key] = self.state['interfaceCalls'].get(key, 0)+1
            before = self.fingerprint(item)
            params = dict(part=self.target(item['part']), body=self.target(item['body']),
                          hole=item['holes'], heldHole=item['heldHoles'], drive=False)
            failure = None
            try:
                self.action('fasten_onto', params)
            except SkillFailed as exc:
                failure = str(exc)
            session_identity.failure(self, item, before, failure)
            check_budget()
            post = self.state.get('lastResult') or {}
            checks = [post.get(k, {}) for k in ('placementCheck', 'placementReadiness', 'supportTargetCheck')]
            if any(c.get('passed') is False for c in checks):
                self.state['contradictions'][key] = deepcopy(checks)
            if post.get('placementCheck', {}).get('passed') is True:
                self.state['contradictions'].pop(key, None)
            from steward.ego import assembly_outcome
            try:
                assembly_outcome.after_interface(self, item, post)
            except SkillFailed as exc:
                check_budget()
                assembly_outcome.park(self)
                self.emit('outcome_repair_pending', reason=str(exc))
            if self.state.get('memoryResumed') and session_identity.enabled() and failure:
                # Next entry consumes another candidate/local evidence, never
                # replays this failed command with the same evidence version.
                self.state['identityPending'] = True
                assembly_outcome.park(self)
                return
            if first_interface.applies(self, item):
                # The first face's local action pass consumes this result as-is.
                # Do not add the old retry observation when a release is known
                # to have failed or when a pending hole still needs an action.
                return
            if self.complete(item):
                return
            if assembly_outcome.enabled() and (any(assembly_outcome.empty(a) for a in self.rows(item)) or
                    self.state.get('outcomeComponents', {}).get(key, {}).get('matches') is False):
                return
            if key in self.state['contradictions']:
                raise SkillFailed('component_placement_contradicted')
            if self.rt.estimator.held_id:
                if axial_release.enabled():
                    assembly_outcome.park(self)
                else:
                    raise SkillFailed('held_load_requires_checked_recovery')
            if self.state['retries'].get(key, 0) >= 1:
                from steward.ego.edge_geometry import enabled as contact_window_enabled
                if contact_window_enabled():
                    # The available estimate has been executed after the one
                    # existing retry. Continue the other face before late repair.
                    assembly_outcome.park(self)
                    self.state.setdefault('unconfirmedInterfaces', {})[key] = str(failure or 'not_observed')
                    self.emit('interface_continued_unconfirmed', reason=str(failure or 'not_observed'),
                        nextAction='remaining_interfaces_then_review')
                    return
                raise SkillFailed('interface_recovery_exhausted: '+str(failure))
            self.state['retries'][key] = 1
            self.state['decision'] = 'recovering'
            self.action('_assembly_observe', dict(body=params['body'], part=params['part']))
            from steward.ego.edge_geometry import enabled as contact_window_enabled
            if before == self.fingerprint(item) and not axial_release.enabled() and not contact_window_enabled():
                raise SkillFailed('no_changed_evidence_or_plan: '+str(failure))
            self.emit('recovery', reason='best_estimate_after_single_view' if before == self.fingerprint(item)
                      else 'changed_evidence_or_pending_holes')
            self.state['decision'] = 'running'

    def review_releases(self):
        """One late visit per released hole, retained across explicit resumes."""
        from steward.ego import assembly_outcome
        if assembly_outcome.enabled():
            return assembly_outcome.review(self)
        from steward.ego.hole_seating import confirmed
        reviewed = self.state.setdefault('reviewedHoles', [])
        items = sorted(self.state['goal']['interfaces'],
                       key=lambda i: self.access(i)['status'] == 'inaccessible')
        self.state['reviewingReleases'] = True
        try:
            for item in items:
                pending = [a for a in self.rows(item) if a.get('releaseExecuted') and not confirmed(a)
                           and item['key']+'/'+a['hole'] not in reviewed]
                if not pending:
                    continue
                self.state['interface'] = item['key']
                # Reserve before motion: interruption cannot renew a visit.
                reviewed.extend(item['key']+'/'+a['hole'] for a in pending)
                self.emit('deferred_review', holes=[a['hole'] for a in pending])
                try:
                    self.prepare_face(item, tools=True)
                    self.action('_assembly_observe', dict(body=self.target(item['body']),
                        part=self.target(item['part'])))
                    for attempt in pending:
                        self.controller._seat_retry_budget = [1]
                        try:
                            self.action('seat_screw', dict(body=self.target(item['body']),
                                part=self.target(item['part']), hole=attempt['hole']))
                        except SkillFailed as exc:
                            check_budget()
                            if self.rt.estimator.held_id:
                                raise
                            self.emit('deferred_review_unconfirmed', hole=attempt['hole'], reason=str(exc))
                except SkillFailed as exc:
                    check_budget()
                    if self.rt.estimator.held_id:
                        raise
                    self.emit('deferred_review_unconfirmed', reason=str(exc))
        finally:
            self.state['reviewingReleases'] = False

    def execute_tools(self):
        policy = self.state['goal']['tools']
        if policy['mode'] == 'none':
            return
        items = self.state['goal']['interfaces']
        required = [(i, a) for i in items for a in self.rows(i)]
        release_time = max(a.get('seatedAt', -1) for _, a in required)
        def performed(a):
            return a.get('driveStrokeExecuted') and a.get('drivenAt', -1) >= release_time
        if policy['mode'] == 'one' and any(performed(a) for _, a in required):
            return
        pending = [(i, a) for i, a in required if not performed(a)
                   and i['key'] not in self.state['contradictions']
                   and a.get('seatCheck', {}).get('passed') is True]
        self.state['deferredToolHoles'] = [a.get('hole') for _, a in required
            if a.get('seatCheck', {}).get('passed') is not True]
        pending.sort(key=lambda pair: self.access(pair[0])['status'] == 'inaccessible')
        for item, attempt in pending:
            check_budget()
            if not all(self.complete(i) for i in items):
                raise SkillFailed('required_releases_no_longer_valid')
            self.state['interface'] = item['key']
            self.prepare_face(item, tools=True)
            if attempt.get('driveStatus') == 'driven' and not performed(attempt):
                raise SkillFailed('old_tool_action_precedes_required_releases')
            if attempt.get('seatCheck', {}).get('passed') is not True:
                continue
            self.action('drive_screw', dict(body=self.target(item['body']), hole=attempt['hole'],
                                          screw=attempt['screw']))
            if not performed(attempt):
                raise SkillFailed('tool_action_not_executed')
            if policy['mode'] == 'one':
                return

    def run(self, job, plan):
        self.job = job
        if not enabled():
            job.postcondition = dict(decision='needs_input', reason='STEWARD_ASSEMBLY_SESSION_DISABLED')
            return
        store = sessions(self.rt)
        if plan.get('resumeSession'):
            self.state = store[plan['resumeSession']]
        else:
            sid = uuid.uuid4().hex
            self.state = store[sid] = dict(id=sid, generation=self.rt._generation,
                goal=deepcopy(plan['sessionGoal']), aliases={}, sequence=0, retries={}, interfaceCalls={}, releaseEpochs={}, contradictions={}, completed=[], flipped=[],
                decision='running', controller=None)
        from steward.ego.fasten import Fasten
        self.controller = self.state['controller'] or Fasten(self.rt)
        self.state['controller'] = self.controller
        for key, value in self.state.pop('savedControllerMemory', {}).items():
            setattr(self.controller, key, value)
        clock = ab.clock(self.rt.executor)
        clock.observe(float(self.rt.executor._state()['stamp']))
        # Explicit simulation clock even when legacy ASSEMBLY_ACTION_BUDGET is disabled.
        if 'deadline' not in self.state:
            self.state['deadline'] = ab.Deadline(clock, self.state['goal']['budgetSimS'], 'user_session_budget')
        self.deadline = self.state['deadline']
        from steward.ego import assembly_outcome
        if assembly_outcome.enabled() and not self.state.get('memoryResumed'):
            self.state['outcomeRefills'] = {}
            self.state['outcomeComponentRetries'] = {}
        token = _active.set(self)
        decision, reason = 'needs_input', None
        try:
            self.state['decision'] = 'running'
            self.state['heartbeatWall'] = time.monotonic()
            from steward.ego import supply_reliability as sr
            self.emit('start', resumed=bool(plan.get('resumeSession')), goal=self.state['goal'],
                config=dict(assembly_session=enabled(), holding=sr.holding_enabled(), size=sr.size_enabled(),
                    role=sr.role_enabled(), crossSupport=sr.cross_support_enabled(),
                    sessionSupply='bound_support_loose_stock', legacyCompletionGate=False,
                    legacyEnvironmentReadiness=False))
            self.normalize_aliases()
            from steward.ego import session_identity
            if session_identity.enabled():
                session_identity.ledger(self.rt.estimator, self.state)
                session_identity.prepare(self)
            check_budget()
            from steward.ego import assembly_final_pose
            assembly_final_pose.initialize(self)
            from steward.ego import first_interface
            first_interface.initialize(self)
            if not plan.get('resumeSession'):
                from steward.ego.session_initial import initialize
                initialize(self)
            else:
                from steward.ego.session_initial import refresh
                refresh(self)
            from steward.ego.hole_seating import confirmed
            done = set(self.state.get('completed', [])) if self.state.get('memoryResumed') else set()
            # A new contradictory observation revokes scheduling completion,
            # while immutable release/attempt history remains available for repair.
            done -= set(self.state.get('contradictions', {}))
            todo = self.state['goal']['interfaces']
            while len(done) < len(todo):
                ready = [i for i in todo if i['key'] not in done and set(i['dependencies']) <= done]
                from steward.ego import session_faces
                if session_faces.enabled():
                    ready.sort(key=lambda i: session_faces.priority(self, i))
                elif first_interface.enabled():
                    from steward.ego.working_face import enabled as observation_first
                    ready.sort(key=lambda i: (observation_first() and self.access(i)['status'] != 'estimated_accessible',
                                             not first_interface.applies(self, i),
                                             first_interface.access_status(self, i) == 'inaccessible'))
                else:
                    from steward.ego.working_face import enabled as observation_first
                    ready.sort(key=lambda i: self.access(i)['status'] != 'estimated_accessible'
                               if observation_first() else self.access(i)['status'] == 'inaccessible')
                item = ready[0]
                from steward.ego import axial_release
                if axial_release.enabled():
                    try:
                        self.execute_interface(item)
                    except (SkillFailed, StopIteration, KeyError, IndexError, AttributeError, TypeError, ValueError) as exc:
                        if isinstance(exc, IndexError) and not first_interface.enabled():
                            raise
                        check_budget()
                        assembly_outcome.park(self)
                        self.emit('interface_continued_unconfirmed',
                            reason=str(exc) or type(exc).__name__, interfaceStatus='executed_unverified')
                else:
                    self.execute_interface(item)
                first_interface.repair(self, item)
                done.add(item['key'])
                if self.complete(item) and item['key'] not in self.state['completed']:
                    self.state['completed'].append(item['key'])
                from steward.ego.hole_seating import confirmed
                uncertain = [h for h,a in zip(item['holes'], self.rows(item)) if not confirmed(a)]
                status = 'released_unconfirmed' if uncertain else 'seated'
                preserved = session_faces.protected(self.state.get('initialInterfaces', {}).get(item['key']))
                if preserved:
                    status = 'existing_structure_unconfirmed'
                if assembly_outcome.enabled() and not self.complete(item):
                    status = 'repair_pending'
                if uncertain:
                    recorded = self.state.setdefault('releasedUnconfirmedInterfaces', [])
                    if item['key'] not in recorded:
                        recorded.append(item['key'])
                        self.emit('interface_continued_unconfirmed', interfaceStatus=status,
                                  unconfirmedHoles=uncertain)
                self.emit('interface_preserved' if preserved else
                          'interface_repair_pending' if status == 'repair_pending' else
                          'interface_released' if any(a.get('releaseExecuted') for a in self.rows(item))
                          else 'interface_initially_completed', interfaceStatus=status)
            from steward.ego.session_initial import review
            review(self)
            self.review_releases()
            self.execute_tools()
            if not all(self.complete(i) for i in todo):
                raise SkillFailed('required_interface_contradicted_after_actions')
            pose_achieved = assembly_final_pose.restore(self)
            if (assembly_outcome.enabled() and self.state.get('disturbedAt', -math.inf) >
                    self.state.get('outcomeReviewedAt', -math.inf)):
                self.review_releases()
            if not all(self.complete(i) for i in todo):
                raise SkillFailed('required_interface_contradicted_after_actions')
            rows = [a for i in todo for a in self.rows(i)]
            tool_rows = [a for a in rows if a.get('driveStrokeExecuted')]
            observed_tools = self.state['goal']['tools']['mode'] == 'none' or (
                bool(tool_rows) and all(a.get('driveStatus') == 'driven' for a in tool_rows))
            observed_seats = not self.state['contradictions'] and all(confirmed(a)
                and a.get('seatCheck', {}).get('observedAt', -1) >
                max(a.get('seatedAt', -1), a.get('transportedAt', -math.inf),
                    self.state.get('disturbedAt', -math.inf)) for a in rows)
            if self.state.get('memoryResumed'):
                # Historical confirmations remain in the ledger but cannot
                # certify a current-process observed completion. No extra action
                # or view is demanded solely to upgrade this reporting label.
                resumed_at = self.state['memoryRestoredAt']
                observed_seats = observed_seats and all(
                    not a.get('seatCheck', {}).get('restoredPrior') and
                    a.get('seatCheck', {}).get('observedAt', -math.inf) >= resumed_at for a in rows)
                observed_tools = observed_tools and (self.state['goal']['tools']['mode'] == 'none' or
                    all(a.get('drivenAt', -math.inf) > resumed_at for a in tool_rows))
            decision = ('complete_observed' if observed_seats and observed_tools and pose_achieved
                        else 'executed_unverified')
            if not pose_achieved:
                reason = self.state['finalPose'].get('reason', 'final_pose_unverified')
        except SessionExpired as exc:
            decision, reason = 'budget_exhausted', str(exc)
        except SkillStopped:
            decision, reason = 'cancelled', 'cancelled_on_request'
        except (SkillFailed, PermissionError) as exc:
            decision, reason = ('budget_exhausted' if ab.expired(self.deadline) else 'needs_input'), str(exc)
        except Exception as exc:
            decision, reason = 'needs_input', f'internal_error: {type(exc).__name__}: {exc}'
        finally:
            if (assembly_outcome.enabled() and decision not in ('budget_exhausted', 'cancelled')
                    and self.rt.estimator.held_id):
                try:
                    assembly_outcome.park(self)
                except SessionExpired as exc:
                    decision, reason = 'budget_exhausted', str(exc)
                except SkillStopped:
                    decision, reason = 'cancelled', 'cancelled_on_request'
                except (SkillFailed, AttributeError, KeyError, TypeError, ValueError) as exc:
                    reason = (reason or '') + '; return_pending: ' + str(exc)
            # Existing stop is a zero-displacement hold, not an unbudgeted
            # retract/park trajectory. Preserve a held load for checked recovery.
            from steward.ego import axial_release
            if axial_release.enabled() and decision == 'needs_input':
                decision = 'continue_assembly'
            if decision in ('needs_input', 'budget_exhausted', 'cancelled') and hasattr(self.rt.executor, 'stop'):
                try:
                    self.rt.executor.stop()
                except Exception as exc:
                    reason = (reason or '') + '; stop_failed: ' + str(exc)
            self.sync_releases()
            if assembly_outcome.enabled() and decision not in ('cancelled', 'budget_exhausted'):
                handled = axial_release.enabled() and all(axial_release.checked(a)
                    for item in self.state['goal']['interfaces'] for a in self.rows(item))
                if not handled and (assembly_outcome.summary(self)['repairPending'] or (decision == 'needs_input' and any(
                        word in (reason or '') for word in ('unconfirmed', 'observation', 'recovery_exhausted', 'no_changed_evidence')))):
                    decision = 'continue_assembly'
            from steward.ego import first_interface
            if first_interface.enabled():
                decision = first_interface.summary(self, dict(decision=decision))['decision']
            from steward.ego import session_identity
            if session_identity.enabled() and self.state.get('memoryResumed') and decision not in ('cancelled', 'budget_exhausted', 'complete_observed'):
                if self.state.get('identityPending') or decision in ('needs_input', 'continue_assembly'):
                    decision = 'identity_continuation'
            self.state['decision'] = decision
            self.state['reason'] = reason or ''
            self.state['action'] = None
            self.emit('exit', decision=decision, reason=reason,
                **({'holeOutcomes':assembly_outcome.summary(self)['holeOutcomes']}
                   if assembly_outcome.enabled() else {}))
            job.postcondition = self.summary(decision, reason)
            job.message = decision
            _active.reset(token)


def track_renamed(est, old, new):
    """Follow an estimator's explicit identity rename; never infer by category.

    Only live session members opt into this migration. An ambiguous merge into
    another bound identity blocks the session instead of laundering ownership.
    """
    states = getattr(est, 'assembly_sessions', {})
    affected = []
    from steward.ego import session_identity
    for state in states.values():
        if session_identity.enabled(): session_identity.ledger(est, state)
        ids = {e for i in state['goal']['interfaces'] for e in (i['body'], i['part'])}
        ids |= set(state['goal']['materials']) | set(state['aliases'].values())
        ids |= set(state.get('supplyEntities', {}))
        ids |= {v['source'] for v in state['goal']['materials'].values()}
        ids.add(state['goal']['tools'].get('entity'))
        if old in ids:
            if new in ids and new != old:
                state['identityConflict'] = 'rename_collides_with_bound_identity'
                continue
            state['aliases'][old] = new
            affected.append(state)
    if not affected:
        return
    def rename_fields(value):
        if isinstance(value, dict):
            for k, v in value.items():
                if k in ('entity', 'body', 'part', 'screw', 'support', 'kitSupport', 'source', 'target', 'object') and isinstance(v, str) and v == old:
                    value[k] = new
                elif isinstance(v, (dict, list)):
                    rename_fields(v)
        elif isinstance(value, list):
            for v in value:
                rename_fields(v)
    for key, attempt in list(est.fastening_attempts.items()):
        if old not in (attempt.get('body'), attempt.get('part'), attempt.get('screw')):
            continue
        rename_fields(attempt)
        new_key = (attempt['body'], attempt['hole'])
        if new_key != key:
            if new_key in est.fastening_attempts:
                from steward.ego import first_interface
                other = est.fastening_attempts[new_key]
                if first_interface.same_release(attempt, other):
                    est.fastening_attempts[new_key] = first_interface.merge_release(attempt, other)
                    est.fastening_attempts.pop(key)
                else:
                    for state in affected:state['identityConflict'] = 'rename_collides_with_occupied_hole'
            else:
                est.fastening_attempts[new_key] = est.fastening_attempts.pop(key)
    if getattr(est, 'held_id', None) == old:
        est.held_id = new
    for mapping in (est.fastener_origins, getattr(est, 'released_interfaces', {}),
                    getattr(getattr(est, 'stationary_assemblies', None), 'records', {})):
        if old in mapping and new not in mapping:
            mapping[new] = mapping.pop(old)
        for value in mapping.values():
            rename_fields(value)

    from steward.ego.session_identity import migrated, enabled as identity_enabled
    if identity_enabled(): migrated(est, old, new)
