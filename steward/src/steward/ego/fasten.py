"""Cancellable fastening over robot perception, encoders and ordinary bounded-risk receipts.

No world connection graph, fastener events, judge, assets or scenario access.
Pre-action perceptual confidence is advisory; post-contact claims need observations.
"""

from steward.assembly_profile import enabled as _assembly_enabled

if _assembly_enabled():
    from steward.ego.phase_timing import timed
    from steward.ego.decision_observation import controller_action
    from steward.ego import driver_batch
    from steward.ego import supply_reliability as sr
from copy import deepcopy
import math
import time
if _assembly_enabled():
    from steward.ego import action_budget as ab
import numpy as np

from steward.ego.assemble import AssembleOnto
from steward.ego.fasten_geometry import directed_tool, follows, head_sank, screw_head, settled_flip_rotation
from steward.ego.fasten_plan import value, tool_estimate, interface_holes, drive_context, support_anchor, mating_holes
from steward.ego.planner import entity_box
if _assembly_enabled():
    from steward.ego.public import public_json
from steward.ego.skills import SkillFailed
from steward.math3d import matrix_from_quat, quat_from_yaw
from steward.scene.schema import Est


if _assembly_enabled():
    from steward.json_types import plain_json


class Fasten(AssembleOnto):
    if _assembly_enabled():
        @controller_action
        def step(self, job, skill, params, *, receipt=None):
            self._observation_job = job
            self.check()
            started = time.monotonic()
            job.log.append('predict '+skill)
            from steward.ego.assembly_session import current
            if current() is not None and skill == 'flip_assembly':
                from steward.ego import first_interface
                first_interface.guard_turn(current(), params['body'])
            if current() is not None and skill == 'seat_screw':
                current().state['pickTargetHole'] = dict(body=params.get('body'), hole=params.get('hole'))
            from steward.ego import edge_geometry
            if edge_geometry.enabled() and skill in ('assemble_onto', 'seat_screw') and params.get('body'):
                chosen = params.get('hole')
                if edge_geometry.guard_force(self, job, params['body'], {chosen} if isinstance(chosen, str) else chosen):
                    receipt = None  # Body motion invalidates the caller's old plan.
            receipt = receipt or self.rt.predict(skill, params)
            predict_s = time.monotonic()-started
            root = job.postcondition
            if receipt.get('preconditions'):
                from steward.ego.preconditions import record_job
                record_job(job, receipt['preconditions'], receipt['id']+':admission')
                root = job.postcondition
            for warning in receipt.get('warnings', []):
                self.warn(**warning)
            row = {'skill':skill, 'wallStart':time.time()-predict_s, 'predictWallS':predict_s, 'receipt':deepcopy(receipt), 'result':None}
            root.setdefault('stages', []).append(row)
            if not receipt.get('withinRiskLimit'):
                row['failure'] = '; '.join(receipt.get('issues', []))
                self.progress(job, skill, False, row['failure'])
                raise SkillFailed(f'{skill}: {row["failure"]}')
            try:
                plan = self.rt._consume_receipt(receipt)
                self.ex._begin(job, skill, receipt)
                self.check()
                if skill == 'pick':
                    from steward.ego.assembly_session import current
                    if current() is not None:
                        current().require_pick(plan['target'])
                    scene = self.rt.ego_state(); target = scene.find(plan['target'])
                    from steward.ego.priors import prior
                    if (sr.role_enabled() or sr.size_enabled()) and (params.get('expectedFastener') or target is not None and
                            prior(target.category.value).get('assemblyRole')=='fastener'):
                        sr.preflight(self,job,plan['target'])
                        plan['protectGraspFrame']=True
                    origins = self.rt.estimator.fastener_origins
                    position = target.pose.value['p'] if target else plan['targetP']
                    origin = dict(support=plan['support'], xy=list(position[:2]))
                    origin['kitSupport'] = origins.get(plan['target'], {}).get('kitSupport', plan['support'])
                    origins[plan['target']] = {**origins.get(plan['target'], {}), **origin}
                    self.rt.estimator.fastener_origins = origins
                    from steward.ego import supply_pregrasp
                    if supply_pregrasp.enabled() and plan['target'] == getattr(self, '_supply_pregrasp_pending_target', None):
                        supply_pregrasp.pick(self, job, plan, params)
                        origin = origins.get(plan['target'], origin)
                    else:
                        self.ex._pick(job, plan)
                    if current() is not None and self.rt.estimator.held_id:
                        current().emit('material_picked', entity=self.rt.estimator.held_id,
                                       requested=plan['target'], material=current().owns(self.rt.estimator.held_id),
                                       **current().state.get('pickTargetHole', {}))
                    if self.rt.estimator.held_id:
                        track = self.rt.estimator.tracks.get(self.rt.estimator.held_id)
                        evidence = (getattr(track, 'grasp_association', None) or {}).get('endpointEvidence')
                        if evidence:
                            row['endpointGraspEvidence'] = deepcopy(evidence)
                        previous_origin = origins.get(self.rt.estimator.held_id, {})
                        origins[self.rt.estimator.held_id] = {**previous_origin, **origin}
                        getattr(self, '_tool_sources', {}).pop(plan['target'], None)
                elif skill == 'place':
                    from steward.ego.place_surface import refresh as refresh_surface
                    plan = refresh_surface(self, job, plan, params)
                    release_after = self.ex._state()['stamp']
                    if plan.get('seatTarget'):
                        from steward.ego.fasten_accounting import allow_release
                        plan.update(seatBody=params['seatBody'], seatHole=params['seatHole'], seatPart=params['on'])
                        allow_release(self.rt.estimator, plan['seatBody'], plan['seatHole'], plan['object'])
                    try:
                        from steward.ego.assembly_precision import placement, post_screw
                        from steward.ego.working_face import placement as stationary_placement
                        with stationary_placement(self, job, plan, params), placement(self, job, plan, params):
                            if (getattr(self, '_seat_only', False) and getattr(self, '_lower_interface', False)
                                    and 'floor' in plan):
                                self.ex._place_floor(job, plan)
                            else:
                                self.ex._place(job, plan)
                        post_screw(self, plan, params)
                    except SkillFailed as exc:
                        self.check()
                        if not (getattr(self, '_seat_only', False) and plan.get('seatTarget')
                                and (plan.get('releaseExecuted') or 'releasedAt' in plan)):
                            raise
                        self.warn('seat_released_exit_failed', detail=str(exc), advisory=True)
                    finally:
                        if plan.get('endpointReleaseEvidence'):
                            row['endpointReleaseEvidence'] = deepcopy(plan['endpointReleaseEvidence'])
                        if plan.get('precision'):
                            row['precision'] = deepcopy(plan['precision'])
                        if plan.get('seatTarget') and (plan.get('releaseExecuted') or 'releasedAt' in plan):
                            from steward.ego.fasten_accounting import record_release
                            with self.rt._lock:
                                record_release(self.rt.estimator, plan, plan.get('releasedAt', release_after))
                                self.rt.estimator.fastening_attempts[(plan['seatBody'], plan['seatHole'])]['precision'] = plan.get('precision')
                    with self.rt._lock:
                        released = getattr(self.rt.estimator, 'fastener_releases', {}).get(plan['object'])
                        if released and released['stamp'] >= release_after:
                            plan['releaseToolFeature'] = deepcopy(released)
                            origin = self.rt.estimator.fastener_origins.get(plan['object'], {})
                            if 'driverRelease' in origin:
                                origin.update(support=plan.get('support', plan.get('on')), xy=list(plan['spot']))
                        t = self.rt.estimator.tracks.get(plan['object'])
                        if t is not None and hasattr(self, '_tool_sources'):
                            self._tool_sources[t.id] = deepcopy(t)
                elif skill == 'navigate':
                    self.ex._drive(job, plan['path'], plan['yaw'])
                elif skill == '_fastener_reset':
                    self.ex._set_joint('wrist_yaw', plan['wristYaw'], tol=.02)
                    self.ex._set_joint('wrist_roll', plan['wristRoll'], tol=.02)
                    if 'wrist_pitch' in self.ex._state()['joints']:
                        self.ex._set_joint('wrist_pitch', plan['wristPitch'], tol=.02)
                elif skill == '_pivot_align':
                    from steward.ego.fasten_align import execute_push
                    execute_push(self, job, plan)
                elif skill == '_simple_driver':
                    self.simple_stroke(job, plan)
                elif skill == '_fastener_contact':
                    self.contact(job, plan)
                elif skill in ('_stage_screw', '_upright_tool'):
                    self.stage_release(job, plan)
                elif skill == 'assemble_onto':
                    AssembleOnto(self.rt).run(job, plan)
                else:
                    child = Fasten(self.rt)
                    child._tool_sources = getattr(self, '_tool_sources', {})
                    child._driver_batch = getattr(self, '_driver_batch', None)
                    self._driver_misses = getattr(self, '_driver_misses', {})
                    child._driver_misses = self._driver_misses
                    child._seat_retry_budget = getattr(self, '_seat_retry_budget', [2])
                    child._seat_retry_counts = getattr(self, '_seat_retry_counts', {})
                    child._seat_only = getattr(self, '_seat_only', False)
                    child._screw_misses = getattr(self, '_screw_misses', {})
                    try:
                        child.run(job, plan)
                    finally:
                        from steward.ego.wrist_target import enabled as wrist_target_enabled
                        if (wrist_target_enabled()
                                and skill == 'seat_screw' and hasattr(child, '_registered_alignment')):
                            self._registered_alignment = child._registered_alignment
                row['result'] = deepcopy(job.postcondition) if job.postcondition is not root else {'executed':True}
                self.progress(job, skill, True, 'executed', result=root)
                return plan
            except Exception as exc:
                row['failure'] = str(exc)
                self.progress(job, skill, False, str(exc), result=root)
                if job.postcondition is not root:
                    row['result'] = deepcopy(job.postcondition)
                raise
            finally:
                row['wallS'] = time.monotonic()-started
                if skill == 'assemble_onto' and job.postcondition.get('supportTargetCheck'):
                    root['supportTargetCheck'] = deepcopy(job.postcondition['supportTargetCheck'])
                if skill == 'assemble_onto' and job.postcondition.get('postflipDestination'):
                    root['postflipDestination'] = deepcopy(job.postcondition['postflipDestination'])
                if skill == 'assemble_onto' and job.postcondition.get('placementReadiness'):
                    root['placementReadiness'] = deepcopy(job.postcondition['placementReadiness'])
                if skill == 'assemble_onto' and job.postcondition.get('inhandAlignment'):
                    root['inhandAlignment'] = deepcopy(job.postcondition['inhandAlignment'])
                if skill == 'assemble_onto' and job.postcondition is not root and job.postcondition.get('componentHoleAlignment'):
                    from steward.ego.component_registration import publish_parent
                    publish_parent(root, job.postcondition['componentHoleAlignment'])
                if skill == 'drive_screw' and job.postcondition is not root:
                    driver_batch.publish(root,job.postcondition,params.get('hole'))
                    if job.postcondition.get('driverMotion'):
                        root['driveStage'] = job.postcondition['driverMotion'].get('stage')
                    contact = job.postcondition.get('driverMotion', {}).get('lowerContact')
                    if contact:
                        from steward.ego.driver_contact import summary
                        root.setdefault('driverContacts', []).append(dict(hole=params.get('hole'), **summary(contact)))
                    timing = job.postcondition.get('driverMotion', {}).get('descentTiming')
                    if timing:
                        root.setdefault('driverDescentTiming', []).append(dict(hole=params.get('hole'), phases=deepcopy(timing)))
                    for key in ('driverClear', 'driveBudget', 'driveGate'):
                        if key in job.postcondition:
                            root[key] = deepcopy(job.postcondition[key])
                if skill == 'seat_screw' and job.postcondition is not root and job.postcondition.get('releaseObservation'):
                    root.setdefault('releaseObservations', []).append(deepcopy(job.postcondition['releaseObservation']))
                if job.postcondition is not root and job.postcondition.get('fastenerReturns'):
                    root.setdefault('fastenerReturns', []).extend(deepcopy(job.postcondition['fastenerReturns']))
                    root['fastenerReturnCount'] = len(root['fastenerReturns'])
                from steward.ego.wrist_fasten import publish_release
                publish_release(root, job.postcondition)
                if job.postcondition is not root and job.postcondition.get('identityChecks'):
                    root.setdefault('identityChecks', []).extend(job.postcondition['identityChecks'])
                if job.postcondition is not root:
                    for key in ('pregraspChecks','supplySelections','supplyDiagnostics','driverPoseChecks','seatEndpointAttempts',
                                'supplyPregrasp'):
                        if job.postcondition.get(key):root.setdefault(key,[]).extend(deepcopy(job.postcondition[key]))
                    if job.postcondition.get('supplyUnconfirmed'):
                        root['supplyUnconfirmed'] = True
                job.postcondition = root
                from steward.ego.assembly_precision import publish
                publish(job)
    else:
        def step(self, job, skill, params, *, receipt=None):
            self._observation_job = job
            self.check()
            started = time.monotonic()
            job.log.append('predict '+skill)
            receipt = receipt or self.rt.predict(skill, params)
            predict_s = time.monotonic()-started
            root = job.postcondition
            for warning in receipt.get('warnings', []):
                self.warn(**warning)
            row = {'skill':skill, 'wallStart':time.time()-predict_s, 'predictWallS':predict_s, 'receipt':deepcopy(receipt), 'result':None}
            root.setdefault('stages', []).append(row)
            if not receipt.get('withinRiskLimit'):
                row['failure'] = '; '.join(receipt.get('issues', []))
                self.progress(job, skill, False, row['failure'])
                raise SkillFailed(f'{skill}: {row["failure"]}')
            try:
                plan = self.rt._consume_receipt(receipt)
                self.ex._begin(job, skill, receipt)
                self.check()
                if skill == 'pick':
                    scene = self.rt.ego_state(); target = scene.find(plan['target'])
                    origins = self.rt.estimator.fastener_origins
                    position = target.pose.value['p'] if target else plan['targetP']
                    origin = dict(support=plan['support'], xy=list(position[:2]))
                    origins[plan['target']] = origin
                    self.rt.estimator.fastener_origins = origins
                    self.ex._pick(job, plan)
                    if self.rt.estimator.held_id:
                        origins[self.rt.estimator.held_id] = origin
                        getattr(self, '_tool_sources', {}).pop(plan['target'], None)
                elif skill == 'place':
                    self.ex._place(job, plan)
                    with self.rt._lock:
                        t = self.rt.estimator.tracks.get(plan['object'])
                        if t is not None and hasattr(self, '_tool_sources'):
                            self._tool_sources[t.id] = deepcopy(t)
                elif skill == 'navigate':
                    self.ex._drive(job, plan['path'], plan['yaw'])
                elif skill == '_fastener_reset':
                    self.ex._set_joint('wrist_yaw', plan['wristYaw'], tol=.02)
                    self.ex._set_joint('wrist_roll', plan['wristRoll'], tol=.02)
                    if 'wrist_pitch' in self.ex._state()['joints']:
                        self.ex._set_joint('wrist_pitch', plan['wristPitch'], tol=.02)
                elif skill == '_simple_driver':
                    self.simple_stroke(job, plan)
                elif skill == '_fastener_contact':
                    self.contact(job, plan)
                elif skill in ('_stage_screw', '_upright_tool'):
                    self.stage_release(job, plan)
                elif skill == 'assemble_onto':
                    AssembleOnto(self.rt).run(job, plan)
                else:
                    child = Fasten(self.rt)
                    child._tool_sources = getattr(self, '_tool_sources', {})
                    child._seat_retry_budget = getattr(self, '_seat_retry_budget', [2])
                    child.run(job, plan)
                row['result'] = deepcopy(job.postcondition) if job.postcondition is not root else {'executed':True}
                self.progress(job, skill, True, 'executed', result=root)
                return plan
            except Exception as exc:
                row['failure'] = str(exc)
                self.progress(job, skill, False, str(exc), result=root)
                if job.postcondition is not root:
                    row['result'] = deepcopy(job.postcondition)
                raise
            finally:
                row['wallS'] = time.monotonic()-started
                job.postcondition = root

    def reference(self, body_id):
        if _assembly_enabled():
            with self.rt._lock:
                records = self.rt.estimator.stationary_assemblies.records
                record = records.get(body_id)
                from steward.ego import edge_retention
                if edge_retention.enabled() and record and record.get('edgeRetained'):
                    return deepcopy(record)
                if record and record.get('status') != 'frozen' and not record.get('hardMotion'):
                    self.warn('stationary_reference_estimated', entity=body_id, status=record.get('status'))
                    records.pop(body_id)
        else:
            with self.rt._lock:
                records = self.rt.estimator.stationary_assemblies.records
                record = records.get(body_id)
                if record and record.get('status') != 'frozen' and not record.get('hardMotion'):
                    self.warn('stationary_reference_estimated', entity=body_id, status=record.get('status'))
                    records.pop(body_id)
        return super().reference(body_id)

    def sample(self, entity):
        with self.rt._lock:
            track = self.rt.estimator.tracks.get(entity)
            return deepcopy(track.manipulation_view) if track else None

    if _assembly_enabled():
        @timed("wait_next_view")
        def wait_view(self, entity, after, *, timeout=2):
            start = self.ex._state()['stamp']; deadline = ab.make(self.ex, timeout, 'caller-specified sensor frame interval') if ab.enabled() else time.monotonic()+300
            while self.ex._state()['stamp']-start < timeout and not ab.expired(deadline):
                self.check()
                view = self.sample(entity)
                if (view and view['stamp'] > after and not view['truncated']
                        and view['invalidDepthFraction'] <= .05):
                    return view
                try:
                    self.ex._wait(.25)
                except SkillFailed as exc:
                    self.check()
                    self.warn('view_wait_timeout_continue', entity=entity, detail=str(exc))
                    break
            view = self.sample(entity)
            self.warn('no_fresh_tool_view', entity=entity, after=after, ageS=self.ex._state()['stamp']-view['stamp'] if view else None)
            return view
    else:
        def wait_view(self, entity, after, *, timeout=2):
            start = self.ex._state()['stamp']; deadline = time.monotonic()+300
            while self.ex._state()['stamp']-start < timeout and time.monotonic() < deadline:
                self.check()
                view = self.sample(entity)
                if (view and view['stamp'] > after and not view['truncated']
                        and view['invalidDepthFraction'] <= .05):
                    return view
                try:
                    self.ex._wait(.25)
                except SkillFailed as exc:
                    self.check()
                    self.warn('view_wait_timeout_continue', entity=entity, detail=str(exc))
                    break
            view = self.sample(entity)
            self.warn('no_fresh_tool_view', entity=entity, after=after, ageS=self.ex._state()['stamp']-view['stamp'] if view else None)
            return view

    def feature(self, entity):
        ent = self.rt.ego_state().find(entity)
        feature = tool_estimate(ent, self.rt.estimator.stamp)
        attempt = next((a for a in self.rt.estimator.fastening_attempts.values() if a['screw'] == entity), None)
        if attempt and (not feature or feature.get('source') != 'rgbd_endpoint_width'
                        or feature['stamp'] <= attempt['seatedAt']):
            from steward.ego.classify import SHAPES
            feature = dict(tip=attempt['entry'], head=attempt['headBeforeRelease'], axis=[0,0,1],
                stamp=attempt['seatedAt'], source='seating_action_estimate', polarityObserved=False,
                lengthM=max(SHAPES['toy_screw']['dims']), radiusM=min(SHAPES['toy_screw']['dims'])/2)
        if not feature:
            raise SkillFailed(f'{entity}: no tool target or seating action to locate it')
        if not feature.get('polarityObserved'):
            self.warn('nominal_tool_endpoints', entity=entity, source=feature['source'])
        return feature

    def choose(self, category, support, *, free_screw=False):
        with self.rt._lock:
            for eid, track in getattr(self, '_tool_sources', {}).items():
                associated = any(t.category == track.category and not t.held
                    and t.last_seen >= track.last_seen and np.linalg.norm(t.pos-track.pos) < .04
                    for t in self.rt.estimator.tracks.values())
                if eid not in self.rt.estimator.tracks and not track.held and not associated:
                    self.rt.estimator.tracks[eid] = deepcopy(track)
                    self.rt.estimator.version += 1
                    self.warn('tool_last_source_restored', entity=eid, observedAt=track.last_seen)
        if _assembly_enabled():
            from steward.ego.assembly_session import current
            session = current()
            if free_screw:
                from steward.ego.fasten_accounting import kit_support
                support = kit_support(self, support)
            scene = self.rt.ego_state()
        else:
            scene = self.rt.ego_state(); used = self.rt.estimator.fastening_attempts.values()
            used_ids = {a['screw'] for a in used}
        candidates = []
        if _assembly_enabled():
            selection = dict(requestedSupport=support, candidates=[])
            for ent in scene.entities:
                from steward.ego import grasp_binding
                if grasp_binding.enabled() and (ent.id in getattr(self.rt.estimator, 'rejected_load_candidates', ())
                        or ent.id.startswith('load:')):
                    continue
                if free_screw:
                    from steward.ego.fasten_supply import selection_candidate
                    detail = dict(entity=ent.id)
                    feature = selection_candidate(self, ent, scene, support, detail)
                    selection['candidates'].append(detail)
                    if feature is None:
                        continue
                else:
                    if session is not None and not session.owns(ent.id, tool=True):
                        continue
                track = self.rt.estimator.tracks.get(ent.id)
                seen_at = track.last_seen if track is not None else (ent.pose.t or -1)
                if not free_screw and ent.id in getattr(self, '_driver_misses', {}) and seen_at <= self._driver_misses[ent.id]:
                    continue
                if category == 'electric_screwdriver':
                    memories = self.rt.estimator.fastener_origins
                    memory = memories.get(ent.id, {}).get('driverRelease')
                    if memory and track is not None and np.any(
                            np.sort(track.size) > np.sort(memory['track'].size)*1.6+.01):
                        continue  # merged envelope; search_driver supplies a release-FK alias
                    if any(r.get('alias') == ent.id and r.get('missedAt', -1) >= r['stamp']
                           for o in memories.values() for r in [o.get('driverRelease', {})] if r):
                        continue
                if not free_screw:
                    if ent.category.value != category:
                        continue
                    feature = tool_estimate(ent, scene.stamp)
                    if not feature:
                        continue
                anchor = self.rt.esim._anchor(scene, ent)
                if anchor.id != support:
                    self.warn('tool_support_estimated', entity=ent.id, expected=support, estimated=anchor.id)
                self.warn('tool_estimate', entity=ent.id, ageS=scene.stamp-feature['stamp'], source=feature['source'])
                candidates.append(ent)
            supply_pregrasp_chosen = None
            from steward.ego import supply_pregrasp
            if free_screw and candidates and supply_pregrasp.enabled():
                supply_pregrasp_chosen = supply_pregrasp.select(self, candidates, scene, support, selection)
            if free_screw:
                if session is not None:
                    for entity in session.state['goal']['materials']:
                        if scene.find(session.canonical(entity)) is None:
                            selection['candidates'].append(dict(entity=entity, eligible=False, reason='bound_track_unavailable'))
                selection.update(available=[e.id for e in candidates], insufficient=not candidates)
                if session is not None:
                    session.emit('supply_selection', **selection)
                job = getattr(self, '_observation_job', None)
                if job is not None:
                    job.postcondition.setdefault('supplyDiagnostics', []).append(deepcopy(selection))
            if not candidates:
                raise SkillFailed(('本套紧固件不足：绑定承载面当前没有可用松散紧固件') if free_screw else
                                  f'no available {category} target on support')
        else:
            for ent in scene.entities:
                if ent.category.value != category:
                    continue
                feature = tool_estimate(ent, scene.stamp)
                if not feature:
                    continue
                if free_screw and (ent.id in used_ids or ent.id in getattr(self, '_retired_screws', set())
                        or ent.id in getattr(self, '_missed_screws', set())
                        or ent.id in getattr(self, '_reserved_screws', set())):
                    continue
                if free_screw and any(np.linalg.norm(np.asarray(ent.pose.value['p'])[:2]-a['entry'][:2]) < .025
                                      for a in self.rt.estimator.fastening_attempts.values()):
                    continue  # includes a new segmentation ID for an already released screw
                if free_screw and any(np.linalg.norm(np.asarray(ent.pose.value['p'])[:2]-xy) < .025
                                      for xy in getattr(self,'_reserved_positions',[])):
                    continue
                anchor = self.rt.esim._anchor(scene, ent)
                if anchor.id != support:
                    self.warn('tool_support_estimated', entity=ent.id, expected=support, estimated=anchor.id)
                self.warn('tool_estimate', entity=ent.id, ageS=scene.stamp-feature['stamp'], source=feature['source'])
                candidates.append(ent)
            if not candidates:
                raise SkillFailed(f'no available {category} target on support')
        o = self.ex._state()['odometry']
        if _assembly_enabled():
            chosen = supply_pregrasp_chosen or min(candidates, key=lambda e: (
                category == 'electric_screwdriver' and bool(getattr(self, '_driver_target', None))
                    and e.id != self._driver_target,
                self.rt.esim._anchor(scene, e).id != support,
                math.hypot(e.pose.value['p'][0]-o['x'], e.pose.value['p'][1]-o['y']), e.id))
            if free_screw and getattr(self, '_lower_interface', False):
                from steward.ego.fasten_supply import measured_fastener
                observed = measured_fastener(chosen, scene.stamp)
                if observed and chosen.id in self.rt.estimator.tracks:
                    with self.rt._lock:
                        self.rt.estimator.tracks[chosen.id].tool_feature = observed
            if free_screw and (current() is not None):
                session.adopt_supply(chosen, session.stock_source(chosen, scene))
                actual=self.rt.esim._anchor(scene,chosen).id
                self._supply_supports=getattr(self,'_supply_supports',{})
                self._supply_supports[chosen.id]=actual
                job=getattr(self,'_observation_job',None)
                if job is not None and actual!=support:
                    job.postcondition.setdefault('supplySelections',[]).append(dict(entity=chosen.id,
                        requestedSupport=support,observedSupport=actual,source='session_bound_material' if current() is not None else 'unoccupied_same_category_other_support'))
            return chosen
        else:
            return min(candidates, key=lambda e: (self.rt.esim._anchor(scene, e).id != support,
                math.hypot(e.pose.value['p'][0]-o['x'], e.pose.value['p'][1]-o['y']), e.id))

    def stage_release(self, job, plan):
        """Same checked endpoint path as contact; release upright on the table."""
        self.ex._dock(job, plan, retract=False)
        self.ex._set_joint('lift', plan['safeLift'])
        self.ex._set_joint('reach', plan['retractReach'])
        for key, field in (('wrist_yaw','wristYaw'),('wrist_roll','wristRoll'),('wrist_pitch','wristPitch')):
            if key in self.ex._state()['joints']:
                self.ex._set_joint(key, plan[field], tol=.02)
        self.ex._align_lateral(plan['gripSpot'], tol=.002)
        self.ex._set_joint('reach', self.ex._reach_to(plan['gripSpot']))
        self.ex._set_joint('lift', plan['lift'])
        self.ex._settle_for_release()
        self.check()
        self.ex._command({'gripper':'open'})
        self.ex._released(plan['object'], plan['goal'][:2])
        job.postcondition['stagingReleaseAt'] = self.ex._state()['stamp']
        self.ex._wait(.15)
        self.ex._set_joint('lift', plan['safeLift'])
        self.ex._withdraw(plan)
        for key in ('wrist_yaw','wrist_roll','wrist_pitch'):
            if key in self.ex._state()['joints']:
                self.ex._set_joint(key, 0., tol=.02)

    def stage_screws(self, job, plan, holes):
        from steward.ego.assembly_quick import staging_spots
        from steward.ego.classify import SHAPES
        if _assembly_enabled():
            self._supply_members = {plan.get('body'), plan.get('part')}
        self.body_id = plan['body']
        self.reference(self.body_id)
        scene = self.rt.ego_state()
        body, part = scene.entity(plan['body']), scene.entity(plan['part'])
        table = support_anchor(self.rt.esim, scene, body, [])
        spots = staging_spots(self.rt.esim, scene, body, part, table, holes)
        staged=[]; self._reserved_screws=set(); self._reserved_positions=[]
        if _assembly_enabled():
            for hole in holes:
                from steward.ego.assembly_session import current
                if current() is not None:
                    current().state['pickTargetHole'] = dict(body=plan['body'], hole=hole['id'])
                candidates=[xy for xy in spots if all(np.linalg.norm(np.asarray(xy)-r['xy'])>.035 for r in staged)]
                if not candidates:
                    self.warn('staging_no_near_table_slot', hole=hole['id']); break
                self._stance_retry_used=False; self._missed_screws=set()
                try:
                    for pick_try in range(2):
                        screw=self.acquire_tool(job,'toy_screw',table.id,free_screw=True)
                        # RGB-D endpoints win; otherwise the catalogue's table-rest
                        # pose is a better hypothesis than a padded tiny OBB axis.
                        from steward.ego.fasten_geometry import nominal_tool
                        with self.rt._lock:
                            t=self.rt.estimator.tracks[screw.id]
                            if not t.tool_feature or t.tool_feature.get('source') != 'rgbd_endpoint_width':
                                dims=SHAPES['toy_screw']['dims']
                                t.tool_feature=nominal_tool('toy_screw',t.pos,dims,np.eye(3),t.last_seen)
                        try:
                            self.reach_retry(job,'pick',dict(target=screw.id,**({'expectedFastener':True} if sr.role_enabled() or sr.size_enabled() else {})),table.id)
                            break
                        except SkillFailed as exc:
                            self.check()
                            if pick_try or not any(x in str(exc) for x in ('grasp missed','grasp mismatch')) or self.rt.estimator.held_id:
                                raise
                            self._missed_screws.add(screw.id)
                    eid=self.rt.estimator.held_id or screw.id
                    snapshot=deepcopy(self.rt.estimator.tracks[eid])
                    last=None
                    for xy in candidates:
                        params=dict(body=body.id,stageOn=table.id,stageXY=xy,operation='seat')
                        receipt=self.rt.predict('_stage_screw',params)
                        if not receipt.get('withinRiskLimit'):
                            last=receipt.get('issues'); continue
                        self.step(job,'_stage_screw',params,receipt=receipt)
                        break
                    else:
                        raise SkillFailed(f'no reachable nearby upright staging slot: {last}')
                    row=dict(entity=eid,xy=list(xy),hole=hole['id'],support=table.id,
                             position=[*xy,entity_box(table)[2]+max(SHAPES['toy_screw']['dims'])/2],
                             stagedAt=self.ex._state()['stamp'])
                    # An action target, explicitly not a new sensory sample.
                    snapshot.pos=np.array(row['position']); snapshot.held=False
                    snapshot.size=np.array(SHAPES['toy_screw']['dims']); snapshot.yaw=0.
                    snapshot.tool_feature=snapshot.held_tool_feature=None
                    snapshot.manipulation_view=None
                    row['track']=snapshot
                    staged.append(row); self._reserved_screws.update((screw.id,eid))
                    self._reserved_positions.append(np.asarray(xy))
                except SkillFailed as exc:
                    self.check(); self.warn('staging_failed_continue',hole=hole['id'],detail=str(exc))
                    if self.rt.estimator.held_id and not self.park(job,table.id):
                        raise
        else:
            for hole in holes:
                candidates=[xy for xy in spots if all(np.linalg.norm(np.asarray(xy)-r['xy'])>.035 for r in staged)]
                if not candidates:
                    self.warn('staging_no_near_table_slot', hole=hole['id']); break
                self._stance_retry_used=False; self._missed_screws=set()
                try:
                    for pick_try in range(2):
                        screw=self.acquire_tool(job,'toy_screw',table.id,free_screw=True)
                        # RGB-D endpoints win; otherwise the catalogue's table-rest
                        # pose is a better hypothesis than a padded tiny OBB axis.
                        from steward.ego.fasten_geometry import nominal_tool
                        with self.rt._lock:
                            t=self.rt.estimator.tracks[screw.id]
                            if not t.tool_feature or t.tool_feature.get('source') != 'rgbd_endpoint_width':
                                dims=SHAPES['toy_screw']['dims']
                                t.tool_feature=nominal_tool('toy_screw',t.pos,dims,np.eye(3),t.last_seen)
                        try:
                            self.reach_retry(job,'pick',dict(target=screw.id),table.id)
                            break
                        except SkillFailed as exc:
                            self.check()
                            if pick_try or 'grasp missed' not in str(exc) or self.rt.estimator.held_id:
                                raise
                            self._missed_screws.add(screw.id)
                    eid=self.rt.estimator.held_id or screw.id
                    snapshot=deepcopy(self.rt.estimator.tracks[eid])
                    last=None
                    for xy in candidates:
                        params=dict(body=body.id,stageOn=table.id,stageXY=xy,operation='seat')
                        receipt=self.rt.predict('_stage_screw',params)
                        if not receipt.get('withinRiskLimit'):
                            last=receipt.get('issues'); continue
                        self.step(job,'_stage_screw',params,receipt=receipt)
                        break
                    else:
                        raise SkillFailed(f'no reachable nearby upright staging slot: {last}')
                    row=dict(entity=eid,xy=list(xy),hole=hole['id'],support=table.id,
                             position=[*xy,entity_box(table)[2]+max(SHAPES['toy_screw']['dims'])/2],
                             stagedAt=self.ex._state()['stamp'])
                    # An action target, explicitly not a new sensory sample.
                    snapshot.pos=np.array(row['position']); snapshot.held=False
                    snapshot.size=np.array(SHAPES['toy_screw']['dims']); snapshot.yaw=0.
                    snapshot.tool_feature=snapshot.held_tool_feature=None
                    snapshot.manipulation_view=None
                    row['track']=snapshot
                    staged.append(row); self._reserved_screws.update((screw.id,eid))
                    self._reserved_positions.append(np.asarray(xy))
                except SkillFailed as exc:
                    self.check(); self.warn('staging_failed_continue',hole=hole['id'],detail=str(exc))
                    if self.rt.estimator.held_id and not self.park(job,table.id):
                        raise
        job.postcondition['stagedScrews']=[{k:v for k,v in r.items() if k!='track'} for r in staged]
        return staged

    if _assembly_enabled():
        @timed("supply_selection")
        def acquire_tool(self, job, category, support, *, free_screw=False):
            if category == 'electric_screwdriver':
                from steward.ego.fasten_driver import search_driver
                support = search_driver(self, job, support)
            if free_screw:
                from steward.ego.fasten_accounting import kit_support
                support = kit_support(self, support)
            scene = self.rt.ego_state(); table = scene.find(support)
            if table is None and free_screw:
                raise SkillFailed('本套紧固件不足：原工作面当前不可定位')
            box,_,_=entity_box(table); o=self.ex._state()['odometry']
            origin=math.atan2(o['y']-box.cy,o['x']-box.cx)
            last = None
            for turn in (None, 0, .65, -.65):
                try:
                    if turn is not None:
                        arrived=self.view(job,table,origin,turn)
                        self.look(self.ex._state()['stamp'] if arrived is None else arrived)
                    return self.choose(category,support,free_screw=free_screw)
                except SkillFailed as exc:
                    last = exc
            raise SkillFailed(str(last))
    else:
        def acquire_tool(self, job, category, support, *, free_screw=False):
            scene = self.rt.ego_state(); table = scene.entity(support)
            box,_,_=entity_box(table); o=self.ex._state()['odometry']
            origin=math.atan2(o['y']-box.cy,o['x']-box.cx)
            last = None
            for turn in (None, 0, .65, -.65):
                try:
                    if turn is not None:
                        arrived=self.view(job,table,origin,turn)
                        self.look(self.ex._state()['stamp'] if arrived is None else arrived)
                    return self.choose(category,support,free_screw=free_screw)
                except SkillFailed as exc:
                    last = exc
            raise SkillFailed(str(last))

    def park(self, job, support):
        """Origin, any estimated table space, then nearby floor; failure is a result."""
        from steward.ego.fasten_simple import offset_on
        held = self.rt.estimator.held_id
        if not held:
            return True
        if _assembly_enabled():
            from steward.ego.wrist_target import allow_return
            if not allow_return(self, job):
                raise SkillFailed('held fastener retained at target; return requires misalignment and exhausted corrections')
        scene = self.rt.ego_state()
        origin = getattr(self.rt.estimator, 'fastener_origins', {}).get(held)
        options = []
        if origin and scene.find(origin['support']):
            options.append(dict(on=origin['support'], offset=offset_on(scene.entity(origin['support']), origin['xy'])))
        tables = [e for e in scene.entities if self.rt.esim._affords(e, 'support-surface')]
        tables.sort(key=lambda e:(e.id != support, np.linalg.norm(np.asarray(e.pose.value['p'])[:2]-self.rt.esim._robot_xy_yaw(scene)[:2])))
        options.extend(dict(on=e.id) for e in tables)
        x,y,yaw = self.rt.esim._robot_xy_yaw(scene)
        options.extend(dict(at=[x+r*math.cos(yaw+a), y+r*math.sin(yaw+a)])
                       for r in (.45,.65) for a in (0, math.pi/2, -math.pi/2, math.pi))
        for params in options:
            try:
                self.step(job, 'place', dict(params, alignHoles=False))
                if not self.rt.estimator.held_id:
                    return True
            except SkillFailed as exc:
                self.check()
                self.warn('parking_candidate_failed', params=params, detail=str(exc))
        self.warn('parking_exhausted', held=self.rt.estimator.held_id)
        return False

    if _assembly_enabled():
        @timed("held_identity")
        def held_kind_check(self, job, support, *, replay_identity=None):
            """held_fastener_kind: never carry a held load the estimator identifies as a non-fastener to a hole."""
            from steward.ego.fasten_identity import held_check
            held_check(self, job, support, replay_identity=replay_identity)
            held = self.rt.ego_state().find(self.rt.estimator.held_id)
            if held is not None and held.category.value != 'toy_screw':
                job.postcondition.setdefault('graspChecks', []).append(dict(
                    status='contradicted', reason='held_not_a_fastener', actual=held.id, category=held.category.value))
                self.park(job, support)
                raise SkillFailed('grasp mismatch: held_not_a_fastener')
            from steward.ego.fasten_upright import ensure
            ensure(self, job, support)

        @timed("held_shape")
        def seat_grasp_check(self, job, requested, support, *, replay_identity=None):
            from steward.ego.fasten_identity import held_check
            held_check(self, job, support, replay_identity=replay_identity)
            from steward.ego.fasten_seat import grasp_check
            check = grasp_check(self.rt.ego_state(), self.rt.estimator.held_id, requested)
            job.postcondition.setdefault('graspChecks', []).append(check)
            if check['status'] == 'contradicted':
                self._screw_misses[requested] = self.ex._state()['stamp']
                self.park(job, support)
                raise SkillFailed('grasp mismatch: '+check['reason'])
            from steward.ego.fasten_upright import ensure
            ensure(self, job, support)

    def simple_seat(self, job, plan):
        from steward.ego.fasten_simple import offset_on
        from steward.ego.fasten_plan import seat_target
        if _assembly_enabled():
            self._supply_members = {plan.get('body'), plan.get('part')}
        self.body_id = plan['body']
        if _assembly_enabled():
            from steward.ego.registered_alignment import register
            from steward.ego import axial_release
        key = (plan['body'], plan['hole'])
        previous = self.rt.estimator.fastening_attempts.get(key)
        if _assembly_enabled():
            from steward.ego import first_interface, assembly_outcome
            refill = bool(first_interface.enabled() and previous and assembly_outcome.refill_ready(previous))
            if previous and axial_release.handled(previous) and not refill:
                job.postcondition.update(deepcopy(previous), executed=True, repeated=True,
                                         confirmedSeatedScrews=int(previous.get('seatConfirmed', False)))
                return
            if not (axial_release.enabled() and previous and previous.get('releaseExecuted')):
                register(self, job, plan)
            from steward.ego import assembly_outcome
            if previous and assembly_outcome.refill_ready(previous) and (not axial_release.enabled() or refill):
                previous = None
            from steward.ego.hole_seating import confirmed
            if confirmed(previous):
                job.postcondition.update(deepcopy(previous), executed=True, repeated=True, confirmedSeatedScrews=1)
                return
        else:
            if previous and previous.get('seatConfirmed'):
                job.postcondition.update(deepcopy(previous), executed=True, repeated=True, confirmedSeatedScrews=1)
                return
        scene = self.rt.ego_state()
        support = support_anchor(self.rt.esim, scene, scene.entity(self.body_id), []).id
        if _assembly_enabled():
            if previous:
                attempt = previous
            else:
                self._stance_retry_used = False
                self._missed_screws = set()
                self._reserved_screws = set(plan.get('excludedScrews', []))
                self._reserved_positions = [np.asarray(xy) for xy in plan.get('excludedPositions', [])]
                for pick_try in range(2):
                    staged_id = plan.get('stagedScrew') if pick_try == 0 else None
                    screw = self.rt.ego_state().find(staged_id) if staged_id else None
                    if screw is not None:
                        from steward.ego.fasten_accounting import belongs, kit_support
                        if not belongs(self, screw, kit_support(self, support)):
                            raise SkillFailed('本套紧固件不足：预备料不属于本工作面')
                    screw = screw or self.acquire_tool(job, 'toy_screw', support, free_screw=True)
                    try:
                        self.reach_retry(job, 'pick', dict(target=screw.id,**({'axialFastener':True} if axial_release.enabled() else {}),**({'expectedFastener':True} if sr.role_enabled() or sr.size_enabled() else {}),
                            **({'stayNear':True} if plan.get('deferVerify') else {})),
                            getattr(self,'_supply_supports',{}).get(screw.id,support))
                        if getattr(self, '_seat_only', False) and getattr(self, '_lower_interface', False):
                            self.seat_grasp_check(job, screw.id, support)
                        else:
                            self.held_kind_check(job, support)  # held_fastener_kind: ordinary path checks only the identified category
                        break
                    except SkillFailed as exc:
                        self.check()
                        if getattr(self, '_seat_only', False) and getattr(self, '_lower_interface', False) and any(x in str(exc) for x in ('grasp missed', 'grasp mismatch')):
                            self._screw_misses[screw.id] = self.ex._state()['stamp']
                        if pick_try or not any(x in str(exc) for x in ('grasp missed', 'grasp mismatch')) or self.rt.estimator.held_id:
                            raise
                        self._missed_screws.add(screw.id)
                        self.warn('screw_pick_retry_other', missed=screw.id, detail=str(exc))
                screw_id = self.rt.estimator.held_id or screw.id
                from steward.ego.assembly_precision import acquired
                acquired(self, job, screw_id)
                from steward.ego.registered_alignment import refresh_if_moved
                refresh_if_moved(self, job, plan)
                scene = self.rt.ego_state()
                target = seat_target(self.rt.esim, scene, plan)
                from steward.ego.fasten_release import component_support
                part = (component_support(scene, target['componentFrame']) if target.get('componentFrame')
                        else scene.entity(plan['part']))
                point = np.asarray(target['point'])
                params = dict(on=part.id, offset=offset_on(part, point[:2]), alignHoles=False, clearance=0.,
                              seatBody=plan['body'], seatHole=plan['hole'])
                if getattr(self, '_seat_only', False) and getattr(self, '_lower_interface', False):
                    params['lowerSeatSurface'] = True
                from steward.ego.fasten_landing import ENABLED as landing_enabled
                if landing_enabled:
                    params['boundedSeatRelease'] = True
                if plan.get('closeRelease'):
                    params['closeRelease'] = True
                if plan.get('deferVerify'):
                    params['stayNear'] = True
                from steward.ego.fasten_release import local_head_baseline
                baseline = local_head_baseline(self.rt.ego_state(), point)
                from steward.ego.wrist_landing import prepare as prepare_landing
                prepare_landing(self, plan)
                place = self.reach_retry(job, 'place', params, support)
                # Preserve the target actually consumed by the placement receipt.
                target = place.get('seatTarget', target)
                wrist = (place.get('precision') or {}).get('wristAlignment', {})
                from steward.ego.wrist_target import adoption_enabled
                if wrist.get('selectedHole') and (not adoption_enabled() or
                        wrist.get('targetDecision') != 'kept_original'):
                    plan['hole'] = wrist['selectedHole']
                    key = (plan['body'], plan['hole'])
                    # The observed hole remains the posterior target even when a bounded
                    # correction stopped short; a command endpoint is not a measured hole.
                    target = dict(target, point=list(wrist['targetPointM']))
                point = np.asarray(target['point'])
                after = place.get('releasedAt', self.ex._state()['stamp'])
                from steward.ego.classify import SHAPES
                height = max(SHAPES['toy_screw']['dims'])
                nominal_tip = [*place['spot'], point[2]]
                attempt = dict(body=plan['body'], part=part.id, hole=plan['hole'], partHole=target['partHole'],
                    screw=screw_id, entry=point.tolist(), headBeforeRelease=(np.asarray(nominal_tip)+[0,0,height]).tolist(),
                    headBeforeSource='ordinary_place_settled_estimate', seatedAt=after,
                    status='released_unverified', verified=False, method='simple', releaseExecuted=True,
                    commandedSpotM=place['spot'], estimatedSettledTipM=nominal_tip, landingErrors=[],
                    targetSource=target.get('targetSource', 'current_component_estimate'),
                    **({'surfaceEvidence':deepcopy(target['surfaceEvidence'])} if target.get('surfaceEvidence') else {}),
                    pairAlignment=deepcopy(target.get('pairAlignment')),
                    componentFrame=deepcopy(target.get('componentFrame')), headPatchBefore=baseline,
                    releaseToolFeature=deepcopy(place.get('releaseToolFeature')),
                    seatMotion=deepcopy(place.get('seatMotion')), releaseGapM=place.get('releaseGapM'),
                    precision=place.get('precision'),
                    nearContactUnconfirmed=bool(place.get('nearContactUnconfirmed')))
                with self.rt._lock:
                    recorded = self.rt.estimator.fastening_attempts.get(key, {})
                    attempt['releaseHistory'] = recorded.get('releaseHistory', [dict(screw=screw_id, stamp=after)])
                    if recorded.get('loadReleaseHistory'):
                        attempt['loadReleaseHistory'] = deepcopy(recorded['loadReleaseHistory'])
                    from steward.ego.grasp_binding import annotate_release
                    annotate_release(self.rt.estimator, attempt, screw_id)
                    self.rt.estimator.fastening_attempts[key] = attempt
                    from steward.ego.assembly_session import current
                    if current() is not None:
                        current().sync_releases()
            from steward.ego.hole_seating import apply as apply_hole
            apply_hole(attempt, locals().get('place'))
            from steward.ego.released_object import recovery_evidence
            if axial_release.enabled():
                axial_release.result_view(self, job, attempt, locals().get('place'))
                self.confirm_seat(job, attempt, plan, observe=False)
                return
            if recovery_evidence(attempt, attempt.get('seatCheck') or {}):
                self.confirm_seat(job, attempt, plan, observe=False)
                return
            from steward.ego.release_observation import ENABLED as release_view, observe as observe_release
            if release_view:
                observe_release(self, job, attempt, locals().get('place'))
                self.confirm_seat(job, attempt, plan, observe=False)
                return
            from steward.ego import wrist_observation as wo
            if wo.release_reporting():
                from steward.ego.wrist_fasten import released
                released(self,job,attempt,locals().get('place'))
            if plan.get('deferVerify'):
                if ((attempt.get('precision') or {}).get('observations') and not getattr(self, '_lower_interface', False)
                        and not attempt.get('holeSeatObservation')):
                    self.wait_view(attempt['screw'], attempt['seatedAt'])
                # On the lower interface, capture each short-lived view before the
                # next source trip. No replay or camera tour until all holes attempt.
                if (getattr(self, '_lower_interface', False) and not attempt.get('seatConfirmed')
                        and not attempt.get('holeSeatObservation')):
                    self.wait_view(attempt['screw'], attempt['seatedAt'])
                # The top interface retains the existing batch timing.
                self.seat_posterior(attempt)
                job.postcondition.update(deepcopy(attempt), executed=True, verificationDeferred=True)
                return
        else:
            if previous:
                attempt = previous
            else:
                self._stance_retry_used = False
                self._missed_screws = set()
                self._reserved_screws = set(plan.get('excludedScrews', []))
                self._reserved_positions = [np.asarray(xy) for xy in plan.get('excludedPositions', [])]
                for pick_try in range(2):
                    staged_id = plan.get('stagedScrew') if pick_try == 0 else None
                    screw = self.rt.ego_state().find(staged_id) if staged_id else None
                    screw = screw or self.acquire_tool(job, 'toy_screw', support, free_screw=True)
                    try:
                        self.reach_retry(job, 'pick', dict(target=screw.id,
                            **({'stayNear':True} if plan.get('deferVerify') else {})), support)
                        break
                    except SkillFailed as exc:
                        self.check()
                        if pick_try or 'grasp missed' not in str(exc) or self.rt.estimator.held_id:
                            raise
                        self._missed_screws.add(screw.id)
                        self.warn('screw_pick_retry_other', missed=screw.id, detail=str(exc))
                screw_id = self.rt.estimator.held_id or screw.id
                scene = self.rt.ego_state()
                target = seat_target(self.rt.esim, scene, plan)
                from steward.ego.fasten_release import component_support
                part = (component_support(scene, target['componentFrame']) if target.get('componentFrame')
                        else scene.entity(plan['part']))
                point = np.asarray(target['point'])
                params = dict(on=part.id, offset=offset_on(part, point[:2]), alignHoles=False, clearance=0.,
                              seatBody=plan['body'], seatHole=plan['hole'])
                if plan.get('closeRelease'):
                    params['closeRelease'] = True
                if plan.get('deferVerify'):
                    params['stayNear'] = True
                place = self.reach_retry(job, 'place', params, support)
                # Preserve the target actually consumed by the placement receipt.
                target = place.get('seatTarget', target)
                point = np.asarray(target['point'])
                after = place.get('releasedAt', self.ex._state()['stamp'])
                from steward.ego.classify import SHAPES
                height = max(SHAPES['toy_screw']['dims'])
                nominal_tip = [*place['spot'], point[2]]
                attempt = dict(body=plan['body'], part=part.id, hole=plan['hole'], partHole=target['partHole'],
                    screw=screw_id, entry=point.tolist(), headBeforeRelease=(np.asarray(nominal_tip)+[0,0,height]).tolist(),
                    headBeforeSource='ordinary_place_settled_estimate', seatedAt=after,
                    status='released_unverified', verified=False, method='simple', releaseExecuted=True,
                    commandedSpotM=place['spot'], estimatedSettledTipM=nominal_tip, landingErrors=[],
                    targetSource=target.get('targetSource', 'current_component_estimate'),
                    componentFrame=deepcopy(target.get('componentFrame')))
                with self.rt._lock:
                    self.rt.estimator.fastening_attempts[key] = attempt
            if plan.get('deferVerify'):
                job.postcondition.update(deepcopy(attempt), executed=True, verificationDeferred=True)
                return
        self.confirm_seat(job, attempt, plan)

    if _assembly_enabled():
        @timed("release_posterior")
        def seat_posterior(self, attempt):
            from steward.ego import axial_release
            if axial_release.checked(attempt):
                return deepcopy(attempt['axialReleaseCheck'])
            from steward.ego.fasten_release import seat_check
            from steward.ego.fasten_simple import landing_error
            scene = self.rt.ego_state()
            screw = scene.find(attempt['screw'])
            from steward.ego import released_object
            if released_object.current_success(attempt):
                return attempt['seatCheck']
            direct = attempt.get('holeSeatObservation') or {}
            direct_current = (direct.get('releasedAt') == attempt.get('seatedAt') and
                              bool(direct) and not attempt.get('transportedAt'))
            if released_object.enabled() and direct.get('check', {}).get('source') == released_object.SOURCE:
                direct_current = (direct.get('releasedAt') == attempt.get('seatedAt') and
                                  attempt.get('transportedAt',-1) < attempt.get('seatedAt',-1))
            check = (deepcopy(direct['check']) if direct_current else
                     seat_check(screw, attempt.get('componentFrame'), attempt['entry'],
                                attempt['seatedAt'], scene.stamp))
            object_check = released_object.enabled() and check.get('source') == released_object.SOURCE
            if direct_current and screw is not None and not object_check:
                later = seat_check(screw, attempt.get('componentFrame'), attempt['entry'],
                                   attempt['seatedAt'], scene.stamp)
                if later.get('passed') is not None and (later.get('observedAt') or -1) > (check.get('observedAt') or -1):
                    check = later
            if check['status'] == 'unknown' and not direct_current:
                matches = []
                for candidate in scene.entities:
                    if candidate.id == attempt['screw']:
                        continue
                    measured = seat_check(candidate, attempt.get('componentFrame'), attempt['entry'],
                                          attempt['seatedAt'], scene.stamp)
                    if not measured['passed']:
                        continue
                    peers = [a for a in self.rt.estimator.fastening_attempts.values() if a is not attempt]
                    if any(a['screw'] == candidate.id or seat_check(candidate, a.get('componentFrame'),
                           a['entry'], a['seatedAt'], scene.stamp)['passed'] for a in peers):
                        continue
                    matches.append((candidate, measured))
                if len(matches) == 1:
                    screw, check = matches[0]
                    attempt.setdefault('screwAliases', []).append(attempt['screw'])
                    attempt['screw'] = screw.id
                    check['reassociated'] = True
                elif len(matches) > 1:
                    check['reason'] = 'ambiguous_separate_screw_candidates'
            if check['status'] == 'unknown' and not direct_current:
                from steward.ego.fasten_release import local_head_check
                local = local_head_check(scene, attempt)
                if local and not any(a is not attempt and a.get('seatConfirmed') and
                        np.linalg.norm(np.asarray(a.get('seatCheck', {}).get('headM', a['entry']))-local['headM']) < .006
                        for a in self.rt.estimator.fastening_attempts.values()):
                    check = local
            # Keep an actual observation made after this release when later occluded.
            # It proves the historical landing only; a new contrary view supersedes it.
            wrist = attempt.get('wristSeatCheck')
            if wrist and wrist['observedAt'] > max(attempt['seatedAt'], attempt.get('transportedAt', 0)) and (check.get('observedAt') or -1) <= wrist['observedAt']:
                check = dict(wrist)
            saved = attempt.get('seatCheck', {})
            frame = attempt.get('componentFrame')
            current_frame = getattr(self.rt.estimator, 'released_interfaces', {}).get(attempt['part'])
            same_release = (frame is None or current_frame is not None and
                            current_frame['releasedAt'] == frame['releasedAt'])
            from steward.ego.fasten_landing import ENABLED as landing_enabled
            if (check['status'] == 'unknown' and (saved.get('passed') is True or
                    landing_enabled and saved.get('passed') is False) and same_release
                    and saved.get('observedAt', -1) > attempt['seatedAt']
                    and not attempt.get('transportedAt')):
                check = dict(saved, retainedObservation=True)
            view = value(scene.find(attempt['part']), 'contactView')
            if (frame and view and view.get('source') == 'rgbd_segment' and
                    frame['releasedAt'] < view['stamp'] <= scene.stamp and scene.stamp-view['stamp'] <= 20
                    and not view.get('truncated', True) and view.get('invalidDepthFraction', 1) <= .05):
                size = np.asarray(view.get('visibleSizeM', [0,0,0]))
                expected = np.asarray(frame['size'])
                # Only a separate, plausible component view can contradict FK.
                # A union envelope cannot shift the target or verify placement.
                separate = np.all(size > expected*.4) and np.all(size < expected*1.6+.01)
                if separate and np.linalg.norm(np.asarray(view['centreM'])-frame['position']) > .035:
                    check.update(status='failed', passed=False, reason='component_displaced_after_release',
                                 componentObservedAt=view['stamp'])
            if check['passed'] and any(a is not attempt and a.get('seatConfirmed') and
                    a['screw'] == attempt['screw'] for a in self.rt.estimator.fastening_attempts.values()):
                check.update(status='failed', passed=False, reason='screw_already_confirmed_in_another_hole')
            from steward.ego.release_observation import ENABLED as release_view
            if (release_view and attempt.get('releaseObservation') and check.get('passed') is True
                    and not check.get('freshEndpointObserved') and not check.get('axisObserved')
                    and check.get('source') != 'post_release_hole_rgbd' and not object_check):
                check = dict(check, status='unknown', passed=None, reason='head_visible_axis_unresolved')
            from steward.ego.assembly_precision import head_check, tolerance
            if check.get('source') != 'post_release_hole_rgbd' and not object_check:
                check = head_check(attempt, check, tolerance(self), scene)
            attempt.setdefault('landingErrors', []).append(
                landing_error(screw, attempt['entry'], scene.stamp, attempt['seatedAt']))
            attempt.update(seatCheck=check, seatConfirmed=check['passed'] is True and not (attempt.get('bindingUnconfirmed') and not attempt.get('transportCommitted')),
                           seatStatus=check['status'])
            from steward.ego.hole_seating import deferred
            attempt['seatDeferred'] = deferred(attempt)
            if not (attempt.get('verified') and check['passed']):
                attempt['status'] = 'seat_'+check['status']
            return check
    else:
        def seat_posterior(self, attempt):
            from steward.ego.fasten_release import seat_check
            from steward.ego.fasten_simple import landing_error
            scene = self.rt.ego_state()
            screw = scene.find(attempt['screw'])
            check = seat_check(screw, attempt.get('componentFrame'), attempt['entry'],
                               attempt['seatedAt'], scene.stamp)
            frame = attempt.get('componentFrame')
            view = value(scene.find(attempt['part']), 'contactView')
            if (frame and view and view.get('source') == 'rgbd_segment' and
                    frame['releasedAt'] < view['stamp'] <= scene.stamp and scene.stamp-view['stamp'] <= 20
                    and not view.get('truncated', True) and view.get('invalidDepthFraction', 1) <= .05):
                size = np.asarray(view.get('visibleSizeM', [0,0,0]))
                expected = np.asarray(frame['size'])
                # Only a separate, plausible component view can contradict FK.
                # A union envelope cannot shift the target or verify placement.
                separate = np.all(size > expected*.4) and np.all(size < expected*1.6+.01)
                if separate and np.linalg.norm(np.asarray(view['centreM'])-frame['position']) > .035:
                    check.update(status='failed', passed=False, reason='component_displaced_after_release',
                                 componentObservedAt=view['stamp'])
            if check['passed'] and any(a is not attempt and a.get('seatConfirmed') and
                    a['screw'] == attempt['screw'] for a in self.rt.estimator.fastening_attempts.values()):
                check.update(status='failed', passed=False, reason='screw_already_confirmed_in_another_hole')
            attempt.setdefault('landingErrors', []).append(
                landing_error(screw, attempt['entry'], scene.stamp, attempt['seatedAt']))
            attempt.update(seatCheck=check, seatConfirmed=check['passed'] is True,
                           seatStatus=check['status'])
            if not (attempt.get('verified') and check['passed']):
                attempt['status'] = 'seat_'+check['status']
            return check

    if _assembly_enabled():
        from steward.ego.key_observation import observed_seat

    if _assembly_enabled():
        @timed("confirmation")
        @observed_seat
        def confirm_seat(self, job, attempt, plan, *, observe=True):
            """Replay only the same positively located departed screw; unknown spends zero."""
            from steward.ego import grasp_binding
            if grasp_binding.enabled() and attempt.get('bindingUnconfirmed') and not attempt.get('transportCommitted'):
                # The ordinary release already executes and acquires its result.
                # A failed seating label does not locate the unbound load, so it
                # cannot authorize pulling a possibly successful fastener back out.
                if not attempt.get('holeSeatObservation'):
                    self.wait_view(attempt['screw'], attempt['seatedAt'])
                self.seat_posterior(attempt)
                attempt.update(seatConfirmed=False, verified=False)
                job.postcondition.update(deepcopy(attempt), executed=True,
                    confirmation='unconfirmed', canContinue=True, resultChecked=True)
                return
            from steward.ego import released_object, axial_release
            if axial_release.enabled():
                self.check()
            from steward.ego.fasten_plan import seat_target
            if axial_release.handled(attempt):
                job.postcondition.update(deepcopy(attempt), executed=True, repeated=True)
                return
            from steward.ego.hole_seating import confirmed
            from steward.ego.assembly_session import current
            session = current()
            if confirmed(attempt):
                axial_release.record(session, attempt)
                if axial_release.enabled(): attempt['axialReleaseHandled'] = attempt.get('seatedAt')
                job.postcondition.update(deepcopy(attempt), executed=True, repeated=True)
                return
            from steward.ego.release_conditions import repair_now
            if not axial_release.enabled() and session is not None and not session.state.get('reviewingReleases') and not repair_now(attempt):
                # The release ledger reserves the hole. Review only after every
                # interface has had its first pass; preserve contrary evidence.
                from steward.ego import assembly_outcome
                if assembly_outcome.enabled() and attempt.get('holeSeatObservation'):
                    direct = attempt['holeSeatObservation'].get('check', {})
                    if direct.get('observedAt', -1) > attempt.get('seatedAt', -1):
                        attempt.update(seatCheck=deepcopy(direct), seatConfirmed=direct.get('passed') is True)
                attempt['seatDeferred'] = True
                job.postcondition.update(deepcopy(attempt), executed=True, verificationDeferred=True)
                return
            if session is not None or axial_release.enabled():
                observe = False  # Native release view already owns this result.
            budget = getattr(self, '_seat_retry_budget', None)
            if budget is None:
                budget = self._seat_retry_budget = [2]
            if axial_release.enabled():
                budget = [1]  # Each physical object gets its own immediate recovery opportunity.
            if observe and not attempt.get('holeSeatObservation'):
                self.wait_view(attempt['screw'], attempt['seatedAt'])
            check = self.seat_posterior(attempt)
            if observe and check['status'] == 'unknown' and not attempt.get('holeSeatObservation'):
                self.close_seat_view(job, [attempt])
                check = self.seat_posterior(attempt)
            if axial_release.enabled():
                axial_release.record(session, attempt)
            elif session is not None:
                from steward.ego.released_object import recovery_evidence
                if check.get('passed') is not True and not recovery_evidence(attempt, check):
                    self.close_seat_view(job, [attempt])
                    check = self.seat_posterior(attempt)
            else:
                from steward.ego.fasten_completion import observe_unknown
                check = observe_unknown(self, job, attempt, check)
            from steward.ego.released_object import recovery_evidence
            if (not axial_release.enabled() and session is None and check.get('passed') is False and attempt.get('holeSeatObservation')
                    and not recovery_evidence(attempt, check)):
                # An empty hole identifies a miss, not the identity of nearby stock.
                # Search using the existing independent identity reacquisition path.
                recovery = attempt.setdefault('landingRecovery', dict(status='searching_same_object'))
                if not recovery.get('searched'):
                    recovery['searched'] = True
                    self.close_seat_view(job, [attempt])
                    self.wait_view(attempt['screw'], attempt['seatedAt'])
                    recovery['status'] = 'searched_identity_unresolved'
            counts = self._seat_retry_counts = getattr(self, '_seat_retry_counts', {})
            key = (attempt['body'], attempt['hole'])
            from steward.ego import first_interface
            retry_limit = (max(1, attempt.get('firstInterfaceRetryBase', 1)) + 1
                if first_interface.enabled() and attempt.get('firstInterfaceRetryGranted') else 1)
            from steward.ego import seating_geometry
            if seating_geometry.enabled() and not first_interface.enabled():
                retry_limit = 1
            moved_component = first_interface.enabled() and bool(attempt.get('firstInterfaceNeedsReseat'))
            from steward.ego.fasten_landing import ENABLED as landing_enabled, replay_evidence
            while check['passed'] is not True and budget[0] > 0:
                if axial_release.enabled() and ((check.get('passed') is not False and not moved_component) or attempt.get('seatRetryCount', 0) >= retry_limit):
                    break
                if session is not None and attempt.get('seatRetryCount', 0) >= retry_limit:
                    break
                direct_history = attempt.get('holeSeatObservation') or any(
                    h.get('check', {}).get('source') == 'post_release_hole_rgbd' or
                    released_object.enabled() and h.get('check', {}).get('source') == released_object.SOURCE
                    for h in attempt.get('seatHistory', []))
                if direct_history and attempt.get('seatRetryCount', 0) >= retry_limit:
                    break
                if landing_enabled and attempt.get('seatRetryCount', 0) >= retry_limit:
                    break
                if (attempt.get('precision') or {}).get('observations') and attempt.get('seatRetryCount', 0) >= retry_limit:
                    break
                from steward.ego.fasten_accounting import departure, belongs, kit_support
                scene = self.rt.ego_state()
                evidence = replay_evidence(scene, attempt, check)
                if evidence is None and axial_release.enabled() and (check.get('passed') is False or moved_component):
                    # The bound original remains the only permissible pickup. Use its
                    # retained estimate after an inconclusive identity view; no new stock.
                    if scene.find(attempt['screw']) is not None and (not attempt.get('transportedAt') or moved_component):
                        evidence = dict(screw=attempt['screw'], observedAt=check.get('observedAt'),
                            source='same_identity_best_estimate', unconfirmed=True, check=deepcopy(check))
                if evidence is None:
                    break
                support = kit_support(self, support_anchor(self.rt.esim, scene, scene.entity(attempt['body']), []).id)
                screw = scene.find(evidence['screw'])
                if screw is None or not belongs(self, screw, support):
                    break
                attempt['departureEvidence'] = evidence
                if attempt.get('landingRecovery'):
                    attempt['landingRecovery'].update(status='same_object_located', evidence=deepcopy(evidence))
                if landing_enabled and check.get('passed') is None and evidence.get('source') != 'independent_original_identity_reacquisition':
                    check = dict(status='failed', passed=False, reason='independent_same_screw_elsewhere',
                                 observedAt=evidence['observedAt'])
                    attempt.update(seatCheck=check, seatConfirmed=False)
                peers = [a for a in self.rt.estimator.fastening_attempts.values()
                         if a is not attempt and a['body'] == attempt['body'] and a['part'] == attempt['part']
                         and not a.get('seatConfirmed') and not counts.get((a['body'], a['hole']))]
                if counts.get(key) and budget[0] <= len(peers):
                    break
                self.check()
                budget[0] -= 1
                counts[key] = counts.get(key, 0)+1
                attempt['seatRetryCount'] = attempt.get('seatRetryCount', 0)+1
                if (session is not None and (axial_release.enabled() or
                        released_object.enabled() and evidence.get('source') == released_object.SOURCE)
                        and (not axial_release.enabled() or hasattr(session, 'emit'))
                        and not session.state.get('reviewingReleases')):
                    if not axial_release.enabled():
                        session.emit('outcome_checked', interface=session.state.get('interface'),
                        body=attempt['body'], hole=attempt['hole'], screw=attempt['screw'], phase='after_release',
                        outcome='observed_not_seated', seatedAt=attempt['seatedAt'],check=deepcopy(check))
                    session.emit('outcome_reseat', interface=session.state.get('interface'),
                        hole=attempt['hole'], screw=attempt['screw'], phase='after_release',
                        check=deepcopy(check), recoveryTarget=deepcopy(evidence.get('recoveryTarget')))
                attempt.setdefault('seatHistory', []).append(dict(
                    screw=attempt['screw'], seatedAt=attempt['seatedAt'], check=deepcopy(check)))
                try:
                    # Positive identity and departure were established above; never substitute supply.
                    if plan.get('method', 'simple') != 'contact' and not axial_release.enabled():
                        from steward.ego.registered_alignment import register
                        register(self, job, attempt)
                    self.step(job, 'pick', dict(target=screw.id,**({'stayNear':True, 'axialFastener':True} if axial_release.enabled() else {}),**({'expectedFastener':True} if sr.role_enabled() or sr.size_enabled() else {})))
                    if getattr(self, '_seat_only', False) and getattr(self, '_lower_interface', False):
                        self.seat_grasp_check(job, screw.id, support, replay_identity=screw.id)
                    else:
                        self.held_kind_check(job, support, replay_identity=screw.id)  # held_fastener_kind: ordinary path checks only the identified category
                    held = self.rt.estimator.held_id or screw.id
                    if plan.get('method', 'simple') == 'contact':
                        self.step(job, '_fastener_contact', dict(plan, screw=held, operation='seat'))
                        replacement = self.rt.estimator.fastening_attempts[(attempt['body'], attempt['hole'])]
                        history = {k:attempt[k] for k in ('seatRetryCount', 'seatHistory')}
                        attempt.update(replacement, **history)
                        self.rt.estimator.fastening_attempts[(attempt['body'], attempt['hole'])] = attempt
                    else:
                        from steward.ego.assembly_precision import acquired
                        from steward.ego.registered_alignment import refresh_if_moved
                        acquired(self, job, held)
                        if not axial_release.enabled():
                            refresh_if_moved(self, job, attempt)
                        target = seat_target(self.rt.esim, self.rt.ego_state(), attempt)
                        from steward.ego.fasten_release import local_head_baseline
                        baseline = local_head_baseline(self.rt.ego_state(), target['point'])
                        params = dict(on=attempt['part'], alignHoles=False, clearance=0.,
                            **({'stayNear':True} if axial_release.enabled() else {}),
                            seatBody=attempt['body'], seatHole=attempt['hole'],
                            **({'closeRelease':True} if plan.get('closeRelease') else {}))
                        if landing_enabled:
                            params['boundedSeatRelease'] = True
                        receipt = None
                        if getattr(self, '_seat_only', False) and getattr(self, '_lower_interface', False):
                            params['lowerSeatSurface'] = True
                            from steward.ego.fasten_seat import replay_release
                            params, receipt = replay_release(self, attempt, check, params)
                        from steward.ego.wrist_landing import prepare as prepare_landing
                        prepare_landing(self, attempt, replay=True)
                        placed = self.step(job, 'place', params, receipt=receipt)
                        if moved_component:
                            attempt.pop('firstInterfaceNeedsReseat', None)
                            moved_component = False
                        target = placed.get('seatTarget', target)
                        recorded = self.rt.estimator.fastening_attempts.get(key, {})
                        attempt.update(releaseHistory=deepcopy(recorded.get('releaseHistory', [])), departureEvidence=None)
                        self.rt.estimator.fastening_attempts[key] = attempt
                        attempt.update(screw=held, entry=list(target['point']),
                            seatedAt=placed.get('releasedAt', self.ex._state()['stamp']),
                            commandedSpotM=placed['spot'], componentFrame=deepcopy(target.get('componentFrame')),
                            pairAlignment=deepcopy(target.get('pairAlignment')),
                            **({'surfaceEvidence':deepcopy(target['surfaceEvidence'])} if target.get('surfaceEvidence') else {}),
                            verified=False, seatConfirmed=False, headPatchBefore=baseline,
                            driveStatus='not_driven', driveStrokeExecuted=False, driveCheck=None,
                            releaseToolFeature=deepcopy(placed.get('releaseToolFeature')),
                            seatMotion=deepcopy(placed.get('seatMotion')), releaseGapM=placed.get('releaseGapM'),
                            precision=placed.get('precision'))
                    from steward.ego.hole_seating import apply as apply_hole
                    attempt.pop('holeSeatObservation', None)
                    attempt.pop('seatDeferred', None)
                    attempt.pop('landingRecovery', None)
                    apply_hole(attempt, locals().get('placed'))
                    from steward.ego.release_observation import ENABLED as release_view, observe as observe_release
                    if axial_release.enabled():
                        axial_release.result_view(self, job, attempt, locals().get('placed'))
                    elif release_view:
                        observe_release(self, job, attempt, locals().get('placed'))
                    else:
                        from steward.ego import wrist_observation as wo
                        if wo.release_reporting():
                            from steward.ego.wrist_fasten import released
                            released(self,job,attempt,locals().get('placed'))
                        self.wait_view(attempt['screw'], attempt['seatedAt'])
                    check = self.seat_posterior(attempt)
                    axial_release.record(session, attempt)
                except SkillFailed as exc:
                    self.check()
                    if getattr(self, '_seat_only', False) and getattr(self, '_lower_interface', False) and 'grasp missed' in str(exc) and screw is not None:
                        self._screw_misses[screw.id] = self.ex._state()['stamp']
                    attempt['seatReplayFailure'] = str(exc)
                    self.warn('seat_replay_failed', detail=str(exc), hole=attempt['hole'])
                    if self.rt.estimator.held_id:
                        self.park(job, support)
            if axial_release.enabled():
                attempt['axialReleaseHandled'] = attempt.get('seatedAt')
            if attempt.get('completionRetry'):
                attempt['completionRetry']['replays'] = attempt.get('seatRetryCount', 0)
            job.postcondition.update(deepcopy(attempt), executed=True,
                confirmedSeatedScrews=int(attempt.get('seatConfirmed', False)))
            if check['status'] == 'failed':
                if axial_release.enabled() or seating_geometry.enabled():
                    job.postcondition.update(confirmation='unverified', canContinue=True, actionStatus='executed_unconfirmed')
                    return
                raise SkillFailed('screw landing contradicted by observation after bounded recovery')
            if check['status'] == 'unknown':
                if seating_geometry.enabled():
                    job.postcondition.update(confirmation='unverified', canContinue=True,
                        actionStatus='executed_unconfirmed')
                    return
                from steward.ego.fasten_completion import REPORT
                if REPORT:
                    job.postcondition.update(confirmation='unverified', canContinue=bool(attempt.get('seatDeferred')), actionStatus='unconfirmed' if attempt.get('seatDeferred') else 'incomplete',
                        guidance='该孔已释放但入座未知；观察预算已用完，不可向可能占用的孔补另一颗。')
                    return
                job.postcondition.update(confirmation='unverified', canContinue=True,
                    actionStatus='executed', guidance='动作已执行、未能目视确认，可继续下一步')
                self.warn('landing_unverified_continue', hole=attempt['hole'],
                          detail='Action executed; visual confirmation unavailable; continue the next task step.')
    else:
        def confirm_seat(self, job, attempt, plan, *, observe=True):
            """Two replays per job, after action and independent perception only."""
            from steward.ego.fasten_plan import seat_target
            budget = getattr(self, '_seat_retry_budget', None)
            if budget is None:
                budget = self._seat_retry_budget = [2]
            if observe:
                self.wait_view(attempt['screw'], attempt['seatedAt'])
            check = self.seat_posterior(attempt)
            if observe and check['status'] == 'unknown':
                self.close_seat_view(job, [attempt])
                check = self.seat_posterior(attempt)
            while check['passed'] is not True and budget[0] > 0:
                self.check()
                budget[0] -= 1
                attempt['seatRetryCount'] = attempt.get('seatRetryCount', 0)+1
                attempt.setdefault('seatHistory', []).append(dict(
                    screw=attempt['screw'], seatedAt=attempt['seatedAt'], check=deepcopy(check)))
                scene = self.rt.ego_state()
                support = support_anchor(self.rt.esim, scene, scene.entity(attempt['body']), []).id
                try:
                    # Only regrasp the old screw when an independent view locates it.
                    # A merged mask/predicted release point is not a safe screw identity.
                    screw = scene.find(attempt['screw']) if check.get('observedAt') is not None else None
                    screw = screw or self.acquire_tool(job, 'toy_screw', support, free_screw=True)
                    self.step(job, 'pick', dict(target=screw.id))
                    held = self.rt.estimator.held_id or screw.id
                    if plan.get('method', 'simple') == 'contact':
                        self.step(job, '_fastener_contact', dict(plan, screw=held, operation='seat'))
                        replacement = self.rt.estimator.fastening_attempts[(attempt['body'], attempt['hole'])]
                        history = {k:attempt[k] for k in ('seatRetryCount', 'seatHistory')}
                        attempt.update(replacement, **history)
                        self.rt.estimator.fastening_attempts[(attempt['body'], attempt['hole'])] = attempt
                    else:
                        target = seat_target(self.rt.esim, self.rt.ego_state(), attempt)
                        placed = self.step(job, 'place', dict(on=attempt['part'], alignHoles=False,
                            clearance=0., seatBody=attempt['body'], seatHole=attempt['hole'],
                            **({'closeRelease':True} if plan.get('closeRelease') else {})))
                        target = placed.get('seatTarget', target)
                        attempt.update(screw=held, entry=list(target['point']),
                            seatedAt=placed.get('releasedAt', self.ex._state()['stamp']),
                            commandedSpotM=placed['spot'], componentFrame=deepcopy(target.get('componentFrame')),
                            verified=False, seatConfirmed=False)
                    self.wait_view(attempt['screw'], attempt['seatedAt'])
                    check = self.seat_posterior(attempt)
                except SkillFailed as exc:
                    self.check()
                    self.warn('seat_replay_failed', detail=str(exc), hole=attempt['hole'])
                    if self.rt.estimator.held_id:
                        self.park(job, support)
            job.postcondition.update(deepcopy(attempt), executed=True,
                confirmedSeatedScrews=int(attempt.get('seatConfirmed', False)))
            if not attempt.get('seatConfirmed'):
                raise SkillFailed('screw landing '+check['status']+' after posterior replay budget')

    if _assembly_enabled():
        @timed("release_observation_trip")
        def close_seat_view(self, job, attempts):
            """One short camera view aimed at the actual interface, after releases."""
            scene = self.rt.ego_state()
            table = support_anchor(self.rt.esim, scene, scene.entity(attempts[0]['body']), [])
            box,_,_ = entity_box(table); o = self.ex._state()['odometry']
            old_close = getattr(self, '_close_view', False)
            old_target = getattr(self, 'observation_target', None)
            self._close_view = True
            self.observation_target = np.mean([a['entry'] for a in attempts], axis=0).tolist()
            try:
                since = self.view(job, table, math.atan2(o['y']-box.cy,o['x']-box.cx), 0.)
                self.look(self.ex._state()['stamp'] if since is None else since, cycles=1)
            except SkillFailed as exc:
                self.check(); self.warn('seat_posterior_view_unavailable', detail=str(exc))
            finally:
                self._close_view = old_close
                if old_target is None:
                    del self.observation_target
                else:
                    self.observation_target = old_target
            for a in attempts:
                self.wait_view(a['screw'], a['seatedAt'], timeout=0)
    else:
        def close_seat_view(self, job, attempts):
            """One short camera view aimed at the actual interface, after releases."""
            scene = self.rt.ego_state()
            table = support_anchor(self.rt.esim, scene, scene.entity(attempts[0]['body']), [])
            box,_,_ = entity_box(table); o = self.ex._state()['odometry']
            old_close = getattr(self, '_close_view', False)
            old_target = getattr(self, 'observation_target', None)
            self._close_view = True
            self.observation_target = np.mean([a['entry'] for a in attempts], axis=0).tolist()
            try:
                since = self.view(job, table, math.atan2(o['y']-box.cy,o['x']-box.cx), 0.)
                self.look(self.ex._state()['stamp'] if since is None else since, cycles=1)
            except SkillFailed as exc:
                self.check(); self.warn('seat_posterior_view_unavailable', detail=str(exc))
            finally:
                self._close_view = old_close
                if old_target is None:
                    del self.observation_target
                else:
                    self.observation_target = old_target
            for a in attempts:
                self.wait_view(a['screw'], a['seatedAt'], timeout=0)

    def reach_retry(self, job, skill, params, support):
        """One changed base station after an unreachable screw pick or placement."""
        try:
            return self.step(job, skill, params)
        except SkillFailed as exc:
            self.check()
            if (getattr(self, '_stance_retry_used', False) or
                    not any(word in str(exc) for word in ('out of range', 'no feasible approach', 'unreachable')) or
                    skill == 'place' and not self.rt.estimator.held_id):
                raise
            self._stance_retry_used = True
            self.warn('screw_reach_retry_station', skill=skill, detail=str(exc))
            table = self.rt.ego_state().entity(support)
            box, _, _ = entity_box(table)
            o = self.ex._state()['odometry']
            origin = math.atan2(o['y']-box.cy, o['x']-box.cx)
            self.view(job, table, origin, .65)
            return self.step(job, skill, params)

    def regrasp_upright(self, job, plan):
        """Place an elongated load vertically, then take it with a top-down grasp.

        All placement rotations use the model's FK and continuous IK path. A
        quarter roll is one solution; a tool-axis wrist can require pitch too.
        """
        from steward.ego.slender_tool import axial_feature
        support = plan['support']
        target = plan.get('target') or self.rt.estimator.held_id
        if not self.rt.estimator.held_id:
            scene = self.rt.ego_state()
            with self.rt._lock:
                track = self.rt.estimator.tracks[target]
                track.tool_feature = track.tool_feature or axial_feature(scene.entity(target), scene.stamp)
            self.step(job, 'pick', dict(target=target))
        if _assembly_enabled():
            for attempt in range(1 if plan.get('fastenerPosture') else 2):
                scene = self.rt.ego_state()
                held = scene.entity(self.rt.estimator.held_id)
                feature = axial_feature(held, scene.stamp)
                if feature is None:
                    raise SkillFailed('no elongated geometry for stand-and-regrasp')
                # Search with the standing footprint, not the horizontal tool box.
                from steward.ego.fasten_driver import support_slots
                table = scene.entity(support)
                candidates = support_slots(self.rt.esim, scene, table, feature['radiusM']+.015)
                if plan.get('fastenerPosture'):
                    origin = self.rt.estimator.fastener_origins.get(held.id, {}).get('xy')
                    origin = origin if origin is not None else self.ex._gripper_xy()
                    candidates = [list(origin), *sorted(candidates, key=lambda xy:np.linalg.norm(np.asarray(xy)-origin))]
                upright = None
                for xy in candidates:
                    params = dict(stageOn=support, stageXY=xy, operation='stand')
                    upright = self.rt.predict('_upright_tool', params)
                    if upright.get('withinRiskLimit'):
                        break
                if upright and upright.get('withinRiskLimit'):
                    break
                issues = (upright or {}).get('issues', ['no free upright footprint'])
                if attempt or plan.get('fastenerPosture'):
                    raise SkillFailed('upright regrasp: '+str(issues))
                self.warn('upright_change_grasp', detail=issues)
                old = held.id
                if not self.park_driver(job, support):
                    raise SkillFailed('upright regrasp could not park the held tool')
                self.step(job, 'pick', dict(target=old, preparePowered=True))
        else:
            for attempt in range(2):
                scene = self.rt.ego_state()
                held = scene.entity(self.rt.estimator.held_id)
                feature = axial_feature(held, scene.stamp)
                if feature is None:
                    raise SkillFailed('no elongated geometry for stand-and-regrasp')
                # Ordinary placement supplies a free support-relative location.
                slot = self.rt.predict('place', dict(on=support, alignHoles=False))
                if not slot.get('withinRiskLimit'):
                    raise SkillFailed('no reachable stand-and-regrasp placement: '+str(slot.get('issues')))
                params = dict(stageOn=support, stageXY=slot['plan']['spot'], operation='stand')
                upright = self.rt.predict('_upright_tool', params)
                if upright.get('withinRiskLimit'):
                    break
                if attempt:
                    raise SkillFailed('upright regrasp: '+str(upright.get('issues')))
                self.warn('upright_change_grasp', detail=upright.get('issues'))
                old = held.id
                self.step(job, 'place', dict(on=support, alignHoles=False), receipt=slot)
                self.step(job, 'pick', dict(target=old, prepareUpright=True))
        eid = self.rt.estimator.held_id
        with self.rt._lock:
            snapshot = deepcopy(self.rt.estimator.tracks[eid])
        placed = self.step(job, '_upright_tool', params, receipt=upright)
        # Cache executed geometry without turning it into a fresh camera sample.
        tip = np.asarray(placed['goal'])
        head = tip + [0., 0., placed['toolLengthM']]
        if _assembly_enabled():
            with self.rt._lock:
                track = self.rt.estimator.tracks.get(eid, snapshot)
                observed = track.tool_feature if (track.tool_feature and
                    track.tool_feature.get('source') == 'rgbd_endpoint_width' and
                    track.tool_feature.get('stamp', 0) > job.postcondition.get('stagingReleaseAt', math.inf)) else None
                track.pos = (tip+head)/2 if observed is None else (np.asarray(observed['tip'])+observed['head'])/2
                if observed is None:
                    track.size = np.array([2*placed['toolRadiusM'], 2*placed['toolRadiusM'], placed['toolLengthM']])
                    track.yaw = 0.
                track.tool_feature = observed or dict(tip=tip.tolist(), head=head.tolist(),
                    axis=[0.,0.,1. if plan.get('fastenerPosture') else -1.],
                    lengthM=placed['toolLengthM'], radiusM=placed['toolRadiusM'],
                    stamp=feature['stamp'], source='upright_release_estimate', polarityObserved=False)
                track.held_tool_feature = None
                self.rt.estimator.tracks[eid] = track
                self.rt.estimator.version += 1
        else:
            with self.rt._lock:
                track = self.rt.estimator.tracks.get(eid, snapshot)
                observed = track.tool_feature if (track.tool_feature and
                    track.tool_feature.get('source') == 'rgbd_endpoint_width' and
                    track.tool_feature.get('stamp', 0) > job.postcondition.get('stagingReleaseAt', math.inf)) else None
                track.pos = (tip+head)/2 if observed is None else (np.asarray(observed['tip'])+observed['head'])/2
                if observed is None:
                    track.size = np.array([2*placed['toolRadiusM'], 2*placed['toolRadiusM'], placed['toolLengthM']])
                    track.yaw = 0.
                track.tool_feature = observed or dict(tip=tip.tolist(), head=head.tolist(), axis=[0.,0.,-1.],
                    lengthM=placed['toolLengthM'], radiusM=placed['toolRadiusM'],
                    stamp=feature['stamp'], source='upright_release_estimate', polarityObserved=False)
                track.held_tool_feature = None
                self.rt.estimator.tracks[eid] = track
                self.rt.estimator.version += 1
        self.step(job, 'pick', dict(target=eid, topGrasp=True))
        job.postcondition['uprightRegrasp'] = dict(entity=self.rt.estimator.held_id,
            executed=True, verified=False, source='executed_release_and_encoder_grasp')

    if _assembly_enabled():
        @driver_batch.timed("acquire")
        def pick_driver(self, job, support):
            """Try, observe the miss, then refresh and retry once; never repeat a stale target."""
            last = None
            for retry in range(2):
                try:
                    driver = self.acquire_tool(job, 'electric_screwdriver', support)
                    target = driver.id
                    params = dict(target=target, preparePowered=True)
                    receipt = self.rt.predict('pick', params)
                    if not receipt.get('withinRiskLimit'):
                        self.warn('driver_prepared_grasp_unavailable', detail=receipt.get('issues'))
                        params = dict(target=target)
                        receipt = self.rt.predict('pick', params)
                    self.step(job, 'pick', params, receipt=receipt)
                    if not self.rt.estimator.held_id:
                        raise SkillFailed('driver grasp returned without a held load')
                    return self.rt.estimator.held_id
                except SkillFailed as exc:
                    self.check()
                    last = exc
                    self.warn('driver_pick_failed', attempt=retry+1, detail=str(exc))
                    if self.rt.estimator.held_id or retry:
                        raise
                    if 'target' in locals():
                        self._driver_misses = getattr(self, '_driver_misses', {})
                        self._driver_misses[target] = self.rt.estimator.stamp
                        for origin in self.rt.estimator.fastener_origins.values():
                            row = origin.get('driverRelease', {})
                            if row.get('alias') == target:
                                row['missedAt'] = self.rt.estimator.stamp
                    # One changed view before trying the current candidates again.
                    scene = self.rt.ego_state(); table = scene.entity(support)
                    box, _, _ = entity_box(table); o = self.ex._state()['odometry']
                    since = self.view(job, table, math.atan2(o['y']-box.cy, o['x']-box.cx), .65)
                    self.look(self.ex._state()['stamp'] if since is None else since, cycles=1)
                    # A missed stale alias is ineligible until seen again. The choice
                    # filter does not hide fresh new ids for the same physical tool.
            raise SkillFailed(str(last))

    def simple_drive(self, job, plan):
        if _assembly_enabled():
            from steward.ego.fasten_driver import head_target, drive_check, observe_head
            pairs = [a for a in self.rt.estimator.fastening_attempts.values()
                     if a['screw'] == plan['screw'] and (not plan.get('body') or a['body'] == plan['body'])
                     and (not plan.get('hole') or a['hole'] == plan['hole'])]
        else:
            pairs = [a for a in self.rt.estimator.fastening_attempts.values() if a['screw'] == plan['screw']]
        attempt = pairs[0] if pairs else deepcopy(plan.get('driveContext') or drive_context(self.rt.esim, self.rt.ego_state(), plan))
        if _assembly_enabled():
            if attempt.get('driveStatus') == 'driven':
                job.postcondition.update(deepcopy(attempt), repeated=True, executed=True)
                return
            if not pairs:
                self.warn('seating_record_estimated', screw=plan['screw'])
            self.rt.estimator.fastening_attempts[(attempt['body'], attempt['hole'])] = attempt
        else:
            if not pairs:
                self.warn('seating_record_estimated', screw=plan['screw'])
                self.rt.estimator.fastening_attempts[(attempt['body'],attempt['hole'])] = attempt
        self.body_id = attempt['body']
        scene = self.rt.ego_state()
        support = support_anchor(self.rt.esim, scene, scene.entity(self.body_id), []).id
        if _assembly_enabled():
            attempt.setdefault('driveHistory', []).append({k:deepcopy(attempt[k]) for k in
                ('driveStatus', 'driveStage', 'driveCheck', 'drivenAt', 'driveStrokeExecuted',
                 'driverMotion', 'driveFailure') if k in attempt})
            previous_status = attempt.get('driveStatus')
            for key in ('driverMotion', 'driverServoCorrections', 'driveFailure', 'driverParkFailure', 'drivenAt'):
                attempt.pop(key, None)
            attempt.update(driveStatus='unknown' if previous_status == 'unknown' else 'not_driven',
                           driveStage='acquire', driveStrokeExecuted=False, driveCheck=None)
            failure = None
            batch = getattr(self,'_driver_batch',None) if driver_batch.enabled() else None
            try:
                held=self.rt.estimator.held_id
                if not (driver_batch.enabled() and plan.get('reuseHeldTool') and held):
                    self.pick_driver(job, support)
                    if batch is not None:
                        batch['tool']=self.rt.estimator.held_id
                        batch['acquisitions']+=1
                elif batch is not None and (held != batch['tool'] or not batch['clear']):
                    raise SkillFailed('retained driver identity or withdrawal clearance changed')
                from steward.ego import driver_pose
                driver_pose.check(self, job, 'acquired' if not plan.get('reuseHeldTool') else 'retained')
                with driver_batch.phase(self,job,'head_observation'):
                    before = observe_head(self, job, attempt)
                attempt['driveStage'] = 'orient'
                # Refresh local posterior after the trip to acquire the driver.
                self.seat_posterior(attempt)
                if attempt.get('seatCheck',{}).get('passed') is not True:
                    raise SkillFailed('seating_unconfirmed_before_drive')
                from steward.ego.fasten_sequence import require_attempt
                require_attempt(self, job, plan, attempt)
                refreshed = head_target(attempt, self.rt.estimator.stamp)
                if (not before.get('observed') or refreshed.get('observed')
                        and refreshed['stamp'] >= before['stamp']):
                    before = refreshed
                attempt['driveTarget'] = deepcopy(before)
                params = dict(body=self.body_id, head=before['head'], hole=attempt['hole'],
                              screw=attempt['screw'], targetSource=before['source'],
                              targetObservedAt=before.get('stamp'),targetObserved=before.get('observed',False))
                with driver_batch.phase(self,job,'stroke_planning'):
                    receipt = self.rt.predict('_simple_driver', params)
                if not receipt.get('withinRiskLimit') and any('transverse regrasp' in s for s in receipt.get('issues', [])):
                    self.warn('driver_transverse_regrasp', detail=receipt['issues'])
                    self.regrasp_upright(job, dict(target=self.rt.estimator.held_id, support=support))
                    driver_pose.check(self, job, 'reacquired')
                    receipt = self.rt.predict('_simple_driver', params)
                attempt['driveStage'] = 'stroke'
                started = self.ex._state()['stamp']
                # Once a stroke starts, a later failure may follow contact: unknown.
                if receipt.get('withinRiskLimit'):
                    attempt['driveStatus'] = 'unknown'
                for pose_try in range(2):
                    try:
                        self.step(job, '_simple_driver', params, receipt=receipt)
                        break
                    except driver_pose.PoseMismatch:
                        self.check()
                        if pose_try or job.postcondition.get('driverClear') is not True:
                            raise
                        self.regrasp_upright(job, dict(target=self.rt.estimator.held_id, support=support))
                        driver_pose.check(self, job, 'reacquired')
                        receipt = self.rt.predict('_simple_driver', params)
                after = self.ex._state()['stamp']
                attempt.update(drivenAt=after, driveStrokeExecuted=True, driveStage='posterior', method='simple')
            except Exception as exc:
                failure = exc
                attempt['driveFailure'] = str(exc)
            finally:
                # Cancellation/session loss propagates before further motion.
                self.check()
                try:
                    retained = bool(batch is not None and failure is None and
                        job.postcondition.get('driverClear') is True and self.rt.estimator.held_id==batch['tool'])
                    if batch is not None:
                        batch['clear']=job.postcondition.get('driverClear') is not False
                        batch['retainedHoles']+=int(retained)
                    parked = (False if retained else self.park_driver(job, support)
                              if job.postcondition.get('driverClear') is not False else False)
                    if retained: attempt['driverRetained']=True
                except SkillFailed as exc:
                    self.check()
                    parked = False
                    attempt['driverParkFailure'] = str(exc)
                    failure = failure or exc
                attempt['driverParked'] = parked
                for key in ('driverMotion', 'driverServoCorrections'):
                    if key in job.postcondition:
                        attempt[key] = deepcopy(job.postcondition[key])
                job.postcondition.update(deepcopy(attempt))
            if failure is not None:
                raise failure
        else:
            driver = self.acquire_tool(job, 'electric_screwdriver', support)
            before = deepcopy(self.feature(attempt['screw']))
            self.step(job, 'pick', dict(target=driver.id))
            params = dict(body=self.body_id, head=before['head'])
            receipt = self.rt.predict('_simple_driver', params)
            if not receipt.get('feasible', False) and any('transverse regrasp' in s for s in receipt.get('issues', [])):
                self.warn('driver_transverse_regrasp', detail=receipt['issues'])
                self.regrasp_upright(job, dict(target=self.rt.estimator.held_id, support=support))
                receipt = self.rt.predict('_simple_driver', params)
            try:
                self.step(job, '_simple_driver', params, receipt=receipt)
                after = self.ex._state()['stamp']
                attempt.update(drivenAt=after, status='driven_unverified', method='simple')
            finally:
                self.check()
                self.park(job, support)
        view = self.wait_view(attempt['screw'], after)
        observed = screw_head(view, before)
        if _assembly_enabled():
            self.seat_posterior(attempt)
            local = head_target(attempt, self.rt.estimator.stamp)
            if observed is None and local.get('observed') and local['stamp'] > after:
                observed = local
            evidence = drive_check(before, observed, attempt['entry'], started)
            attempt.update(driveStatus=evidence['status'], driveCheck=evidence,
                           verified=attempt.get('verified', False) or evidence['status'] == 'driven',
                           driveStage='complete', status='head_sink_observed' if evidence['status'] == 'driven' else 'attachment_unknown')
        else:
            verified = head_sank(before, observed, attempt['entry'])
            attempt.update(verified=verified, status='head_sink_observed' if verified else 'attachment_unknown',
                           evidence=dict(kind='head_sink', before=before, after=observed))
        job.postcondition.update(deepcopy(attempt), executed=True)
        if _assembly_enabled():
            if not parked and not retained:
                raise SkillFailed('driver parking incomplete; park held tool before resuming')

    if _assembly_enabled():
        @driver_batch.timed("park")
        def park_driver(self, job, support):
            """Use free support space away from the installed interface; no floor fallback."""
            if not self.rt.estimator.held_id:
                return True
            scene = self.rt.ego_state()
            from steward.ego.fasten_driver import support_slots
            from steward.ego.fasten_simple import offset_on
            held = scene.find(self.rt.estimator.held_id)
            feature = tool_estimate(held, scene.stamp) if held else None
            if held is None:
                self.warn('driver_parking_unavailable', reason='held_geometry_missing')
                return False
            box, low, high = entity_box(held)
            radius = (feature or {}).get('lengthM', max(2*box.hx, 2*box.hy, high-low))/2+.025
            tables = [e for e in scene.entities if self.rt.esim._affords(e, 'support-surface')]
            tables.sort(key=lambda e:e.id != support)
            for table in tables:
                for xy in support_slots(self.rt.esim, scene, table, radius):
                    if any(np.linalg.norm(np.asarray(xy)-np.asarray(a['entry'])[:2]) < radius+.06
                           for a in self.rt.estimator.fastening_attempts.values()):
                        continue
                    params = dict(on=table.id, offset=offset_on(table, xy), alignHoles=False)
                    receipt = self.rt.predict('place', params)
                    if not receipt.get('withinRiskLimit'):
                        continue
                    spot = receipt['plan']['spot']
                    if any(np.linalg.norm(np.asarray(spot)-np.asarray(a['entry'])[:2]) < radius+.06
                           for a in self.rt.estimator.fastening_attempts.values()):
                        continue
                    try:
                        self.step(job, 'place', params, receipt=receipt)
                        if not self.rt.estimator.held_id:
                            return True
                    except SkillFailed as exc:
                        self.check(); self.warn('driver_parking_failed', detail=str(exc))
            self.warn('driver_parking_unavailable', held=self.rt.estimator.held_id)
            return False

    def simple_stroke(self, job, plan):
        if _assembly_enabled():
            job.postcondition['driverClear'] = False
            from steward.ego.fasten_driver import settle_joint, motion_sample, correct_tip_xy
            motion = job.postcondition['driverMotion'] = dict(
                stage='docking', goal=plan['goal'], targetSource=plan.get('targetSource'),
                pressureControl='position_only_no_force_sensor',
                tipGeometry=deepcopy(plan.get('driverTipGeometry')))
            with driver_batch.phase(self,job,'dock'):
                self.ex._dock(job, plan, retract=False)
            stages=driver_batch.Stages(self,job)
        else:
            self.ex._dock(job, plan, retract=False)
        self.check()
        if _assembly_enabled():
            failure = None
            try:
                stages.mark('clearance')
                motion['stage'] = 'clearance'
                from steward.ego.fasten_sequence import ENABLED as batch_drives
                if not batch_drives:
                    self.ex._set_joint('lift', plan['safeLift'])
                settle_joint(self, job, 'lift', plan['safeLift'], tol=.002)
                stages.mark('orient')
                motion['stage'] = 'orient'
                self.ex._set_joint('reach', plan['retractReach'])
                self.ex._set_joint('wrist_yaw', plan['wristYaw'], tol=.02)
                self.ex._set_joint('wrist_roll', plan['wristRoll'], tol=.02)
                if 'wrist_pitch' in self.ex._state()['joints']:
                    self.ex._set_joint('wrist_pitch', plan['wristPitch'], tol=.02)
                from steward.ego import driver_pose
                driver_pose.check(self, job, 'oriented', since=self.ex._state()['stamp'], require=True)
                stages.mark('align')
                self.ex._align_lateral(plan['gripSpot'], tol=.002)
                self.ex._set_joint('reach', self.ex._reach_to(plan['gripSpot']))
                settle_joint(self, job, 'reach', self.ex._reach_to(plan['gripSpot']), tol=.002)
                correct_tip_xy(self, job, plan, motion)
                motion.update(stage='aligned', aligned=motion_sample(self, plan))
                stages.mark('lower')
                from steward.ego import driver_contact
                driver_contact.lower(self, plan, motion)
                stages.mark('dwell')
                job.postcondition['driverMotion']['stage'] = 'dwell'
                motion['contact'] = motion_sample(self, plan)
                motion['holdSamples'] = [motion['contact']]
                start = self.ex._state()['stamp']
                while self.ex._state()['stamp']-start < plan.get('dwellS', .5):
                    self.ex._wait(.1)
                    self.check()
                    motion['holdSamples'].append(motion_sample(self, plan))
                motion['heldForS'] = self.ex._state()['stamp']-start
                from steward.ego.tip_entry_servo import driver_after_contact
                driver_after_contact(self, plan, motion)
                self.check()
            except Exception as exc:
                failure = exc
                motion['failureStage'] = motion['stage']
                motion['failureSample'] = motion_sample(self, plan)
                raise
            finally:
                stages.mark('withdraw')
                self.check()
                job.postcondition['driverClear'] = False
                try:
                    self.ex._set_joint('lift', plan['safeLift'])
                    settle_joint(self, job, 'lift', plan['safeLift'], tol=.002)
                    self.ex._set_joint('reach', plan['retractReach'])
                    for joint, value in plan['parkWrist'].items():
                        if joint in self.ex._state()['joints']:
                            self.ex._set_joint(joint, value, tol=.02)
                    motion['withdrawn'] = motion_sample(self, plan)
                    self.ex._backoff(plan)
                    job.postcondition['driverClear'] = True
                except SkillFailed as exc:
                    self.check()
                    self.warn('driver_withdrawal_failed', detail=str(exc))
                    if failure is None:
                        raise
                finally:
                    stages.mark()
        else:
            self.ex._set_joint('lift', plan['safeLift'])
            self.ex._set_joint('reach', plan['retractReach'])
            self.ex._set_joint('wrist_yaw', plan['wristYaw'], tol=.02)
            self.ex._set_joint('wrist_roll', plan['wristRoll'], tol=.02)
            if 'wrist_pitch' in self.ex._state()['joints']:
                self.ex._set_joint('wrist_pitch', plan['wristPitch'], tol=.02)
            self.ex._align_lateral(plan['gripSpot'], tol=.002)
            self.ex._set_joint('reach', self.ex._reach_to(plan['gripSpot']))
            self.ex._set_joint('lift', plan['lift'])
            self.ex._wait(.5)
            self.check()
            self.ex._set_joint('lift', plan['safeLift'])
            self.ex._set_joint('reach', plan['retractReach'])
            for joint, value in plan['parkWrist'].items():
                if joint in self.ex._state()['joints']:
                    self.ex._set_joint(joint, value, tol=.02)
            self.ex._backoff(plan)

    def contact_error(self, plan):
        feature = self.feature(plan['object'])
        position = float(np.linalg.norm(np.asarray(feature['tip'])-plan['goal']))
        angle = math.acos(float(np.clip(np.asarray(feature['axis']) @ plan['desiredAxis'], -1, 1)))
        return position, angle

    def contact(self, job, plan):
        if _assembly_enabled():
            try:
                return self._contact(job, plan)
            except SkillFailed:
                if plan['operation'] != 'seat':
                    try:
                        self.ex._set_joint('lift', plan['safeLift'])
                        self.ex._withdraw(plan)
                        job.postcondition['driverClear'] = True
                    except SkillFailed as exc:
                        job.postcondition['driverClear'] = False
                        job.postcondition['withdrawalFailure'] = str(exc)
                raise
        else:
            """Execute the exact checked approach; no unplanned IK fallback or target shifting."""
            contact_started = self.ex._state()['stamp']
            self.ex._dock(job, plan, retract=False)
            self.check()
            self.ex._set_joint('lift', plan['safeLift'])
            self.ex._set_joint('reach', plan['retractReach'])
            self.ex._set_joint('wrist_yaw', plan['wristYaw'], tol=.02)
            self.ex._set_joint('wrist_roll', plan['wristRoll'], tol=.02)
            if 'wrist_pitch' in self.ex._state()['joints']:
                self.ex._set_joint('wrist_pitch', plan['wristPitch'], tol=.02)
            self.ex._align_lateral(plan['gripSpot'], tol=.002)
            self.ex._set_joint('reach', self.ex._reach_to(plan['gripSpot']))
            from steward.ego.fasten_plan import seat_target
            from steward.ego.esim import PredictionError
            try:
                target = seat_target(self.rt.esim, self.rt.ego_state(), plan['contactParams'])
            except PredictionError as exc:
                self.warn('entry_recheck_unavailable', detail=str(exc))
                target = plan['targetEvidence']
            if np.linalg.norm(np.asarray(target['point'])-plan['targetEvidence']['point']) > .003:
                self.warn('entry_surface_shift', distanceM=float(np.linalg.norm(np.asarray(target['point'])-plan['targetEvidence']['point'])), toleranceM=.003)
            self.ex._set_joint('lift', plan['lift'])
            self.ex._set_joint('reach', self.ex._reach_to(plan['gripSpot']), ramp=False, tol=.0015)
            self.ex._set_joint('lift', plan['lift'], ramp=False, tol=.0015)
            self.ex._settle_for_release()
            position, angle = self.contact_error(plan)
            tolerance = .0025 if plan['operation'] == 'seat' else .008
            if position > tolerance or angle > math.radians(8):
                if self.feature(plan['object']).get('polarityObserved', True):
                    raise SkillFailed(f'contact missed: endpoint={position:.4f} m axis={math.degrees(angle):.1f} deg; keeping tool')
                self.warn('contact_estimate_error', distanceM=position, angleDeg=math.degrees(angle), toleranceM=tolerance)
            # Direct measured body motion is a collision risk; missing registration
            # remains advisory and does not make predicted contact into evidence.
            with self.rt._lock:
                memory = self.rt.estimator.stationary_assemblies
                memory.update(self.rt.estimator.tracks, self.rt.estimator.stamp)
                for key in ('body', 'part'):
                    record = memory.records.get(plan['contactParams'][key])
                    if record and record.get('hardMotion'):
                        raise SkillFailed('contact registration invalidated by direct body motion')
                    if record is None or record['status'] != 'frozen':
                        self.warn('contact_reference_missing', entity=plan['contactParams'][key], stamp=self.rt.estimator.stamp)
            if plan['operation'] == 'seat':
                feature = self.feature(plan['object'])
                raw = self.sample(plan['object'])
                before_release = None
                if (raw and raw['stamp'] >= contact_started and raw.get('source') == 'rgbd_segment'
                        and not raw.get('truncated', True) and raw.get('invalidDepthFraction', 1) <= .05):
                    observed = directed_tool(raw['points'], 'toy_screw', raw['stamp'])
                    if observed and np.linalg.norm(np.asarray(observed['tip'])-plan['goal']) <= .012:
                        before_release = observed
                self.ex._command({'gripper':'open'})
                # Save the action immediately, even if retreat or observation then fails.
                with self.rt._lock:
                    p = plan['contactParams']; key = (p['body'], p['hole'])
                    self.rt.estimator.fastening_attempts[key] = {
                        'body':p['body'], 'part':p['part'], 'hole':p['hole'], 'partHole':plan['targetEvidence']['partHole'],
                        'screw':plan['object'], 'entry':plan['targetEvidence']['point'],
                        'seatedAt':self.ex._state()['stamp'], 'status':'released_unverified',
                        'verified':False, 'headBeforeRelease':feature['head'],
                        'headBeforeSource':feature.get('source', 'nominal'),
                        'headObservationBeforeRelease':before_release,
                        'componentFrame':deepcopy(plan['targetEvidence'].get('componentFrame'))}
                # Use the current sensed-grasp endpoint frame, not a segmentation-old
                # fused centre left at the carrying pose.
                xy = ((np.asarray(feature['tip'])+feature['head'])/2)[:2].tolist()
                self.ex._released(plan['object'], xy)
                job.postcondition['releaseExecuted'] = True
                self.ex._wait(.4)
            else:
                # The preceding descent brings the bit just above the measured head.
                # Hold without adding an unplanned displacement or force command.
                began = self.ex._state()['stamp']
                deadline = time.monotonic()+30
                while self.ex._state()['stamp']-began < .4:
                    if time.monotonic() >= deadline:
                        raise SkillFailed('contact hold timed out')
                    self.check()
                    distance, angle = self.contact_error(plan)
                    if distance > tolerance or angle > math.radians(10):
                        if self.feature(plan['object']).get('polarityObserved', True):
                            raise SkillFailed('driver contact was not maintained; no fastening claim')
                        self.warn('driver_contact_estimate_error', distanceM=distance, angleDeg=math.degrees(angle))
                    self.ex._wait(.04)
                job.postcondition['contactHeldS'] = self.ex._state()['stamp']-began
            self.ex._set_joint('lift', plan['safeLift'])
            self.ex._withdraw(plan)
            if plan['operation'] == 'seat':
                for key in ('wrist_yaw','wrist_roll','wrist_pitch'):
                    if key in self.ex._state()['joints']:
                        self.ex._set_joint(key, 0., tol=.02)
            self.ex._backoff(plan)

    if _assembly_enabled():
        def _contact(self, job, plan):
            """Execute the exact checked approach; no unplanned IK fallback or target shifting."""
            contact_started = self.ex._state()['stamp']
            self.ex._dock(job, plan, retract=False)
            self.check()
            self.ex._set_joint('lift', plan['safeLift'])
            self.ex._set_joint('reach', plan['retractReach'])
            self.ex._set_joint('wrist_yaw', plan['wristYaw'], tol=.02)
            self.ex._set_joint('wrist_roll', plan['wristRoll'], tol=.02)
            if 'wrist_pitch' in self.ex._state()['joints']:
                self.ex._set_joint('wrist_pitch', plan['wristPitch'], tol=.02)
            self.ex._align_lateral(plan['gripSpot'], tol=.002)
            self.ex._set_joint('reach', self.ex._reach_to(plan['gripSpot']))
            from steward.ego.fasten_plan import seat_target
            from steward.ego.esim import PredictionError
            try:
                target = seat_target(self.rt.esim, self.rt.ego_state(), plan['contactParams'])
            except PredictionError as exc:
                self.warn('entry_recheck_unavailable', detail=str(exc))
                target = plan['targetEvidence']
            if np.linalg.norm(np.asarray(target['point'])-plan['targetEvidence']['point']) > .003:
                self.warn('entry_surface_shift', distanceM=float(np.linalg.norm(np.asarray(target['point'])-plan['targetEvidence']['point'])), toleranceM=.003)
            from steward.ego import tip_entry_servo as servo
            from steward.ego.fasten_seat import lift_to
            opening=self.rt.esim.arm.geometry.jaw_opening
            lift_to(self.ex,min(plan['safeLift'],plan['lift']+opening/4))
            current=dict(object=plan['object'],seatTarget=target,seatEndpoint=plan,
                seatHole=plan['contactParams'].get('hole'),body=plan['contactParams'].get('body'))
            if plan['operation']=='seat':
                servo.run(self,current,job.postcondition)
                servo.near_endpoint(self,current,job.postcondition,plan)
                lift_to(self.ex,plan['lift'])
            else:
                from steward.ego.driver_contact import lower
                plan['head']=list(target['point'])
                lower(self,plan,job.postcondition)
            self.ex._settle_for_release()
            tolerance = .0025 if plan['operation'] == 'seat' else .008
            # Direct measured body motion is a collision risk; missing registration
            # remains advisory and does not make predicted contact into evidence.
            with self.rt._lock:
                memory = self.rt.estimator.stationary_assemblies
                memory.update(self.rt.estimator.tracks, self.rt.estimator.stamp)
                for key in ('body', 'part'):
                    record = memory.records.get(plan['contactParams'][key])
                    if record and record.get('hardMotion'):
                        raise SkillFailed('contact registration invalidated by direct body motion')
                    if record is None or record['status'] != 'frozen':
                        self.warn('contact_reference_missing', entity=plan['contactParams'][key], stamp=self.rt.estimator.stamp)
            if plan['operation'] == 'seat':
                feature = self.feature(plan['object'])
                raw = self.sample(plan['object'])
                before_release = None
                if (raw and raw['stamp'] >= contact_started and raw.get('source') == 'rgbd_segment'
                        and not raw.get('truncated', True) and raw.get('invalidDepthFraction', 1) <= .05):
                    observed = directed_tool(raw['points'], 'toy_screw', raw['stamp'])
                    if observed and np.linalg.norm(np.asarray(observed['tip'])-plan['goal']) <= .012:
                        before_release = observed
                from steward.ego.fasten_accounting import allow_release
                p = plan['contactParams']
                self.rt._check_skill_preconditions(plan)
                allow_release(self.rt.estimator, p['body'], p['hole'], plan['object'])
                self.ex._command({'gripper':'open'})
                from steward.ego.grasp_endpoints import released
                job.postcondition['endpointReleaseEvidence'] = released(plan, self.ex._state()['stamp'])
                # Save the action immediately, even if retreat or observation then fails.
                with self.rt._lock:
                    p = plan['contactParams']; key = (p['body'], p['hole'])
                    previous = self.rt.estimator.fastening_attempts.get(key, {})
                    history = deepcopy(previous.get('releaseHistory', []))
                    if previous.get('releaseExecuted') and not history:
                        history.append(dict(screw=previous['screw'], stamp=previous['seatedAt']))
                    history.append(dict(screw=plan['object'], stamp=self.ex._state()['stamp']))
                    self.rt.estimator.fastening_attempts[key] = {
                        'releaseHistory':history, 'departureEvidence':None,
                        'nearContactUnconfirmed':bool(current.get('nearContactUnconfirmed')),
                        'body':p['body'], 'part':p['part'], 'hole':p['hole'], 'partHole':plan['targetEvidence']['partHole'],
                        'screw':plan['object'], 'entry':plan['targetEvidence']['point'],
                        'seatedAt':self.ex._state()['stamp'], 'status':'released_unverified', 'releaseExecuted':True,
                        'verified':False, 'headBeforeRelease':feature['head'],
                        'headBeforeSource':feature.get('source', 'nominal'),
                        'headObservationBeforeRelease':before_release,
                        'componentFrame':deepcopy(plan['targetEvidence'].get('componentFrame'))}
                from steward.ego.grasp_binding import annotate_release
                annotate_release(self.rt.estimator, self.rt.estimator.fastening_attempts[key], plan['object'])
                # Use the current sensed-grasp endpoint frame, not a segmentation-old
                # fused centre left at the carrying pose.
                xy = ((np.asarray(feature['tip'])+feature['head'])/2)[:2].tolist()
                self.ex._released(plan['object'], xy)
                job.postcondition['releaseExecuted'] = True
                current.update(releaseExecuted=True, releasedAt=self.ex._state()['stamp'])
                servo.finish(current)
                self.ex._wait(.4)
            else:
                # The preceding descent brings the bit just above the measured head.
                # Hold without adding an unplanned displacement or force command.
                began = self.ex._state()['stamp']
                deadline = ab.make(self.ex, .4+self.ex.config.tick_s, 'contact dwell plus one control tick') if ab.enabled() else time.monotonic()+30
                while self.ex._state()['stamp']-began < .4:
                    if ab.expired(deadline):
                        raise SkillFailed('contact hold timed out')
                    self.check()
                    distance, angle = self.contact_error(plan)
                    if distance > tolerance or angle > math.radians(10):
                        if self.feature(plan['object']).get('polarityObserved', True):
                            raise SkillFailed('driver contact was not maintained; no fastening claim')
                        self.warn('driver_contact_estimate_error', distanceM=distance, angleDeg=math.degrees(angle))
                    self.ex._wait(.04)
                job.postcondition['contactHeldS'] = self.ex._state()['stamp']-began
                servo.driver_after_contact(self,plan,job.postcondition)
            self.ex._set_joint('lift', plan['safeLift'])
            self.ex._withdraw(plan)
            if plan['operation'] == 'seat':
                for key in ('wrist_yaw','wrist_roll','wrist_pitch'):
                    if key in self.ex._state()['joints']:
                        self.ex._set_joint(key, 0., tol=.02)
            self.ex._backoff(plan)

    def seat(self, job, plan):
        if plan.get('method', 'simple') == 'simple':
            return self.simple_seat(job, plan)
        if _assembly_enabled():
            self._supply_members = {plan.get('body'), plan.get('part')}
        self.body_id = plan['body']
        self.reference(self.body_id)
        key = (self.body_id, plan['hole'])
        with self.rt._lock:
            previous = self.rt.estimator.fastening_attempts.get(key)
        if previous:
            self.warn('reuse_seating_attempt', body=self.body_id, hole=plan['hole'])
            self.confirm_seat(job, previous, plan)
            return
        scene = self.rt.ego_state()
        support = support_anchor(self.rt.esim, scene, scene.entity(self.body_id), job.postcondition.setdefault('warnings', [])).id
        screw = self.acquire_tool(job, 'toy_screw', support, free_screw=True)
        if _assembly_enabled():
            for pick_try in range(2):
                try:
                    self.step(job, 'pick', dict(target=screw.id,**({'expectedFastener':True} if sr.role_enabled() or sr.size_enabled() else {})))
                    self.held_kind_check(job, support)
                    break
                except SkillFailed as exc:
                    self.check()
                    if pick_try or 'grasp mismatch' not in str(exc) or self.rt.estimator.held_id:
                        raise
                    screw = self.acquire_tool(job, 'toy_screw', support, free_screw=True)
        else:
            self.step(job, 'pick', {'target':screw.id})
        held = self.rt.estimator.held_id
        if held != screw.id:
            self.warn('screw_alias_changed', requested=screw.id, held=held, stamp=self.rt.estimator.stamp)
        self.reference(self.body_id)
        self.step(job, '_fastener_contact', {**plan, 'screw':held, 'operation':'seat'})
        attempt = self.rt.estimator.fastening_attempts[key]
        if _assembly_enabled():
            job.postcondition.update(deepcopy(attempt))
        else:
            job.postcondition.update(deepcopy(attempt), releaseExecuted=True)
        view = self.wait_view(screw.id, attempt['seatedAt'])
        from steward.ego.fasten_simple import landing_error
        scene = self.rt.ego_state()
        attempt['landingErrors'] = [landing_error(scene.find(screw.id), attempt['entry'], scene.stamp, attempt['seatedAt'])]
        job.postcondition['landingErrors'] = deepcopy(attempt['landingErrors'])
        feature = self.feature(screw.id)
        seated = bool(view and feature.get('source') == 'rgbd_endpoint_width'
                      and feature['stamp'] >= view['stamp'] > attempt['seatedAt']
                      and np.linalg.norm(np.asarray(feature['tip'])-attempt['entry']) <= .012
                      and feature['axis'][2] >= math.cos(math.radians(10)))
        # Seat mode may already have buried the shaft before a complete tool view.
        # Head motion is independent evidence, regardless of whether drive ran.
        before = attempt.get('headObservationBeforeRelease') or dict(
            head=attempt['headBeforeRelease'], stamp=attempt['seatedAt'],
            source=attempt.get('headBeforeSource', 'nominal'))
        observed = screw_head(view, before)
        attached = head_sank(before, observed, attempt['entry'])
        if _assembly_enabled():
            with self.rt._lock:
                attempt.update(status='head_sink_observed' if attached else 'seated_observed' if seated else 'released_unverified',
                               verified=attached and not (attempt.get('bindingUnconfirmed') and not attempt.get('transportCommitted')),
                               seatedObserved=seated and not (attempt.get('bindingUnconfirmed') and not attempt.get('transportCommitted')), beforeDrive=deepcopy(feature))
        else:
            with self.rt._lock:
                attempt.update(status='head_sink_observed' if attached else 'seated_observed' if seated else 'released_unverified',
                               verified=attached, seatedObserved=seated, beforeDrive=deepcopy(feature))
        if not seated and not attached:
            self.warn('seating_unverified', screw=screw.id)
        self.confirm_seat(job, attempt, plan, observe=False)

    def drive(self, job, plan):
        if plan.get('method', 'simple') == 'simple':
            return self.simple_drive(job, plan)
        if _assembly_enabled():
            with self.rt._lock:
                attempts = self.rt.estimator.fastening_attempts
                pairs = [(k,a) for k,a in attempts.items() if a['screw'] == plan['screw']]
                if not pairs:
                    context = plan.get('driveContext') or drive_context(self.rt.esim, self.rt.ego_state(), plan)
                    key = (context['body'], context['hole'])
                    attempts[key] = deepcopy(context)
                    pairs = [(key, attempts[key])]
                    self.warn('seating_record_estimated', screw=plan['screw'])
                key, attempt = pairs[0]
                if attempt.get('driveStatus') == 'driven':
                    job.postcondition.update(deepcopy(attempt)); return
                params = {k:attempt[k] for k in ('body','part','hole','screw')}
        else:
            with self.rt._lock:
                attempts = self.rt.estimator.fastening_attempts
                pairs = [(k,a) for k,a in attempts.items() if a['screw'] == plan['screw']]
                if not pairs:
                    context = plan.get('driveContext') or drive_context(self.rt.esim, self.rt.ego_state(), plan)
                    key = (context['body'], context['hole'])
                    attempts[key] = deepcopy(context)
                    pairs = [(key, attempts[key])]
                    self.warn('seating_record_estimated', screw=plan['screw'])
                key, attempt = pairs[0]
                if attempt.get('verified'):
                    job.postcondition.update(deepcopy(attempt)); return
                params = {k:attempt[k] for k in ('body','part','hole','screw')}
        self.body_id = params['body']; self.reference(self.body_id)
        scene = self.rt.ego_state()
        support = support_anchor(self.rt.esim, scene, scene.entity(self.body_id), job.postcondition.setdefault('warnings', [])).id
        if _assembly_enabled():
            from steward.ego.fasten_driver import head_target, drive_check
            attempt.update(driveStatus='not_driven', driveStage='acquire', driveStrokeExecuted=False)
            try:
                self.pick_driver(job, support)
                self.seat_posterior(attempt)
                if attempt.get('seatCheck',{}).get('passed') is not True:
                    raise SkillFailed('seating_unconfirmed_before_drive')
                from steward.ego.fasten_sequence import require_attempt
                require_attempt(self, job, plan, attempt)
                before = head_target(attempt, self.rt.estimator.stamp)
                attempt['driveTarget'] = deepcopy(before)
                attempt['driveStage'] = 'stroke'
                started = self.ex._state()['stamp']
                receipt = self.rt.predict('_fastener_contact', {**params, 'operation':'drive'})
                if receipt.get('withinRiskLimit'):
                    attempt['driveStatus'] = 'unknown'
                self.step(job, '_fastener_contact', {**params, 'operation':'drive'}, receipt=receipt)
                after = self.ex._state()['stamp']
                attempt.update(status='driven_unverified', drivenAt=after, driveStrokeExecuted=True)
            finally:
                self.check()
                if self.rt.estimator.held_id:
                    try:
                        self.step(job, '_fastener_reset', {})
                    except SkillFailed as exc:
                        self.check(); self.warn('driver_reset_failed', detail=str(exc))
                attempt['driverParked'] = self.park_driver(job, support)
                job.postcondition.update(deepcopy(attempt))
        else:
            driver = self.acquire_tool(job, 'electric_screwdriver', support)
            before = deepcopy(self.feature(params['screw']))
            self.step(job, 'pick', {'target':driver.id, 'flip':True, 'prepareDriver':True})
            if self.rt.estimator.held_id != driver.id:
                self.warn('driver_alias_changed', requested=driver.id, held=self.rt.estimator.held_id, stamp=self.rt.estimator.stamp)
            self.step(job, '_fastener_contact', {**params, 'operation':'drive'})
            after = self.ex._state()['stamp']
            with self.rt._lock:
                attempt.update(status='driven_unverified', drivenAt=after)
            self.step(job, '_fastener_reset', {})
            self.park(job, support)
        view = self.wait_view(params['screw'], after)
        observed = screw_head(view, before)
        if _assembly_enabled():
            evidence = drive_check(before, observed, attempt['entry'], started)
            attempt.update(driveStatus=evidence['status'], driveCheck=evidence, driveStage='complete',
                verified=attempt.get('verified', False) or evidence['status'] == 'driven',
                status='head_sink_observed' if evidence['status'] == 'driven' else 'attachment_unknown')
            job.postcondition.update(deepcopy(attempt), executed=True)
            if not attempt['driverParked']:
                raise SkillFailed('driver parking incomplete; park held tool before resuming')
        else:
            verified = head_sank(before, observed, attempt['entry'])
            with self.rt._lock:
                attempt.update(verified=verified, status='head_sink_observed' if verified else 'attachment_unknown',
                               evidence={'kind':'head_sink', 'before':before, 'after':observed})
            job.postcondition.update(deepcopy(attempt))
            job.postcondition['executed'] = True
            if not verified:
                self.warn('attachment_unverified_after_drive', screw=params['screw'])

    def recover_component(self, job, plan, part, source):
        with self.rt._lock:
            records=self.rt.estimator.stationary_assemblies.records
            records.pop(part,None)
            records.pop(source['entity'],None)
            records[plan['body']]['parts'][:]=[p for p in records[plan['body']]['parts'] if p['entity']!=part]
            track=self.rt.estimator.tracks.get(source['entity'])
            if track is not None:
                track.pos=np.asarray(source['position'],dtype=float)
        if plan.get('recoverDropped') and source.get('mountingFaceUp') is False:
            from steward.ego.fasten_simple import offset_on
            scene=self.rt.ego_state(); table=support_anchor(self.rt.esim,scene,scene.entity(plan['body']),[])
            self.warn('restore_component_face_on_table',entity=source['entity'])
            self.step(job,'pick',dict(target=source['entity'],flip=True))
            self.step(job,'place',dict(on=table.id,offset=offset_on(table,source['position'][:2]),
                                       flip=True,alignHoles=False,closeRelease=True))

    def fasten(self, job, plan):
        if _assembly_enabled():
            from steward.ego import axial_release
            self._supply_members = {plan.get('body'), plan.get('part')}
        self.body_id = plan['body']
        if not hasattr(self, '_seat_retry_budget'):
            self._seat_retry_budget = [2]
        if _assembly_enabled():
            if not hasattr(self, '_seat_retry_counts'):
                self._seat_retry_counts = {}
        job.postcondition['confirmedSeatedScrews'] = 0
        scene = self.rt.ego_state()
        holes = mating_holes(self.rt.esim, scene, plan)
        selected_ids = {h['id'] for h in holes}
        if _assembly_enabled():
            from steward.ego.interface_entry import check as entry_check
            entry_check(self, job, plan['part'], plan['body'], holes)
        job.postcondition['selectedHoleIds'] = sorted(selected_ids)
        job.postcondition['requestedScrews'] = len(selected_ids)
        # Staging precedes placing the component, including all long source trips.
        with self.rt._lock:
            self._tool_sources = {t.id:deepcopy(t) for t in self.rt.estimator.tracks.values()
                if not t.held and tool_estimate(scene.find(t.id), scene.stamp)}
        if _assembly_enabled():
            if not plan.get('resumeReleased') and getattr(self, '_seat_only', False):
                from steward.ego.fasten_release import released_interface
                frame = released_interface(scene, plan['part'], self.body_id, allow_alias=True)
                if frame:
                    plan = dict(plan, part=frame['entity'], resumeReleased=True, releaseFrame=frame)
                elif any(a.get('releaseExecuted') and a['body'] == self.body_id and a['hole'] in selected_ids
                         for a in self.rt.estimator.fastening_attempts.values()):
                    raise SkillFailed('released interface unavailable; preserve the component and occupied holes')
        source_track = deepcopy(self.rt.estimator.tracks.get(plan['part']))
        if _assembly_enabled():
            from steward.ego.fasten_accounting import kit_support
            kit_support(self, plan['support'])
            from steward.ego.session_initial import occupied as initially_occupied
            unstaged = [h for h in holes if not initially_occupied(self.body_id, h['id']) and not self.rt.estimator.fastening_attempts.get((self.body_id, h['id']), {}).get('releaseExecuted')]
            staged = self.stage_screws(job, plan, unstaged) if plan.get('stageScrews') and unstaged else []
            staged_by_hole = dict(zip([h['id'] for h in unstaged], staged))
        else:
            staged = self.stage_screws(job, plan, holes) if plan.get('stageScrews') else []
        if source_track is not None and (plan['part'] not in self.rt.estimator.tracks or
                self.rt.estimator.tracks[plan['part']].category != source_track.category):
            self.rt.estimator.tracks[plan['part']] = source_track
            self.warn('component_last_source_restored', entity=plan['part'])
        quick = plan.get('method', 'simple') == 'simple'
        params = {k:plan[k] for k in ('part','body','hole','heldHole') if k in plan}
        params['hole'] = sorted(selected_ids)
        if plan.get('selectedHeldHoleIds'):
            params['heldHole'] = plan['selectedHeldHoleIds']
        params.update(deferVerify=quick, closeRelease=bool(plan.get('closeRelease')))
        if _assembly_enabled():
            lower_seat = getattr(self, '_seat_only', False) and getattr(self, '_lower_interface', False)
            if lower_seat:
                params['lowerSeatPlacement'] = True
            if plan.get('resumeReleased'):
                from steward.ego.fasten_release import released_interface
                frame = released_interface(self.rt.ego_state(), plan['part'], self.body_id)
                if not frame or frame['releasedAt'] != plan['releaseFrame']['releasedAt']:
                    raise SkillFailed('release frame changed; observe and predict fasten_onto again')
                from steward.ego.assembly_landing import ENABLED as landing_guard
                if landing_guard:
                    from steward.ego.placement_ready import assess as assess_placement
                    readiness, _ = assess_placement(self, dict(object=frame['entity'], releasedAt=frame['releasedAt']))
                    if readiness.get('passed') is False or frame.get('placementFailure'):
                        failure = readiness if readiness.get('passed') is False else frame['placementFailure']
                        job.postcondition['placementReadiness'] = deepcopy(failure)
                        raise SkillFailed('placement readiness failed: '+failure['reason'])
                assembly = dict(part=frame['entity'], releaseExecuted=True, releasedAt=frame['releasedAt'],
                                resumed=True, placementCheck={'passed':None, 'status':'unknown'})
                from steward.ego import session_observation
                if session_observation.enabled() and frame.get('preExisting'):
                    assembly.update(releaseExecuted=False, preExisting=True,
                                    observedAt=frame.get('observedAt'), source=frame.get('source'))
                check = assembly['placementCheck']
                placement_retries = 0
                retry = 0
            else:
                placement_retries = max(0, 2-plan.get('_dropRetries',0)) if plan.get('recoverDropped') else 1
                pickup_retry = False
                for retry in range(placement_retries+1):
                    try:
                        self.step(job, 'assemble_onto', params)
                    except SkillFailed as exc:
                        self.check()
                        assembly = job.postcondition['stages'][-1].get('result') or {}
                        if assembly.get('supportTargetCheck', {}).get('passed') is False:
                            job.postcondition['supportTargetCheck'] = deepcopy(assembly['supportTargetCheck'])
                            raise
                        if assembly.get('placementReadiness', {}).get('passed') is False:
                            job.postcondition['placementReadiness'] = deepcopy(assembly['placementReadiness'])
                            raise
                        if not assembly.get('releaseExecuted'):
                            source = plan.get('partSource', {})
                            with self.rt._lock:
                                fresh = [t for t in self.rt.estimator.tracks.values()
                                    if source and self.matches(t, source) and t.id != params['part']
                                    and t.last_seen >= (source_track.last_seen if source_track else 0.)]
                            if (not pickup_retry and not self.rt.estimator.held_id and fresh
                                    and any(x in str(exc) for x in ('grasp missed','grasp mismatch'))):
                                replacement = max(fresh, key=lambda t:t.last_seen)
                                pickup_retry = True
                                params['part'] = replacement.id
                                self.warn('component_pick_retry_fresh_source', entity=replacement.id)
                                self.step(job, 'assemble_onto', params)
                            elif (plan.get('wingRetry') and not pickup_retry and not self.rt.estimator.held_id
                                    and any(x in str(exc) for x in ('grasp missed','out of range','no feasible approach'))):
                                pickup_retry = True
                                self.warn('component_pick_retry_station', detail=str(exc))
                                table=self.rt.ego_state().entity(plan['support'])
                                box,_,_=entity_box(table); o=self.ex._state()['odometry']
                                self.view(job,table,math.atan2(o['y']-box.cy,o['x']-box.cx),.65)
                                self.step(job,'assemble_onto',params)
                            else:
                                raise
                        else:
                            self.warn('assembly_postcheck_failed_continue', detail=str(exc))
                    assembly = job.postcondition['stages'][-1]['result']
                    check = assembly.get('placementCheck', {})
                    if (getattr(self, '_seat_only', False) and assembly.get('releaseExecuted') or
                            check.get('passed') is not False or not check.get('retrySource') or retry == placement_retries):
                        break
                    source = check['retrySource']
                    if not plan.get('recoverDropped') and source.get('sourceDistanceM',0) > .04:
                        break
                    self.warn('placement_retry_from_observed_source', attempt=retry+1, evidence=deepcopy(check))
                    job.postcondition.setdefault('previousAssemblies', []).append(deepcopy(assembly))
                    self.recover_component(job, plan, assembly['part'], source)
                    params['part']=source['entity']
        else:
            placement_retries = max(0, 2-plan.get('_dropRetries',0)) if plan.get('recoverDropped') else 1
            pickup_retry = False
            for retry in range(placement_retries+1):
                try:
                    self.step(job, 'assemble_onto', params)
                except SkillFailed as exc:
                    self.check()
                    assembly = job.postcondition['stages'][-1].get('result') or {}
                    if not assembly.get('releaseExecuted'):
                        source = plan.get('partSource', {})
                        with self.rt._lock:
                            fresh = [t for t in self.rt.estimator.tracks.values()
                                if source and self.matches(t, source) and t.id != params['part']
                                and t.last_seen >= (source_track.last_seen if source_track else 0.)]
                        if (not pickup_retry and not self.rt.estimator.held_id and fresh
                                and 'grasp missed' in str(exc)):
                            replacement = max(fresh, key=lambda t:t.last_seen)
                            pickup_retry = True
                            params['part'] = replacement.id
                            self.warn('component_pick_retry_fresh_source', entity=replacement.id)
                            self.step(job, 'assemble_onto', params)
                        elif (plan.get('wingRetry') and not pickup_retry and not self.rt.estimator.held_id
                                and any(x in str(exc) for x in ('grasp missed','out of range','no feasible approach'))):
                            pickup_retry = True
                            self.warn('component_pick_retry_station', detail=str(exc))
                            table=self.rt.ego_state().entity(plan['support'])
                            box,_,_=entity_box(table); o=self.ex._state()['odometry']
                            self.view(job,table,math.atan2(o['y']-box.cy,o['x']-box.cx),.65)
                            self.step(job,'assemble_onto',params)
                        else:
                            raise
                    else:
                        self.warn('assembly_postcheck_failed_continue', detail=str(exc))
                assembly = job.postcondition['stages'][-1]['result']
                check = assembly.get('placementCheck', {})
                if check.get('passed') is not False or not check.get('retrySource') or retry == placement_retries:
                    break
                source = check['retrySource']
                if not plan.get('recoverDropped') and source.get('sourceDistanceM',0) > .04:
                    break
                self.warn('placement_retry_from_observed_source', attempt=retry+1, evidence=deepcopy(check))
                job.postcondition.setdefault('previousAssemblies', []).append(deepcopy(assembly))
                self.recover_component(job, plan, assembly['part'], source)
                params['part']=source['entity']
        job.postcondition['placementCheck'] = deepcopy(check)
        job.postcondition.update(releaseExecuted=assembly.get('releaseExecuted',False), assembly=assembly)
        if not _assembly_enabled():
            if check.get('passed') is False:
                count = 'one retry' if placement_retries == 1 else f'{placement_retries} retries'
                raise SkillFailed(f'component placement contradicted by fresh RGB-D after {count}')
        part = assembly['part']; body = plan['body']
        released_at = assembly.get('releasedAt')
        if _assembly_enabled():
            entry = entry_check(self, job, part, body, holes, placed=True)
            if check.get('passed') is False and assembly.get('releaseExecuted'):
                job.postcondition['previousPlacementCheck'] = deepcopy(check)
                job.postcondition['placementCheck'] = dict(passed=None, status='unknown',
                    reason=entry['reason'], entryCheckId=entry['id'])
            from steward.ego.component_alignment import before_fasteners
            before_fasteners(self, job, part, body, holes)
        else:
            if released_at is not None:
                with self.rt._lock:
                    old = self.rt.estimator.fastening_attempts
                    stale = [k for k,a in old.items() if a['body'] == body and
                             a['hole'] in selected_ids and a['seatedAt'] < released_at]
                    for k in stale:
                        job.postcondition.setdefault('supersededAttempts', []).append(deepcopy(old.pop(k)))
        scene = self.rt.ego_state(); base = scene.find(body)
        # Retain the pre-placement interface selection even if the body's mask
        # changes category or loses its hole catalogue after adding the part.
        if _assembly_enabled():
            from steward.ego.fasten_align import hole_order, align_after_first, save_measurement
            holes, order_source, first_frame = hole_order(self.rt.esim, scene, part, body, holes)
            if first_frame is not None:
                save_measurement(self, dict(status='estimated', frame=first_frame, reason='existing_view_before_first_screw'))
            job.postcondition['holeOrder'] = dict(ids=[h['id'] for h in holes], source=order_source)
            precision = assembly.get('precision') or {}
            # The close-view loop already owns the one post-release component move.
            # Keep the old fallback only when it had no usable close observations.
            alignment_done = any(m.get('status') == 'estimated' for m in precision.get('observations', [])) or (
                precision.get('postRelease', {}).get('status') == 'estimated')
            from steward.ego.fasten_sequence import ENABLED as batch_drives
            from steward.ego import interface_sequence
            batch_drives = batch_drives or interface_sequence.enabled()
            self._driver_alignment = bool(plan.get('drive', True))
        # Retain the pre-placement interface selection even if the body's mask
        # changes category or loses its hole catalogue after adding the part.
        failures = []
        if _assembly_enabled():
            pending_seats = {}
            for hole_index, hole in enumerate(holes):
                if initially_occupied(body, hole['id']):
                    continue
                key = (body, hole['id'])
                if plan.get('drive', True) and hole_index and not alignment_done:
                    first = next((self.rt.estimator.fastening_attempts.get((body, h['id']))
                                  for h in holes[:hole_index]
                                  if self.rt.estimator.fastening_attempts.get((body, h['id']), {}).get('releaseExecuted')), None)
                    if first and not self.rt.estimator.held_id:
                        alignment_done = True
                        from steward.ego.fasten_efficiency import align_next
                        align_next(self, job, first)
                for skill in (('seat_screw',) if batch_drives else ('seat_screw', 'drive_screw')):
                    attempt = self.rt.estimator.fastening_attempts.get(key)
                    from steward.ego.hole_seating import confirmed
                    if skill == 'seat_screw' and confirmed(attempt):
                        continue
                    if (skill == 'seat_screw' and plan.get('drive', True) and attempt
                            and attempt.get('releaseExecuted') and (attempt.get('driveStatus') == 'driven'
                                or attempt.get('seatCheck', {}).get('passed') is not False)):
                        continue
                    if skill == 'drive_screw' and (not plan.get('drive', True) or not attempt or attempt.get('driveStatus') == 'driven'
                            or not plan.get('driveExplicit') and attempt.get('verified')):
                        continue
                    if skill == 'drive_screw' and attempt.get('seatCheck',{}).get('passed') is not True:
                        failures.append(dict(hole=hole['id'],skill=skill,failure='seating_unconfirmed_before_drive'))
                        continue
                    params = {'part':part, 'body':body, 'hole':hole['id']} if skill == 'seat_screw' else {'screw':attempt['screw'], 'body':body, 'hole':hole['id']}
                    try:
                        params.update(method=plan.get('method', 'simple'), correctOnce=bool(plan.get('correctOnce', True)))
                        if skill == 'seat_screw':
                            params.update(deferVerify=quick, closeRelease=bool(plan.get('closeRelease')))
                            if hole['id'] in staged_by_hole:
                                row=staged_by_hole[hole['id']]
                                with self.rt._lock:
                                    nearby=[t for t in self.rt.estimator.tracks.values() if t.category=='toy_screw'
                                        and not t.held and t.last_seen>row['stagedAt']
                                        and np.linalg.norm(t.pos[:2]-row['xy'])<.025]
                                    t=min(nearby,key=lambda t:np.linalg.norm(t.pos[:2]-row['xy'])) if nearby else deepcopy(row['track'])
                                    self.rt.estimator.tracks[t.id]=t
                                params['stagedScrew']=t.id
                            params['excludedScrews'] = [r['entity'] for r in staged if r['entity'] != params.get('stagedScrew')]
                            params['excludedScrews'] += list(getattr(self,'_retired_screws',set()))
                            params['excludedPositions'] = [r['xy'] for h,r in staged_by_hole.items() if h != hole['id']]
                            if lower_seat:
                                pending_seats[hole['id']] = deepcopy(params)
                        self.step(job, skill, params)
                    except SkillFailed as exc:
                        self.check()  # cancellation/session reset is never a recoverable screw failure
                        failures.append(dict(hole=hole['id'], skill=skill, failure=str(exc)))
                        if '本套紧固件不足' in str(exc):
                            job.postcondition['materialStatus'] = '本套紧固件不足'
                        self.warn('fastener_attempt_failed', **failures[-1])
                        # A failed contact can leave a screw/driver in hand. Free the
                        # hand through its own checked receipt before the next attempt.
                        if self.rt.estimator.held_id:
                            child_result = job.postcondition['stages'][-1].get('result') or {}
                            if skill == 'drive_screw' and child_result.get('driverClear') is False:
                                raise SkillFailed('driver withdrawal failed; tool remains held, inspect before recovery')
                            scene = self.rt.ego_state()
                            support = support_anchor(self.rt.esim, scene, scene.entity(body), job.postcondition.setdefault('warnings', [])).id
                            if plan.get('method', 'simple') == 'contact' and self.rt.estimator.tracks[self.rt.estimator.held_id].category == 'electric_screwdriver':
                                try:
                                    self.step(job, '_fastener_reset', {})
                                except SkillFailed as reset_error:
                                    self.check()
                                    self.warn('driver_reset_failed', detail=str(reset_error))
                            if skill == 'drive_screw':
                                self.park_driver(job, support)
                            else:
                                self.park(job, support)
                    alignment_stage = 'drive_screw' if plan.get('drive', True) and not batch_drives else 'seat_screw'
                    if skill == alignment_stage and len(holes) > 1 and not alignment_done:
                        first = self.rt.estimator.fastening_attempts.get(key)
                        if first and first.get('releaseExecuted') and not self.rt.estimator.held_id:
                            alignment_done = True
                            from steward.ego.fasten_efficiency import align_next
                            align_next(self, job, first)
            if lower_seat:
                from steward.ego.fasten_lower import retry_unreleased
                recovered = retry_unreleased(self, job, pending_seats, plan['support'])
                failures = [f for f in failures if not (f['skill'] == 'seat_screw' and f['hole'] in recovered)]
            attempts = [a for a in self.rt.estimator.fastening_attempts.values()
                        if a['body'] == body and a['part'] == part and a['hole'] in selected_ids]
            if quick:
                # All holes precede screw replay; the first-pin alignment already ran.
                from steward.ego.fasten_simple import landing_error
                for a in attempts:
                    scene=self.rt.ego_state()
                    a['landingErrors']=[landing_error(scene.find(a['screw']),a['entry'],scene.stamp,a['seatedAt'])]
                released_at=assembly.get('releasedAt')
                if released_at is not None and attempts:
                    job.postcondition['releaseToFirstScrewS']=min(a['seatedAt'] for a in attempts)-released_at
                # Now take the deferred view, and report the same independent position check.
                from steward.ego.fasten_efficiency import final_view
                from steward.ego import axial_release
                if not axial_release.enabled() or not all(axial_release.checked(a) for a in attempts):
                    final_view(self, job, attempts)
                scene = self.rt.ego_state()
                for a in attempts:
                    a['landingErrors'].append(landing_error(scene.find(a['screw']),a['entry'],scene.stamp,a['seatedAt']))
                from steward.ego.assembly_motion import placement_check
                record=self.rt.estimator.stationary_assemblies.records.get(body,{})
                expected=next((p for p in record.get('parts',[]) if p['entity']==part),None)
                if expected and released_at is not None:
                    check=placement_check(self.rt.estimator.tracks.values(),plan['partSource'],expected,
                                          released_at,self.rt.estimator.stamp)
                    job.postcondition['placementCheck']=check
            if quick and check.get('passed') is False and getattr(self, '_seat_only', False):
                raise SkillFailed('component placement contradicted; preserve the released component and screw ledger; observe before continuation')
            if (quick and check.get('passed') is False and not plan.get('resumeReleased')
                    and not getattr(self, '_seat_only', False)):
                used=plan.get('_dropRetries',0)+retry
                source=check.get('retrySource')
                limit = 2 if plan.get('recoverDropped') else 1
                retry_source = source and (plan.get('recoverDropped') or source.get('sourceDistanceM', 0) <= .04)
                if retry_source and used < limit:
                    job.postcondition.setdefault('discardedAttempts',[]).extend(deepcopy(attempts))
                    self._retired_screws=getattr(self,'_retired_screws',set()) | {a['screw'] for a in attempts}
                    self.recover_component(job,plan,part,source)
                    next_plan={**plan,'part':source['entity'],'_dropRetries':used+1,
                               'partSource':dict(id=source['entity'],category=plan['partSource']['category'],position=source['position'])}
                    job.postcondition['dropRecoveryCount']=used+1
                    return self.fasten(job,next_plan)
                raise SkillFailed('component placement contradicted after one retry; recovery budget exhausted or source unknown')
            # The first pass still batches the holes. A shared posterior view then
            # triggers at most two replays in this same job, independent of component_staging flags.
            if quick and (not plan.get('drive', True) or batch_drives):
                for a in attempts:
                    root = job.postcondition
                    job.postcondition = {}
                    try:
                        self.confirm_seat(job, a, dict(plan, body=body, part=part, hole=a['hole']), observe=False)
                    except SkillFailed as exc:
                        self.check()
                        failures.append(dict(hole=a['hole'], skill='seat_screw', failure=str(exc)))
                    finally:
                        postcheck = job.postcondition
                        if batch_drives and plan.get('drive', True):
                            postcheck = {k:postcheck[k] for k in ('hole','screw','seatCheck','seatRetryCount') if k in postcheck}
                        root.setdefault('seatPostchecks', []).append(deepcopy(postcheck))
                        job.postcondition = root
                    for stage in job.postcondition['stages']:
                        if stage['skill'] == 'seat_screw' and stage.get('receipt', {}).get('params', {}).get('hole') == a['hole']:
                            stage['result'] = deepcopy(a)
                            if a.get('seatCheck', {}).get('passed') is False:
                                stage['failure'] = 'landing '+a.get('seatCheck', {}).get('status', 'unknown')
            if (batch_drives and plan.get('drive', True) and len(attempts)==len(selected_ids)
                    and any(a.get('seatConfirmed') and a.get('seatCheck',{}).get('passed') is True for a in attempts)):
                from steward.ego.fasten_sequence import drive_all
                drive_holes = [h for h in holes if any(a['hole'] == h['id'] and
                    a.get('seatCheck', {}).get('passed') is True for a in attempts)]
                drive_all(self, job, dict(plan, part=part) if interface_sequence.enabled() else plan, drive_holes, failures)
        else:
            for hole_index, hole in enumerate(holes):
                key = (body, hole['id'])
                for skill in ('seat_screw', 'drive_screw'):
                    attempt = self.rt.estimator.fastening_attempts.get(key)
                    if skill == 'drive_screw' and (not plan.get('drive', True) or not attempt or attempt.get('verified')):
                        continue
                    params = {'part':part, 'body':body, 'hole':hole['id']} if skill == 'seat_screw' else {'screw':attempt['screw']}
                    try:
                        params.update(method=plan.get('method', 'simple'), correctOnce=bool(plan.get('correctOnce', True)))
                        if skill == 'seat_screw':
                            params.update(deferVerify=quick, closeRelease=bool(plan.get('closeRelease')))
                            if hole_index < len(staged):
                                row=staged[hole_index]
                                with self.rt._lock:
                                    nearby=[t for t in self.rt.estimator.tracks.values() if t.category=='toy_screw'
                                        and not t.held and t.last_seen>row['stagedAt']
                                        and np.linalg.norm(t.pos[:2]-row['xy'])<.025]
                                    t=min(nearby,key=lambda t:np.linalg.norm(t.pos[:2]-row['xy'])) if nearby else deepcopy(row['track'])
                                    self.rt.estimator.tracks[t.id]=t
                                params['stagedScrew']=t.id
                            params['excludedScrews'] = [r['entity'] for r in staged if r['entity'] != params.get('stagedScrew')]
                            params['excludedScrews'] += list(getattr(self,'_retired_screws',set()))
                            params['excludedPositions'] = [r['xy'] for i,r in enumerate(staged) if i != hole_index]
                        self.step(job, skill, params)
                    except SkillFailed as exc:
                        self.check()  # cancellation/session reset is never a recoverable screw failure
                        failures.append(dict(hole=hole['id'], skill=skill, failure=str(exc)))
                        self.warn('fastener_attempt_failed', **failures[-1])
                        # A failed contact can leave a screw/driver in hand. Free the
                        # hand through its own checked receipt before the next attempt.
                        if self.rt.estimator.held_id:
                            scene = self.rt.ego_state()
                            support = support_anchor(self.rt.esim, scene, scene.entity(body), job.postcondition.setdefault('warnings', [])).id
                            if plan.get('method', 'simple') == 'contact' and self.rt.estimator.tracks[self.rt.estimator.held_id].category == 'electric_screwdriver':
                                try:
                                    self.step(job, '_fastener_reset', {})
                                except SkillFailed as reset_error:
                                    self.check()
                                    self.warn('driver_reset_failed', detail=str(reset_error))
                            self.park(job, support)
            attempts = [a for a in self.rt.estimator.fastening_attempts.values() if a['body'] == body and a['part'] == part]
            if quick:
                # Both holes were attempted before any optional wait or correction.
                from steward.ego.fasten_simple import landing_error
                for a in attempts:
                    scene=self.rt.ego_state()
                    a['landingErrors']=[landing_error(scene.find(a['screw']),a['entry'],scene.stamp,a['seatedAt'])]
                released_at=assembly.get('releasedAt')
                if released_at is not None and attempts:
                    job.postcondition['releaseToFirstScrewS']=min(a['seatedAt'] for a in attempts)-released_at
                # Now take the deferred view, and report the same independent position check.
                if attempts:
                    self.close_seat_view(job, attempts)
                scene = self.rt.ego_state()
                for a in attempts:
                    a['landingErrors'].append(landing_error(scene.find(a['screw']),a['entry'],scene.stamp,a['seatedAt']))
                from steward.ego.assembly_motion import placement_check
                record=self.rt.estimator.stationary_assemblies.records.get(body,{})
                expected=next((p for p in record.get('parts',[]) if p['entity']==part),None)
                if expected and released_at is not None:
                    check=placement_check(self.rt.estimator.tracks.values(),plan['partSource'],expected,
                                          released_at,self.rt.estimator.stamp)
                    job.postcondition['placementCheck']=check
            if quick and check.get('passed') is False:
                used=plan.get('_dropRetries',0)+retry
                source=check.get('retrySource')
                limit = 2 if plan.get('recoverDropped') else 1
                retry_source = source and (plan.get('recoverDropped') or source.get('sourceDistanceM', 0) <= .04)
                if retry_source and used < limit:
                    job.postcondition.setdefault('discardedAttempts',[]).extend(deepcopy(attempts))
                    self._retired_screws=getattr(self,'_retired_screws',set()) | {a['screw'] for a in attempts}
                    with self.rt._lock:
                        for a in attempts:
                            self.rt.estimator.fastening_attempts.pop((body,a['hole']),None)
                    self.recover_component(job,plan,part,source)
                    next_plan={**plan,'part':source['entity'],'_dropRetries':used+1,
                               'partSource':dict(id=source['entity'],category=plan['partSource']['category'],position=source['position'])}
                    job.postcondition['dropRecoveryCount']=used+1
                    return self.fasten(job,next_plan)
                raise SkillFailed('component placement contradicted after one retry; recovery budget exhausted or source unknown')
            # The first pass still batches the holes. A shared posterior view then
            # triggers at most two replays in this same job, independent of component_staging flags.
            if quick:
                for a in attempts:
                    root = job.postcondition
                    job.postcondition = {}
                    try:
                        self.confirm_seat(job, a, dict(plan, body=body, part=part, hole=a['hole']), observe=False)
                    except SkillFailed as exc:
                        self.check()
                        failures.append(dict(hole=a['hole'], skill='seat_screw', failure=str(exc)))
                    finally:
                        root.setdefault('seatPostchecks', []).append(deepcopy(job.postcondition))
                        job.postcondition = root
                    for stage in job.postcondition['stages']:
                        if stage['skill'] == 'seat_screw' and stage.get('receipt', {}).get('params', {}).get('hole') == a['hole']:
                            stage['result'] = deepcopy(a)
                            if not a.get('seatConfirmed'):
                                stage['failure'] = 'landing '+a.get('seatCheck', {}).get('status', 'unknown')
        confirmed = sum(bool(a.get('seatConfirmed')) for a in attempts)
        if _assembly_enabled():
            job.postcondition.update(verified=any(a.get('verified') for a in attempts), executed=True,
                part=part, body=body, attempts=deepcopy(attempts), partialFailures=failures,
                confirmedSeatedScrews=confirmed, requestedScrews=len(selected_ids),
                seatRetriesUsed=(sum(a.get('seatRetryCount', 0) for a in attempts) if axial_release.enabled()
                                 else 2-self._seat_retry_budget[0]),
                introducedScrewCount=sum(bool(a.get('releaseExecuted')) for a in attempts),
                releaseEventCount=sum(len(a.get('releaseHistory', [a])) for a in attempts if a.get('releaseExecuted')),
                reason='landing count uses independent screw geometry; attachment remains separate')
            released = {a['hole'] for a in attempts if a.get('releaseExecuted')}
            unknown = sum(a.get('seatCheck', {}).get('status') == 'unknown' for a in attempts)
            unresolved = [f for f in failures if not (f['skill'] == 'seat_screw' and any(
                a['hole'] == f['hole'] and a.get('seatConfirmed') for a in attempts))]
            from steward.ego.hole_seating import continuable
            initial_ids = {h for h in selected_ids if initially_occupied(body, h)}
            can_continue = len({a['hole'] for a in attempts} | initial_ids) == len(selected_ids) and all(continuable(a) for a in attempts)
            job.postcondition.update(unresolvedFailures=unresolved, executedScrewHoles=sorted(released), unconfirmedScrews=unknown,
                confirmation='confirmed' if confirmed == len(selected_ids) else 'unverified',
                canContinue=not unresolved and can_continue,
                actionStatus='partial' if unresolved or confirmed != len(selected_ids) else 'executed')
            if unresolved or not can_continue:
                shortage = job.postcondition.get('materialStatus', '')
                raise SkillFailed((shortage+'；' if shortage else '')+f'fastening actions incomplete; visually confirmed {confirmed}/{len(selected_ids)}; '
                                  'see partialFailures and seatCheck; unresolved seating blocks interface continuation')
            if unknown:
                job.postcondition['guidance'] = '入座未确认；保留该孔占用记录，禁止补料和工具顶压，继续其余可做步骤。'
        else:
            job.postcondition.update(verified=any(a.get('verified') for a in attempts), executed=True,
                part=part, body=body, attempts=deepcopy(attempts), partialFailures=failures,
                confirmedSeatedScrews=confirmed, requestedScrews=len(selected_ids),
                seatRetriesUsed=2-self._seat_retry_budget[0],
                reason='landing count uses independent screw geometry; attachment remains separate')
            if confirmed != len(selected_ids):
                raise SkillFailed(f'confirmed screw landings {confirmed}/{len(selected_ids)}; failed or unknown posterior')

    def flip(self, job, plan):
        if _assembly_enabled():
            from steward.ego.base_command_ledger import invalidate
            invalidate(self.ex, 'flip_started')
            from steward.ego.working_face import flip_guard
            protected = flip_guard(self.rt.esim, self.rt.ego_state(), plan)
            if protected:
                job.postcondition.update(protected)
                job.log.append(protected['guidance'])
                return protected
            from steward.ego.fasten_landing import flip_advice
            job.postcondition['flipAdvice'] = flip_advice(
                getattr(self.rt.estimator, 'released_interfaces', {}).values(),
                self.rt.estimator.fastening_attempts.values(), plan['body'])
            self._supply_members = {plan.get('body'), plan.get('part')}
        self.body_id = plan['body']
        if _assembly_enabled():
            from steward.ego import edge_retention
            saved = edge_retention.retained_record(self.rt.estimator.stationary_assemblies.records, self.body_id)
            from steward.ego import placement_support_margin
            if placement_support_margin.enabled():
                saved = placement_support_margin.retained(self.rt.estimator.stationary_assemblies.records, self.body_id) or saved
            if edge_retention.enabled() and saved and (not plan.get('_edgeAdjusted') or
                    placement_support_margin.enabled() and not saved.get('edgeCorrectionTarget')):
                return edge_retention.retained_result(job, saved)
        record = self.reference(self.body_id)
        if _assembly_enabled():
            from steward.ego.flip_grasp import reference
            with reference(self.rt, record, job) as release_reference:
                return self._flip_registered(job, plan, record, release_reference)
        else:
            if plan.get('levelBelly'):
                from steward.ego.assembly_quick import level_assessment
                job.postcondition['levelBelly'] = level_assessment(record)
                self.warn('level_belly_assessment', assessment=job.postcondition['levelBelly'])
            from steward.ego.assembly_motion import restore_track, released_record, front_spots, local_bounds
            with self.rt._lock:
                body_track = self.rt.estimator.tracks.get(self.body_id)
                if body_track is None:
                    body_track = restore_track(self.rt.estimator, record)
                body_track.pos = np.array(record['reference']['position'])
                body_track.size = np.array(record['reference']['size'])
                body_track.yaw = record['reference'].get('yaw', body_track.yaw)
            support_track = deepcopy(self.rt.estimator.tracks.get(record['reference'].get('support')))
            members = record['parts']
            turn_over = plan.get('turnOver', True)
            attempts = self.rt.estimator.fastening_attempts
            from steward.ego.fasten_release import loose_components
            loose = loose_components(self.rt.ego_state(), self.body_id)
            if loose:
                self.warn('loose_components_before_flip', parts=[p['entity'] for p in loose],
                          confirmedSeatedScrews=0, advisory=True)
            if not members or any(not any(a['part'] == p['entity'] and a['body'] == self.body_id and a.get('verified')
                                          for a in attempts.values()) for p in members):
                self.warn('attachment_unverified_before_flip', members=len(members), verifiedAttempts=sum(bool(a.get('verified')) for a in attempts.values()))
            # Conservative circumsphere includes every known released component through
            # rotation, also for pick/navigation. It does not create a connection belief.
            root = np.asarray(record['reference']['position'])
            sphere = max([np.linalg.norm(np.asarray(record['reference']['size'])/2)] +
                [np.linalg.norm(np.asarray(p['position'])-root)+np.linalg.norm(np.asarray(p['size'])/2) for p in members])
            before = {p['entity']:self.sample(p['entity']) for p in members}
            body_before = self.sample(self.body_id)
            now = self.ex._state()['stamp']
            if (not body_before or not all(before.values()) or
                    any(now-v['stamp'] > 20 for v in [body_before, *before.values()] if v)):
                self.warn('flip_previews_incomplete', bodyAvailable=bool(body_before), availableMembers=sum(bool(v) for v in before.values()), stamp=now)
            def independent(v):
                return v if (v and 0 <= now-v['stamp'] <= 20 and not v.get('truncated', True)
                             and v.get('invalidDepthFraction', 1) <= .05) else None
            body_before = independent(body_before)
            before = {eid:independent(v) for eid,v in before.items()}
            with self.rt._lock:
                self.rt.estimator.manipulation_envelopes[self.body_id] = float(2*sphere)
            try:
                self.step(job, 'pick', {'target':self.body_id, 'flip':turn_over, 'assemblyGrip':True})
                if self.rt.estimator.held_id != self.body_id:
                    self.warn('body_alias_changed_at_grasp', requested=self.body_id, held=self.rt.estimator.held_id, stamp=self.rt.estimator.stamp)
                after = self.ex._state()['stamp']; observations = []
                for _ in range(2):
                    body_view = self.wait_view(self.body_id, after)
                    row = {}
                    for p in members:
                        row[p['entity']] = self.wait_view(p['entity'], after)
                    observations.append((body_view,row))
                    after = max([after] + [v['stamp'] for v in [body_view, *row.values()] if v])
                follow = {p['entity']:bool(body_before and before[p['entity']] and all(b and rows[p['entity']] for b,rows in observations)) and follows([body_before, before[p['entity']]],
                    [[b,rows[p['entity']]] for b,rows in observations]) for p in members}
                job.postcondition['followEvidence'] = follow
                job.postcondition['followSamples'] = [
                    [{k:v for k,v in (b or {}).items() if k != 'points'},
                     {eid:{k:v for k,v in (row or {}).items() if k != 'points'} for eid,row in rows.items()}]
                    for b,rows in observations]
                if not all(follow.values()):
                    self.warn('following_unverified', observedMembers=sum(follow.values()), totalMembers=len(members))
                with self.rt._lock:
                    est = self.rt.estimator
                    gr = est.gripper_rotation(est.held_gripper)
                    gp = est.gripper_position(est.held_gripper)
                    # Keep component identities and encoder-predicted poses while carried.
                    # Independent raw RGB-D samples remain untouched for contradiction checks.
                    for eid, raw in observations[-1][1].items():
                        if not raw or not follow[eid]:
                            continue
                        est.rigid_carry[eid] = {'offset':gr.T @ (np.asarray(raw['centreM'])-gp),
                            'rotation':gr.T @ matrix_from_quat(quat_from_yaw(raw['boxYawRad'])),
                            'observedAt':raw['stamp']}
                support = record['reference']['support']
                # The original body frame is now invalid. Keep it only as a pre-motion
                # hypothesis and re-register from new views after the executed flip.
                with self.rt._lock:
                    est = self.rt.estimator
                    rot0 = est.gripper_rotation(est.held_gripper).copy()
                    est.flip_capture = {'body':self.body_id, 'rotation':rot0, 'record':deepcopy(record)}
                scene = self.rt.ego_state()
                table = scene.find(support) or record['support']
                low, high = local_bounds(record)
                half = float(max(abs(low[:2]).max(), abs(high[:2]).max()))
                last = None
                from steward.ego.fasten_simple import offset_on
                tables = [table] + [e for e in scene.entities if e.id != table.id and
                                   self.rt.esim._affords(e, 'support-surface')]
                candidates = []
                for surface in tables:
                    offsets = front_spots(self.rt.esim, scene, surface, half)
                    if surface.id == support:
                        offsets.insert(0, offset_on(surface, root[:2]))
                    for offset in [*offsets, None]:
                        candidates.append(dict(on=surface.id, flip=turn_over, alignHoles=False,
                            assemblyBody=self.body_id, **({'offset':offset} if offset is not None else {})))
                for candidate in candidates:
                    receipt = self.rt.predict('place', candidate)
                    if not receipt.get('withinRiskLimit'):
                        last = '; '.join(receipt.get('issues', []))
                        self.warn('assembly_place_candidate_failed', on=candidate['on'], offset=candidate.get('offset'), detail=last)
                        continue
                    place_plan = self.step(job, 'place', candidate, receipt=receipt)
                    support = candidate['on']
                    table = next(t for t in tables if t.id == support)
                    break
                else:
                    raise SkillFailed('assembly placement has no reachable table candidate: '+str(last))
                with self.rt._lock:
                    captured = self.rt.estimator.flip_capture
                    self.rt.estimator.flip_capture = None
                if not captured or 'releaseRotation' not in captured:
                    self.warn('flip_release_frame_estimated')
                    axis = matrix_from_quat(quat_from_yaw(record['reference']['yaw']))
                    delta = axis @ np.diag([1.,-1.,-1.]) @ axis.T if turn_over else np.eye(3)
                    after = self.ex._state()['stamp']
                else:
                    delta = captured['releaseRotation'] @ rot0.T
                    after = captured['releasedAt']
                fresh = self.wait_view(self.body_id, after)
                orientation_observed = False
                if turn_over and (delta @ np.array([0.,0.,1.]))[2] > -math.cos(math.radians(10)):
                    second = self.wait_view(self.body_id, fresh['stamp'] if fresh else after)
                    settled = settled_flip_rotation(delta, body_before, [fresh,second]) if body_before and fresh and second else None
                    if settled is None:
                        self.warn('flip_face_unverified', releaseUpZ=float(delta[2,2]))
                        # Nominal completed half turn: preserve the observed horizontal
                        # axis as best possible. This transports priors, not evidence.
                        u, _, vt = np.linalg.svd(delta[:2,:2])
                        xy = u @ vt
                        if np.linalg.det(xy) > 0:
                            u[:,-1] *= -1
                            xy = u @ vt
                        delta = np.zeros((3,3)); delta[:2,:2] = xy; delta[2,2] = -1
                    else:
                        delta = settled; fresh = second; orientation_observed = True
                        job.postcondition['faceReversalObserved'] = True
                # Rebase directly from the completed release. Old pre-flip tracks and
                # registrations cannot supply a stationary reference for the next face.
                scene = self.rt.ego_state()
                table = scene.find(support) or table
                new = released_record(record, place_plan['spot'], table, delta, after, fresh)
                new_ref = new['reference']
                # One cheap look from the release retreat. Neither a half-turn command
                # nor propagated tracks count as a visually completed flip.
                self.observation_target = new_ref['position']
                self.look(after, cycles=1)
                with self.rt._lock:
                    fresh_bodies = []
                    for t in self.rt.estimator.tracks.values():
                        v = t.manipulation_view
                        if (not v or t.held or t.category not in (record['reference']['category'], record['reference'].get('family'))
                                or v['stamp'] <= after or self.rt.estimator.stamp-v['stamp'] > 20
                                or v.get('truncated', True) or v.get('invalidDepthFraction', 1) > .05
                                or not np.allclose(np.sort(v['visibleSizeM'][:2]),
                                                   np.sort(record['reference']['size'][:2]), atol=.025)):
                            continue
                        d = min(np.linalg.norm(np.asarray(v['centreM'])[:2]-root[:2]),
                                np.linalg.norm(np.asarray(v['centreM'])[:2]-new_ref['position'][:2]))
                        if d <= .10:
                            fresh_bodies.append(deepcopy(v))
                fresh = fresh_bodies[0] if len(fresh_bodies) == 1 else None
                from steward.ego.assembly_motion import flip_check
                check = flip_check(body_before, fresh, before,
                    {p['entity']:self.sample(p['entity']) for p in members}, after, self.rt.estimator.stamp)
                if (fresh and np.linalg.norm(root[:2]-new_ref['position'][:2]) > .10 and
                        np.linalg.norm(np.asarray(fresh['centreM'])[:2]-root[:2]) < .04):
                    check = dict(passed=False, status='fail', reason='body_still_at_pickup',
                                 observedPositionM=fresh['centreM'], expectedPositionM=new_ref['position'])
                job.postcondition['orientationCheck'] = check
                orientation_observed = check['passed'] is True
                if check['passed'] is False and fresh:
                    # A measured unchanged face invalidates the commanded turn
                    # hypothesis. Keep the observed location and the previous face.
                    landing_table = record['support'] if check['reason'] == 'body_still_at_pickup' else table
                    new = released_record(record, fresh['centreM'][:2], landing_table, np.eye(3), after, fresh)
                    new_ref = new['reference']
                with self.rt._lock:
                    est = self.rt.estimator
                    for eid in {self.body_id, *(p['entity'] for p in members)}:
                        est.stationary_assemblies.records.pop(eid, None)
                        est.assembly_memory.invalidate(eid)
                    if self.body_id not in est.tracks:
                        self.warn('body_track_restored_after_flip', entity=self.body_id)
                    t = restore_track(est, new)
                    t.held = False
                    t.pos = np.asarray(new_ref['position'])
                    t.yaw = new_ref['yaw']
                    t.size = np.asarray(new_ref['size'])
                    t.interface_views.clear(); t.interface_best_near = None
                    est.stationary_assemblies.records[self.body_id] = new
                    if support == record['reference']['support'] and support not in est.tracks and support_track is not None:
                        est.tracks[support] = support_track
                    for a in attempts.values():
                        if a['body'] == self.body_id:
                            a['entry'] = (np.asarray(new_ref['position'])+delta @ (np.asarray(a['entry'])-root)).tolist()
                            a['transportedAt'] = after
                    job.postcondition.update(verified=(orientation_observed if turn_over else bool(members) and all(follow.values())),
                        executed=True, releaseExecuted=True, body=self.body_id, reference=deepcopy(new_ref),
                        flipped=check['passed'] if turn_over else False, commandedTurnOver=turn_over,
                        orientationSource='commanded_relative_half_turn' if check['passed'] is not False else 'observed_unchanged_face', upright=None,
                        passed=check['passed'] if turn_over else None)
                    # Independently observed following also confirms seat-mode attachment.
                    for a in attempts.values():
                        if a['body'] == self.body_id and follow.get(a['part']):
                            a.update(verified=True, status='following_observed')
            finally:
                with self.rt._lock:
                    # While still held, retain the full load envelope for safe recovery.
                    if self.rt.estimator.held_id != self.body_id:
                        self.rt.estimator.manipulation_envelopes.pop(self.body_id, None)
                    self.rt.estimator.flip_capture = None

    if _assembly_enabled():
        def _flip_registered(self, job, plan, record, release_reference):
            if plan.get('levelBelly'):
                from steward.ego.assembly_quick import level_assessment
                job.postcondition['levelBelly'] = level_assessment(record)
                self.warn('level_belly_assessment', assessment=job.postcondition['levelBelly'])
            from steward.ego.assembly_motion import restore_track, released_record, front_spots, local_bounds
            with self.rt._lock:
                body_track = self.rt.estimator.tracks.get(self.body_id)
            support_track = deepcopy(self.rt.estimator.tracks.get(record['reference'].get('support')))
            from steward.ego import edge_geometry
            if edge_geometry.enabled() and not plan.get('_edgeAdjusted'):
                try:
                    edge_geometry.recover_members(self, record)
                except (AttributeError, KeyError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
                    record['bearingRecovery'] = dict(members=len(record['parts']), reason=type(exc).__name__, observed=False)
                self.rt.estimator.stationary_assemblies.records[self.body_id] = deepcopy(record)
            members = record['parts']
            released_frames = deepcopy(getattr(self.rt.estimator, 'released_interfaces', {}))
            turn_over = plan.get('turnOver', True)
            attempts = self.rt.estimator.fastening_attempts
            from steward.ego.fasten_release import loose_components
            loose = loose_components(self.rt.ego_state(), self.body_id)
            if loose:
                self.warn('loose_components_before_flip', parts=[p['entity'] for p in loose],
                          confirmedSeatedScrews=0, advisory=True)
            if not members or any(not any(a['part'] == p['entity'] and a['body'] == self.body_id and a.get('verified')
                                          for a in attempts.values()) for p in members):
                self.warn('attachment_unverified_before_flip', members=len(members), verifiedAttempts=sum(bool(a.get('verified')) for a in attempts.values()))
            # Conservative circumsphere includes every known released component through
            # rotation, also for pick/navigation. It does not create a connection belief.
            root = np.asarray(record['reference']['position'])
            sphere = max([np.linalg.norm(np.asarray(record['reference']['size'])/2)] +
                [np.linalg.norm(np.asarray(p['position'])-root)+np.linalg.norm(np.asarray(p['size'])/2) for p in members])
            before = {p['entity']:self.sample(p['entity']) for p in members}
            body_before = self.sample(self.body_id)
            from steward.ego.assembly_edge import preflip_view
            from steward.ego import edge_result
            body_before, pre_observation = (edge_result.preflight(self, self.body_id, body_before)
                if edge_result.enabled() else preflip_view(self, self.body_id, body_before))
            now = self.ex._state()['stamp']
            if (not body_before or not all(before.values()) or
                    any(now-v['stamp'] > 20 for v in [body_before, *before.values()] if v)):
                self.warn('flip_previews_incomplete', bodyAvailable=bool(body_before), availableMembers=sum(bool(v) for v in before.values()), stamp=now)
            def independent(v):
                return v if (v and v.get('source') == 'rgbd_segment' and 0 <= now-v['stamp'] <= 20 and not v.get('truncated', True)
                             and v.get('invalidDepthFraction', 1) <= .05) else None
            geometry_view = body_before  # Merged/local geometry is not independent motion evidence.
            if edge_geometry.enabled() and not plan.get('_edgeAdjusted'):
                try:
                    edge_geometry.recover_body_face(self, record, geometry_view)
                except (AttributeError, KeyError, TypeError, ValueError, np.linalg.LinAlgError):
                    pass  # Retain the registered extent as an explicit estimate.
            from steward.ego.flip_grasp import local_pinches
            grasp_sections = local_pinches(record['reference'], geometry_view,
                self.rt.esim.arm.geometry.jaw_opening, now, loose)
            job.postcondition['graspSections'] = grasp_sections
            with self.rt._lock:
                record['reference']['localGraspSections'] = grasp_sections
                self.rt.estimator.stationary_assemblies.records[self.body_id]['reference']['localGraspSections'] = grasp_sections
            body_before = independent(body_before)
            before = {eid:independent(v) for eid,v in before.items()}
            from steward.ego.assembly_edge import geometry
            geometry_diag = {}
            if not plan.get('_edgeAdjusted'):
                record['edgeGeometry'] = geometry(record, turn_over, geometry_view, body_face=True,
                                                   stable_bearing=True, diagnostics=geometry_diag)
                if record['edgeGeometry'] is None and geometry_view and geometry_view.get('source') == 'rgbd_assembly':
                    from steward.ego.assembly_edge_observation import native_view
                    self._edge_observation_diag = {}
                    native = edge_result.cached_view(self, record) if edge_result.enabled() else native_view(self, record)
                    pre_observation['geometryRetry'] = self._edge_observation_diag.get('native', {})
                    if native:
                        retry_diag = {}
                        shape = geometry(record, turn_over, native, body_face=True,
                                         stable_bearing=True, diagnostics=retry_diag)
                        geometry_diag = dict(retry_diag, previousSource='rgbd_assembly',
                                             previousBelowBearing=geometry_diag.get('belowBearingPoints'))
                        geometry_view = native
                        record['edgeGeometry'] = shape
            hint_report = None
            if not plan.get('_edgeAdjusted') and plan.get('downwardProtrusion') is not None:
                from steward.ego.assembly_edge_hint import apply
                record['edgeGeometry'], hint_report = apply(record, turn_over, record.get('edgeGeometry'),
                                                            plan['downwardProtrusion'])
            if (edge_result.enabled() and not plan.get('_edgeAdjusted')
                    and (edge_geometry.enabled() and edge_geometry.requires_prior(record) or members or record.get('edgeGeometry') or self.body_id in self.rt.estimator.seat_assemblies)):
                record['edgeGeometry'] = (edge_geometry.prepare(record, turn_over, geometry_view, record.get('edgeGeometry'), geometry_diag)
                    if edge_geometry.enabled() else edge_result.prepare(record, turn_over, geometry_view, record.get('edgeGeometry'), geometry_diag))
            if edge_geometry.active(record.get('edgeGeometry')):
                record['reference']['edgeGraspOutward'] = record['edgeGeometry']['outwardAxis']
                from steward.ego import edge_retention
                if edge_retention.enabled() and turn_over:
                    axis = matrix_from_quat(quat_from_yaw(record['reference']['yaw']))
                    inverse_turn = axis @ np.diag([1., -1., -1.]) @ axis.T
                    record['reference']['edgeGraspOutward'] = (inverse_turn @
                        np.asarray(record['edgeGeometry']['outwardAxis'])).tolist()
            if record.get('edgeGeometry'):
                sphere = max(sphere, float(np.max(np.linalg.norm(record['edgeGeometry']['envelope'], axis=1))))
            edge_enabled = bool(record.get('edgeGeometry')) or bool(members) or self.body_id in self.rt.estimator.seat_assemblies
            job.postcondition['levelAlignment'] = dict(attempts=1, successes=0, fallbacks=0,
                fallbackReasons={}, measuredTiltDeg=None, thresholdDeg=5., adjustmentAttempts=0,
                preObservation=pre_observation)
            if edge_enabled:
                job.postcondition['edgePlacement'] = dict(used=False, estimatedTiltDeg=None,
                    observed=False, candidateCount=0, rejected={}, preObservation=pre_observation,
                    source=(record.get('edgeGeometry') or {}).get('source', (geometry_view or {}).get('source', 'none')),
                    fallback=False, fallbackResult='not_triggered', tiltBefore=None, tiltAfter=None,
                    reason=('no_protrusion_geometry' if geometry_view else 'no_fresh_local_geometry')
                    if record.get('edgeGeometry') is None else 'no_suitable_reachable_edge')
            from steward.ego import placement_support_margin
            if placement_support_margin.enabled() and not turn_over and not plan.get('_edgeAdjusted'):
                edge_enabled = False  # An ordinary transfer must keep its full bearing inside.
            if edge_enabled:
                job.postcondition['edgePlacement']['geometryDiagnostics'] = geometry_diag
            if edge_enabled and hint_report is not None:
                job.postcondition['edgePlacement']['hint'] = hint_report
            with self.rt._lock:
                self.rt.estimator.manipulation_envelopes[self.body_id] = float(2*sphere)
                if edge_enabled:
                    self.rt.estimator.stationary_assemblies.records[self.body_id]['edgeGeometry'] = deepcopy(record.get('edgeGeometry'))
                    from steward.ego import edge_retention
                    if edge_retention.enabled() and record['reference'].get('edgeGraspOutward') is not None:
                        self.rt.estimator.stationary_assemblies.records[self.body_id]['reference']['edgeGraspOutward'] = list(record['reference']['edgeGraspOutward'])
            try:
                lift_started = self.ex._state()['stamp']
                from steward.ego import grasp_binding
                for binding_try in range(2):
                    excluded = getattr(self.rt.estimator, 'rejected_grasp_points', {}).get(self.body_id, [])
                    params = dict(target=self.body_id, flip=turn_over, assemblyGrip=True)
                    if grasp_binding.enabled() and excluded: params['excludedGraspPoints'] = deepcopy(excluded)
                    try:
                        self.step(job, 'pick', params)
                        break
                    except SkillFailed as exc:
                        if (not grasp_binding.enabled() or binding_try or self.rt.estimator.held_id
                                or 'grasp mismatch: independent load geometry' not in str(exc)):
                            raise
                        self.warn('load_binding_retry_next_pinch', body=self.body_id)

                release_reference('grasp_completed')  # Release views must stay independent.
                from steward.ego import grasp_binding
                load_binding = grasp_binding.receipt(self.rt.estimator)
                if load_binding:
                    job.postcondition.update(grasp_binding.public(load_binding))
                binding_pending = bool(load_binding and not load_binding['transportCommitted'])
                from steward.ego.flip_evidence import body_grasp, resolve_face, reject_mismatch
                grasp = body_grasp(self.rt.estimator, record)
                job.postcondition['bodyGraspCheck'] = grasp
                reject_mismatch(job, grasp)
                after = self.ex._state()['stamp']; observations = []
                for _ in range(0 if binding_pending else 2):
                    # A corrective reposition reuses registrations and grip encoders;
                    # its only new observation is the normal post-release look below.
                    cached_adjustment = edge_result.enabled() and plan.get('_edgeAdjusted')
                    body_view = None if cached_adjustment else self.wait_view(self.body_id, after)
                    row = {}
                    for p in members:
                        row[p['entity']] = None if cached_adjustment else self.wait_view(p['entity'], after)
                    observations.append((body_view,row))
                    after = max([after] + [v['stamp'] for v in [body_view, *row.values()] if v])
                from steward.ego.preconditions import lift_check, fresh_view, _margin
                lift = lift_check(body_before, [v for v, _ in observations],
                    fresh_view(support_track, lift_started), lift_started,
                    self.rt.estimator.stamp, _margin(body_track, support_track))
                job.postcondition['liftCheck'] = lift
                grasp = body_grasp(self.rt.estimator, record, [v for v, _ in observations], lift_started, lift)
                job.postcondition['bodyGraspCheck'] = grasp
                reject_mismatch(job, grasp, lift)
                follow = {p['entity']:bool(not binding_pending and body_before and before[p['entity']] and all(b and rows[p['entity']] for b,rows in observations)) and follows([body_before, before[p['entity']]],
                    [[b,rows[p['entity']]] for b,rows in observations]) for p in members}
                job.postcondition['followEvidence'] = follow
                job.postcondition['followSamples'] = [
                    [{k:v for k,v in (b or {}).items() if k != 'points'},
                     {eid:{k:v for k,v in (row or {}).items() if k != 'points'} for eid,row in rows.items()}]
                    for b,rows in observations]
                if not all(follow.values()):
                    self.warn('following_unverified', observedMembers=sum(follow.values()), totalMembers=len(members))
                with self.rt._lock:
                    est = self.rt.estimator
                    gr = est.gripper_rotation(est.held_gripper)
                    gp = est.gripper_position(est.held_gripper)
                    # Keep component identities and encoder-predicted poses while carried.
                    # Independent raw RGB-D samples remain untouched for contradiction checks.
                    for eid, raw in (observations[-1][1] if observations else {}).items():
                        if not raw or not follow[eid]:
                            continue
                        est.rigid_carry[eid] = {'offset':gr.T @ (np.asarray(raw['centreM'])-gp),
                            'rotation':gr.T @ matrix_from_quat(quat_from_yaw(raw['boxYawRad'])),
                            'observedAt':raw['stamp']}
                support = record['reference']['support']
                # The original body frame is now invalid. Keep it only as a pre-motion
                # hypothesis and re-register from new views after the executed flip.
                with self.rt._lock:
                    est = self.rt.estimator
                    rot0 = np.asarray(grasp['association'].get('gripperRotationAtGrasp',
                        est.gripper_rotation(est.held_gripper)), float).copy()
                    from steward.ego import directed_orientation
                    if not binding_pending and directed_orientation.enabled() and edge_enabled and record.get('edgeGeometry'):
                        directed_orientation.bind_grasp(record, rot0)
                        est.stationary_assemblies.records[self.body_id]['reference']['directedGraspBasis'] = deepcopy(record['reference']['directedGraspBasis'])
                    est.flip_capture = None if binding_pending else {'body':self.body_id, 'rotation':rot0, 'record':deepcopy(record)}
                scene = self.rt.ego_state()
                table = scene.find(support) or record['support']
                low, high = local_bounds(record)
                half = float(max(abs(low[:2]).max(), abs(high[:2]).max()))
                last = None
                from steward.ego.fasten_simple import offset_on
                from steward.ego import edge_retention
                pinned_correction = edge_retention.enabled() and bool(record.get('edgeCorrectionTarget'))
                tables = [table] + [e for e in scene.entities if e.id != table.id and
                                   self.rt.esim._affords(e, 'support-surface')]
                candidates = []
                if edge_enabled and record.get('edgeGeometry'):
                    from steward.ego.assembly_edge_hint import options
                    for edge in options(record, table, self.rt.esim._sigma_xy(table),
                                        plan.get('downwardProtrusion'), job.postcondition['edgePlacement']):
                        candidates.append(dict(on=table.id, flip=turn_over, alignHoles=False,
                            assemblyBody=self.body_id, assemblyEdge=True, offset=edge['offset'],
                            yaw=edge['yaw'], **({'assemblyEdgeSide':[edge['edgeAxis'], edge['edgeSign']]}
                                if edge_result.enabled() else {})))
                from steward.ego import placement_body_frame
                if placement_body_frame.enabled() and edge_enabled and record.get('edgeGeometry'):
                    for edge in placement_body_frame.neighbour_options(record, table, self.rt.esim._sigma_xy(table)):
                        candidates.append(dict(on=table.id, flip=turn_over, alignHoles=False,
                            assemblyBody=self.body_id, assemblyEdge=True, assemblyEdgeNeighbours=True,
                            offset=edge['offset'], yaw=edge['yaw'],
                            assemblyEdgeSide=[edge['edgeAxis'], edge['edgeSign']]))
                if edge_result.enabled() and plan.get('_edgeRestore'):
                    candidates.insert(0, dict(plan['_edgeRestore'], flip=False, alignHoles=False,
                        assemblyBody=self.body_id))
                if edge_result.enabled() and plan.get('_edgeAvoid'):
                    avoided = plan['_edgeAvoid']
                    candidates = [c for c in candidates if not any(
                        c.get('on') == a['on'] and np.allclose(c.get('offset'), a['offset'], atol=1e-6)
                        and abs((c.get('yaw', 0.)-a['yaw']+math.pi) % (2*math.pi)-math.pi) < 1e-6
                        for a in avoided)]
                if edge_result.enabled() and plan.get('_edgeLevelSearch'):
                    offsets = [offset_on(table, root[:2]), *front_spots(self.rt.esim, scene, table, half)[:1]]
                    for offset in offsets:
                        for angle in (math.pi/2, -math.pi/2, math.pi):
                            candidates.append(dict(on=table.id, flip=False, alignHoles=False,
                                assemblyBody=self.body_id, offset=offset, yaw=record['reference']['yaw']+angle))
                if (edge_enabled or not placement_support_margin.enabled()) and edge_geometry.active(record.get('edgeGeometry')) and not pinned_correction:
                    # Try other support edges before any ordinary full-footprint
                    # fallback; never select the known tilted original pose.
                    for surface in tables[1:]:
                        for edge in options(record, surface, self.rt.esim._sigma_xy(surface), None, {}):
                            candidates.append(dict(on=surface.id, flip=turn_over, alignHoles=False,
                                assemblyBody=self.body_id, assemblyEdge=True, offset=edge['offset'],
                                yaw=edge['yaw'], assemblyEdgeSide=[edge['edgeAxis'], edge['edgeSign']]))
                for surface in ([] if pinned_correction else tables):
                    offsets = front_spots(self.rt.esim, scene, surface, half)
                    contact_window = edge_geometry.active(record.get('edgeGeometry'))
                    if surface.id == support and not contact_window:
                        offsets.insert(0, offset_on(surface, root[:2]))
                    for offset in [*offsets, None]:
                        # Only mechanical/collision exhaustion reaches these checked
                        # temporary placements. COM never routes here, even negative.
                        candidates.append(dict(on=surface.id, flip=turn_over, alignHoles=False,
                            assemblyBody=self.body_id, **({'yaw':record['reference']['yaw']+math.pi/2}
                                if contact_window else {}),
                            **({'offset':offset} if offset is not None else {})))
                for candidate in candidates:
                    counts = job.postcondition['levelAlignment']
                    if candidate.get('assemblyEdge'):
                        counts['candidateAttempts'] = counts.get('candidateAttempts', 0)+1
                    else:
                        reason = job.postcondition.get('edgePlacement', {}).get('reason', 'no_protrusion_geometry')
                        counts.update(fallbacks=1, fallbackReasons={reason:1})
                    from steward.ego.assembly_final_pose import active_restore, landing
                    if active_restore() and not candidate.get('assemblyEdge'):
                        stability = landing(record, turn_over)
                        job.postcondition['finalPoseStability'] = stability
                        if stability['passed'] is not True:
                            last = stability['reason']
                            continue
                    receipt = self.rt.predict('place', candidate)
                    if not receipt.get('withinRiskLimit'):
                        last = '; '.join(receipt.get('issues', []))
                        if candidate.get('assemblyEdge'):
                            from steward.ego.assembly_edge import _reject
                            text = last.lower()
                            why = 'unreachable' if (not receipt.get('feasible', True) or
                                any(word in text for word in ('reach', 'kinematic', 'no path'))) else 'risk'
                            if edge_result.enabled():
                                if 'no supported edge' in text:
                                    why = 'contact_revalidation'
                                elif any(word in text for word in ('occupied', 'obstacle', 'collision', 'obstruction')):
                                    why = 'collision'
                                elif not any(word in text for word in ('reach', 'kinematic', 'no path')):
                                    why = 'admission' if not receipt.get('feasible', True) else 'risk'
                            _reject(job.postcondition['edgePlacement'], why)
                        self.warn('assembly_place_candidate_failed', on=candidate['on'], offset=candidate.get('offset'), detail=last)
                        continue
                    if active_restore() and candidate.get('assemblyEdge'):
                        job.postcondition['finalPoseStability'] = dict(passed=True, observed=False,
                            source='registered_edge_support_polygon', reason='admitted_edge_contact',
                            evidence=deepcopy(receipt.get('plan', {}).get('edgePlacement')))
                    place_plan = self.step(job, 'place', candidate, receipt=receipt)
                    support = candidate['on']
                    table = next(t for t in tables if t.id == support)
                    if edge_enabled and place_plan.get('edgePlacement'):
                        job.postcondition['edgePlacement'].update(place_plan['edgePlacement'], used=True, reason='protrusions_over_edge')
                        if edge_result.enabled():
                            job.postcondition['edgePlacement']['predictedTiltDeg'] = place_plan['edgePlacement']['estimatedTiltDeg']
                    break
                else:
                    raise SkillFailed('assembly placement has no reachable table candidate: '+str(last))
                with self.rt._lock:
                    captured = self.rt.estimator.flip_capture
                    self.rt.estimator.flip_capture = None
                measured_delta = None
                if not captured or 'releaseRotation' not in captured:
                    self.warn('flip_release_frame_estimated')
                    axis = matrix_from_quat(quat_from_yaw(record['reference']['yaw']))
                    delta = axis @ np.diag([1.,-1.,-1.]) @ axis.T if turn_over else np.eye(3)
                    if place_plan.get('edgePlacement'):
                        delta = matrix_from_quat(quat_from_yaw(place_plan['edgePlacement']['yawDeltaRad'])) @ delta
                    after = self.ex._state()['stamp']
                else:
                    delta = captured['releaseRotation'] @ rot0.T
                    measured_delta = delta.copy()
                    job.postcondition.setdefault('flipRotationEvents', []).append(dict(
                        source='grasp_to_release_wrist_encoders', observed=False,
                        upDot=float(delta[2, 2]), reversed=bool(delta[2, 2] < 0),
                        turnOver=bool(turn_over), bodyIdentityVerified=grasp['identityVerified'],
                        bodyGraspStatus=grasp['status']))
                    after = captured['releasedAt']
                fresh = self.wait_view(self.body_id, after)
                if binding_pending:
                    # Motion and the post-action view have completed. Neither that
                    # motion nor an unbound release authorizes a body-frame rebase.
                    job.postcondition.update(grasp_binding.public(load_binding),
                        executed=True, newReleaseExecuted=True, releaseExecuted=False,
                        specifiedTargetReleased=False, verified=False, referenceUnchanged=True,
                        reference=public_json(record['reference']), resultChecked=True,
                        postActionObservation=plain_json(fresh),
                        guidance='动作已完成，目标身份未确认；保留原参考，按动作后观测继续补救。')
                    return
                orientation_observed = False
                if turn_over and (delta @ np.array([0.,0.,1.]))[2] > -math.cos(math.radians(10)):
                    second = self.wait_view(self.body_id, fresh['stamp'] if fresh else after)
                    settled = settled_flip_rotation(delta, body_before, [fresh,second]) if body_before and fresh and second else None
                    if settled is None:
                        self.warn('flip_face_unverified', releaseUpZ=float(delta[2,2]))
                        # Nominal completed half turn: preserve the observed horizontal
                        # axis as best possible. This transports priors, not evidence.
                        u, _, vt = np.linalg.svd(delta[:2,:2])
                        xy = u @ vt
                        if np.linalg.det(xy) > 0:
                            u[:,-1] *= -1
                            xy = u @ vt
                        delta = np.zeros((3,3)); delta[:2,:2] = xy; delta[2,2] = -1
                    else:
                        delta = settled; fresh = second; orientation_observed = True
                        job.postcondition['faceReversalObserved'] = True
                # Rebase directly from the completed release. Old pre-flip tracks and
                # registrations cannot supply a stationary reference for the next face.
                scene = self.rt.ego_state()
                table = scene.find(support) or table
                release_xy = place_plan['spot']
                measured_position = captured.get('releasePosition') if captured else None
                if measured_position is not None:
                    residual = float(np.linalg.norm(np.asarray(measured_position)[:2]-release_xy))
                    job.postcondition['releasePositionCheck'] = dict(commandXY=release_xy,
                        encoderPosition=measured_position, residualM=residual,
                        applied=residual <= .03, observed=False)
                    if edge_geometry.enabled() and place_plan.get('edgeReleaseHeight'):
                        job.postcondition['releasePositionCheck']['residualZM'] = float(measured_position[2]-place_plan['edgeReleaseHeight']['targetBodyZM'])
                    # Large disagreement may be a slipped/aliased grasp. Do not move
                    # every interface to that hypothesis without an independent view.
                    if residual <= .03:
                        release_xy = measured_position[:2]
                new = released_record(record, release_xy, table, delta, after, fresh)
                from steward.ego import directed_orientation
                if directed_orientation.enabled(): new['reference'].pop('directedGraspBasis', None)
                if place_plan.get('edgePlacement'):
                    from steward.ego.assembly_edge import rebase_height
                    rebase_height(new, record, delta, place_plan['edgePlacement'])
                    if edge_geometry.enabled():
                        from steward.ego.planner import entity_box
                        sb, _, _ = entity_box(table)
                        new['edgeSupport'] = dict(centre=[sb.cx, sb.cy], half=[sb.hx, sb.hy], yaw=sb.yaw)
                        if edge_retention.enabled():
                            new['edgeAnchor'] = deepcopy(place_plan['edgePlacement'])
                            new.pop('edgeCorrectionTarget', None)
                from steward.ego import assembly_release_height
                if assembly_release_height.enabled() and place_plan.get('assemblyReleaseHeight'):
                    assembly_release_height.rebase(new, record, delta, place_plan['assemblyReleaseHeight'])
                new['reference']['positionSource'] = 'release_fk' if measured_position is not None and residual <= .03 else 'command_spot'
                new_ref = new['reference']
                # One cheap look from the release retreat. Neither a half-turn command
                # nor propagated tracks count as a visually completed flip.
                self.observation_target = list(new_ref['position'])
                if edge_geometry.enabled():
                    self.observation_target[2] = new_ref.get('landingTopM', new_ref['position'][2])
                settle_view = deepcopy(fresh)
                self.look(after, cycles=1)
                with self.rt._lock:
                    fresh_bodies = []
                    for t in self.rt.estimator.tracks.values():
                        v = t.manipulation_view
                        if (not v or t.held or t.category not in (record['reference']['category'], record['reference'].get('family'))
                                or v['stamp'] <= after or self.rt.estimator.stamp-v['stamp'] > 20
                                or v.get('truncated', True) or v.get('invalidDepthFraction', 1) > .05
                                or not np.allclose(np.sort(v['visibleSizeM'][:2]),
                                                   np.sort(record['reference']['size'][:2]), atol=.025)):
                            continue
                        d = min(np.linalg.norm(np.asarray(v['centreM'])[:2]-root[:2]),
                                np.linalg.norm(np.asarray(v['centreM'])[:2]-new_ref['position'][:2]))
                        if d <= .10:
                            fresh_bodies.append(deepcopy(v))
                fresh = fresh_bodies[0] if len(fresh_bodies) == 1 else None
                from steward.ego.assembly_motion import flip_check
                check = flip_check(body_before, fresh, before,
                    {p['entity']:self.sample(p['entity']) for p in members}, after, self.rt.estimator.stamp)
                if (fresh and np.linalg.norm(root[:2]-new_ref['position'][:2]) > .10 and
                        np.linalg.norm(np.asarray(fresh['centreM'])[:2]-root[:2]) < .04):
                    check = dict(passed=False, status='fail', reason='body_still_at_pickup',
                                 observedPositionM=fresh['centreM'], expectedPositionM=new_ref['position'])
                if check['passed'] is None and orientation_observed:
                    check = dict(passed=True, status='ok', reason='rgbd_settled_face_reversal')
                job.postcondition['orientationCheck'] = check
                from steward.ego.assembly_status import face_advice
                job.postcondition['faceAdvice'] = face_advice(check)
                orientation_observed = check['passed'] is True
                if edge_enabled:
                    from steward.ego.assembly_edge import observe_level
                    self._edge_measure_record = new
                    self._edge_contact_measure = edge_geometry.enabled()
                    try:
                        face_level = (edge_result.measure(self, new_ref, fresh, after) if edge_result.enabled()
                                      else observe_level(self, new_ref, fresh, after))
                    finally:
                        self.__dict__.pop('_edge_measure_record', None)
                        self.__dict__.pop('_edge_contact_measure', None)
                    job.postcondition['edgePlacement'].update(
                        {k:v for k,v in face_level.items() if k != 'reason'}, levelReason=face_level['reason'])
                    from steward.ego import directed_orientation
                    if directed_orientation.enabled():
                        # Keep the existing operational flatness tolerance. A very
                        # precise plane fit must not reinterpret harmless surface
                        # slope as evidence that the directed end was reversed.
                        tolerance = job.postcondition['levelAlignment']['thresholdDeg']
                        angle = face_level.get('estimatedTiltDeg')
                        job.postcondition['edgePlacement']['measuredLevel'] = (face_level.get('level') is True or
                            bool(face_level.get('observed') and angle is not None and angle <= tolerance))
                        job.postcondition['edgePlacement']['measuredLevelToleranceDeg'] = tolerance
                    if (settle_view and fresh and after < settle_view['stamp'] < fresh['stamp']
                            and not settle_view.get('truncated', True)
                            and settle_view.get('invalidDepthFraction', 1) <= .05):
                        movement = float(np.linalg.norm(np.asarray(fresh['centreM'])-settle_view['centreM']))
                        job.postcondition['edgePlacement']['settlingMotionM'] = movement
                        if movement > .008:
                            job.postcondition['edgePlacement'].update(level=False, levelReason='post_release_motion')
                if check['passed'] is False and fresh:
                    # A measured unchanged face invalidates the commanded turn
                    # hypothesis. Keep the observed location and the previous face.
                    landing_table = record['support'] if check['reason'] == 'body_still_at_pickup' else table
                    delta = np.eye(3)
                    new = released_record(record, fresh['centreM'][:2], landing_table, delta, after, fresh)
                    new_ref = new['reference']
                new_ref['faceCheck'] = deepcopy(check)
                # Internal records/scenes retain all points for later face checks.
                # Receipts and JSON exports take the public view instead.
                new_ref['faceBefore'] = plain_json(body_before)
                new_ref['faceSince'] = after
                new_ref['rotationBefore'] = deepcopy(record['reference'].get('rotationM', np.eye(3).tolist()))
                resolution = resolve_face(record['reference'], measured_delta, grasp, check, after)
                new_ref['faceResolution'] = deepcopy(resolution)
                job.postcondition['faceResolution'] = resolution
                if resolution['status'] in ('estimated', 'observed'):
                    job.postcondition['faceAdvice'].update(nextAction='continue_selected_interface',
                        interfaceSource=resolution['source'], exposedInterfaces=resolution['exposedInterfaces'])
                with self.rt._lock:
                    est = self.rt.estimator
                    for eid, frame in released_frames.items():
                        if frame['body'] != self.body_id:
                            continue
                        def moved(p):
                            return (np.asarray(new_ref['position']) + delta @ (np.asarray(p)-root)).tolist()
                        frame['position'] = moved(frame['position'])
                        frame['rotation'] = (delta @ np.asarray(frame['rotation'])).tolist()
                        for h in frame['holes']:
                            h['point'] = moved(h['point'])
                        if 'bodyHolePoints' in frame:
                            frame['bodyHolePoints'] = {h:moved(p) for h,p in frame['bodyHolePoints'].items()}
                        frame.pop('alignmentEvidence', None)
                        frame.update(releasedAt=after, transportedAt=after, observed=False, outlineObserved=False,
                                     source='release_fk_transported_by_body_motion')
                        est.released_interfaces[eid] = frame
                    for eid in {self.body_id, *(p['entity'] for p in members)}:
                        est.stationary_assemblies.records.pop(eid, None)
                        est.assembly_memory.invalidate(eid)
                    if self.body_id not in est.tracks:
                        self.warn('body_track_restored_after_flip', entity=self.body_id)
                    t = restore_track(est, new)
                    t.held = False
                    t.pos = np.asarray(new_ref['position'])
                    t.yaw = new_ref['yaw']
                    t.size = np.asarray(new_ref['size'])
                    t.interface_views.clear(); t.interface_best_near = None
                    est.stationary_assemblies.records[self.body_id] = new
                    if support == record['reference']['support'] and support not in est.tracks and support_track is not None:
                        est.tracks[support] = support_track
                    for a in attempts.values():
                        if a['body'] == self.body_id:
                            a['entry'] = (np.asarray(new_ref['position'])+delta @ (np.asarray(a['entry'])-root)).tolist()
                            a['transportedAt'] = after
                            from steward.ego.hole_seating import transported
                            transported(a)
                            a['headPatchBefore'] = None
                            a['componentFrame'] = deepcopy(est.released_interfaces.get(a['part']))
                    job.postcondition.update(verified=(orientation_observed if turn_over else bool(members) and all(follow.values())),
                        executed=True, releaseExecuted=True, body=self.body_id, reference=public_json(new_ref),
                        flipped=check['passed'] if turn_over else False, commandedTurnOver=turn_over,
                        orientationSource='observed_face_change' if check['passed'] is True else
                            'commanded_relative_half_turn' if check['passed'] is None else 'observed_unchanged_face', upright=None,
                        passed=check['passed'] if turn_over else None)
                    # Independently observed following also confirms seat-mode attachment.
                    for a in attempts.values():
                        if a['body'] == self.body_id and follow.get(a['part']):
                            a.update(verified=True, status='following_observed')
                if (edge_enabled and not plan.get('_edgeAdjusted') and
                        (job.postcondition['edgePlacement'].get('level') is False or
                         job.postcondition['edgePlacement']['used'] and job.postcondition['edgePlacement'].get('level') is None)):
                    from steward.ego.assembly_edge import adjust_once
                    if edge_geometry.enabled():
                        edge_geometry.adjust(self, job, plan)
                    elif edge_result.enabled():
                        edge_result.adjust(self, job, plan)
                    else:
                        adjust_once(self, job, plan)
                    final_ref = self.rt.estimator.stationary_assemblies.records[self.body_id]['reference']
                    if final_ref.get('ownReleaseAt', after) > after:
                        # Repositioning can change the face again. Only a view after
                        # the final release can support the final orientation claim.
                        check = flip_check(body_before, self.sample(self.body_id), before,
                            {p['entity']:self.sample(p['entity']) for p in members},
                            final_ref['ownReleaseAt'], self.rt.estimator.stamp)
                        job.postcondition['orientationCheck'] = check
                        job.postcondition['faceAdvice'] = face_advice(check)
                        job.postcondition['faceResolution'] = deepcopy(final_ref.get('faceResolution', {}))
                from steward.ego.assembly_edge import observe_level
                edge = job.postcondition.get('edgePlacement', {})
                from steward.ego import directed_orientation
                if directed_orientation.enabled() and edge_enabled:
                    edge['orientationDecision'] = directed_orientation.classify(edge)
                measured = edge if edge_enabled else (edge_result.measure(self, new_ref, fresh, after)
                    if edge_result.enabled() else observe_level(self, new_ref, fresh, after))
                if edge_result.enabled():
                    edge.setdefault('finalTiltDeg', measured.get('estimatedTiltDeg'))
                    if edge_geometry.enabled():
                        edge['tiltAfter'] = measured.get('estimatedTiltDeg')
                        edge.setdefault('tiltBefore', measured.get('estimatedTiltDeg'))
                    displacement = np.asarray(job.postcondition['reference']['position'][:2])-root[:2]
                    edge['translationM'] = float(np.linalg.norm(displacement))
                    edge['translationXYM'] = displacement.tolist()
                    edge['unconfirmed'] = (edge.get('unconfirmed', False) if edge_geometry.enabled() else False) or not measured.get('observed', False)
                if edge_retention.enabled() and not plan.get('_edgeAdjusted') and not self.rt.estimator.held_id:
                    edge_retention.retain(self.rt.estimator.stationary_assemblies.records[self.body_id], edge)
                counts = job.postcondition['levelAlignment']
                reason = edge.get('reason', 'no_protrusion_geometry')
                counts.update(successes=int(measured.get('level') is True),
                    fallbacks=int(not edge.get('used', False)),
                    fallbackReasons={} if edge.get('used') else {reason:1},
                    measuredTiltDeg=measured.get('estimatedTiltDeg'),
                    observationReason=measured.get('levelReason', measured.get('reason', 'no_fresh_local_face')),
                    adjustmentAttempts=counts['adjustmentAttempts'])
                from steward.ego.preconditions import finish_flip
                finish_flip(job, check, lift, turn_over=turn_over)
                if job.postcondition.get('faceResolution', {}).get('status') in ('estimated', 'observed'):
                    job.postcondition['guidance'] = ('已由机器人证据确定朝上接口，可继续。' if check['passed'] is True else
                        '已由编码器与接口布局确定朝上接口；独立视觉未确认，可继续。')
                    job.message = job.postcondition['guidance']
            finally:
                with self.rt._lock:
                    # While still held, retain the full load envelope for safe recovery.
                    if self.rt.estimator.held_id != self.body_id:
                        self.rt.estimator.manipulation_envelopes.pop(self.body_id, None)
                    self.rt.estimator.flip_capture = None

    def run(self, job, plan):
        skill = plan['fastenSkill']
        if _assembly_enabled():
            if skill == 'flip_assembly':
                from steward.ego import first_interface
                first_interface.guard_runtime_turn(self.rt, plan['body'])
            if skill == 'fasten_onto':
                self._seat_only = not plan.get('drive', True)
                if self._seat_only:
                    self.rt.estimator.seat_assemblies.add(plan['body'])
                else:
                    self.rt.estimator.seat_assemblies.discard(plan['body'])
            elif skill == 'seat_screw' and not hasattr(self, '_seat_only'):
                self._seat_only = True
            if skill == 'seat_screw' and getattr(self, '_seat_only', False):
                self.rt.estimator.seat_assemblies.add(plan['body'])
            if getattr(self, '_seat_only', False):
                self._screw_misses = getattr(self, '_screw_misses', {})
            from steward.ego.fasten_plan import lower_interface
            self._lower_interface = lower_interface(self.rt.esim, self.rt.ego_state().find(plan.get('body')),
                set(plan.get('selectedHoleIds', [plan.get('hole')]))) if skill in ('fasten_onto', 'seat_screw') else False
            from steward.ego import working_face
            scene = self.rt.ego_state()
            if self._lower_interface and working_face.ordinary_path(self.rt.esim, scene,
                    scene.find(plan.get('body')), plan.get('selectedHoleIds', [plan.get('hole')])):
                self._lower_interface = False  # Exposed face uses the ordinary component/fastener path.
        self._observation_job = job
        job.postcondition = {'skill':skill, 'verified':False, 'stages':[]}
        if _assembly_enabled():
            release_serial = getattr(self.rt.estimator, '_load_release_serial', 0)
            historical_release = bool(release_serial or any(
                a.get('releaseExecuted') or a.get('releaseHistory') or a.get('loadReleaseHistory')
                for a in self.rt.estimator.fastening_attempts.values()))
            try:
                action = {'seat_screw':self.seat, 'drive_screw':self.drive,
                    'fasten_onto':self.fasten, 'flip_assembly':self.flip, 'regrasp_upright':self.regrasp_upright}[skill]
                if skill == 'drive_screw':
                    from steward.ego.fasten_sequence import bounded
                    from steward.ego.fasten_sequence import require_attempt
                    require_attempt(self, job, plan)
                    bounded(self, job, lambda:action(job, plan))
                else:
                    action(job, plan)
            except Exception as exc:
                job.postcondition['failure'] = str(exc)
                if skill == 'flip_assembly' and 'flip_result' not in getattr(job, '_precondition_events', {}):
                    from steward.ego.preconditions import empty_report, result, record_job
                    report = empty_report()
                    report['flip'] = result(reason='flip_motion_incomplete', message='翻面动作未完成，结果未确认。')
                    record_job(job, report, 'flip_result')
                    job.postcondition.update(actionStatus='failed', confirmation='unconfirmed', verified=False)
                with self.rt._lock:
                    attempts = [a for a in self.rt.estimator.fastening_attempts.values()
                                if not plan.get('body') or a['body'] == plan['body']]
                    if skill == 'fasten_onto':
                        selected = set(job.postcondition.get('selectedHoleIds', []))
                        attempts = [a for a in attempts if a['hole'] in selected]
                    job.postcondition['attempts'] = deepcopy(attempts)
                    if skill == 'seat_screw':
                        selected = set(job.postcondition.get('selectedHoleIds', [plan.get('hole')]))
                        job.postcondition['confirmedSeatedScrews'] = sum(
                            bool(a.get('seatConfirmed')) for a in attempts if a['hole'] in selected)
                raise
            finally:
                from steward.ego import grasp_binding
                if grasp_binding.enabled() and (getattr(self.rt.estimator, 'load_bindings', {})
                        or hasattr(self.ex, '_verify_load_after_lift')):
                    job.postcondition['newReleaseExecuted'] = getattr(self.rt.estimator, '_load_release_serial', 0) > release_serial
                    job.postcondition['historicalReleaseExecuted'] = historical_release
                sr.publish(self,job)
                from steward.ego.assembly_status import assembly_state
                if skill == 'fasten_onto':
                    from steward.ego.fasten_supply import hole_progress
                    selected = set(job.postcondition.get('selectedHoleIds', plan.get('selectedHoleIds', [])))
                    attempts = [a for a in self.rt.estimator.fastening_attempts.values()
                                if a['body'] == plan['body'] and a['hole'] in selected]
                    progress = hole_progress(selected, attempts,
                        job.postcondition.get('part', plan['part']), plan['body'])
                    if plan.get('drive', True) and not progress.get('failedHoles'):
                        progress.pop('interfaceSummary', None)
                    job.postcondition.update(progress)
                    job.postcondition.update(
                        introducedScrewCount=sum(bool(a.get('releaseExecuted')) for a in attempts),
                        releaseEventCount=sum(len(a.get('releaseHistory', [a])) for a in attempts if a.get('releaseExecuted')))
                if skill == 'fasten_onto' and plan.get('drive', True):
                    from steward.ego.fasten_driver import progress
                    selected = set(job.postcondition.get('selectedHoleIds', plan.get('selectedHoleIds', [])))
                    attempts = [a for a in self.rt.estimator.fastening_attempts.values() if a['body'] == plan['body']]
                    job.postcondition.update(progress(selected, attempts,
                        job.postcondition.get('part', plan['part']), plan['body']))
                state = assembly_state(self.rt, plan, job.postcondition)
                if skill == 'flip_assembly':
                    state['guidance'] = job.postcondition.get('guidance', '翻面未确认。')
                    if job.postcondition.get('edgePlacement'):
                        # Completed execution paths are bulky and repeated on the
                        # one allowed adjustment. Retain admission and release geometry.
                        for stage in job.postcondition.get('stages', []):
                            saved = stage.get('receipt', {})
                            from steward.ego.edge_geometry import enabled as contact_window_enabled
                            if contact_window_enabled():
                                saved.pop('warnings', None)  # Already in the parent warning ledger.
                            # Admission remains reviewable; omit repeated request and
                            # generic precondition prose from completed child stages.
                            for key in ('params', 'esVersion', 'createdAt', 'expiresAt', 'maxRisk', 'preconditions'):
                                saved.pop(key, None)
                            if 'plan' in saved:
                                saved['plan'] = {k:v for k,v in saved['plan'].items() if k in (
                                    'object', 'target', 'support', 'spot', 'yaw', 'lift',
                                    'assemblyTurnLift', 'edgePlacement', 'wristRoll', 'wristYaw',
                                    'edgeLowRelease', 'edgeDescentM', 'edgeVerticalClearanceM',
                                    'edgeGripperClearanceM', 'edgeReleaseUnconfirmed', 'releaseXYCheck', 'edgeReleaseHeight',
                                    'assemblyReleaseHeight', 'assemblyVerticalRelease')}
                                if contact_window_enabled() and saved['plan'].get('edgeReleaseHeight'):
                                    saved['plan']['edgeReleaseHeight'] = {k:v for k,v in
                                        saved['plan']['edgeReleaseHeight'].items() if k in ('targetBodyZM', 'commandedLiftM')}
                                if contact_window_enabled() and saved['plan'].get('edgePlacement'):
                                    for duplicate in ('bearingPolygonWorld', 'massSource', 'comOffsetWorldM', 'observed',
                                            'orientationKinematics', 'protrusionEndSource',
                                            'contactWindow', 'geometryConfirmed', 'unconfirmed', 'bodyGraspZM',
                                            'estimatedTiltDeg', 'offset', 'spot', 'yaw', 'bearingZM', 'comUncertaintyM',
                                            'tableComMarginM', 'bearingComMarginM'):
                                        saved['plan']['edgePlacement'].pop(duplicate, None)

                from steward.ego import edge_retention
                edge_trials = len(job.postcondition.get('edgePlacement', {}).get('adjustments', []))
                if skill == 'flip_assembly' and (edge_trials > 1 or edge_retention.enabled() and edge_trials):
                    for stage in job.postcondition.get('stages', []):
                        saved = stage.get('receipt', {})
                        if saved.get('plan', {}).get('edgePlacement'):
                            detail = saved['plan']['edgePlacement']
                            saved['plan']['edgePlacement'] = {k:v for k,v in detail.items() if k in (
                                'edgeAxis','edgeSign','comMarginM','requiredComMarginM','protrusionClearanceM',
                                'comOffsetWorldM','bodyCentreZM','bearingRole','yawDeltaRad','validation',
                                'protrusionEndSource','plannedYawRad','orientationKinematics')}
                        # Detailed grasp evidence already appears in the final bodyGraspCheck.
                        if 'endpointGraspEvidence' in stage:
                            stage['endpointGraspEvidence'] = {k:v for k,v in stage['endpointGraspEvidence'].items()
                                if k in ('status','identityVerified','observed','reason','source')}

                from steward.ego import assembly_release_height
                if skill == 'flip_assembly' and assembly_release_height.enabled():
                    assembly_release_height.compact_stages(job.postcondition)

                if (job.postcondition.get('driveResumeSuggestion') and not self.rt.estimator.held_id
                        and not job.postcondition.get('failedHoles')
                        and not job.postcondition.get('driveBudget', {}).get('timedOut')):
                    state['nextAction'] = dict(job.postcondition['driveResumeSuggestion'], reason='resume_missing_powered_drives')
                if job.postcondition.get('driveBudget', {}).get('timedOut') and not self.rt.estimator.held_id:
                    state['nextAction'] = dict(skill='observe', reason='drive_budget_exhausted_continue_remaining_task_steps')
                if job.postcondition.get('driveGate', {}).get('nextAction'):
                    state['nextAction'] = deepcopy(job.postcondition['driveGate']['nextAction'])
                if skill == 'fasten_onto' and getattr(self, '_seat_only', False):
                    from steward.ego.fasten_seat import summary
                    from steward.ego.fasten_supply import hole_progress
                    selected = set(job.postcondition.get('selectedHoleIds', plan.get('selectedHoleIds', [])))
                    rows = [a for a in self.rt.estimator.fastening_attempts.values() if a['body'] == plan['body']]
                    part = job.postcondition.get('part', plan['part'])
                    progress = hole_progress(selected, rows, part, plan['body'])
                    line = progress.get('interfaceSummary') or summary(part, plan['body'], progress['holeProgress'])
                    if job.postcondition.get('materialStatus'):
                        line += (' 本套紧固件不足，不去其他工作面取料。')
                    job.postcondition = {'interfaceSummary':line, **job.postcondition}
                    state['guidance'] = line
                    job.message = line
                job.postcondition = {**({'interfaceSummary':job.postcondition['interfaceSummary']}
                    if 'interfaceSummary' in job.postcondition else {}),
                    **({'flipAdvice':job.postcondition['flipAdvice']} if 'flipAdvice' in job.postcondition else {}),
                    'assemblyState':state, **job.postcondition}
                job.postcondition['retry'] = dict(allowed=True, before=state['nextAction'],
                    reason='use new perception and a fresh receipt; preserve confirmed holes')
                from steward.ego.assembly_precision import publish
                publish(job)
                if job.postcondition.get('precisionSummary'):
                    job.message = job.postcondition['precisionSummary']
                from steward.ego.preconditions import publish
                publish(job)
                from steward.ego.fasten_completion import REPORT
                if REPORT and job.postcondition.get('remainingHoles') and not job.postcondition.get('canContinue'):
                    job.postcondition.update(canContinue=False, confirmation='unconfirmed')
                    state['nextAction']=dict(skill='fasten_onto',part=plan.get('part'),body=plan.get('body'),
                        holes=job.postcondition['remainingHoles'],reason='complete_unconfirmed_interface')
                    state['guidance']='尚有未确认孔；按逐孔重试上限观察或拾回原颗，不向未知占用孔补另一颗。'
            job.message = ((job.postcondition["interfaceSummary"] + "\n" +
                            "明确未入座的孔可以补一次（同一颗）；未确认的不补；保留已放上件。")
                           if job.postcondition.get("interfaceSummary") and
                           job.postcondition.get("confirmedSeatedScrews", 0) < job.postcondition.get("requestedScrews", 0) else
                           job.postcondition.get("interfaceSummary")) or (f"{skill}: action executed; visually confirmed {job.postcondition.get('confirmedSeatedScrews', '?')}/"
                           f"{job.postcondition.get('requestedScrews', '?')}; "
                           f"next={state['nextAction']['skill']}; "
                           f"{job.postcondition.get('guidance', 'continue remaining task steps; attachment unverified')}")

            from steward.ego.fasten_completion import REPORT
            if REPORT and job.postcondition.get('interfaceSummary'):
                job.message = job.postcondition['interfaceSummary']
            if skill == 'flip_assembly':
                job.message = job.postcondition.get('guidance', '翻面未确认。')
            if skill == 'flip_assembly' and job.postcondition.get('edgePlacement'):
                edge = job.postcondition['edgePlacement']
                if edge.get('contactWindow'):
                    edge.pop('bearingPolygonWorld', None)
                angle = edge.get('estimatedTiltDeg')
                job.message = ('边沿放置：'+('已采用' if edge['used'] else '未采用，退回原放置')+
                    '；估计倾角 '+(f'{angle:.1f}°' if angle is not None else '未知')+
                    '；'+edge['reason']+'。\n'+job.message)
            if skill == 'flip_assembly' and job.postcondition.get('flipAdvice', {}).get('remainingHoles'):
                job.message = job.postcondition['flipAdvice']['message'] + '\n' + job.message

            if job.postcondition.get('precisionSummary'):
                job.message += '\n' + job.postcondition['precisionSummary']
        else:
            try:
                {'seat_screw':self.seat, 'drive_screw':self.drive,
                 'fasten_onto':self.fasten, 'flip_assembly':self.flip, 'regrasp_upright':self.regrasp_upright}[skill](job,plan)
            except Exception as exc:
                job.postcondition['failure'] = str(exc)
                with self.rt._lock:
                    attempts = [a for a in self.rt.estimator.fastening_attempts.values()
                                if not plan.get('body') or a['body'] == plan['body']]
                    job.postcondition['attempts'] = deepcopy(attempts)
                    if skill == 'seat_screw':
                        selected = set(job.postcondition.get('selectedHoleIds', [plan.get('hole')]))
                        job.postcondition['confirmedSeatedScrews'] = sum(
                            bool(a.get('seatConfirmed')) for a in attempts if a['hole'] in selected)
                raise
            job.message = f'{skill}: see sensory evidence and unresolved limits in postcondition'
