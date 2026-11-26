"""Receipt planning for staged fastening and precise, measured feature contact."""

from steward.assembly_profile import enabled as _assembly_enabled

from itertools import product
import math
import numpy as np

from steward.ego.esim import PredictionError
from steward.ego.fasten_geometry import surface_patch, nominal_tool
from steward.ego.planner import entity_box
from steward.math3d import matrix_from_quat, quat_from_yaw

SKILLS = ('regrasp_upright', 'seat_screw', 'drive_screw', 'fasten_onto', 'flip_assembly')


def value(entity, key, default=None):
    item = entity.state_vars.get(key) if entity else None
    return item.value if item else default


def tool_estimate(entity, stamp):
    feature = value(entity, 'toolFeature')
    if feature or entity is None:
        return feature
    box, low, high = entity_box(entity)
    return nominal_tool(entity.category.value, entity.pose.value['p'],
                        [2*box.hx, 2*box.hy, high-low],
                        matrix_from_quat(entity.pose.value['q']), stamp)


def interface_holes(planner, entity):
    from steward.ego.hole_registration import register_hole_template
    from steward.ego.priors import SCREW_HOLE_FRAMES, screw_holes
    holes = planner._screw_holes(entity)
    if _assembly_enabled():
        if holes and all(h.get('frame') == 'entity' for h in holes):
            # A visible face does not erase the catalogue's other interfaces. Keep
            # measured groups intact (including incomplete ones); add only entirely
            # hidden groups as registered priors, with their original provenance.
            box, low, high = entity_box(entity)
            template, _ = register_hole_template(screw_holes(entity.category.value),
                SCREW_HOLE_FRAMES.get(entity.category.value),
                [2*box.hx, 2*box.hy, high-low], best_effort=True)
            catalogue = {h['id']: h for h in template}
            holes = [{**{k: v for k, v in catalogue.get(h['id'], {}).items() if k in ('group', 'mates')},
                      **h} for h in holes]
            visible = {h.get('group') for h in holes}
            ids = {h['id'] for h in holes}
            return holes + [h for h in template if h['id'] not in ids and h.get('group') and h['group'] not in visible]
    else:
        if holes and all(h.get('frame') == 'entity' for h in holes):
            return holes
    box, low, high = entity_box(entity)
    category = entity.category.value
    holes, _ = register_hole_template(screw_holes(category), SCREW_HOLE_FRAMES.get(category),
                                      [2*box.hx, 2*box.hy, high-low], best_effort=True)
    return holes


def mating_holes(planner, es, plan):
    """Select the component's compatible interface, never all unrelated body holes."""
    if _assembly_enabled():
        if plan.get('resumeReleased'):
            return [dict(id=h) for h in plan['selectedHoleIds']]
    body, part = es.entity(plan['body']), es.entity(plan['part'])
    bh = planner._select_holes(interface_holes(planner, body), plan.get('selectedHoleIds', plan.get('hole')), body.id)
    if plan.get('selectedHoleIds'):
        return bh
    ph = planner._select_holes(interface_holes(planner, part), plan.get('heldHole'), part.id)
    pairs, grouped = planner._hole_group_pairs(ph, bh,
        plan.get('hole') is None or plan.get('heldHole') is None)
    if grouped and pairs:
        # Same geometric choice as placement, including its minimum-turn tie break.
        fits = planner._hole_fit_candidates(part, ph, body, bh)
        _, _, matched = planner._choose_hole_fit(fits, 'offset',
            preferred_yaw=entity_box(part)[0].yaw)
        plan['selectedHeldHoleIds'] = [a for a, _ in matched]
        ids = {b for _, b in matched}
        return [h for h in bh if h['id'] in ids]
    if grouped:
        raise PredictionError('no compatible component interface')
    return bh


def support_anchor(planner, es, entity, warnings):
    anchor = planner._anchor(es, entity)
    if anchor is not entity and planner._affords(anchor, 'support-surface'):
        return anchor
    candidates = [e for e in es.entities if e.id != entity.id and planner._affords(e, 'support-surface')]
    if not candidates:
        raise PredictionError('no support geometry available for checked contact approach')
    anchor = min(candidates, key=lambda e: np.linalg.norm(
        np.asarray(e.pose.value['p'])[:2]-entity.pose.value['p'][:2]))
    warnings.append(dict(reason='support_relation_estimated', entity=entity.id, support=anchor.id))
    return anchor


def fresh_view(entity, stamp):
    view = value(entity, 'contactView')
    if not view:
        return None
    warnings = []
    if stamp-view['stamp'] > 20 or view['truncated'] or view['invalidDepthFraction'] > .05:
        warnings.append(dict(reason='contact_view_quality', ageS=stamp-view['stamp'],
            truncated=view['truncated'], invalidDepthFraction=view['invalidDepthFraction']))
    return {**view, 'warnings':warnings}


def hole_point(planner, entity, selector):
    holes = planner._select_holes(interface_holes(planner, entity), selector, entity.id)
    if len(holes) != 1 or holes[0].get('frame') != 'entity':
        raise PredictionError('seat_screw needs one registered body hole id, not a group or unregistered prior')
    h = holes[0]
    # XY only: outline registration does not calibrate template root Z.
    r = matrix_from_quat(entity.pose.value['q'])
    point = np.asarray(entity.pose.value['p']) + r @ np.asarray(h['p'])
    return point[:2]


if _assembly_enabled():
    def lower_interface(planner, body, selected):
        """A lower layer of opposed interfaces, independent of labels or world pose."""
        if body is None:
            return False
        holes = interface_holes(planner, body)
        levels = [float(h['p'][2]) for h in holes]
        chosen = [float(h['p'][2]) for h in holes if h['id'] in selected]
        return bool(levels and chosen and max(levels)-min(levels) > .01
                    and max(chosen) < (min(levels)+max(levels))/2-.005)


def seat_target(planner, es, params):
    from steward.ego.fasten_release import released_interface
    frame = released_interface(es, params.get('part'), params.get('body'))
    if _assembly_enabled():
        if frame:
            pair = next((p for p in frame['holePairs'] if p['bodyHole'] == params.get('hole')), None)
            h = next((h for h in frame['holes'] if pair and h['id'] == pair['partHole']), None)
            if h is None:
                raise PredictionError('release frame has no correspondence for this interface hole')
            body = es.find(params.get('body'))
            pair_error = None
            if body:
                try:
                    base_point = frame.get('bodyHolePoints', {}).get(params['hole'])
                    base_xy = base_point[:2] if base_point is not None else hole_point(planner, body, params['hole'])
                    pair_error = float(np.linalg.norm(np.asarray(h['point'])[:2]-base_xy))
                except PredictionError:
                    pass  # A lost body catalogue must not veto the saved release target.
            result = dict(point=list(h['point']), normal=[0.,0.,1.], partHole=h['id'],
                        pairAlignment=dict(distanceXYM=pair_error, source='registered_frames',
                            observed=False, throughBothLayers=None,
                            guidance='上件孔落位不证明穿过两层；上下孔存在偏移时需重新对孔'),
                        observedAt=frame['observedAt'], releasedAt=frame['releasedAt'],
                        targetSource=frame['source'], componentFrame=frame, warnings=[
                            dict(reason=('component_interface_estimate' if frame.get('outlineObserved') or
                                         frame.get('alignmentMovedAt') is not None else 'release_fk_interface_estimate'),
                                 source=frame['source'], observed=False,
                                 heightSource=frame['heightSource'])])
            surface = frame.get('entrySurface')
            if surface and 0 <= es.stamp-surface['checkedAt'] <= 20:
                from steward.ego.place_surface import height_at
                result['entrySurface'] = surface
                if surface['observed']:
                    result['point'][2] = height_at(surface['patch'], result['point'][:2])
                    result['normal'] = surface['normal']
                    result['targetSource'] = 'native_upper_contact_plane'
                    result['observedAt'] = surface['observedAt']
                else:
                    result['warnings'].append(dict(reason='contact_surface_assumed', uncertaintyM=surface['uncertaintyM']))
            from steward.ego.fasten_target import ENABLED, RADIAL_TOLERANCE_M
            registered = frame.get('registeredSeatTargets', {}).get(params.get('hole')) if ENABLED else None
            if registered:
                result.update(point=list(registered['point']), targetSource=registered['targetSource'],
                              observedAt=registered['observedAt'])
                result['pairAlignment']['distanceXYM'] = registered['pairDistanceM']
                if (registered['pairDistanceM'] or 0.) > RADIAL_TOLERANCE_M:
                    result['warnings'].append(dict(reason='upper_component_misaligned',
                        distanceM=registered['pairDistanceM'], toleranceM=RADIAL_TOLERANCE_M))
            return result
    else:
        if frame:
            pair = next((p for p in frame['holePairs'] if p['bodyHole'] == params.get('hole')), None)
            h = next((h for h in frame['holes'] if pair and h['id'] == pair['partHole']), None)
            if h is None:
                raise PredictionError('release frame has no correspondence for this interface hole')
            return dict(point=list(h['point']), normal=[0.,0.,1.], partHole=h['id'],
                        observedAt=frame['observedAt'], releasedAt=frame['releasedAt'],
                        targetSource=frame['source'], componentFrame=frame, warnings=[
                            dict(reason='release_fk_interface_estimate', observed=False,
                                 heightSource=frame['heightSource'])])
    part, body = es.find(str(params.get('part',''))), es.find(str(params.get('body','')))
    if part is None or body is None or part.id == body.id:
        raise PredictionError('seat_screw needs distinct current part and body ids')
    xy = hole_point(planner, body, params.get('hole'))
    part_holes = [h for h in interface_holes(planner, part) if h.get('frame') == 'entity']
    if not part_holes:
        raise PredictionError('released component has no registered interface')
    r = matrix_from_quat(part.pose.value['q']); p = np.asarray(part.pose.value['p'])
    candidates = [(np.linalg.norm((p+r @ h['p'])[:2]-xy), h, (p+r @ h['p'])[:2]) for h in part_holes]
    distance, h, upper_xy = min(candidates, key=lambda x: x[0])
    warnings = [dict(reason='hole_pair_error', distanceM=float(distance), toleranceM=.020)] if distance > .020 else []
    view = fresh_view(part, es.stamp)
    patch = surface_patch(view['points'], upper_xy) if view else None
    if patch is None:
        # Catalogue interface height is relative to the template base, not box centre.
        # In particular, crossbar height must not become the top of the legs.
        from steward.ego.classify import SHAPES
        height = SHAPES.get(part.category.value, {}).get('dims', [0,0,entity_box(part)[2]-entity_box(part)[1]])[2]
        z = p[2]-height/2+float(h['p'][2])
        patch = dict(point=[*upper_xy.tolist(), float(z)], normal=[0.,0.,1.])
        warnings.append(dict(reason='entry_surface_estimated', heightM=patch['point'][2], observedPoints=len(view['points']) if view else 0))
    warnings.append(dict(reason='catalogue_interface_hypothesis', part=part.id, body=body.id))
    return {**patch, 'partHole': h['id'], 'observedAt': view['stamp'] if view else part.pose.t,
            'warnings':warnings+(view.get('warnings', []) if view else [])}


def drive_context(planner, es, params):
    """Prefer the action ledger; otherwise associate the nearest catalogue interface.

    This supplies an execution target, never a claim that a release occurred.
    """
    attempts = {}
    for ent in es.entities:
        for row in value(ent, 'fasteningEvidence', []):
            if (params.get('screw') and row['screw'] == params['screw'] or
                    not params.get('screw') and row['body'] == params.get('body') and row['hole'] == params.get('hole')):
                attempts[(row['body'], row['hole'])] = row
    if attempts:
        return next(iter(attempts.values()))
    screws = [e for e in es.entities if e.category.value == 'toy_screw'
              and (not params.get('screw') or e.id == params['screw'])]
    choices = []
    for body in es.entities:
        if params.get('body') and body.id != params['body']:
            continue
        for hole in interface_holes(planner, body):
            if params.get('hole') and hole['id'] != params['hole']:
                continue
            for part in es.entities:
                if part.id == body.id or params.get('part') and part.id != params['part']:
                    continue
                peers = interface_holes(planner, part)
                if not any(h.get('group') in hole.get('mates', ()) for h in peers):
                    continue
                target = seat_target(planner, es, dict(part=part.id, body=body.id, hole=hole['id']))
                for screw in screws:
                    feature = tool_estimate(screw, es.stamp)
                    error = np.linalg.norm(np.asarray(feature['tip'])-target['point'])
                    # Also prefer a component whose interface is already near this body.
                    xy = hole_point(planner, body, hole['id'])
                    error += np.linalg.norm(np.asarray(target['point'])[:2]-xy)
                    choices.append((error, dict(body=body.id, part=part.id, hole=hole['id'],
                        partHole=target['partHole'], screw=screw.id, entry=target['point'],
                        headBeforeRelease=feature['head'], headBeforeSource=feature['source'],
                        seatedAt=feature['stamp'], verified=False, status='seating_record_estimated')))
    if not choices:
        raise PredictionError('drive has no screw/interface target in current estimates')
    return min(choices, key=lambda pair:pair[0])[1]


def predict(planner, es, skill, params):
    if params.get('method', 'simple') not in ('simple', 'contact'):
        raise PredictionError('screw method must be simple or contact')
    if _assembly_enabled():
        if skill == 'fasten_onto':
            from steward.ego.fasten_release import released_interface
            frame = released_interface(es, params.get('part'), params.get('body'),
                                       allow_alias=not params.get('drive', True))
            if frame:
                if planner._held(es) is not None:
                    raise PredictionError('fasten_onto starts with an empty hand; park the held load then retry')
                body = es.entity(frame['body'])
                ids = [p['bodyHole'] for p in frame['holePairs']]
                if not ids:
                    raise PredictionError('released interface has no hole correspondence; observe and register before resuming')
                if params.get('hole') is not None:
                    chosen = planner._select_holes(interface_holes(planner, body), params['hole'], body.id)
                    if {h['id'] for h in chosen} != set(ids):
                        raise PredictionError('resume requires the complete released interface, not a subset or another group')
                from steward.ego.assembly_status import require_interface_access
                access = require_interface_access(planner, body, set(ids))
                support = support_anchor(planner, es, body, []).id
                result = planner._receipt(es, skill, params, .02, [], dict(
                    part=frame['entity'], body=body.id, support=support, resumeReleased=True,
                    partSource=dict(id=frame['entity'], category=frame['category'], position=frame['position']),
                    releaseFrame=frame, selectedHoleIds=ids, interfaceAccess=access, stages=['resume_released_interface']))
                if access['status'] != 'estimated_accessible':
                    result['warnings'].append(dict(access, reason='interface_face_unresolved'))
            else:
                result = planner._predict_assemble_onto(es, params)
            if not frame and not params.get('drive', True):
                selected = {h['id'] for h in mating_holes(planner, es, result['plan'])}
                frames = {f['entity']:f for e in es.entities for f in value(e, 'releasedInterfaces', [])
                          if f['body'] == params['body'] and
                          selected & {p['bodyHole'] for p in f.get('holePairs', [])}}
                if len(frames) == 1:
                    saved = next(iter(frames.values()))
                    return predict(planner, es, skill, dict(params, part=saved['entity']))
                attempts = value(es.entity(params['body']), 'fasteningEvidence', [])
                if any(a.get('releaseExecuted') and a['hole'] in selected for a in attempts):
                    raise PredictionError('released holes remain occupied; recover the registered interface before continuation; do not regrasp component')
            result.update(skill=skill, params=dict(params))
            result['plan'].update(method=params.get('method', 'simple'), correctOnce=bool(params.get('correctOnce', True)), drive=bool(params.get('drive', True)), driveExplicit=params.get('drive') is True)
            for key in ('stageScrews', 'recoverDropped', 'closeRelease', 'wingRetry'):
                result['plan'][key] = bool(params.get(key, False))
            result['plan']['fastenSkill'] = skill
            result['plan']['selectedHoleIds'] = [h['id'] for h in mating_holes(planner, es, result['plan'])]
            result['plan']['stages'] += ['seat_each_screw', 'drive_each_screw', 'verify_attachment']
            return result
    else:
        if skill == 'fasten_onto':
            result = planner._predict_assemble_onto(es, params)
            result.update(skill=skill, params=dict(params))
            result['plan'].update(method=params.get('method', 'simple'), correctOnce=bool(params.get('correctOnce', True)), drive=bool(params.get('drive', True)))
            for key in ('stageScrews', 'recoverDropped', 'closeRelease', 'wingRetry'):
                result['plan'][key] = bool(params.get(key, False))
            result['plan']['fastenSkill'] = skill
            result['plan']['selectedHoleIds'] = [h['id'] for h in mating_holes(planner, es, result['plan'])]
            result['plan']['stages'] += ['seat_each_screw', 'drive_each_screw', 'verify_attachment']
            return result
    if skill == 'regrasp_upright':
        target = es.find(str(params.get('target', '')))
        from steward.ego.slender_tool import axial_feature
        if target is None or axial_feature(target, es.stamp) is None:
            raise PredictionError('regrasp_upright needs an estimated elongated target')
        held = planner._held(es)
        if held is not None and held.id != target.id:
            raise PredictionError('regrasp target differs from the held load')
        support = es.find(str(params.get('on', ''))) if params.get('on') else support_anchor(planner, es, target, [])
        if support is None:
            raise PredictionError('regrasp support unavailable')
        return planner._receipt(es, skill, params, .02, [],
            dict(params, support=support.id, fastenSkill=skill, stagedPrediction=True))
    admission_warnings = []
    if _assembly_enabled():
        held = planner._held(es)
        from steward.ego.driver_batch import enabled as batch_enabled
        reuse = (batch_enabled() and skill == 'drive_screw' and params.get('reuseHeldTool')
                 and params.get('method','simple') == 'simple' and held is not None
                 and planner._affords(held, 'screwdriver'))
        if held is not None and not reuse:
            raise PredictionError(f'{skill} starts with an empty hand; park the current load using place')
    else:
        if planner._held(es) is not None:
            raise PredictionError(f'{skill} starts with an empty hand; park the current load using place')
    body = es.find(str(params.get('body', '')))
    if _assembly_enabled():
        if skill == 'drive_screw':
            context = drive_context(planner, es, params)
            params = {**params, 'screw':context['screw'], 'driveContext':context}
            admission_warnings.append(dict(reason='drive_target_best_estimate', screw=params['screw'], status=context['status']))
        else:
            if body is None:
                raise PredictionError(f'{skill}: body id is absent from current Ego state')
            if skill == 'seat_screw':
                admission_warnings.extend(seat_target(planner, es, params).get('warnings', []))
            else:
                attempts = value(body, 'fasteningEvidence', [])
                from steward.ego.fasten_landing import flip_advice
                advice = flip_advice(value(body, 'releasedInterfaces', []), attempts, body.id)
                params = {**params, 'flipAdvice':advice}
                admission_warnings.append(dict(reason='unfinished_holes_before_flip', **advice))
                admission_warnings.append(dict(reason='attachment_confidence', verifiedAttempts=sum(bool(a.get('verified')) for a in attempts)))
                from steward.ego.fasten_release import loose_components
                loose = loose_components(es, body.id)
                if loose:
                    admission_warnings.append(dict(reason='loose_components_before_flip',
                        parts=[p['entity'] for p in loose], confirmedSeatedScrews=0,
                        advisory=True))
    else:
        if skill == 'drive_screw':
            context = drive_context(planner, es, params)
            params = {**params, 'screw':context['screw'], 'driveContext':context}
            admission_warnings.append(dict(reason='drive_target_best_estimate', screw=params['screw'], status=context['status']))
        else:
            if body is None:
                raise PredictionError(f'{skill}: body id is absent from current Ego state')
            if skill == 'seat_screw':
                admission_warnings.extend(seat_target(planner, es, params).get('warnings', []))
            else:
                attempts = value(body, 'fasteningEvidence', [])
                admission_warnings.append(dict(reason='attachment_confidence', verifiedAttempts=sum(bool(a.get('verified')) for a in attempts)))
                from steward.ego.fasten_release import loose_components
                loose = loose_components(es, body.id)
                if loose:
                    admission_warnings.append(dict(reason='loose_components_before_flip',
                        parts=[p['entity'] for p in loose], confirmedSeatedScrews=0,
                        advisory=True))
    # Attachment confidence is reported by execution, not an admission gate.
    receipt = planner._receipt(es, skill, params, .02,
        ['Staged skill; every motion uses a new receipt. Occluded evidence remains unknown; later stages can fail.'],
        {**params, 'fastenSkill': skill, 'stagedPrediction': True})
    receipt['warnings'].extend(admission_warnings)
    if _assembly_enabled():
        if skill == 'flip_assembly':
            from steward.ego.working_face import flip_guard
            protected = flip_guard(planner, es, params)
            if protected:
                receipt.update(protected, feasible=False)
                receipt['issues'].insert(0, protected['guidance'])
                return receipt
            from steward.ego.assembly_edge_hint import preview
            receipt['edgeOptions'] = preview(planner, es, body, params)
    return receipt


def contact(planner, es, params):
    """Plan a measured tool endpoint onto a measured surface, without arbitrary goal coordinates."""
    held = planner._held(es)
    if held is None:
        raise PredictionError('contact needs the requested tool held')
    operation = params.get('operation')
    standing = operation == 'stand'
    expected = 'toy_screw' if operation == 'seat' else 'electric_screwdriver'
    if not standing and (operation not in ('seat', 'drive') or held.category.value != expected):
        raise PredictionError('contact tool identity does not match the operation')
    from steward.ego.slender_tool import axial_feature
    feature = axial_feature(held, es.stamp) if standing else tool_estimate(held, es.stamp)
    if not feature:
        raise PredictionError('held object has no supported tool geometry')
    if 'grip' not in feature:
        # Recovery for a restored held state without the grasp snapshot.
        feature = {**feature, 'grip':list(held.pose.value['p']),
                   'warnings':[*feature.get('warnings', []), dict(reason='grasp_centre_estimated')]}
    body = es.find(str(params.get('body', '')))
    if standing:
        body = es.find(str(params.get('stageOn', '')))
    if body is None:
        raise PredictionError('contact body disappeared')
    staging = bool(params.get('stageOn'))
    if staging:
        anchor = es.entity(params['stageOn'])
        if not planner._affords(anchor, 'support-surface'):
            raise PredictionError('screw staging requires a perceived table surface')
        target = dict(point=[*params['stageXY'], entity_box(anchor)[2]], normal=[0,0,1], warnings=[])
    else:
        target = seat_target(planner, es, params)
    if _assembly_enabled():
        if operation == 'seat' and params.get('refineSeatSurface'):
            from steward.ego.fasten_seat import surface_target
            target = surface_target(es, params['part'], target)
        registered = params.get('registeredEntry')
        if registered is not None:
            from steward.ego.wrist_target import adoption_enabled, entry_check
            wrist_check = None
            if adoption_enabled() and registered.get('measurementSource') in ('wrist_rgbd_empty_recess', 'preplace_layout_rgbd'):
                wrist_check = entry_check(es, params, target, registered)
                if wrist_check['mismatches']:
                    error = PredictionError('registered entry does not match this placement: '+
                                            ', '.join(wrist_check['mismatches']))
                    error.wrist_admission = wrist_check
                    raise error
            if wrist_check is None and (operation != 'seat' or registered.get('entity') != params.get('part')
                    or registered.get('body') != params.get('body') or registered.get('hole') != params.get('hole')
                    or not (0 <= es.stamp-registered.get('observedAt', -math.inf) <= 90 or
                        registered.get('persistent') and target.get('componentFrame', {}).get(
                            'registeredSeatTargets', {}).get(params.get('hole'), {}).get('observedAt') == registered.get('observedAt'))):
                from steward.ego.wrist_target import local_range_enabled
                if local_range_enabled():
                    mismatches = [name for name, valid in (
                        ('operation', operation == 'seat'),
                        ('entity', registered.get('entity') == params.get('part')),
                        ('body', registered.get('body') == params.get('body')),
                        ('hole', registered.get('hole') == params.get('hole')),
                        ('observation_age_or_persistent_stamp',
                         0 <= es.stamp-registered.get('observedAt', -math.inf) <= 90 or
                         registered.get('persistent') and target.get('componentFrame', {}).get(
                             'registeredSeatTargets', {}).get(params.get('hole'), {}).get('observedAt') == registered.get('observedAt')),
                    ) if not valid]
                    raise PredictionError('registered entry does not match this placement: '+', '.join(mismatches))
                raise PredictionError('registered entry does not match this placement')
            point = np.asarray(registered.get('point'), float)
            if point.shape != (3,) or not np.isfinite(point).all():
                raise PredictionError('invalid registered entry')
            target = dict(target, point=point.tolist(), targetSource='registered_hole_fk')
            if wrist_check is not None:
                target['wristAdmission'] = wrist_check
    goal = np.asarray(target['point'])
    if _assembly_enabled():
        contact_surface = None
        if operation == 'drive':
            from steward.ego.fasten_driver import head_target
            context = params.get('driveContext') or drive_context(planner, es, params)
            observed = head_target(context, es.stamp)
            target.setdefault('warnings', []).append(dict(reason='per_hole_head_target', **observed))
            goal = np.asarray(observed['head']) + [0, 0, .002]
        else:
            gap = -.001 if operation == 'seat' and params.get('precisionInsert') else .001
            if target.get('entrySurface'):
                from steward.ego.place_surface import placement_envelope
                surface_z, gap, contact_surface = placement_envelope(target['entrySurface'],
                    goal[:2], [0., 0.], 0., float(goal[2]), .002,
                    max(.002, float(feature.get('uncertaintyM', .01))))
                goal[2] = surface_z
            goal = goal + [0, 0, gap]  # unobserved endpoints retain a positive surface margin
    else:
        if operation == 'drive':
            screw = es.find(str(params.get('screw', '')))
            observed = value(screw, 'toolFeature')
            attempts = value(body, 'fasteningEvidence', [])
            attempt = next((a for a in attempts if a['screw'] == params.get('screw')), {})
            if not observed or observed.get('stamp', 0) <= attempt.get('seatedAt', -1):
                from steward.ego.classify import SHAPES
                head = attempt.get('headBeforeRelease', (goal+[0,0,max(SHAPES['toy_screw']['dims'])]).tolist())
                observed = dict(head=head, tip=goal.tolist(), axis=[0,0,1], stamp=es.stamp)
                target.setdefault('warnings', []).append(dict(reason='seated_head_nominal', screw=params.get('screw')))
            target.setdefault('warnings', []).append(dict(reason='screw_head_estimate',
                ageS=es.stamp-observed['stamp'], axisZ=float(observed['axis'][2]),
                entryErrorM=float(np.linalg.norm(np.asarray(observed['tip'])-goal))))
            goal = np.asarray(observed['head']) + [0, 0, .002]
        else:
            goal = goal + [0, 0, .001]  # controlled release, not the ordinary 20 mm place drop
    anchor = anchor if staging else support_anchor(planner, es, body, target.setdefault('warnings', []))
    current = {'wrist_pitch':0., 'wrist_roll':0., 'wrist_yaw':0., **planner._arm_joints(es)}
    _, _, base_yaw = planner._robot_xy_yaw(es)
    # A robot-side actual pinch point is carried in the inferred feature, not world pose.
    grip = np.asarray(feature['grip'])
    pitch = float(current.get('wrist_pitch', 0))
    r0 = matrix_from_quat(quat_from_yaw(base_yaw)) @ planner.arm.tool_rotation(
        current.get('wrist_yaw', 0), current.get('wrist_roll', 0), pitch)
    tip_local = r0.T @ (np.asarray(feature['tip'])-grip)
    axis_local = r0.T @ np.asarray(feature['axis'])
    if _assembly_enabled():
        want = (np.asarray(target['normal']) if operation == 'seat' and contact_surface
                and contact_surface['observed'] else np.array([0., 0., 1. if operation == 'seat' else -1.]))
        solutions = []; closest = math.pi
    else:
        want = np.array([0., 0., 1. if operation == 'seat' else -1.])
        solutions = []
    # Some wrists roll about the tool axis, so roll alone cannot stand a shaft.
    # Search only the body's existing pitch envelope; never expand joint/risk limits.
    pitches = sorted(set([pitch, *np.linspace(0, planner.arm.geometry.wrist_pitch_max, 13)]))
    if _assembly_enabled():
        for trial_pitch, roll in product(pitches, np.linspace(-planner.arm.geometry.wrist_roll_max,
                                                            planner.arm.geometry.wrist_roll_max, 49)):
            base_rotation = planner.arm.tool_rotation(0., roll, trial_pitch)
            error = math.acos(float(np.clip((base_rotation @ axis_local) @ want, -1, 1)))
            closest = min(closest, error)
            measured_normal = bool(contact_surface and contact_surface['observed'])
            if not measured_normal and error > math.radians(6):
                continue
            for alpha in np.linspace(-math.pi, math.pi, 25):
                r = matrix_from_quat(quat_from_yaw(alpha)) @ base_rotation
                # Unlike a vertical axis, an observed inclined normal changes its
                # alignment error when the base/world heading changes.
                if measured_normal:
                    error = math.acos(float(np.clip((r @ axis_local) @ want, -1, 1)))
                    closest = min(closest, error)
                    if error > math.radians(6):
                        continue
                endpoint = goal-r @ tip_local
                turn = abs(roll-current.get('wrist_roll', 0))+abs(trial_pitch-pitch)
                solutions.append((error, turn, alpha, roll, trial_pitch, endpoint))
        from steward.ego.axial_release import refine_solutions
        if operation == 'seat':
            solutions = refine_solutions(planner.arm, solutions, axis_local, want, tip_local, goal, current)
        if not solutions:
            raise PredictionError('cannot stand the estimated shaft upright with this grasp; park and regrasp'
                                  f' (closest {math.degrees(closest):.0f} deg from upright,'
                                  f' feature {feature.get("source")})')
    else:
        for trial_pitch, roll in product(pitches, np.linspace(-planner.arm.geometry.wrist_roll_max,
                                                            planner.arm.geometry.wrist_roll_max, 49)):
            base_rotation = planner.arm.tool_rotation(0., roll, trial_pitch)
            error = math.acos(float(np.clip((base_rotation @ axis_local) @ want, -1, 1)))
            if error > math.radians(6):
                continue
            for alpha in np.linspace(-math.pi, math.pi, 25):
                r = matrix_from_quat(quat_from_yaw(alpha)) @ base_rotation
                endpoint = goal-r @ tip_local
                turn = abs(roll-current.get('wrist_roll', 0))+abs(trial_pitch-pitch)
                solutions.append((error, turn, alpha, roll, trial_pitch, endpoint))
        if not solutions:
            raise PredictionError('cannot stand the estimated shaft upright with this grasp; park and regrasp')
    radius = max(feature['lengthM'], float(np.linalg.norm(tip_local))) + .025
    top = max(entity_box(e)[2] for e in (anchor, body, held) if e is not held)
    if _assembly_enabled():
        safe = max(grip[2], top + radius + .04, goal[2]+radius+.04 if contact_surface else -math.inf)
        if safe > planner.lift_max:
            raise PredictionError('insufficient lift clearance for the held tool rotation'
                f' (required={safe:.6f}, maximum={planner.lift_max:.6f}, gripZ={grip[2]:.6f}, goalZ={goal[2]:.6f}, sweepRadius={radius:.6f})')
    else:
        safe = max(grip[2], top + radius + .04)
        if safe > planner.lift_max:
            raise PredictionError('insufficient lift clearance for the held tool rotation')
    failures = []
    if _assembly_enabled():
        from steward.ego.axial_release import order_for_entry
        clearance_reports = {}
        ordered = (order_for_entry(planner, es, params, solutions, safe, clearance_reports)
                   if operation == 'seat' else sorted(solutions, key=lambda x: x[:2])[:24])
        for solution in ordered:
            _, _, alpha, roll, trial_pitch, endpoint = solution
            def wrist(heading):
                yaw = (alpha-heading+math.pi) % (2*math.pi)-math.pi
                return yaw, float(roll)
            def heading_issue(heading):
                return 'wrist yaw limit' if abs(wrist(heading)[0]) > planner.arm.geometry.wrist_yaw_max else None
            def path(heading, reach, near, _pitch):
                yaw, rr = wrist(heading)
                at = dict(current)
                stages = [{'lift': safe}, {'reach': near}, {'wrist_yaw': yaw}, {'wrist_roll': rr}, {'wrist_pitch':float(trial_pitch)},
                          {'reach': reach}, {'lift': float(endpoint[2])}, {'lift': safe}, {'reach': near}]
                if operation in ('seat', 'stand'):
                    stages += [{'wrist_yaw':0.}, {'wrist_roll':0.}, {'wrist_pitch':0.}]
                targets = [at]
                for change in stages:
                    end = {**at, **change}
                    n = max(1, *(math.ceil(abs(end[k]-at[k])/(.01 if k in ('lift','reach') else .1)) for k in change))
                    targets.extend({k: at[k]+(end[k]-at[k])*i/n for k in at} for i in range(1, n+1))
                    at = end
                return planner.arm.reachable_path(targets,
                    start_joints={j.id:j.q.value[0] for j in planner._self(es).rigid.joints if j.q.value})
            try:
                approach = planner._approach(es, anchor, endpoint[:2], float(endpoint[2]), {held.id},
                    pitches=[float(trial_pitch)], lifts=[safe, float(endpoint[2])], wrist=wrist,
                    heading_issue=heading_issue, approach_path=path)
            except PredictionError as exc:
                failures.append(str(exc)); continue
            # Do not silently displace a contact goal to a nearby free place.
            sbox, _, _ = entity_box(anchor)
            dx,dy=goal[:2]-[sbox.cx,sbox.cy]; c,s=math.cos(sbox.yaw),math.sin(sbox.yaw)
            if abs(c*dx+s*dy)+radius > sbox.hx or abs(-s*dx+c*dy)+radius > sbox.hy:
                raise PredictionError('tool sweep too near support edge')
            # A re-detection of the mating stack as one merged entity (it covers the body centre) lies inside the stack: whatever
            # does not rise above the stack's own top cannot block a descent that ends on that top.
            stack_top = -math.inf if staging else max(entity_box(e)[2] for e in es.entities
                                                      if e.id in (body.id, params.get('part')))
            for other in es.entities:
                if other.id in (held.id, None if staging else body.id, params.get('part'), params.get('screw'), anchor.id, es.self_id):
                    continue
                box, low, high = entity_box(other)
                if high <= stack_top+.005 and not staging:
                    bx, by = entity_box(body)[0].cx-box.cx, entity_box(body)[0].cy-box.cy
                    if (abs(math.cos(box.yaw)*bx+math.sin(box.yaw)*by) < box.hx and
                            abs(-math.sin(box.yaw)*bx+math.cos(box.yaw)*by) < box.hy):
                        continue
                dx,dy=goal[:2]-[box.cx,box.cy]; c,s=math.cos(box.yaw),math.sin(box.yaw)
                # Footprint test in the entity's own frame: a bounding circle would make a long thin
                # obstacle (a wall) block every goal within half its length.
                intersects = (abs(c*dx+s*dy)<box.hx+feature['radiusM']+.01 and
                              abs(-s*dx+c*dy)<box.hy+feature['radiusM']+.01)
                if staging:
                    from steward.ego.assembly_quick import occupied_at
                    intersects = occupied_at(other,goal[:2],feature['radiusM']+.01)
                if high > goal[2]+.008 and low < safe and intersects:
                    raise PredictionError(f'contact corridor obstructed by {other.id}')
            yaw, rr = wrist(approach['yaw'])
            plan = {**approach, 'object':held.id, 'support':anchor.id, 'gripSpot':endpoint[:2].tolist(),
                'lift':float(endpoint[2]), 'safeLift':safe, 'wristYaw':yaw, 'wristRoll':rr, 'wristPitch':float(trial_pitch),
                'goal':goal.tolist(), 'desiredAxis':want.tolist(), 'operation':operation,
                'targetEvidence':target, 'contactParams':dict(params),
                'toolLengthM':feature['lengthM'], 'toolRadiusM':feature['radiusM'],
                'polarityObserved':feature.get('polarityObserved', False)}
            if feature.get('offsetEvidence'):
                from copy import deepcopy
                plan['offsetEvidence'] = deepcopy(feature['offsetEvidence'])
            if id(solution) in clearance_reports:
                plan['jawClearanceEstimate'] = clearance_reports[id(solution)]
            if contact_surface is not None:
                plan['placementSurface'] = contact_surface
            receipt = planner._receipt(es, '_upright_tool' if standing else ('_stage_screw' if staging else '_fastener_contact'), params, .03,
                ['Endpoint pose uses best available shape plus grasp/encoders; verify independently after contact.'], plan)
            receipt['warnings'].extend(feature.get('warnings', []))
            return receipt
    else:
        for _, _, alpha, roll, trial_pitch, endpoint in sorted(solutions, key=lambda x: x[:2])[:24]:
            def wrist(heading):
                yaw = (alpha-heading+math.pi) % (2*math.pi)-math.pi
                return yaw, float(roll)
            def heading_issue(heading):
                return 'wrist yaw limit' if abs(wrist(heading)[0]) > planner.arm.geometry.wrist_yaw_max else None
            def path(heading, reach, near, _pitch):
                yaw, rr = wrist(heading)
                at = dict(current)
                stages = [{'lift': safe}, {'reach': near}, {'wrist_yaw': yaw}, {'wrist_roll': rr}, {'wrist_pitch':float(trial_pitch)},
                          {'reach': reach}, {'lift': float(endpoint[2])}, {'lift': safe}, {'reach': near}]
                if operation in ('seat', 'stand'):
                    stages += [{'wrist_yaw':0.}, {'wrist_roll':0.}, {'wrist_pitch':0.}]
                targets = [at]
                for change in stages:
                    end = {**at, **change}
                    n = max(1, *(math.ceil(abs(end[k]-at[k])/(.01 if k in ('lift','reach') else .1)) for k in change))
                    targets.extend({k: at[k]+(end[k]-at[k])*i/n for k in at} for i in range(1, n+1))
                    at = end
                return planner.arm.reachable_path(targets,
                    start_joints={j.id:j.q.value[0] for j in planner._self(es).rigid.joints if j.q.value})
            try:
                approach = planner._approach(es, anchor, endpoint[:2], float(endpoint[2]), {held.id},
                    pitches=[float(trial_pitch)], lifts=[safe, float(endpoint[2])], wrist=wrist,
                    heading_issue=heading_issue, approach_path=path)
            except PredictionError as exc:
                failures.append(str(exc)); continue
            # Do not silently displace a contact goal to a nearby free place.
            sbox, _, _ = entity_box(anchor)
            dx,dy=goal[:2]-[sbox.cx,sbox.cy]; c,s=math.cos(sbox.yaw),math.sin(sbox.yaw)
            if abs(c*dx+s*dy)+radius > sbox.hx or abs(-s*dx+c*dy)+radius > sbox.hy:
                raise PredictionError('tool sweep too near support edge')
            for other in es.entities:
                if other.id in (held.id, None if staging else body.id, params.get('part'), params.get('screw'), anchor.id, es.self_id):
                    continue
                box, low, high = entity_box(other)
                dx,dy=goal[:2]-[box.cx,box.cy]; c,s=math.cos(box.yaw),math.sin(box.yaw)
                intersects = (abs(c*dx+s*dy)<box.hx+feature['radiusM']+.01 and
                              abs(-s*dx+c*dy)<box.hy+feature['radiusM']+.01) if staging else (
                                  math.hypot(dx,dy)<max(box.hx,box.hy)+feature['radiusM']+.01)
                if staging:
                    from steward.ego.assembly_quick import occupied_at
                    intersects = occupied_at(other,goal[:2],feature['radiusM']+.01)
                if high > goal[2]+.008 and low < safe and intersects:
                    raise PredictionError(f'contact corridor obstructed by {other.id}')
            yaw, rr = wrist(approach['yaw'])
            plan = {**approach, 'object':held.id, 'support':anchor.id, 'gripSpot':endpoint[:2].tolist(),
                'lift':float(endpoint[2]), 'safeLift':safe, 'wristYaw':yaw, 'wristRoll':rr, 'wristPitch':float(trial_pitch),
                'goal':goal.tolist(), 'desiredAxis':want.tolist(), 'operation':operation,
                'targetEvidence':target, 'contactParams':dict(params),
                'toolLengthM':feature['lengthM'], 'toolRadiusM':feature['radiusM'],
                'polarityObserved':feature.get('polarityObserved', False)}
            receipt = planner._receipt(es, '_upright_tool' if standing else ('_stage_screw' if staging else '_fastener_contact'), params, .03,
                ['Endpoint pose uses best available shape plus grasp/encoders; verify independently after contact.'], plan)
            receipt['warnings'].extend(feature.get('warnings', []))
            return receipt
    raise PredictionError('no continuous contact approach: ' + '; '.join(failures[-1:]))


def reset_tool(planner, es, params):
    """Restore grasp-time roll at the current clear, retracted carrying pose."""
    held = planner._held(es); feature = tool_estimate(held, es.stamp)
    if feature and 'grip' not in feature:
        feature = {**feature, 'grip':held.pose.value['p']}
    if not held or held.category.value != 'electric_screwdriver' or not feature or 'grip' not in feature:
        raise PredictionError('driver roll reset needs a held driver target')
    grip = np.asarray(feature['grip'])
    radius = max(np.linalg.norm(np.asarray(feature[k])-grip) for k in ('tip','head'))+feature['radiusM']+.02
    for other in es.entities:
        if other.id in (held.id,es.self_id):
            continue
        box,lo,hi = entity_box(other)
        if hi < grip[2]-radius or lo > grip[2]+radius:
            continue
        c,s=math.cos(box.yaw),math.sin(box.yaw);dx,dy=grip[:2]-[box.cx,box.cy]
        if abs(c*dx+s*dy) < box.hx+radius and abs(-s*dx+c*dy) < box.hy+radius:
            raise PredictionError(f'tool reset sweep obstructed by {other.id}')
    current={'wrist_pitch':0., 'wrist_roll':0., 'wrist_yaw':0., **planner._arm_joints(es)}
    grasp=feature.get('graspWrist', {'wrist_roll':0.,'wrist_pitch':0.})
    target={k:float(grasp.get(k,0.)) for k in ('wrist_roll','wrist_pitch')}
    target['wrist_yaw']=0.
    path=[current]
    for key in ('wrist_yaw','wrist_roll','wrist_pitch'):
        begin=path[-1];steps=max(1,math.ceil(abs(begin.get(key,0)-target[key])/.1))
        path.extend({**begin,key:begin.get(key,0)+(target[key]-begin.get(key,0))*i/steps} for i in range(1,steps+1))
    if not planner.arm.reachable_path(path,
            start_joints={j.id:j.q.value[0] for j in planner._self(es).rigid.joints if j.q.value}):
        raise PredictionError('no continuous arm path to park the driver horizontally')
    receipt = planner._receipt(es, '_fastener_reset', params, .02, [],
        {'wristYaw':0.,'wristRoll':target['wrist_roll'],'wristPitch':target['wrist_pitch']})
    receipt['warnings'].extend(feature.get('warnings', []))
    if 'graspWrist' not in feature:
        receipt['warnings'].append(dict(reason='nominal_grasp_wrist_reset'))
    return receipt


def driver_grasp_heading(planner, es, target, pitch):
    """Choose a docking side from estimated shaft direction before gripping it.

    A poor grasp around a tool-axis wrist may leave no reachable tip-down pose,
    even though ordinary pick succeeds. Do not discover that only after lifting.
    """
    from steward.ego.slender_tool import axial_feature
    feature=axial_feature(target, es.stamp)
    if not feature:
        raise PredictionError('driver grasp requires a driver target')
    axis=np.asarray(feature['axis'])
    future=[planner.arm.tool_rotation(0.,rr,pp)[2] for pp,rr in product(
        np.linspace(0,planner.arm.geometry.wrist_pitch_max,13),np.linspace(-math.pi,math.pi,25))]
    future=np.asarray(future)
    def issue(heading):
        grasp=matrix_from_quat(quat_from_yaw(heading)) @ planner.arm.tool_rotation(0.,0.,pitch)
        local=grasp.T @ axis
        return None if float(np.min(future @ local))<=-math.cos(math.radians(6)) else 'this grasp leaves no reachable driver-tip-down orientation'
    return issue
