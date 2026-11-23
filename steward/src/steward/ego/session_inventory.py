"""Body-centred inventory from robot RGB-D and generic interface descriptors.

A layer reference is a session-local planning proxy, never a new sensor track.
Catalogue geometry proposes searches; only current touching surfaces admit an
omitted interface. Lost separation retains the last observed stack explicitly
as an estimate, and never authorizes dismantling it.
"""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np

from steward.ego import capabilities, session_faces as faces, session_observation as current
from steward.ego.fasten_plan import interface_holes
from steward.ego.esim import PredictionError, Esim


def enabled():
    return capabilities.get('STEWARD_ASSEMBLY_SESSION_INVENTORY') != '0'


def proxy(body, category, identity, size, holes):
    from steward.scene.schema import Est
    ent = SimpleNamespace(id=identity, category=Est(value=category, src='inferred'),
                          pose=deepcopy(body.pose), state_vars={})
    ent.rigid = SimpleNamespace(links=[SimpleNamespace(
        geometry=dict(collision=[dict(type='box', size=list(size))]),
        affordances=[Est(value={'screw-hole':deepcopy(h)}, src='inferred') for h in holes])])
    ent.holes = deepcopy(holes)
    return ent


def groups(holes):
    result = {}
    for h in holes:
        result.setdefault(h.get('group', ''), []).append(h)
    return list(result.values())


def nearby(controller, member, body):
    """A search bound only; this is not evidence of contact or attachment."""
    from steward.ego.assembly_precision import tolerance
    p, size, _ = current.prior_box(member)
    b, base, _ = current.prior_box(body)
    return np.linalg.norm(p[:2]-b[:2]) <= (np.linalg.norm(size[:2])+np.linalg.norm(base[:2]))/2+tolerance(controller)


def bodies(controller, requests):
    scene = controller.rt.ego_state()
    result = {}
    for request in requests:
        anchor = scene.find(request.get('body')) or scene.find(request.get('part'))
        if anchor is None:
            continue
        candidates = [anchor]
        candidates += [e for e in scene.entities if e.id != anchor.id and nearby(controller, anchor, e)]
        candidates = [e for e in candidates if len(groups(interface_holes(controller.rt.esim, e))) > 1]
        # Prefer the bound body; otherwise the nearest multi-interface member
        # of the same observed region. Unrelated scene bodies are never scanned.
        if candidates:
            body = min(candidates, key=lambda e:(e.id != anchor.id,
                np.linalg.norm(np.asarray(e.pose.value['p'])-anchor.pose.value['p'])))
            result[body.id] = body
        elif interface_holes(controller.rt.esim, anchor):
            result[anchor.id] = anchor
    return list(result.values())


def hypotheses(controller, body):
    """All compatible models, independent of caller membership and track count."""
    from steward.ego.priors import SCREW_HOLES, SCREW_HOLE_FRAMES
    from steward.ego.classify import SHAPES
    scene = controller.rt.ego_state(); planner = controller.rt.esim
    base_holes = interface_holes(planner, body)
    models = list(scene.entities)
    for category, holes in SCREW_HOLES.items():
        template = SCREW_HOLE_FRAMES.get(category)
        shape = SHAPES.get(category, {})
        if not template or not shape.get('dims'):
            continue
        size = [*template['sizeXYM'], shape['dims'][2]]
        models.append(proxy(body, category, 'model:'+category, size, holes))
    items, members, signatures = [], {}, set()
    for ent in models:
        if ent.id == body.id:
            continue
        ph = interface_holes(planner, ent)
        if not ph:
            continue
        pairs, declared = Esim._hole_group_pairs(ph, base_holes)
        for held, selected in pairs:
            if not held or len(held) != len(selected):
                continue
            # Undeclared compatibility may use observed complete layouts, but
            # never an arbitrary catalogue model with no mating declaration.
            if not declared and ent.id.startswith('model:'):
                continue
            spec = faces.model(controller, ent)
            layout = [h for h in spec['holes'] if h['id'] in {p['id'] for p in held}]
            signature = (tuple(h['id'] for h in selected), tuple(np.round(spec['size'], 8)),
                         tuple(tuple(h['p']) for h in layout))
            if signature in signatures:
                continue
            signatures.add(signature)
            key = str(len(items))
            eid = 'stack:'+body.id+':'+key
            model = deepcopy(ent); model.id = eid
            # Place search priors at the body; a loose duplicate is just a model.
            model.pose = deepcopy(body.pose)
            members[eid] = model
            items.append(dict(key=key, part=eid, body=body.id,
                holes=[h['id'] for h in selected], heldHoles=[h['id'] for h in held], dependencies=[]))
    return items, members


def associate(controller, body, layer, model):
    """Match geometry and current position, never an entity number or label."""
    from steward.ego.assembly_precision import tolerance
    scene = controller.rt.ego_state(); margin = tolerance(controller)
    matches = []
    for ent in scene.entities:
        if ent.id == body.id:
            continue
        measured = current.layer(controller, ent)
        if measured is None:
            continue
        delta = np.linalg.norm(measured['position']-layer['position'])
        extent = np.linalg.norm(np.sort(measured['size'][:2])-np.sort(layer['size'][:2]))
        if delta <= margin+max(layer['size'][2], measured['size'][2]) and extent <= 2*margin:
            matches.append((delta+extent, ent.id))
    return min(matches)[1] if matches else model.id


def scan(controller, requests, samples):
    """Inventory current surfaces on both sides, including merged members."""
    if not enabled():
        return []
    scene = controller.rt.ego_state(); found = []
    for body in bodies(controller, requests):
        items, models = hypotheses(controller, body)
        if not items:
            continue
        overlay = SimpleNamespace(stamp=scene.stamp, entities=[*scene.entities, *models.values()],
            relations=getattr(scene, 'relations', []),
            entity=lambda eid:models.get(eid) or scene.entity(eid),
            find=lambda eid:models.get(eid) or scene.find(eid))
        rt = SimpleNamespace(ego_state=lambda:overlay, esim=controller.rt.esim)
        view = SimpleNamespace(rt=rt, ex=controller.ex)
        recovered = faces.regional_layers(view, body, items, samples)
        # Independent measured masks also cover scenes whose base is partly
        # occluded. They are checked against every compatible interface.
        for item in items:
            if item['key'] in recovered:
                continue
            model = models[item['part']]
            spec = faces.model(view, model)
            for ent in scene.entities:
                if ent.id == body.id:
                    continue
                for sample in [None, *samples]:
                    upper = current.layer(controller, ent, sample)
                    lower = current.layer(controller, body, sample)
                    if upper is None or lower is None:
                        continue
                    from steward.ego.assembly_precision import tolerance
                    margin = tolerance(controller)
                    if np.any(abs(np.sort(upper['size'][:2])-np.sort(spec['size'][:2])) > spec['slack']+2*margin):
                        continue
                    on, exposed, _ = current.stacked(controller, upper, lower)
                    if on is not False and exposed is not None:
                        recovered[item['key']] = dict(upper, outlineErrorM=0.,
                            measuredMember=ent.id), exposed
                        break
                if item['key'] in recovered:
                    break
        sides = faces.sides(view, body, items)
        votes = [sides[k]*(1 if exposed else -1) for k,(_,exposed) in recovered.items() if k in sides]
        up = votes[0] if votes and len(set(votes)) == 1 else None
        used = set(); assigned_layers = []
        for item in items:
            if item['key'] not in recovered:
                continue
            layer, exposed = recovered[item['key']]
            identity = tuple(item['holes'])
            if identity in used:
                continue
            from steward.ego.assembly_precision import tolerance
            margin = tolerance(controller)
            if any(np.linalg.norm(layer['position']-p) <= margin and
                   np.all(abs(layer['size']-size) <= margin) for p,size in assigned_layers):
                continue  # One visible surface cannot fill two interfaces.
            used.add(identity)
            assigned_layers.append((layer['position'], layer['size']))
            model = models[item['part']]
            measured_member = scene.find(layer.get('measuredMember'))
            measured_holes = interface_holes(controller.rt.esim, measured_member) if measured_member is not None else []
            base_holes = [h for h in interface_holes(controller.rt.esim, body) if h['id'] in item['holes']]
            compatible, _ = Esim._hole_group_pairs(measured_holes, base_holes)
            registered = (measured_member is not None and
                          set(item['heldHoles']) <= {h['id'] for h in measured_holes} and bool(compatible))
            if registered:
                # Preserve the measured layout's registration, including its
                # directed pose; the model-only regional path fits its outline.
                measured_item = dict(item, part=layer['measuredMember'])
                frame = current.make_frame(controller, measured_item, layer)
            else:
                frame = faces.make_frame(view, item, layer)
            bound = associate(controller, body, layer, model)
            frame['entity'] = bound
            frame['inventoryLayer'] = True
            frame['source'] = 'body_inventory_rgbd_layer'
            item = dict(item, part=bound)
            row = dict(component=dict(onBody=True, source='current_rgbd_stack', confirmed=False,
                bestEstimate=True, reason='body_inventory_touching_surface', observedAt=layer['stamp'],
                faceExposed=exposed, faceObserved=True, layerSources=[layer['source']]), frame=frame,
                face=dict(exposed=exposed, bodyUpZEstimate=up, confirmed=up is not None,
                    resolvedBeforeHoles=True, source='body_inventory_surface_order'),
                holes={h:dict(occupied=None, confirmed=False, source='unavailable') for h in item['holes']},
                holeMapping=dict(method='body_inventory_then_visible_entries', selectedHoles=item['holes'],
                    faceExposed=exposed, acceptedVisibleHoles=[], observedEntries=[],
                    rejectedReason=None if exposed else 'selected_face_hidden'))
            found.append(dict(item=item, report=row))
    return found


def safe_scan(controller, requests):
    from steward.ego.session_initial import samples
    try:
        native = list(samples(controller))
        # No RGB-D and no independent segmented measurements: preserve the
        # ordinary from-zero lifecycle. Catalogue presence alone adds nothing.
        scene = controller.rt.ego_state()
        if not native and not any(current.layer(controller, e) is not None for e in scene.entities):
            return []
        return scan(controller, requests, native)
    except (AttributeError, KeyError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
        controller._inventory_unavailable = type(exc).__name__
        return []


def compile_requests(rt, scene, requests):
    """Repair malformed membership before legacy goal validation, using pixels."""
    if not enabled() or not isinstance(requests, list) or not requests:
        return requests, {}
    from steward.ego.fasten import Fasten
    controller = Fasten(rt)
    found = safe_scan(controller, requests)
    if not found:
        found = provisional(controller, requests)
    if not found:
        return requests, {}
    result = []; reports = {}; used = set()
    for request in requests:
        if not isinstance(request, dict):
            result.append(request); continue
        roots = {b.id for b in bodies(controller, [request])}
        options = [r for r in found if r['item']['body'] in roots]
        # A caller's selected face remains selected when it is present. The
        # other observed face is added below even if the caller omitted it.
        selector = request.get('hole')
        if selector is not None:
            selected = []
            for r in options:
                try:
                    hs = rt.esim._select_holes(interface_holes(rt.esim, scene.find(r['item']['body'])), selector, r['item']['body'])
                    if {h['id'] for h in hs} <= set(r['item']['holes']): selected.append(r)
                except PredictionError:
                    pass
            options = selected
        match = next((r for r in options if r['item']['part'] == request.get('part')), None)
        if match is None and len(options) == 1: match = options[0]
        if match is None and options and selector is None:
            part = scene.find(request.get('part'))
            if part is not None:
                compatible = []
                for record in options:
                    bh = interface_holes(rt.esim, scene.find(record['item']['body']))
                    bh = [h for h in bh if h['id'] in record['item']['holes']]
                    pairs, declared = Esim._hole_group_pairs(interface_holes(rt.esim, part), bh)
                    if declared and pairs: compatible.append(record)
                if len(compatible) == 1: match = compatible[0]
            if match is None:
                match = next((r for r in options if r['report']['face']['exposed']), options[0])
        if match is not None:
            item = match['item']; token = (item['body'], tuple(item['holes']))
            used.add(token)
            if not any(q.get('body') == item['body'] and q.get('hole') == item['holes'] for q in result if isinstance(q, dict)):
                result.append(dict(request, part=item['part'], body=item['body'],
                                   hole=item['holes'], heldHole=item['heldHoles']))
            reports[token] = match['report']
        else:
            result.append(request)
    for record in found:
        item = record['item']; token = (item['body'], tuple(item['holes']))
        if token not in used:
            # An existing caller selector may already name this interface.
            covered = next((q for q in result if isinstance(q, dict) and q.get('body') == item['body'] and
                            q.get('hole') == item['holes']), None)
            if covered is not None:
                covered.update(part=item['part'], heldHole=item['heldHoles'])
            else:
                result.append(dict(part=item['part'], body=item['body'], hole=item['holes'], heldHole=item['heldHoles']))
            reports[token] = record['report']; used.add(token)
    return result, reports


def provisional(controller, requests):
    """Admit a missing member for one view, then a non-destructive estimate.

    This applies only to an incomplete caller binding. A valid separate member
    keeps its existing from-zero placement path. No hidden interface is added
    on catalogue knowledge alone.
    """
    scene = controller.rt.ego_state(); result = []
    for request in requests:
        if not isinstance(request, dict): continue
        part = scene.find(request.get('part'))
        if part is not None and part.id != request.get('body'): continue
        try:
            roots = bodies(controller, [request])
            if not roots: continue
            body = roots[0]; items, models = hypotheses(controller, body)
            selector = request.get('hole')
            if selector is not None:
                selected = controller.rt.esim._select_holes(interface_holes(controller.rt.esim, body), selector, body.id)
                items = [i for i in items if {h['id'] for h in selected} <= set(i['holes'])]
            if not items: continue
            item = items[0]; model = models[item['part']]
            base = faces.model(controller, body); spec = faces.model(controller, model)
            measured = current.layer(controller, body)
            position = np.array(measured['position'] if measured else base['centre'], copy=True)
            base_size = measured['size'] if measured else base['size']
            position[2] += (base_size[2]+spec['size'][2])/2
            layer = dict(position=position, size=spec['size'], yaw=base['yaw'],
                stamp=body.pose.t if body.pose.t is not None else -1)
            overlay = SimpleNamespace(find=lambda eid:models.get(eid) or scene.find(eid))
            view = SimpleNamespace(rt=SimpleNamespace(ego_state=lambda:overlay,esim=controller.rt.esim))
            frame = faces.make_frame(view, item, layer)
            frame.update(inventoryLayer=True, source='body_inventory_prior_estimate', heightSource='unconfirmed_body_surface')
            report = dict(frame=frame, component=dict(onBody=True, confirmed=False, bestEstimate=True,
                source='last_perceived_overlap_estimate', reason='missing_member_preserve_bound_region',
                observedAt=layer['stamp'], faceExposed=None, faceObserved=False),
                face=dict(exposed=None, bodyUpZEstimate=None, confirmed=False,
                    resolvedBeforeHoles=True, source='body_inventory_prior_estimate'),
                holes={h:dict(occupied=None,confirmed=False,source='unavailable') for h in item['holes']})
            result.append(dict(item=item,report=report))
        except (AttributeError, KeyError, TypeError, ValueError, np.linalg.LinAlgError):
            continue
    return result


def member(scene, identity, report):
    frame = report['frame']; body = scene.find(frame['body'])
    return proxy(body, frame['category'], identity, frame['size'], frame['holes'])


def prepare(session, items):
    """Refresh/add/rebind inventory before any caller-scoped initial decision."""
    if not enabled(): return {}
    found = safe_scan(session.controller, items)
    reports = {}
    for record in found:
        candidate = record['item']
        item = next((i for i in items if session.canonical(i['body']) == candidate['body'] and
                     set(i['holes']) == set(candidate['holes'])), None)
        if item is None:
            keys = {i['key'] for i in items}; key = str(len(items))
            while key in keys: key = 'inventory:'+key
            item = dict(candidate, key=key)
            items.append(item)
        old = item['part']
        item['part'] = candidate['part']; item['body'] = candidate['body']
        report = record['report']
        item['inventoryReport'] = deepcopy(report)
        reports[item['key']] = report
        session.emit('stack_inventory', interface=item['key'], body=item['body'], part=item['part'],
                     previousPart=old, component=report['component'], face=report['face'], holes=item['holes'])
    # A lost mask after the sole view change is not evidence of disassembly.
    for item in items:
        if item['key'] not in reports and item.get('inventoryReport'):
            try:
                from steward.ego.session_initial import component
                measured, _ = component(session.controller, dict(item,
                    part=session.canonical(item['part']), body=session.canonical(item['body'])))
                if measured.get('onBody') is False and measured.get('source') == 'current_rgbd_stack':
                    # A current, separate layer is a contradiction, not a lost
                    # mask. Let the existing outcome repair inspect it.
                    continue
            except (AttributeError, KeyError, TypeError, ValueError):
                pass
            report = deepcopy(item['inventoryReport'])
            report['component'].update(confirmed=False, bestEstimate=True,
                source='last_perceived_overlap_estimate', reason='inventory_unresolved_preserve_last_stack')
            report['face']['confirmed'] = False
            reports[item['key']] = report
    if reports:
        previous = session.state.get('stackInventory', {}).get('protectedEntities', [])
        regions = []
        scene = session.rt.ego_state()
        for item in items:
            if item['key'] not in reports: continue
            frame = reports[item['key']]['frame']
            regions.append(dict(position=frame['position'], size=frame['size']))
            body = scene.find(session.canonical(item['body']))
            if body is not None:
                position, size, _ = current.prior_box(body)
                regions.append(dict(position=position.tolist(), size=size.tolist()))
        session.state['stackInventory'] = dict(source='robot_rgbd', interfaces=list(reports),
            regions=regions,
            protectedEntities=sorted({session.canonical(e) for e in previous} |
                {session.canonical(i[k]) for i in items if i['key'] in reports for k in ('body','part')}))
    for item in items:
        item['dependencies'] = [i['key'] for i in items if i['key'] != item['key'] and
                                session.canonical(i['part']) == session.canonical(item['body'])]
    return reports


def guard(rt, skill, params):
    if not enabled() or skill not in ('fasten_onto', 'assemble_onto', 'flip_assembly'):
        return None
    from steward.ego.assembly_session import current as active
    own = active()
    for state in getattr(rt.estimator, 'assembly_sessions', {}).values():
        if state.get('generation') != rt._generation or own is not None and own.rt is rt and own.state is state:
            continue
        aliases = state.get('aliases', {})
        def canonical(e):
            seen = set()
            while e in aliases and e not in seen:
                seen.add(e); e = aliases[e]
            return e
        protected = {canonical(e) for e in state.get('stackInventory', {}).get('protectedEntities', [])}
        if state.get('decision') in ('running', 'recovering'):
            protected |= {canonical(i[k]) for i in state.get('goal', {}).get('interfaces', []) for k in ('body','part')}
        targets = {canonical(params.get(k)) for k in ('part','body','target')}
        matched = bool(protected & targets)
        if not matched and state.get('stackInventory', {}).get('regions'):
            from steward.ego.assembly_precision import tolerance
            controller = state.get('controller') or SimpleNamespace(rt=rt, ex=rt.executor)
            margin = tolerance(controller)
            scene = rt.ego_state()
            for target in targets:
                ent = scene.find(target)
                if ent is None: continue
                measured = current.layer(controller, ent)
                if measured is None: continue
                if any(np.linalg.norm(measured['position']-r['position']) <= 2*margin and
                       np.all(abs(np.sort(measured['size'])-np.sort(r['size'])) <= 2*margin)
                       for r in state['stackInventory']['regions']):
                    matched = True
                    break
        if matched:
            return dict(session=state['id'], reason='stack_owned_by_assembly_session',
                message='这些零件由装配会话保护。请续做 complete_assembly session='+state['id']+'；不要另行调用单步技能去移动这些零件。')
    return None


def narrative(session):
    if not enabled() or not session.state.get('stackInventory'):
        return None
    notes = []
    for index, item in enumerate(session.state['goal']['interfaces'], 1):
        row = session.state.get('initialInterfaces', {}).get(item['key'], {})
        if not row.get('component', {}).get('onBody'): continue
        if faces.protected(row):
            step = '下方已有结构，保持不拆'
        elif session.complete(item):
            step = '在位接口已处理，接下来查看结果并补救'
        else:
            empty = sum(h.get('occupied') is False for h in row.get('holes', {}).values())
            step = ('上件已在位，'+str(empty)+' 个入口空，接下来放入紧固件' if empty else
                    '上件已在位，入口状态未确认，接下来按最好估计紧固，再查看结果并补救')
        notes.append('接口 '+str(index)+'：'+step)
    return '；'.join(notes)+'。不要另行调用单步技能去移动这些零件。续做本会话。'
