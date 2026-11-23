"""Contact-window construction and feedback using robot geometry only (ASSEMBLY_EDGE_GEOMETRY)."""
from copy import deepcopy
from itertools import product
import math
import numpy as np
from steward.ego import capabilities
from steward.math3d import matrix_from_quat, quat_from_yaw


def enabled():
    return capabilities.get('STEWARD_ASSEMBLY_EDGE_GEOMETRY') != '0'


def requires_prior(record):
    # A composite/ambiguous outline can hide an installed bearing. An isolated
    # well-registered slab without attachments needs no invented obstruction.
    return bool(record.get('parts') or record.get('reference', {}).get('registration', {}).get('lowConfidence'))


def active(shape):
    return enabled() and bool((shape or {}).get('contactWindow'))


def signed_margin(points, point):
    from steward.ego.assembly_edge import support_polygon
    poly = support_polygon(points)
    if len(poly) < 3:
        return -math.inf
    edges = np.roll(poly, -1, axis=0)-poly
    inward = np.c_[-edges[:, 1], edges[:, 0]]/np.linalg.norm(edges, axis=1)[:, None]
    return float(np.min(np.sum((np.asarray(point)[:2]-poly)*inward, axis=1)))


def recover_members(controller, record):
    """Recover pre-existing bearing envelopes from cached stack evidence.

    Does not claim a connection, a seated fastener, or task completion. The
    recovered member remains an unconfirmed rigid-carry hypothesis.
    """
    if record.get('parts'):
        return
    from steward.ego import session_initial, session_faces
    from steward.ego.fasten_plan import interface_holes
    rt = controller.rt; scene = rt.ego_state(); ref = record['reference']
    body = scene.find(ref['entity'])
    if body is None:
        return
    frames = [deepcopy(f) for f in getattr(rt.estimator, 'released_interfaces', {}).values()
              if f.get('body') == ref['entity']]
    if not frames:
        bh = interface_holes(rt.esim, body); items = []
        for ent in scene.entities:
            if ent.id == body.id:
                continue
            ph = interface_holes(rt.esim, ent)
            pairs = [(b, p) for b in bh for p in ph
                     if b.get('group') in p.get('mates', ()) and p.get('group') in b.get('mates', ())]
            if not pairs:
                continue
            items.append(dict(key=ent.id, body=body.id, part=ent.id,
                holes=list(dict.fromkeys(b['id'] for b, _ in pairs)),
                heldHoles=list(dict.fromkeys(p['id'] for _, p in pairs))))
        for item in items:
            report, frame = session_initial.component(controller, item)
            if frame and report.get('onBody') is True:
                frames.append(frame)
        if not frames and items:
            layers = session_faces.regional_layers(controller, body, items, list(session_initial.samples(controller)))
            frames = [session_faces.make_frame(controller, i, layers[i['key']][0])
                      for i in items if i['key'] in layers]
    seen = set()
    for frame in frames:
        if frame['entity'] in seen:
            continue
        seen.add(frame['entity'])
        rotation = np.asarray(frame.get('rotation', np.eye(3)))
        record['parts'].append(dict(frame, yaw=math.atan2(rotation[1, 0], rotation[0, 0]),
            status='unconfirmed_existing_bearing', connectionObserved=False))
    record['bearingRecovery'] = dict(members=len(record['parts']),
        source='existing_registration_or_cached_stack', observed=False, extraObservations=0)


def prepare(record, turn_over, view, old, diagnostics):
    from steward.ego import assembly_edge as edge
    ref = record['reference']; origin = np.asarray(ref['position'])
    r = matrix_from_quat(quat_from_yaw(ref['yaw']))
    delta = r @ np.diag([1., -1., -1.]) @ r.T if turn_over else np.eye(3)
    # The broadest low registered member is the bearing even with tiny/negative
    # COM reserve. Reserve is evaluated later and never discards its geometry.
    shape = edge.geometry(record, turn_over, body_face=True, allow_flat=True)
    if shape is None:
        return None
    contact_ref = record.get('contactReference', ref)
    core = edge.corners(contact_ref, origin, delta)
    if contact_ref is not ref:
        items = [contact_ref, *record.get('parts', [])]
        weights = [max(float(np.prod(p['size'])), np.finfo(float).eps) for p in items]
        shape['centreOfMass'] = np.average([delta @ (np.asarray(p['position'])-origin) for p in items], axis=0, weights=weights).tolist()
        shape['core'] = core.tolist()
        shape['upperBodyZM'] = float(core[:, 2].max())
    contact = core[np.isclose(core[:, 2], core[:, 2].min())]
    bearing = float(contact[0, 2]); role = 'body_face'
    for part in record.get('parts', []):
        # Small isolated obstructions do not replace the broad bearing slab.
        if np.prod(part['size'][:2]) < .15*np.prod(contact_ref['size'][:2]):
            continue
        pts = edge.corners(part, origin, delta); z = float(pts[:, 2].min())
        face = pts[np.isclose(pts[:, 2], z, atol=1e-6)]
        if z < bearing and len(edge.support_polygon(face)) >= 3:
            bearing, contact, role = z, face, 'spanning_member'
    shape.update(bearingZM=bearing, bearingPolygon=contact.tolist(), bearingRole=role,
        bodyModelSource=contact_ref.get('source', 'registered_extent'))
    axis = int(np.argmax(contact_ref['size'][:2]))
    direction = delta @ matrix_from_quat(quat_from_yaw(contact_ref['yaw']))[:, axis]
    from steward.ego import directed_orientation
    end_report = directed_orientation.end_evidence(record, view) if directed_orientation.enabled() and turn_over else None
    points = np.asarray((view or {}).get('points', []), float)
    protrusions = np.empty((0, 3)); inferred = True
    if points.ndim == 2 and points.shape[1] == 3 and len(points) >= 12:
        raw = (points-origin) @ delta.T
        low = raw[raw[:, 2] < bearing-.02*ref['size'][2]]
        # A registered bearing is not an obstruction. Strip its footprint.
        for part in record.get('parts', []):
            pr = delta @ matrix_from_quat(quat_from_yaw(part['yaw']))
            centre = delta @ (np.asarray(part['position'])-origin)
            local = (low-centre) @ pr
            low = low[~np.all(abs(local) <= np.asarray(part['size'])/2+.02*min(part['size']), axis=1)]
        if len(low) >= 6:
            # Only an end-localised cluster describes an obstruction. A broad
            # merged lower face must not become a protrusion spanning the COM.
            projection = low @ direction
            end = 1. if np.median(projection) >= 0 else -1.
            cutoff = np.quantile(end*projection, .10)
            if cutoff > 0:
                protrusions = low; direction *= end; inferred = False
    if not len(protrusions) and old and not old.get('inferred') and old.get('source') != 'caller_hint' and old.get('protrusions'):
        known = np.asarray(old['protrusions'])
        low = known[known[:, 2] < bearing-.02*ref['size'][2]]
        if len(low) and np.min(low @ direction)*np.max(low @ direction) > 0:
            protrusions = low
            inferred = False
    if not len(protrusions):
        if old and old.get('protrusions'):
            sign = np.sign(np.mean(old['protrusions'], axis=0) @ direction)
        elif len(points) >= 12 and points.ndim == 2:
            # Before turning, the high end is the future obstruction.
            raw = (points-origin) @ delta.T
            extrema = raw[raw[:, 2] <= np.quantile(raw[:, 2], .20)]
            sign = np.sign(np.mean(extrema, axis=0) @ direction)
        else:
            # Offset bearing gives the free end; symmetric geometry stays an
            # explicit end hypothesis, corrected by the release-face slope.
            sign = -np.sign(np.mean(contact, axis=0) @ direction)
        if end_report and end_report['source'].startswith('cached_rgbd'):
            sign = np.sign((delta @ np.asarray(end_report['directionWorld'])) @ direction)
        direction *= sign if sign else 1.
        along = core @ direction
        bearing_end = float(np.max(contact @ direction))
        # Conservative uncertain end: extend it inward to the bearing boundary
        # when available. This maximises overhang without hanging the bearing.
        inner = max(0., bearing_end) if role == 'spanning_member' else .15*np.ptp(along)
        inner = min(inner, float(along.max())*.90)
        from steward.ego.assembly_edge import support_polygon
        poly = support_polygon(core); end = []
        for a, b in zip(poly, np.roll(poly, -1, axis=0)):
            da, db = a @ direction[:2]-inner, b @ direction[:2]-inner
            if da >= 0: end.append(a)
            if (da >= 0) != (db >= 0): end.append(a+(b-a)*da/(da-db))
        protrusions = np.c_[end, np.full(len(end), bearing-ref['size'][2])]
        if role == 'body_face':
            # No separate member was recovered: use the remaining body face as
            # a support hypothesis, never claim an observed full member.
            clipped = []
            boundary = inner-.02*min(ref['size'])
            for a, b in zip(poly, np.roll(poly, -1, axis=0)):
                da, db = boundary-a @ direction[:2], boundary-b @ direction[:2]
                if da >= 0: clipped.append(a)
                if (da >= 0) != (db >= 0): clipped.append(a+(b-a)*da/(da-db))
            contact = np.c_[clipped, np.full(len(clipped), bearing)]
            shape['bearingPolygon'] = contact.tolist()
            shape['bearingEstimated'] = True
    else:
        direction *= 1 if np.mean(protrusions, axis=0) @ direction >= 0 else -1
    if role == 'body_face' and not inferred:
        # The exposed body's contact is its intersection with the support. A
        # separate spanning member, in contrast, must remain entirely inside.
        poly = edge.support_polygon(core); clipped = []
        boundary = float(np.min(protrusions @ direction))-.02*min(ref['size'])
        for a, b in zip(poly, np.roll(poly, -1, axis=0)):
            da, db = boundary-a @ direction[:2], boundary-b @ direction[:2]
            if da >= 0: clipped.append(a)
            if (da >= 0) != (db >= 0): clipped.append(a+(b-a)*da/(da-db))
        shape['bearingPolygon'] = np.c_[clipped, np.full(len(clipped), bearing)].tolist()
    # If registration includes a broad whole-body box, the observed bearing
    # still has priority. Never label the body box as a measured member.
    shape.update(protrusions=protrusions.tolist(),
        envelope=np.vstack([np.asarray(shape['envelope']), protrusions]).tolist(),
        contactWindow=True, finalContactOnly=True, outwardAxis=direction.tolist(),
        inferred=inferred, source=((old or {}).get('source') or (view or {}).get('source') or 'registered_contact_window')
            if not inferred else ((old or {}).get('source') if (old or {}).get('source') == 'caller_hint' else 'estimated_contact_window'),
        geometryConfirmed=not inferred,
        bodyLongAxis=direction.tolist())
    if directed_orientation.enabled():
        inverse = delta.T @ direction
        local = r.T @ inverse
        dim = int(np.argmax(abs(local[:2])))
        cached = bool(view and (view.get('source') or '').startswith('rgbd'))
        shape['protrusionEndSource'] = dict(end_report or {},
            source=(end_report or {}).get('source', 'registered_or_observed_protrusions') if inferred else 'cached_protrusion_geometry' if cached else 'registered_protrusion_geometry',
            confidence=(end_report or {}).get('confidence', 'unconfirmed') if inferred else 'observed' if cached else 'inferred',
            bodyEnd=('+' if local[dim] >= 0 else '-')+'xy'[dim],
            directionBody=local.tolist(), directionWorld=inverse.tolist())
    diagnostics.update(contactWindow=True, bearingRole=role, inferred=inferred,
        geometryConfirmed=shape['geometryConfirmed'], registeredParts=len(record.get('parts', [])),
        bearingRecovery=record.get('bearingRecovery'), sweepSeparatedFromContact=True)
    return shape


def candidates(record, table, sigma=0., diagnostics=None):
    from steward.ego import edge_retention
    if edge_retention.enabled():
        return edge_retention.candidates(record, table, sigma, diagnostics)
    from steward.ego.assembly_edge import entity_box
    from steward.ego.edge_result import rotated
    shape = record['edgeGeometry']; box, _, top = entity_box(table)
    tr = matrix_from_quat(quat_from_yaw(box.yaw))[:2, :2]; half = np.array([box.hx, box.hy])
    axis = np.asarray(shape['outwardAxis'])[:2]
    heading = math.atan2(axis[1], axis[0]); results = []
    root = tr.T @ (np.asarray(record['reference']['position'])[:2]-[box.cx, box.cy])
    for dim, sign in product(range(2), (-1, 1)):
        wanted = box.yaw+dim*math.pi/2+(math.pi if sign < 0 else 0.)
        angle = (wanted-heading+math.pi) % (2*math.pi)-math.pi
        s = rotated(shape, angle)
        bearing = np.asarray(s['bearingPolygon'])[:, :2] @ tr
        protrusion = np.asarray(s['protrusions'])[:, :2] @ tr
        env = np.asarray(s['envelope'])[:, :2] @ tr
        com = tr.T @ np.asarray(s['centreOfMass'])[:2]
        force = np.asarray(record.get('forcePoints', []), dtype=float).reshape(-1, 3)
        rotation = matrix_from_quat(quat_from_yaw(angle))
        force = (force @ rotation.T)[:, :2] @ tr
        inward_protrusion = float(np.min(sign*protrusion[:, dim]))
        outward_bearing = float(np.max(sign*bearing[:, dim]))
        window = inward_protrusion-outward_bearing
        # Edge line lies nearest the protrusion's inner side. Cap measurement
        # slack by the available window instead of consuming a small COM reserve.
        scale = min(record['reference']['size'])
        slack = min(max(0., window)*.20, max(scale*.02, sigma))
        line = inward_protrusion-slack
        extra = max(0., record.get('edgeClearanceExtraM', 0.))
        line -= min(extra, max(0., line-outward_bearing))
        normal = sign*(half[dim]-line)
        tangent = 1-dim
        lo = -half[tangent]-env[:, tangent].min()
        hi = half[tangent]-env[:, tangent].max()
        if len(force):
            lo = max(lo, -half[tangent]-force[:, tangent].min())
            hi = min(hi, half[tangent]-force[:, tangent].max())
        if lo > hi:
            continue
        # Centre the complete contact/envelope along the edge before considering
        # nearer approaches. No full-body footprint inset on the overhanging end.
        along = float(np.clip(-com[tangent], lo, hi))
        for value in dict.fromkeys([along, float(np.clip(root[tangent], lo, hi))]):
            offset = np.zeros(2); offset[dim] = normal; offset[tangent] = value
            contact_world = bearing+offset
            contact_inside = float(np.min(half-abs(contact_world)))
            clearance = float(np.min(half-abs(offset+com)))
            bearing_margin = signed_margin(s['bearingPolygon'], s['centreOfMass'])
            margin = min(clearance, bearing_margin)
            force_margin = float(np.min(half-abs(force+offset))) if len(force) else None
            # A missing member may leave an overlapping body-box estimate. Keep
            # the best executable edge hypothesis and expose the inconsistency.
            results.append(dict(offset=offset.tolist(), spot=(np.array([box.cx, box.cy])+tr @ offset).tolist(),
                edgeAxis=dim, edgeSign=sign, yaw=record['reference']['yaw']+angle, yawDeltaRad=angle,
                overhangM=float(np.max(sign*(env[:, dim]+offset[dim]))-half[dim]),
                protrusionClearanceM=float(inward_protrusion-line), bearingInsideMarginM=contact_inside,
                supportWindowM=window, edgeLineBodyM=line, protrusionInnerM=inward_protrusion,
                bearingOuterM=outward_bearing, longAxisEdgeAngleDeg=90.,
                comMarginM=margin, tableComMarginM=clearance, bearingComMarginM=bearing_margin,
                requiredComMarginM=0., comUncertaintyM=shape['comUncertaintyM']+2*sigma,
                forceInsideMarginM=force_margin,
                bodyCentreZM=top-s['bearingZM'], bearingZM=s['bearingZM'],
                bodyGraspZM=top+(s['upperBodyZM']-s['bearingZM'])/2,
                comOffsetWorldM=s['centreOfMass'][:2], massSource=s['massSource'], bearingRole=s['bearingRole'],
                contactWindow=True, geometryConfirmed=shape.get('geometryConfirmed', False),
                geometryConflict=window < 0 or contact_inside < -1e-8,
                bearingPolygonWorld=(np.c_[np.asarray(s['bearingPolygon'])[:, :2]+
                    (np.array([box.cx, box.cy])+tr @ offset), np.full(len(bearing), top)]).tolist(),
                estimatedTiltDeg=0., observed=False))
    positive = [c for c in results if c['comMarginM'] >= 0 and not c['geometryConflict']]
    # Keep all negative candidates when no positive is available; best margin
    # wins. Mechanical reach/collision checks still apply to every candidate.
    selected = positive or results
    force_safe = [c for c in selected if c['forceInsideMarginM'] is None or c['forceInsideMarginM'] >= 0]
    if positive and force_safe: selected = force_safe
    selected.sort(key=lambda c:(-c['comMarginM'], c['geometryConflict'],
        np.linalg.norm(np.asarray(c['spot'])-record['reference']['position'][:2])))
    for c in selected:
        c['allMarginsNegative'] = bool(selected and max(q['comMarginM'] for q in selected) < 0)
        c['unconfirmed'] = c['allMarginsNegative'] or c['geometryConflict'] or not c['geometryConfirmed']
    if diagnostics is not None:
        diagnostics.update(candidateCount=len(selected), contactWindow=True, rejected={},
            allMarginsNegative=bool(selected and selected[0]['allMarginsNegative']))
    return selected


def validate(record, table, offset, sigma, yaw):
    if offset is None:
        return None
    # Recompute the constrained target; no legacy uncertainty-disc veto.
    return next((c for c in candidates(record, table, sigma)
        if np.allclose(c['offset'], offset, atol=1e-6) and
        (yaw is None or abs((c['yaw']-yaw+math.pi) % (2*math.pi)-math.pi) < 1e-6)), None)


def correction(record, report):
    """Advance toward the high end within the remaining bearing window."""
    from steward.ego import edge_retention
    if edge_retention.enabled():
        return edge_retention.correction(record, report)
    patch = report.get('patch') or {}; normal = np.asarray(patch.get('normal', [0., 0., 1.]))
    shape = record.get('edgeGeometry')
    if not shape or report.get('level') is True:
        return False
    high = -normal[:2]
    axis = np.asarray(shape['outwardAxis'])
    if np.linalg.norm(high) and high @ axis[:2] < 0:
        # The observed high end supersedes an uncertain end prior.
        from steward.ego.assembly_edge_hint import apply
        r = matrix_from_quat(quat_from_yaw(record['reference']['yaw']))
        local = r.T @ np.r_[high, 0.]
        dim = int(np.argmax(record['reference']['size'][:2]))
        end = ('+' if local[dim] >= 0 else '-')+'xy'[dim]
        hinted, _ = apply(record, False, None, dict(end=end, uncertain=True))
        record['edgeGeometry'] = prepare(record, False, None, hinted, {})
        record['edgeClearanceExtraM'] = 0.
        return True
    size = record['reference']['size']; span = max(size[:2])
    window = max(0., report.get('supportWindowM', 0.))
    reserve = max(0., report.get('comMarginM', 0.))
    rise = span*np.linalg.norm(high)/max(normal[2], np.finfo(float).eps)
    # Consume at most half of the remaining support window/reserve. A fresh
    # slope determines direction; no along-edge blind move can substitute.
    step = min(span*.10, max(rise*.5, span*.01))
    if window > 0: step = min(step, .5*window)
    if reserve > 0: step = min(step, .5*reserve)
    record['edgeClearanceExtraM'] = record.get('edgeClearanceExtraM', 0.)+step
    record['edgeCorrectionDirection'] = high.tolist()
    return True


def adjust(controller, job, plan):
    from steward.ego.skills import SkillFailed
    records = controller.rt.estimator.stationary_assemblies.records
    post = job.postcondition; initial = deepcopy(post['edgePlacement']); history = []
    start = np.asarray(records[plan['body']]['reference']['position'][:2])
    from steward.ego import placement_support_margin, directed_orientation
    directed = directed_orientation.enabled() and placement_support_margin.enabled()
    for attempt in range(2 if directed else 1 if placement_support_margin.enabled() else 2):
        report = post['edgePlacement']
        # The first release already executed the best geometric estimate. With
        # no depth, finish unconfirmed rather than adding a blind grasp/move.
        from steward.ego import edge_retention
        if edge_retention.enabled() and (not report.get('observed') or report.get('estimatedTiltDeg') is None):
            report.update(unconfirmed=True, correctionStopped='tilt_unobserved_keep_position')
            break
        if report.get('level') is True:
            break
        if not report.get('observed') and (attempt or not records[plan['body']].get('edgeGeometry', {}).get('geometryConfirmed')):
            break
        record = records[plan['body']]
        if not (directed_orientation.correction(record, report) if directed else correction(record, report)): break
        if placement_support_margin.enabled():
            if not directed and record.get('edgeCorrectionAttempted'):
                report['correctionStopped']='outward_attempt_already_spent'
                break
            ready, reason = placement_support_margin.correction_ready(controller, record)
            if not ready:
                record.pop('edgeCorrectionTarget',None)
                report.update(correctionStopped=reason, unconfirmed=True)
                from steward.ego import placement_body_frame
                if placement_body_frame.enabled():
                    report['correctionPreflightChecks']=record.get('edgeCorrectionPreflight',[])
                    report['correctionSearchExhausted']=True
                break
            record['edgeCorrectionAttempted']=True
            if directed and record.get('edgeCorrectionAction') == 'rotate_vertical_half_turn':
                record['edgeHalfTurnAttempted'] = True
        child = {}; job.postcondition = child; failure = None
        try:
            controller.flip(job, dict(plan, turnOver=False, _edgeAdjusted=True))
        except SkillFailed as exc:
            failure = str(exc)
            controller.check()
            if controller.rt.estimator.held_id:
                # A failed read must never leave the body held: execute the
                # current best estimate using the normal checked place path.
                release_held(controller, job, record)
            if not child.get('releaseExecuted'):
                report.update(adjustmentFailure=str(exc), unconfirmed=True)
                break
        finally:
            job.postcondition = post
        if 'edgePlacement' not in child:
            if child.get('releaseExecuted'):
                # A recovery release invalidates the earlier measured tilt even
                # when its postcheck failed. Never publish stale success.
                post['edgePlacement'] = dict(report, estimatedTiltDeg=None,
                    observed=False, level=None, unconfirmed=True,
                    levelReason='recovery_release_not_measured', adjustmentFailure=failure)
                post.setdefault('stages', []).extend(child.get('stages', []))
            break
        final = child['edgePlacement']
        if failure: final['adjustmentFailure'] = failure
        from steward.ego import placement_body_frame
        if placement_body_frame.enabled() and final.get('correctionRecovery'):
            records[plan['body']]['edgeHalfTurnAttempted'] = record.get('edgeHalfTurnCompleted', False)
        elif placement_body_frame.enabled() and record.get('edgeCorrectionAction') == 'rotate_vertical_half_turn':
            records[plan['body']]['edgeHalfTurnCompleted'] = True
        history.append(dict(tiltDeg=final.get('estimatedTiltDeg'), used=final.get('used'),
            **(dict(stepM=record.get('edgeCorrectionStepM'),
                edgeNormalWorld=record.get('edgeCorrectionDirection'),
                reason=report.get('orientationDecision') if directed else 'observed_tilt_outward_only') if edge_retention.enabled() else {}),
            translationM=float(np.linalg.norm(np.asarray(records[plan['body']]['reference']['position'][:2])-start)),
            highEndDirection=(-np.asarray(report['patch']['normal'])[:2]).tolist() if directed and report.get('patch') else record.get('edgeCorrectionDirection'), offset=final.get('offset'),
            **(dict(decision=report.get('orientationDecision'), action=record.get('edgeCorrectionAction'),
                correctionTiltAfter=final.get('estimatedTiltDeg')) if directed else {})))
        if placement_body_frame.enabled():
            target=record.get('edgeCorrectionTarget', {})
            history[-1].update(requestedSpot=target.get('spot'), plannedSpot=final.get('spot'),
                plannedYawRad=final.get('yaw'),
                preflightAttempts=len(record.get('edgeCorrectionPreflight',[])))
            for key in ('correctionRetry','correctionRecovery'):
                if final.get(key): history[-1][key]=True
        post['edgePlacement'] = final
        post['reference'] = child['reference']
        post.setdefault('stages', []).extend(child.get('stages', []))
        if failure or not final.get('patch'): break
        before_angle, after_angle = report.get('estimatedTiltDeg'), final.get('estimatedTiltDeg')
        if not directed and before_angle is not None and after_angle is not None and after_angle >= before_angle:
            final['correctionStopped'] = 'no_measured_improvement'
            break
    final = post['edgePlacement']
    final.update(tiltBefore=initial.get('estimatedTiltDeg'), tiltAfter=final.get('estimatedTiltDeg'),
        finalTiltDeg=final.get('estimatedTiltDeg'), adjustments=history,
        bestMeasuredTiltDeg=min([h['tiltDeg'] for h in history if h['tiltDeg'] is not None]+
            ([initial['estimatedTiltDeg']] if initial.get('estimatedTiltDeg') is not None else []), default=None),
        adjustment=(('outward_normal_steps' if history else 'kept_release_position') if edge_retention.enabled()
            else 'one_body_reposition' if len(history)==1 else 'bounded_high_end_translation'),
        fallback=bool(history), fallbackResult=('used' if final.get('used') else 'not_admitted') if history else final.get('fallbackResult','not_triggered'),
        restoredBestPose=False,
        adjustmentTranslationM=float(np.linalg.norm(np.asarray(records[plan['body']]['reference']['position'][:2])-start)),
        unconfirmed=final.get('unconfirmed', False) or not final.get('observed', False))
    if directed:
        final['orientationDecision'] = directed_orientation.classify(final)
        final['adjustment'] = 'bounded_directional_corrections' if history else 'kept_release_position'
        final['protrusionEndSource'] = initial.get('protrusionEndSource')
        final['initialPlannedYawRad'] = initial.get('yaw')
    if final.get('correctionSearchExhausted'):
        final['adjustment'] = 'edge_search_exhausted_release_unconfirmed'
    post['levelAlignment']['adjustmentAttempts'] = len(history)


def release_held(controller, job, record):
    """Finish the held estimate without re-entering perception or session waits."""
    if not controller.rt.estimator.held_id: return
    table = controller.rt.ego_state().find(record['reference']['support'])
    if table is None: return
    for c in candidates(record, table, controller.rt.esim._sigma_xy(table)):
        params = dict(on=table.id, alignHoles=False, assemblyBody=record['reference']['entity'],
            assemblyEdge=True, offset=c['offset'], yaw=c['yaw'],
            assemblyEdgeSide=[c['edgeAxis'], c['edgeSign']])
        receipt = controller.rt.predict('place', params)
        if receipt.get('withinRiskLimit'):
            controller.step(job, 'place', params, receipt=receipt)
            job.postcondition.update(releaseExecuted=True, unconfirmed=True)
            from steward.ego import edge_retention
            if edge_retention.enabled():
                edge_retention.recovery_record(controller, job, record, table, c)
            return


def force_points(record, holes=None):
    ref = record['reference']
    r = np.asarray(ref.get('rotationM', matrix_from_quat(quat_from_yaw(ref['yaw']))))
    return [(np.asarray(ref['position'])+r @ np.asarray(h['p'])).tolist()
            for h in ref.get('holes', []) if holes is None or h['id'] in holes]


def force_report(record, points):
    shape = record.get('edgeGeometry')
    if not active(shape): return None
    origin = np.asarray(record['reference']['position'])
    from steward.ego import edge_retention
    contact = record.get('edgeAnchor', {}).get('bearingPolygonWorld') if edge_retention.enabled() else None
    margins = ([signed_margin(contact, p) for p in points] if contact else
        [signed_margin(shape['bearingPolygon'], np.asarray(p)-origin) for p in points])
    table = record.get('edgeSupport')
    if table:
        rot = matrix_from_quat(quat_from_yaw(table['yaw']))[:2, :2]
        for p in points:
            local = (np.asarray(p)[:2]-table['centre']) @ rot
            margins.append(float(np.min(np.asarray(table['half'])-abs(local))))
    return dict(direction=[0., 0., -1.], inside=bool(margins and min(margins) >= 0),
        marginM=min(margins, default=None), source='registered_bearing_polygon', observed=False)


def guard_force(controller, job, body, holes=None):
    if controller.rt.estimator.held_id: return
    from steward.ego import edge_retention
    records = controller.rt.estimator.stationary_assemblies.records
    record = (edge_retention.retained_record(records, body) if edge_retention.enabled() else None) or records.get(body)
    if not record: return
    report = force_report(record, force_points(record, holes))
    if report is None: return
    job.postcondition['supportForce'] = report
    from steward.ego.working_face import enabled as observation_first
    if observation_first() and record.get('workingFaceKeepBody'):
        report.update(bodyRetained=True, adjusted=False, unconfirmed=not report['inside'])
        return False
    from steward.ego import edge_retention
    if edge_retention.enabled() and record.get('edgeRetained'):
        report.update(bodyRetained=True, adjusted=False, unconfirmed=not report['inside'])
        if not report['inside']: controller.warn('force_point_outside_support', **report)
        return False
    if report['inside']: return
    controller.warn('force_point_outside_support', **report)
    key = tuple(sorted(holes or (h['id'] for h in record['reference'].get('holes', []))))
    tried = record.setdefault('forceGuardAttempts', [])
    if list(key) in tried:
        report['unconfirmed'] = True
        return False
    tried.append(list(key))
    # Include future load points in edge containment and recompute before taking
    # the next loose object. A translation cannot fix a bearing-internal moment;
    # keep that residual explicit instead of declaring it safe after movement.
    record['forcePoints'] = [(np.asarray(p)-record['reference']['position']).tolist()
                             for p in force_points(record, holes)]
    child = {}; root = job.postcondition; job.postcondition = child
    try:
        controller.flip(job, dict(body=body, turnOver=False, _edgeAdjusted=True))
    finally:
        job.postcondition = root
    root.setdefault('stages', []).extend(child.get('stages', []))
    latest = controller.rt.estimator.stationary_assemblies.records.get(body, record)
    final = force_report(latest, force_points(latest, holes))
    root['supportForce'] = dict(final or report, adjusted=bool(child.get('releaseExecuted')),
        unconfirmed=not (final or report)['inside'])
    return bool(child.get('releaseExecuted'))


def recover_body_face(controller, record, view):
    """Disentangle a merged assembly outline using the existing shape model.

    The result is contact geometry, not a replacement identity or a fabricated
    hole registration. All offsets remain relative to the execution reference.
    """
    if not view or not len(view.get('points', [])):
        return
    from steward.ego import session_faces
    from steward.ego.assembly_precision import tolerance
    ent = controller.rt.ego_state().find(record['reference']['entity'])
    if ent is None: return
    spec = session_faces.model(controller, ent)
    patches = session_faces.bands(np.asarray(view['points']), tolerance(controller))
    fits = [session_faces.fit_band(p, spec, tolerance(controller), record['reference']['position']) for p in patches]
    fits = [f for f in fits if f is not None and
        np.linalg.norm(f['position'][:2]-record['reference']['position'][:2]) < max(record['reference']['size'][:2])/2]
    if not fits: return
    match = min(fits, key=lambda f:f['outlineErrorM'])
    record['contactReference'] = dict(position=match['position'].tolist(), size=match['size'].tolist(),
        yaw=match['yaw'], source='cached_body_face_outline', observedAt=view['stamp'],
        outlineErrorM=match['outlineErrorM'])
