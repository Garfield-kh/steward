"""Contradiction-only supply checks using robot priors and perception."""
from steward.ego import capabilities
import numpy as np
from steward.ego.classify import SHAPES
from steward.ego.fasten_plan import value

GENTLE = capabilities.get('STEWARD_ASSEMBLY_GENTLE_IDENTITY') != '0'

ENABLED = capabilities.get('STEWARD_ASSEMBLY_FASTENER_IDENTITY') != '0'


def check(entity, stamp, attempts=(), frames=(), *, measured_only=False, members=()):
    if not ENABLED or entity is None:
        return dict(status='unknown', reason='disabled_or_missing_entity')
    eid = entity.id
    from steward.ego.priors import screw_holes, prior
    fastener = prior(entity.category.value).get('assemblyRole') == 'fastener'
    warnings = []
    if GENTLE and eid in members:
        return dict(status='contradicted', reason='current_assembly_member', actual=eid)
    for a in attempts:
        if eid in {a.get('body'), a.get('part')}:
            return dict(status='contradicted', reason='known_assembly_member', actual=eid)
        ids = {a.get('screw'), *a.get('screwAliases', []),
               *(h.get('screw') for h in a.get('seatHistory', []))}
        from steward.ego.assembly_outcome import refill_ready
        if eid in ids and not refill_ready(a) and (a.get('releaseExecuted') or a.get('seatConfirmed') or a.get('verified')):
            return dict(status='contradicted', reason='previously_released_fastener', actual=eid)
    holes = [a.value['screw-hole'] for link in entity.rigid.links for a in link.affordances
             if isinstance(a.value, dict) and 'screw-hole' in a.value]
    frame_member = any(eid in (f.get('body'), f.get('entity')) for f in frames)
    registered = len([h for h in holes if h.get('frame') == 'entity']) >= 2
    sources = dict(categoryPrior=bool(screw_holes(entity.category.value)),
        holeAttributes=len(holes), registeredMultiHole=registered, releasedInterfaceMember=frame_member)
    if any(sources.values()):
        hard = not GENTLE or frame_member or registered and not fastener
        if hard:
            return dict(status='contradicted', reason='known_interface_member', actual=eid,
                interfaceEvidence=sources)
        warnings.append(dict(reason='known_interface_member', interfaceEvidence=sources))
    def report(**fields):
        return dict(actual=eid, warnings=warnings, **fields)
    view = value(entity, 'contactView')
    if view and (view.get('source') != 'rgbd_segment' or view.get('truncated', True)
            or view.get('invalidDepthFraction', 1) > .05
            or not 0 <= stamp-view.get('stamp', -1) <= 20):
        return report(status='unknown', reason='no_fresh_measured_extent')
    if view:
        size = np.asarray(view.get('visibleSizeM', []), float)
        observed_at = view['stamp']
    elif (not measured_only and entity.pose.src in ('observed', 'fused') and entity.pose.t is not None
            and 0 <= stamp-entity.pose.t <= 20):
        # Ego collision extents derive from the observed track, unlike the
        # nominal tool endpoints. The bound below includes their inflation.
        size = np.asarray(entity.rigid.links[0].geometry['collision'][0]['size'], float)
        observed_at = entity.pose.t
    else:
        return report(status='unknown', reason='no_fresh_measured_extent')
    if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
        return report(status='unknown', reason='no_measured_extent')
    prior = SHAPES['toy_screw']
    # Any rotated extent fits within the prior diagonal. Existing perception
    # extent slack and two-sigma pose uncertainty make this a generous veto.
    cov = (entity.pose.unc or {}).get('cov', [])
    sigma = np.sqrt(max([float(cov[i]) for i in (0, 7, 14)] + [0.])) if len(cov) == 36 else 0.
    bound = float(np.linalg.norm(prior['dims'])*max(prior.get('scale', (1., 1.)))
                  + prior.get('extent_slack', 0.) + 2*sigma)
    mismatch = bool(max(size) > bound)
    if mismatch and GENTLE:
        warnings.append(dict(reason='measured_extent_exceeds_fastener_prior'))
    return report(status='contradicted' if mismatch and not GENTLE else 'unknown',
        reason='measured_extent_exceeds_fastener_prior' if mismatch else 'extent_not_contradicted',
        measuredExtentM=size.tolist(), extentBoundM=bound,
        uncertaintyM=float(2*sigma), observedAt=observed_at)


def held_check(controller, job, support, *, replay_identity=None):
    """Return an incompatible load to its recorded pickup, never to a hole."""
    if not ENABLED:
        return
    scene = controller.rt.ego_state()
    held = controller.rt.estimator.held_id
    result = check(scene.find(held), scene.stamp,
        [a for a in controller.rt.estimator.fastening_attempts.values()
         if not (held == replay_identity and held in {a.get('screw'), *a.get('screwAliases', [])})],
        list(getattr(controller.rt.estimator, 'released_interfaces', {}).values()),
        measured_only=True, members=getattr(controller, '_supply_members', ()))
    job.postcondition.setdefault('identityChecks', []).append(result)
    if result['status'] != 'contradicted':
        return
    from steward.ego.skills import SkillFailed
    from steward.ego.fasten_simple import offset_on
    origin = controller.rt.estimator.fastener_origins.get(held)
    controller._missed_screws = getattr(controller, '_missed_screws', set()) | {held}
    if origin and scene.find(origin['support']):
        controller.step(job, 'place', dict(on=origin['support'],
            offset=offset_on(scene.entity(origin['support']), origin['xy']), alignHoles=False))
    # Missing origin leaves the object held; no fallback onto another surface.
    raise SkillFailed('grasp mismatch: '+result['reason'])
