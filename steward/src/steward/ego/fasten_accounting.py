"""Material ownership and irreversible release accounting from robot evidence."""
from copy import deepcopy
from steward.ego import supply_reliability as sr
import numpy as np

from steward.ego.fasten_plan import value
from steward.ego.skills import SkillFailed


def kit_support(controller, support):
    """Remember the stock bearing surface across jobs and body flips."""
    from steward.ego.assembly_session import current
    session = current()
    if session is not None:
        return session.source_support(support)
    est = controller.rt.estimator
    body = getattr(controller, 'body_id', None)
    if body is None:
        return support
    original = est.fastener_kits.setdefault(body, support)
    return original


def belongs(controller, entity, support):
    from steward.ego.assembly_session import current
    session = current()
    if session is not None:
        return session.owns(entity.id)
    scene = controller.rt.ego_state()
    origin = controller.rt.estimator.fastener_origins.get(entity.id, {})
    return (controller.rt.esim._anchor(scene, entity).id == support
            and origin.get('kitSupport', support) == support)


def departure(scene, attempt):
    """Locate this very screw outside its hole in a new independent view.

    Bad orientation, a displaced component, an empty mask or a stale pose do
    not establish departure. A newly named loose screw is not an identity match.
    """
    from steward.ego.fasten_supply import measured_fastener
    since = max(attempt.get('seatedAt', 0), attempt.get('transportedAt', 0))
    for eid in dict.fromkeys([attempt['screw'], *attempt.get('screwAliases', [])]):
        entity = scene.find(eid)
        if entity is None or value(entity, 'held', False):
            continue
        feature = measured_fastener(entity, scene.stamp)
        if feature is None:
            feature = value(entity, 'toolFeature')
        if (not feature or feature.get('source') != 'rgbd_endpoint_width'
                or not since < feature.get('stamp', -1) <= scene.stamp
                or scene.stamp-feature['stamp'] > 20):
            continue
        # Both ends must be clear of the hole capture neighbourhood. This also
        # excludes a shaft merely lying across the hole with its head displaced.
        entry = np.asarray(attempt['entry'])
        tip, head = np.asarray(feature['tip']), np.asarray(feature['head'])
        segment = head[:2]-tip[:2]
        t = np.clip(np.dot(entry[:2]-tip[:2], segment)/max(np.dot(segment, segment), 1e-12), 0, 1)
        distance = float(np.linalg.norm(tip[:2]+t*segment-entry[:2]))
        vertical_clear = (min(tip[2], head[2])-entry[2] > feature.get('lengthM', np.linalg.norm(head-tip))+.012
                          or max(tip[2], head[2])-entry[2] < -.025)
        if distance > .025+feature.get('radiusM', 0.) or vertical_clear:
            return dict(screw=eid, observedAt=feature['stamp'], distanceXYM=distance,
                        source='independent_same_screw_elsewhere')
    return None


def allow_release(est, body, hole, screw):
    previous = est.fastening_attempts.get((body, hole))
    if not previous or not previous.get('releaseExecuted'):
        return
    from steward.ego import seating_geometry
    if (seating_geometry.enabled() and previous.get('seatRetryCount') == 1
            and previous.get('screw') == screw == getattr(est, 'held_id', None)):
        # The single same-object pickup removed that object's old occupancy.
        return
    from steward.ego.assembly_outcome import refill_ready
    if refill_ready(previous):
        return
    from steward.ego import first_interface
    if (first_interface.enabled() and previous.get('firstInterfaceRetryGranted') and
            previous.get('screw') == screw == getattr(est, 'held_id', None)):
        # The admitted identity-checked pickup already removed this very object.
        # Its old release cannot also occupy the hole; no extra departure view.
        return
    evidence = previous.get('departureEvidence') or {}
    if (evidence.get('screw') != screw or evidence.get('observedAt', -1) <=
            max(previous.get('seatedAt', 0), previous.get('transportedAt', 0))):
        raise SkillFailed('hole already occupied by an actual release; no new screw permitted')


def record_release(est, plan, stamp):
    """Called in the place finally block, including cancellation after opening."""
    target = plan.get('seatTarget')
    if not target or plan.get('_releaseAccounted'):
        return
    body, hole = plan['seatBody'], plan['seatHole']
    key = body, hole
    previous = est.fastening_attempts.get(key, {})
    point = list(target['point'])
    history = deepcopy(previous.get('releaseHistory', []))
    if previous.get('releaseExecuted') and not history:
        history.append(dict(screw=previous['screw'], stamp=previous['seatedAt']))
    history.append(dict(screw=plan['object'], stamp=stamp))
    est.fastening_attempts[key] = dict(previous, body=body, hole=hole,
        part=target.get('componentFrame', {}).get('entity') if target.get('componentFrame') else plan['seatPart'],
        partHole=target['partHole'], screw=plan['object'], entry=point,
        headBeforeRelease=point, headBeforeSource='release_action_only',
        seatedAt=stamp, releaseExecuted=True, restoredPrior=False, status='released_unverified',
        verified=False, seatConfirmed=False, seatStatus='unknown', seatDeferred=True,
        seatCheck=dict(status='unknown', passed=None, reason='release_awaiting_observation'),
        componentFrame=deepcopy(target.get('componentFrame')), releaseHistory=history,
        departureEvidence=None)
    from steward.ego.grasp_binding import annotate_release
    annotate_release(est, est.fastening_attempts[key], plan['object'])
    from steward.ego import assembly_outcome
    if assembly_outcome.enabled():
        est.fastening_attempts[key].pop('refillAuthorized', None)
    if plan.get('_wristReleaseFrame'):
        est.fastening_attempts[key]['releaseFrame'] = deepcopy(plan['_wristReleaseFrame'])
    from steward.ego.hole_seating import apply as apply_hole
    apply_hole(est.fastening_attempts[key], plan)
    est.fastening_attempts[key]['nearContactUnconfirmed'] = bool(plan.get('nearContactUnconfirmed'))
    est.fastening_attempts[key]['releaseHeight'] = deepcopy(plan.get('releaseHeight', {}))
    from steward.ego.release_conditions import enabled as release_conditions_enabled
    if release_conditions_enabled() and plan.get('releaseConditions'):
        est.fastening_attempts[key]['releaseConditions'] = deepcopy(plan['releaseConditions'])
    plan['_releaseAccounted'] = True
    from steward.ego.assembly_session import current
    if current() is not None:
        current().sync_releases()
        current().emit('release_checked', body=body, hole=hole, screw=plan['object'],
            seatedAt=stamp, nearContactUnconfirmed=bool(plan.get('nearContactUnconfirmed')),
            check=deepcopy(est.fastening_attempts[key]['seatCheck']))
