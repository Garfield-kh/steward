"""Observe both released layers before fetching fasteners; one local re-placement.

Only independent current RGB-D outlines and the registered hole correspondence
are used. A target cache, release FK or a covered hole is not a measurement.
"""
from copy import deepcopy
import uuid

from steward.ego import assembly_precision as ap
from steward.ego.fasten_release import released_interface
from steward.ego.skills import SkillFailed


def measure(controller, part, body):
    scene = controller.rt.ego_state()
    frame = released_interface(scene, part, body)
    if not frame:
        return dict(status='unknown', residualM=None, reason='no_interface_frame')
    # Both masks must follow release and any subsequent body/component motion.
    after = max(frame['releasedAt'], frame.get('transportedAt', frame['releasedAt']),
                frame.get('alignmentMovedAt', frame['releasedAt']))
    result = ap.pair_measurement(controller.rt.esim, scene, frame, after)
    if result.get('status') == 'estimated':
        if result['partPose']['observedAt'] != result['bodyPose']['observedAt']:
            return dict(status='unknown', residualM=None, reason='layers_from_different_frames')
        # The existing robot placement precision is a control target, not a
        # claim that nominal holes or an evaluator's snap radius are observed.
        limit = ap.tolerance(controller)
        result.update(limitM=limit, limitSource='robot_interface_precision',
            inBounds=result['residualM'] + result['uncertaintyM'] <= limit,
            needsCorrection=result['residualM'] - result['uncertaintyM'] > limit,
            throughBothLayers=None)
    return result


def publish(controller, job, report):
    job.postcondition['componentAlignment'] = report
    from steward.ego.assembly_session import current
    session = current()
    if session is not None:
        root = job.postcondition
        try:
            session.emit('component_alignment', alignment=deepcopy(report))
        finally:
            job.postcondition = root


def before_fasteners(controller, job, part, body, holes):
    """No visibility failure can hold a load, abort the interface or request input."""
    controller.check()
    est = controller.rt.estimator
    from steward.ego import placement_xy_compensation
    previous = est.released_interfaces.get(part, {}).get('placementXYCompensation')
    if placement_xy_compensation.enabled() and previous:
        job.postcondition['placementXYCompensation'] = deepcopy(previous)
        return previous  # post_component already spent the shared replay budget
    report = dict(schema='component_alignment.v1', id=uuid.uuid4().hex, part=part, body=body,
        holes=[h['id'] for h in holes], observations=[], viewChanges=0,
        replayAttempts=0, replays=0, maxReplays=1, confirmed=False,
        reason='unconfirmed_continue', throughBothLayers=None)
    frame = released_interface(controller.rt.ego_state(), part, body)
    # Resume of an occupied interface must not pull against an existing fastener.
    occupied = any(a.get('releaseExecuted') and a.get('body') == body
        and a.get('part') == part for a in est.fastening_attempts.values())
    from steward.ego.session_initial import occupied as initially_occupied
    occupied |= any(initially_occupied(body, h['id']) for h in holes)
    from steward.ego.session_observation import preserve_component
    preexisting = preserve_component(body, [h['id'] for h in holes], part)
    already_occupied = occupied
    occupied |= preexisting
    old = est.released_interfaces.get(part, {}).get('componentAlignment') or {}
    budgets = getattr(est, '_component_alignment_attempts', {})
    est._component_alignment_attempts = budgets
    budget_key = (body, part)
    replay_used = max(old.get('replayAttempts', 0), budgets.get(budget_key, 0))

    def observe(*, refresh=False):
        try:
            if refresh:
                ap.acquire(controller, [part, body])
            return measure(controller, part, body)
        except (SkillFailed, AttributeError, KeyError, TypeError, ValueError) as exc:
            controller.check()
            return dict(status='unknown', residualM=None, reason='independent_views_unavailable',
                        detail=type(exc).__name__)

    measurement = observe()
    report['observations'].append(ap.compact(measurement))
    if measurement['status'] == 'estimated':
        report['confirmed'] = measurement['inBounds']
        report['reason'] = 'within_control_bound' if report['confirmed'] else 'alignment_unconfirmed'
        if measurement['needsCorrection'] and not occupied and not replay_used and not est.held_id:
            report['replayAttempts'] = 1
            budgets[budget_key] = 1
            report['reason'] = 'replay_pending'
            # Reserve the budget before picking, including on interrupted calls.
            with controller.rt._lock:
                stored = est.released_interfaces.get(part)
                if stored is not None: stored['componentAlignment'] = deepcopy(report)
                ap.install_pose(est, part, measurement['partPose'])
                ap.install_pose(est, body, measurement['bodyPose'])
                # Measured layer poses supersede the pre-placement fit. Keeping
                # it would send the correction straight back to the old target.
                record = est.stationary_assemblies.records.get(body)
                if record:
                    record['reference'].pop('componentHoleRegistration', None)
                getattr(est, '_seat_registrations', {}).pop(part, None)
                controller._registered_alignment = {}
                est.version += 1
            params = dict(on=body, alignHoles=True, hole=report['holes'],
                heldHole=[p['partHole'] for p in measurement['frame']['holePairs']],
                stayNear=True, closeRelease=True)
            publish(controller, job, report)
            released_before = (frame or {}).get('releasedAt', float('-inf'))
            try:
                controller.step(job, 'pick', dict(target=part, stayNear=True))
                # Fresh ordinary admission owns reachability and collision checks.
                corrected = controller.step(job, 'place', params)
                report['replays'] = int(bool(corrected.get('releaseExecuted') or 'releasedAt' in corrected))
            except SkillFailed as exc:
                controller.check()
                report.update(reason='replay_execution_failed', failure=str(exc))
                # Actual motion/admission failure is distinct from missing imagery.
                if est.held_id:
                    publish(controller, job, report)
                    raise
            # Release can precede a failed withdrawal. Count and inspect the
            # actual new release even when the primitive did not return a plan.
            latest = est.released_interfaces.get(part, {})
            if not est.held_id and latest.get('releasedAt', released_before) > released_before:
                report['replays'] = 1
            if report['replays']:
                post = observe(refresh=True)
                report['observations'].append(ap.compact(post))
                report['confirmed'] = post.get('inBounds', False)
                report['reason'] = 'within_control_bound' if report['confirmed'] else 'post_replay_unconfirmed_continue'
        elif occupied:
            report['reason'] = 'occupied_interface_preserved' if already_occupied else 'preexisting_component_preserved'
        elif replay_used:
            report['reason'] = 'replay_budget_already_used'
    report['replayAttempts'] = max(report['replayAttempts'], replay_used)
    with controller.rt._lock:
        stored = est.released_interfaces.get(part)
        if stored is not None:
            stored['componentAlignment'] = deepcopy(report)
    publish(controller, job, report)
    return report
