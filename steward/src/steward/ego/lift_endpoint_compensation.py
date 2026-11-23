"""Bounded lift endpoint compensation from stable encoder samples only."""
import math
import numpy as np
from steward.ego import capabilities

MAX_CORRECTIONS = 2


def enabled():
    return capabilities.get('STEWARD_ASSEMBLY_LIFT_ENDPOINT_COMPENSATION') != '0'


def available(ex):
    return (callable(getattr(ex, '_tick', None)) and callable(getattr(ex, '_command', None))
            and getattr(ex, '_liftEndpointCompensationArm', None) is not None)


def stable_sample(ex, tolerance, row):
    """Wait for a full quiet window; a delayed command is not a static bias."""
    from steward.ego.skills import SkillFailed
    window = max(ex.config.settle_s, 2*ex.config.tick_s)
    timeout = max(4*window, 2*ex.config.release_settle_s)
    start = ex._state()['stamp']; history = []; previous_stamp = start
    for _ in range(math.ceil(timeout/ex.config.tick_s)+2):
        state = ex._tick(); stamp = state['stamp']; value = state['joints']['lift']
        if not math.isfinite(stamp) or not math.isfinite(value) or stamp <= previous_stamp:
            raise SkillFailed('lift_encoder_timestamp_or_value_invalid')
        previous_stamp = stamp
        history.append((stamp, value))
        while len(history)>2 and history[1][0] <= stamp-window:
            history.pop(0)
        if history[-1][0]-history[0][0] >= window:
            values = np.array([q for _,q in history])
            spread = float(np.ptp(values))
            if spread <= tolerance/2:
                sample = dict(fromStamp=history[0][0], toStamp=stamp, count=len(values),
                    medianM=float(np.median(values)), actualM=float(value), spreadM=spread,
                    physicalResidualM=float(value-row['physicalTargetM']))
                row['samples'].append(sample)
                return sample, state
        if stamp-start >= timeout: break
    raise SkillFailed('lift_encoder_not_steady')


def endpoint(ex, target, tolerance, row):
    """Keep the ordinary ramp; change only its bounded command endpoint.

    The compensation cap is the executor's existing coarse joint envelope,
    independent of measured task outcomes. Physical residuals and release
    setpoints always refer to target, never to the compensated command.
    """
    from steward.ego.skills import SkillFailed
    limit = ex.config.joint_tol
    row.update(physicalTargetM=float(target), commandLimitM=float(limit),
               toleranceM=float(tolerance), commands=[], samples=[], confirmed=False, reason=None)
    command = float(target); previous_error = None
    try:
        try:
            ex._set_joint('lift', target)
        except SkillFailed as exc:
            row['ordinaryMotionFailure'] = str(exc)
        for correction in range(MAX_CORRECTIONS+1):
            sample, state = stable_sample(ex, tolerance, row)
            error = target-sample['medianM']
            # Verify the original physical target across the whole stable band.
            if abs(error)+sample['spreadM'] <= tolerance:
                row.update(confirmed=True, reason='physical_target_reached',
                           finalResidualM=sample['physicalResidualM'])
                return
            if previous_error is not None and abs(error) >= previous_error-tolerance/4:
                raise SkillFailed('lift_compensation_not_converging')
            if correction == MAX_CORRECTIONS:
                raise SkillFailed('lift_compensation_iteration_limit')
            proposed = command+error
            if not math.isfinite(proposed) or abs(proposed-target) > limit:
                raise SkillFailed('lift_compensation_bound_exceeded')
            start = state['joints']
            path = [dict(start, lift=float(v)) for v in np.linspace(start['lift'], proposed, 5)]
            if not ex._liftEndpointCompensationArm.reachable_path(path, start_joints=start):
                raise SkillFailed('lift_compensation_path_unreachable')
            row['commands'].append(dict(commandM=float(proposed), compensationM=float(proposed-target),
                measuredSteadyErrorM=float(error), fromStamp=state['stamp']))
            command = proposed; previous_error = abs(error)
            ex._command({'lift': command})
            # _settle_for_release must not check against the offset command.
            ex._setpoints['lift'] = target
    except (SkillFailed, KeyError, TypeError, ValueError) as exc:
        row['reason'] = str(exc) if isinstance(exc, SkillFailed) else 'lift_feedback_inputs_unavailable:'+type(exc).__name__
        try:
            residual = float(ex._state()['joints']['lift']-target)
            row['finalResidualM'] = residual if math.isfinite(residual) else None
        except (KeyError, TypeError, ValueError):
            row['finalResidualM'] = None
        if command != target:
            ex._command({'lift': target})
            ex._setpoints['lift'] = target
            row['compensatedCommandWithdrawn'] = True
        if not isinstance(exc, SkillFailed):
            raise SkillFailed(row['reason']) from exc
        raise
