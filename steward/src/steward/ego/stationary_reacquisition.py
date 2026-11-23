"""Whole-window stationary evidence and one command-free reacquisition.

Only declared odometry noise, encoder FK, sensor clocks and command history.
A statistical outlier is unresolved evidence, never permission to discard it.
"""
from copy import deepcopy
from functools import lru_cache
import math
from statistics import NormalDist
import numpy as np
from steward.ego import capabilities

# Family-wise probability for all two-sided contrasts in all three dimensions.
# This is a sensor-model error budget, independent of placement tolerances.
WINDOW_ALPHA = .005


def enabled():
    return capabilities.get('STEWARD_ASSEMBLY_STATIONARY_REACQUISITION') != '0'


@lru_cache(maxsize=128)
def critical(count):
    return NormalDist().inv_cdf(1-WINDOW_ALPHA/(2*max(1, count)))


def contrasts(times):
    """Linear trend and every sustained prefix/suffix shift (>=2 samples)."""
    n = len(times)
    time = times-times.mean()
    rows = [time / max(np.linalg.norm(time), np.finfo(float).eps)]
    for cut in range(2, n-1):
        rows.append(np.r_[np.full(cut, -1/cut), np.full(n-cut, 1/(n-cut))])
    return np.asarray(rows)


def odometry_evidence(states):
    n = len(states)
    times = np.asarray([s['stamp'] for s in states], float)
    weights = contrasts(times)
    pairs = np.triu_indices(n, 1)
    bound = critical(3*(len(pairs[0])+len(weights)))
    axes = {}; suspect = []; sustained = []
    for key in ('x', 'y', 'yaw'):
        values = np.asarray([s['odometry'][key] for s in states], float)
        sigma = np.asarray([s['odometry'].get('sigma', {}).get('yaw' if key == 'yaw' else 'pos', 0.) for s in states], float)
        if not np.isfinite(values).all() or not np.isfinite(sigma).all() or np.any(sigma < 0):
            return dict(decision='reject', reason='invalid_odometry_noise_or_value', triggerDimensions=[key], axes=axes)
        if key == 'yaw': values = np.unwrap(values)
        floor = np.finfo(float).eps * max(1., float(np.max(abs(values))))
        pair_z = abs(values[pairs[1]]-values[pairs[0]]) / np.maximum(np.hypot(sigma[pairs[0]], sigma[pairs[1]]), floor)
        centered = values-np.median(values)
        zs = abs(weights @ centered) / np.maximum(np.sqrt(weights**2 @ sigma**2), floor)
        crossing = np.flatnonzero(pair_z > bound)
        # The median lies between observed samples. Its conservative envelope
        # follows from the simultaneous pair bounds, without an extra test or
        # assuming an estimated (potentially inflated) sample variance.
        center_z = abs(centered) / np.maximum(np.hypot(sigma, np.max(sigma)), floor)
        centered_outliers = np.flatnonzero(center_z > bound)
        anomalous = sorted(set(pairs[0][crossing]) | set(pairs[1][crossing]))
        moving = bool(np.any(zs > bound))
        # A single arbitrarily large sample can create a slope or a mean shift.
        # Require the temporal evidence to survive removal of ANY one sample
        # before calling it clear motion. Never use this deletion to accept.
        persistent = moving and n >= 4
        if persistent:
            for i in range(n):
                keep = np.arange(n) != i
                w = contrasts(times[keep]); v = centered[keep]; noise = sigma[keep]
                scores = abs(w @ v) / np.maximum(np.sqrt(w**2 @ noise**2), floor)
                if not np.any(scores > bound):
                    persistent = False
                    break
        if len(crossing) or moving: suspect.append(key)
        if persistent: sustained.append(key)
        axes[key] = dict(robustCenter=float(np.median(values)), noiseScale=sigma.tolist(),
            anomalousSamples=[dict(index=int(i), stamp=float(times[i]), value=float(values[i]),
                                   centered=float(centered[i]), centeredZ=float(center_z[i]),
                                   outsideRobustCenterEnvelope=bool(i in centered_outliers)) for i in anomalous],
            crossingPairs=[dict(indices=[int(pairs[0][i]), int(pairs[1][i])], z=float(pair_z[i])) for i in crossing],
            maxPairZ=float(np.max(pair_z)), trendZ=float(zs[0]),
            sustainedShiftZ=float(max(zs[1:], default=0.)),
            trendJudgment='persistent_motion' if persistent else ('suspected' if moving else 'noise_consistent'),
            survivesAnySingleSampleDeletion=bool(persistent))
    return dict(decision='reject' if sustained else ('suspect' if suspect else 'accept'),
        reason='base_motion_evidence' if suspect else None,
        triggerDimensions=suspect, motionDimensions=sustained, axes=axes, familyAlpha=WINDOW_ALPHA,
        comparisonCount=3*(len(pairs[0])+len(weights)), simultaneousZ=bound,
        noiseModel='independent_declared_gaussian_samples', yawUnwrapped=True)


def assess(states, max_gap, tolerance, est, *, vertical=False):
    from steward.ego.relative_entrance_servo import pose
    from steward.ego.stationary_entrance_layers import local_state, native
    result = dict(schema='stationary_evidence.stationary_reacquisition', decision='reject', reason=None,
                  triggerDimensions=[], resampleCount=0, sampleCount=len(states), states=deepcopy(states))
    def reject(reason):
        result['reason'] = reason
        return native(result)
    if len(states) < 2: return reject('continuous_motion_record_unavailable')
    stamps = np.asarray([s['stamp'] for s in states], float)
    if not np.isfinite(stamps).all() or np.any(np.diff(stamps) <= 0): return reject('motion_clock_not_monotonic')
    if np.max(np.diff(stamps)) > max_gap+1e-8: return reject('motion_record_gap')
    first = states[0]; g0, r0 = pose(est, local_state(first))
    for state in states:
        if state.get('episode') != first.get('episode'): return reject('episode_changed')
        if not state.get('gripper', {}).get('holding'): return reject('grasp_not_held')
        if state.get('gripper') != first.get('gripper'): return reject('grasp_changed')
        g, r = pose(est, local_state(state))
        delta = g-g0
        if vertical: delta[2] = 0.
        if not np.isfinite(g).all() or not np.isfinite(r).all(): return reject('invalid_encoder_pose')
        if np.linalg.norm(delta) > tolerance/2 or np.linalg.norm(r-r0) > tolerance/max(np.linalg.norm(g0), tolerance):
            return reject('actuator_not_holding')
    result.update(odometry_evidence(states))
    return native(result)


def join_traces(before, after):
    """Deduplicate the shared boundary only; never sort away a clock reset."""
    if before and after and before[-1]['stamp'] == after[0]['stamp']:
        return before[:-1]+after
    return before+after


def refresh(controller, plan, observation):
    """One fresh pair at the unchanged current command, including final descent.

    A clean extra window alone cannot authorize propagation of old entrances.
    The caller must use the returned NEW joint observation or its old fallback.
    """
    from steward.ego import stationary_entrance_layers as p91
    saved = plan['_relative_entrance_servo']; row = plan['relative_entrance_servo']
    evidence = observation.get('propagationEvidence', {})
    if evidence.get('decision') != 'suspect' or saved.get('stationaryReacquisitionPropagationResamples', 0):
        return None
    saved['stationaryReacquisitionPropagationResamples'] = 1
    record = p91.monitor(controller.ex)
    sequence = record.sequence
    attempt = dict(trigger=deepcopy(evidence), resampleCount=1,
                   commandSequenceBefore=sequence, holdTargetsBefore=deepcopy(record.targets))
    row.setdefault('stationaryReacquisitions', []).append(attempt)
    fresh, detail = p91.measure(controller, plan, row['camera'], controller.ex._state()['stamp'],
                               resample=False, prior=evidence)
    attempt.update(measurement=detail, commandSequenceAfter=record.sequence,
                   holdTargetsAfter=deepcopy(record.targets), finalDecision='accept' if fresh else 'reject')
    if record.sequence != sequence:
        attempt.update(finalDecision='reject', reason='command_during_reacquisition')
        return None
    if fresh is not None:
        saved['last'] = fresh
        row['observations'].append(fresh)
        row['lastVisibleError'] = deepcopy(fresh['error'])
    return fresh
