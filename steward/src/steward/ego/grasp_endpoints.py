"""Preserve observed endpoint polarity across a grasp and encoder transport.

An observation transported with FK is an estimate, not a new in-hand image.
Unknown direction keeps the existing action. No task layout or object template.
"""
from copy import deepcopy
import math
import uuid

import numpy as np

from steward.ego import capabilities
from steward.ego.capability_usage import record


def enabled():
    return capabilities.get('STEWARD_ASSEMBLY_GRASP_ENDPOINTS') != '0'


def observed(feature):
    if not feature or feature.get('polarityObserved') is not True:
        return False
    try:
        return (all(np.asarray(feature[k]).shape == (3,) and np.isfinite(feature[k]).all()
                    for k in ('tip', 'head', 'axis'))
                and math.isfinite(feature['stamp']) and feature['lengthM'] > 0
                and np.linalg.norm(np.asarray(feature['head'])-feature['tip']) > 0)
    except (KeyError, TypeError, ValueError):
        return False


def before_grasp(track, now, same_identity=True):
    """Only the latest independent view, after the last manipulation, is eligible.

    Reuse the existing endpoint detector and its visibility checks. Targeted
    native RGB-D association can supersede the ordinary fused track's feature.
    A newer ambiguous view must not resurrect an older directed observation.
    """
    from steward.ego.fasten_geometry import directed_tool
    feature = track.tool_feature
    view = track.manipulation_view
    diagnostic = dict(detector=dict(called=False, reason='no_independent_view'),
                      featureSource=(feature or {}).get('source'),
                      featureStamp=(feature or {}).get('stamp'))
    track.grasp_direction_diagnostic = diagnostic
    reason = 'no_directed_observation'
    if view and view.get('stamp', -math.inf) >= (feature or {}).get('stamp', -math.inf):
        diagnostic['view'] = {k:deepcopy(v) for k,v in view.items() if k != 'points'}
        feature = None
        if (view.get('source') == 'rgbd_segment' and not view.get('truncated', True)
                and view.get('invalidDepthFraction', 1) <= .05):
            feature = directed_tool(view['points'], track.category, view['stamp'], diagnostic['detector'])
            reason = diagnostic['detector']['reason']
        else:
            reason = 'incomplete_view'
    if not same_identity:
        return None, 'grasp_identity_changed'
    if not observed(feature) or feature.get('source') != 'rgbd_endpoint_width':
        return None, reason
    # Same freshness allowance as the existing stale_grasp_endpoint warning.
    if not 0 <= now-feature['stamp'] <= 20:
        return None, 'stale_or_future_observation'
    if feature['stamp'] <= track.interface_motion_at:
        return None, 'observation_precedes_manipulation'
    return deepcopy(feature), None


def grasp_evidence(track, feature, reason, since, adopted):
    row = dict(graspId=uuid.uuid4().hex, entity=track.id, category=track.category,
               fastener=track.category == 'toy_screw', graspedAt=float(since),
               pregraspDirectionObserved=bool(feature),
               status='observed_basis' if feature and adopted else 'unknown',
               basis='pregrasp_rgbd_plus_grasp_fk' if feature and adopted else 'nominal',
               observedAt=feature['stamp'] if feature else None,
               reason=reason if not feature else None if adopted else 'grasp_endpoints_disabled',
               identityVerified=False, rigidTransportAssumed=True)
    if feature:
        row['pregraspFeature'] = deepcopy(feature)
    track.grasp_association['endpointEvidence'] = deepcopy(row)
    return row


def annotate(feature, evidence):
    if feature is not None:
        feature['offsetEvidence'] = deepcopy(evidence)
        feature['offsetEvidence']['tipInGripperM'] = deepcopy(feature['tip'])
        feature['offsetEvidence']['headInGripperM'] = deepcopy(feature['head'])


def inhand_observation(track):
    """A same-frame RGB-D/FK calibration supersedes the pregrasp estimate."""
    feature = track.held_tool_feature
    if not observed(feature):
        return
    row = deepcopy((getattr(track, 'grasp_association', None) or {}).get('endpointEvidence') or {})
    row.update(status='observed_basis', basis='inhand_rgbd_plus_same_frame_fk',
               observedAt=feature['stamp'], reason=None)
    annotate(feature, row)
    if enabled():
        track.held_grasp_geometry = deepcopy(feature)
    emit(feature['offsetEvidence'], 'offset')


def emit(evidence, stage, *, stamp=None):
    if not evidence or not evidence.get('graspId'):
        return
    record('grasp_endpoint', entity=evidence.get('entity'),
           stamp=stamp if stamp is not None else evidence.get('observedAt'),
           measured=evidence.get('status') == 'observed_basis',
           action=stage == 'release', stage=stage, reason=evidence.get('reason'),
           event_id=evidence['graspId']+':'+stage+((':'+str(stamp)) if stage == 'release' else ''),
           evidence=evidence)


def released(plan, stamp):
    """Called only after the open command, using the consumed endpoint plan."""
    evidence = deepcopy(plan.get('offsetEvidence'))
    if evidence:
        evidence['releasedAt'] = float(stamp)
    emit(evidence, 'release', stamp=stamp)
    return evidence
