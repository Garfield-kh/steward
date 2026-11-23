"""Robot public JSON views. Never feed these summaries back into geometry checks.

Internal SceneState entities retain measurements; only the exported dictionary is
cropped. This policy does not apply to World/sensor/actuator transport.
"""
import numpy as np

from steward.json_types import numpy_json
from steward.scene.schema import SceneState


MAX_PUBLIC_ARRAY_ELEMENTS = 64
OBSERVATION_PUBLIC_FIELDS = frozenset((
    'source', 'stamp', 'centreM', 'visibleSizeM', 'boxYawRad', 'rangeM',
    'truncated', 'invalidDepthFraction', 'camera', 'pointCount',
))


def precision_receipt(item):
    """Keep every placement measurement once, without nested planning transcripts.

    Only precision-bearing skill results use this view; accounting and internal
    receipts retain the original records. No numeric-array limit is changed.
    """
    result = dict(item)
    if any(r.get('supply_pregrasp') for r in result.get('supplyDiagnostics', [])):
        result['supplyDiagnostics'] = [dict(
            {k:v for k,v in r.items() if k not in ('candidates', 'available')},
            candidateEligibility={c['entity']:c.get('reason', 'unknown') for c in r.get('candidates', [])},
            sourceSupports={c['entity']:c['sourceSupport'] for c in r.get('candidates', [])
                            if c.get('sourceSupport') and c['sourceSupport'] != r.get('requestedSupport')})
            if r.get('supply_pregrasp') and 'candidates' in r else r for r in result['supplyDiagnostics']]
    if result.get('precisionReports'):
        # New release diagnostics share the existing receipt budget. The
        # concatenated prose is already in Job.message and each report.
        if any(r.get('releaseConditions') or r.get('referenceSource') for r in result['precisionReports']) and (
                result.get('precisionSummary') == ' '.join(r.get('message','') for r in result['precisionReports'])):
            result.pop('precisionSummary', None)
        conditions=result.get('releaseConditions')
        if conditions and any(r.get('releaseConditions',{}).get('id')==conditions.get('id')
                for r in result['precisionReports']):
            result.pop('releaseConditions')
        if result.get('insertionSearch') and any(
                r.get('insertionSearch') == result['insertionSearch'] for r in result['precisionReports']):
            result.pop('insertionSearch')
        # The same report is already exported as precisionReports; raw copies
        # carry whole observation transcripts and grow on every posterior.
        result.pop('precision', None)
        if any(r.get('seatDelivery', {}).get('releaseHeightM') is not None
               for r in result['precisionReports']):
            result.pop('releaseGapM', None)
        if result.get('nearContactUnconfirmed'):
            # The servo transcript owns the executed target. Keep the nominal
            # frame's provenance without repeating its full hole layout here.
            # Live geometry and session checkpoints remain untouched.
            frame = result.pop('componentFrame', None)
            if frame:
                result['componentFrameSummary'] = {k:v for k,v in frame.items()
                    if k in ('entity','body','releasedAt','transportedAt','source','observed')}
            if any(r.get('seatDelivery') == result.get('seatDelivery')
                   for r in result['precisionReports']):
                result.pop('seatDelivery', None)
            feature = result.pop('releaseToolFeature', None)
            if feature:
                result['releaseToolFeatureSummary'] = {k:v for k,v in feature.items()
                    if k in ('stamp','source','polarityObserved')}
    if 'seatEndpointAttempts' in result:
        # The live capability event carries identity/policy. Keep receipt-only
        # counting possible without repeating the surrounding hole reports.
        result['seatEndpointAttempts'] = [{
            k: (round(v, 6) if isinstance(v, float) else v)
            for k, v in row.items() if k in ('id', 'arrivedAt', 'releasedAt',
                'gripperResidualM', 'failure') or k == 'endpointBlocked' and v
        } for row in result['seatEndpointAttempts']]
    for key in ('stages', 'motionPlans'):
        if key in result:
            result[key] = [r if 'plan' not in r and not isinstance(r.get('receipt'), dict) else dict(skill=r.get('skill'), failure=r.get('failure'),
                executed=(r.get('result') or {}).get('executed'),
                receipt=(r.get('receipt') or {}).get('id'),
                withinRiskLimit=(r.get('receipt') or r).get('withinRiskLimit')) for r in result[key]]
    from steward.ego.fasten_completion import REPORT
    if REPORT and len(result.get('landingErrors', [])) > 2:
        result['landingErrorSamples'] = len(result['landingErrors'])
        result['landingErrors'] = [result['landingErrors'][0], result['landingErrors'][-1]]
    identity = ('part', 'body', 'screw', 'hole', 'partHole', 'status', 'executed',
        'releaseExecuted', 'releasedAt', 'seatedAt', 'releaseHistory', 'verified',
        'seatConfirmed', 'seatStatus', 'seatCheck', 'driveStatus', 'driveStrokeExecuted',
        'driverParked', 'seatRetryCount', 'releaseGapM', 'failure', 'resumed', 'completionRetry', 'evidenceIn',
        'seatDeferred', 'holeSeatObservation', 'landingRecovery', 'nearContactUnconfirmed')
    for key in ('attempts', 'previousAssemblies', 'seatPostchecks'):
        if key in result:
            result[key] = [{k:v for k,v in a.items() if k in identity} for a in result[key]]
    if REPORT and result.get('attempts') and result.get('seatPostchecks'):
        for row in result['seatPostchecks']:
            if row.get('seatCheck') and any(all(k not in row or row[k]==a.get(k) for k in ('body','hole','screw','seatedAt','seatCheck'))
                   for a in result['attempts']):
                saved={k:row[k] for k in ('body','hole','screw','seatedAt','executed') if k in row}
                row.clear();row.update(saved, evidenceIn='attempts')
    if 'releaseObservations' in result:
        result['releaseObservations'] = [dict(
            {k:v for k,v in r.items() if k != 'check'}, checkHole=r.get('hole'))
            for r in result['releaseObservations']]
    if 'assembly' in result:
        result['assembly'] = {k:v for k,v in result['assembly'].items() if k in identity}
    return result


def public_json(value, *, max_array_elements=MAX_PUBLIC_ARRAY_ELEMENTS):
    """Copy public metadata, replacing raw RGB-D and large numeric arrays.

    Observation fields use a whitelist regardless of point count or parent key.
    Other numeric arrays use a size rule regardless of name or list/ndarray type.
    Short vectors/matrices, records and text logs retain their normal structure.
    Unknown objects and invalid numbers still fail strict JSON encoding.
    """
    def visit(item):
        if isinstance(item, dict):
            if 'precisionReports' in item and ('stages' in item or 'motionPlans' in item):
                item = precision_receipt(item)
            if item.get('source') == 'rgbd_segment' and 'points' in item:
                points = np.asarray(item['points'], dtype=float)
                if not np.isfinite(points).all():
                    raise ValueError('Out of range float in RGB-D observation')
                return {**{k: visit(v) for k, v in item.items() if k in OBSERVATION_PUBLIC_FIELDS},
                        'pointCount': len(points), 'rawSamplesOmitted': True}
            return {k: visit(v) for k, v in item.items()}
        if isinstance(item, (np.ndarray, list, tuple)):
            try:
                array = np.asarray(item)
            except (TypeError, ValueError):  # heterogeneous records, not a numeric array
                array = None
            if (array is not None and array.dtype.kind in 'biuf'
                    and array.size > max_array_elements):
                if not np.isfinite(array).all():
                    raise ValueError('Out of range float in public array')
                return dict(shape=list(array.shape), elementCount=int(array.size), rawSamplesOmitted=True)
            if isinstance(item, np.ndarray):
                item = numpy_json(item)
            if isinstance(item, (list, tuple)):
                return [visit(v) for v in item]
        if isinstance(item, np.generic):
            return visit(numpy_json(item))
        return item

    return visit(value)


class EgoSceneState(SceneState):
    """Keep internal scene geometry intact while making JSON exports compact."""

    def to_dict(self):
        return public_json(super().to_dict())
