"""Recovery precision admission. Only robot measurements can renew an interface."""
from copy import deepcopy
from itertools import combinations, permutations
import math
import dataclasses
from collections import deque
import os
import uuid
import numpy as np


def pending(ref, stamp, reason):
    old = ref.get('geometryRelocalization', {})
    ref['geometryRelocalization'] = dict(status='pending', requiredAt=max(
        float(stamp), old.get('requiredAt', -math.inf)), reason=reason)
    if ref.get('componentHoleRegistration'):
        ref['componentHoleRegistration']['historicalPrior'] = True


def eligible(ref, reg):
    gate = ref.get('geometryRelocalization')
    if not gate:
        return not bool((reg or {}).get('historicalPrior') or (reg or {}).get('restoredPrior'))
    return bool(reg and not reg.get('restoredPrior') and not reg.get('historicalPrior')
        and reg.get('observedAt', -math.inf) > gate['requiredAt']
        and reg.get('geometryIdentity', {}).get('confirmed')
        and reg.get('prePickRelocalized'))


def identity(ref, pairs, points, detected, outline, stamp, tolerance):
    """Compare unlabeled pair hypotheses against independent structure.

    Hole IDs select the requested catalogue subset; they never score a fit.
    Competing equal-distance subsets must be ruled out by an extra non-collinear
    hole or a fresh, separate body outline. Half-turns of the SAME unordered
    physical pair are equivalent; this does not establish directed body yaw.
    """
    gate = ref['geometryRelocalization']
    unknown = dict(confirmed=False, reason='independent_structure_unavailable')
    if stamp <= gate['requiredAt'] or len(points) != 2:
        return dict(unknown, reason='fresh_pair_required')
    q = np.array([h['point'] for h in points], float)
    normals = np.array([h.get('normal', [0., 0., 0.]) for h in points])
    if (not np.isfinite(q).all() or not np.isfinite(normals).all()
            or np.any(normals[:, 2] < math.cos(math.radians(25)))
            or np.dot(normals[0], normals[1]) < math.cos(math.radians(10))):
        return dict(unknown, reason='support_plane_unresolved')
    holes = ref.get('holes', [])
    wanted = {p['bodyHole'] for p in pairs}
    if len(wanted) != 2 or not wanted <= {h['id'] for h in holes}:
        return dict(unknown, reason='catalogue_pair_unavailable')
    # Use the same outline completeness checks as stationary-body registration.
    outline_ok = bool(outline and len(outline.get('visibleSizeM', [])) == 3 and outline.get('stamp', -math.inf) > gate['requiredAt']
        and 0 <= stamp-outline['stamp'] <= 2.
        and not outline.get('truncated', True) and outline.get('invalidDepthFraction', 1.) <= .05
        and np.allclose(np.sort(outline.get('visibleSizeM', [])[:2]), np.sort(ref['size'][:2]), atol=.025))
    noise = max(tolerance, *(float(h.get('relUncM', h.get('uncertaintyM', 0.))) for h in points))
    candidates = []
    for a, b in combinations(holes, 2):
        p = np.array([a['p'], b['p']], float)
        if abs(p[0, 2]-p[1, 2]) > noise: continue
        if abs(np.linalg.norm(p[1, :2]-p[0, :2])-np.linalg.norm(q[1, :2]-q[0, :2])) > 2*noise: continue
        for indices in permutations(range(2)):
            seen = q[list(indices), :2]
            da, db = p[1, :2]-p[0, :2], seen[1]-seen[0]
            angle = math.atan2(db[1], db[0])-math.atan2(da[1], da[0])
            c, s = math.cos(angle), math.sin(angle); r = np.array([[c, -s], [s, c]])
            centre = seen.mean(0)-r @ p[:, :2].mean(0)
            outline_match = False
            if outline_ok:
                outline_match = np.linalg.norm(centre-np.asarray(outline['centreM'])[:2]) <= 2*noise
                if 'boxYawRad' in outline:
                    outline_match &= abs((angle-outline['boxYawRad']+math.pi/2) % math.pi-math.pi/2) <= math.radians(10)
                if not outline_match: continue
            extra = False
            for h in holes:
                hp = np.asarray(h['p'])
                if h['id'] in (a['id'], b['id']) or abs(hp[2]-p[0, 2]) > noise: continue
                if abs(np.cross(da, hp[:2]-p[0, :2])) <= noise*np.linalg.norm(da): continue
                predicted = r @ hp[:2]+centre
                for d in detected:
                    dp = np.asarray(d['point'])
                    if min(np.linalg.norm(dp-x) for x in q) <= 2*noise: continue
                    if (np.linalg.norm(dp[:2]-predicted) <= 2*noise and
                            abs(dp[2]-q[:, 2].mean()) <= 2*noise and
                            np.dot(d.get('normal', [0, 0, 0]), normals.mean(0)) >= math.cos(math.radians(10))):
                        extra = True
            if outline_match or extra:
                candidates.append(dict(ids={a['id'], b['id']}, extra=extra, outline=outline_match))
    if not candidates: return unknown
    if any(c['ids'] != wanted for c in candidates):
        return dict(unknown, reason='ambiguous_interface_geometry', hypotheses=len(candidates))
    return dict(confirmed=True, reason='independent_structure_and_plane',
        basis='third_noncollinear_hole' if all(c['extra'] for c in candidates) else 'fresh_separate_body_outline',
        observedAt=stamp, planeNormal=normals.mean(0).tolist(), planePoint=q.mean(0).tolist(),
        directedYawConfirmed=False, unorderedPair=True)


def current_reference(target, est=None):
    ref = (target or {}).get('mateReference', {})
    records = getattr(getattr(est, 'stationary_assemblies', None), 'records', {})
    return records.get(ref.get('entity'), {}).get('reference') or ref


def provenance(target, released_at, est=None, *, restored=False):
    ref = (target or {}).get('mateReference', {})
    reg = (target or {}).get('componentHoleRegistration')
    memory = getattr(est, '_memory_checkpoint', None)
    restored_at = (getattr(memory, 'restore_report', None) or {}).get('restoredAt')
    current = bool(reg and eligible(current_reference(target, est), reg))
    source = ('current_registration' if current else 'coarse_observation' if
        ref.get('localizationSource') == 'coarse_observation' else 'restored_prior' if
        ref.get('restoredPrior') or ref.get('geometryRelocalization') else 'coarse_observation')
    return dict(restoredPrior=bool(restored), processId=os.getpid(), memoryRestoredAt=restored_at,
        placementId=(target or {}).get('placementId'),
        releasedAfterMemoryRestore=(bool(not restored and released_at is not None and released_at > restored_at)
                                    if restored_at is not None else None),
        referenceSource=source, referencePrecision='precise' if current else 'coarse',
        geometryUnconfirmed=not current)


def annotate_release(frame, target, est):
    frame.update(provenance(target, frame.get('releasedAt'), est))
    identity = (target.setdefault('placementId', uuid.uuid4().hex) if target is not None else uuid.uuid4().hex)
    frame.setdefault('id', identity)
    frame['placementId'] = identity


def historical_receipts(value, restored_at):
    """Annotate history in memory; no release event is generated by recovery."""
    if dataclasses.is_dataclass(value):
        for child in vars(value).values(): historical_receipts(child, restored_at)
    elif isinstance(value, dict):
        if 'releasedAt' in value:
            value.update(restoredPrior=True, memoryRestoredAt=restored_at,
                releasedAfterMemoryRestore=False, referenceSource=value.get('referenceSource', 'restored_prior'),
                referencePrecision=value.get('referencePrecision', 'coarse'),
                originalProcessId=value.get('processId'), processId=os.getpid())
        for child in list(value.values()): historical_receipts(child, restored_at)
    elif isinstance(value, (list, tuple, deque)):
        for child in value: historical_receipts(child, restored_at)


def event_view(value):
    """Historical releases are references in snapshots, never fresh result rows."""
    if isinstance(value, dict):
        if value.get('restoredPrior') and 'releasedAt' in value:
            return dict(historicalOnly=True, **{k:deepcopy(value[k]) for k in (
                'id', 'placementId', 'entity', 'object', 'body', 'releasedAt', 'restoredPrior', 'processId',
                'originalProcessId', 'memoryRestoredAt', 'releasedAfterMemoryRestore',
                'referenceSource', 'referencePrecision') if k in value})
        return {k:event_view(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [event_view(v) for v in value]
    return value
