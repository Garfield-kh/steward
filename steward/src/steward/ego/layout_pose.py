"""Rigid hole-layout correspondence. Memory ranks identities, never bounds motion."""
from itertools import combinations, permutations
import math
import numpy as np


def rotation(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]])


def transform(point, fit):
    p = np.asarray(point, float).copy()
    p[:2] = rotation(fit['yawRad']) @ p[:2] + fit['offsetM']
    return p.tolist()


def match(rows, features, occupied=()):
    """Fit observed pair distances, then resolve symmetric identities with the prior.

    A single centre cannot measure heading or prove identity: its nearest prior
    identity is explicitly retained. No maximum displacement/yaw is imposed.
    Z is intentionally unchanged; this is planar body localization, not seating.
    """
    if not rows or not features:
        return None, 'holes_unseen_or_layout_missing'
    p = np.array([r['point'][:2] for r in rows], float)
    q = np.array([h['point'][:2] for h in features], float)
    if not np.isfinite(p).all() or not np.isfinite(q).all():
        return None, 'nonfinite_geometry'
    # Relative sensor uncertainty only; common odometry error is the correction.
    noise = np.array([max(float(h.get('relUncM', h.get('uncertaintyM', 0.))),
                         np.finfo(float).eps) for h in features])
    allowed = np.array([[(r.get('bodyHole') in occupied) == (h.get('kind', 'recess') == 'head')
                         for h in features] for r in rows])
    if any(np.linalg.norm(a-b) <= max(noise[i], noise[j])
           for i, a in enumerate(q) for j, b in enumerate(q) if i < j):
        return None, 'duplicate_or_unresolved_centres'
    hypotheses = []
    if min(int(allowed.any(1).sum()), int(allowed.any(0).sum())) <= 1:
        hypotheses = [(0., q[j]-p[i]) for i, j in zip(*np.where(allowed))]
    else:
        if math.comb(len(p), 2)*len(q)*(len(q)-1) > 4096:
            return None, 'layout_search_limit'
        for a, b in combinations(range(len(p)), 2):
            dp = p[b]-p[a]; span = np.linalg.norm(dp)
            for c, d in permutations(range(len(q)), 2):
                if not (allowed[a,c] and allowed[b,d]): continue
                dq = q[d]-q[c]
                if span <= noise[c]+noise[d] or abs(np.linalg.norm(dq)-span) > noise[c]+noise[d]:
                    continue
                angle = math.atan2(dq[1],dq[0])-math.atan2(dp[1],dp[0])
                angle = (angle+math.pi) % (2*math.pi)-math.pi
                hypotheses.append((angle, (q[c]+q[d])/2-rotation(angle) @ ((p[a]+p[b])/2)))
    choices = {}
    for angle, shift in hypotheses:
        predicted = p @ rotation(angle).T+shift
        distances = np.linalg.norm(predicted[:,None]-q[None,:], axis=2)
        valid = (distances <= 2*noise[None,:]) & allowed
        pairs = [(i,j) for i,j in zip(*np.where(valid)) if valid[i].sum()==1 and valid[:,j].sum()==1]
        if len(pairs) < min(2, int(allowed.any(1).sum()), int(allowed.any(0).sum())) or not pairs: continue
        ii, jj = np.array(pairs).T
        a, b = p[ii], q[jj]
        if len(pairs) > 1:
            u, _, vt = np.linalg.svd((b-b.mean(0)).T @ (a-a.mean(0)))
            r = u @ vt
            if np.linalg.det(r) < 0:
                u[:,-1] *= -1; r = u @ vt
            angle = math.atan2(r[1,0],r[0,0])
        else: r = np.eye(2)
        shift = b.mean(0)-r @ a.mean(0)
        errors = np.linalg.norm(a @ r.T+shift-b,axis=1)
        # Every observed pair must preserve registered spacing within sensor bounds.
        if any(abs(np.linalg.norm(a[i]-a[j])-np.linalg.norm(b[i]-b[j])) > noise[jj[i]]+noise[jj[j]]
               for i,j in combinations(range(len(pairs)),2)): continue
        if np.any(errors > 2*noise[jj]): continue
        key = tuple(pairs)
        prior_cost = float(np.mean(np.linalg.norm(a-b,axis=1)))
        choices[key] = dict(yawRad=angle, offsetM=shift.tolist(), matches=[
            dict(hole=rows[i]['bodyHole'], row=int(i), feature=int(j), kind=features[j].get('kind','recess'),
                 point=features[j]['point']) for i,j in pairs], visibleHoles=len(pairs),
            residualM=float(errors.max()), uncertaintyM=float(max(noise[jj])),
            identityBasis='single_hole_prior' if len(pairs)==1 else 'pair_distances_then_pose_prior',
            priorCostM=prior_cost)
    if not choices: return None, 'layout_inconsistent'
    supported = [c for c in choices.values() if c['visibleHoles']==max(f['visibleHoles'] for f in choices.values())]
    closest = min(c['priorCostM'] for c in supported)
    # Numerically equal translation costs occur for an axially shifted pair.
    # Resolve that symmetry by heading, not by roundoff in a distance sum.
    ranked = sorted((c for c in supported if c['priorCostM'] <= closest+c['uncertaintyM']),
                    key=lambda f:(abs(f['yawRad']),f['priorCostM']))
    best = ranked[0]
    if len(ranked)>1 and abs(ranked[1]['priorCostM']-best['priorCostM']) <= np.finfo(float).eps*max(1.,best['priorCostM']) and abs(
            abs(ranked[1]['yawRad'])-abs(best['yawRad'])) <= np.finfo(float).eps:
        return None, 'ambiguous_layout_identity'
    best['yawMeasured'] = best['visibleHoles'] >= 2
    return best, None
