"""Diagnostic only: record what each precision-association gate saw. No decision reads this."""
import json
from pathlib import Path

import numpy as np


def _top(counts):
    if not counts:
        return None
    label = max(counts, key=counts.get)
    total = sum(counts.values())
    return [str(label), round(float(counts[label]) / total, 3) if total > 0 else 0.0]


def record(path, tracks, measurements, targets, stamp, camera, shape, n_segments, n_body, accepted):
    try:
        rec = dict(stamp=round(float(stamp), 3), camera=camera, shape=[int(v) for v in shape],
                   nSegments=int(n_segments), nBodyDropped=int(n_body), targets=list(targets),
                   accepted=sorted(accepted), tracks={}, meas=[])
        for eid in targets:
            t = tracks.get(eid)
            if t is None:
                rec['tracks'][eid] = None
                continue
            rec['tracks'][eid] = dict(category=str(t.category), held=bool(t.held),
                semTop=_top(t.semantic_counts), catTop=_top(t.category_counts),
                pos=[round(float(v), 4) for v in t.pos], size=[round(float(v), 4) for v in np.sort(t.size)])
        for i, m in enumerate(measurements):
            seg = m['seg']
            points = np.asarray(seg.points)
            scores = seg.semantics or {}
            total = sum(scores.values()) if scores else 0.
            row = dict(i=i, border=bool(seg.border), inv=round(float(seg.invalid_frac), 4),
                       range=round(float(seg.range), 4), n=int(len(points)),
                       finite=bool(np.isfinite(points).all()), sem=_top(scores), semTotal=round(float(total), 3),
                       centre=[round(float(v), 4) for v in m['centre']],
                       size=[round(float(v), 4) for v in np.sort(m['size'])], near=[])
            for eid, t in tracks.items():
                d = float(np.linalg.norm(np.asarray(m['centre']) - t.pos))
                if d > .25 and eid not in targets:
                    continue
                if d > .6:
                    continue
                reference = t.semantic_counts or t.category_counts
                rt = sum(reference.values())
                overlap = (sum(min(s / total, reference.get(label, 0.) / rt) for label, s in scores.items())
                           if rt > 0 and total > 0 else 0.)
                expected, measured = np.sort(t.size), np.sort(m['size'])
                excess = float(np.max(abs(measured - expected) - np.maximum(.006, .2 * expected)))
                ratio = float(np.max(abs(measured - expected) / np.maximum(expected, 1e-6)))
                row['near'].append([eid, round(overlap, 3), round(d, 4), round(excess, 4), round(ratio, 3)])
            rec['meas'].append(row)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'a') as f:
            f.write(json.dumps(rec) + '\n')
    except Exception as exc:  # the probe must never change behaviour
        try:
            with open(path, 'a') as f:
                f.write(json.dumps(dict(probeError=repr(exc))) + '\n')
        except Exception:
            pass
