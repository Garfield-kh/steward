"""Ego perception: the hull of a segmented region's points."""

from __future__ import annotations

import math

import numpy as np
import pytest

from steward.ego.perception import _hull


def _reference_hull(xy: np.ndarray) -> np.ndarray:
    """The monotone chain over every point, as perception ran it before the fix."""
    pts = np.unique(np.round(xy, 4), axis=0)
    if len(pts) < 3:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2 and ((out[-1][0] - out[-2][0]) * (p[1] - out[-2][1])
                                     - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0])) <= 0:
                out.pop()
            out.append(p)
        return out

    lower, upper = half(pts), half(pts[::-1])
    return np.array(lower[:-1] + upper[:-1])


@pytest.mark.parametrize("kind", ["gauss", "rotated-rect", "grid-ties", "thin-line", "tiny"])
def test_the_hull_of_a_dense_cloud_is_the_full_monotone_chain_hull(kind):
    """F: a depth cloud fills its footprint, and the pure-Python chain over all of its points held
    the belief lock for seconds per frame (phase1 /ego/summary timed out). Points strictly inside
    the extreme octagon are skipped; the hull must stay exactly the same."""
    rng = np.random.default_rng(len(kind))
    for n in (3, 5, 70, 4000, 30000):
        if kind == "gauss":
            xy = rng.normal(size=(n, 2)) * [0.4, 0.1]
        elif kind == "rotated-rect":
            t = 0.7
            xy = rng.uniform(-1, 1, (n, 2)) * [0.3, 0.05] @ np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
        elif kind == "grid-ties":
            xy = np.round(rng.uniform(0, 0.3, (n, 2)), 3)
        elif kind == "thin-line":
            xy = np.c_[rng.uniform(0, 1, n), np.zeros(n)] + rng.normal(0, 1e-5, (n, 2))
        else:
            xy = rng.uniform(0, 1e-4, (n, 2))
        np.testing.assert_allclose(_hull(xy), _reference_hull(xy))
