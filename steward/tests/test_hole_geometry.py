"""Observed feature fitting is independent of category names, order and world rotation."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from steward.ego.esim import Esim, PredictionError
from steward.math3d import quat_from_yaw, wrap_angle


def obj(yaw=0):
    return SimpleNamespace(pose=SimpleNamespace(value={"p": [0, 0, 0], "q": quat_from_yaw(yaw)}))


def holes(points):
    return [{"id": f"feature_{i}", "p": [*p, 0]} for i, p in enumerate(points)]


def fit(hp, mp, yaw=None, base=0):
    return Esim.__new__(Esim)._mate_offset_yaw(obj(.37), hp, obj(base), mp, yaw)


def test_two_holes_keep_both_solutions_and_default_is_independent_of_list_order():
    a = holes([[-.04, 0], [.04, 0]])
    b = holes([[0, -.04], [0, .04]])
    for aa in (a, a[::-1]):
        for bb in (b, b[::-1]):
            candidates = Esim.__new__(Esim)._hole_fit_candidates(obj(.37), aa, obj(), bb)
            assert len(candidates) == 2
            off, angle, _ = fit(aa, bb)
            assert off == pytest.approx([0, 0])
            assert angle == pytest.approx(math.pi / 2)  # least rotation from .37
    offset, yaw, _ = fit(a, b, math.pi / 2)
    assert offset == pytest.approx([0, 0])
    assert yaw == pytest.approx(math.pi / 2)


def test_asymmetric_observed_constellation_resolves_rotation_after_shuffling():
    pts = np.array([[.02, .01], [.11, -.01], [-.01, .16]])
    angle = .63
    r = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    target = pts @ r.T + [.21, -.14]
    a, b = holes(pts.tolist()), holes(target.tolist())
    off, yaw, _ = fit(a[::-1], [b[1], b[2], b[0]], base=-.28)
    assert off == pytest.approx([.21, -.14])
    assert wrap_angle(yaw - (angle - .28)) == pytest.approx(0)


def test_explicit_yaw_recomputes_offset_for_off_centre_hole():
    a, b = holes([[.1, 0]]), holes([[.3, .2]])
    off, yaw, _ = fit(a, b, math.pi / 2)
    assert off == pytest.approx([.3, .1])
    assert yaw == pytest.approx(math.pi / 2)
    _, unconstrained, _ = fit(a, b)
    assert unconstrained == pytest.approx(.37)  # never set a one-hole part to the mate's yaw


def test_unlabelled_mounts_keep_alternatives_with_stable_default():
    a = holes([[-.04, 0], [.04, 0]])
    b = holes([[-.04, .1], [.04, .1], [-.04, -.1], [.04, -.1]])
    candidates = Esim.__new__(Esim)._hole_fit_candidates(obj(), a, obj(), b, 0)
    assert len(candidates) == 2
    for aa, bb in ((a, b), (a[::-1], b[::-1])):
        default, _, _ = fit(aa, bb, 0)
        assert default == pytest.approx([0, .1])
    off, _, _ = fit(a, b[:2], 0)
    assert off == pytest.approx([0, .1])


def test_no_truncation_and_no_nonrigid_fit():
    with pytest.raises(PredictionError, match="will not be truncated"):
        fit(holes([[0, 0], [.1, 0]]), holes([[0, 0]]))
    with pytest.raises(PredictionError, match="no rigid planar fit"):
        fit(holes([[0, 0], [.1, 0]]), holes([[0, 0], [.2, 0]]))
    with pytest.raises(PredictionError, match="no rigid planar fit"):
        fit(holes([[0, 0], [.1, 0]]), holes([[0, 0], [.1, 0]]), math.pi / 2)


def test_ambiguous_single_mount_defaults_but_large_search_requires_selection():
    a, b = holes([[0, 0]]), holes([[0, 0], [.1, 0]])
    assert len(Esim.__new__(Esim)._hole_fit_candidates(obj(.37), a, obj(), b)) == 2
    off, yaw, _ = fit(a, b)
    assert off == pytest.approx([0, 0])
    assert yaw == pytest.approx(.37)
    with pytest.raises(PredictionError, match="too many possible assignments"):
        fit(holes([[i, 0] for i in range(8)]), holes([[i, 0] for i in range(8)]))


def test_selector_exact_id_precedes_prefix_and_no_substring_fallback():
    h = [{"id": name, "p": [0, 0, 0]} for name in ("pin", "pin_1", "spin")]
    assert Esim._select_holes(h, "pin", "object") == h[:1]
    with pytest.raises(PredictionError, match="no hole matching"):
        Esim._select_holes(h, "in", "object")
