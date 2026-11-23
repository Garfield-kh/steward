"""Catalogue interfaces constrain fitting without knowledge of object categories."""
import pytest

from steward.ego.esim import Esim, PredictionError
from test_hole_geometry import holes, obj
from test_place_align_holes import _assembly_scene


def group(points, name, mates):
    return [{**h, "group": name, "mates": mates} for h in holes(points)]


def test_catalogue_pairing_removes_cross_group_candidates_and_survives_estimation():
    _, scene = _assembly_scene()
    held, mate = scene.entity("e:wing"), scene.entity("e:fuse")
    sim = Esim.__new__(Esim)
    hh, mh = sim._screw_holes(held), sim._screw_holes(mate)
    candidates = sim._hole_fit_candidates(held, hh, mate, mh)
    assert len(candidates) == 4  # two signs of offset times two yaw directions
    assert all({b for a, b in pairs} == {"wing_0", "wing_1"} for _, _, pairs in candidates)
    assert len(sim._hole_fit_candidates(held, hh, mate, mh, use_catalog=False)) == 14


def test_synthetic_interface_names_and_reciprocal_declarations_drive_matching():
    hh = group([[-.04, 0], [.04, 0]], "socket-x", ("interface-y",))
    good = group([[-.04, .1], [.04, .1]], "interface-y", ("socket-x",))
    bad = group([[-.04, -.1], [.04, -.1]], "other", ("socket-x",))
    for h in bad:
        h["id"] += "_other"
    sim = Esim.__new__(Esim)
    for mh in (good + bad, (good + bad)[::-1]):
        candidates = sim._hole_fit_candidates(obj(), hh, obj(), mh)
        assert len(candidates) == 2
        assert all(o == pytest.approx([0, .1]) for o, _, _ in candidates)
    with pytest.raises(PredictionError, match="RSI_HOLE_GROUP"):
        sim._hole_fit_candidates(obj(), hh, obj(), bad)
    assert len(sim._hole_fit_candidates(obj(), hh, obj(), bad, use_catalog=False)) == 2


def test_missing_metadata_preserves_geometry_and_groups_are_not_truncated():
    sim = Esim.__new__(Esim)
    hh = group([[-.04, 0], [.04, 0]], "x", ("y",))
    plain = holes([[-.04, .1], [.04, .1]])
    assert len(sim._hole_fit_candidates(obj(), hh, obj(), plain)) == 2
    incomplete = group([[0, 0]], "y", ("x",))
    with pytest.raises(PredictionError, match="RSI_HOLE_GROUP"):
        sim._hole_fit_candidates(obj(), hh, obj(), incomplete)
