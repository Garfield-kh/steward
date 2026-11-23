"""Frame provenance must survive the public belief and the planner boundary."""
from types import SimpleNamespace

import numpy as np
import pytest

from steward.ego.estimator import EgoEstimator, Track
from steward.ego.esim import Esim, PredictionError
from steward.math3d import quat_from_yaw
from steward.scene.schema import Est


def entity(name, holes=(), source="observed", yaw=0):
    return SimpleNamespace(id=name, pose=Est(value={"p": [0, 0, 0], "q": quat_from_yaw(yaw)}),
                           rigid=SimpleNamespace(links=[SimpleNamespace(affordances=[
                               Est(value={"screw-hole": h}, src=source) for h in holes])]))


def test_unregistered_features_cannot_create_an_alignment_receipt():
    holes = [{"id": "a", "p": [0, 0, 0]}]
    held, mate = entity("load", holes, "prior"), entity("mate", holes)
    es = SimpleNamespace(find=lambda eid: mate)
    sim = Esim.__new__(Esim)
    explicit = sim._place_hole_align(es, held, mate, {"alignTo": "mate"})
    assert explicit[1] is not None and any("RSI_HOLE_FRAME" in issue for issue in explicit[3])
    result = sim._place_hole_align(es, held, mate, {})
    assert result[1] is not None
    assert "RSI_HOLE_FRAME" in result[3][0]
    assert sim._place_hole_align(es, held, mate, {"alignHoles": False}) == (None, None, None, [])


@pytest.mark.parametrize("size,period", [([.4, .1, .03], 180), ([.4, .39, .03], 90)])
def test_shape_axis_is_never_a_directed_semantic_heading(size, period):
    t = Track("load", {"unseen_category": 1}, np.zeros(3), np.ones(3)*.001,
              .4, .001, np.array(size), 3, 0., 1.)
    t.screw_holes = [{"id": "feature", "p": [0., 0., 0.]}]
    est = EgoEstimator.__new__(EgoEstimator)
    ent = est._track_entity(t)
    orientation = ent.state_vars["orientation"].value
    assert orientation["axisPeriodDeg"] == period
    assert orientation["directedYawObserved"] is False


def test_objects_without_alignment_features_or_yaw_target_keep_parent_state_shape():
    t = Track("load", {"unseen_category": 1}, np.zeros(3), np.ones(3)*.001,
              .4, .001, np.array([.4, .1, .03]), 3, 0., 1.)
    est = EgoEstimator.__new__(EgoEstimator)
    assert "orientation" not in est._track_entity(t).state_vars


def test_feature_provenance_is_not_discarded_by_esim():
    holes = [{"id": "a", "p": [.1, 0, 0], "frame": "unregistered"}]
    assert Esim._screw_holes(entity("load", holes))[0]["frame"] == "unregistered"


def test_observed_holes_are_reexpressed_after_pose_filtering():
    import math
    t = Track("load", {"new": 1}, np.array([.5, -.2, .8]), np.ones(3)*.001,
              math.pi/2, .001, np.array([.3, .1, .03]), 3, 0., 1.)
    det = {"screwHoles": [{"id": "landmark", "p": [.1, 0., .01]}]}
    EgoEstimator._remember_screw_holes(t, det, np.array([.52, -.18, .8]), 0.)
    # World feature is (.62, -.18, .81); track frame is rotated by 90 degrees.
    assert t.screw_holes[0]["p"] == pytest.approx([.02, -.12, .01])
