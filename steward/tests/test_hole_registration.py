"""Synthetic template frames exercise registration, ambiguity and executable selection."""
import ast
import math
from types import SimpleNamespace

import numpy as np
import pytest

from steward.ego.esim import Esim, PredictionError
from steward.ego.estimator import EgoEstimator, Track
from steward.ego.hole_registration import register_hole_template
from steward.ego.priors import SCREW_HOLE_FRAMES, SCREW_HOLES
from steward.math3d import quat_from_yaw, wrap_angle
from steward.scene.schema import Est


def template(size=(.12, .48), centre=(0., 0.)):
    return {"sizeXYM": list(size), "centerXYM": list(centre)}


def pair():
    return [{"id": "a", "p": [0., .06, .01]}, {"id": "b", "p": [0., -.06, .01]}]


def entity(name, holes, yaw=0., source="prior", pos=(0., 0., 0.)):
    return SimpleNamespace(id=name, pose=Est(value={"p": list(pos), "q": quat_from_yaw(yaw)}),
                           rigid=SimpleNamespace(links=[SimpleNamespace(affordances=[
                               Est(value={"screw-hole": h}, src=source) for h in holes])]))


def alternatives(issues):
    message = next(i for i in issues if "geometric alternatives: " in i)
    assert "Verify after release" in message and "re-place" in message
    return ast.literal_eval(message.split("geometric alternatives: ")[1].split(";", 1)[0])


def test_y_long_template_maps_to_x_long_observation_without_changing_hole_spacing():
    original = pair()
    hs, info = register_hole_template(original, template(), [.50, .13, .03])
    assert info["status"] == "axis_registered"
    assert hs[0]["p"] == pytest.approx([.06, 0, .01])
    assert hs[1]["p"] == pytest.approx([-.06, 0, .01])
    assert hs[0]["positionsByRegistration"][1] == pytest.approx([-.06, 0, .01])
    assert original == pair()  # never mutate the catalogue
    assert info["directedYawObserved"] is False


def test_registration_handles_centred_translation_and_either_observed_axis():
    hs, _ = register_hole_template([{"id": "q", "p": [.2, -.04, .03]}],
                                   template(centre=(.2, -.1)), [.12, .48, .03])
    assert hs[0]["p"] == pytest.approx([0, .06, .03])


@pytest.mark.parametrize("spec,observed,reason", [
    (template(), [.48, .47, .03], "near-square"),
    (template((.48, .47)), [.48, .12, .03], "near-square"),
    (template(), [.8, .12, .03], "incompatible"),
    (template(), [.48, .02, .03], "incompatible"),
    (None, [.48, .12, .03], "missing"),
    (template(), [float("nan"), .12, .03], "invalid"),
])
def test_unregistrable_templates_reject_explicit_alignment(spec, observed, reason):
    hs, info = register_hole_template(pair(), spec, observed)
    assert info["status"] == "unregistered"
    assert reason in info["reason"]
    sim = Esim.__new__(Esim)
    held, mate = entity("load", hs), entity("base", pair(), source="observed")
    result = sim._place_hole_align(None, held, mate, {"alignHoles": True})
    assert result[1] is not None and any("RSI_HOLE_FRAME" in issue for issue in result[3])
    assert sim._place_hole_align(None, held, mate, {"alignHoles": False}) == (None, None, None, [])


@pytest.mark.parametrize("mate_yaw", [0., .73, -2.1])
def test_registered_pair_returns_both_world_yaws_and_each_can_be_selected(mate_yaw):
    hs, _ = register_hole_template(pair(), template(), [.48, .12, .03])
    ms, _ = register_hole_template(pair(), template((.48, .12)), [.48, .12, .03])
    held, mate = entity("load", hs), entity("base", ms, mate_yaw)
    sim = Esim.__new__(Esim)
    default = sim._place_hole_align(None, held, mate, {"alignHoles": True})
    choices = alternatives(default[3])
    assert default[1] == pytest.approx(choices[0]["offset"])
    assert default[2] == pytest.approx(choices[0]["yaw"])
    assert len(choices) == 2
    assert abs(wrap_angle(choices[0]["yaw"] - choices[1]["yaw"])) == pytest.approx(math.pi)
    for c in choices:
        assert abs(wrap_angle(c["yaw"] - mate_yaw)) == pytest.approx(math.pi / 2)
        result = sim._place_hole_align(None, held, mate, {"alignHoles": True, "yaw": c["yaw"]})
        assert result[1] == pytest.approx(c["offset"])
        assert wrap_angle(result[2] - c["yaw"]) == pytest.approx(0)
        assert any("RSI_HOLE_REGISTER" in issue for issue in result[3])


def test_off_centre_template_signs_survive_selection_and_support_frame_conversion(monkeypatch):
    hs, _ = register_hole_template(pair(), template(), [.48, .12, .03])
    mh = [{**h, "p": [h["p"][0] + .03, *h["p"][1:]]} for h in pair()]
    ms, _ = register_hole_template(mh, template((.48, .12)), [.48, .12, .03])
    held, mate = entity("load", hs), entity("base", ms, .4, pos=(.2, -.1, 0))
    support = SimpleNamespace(id="support")
    box = SimpleNamespace(cx=.1, cy=.2, yaw=-.3)
    monkeypatch.setattr("steward.ego.esim.entity_box", lambda _: (box, 0, 1))
    es = SimpleNamespace(find=lambda _: mate)
    sim = Esim.__new__(Esim)
    params = {"alignTo": "base", "yaw": .4 + math.pi / 2}
    default = sim._place_hole_align(es, held, support, params)
    choices = alternatives(default[3])
    assert len(choices) == 2
    assert np.linalg.norm(np.array(choices[0]["offset"]) - choices[1]["offset"]) == pytest.approx(.06)
    for c in choices:
        result = sim._place_hole_align(es, held, support, {**params, "offset": c["offset"]})
        assert result[1] == pytest.approx(c["offset"])
    with pytest.raises(PredictionError, match="no rigid planar fit"):
        sim._place_hole_align(es, held, support, {**params, "offset": [5, 5]})


def test_estimator_uses_raw_extent_and_keeps_prior_provenance(monkeypatch):
    monkeypatch.setitem(SCREW_HOLES, "synthetic", pair())
    monkeypatch.setitem(SCREW_HOLE_FRAMES, "synthetic", template())
    track = Track("load", {"synthetic": 1}, np.zeros(3), np.ones(3)*.1,
                  .4, .001, np.array([.48, .12, .03]), 3, 0., 1.)
    est = EgoEstimator.__new__(EgoEstimator)
    ent = est._track_entity(track)  # uncertainty inflation must not change registration
    hs = Esim._screw_holes(ent)
    assert all(h["source"] == "prior" and h["frame"] == "entity" for h in hs)
    assert hs[0]["p"] == pytest.approx([.06, 0, .01])
    assert ent.state_vars["orientation"].value["holeRegistration"]["status"] == "axis_registered"
    track.size[:2] = [.48, .47]
    info = est._track_entity(track).state_vars['orientation'].value['holeRegistration']
    assert info['status'] == 'axis_registered' and info['warnings']


def test_registered_prior_candidates_produce_feasible_full_place_receipts():
    # Existing robot fixture supplies only kinematics; all feature geometry below is synthetic.
    from test_place_align_holes import _assembly_scene, _esim
    model, scene = _assembly_scene()
    held, mate = scene.entity("e:wing"), scene.entity("e:fuse")
    hs, _ = register_hole_template(pair(), template(), [.48, .12, .03])
    ms, _ = register_hole_template(pair(), template((.48, .12)), [.48, .12, .03])
    for obj, holes in ((held, hs), (mate, ms)):
        obj.rigid.links[0].affordances = [Est(value={"screw-hole": h}, src="prior") for h in holes]
    sim = _esim(model)
    result = sim.predict(scene, "place", {"on": mate.id})
    assert result["feasible"], result["issues"]
    choices = alternatives(result["issues"])
    assert result["plan"]["placementTarget"]["yawRad"] == pytest.approx(choices[0]["yaw"])
    assert len(choices) == 2
    for choice in choices:
        selected = sim.predict(scene, "place", {"on": mate.id, "yaw": choice["yaw"]})
        assert selected["feasible"], selected["issues"]
        assert selected["plan"]["alignTo"] == mate.id
        assert selected["plan"]["placementTarget"]["yawRad"] == pytest.approx(choice["yaw"])


def test_centred_feature_registers_without_a_principal_axis():
    """A feature on the outline centre is rotation-invariant: a round or square outline still registers."""
    from steward.ego.hole_registration import register_hole_template
    holes, info = register_hole_template([{"id": "c", "p": [0.0, 0.0, 0.01]}],
                                         {"sizeXYM": [0.05, 0.05], "centerXYM": [0.0, 0.0]}, [0.052, 0.049, 0.02])
    assert info["status"] == "axis_registered"
    assert all(abs(v) < 1e-9 for ps in holes[0]["positionsByRegistration"] for v in ps[:2])
    off, info2 = register_hole_template([{"id": "c", "p": [0.01, 0.0, 0.01]}],
                                        {"sizeXYM": [0.05, 0.05], "centerXYM": [0.0, 0.0]}, [0.052, 0.049, 0.02])
    assert info2["status"] == "unregistered"


def test_small_part_outline_error_within_slack_still_registers():
    """A thin part seen a couple of centimetres too wide keeps a distinguishable long axis and registers."""
    from steward.ego.hole_registration import register_hole_template
    tpl = {"sizeXYM": [0.03, 0.08], "centerXYM": [0.0, 0.0]}
    hole = [{"id": "a", "p": [0.0, 0.012, 0.02]}]
    assert register_hole_template(hole, tpl, [0.10, 0.05, 0.03])[1]["status"] == "unregistered"
    assert register_hole_template(hole, tpl, [0.10, 0.05, 0.03], extent_slack_m=0.03)[1]["status"] == "axis_registered"
    assert register_hole_template(hole, tpl, [0.25, 0.11, 0.03], extent_slack_m=0.03)[1]["status"] == "unregistered"
