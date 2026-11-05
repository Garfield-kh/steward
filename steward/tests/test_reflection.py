"""Reflection evidence: naming doubts, candidates for a sought object and failure analysis."""

import numpy as np
import pytest

from steward.assets.generators import steward_bot
from steward.ego.estimator import Track
from steward.ego.runtime import EgoRuntime, RobotLink
from steward.ego.reflection import Evidence, ReflectionConfig, candidates, mentioned, reflect, size_fit

CFG = ReflectionConfig()
WORDS = {"picture": ["photo", "picture"], "bowl": ["bowl"]}


def _obj(oid, category, extent, **extra):
    return {"id": oid, "category": category, "role": None, "extentM": extent, "color": None, "on": None,
            "position": [0.0, 0.0, 0.5], "lastSeenAgoS": 0.0, "categories": {category: 1.0}, **extra}


# A puzzle box lid printed with a picture: the segmenter matched "photo", the shape has no prototype.
LID = _obj("e:5", "picture", [0.46, 0.46, 0.02])
BOWL = _obj("e:6", "bowl", [0.17, 0.17, 0.06])
TABLE = _obj("e:2", "table", [1.2, 0.8, 0.75])
EVIDENCE = {
    "e:5": Evidence({"picture": 0.55, "plate": 0.3, "unknown": 0.15}, {"picture": 0.8}),
    "e:6": Evidence({"bowl": 0.95}, {"bowl": 0.9}),
    "e:2": Evidence({"table": 0.97}, {}),
}
OBJECTS = [TABLE, LID, BOWL]


def test_a_weakly_perceived_name_is_doubted_with_the_segmenter_words_behind_it():
    [lid] = candidates(OBJECTS, EVIDENCE, WORDS, CFG)
    assert lid["id"] == "e:5"
    assert lid["doubts"] == ["the robot has no shape knowledge of a picture: the name comes from appearance alone",
                             "only 55% sure it is a picture"]
    assert lid["segmenter"] == {"picture": {"score": 0.8, "prompts": ["photo", "picture"]}}
    assert lid["perceived"] == {"picture": 0.55, "plate": 0.3, "unknown": 0.15}


def test_a_segmenter_name_disagreeing_with_the_shape_and_an_odd_size_are_doubted():
    plate = _obj("e:7", "plate", [0.46, 0.46, 0.02])
    [entry] = candidates([plate], {"e:7": Evidence({"plate": 0.9}, {"picture": 0.7})}, WORDS, CFG)
    assert entry["doubts"] == [
        "the segmenter saw it as picture (prompt photo/picture, score 0.70), not plate",
        "its size 0.46x0.46x0.02 m is unlike a typical plate (0.24x0.24x0.02 m)",
    ]


def test_an_object_sought_by_size_ranks_the_misnamed_lid_first():
    ranked = candidates(OBJECTS, EVIDENCE, WORDS, CFG, name="box", size=[0.46, 0.46])
    assert [c["id"] for c in ranked][:1] == ["e:5"]
    assert ranked[0]["why"][0].startswith("size fit 1.00")


def test_an_object_sought_by_a_perceived_alternative_counts_that_probability():
    [first, *_] = candidates(OBJECTS, EVIDENCE, WORDS, CFG, name="plate")
    assert first["id"] == "e:5"
    assert "perception gives plate 30%" in first["why"]


def test_a_thin_top_floating_above_its_support_is_doubted_and_fits_a_taller_box():
    table = _obj("e:4", "table", [2.22, 1.11, 0.755], position=[0.4, -0.63, 0.378])
    lid = _obj("e:5", "picture", [0.44, 0.43, 0.01], position=[-0.41, -0.45, 0.81], on="e:4")
    evidence = {"e:5": Evidence({"picture": 0.54, "chair": 0.14}, {"picture": 0.82}), "e:4": Evidence({"table": 0.64}, {})}
    [entry] = candidates([table, lid], evidence, WORDS, CFG, name="box", size=[0.46, 0.46, 0.12], limit=1)
    assert entry["id"] == "e:5"
    assert ("its bottom lies 0.05 m above the top of e:4: a part below it is not seen "
            "(it may be the top of something taller, or rest on something unseen)") in entry["doubts"]
    assert entry["why"][0].startswith("size fit 0.82")


def test_a_name_sharing_a_word_with_the_sought_one_ranks_first():
    table = _obj("e:4", "dinner table", [2.39, 1.14, 0.75])
    box = _obj("e:2", "picture", [0.54, 0.54, 0.05])
    ranked = candidates([box, table], {}, WORDS, CFG, name="table")
    assert ranked[0]["id"] == "e:4"
    assert ranked[0]["why"][0] == "its name dinner table shares a word with table"


def test_a_smaller_measurement_costs_less_than_a_larger_one():
    assert size_fit([0.46, 0.46, 0.03], [0.46, 0.46, 0.12], 0.35, 0.25) > size_fit([0.46, 0.46, 0.48], [0.46, 0.46, 0.12], 0.35, 0.25)


def test_a_sought_name_without_any_evidence_leaves_well_named_objects_out():
    assert [c["id"] for c in candidates(OBJECTS, EVIDENCE, WORDS, CFG, name="box")] == ["e:5"]


def test_the_runtime_reads_naming_evidence_from_its_tracks():
    runtime = EgoRuntime(RobotLink("http://127.0.0.1:1"), steward_bot().to_dict())
    track = Track(id="e:3", category_counts={"picture": 3.0, "plate": 2.0}, pos=np.array([0.5, 0.0, 0.8]),
                  var=np.full(3, 1e-4), yaw=0.0, yaw_var=1e-3, size=np.array([0.46, 0.46, 0.02]), observations=10,
                  first_seen=0.0, last_seen=0.0, existence=1.0, semantic_counts={"picture": 2.4}, semantic_views=3)
    runtime.estimator.tracks["e:3"] = track
    [entry] = runtime.candidates({"name": "box", "size": [0.46, 0.46]})["candidates"]
    assert entry["perceived"] == {"picture": 0.6, "plate": 0.4}
    assert entry["segmenter"] == {"picture": {"score": 0.8, "prompts": []}}
    with pytest.raises(ValueError, match="size"):
        runtime.candidates({"size": [0.4, -1]})
    runtime.reflect({"tool": "ego_query", "arguments": {"category": "box"}, "message": "no box in the ego state"})
    [entry] = [e for e in runtime.activity_since(0) if e["kind"] == "reflection"]
    assert entry["text"].split("\n") == [
        "ego_query：no box in the ego state", "归类：感知缺失：需要的物体不在 Ego State 里",
        "候选（box）：e:3 picture 吻合 0.20（存疑：the robot has no shape knowledge of a picture: the name comes from appearance alone）"]


def test_a_label_is_reported_with_what_perception_alone_calls_the_object():
    lid = {**LID, "category": "lid"}
    [entry] = candidates([lid], {"e:5": Evidence({"picture": 0.9}, {}, label="lid")}, WORDS, CFG)
    assert entry["doubts"] == ["you labelled it lid; perception alone calls it picture"]
    assert entry["label"] == "lid"


def test_ids_come_from_the_call_arguments_and_the_failure_message():
    assert mentioned({"skill": "pick", "params": {"target": "e:5"}}, "e:9 (chair) is in the way") == ["e:5", "e:9"]
    assert mentioned({"entities": ["e:1", "e:2"], "reference": "e:3"}, "") == ["e:3", "e:1", "e:2"]
    assert mentioned(None, "no box in the ego state") == []


def test_a_failed_lookup_by_name_classifies_the_cause_and_offers_the_doubtful_objects():
    result = reflect(OBJECTS, EVIDENCE, WORDS, CFG,
                     {"tool": "ego_query", "arguments": {"category": "box"}, "message": "no box in the ego state"})
    assert result["cause"]["key"] == "perception-missing"
    assert result["sought"] == {"name": "box", "size": None}
    assert result["targets"] == [] and result["missing"] == []
    assert result["candidates"][0]["id"] == "e:5"


def test_a_failed_pick_reports_its_target_and_ids_no_longer_known():
    result = reflect(OBJECTS, EVIDENCE, WORDS, CFG, {
        "tool": "ego_predict", "arguments": {"skill": "pick", "params": {"target": "e:5"}},
        "message": "height 0.02 m is outside the lift range [0.3, 1.2]; e:11 collides"})
    assert result["cause"]["key"] == "out-of-reach"
    assert [t["id"] for t in result["targets"]] == ["e:5"]
    assert result["missing"] == ["e:11"]
    assert all(c["id"] != "e:5" for c in result["candidates"])


def test_a_blocked_path_offers_no_candidates():
    result = reflect(OBJECTS, EVIDENCE, WORDS, CFG, {
        "tool": "ego_predict", "arguments": {"skill": "navigate", "params": {"x": 1.0, "y": 0.0}},
        "message": "navigate is not feasible: goal pose collides with e:5"})
    assert result["cause"]["key"] not in CFG.candidate_causes
    assert [t["id"] for t in result["targets"]] == ["e:5"]
    assert result["candidates"] == []


def test_reflect_needs_the_failure_message():
    with pytest.raises(ValueError, match="failure message"):
        reflect(OBJECTS, EVIDENCE, WORDS, CFG, {"tool": "skill_run"})
