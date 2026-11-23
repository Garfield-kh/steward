"""Post-release evidence uses RGB-D samples, not the intended position written to belief."""
from itertools import product

import numpy as np
import pytest

from steward.assets.generators import steward_bot
from steward.ego.estimator import EgoEstimator, Track
from steward.ego.perception import Segment
from steward.ego.skills import Job, SkillExecutor


def setup_release():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    track = Track(id="load", category_counts={"box": 1}, pos=np.array([0., 0., .8]),
                  var=np.full(3, .01**2), yaw=0, yaw_var=.01, size=np.array([.08, .06, .04]),
                  observations=5, first_seen=0, last_seen=1)
    est.tracks[track.id] = track
    est.stamp = 1
    est.release_at(track.id, [.2, .3], body={"stamp": 10})
    return est, track


def see(est, track, stamp=11, border=False, invalid=0., centre=(.35, .28, .8)):
    centre = np.array(centre)
    points = np.array(list(product([-.04, .04], [-.03, .03], [-.02, .02]))) + centre
    seg = Segment(points=points, color=np.array([.5, .5, .5]), pixels=len(points),
                  invalid_frac=invalid, border=border, range=1.)
    est.stamp = stamp
    est._fuse_cloud(track, {"seg": seg, "probs": {"box": 1}, "centre": centre,
                            "size": np.array([.08, .06, .04]), "yaw": 0}, np.zeros(3))


def test_release_prediction_is_never_an_observation():
    est, track = setup_release()
    assert track.pos[:2] == pytest.approx([.2, .3])
    evidence = est.placement_evidence(track.id)
    assert evidence["ok"] is None and evidence["status"] == "unknown"
    assert "observation" not in evidence and "deltaXYM" not in evidence


@pytest.mark.parametrize("stamp", [9, 10])
def test_pre_release_and_same_timestamp_frames_do_not_confirm_release(stamp):
    est, track = setup_release()
    see(est, track, stamp=stamp)
    assert est.placement_evidence(track.id)["status"] == "unknown"


def test_fresh_sensor_measurement_reports_residual_independent_of_belief():
    est, track = setup_release()
    see(est, track)
    track.pos[:] = [99, 99, 99]  # later prediction/annotation must not become sensor evidence
    evidence = est.placement_evidence(track.id)
    assert evidence["status"] == "observed"
    assert evidence["observation"]["centreM"] == pytest.approx([.35, .28, .8])
    assert evidence["deltaXYM"] == pytest.approx([.15, -.02])
    assert evidence["ok"] is None  # no claim of orientation, support, alignment or fastening
    evidence["observation"]["centreM"][0] = -999
    assert est.placement_evidence(track.id)["deltaXYM"] == pytest.approx([.15, -.02])


@pytest.mark.parametrize("border,invalid", [(True, 0.), (False, .2)])
def test_partial_or_invalid_depth_stays_unknown(border, invalid):
    est, track = setup_release()
    see(est, track, border=border, invalid=invalid)
    evidence = est.placement_evidence(track.id)
    assert evidence["status"] == "unknown" and evidence["ok"] is None
    assert "deltaXYM" in evidence  # diagnostic, explicitly not a success verdict


def test_missing_track_is_not_replaced_with_nearest_object():
    est, track = setup_release()
    see(est, track)
    del est.tracks[track.id]
    assert est.placement_evidence(track.id)["status"] == "unknown"


def test_next_release_requires_a_new_observation():
    est, track = setup_release()
    see(est, track)
    est.release_at(track.id, [2, 3], body={"stamp": 12})
    assert est.placement_evidence(track.id)["status"] == "unknown"


def test_zero_residual_still_does_not_certify_the_required_relation():
    est, track = setup_release()
    see(est, track, centre=(.2, .3, .8))
    evidence = est.placement_evidence(track.id)
    assert evidence["deltaXYM"] == pytest.approx([0, 0])
    assert evidence["ok"] is None


def test_regrasp_invalidates_previous_placement_evidence():
    est, track = setup_release()
    see(est, track)
    est.mark_held(track.id, at=track.pos.tolist())
    assert track.placement is None
    assert est.placement_evidence(track.id)["status"] == "unknown"


def test_held_views_do_not_verify_a_release():
    est, track = setup_release()
    track.held = True
    see(est, track)
    assert est.placement_evidence(track.id)["status"] == "unknown"


def test_rejected_cloud_does_not_produce_evidence():
    est, track = setup_release()
    track.pinned = True
    see(est, track)
    assert est.placement_evidence(track.id)["status"] == "unknown"


@pytest.mark.parametrize("floor", [False, True])
def test_place_reports_unknown_without_changing_release_commands(floor):
    est, track = setup_release()
    commands = []
    executor = SkillExecutor(commands.append, lambda: {"stamp": 10, "joints": {"reach": .1}}, lambda eid: eid,
                             on_release=lambda eid, spot: est.release_at(eid, spot, body={"stamp": 10}),
                             placement_evidence=est.placement_evidence)
    # Actuator stubs exercise the real place sequence, without a simulator or body connection.
    for name in ("_dock", "_set_joint", "_ramp_joint", "_align_lateral", "_reach_over", "_wait",
                 "_settle_for_release", "_withdraw", "_untilt", "_backoff"):
        setattr(executor, name, lambda *a, **kw: None)
    executor._over_support = lambda plan: True
    job = Job("j", "place", "r")
    plan = {"object": track.id, "support": "surface", "spot": [.2, .3], "floor": [.2, .3], "lift": .8}
    (executor._place_floor if floor else executor._place)(job, plan)
    assert commands == [{"gripper": "open"}]
    assert job.to_dict()["postcondition"]["ok"] is None
    assert "unverified" in job.message
    assert any("[RSI_PLACE_EVIDENCE]" in line for line in job.log)


def test_untriggered_job_serialization_and_optional_callback_are_unchanged():
    job = Job("j", "navigate", "r", message="original")
    executor = SkillExecutor(lambda cmd: None, lambda: {}, lambda eid: eid)
    executor._report_placement(job, "load")
    assert "postcondition" not in job.to_dict()
    assert job.message == "original" and job.log == []


def test_evidence_failure_after_release_does_not_imply_rollback():
    def unavailable(eid):
        raise RuntimeError("sensor disconnected")
    executor = SkillExecutor(lambda cmd: None, lambda: {}, lambda eid: eid, placement_evidence=unavailable)
    job = Job("j", "place", "r")
    executor._report_placement(job, "load")
    assert job.postcondition["ok"] is None
    assert job.postcondition["status"] == "unknown"
