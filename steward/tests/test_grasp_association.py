"""A renamed held track retains spatial evidence from before it is transported."""
import numpy as np
import pytest
from test_mating_transport_yaw import tracked_part


def test_missing_requested_track_records_actual_candidate_before_transport():
    est, track = tracked_part()
    old = track.pos.copy()
    assert est.mark_held('gone', at=(old + [.01, 0, 0]).tolist()) == track.id
    evidence = dict(track.grasp_association)
    assert evidence['requested'] == 'gone'
    assert evidence['held'] == track.id
    assert evidence['category'] == 'car_wheel'
    assert evidence['positionAtGrasp'] == pytest.approx(old)
    assert evidence['distanceToGripperM'] == pytest.approx(.01)
    assert evidence['identityVerified'] is False
    track.pos += [.2, .3, .1]
    assert track.grasp_association == evidence
    est.release_held()
    assert track.grasp_association is None


def test_same_id_and_regrasp_get_new_snapshots():
    est, track = tracked_part()
    est.mark_held(track.id, at=track.pos.tolist())
    first = dict(track.grasp_association)
    est.release_held()
    track.pos += [.1, 0, 0]
    est.mark_held(track.id, at=track.pos.tolist())
    assert track.grasp_association['requested'] == track.id
    assert track.grasp_association['positionAtGrasp'] != first['positionAtGrasp']


def test_missing_target_with_no_near_track_still_refuses():
    est, track = tracked_part()
    with pytest.raises(KeyError):
        est.mark_held('gone', at=(track.pos + np.ones(3)).tolist())
    assert track.grasp_association is None
