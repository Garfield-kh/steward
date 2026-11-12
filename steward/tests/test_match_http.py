"""Face matching releases input scheduling and observes a real HTTP peer disconnect."""

from concurrent.futures import CancelledError, ThreadPoolExecutor
import http.client
import json
import socket
import threading

import numpy as np
import pytest

from steward.assets.generators import steward_bot
from steward.ego.estimator import Appearance, Track
from steward.ego.runtime import EgoRuntime, RobotLink
from steward.ego.server import ego_api


@pytest.fixture
def match_runtime(monkeypatch):
    runtime = EgoRuntime(RobotLink("http://127.0.0.1:9"), steward_bot().to_dict())
    monkeypatch.setattr(runtime.link, "wait_until", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(runtime.link, "release", lambda *args: None)
    monkeypatch.setattr(runtime, "summary", lambda: {"objects": []})
    monkeypatch.setattr(runtime, "_camera_frames", lambda: [{"header": {"stamp": 1.0, "odometry": {}}}])
    monkeypatch.setattr(runtime, "_look_at_faces", lambda *args: None)
    rng = np.random.default_rng(0)
    rgb = rng.uniform(0, 255, (24, 24, 3))
    for eid, category, size, face in (("e:1", "lid", 0.18, rgb),
                                      ("e:2", "puzzle_piece", 0.06, rgb[8:16, 8:16])):
        track = Track(eid, {category: 1.0}, np.array([0.5, 0.0, 0.75]), np.ones(3) * 1e-5,
                      0.0, 1e-4, np.array([size, size, 0.006]), 10, 0.0, 1.0)
        track.appearance = Appearance(face.copy(), (size, size), 1.0)
        runtime.estimator.tracks[eid] = track
    yield runtime
    runtime.stop()


def test_state_queries_finish_before_matching_computation_is_released(match_runtime, monkeypatch):
    from steward.ego import runtime as runtime_module

    entered, release, settled = threading.Event(), threading.Event(), threading.Event()
    original = runtime_module.match_face

    def held_match(*args, **kwargs):
        entered.set()
        try:
            assert release.wait(10), "test did not release computation"
            return original(*args, **kwargs)
        finally:
            settled.set()

    monkeypatch.setattr(runtime_module, "match_face", held_match)
    api = ego_api(match_runtime).start()

    def request(path, body=None):
        connection = http.client.HTTPConnection("127.0.0.1", api.port, timeout=5)
        try:
            connection.request("GET" if body is None else "POST", path,
                               None if body is None else json.dumps(body))
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            future = executor.submit(request, "/ego/match-face", {"entities": ["e:2"], "reference": "e:1", "px": 8})
            try:
                assert entered.wait(5)
                summary = executor.submit(request, "/ego/summary")
                assert summary.result(timeout=5) == (200, {"objects": []})
                assert not settled.is_set()
            finally:
                release.set()
            status, result = future.result(timeout=5)
            assert status == 200 and result["results"][0]["matches"][0]["score"] == 1.0
            assert match_runtime.inputs[0]["path"] == "/ego/match-face"
    finally:
        release.set()
        api.stop()


def test_disconnecting_match_request_stops_before_further_correlation(match_runtime, monkeypatch):
    from steward.ego import runtime as runtime_module

    entered, disconnected, release, settled = (threading.Event() for _ in range(4))
    cancelled_seen = []

    def held_match(*args, cancelled=None, **kwargs):
        entered.set()
        try:
            assert disconnected.wait(10)
            # Client shutdown and peer EOF visibility are separate kernel events. Keep reaching
            # cooperative checkpoints, as the real matcher does between rotations and pieces.
            while not release.is_set():
                if cancelled is not None and cancelled():
                    cancelled_seen.append(True)
                    raise CancelledError("peer disconnected")
                release.wait(0.01)
            return {"matches": []}
        finally:
            settled.set()

    monkeypatch.setattr(runtime_module, "match_face", held_match)
    api = ego_api(match_runtime).start()
    connection = http.client.HTTPConnection("127.0.0.1", api.port, timeout=5)
    try:
        connection.request("POST", "/ego/match-face", json.dumps({"entities": ["e:2"], "reference": "e:1", "px": 8}))
        assert entered.wait(5)
        connection.sock.shutdown(socket.SHUT_RDWR)
        connection.close()
        disconnected.set()
        assert settled.wait(5), "matching continued after the peer disconnected"
        assert cancelled_seen == [True]
        assert not match_runtime.alerts, "normal request cancellation is not a runtime fault"
    finally:
        disconnected.set()
        release.set()
        assert settled.wait(10)
        connection.close()
        api.stop()


def test_prepared_match_uses_copied_pixels_and_pose(match_runtime):
    snapshot = match_runtime.prepare_face_match(["e:2"], "e:1", px=8)
    expected = match_runtime.compute_face_match(snapshot)
    for track in match_runtime.estimator.tracks.values():
        track.appearance.rgb[:] = 0
        track.size[:] *= 2
        track.yaw = 1.0
    assert match_runtime.compute_face_match(snapshot) == expected


def test_correlation_checks_cancellation_between_rotations(monkeypatch):
    from steward.ego import perception

    rotations = []

    def scores(turned, reference, cache):
        rotations.append(True)
        return np.zeros((5, 5))

    monkeypatch.setattr(perception, "_face_score_map", scores)
    with pytest.raises(CancelledError):
        perception.match_face(np.ones((4, 4, 3)), np.ones((8, 8, 3)), cancelled=lambda: bool(rotations))
    assert len(rotations) == 1


def test_captured_matching_inputs_are_independent_of_later_estimator_changes(match_runtime):
    snapshot = match_runtime.prepare_face_match(["e:2"], "e:1", 8)
    before = match_runtime.compute_face_match(snapshot)
    for track in match_runtime.estimator.tracks.values():
        track.appearance.rgb[:] = 0
        track.size[:] *= 2
        track.yaw = 1.7
    match_runtime.estimator.tracks.clear()
    assert match_runtime.compute_face_match(snapshot) == before
    assert before["results"][0]["matches"][0]["score"] == 1.0
