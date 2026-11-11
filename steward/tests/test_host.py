"""Steward host behaviour that depends on the main-thread OpenGL rule."""

from __future__ import annotations

import threading

import pytest

from steward.net import HttpError, call
from steward.world.server import StewardHost


def _pump_while(host: StewardHost, work) -> dict:
    result: dict = {}

    def run() -> None:
        try:
            result["value"] = work()
        except Exception as exc:  # noqa: BLE001 - captured for the assertion
            result["error"] = exc

    thread = threading.Thread(target=run)
    thread.start()
    while thread.is_alive():
        host.pump(0.05)
    return result


def test_episode_load_and_reset_over_wca_run_on_main_thread(kitchen_root, tmp_path):
    host = StewardHost(kitchen_root, scenario="phase0-kitchen", run_dir=tmp_path).start()
    try:
        wca = host.wca.url
        png = _pump_while(host, lambda: call("GET", f"{wca}/wca/render?camera=overview&width=160&height=120"))
        assert png["value"].startswith(b"\x89PNG")
        jpeg = _pump_while(host, lambda: call("GET", f"{wca}/wca/render?camera=overview&width=160&height=120&format=jpeg"))
        assert jpeg["value"].startswith(b"\xff\xd8")
        bad = _pump_while(host, lambda: call("GET", f"{wca}/wca/render?camera=overview&format=gif"))
        assert bad["error"].status == 400
        loaded = _pump_while(host, lambda: call("POST", f"{wca}/wca/episode",
                                                {"action": "load", "scenario": "phase0-kitchen"}))
        assert loaded["value"]["scenario"] == "phase0-kitchen"
        reset = _pump_while(host, lambda: call("POST", f"{wca}/wca/episode", {"action": "reset"}))
        assert reset["value"] == {"reset": True}
        assert call("GET", f"{wca}/wca/status")["t"] < 5.0
    finally:
        host.stop()


def test_rejected_misnamed_phase_preserves_the_sensors_and_the_next_episode(kitchen_root, tmp_path):
    host = StewardHost(kitchen_root, scenario="phase0-chores", phase="phase1", run_dir=tmp_path).start()
    try:
        wca = host.wca.url
        robot = host.robot_servers["w:robot-1"].url
        with pytest.raises(HttpError, match="no misnames configured") as rejected:
            call("POST", f"{wca}/wca/phase", {"level": "phase0-misnamed"})
        assert rejected.value.status == 400
        assert call("GET", f"{wca}/wca/phase")["level"] == "phase1"
        sensors = call("GET", f"{robot}/sensors")
        assert sensors["imaging"] == "rgbd" and sensors["masks"]

        loaded = _pump_while(host, lambda: call("POST", f"{wca}/wca/episode",
                                                {"action": "load", "scenario": "phase0-shuttle"}))
        assert loaded["value"]["scenario"] == "phase0-shuttle"
        assert call("GET", f"{wca}/wca/phase")["level"] == "phase1"
        sensors = call("GET", f"{robot}/sensors")
        assert sensors["imaging"] == "rgbd" and sensors["masks"]
    finally:
        host.stop()


def test_ego_runtime_forgets_ego_state_when_the_episode_restarts(kitchen_root, tmp_path):
    from steward.assets.generators import steward_bot
    from steward.ego.runtime import EgoRuntime, RobotLink

    host = StewardHost(kitchen_root, scenario="phase0-kitchen", run_dir=tmp_path)
    runtime = EgoRuntime(RobotLink(host.robot_servers["w:robot-1"].url), steward_bot().to_dict())
    host.sim.step_seconds(1.0)
    runtime._poll_once()
    assert runtime.estimator.tracks
    old = runtime.estimator
    host.sim.reset()
    runtime._poll_once()
    assert runtime.estimator is not old
    assert runtime.estimator.version == 1
    host.stop()


def test_ego_runtime_forgets_ego_state_on_a_reset_before_the_clock_moved(kitchen_root, tmp_path):
    """With the world frozen while idle, a reset can land at the time the episode started; the new
    episode number still makes the robot forget all it believed, ids and roles included."""
    from steward.assets.generators import steward_bot
    from steward.ego.runtime import EgoRuntime, RobotLink

    host = StewardHost(kitchen_root, scenario="phase0-kitchen", run_dir=tmp_path)
    runtime = EgoRuntime(RobotLink(host.robot_servers["w:robot-1"].url), steward_bot().to_dict())
    runtime._poll_once()
    old = runtime.estimator
    assert old.tracks and old.stamp == 0.0
    runtime._roles[next(iter(old.tracks))] = "table"
    host.sim.reset()
    runtime._poll_once()
    assert runtime.estimator is not old and runtime.estimator.version == 1
    assert not runtime._roles
    host.stop()
