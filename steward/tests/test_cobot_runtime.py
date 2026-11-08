"""The Cobot Magic twin's Ego Runtime behind a robot port: it perceives through the port, nests
the bowls on its own belief, and starts over when the world does."""

from __future__ import annotations

import time

import pytest

from steward.assets.cobot import cobot_magic
from steward.authoring.service import AuthoringService, build_cobot_kitchen
from steward.ego.cobot_runtime import CobotEgoRuntime
from steward.ego.cobot_skills import MotionConfig
from steward.ego.runtime import RobotLink
from steward.world.server import StewardHost

ROBOT = "w:cobot-1"


@pytest.fixture(scope="module")
def rig(tmp_path_factory):
    root = tmp_path_factory.mktemp("cobot-runtime")
    build_cobot_kitchen(AuthoringService(root))
    host = StewardHost(root, scenario="cobot-kitchen", speed=3.0, run_dir=root / "run", phase="phase1").start()
    try:
        yield host
    finally:
        host.stop()


def _pump_until(host: StewardHost, done, timeout: float) -> None:
    deadline = time.time() + timeout
    while not done() and time.time() < deadline:
        host.pump(0.05)
    assert done(), "timed out"


def _bowl_heights(host: StewardHost) -> dict[str, float]:
    return {e["id"]: float(e["pose"]["value"]["p"][2]) for e in host.sim.world_state()["entities"]
            if e["id"].startswith("w:bowl-")}


def test_the_runtime_reads_the_kind_of_sensor_data_the_world_delivers(rig) -> None:
    runtime = CobotEgoRuntime(RobotLink(rig.robot_servers[ROBOT].url))
    frame, rgbd = runtime.frame_source()
    assert rgbd and frame == runtime.rgbd_frame
    rig.set_phase("phase0")
    try:
        frame, rgbd = runtime.frame_source()
        assert not rgbd and "detections" in frame()
    finally:
        rig.set_phase("phase1")
    with pytest.raises(ValueError, match="unknown mission"):
        CobotEgoRuntime(RobotLink(rig.robot_servers[ROBOT].url), task="fold-laundry")


def test_the_twins_ego_api_speaks_the_alerts_shape_the_console_reads(rig) -> None:
    """The World Harness reads ["active"] from /ego/alerts; the twin used to answer without it,
    and the KeyError flashed on the console's chat poll forever."""
    from steward.ego.cobot_server import cobot_ego_api
    from steward.ego.estimator import EgoEstimator
    from steward.net import call

    live = cobot_ego_api(EgoEstimator(robot_model=cobot_magic().to_dict()), port=0).start()
    try:
        alerts = call("GET", f"{live.url}/ego/alerts")
        assert "active" in alerts and isinstance(alerts["active"], list)
    finally:
        live.stop()


def test_the_runtime_nests_the_bowls_over_the_port_and_reboots_with_the_world(rig) -> None:
    reports: list[dict] = []
    runtime = CobotEgoRuntime(RobotLink(rig.robot_servers[ROBOT].url), task="nest-bowls",
                              config=MotionConfig(tick_s=0.01), report=reports.append)
    runtime.start()
    try:
        _pump_until(rig, lambda: runtime.rounds >= 1, timeout=150)
        result = runtime.results[0]
        assert result["ok"], "\n".join(result.get("log", []))
        rig.sim.step(120)
        heights = _bowl_heights(rig)
        red, green, blue = heights["w:bowl-red"], heights["w:bowl-green"], heights["w:bowl-blue"]
        assert green - red == pytest.approx(0.008, abs=0.006), f"green at {green:.3f} on red at {red:.3f}"
        assert blue - green == pytest.approx(0.008, abs=0.006), f"blue at {blue:.3f} on green at {green:.3f}"
        assert runtime.last_error is None
        # The settling look after the round: nothing is believed held any more, and the bowl it
        # put down is where it lies. Without that look the belief kept the load in the hand and
        # the Ego State showed the bowl riding up with the parked arm, all the way to the reset.
        assert runtime.estimator.held_id is None
        assert not [t for t in runtime.estimator.tracks.values() if t.held], \
            {t.id: t.category for t in runtime.estimator.tracks.values() if t.held}

        # Between rounds the twin keeps the Observer fed with its belief, as the mobile robot does.
        _pump_until(rig, lambda: rig.observer.latest_ego.get(ROBOT) is not None, timeout=10)
        assert "leak" not in rig.observer.latest[ROBOT]

        # The operator's reset starts a new episode: a reboot, then the next round on a fresh belief.
        version = runtime.estimator.version
        rig.sim.reset()
        _pump_until(rig, lambda: any(r.get("cobotEgo") == "reboot" for r in reports), timeout=20)
        assert runtime.estimator.version > version
    finally:
        runtime.stop()
