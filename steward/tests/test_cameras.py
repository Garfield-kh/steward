"""The robot's two head cameras: the narrow head camera and the short-focal-length wide camera."""

from __future__ import annotations

import threading
import time

import pytest

from steward.assets.generators import steward_bot
from steward.ego.perception import decode_rgbd
from steward.ego.runtime import EgoRuntime, RobotLink, RuntimeConfig
from steward.net import HttpError, call
from steward.world.server import StewardHost
from steward.world.sim import WorldSim


def _pump_until(host, done, timeout=60.0):
    deadline = time.time() + timeout
    while not done() and time.time() < deadline:
        host.pump(0.05)
    assert done(), "timed out"


def _off_main(host, fn):
    """Run ``fn`` on a worker thread while the main thread serves the world's render requests."""
    out = {}
    thread = threading.Thread(target=lambda: out.setdefault("value", fn()), daemon=True)
    thread.start()
    _pump_until(host, lambda: not thread.is_alive())
    return out["value"]


def test_each_camera_gives_its_own_frame_and_an_unknown_camera_is_refused(kitchen_scene):
    sim = WorldSim(kitchen_scene, seed=1)
    sim.perception.mode = "rgbd"
    assert [c["id"] for c in sim.sensors("w:robot-1")["cameras"]] == ["head_camera", "wide_camera"]
    head, wide = (decode_rgbd(sim.rgbd("w:robot-1", c)) for c in ("head_camera", "wide_camera"))
    assert head["camera"] == "head_camera" and head["rgb"].shape == (240, 320, 3) and head["fovy"] == 60.0
    assert wide["camera"] == "wide_camera" and wide["rgb"].shape == (360, 480, 3) and wide["fovy"] == 100.0
    assert decode_rgbd(sim.rgbd("w:robot-1"))["camera"] == "head_camera", "no camera named: the primary one"
    with pytest.raises(KeyError):
        sim.rgbd("w:robot-1", "tail_camera")


def test_the_robot_port_names_its_cameras_and_refuses_others(kitchen_root, tmp_path):
    host = StewardHost(kitchen_root, scenario="phase0-kitchen", run_dir=tmp_path, phase="phase1").start()
    try:
        url = host.robot_servers["w:robot-1"].url
        with pytest.raises(HttpError) as err:
            call("GET", f"{url}/sensors/rgbd?camera=tail_camera")
        assert err.value.status == 404
    finally:
        host.stop()


class OneCameraLink:
    """A robot port whose robot has only the head camera, as in a scene built with an older model."""

    def __init__(self, link: RobotLink):
        self.link = link

    def __getattr__(self, name):
        return getattr(self.link, name)

    def sensors(self):
        frame = self.link.sensors()
        frame["cameras"] = [c for c in frame["cameras"] if c["id"] == "head_camera"]
        frame["detections"] = [d for d in frame.get("detections", []) if d["camera"] == "head_camera"]
        return frame


def test_a_camera_the_world_lacks_is_reported_once_and_perception_goes_on_without_it(kitchen_root, tmp_path):
    host = StewardHost(kitchen_root, scenario="phase0-kitchen", run_dir=tmp_path, speed=4.0).start()
    runtime = EgoRuntime(OneCameraLink(RobotLink(host.robot_servers["w:robot-1"].url)), steward_bot().to_dict())
    runtime.start()
    try:
        _pump_until(host, lambda: len(runtime.estimator.tracks) > 3)
        for _ in range(20):
            host.pump(0.05)
        alerts = [a for a in runtime.alerts if "wide_camera" in a["message"]]
        assert len(alerts) == 1 and alerts[0]["count"] == 1
    finally:
        runtime.stop()
        host.stop()


def test_perception_cameras_must_be_cameras_of_the_robot():
    with pytest.raises(ValueError, match="tail_camera"):
        EgoRuntime(RobotLink("http://127.0.0.1:9"), steward_bot().to_dict(), RuntimeConfig(cameras=("tail_camera",)))


def test_at_the_table_edge_the_wide_camera_reads_the_face_the_head_camera_cannot_see(kitchen_root, tmp_path):
    host = StewardHost(kitchen_root, scenario="phase0-kitchen", run_dir=tmp_path, phase="phase1", speed=4.0).start()
    runtime = EgoRuntime(RobotLink(host.robot_servers["w:robot-1"].url), steward_bot().to_dict())
    runtime.start()
    try:
        plate = host.sim.entity_obb("w:plate-1").center
        host.sim.command("w:robot-1", {"base": [float(plate[0]) - 0.62, float(plate[1]), 0.0]})

        def plate_track():
            return next((t for t in runtime.estimator.tracks.values() if t.category == "plate"
                         and abs(t.pos[0] - plate[0]) < 0.1 and abs(t.pos[1] - plate[1]) < 0.1), None)

        _pump_until(host, lambda: abs(runtime.estimator.odometry["x"] - (plate[0] - 0.62)) < 0.03 and plate_track())
        track_id = plate_track().id
        seen = _off_main(host, lambda: runtime.inspect(track_id, 8))
        assert seen["camera"] == "wide_camera" and seen["coverage"] > 0.8
        head = RuntimeConfig(cameras=("head_camera",))
        only_head = EgoRuntime(runtime.link, steward_bot().to_dict(), head)
        only_head.estimator = runtime.estimator
        narrow = _off_main(host, lambda: only_head.inspect(track_id, 8))
        assert narrow["camera"] == "head_camera" and narrow["coverage"] < seen["coverage"] / 1.5
        assert narrow["hint"].startswith("too close"), "the head camera alone cannot see the strip next to the edge"
    finally:
        runtime.stop()
        host.stop()
