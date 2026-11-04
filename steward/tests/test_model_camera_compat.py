"""Model evidence preserves the existing head/wide camera rig and deterministic sensor readings."""
import threading

import numpy as np

from steward.assets.generators import steward_bot
from steward.ego.runtime import EgoRuntime, RobotLink
from steward.world.server import StewardHost
from steward.world.sim import WorldSim
from steward.world.sensor_cameras import SensorCameras


def test_model_snapshot_uses_existing_cameras_with_their_own_intrinsics(kitchen_root):
    host = StewardHost(kitchen_root, scenario="phase0-kitchen")
    runtime = EgoRuntime(RobotLink(host.robot_servers["w:robot-1"].url), steward_bot().to_dict())
    try:
        before = host.sim.sensors("w:robot-1")
        mounts = host.sim.model.cam_quat.copy()
        captured = runtime.capture_cameras()
        assert set(captured["views"]) == {"head_camera", "wide_camera"}
        assert set(captured["gripper_pose"]["gripper_to_camera"]) == {"head_camera", "wide_camera"}
        assert captured["views"]["head_camera"]["K"] != captured["views"]["wide_camera"]["K"]
        np.testing.assert_allclose(captured["views"]["head_camera"]["camera_to_reference"], np.eye(4))
        np.testing.assert_array_equal(host.sim.model.cam_quat, mounts)
        assert host.sim.sensors("w:robot-1") == before
        assert "camera_to_world" not in str(captured)
    finally:
        host.stop()


def test_recording_advances_freeze_idle_clock_and_releases_its_clock_hold(kitchen_scene):
    sim = WorldSim(kitchen_scene, freeze_idle=True)
    cameras = SensorCameras(lambda: sim)
    sim.start()
    try:
        result = cameras.recordings.start("w:robot-1", {"fps": 2, "max_frames": 3, "request_key": "freeze-record-001"})
        job = cameras.recordings.get("w:robot-1", result["recording_id"])
        assert job.done.wait(15)
        assert job.status()["status"] == "succeeded", job.status()
        assert job.status()["written_frames"] == 3
        assert job.id not in sim._holds and not sim._step_listeners
        assert all("images" not in frame for frame in job.result()["frames"])
    finally:
        cameras.close(); sim.stop()
    assert not any(t.name.startswith(("sensor-evidence", "rgb-record-")) for t in threading.enumerate())
