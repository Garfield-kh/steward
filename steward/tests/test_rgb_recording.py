"""Clock-driven sensor recording and independent NovelAnything scene export geometry."""
import base64
import io
import json
import threading
import zipfile

import numpy as np
import pytest
from PIL import Image

from steward.assets.generators import steward_bot
from steward.ego.novelanything_scene import prepare_gripper_scene
from steward.ego.runtime import EgoRuntime, RobotLink
from steward.ego.server import ego_api
from steward.net import HttpError, call
from steward.math3d import matrix_from_quat
from steward.world.sensor_cameras import SensorCameras
from steward.world.rgb_recording import RecordingConfig, SensorRecordings
from steward.world.server import StewardHost
from steward.world.sim import WorldSim


def test_recorder_samples_simulation_clock_independently_of_tool_calls_and_saves_only_middle_rgb(kitchen_scene):
    sim = WorldSim(kitchen_scene)
    pipeline = SensorCameras(lambda: sim)
    try:
        assert set(sim.build.names["w:robot-1"].cameras) == {"head_camera", "wide_camera"}
        body = {"fps": 2, "max_frames": 3, "request_key": "record-timing-001"}
        started = pipeline.recordings.start("w:robot-1", body)
        assert pipeline.recordings.start("w:robot-1", body)["recording_id"] == started["recording_id"]
        job = pipeline.recordings.get("w:robot-1", started["recording_id"])
        sim.step(500)
        assert job.done.wait(20)
        result = job.result()
        assert result["written_frames"] == 3
        np.testing.assert_allclose([frame["stamp"] for frame in result["frames"]], [0, 0.5, 1], atol=1e-10)
        assert result["clock"] == "simulation_time"
        text = json.dumps(result)
        assert "camera_to_world" not in text and "simulation_ground_truth" not in text
        assert "w:robot" not in text
        for frame in result["frames"]:
            assert "image" in frame["middle"] and "depth_camera" not in frame
            assert all("image" not in view for view in frame["views"].values())
        with pytest.raises(HttpError, match="different recording"):
            pipeline.recordings.start("w:robot-1", body | {"fps": 1})
    finally:
        pipeline.close(); sim.stop()
    assert not any(thread.name.startswith("rgb-record-") for thread in threading.enumerate())


def test_ego_export_builds_real_scene_zip_and_current_gripper_target_without_motion_or_world_poses(kitchen_root):
    host = StewardHost(kitchen_root, scenario="phase0-kitchen")
    host.sim.perception.odom_pos_sigma = host.sim.perception.odom_yaw_sigma = 0
    runtime = EgoRuntime(RobotLink(host.robot_servers["w:robot-1"].url), steward_bot().to_dict())
    api = ego_api(runtime).start()
    try:
        start = call("POST", f"{api.url}/ego/rgb/record/start", {"fps": 2, "max_frames": 3, "request_key": "ego-record-001"})
        host.sim.step(500)
        job = host.sensor_cameras.recordings.get("w:robot-1", start["recording_id"])
        assert job.done.wait(20)
        exported = call("POST", f"{api.url}/ego/rgb/prepare-novelanything", {"recording_id": job.id, "num_source_views": 2})
        assert exported["context"] == [0, 1] and exported["target"] == [2]
        zipped = zipfile.ZipFile(io.BytesIO(base64.b64decode(exported["archive_base64"])))
        scene = json.loads(zipped.read("scene.json"))
        evidence = json.loads(zipped.read("evidence/manifest.json"))
        assert len(evidence["frames"]) == 3 and len(scene["frames"]) == 3
        assert scene["n_input"] == 2 and scene["n_target"] == 1
        assert scene["metadata"]["camera_id"] == "head_camera"
        assert scene["metadata"]["label"] == "gripper_view_prediction"
        assert scene["metadata"]["source_frame_indices"] == [0, 2]
        assert "image_path" not in scene["frames"][-1]
        camera = np.linalg.inv(np.asarray(scene["frames"][-1]["w2c"]))
        hand = host.sim.gripper_pose("w:robot-1")
        rotation = matrix_from_quat(hand["q"])
        np.testing.assert_allclose(camera[:3, 3], hand["p"], atol=1e-6)
        np.testing.assert_allclose(camera[:3, 2], rotation[:, 0], atol=1e-6)
        np.testing.assert_allclose(-camera[:3, 1], rotation[:, 2], atol=1e-6)
        assert scene["frames"][-1]["fxfycxcy"] == scene["frames"][0]["fxfycxcy"]
        assert scene["metadata"]["input_modalities"] == ["rgb"]
        assert all("depth_path" not in frame for frame in scene["frames"])
        assert "depth" not in json.dumps(evidence)
        assert all("depth" not in name for name in zipped.namelist())
        for frame in scene["frames"][:-1]:
            image = Image.open(io.BytesIO(zipped.read(frame["image_path"])))
            assert image.format == "JPEG" and image.size == (2560, 1920)
        assert all("head_camera_left" not in name and "head_camera_right" not in name for name in zipped.namelist())
        assert "simulation_ground_truth" not in json.dumps(evidence)
        temporal = call("POST", f"{api.url}/ego/rgb/record/export", {"recording_id": job.id})
        assert temporal["reference_frame"] == "first_recorded_head_camera_opencv"
        assert temporal["input_modalities"] == ["rgb"]
        assert len(temporal["frames"]) == 3
        assert [frame["stamp"] for frame in temporal["frames"]] == pytest.approx([0, 0.5, 1])
        np.testing.assert_allclose(temporal["frames"][0]["camera_to_reference"], np.eye(4), atol=1e-10)
        assert all("image" in frame and "camera_to_reference" in frame for frame in temporal["frames"])
        assert "camera_to_world" not in json.dumps(temporal)
        first_c2w = np.asarray(evidence["frames"][0]["camera_to_odometry"])
        expected = first_c2w @ np.asarray(temporal["gripper_to_reference"])
        np.testing.assert_allclose(expected[:3, 3], hand["p"], atol=1e-6)
        host.sim.reset()
        with pytest.raises(ValueError, match="odometry frame"):
            runtime.recording_evidence(job.id)
        with pytest.raises(ValueError, match="odometry frame"):

            prepare_gripper_scene(steward_bot().to_dict(), job.result(), runtime.link.rgb_pose(), 2)
    finally:
        api.stop(); host.stop()


def test_finish_and_cancel_are_independent_of_motion_and_failed_recordings_cannot_export(kitchen_scene):
    sim = WorldSim(kitchen_scene)
    manager = SensorRecordings(lambda: sim)
    try:
        one = manager.start("w:robot-1", {"fps": 2, "max_frames": 4, "request_key": "finish-first-001"})
        job = manager.get("w:robot-1", one["recording_id"])
        sim.step(250)
        assert job.finish()["status"] == "succeeded"
        assert len(job.result()["frames"]) == 2
        two = manager.start("w:robot-1", {"fps": 2, "max_frames": 4, "request_key": "cancel-second-001"})
        cancelled = manager.get("w:robot-1", two["recording_id"])
        assert cancelled.finish(cancel=True)["status"] == "cancelled"
        with pytest.raises(HttpError, match="cancelled"):
            cancelled.result()
        with pytest.raises(HttpError):
            manager.start("w:robot-1", {"fps": 3, "request_key": "invalid-rate-001"})
    finally:
        manager.close(); sim.stop()


def test_render_backpressure_fails_recording_without_silently_dropping_sensor_frames(kitchen_scene, monkeypatch):
    from steward.world.sensor_cameras import CameraRenderer
    entered, release = threading.Event(), threading.Event()
    original = CameraRenderer.render_data

    def blocked(renderer, *args):
        entered.set()
        if not release.wait(20):
            raise RuntimeError("render test barrier timed out")
        return original(renderer, *args)

    monkeypatch.setattr(CameraRenderer, "render_data", blocked)
    sim = WorldSim(kitchen_scene)
    manager = SensorRecordings(lambda: sim, RecordingConfig(queue_capacity=1))
    try:
        started = manager.start("w:robot-1", {"fps": 2, "max_frames": 4, "request_key": "queue-full-001"})
        job = manager.get("w:robot-1", started["recording_id"])
        assert entered.wait(20)
        sim.step(500)
        release.set()
        assert job.done.wait(20)
        assert job.status()["status"] == "failed" and "queue is full" in job.status()["error"]
        with pytest.raises(HttpError, match="failed"):
            job.result()
    finally:
        release.set(); manager.close(); sim.stop()
