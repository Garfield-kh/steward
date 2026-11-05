"""Prepare a portable NovelAnything scene from measured middle-camera RGB evidence."""
from __future__ import annotations

import base64
import io
import json
import zipfile

import numpy as np

from steward.ego.kinematics import gripper_pose, reference_to_odometry


def _image(value: str, media: str) -> bytes:
    prefix = f"data:image/{media};base64,"
    if not value.startswith(prefix):
        raise ValueError(f"Expected inline {media} sensor image")
    return base64.b64decode(value[len(prefix):], validate=True)


def prepare_gripper_scene(model: dict, recording: dict, target: dict, source_count: int | None = None) -> dict:
    """Build a scene ZIP without navigation, file-authoring privileges or remote inference.

    Contexts use one measured middle RGB camera in a shared odometry frame. The target camera
    is at the current gripper link origin, forward +X, image-up +Z, with the middle RGB K.
    """
    frames = recording["frames"]
    if recording["status"] != "succeeded" or len(frames) < 2:
        raise ValueError("Finish a recording with at least two middle-camera RGB samples before preparation")
    count = min(16, len(frames)) if source_count is None else source_count
    if type(count) is not int or not 2 <= count <= min(16, len(frames)):
        raise ValueError("num_source_views must be 2–16 and cannot exceed the recorded frame count")
    if target["odometry_frame"] != frames[0]["odometry_frame"] or target["stamp"] < frames[-1]["stamp"]:
        raise ValueError("Recording and current target no longer share the robot odometry frame; record again")
    selected = np.rint(np.linspace(0, len(frames) - 1, count)).astype(int).tolist()
    scene_frames, evidence = [], []
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipped:
        for index, packet in enumerate(frames):
            if (packet["odometry_frame"] != target["odometry_frame"]
                    or abs(packet["stamp"] - frames[0]["stamp"] - index / recording["fps"]) > 1e-6):
                raise ValueError("Recorded sensor timestamps or odometry frames are inconsistent")
            middle = packet["middle"]
            rgb_name = f"evidence/{index:06d}/rgb.jpg"
            zipped.writestr(rgb_name, _image(middle["image"], "jpeg"))
            c2w = reference_to_odometry(model, packet)
            K = np.asarray(middle["K"], dtype=float)
            evidence.append({"stamp": packet["stamp"], "camera_id": "head_camera", "image_path": rgb_name,
                             "K": middle["K"], "camera_to_odometry": c2w.tolist(),
                             "odometry": packet["odometry"], "joint_encoders": packet["joint_encoders"]})
            if index not in selected:
                continue
            scene_frames.append({"image_path": rgb_name,
                                 "w2c": np.linalg.inv(c2w).tolist(),
                                 "fxfycxcy": [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])],
                                 "image_size": [middle["width"], middle["height"]]})
        hand = gripper_pose(model, target)
        hand_cv = np.asarray(hand["gripper_to_camera"]["head_camera"])
        # Target OpenCV right/down/forward correspond to gripper -Y/-Z/+X.
        gripper_from_camera = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        target_c2w = reference_to_odometry(model, target) @ hand_cv @ gripper_from_camera
        scene_frames.append({"w2c": np.linalg.inv(target_c2w).tolist(),
                             "fxfycxcy": scene_frames[0]["fxfycxcy"], "image_size": scene_frames[0]["image_size"]})
        metadata = {"model": "NovelAnything", "camera_id": "head_camera", "recording_id": recording["recording_id"],
                    "input_modalities": ["rgb"], "fps": recording["fps"], "recorded_frames": len(frames), "source_frame_indices": selected,
                    "pose_source": "robot_odometry_and_joint_encoders", "coordinate_frame": "robot_odometry",
                    "label": "gripper_view_prediction", "target_stamp": target["stamp"],
                    "target_camera_to_odometry": target_c2w.tolist(), "gripper_pose": hand}
        scene = {"scene_name": recording["recording_id"], "n_input": count, "n_target": 1,
                 "frames": scene_frames, "metadata": metadata}
        zipped.writestr("scene.json", json.dumps(scene, indent=2) + "\n")
        zipped.writestr("evidence/manifest.json", json.dumps(metadata | {"frames": evidence}, indent=2) + "\n")
    data = archive.getvalue()
    return {"archive_base64": base64.b64encode(data).decode("ascii"), "bytes": len(data),
            "num_source_views": count, "context": list(range(count)), "target": [count], "metadata": metadata}
