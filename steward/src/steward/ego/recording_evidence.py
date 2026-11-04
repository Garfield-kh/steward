"""Export RGB recordings with measured camera poses in the first middle-camera frame."""
from __future__ import annotations

import numpy as np

from steward.ego.kinematics import gripper_pose, reference_to_odometry


def recording_evidence(model: dict, recording: dict, target: dict) -> dict:
    """Export all successful samples and current gripper geometry without motion or inference."""
    frames = recording["frames"]
    if recording["status"] != "succeeded" or len(frames) < 2:
        raise ValueError("Finish an RGB recording with at least two frames before exporting evidence")
    if target["odometry_frame"] != frames[0]["odometry_frame"] or target["stamp"] < frames[-1]["stamp"]:
        raise ValueError("Recording and target no longer share the robot odometry frame; record again")
    first_inverse = np.linalg.inv(reference_to_odometry(model, frames[0]))
    exported = []
    for index, packet in enumerate(frames):
        if (packet["odometry_frame"] != target["odometry_frame"]
                or abs(packet["stamp"] - frames[0]["stamp"] - index / recording["fps"]) > 1e-6):
            raise ValueError("Recorded timestamps or odometry frames are inconsistent")
        exported.append({"stamp": packet["stamp"], **packet["middle"],
                         "camera_to_reference": (first_inverse @ reference_to_odometry(model, packet)).tolist(),
                         "odometry": packet["odometry"], "joint_encoders": packet["joint_encoders"]})
    hand = gripper_pose(model, target)
    hand_to_reference = first_inverse @ reference_to_odometry(model, target) @ np.asarray(hand["gripper_to_camera"]["head_camera"])
    return {"recording_id": recording["recording_id"], "status": "succeeded", "fps": recording["fps"],
            "camera_id": "head_camera", "input_modalities": ["rgb"], "frames": exported,
            "reference_frame": "first_recorded_head_camera_opencv", "target_stamp": target["stamp"],
            "pose_source": "robot_odometry_and_joint_encoders", "gripper_to_reference": hand_to_reference.tolist()}
