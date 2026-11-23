"""Capture the robot's existing cameras without changing their mounts or perception inputs."""
from __future__ import annotations

from steward.assembly_profile import enabled as _assembly_enabled


import base64
import copy
import math
import threading
from concurrent.futures import ThreadPoolExecutor

import mujoco
import numpy as np

from steward.glrender import close_renderer, encode_image
from steward.math3d import matrix_from_quat
from steward.net import HttpError


def _intrinsics(sensor):
    width, height = sensor["width"], sensor["height"]
    focal = height / (2 * math.tan(math.radians(sensor["fovy"]) / 2))
    return {"width": width, "height": height,
            "K": [[focal, 0, (width - 1) / 2], [0, focal, (height - 1) / 2], [0, 0, 1]]}


def _sensor_kinematics(sim, robot):
    """Read encoders and fixed camera-to-``head_camera`` mounts under the simulation lock.

    Only cameras welded to the same rigid body as ``head_camera`` enter the packet: wrist
    cameras ride moving links and need FK, so they stay out of this fixed-rig calibration.
    """
    names = sim.build.names[robot]
    if _assembly_enabled():
        from steward.ego.kinematics import camera_rig
        encoders = {key: sim._joint_q(name) for key, name in names.joints.items()}
        return {"stamp": sim.time, "reference_camera": "head_camera", "calibration_source": "rig_calibration",
                "camera_axes": "opencv_x_right_y_down_z_forward", "joint_encoders": encoders,
                "views": camera_rig(sim.template.entity(robot).to_dict(), encoders)}
    else:
        transforms, bodies = {}, {}
        for sensor in sim.robot_cameras(robot):
            key = sensor["id"]
            cid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, names.cameras[key])
            bodies[key] = int(sim.model.cam_bodyid[cid])
            transform = np.eye(4)
            transform[:3, :3] = matrix_from_quat(sim.model.cam_quat[cid]) @ np.diag([1, -1, -1])
            transform[:3, 3] = sim.model.cam_pos[cid]
            transforms[key] = transform
        if "head_camera" not in transforms:
            raise ValueError("Camera calibration requires head_camera")
        ref_body = bodies["head_camera"]
        shared = {key: value for key, value in transforms.items() if bodies[key] == ref_body}
        if not shared:
            raise ValueError("Camera calibration requires a shared rigid mount with head_camera")
        inverse = np.linalg.inv(shared["head_camera"])
        return {"stamp": sim.time, "reference_camera": "head_camera", "calibration_source": "rig_calibration",
                "camera_axes": "opencv_x_right_y_down_z_forward",
                "joint_encoders": {key: sim._joint_q(name) for key, name in names.joints.items()},
                "views": {key: {"camera_to_reference": np.round(inverse @ value, 12).tolist()}
                          for key, value in shared.items()}}


class CameraRenderer:
    """Own a private physics snapshot and GL resources on one rendering thread."""
    def __init__(self):
        self.source = self.model = self.data = None
        self.renderers = {}

    def render(self, sim, robot):
        if _assembly_enabled():
            with sim.lock:
                if self.source is not sim.model:
                    self.close()
                    self.source = sim.model
                    self.model = copy.deepcopy(sim.model)
                    self.data = mujoco.MjData(self.model)
                mujoco.mj_copyData(self.data, self.model, sim.data)
                calibration = {s["id"]: dict(s["intrinsics"], **{k:s[k] for k in ("renderNearM",) if k in s}) for s in sim.robot_cameras(robot)}
                packet = _sensor_kinematics(sim, robot)
        else:
            with sim.lock:
                if self.source is not sim.model:
                    self.close()
                    self.source = sim.model
                    self.model = copy.deepcopy(sim.model)
                    self.data = mujoco.MjData(self.model)
                mujoco.mj_copyData(self.data, self.model, sim.data)
                packet = _sensor_kinematics(sim, robot)
                # Match the fixed-rig packet: skip wrist (and any other non-head-body) cameras.
                calibration = {s["id"]: dict(s["intrinsics"]) for s in sim.robot_cameras(robot)
                               if s["id"] in packet["views"]}
        return {**packet, "images": self.render_data(robot, tuple(calibration), calibration)["views"]}

    def render_data(self, robot, ids, calibration, epoch=None):
        """Render existing cameras at their own resolution and field of view from installed data."""
        mujoco.mj_forward(self.model, self.data)
        views = {}
        if _assembly_enabled():
            for camera in ids:
                intrinsic = _intrinsics(calibration[camera])
                key = (intrinsic["width"], intrinsic["height"])
                if key not in self.renderers:
                    self.renderers[key] = mujoco.Renderer(self.model, key[1], key[0])
                renderer = self.renderers[key]
                from steward.world.camera_optics import clip_planes
                with clip_planes(self.model, calibration[camera]):
                    renderer.update_scene(self.data, camera=f"{robot}::{camera}")
                    views[camera] = intrinsic | {"rgb": renderer.render().copy()}
        else:
            for camera in ids:
                intrinsic = _intrinsics(calibration[camera])
                key = (intrinsic["width"], intrinsic["height"])
                if key not in self.renderers:
                    self.renderers[key] = mujoco.Renderer(self.model, key[1], key[0])
                renderer = self.renderers[key]
                renderer.update_scene(self.data, camera=f"{robot}::{camera}")
                views[camera] = intrinsic | {"rgb": renderer.render().copy()}
        return {"stamp": float(self.data.time), "views": views}

    def close(self):
        for renderer in self.renderers.values():
            close_renderer(renderer)
        self.renderers.clear()
        self.source = self.model = self.data = None


class SensorCameras:
    """Serve optional evidence tools independently of the original head/wide perception loop."""
    def __init__(self, sim):
        from steward.world.rgb_recording import SensorRecordings
        self.sim = sim
        self.lock = threading.RLock()
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sensor-evidence")
        self.renderer = CameraRenderer()
        self.closed = False
        self.recordings = SensorRecordings(sim)

    def sensor_snapshot(self, robot):
        with self.lock:
            if self.closed:
                raise HttpError(503, "Camera evidence service is closed")
            packet = self.worker.submit(self.renderer.render, self.sim(), robot).result()
        images = packet.pop("images")
        for key, view in images.items():
            packet["views"][key].update({k: view[k] for k in ("width", "height", "K")})
            packet["views"][key]["image"] = "data:image/jpeg;base64," + base64.b64encode(encode_image(view["rgb"], "jpeg")).decode("ascii")
        return packet

    def kinematics(self, robot):
        with self.lock:
            if self.closed:
                raise HttpError(503, "Camera evidence service is closed")
            sim = self.sim()
            with sim.lock:
                return _sensor_kinematics(sim, robot)

    def calibration(self, robot):
        if _assembly_enabled():
            with self.lock:
                packet = self.kinematics(robot)
                sim = self.sim()
                with sim.lock:
                    for sensor in sim.robot_cameras(robot):
                        packet["views"][sensor["id"]].update(_intrinsics(sensor["intrinsics"]))
                        packet["views"][sensor["id"]].update({k:sensor[k] for k in
                            ("renderNearM", "depthRangeM", "housingSizeM", "tipPlaneClearanceM") if k in sensor})
            # Moving camera extrinsics belong to this encoder timestamp.
            if "wrist_camera" not in packet["views"]:
                packet.pop("stamp")
                packet.pop("joint_encoders")
        else:
            with self.lock:
                packet = self.kinematics(robot)
                sim = self.sim()
                with sim.lock:
                    for sensor in sim.robot_cameras(robot):
                        packet["views"][sensor["id"]].update(_intrinsics(sensor["intrinsics"]))
            packet.pop("stamp")
            packet.pop("joint_encoders")
        return packet

    def odometry_kinematics(self, robot):
        with self.lock:
            sim = self.sim()
            with sim.lock:
                return self.kinematics(robot) | {"odometry": sim.odometry(robot), "odometry_frame": sim.odometry_frame_id}

    def close(self):
        self.recordings.close()
        with self.lock:
            if self.closed:
                return
            self.closed = True
            self.worker.submit(self.renderer.close).result()
            self.worker.shutdown(wait=True)
