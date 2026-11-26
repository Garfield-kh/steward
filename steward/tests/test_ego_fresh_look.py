"""Fresh look without a robot-side segmenter: the frame's own anonymous masks re-anchor a target."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from steward.assets.generators import steward_bot
from steward.ego.estimator import EgoEstimator
from steward.ego.perception import decode_rgbd
from steward.ego.runtime import RuntimeConfig, _anonymous_reading, _detection_reading
from steward.world.sim import PerceptionConfig, WorldSim

CUP = "w:cup-1"


def _sim(kitchen_scene, base: list[float]) -> WorldSim:
    sim = WorldSim(kitchen_scene, seed=7, perception=PerceptionConfig(mode="rgbd", masks=True))
    sim.command("w:robot-1", {"base": base})
    sim.step_seconds(3.0)
    return sim


def _perceive(sim: WorldSim, frames: int) -> EgoEstimator:
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    for _ in range(frames):
        est.update_rgbd(decode_rgbd(sim.rgbd("w:robot-1")))
        sim.step_seconds(0.5)
    return est


def _cup_track(sim: WorldSim, est: EgoEstimator):
    gt = sim.entity_obb(CUP)
    track = min(est.tracks.values(), key=lambda t: float(np.linalg.norm(t.pos - gt.center)))
    return track, gt


class Rig:
    """Just the surface ``EgoRuntime._fresh_look`` touches; no runtime process, no server."""


def test_a_fresh_look_without_a_segmenter_reads_the_object_from_the_worlds_own_masks(kitchen_scene):
    """With no STEWARD_SEGMENTER_URL the world's frames carry class-agnostic instance masks; the
    fresh look takes one now and reads the instance standing at the (drifted) belief, so a pick
    re-anchors instead of being silently skipped."""
    sim = _sim(kitchen_scene, [-0.5, 1.0, 1.5708])
    est = _perceive(sim, 2)
    track, gt = _cup_track(sim, est)
    assert float(np.linalg.norm(track.pos - gt.center)) < 0.10, "the cup is perceived"

    # A stale belief a few centimetres off, as a fused track carries: the look restates it.
    # One look reads the mask's own depth pixels (their median sits within the object, a few cm
    # from the box centre, exactly as a segmented look's median does).
    believed = np.asarray(track.pos, float) + np.array([0.05, -0.03, 0.0])
    frame = decode_rgbd(sim.rgbd("w:robot-1"))
    reading = _anonymous_reading(frame, est.camera_pose(), believed, RuntimeConfig())
    assert reading is not None
    assert float(np.linalg.norm(reading - gt.center)) < 0.05, f"the mask's own depth says where it stands: {reading}"
    assert float(np.linalg.norm(reading - gt.center)) < float(np.linalg.norm(believed - gt.center)), \
        "the look pulled the drifted belief back onto the object"

    # A belief well below the counter top is nobody's instance: every instance in the frame
    # disagrees in height, so the look reports "nothing there" instead of guessing.
    far = np.asarray(track.pos, float) + np.array([0.6, 0.0, -0.3])
    assert _anonymous_reading(frame, est.camera_pose(), far, RuntimeConfig()) is None

    # An instance at another height is another thing (a table edge, the floor).
    high = np.asarray(track.pos, float) + np.array([0.0, 0.0, 0.4])
    assert _anonymous_reading(frame, est.camera_pose(), high, RuntimeConfig()) is None


def test_the_runtime_fresh_look_relocates_without_a_segmenter_url(kitchen_scene, monkeypatch, capsys):
    """EgoRuntime._fresh_look, the path a pick and a place posterior take: with the segmenter URL
    unset it now re-places the track from the world's masks instead of skipping."""
    from steward.ego.runtime import EgoRuntime

    sim = _sim(kitchen_scene, [-0.5, 1.0, 1.5708])
    est = _perceive(sim, 2)
    track, gt = _cup_track(sim, est)
    drifted = np.asarray(track.pos, float) + np.array([0.06, 0.0, 0.0])
    track.pos = drifted
    track.var = np.full(3, 0.05 ** 2)

    monkeypatch.delenv("STEWARD_SEGMENTER_URL", raising=False)

    rig = Rig()
    rig.estimator = est
    rig.config = RuntimeConfig()
    rig._lock = threading.Lock()
    rig._latest_frame = decode_rgbd(sim.rgbd("w:robot-1"))  # what the perception loop last kept
    rig._fresh_look_latest = lambda target_id, track: EgoRuntime._fresh_look_latest(rig, target_id, track)

    assert EgoRuntime._fresh_look(rig, track.id) is True
    assert float(np.linalg.norm(est.tracks[track.id].pos - gt.center)) < 0.05, "re-anchored at the object"
    assert "fresh look placed" in capsys.readouterr().err

    # A target no instance stands beside is reported, not silently kept.
    est.tracks[track.id].pos = drifted + np.array([0.0, 0.0, -0.3])
    assert EgoRuntime._fresh_look(rig, track.id) is False
    assert "no instance" in capsys.readouterr().err


def test_a_fresh_look_without_a_segmenter_reads_the_frame_s_own_detections(kitchen_scene):
    """Phase0 ideal frames carry the detector's own noisy boxes instead of masks; the fresh look
    reads the detection at the belief under the same gates, so a pick re-anchors there too."""
    sim = WorldSim(kitchen_scene, seed=7)  # default perception: ideal detections
    sim.command("w:robot-1", {"base": [-0.5, 1.0, 1.5708]})
    sim.step_seconds(3.0)
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    for _ in range(2):
        est.update(sim.sensors("w:robot-1"))
        sim.step_seconds(0.5)
    track, gt = _cup_track(sim, est)
    assert float(np.linalg.norm(track.pos - gt.center)) < 0.10, "the cup is perceived"

    sensors = sim.sensors("w:robot-1")
    frame = {**sensors, "cameras": [c for c in sensors.get("cameras", []) if c["id"] in est.cameras],
             "detections": [d for d in sensors.get("detections", []) if d.get("camera") in est.cameras]}
    cams = {c: est.camera_pose(camera=c) for c in est.cameras}
    believed = np.asarray(track.pos, float) + np.array([0.05, -0.03, 0.0])
    reading = _detection_reading(frame, cams, believed, RuntimeConfig())
    assert reading is not None
    assert float(np.linalg.norm(reading - gt.center)) < 0.04, f"the detector's own box says where it stands: {reading}"
    assert float(np.linalg.norm(reading - gt.center)) < float(np.linalg.norm(believed - gt.center)), \
        "the look pulled the drifted belief back onto the object"

    # A belief well below the counter top matches no detection's height: reported, not guessed.
    assert _detection_reading(frame, cams, believed + np.array([0.6, 0.0, -0.3]), RuntimeConfig()) is None
    assert _detection_reading(frame, cams, believed + np.array([0.0, 0.0, 0.4]), RuntimeConfig()) is None


def test_the_runtime_fresh_look_relocates_from_detections_without_a_segmenter_url(kitchen_scene, monkeypatch, capsys):
    """The detections branch of perception caches its frame the same way, so the fresh look (and
    with it the place posterior) is no longer a no-op in phase0 ideal episodes."""
    from steward.ego.runtime import EgoRuntime

    sim = WorldSim(kitchen_scene, seed=7)  # default perception: ideal detections
    sim.command("w:robot-1", {"base": [-0.5, 1.0, 1.5708]})
    sim.step_seconds(3.0)
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    for _ in range(2):
        est.update(sim.sensors("w:robot-1"))
        sim.step_seconds(0.5)
    track, gt = _cup_track(sim, est)
    drifted = np.asarray(track.pos, float) + np.array([0.06, 0.0, 0.0])
    track.pos = drifted
    track.var = np.full(3, 0.05 ** 2)

    monkeypatch.delenv("STEWARD_SEGMENTER_URL", raising=False)
    sensors = sim.sensors("w:robot-1")
    rig = Rig()
    rig.estimator = est
    rig.config = RuntimeConfig()
    rig._lock = threading.Lock()
    rig._latest_frame = {**sensors, "cameras": [c for c in sensors.get("cameras", []) if c["id"] in est.cameras],
                         "detections": [d for d in sensors.get("detections", []) if d.get("camera") in est.cameras]}
    rig._fresh_look_latest = lambda target_id, track: EgoRuntime._fresh_look_latest(rig, target_id, track)

    assert EgoRuntime._fresh_look(rig, track.id) is True
    assert float(np.linalg.norm(est.tracks[track.id].pos - gt.center)) < 0.05, "re-anchored at the object"
    assert "fresh look placed" in capsys.readouterr().err
