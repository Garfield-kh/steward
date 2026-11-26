"""P1 birth gates: a held object's echo, a dropped spot's lineage, and births of kinds that never
stand where they were born. All on the detections path the missions actually run."""

from __future__ import annotations

import math

import numpy as np
import pytest

from steward.assets.generators import steward_bot
from steward.ego.estimator import EgoEstimator, Track
from steward.math3d import pose_compose, pose_inverse
from steward.world.sim import WorldSim


def _est() -> EgoEstimator:
    return EgoEstimator(robot_model=steward_bot().to_dict())


def _track(tid: str, category: str, pos, size, obs: int = 1) -> Track:
    return Track(id=tid, category_counts={category: float(obs)}, pos=np.array(pos, float),
                 var=np.full(3, 0.01 ** 2), yaw=0.0, yaw_var=1e-3, size=np.array(size, float),
                 observations=obs, first_seen=0.0, last_seen=0.0, existence=1.0)


def _det_at(est: EgoEstimator, category: str, world_p, size, camera: str = "head_camera") -> dict:
    cam = est.camera_pose(camera=camera)
    pose = pose_compose(pose_inverse(cam), {"p": [float(v) for v in world_p], "q": cam["q"]})
    return {"category": category, "confidence": 0.9, "pose": pose, "size": [float(v) for v in size],
            "range": 2.0, "camera": camera}


def _frame(est: EgoEstimator, *dets: dict, holding: bool = False) -> dict:
    return {"stamp": est.stamp + 0.1, "odometry": dict(est.odometry), "joints": dict(est.joints),
            "gripper": {**est.gripper, "state": "closed" if holding else "open", "holding": holding},
            "cameras": [{"id": c} for c in est.cameras],
            "detections": list(dets)}


def test_a_held_object_takes_back_its_echo_from_a_phantom_beside_it():
    """The load hangs where it was just taken from and the world still sees it there: its
    detections fuse into the held track (one per camera, ahead of the nearest-first race) and
    never feed the free phantom standing beside it — the held ghost of the A/B run (e:31, 931
    observations of a body the gripper was holding)."""
    est = _est()
    held_pos = np.array([1.07, -0.60, 0.78])
    est.tracks["e:3"] = _track("e:3", "car_body", held_pos, [0.16, 0.11, 0.065], obs=10)
    est.mark_held("e:3", since=est.stamp)
    # a phantom born beside the load, a hair nearer the echo detections than the load itself
    phantom_pos = held_pos + np.array([0.008, 0.004, 0.0])
    est.tracks["e:31"] = _track("e:31", "car_body", phantom_pos, [0.16, 0.11, 0.065], obs=900)
    echo = held_pos + np.array([0.01, 0.005, 0.0])
    for _ in range(20):
        est.update(_frame(est,
                          _det_at(est, "car_body", echo, [0.16, 0.11, 0.065], "head_camera"),
                          _det_at(est, "car_body", echo + np.array([0.002, 0.001, 0.0]), [0.16, 0.11, 0.065], "wide_camera"),
                          holding=True))
    assert est.tracks["e:3"].held
    assert "e:31" in est.tracks, "a held pair does not merge"
    assert est.tracks["e:3"].observations > 10, "the echo fused into the load"
    assert est.tracks["e:31"].observations == 900, "the phantom was starved of the echo"
    assert est.tracks["e:31"].existence < 1.0, "and it fades where nothing confirms it"


def test_a_birth_where_a_same_kind_track_was_just_merged_reconciles_with_the_survivor():
    """A detection at the spot of a just-merged same-kind track belongs to that track's survivor,
    not to a fresh id — the per-cycle churn of the A/B run (e:38/e:40 born at the body's old spot
    with two observations each, merged away the next frame)."""
    est = _est()
    survivor = _track("e:2", "cup", [0.9, -0.6, 0.9], [0.08, 0.08, 0.10], obs=6)
    dropped = _track("e:1", "cup", [1.3, -0.6, 0.8], [0.08, 0.08, 0.10], obs=4)
    est.tracks = {"e:1": dropped, "e:2": survivor}
    est._drop(dropped, "merged: duplicate", into=survivor)
    assert est.merged_into == {"e:1": "e:2"}
    det_pos = [1.3, -0.6, 0.8]
    for _ in range(3):
        est.update(_frame(est, _det_at(est, "cup", det_pos, [0.08, 0.08, 0.10])))
    assert sorted(est.tracks) == ["e:2"], f"no fresh id at the dropped spot: {sorted(est.tracks)}"
    assert survivor.observations > 6, "the returning detection fused into the survivor"


def test_a_floor_birth_of_a_kind_that_never_stands_on_floors_starts_doubted():
    """A car wheel lying on the floor borrows a prior whose home is a hub: born doubted, it takes
    further views to become pickable, and silence fades it — the floor-ripple door to wheel
    ghosts. The same kind born on a part (or a table) starts as any track does."""
    est = _est()
    est.update(_frame(est, _det_at(est, "car_wheel", [1.2, -0.6, 0.028], [0.05, 0.05, 0.017])))
    wheel = next(t for t in est.tracks.values() if "car_wheel" in t.category_counts)
    assert wheel.existence == pytest.approx(est.config.floor_suspect_existence)
    assert wheel.existence < 0.4, "below the pick gate until it is seen again"

    est.tracks["t:1"] = _track("t:1", "table", [1.0, -0.6, 0.375], [1.2, 0.8, 0.75], obs=10)
    est.update(_frame(est, _det_at(est, "car_wheel", [1.0, -0.6, 0.767], [0.05, 0.05, 0.017])))
    seated = [t for t in est.tracks.values() if "car_wheel" in t.category_counts
              and float(np.linalg.norm(t.pos[:2] - [1.0, -0.6])) < 0.05]
    assert len(seated) == 1 and seated[0].existence == pytest.approx(0.5)


def test_a_long_absent_track_is_recognized_when_met_again_instead_of_reborn():
    """The predict step: 400 s of silence loosens a track's hold on where it was last seen (the
    frozen-covariance hole), so the object met 0.40 m away re-associates instead of opening a
    fresh id beside itself — the e:38/e:40 churn of the A/B run, at association level."""
    est = _est()
    est.tracks["e:2"] = _track("e:2", "cup", [1.3, -0.6, 0.8], [0.08, 0.08, 0.10], obs=6)
    est.tracks["t:1"] = _track("t:1", "table", [1.7, -0.6, 0.375], [1.2, 0.8, 0.75], obs=10)
    est.stamp = 400.0  # the frame below then advances the clock; the growth caps at 0.15 m sigma
    det_pos = [1.7, -0.6, 0.8]  # 0.40 m from the last seen spot: beyond the 0.30 m floor gate
    est.update(_frame(est, _det_at(est, "cup", det_pos, [0.08, 0.08, 0.10])))
    assert sorted(est.tracks) == ["e:2", "t:1"], f"re-associated, not reborn: {sorted(est.tracks)}"
    assert est.tracks["e:2"].observations > 6
    assert float(np.linalg.norm(est.tracks["e:2"].pos - np.asarray(det_pos))) < 0.05


def test_the_grasp_window_echo_does_not_birth_while_the_grip_is_closed_but_unmarked():
    """Between the gripper closing and mark_held, held_id is unset and the load's echo stands at
    the pinch point: it births no track there, while real detections elsewhere still do."""
    est = _est()
    est.odometry = {"x": 0.5, "y": -0.3, "yaw": 0.0, "sigma": {"pos": 0.01, "yaw": 0.01}}
    est.joints = {"lift": 0.78, "reach": 0.5, "wrist_yaw": 0.0, "wrist_roll": 0.0}
    pinch = np.asarray(est.gripper_position(), float)
    est.tracks["t:1"] = _track("t:1", "table", [float(pinch[0]), float(pinch[1]), 0.375],
                               [1.2, 0.8, 0.75], obs=10)
    echo = pinch + np.array([0.02, 0.01, 0.0])
    other = pinch + np.array([0.45, 0.1, 0.0])
    est.update(_frame(est,
                      _det_at(est, "car_body", echo, [0.16, 0.11, 0.065]),
                      _det_at(est, "car_body", other, [0.16, 0.11, 0.065]),
                      holding=True))
    near = [t for t in est.tracks.values() if "car_body" in t.category_counts
            and float(np.linalg.norm(t.pos - pinch)) <= 0.1]
    assert near == [], f"the unmarked load's echo births no track: {[(t.id, t.pos.tolist()) for t in near]}"
    far = [t for t in est.tracks.values() if "car_body" in t.category_counts
           and float(np.linalg.norm(t.pos - other)) <= 0.1]
    assert len(far) == 1, "a real detection away from the pinch still births"


def test_holding_a_cup_does_not_spawn_a_copy_at_the_dock(kitchen_scene):
    """The detections-path integration with a real grasp: while the load hangs at the dock, the
    world's own detections of it never open a second track beside the held one."""
    sim = WorldSim(kitchen_scene, seed=6)
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    gt = np.asarray(sim.entity_obb("w:cup-1").center)
    # Dock 0.2 m short of the cup, facing it, with the arm's lateral line accounted for.
    theta = math.pi / 2
    heading = np.array([math.cos(theta), math.sin(theta)])
    probe = est.gripper_position(joints={"lift": float(gt[2]), "reach": 0.3, "wrist_yaw": 0.0, "wrist_roll": 0.0},
                                 odometry={"x": 0.0, "y": 0.0, "yaw": theta})
    dock = gt[:2] - np.asarray(probe[:2]) - heading * 0.2
    sim.command("w:robot-1", {"base": [float(dock[0]), float(dock[1]), theta]})
    sim.step_seconds(2.0)
    for _ in range(10):
        est.update(sim.sensors("w:robot-1"))
        sim.step_seconds(0.1)
    cup = min(est.tracks.values(), key=lambda t: float(np.linalg.norm(t.pos - gt)))
    assert float(np.linalg.norm(cup.pos - gt)) < 0.10, "the cup is perceived from the dock"
    best, bf = None, 1e9
    for reach in np.arange(0.30, 0.62, 0.01):
        for lift in np.arange(float(gt[2]) - 0.06, float(gt[2]) + 0.06, 0.01):
            joints = {"lift": float(lift), "reach": float(reach), "wrist_yaw": 0.0, "wrist_roll": 0.0}
            tcp = est.gripper_position(joints=joints, odometry={"x": float(dock[0]), "y": float(dock[1]), "yaw": theta})
            d = float(np.linalg.norm(tcp - gt))
            if d < bf:
                best, bf = joints, d
    assert bf < 0.03, f"no arm posture reaches the cup ({bf:.3f} m off)"
    sim.command("w:robot-1", {"lift": best["lift"], "reach": best["reach"], "wrist_yaw": 0.0, "wrist_roll": 0.0})
    sim.step_seconds(1.5)
    est.update(sim.sensors("w:robot-1"))  # the belief's own body pose is current for the grasp offset
    sim.command("w:robot-1", {"gripper": "close"})
    sim.step_seconds(0.5)
    est.update(sim.sensors("w:robot-1"))
    assert sim.sensors("w:robot-1")["gripper"]["holding"], "the grasp attached"
    est.mark_held(cup.id, since=est.stamp)

    for _ in range(40):
        est.update(sim.sensors("w:robot-1"))
        sim.step_seconds(0.1)
    cups = [t for t in est.tracks.values() if "cup" in t.category_counts
            and float(np.linalg.norm(t.pos - gt)) < 0.30]
    assert len(cups) == 1 and cups[0].held, \
        f"one cup, the held one: {[(t.id, t.held, np.round(t.pos, 2).tolist()) for t in cups]}"
