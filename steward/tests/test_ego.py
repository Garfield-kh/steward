from __future__ import annotations

import numpy as np

from steward.assets.generators import steward_bot
from steward.ego.esim import Esim
from steward.ego.estimator import EgoEstimator, EstimatorConfig
from steward.scene.validate import validate_ego
from steward.world.sim import WorldSim


def _observe(sim: WorldSim, est: EgoEstimator, frames: int) -> None:
    for _ in range(frames):
        est.update(sim.sensors("w:robot-1"))
        sim.step_seconds(0.1)


def _cup_track(est: EgoEstimator):
    """The track most often labelled cup; the detector occasionally confuses cups and bowls."""
    return max((t for t in est.tracks.values() if "cup" in t.category_counts),
               key=lambda t: t.category_counts["cup"])


def test_ego_state_converges_toward_ground_truth(kitchen_scene):
    sim = WorldSim(kitchen_scene, seed=3)
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    sim.command("w:robot-1", {"base": [-0.6, -0.2, 0.9]})
    sim.step_seconds(3.0)
    gt = np.asarray(sim.entity_obb("w:cup-1").center)
    _observe(sim, est, 3)
    early = float(np.linalg.norm(_cup_track(est).pos - gt))
    early_sigma = _cup_track(est).sigma
    _observe(sim, est, 57)
    late = float(np.linalg.norm(_cup_track(est).pos - gt))
    assert late < early
    assert _cup_track(est).sigma < 0.75 * early_sigma
    assert late < 0.02


def test_ego_state_is_partial_and_leak_free(kitchen_scene):
    sim = WorldSim(kitchen_scene, seed=4)
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    _observe(sim, est, 5)
    es = est.to_scene().to_dict()
    assert validate_ego(es) == []
    categories = {e["category"]["value"] for e in es["entities"]}
    assert "cup" not in categories, "the cup on the counter is outside the initial field of view"
    by_camera = {}
    for det in sim.sensors("w:robot-1")["detections"]:
        by_camera.setdefault(det["camera"], set()).add(det["category"])
    assert "counter" in by_camera["wide_camera"] and "counter" not in by_camera["head_camera"], \
        "the wide camera sees the counter beside the robot, which the head camera misses"


def test_esim_refuses_uncertain_pick_and_accepts_after_observation(kitchen_scene):
    sim = WorldSim(kitchen_scene, seed=5)
    model = steward_bot().to_dict()
    sim.perception.pos_sigma_rows = 1920  # a detector as coarse at the head camera's 1920 rows as at 240
    est = EgoEstimator(robot_model=model, config=EstimatorConfig(pos_sigma_rows=1920))
    esim = Esim(model)
    sim.command("w:robot-1", {"base": [-1.8, -1.2, 1.2]})
    sim.step_seconds(3.0)
    _observe(sim, est, 4)
    cup_id = _cup_track(est).id
    far = esim.predict(est.to_scene(), "pick", {"target": cup_id})
    assert far["feasible"] and far["risk"] > 0.35
    sim.command("w:robot-1", {"base": [-0.5, 1.0, 1.5708]})
    sim.step_seconds(4.0)
    _observe(sim, est, 30)
    near = esim.predict(est.to_scene(), "pick", {"target": cup_id})
    assert near["feasible"] and near["risk"] < 0.05
    assert near["plan"]["support"] != cup_id


def test_translate_keeps_heading_in_the_odometry_yaw_frame():
    from steward.ego.skills import SkillExecutor

    commands: list[dict] = []
    stamp = {"t": 0.0}

    def state() -> dict:
        stamp["t"] += 1.0
        return {"stamp": stamp["t"], "odometry": {"x": 0.0, "y": 0.0, "yaw": 2 * 3.14159265 + 0.1}, "joints": {},
                "gripper": {}, "entities": {}, "toolOffset": 0.26}

    executor = SkillExecutor(commands.append, state, lambda _id: None)
    executor.config.tick_s = 0.0
    executor._translate_to(0.0, 0.0, 0.1)
    assert abs(commands[-1]["base"][2] - (2 * 3.14159265 + 0.1)) < 1e-6


def _with_detection(est: EgoEstimator, frame: dict, category: str, world_p: list[float], size: list[float],
                    camera: str = "head_camera") -> dict:
    """``frame`` with one more detection at ``world_p``, as the robot's own ``camera`` would place it."""
    from steward.math3d import pose_compose, pose_inverse

    cam = est.camera_pose(camera=camera)
    pose = pose_compose(pose_inverse(cam), {"p": world_p, "q": cam["q"]})
    det = {"category": category, "confidence": 0.9, "pose": pose, "size": size, "range": 2.0, "camera": camera}
    return {**frame, "detections": [*frame["detections"], det]}


def _tracks_near(est: EgoEstimator, p: list[float]) -> list:
    return [t for t in est.tracks.values() if np.linalg.norm(t.pos - np.asarray(p)) < 0.05]


def test_a_detection_floating_clear_of_everything_gets_no_track(kitchen_scene):
    """Something seen in mid-air is falling, not resting: the belief waits to perceive it where it
    lands, while a thing seen on a surface is taken in at once."""
    sim = WorldSim(kitchen_scene, seed=6)
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    _observe(sim, est, 3)
    table = next(t for t in est.tracks.values() if t.category == "table")
    resting = [float(table.pos[0]) + 0.3, float(table.pos[1]) - 0.1, table.top() + 0.02]
    floating = [float(table.pos[0]) + 0.3, float(table.pos[1]) + 0.25, table.top() + 0.3]
    beyond = [float(table.pos[0]), float(table.pos[1]) + 1.2, table.top() + 0.02]
    frame = sim.sensors("w:robot-1")
    for p in (resting, floating, beyond):
        frame = _with_detection(est, frame, "book", p, [0.15, 0.1, 0.04])
    est.update(frame)
    assert len(_tracks_near(est, resting)) == 1
    assert _tracks_near(est, floating) == [] and _tracks_near(est, beyond) == []


def test_one_piece_seen_by_both_cameras_is_one_track_and_overlapping_pieces_are_two(kitchen_scene):
    """An object yields one detection per camera, so both cameras' views of a piece join its one
    track; two pieces one camera sees apart in a frame stay two, even lying half over each other."""
    sim = WorldSim(kitchen_scene, seed=6)
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    _observe(sim, est, 3)
    table = next(t for t in est.tracks.values() if t.category == "table")
    single = [float(table.pos[0]) + 0.3, float(table.pos[1]) - 0.15, table.top() + 0.003]
    a = [float(table.pos[0]) + 0.3, float(table.pos[1]) + 0.1, table.top() + 0.003]
    b = [a[0] + 0.02, a[1], a[2] + 0.006]
    size = [0.0575, 0.0575, 0.006]
    first = None
    for _ in range(5):
        frame = sim.sensors("w:robot-1")
        for camera in ("head_camera", "wide_camera"):
            frame = _with_detection(est, frame, "puzzle_piece", single, size, camera)
        for p in (a, b):
            frame = _with_detection(est, frame, "puzzle_piece", p, size)
        est.update(frame)
        pieces = sorted(t.id for t in est.tracks.values() if t.category == "puzzle_piece")
        assert len(pieces) == 3
        first = first or pieces
        assert pieces == first, "the same tracks every frame, none born and merged again"
    assert not [d for d in est.dropped if d["category"] == "puzzle_piece"]


def test_views_of_the_load_in_the_gripper_correct_where_it_sits_unless_a_neighbour_is_as_near(kitchen_scene):
    """The grasp takes a piece where it really lies, not where the estimate put it; views of it in
    the gripper move the held offset there, so a place puts the real piece down at the spot."""
    sim = WorldSim(kitchen_scene, seed=6)
    sim.perception.odom_pos_sigma = sim.perception.odom_yaw_sigma = 0.0  # detections are placed with the last frame's pose
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    _observe(sim, est, 3)
    table = next(t for t in est.tracks.values() if t.category == "table")
    believed = [float(table.pos[0]) + 0.3, float(table.pos[1]) - 0.15, table.top() + 0.003]
    real = np.asarray(believed) + np.array([0.02, -0.015, 0.0])
    size = [0.0575, 0.0575, 0.006]

    def frame_with(*points):
        sim.step_seconds(0.1)
        frame = sim.sensors("w:robot-1")
        for p in points:
            frame = _with_detection(est, frame, "puzzle_piece", list(p), size)
        return frame

    for _ in range(5):
        est.update(frame_with(believed))
    piece = _tracks_near(est, believed)[0]
    est.mark_held(piece.id, since=est.stamp)
    for _ in range(30):
        est.update({**frame_with(real), "gripper": {"state": "closed", "holding": True}})
    assert np.linalg.norm(est.held_position() - real) < 0.004
    neighbour = real + np.array([0.0575, 0.0, 0.0])
    for _ in range(3):
        est.update({**frame_with(real, neighbour), "gripper": {"state": "closed", "holding": True}})
    assert len(_tracks_near(est, neighbour)) == 1
    settled = est.held_position().copy()
    est.update({**frame_with(neighbour, real + np.array([0.03, 0.0, 0.0])), "gripper": {"state": "closed", "holding": True}})
    assert np.allclose(est.held_position(), settled), "a view as near the neighbour may be the neighbour"


def test_an_implausible_object_is_flagged_and_forgotten_only_once_unseen(kitchen_scene):
    sim = WorldSim(kitchen_scene, seed=6)
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    _observe(sim, est, 3)
    plate = next(t for t in est.tracks.values() if t.category == "plate")
    assert not est.unsupported(plate)
    wall = next(t for t in est.tracks.values() if t.category == "wall")
    wall.pos = wall.pos + np.array([0.0, 0.0, 0.1])
    assert not est.unsupported(wall), "a wall stands on the floor, whatever its estimated bottom"
    plate.pos = plate.pos + np.array([0.0, 0.0, 0.4])
    assert est.unsupported(plate)
    plate.last_seen = est.stamp
    try:
        est.forget(plate.id, "floating")
        raise AssertionError("an object perceived a moment ago must not be forgotten")
    except ValueError as exc:
        assert "still seen" in str(exc)
    plate.last_seen = est.stamp - est.config.forget_unseen_s - 1.0
    est.forget(plate.id, "floats above the table; not seen on two looks")
    assert plate.id not in est.tracks
    assert est.dropped[-1]["reason"] == "forgotten by the agent: floats above the table; not seen on two looks"


def test_esim_place_requires_holding(kitchen_scene):
    sim = WorldSim(kitchen_scene, seed=6)
    model = steward_bot().to_dict()
    est = EgoEstimator(robot_model=model)
    _observe(sim, est, 3)
    table = next(t for t in est.tracks.values() if t.category == "table")
    receipt = Esim(model).predict(est.to_scene(), "place", {"on": table.id})
    assert not receipt["feasible"]
    assert "not holding" in receipt["issues"][0]


def test_motion_state_reads_the_live_body_not_the_perception_clock(kitchen_scene):
    """A real segmenter can be seconds behind. Motion grades itself on the body's own header;
    the belief only supplies where the objects are."""
    from steward.ego.runtime import EgoRuntime, RobotLink

    runtime = EgoRuntime(RobotLink("http://127.0.0.1:1"), steward_bot().to_dict())
    runtime.estimator.stamp = 5.0
    runtime.estimator.odometry = {"x": 0.0, "y": 0.0, "yaw": 0.0}

    class Live:
        @staticmethod
        def sensors() -> dict:
            return {"stamp": 9.0, "odometry": {"x": 1.0, "y": 2.0, "yaw": 0.5},
                    "joints": {"lift": 0.4, "reach": 0.2},
                    "gripper": {"state": "closed", "holding": True}}

    class Dead:
        @staticmethod
        def sensors() -> dict:
            raise OSError("robot port unavailable")

    runtime.link = Live()
    state = runtime._motion_state()
    assert state["stamp"] == 9.0 and state["odometry"] == {"x": 1.0, "y": 2.0, "yaw": 0.5}
    assert state["joints"] == {"lift": 0.4, "reach": 0.2} and state["gripper"]["holding"]
    runtime.link = Dead()
    assert runtime._motion_state()["stamp"] == 5.0, "an unreachable body leaves the belief as the answer"
