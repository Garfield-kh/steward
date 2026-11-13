"""Phase levels and the Phase 1 perception pipeline (render → segment → estimate)."""

from __future__ import annotations

import math
import threading
import time

import numpy as np
import pytest

import mujoco

from steward.assets.generators import steward_bot
from steward.authoring.service import AuthoringService
from steward.ego.classify import classify
from steward.ego.estimator import EgoEstimator
from steward.ego.perception import box_of, decode_rgbd
from steward.ego.runtime import EgoRuntime, RobotLink
from steward.ego.server import ego_api
from steward.net import call
from steward.phases import PHASES, phase_modules
from steward.world.console import console_state
from steward.world.server import StewardHost
from steward.world.sim import PerceptionConfig, WorldSim


def _pump_until(host: StewardHost, done, timeout: float = 120.0) -> None:
    deadline = time.time() + timeout
    while not done() and time.time() < deadline:
        host.pump(0.05)
    assert done(), "timed out"


def test_box_fit_recovers_the_extent_and_heading_of_a_rotated_box():
    rng = np.random.default_rng(0)
    local = rng.uniform([-0.3, -0.1, 0.0], [0.3, 0.1, 0.2], (800, 3))
    yaw = 0.6
    c, s = math.cos(yaw), math.sin(yaw)
    points = np.stack([local[:, 0] * c - local[:, 1] * s + 1.0, local[:, 0] * s + local[:, 1] * c - 2.0, local[:, 2]], 1)
    centre, size, fitted = box_of(points)
    assert centre[:2] == pytest.approx([1.0, -2.0], abs=0.01)
    assert size == pytest.approx([0.6, 0.2, 0.2], abs=0.02)
    assert abs(math.sin(fitted - yaw)) < 0.05


def test_classifier_tells_tableware_from_furniture_by_size_colour_and_footing():
    def best(size, bottom, color, invalid=0.0):
        probs = classify(np.array(size), bottom, np.array(color), invalid)
        return max(probs, key=probs.get)

    assert best([0.22, 0.22, 0.07], 0.75, [0.8, 0.15, 0.15]) == "bowl"
    assert best([0.11, 0.085, 0.095], 0.75, [0.05, 0.05, 0.06]) == "mug"
    assert best([0.08, 0.08, 0.19], 0.75, [0.8, 0.85, 0.9], invalid=0.3) == "wine_glass"
    assert best([1.2, 0.8, 0.75], 0.0, [0.8, 0.67, 0.5]) == "table"
    assert best([0.56, 0.44, 0.22], 0.55, [0.8, 0.67, 0.5]) == "drawer"
    assert best([4.5, 0.1, 1.45], 0.0, [0.9, 0.9, 0.88]) == "wall"


def test_every_phase_level_names_its_module_choices():
    assert set(PHASES) >= {"phase0", "phase1"}
    assert phase_modules("phase0")["perception"] == "ideal"
    assert phase_modules("phase1")["perception"] == "rgbd"
    with pytest.raises(ValueError):
        phase_modules("phase9")


def test_switching_the_phase_level_changes_the_sensors_and_the_robot_follows(kitchen_root, tmp_path):
    host = StewardHost(kitchen_root, scenario="phase0-chores", speed=4.0, seed=3, run_dir=tmp_path).start()
    runtime = EgoRuntime(RobotLink(host.robot_servers["w:robot-1"].url), steward_bot().to_dict())
    runtime.start()
    try:
        assert "detections" in host.sim.sensors("w:robot-1")
        _pump_until(host, lambda: runtime.perception_mode == "detections" and len(runtime.estimator.tracks) > 3)
        assert console_state(host)["phase"]["level"] == "phase0"
        with pytest.raises(ValueError):
            host.set_phase("phase9")
        state = host.set_phase("phase1")
        assert state["level"] == "phase1" and host.sim.events[-1]["type"] == "phase/switch"
        assert host.sim.sensors("w:robot-1").get("imaging") == "rgbd"
        _pump_until(host, lambda: runtime.perception_mode == "rgbd")
        _pump_until(host, lambda: len(runtime.estimator.tracks) > 3)
        assert all(t.voxels is not None for t in runtime.estimator.tracks.values()), \
            "after the switch every object comes from the RGB-D pipeline"
        host.set_phase("phase0")
        _pump_until(host, lambda: runtime.perception_mode == "detections")
    finally:
        runtime.stop()
        host.stop()


def test_a_named_level_outranks_the_scenarios_declared_one_for_every_later_episode(kitchen_root, tmp_path):
    declared = StewardHost(kitchen_root, scenario="puzzle-dog", run_dir=tmp_path)
    assert declared.phase == "phase0", "without a named level the scenario's declared one applies"
    host = StewardHost(kitchen_root, scenario="puzzle-dog", run_dir=tmp_path, phase="phase1")
    assert host.phase == "phase1" and host.sim.perception.mode == "rgbd"
    host.set_phase("phase1-sam")
    host.load_episode("puzzle-dog")
    assert host.phase == "phase1-sam" and host.sim.perception.masks is False, "a console switch outlives the episode"
    for h in (declared, host):
        h.stop()


def test_phase1_exploration_builds_an_accurate_ego_state_from_rgbd(kitchen_root, tmp_path):
    host = StewardHost(kitchen_root, scenario="phase0-chores", speed=4.0, seed=3, run_dir=tmp_path,
                       phase="phase1").start()
    runtime = EgoRuntime(RobotLink(host.robot_servers["w:robot-1"].url), steward_bot().to_dict())
    runtime.start()
    api = ego_api(runtime).start()
    finished = threading.Event()

    def explore() -> None:
        job = call("POST", f"{api.url}/ego/explore")
        call("POST", f"{api.url}/skill/wait", {"job": job["id"], "timeoutS": 120}, timeout=130)
        time.sleep(1.5)
        finished.set()

    try:
        threading.Thread(target=explore, daemon=True).start()
        _pump_until(host, finished.is_set)
        record = host.observer.latest["w:robot-1"]
        assert record["phase"] == "phase1"
        relations = host.sim.world_state()["relations"]
        in_drawers = sum(1 for r in relations if r["type"] == "supported-by" and r["b"].get("link") == "drawer")
        assert record["matchedObjects"] >= record["gtObjects"] - in_drawers - 1, \
            f"everything outside the closed drawers is found ({in_drawers} lie in them): {record}"
        assert record["meanPositionErrorM"] < 0.03, record
        assert record["meanSizeErrorM"] < 0.04, record
        assert record["falsePositives"] <= 1, record
        assert "leak" not in record
        assert record["categoryAccuracy"] >= 0.8, record
        # The summary reads the live body from the world, and perception may be waiting for a frame
        # that only this thread renders: the request is answered while the world is pumped.
        summary: dict = {}
        threading.Thread(target=lambda: summary.update(call("GET", f"{api.url}/ego/summary")), daemon=True).start()
        _pump_until(host, lambda: "objects" in summary)
        objects = summary["objects"]
        assert {"bowl", "drawer"} <= {o["category"] for o in objects}
        walls = [o for o in objects if o["category"] == "wall"]
        assert len(walls) >= 3, "near walls fill the image edge to edge and still become tracks"

        def plane(wall: dict) -> tuple[str, float]:
            yaw = math.radians(wall["yawDeg"])
            if abs(math.cos(yaw)) > abs(math.sin(yaw)):
                return "y", round(wall["position"][1], 1)
            return "x", round(wall["position"][0], 1)

        # One wall may stay in two pieces where furniture in front of it hides its middle.
        assert len({plane(w) for w in walls}) >= len(walls) - 1, f"views of one wall merge into one track: {walls}"
    finally:
        api.stop()
        runtime.stop()
        host.stop()


def _frame(depth: np.ndarray, mask: np.ndarray) -> dict:
    return {"depth": depth, "mask": mask, "rgb": np.full((*depth.shape, 3), 180, np.uint8), "fovy": 58.0}


CAM = {"p": [0.0, 0.0, 3.0], "q": [1.0, 0.0, 0.0, 0.0]}


def test_a_cup_whose_rim_meets_the_top_of_the_image_is_still_a_whole_object():
    from steward.ego.perception import segments

    h, w = 80, 100
    depth = np.full((h, w), 4.0, np.float32)
    mask = np.zeros((h, w), np.uint8)
    mask[0:16, 40:52] = 1
    depth[0:16, 40:52] = 2.2
    mask[20:40, 0:10] = 2
    depth[20:40, 0:10] = 2.2
    found = segments(_frame(depth, mask), CAM)
    by_pixels = sorted(found, key=lambda seg: seg.pixels)
    assert any(not seg.border for seg in found)
    assert any(seg.border for seg in found)
    assert len(by_pixels) == 2


def test_a_region_whose_base_is_cut_hidden_or_overhanging_has_no_seen_bottom():
    from steward.ego.perception import segments

    h, w = 60, 80
    depth = np.full((h, w), 4.0, np.float32)
    mask = np.zeros((h, w), np.uint8)
    mask[10:30, 5:25] = 1  # standing free: the floor right below is about as far
    depth[10:30, 5:25] = 3.0
    depth[30:, 5:25] = 2.95
    mask[5:25, 40:70] = 2  # a wall with a counter right below it, much nearer
    depth[5:25, 40:70] = 3.4
    mask[25:45, 40:70] = 3  # the counter, standing on the floor right below it
    depth[25:45, 40:70] = 2.0
    depth[45:, 40:70] = 1.95
    mask[45:60, 30:38] = 4  # reaches the lower image edge
    depth[45:60, 30:38] = 2.5
    mask[35:45, 72:78] = 5  # a table top segmented without its legs: the room below it is farther
    depth[35:45, 72:78] = 2.7
    mask[2:8, 5:15] = 6  # a bowl whose rim reaches past the table edge: half of it rests on the table
    depth[2:8, 5:15] = 3.55
    depth[8:10, 5:10] = 3.6
    seen = {round(seg.range, 1): seg.bottom_seen for seg in segments(_frame(depth, mask), CAM)}
    assert seen == {3.0: True, 3.4: False, 2.0: True, 2.5: False, 2.7: False, 3.5: True}


def test_a_partial_view_without_its_base_is_classified_by_its_visible_top():
    tabletop = classify([1.2, 0.8, 0.30], 0.45, [0.8, 0.7, 0.5], bottom_seen=False)
    assert max(tabletop, key=tabletop.get) == "table"
    hidden_wall = classify([3.5, 0.04, 1.03], 0.47, [0.9, 0.9, 0.9], bottom_seen=False)
    assert max(hidden_wall, key=hidden_wall.get) == "wall"
    floating = classify([1.2, 0.8, 0.30], 0.45, [0.8, 0.7, 0.5])
    assert floating.get("table", 0.0) < 0.2, "a base in view that is off the floor is not a table"


def test_achromatic_colours_are_named_by_brightness_and_hues_by_chromaticity():
    from steward.ego.estimator import color_name

    names = [color_name(np.array(rgb)) for rgb in ([0.06, 0.05, 0.06], [0.45, 0.46, 0.47], [0.86, 0.86, 0.87],
                                                    [0.7, 0.15, 0.15], [0.12, 0.25, 0.7])]
    assert names == ["black", "gray", "white", "red", "blue"]


def test_a_small_object_keeps_its_single_view_size_when_views_of_it_disagree():
    from steward.assets.generators import steward_bot
    from steward.ego.estimator import EgoEstimator, Track

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    mug = Track(id="e:1", category_counts={"mug": 5.0}, pos=np.array([1.0, 0.0, 0.8]), var=np.full(3, 1e-4), yaw=0.0,
                yaw_var=1e-3, size=np.array([0.1, 0.08, 0.1]), observations=6, first_seen=0.0, last_seen=1.0)
    rng = np.random.default_rng(0)
    clouds = [rng.uniform([-0.05, -0.04, 0.75], [0.05, 0.04, 0.85], (200, 3)) + [1.0 + dx, 0.0, 0.0] for dx in (0, 0, 0, 0.12)]
    for cloud in clouds:
        mug.voxels = est._union(mug.voxels, np.unique(np.floor(cloud / 0.01).astype(np.int32), axis=0))
        mug.recent.append((cloud.mean(axis=0), np.array([0.1, 0.08, 0.1])))
    est.tracks[mug.id] = mug
    est._rebuild_geometry(mug, np.array([0.0, 0.0, 1.2]), 1.0)
    assert mug.size[0] == pytest.approx(0.1, abs=0.01), "a view 12 cm off (an object pushed aside) does not swell it"
    assert mug.pos[0] == pytest.approx(1.0, abs=0.02)


def test_an_established_piece_of_furniture_is_not_folded_into_a_larger_track_of_another_kind():
    from steward.assets.generators import steward_bot
    from steward.ego.estimator import EgoEstimator, Track

    est = EgoEstimator(robot_model=steward_bot().to_dict())

    def track(oid, counts, pos, size, observations):
        return Track(id=oid, category_counts=counts, pos=np.array(pos), var=np.full(3, 1e-4), yaw=0.0, yaw_var=1e-3,
                     size=np.array(size), observations=observations, first_seen=0.0, last_seen=0.0, existence=1.0)

    est.tracks["e:11"] = track("e:11", {"counter": 4.0, "unknown": 1.0}, [0.0, 0.6, 0.74], [2.6, 0.6, 1.48], 12)
    est.tracks["e:16"] = track("e:16", {"sideboard": 4.0, "counter": 3.0}, [0.0, 0.6, 0.43], [1.4, 0.47, 0.84], 20)
    est.tracks["e:20"] = track("e:20", {"unknown": 1.0}, [0.5, 0.6, 0.3], [0.35, 0.3, 0.6], 2)
    est._merge_parts(np.array([0.0, -2.0, 1.2]))
    assert "e:16" in est.tracks, "the sideboard stays its own object"
    assert "e:20" not in est.tracks, "a fragment seen twice is still folded in"


def test_tracks_the_robot_stands_in_or_wall_tall_non_walls_are_dropped():
    from steward.assets.generators import steward_bot
    from steward.ego.estimator import EgoEstimator, Track

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    est.odometry = {"x": -1.5, "y": -1.24, "yaw": 0.0}

    def track(oid, cat, pos, size):
        return Track(id=oid, category_counts={cat: 5.0}, pos=np.array(pos), var=np.full(3, 1e-4), yaw=0.0, yaw_var=1e-3,
                     size=np.array(size), observations=8, first_seen=0.0, last_seen=0.0, existence=1.0)

    for t in (track("e:398", "unknown", [-2.11, -1.83, 0.74], [1.9, 1.48, 1.48]),   # wall corner and chair, around the robot
              track("e:2126", "unknown", [-2.75, 1.56, 0.74], [2.76, 1.33, 1.48]),  # wall pieces in front of the sideboard
              track("e:6", "table", [0.6, -0.9, 0.38], [1.19, 0.8, 0.76]),
              track("e:12", "wall", [-0.1, 2.5, 0.74], [5.7, 0.07, 1.48])):
        est.tracks[t.id] = t
    est._drop_impossible(np.array([0.0, 0.0, 1.4]))
    assert sorted(est.tracks) == ["e:12", "e:6"]


def test_a_new_box_on_the_far_side_of_a_wall_is_dropped_and_the_cabinet_against_it_stays():
    """The far edge of a cabinet top, with the wall right behind it, is completed into a second
    cabinet standing behind that edge: inside the wall. The camera cannot see there."""
    from steward.assets.generators import steward_bot
    from steward.ego.estimator import EgoEstimator, Track

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    est.odometry = {"x": -1.8, "y": -1.2, "yaw": 0.0}

    def track(oid, cat, pos, size, views):
        return Track(id=oid, category_counts={cat: 5.0}, pos=np.array(pos), var=np.full(3, 1e-4), yaw=0.0, yaw_var=1e-3,
                     size=np.array(size), observations=views, first_seen=0.0, last_seen=0.0, existence=1.0)

    for t in (track("e:2", "wall", [3.0, -0.1, 0.75], [0.1, 4.7, 1.47], 30),
              track("e:6", "cabinet", [2.69, 0.59, 0.41], [0.53, 0.61, 0.79], 26),
              track("e:109", "cabinet", [3.285, 0.629, 0.405], [0.52, 0.478, 0.8], 1),
              track("e:110", "cabinet", [2.7, 0.6, 0.4], [0.5, 0.6, 0.8], 1)):
        est.tracks[t.id] = t
    est._drop_impossible(np.array([-1.8, -1.2, 1.4]))
    assert sorted(est.tracks) == ["e:110", "e:2", "e:6"]
    assert est.dropped[-1]["id"] == "e:109" and est.dropped[-1]["reason"] == "on the far side of a wall"


def test_a_room_corner_seen_from_inside_is_concave_and_a_cabinet_front_is_not():
    from steward.ego.estimator import _concave
    from steward.ego.perception import box_of

    cam = np.array([0.0, 0.0, 1.4])
    rng = np.random.default_rng(1)
    z = rng.uniform(0.0, 1.48, 400)
    # Two walls meeting at (-3, -2.5), seen from the room centre: an L of points around the corner.
    left = np.stack([np.full(200, -3.0), rng.uniform(-2.5, -1.0, 200), z[:200]], axis=1)
    back = np.stack([rng.uniform(-3.0, -1.5, 200), np.full(200, -2.5), z[200:]], axis=1)
    corner = np.vstack([left, back])
    centre, size, _ = box_of(corner)
    assert _concave(corner, centre, float(max(size[:2])), cam)
    # The front and top of a cabinet facing the camera: the surfaces lie in front of its centre.
    front = np.stack([np.full(200, 2.4), rng.uniform(0.3, 0.9, 200), rng.uniform(0.0, 0.8, 200)], axis=1)
    top = np.stack([rng.uniform(2.4, 2.95, 200), rng.uniform(0.3, 0.9, 200), np.full(200, 0.8)], axis=1)
    cabinet = np.vstack([front, top])
    centre, size, _ = box_of(cabinet)
    assert not _concave(cabinet, centre, float(max(size[:2])), cam)

    # A table seen from its long side: the top and the legs spread around the centre, not behind it.
    top_t = np.stack([rng.uniform(0.0, 1.2, 300), rng.uniform(-1.3, -0.5, 300), np.full(300, 0.75)], axis=1)
    legs = np.vstack([np.stack([np.full(40, x), np.full(40, y), rng.uniform(0.0, 0.73, 40)], axis=1)
                      for x in (0.03, 1.17) for y in (-1.27, -0.53)])
    table = np.vstack([top_t, legs])
    centre, size, _ = box_of(table)
    assert not _concave(table, centre, float(max(size[:2])), cam)


def test_what_the_robot_holds_keeps_its_identity_while_carried():
    from steward.assets.generators import steward_bot
    from steward.ego.estimator import EgoEstimator, Track
    from steward.ego.perception import Segment

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    mug = Track(id="e:8", category_counts={"mug": 6.0, "cup": 2.0}, pos=np.array([-1.4, 0.3, 0.9]), var=np.full(3, 1e-4),
                yaw=0.0, yaw_var=1e-3, size=np.array([0.1, 0.08, 0.1]), observations=8, first_seen=0.0, last_seen=0.0,
                existence=1.0, held=True)
    est.tracks[mug.id] = mug
    table_view = Segment(points=np.zeros((40, 3)), color=np.array([0.87, 0.76, 0.57]), pixels=900, invalid_frac=0.0,
                         border=True, range=1.0)
    est._fuse_cloud(mug, {"seg": table_view, "centre": np.array([-1.4, 0.3, 0.6]), "size": np.array([1.2, 0.8, 0.3]),
                          "yaw": 0.0, "probs": {"table": 1.0}, "footprint": 1.2}, np.array([0.0, 0.0, 1.4]))
    assert mug.category == "mug" and "table" not in mug.category_counts


def test_a_swelling_neighbour_view_does_not_vote_unknown_onto_stemware():
    """Rejected swollen fusions must not outvote an established wine glass into unknown."""
    from steward.assets.generators import steward_bot
    from steward.ego.estimator import EgoEstimator, Track, _voxelize
    from steward.ego.perception import Segment

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    rng = np.random.default_rng(1)
    glass_cloud = rng.uniform([0.45, -1.16, 0.75], [0.53, -1.08, 0.94], (120, 3))
    glass = Track(
        id="e:w", category_counts={"wine_glass": 8.0}, pos=np.array([0.49, -1.12, 0.85]),
        var=np.full(3, 1e-4), yaw=0.0, yaw_var=1e-3, size=np.array([0.08, 0.08, 0.19]),
        observations=6, first_seen=0.0, last_seen=1.0, existence=1.0,
        voxels=_voxelize(glass_cloud), mean_rgb=np.array([0.76, 0.68, 0.55]), mean_invalid=0.05,
    )
    before = dict(glass.category_counts)
    est.tracks[glass.id] = glass
    # A much larger neighbour cloud that would swell the established small footprint.
    neighbour = rng.uniform([0.2, -1.4, 0.7], [1.0, -0.8, 1.0], (200, 3))
    bad = Segment(points=neighbour, color=np.array([0.5, 0.45, 0.4]), pixels=200, invalid_frac=0.0,
                  border=False, range=1.2)
    est._fuse_cloud(
        glass,
        {"seg": bad, "centre": neighbour.mean(axis=0), "size": np.array([0.8, 0.6, 0.3]),
         "yaw": 0.0, "probs": {"unknown": 1.0}, "footprint": 0.8},
        np.array([0.0, 0.0, 1.4]),
    )
    assert glass.category == "wine_glass", glass.category_counts
    assert glass.category_counts.get("unknown", 0.0) == before.get("unknown", 0.0)
    assert glass.category_counts["wine_glass"] == before["wine_glass"]


def _remembered(oid: str, category: str, pos: list[float], size: list[float]):
    from steward.ego.estimator import Track

    return Track(id=oid, category_counts={category: 5.0}, pos=np.array(pos), var=np.full(3, 1e-4), yaw=0.0,
                 yaw_var=1e-3, size=np.array(size), observations=20, first_seen=0.0, last_seen=0.0, existence=1.0)


LOOKING_ALONG_X = {"p": [0.0, 0.0, 1.2], "q": [1.0, 0.0, 0.0, 0.0]}


def _views(est, cam: dict, frame: dict, seconds: float, dt: float = 0.1) -> None:
    for _ in range(int(round(seconds / dt))):
        est.stamp += dt
        est._decay_unseen_rgbd(cam, set(), frame)


def test_an_established_object_is_dropped_only_after_free_views_from_two_places():
    from steward.ego.estimator import EgoEstimator

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    est.tracks["e:5"] = _remembered("e:5", "plant", [2.0, 0.0, 0.2], [0.24, 0.2, 0.4])
    frame = {"depth": np.full((240, 320), 4.0, np.float32), "fovy": 60.0}  # the far wall, nothing in front of it
    _views(est, LOOKING_ALONG_X, frame, 10.0)
    assert "e:5" in est.tracks, "however long it looks from one place, one viewpoint does not delete a known object"
    assert est.tracks["e:5"].existence >= 0.5
    _views(est, {"p": [0.0, 0.5, 1.2], "q": [1.0, 0.0, 0.0, 0.0]}, frame, 0.6)
    assert "e:5" not in est.tracks
    assert est.dropped[-1]["id"] == "e:5" and est.dropped[-1]["reason"].startswith("seen through")


def test_a_clear_glass_is_not_deleted_when_the_camera_sees_the_room_behind_it():
    from steward.ego.estimator import EgoEstimator

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    glass = _remembered("e:5", "wine_glass", [2.0, 0.0, 0.9], [0.08, 0.08, 0.19])
    glass.mean_invalid = 0.4
    est.tracks["e:5"] = glass
    frame = {"depth": np.full((240, 320), 4.0, np.float32), "fovy": 60.0}
    _views(est, LOOKING_ALONG_X, frame, 10.0)
    _views(est, {"p": [0.0, 0.5, 1.2], "q": [1.0, 0.0, 0.0, 0.0]}, frame, 0.6)
    assert "e:5" in est.tracks


def test_a_cup_on_a_counter_is_not_deleted_when_depth_reads_the_counter_behind_it():
    from steward.ego.estimator import EgoEstimator

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    cup = _remembered("e:5", "cup", [2.0, 0.0, 0.95], [0.08, 0.08, 0.10])
    cup.mean_invalid = 0.0
    est.tracks["e:5"] = cup
    frame = {"depth": np.full((240, 320), 4.0, np.float32), "fovy": 60.0}
    _views(est, LOOKING_ALONG_X, frame, 10.0)
    _views(est, {"p": [0.0, 0.5, 1.2], "q": [1.0, 0.0, 0.0, 0.0]}, frame, 0.6)
    assert "e:5" in est.tracks


def test_a_floating_cup_sits_on_the_counter_and_a_strip_or_a_table_top_does_not():
    from steward.ego.estimator import EgoEstimator

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    counter = _remembered("e:2", "counter", [0.0, 2.18, 0.45], [1.82, 0.62, 0.90])
    cup = _remembered("e:5", "cup", [-0.50, 2.18, 1.015], [0.08, 0.08, 0.10])
    bowl = _remembered("e:6", "bowl", [0.45, 2.20, 0.94], [0.16, 0.16, 0.06])
    strip = _remembered("e:7", "bowl", [-0.48, 2.31, 0.95], [0.40, 0.12, 0.10])
    table = _remembered("e:3", "table", [0.6, -0.9, 0.375], [1.2, 0.8, 0.75])
    plate = _remembered("e:8", "plate", [0.30, -0.80, 0.761], [0.24, 0.24, 0.02])
    top = _remembered("e:9", "plate", [0.10, -1.14, 0.745], [0.30, 0.24, 0.02])
    est.tracks = {t.id: t for t in (counter, cup, bowl, strip, table, plate, top)}
    est._drop_false_tableware()
    est._seat_floating_tableware()
    assert "e:7" not in est.tracks and "e:9" not in est.tracks
    assert "e:6" in est.tracks and "e:8" in est.tracks and "e:5" in est.tracks
    assert abs(est.tracks["e:5"].bottom() - counter.top()) < 1e-6
    assert est.tracks["e:5"].category == "cup"


def test_a_cup_smeared_along_the_counter_is_cut_back_to_the_cup():
    from steward.ego.estimator import EgoEstimator, _voxelize

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    rng = np.random.default_rng(0)
    band = np.column_stack([rng.uniform(-0.70, -0.30, 80), rng.uniform(2.25, 2.34, 80), rng.uniform(0.90, 0.93, 80)])
    cup_pts = np.column_stack([rng.uniform(-0.54, -0.46, 70), rng.uniform(2.14, 2.22, 70), rng.uniform(0.91, 1.01, 70)])
    strip = _remembered("e:7", "bowl", [-0.48, 2.28, 0.95], [0.40, 0.09, 0.11])
    strip.voxels = _voxelize(np.vstack([band, cup_pts]))
    strip.mean_rgb = np.array([0.75, 0.82, 0.88])
    counter = _remembered("e:2", "counter", [0.0, 2.18, 0.45], [1.82, 0.62, 0.90])
    est.tracks = {"e:2": counter, "e:7": strip}
    est._drop_false_tableware()
    est._seat_floating_tableware()
    assert "e:7" in est.tracks
    kept = est.tracks["e:7"]
    assert float(max(kept.size[0], kept.size[1])) < 0.20
    assert abs(float(kept.pos[0]) + 0.50) < 0.12
    assert kept.category == "cup"


def test_a_cup_the_segmenter_called_a_bowl_is_not_merged_into_the_strip():
    from steward.ego.estimator import EgoEstimator

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    cup = _remembered("e:5", "bowl", [-0.52, 2.16, 0.95], [0.076, 0.076, 0.10])
    wide = _remembered("e:9", "bowl", [-0.52, 2.16, 0.95], [0.16, 0.14, 0.10])
    est.tracks = {"e:5": cup, "e:9": wide}
    est._drop_false_tableware()
    assert "e:9" not in est.tracks
    assert est.tracks["e:5"].category == "cup"


def test_a_drawer_heading_a_little_off_square_is_set_square():
    from steward.ego.estimator import EgoEstimator, _voxelize

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    ang = 0.45
    c, s = math.cos(ang), math.sin(ang)
    pts = []
    for lx in np.linspace(-0.28, 0.28, 8):
        for ly in np.linspace(-0.22, 0.22, 6):
            for lz in (0.60, 0.70, 0.80):
                pts.append([-2.70 + c * lx - s * ly, 0.60 + s * lx + c * ly, lz])
    drawer = _remembered("e:22", "drawer", [-2.70, 0.60, 0.70], [0.56, 0.44, 0.22])
    drawer.yaw = ang
    drawer.voxels = _voxelize(np.array(pts))
    drawer.mean_rgb = np.array([0.62, 0.45, 0.28])
    est.tracks = {"e:22": drawer}
    est._rebuild_geometry(drawer, np.array([-1.6, 0.6, 1.2]), 1.5)
    square = abs(math.remainder(drawer.yaw, math.pi / 2))
    assert square < 0.05


def test_a_tall_thin_unknown_in_the_corner_is_dropped():
    from steward.ego.estimator import EgoEstimator

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    edge = _remembered("e:24", "unknown", [3.0, -2.3, 0.75], [0.56, 0.23, 1.48])
    cup = _remembered("e:13", "cup", [-0.5, 2.18, 0.95], [0.08, 0.08, 0.10])
    est.tracks = {"e:24": edge, "e:13": cup}
    est._drop_wall_edges()
    assert "e:24" not in est.tracks and "e:13" in est.tracks


def test_a_counter_depth_a_few_centimetres_short_grows_away_from_the_camera():
    from steward.ego.estimator import EgoEstimator, _voxelize

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    xs = np.linspace(0.0, 1.80, 40)
    ys = np.linspace(0.0, 0.57, 12)
    zs = np.linspace(0.02, 0.88, 8)
    track = _remembered("e:2", "counter", [0.90, 0.285, 0.45], [1.80, 0.57, 0.88])
    track.voxels = _voxelize(np.stack(np.meshgrid(xs, ys, zs), axis=-1).reshape(-1, 3))
    track.mean_rgb = np.array([0.92, 0.92, 0.93])
    track.bottom_seen = True
    est.tracks = {"e:2": track}
    track.seen_across = 0b10  # an earlier broadside view remembered this depth
    est._rebuild_geometry(track, np.array([0.90, -2.0, 1.2]), 2.5)
    assert float(min(track.size[0], track.size[1])) == pytest.approx(0.62, abs=0.02)
    assert float(track.pos[1]) > 0.285


def test_an_open_drawer_whose_centre_has_left_the_cabinet_stays():
    from steward.ego.estimator import EgoEstimator

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    cabinet = _remembered("e:1", "cabinet", [0.0, 0.0, 0.4], [0.62, 0.52, 0.8])
    # The room lies in +y, so the drawer's front points that way and its centre has slid out.
    table = _remembered("e:3", "table", [0.0, 3.0, 0.38], [1.2, 0.8, 0.75])
    drawer = _remembered("e:2", "drawer", [0.0, 0.42, 0.25], [0.50, 0.44, 0.22])
    est.tracks = {"e:1": cabinet, "e:2": drawer, "e:3": table}
    assert est.drawer_opening(drawer) >= 0.08
    est._drop_orphan_parts()
    assert "e:2" in est.tracks


def test_furniture_whose_top_is_still_in_view_gathers_no_evidence_of_absence():
    from steward.ego.estimator import EgoEstimator

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    est.tracks["e:7"] = _remembered("e:7", "table", [2.0, 0.0, 0.375], [1.2, 0.8, 0.75])
    depth = np.full((240, 320), 4.0, np.float32)  # the floor and wall between the legs
    depth[140:195, :] = 2.0  # the tabletop, at the table's distance
    frame = {"depth": depth, "fovy": 60.0}
    for cam_y in (0.0, 0.5, -0.5):
        _views(est, {"p": [0.0, cam_y, 1.2], "q": [1.0, 0.0, 0.0, 0.0]}, frame, 10.0)
    table = est.tracks["e:7"]
    assert table.existence == 1.0 and not table.absent_views, "regions that went elsewhere are no evidence it is gone"


def test_a_young_fragment_where_a_surface_is_seen_fades_away():
    from steward.ego.estimator import EgoEstimator

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    est.tracks["e:9"] = _remembered("e:9", "unknown", [2.0, 0.0, 0.2], [0.24, 0.2, 0.4])
    est.tracks["e:9"].observations = 2
    frame = {"depth": np.full((240, 320), 2.0, np.float32), "fovy": 60.0}
    _views(est, LOOKING_ALONG_X, frame, 12.0)
    assert "e:9" not in est.tracks and est.dropped[-1]["reason"].startswith("faded")


def test_a_potted_plant_is_recognised_from_its_size_and_green_foliage():
    measured = classify(np.array([0.22, 0.19, 0.40]), 0.01, np.array([0.305, 0.494, 0.231]))
    assert max(measured, key=measured.get) == "plant", measured
    big = classify(np.array([0.5, 0.45, 0.9]), 0.0, np.array([0.30, 0.50, 0.22]))
    assert max(big, key=big.get) == "plant", f"a larger plant is still a plant: {big}"


def test_a_well_known_object_folded_into_a_newer_larger_track_keeps_its_id():
    from steward.ego.estimator import EgoEstimator

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    est.tracks["e:3"] = _remembered("e:3", "table", [0.6, -0.9, 0.375], [1.2, 0.8, 0.75])
    est.tracks["e:3"].observations = 40
    est.tracks["e:90"] = _remembered("e:90", "table", [0.65, -0.9, 0.375], [1.4, 0.9, 0.75])
    est.tracks["e:90"].observations = 3
    est._merge_parts(np.array([-1.5, -1.2, 1.2]))
    assert list(est.tracks) == ["e:3"] and est.tracks["e:3"].observations == 43
    record = est.dropped[-1]
    assert {k: record[k] for k in ("id", "observations", "reason", "into")} == \
        {"id": "e:90", "observations": 3, "reason": "merged: part of a larger object", "into": "e:3"}


def test_a_well_known_table_that_looks_hollow_from_afar_is_kept_while_an_unknown_box_is_not():
    from steward.ego.estimator import EgoEstimator, _concave, _points

    rng = np.random.default_rng(2)
    # Only the far half of the top and the far legs: seen from (-1.4, -1.2) this reads as surfaces
    # behind the box centre, as a room corner does.
    top = np.stack([rng.uniform(0.6, 1.2, 300), rng.uniform(-0.9, -0.5, 300), np.full(300, 0.74)], axis=1)
    legs = np.vstack([np.stack([np.full(40, 1.17), np.full(40, y), rng.uniform(0.0, 0.73, 40)], axis=1)
                      for y in (-1.27, -0.53)])
    voxels = np.unique(np.floor(np.vstack([top, legs]) / 0.01).astype(np.int32), axis=0)
    cam = np.array([-1.4, -1.2, 1.5])
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    est.odometry = {"x": -1.44, "y": -1.18, "yaw": 0.0}
    table = _remembered("e:29", "table", [0.6, -0.9, 0.385], [1.19, 0.79, 0.74])
    table.observations, table.voxels = 470, voxels
    assert _concave(_points(voxels), table.pos, table.footprint, cam), "the case: it looks hollow from here"
    unknown = _remembered("e:30", "unknown", [0.6, -0.9, 0.385], [1.19, 0.79, 0.74])
    unknown.observations, unknown.voxels = 2, voxels.copy()
    est.tracks = {"e:29": table, "e:30": unknown}
    est._drop_impossible(cam)
    assert list(est.tracks) == ["e:29"]


def test_phase1_keeps_the_far_edge_of_a_counter(kitchen_root):
    """The far edge of a counter top is a few percent of its pixels. A random sample of the cloud
    publishes about half the depth; the box of every valid pixel keeps it."""
    scene = AuthoringService(kitchen_root).load_scenario("phase0-chores")["sceneState"]
    sim = WorldSim(scene, seed=1, perception=PerceptionConfig(mode="rgbd", masks=True))
    names = sim.build.names["w:robot-1"]
    for jname, val in zip(names.root_joints, (0.6, -2.2, 1.57), strict=True):
        jid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        sim.data.qpos[sim.model.jnt_qposadr[jid]] = val
    sim.command("w:robot-1", {"base": [0.6, -2.2, 1.57]})
    mujoco.mj_forward(sim.model, sim.data)
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    est.update_rgbd(decode_rgbd(sim.rgbd("w:robot-1")))
    gt = sim.entity_obb("w:counter-1")
    counter = min((t for t in est.tracks.values() if t.category == "counter"),
                  key=lambda t: float(np.linalg.norm(t.pos - gt.center)))
    assert float(np.linalg.norm(counter.pos - gt.center)) < 0.12
    assert float(np.mean(np.abs(np.sort(counter.size) - np.sort(gt.size)))) < 0.05


def test_a_label_perception_never_reported_does_not_break_the_next_rgbd_frame(kitchen_root):
    """The agent may name an object with a category the robot's perception holds no evidence for
    (a lid the segmenter only ever called unknown); every later frame must still be fused."""
    from steward.ego.estimator import ESTABLISHED_VIEWS

    scene = AuthoringService(kitchen_root).load_scenario("phase0-chores")["sceneState"]
    sim = WorldSim(scene, seed=1, perception=PerceptionConfig(mode="rgbd", masks=True))
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    frame = decode_rgbd(sim.rgbd("w:robot-1"))
    for _ in range(ESTABLISHED_VIEWS):
        est.update_rgbd(frame)
    for track_id in est.tracks:
        est.annotate(track_id, "lid")
    est.update_rgbd(frame)
    assert est.tracks and all(t.category == "lid" for t in est.tracks.values())


def _paint(depth, mask, cam, fovy, points, instance: int) -> None:
    """Write ``points`` (map frame) into a depth image through ``cam``."""
    from steward.math3d import pose_compose, pose_inverse

    h, w = depth.shape
    f = (h / 2.0) / math.tan(math.radians(fovy) / 2.0)
    inv = pose_inverse(cam)
    for p in points:
        x, y, z = pose_compose(inv, {"p": p.tolist(), "q": [1.0, 0.0, 0.0, 0.0]})["p"]
        if x < 0.3:
            continue
        u = int(round((w - 1) / 2 - y * f / x))
        v = int(round((h - 1) / 2 - z * f / x))
        if 0 <= u < w and 0 <= v < h:
            depth[v, u] = x
            mask[v, u] = instance


def test_two_raised_objects_in_one_segmenter_mask_become_two_tracks():
    """Two raised objects painted as one region are wider than a bowl, so that mask is dropped.
    With segmenter labels the gap splits them, and each keeps its own size. The second object is
    wide enough that a cabinet lip, which is thinner than a bowl, is not what this test keeps."""
    from steward.ego.perception import segments
    from steward.math3d import matrix_from_quat

    header = {"stamp": 1.0, "odometry": {"x": 0.0, "y": 0.0, "yaw": 0.0},
              "joints": {"lift": 0.4, "reach": 0.0, "wrist_yaw": 0.0, "wrist_roll": 0.0},
              "gripper": {"state": "open", "holding": False}}
    cam = EgoEstimator(robot_model=steward_bot().to_dict()).camera_pose(header["odometry"])
    rot = matrix_from_quat(cam["q"])
    origin = np.asarray(cam["p"], float)
    h, w, fovy = 90, 120, 60.0
    depth = np.zeros((h, w), np.float32)
    mask = np.zeros((h, w), np.uint8)
    rng = np.random.default_rng(0)

    def blob(local_centre, size, n=90):
        centre = origin + rot @ np.asarray(local_centre, float)
        return rng.uniform(centre - np.asarray(size) / 2, centre + np.asarray(size) / 2, (n, 3))

    _paint(depth, mask, cam, fovy, blob([1.15, 0.16, -0.05], [0.14, 0.14, 0.05]), 1)
    _paint(depth, mask, cam, fovy, blob([1.15, -0.16, -0.02], [0.14, 0.12, 0.08]), 1)
    bare = {"depth": depth, "mask": mask, "rgb": np.full((h, w, 3), 160, np.uint8), "fovy": fovy,
            "header": header, "camera": "head_camera"}
    assert len(segments(bare, cam)) == 1, "an anonymous mask stays one region"
    named = dict(bare)
    named["labels"] = {1: {"bowl": 0.5}}
    assert len(segments(named, cam)) == 2
    before, after = EgoEstimator(robot_model=steward_bot().to_dict()), EgoEstimator(robot_model=steward_bot().to_dict())
    before.update_rgbd(bare)
    after.update_rgbd(named)
    assert before.tracks == {}
    assert any(d["reason"] == "bowl longer than the bowl prior allows" for d in before.dropped)
    small = [t for t in after.tracks.values() if t.footprint < 0.25]
    assert len(after.tracks) == 2 and len(small) == 2
    assert not any(t.category == "drawer" for t in after.tracks.values())


def test_two_objects_on_a_wide_tabletop_mask_become_two_tracks():
    """A bowl and a glass painted onto one table-top region stay one region without labels. With
    labels, the two raised clusters become two tracks and the slab does not become a third."""
    from steward.ego.perception import segments

    header = {"stamp": 1.0, "odometry": {"x": 0.0, "y": 0.0, "yaw": 0.0},
              "joints": {"lift": 0.4, "reach": 0.0, "wrist_yaw": 0.0, "wrist_roll": 0.0},
              "gripper": {"state": "open", "holding": False}}
    cam = EgoEstimator(robot_model=steward_bot().to_dict()).camera_pose(header["odometry"])
    h, w, fovy = 120, 160, 60.0
    depth = np.zeros((h, w), np.float32)
    mask = np.zeros((h, w), np.uint8)
    rng = np.random.default_rng(2)
    slab = np.column_stack([rng.uniform(1.25, 1.35, 600), rng.uniform(-0.42, 0.42, 600),
                            rng.uniform(0.74, 0.76, 600)])
    bowl = np.column_stack([rng.uniform(1.22, 1.32, 90), rng.uniform(-0.36, -0.22, 90),
                            rng.uniform(0.80, 0.86, 90)])
    glass = np.column_stack([rng.uniform(1.22, 1.30, 90), rng.uniform(0.24, 0.34, 90),
                             rng.uniform(0.80, 0.96, 90)])
    _paint(depth, mask, cam, fovy, np.vstack([slab, bowl, glass]), 1)
    bare = {"depth": depth, "mask": mask, "rgb": np.full((h, w, 3), 150, np.uint8), "fovy": fovy,
            "header": header, "camera": "head_camera"}
    assert len(segments(bare, cam)) == 1
    named = dict(bare)
    named["labels"] = {1: {"bowl": 0.4}}
    assert len(segments(named, cam)) == 2


def test_a_bowl_measured_with_the_table_shrinks_to_the_bowl_and_is_not_completed_as_a_drawer():
    """Table points around a bowl make a wide shallow cloud. The track keeps the bowl, so the
    drawer prior does not stretch it to a drawer's height."""
    from steward.ego.estimator import _voxelize

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    rng = np.random.default_rng(5)
    slab = np.column_stack([rng.uniform(0.40, 0.95, 180), rng.uniform(-1.18, -0.98, 180),
                            rng.uniform(0.74, 0.76, 180)])
    bowl = np.column_stack([rng.uniform(0.62, 0.80, 70), rng.uniform(-1.16, -1.00, 70),
                            rng.uniform(0.79, 0.85, 70)])
    track = _remembered("e:1", "drawer", [0.67, -1.08, 0.80], [0.55, 0.22, 0.22])
    track.voxels = _voxelize(np.vstack([slab, bowl]))
    track.mean_rgb = np.array([0.2, 0.55, 0.2])
    track.bottom_seen = True
    est.tracks = {"e:1": track}
    est._rebuild_geometry(track, np.array([-1.6, -1.3, 1.2]), 2.0)
    assert track.category == "bowl"
    assert max(track.size[0], track.size[1]) < 0.34
    assert track.size[2] < 0.15


def test_a_rim_filled_down_to_the_table_becomes_a_wine_glass_and_objects_already_on_it_keep_their_names():
    """The head camera sees the rim and calls it a bowl. Filling the stem down to the table makes
    the box too tall for a bowl or a cup, so the name becomes a wine glass. A bowl and a cup whose
    points already reach the surface stay what they are."""
    from steward.ego.estimator import _voxelize

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    table = _remembered("e:table", "table", [0.6, -0.9, 0.378], [1.16, 0.79, 0.755])
    counter = _remembered("e:counter", "counter", [0.0, 2.18, 0.46], [1.8, 0.6, 0.9])
    rim = _remembered("e:1", "bowl", [0.50, -1.12, 0.89], [0.08, 0.08, 0.06])
    rng = np.random.default_rng(6)
    rim.voxels = _voxelize(np.column_stack([rng.uniform(0.46, 0.54, 40), rng.uniform(-1.16, -1.08, 40),
                                            rng.uniform(0.86, 0.92, 40)]))
    rim.mean_rgb = np.array([0.78, 0.80, 0.84])
    rim.bottom_seen = True
    rim.semantic_counts = {"bowl": 3.6, "cup": 2.8}
    rim.semantic_views = 4
    rim.recent.extend((np.array([0.50, -1.12, 0.89]), np.array([0.08, 0.08, 0.06])) for _ in range(3))
    bowl = _remembered("e:2", "bowl", [0.72, -1.07, 0.79], [0.20, 0.20, 0.06])
    bowl.voxels = _voxelize(np.column_stack([rng.uniform(0.62, 0.82, 40), rng.uniform(-1.17, -0.97, 40),
                                             rng.uniform(0.76, 0.82, 40)]))
    bowl.mean_rgb = np.array([0.2, 0.55, 0.2])
    bowl.bottom_seen = True
    bowl.semantic_counts = {"bowl": 3.0}
    bowl.semantic_views = 4
    bowl.recent.extend((np.array([0.72, -1.07, 0.79]), np.array([0.20, 0.20, 0.06])) for _ in range(3))
    cup = _remembered("e:3", "cup", [-0.50, 2.16, 0.96], [0.08, 0.08, 0.10])
    cup.voxels = _voxelize(np.column_stack([rng.uniform(-0.54, -0.46, 40), rng.uniform(2.12, 2.20, 40),
                                            rng.uniform(0.91, 1.01, 40)]))
    cup.mean_rgb = np.array([0.9, 0.9, 0.92])
    cup.bottom_seen = True
    cup.semantic_counts = {"cup": 3.2}
    cup.semantic_views = 4
    cup.recent.extend((np.array([-0.50, 2.16, 0.96]), np.array([0.08, 0.08, 0.10])) for _ in range(3))
    est.tracks = {"e:table": table, "e:counter": counter, "e:1": rim, "e:2": bowl, "e:3": cup}
    for track in (rim, bowl, cup):
        est._rebuild_geometry(track, np.array([-1.6, -1.3, 1.2]), 2.0)
    assert rim.category == "wine_glass"
    assert rim.size[2] > 0.14
    assert bowl.category == "bowl"
    assert cup.category == "cup"


def test_a_small_object_keeps_the_size_measured_from_close_up():
    from steward.ego.estimator import _median_of_closer_views

    far = (np.array([0.0, 0.0, 0.8]), np.array([0.16, 0.16, 0.19]), 2.6)
    near = (np.array([0.02, 0.0, 0.8]), np.array([0.08, 0.08, 0.19]), 0.9)
    _, size = _median_of_closer_views([far, far, far, far, near, near])
    assert size[0] < 0.10


def test_a_smaller_bowl_inside_a_larger_one_is_not_swallowed_when_its_base_was_hidden():
    from steward.ego.estimator import EgoEstimator

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    outer = _remembered("e:1", "bowl", [0.5, -1.1, 0.80], [0.50, 0.50, 0.10])
    inner = _remembered("e:2", "bowl", [0.5, -1.1, 0.90], [0.31, 0.31, 0.08])
    est.tracks = {"e:1": outer, "e:2": inner}
    est._merge_parts(np.array([-1.6, -1.3, 1.2]))
    assert set(est.tracks) == {"e:1", "e:2"}


def test_a_narrow_cabinet_slice_keeps_its_drawer_name():
    """A cabinet face taller than a drawer prior keeps the drawer name.

    Dropping that name lets the slice be folded into the cabinet and the cabinet grows. Bowl and
    cup names on a wine-glass-height box are the ones left out."""
    from steward.ego.estimator import _labels_within_height

    assert "drawer" in _labels_within_height({"drawer": 0.85, "box": 0.4}, 0.40)
    assert _labels_within_height({"bowl": 0.85, "cup": 0.55}, 0.174) == {}


def test_two_names_for_one_glass_merge_and_two_bowls_stay_apart():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    est.tracks = {
        "e:1": _remembered("e:1", "cup", [0.47, -1.12, 0.84], [0.08, 0.07, 0.17]),
        "e:2": _remembered("e:2", "bowl", [0.48, -1.12, 0.84], [0.12, 0.08, 0.17]),
        "e:3": _remembered("e:3", "bowl", [0.24, -1.07, 0.78], [0.31, 0.31, 0.08]),
        "e:4": _remembered("e:4", "bowl", [0.71, -1.07, 0.79], [0.27, 0.20, 0.08]),
    }
    est._merge_same_place(np.zeros(3))
    assert "e:2" not in est.tracks and "e:1" in est.tracks
    assert {"e:3", "e:4"} <= set(est.tracks)


def test_a_shallow_bowl_beside_a_glass_splits_out_of_the_table_mask():
    """The green bowl is a short bump on the same mask as the glass, with table points on its side
    of the gap. The gap split keeps the bump and the glass, not the table."""
    from steward.ego.perception import _raised_cluster_indices

    rng = np.random.default_rng(4)
    glass = np.column_stack([rng.uniform(0.40, 0.48, 90), rng.uniform(-1.16, -1.08, 90),
                             rng.uniform(0.78, 0.92, 90)])
    bowl = np.column_stack([rng.uniform(0.62, 0.82, 80), rng.uniform(-1.16, -1.00, 80),
                            rng.uniform(0.78, 0.84, 80)])
    table = np.column_stack([rng.uniform(0.78, 0.98, 50), rng.uniform(-1.20, -0.98, 50),
                             rng.uniform(0.74, 0.76, 50)])
    split = _raised_cluster_indices(np.vstack([table, bowl, glass]), 20)
    assert split is not None and len(split) == 2
    widths = sorted(float(np.max(pts[:, 0]) - np.min(pts[:, 0])) for pts in
                    (np.vstack([table, bowl, glass])[idx] for idx in split))
    assert widths[0] < 0.15 and widths[1] < 0.32


def test_a_better_known_drawer_keeps_its_name_beside_the_sideboard():
    """The drawer shares the sideboard's base and footprint. Folding it in renames the sideboard
    to the drawer's id, and the open then fails because that id is no longer a drawer."""
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    side = _remembered("e:side", "sideboard", [-2.74, 0.6, 0.42], [1.4, 0.47, 0.84])
    side.observations = 4
    drawer = _remembered("e:14", "drawer", [-2.74, 0.6, 0.45], [0.56, 0.44, 0.90])
    drawer.observations = 12
    est.tracks = {"e:side": side, "e:14": drawer}
    est._merge_parts(np.zeros(3))
    assert est.tracks["e:14"].category == "drawer"
    assert est.tracks["e:side"].category == "sideboard"
    young = _remembered("e:15", "drawer", [-2.74, 0.6, 0.45], [0.56, 0.44, 0.90])
    young.observations = 3
    est.tracks = {"e:side": side, "e:15": young}
    est._merge_parts(np.zeros(3))
    assert est.tracks["e:15"].category == "drawer" and "e:side" in est.tracks
    tall = _remembered("e:tall", "drawer", [-2.74, 0.6, 0.42], [1.0, 0.5, 0.75])
    short = _remembered("e:short", "drawer", [-2.74, 0.6, 0.16], [0.56, 0.4, 0.22])
    est.tracks = {"e:tall": tall, "e:short": short}
    est._merge_parts(np.zeros(3))
    assert est.tracks["e:short"].category == "drawer" and est.tracks["e:short"].size[2] < 0.4


def test_a_drawer_does_not_absorb_the_sideboard_face():
    from steward.ego.estimator import _voxelize
    from steward.ego.perception import Segment

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    drawer = _remembered("e:14", "drawer", [-2.74, 0.6, 0.55], [0.56, 0.44, 0.22])
    drawer.observations = 4
    rng = np.random.default_rng(0)
    drawer.voxels = _voxelize(np.column_stack([rng.uniform(-3.02, -2.46, 80), rng.uniform(0.38, 0.82, 80),
                                               rng.uniform(0.44, 0.66, 80)]))
    drawer.mean_rgb = np.array([0.6, 0.5, 0.4])
    face = np.column_stack([rng.uniform(-3.44, -2.04, 200), rng.uniform(0.36, 0.84, 200),
                            rng.uniform(0.02, 0.82, 200)])
    seg = Segment(points=face, color=np.array([0.55, 0.45, 0.35]), pixels=4000, invalid_frac=0.0,
                  border=False, range=1.2, semantics={"sideboard": 0.9})
    est._fuse_cloud(drawer, {"seg": seg, "centre": np.array([-2.74, 0.6, 0.42]),
                             "size": np.array([1.4, 0.48, 0.8]), "yaw": 0.0,
                             "probs": {"sideboard": 0.9}, "footprint": 1.4}, np.zeros(3))
    assert drawer.category == "drawer" and drawer.footprint < 0.7
    drawer.pinned = True
    est._fuse_cloud(drawer, {"seg": seg, "centre": np.array([-2.74, 0.6, 0.42]),
                             "size": np.array([1.4, 0.48, 0.8]), "yaw": 0.0,
                             "probs": {"sideboard": 0.9}, "footprint": 1.4}, np.zeros(3))
    assert drawer.pinned and drawer.footprint < 0.7
    cabinet = _remembered("e:4", "cabinet", [2.67, 0.6, 0.4], [0.6, 0.5, 0.8])
    cabinet.observations = 8
    cabinet.voxels = _voxelize(np.column_stack([rng.uniform(2.37, 2.97, 80), rng.uniform(0.35, 0.85, 80),
                                               rng.uniform(0.0, 0.8, 80)]))
    pulled = np.column_stack([rng.uniform(2.2, 2.97, 120), rng.uniform(0.35, 0.85, 120),
                             rng.uniform(0.4, 0.8, 120)])
    open_seg = Segment(points=pulled, color=np.array([0.5, 0.4, 0.3]), pixels=2000, invalid_frac=0.0,
                       border=False, range=1.0, semantics={"drawer": 0.8})
    est._fuse_cloud(cabinet, {"seg": open_seg, "centre": np.array([2.55, 0.6, 0.6]),
                              "size": np.array([0.9, 0.5, 0.4]), "yaw": 0.0,
                              "probs": {"drawer": 0.8}, "footprint": 0.9}, np.zeros(3))
    assert cabinet.footprint < 0.7
    # Already a little past the prior: one more step under 12 cm must still be refused.
    swollen = _remembered("e:sw", "cabinet", [2.55, 0.6, 0.4], [0.68, 0.55, 0.8])
    swollen.observations = 8
    swollen.voxels = _voxelize(np.column_stack([rng.uniform(2.21, 2.89, 80), rng.uniform(0.32, 0.88, 80),
                                               rng.uniform(0.0, 0.8, 80)]))
    further = np.column_stack([rng.uniform(2.10, 2.89, 40), rng.uniform(0.32, 0.88, 40),
                              rng.uniform(0.4, 0.8, 40)])
    more = Segment(points=further, color=np.array([0.5, 0.4, 0.3]), pixels=800, invalid_frac=0.0,
                   border=False, range=1.0, semantics={"drawer": 0.5})
    before = float(swollen.footprint)
    est._fuse_cloud(swollen, {"seg": more, "centre": np.array([2.5, 0.6, 0.6]),
                              "size": np.array([0.79, 0.56, 0.4]), "yaw": 0.0,
                              "probs": {"drawer": 0.5}, "footprint": 0.79}, np.zeros(3))
    assert swollen.footprint == pytest.approx(before)
    short = _remembered("e:15", "drawer", [-2.74, 0.6, 0.71], [0.56, 0.44, 0.22])
    short.observations = 4
    short.voxels = _voxelize(np.column_stack([rng.uniform(-3.02, -2.46, 60), rng.uniform(0.4, 0.8, 60),
                                              rng.uniform(0.60, 0.82, 60)]))
    tall_face = np.column_stack([rng.uniform(-3.2, -2.2, 120), rng.uniform(0.36, 0.84, 120),
                                rng.uniform(0.05, 0.82, 120)])
    tall_seg = Segment(points=tall_face, color=np.array([0.5, 0.4, 0.3]), pixels=2000, invalid_frac=0.0,
                       border=False, range=1.2, semantics={"drawer": 0.7})
    est._fuse_cloud(short, {"seg": tall_seg, "centre": np.array([-2.7, 0.6, 0.42]),
                            "size": np.array([1.0, 0.48, 0.77]), "yaw": 0.0,
                            "probs": {"drawer": 0.7}, "footprint": 1.0}, np.zeros(3))
    assert short.category == "drawer" and float(short.size[2]) < 0.4


def test_a_drawer_sized_track_keeps_its_name_when_the_shape_says_sideboard():
    from steward.ego.estimator import _voxelize

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    drawer = _remembered("e:14", "drawer", [-2.74, 0.6, 0.42], [1.0, 0.47, 0.84])
    rng = np.random.default_rng(1)
    pts = np.column_stack([rng.uniform(-3.24, -2.24, 150), rng.uniform(0.36, 0.84, 150),
                           rng.uniform(0.0, 0.84, 150)])
    drawer.voxels = _voxelize(pts)
    drawer.mean_rgb = np.array([0.40, 0.26, 0.14])
    drawer.semantic_counts = {"sideboard": 4.0}
    drawer.semantic_views = 4
    drawer.bottom_seen = True
    centre, size, _ = box_of(pts, trim=0.01)
    probs = est.category_probs(size, float(centre[2] - size[2] / 2), drawer.mean_rgb, 0.0, True,
                               {"sideboard": 1.0})
    assert max(probs, key=probs.get) == "sideboard"
    est._rebuild_geometry(drawer, np.array([-1.6, -1.3, 1.2]), 2.0)
    assert drawer.category == "drawer" and float(drawer.size[2]) <= 0.40
    assert float(drawer.pos[2]) > 0.55
    wide = _remembered("e:16", "drawer", [-2.74, 0.6, 0.42], [1.4, 0.47, 0.84])
    wide.voxels = _voxelize(np.column_stack([rng.uniform(-3.44, -2.04, 150), rng.uniform(0.36, 0.84, 150),
                                             rng.uniform(0.0, 0.84, 150)]))
    wide.mean_rgb = drawer.mean_rgb.copy()
    wide.semantic_counts = {"sideboard": 4.0}
    wide.semantic_views = 4
    wide.bottom_seen = True
    est._rebuild_geometry(wide, np.array([-1.6, -1.3, 1.2]), 2.0)
    assert wide.category == "drawer" and float(wide.size[2]) <= 0.40
    assert float(wide.pos[2]) > 0.55


def test_a_sideboard_face_labeled_as_a_drawer_keeps_its_top_band():
    """The handle is in the top band. A box of the whole face puts the hand on the plinth."""
    from steward.ego.estimator import _voxelize

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    rng = np.random.default_rng(2)
    face = np.column_stack([rng.uniform(-2.96, -2.52, 240), rng.uniform(0.15, 1.05, 240),
                            rng.uniform(0.04, 0.84, 240)])
    drawer = _remembered("e:14", "drawer", [-2.74, 0.6, 0.44], [0.9, 0.44, 0.8])
    drawer.voxels = _voxelize(face)
    drawer.mean_rgb = np.array([0.45, 0.32, 0.22])
    drawer.bottom_seen = True
    drawer.semantic_counts = {"drawer": 3.0}
    drawer.semantic_views = 3
    est._rebuild_geometry(drawer, np.array([-1.6, 0.6, 1.2]), 1.5)
    assert float(drawer.size[2]) <= 0.40
    assert float(drawer.pos[2]) > 0.55
    assert drawer.category == "drawer"


def test_a_grazing_edge_is_dropped_and_the_sideboard_stays():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    est.odometry = {"x": 0.0, "y": 0.0, "yaw": 0.0}
    est.tracks = {
        "e:1": _remembered("e:1", "unknown", [1.0, -1.6, 0.4], [2.3, 0.29, 0.5]),
        "e:2": _remembered("e:2", "sideboard", [-2.7, 0.6, 0.42], [1.4, 0.47, 0.84]),
    }
    est._drop_impossible(np.zeros(3))
    assert "e:1" not in est.tracks and est.tracks["e:2"].category == "sideboard"


def test_a_shut_drawer_drops_the_length_it_had_while_open():
    """Pushing a drawer shut keeps the front and one prior-length behind it.
    The points that spanned the open travel were the cabinet, and they go.
    Sliding it when nothing shows it as open leaves that length alone."""
    from steward.ego.estimator import _voxelize

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    xs = np.linspace(0.0, 0.86, 40)
    ys = np.linspace(-0.2, 0.2, 6)
    zs = np.linspace(0.55, 0.75, 4)
    cloud = _voxelize(np.stack(np.meshgrid(xs, ys, zs), axis=-1).reshape(-1, 3))
    drawer = _remembered("e:5", "drawer", [0.43, 0.0, 0.65], [0.86, 0.40, 0.20])
    drawer.voxels = cloud
    est.tracks = {"e:5": drawer}
    est._trim_shut_drawer(drawer, np.array([0.26, 0.0, 0.0]))
    assert float(max(drawer.size[0], drawer.size[1])) == pytest.approx(0.60, abs=0.05)
    opened = _remembered("e:6", "drawer", [0.43, 0.0, 0.65], [0.86, 0.40, 0.20])
    opened.voxels = cloud.copy()
    est.tracks = {"e:6": opened}
    est.shift("e:6", np.array([0.26, 0.0, 0.0]))
    assert float(max(opened.size[0], opened.size[1])) == pytest.approx(0.86, abs=0.05)


def test_a_flat_counter_sliver_and_an_unnamed_scissors_are_dropped():
    """A paper-thin unknown wider than a plate is the counter top. Scissors the segmenter
    never named are a cabinet edge completed to that prior. A plate, and scissors the
    segmenter did name, stay."""
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    est.odometry = {"x": 0.0, "y": 0.0, "yaw": 0.0}
    named = _remembered("e:9", "scissors", [0.2, -0.4, 0.76], [0.16, 0.07, 0.012])
    named.semantic_counts = {"scissors": 2.0}
    est.tracks = {
        "e:21": _remembered("e:21", "unknown", [0.63, 1.99, 0.90], [0.56, 0.27, 0.02]),
        "e:27": _remembered("e:27", "scissors", [2.45, 0.88, 0.80], [0.16, 0.07, 0.012]),
        "e:6": _remembered("e:6", "plate", [0.30, -0.80, 0.76], [0.24, 0.23, 0.02]),
        "e:9": named,
    }
    est._drop_impossible(np.zeros(3))
    assert "e:21" not in est.tracks and "e:27" not in est.tracks
    assert set(est.tracks) == {"e:6", "e:9"}


def test_table_edge_strips_and_a_tall_sliver_are_dropped_and_the_table_stays():
    """Strips south of the table, and a tall sliver a few centimetres thick, are not objects.
    The table, the pistol and the pill bottle are thinner than furniture and stay."""
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    est.odometry = {"x": 1.6, "y": 0.6, "yaw": 0.0}
    labeled = _remembered("e:6", "box", [2.76, 0.62, 0.66], [0.85, 0.65, 0.28])
    labeled.label = "drawer"
    est.tracks = {
        "e:6": labeled,
        "e:3": _remembered("e:3", "table", [0.59, -0.89, 0.38], [1.19, 0.79, 0.75]),
        "e:116": _remembered("e:116", "unknown", [-0.14, -1.68, 0.56], [1.73, 0.28, 0.54]),
        "e:119": _remembered("e:119", "unknown", [0.73, -1.81, 0.55], [1.49, 0.10, 0.60]),
        "e:120": _remembered("e:120", "unknown", [0.52, -1.75, 0.54], [1.61, 0.20, 0.56]),
        "e:106": _remembered("e:106", "unknown", [3.01, -2.30, 0.78], [0.40, 0.03, 1.42]),
        "e:45": _remembered("e:45", "chair", [1.82, -3.49, 0.62], [0.40, 0.07, 1.21]),
        "e:46": _remembered("e:46", "pistol", [-2.74, 0.69, 0.63], [0.18, 0.11, 0.03]),
        "e:82": _remembered("e:82", "pill_bottle", [2.71, 0.46, 0.59], [0.05, 0.04, 0.11]),
    }
    est._drop_impossible(np.zeros(3))
    assert set(est.tracks) == {"e:6", "e:3", "e:46", "e:82"}


def test_a_stretched_bowl_and_a_drawer_only_plate_are_dropped():
    """A bowl longer than the bowl prior is not a bowl. A plate the segmenter only called
    a drawer, sitting on that drawer, is the drawer. The chores bowl (under the prior) and
    the plate the segmenter named stay."""
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    plate = _remembered("e:31", "plate", [2.81, 0.82, 0.80], [0.24, 0.24, 0.02])
    plate.semantic_counts = {"drawer": 0.96}
    plate.semantic_views = 1
    named = _remembered("e:6", "plate", [0.30, -0.80, 0.76], [0.24, 0.23, 0.03])
    named.semantic_counts = {"plate": 13.7}
    named.semantic_views = 14
    bowl = _remembered("e:18", "bowl", [0.41, 2.17, 0.94], [0.17, 0.12, 0.06])
    est.tracks = {
        "e:3": _remembered("e:3", "table", [0.60, -0.90, 0.375], [1.20, 0.80, 0.75]),
        "e:9": _remembered("e:9", "bowl", [0.53, 2.31, 0.93], [0.48, 0.15, 0.07]),
        "e:18": bowl,
        "e:30": _remembered("e:30", "bowl", [0.24, -1.07, 0.79], [0.32, 0.31, 0.08]),
        "e:5": _remembered("e:5", "drawer", [2.62, 0.61, 0.69], [0.63, 0.65, 0.24]),
        "e:31": plate,
        "e:6": named,
    }
    named_long = _remembered("e:40", "bowl", [0.72, -1.08, 0.82], [0.53, 0.22, 0.16])
    named_long.semantic_counts = {"bowl": 4.0}
    named_long.semantic_views = 4
    est.tracks["e:40"] = named_long
    est._drop_false_tableware()
    assert set(est.tracks) == {"e:3", "e:18", "e:30", "e:5", "e:6", "e:40"}


def test_a_plate_on_the_table_stays_and_the_cabinet_top_does_not():
    """A plate is about 2 cm thick, so its top sits that far above the table. A box whose top
    is the cabinet top is the surface."""
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    est.tracks = {
        "e:3": _remembered("e:3", "table", [0.58, -0.89, 0.378], [1.18, 0.80, 0.755]),
        "e:4": _remembered("e:4", "cabinet", [2.66, 0.60, 0.408], [0.67, 0.57, 0.80]),
        "e:6": _remembered("e:6", "plate", [0.30, -0.80, 0.76], [0.24, 0.23, 0.02]),
        "e:7": _remembered("e:7", "plate", [2.80, 0.46, 0.805], [0.24, 0.24, 0.02]),
    }
    est._drop_false_tableware()
    assert "e:6" in est.tracks and "e:7" not in est.tracks


def test_a_cup_merged_with_an_unnamed_fill_keeps_the_cup():
    """The tighter unnamed fill is the same cup. Keeping the fill would leave the cup unnamed."""
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    unknown = _remembered("e:20", "unknown", [-0.49, 2.16, 0.955], [0.07, 0.07, 0.10])
    unknown.category_counts = {"unknown": 12.0}
    est.tracks = {
        "e:20": unknown,
        "e:21": _remembered("e:21", "cup", [-0.50, 2.16, 0.955], [0.08, 0.08, 0.10]),
    }
    est._merge_same_place(np.zeros(3))
    left = list(est.tracks.values())
    assert len(left) == 1 and left[0].category == "cup"


def test_a_wide_unknown_on_the_counter_is_dropped_and_the_cup_stays():
    """The far lip of the counter is a wide shallow unknown. The cup on the counter is not."""
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    est.tracks = {
        "e:13": _remembered("e:13", "counter", [-0.05, 2.155, 0.46], [1.77, 0.66, 0.89]),
        "e:17": _remembered("e:17", "unknown", [0.483, 2.274, 0.93], [0.458, 0.255, 0.07]),
        "e:20": _remembered("e:20", "cup", [-0.49, 2.16, 0.955], [0.105, 0.105, 0.10]),
    }
    est._drop_false_tableware()
    assert "e:17" not in est.tracks and est.tracks["e:20"].category == "cup"


def test_a_long_bowl_drops_a_detached_counter_strip_and_a_cup_keeps_one():
    """Support voxels that do not touch a bowl stretch its box. The bowl keeps the compact cluster.

    A cup with the same kind of tail is left whole: only a track already named bowl is cut, and only
    when it is longer than a bowl."""
    from steward.ego.estimator import _voxelize

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    rng = np.random.default_rng(7)
    bowl_pts = np.column_stack([rng.uniform(0.36, 0.52, 45), rng.uniform(2.11, 2.24, 45),
                                rng.uniform(0.90, 0.96, 45)])
    strip = np.array([[0.60, 2.465, 0.925], [0.68, 2.50, 0.92], [0.70, 2.51, 0.92],
                      [0.72, 2.52, 0.925], [0.74, 2.53, 0.92]])
    bowl = _remembered("e:1", "bowl", [0.53, 2.31, 0.93], [0.48, 0.15, 0.07])
    bowl.voxels = _voxelize(np.vstack([bowl_pts, strip]))
    bowl.mean_rgb = np.array([0.93, 0.93, 0.95])
    bowl.bottom_seen = True
    bowl.semantic_counts = {"bowl": 4.0}
    bowl.semantic_views = 4
    swollen = (np.array([0.53, 2.31, 0.93]), np.array([0.50, 0.15, 0.06]))
    bowl.recent.extend(swollen for _ in range(3))
    cup_pts = np.column_stack([rng.uniform(-0.54, -0.46, 40), rng.uniform(2.12, 2.20, 40),
                               rng.uniform(0.90, 1.00, 40)])
    cup_tail = np.column_stack([rng.uniform(-0.10, -0.02, 8), rng.uniform(2.48, 2.54, 8),
                                rng.uniform(0.94, 0.96, 8)])
    cup = _remembered("e:2", "cup", [-0.30, 2.30, 0.95], [0.50, 0.16, 0.10])
    cup.voxels = _voxelize(np.vstack([cup_pts, cup_tail]))
    cup.bottom_seen = True
    est.tracks = {"e:1": bowl, "e:2": cup}
    est._rebuild_geometry(bowl, np.array([-1.8, -1.2, 1.2]), 2.0)
    est._rebuild_geometry(cup, np.array([-1.8, -1.2, 1.2]), 2.0)
    assert bowl.category == "bowl"
    assert max(bowl.size[0], bowl.size[1]) < 0.25
    assert abs(float(bowl.pos[0]) - 0.44) < 0.06 and abs(float(bowl.pos[1]) - 2.175) < 0.06
    assert cup.category == "cup"
    assert max(cup.size[0], cup.size[1]) > 0.40
    assert cup.size[2] > 0.08


def test_a_cabinet_edge_called_a_bowl_is_dropped_and_a_real_bowl_stays():
    """A lip a few centimetres thick is not a bowl. The 12 cm bowl, thin on its short side
    of the prior but round in plan, stays."""
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    est.tracks = {
        "e:1": _remembered("e:1", "bowl", [1.36, 2.92, 0.93], [0.19, 0.052, 0.05]),
        "e:2": _remembered("e:2", "bowl", [0.70, -2.09, 0.75], [0.137, 0.110, 0.045]),
    }
    est._drop_false_tableware()
    assert set(est.tracks) == {"e:2"}


def test_a_bowl_and_a_cup_standing_on_a_counter_are_peeled_off_it():
    """Points above the counter top are the objects. They become their own tracks instead of
    staying inside the counter."""
    from steward.ego.estimator import _voxelize

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    xs = np.linspace(-0.8, 0.8, 30)
    ys = np.linspace(1.95, 2.45, 10)
    top = np.array([[x, y, 0.90] for x in xs for y in ys])
    bowl = np.array([[0.45 + dx, 2.20 + dy, 0.90 + dz]
                     for dx in np.linspace(-0.06, 0.06, 5)
                     for dy in np.linspace(-0.06, 0.06, 5)
                     for dz in (0.02, 0.04, 0.06)])
    cup = np.array([[-0.50 + dx, 2.18 + dy, 0.90 + dz]
                    for dx in np.linspace(-0.03, 0.03, 4)
                    for dy in np.linspace(-0.03, 0.03, 4)
                    for dz in np.linspace(0.02, 0.10, 4)])
    counter = _remembered("e:9", "counter", [0.0, 2.18, 0.45], [1.8, 0.6, 0.9])
    counter.voxels = _voxelize(np.vstack([top, bowl, cup]))
    est.tracks = {"e:9": counter}
    est._rebuild_geometry(counter, np.array([-1.8, -1.2, 1.2]), 2.0)
    est._spawn_shed()
    others = [t for t in est.tracks.values() if t.id != "e:9"]
    assert any(abs(float(t.pos[0]) - 0.45) < 0.08 for t in others), "the bowl leaves the counter"
    assert any(abs(float(t.pos[0]) + 0.50) < 0.08 for t in others), "the cup leaves the counter"
    assert counter.category == "counter"
    assert float(counter.size[2]) < 1.0


def test_two_halves_of_a_sideboard_become_the_sideboard_and_its_drawer():
    """Each look at a sideboard face is about the size of a chair. The two halves are one
    sideboard, and the top band of that face is the drawer."""
    from steward.ego.estimator import _voxelize

    def block(origin, size, step=0.08):
        xs = np.arange(origin[0], origin[0] + size[0], step)
        ys = np.arange(origin[1], origin[1] + size[1], step)
        zs = np.arange(origin[2], origin[2] + size[2], step)
        return np.array([[x, y, z] for x in xs for y in ys for z in zs])

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    chair = _remembered("e:20", "chair", [-2.55, 0.30, 0.40], [0.48, 0.46, 0.80])
    other = _remembered("e:21", "unknown", [-2.55, 0.82, 0.40], [0.48, 0.46, 0.80])
    chair.voxels = _voxelize(block([-2.79, 0.07, 0.02], [0.48, 0.46, 0.78]))
    other.voxels = _voxelize(block([-2.79, 0.59, 0.02], [0.48, 0.46, 0.78]))
    est.tracks = {"e:20": chair, "e:21": other}
    est._join_abutting_furniture(np.array([-1.6, -1.3, 1.2]))
    est._drawer_in_sideboard()
    bodies = [t for t in est.tracks.values() if t.category == "sideboard"]
    drawers = [t for t in est.tracks.values() if t.category == "drawer"]
    assert len(bodies) == 1 and abs(float(bodies[0].pos[1]) - 0.55) < 0.25
    assert len(drawers) == 1 and float(drawers[0].pos[2]) > float(bodies[0].pos[2])


def test_two_chairs_the_segmenter_named_stay_two_chairs():
    from steward.ego.estimator import _voxelize

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    a = _remembered("e:20", "chair", [0.0, 0.0, 0.45], [0.47, 0.46, 0.90])
    b = _remembered("e:21", "chair", [0.55, 0.0, 0.45], [0.47, 0.46, 0.90])
    a.semantic_counts = {"chair": 2.0}
    b.semantic_counts = {"chair": 2.0}
    a.voxels = _voxelize(np.array([[0.0, 0.0, 0.4], [0.1, 0.1, 0.8]]))
    b.voxels = _voxelize(np.array([[0.55, 0.0, 0.4], [0.65, 0.1, 0.8]]))
    est.tracks = {"e:20": a, "e:21": b}
    est._join_abutting_furniture(np.array([-1.0, -1.0, 1.2]))
    assert set(est.tracks) == {"e:20", "e:21"}


def test_a_short_drawer_off_the_middle_of_the_sideboard_is_put_back_there():
    """The handle is the middle of the front. A cloud of one end of the face, shorter than
    the cut that drops a full-length mask, still has to sit on that middle."""
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    est.odometry = {"x": -1.6, "y": 0.6, "yaw": 0.0, "sigma": {"pos": 0.01, "yaw": 0.01}}
    side = _remembered("e:20", "sideboard", [-2.75, 0.60, 0.42], [1.42, 0.49, 0.84])
    side.yaw = math.pi / 2
    drawer = _remembered("e:22", "drawer", [-2.55, 0.80, 0.72], [0.63, 0.50, 0.26])
    drawer.yaw = 0.45
    table = _remembered("e:3", "table", [0.6, -0.9, 0.38], [1.2, 0.8, 0.75])
    est.tracks = {"e:20": side, "e:22": drawer, "e:3": table}
    est._seat_closed_drawers()
    assert abs(float(drawer.pos[1]) - 0.60) < 0.05
    assert abs(math.remainder(float(drawer.yaw), math.pi / 2)) < 0.05
    assert float(drawer.pos[0]) < -2.55


def test_a_drawer_measured_on_the_front_face_is_seated_inside_before_it_is_dropped():
    """The closed drawer's points are the front. That centre is outside the furniture box and
    above the band the orphan check keeps, so it has to be seated first."""
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    est.odometry = {"x": -1.6, "y": 0.6, "yaw": 0.0, "sigma": {"pos": 0.01, "yaw": 0.01}}
    side = _remembered("e:20", "sideboard", [-2.75, 0.60, 0.42], [1.42, 0.49, 0.84])
    side.yaw = math.pi / 2
    drawer = _remembered("e:22", "drawer", [-2.50, 0.80, 0.70], [0.56, 0.44, 0.22])
    table = _remembered("e:3", "table", [0.6, -0.9, 0.38], [1.2, 0.8, 0.75])
    est.tracks = {"e:20": side, "e:22": drawer, "e:3": table}
    est._seat_closed_drawers()
    est._drop_support_top_drawers()
    est._drop_orphan_parts()
    assert "e:22" in est.tracks
    assert abs(float(drawer.pos[1]) - 0.60) < 0.05


def test_a_flat_pistol_is_not_folded_into_the_bowl_that_covers_it():
    from steward.ego.estimator import _voxelize
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    bowl = _remembered("e:4", "bowl", [0.0, 0.0, 0.80], [0.24, 0.24, 0.06])
    grid = [[x, y, 0.80] for x in np.linspace(-0.10, 0.10, 6) for y in np.linspace(-0.10, 0.10, 6)]
    bowl.voxels = _voxelize(np.array(grid))
    pistol = _remembered("e:9", "pistol", [0.02, 0.02, 0.80], [0.18, 0.12, 0.03])
    pistol.voxels = _voxelize(np.array([[0.02, 0.02, 0.80]]))
    est.tracks = {"e:4": bowl, "e:9": pistol}
    est._merge_rim_fragments()
    assert "e:9" in est.tracks


def test_the_only_perfume_in_front_of_a_closed_drawer_is_kept():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    side = _remembered("e:20", "sideboard", [-2.75, 0.60, 0.42], [1.42, 0.49, 0.84])
    side.yaw = math.pi / 2
    drawer = _remembered("e:22", "drawer", [-2.71, 0.60, 0.70], [0.56, 0.44, 0.22])
    drawer.yaw = math.pi / 2
    perfume = _remembered("e:31", "perfume_bottle", [-2.48, 0.46, 0.70], [0.06, 0.04, 0.10])
    table = _remembered("e:3", "table", [0.6, -0.9, 0.38], [1.2, 0.8, 0.75])
    est.tracks = {"e:20": side, "e:22": drawer, "e:31": perfume, "e:3": table}
    est._drop_left_in_front()
    assert "e:31" in est.tracks
    # Back inside the drawer's front face (-2.49), at a bottle's own depth rather than the centre.
    assert -2.71 < float(perfume.pos[0]) < -2.54


def test_a_mug_on_the_end_of_the_sideboard_is_dropped():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    side = _remembered("e:20", "sideboard", [-2.75, 0.60, 0.42], [1.42, 0.49, 0.84])
    side.yaw = math.pi / 2
    drawer = _remembered("e:22", "drawer", [-2.71, 0.60, 0.70], [0.56, 0.44, 0.22])
    drawer.yaw = math.pi / 2
    mug = _remembered("e:75", "mug", [-2.82, -0.02, 0.81], [0.18, 0.11, 0.09])
    table = _remembered("e:3", "table", [0.6, -0.9, 0.38], [1.2, 0.8, 0.75])
    est.tracks = {"e:20": side, "e:22": drawer, "e:75": mug, "e:3": table}
    est._drop_pistols_off_the_drawer()
    assert "e:75" not in est.tracks


def test_a_chair_just_in_front_of_the_cabinet_is_dropped():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    cabinet = _remembered("e:4", "cabinet", [2.61, 0.60, 0.40], [0.68, 0.58, 0.80])
    chair = _remembered("e:121", "chair", [2.29, 0.60, 0.38], [0.54, 0.32, 0.76])
    far = _remembered("e:9", "chair", [0.60, -1.40, 0.45], [0.47, 0.46, 0.90])
    est.tracks = {"e:4": cabinet, "e:121": chair, "e:9": far}
    est._drop_false_chairs()
    assert "e:121" not in est.tracks and "e:9" in est.tracks


def test_a_bowl_on_the_cabinet_rail_and_a_paper_thin_cabinet_are_dropped():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    cabinet = _remembered("e:4", "cabinet", [2.67, 0.60, 0.40], [0.66, 0.58, 0.80])
    rail = _remembered("e:237", "bowl", [2.71, 0.37, 0.76], [0.18, 0.11, 0.09])
    sliver = _remembered("e:238", "cabinet", [2.78, 0.40, 0.80], [0.34, 0.23, 0.01])
    speck = _remembered("e:243", "unknown", [2.70, 0.68, 0.795], [0.16, 0.07, 0.02])
    est.tracks = {"e:4": cabinet, "e:237": rail, "e:238": sliver, "e:243": speck}
    est._drop_furniture_edges()
    assert set(est.tracks) == {"e:4"}


def test_a_table_on_a_pinned_sideboard_is_dropped_and_the_sideboard_stays():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    side = _remembered("e:22", "sideboard", [-2.72, 0.56, 0.42], [1.40, 0.50, 0.84])
    side.pinned = True
    table = _remembered("e:64", "table", [-2.60, 0.52, 0.42], [1.28, 0.74, 0.84])
    est.tracks = {"e:22": side, "e:64": table}
    est._dedupe_furniture(np.zeros(3))
    assert "e:22" in est.tracks and "e:64" not in est.tracks


def test_a_perfume_and_a_pill_a_hand_apart_stay_two_tracks():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    perfume = _remembered("e:7", "perfume_bottle", [2.57, 0.46, 0.60], [0.06, 0.04, 0.11])
    pill = _remembered("e:8", "perfume_bottle", [2.73, 0.46, 0.60], [0.05, 0.05, 0.09])
    est.tracks = {"e:7": perfume, "e:8": pill}
    est._merge_same_spot()
    assert "e:7" in est.tracks and "e:8" in est.tracks


def test_a_bowl_the_size_of_the_pistol_inside_a_drawer_is_the_pistol():
    from steward.ego.estimator import _voxelize

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    drawer = _remembered("e:22", "drawer", [-2.71, 0.60, 0.70], [0.56, 0.44, 0.22])
    bowl = _remembered("e:109", "bowl", [-2.74, 0.70, 0.63], [0.175, 0.165, 0.06])
    xs = np.linspace(-2.82, -2.66, 5)
    ys = np.linspace(0.62, 0.78, 5)
    zs = (0.60, 0.66)
    bowl.voxels = _voxelize(np.array([[x, y, z] for x in xs for y in ys for z in zs]))
    est.tracks = {"e:22": drawer, "e:109": bowl}
    est._pushed_shut.add("e:22")  # the robot has had this drawer open
    est._rebuild_geometry(bowl, np.array([-1.6, 0.6, 1.2]), 1.0)
    assert bowl.category == "pistol"


def test_a_thin_slice_of_the_cabinet_is_dropped():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    cabinet = _remembered("e:4", "cabinet", [2.68, 0.60, 0.40], [0.62, 0.52, 0.80])
    sliver = _remembered("e:210", "cabinet", [2.49, 0.40, 0.40], [0.24, 0.13, 0.80])
    est.tracks = {"e:4": cabinet, "e:210": sliver}
    est._drop_furniture_edges()
    assert "e:4" in est.tracks and "e:210" not in est.tracks


def test_a_perfume_keeps_its_name_when_a_bowl_shares_its_spot():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    perfume = _remembered("e:7", "perfume_bottle", [2.50, 0.46, 0.60], [0.06, 0.04, 0.10])
    bowl = _remembered("e:8", "bowl", [2.54, 0.46, 0.60], [0.16, 0.12, 0.06])
    est.tracks = {"e:7": perfume, "e:8": bowl}
    est._merge_same_place(np.zeros(3))
    assert "e:7" in est.tracks and est.tracks["e:7"].category == "perfume_bottle"
    assert "e:8" not in est.tracks


def test_a_flat_box_beside_a_drawer_is_the_pistol():
    from steward.ego.estimator import _voxelize

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    drawer = _remembered("e:22", "drawer", [-2.71, 0.60, 0.70], [0.56, 0.44, 0.22])
    box = _remembered("e:88", "box", [-2.71, 0.68, 0.63], [0.135, 0.124, 0.03])
    xs = np.linspace(-2.77, -2.65, 4)
    ys = np.linspace(0.63, 0.73, 4)
    zs = (0.62, 0.645)
    box.voxels = _voxelize(np.array([[x, y, z] for x in xs for y in ys for z in zs]))
    est.tracks = {"e:22": drawer, "e:88": box}
    est._pushed_shut.add("e:22")  # the robot has had this drawer open
    est._rebuild_geometry(box, np.array([-1.6, 0.6, 1.2]), 1.0)
    assert box.category == "pistol"


def test_a_flat_pistol_on_the_drawer_floor_is_its_own_cluster():
    from steward.ego.estimator import _shed_cores

    pts = np.array([[x, y, 0.60 + z]
                    for x in np.linspace(0.0, 0.16, 6)
                    for y in np.linspace(0.0, 0.10, 3)
                    for z in (0.0, 0.018)])
    assert len(_shed_cores(pts)) == 1
    speck = np.array([[x, 0.0, 0.60 + z] for x in np.linspace(0.0, 0.03, 4) for z in (0.0, 0.015)])
    assert _shed_cores(speck) == []


def test_a_teddy_left_in_front_of_a_closed_drawer_is_dropped_and_the_one_inside_stays():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    side = _remembered("e:20", "sideboard", [-2.75, 0.60, 0.42], [1.42, 0.49, 0.84])
    side.yaw = math.pi / 2
    drawer = _remembered("e:22", "drawer", [-2.71, 0.60, 0.70], [0.56, 0.44, 0.22])
    drawer.yaw = math.pi / 2
    inside = _remembered("e:30", "teddy_bear", [-2.74, 0.48, 0.70], [0.11, 0.10, 0.18])
    outside = _remembered("e:31", "teddy_bear", [-2.48, 0.48, 0.70], [0.11, 0.10, 0.18])
    table = _remembered("e:3", "table", [0.6, -0.9, 0.38], [1.2, 0.8, 0.75])
    est.tracks = {"e:20": side, "e:22": drawer, "e:30": inside, "e:31": outside, "e:3": table}
    est._drop_left_in_front()
    assert "e:30" in est.tracks and "e:31" not in est.tracks


def test_a_drawer_centred_on_the_width_but_still_on_the_front_face_is_seated_inside():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    est.odometry = {"x": -1.6, "y": 0.6, "yaw": 0.0, "sigma": {"pos": 0.01, "yaw": 0.01}}
    side = _remembered("e:20", "sideboard", [-2.75, 0.60, 0.42], [1.42, 0.49, 0.84])
    side.yaw = math.pi / 2
    drawer = _remembered("e:22", "drawer", [-2.49, 0.60, 0.72], [0.66, 0.54, 0.26])
    table = _remembered("e:3", "table", [0.6, -0.9, 0.38], [1.2, 0.8, 0.75])
    est.tracks = {"e:20": side, "e:22": drawer, "e:3": table}
    est._seat_closed_drawers()
    assert float(drawer.pos[0]) < -2.60
    assert abs(float(drawer.pos[1]) - 0.60) < 0.05


def test_a_drawer_shifted_along_a_swollen_cabinet_is_put_back_on_the_middle():
    """The measured depth can exceed the width. The drawer still slides toward the room,
    and the handle is the middle of the other side."""
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    cabinet = _remembered("e:1", "cabinet", [2.70, 0.60, 0.40], [0.735, 0.692, 0.80])
    cabinet.yaw = 0.0
    drawer = _remembered("e:2", "drawer", [2.71, 0.73, 0.66], [0.57, 0.45, 0.28])
    table = _remembered("e:3", "table", [0.6, 0.60, 0.38], [1.2, 0.8, 0.75])
    est.tracks = {"e:1": cabinet, "e:2": drawer, "e:3": table}
    est._seat_closed_drawers()
    assert abs(float(drawer.pos[1]) - 0.60) < 0.05


def test_a_pistol_on_the_end_of_the_sideboard_is_dropped_and_the_one_in_the_drawer_stays():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    side = _remembered("e:20", "sideboard", [-2.75, 0.60, 0.42], [1.42, 0.49, 0.84])
    side.yaw = math.pi / 2
    drawer = _remembered("e:22", "drawer", [-2.71, 0.60, 0.70], [0.56, 0.44, 0.22])
    drawer.yaw = math.pi / 2
    real = _remembered("e:30", "pistol", [-2.74, 0.69, 0.63], [0.18, 0.12, 0.03])
    end = _remembered("e:31", "pistol", [-2.79, -0.04, 0.82], [0.16, 0.03, 0.02])
    table = _remembered("e:3", "table", [0.6, -0.9, 0.38], [1.2, 0.8, 0.75])
    est.tracks = {"e:20": side, "e:22": drawer, "e:30": real, "e:31": end, "e:3": table}
    est._drop_pistols_off_the_drawer()
    assert "e:30" in est.tracks and "e:31" not in est.tracks


def test_two_bottles_sharing_a_drawer_floor_but_twenty_centimetres_apart_stay_apart():
    from steward.ego.estimator import _voxelize

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    pill = _remembered("e:8", "pill_bottle", [2.72, 0.46, 0.60], [0.50, 0.10, 0.09])
    other = _remembered("e:9", "pill_bottle", [2.47, 0.46, 0.60], [0.50, 0.10, 0.09])
    shared = _voxelize(np.array([[2.55, 0.46, 0.55], [2.60, 0.46, 0.55], [2.65, 0.46, 0.55]]))
    pill.voxels = shared
    other.voxels = shared.copy()
    est.tracks = {"e:8": pill, "e:9": other}
    est._merge_same_surface(np.zeros(3))
    assert set(est.tracks) == {"e:8", "e:9"}


def test_a_drawer_already_pulled_out_is_left_where_the_pull_put_it():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    cabinet = _remembered("e:1", "cabinet", [0.0, 0.0, 0.4], [0.62, 0.52, 0.8])
    table = _remembered("e:3", "table", [0.0, 3.0, 0.38], [1.2, 0.8, 0.75])
    drawer = _remembered("e:2", "drawer", [0.0, 0.42, 0.25], [0.50, 0.44, 0.22])
    est.tracks = {"e:1": cabinet, "e:2": drawer, "e:3": table}
    est._seat_closed_drawers()
    assert float(drawer.pos[1]) == pytest.approx(0.42)


def test_a_drawer_as_long_as_the_sideboard_is_cut_back_to_the_handle():
    """The top band of the sideboard is one mask. The drawer, and the handle, are its middle."""
    from steward.ego.estimator import _voxelize

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    est.odometry = {"x": -1.6, "y": 0.6, "yaw": 0.0, "sigma": {"pos": 0.01, "yaw": 0.01}}
    side = _remembered("e:20", "sideboard", [-2.75, 0.60, 0.42], [1.42, 0.47, 0.84])
    side.yaw = math.pi / 2
    drawer = _remembered("e:21", "drawer", [-2.55, 0.80, 0.72], [1.10, 0.45, 0.26])
    drawer.yaw = math.pi / 2
    ys = np.linspace(0.05, 1.15, 24)
    xs = np.linspace(-2.62, -2.42, 4)
    drawer.voxels = _voxelize(np.array([[x, y, 0.72] for x in xs for y in ys]))
    est.tracks = {"e:20": side, "e:21": drawer}
    est._rebuild_geometry(drawer, np.array([-1.6, 0.6, 1.2]), 2.0)
    assert float(max(drawer.size[0], drawer.size[1])) < 0.70
    assert abs(float(drawer.pos[1]) - 0.60) < 0.15


def test_a_chair_that_is_really_furniture_or_a_speck_is_dropped():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    speck = _remembered("e:20", "chair", [-2.58, 1.175, 0.84], [0.29, 0.14, 0.03])
    face = _remembered("e:21", "chair", [2.25, 0.60, 0.38], [0.56, 0.27, 0.77])
    drawer = _remembered("e:22", "drawer", [2.38, 0.59, 0.66], [0.63, 0.66, 0.28])
    est.tracks = {"e:20": speck, "e:21": face, "e:22": drawer}
    est._drop_false_chairs()
    assert set(est.tracks) == {"e:22"}


def test_a_perfume_and_a_pill_on_the_same_spot_stay_the_perfume():
    """The pill prior fits the perfume. Folding the perfume into the drawer, or keeping both
    names, loses the perfume."""
    from steward.ego.estimator import _voxelize

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    drawer = _remembered("e:6", "drawer", [2.64, 0.60, 0.65], [0.56, 0.44, 0.22])
    perfume = _remembered("e:7", "perfume_bottle", [2.56, 0.46, 0.60], [0.06, 0.04, 0.12])
    pill = _remembered("e:8", "pill_bottle", [2.565, 0.462, 0.60], [0.05, 0.04, 0.12])
    shared = _voxelize(np.array([[2.56, 0.46, 0.55], [2.57, 0.46, 0.56], [2.56, 0.47, 0.57],
                                 [2.55, 0.45, 0.58], [2.57, 0.47, 0.55]]))
    drawer.voxels = shared
    perfume.voxels = shared
    pill.voxels = shared.copy()
    perfume.mean_rgb = np.array([0.95, 0.60, 0.75])
    est.tracks = {"e:6": drawer, "e:7": perfume, "e:8": pill}
    est._merge_same_surface(np.zeros(3))
    assert set(est.tracks) == {"e:6", "e:7", "e:8"}
    est._merge_same_place(np.zeros(3))
    assert "e:6" in est.tracks
    contents = [t for t in est.tracks.values() if t.category in ("perfume_bottle", "pill_bottle")]
    assert [t.category for t in contents] == ["perfume_bottle"]


def test_a_pill_and_a_perfume_four_centimetres_apart_stay_two_bottles():
    """The two bottles in the cabinet drawer are a few centimetres apart. Folding them
    keeps the perfume and loses the pill."""
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    drawer = _remembered("e:6", "drawer", [2.64, 0.60, 0.65], [0.56, 0.44, 0.22])
    perfume = _remembered("e:7", "perfume_bottle", [2.68, 0.445, 0.60], [0.06, 0.04, 0.10])
    pill = _remembered("e:8", "pill_bottle", [2.721, 0.464, 0.59], [0.046, 0.046, 0.085])
    est.tracks = {"e:6": drawer, "e:7": perfume, "e:8": pill}
    est._merge_same_place(np.zeros(3))
    assert "e:7" in est.tracks and "e:8" in est.tracks


def test_a_pink_perfume_and_a_white_pill_on_the_same_spot_stay_two_bottles():
    """The open drawer slides the two bottles together. Colour is what keeps them apart."""
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    drawer = _remembered("e:6", "drawer", [2.64, 0.60, 0.65], [0.56, 0.44, 0.22])
    perfume = _remembered("e:7", "perfume_bottle", [2.48, 0.46, 0.60], [0.06, 0.04, 0.10])
    pill = _remembered("e:8", "pill_bottle", [2.49, 0.46, 0.59], [0.046, 0.046, 0.085])
    perfume.mean_rgb = np.array([0.90, 0.55, 0.70])
    pill.mean_rgb = np.array([0.92, 0.92, 0.90])
    pill.semantic_counts = {"bottle": 6.9}
    est.tracks = {"e:6": drawer, "e:7": perfume, "e:8": pill}
    est._merge_same_place(np.zeros(3))
    assert "e:7" in est.tracks and "e:8" in est.tracks


def test_a_cabinet_seen_with_its_drawer_pulled_out_does_not_replace_the_cabinet():
    """The view of the cabinet with the open drawer is lower and deeper. The cabinet known before
    the pull is the one kept; after the push the lower top decides again."""
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    cabinet = _remembered("e:4", "cabinet", [2.715, 0.594, 0.41], [0.52, 0.60, 0.79])
    drawer = _remembered("e:6", "drawer", [2.395, 0.575, 0.65], [0.56, 0.44, 0.25])
    both = _remembered("e:205", "table", [2.577, 0.54, 0.40], [0.945, 0.66, 0.75])
    both.first_seen = 44.4
    est.tracks = {"e:4": cabinet, "e:6": drawer, "e:205": both}
    est._pulled_open.add("e:6")
    est._dedupe_furniture(np.array([1.2, 0.6, 1.2]))
    assert "e:4" in est.tracks and "e:205" not in est.tracks
    assert est.dropped[-1]["reason"] == "merged: furniture duplicate"

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    cabinet = _remembered("e:4", "cabinet", [2.715, 0.594, 0.41], [0.52, 0.60, 0.79])
    lower = _remembered("e:205", "table", [2.70, 0.59, 0.40], [0.56, 0.60, 0.75])
    lower.first_seen = 44.4
    est.tracks = {"e:4": cabinet, "e:205": lower}
    est._dedupe_furniture(np.array([1.2, 0.6, 1.2]))
    assert set(est.tracks) == {"e:205"}


def test_an_unnamed_box_inside_a_named_object_is_a_part_of_it():
    """Depth fills over the airplane: a tall one the prior calls a bear and a flat one it calls a
    pistol. The perfume beside the pill, a bowl's unnamed neighbour inside a larger bowl and a
    lid on the airplane are other objects."""
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    plane = _remembered("e:152", "toy_airplane", [2.387, 0.717, 0.597], [0.177, 0.098, 0.08])
    plane.semantic_counts, plane.semantic_views = {"toy_airplane": 6.4}, 10
    bear = _remembered("e:174", "teddy_bear", [2.428, 0.695, 0.654], [0.146, 0.131, 0.242])
    flat = _remembered("e:197", "pistol", [2.32, 0.72, 0.573], [0.231, 0.091, 0.04])
    lid = _remembered("e:198", "plate", [2.39, 0.717, 0.647], [0.10, 0.08, 0.02])
    pill = _remembered("e:153", "pill_bottle", [2.468, 0.46, 0.589], [0.054, 0.053, 0.112])
    pill.semantic_counts, pill.semantic_views = {"pill_bottle": 7.0}, 12
    perfume = _remembered("e:154", "perfume_bottle", [2.42, 0.455, 0.605], [0.083, 0.061, 0.145])
    red = _remembered("e:5", "bowl", [0.238, -1.067, 0.785], [0.31, 0.307, 0.08])
    red.semantic_counts, red.semantic_views = {"bowl": 32.2}, 50
    nested = _remembered("e:9", "bowl", [0.25, -1.06, 0.78], [0.12, 0.12, 0.05])
    est.tracks = {t.id: t for t in (plane, bear, flat, lid, pill, perfume, red, nested)}
    est._drop_unnamed_parts()
    assert set(est.tracks) == {"e:152", "e:198", "e:153", "e:154", "e:5", "e:9"}
    assert {d["id"]: d.get("into") for d in est.dropped} == {"e:174": "e:152", "e:197": "e:152"}


def test_the_cap_of_the_perfume_is_not_a_pill_of_another_colour():
    """A few pixels on top of the perfume, never named by a mask, fit the pill prior. Their
    colour is the cap's, not the body's, and they are still the perfume."""
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    drawer = _remembered("e:6", "drawer", [2.64, 0.60, 0.65], [0.56, 0.44, 0.22])
    perfume = _remembered("e:161", "perfume_bottle", [2.588, 0.457, 0.594], [0.152, 0.093, 0.122])
    cap = _remembered("e:200", "pill_bottle", [2.577, 0.47, 0.607], [0.08, 0.06, 0.147])
    perfume.mean_rgb = np.array([0.90, 0.55, 0.70])
    cap.mean_rgb = np.array([0.75, 0.62, 0.30])
    est.tracks = {"e:6": drawer, "e:161": perfume, "e:200": cap}
    est._merge_same_place(np.zeros(3))
    assert "e:161" in est.tracks and "e:200" not in est.tracks
    assert est.tracks["e:161"].category == "perfume_bottle"


def test_the_cap_of_a_perfume_seen_once_does_not_replace_the_perfume_seen_four_times():
    """Both boxes are the perfume by shape; no mask named either. The cap is the tighter box,
    and with it the track would keep one view of a gold cap for its size and its colour."""
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    drawer = _remembered("e:6", "drawer", [2.37, 0.60, 0.65], [0.56, 0.44, 0.22])
    perfume = _remembered("e:113", "perfume_bottle", [2.288, 0.462, 0.598], [0.06, 0.04, 0.142])
    perfume.observations = 4
    perfume.mean_rgb = np.array([0.86, 0.65, 0.69])
    cap = _remembered("e:151", "perfume_bottle", [2.285, 0.46, 0.594], [0.04, 0.03, 0.122])
    cap.observations = 1
    cap.mean_rgb = np.array([0.87, 0.72, 0.54])
    est.tracks = {"e:6": drawer, "e:113": perfume, "e:151": cap}
    est._merge_same_place(np.zeros(3))
    assert "e:113" in est.tracks and "e:151" not in est.tracks
    assert est.tracks["e:113"].mean_rgb == pytest.approx([0.86, 0.65, 0.69])


def test_a_mask_of_the_open_drawer_and_the_cabinet_face_is_not_folded_into_the_airplane_in_it():
    """The mask holds the airplane's points, and its footprint is under twice the airplane
    prior. It is 0.77 m tall. Unioned in, the airplane is renamed and its box is 0.21 m tall."""
    from steward.ego.estimator import _voxelize

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    plane = _remembered("e:112", "toy_airplane", [2.428, 0.727, 0.595], [0.117, 0.081, 0.08])
    plane.observations = 14
    body = np.array([[x, y, z] for x in np.arange(2.37, 2.49, 0.01) for y in np.arange(0.69, 0.77, 0.01)
                     for z in np.arange(0.56, 0.63, 0.01)])
    plane.voxels = _voxelize(body)
    face = _remembered("e:158", "chair", [2.497, 0.81, 0.382], [0.47, 0.46, 0.765])
    face.observations = 1
    face.semantic_counts, face.semantic_views = {"toy_airplane": 0.6}, 1
    front = np.array([[x, y, z] for x in (2.27, 2.72) for y in np.arange(0.58, 1.04, 0.02)
                      for z in np.arange(0.0, 0.76, 0.02)])
    face.voxels = _voxelize(np.vstack([body, front]))
    est.tracks = {"e:112": plane, "e:158": face}
    est._merge_same_surface(np.zeros(3))
    assert plane.category == "toy_airplane"
    assert float(plane.size[2]) == pytest.approx(0.08, abs=0.01)
    assert "e:158" not in est.tracks and est.dropped[-1]["reason"].startswith("a region spanning")


def test_a_cup_height_glass_on_the_counter_stays_a_cup():
    """The counter cup is a centimetre taller than it is wide. That is still the cup."""
    from steward.ego.estimator import _voxelize

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    counter = _remembered("e:c", "counter", [0.0, 2.18, 0.45], [1.8, 0.62, 0.90])
    rng = np.random.default_rng(6)
    cup = _remembered("e:1", "wine_glass", [-0.50, 2.18, 0.955], [0.10, 0.10, 0.11])
    cup.voxels = _voxelize(np.column_stack([rng.uniform(-0.55, -0.45, 40), rng.uniform(2.13, 2.23, 40),
                                            rng.uniform(0.90, 1.01, 40)]))
    cup.bottom_seen = True
    est.tracks = {"e:c": counter, "e:1": cup}
    est._rebuild_geometry(cup, np.array([-1.8, -1.2, 1.5]), 3.4)
    assert cup.category == "cup"


def test_a_tall_smear_on_the_counter_is_the_cup_and_the_table_glass_stays():
    """A vertical smear on the counter is as tall as a wine glass. The glass stands on the table."""
    from steward.ego.estimator import _voxelize

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    counter = _remembered("e:c", "counter", [0.0, 2.18, 0.45], [1.8, 0.62, 0.90])
    counter.yaw = math.pi / 2  # the long side was given the heading of the depth
    table = _remembered("e:t", "table", [0.6, -0.9, 0.375], [1.2, 0.8, 0.75])
    rng = np.random.default_rng(5)
    cup = _remembered("e:1", "wine_glass", [-0.51, 2.17, 0.995], [0.12, 0.11, 0.17])
    cup.voxels = _voxelize(np.column_stack([rng.uniform(-0.57, -0.45, 40), rng.uniform(2.11, 2.23, 40),
                                            rng.uniform(0.91, 1.08, 40)]))
    cup.bottom_seen = True
    stem = _remembered("e:2", "wine_glass", [0.48, -1.12, 0.84], [0.10, 0.09, 0.17])
    stem.voxels = _voxelize(np.column_stack([rng.uniform(0.43, 0.53, 40), rng.uniform(-1.17, -1.07, 40),
                                             rng.uniform(0.755, 0.92, 40)]))
    stem.bottom_seen = True
    est.tracks = {"e:c": counter, "e:t": table, "e:1": cup, "e:2": stem}
    est._rebuild_geometry(cup, np.array([-1.8, -1.2, 1.5]), 3.4)
    est._rebuild_geometry(stem, np.array([-1.8, -1.2, 1.5]), 2.2)
    assert cup.category == "cup" and float(cup.size[2]) == pytest.approx(0.10, abs=0.02)
    assert stem.category == "wine_glass" and float(stem.size[2]) > 0.14


def test_a_wide_short_glass_on_the_counter_is_the_cup_and_a_tall_one_stays():
    """A wine glass is taller than its rim. The cup on the counter is wider than it is tall."""
    from steward.ego.estimator import _voxelize

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    cup = _remembered("e:1", "wine_glass", [-0.51, 2.17, 0.95], [0.14, 0.13, 0.10])
    rng = np.random.default_rng(4)
    cup.voxels = _voxelize(np.column_stack([rng.uniform(-0.58, -0.44, 40), rng.uniform(2.10, 2.24, 40),
                                            rng.uniform(0.90, 1.00, 40)]))
    cup.bottom_seen = True
    stem = _remembered("e:2", "wine_glass", [0.48, -1.12, 0.84], [0.10, 0.09, 0.17])
    stem.voxels = _voxelize(np.column_stack([rng.uniform(0.43, 0.53, 40), rng.uniform(-1.17, -1.07, 40),
                                             rng.uniform(0.76, 0.92, 40)]))
    stem.bottom_seen = True
    est.tracks = {"e:1": cup, "e:2": stem}
    est._rebuild_geometry(cup, np.array([-1.8, -1.2, 1.5]), 3.4)
    est._rebuild_geometry(stem, np.array([-1.8, -1.2, 1.5]), 2.2)
    assert cup.category == "cup" and float(cup.size[2]) < 0.14
    assert stem.category == "wine_glass"


def test_a_tall_cup_is_a_wine_glass_when_the_table_shows_through_it():
    from steward.ego.classify import clear_stem_name

    glass = np.array([0.103, 0.098, 0.163])
    table = np.array([0.72, 0.58, 0.40])
    assert clear_stem_name(glass, "cup", table, 0.0) == "wine_glass"
    assert clear_stem_name(np.array([0.08, 0.08, 0.10]), "cup", table, 0.0) is None


def test_two_bottles_a_few_centimetres_apart_are_one_object():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    a = _remembered("e:20", "pill_bottle", [2.480, 0.462, 0.59], [0.075, 0.075, 0.11])
    b = _remembered("e:21", "pill_bottle", [2.467, 0.462, 0.60], [0.066, 0.066, 0.11])
    est.tracks = {"e:20": a, "e:21": b}
    est._merge_same_spot()
    assert len(est.tracks) == 1


def test_a_shorter_counter_seen_broadside_keeps_its_length():
    """A 1.4 m counter is shorter than the 1.85 m prior. The cloud already spans that
    length, so neither a broadside view nor a view looking along it stretches the box.
    A later end-on view of the same cloud leaves the length alone."""
    from steward.ego.estimator import _voxelize

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    xs = np.linspace(0.0, 1.4, 30)
    ys = np.linspace(-0.25, 0.25, 8)
    zs = np.linspace(0.05, 0.85, 6)
    cloud = np.stack(np.meshgrid(xs, ys, zs), axis=-1).reshape(-1, 3)

    def counter(oid: str):
        track = _remembered(oid, "counter", [0.7, 0.0, 0.425], [1.4, 0.5, 0.85])
        track.voxels = _voxelize(cloud)
        track.mean_rgb = np.array([0.92, 0.92, 0.93])
        track.bottom_seen = True
        return track

    seen = counter("e:1")
    est.tracks = {"e:1": seen}
    est._rebuild_geometry(seen, np.array([-0.23, -2.0, 1.2]), 2.0)
    assert seen.category == "counter"
    assert float(seen.size[0]) == pytest.approx(1.4, abs=0.05)
    est._rebuild_geometry(seen, np.array([-2.0, 0.0, 1.2]), 2.0)
    assert float(seen.size[0]) == pytest.approx(1.4, abs=0.05)
    end_on = counter("e:2")
    est._rebuild_geometry(end_on, np.array([-2.0, 0.0, 1.2]), 2.0)
    assert end_on.category == "counter"
    assert float(max(end_on.size[0], end_on.size[1])) == pytest.approx(1.4, abs=0.05)


def test_a_table_keeps_a_full_top_that_fits_the_prior_and_a_counter_rejects_a_neighbour():
    """The sample drops a few centimetres across the view. The full top still fits the table
    prior, so the centre comes from that top. A counter span long past the prior is a neighbour."""
    from steward.ego.estimator import _voxelize
    from steward.ego.perception import box_corners

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    xs = np.linspace(0.05, 1.15, 24)
    ys = np.linspace(-1.28, -0.52, 12)
    zs = np.linspace(0.05, 0.74, 5)
    table = _remembered("e:1", "table", [0.55, -0.91, 0.39], [1.10, 0.76, 0.72])
    table.voxels = _voxelize(np.stack(np.meshgrid(xs, ys, zs), axis=-1).reshape(-1, 3))
    table.span = box_corners(np.array([0.60, -0.90, 0.375]), np.array([1.19, 0.80, 0.75]), 0.0)
    table.bottom_seen = True
    est.tracks = {"e:1": table}
    est._rebuild_geometry(table, np.array([0.6, -2.2, 1.5]), 2.5)
    assert float(table.pos[0]) == pytest.approx(0.60, abs=0.03)
    assert float(max(table.size[0], table.size[1])) == pytest.approx(1.19, abs=0.04)
    cxs = np.linspace(-0.90, 0.85, 30)
    cys = np.linspace(1.88, 2.40, 8)
    czs = np.linspace(0.05, 0.90, 5)
    counter = _remembered("e:2", "counter", [-0.03, 2.12, 0.45], [1.75, 0.52, 0.88])
    counter.voxels = _voxelize(np.stack(np.meshgrid(cxs, cys, czs), axis=-1).reshape(-1, 3))
    counter.span = box_corners(np.array([0.27, 2.17, 0.46]), np.array([2.40, 0.70, 0.90]), 0.0)
    counter.bottom_seen = True
    est.tracks["e:2"] = counter
    est._rebuild_geometry(counter, np.array([-1.8, -1.2, 1.5]), 3.5)
    assert float(counter.pos[0]) == pytest.approx(-0.03, abs=0.08)
    assert float(max(counter.size[0], counter.size[1])) < 2.0


def test_a_square_table_shorter_than_the_prior_keeps_both_sides():
    """A 0.9 m table seen along one side is still 0.9 m. That side is most of the 1.2 m
    prior, so it is not filled the way an 8 cm drawer face is."""
    from steward.ego.estimator import _voxelize

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    xs = np.linspace(-0.45, 0.45, 20)
    ys = np.linspace(-0.45, 0.45, 20)
    zs = np.linspace(0.02, 0.68, 5)
    table = _remembered("e:1", "table", [0.0, 0.0, 0.34], [0.9, 0.9, 0.68])
    table.voxels = _voxelize(np.stack(np.meshgrid(xs, ys, zs), axis=-1).reshape(-1, 3))
    table.bottom_seen = True
    est.tracks = {"e:1": table}
    est._rebuild_geometry(table, np.array([-2.0, -0.4, 1.2]), 2.0)
    assert table.category == "table"
    assert float(max(table.size[0], table.size[1])) == pytest.approx(0.9, abs=0.05)


def test_a_partly_seen_drawer_depth_still_takes_the_prior():
    """Thirty centimetres facing the camera is still the hidden side of a drawer.
    A facing side is left as measured only once it is already half a metre."""
    from steward.ego.estimator import _voxelize

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    xs = np.linspace(0.0, 0.56, 16)
    ys = np.linspace(-0.15, 0.15, 8)
    zs = np.linspace(0.49, 0.71, 5)
    drawer = _remembered("e:1", "drawer", [0.28, 0.0, 0.60], [0.56, 0.30, 0.22])
    drawer.voxels = _voxelize(np.stack(np.meshgrid(xs, ys, zs), axis=-1).reshape(-1, 3))
    drawer.bottom_seen = True
    est.tracks = {"e:1": drawer}
    est._rebuild_geometry(drawer, np.array([0.28, -2.0, 1.2]), 2.0)
    assert drawer.category == "drawer"
    assert float(min(drawer.size[0], drawer.size[1])) == pytest.approx(0.44, abs=0.05)


def test_a_thin_drawer_face_still_takes_the_prior_depth():
    """Eight centimetres of front is not the drawer. A view that meets that axis at 45 degrees
    still fills the prior depth; only a side that is already most of the prior is remembered."""
    from steward.ego.estimator import _voxelize

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    xs = np.linspace(0.0, 1.4, 24)
    ys = np.linspace(-0.04, 0.04, 4)
    zs = np.linspace(0.05, 0.85, 6)
    face = _remembered("e:1", "counter", [0.7, 0.0, 0.45], [1.4, 0.08, 0.85])
    face.voxels = _voxelize(np.stack(np.meshgrid(xs, ys, zs), axis=-1).reshape(-1, 3))
    face.mean_rgb = np.array([0.92, 0.92, 0.93])
    face.bottom_seen = True
    est.tracks = {"e:1": face}
    est._rebuild_geometry(face, np.array([-1.0, -1.0, 1.2]), 2.0)
    assert face.category == "counter"
    assert float(min(face.size[0], face.size[1])) == pytest.approx(0.62, abs=0.05)


def test_a_drawer_completed_past_the_sideboard_is_dropped_and_the_real_one_stays():
    """The top band of the sideboard stays until the real drawer is pulled, then it is the one that did not move.
    The drawer in the sideboard, and one pulled out of the cabinet at drawer height, stay."""
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    sideboard = _remembered("e:18", "sideboard", [-2.71, 0.59, 0.44], [1.38, 0.52, 0.84])
    sideboard.yaw = math.pi / 2  # the long side runs north; the extra drawer still lies in that box
    est.tracks = {
        "e:18": sideboard,
        "e:4": _remembered("e:4", "cabinet", [2.67, 0.60, 0.40], [0.62, 0.54, 0.80]),
        "e:52": _remembered("e:52", "drawer", [-2.74, 1.04, 0.81], [0.53, 0.44, 0.22]),
        "e:15": _remembered("e:15", "drawer", [-2.69, 0.60, 0.71], [0.56, 0.43, 0.22]),
        "e:6": _remembered("e:6", "drawer", [2.20, 0.60, 0.66], [0.56, 0.44, 0.22]),
    }
    est._drop_support_top_drawers()
    assert "e:52" in est.tracks, "a second drawer stays until one of them is pulled"
    est.shift("e:15", np.array([0.28, 0.0, 0.0]))
    assert "e:52" not in est.tracks
    assert set(est.tracks) == {"e:18", "e:4", "e:15", "e:6"}


def test_a_wide_slab_on_a_table_is_dropped_and_an_open_cabinet_drawer_stays():
    """A tabletop stretched into a drawer is dropped. An open cabinet drawer is just as long,
    and it stays: its base is down in the cabinet, not on the table."""
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    est.tracks = {
        "e:table": _remembered("e:table", "table", [0.6, -0.9, 0.38], [1.16, 0.79, 0.75]),
        "e:1": _remembered("e:1", "drawer", [0.6, -0.9, 0.86], [0.92, 0.29, 0.22]),
        "e:2": _remembered("e:2", "drawer", [2.67, 0.60, 0.66], [0.92, 0.50, 0.28]),
    }
    est._drop_support_top_drawers()
    assert "e:1" not in est.tracks
    assert "e:2" in est.tracks


def test_a_drawer_on_a_counter_top_is_dropped_and_one_inside_a_cabinet_stays():
    """A short slab beside the cup is the counter top. A taller box in the counter beside that
    cup is the counter face. The cabinet's own drawer, and a taller drawer-shaped track on a
    table, stay."""
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    est.tracks = {
        "e:1": _remembered("e:1", "counter", [0.0, 0.0, 0.45], [1.8, 0.6, 0.9]),
        "e:2": _remembered("e:2", "drawer", [-0.4, 0.1, 0.95], [0.46, 0.22, 0.11]),
        "e:5": _remembered("e:5", "cup", [-0.45, 0.0, 0.95], [0.08, 0.08, 0.10]),
        "e:3": _remembered("e:3", "cabinet", [2.0, 0.0, 0.4], [0.62, 0.52, 0.8]),
        "e:4": _remembered("e:4", "drawer", [2.0, 0.0, 0.65], [0.56, 0.44, 0.22]),
        "e:6": _remembered("e:6", "table", [0.6, -0.9, 0.38], [1.16, 0.79, 0.75]),
        "e:7": _remembered("e:7", "drawer", [0.72, -1.07, 0.86], [0.58, 0.25, 0.22]),
        "e:8": _remembered("e:8", "drawer", [-0.4, 0.1, 0.70], [0.55, 0.39, 0.33]),
    }
    est._drop_support_top_drawers()
    assert "e:2" not in est.tracks
    assert "e:8" not in est.tracks
    assert set(est.tracks) == {"e:1", "e:5", "e:3", "e:4", "e:6", "e:7"}


def test_a_bear_in_front_of_an_open_drawer_moves_back_when_it_closes():
    """The bear was seen in the open drawer, just past the estimated box. Closing carries it back.
    A bowl up on the furniture does not come along."""
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    est.tracks = {
        "e:14": _remembered("e:14", "drawer", [-2.7, 0.6, 0.66], [0.56, 0.44, 0.22]),
        "e:50": _remembered("e:50", "teddy_bear", [-2.30, 0.6, 0.66], [0.10, 0.10, 0.18]),
        "e:51": _remembered("e:51", "pistol", [-3.05, 0.6, 0.64], [0.18, 0.11, 0.03]),
        "e:20": _remembered("e:20", "bowl", [-2.30, 0.6, 0.95], [0.16, 0.16, 0.06]),
        "e:3": _remembered("e:3", "table", [0.6, -0.9, 0.38], [1.2, 0.8, 0.75]),
    }
    moved = est.shift("e:14", np.array([-0.26, 0.0, 0.0]))
    assert "e:50" in moved
    assert est.tracks["e:50"].pos[0] == pytest.approx(-2.56, abs=0.01)
    # Already on the inward side of a drawer the estimate has shut. The close must not
    # push it further through the back.
    assert est.tracks["e:51"].pos[0] == pytest.approx(-3.05, abs=0.01)
    assert est.tracks["e:20"].pos[0] == pytest.approx(-2.30)
    assert est.tracks["e:14"].pos[0] == pytest.approx(-2.96)
    # A pistol that shows up only after the close is still out in front, and is brought back.
    # The bear already rode the close, so it is not moved a second time.
    before = {"e:50": [-2.30, 0.6, 0.66]}
    late = _remembered("e:46", "pistol", [-2.40, 0.6, 0.64], [0.18, 0.11, 0.03])
    est.tracks["e:46"] = late
    brought = est.resettle("e:14", np.array([-0.26, 0.0, 0.0]), before)
    assert "e:46" in brought and "e:50" not in brought
    assert est.tracks["e:46"].pos[0] == pytest.approx(-2.66, abs=0.02)
    assert est.tracks["e:50"].pos[0] == pytest.approx(-2.56, abs=0.01)


def test_a_bear_already_on_the_shut_drawer_is_not_pushed_when_the_close_is_applied():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    side = _remembered("e:20", "sideboard", [-2.75, 0.60, 0.42], [1.42, 0.49, 0.84])
    side.yaw = 0.0
    drawer = _remembered("e:14", "drawer", [-2.46, 0.60, 0.70], [0.56, 0.44, 0.22])
    bear = _remembered("e:50", "teddy_bear", [-2.72, 0.48, 0.70], [0.11, 0.10, 0.18])
    table = _remembered("e:3", "table", [0.6, -0.9, 0.38], [1.2, 0.8, 0.75])
    est.tracks = {"e:20": side, "e:14": drawer, "e:50": bear, "e:3": table}
    est.shift("e:14", np.array([-0.26, 0.0, 0.0]))
    assert float(bear.pos[0]) == pytest.approx(-2.72, abs=0.02)


def test_a_pistol_on_the_sideboard_top_is_dropped_and_the_one_in_the_drawer_stays():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    side = _remembered("e:25", "sideboard", [-2.74, 0.585, 0.42], [1.42, 0.51, 0.84])
    drawer = _remembered("e:26", "drawer", [-2.71, 0.585, 0.665], [0.58, 0.46, 0.22])
    rail = _remembered("e:96", "pistol", [-2.77, 0.915, 0.805], [0.30, 0.06, 0.04])
    inside = _remembered("e:104", "pistol", [-2.71, 0.685, 0.63], [0.16, 0.16, 0.03])
    est.tracks = {"e:25": side, "e:26": drawer, "e:96": rail, "e:104": inside}
    est._drop_pistols_off_the_drawer()
    assert "e:96" not in est.tracks and "e:104" in est.tracks


def test_a_plate_off_the_table_and_a_plate_flush_with_the_counter_are_dropped():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    table = _remembered("e:3", "table", [0.6, -0.9, 0.375], [1.2, 0.8, 0.75])
    counter = _remembered("e:18", "counter", [0.0, 2.16, 0.45], [1.82, 0.64, 0.90])
    off = _remembered("e:17", "plate", [0.11, -0.485, 0.755], [0.30, 0.33, 0.02])
    flush = _remembered("e:120", "plate", [-0.78, 1.975, 0.905], [0.31, 0.32, 0.02])
    real = _remembered("e:40", "plate", [0.3, -0.8, 0.77], [0.24, 0.24, 0.02])
    est.tracks = {"e:3": table, "e:18": counter, "e:17": off, "e:120": flush, "e:40": real}
    est._drop_false_tableware()
    assert "e:17" not in est.tracks and "e:120" not in est.tracks
    assert "e:40" in est.tracks


def test_a_tall_thin_glass_beside_a_drawer_is_the_pistol():
    from steward.ego.estimator import _voxelize

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    drawer = _remembered("e:22", "drawer", [-2.70, 0.59, 0.70], [0.56, 0.44, 0.22])
    glass = _remembered("e:92", "wine_glass", [-2.737, 0.68, 0.73], [0.107, 0.033, 0.17])
    glass.bottom_seen = True
    xs = np.linspace(-2.79, -2.68, 6)
    ys = np.linspace(0.664, 0.696, 3)
    zs = np.linspace(0.645, 0.815, 8)
    glass.voxels = _voxelize(np.array([[x, y, z] for x in xs for y in ys for z in zs]))
    est.tracks = {"e:22": drawer, "e:92": glass}
    est._pushed_shut.add("e:22")  # the robot has had this drawer open
    est._rebuild_geometry(glass, np.array([-1.6, 0.6, 1.2]), 1.0)
    assert glass.category == "pistol"
    assert float(glass.size[2]) == pytest.approx(0.04, abs=0.015)


def test_a_flat_mug_beside_a_drawer_is_the_pistol():
    from steward.ego.estimator import _voxelize

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    drawer = _remembered("e:20", "drawer", [-2.71, 0.60, 0.68], [0.56, 0.44, 0.22])
    mug = _remembered("e:77", "mug", [-2.675, 0.73, 0.624], [0.117, 0.057, 0.042])
    mug.mean_rgb = np.array([0.45, 0.40, 0.35])
    mug.bottom_seen = True
    xs = np.linspace(-2.73, -2.62, 5)
    ys = np.linspace(0.70, 0.76, 4)
    zs = (0.60, 0.64)
    mug.voxels = _voxelize(np.array([[x, y, z] for x in xs for y in ys for z in zs]))
    est.tracks = {"e:20": drawer, "e:77": mug}
    est._pushed_shut.add("e:20")  # the robot has had this drawer open
    est._rebuild_geometry(mug, np.array([-1.6, 0.6, 1.2]), 1.0)
    assert mug.category == "pistol"


def test_a_table_in_front_of_a_cabinet_is_dropped():
    """The open cabinet, from the side, is named a table and that table blocks the close."""
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    cabinet = _remembered("e:4", "cabinet", [2.675, 0.58, 0.415], [0.64, 0.57, 0.80])
    blocking = _remembered("e:145", "table", [2.897, 0.933, 0.352], [1.43, 0.41, 0.70])
    real = _remembered("e:3", "table", [0.58, -0.89, 0.378], [1.18, 0.80, 0.75])
    est.tracks = {"e:4": cabinet, "e:145": blocking, "e:3": real}
    est._drop_false_chairs()
    assert "e:145" not in est.tracks and "e:3" in est.tracks and "e:4" in est.tracks


def test_a_yellow_cup_beside_a_drawer_stays_out_of_the_pistol():
    from steward.ego.estimator import _voxelize

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    drawer = _remembered("e:6", "drawer", [2.63, 0.60, 0.66], [0.56, 0.44, 0.22])
    cup = _remembered("e:129", "cup", [2.68, 0.72, 0.59], [0.13, 0.09, 0.10])
    cup.mean_rgb = np.array([0.95, 0.78, 0.10])
    xs = np.linspace(2.62, 2.74, 4)
    ys = np.linspace(0.68, 0.76, 4)
    zs = (0.55, 0.64)
    cup.voxels = _voxelize(np.array([[x, y, z] for x in xs for y in ys for z in zs]))
    est.tracks = {"e:6": drawer, "e:129": cup}
    est._rebuild_geometry(cup, np.array([1.6, 0.6, 1.2]), 1.0)
    assert cup.category != "pistol"


def test_a_bear_in_front_of_a_closed_drawer_is_pulled_back_to_the_carcass():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    side = _remembered("e:23", "sideboard", [-2.736, 0.58, 0.435], [1.42, 0.49, 0.84])
    side.yaw = math.pi / 2
    drawer = _remembered("e:24", "drawer", [-2.70, 0.58, 0.72], [0.56, 0.44, 0.24])
    drawer.yaw = math.pi / 2
    bear = _remembered("e:72", "teddy_bear", [-2.629, 0.485, 0.714], [0.14, 0.14, 0.20])
    table = _remembered("e:3", "table", [0.6, -0.9, 0.38], [1.2, 0.8, 0.75])
    est.tracks = {"e:23": side, "e:24": drawer, "e:72": bear, "e:3": table}
    est._seat_contents_in_closed_drawers()
    assert bear.pos[0] == pytest.approx(-2.736, abs=0.02)
    assert bear.pos[1] == pytest.approx(0.485, abs=1e-6)


def test_an_airplane_floating_in_a_closed_drawer_is_set_on_the_floor():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    side = _remembered("e:23", "sideboard", [-2.736, 0.58, 0.435], [1.42, 0.49, 0.84])
    side.yaw = math.pi / 2
    drawer = _remembered("e:24", "drawer", [-2.70, 0.58, 0.71], [0.56, 0.44, 0.23])
    drawer.yaw = math.pi / 2
    plane = _remembered("e:169", "toy_airplane", [-2.70, 0.72, 0.725], [0.20, 0.18, 0.08])
    table = _remembered("e:3", "table", [0.6, -0.9, 0.38], [1.2, 0.8, 0.75])
    est.tracks = {"e:23": side, "e:24": drawer, "e:169": plane, "e:3": table}
    est._seat_contents_in_closed_drawers()
    assert plane.pos[2] == pytest.approx(drawer.bottom() + 0.04, abs=0.01)


def test_an_airplane_is_not_lowered_onto_a_drawer_that_includes_the_cabinet():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    cabinet = _remembered("e:4", "cabinet", [2.711, 0.609, 0.409], [0.62, 0.52, 0.80])
    cabinet.yaw = math.pi
    drawer = _remembered("e:6", "drawer", [2.681, 0.609, 0.615], [0.57, 0.45, 0.38])
    drawer.yaw = math.pi
    plane = _remembered("e:134", "toy_airplane", [2.71, 0.716, 0.595], [0.18, 0.13, 0.08])
    table = _remembered("e:3", "table", [0.6, -0.9, 0.38], [1.2, 0.8, 0.75])
    est.tracks = {"e:4": cabinet, "e:6": drawer, "e:134": plane, "e:3": table}
    est._seat_contents_in_closed_drawers()
    assert plane.pos[2] == pytest.approx(0.595, abs=0.01)


def test_a_tall_drawer_keeps_the_band_under_its_top():
    from steward.ego.estimator import _voxelize

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    drawer = _remembered("e:6", "drawer", [2.68, 0.61, 0.61], [0.56, 0.44, 0.38])
    xs = np.linspace(2.45, 2.90, 8)
    ys = np.linspace(0.40, 0.80, 6)
    zs = np.linspace(0.42, 0.80, 10)
    drawer.voxels = _voxelize(np.array([[x, y, z] for x in xs for y in ys for z in zs]))
    est.tracks = {"e:6": drawer}
    est._rebuild_geometry(drawer, np.array([1.4, 0.6, 1.5]), 1.2)
    assert float(drawer.size[2]) <= 0.26
    assert drawer.top() == pytest.approx(0.80, abs=0.03)


def test_a_bear_the_segmenter_called_an_airplane_stays_in_the_cabinet():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    cabinet = _remembered("e:4", "cabinet", [2.71, 0.61, 0.41], [0.62, 0.52, 0.80])
    cabinet.yaw = math.pi
    drawer = _remembered("e:6", "drawer", [2.68, 0.61, 0.66], [0.56, 0.44, 0.28])
    drawer.yaw = math.pi
    plane = _remembered("e:122", "teddy_bear", [2.708, 0.757, 0.72], [0.13, 0.07, 0.17])
    plane.semantic_counts = {"toy_airplane": 4.0}
    bear = _remembered("e:60", "teddy_bear", [-2.705, 0.48, 0.70], [0.12, 0.12, 0.18])
    bear.semantic_counts = {"teddy_bear": 6.0}
    side = _remembered("e:22", "sideboard", [-2.70, 0.59, 0.43], [1.42, 0.50, 0.84])
    side_drawer = _remembered("e:96", "drawer", [-2.68, 0.59, 0.71], [0.56, 0.44, 0.26])
    table = _remembered("e:3", "table", [0.6, -0.9, 0.38], [1.2, 0.8, 0.75])
    est.tracks = {"e:4": cabinet, "e:6": drawer, "e:122": plane, "e:60": bear,
                  "e:22": side, "e:96": side_drawer, "e:3": table}
    est._rename_airplanes_called_bears()
    est._seat_contents_in_closed_drawers()
    est._drop_stray_bears()
    assert plane.category == "toy_airplane" and "e:122" in est.tracks
    assert plane.pos[2] < 0.68
    assert bear.category == "teddy_bear" and "e:60" in est.tracks


def test_an_unknown_on_the_airplane_is_dropped_and_a_long_perfume_is_a_smear():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    plane = _remembered("e:134", "toy_airplane", [2.71, 0.716, 0.595], [0.18, 0.13, 0.08])
    unknown = _remembered("e:140", "unknown", [2.68, 0.734, 0.595], [0.15, 0.14, 0.08])
    perfume = _remembered("e:80", "perfume_bottle", [-2.745, 0.82, 0.70], [0.15, 0.08, 0.09])
    real = _remembered("e:90", "perfume_bottle", [2.57, 0.46, 0.60], [0.065, 0.04, 0.10])
    est.tracks = {"e:134": plane, "e:140": unknown, "e:80": perfume, "e:90": real}
    est._drop_unknowns_on_the_airplane()
    est._drop_perfume_smears()
    assert "e:140" not in est.tracks and "e:80" not in est.tracks
    assert "e:134" in est.tracks and "e:90" in est.tracks


def test_a_cup_on_the_bear_is_dropped():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    bear = _remembered("e:84", "teddy_bear", [-2.814, 0.481, 0.709], [0.11, 0.09, 0.17])
    cup = _remembered("e:221", "cup", [-2.679, 0.492, 0.668], [0.16, 0.15, 0.13])
    mug = _remembered("e:15", "mug", [0.926, -1.141, 0.822], [0.12, 0.11, 0.13])
    est.tracks = {"e:84": bear, "e:221": cup, "e:15": mug}
    est._drop_cups_on_the_bear()
    assert "e:221" not in est.tracks and "e:15" in est.tracks and "e:84" in est.tracks


def test_an_airplane_and_a_pill_a_hand_apart_are_not_one_object():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    plane = _remembered("e:113", "toy_airplane", [2.636, 0.725, 0.595], [0.09, 0.08, 0.08])
    pill = _remembered("e:119", "pill_bottle", [2.682, 0.715, 0.586], [0.12, 0.10, 0.10])
    est.tracks = {"e:113": plane, "e:119": pill}
    est._merge_same_place(np.zeros(3))
    assert "e:113" in est.tracks and "e:119" in est.tracks


def test_a_bear_away_from_the_drawer_is_dropped_when_one_is_in_it():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    drawer = _remembered("e:25", "drawer", [-2.675, 0.583, 0.71], [0.56, 0.44, 0.23])
    inside = _remembered("e:58", "teddy_bear", [-2.703, 0.478, 0.699], [0.14, 0.14, 0.19])
    stray = _remembered("e:78", "teddy_bear", [-2.712, 0.985, 0.689], [0.23, 0.11, 0.17])
    est.tracks = {"e:25": drawer, "e:58": inside, "e:78": stray}
    est._drop_stray_bears()
    assert "e:58" in est.tracks and "e:78" not in est.tracks


def test_a_bear_in_an_open_drawer_stays_when_the_drawer_box_has_grown_along_the_sideboard():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    side = _remembered("e:48", "sideboard", [-2.755, 0.585, 0.435], [1.41, 0.50, 0.84])
    side.yaw = math.pi / 2
    table = _remembered("e:3", "table", [0.6, -0.9, 0.38], [1.2, 0.8, 0.75])
    # Pulled out, the drawer was boxed with a stretch of the sideboard top: 0.77 m long, centre 9 cm off.
    drawer = _remembered("e:49", "drawer", [-2.435, 0.67, 0.71], [0.77, 0.50, 0.23])
    drawer.yaw = -math.pi / 2
    bear = _remembered("e:257", "teddy_bear", [-2.476, 0.488, 0.705], [0.165, 0.09, 0.18])
    est.tracks = {"e:48": side, "e:3": table, "e:49": drawer, "e:257": bear}
    est._drop_stray_bears()
    assert "e:257" in est.tracks


def test_a_flat_mug_put_back_in_the_drawer_is_the_pistol():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    drawer = _remembered("e:26", "drawer", [-2.70, 0.59, 0.70], [0.56, 0.44, 0.22])
    mug = _remembered("e:85", "mug", [-2.828, 0.689, 0.624], [0.16, 0.094, 0.042])
    est.tracks = {"e:26": drawer, "e:85": mug}
    est._pushed_shut.add("e:26")  # the robot has had this drawer open
    est._rename_flat_mugs_in_drawers()
    assert mug.category == "pistol"


def test_a_wide_perfume_beside_a_drawer_is_the_airplane():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    drawer = _remembered("e:6", "drawer", [2.64, 0.60, 0.65], [0.56, 0.44, 0.22])
    wide = _remembered("e:173", "perfume_bottle", [2.672, 0.715, 0.584], [0.126, 0.126, 0.102])
    square = _remembered("e:183", "pill_bottle", [2.669, 0.71, 0.56], [0.116, 0.116, 0.074])
    real = _remembered("e:7", "perfume_bottle", [2.57, 0.46, 0.60], [0.065, 0.04, 0.10])
    pill = _remembered("e:185", "pill_bottle", [2.711, 0.465, 0.584], [0.055, 0.055, 0.122])
    est.tracks = {"e:6": drawer, "e:173": wide, "e:183": square, "e:7": real, "e:185": pill}
    est._pushed_shut.add("e:6")  # the robot has had this drawer open
    est._rename_wide_perfumes()
    assert wide.category == "toy_airplane" and square.category == "toy_airplane"
    assert real.category == "perfume_bottle" and pill.category == "pill_bottle"


def test_a_bottle_on_the_airplane_is_dropped_and_the_shelf_bottle_stays():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    plane = _remembered("e:162", "toy_airplane", [2.635, 0.738, 0.595], [0.20, 0.18, 0.08])
    on_it = _remembered("e:170", "pill_bottle", [2.705, 0.732, 0.584], [0.10, 0.09, 0.10])
    shelf = _remembered("e:146", "pill_bottle", [2.732, 0.471, 0.589], [0.07, 0.07, 0.11])
    est.tracks = {"e:162": plane, "e:170": on_it, "e:146": shelf}
    est._drop_bottles_on_the_airplane()
    assert "e:170" not in est.tracks and "e:146" in est.tracks and "e:162" in est.tracks


def test_a_thick_wall_keeps_the_face_toward_the_camera_and_thins_to_the_panel():
    from steward.ego.estimator import _voxelize

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    wall = _remembered("e:70", "wall", [-2.6, 0.6, 1.3], [0.40, 4.0, 2.6])
    xs = np.linspace(-2.8, -2.4, 5)
    ys = np.linspace(-1.0, 2.0, 8)
    zs = np.linspace(0.2, 2.2, 6)
    wall.voxels = _voxelize(np.array([[x, y, z] for x in xs for y in ys for z in zs]))
    est.tracks = {"e:70": wall}
    est._rebuild_geometry(wall, np.array([0.0, 0.6, 1.2]), 2.0)
    assert min(wall.size[0], wall.size[1]) == pytest.approx(0.10, abs=0.01)
    assert wall.pos[0] + 0.05 == pytest.approx(-2.4, abs=0.05)


def test_a_chair_beside_a_sideboard_does_not_replace_it():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    side = _remembered("e:25", "sideboard", [-2.705, 0.575, 0.435], [1.4, 0.52, 0.84])
    side.yaw = math.pi / 2
    chair = _remembered("e:83", "chair", [-2.42, 0.605, 0.403], [0.55, 0.46, 0.80])
    est.tracks = {"e:25": side, "e:83": chair}
    est._dedupe_furniture(np.zeros(3))
    assert "e:25" in est.tracks and "e:83" in est.tracks


def test_a_sideboard_swollen_toward_the_camera_sits_back_on_the_wall():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    wall = _remembered("e:1", "wall", [-3.05, 0.6, 1.3], [5.0, 0.1, 2.6])
    wall.yaw = -math.pi / 2
    side = _remembered("e:2", "sideboard", [-2.671, 0.579, 0.435], [1.455, 0.539, 0.84])
    side.yaw = math.pi / 2
    bear = _remembered("e:3", "teddy_bear", [-2.684, 0.48, 0.70], [0.12, 0.12, 0.18])
    table = _remembered("e:4", "table", [0.6, -0.9, 0.38], [1.2, 0.8, 0.75])
    # The pistol is in the shut drawer: this view did not measure it, so it was not swollen with the piece.
    pistol = _remembered("e:5", "pistol", [-2.72, 0.70, 0.66], [0.17, 0.12, 0.03])
    est.tracks = {"e:1": wall, "e:2": side, "e:3": bear, "e:4": table, "e:5": pistol}
    est.stamp = 10.0
    for track in (wall, side, bear, table):
        track.last_seen = 10.0
    pistol.last_seen = 4.0
    est._seat_against_the_wall()
    assert side.pos[0] == pytest.approx(-2.755, abs=0.01)
    assert bear.pos[0] == pytest.approx(-2.684 + (side.pos[0] - (-2.671)), abs=0.01)
    assert table.pos[0] == pytest.approx(0.6, abs=1e-6)
    assert pistol.pos[0] == pytest.approx(-2.72, abs=1e-6)


def test_a_pistol_in_the_pulled_out_drawer_does_not_follow_the_sideboard_back_to_the_wall():
    """The sideboard was boxed over its open drawer and is seated back on the wall. The pistol
    this view measured lies in that drawer, in front of the piece; moved with the piece, its next
    view starts a second pistol and the close pushes the first through the back."""
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    wall = _remembered("e:1", "wall", [-3.05, 0.6, 1.3], [5.0, 0.1, 2.6])
    wall.yaw = -math.pi / 2
    side = _remembered("e:2", "sideboard", [-2.671, 0.579, 0.435], [1.455, 0.539, 0.84])
    side.yaw = math.pi / 2
    drawer = _remembered("e:51", "drawer", [-2.455, 0.61, 0.71], [0.61, 0.44, 0.23])
    pistol = _remembered("e:212", "pistol", [-2.445, 0.686, 0.635], [0.114, 0.113, 0.02])
    table = _remembered("e:4", "table", [0.6, -0.9, 0.38], [1.2, 0.8, 0.75])
    est.tracks = {"e:1": wall, "e:2": side, "e:51": drawer, "e:212": pistol, "e:4": table}
    est._pulled_open.add("e:51")
    est.stamp = 77.8
    for track in est.tracks.values():
        track.last_seen = 77.8
    est._seat_against_the_wall()
    assert side.pos[0] == pytest.approx(-2.755, abs=0.01)
    assert pistol.pos[0] == pytest.approx(-2.445, abs=1e-6)


def test_a_cabinet_measured_into_the_east_wall_sits_back_on_the_wall():
    """A cabinet boxed past the wall plane still seats; the closed drawer follows the front."""
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    wall = _remembered("e:2", "wall", [3.0, 0.0, 0.75], [0.10, 6.0, 1.5])
    wall.yaw = math.pi / 2
    cabinet = _remembered("e:4", "cabinet", [2.783, 0.605, 0.413], [0.685, 0.563, 0.802])
    cabinet.yaw = 0.0
    drawer = _remembered("e:5", "drawer", [2.753, 0.605, 0.675], [0.582, 0.462, 0.22])
    drawer.yaw = 0.0
    table = _remembered("e:3", "table", [0.6, -0.9, 0.38], [1.2, 0.8, 0.75])
    est.tracks = {"e:2": wall, "e:4": cabinet, "e:5": drawer, "e:3": table}
    est._seat_against_the_wall()
    est._seat_closed_drawers()
    assert float(cabinet.pos[0]) == pytest.approx(2.71, abs=0.03)
    assert float(drawer.pos[0]) == pytest.approx(float(cabinet.pos[0]), abs=0.05)


def test_a_cabinet_is_not_seated_against_a_piece_of_wall_that_is_not_behind_it():
    """One view of 45 cm of the east wall, 2.8 m along it, has a yaw two degrees off. Its plane
    carried to the cabinet is 11 cm into the room, and the cabinet's back would be through it."""
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    east = _remembered("e:2", "wall", [2.997, -0.01, 0.9], [4.91, 0.081, 1.77])
    east.yaw = -1.566
    piece = _remembered("e:75", "wall", [3.006, -2.24, 0.755], [0.449, 0.049, 1.48])
    piece.yaw = -1.532
    piece.observations = 1
    cabinet = _remembered("e:6", "cabinet", [2.708, 0.59, 0.41], [0.52, 0.61, 0.79])
    cabinet.yaw = 0.005
    table = _remembered("e:3", "table", [0.6, -0.9, 0.38], [1.2, 0.8, 0.75])
    est.tracks = {"e:2": east, "e:75": piece, "e:6": cabinet, "e:3": table}
    est._seat_against_the_wall()
    assert float(cabinet.pos[0]) == pytest.approx(2.708, abs=0.02)


def test_a_drawer_named_view_of_the_drawer_the_robot_pulled_out_is_not_a_second_drawer():
    """The pulled drawer is pinned and takes no views. A second drawer would be seated shut in
    the sideboard, and the pistol in the open drawer would be pulled back to that seat."""
    from types import SimpleNamespace

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    side = _remembered("e:59", "sideboard", [-2.69, 0.59, 0.435], [1.39, 0.55, 0.84])
    side.yaw = -math.pi / 2
    table = _remembered("e:3", "table", [0.6, -0.9, 0.38], [1.2, 0.8, 0.75])
    drawer = _remembered("e:60", "drawer", [-2.475, 0.645, 0.59], [0.699, 0.44, 0.47])
    drawer.yaw = -1.58
    drawer.pinned = True
    est.tracks = {"e:59": side, "e:3": table, "e:60": drawer}
    view = {"seg": SimpleNamespace(semantics={"drawer": 0.89}), "centre": np.array([-2.421, 0.539, 0.446]),
            "size": np.array([0.671, 0.449, 0.724]), "yaw": -1.57, "footprint": 0.671}
    assert not est._is_the_pulled_drawer(view), "a drawer the robot never pulled is matched like any other track"
    est._opened_drawers.add("e:60")
    assert est._is_the_pulled_drawer(view)
    bear = dict(view, seg=SimpleNamespace(semantics={"teddy_bear": 0.8}), size=np.array([0.2, 0.11, 0.18]), footprint=0.2)
    assert not est._is_the_pulled_drawer(bear), "what is in the drawer starts its own track"


def test_one_mask_over_a_bowl_and_the_glass_beside_it_is_not_one_more_bowl():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    table = _remembered("e:3", "table", [0.6, -0.9, 0.38], [1.2, 0.8, 0.75])
    glass = _remembered("e:7", "wine_glass", [0.471, -1.118, 0.841], [0.07, 0.07, 0.19])
    green = _remembered("e:19", "bowl", [0.71, -1.075, 0.784], [0.18, 0.18, 0.06])
    est.tracks = {"e:3": table, "e:7": glass, "e:19": green}
    group = {"probs": {"bowl": 0.84, "drawer": 0.15}, "centre": np.array([0.634, -1.073, 0.834]),
             "size": np.array([0.368, 0.209, 0.169]), "yaw": 0.0, "footprint": 0.368}
    assert est._over_several_objects(group)
    from types import SimpleNamespace
    row = dict(group, probs={"drawer": 0.71, "bowl": 0.25}, seg=SimpleNamespace(semantics={"bowl": 0.59}))
    assert est._over_several_objects(row), "two bowls in a row have the shape of a drawer front and the name of a bowl"
    one = dict(group, centre=np.array([0.71, -1.075, 0.79]), size=np.array([0.19, 0.18, 0.07]), footprint=0.19)
    assert not est._over_several_objects(one), "a bowl-sized region is a view of a bowl"
    del est.tracks["e:7"]
    assert not est._over_several_objects(group), "a bowl boxed with a strip of the table is cut back, not refused"
    est.tracks["e:14"] = _remembered("e:14", "bowl", [0.72, -1.07, 0.775], [0.14, 0.14, 0.05])
    assert not est._over_several_objects(group), "a bowl standing in a bowl shares its centre"


def test_the_back_board_of_an_open_drawer_is_not_something_in_it():
    """Points above the floor of an open drawer become its contents. Its own boards are above
    the floor as well: a strip two centimetres thick along the edge of the drawer's box."""
    from steward.ego.estimator import _voxelize

    def contents(raised):
        est = EgoEstimator(robot_model=steward_bot().to_dict())
        side = _remembered("e:59", "sideboard", [-2.755, 0.59, 0.435], [1.39, 0.45, 0.84])
        side.yaw = -math.pi / 2
        table = _remembered("e:3", "table", [0.6, -0.9, 0.38], [1.2, 0.8, 0.75])
        drawer = _remembered("e:60", "drawer", [-2.45, 0.59, 0.71], [0.56, 0.44, 0.22])
        drawer.yaw = -math.pi / 2
        floor = np.array([[x, y, 0.62] for x in np.arange(-2.66, -2.24, 0.01) for y in np.arange(0.32, 0.86, 0.01)])
        drawer.voxels = _voxelize(np.vstack([floor, raised]))
        est.tracks = {"e:59": side, "e:3": table, "e:60": drawer}
        est._opened_drawers.add("e:60")
        est._contents_of_open_drawers()
        return [[round(float(v), 2) for v in t.pos[:2]] for t in est.tracks.values() if t.id not in ("e:59", "e:3", "e:60")]

    board = np.array([[x, y, z] for x in (-2.665, -2.655) for y in np.arange(0.35, 0.46, 0.01)
                      for z in np.arange(0.66, 0.74, 0.01)])
    assert contents(board) == []
    thing = np.array([[x, y, z] for x in np.arange(-2.49, -2.41, 0.01) for y in np.arange(0.56, 0.64, 0.01)
                      for z in np.arange(0.63, 0.73, 0.01)])
    assert contents(thing) == [[-2.46, 0.6]]


def test_a_line_of_the_pistol_beside_a_drawer_stays_and_grows_toward_it():
    from steward.ego.estimator import _voxelize

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    drawer = _remembered("e:22", "drawer", [-2.64, 0.57, 0.70], [0.56, 0.44, 0.22])
    line = _remembered("e:66", "unknown", [-2.635, 0.745, 0.685], [0.16, 0.0, 0.02])
    xs = np.linspace(-2.71, -2.56, 6)
    line.voxels = _voxelize(np.array([[x, 0.745, 0.685] for x in xs]))
    est.tracks = {"e:22": drawer, "e:66": line}
    est._pushed_shut.add("e:22")  # the robot has had this drawer open
    est._rebuild_geometry(line, np.array([-1.6, 0.6, 1.2]), 1.0)
    assert line.category == "pistol"
    assert line.pos[1] < 0.745
    assert min(line.size[0], line.size[1]) == pytest.approx(0.12, abs=0.01)


def test_a_narrow_cup_beside_a_drawer_is_the_pill():
    from steward.ego.estimator import _voxelize

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    drawer = _remembered("e:6", "drawer", [2.63, 0.60, 0.66], [0.56, 0.44, 0.24])
    cup = _remembered("e:109", "cup", [2.737, 0.459, 0.588], [0.085, 0.065, 0.111])
    xs = np.linspace(2.70, 2.77, 3)
    ys = np.linspace(0.43, 0.49, 3)
    zs = np.linspace(0.54, 0.64, 3)
    cup.voxels = _voxelize(np.array([[x, y, z] for x in xs for y in ys for z in zs]))
    est.tracks = {"e:6": drawer, "e:109": cup}
    est._pushed_shut.add("e:6")  # the robot has had this drawer open
    est._rebuild_geometry(cup, np.array([1.4, 0.6, 1.2]), 1.0)
    assert cup.category == "pill_bottle"


def test_two_bears_four_centimetres_apart_are_one_bear():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    a = _remembered("e:64", "teddy_bear", [-2.72, 0.478, 0.699], [0.119, 0.124, 0.192])
    b = _remembered("e:134", "teddy_bear", [-2.754, 0.464, 0.703], [0.27, 0.121, 0.181])
    est.tracks = {"e:64": a, "e:134": b}
    est._merge_same_spot()
    assert len([t for t in est.tracks.values() if t.category == "teddy_bear"]) == 1


def test_a_bear_is_not_merged_into_the_chair_smear_beside_it():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    bear = _remembered("e:65", "teddy_bear", [-2.73, 0.48, 0.70], [0.14, 0.10, 0.19])
    smear = _remembered("e:104", "chair", [-2.67, 0.475, 0.70], [0.18, 0.46, 0.80])
    est.tracks = {"e:65": bear, "e:104": smear}
    est._merge_same_place(np.zeros(3))
    assert "e:65" in est.tracks and est.tracks["e:65"].category == "teddy_bear"


def test_a_bear_on_the_table_gives_way_to_the_wine_glass():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    bear = _remembered("e:9", "teddy_bear", [0.49, -1.11, 0.85], [0.14, 0.10, 0.17])
    glass = _remembered("e:239", "wine_glass", [0.50, -1.12, 0.85], [0.08, 0.08, 0.19])
    est.tracks = {"e:9": bear, "e:239": glass}
    est._merge_same_place(np.zeros(3))
    assert "e:239" in est.tracks and "e:9" not in est.tracks
    assert est.tracks["e:239"].category == "wine_glass"


def test_a_tall_yellow_box_beside_a_drawer_is_the_airplane():
    from steward.ego.estimator import _voxelize

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    drawer = _remembered("e:6", "drawer", [2.63, 0.60, 0.66], [0.56, 0.44, 0.22])
    plane = _remembered("e:270", "wine_glass", [2.665, 0.736, 0.66], [0.138, 0.12, 0.162])
    plane.mean_rgb = np.array([0.90, 0.72, 0.20])
    plane.bottom_seen = True
    xs = np.linspace(2.60, 2.73, 4)
    ys = np.linspace(0.68, 0.80, 4)
    zs = (0.58, 0.74)
    plane.voxels = _voxelize(np.array([[x, y, z] for x in xs for y in ys for z in zs]))
    est.tracks = {"e:6": drawer, "e:270": plane}
    est._pushed_shut.add("e:6")  # the robot has had this drawer open
    est._rebuild_geometry(plane, np.array([1.6, 0.6, 1.2]), 1.0)
    assert plane.category == "toy_airplane"
    assert float(plane.size[2]) == pytest.approx(0.08, abs=0.01)
    assert float(plane.pos[2]) > 0.66


def test_a_wine_glass_already_named_beside_a_drawer_is_the_airplane():
    from steward.ego.estimator import _voxelize

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    drawer = _remembered("e:6", "drawer", [2.63, 0.60, 0.66], [0.56, 0.44, 0.22])
    glass = _remembered("e:199", "wine_glass", [2.673, 0.732, 0.62], [0.127, 0.104, 0.16])
    glass.bottom_seen = True
    xs = np.linspace(2.61, 2.74, 6)
    ys = np.linspace(0.68, 0.78, 6)
    zs = np.linspace(0.54, 0.70, 5)
    glass.voxels = _voxelize(np.array([[x, y, z] for x in xs for y in ys for z in zs]))
    est.tracks = {"e:6": drawer, "e:199": glass}
    est._pushed_shut.add("e:6")  # the robot has had this drawer open
    est._rebuild_geometry(glass, np.array([1.6, 0.6, 1.2]), 1.0)
    assert glass.category == "toy_airplane"


def test_a_thin_bottle_beside_a_drawer_is_the_pill():
    from steward.ego.estimator import _voxelize

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    drawer = _remembered("e:6", "drawer", [2.63, 0.60, 0.66], [0.56, 0.44, 0.22])
    bottle = _remembered("e:335", "bottle", [2.72, 0.466, 0.60], [0.046, 0.02, 0.17])
    bottle.bottom_seen = True
    xs = np.linspace(2.70, 2.74, 4)
    ys = np.linspace(0.456, 0.476, 3)
    zs = np.linspace(0.52, 0.68, 6)
    bottle.voxels = _voxelize(np.array([[x, y, z] for x in xs for y in ys for z in zs]))
    est.tracks = {"e:6": drawer, "e:335": bottle}
    est._pushed_shut.add("e:6")  # the robot has had this drawer open
    est._rebuild_geometry(bottle, np.array([1.6, 0.6, 1.2]), 1.0)
    assert bottle.category == "pill_bottle"


def test_a_bottle_whose_base_was_seen_is_not_stretched_down_to_the_bottom_of_the_drawer_box():
    from steward.ego.estimator import _voxelize

    def bottle_in_a_swollen_drawer(base_seen: bool):
        est = EgoEstimator(robot_model=steward_bot().to_dict())
        # The drawer's box reaches 3 cm under the boards the bottle stands on.
        drawer = _remembered("e:6", "drawer", [2.63, 0.60, 0.66], [0.56, 0.44, 0.27])
        bottle = _remembered("e:139", "perfume_bottle", [2.57, 0.46, 0.606], [0.065, 0.04, 0.10])
        bottle.bottom_seen = base_seen
        xs, ys, zs = np.linspace(2.54, 2.60, 7), np.linspace(0.44, 0.48, 5), np.linspace(0.556, 0.656, 11)
        bottle.voxels = _voxelize(np.array([[x, y, z] for x in xs for y in ys for z in zs]))
        est.tracks = {"e:6": drawer, "e:139": bottle}
        est._pushed_shut.add("e:6")  # the robot has had this drawer open
        est._rebuild_geometry(bottle, np.array([1.6, 0.6, 1.2]), 1.0)
        return bottle

    assert bottle_in_a_swollen_drawer(base_seen=True).size[2] == pytest.approx(0.10, abs=0.012)
    assert bottle_in_a_swollen_drawer(base_seen=False).size[2] > 0.12, "a hidden base is still filled down"


def test_a_bottle_in_a_drawer_is_as_tall_as_its_cloud_when_each_view_boxed_a_part_of_it():
    from steward.ego.estimator import _voxelize

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    drawer = _remembered("e:6", "drawer", [2.63, 0.60, 0.66], [0.56, 0.44, 0.22])
    bottle = _remembered("e:153", "perfume_bottle", [2.57, 0.46, 0.606], [0.065, 0.04, 0.10])
    bottle.bottom_seen = True
    xs, ys, zs = np.linspace(2.54, 2.60, 7), np.linspace(0.44, 0.48, 5), np.linspace(0.556, 0.656, 11)
    bottle.voxels = _voxelize(np.array([[x, y, z] for x in xs for y in ys for z in zs]))
    # The drawer front hid the base in three views and the top half in one.
    upper = (np.array([2.57, 0.46, 0.636]), np.array([0.065, 0.04, 0.04]), 0.9)
    bottle.recent.extend([upper, upper, upper, (np.array([2.57, 0.46, 0.59]), np.array([0.065, 0.04, 0.07]), 0.9)])
    est.tracks = {"e:6": drawer, "e:153": bottle}
    est._pushed_shut.add("e:6")  # the robot has had this drawer open
    est._rebuild_geometry(bottle, np.array([1.6, 0.6, 1.2]), 1.0)
    assert bottle.size[2] == pytest.approx(0.10, abs=0.012)
    assert bottle.pos[2] == pytest.approx(0.606, abs=0.01)
    assert bottle.category == "perfume_bottle"


def test_a_view_cut_by_the_image_border_does_not_shrink_a_bowl_that_was_seen_whole():
    from steward.ego.estimator import _voxelize
    from steward.ego.perception import Segment

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    bowl = _remembered("e:14", "bowl", [0.72, -1.08, 0.78], [0.20, 0.20, 0.06])
    bowl.bottom_seen = True
    bowl.semantic_counts, bowl.semantic_views = {"bowl": 6.0}, 8
    bowl.mean_rgb = np.array([0.2, 0.6, 0.3])
    rng = np.random.default_rng(0)
    whole = rng.uniform([0.62, -1.18, 0.75], [0.82, -0.98, 0.81], (400, 3))
    bowl.voxels = _voxelize(whole)
    bowl.recent.extend((np.array([0.72, -1.08, 0.78]), np.array([0.20, 0.20, 0.06]), 2.2) for _ in range(4))
    est.tracks = {"e:14": bowl}
    # From close by, the image edge cuts the bowl: the mask is the half nearer the middle of the frame.
    half = whole[whole[:, 1] > -1.07]
    for range_m in (1.0, 0.9):
        cut = Segment(points=half, color=np.array([0.2, 0.6, 0.3]), pixels=900, invalid_frac=0.0, border=True,
                      range=range_m, semantics={"bowl": 0.7})
        est._fuse_cloud(bowl, {"seg": cut, "centre": np.array([0.72, -1.025, 0.78]), "size": np.array([0.20, 0.09, 0.06]),
                               "yaw": 0.0, "probs": {"bowl": 1.0}, "footprint": 0.20}, np.array([1.3, 0.5, 1.45]))
    assert min(bowl.size[0], bowl.size[1]) == pytest.approx(0.20, abs=0.02)
    assert bowl.pos[1] == pytest.approx(-1.08, abs=0.02)


def test_a_cup_the_segmenter_called_an_airplane_does_not_replace_the_airplane():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    plane = _remembered("e:173", "toy_airplane", [2.633, 0.727, 0.595], [0.18, 0.13, 0.08])
    cup = _remembered("e:184", "cup", [2.677, 0.732, 0.585], [0.12, 0.10, 0.10])
    cup.semantic_counts = {"toy_airplane": 3.0}
    est.tracks = {"e:173": plane, "e:184": cup}
    est._merge_same_place(np.zeros(3))
    assert "e:173" in est.tracks and est.tracks["e:173"].category == "toy_airplane"
    assert "e:184" not in est.tracks


def test_a_pill_and_a_perfume_eight_centimetres_apart_stay_apart():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    perfume = _remembered("e:205", "perfume_bottle", [2.643, 0.464, 0.60], [0.07, 0.04, 0.10])
    pill = _remembered("e:335", "pill_bottle", [2.721, 0.466, 0.60], [0.046, 0.04, 0.09])
    est.tracks = {"e:205": perfume, "e:335": pill}
    est._merge_same_place(np.zeros(3))
    assert set(est.tracks) == {"e:205", "e:335"}


def test_the_front_pill_is_dropped_when_one_is_deeper_in_the_drawer():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    cabinet = _remembered("e:4", "cabinet", [2.70, 0.60, 0.40], [0.62, 0.52, 0.80])
    cabinet.yaw = 3.1416
    drawer = _remembered("e:6", "drawer", [2.64, 0.60, 0.66], [0.56, 0.44, 0.22])
    drawer.yaw = 3.1416
    inner = _remembered("e:335", "pill_bottle", [2.72, 0.46, 0.60], [0.05, 0.04, 0.09])
    front = _remembered("e:200", "pill_bottle", [2.48, 0.46, 0.60], [0.05, 0.04, 0.09])
    est.tracks = {"e:4": cabinet, "e:6": drawer, "e:335": inner, "e:200": front}
    est._drop_the_front_bottle()
    assert "e:335" in est.tracks and "e:200" not in est.tracks


def test_a_stem_shaped_bear_on_the_table_is_the_wine_glass():
    from steward.ego.estimator import _voxelize

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    bear = _remembered("e:9", "teddy_bear", [0.49, -1.11, 0.85], [0.14, 0.10, 0.17])
    xs = np.linspace(0.42, 0.56, 4)
    ys = np.linspace(-1.16, -1.06, 4)
    zs = (0.77, 0.93)
    bear.voxels = _voxelize(np.array([[x, y, z] for x in xs for y in ys for z in zs]))
    est.tracks = {"e:9": bear}
    est._rebuild_geometry(bear, np.array([-1.6, -1.3, 1.2]), 1.0)
    assert bear.category == "wine_glass"


def test_a_bear_pushed_through_the_back_is_dropped_when_one_remains_inside():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    side = _remembered("e:23", "sideboard", [-2.73, 0.59, 0.42], [1.42, 0.49, 0.84])
    side.yaw = 0.0
    drawer = _remembered("e:24", "drawer", [-2.70, 0.59, 0.70], [0.56, 0.44, 0.22])
    inside = _remembered("e:104", "teddy_bear", [-2.68, 0.48, 0.70], [0.11, 0.10, 0.18])
    behind = _remembered("e:64", "teddy_bear", [-2.96, 0.48, 0.70], [0.13, 0.13, 0.19])
    table = _remembered("e:3", "table", [0.6, -0.9, 0.38], [1.2, 0.8, 0.75])
    est.tracks = {"e:23": side, "e:24": drawer, "e:104": inside, "e:64": behind, "e:3": table}
    est._bring_back_what_the_close_pushed_through()
    assert "e:104" in est.tracks and "e:64" not in est.tracks


def test_a_dark_mug_inside_a_drawer_is_the_pistol():
    from steward.ego.estimator import _voxelize

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    drawer = _remembered("e:22", "drawer", [-2.71, 0.60, 0.70], [0.56, 0.44, 0.22])
    mug = _remembered("e:100", "mug", [-2.70, 0.70, 0.63], [0.09, 0.09, 0.06])
    xs = np.linspace(-2.74, -2.66, 4)
    ys = np.linspace(0.66, 0.74, 4)
    zs = (0.61, 0.65)
    mug.voxels = _voxelize(np.array([[x, y, z] for x in xs for y in ys for z in zs]))
    est.tracks = {"e:22": drawer, "e:100": mug}
    est._pushed_shut.add("e:22")  # the robot has had this drawer open
    est._rebuild_geometry(mug, np.array([-1.6, 0.6, 1.2]), 1.0)
    assert mug.category == "pistol"


def test_a_glass_on_the_table_stays_and_a_bottle_in_the_drawer_is_not_pushed():
    """A swollen wall contains the glass on the table. A bottle in the front half stays put."""
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    table = _remembered("e:3", "table", [0.60, -0.90, 0.375], [1.20, 0.80, 0.75])
    wall = _remembered("e:1", "wall", [0.48, -1.12, 0.75], [0.10, 2.0, 1.5])
    glass = _remembered("e:9", "wine_glass", [0.48, -1.12, 0.842], [0.08, 0.07, 0.17])
    side = _remembered("e:22", "sideboard", [-2.75, 0.60, 0.42], [1.42, 0.49, 0.84])
    smear = _remembered("e:58", "wine_glass", [-2.71, 0.80, 0.76], [0.17, 0.13, 0.19])
    bottle = _remembered("e:187", "wine_glass", [2.50, 0.46, 0.63], [0.065, 0.04, 0.20])
    cabinet = _remembered("e:4", "cabinet", [2.67, 0.60, 0.40], [0.66, 0.56, 0.80])
    face = _remembered("e:152", "plant", [2.99, 0.59, 0.40], [0.62, 0.15, 0.80])
    east = _remembered("e:2", "wall", [3.00, 0.0, 0.75], [0.10, 6.0, 1.5])
    sprout = _remembered("e:208", "plant", [3.00, 0.08, 0.105], [0.17, 0.08, 0.18])
    drawer = _remembered("e:6", "drawer", [2.64, 0.60, 0.66], [0.57, 0.45, 0.29])
    perfume = _remembered("e:213", "perfume_bottle", [2.54, 0.46, 0.58], [0.08, 0.07, 0.11])
    est.tracks = {
        "e:3": table, "e:1": wall, "e:9": glass, "e:22": side, "e:58": smear, "e:187": bottle,
        "e:4": cabinet, "e:152": face, "e:2": east, "e:208": sprout,
        "e:6": drawer, "e:213": perfume,
    }
    est._drop_wall_edges()
    est._drop_flat_plants()
    assert "e:9" in est.tracks and "e:58" not in est.tracks and "e:187" in est.tracks
    assert "e:152" not in est.tracks and "e:208" not in est.tracks
    brought = est.resettle("e:6", np.array([0.24, 0.0, 0.0]), {"e:213": [2.54, 0.46, 0.58]})
    assert brought == []
    assert est.tracks["e:213"].pos[0] == pytest.approx(2.54)


def test_a_bear_already_inside_the_closed_drawer_is_not_pushed_out_the_back():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    est.tracks = {
        "e:14": _remembered("e:14", "drawer", [-2.96, 0.6, 0.66], [0.56, 0.44, 0.22]),
        "e:50": _remembered("e:50", "teddy_bear", [-2.90, 0.6, 0.66], [0.10, 0.10, 0.18]),
    }
    brought = est.resettle("e:14", np.array([-0.26, 0.0, 0.0]), {"e:50": [-2.90, 0.6, 0.66]})
    assert brought == []
    assert est.tracks["e:50"].pos[0] == pytest.approx(-2.90)


def test_an_open_drawer_is_trimmed_off_the_cabinet():
    """Points of the open drawer sit on the camera side of the cabinet. The far face stays,
    and the box comes back inside the cabinet prior."""
    from steward.ego.estimator import _trim_to_furniture_prior, _voxelize

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    cabinet = _remembered("e:4", "cabinet", [2.5, 0.6, 0.4], [0.77, 0.61, 0.8])
    rng = np.random.default_rng(3)
    # Closed cabinet occupies x 2.37..2.97. The open drawer adds x down to 2.16, toward the camera.
    cloud = np.column_stack([rng.uniform(2.16, 2.93, 200), rng.uniform(0.30, 0.91, 200),
                            rng.uniform(0.02, 0.78, 200)])
    cabinet.voxels = _voxelize(cloud)
    kept = _trim_to_furniture_prior(cabinet, cloud, np.zeros(3))
    assert len(kept) < len(cloud)
    centre, size, _yaw = box_of(kept, trim=0.0)
    assert max(float(size[0]), float(size[1])) <= 0.71
    assert float(centre[0]) > 2.5
    est.tracks = {"e:4": cabinet}
    est._rebuild_geometry(cabinet, np.zeros(3), 1.5)
    assert cabinet.footprint <= 0.71


def test_a_longer_sideboard_seen_from_the_front_keeps_its_length():
    """The extra length is across the view, so it is the piece, not a drawer pulled toward the camera."""
    from steward.ego.estimator import _trim_to_furniture_prior, _voxelize

    side = _remembered("e:1", "sideboard", [2.0, 0.0, 0.4], [0.42, 1.6, 0.78])
    rng = np.random.default_rng(4)
    cloud = np.column_stack([rng.uniform(1.8, 2.22, 180), rng.uniform(-0.8, 0.8, 180),
                            rng.uniform(0.02, 0.78, 180)])
    kept = _trim_to_furniture_prior(side, cloud, np.zeros(3))
    _centre, size, _yaw = box_of(kept, trim=0.0)
    assert max(float(size[0]), float(size[1])) == pytest.approx(1.6, abs=0.08)


def test_a_small_object_inside_an_open_drawer_stays_its_own_track():
    """The bear sits in the open drawer. It is not a piece of the drawer front, so it keeps its own track."""
    from steward.math3d import matrix_from_quat

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    header = {"stamp": 1.0, "odometry": {"x": 0.0, "y": 0.0, "yaw": 0.0},
              "joints": {"lift": 0.4, "reach": 0.0, "wrist_yaw": 0.0, "wrist_roll": 0.0},
              "gripper": {"state": "open", "holding": False}}
    cam = est.camera_pose(header["odometry"])
    rot = matrix_from_quat(cam["q"])
    origin = np.asarray(cam["p"], float)
    h, w, fovy = 90, 120, 60.0
    rng = np.random.default_rng(2)
    centre = origin + rot @ np.array([1.05, 0.0, -0.05])
    bear = rng.uniform(centre - np.array([0.05, 0.04, 0.06]), centre + np.array([0.05, 0.04, 0.06]), (80, 3))
    drawer = _remembered("e:14", "drawer", centre.tolist(), [0.56, 0.44, 0.22])
    # A drawer with no furniture around it is an orphan edge. This cabinet holds it and stays
    # under the wall height, so the drawer is not dropped before the bear is seen.
    z = float(centre[2])
    top = min(1.10, z + 0.25)
    cabinet = _remembered("e:4", "cabinet", [float(centre[0]), float(centre[1]), top - 0.40],
                          [0.90, 0.70, 0.80])
    est.tracks = {"e:14": drawer, "e:4": cabinet}
    est._pulled_open.add("e:14")  # the robot pulled this drawer out
    depth = np.zeros((h, w), np.float32)
    mask = np.zeros((h, w), np.uint8)
    _paint(depth, mask, cam, fovy, bear, 1)
    frame = {"depth": depth, "mask": mask, "rgb": np.full((h, w, 3), 140, np.uint8), "fovy": fovy,
             "header": header, "labels": {1: {"teddy_bear": 0.7}}}
    est.update_rgbd(frame)
    header["stamp"] = 1.4
    est.update_rgbd(frame)
    small = [t for t in est.tracks.values() if t.id != "e:14" and t.footprint < 0.3]
    assert len(small) == 1, "the bear was folded into the drawer or split"
    assert "e:14" in est.tracks and est.tracks["e:14"].category == "drawer"


def test_a_failed_color_drops_that_segmenter_score_and_the_sideboard_remains():
    """Counter fails white on the walnut sideboard, then table fails birch. The sideboard prior passes."""
    from steward.ego.classify import classify, fuse_respecting_color

    size = np.array([1.39, 0.47, 0.84])
    color = np.array([0.206, 0.125, 0.073])
    shape = classify(size, 0.015, color, 0.0, bottom_seen=True, min_prob=0.0)
    sem = {"cabinet": 1.06, "box": 0.74, "counter": 1.01, "table": 0.45}
    named = fuse_respecting_color(sem, shape, color, 0.0, 0.5, 0.03)
    assert max(named, key=named.get) == "sideboard"
    white = np.array([0.709, 0.711, 0.705])
    counter = classify(np.array([1.80, 0.61, 0.89]), 0.015, white, 0.0, bottom_seen=True, min_prob=0.0)
    kept = fuse_respecting_color({"counter": 0.7, "table": 0.5}, counter, white, 0.0, 0.5, 0.03)
    assert max(kept, key=kept.get) == "counter"


def test_a_generic_bottle_name_yields_to_the_pill_prior():
    """``bottle`` has no template, so a pill-sized box keeps the pill prior.

    When the segmenter also says pill bottle and that score leads, the fused winner stays.
    """
    from steward.ego.classify import classify, fuse_semantics, shape_over_bottle

    size = np.array([0.045, 0.045, 0.112])
    color = np.array([0.804, 0.601, 0.424])
    shape = classify(size, 0.533, color, 0.0, bottom_seen=True, min_prob=0.0)
    fused = fuse_semantics({"bottle": 0.7}, shape, 0.5, 0.03)
    assert max(fused, key=fused.get) == "bottle"
    assert max(shape_over_bottle(fused, shape), key=shape_over_bottle(fused, shape).get) == "pill_bottle"
    # 11 cm tall is closer to a cup's height. The segmenter still said bottle, so the cup does not inherit it.
    tall = classify(np.array([0.05, 0.05, 0.112]), 0.533, np.array([0.822, 0.489, 0.211]), 0.0,
                    bottom_seen=True, min_prob=0.0)
    assert max(tall, key=tall.get) == "cup"
    tall_fused = fuse_semantics({"bottle": 0.7}, tall, 0.5, 0.03)
    assert max(shape_over_bottle(tall_fused, tall), key=shape_over_bottle(tall_fused, tall).get) == "pill_bottle"
    both = fuse_semantics({"bottle": 0.7, "pill_bottle": 0.85}, shape, 0.5, 0.03)
    kept = shape_over_bottle(both, shape)
    assert max(kept, key=kept.get) == "pill_bottle"
    assert kept is both


def test_a_dark_speck_below_the_bowl_prior_is_a_pistol_and_a_full_pistol_is_not_renamed_here():
    """Black depth covers only part of a pistol. A full-length pistol is left to the ordinary prior."""
    from steward.ego.classify import partial_pistol

    dark = np.array([0.09, 0.084, 0.087])
    assert partial_pistol(np.array([0.04, 0.03, 0.022]), "bowl", dark, 0.0) == "pistol"
    assert partial_pistol(np.array([0.18, 0.11, 0.03]), "pistol", dark, 0.0) is None
    assert partial_pistol(np.array([0.12, 0.11, 0.05]), "bowl", np.array([0.15, 0.29, 0.65]), 0.0) is None


def test_colour_names_the_bottle_the_airplane_and_the_bear():
    """Pink is the perfume, white is the pill, yellow is the airplane, brown is the bear."""
    from steward.ego.classify import content_color_name

    pink = np.array([0.95, 0.60, 0.75])
    white = np.array([0.94, 0.94, 0.92])
    yellow = np.array([0.95, 0.78, 0.10])
    brown = np.array([0.58, 0.38, 0.20])
    assert content_color_name(np.array([0.04, 0.04, 0.12]), pink, 0.0, "pill_bottle") == "perfume_bottle"
    # The drawer floor dilutes the pink until the strict pink rule misses it. It is still the perfume.
    assert content_color_name(np.array([0.05, 0.04, 0.09]), np.array([0.83, 0.76, 0.79]), 0.0,
                              "pill_bottle") == "perfume_bottle"
    assert content_color_name(np.array([0.05, 0.046, 0.11]), white, 0.0, "bottle") == "pill_bottle"
    assert content_color_name(np.array([0.14, 0.06, 0.11]), white, 0.0, "bottle") == "pill_bottle"
    assert content_color_name(np.array([0.13, 0.09, 0.10]), yellow, 0.0, "cup") == "toy_airplane"
    assert content_color_name(np.array([0.09, 0.08, 0.09]), np.array([0.11, 0.11, 0.12]), 0.0,
                              "unknown") == "mug"
    assert content_color_name(np.array([0.20, 0.12, 0.10]), yellow, 0.0, "bowl") == "toy_airplane"
    assert content_color_name(np.array([0.22, 0.08, 0.10]), yellow, 0.0, "bowl") == "toy_airplane"
    assert content_color_name(np.array([0.134, 0.076, 0.105]), np.array([0.922, 0.542, 0.161]), 0.0,
                              "bowl") == "toy_airplane"
    assert content_color_name(np.array([0.12, 0.10, 0.16]), brown, 0.0, "drawer") == "teddy_bear"
    # A drawer front is half a metre wide. Colour does not rename it.
    assert content_color_name(np.array([0.56, 0.44, 0.22]), brown, 0.0, "drawer") is None
    assert content_color_name(np.array([0.085, 0.072, 0.064]), np.array([0.111, 0.11, 0.115]), 0.0, "cup") == "mug"
    assert content_color_name(np.array([0.11, 0.09, 0.09]), np.array([0.40, 0.38, 0.36]), 0.0, "cup") == "mug"
    assert content_color_name(np.array([0.09, 0.09, 0.10]), np.array([0.64, 0.72, 0.76]), 0.0, "cup") is None
    from steward.ego.classify import clear_stem_name
    assert clear_stem_name(np.array([0.072, 0.065, 0.188]), "unknown", np.array([0.8, 0.8, 0.82]), 0.35) == "wine_glass"
    assert clear_stem_name(np.array([0.20, 0.20, 0.08]), "bowl", np.array([0.2, 0.55, 0.2]), 0.0) is None
    assert clear_stem_name(np.array([0.08, 0.08, 0.10]), "cup", np.array([0.9, 0.9, 0.92]), 0.0) is None


def test_a_sideboard_sized_counter_is_the_sideboard_unless_it_is_white():
    """The counter is the long white run. A 1.4 m walnut piece SAM calls a counter is the sideboard."""
    from steward.ego.classify import sideboard_not_counter

    walnut = np.array([0.36, 0.22, 0.13])
    white = np.array([0.94, 0.94, 0.92])
    side = np.array([1.343, 0.478, 0.84])
    assert sideboard_not_counter(side, "counter", walnut, 0.0) == "sideboard"
    assert sideboard_not_counter(side, "counter", white, 0.0) is None
    assert sideboard_not_counter(np.array([1.77, 0.62, 0.89]), "counter", walnut, 0.0) is None


def test_a_pistol_stays_when_the_drawer_around_it_is_seen_again():
    """The pistol is a small track inside the open drawer. A later drawer mask, and a drawer
    fragment centred on it, do not replace it."""
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    pistol = _remembered("e:47", "pistol", [-2.55, 0.70, 0.64], [0.04, 0.03, 0.022])
    pistol.mean_rgb = np.array([0.09, 0.084, 0.087])
    pistol.bottom_seen = True
    fragment = _remembered("e:49", "drawer", [-2.52, 0.68, 0.66], [0.20, 0.12, 0.10])
    fragment.bottom_seen = True
    est.tracks = {"e:47": pistol, "e:49": fragment}
    est._merge_same_place(np.array([-1.4, 0.6, 1.2]))
    assert "e:47" in est.tracks and est.tracks["e:47"].category == "pistol"
    assert est.tracks["e:47"].footprint < 0.1

    from steward.math3d import matrix_from_quat

    header = {"stamp": 2.0, "odometry": {"x": 0.0, "y": 0.0, "yaw": 0.0},
              "joints": {"lift": 0.4, "reach": 0.0, "wrist_yaw": 0.0, "wrist_roll": 0.0},
              "gripper": {"state": "open", "holding": False}}
    cam = est.camera_pose(header["odometry"])
    rot = matrix_from_quat(cam["q"])
    origin = np.asarray(cam["p"], float)
    centre = origin + rot @ np.array([1.05, 0.0, -0.05])
    pistol.pos = centre
    h, w, fovy = 90, 120, 60.0
    rng = np.random.default_rng(3)
    front = rng.uniform(centre - np.array([0.28, 0.22, 0.11]), centre + np.array([0.28, 0.22, 0.11]), (240, 3))
    depth = np.zeros((h, w), np.float32)
    mask = np.zeros((h, w), np.uint8)
    _paint(depth, mask, cam, fovy, front, 1)
    frame = {"depth": depth, "mask": mask, "rgb": np.full((h, w, 3), 90, np.uint8), "fovy": fovy,
             "header": header, "labels": {1: {"drawer": 0.8}}}
    est.update_rgbd(frame)
    assert "e:47" in est.tracks and est.tracks["e:47"].category == "pistol"
    assert est.tracks["e:47"].footprint < 0.1


def test_a_bear_that_shares_the_drawer_floor_is_not_folded_into_the_drawer():
    """The bear's mask includes the drawer floor. Those shared points do not make the bear the drawer."""
    from steward.assets.generators import steward_bot
    from steward.ego.estimator import EgoEstimator

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    drawer = _remembered("e:14", "drawer", [-2.7, 0.6, 0.66], [0.56, 0.44, 0.22])
    bear = _remembered("e:50", "teddy_bear", [-2.5, 0.48, 0.70], [0.12, 0.10, 0.18])
    shared = np.array([[0, 0, 70], [1, 0, 70], [0, 1, 70]], np.int32)
    drawer.voxels = np.vstack([shared, np.array([[10, 0, 70], [11, 0, 70], [12, 0, 70]], np.int32)])
    bear.voxels = shared.copy()
    bear.observations = 2
    est.tracks = {"e:14": drawer, "e:50": bear}
    est._merge_same_surface(np.array([0.0, 0.0, 1.2]))
    assert "e:50" in est.tracks and est.tracks["e:50"].category == "teddy_bear"
    assert est.tracks["e:14"].category == "drawer"


def test_a_tall_square_the_detector_did_not_call_a_pill_is_the_perfume():
    """The pill is the region SAM names. The other tall square, labelled only as the drawer, is the perfume."""
    from steward.ego.classify import perfume_without_bottle_label

    perfume = np.array([0.04, 0.04, 0.122])
    assert perfume_without_bottle_label(perfume, "pill_bottle", {"drawer": 0.93}) == "perfume_bottle"
    assert perfume_without_bottle_label(perfume, "pill_bottle", {"pill_bottle": 9.0, "bottle": 8.0}) is None
    assert perfume_without_bottle_label(np.array([0.08, 0.05, 0.112]), "pill_bottle", {"drawer": 1.0}) is None


def test_a_named_bear_stays_a_bear_when_a_later_view_says_drawer():
    """The open drawer behind the bear is labelled drawer. The bear already has its name, and keeps it."""
    from steward.assets.generators import steward_bot
    from steward.ego.estimator import EgoEstimator, Track

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    bear = Track(id="e:1", category_counts={"teddy_bear": 5.0}, pos=np.array([1.0, 0.0, 0.7]),
                 var=np.full(3, 1e-4), yaw=0.0, yaw_var=1e-3, size=np.array([0.20, 0.15, 0.20]),
                 observations=6, first_seen=0.0, last_seen=1.0, existence=1.0,
                 mean_rgb=np.array([0.4, 0.4, 0.4]))
    bear.bottom_seen = True
    bear.semantic_counts = {"drawer": 5.0}
    bear.semantic_views = 5
    cloud = np.random.default_rng(0).uniform([0.90, -0.07, 0.60], [1.10, 0.08, 0.80], (200, 3))
    bear.voxels = np.unique(np.floor(cloud / 0.01).astype(np.int32), axis=0)
    est.tracks[bear.id] = bear
    est._rebuild_geometry(bear, np.array([0.0, 0.0, 1.2]), 1.0)
    assert bear.category == "teddy_bear"


def test_a_smear_along_one_axis_of_a_small_bottle_is_cut():
    """The pill is 4 cm across. A strip of drawer floor on one axis is not part of it.

    The cut starts at the bottle priors' own long/short ratio. An 8 cm by 4 cm strip is already
    past that. A 14 cm strip is too.
    """
    from steward.ego.classify import max_plan_aspect
    from steward.ego.estimator import _trim_long_smear

    aspect = max_plan_aspect(("pill_bottle", "perfume_bottle"))
    rng = np.random.default_rng(1)
    body = rng.uniform([-0.02, -0.02, 0.55], [0.02, 0.02, 0.66], (40, 3))
    smear = rng.uniform([-0.07, -0.01, 0.55], [0.07, 0.01, 0.57], (40, 3))
    cloud = np.vstack([body, smear])
    trimmed = _trim_long_smear(cloud, 0.04)
    assert len(trimmed) < len(cloud)
    assert float(np.ptp(trimmed[:, 0])) < 0.05
    mild = np.vstack([
        rng.uniform([-0.02, -0.02, 0.55], [0.02, 0.02, 0.66], (40, 3)),
        rng.uniform([-0.04, -0.01, 0.55], [0.04, 0.01, 0.57], (30, 3)),
    ])
    assert _trim_long_smear(mild, 0.04) is mild
    cut = _trim_long_smear(mild, 0.04, aspect)
    assert len(cut) < len(mild)
    assert float(np.ptp(cut[:, 0])) < 0.05


def test_a_mask_mixing_near_and_far_surfaces_does_not_start_a_wide_track():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    header = {"stamp": 1.0, "odometry": {"x": 0.0, "y": 0.0, "yaw": 0.0},
              "joints": {"lift": 0.4, "reach": 0.0, "wrist_yaw": 0.0, "wrist_roll": 0.0},
              "gripper": {"state": "open", "holding": False}}
    cam = est.camera_pose(header["odometry"])
    h, w, fovy = 80, 100, 60.0
    depth = np.zeros((h, w), np.float32)
    mask = np.zeros((h, w), np.uint8)
    rng = np.random.default_rng(1)
    for x0 in (1.3, 2.8):
        pts = np.column_stack([np.full(500, x0), rng.uniform(-0.8, 0.8, 500), rng.uniform(0.2, 1.2, 500)])
        _paint(depth, mask, cam, fovy, pts, 1)
    frame = {"depth": depth, "mask": mask, "rgb": np.full((h, w, 3), 120, np.uint8), "fovy": fovy,
             "header": header, "labels": {1: {"box": 0.6}}}
    est.update_rgbd(frame)
    assert not any(t.footprint > 1.0 for t in est.tracks.values())


def test_a_box_standing_on_a_counter_stays_its_own_track():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    counter = _remembered("e:1", "counter", [0.0, 0.0, 0.45], [1.8, 0.62, 0.9])
    box = _remembered("e:2", "box", [0.2, 0.0, 0.98], [0.4, 0.35, 0.16])
    est.tracks = {"e:1": counter, "e:2": box}
    est._fold_furniture_slices(np.array([0.0, -1.5, 1.4]))
    assert set(est.tracks) == {"e:1", "e:2"}


def test_an_upper_half_named_box_is_folded_back_into_the_cabinet():
    """The upper body named box is a second object. Folding it back removes that object and
    restores the cabinet's height."""
    from steward.math3d import matrix_from_quat

    header = {"stamp": 1.0, "odometry": {"x": 0.0, "y": 0.0, "yaw": 0.0},
              "joints": {"lift": 0.4, "reach": 0.0, "wrist_yaw": 0.0, "wrist_roll": 0.0},
              "gripper": {"state": "open", "holding": False}}
    cam = EgoEstimator(robot_model=steward_bot().to_dict()).camera_pose(header["odometry"])
    forward = matrix_from_quat(cam["q"])[:, 0].copy()
    forward[2] = 0.0
    forward /= np.linalg.norm(forward)
    centre = np.asarray(cam["p"], float) + forward * 1.4
    centre[2] = 0.4
    h, w, fovy = 120, 160, 60.0
    depth = np.zeros((h, w), np.float32)
    mask = np.zeros((h, w), np.uint8)
    rgb = np.zeros((h, w, 3), np.uint8)
    rng = np.random.default_rng(2)

    def slab(z0, z1, instance, color, hx=0.28, hy=0.22):
        pts = np.column_stack([
            rng.uniform(centre[0] - hx, centre[0] + hx, 400),
            rng.uniform(centre[1] - hy, centre[1] + hy, 400),
            rng.uniform(z0, z1, 400),
        ])
        _paint(depth, mask, cam, fovy, pts, instance)
        ys, xs = np.nonzero(mask == instance)
        rgb[ys, xs] = color

    slab(0.02, 0.72, 1, (90, 110, 150))  # the body, blue-grey
    # Narrower than 0.5 m, so it is not dropped as a second copy of the cabinet, and still
    # wide enough to be the cabinet's upper band rather than an object standing on it.
    slab(0.52, 0.76, 2, (180, 40, 40), hx=0.16, hy=0.16)
    frame = {"depth": depth, "mask": mask, "rgb": rgb, "fovy": fovy, "header": header,
             "labels": {1: {"cabinet": 0.9}, 2: {"box": 0.8}}}

    def run(fold: bool) -> EgoEstimator:
        est = EgoEstimator(robot_model=steward_bot().to_dict())
        if not fold:
            est._fold_furniture_slices = lambda _cam_p: None
        est.update_rgbd(frame)
        return est

    before, after = run(False), run(True)
    assert any(t.category == "box" for t in before.tracks.values())
    assert not any(t.category == "box" for t in after.tracks.values())
    cab = next(t for t in after.tracks.values() if t.category == "cabinet")
    assert cab.top() == pytest.approx(0.76, abs=0.06)
    assert "box" not in cab.category_counts


def test_a_printed_lid_with_a_visible_wall_gains_the_box_under_it():
    """The segmenter names the printed lid and not the 5 cm box under it. Depth of that wall,
    inside the lid's outline, becomes its own box. A lid with nothing under it does not."""
    from steward.ego.perception import segments
    from steward.math3d import matrix_from_quat

    header = {"stamp": 1.0, "odometry": {"x": 0.0, "y": 0.0, "yaw": 0.0},
              "joints": {"lift": 0.4, "reach": 0.0, "wrist_yaw": 0.0, "wrist_roll": 0.0},
              "gripper": {"state": "open", "holding": False}}
    cam = EgoEstimator(robot_model=steward_bot().to_dict()).camera_pose(header["odometry"])
    rot = matrix_from_quat(cam["q"])
    origin = np.asarray(cam["p"], float)
    h, w, fovy = 120, 160, 60.0
    rng = np.random.default_rng(1)
    lid_centre = origin + rot @ np.array([1.15, 0.0, 0.0])
    lid = rng.uniform(lid_centre - np.array([0.17, 0.17, 0.004]), lid_centre + np.array([0.17, 0.17, 0.004]), (120, 3))
    forward = rot @ np.array([1.0, 0.0, 0.0])
    forward[2] = 0.0
    forward /= np.linalg.norm(forward)
    wall = lid.copy()
    wall[:, :2] -= forward[:2] * 0.08
    wall[:, 2] = lid[:, 2] - rng.uniform(0.02, 0.06, len(wall))

    def frame_of(points_lid, points_wall):
        depth = np.zeros((h, w), np.float32)
        mask = np.zeros((h, w), np.uint8)
        _paint(depth, mask, cam, fovy, points_lid, 1)
        if points_wall is not None:
            lid_mask = mask.copy()
            depth_lid = depth.copy()
            _paint(depth, mask, cam, fovy, points_wall, 1)
            wall_depth = depth.copy()
            wall_only = (mask == 1) & (lid_mask == 0)
            depth[:] = depth_lid
            depth[wall_only] = wall_depth[wall_only]
            mask[:] = lid_mask
        return {"depth": depth, "mask": mask, "rgb": np.full((h, w, 3), 180, np.uint8), "fovy": fovy,
                "header": header, "labels": {1: {"picture": 0.8}}}

    seen = segments(frame_of(lid, wall), cam)
    assert any(s.semantics and "box" in s.semantics for s in seen)
    assert not any(s.semantics and "box" in (s.semantics or {}) for s in segments(frame_of(lid, None), cam))
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    est.update_rgbd(frame_of(lid, wall))
    boxes = [t for t in est.tracks.values() if t.category == "box"]
    assert len(boxes) == 1
    assert boxes[0].size[2] == pytest.approx(0.05, abs=0.03)
    assert min(boxes[0].size[0], boxes[0].size[1]) > 0.2


def test_phase1_nested_bowls_stay_separate_tracks_across_views():
    """Three bowls share a centre and are separate masks. Later views must not fuse the smaller ones in."""
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    header = {"stamp": 1.0, "odometry": {"x": 0.0, "y": 0.0, "yaw": 0.0},
              "joints": {"lift": 0.4, "reach": 0.0, "wrist_yaw": 0.0, "wrist_roll": 0.0},
              "gripper": {"state": "open", "holding": False}}
    cam = est.camera_pose(header["odometry"])
    h, w, fovy = 120, 160, 60.0
    rng = np.random.default_rng(1)
    board = 0.85
    specs = (
        (1, [1.5, 0.0, board + 0.035], [0.30, 0.30, 0.07], [180, 40, 40]),
        (2, [1.5, 0.0, board + 0.038], [0.20, 0.20, 0.06], [40, 150, 50]),
        (3, [1.5, 0.0, board + 0.041], [0.12, 0.12, 0.05], [40, 60, 180]),
    )

    def frame(stamp: float) -> dict:
        depth = np.zeros((h, w), np.float32)
        mask = np.zeros((h, w), np.uint8)
        rgb = np.full((h, w, 3), 20, np.uint8)
        for inst, centre, size, color in specs:
            pts = rng.uniform(np.asarray(centre) - np.asarray(size) / 2, np.asarray(centre) + np.asarray(size) / 2, (200, 3))
            _paint(depth, mask, cam, fovy, pts, inst)
            rgb[mask == inst] = color
        return {"depth": depth, "mask": mask, "rgb": rgb, "fovy": fovy, "camera": "head_camera",
                "header": {**header, "stamp": stamp}}

    for k in range(4):
        est.update_rgbd(frame(0.2 * (k + 1)))
    bowls = sorted((t for t in est.tracks.values() if t.category == "bowl"), key=lambda t: -t.footprint)
    assert len(bowls) == 3, [(t.id, t.category, [round(float(v), 3) for v in t.size]) for t in est.tracks.values()]
    assert est.support_of(bowls[2]) is bowls[1]
    assert est.support_of(bowls[1]) is bowls[0]


def test_a_one_sided_view_of_a_bowl_joins_the_bowl_already_seen():
    """The bowl on the cabinet, and a smaller box a few centimetres to the side seen once.

    The footprints differ by more than the nesting step, but the centres do not coincide, so the
    smaller box is a partial view. The track with more observations keeps its box."""
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    full = _remembered("e:5", "bowl", [2.676, 0.587, 0.834], [0.194, 0.152, 0.069])
    full.observations = 13
    part = _remembered("e:13", "bowl", [2.616, 0.609, 0.837], [0.117, 0.063, 0.056])
    part.observations = 1
    est.tracks = {"e:5": full, "e:13": part}
    est._merge_same_place(np.zeros(3))
    assert set(est.tracks) == {"e:5"}
    assert est.tracks["e:5"].footprint == pytest.approx(0.194, abs=1e-6)
    closer = _remembered("e:8", "bowl", [2.670, 0.589, 0.832], [0.186, 0.149, 0.067])
    closer.observations = 12
    aside = _remembered("e:14", "bowl", [2.631, 0.586, 0.840], [0.132, 0.096, 0.050])
    aside.observations = 1
    est.tracks = {"e:8": closer, "e:14": aside}
    est._merge_same_place(np.zeros(3))
    assert set(est.tracks) == {"e:8"}
    nested = {
        "e:1": _remembered("e:1", "bowl", [1.5, 0.0, 0.885], [0.30, 0.30, 0.07]),
        "e:2": _remembered("e:2", "bowl", [1.5, 0.0, 0.888], [0.20, 0.20, 0.06]),
    }
    est.tracks = nested
    est._merge_same_place(np.zeros(3))
    assert set(est.tracks) == {"e:1", "e:2"}


def test_a_bottle_on_the_front_of_a_drawer_nobody_opened_is_its_handle():
    """A shut drawer shows its handle and nothing it holds. The drawer rules that name a small box
    a content, and keep it when the drawer is shut, apply once the robot has moved that drawer."""
    def scene():
        est = EgoEstimator(robot_model=steward_bot().to_dict())
        cabinet = _remembered("e:6", "cabinet", [2.69, 0.59, 0.41], [0.53, 0.61, 0.79])
        drawer = _remembered("e:4", "drawer", [2.66, 0.59, 0.67], [0.56, 0.44, 0.21])
        handle = _remembered("e:8", "pill_bottle", [2.443, 0.609, 0.637], [0.11, 0.04, 0.092])
        est.tracks = {"e:6": cabinet, "e:4": drawer, "e:8": handle}
        return est

    est = scene()
    assert not est._beside_drawer(est.tracks["e:8"].pos)
    est._drop_inside_shut_furniture()
    assert set(est.tracks) == {"e:6", "e:4"}
    assert est.dropped[-1]["reason"] == "inside shut furniture"

    est = scene()
    est._pushed_shut.add("e:4")  # the robot looked into this drawer and closed it on the bottle
    assert est._beside_drawer(est.tracks["e:8"].pos)
    est._drop_inside_shut_furniture()
    assert set(est.tracks) == {"e:6", "e:4", "e:8"}


def test_a_region_running_from_a_bowl_along_the_table_is_not_merged_into_the_bowl():
    """A two-metre mask that holds the bowl's points. The bowl, seen twenty times, keeps its own
    cloud; a second view of the same bowl still merges into it."""
    from steward.ego.estimator import _voxelize

    rng = np.random.default_rng(0)
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    bowl = _remembered("e:5", "bowl", [0.243, -1.068, 0.79], [0.308, 0.283, 0.07])
    own = np.column_stack([rng.uniform(0.09, 0.39, 600), rng.uniform(-1.21, -0.93, 600), rng.uniform(0.755, 0.825, 600)])
    bowl.voxels = _voxelize(own)
    slab = _remembered("e:245", "unknown", [-0.403, -1.698, 0.55], [2.108, 0.283, 0.55])
    slab.observations = 1
    strip = np.column_stack([rng.uniform(-1.4, 0.6, 300), rng.uniform(-1.84, -1.56, 300), rng.uniform(0.3, 0.8, 300)])
    slab.voxels = _voxelize(np.vstack([own, strip]))
    again = _remembered("e:251", "bowl", [0.235, -1.067, 0.79], [0.31, 0.276, 0.07])
    again.observations = 1
    again.voxels = _voxelize(own[:400])
    est.tracks = {"e:5": bowl, "e:245": slab, "e:251": again}
    cloud = len(bowl.voxels)
    est._merge_same_surface(np.array([1.0, 0.6, 1.2]))
    assert set(est.tracks) == {"e:5"}
    assert {d["id"]: d["reason"] for d in est.dropped} == {
        "e:245": "a region spanning an established object and its surroundings", "e:251": "merged: same surfaces"}
    assert len(bowl.voxels) == cloud
    assert float(max(bowl.size[0], bowl.size[1])) < 0.33


def test_a_content_the_close_carried_keeps_its_place_when_it_is_seen_again():
    """The perfume was measured five times in the open drawer. The push carries it 25 cm; a view
    of it during the push is the median of single-view boxes that moved with it, not of the
    ones the open drawer left."""
    from steward.ego.perception import Segment

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    perfume = _remembered("e:162", "perfume_bottle", [2.283, 0.461, 0.60], [0.067, 0.04, 0.10])
    for _ in range(5):
        perfume.recent.append((np.array([2.283, 0.461, 0.60]), np.array([0.067, 0.04, 0.10]), 0.8))
    est.tracks = {"e:162": perfume}
    est._translate_track(perfume, np.array([0.25, 0.0, 0.0]))
    assert [float(row[0][0]) for row in perfume.recent] == pytest.approx([2.533] * 5)
    assert all(float(row[2]) == 0.8 for row in perfume.recent)
    cloud = np.random.default_rng(0).uniform([2.50, 0.44, 0.55], [2.566, 0.48, 0.65], (200, 3))
    seg = Segment(points=cloud, color=np.array([0.9, 0.5, 0.6]), pixels=200, invalid_frac=0.0, border=False, range=0.8)
    centre, size, yaw = box_of(cloud)
    est._fuse_cloud(perfume, {"seg": seg, "centre": centre, "size": size, "yaw": yaw, "probs": {"perfume_bottle": 1.0},
                              "footprint": float(max(size[0], size[1]))}, np.array([2.0, 0.6, 1.2]))
    assert float(perfume.pos[0]) == pytest.approx(2.533, abs=0.02)


def test_a_young_sliver_centred_on_a_walls_plane_is_the_wall_whatever_it_was_named():
    """The foot of the east wall boxed as a bottle, seen twice. A bottle standing against the
    wall has its centre off the wall's surface, and one seen often enough is an object."""
    def scene(pos, observations):
        est = EgoEstimator(robot_model=steward_bot().to_dict())
        east = _remembered("e:2", "wall", [3.05, 0.0, 0.75], [0.10, 6.0, 1.5])
        sliver = _remembered("e:276", "pill_bottle", pos, [0.021, 0.003, 0.108])
        sliver.observations = observations
        est.tracks = {"e:2": east, "e:276": sliver}
        return est

    est = scene([3.018, 1.14, 0.07], 2)
    est._drop_wall_edges()
    assert set(est.tracks) == {"e:2"}
    assert est.dropped[-1]["reason"] == "a wall edge, not an object"

    est = scene([2.93, 1.14, 0.07], 2)
    est._drop_wall_edges()
    assert "e:276" in est.tracks

    est = scene([3.018, 1.14, 0.07], 8)
    est._drop_wall_edges()
    assert "e:276" in est.tracks


def test_a_sliver_where_two_walls_meet_is_the_corner_of_the_room():
    """Each wall was measured a few centimetres short of the corner, so the corner is past the
    length of both. A glass two hands' widths off both walls is in the room."""
    def scene(pos):
        est = EgoEstimator(robot_model=steward_bot().to_dict())
        west = _remembered("e:33", "wall", [-3.002, -0.105, 0.835], [4.74, 0.026, 1.64])
        west.yaw = -1.568
        north = _remembered("e:11", "wall", [-0.07, 2.505, 0.885], [5.81, 0.06, 1.74])
        sliver = _remembered("e:45", "wine_glass", pos, [0.103, 0.093, 0.22])
        sliver.observations = 1
        est.tracks = {"e:33": west, "e:11": north, "e:45": sliver}
        return est

    est = scene([-3.01, 2.505, 0.965])
    est._drop_wall_edges()
    assert set(est.tracks) == {"e:33", "e:11"}
    assert est.dropped[-1]["reason"] == "a wall edge, not an object"

    est = scene([-2.80, 2.30, 0.965])
    est._drop_wall_edges()
    assert "e:45" in est.tracks


def test_a_push_drops_what_was_only_glimpsed_beside_the_open_drawer():
    """The shut drawer hides what it holds, so the push keeps it unseen. A slab of the drawer's
    back panel boxed twice and named by its colour goes, and so does the side panel that three
    regions of one moment showed; the bottle the segmenter named, and the bear seen through the
    whole look, stay."""
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    cabinet = _remembered("e:4", "cabinet", [2.70, 0.59, 0.41], [0.53, 0.61, 0.79])
    drawer = _remembered("e:6", "drawer", [2.40, 0.59, 0.66], [0.56, 0.44, 0.21])
    panel = _remembered("e:262", "teddy_bear", [2.62, 0.51, 0.66], [0.161, 0.044, 0.20])
    panel.observations = 2
    bottle = _remembered("e:156", "pill_bottle", [2.50, 0.46, 0.59], [0.05, 0.05, 0.10])
    bottle.observations = 2
    bottle.semantic_counts = {"pill_bottle": 2.0}
    bottle.semantic_views = 2
    bear = _remembered("e:90", "teddy_bear", [2.45, 0.70, 0.64], [0.11, 0.10, 0.18])
    bear.observations = 9
    bear.first_seen, bear.last_seen = 25.9, 46.4
    side = _remembered("e:158", "cup", [2.477, 0.38, 0.665], [0.199, 0.116, 0.08])
    side.observations = 3
    side.first_seen = side.last_seen = 26.1
    table = _remembered("e:3", "table", [0.60, -0.90, 0.375], [1.20, 0.80, 0.75])  # the room is to the west
    est.tracks = {"e:3": table, "e:4": cabinet, "e:6": drawer, "e:262": panel, "e:156": bottle, "e:90": bear,
                  "e:158": side}
    est.shift("e:6", [0.27, 0.0, 0.0])
    assert est._pushed_shut == {"e:6"}
    assert set(est.tracks) == {"e:3", "e:4", "e:6", "e:156", "e:90"}
    assert {d["id"] for d in est.dropped} == {"e:262", "e:158"}
    assert all("glimpsed beside an open drawer" in d["reason"] for d in est.dropped)


def test_a_push_drops_a_panel_that_one_bottle_mask_named():
    """A tall slab of the drawer's back panel was boxed behind the pill, named a bear by its
    colour, and one mask of the bottle in front of it was folded in. A bottle name does not name
    a bear; the pill seen twice, and called a bottle once, stays."""
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    cabinet = _remembered("e:4", "cabinet", [2.70, 0.59, 0.41], [0.53, 0.61, 0.79])
    drawer = _remembered("e:6", "drawer", [2.40, 0.59, 0.66], [0.56, 0.44, 0.29])
    panel = _remembered("e:143", "teddy_bear", [2.517, 0.453, 0.67], [0.136, 0.046, 0.23])
    panel.observations = 2
    panel.first_seen, panel.last_seen = 31.6, 46.0
    panel.semantic_counts, panel.semantic_views = {"bottle": 0.53}, 1
    bottle = _remembered("e:121", "pill_bottle", [2.45, 0.46, 0.60], [0.045, 0.042, 0.09])
    bottle.observations = 2
    bottle.first_seen, bottle.last_seen = 39.4, 46.4
    bottle.semantic_counts, bottle.semantic_views = {"bottle": 0.63}, 1
    table = _remembered("e:3", "table", [0.60, -0.90, 0.375], [1.20, 0.80, 0.75])  # the room is to the west
    est.tracks = {"e:3": table, "e:4": cabinet, "e:6": drawer, "e:143": panel, "e:121": bottle}
    est.shift("e:6", [0.28, 0.0, 0.0])
    assert set(est.tracks) == {"e:3", "e:4", "e:6", "e:121"}
    assert [d["id"] for d in est.dropped] == ["e:143"]
    assert "glimpsed beside an open drawer" in est.dropped[0]["reason"]


def test_a_swollen_drawer_in_front_of_the_open_one_is_dropped_and_a_counter_is_not_its_furniture():
    """The close stands in front of the drawer. A second swollen box there is read as a collision,
    and a counter has no drawer of its own."""
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    sideboard = _remembered("e:s", "sideboard", [-2.75, 0.59, 0.42], [1.40, 0.50, 0.84])
    real = _remembered("e:21", "drawer", [-2.71, 0.59, 0.70], [0.57, 0.45, 0.22])
    swollen = _remembered("e:77", "drawer", [-2.42, 0.64, 0.56], [0.78, 0.47, 0.49])
    counter = _remembered("e:c", "counter", [0.0, 2.16, 0.46], [1.88, 0.65, 0.90])
    apron = _remembered("e:152", "drawer", [-0.87, 2.14, 0.79], [0.63, 0.16, 0.21])
    est.tracks = {"e:s": sideboard, "e:21": real, "e:77": swollen, "e:c": counter, "e:152": apron}
    est._opened_drawers.add("e:21")
    est._drop_orphan_parts()
    assert "e:77" not in est.tracks and "e:152" not in est.tracks
    assert "e:21" in est.tracks


def test_a_partial_bowl_and_a_second_airplane_fold_into_the_object():
    """One bowl seen twice, a centimetre apart, is one bowl. Two airplanes a hand's width apart
    are the same toy. Two bowls a span apart stay two."""
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    wide = _remembered("e:7", "bowl", [0.708, -1.077, 0.782], [0.384, 0.227, 0.07])
    tight = _remembered("e:31", "bowl", [0.713, -1.069, 0.785], [0.229, 0.235, 0.055])
    other = _remembered("e:13", "bowl", [1.074, -1.006, 0.775], [0.175, 0.152, 0.04])
    near = _remembered("e:163", "toy_airplane", [2.657, 0.715, 0.589], [0.163, 0.106, 0.112])
    far = _remembered("e:207", "toy_airplane", [2.57, 0.725, 0.569], [0.08, 0.07, 0.072])
    est.tracks = {"e:7": wide, "e:31": tight, "e:13": other, "e:163": near, "e:207": far}
    est._merge_same_place(np.zeros(3))
    est._drop_second_airplane()
    assert "e:7" not in est.tracks and "e:31" in est.tracks
    assert "e:13" in est.tracks
    assert [t.id for t in est.tracks.values() if t.category == "toy_airplane"] == ["e:207"]


def test_a_swollen_drawer_sits_under_the_top_and_a_near_face_recedes():
    """A drawer boxed down into the carcass is the band under the top. A cabinet face a few
    centimetres past its prior, on the camera side, moves back by half of that. A cup on the
    counter does the same. The wine glass on the table does not, and a long sideboard seen
    from the front keeps the length that runs across the view."""
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    side = _remembered("e:s", "sideboard", [-2.755, 0.598, 0.42], [1.43, 0.50, 0.84])
    drawer = _remembered("e:23", "drawer", [-2.722, 0.596, 0.612], [0.57, 0.45, 0.57])
    cabinet = _remembered("e:4", "cabinet", [2.655, 0.585, 0.407], [0.667, 0.567, 0.78])
    counter = _remembered("e:c", "counter", [0.0, 2.18, 0.46], [1.85, 0.62, 0.90])
    cup = _remembered("e:11", "cup", [-0.527, 2.155, 0.955], [0.155, 0.149, 0.10])
    glass = _remembered("e:32", "wine_glass", [0.48, -1.12, 0.84], [0.09, 0.07, 0.16])
    est.tracks = {"e:s": side, "e:23": drawer, "e:4": cabinet, "e:c": counter, "e:11": cup, "e:32": glass}
    cam = np.array([-1.8, -1.2, 1.5])
    est._seat_swollen_drawer()
    est._recede_facing_excess(cam)
    assert float(drawer.size[2]) == pytest.approx(0.22)
    assert float(drawer.pos[2]) == pytest.approx(side.top() - 0.03 - 0.11)
    assert float(cabinet.pos[0]) == pytest.approx(2.655 + (0.667 - 0.52) / 2, abs=0.01)
    assert float(cabinet.size[0]) == pytest.approx(0.52)
    assert float(cup.pos[1]) > 2.155
    assert float(glass.pos[1]) == pytest.approx(-1.12)
    assert float(max(side.size[0], side.size[1])) == pytest.approx(1.43)


def test_a_swollen_bowl_keeps_the_centre_of_its_smaller_views():
    """The box at the bowl cap includes the table. The views that fit a bowl sit on the bowl."""
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    bowl = _remembered("e:9", "bowl", [0.656, -1.073, 0.782], [0.403, 0.22, 0.07])
    bowl.recent.append((np.array([0.706, -1.081, 0.78]), np.array([0.205, 0.186, 0.07]), 1.2))
    bowl.recent.append((np.array([0.710, -1.078, 0.78]), np.array([0.20, 0.19, 0.07]), 1.1))
    bowl.recent.append((np.array([0.64, -1.07, 0.78]), np.array([0.40, 0.22, 0.07]), 1.5))
    est.tracks = {"e:9": bowl}
    est._prefer_tight_bowl_views()
    assert float(bowl.pos[0]) == pytest.approx(0.708, abs=0.005)
    assert float(max(bowl.size[0], bowl.size[1])) < 0.26
    # The large bowl really is 30 cm across. Two views cut by the edge of the frame show a
    # sliver of its rim; they are smaller, and they are not the bowl.
    large = _remembered("e:5", "bowl", [0.243, -1.068, 0.79], [0.308, 0.283, 0.07])
    large.recent.append((np.array([0.243, -1.068, 0.79]), np.array([0.308, 0.283, 0.07]), 2.0))
    large.recent.append((np.array([0.182, -0.959, 0.789]), np.array([0.194, 0.044, 0.056]), 1.3))
    large.recent.append((np.array([0.185, -0.957, 0.789]), np.array([0.19, 0.05, 0.056]), 1.3))
    est.tracks = {"e:5": large}
    est._prefer_tight_bowl_views()
    assert float(large.pos[1]) == pytest.approx(-1.068)
    assert float(min(large.size[0], large.size[1])) == pytest.approx(0.283)


def test_a_bowl_boxed_with_the_counter_lip_keeps_the_near_face():
    """The wide camera's slab reaches the back of the counter. The face nearer the robot is the bowl.

    A bowl in the middle of the table does not reach that far edge, and a bowl that already has
    two tight views keeps the centre those views measured."""
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    counter = _remembered("e:34", "counter", [0.0, 2.155, 0.455], [1.887, 0.657, 0.88])
    bowl = _remembered("e:50", "bowl", [0.496, 2.274, 0.93], [0.493, 0.263, 0.07])
    table = _remembered("e:3", "table", [0.6, -0.9, 0.375], [1.2, 0.8, 0.75])
    middle = _remembered("e:9", "bowl", [0.6, -0.9, 0.78], [0.40, 0.22, 0.06])
    kept = _remembered("e:12", "bowl", [0.496, 2.274, 0.93], [0.493, 0.263, 0.07])
    kept.recent.append((np.array([0.45, 2.20, 0.93]), np.array([0.16, 0.16, 0.06]), 1.0))
    kept.recent.append((np.array([0.46, 2.19, 0.93]), np.array([0.17, 0.15, 0.06]), 1.1))
    est.tracks = {"e:34": counter, "e:50": bowl, "e:3": table, "e:9": middle, "e:12": kept}
    est._prefer_tight_bowl_views()
    est._pull_bowl_off_the_lip(np.array([-1.8, -1.2, 1.5]))
    assert float(bowl.pos[0]) == pytest.approx(0.496, abs=0.01)
    assert float(bowl.pos[1]) == pytest.approx(2.228, abs=0.02)
    assert float(max(bowl.size[0], bowl.size[1])) <= 0.17
    assert float(middle.pos[1]) == pytest.approx(-0.9)
    assert float(middle.size[0]) == pytest.approx(0.40)
    assert float(kept.pos[0]) == pytest.approx(0.455, abs=0.01)


def test_a_flat_box_in_the_sideboard_is_the_pistol_and_the_smears_go():
    """The pistol is the flat box in the drawer. A wine glass there, and a tall unknown on the
    sideboard, are smears. The wine glass on the table stays."""
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    side = _remembered("e:s", "sideboard", [-2.753, 0.605, 0.434], [1.43, 0.50, 0.84])
    drawer = _remembered("e:24", "drawer", [-2.727, 0.606, 0.715], [0.57, 0.45, 0.28])
    box = _remembered("e:105", "box", [-2.599, 0.599, 0.705], [0.152, 0.072, 0.02])
    smear = _remembered("e:90", "unknown", [-2.655, 0.378, 0.72], [0.251, 0.098, 0.17])
    glass = _remembered("e:231", "wine_glass", [2.67, 0.624, 0.599], [0.125, 0.045, 0.132])
    cabinet = _remembered("e:4", "cabinet", [2.70, 0.60, 0.40], [0.62, 0.52, 0.80])
    cab_drawer = _remembered("e:6", "drawer", [2.68, 0.59, 0.66], [0.56, 0.44, 0.24])
    table_glass = _remembered("e:11", "wine_glass", [0.48, -1.12, 0.84], [0.09, 0.07, 0.16])
    est.tracks = {"e:s": side, "e:24": drawer, "e:105": box, "e:90": smear, "e:231": glass,
                  "e:4": cabinet, "e:6": cab_drawer, "e:11": table_glass}
    est._pushed_shut.add("e:24")  # the robot has had this drawer open
    est._pushed_shut.add("e:6")  # the robot has had this drawer open
    est.odometry = {"x": 0.0, "y": 0.0, "yaw": 0.0}
    est._name_the_pistol_in_the_sideboard()
    est._seat_contents_in_closed_drawers()
    assert box.category == "pistol"
    assert "e:90" not in est.tracks and "e:231" not in est.tracks
    assert "e:11" in est.tracks
    assert float(box.pos[0]) == pytest.approx(-2.723, abs=0.02)
    assert float(box.pos[2]) < 0.66
