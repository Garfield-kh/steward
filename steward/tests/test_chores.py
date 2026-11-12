"""Chores task: drawers, nesting bowls, size-ordered rows, and one full round checked against World State."""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from steward.assets.generators import steward_bot
from steward.authoring.service import AuthoringService
from steward.chores import PHASES, Chores, row_offsets
from steward.ego.estimator import EgoEstimator
from steward.ego.planner import Box2D, protrusion
from steward.ego.runtime import EgoRuntime, RobotLink
from steward.ego.server import ego_api
from steward.mission import _approach_to_observe, _explore
from steward.world.chores_check import check_sideboard, check_swap, check_table, drawer_contents
from steward.world.console import console_state
from steward.world.server import StewardHost
from steward.world.sim import WorldSim


@pytest.fixture
def chores_scene(kitchen_root) -> dict:
    return AuthoringService(kitchen_root).load_scenario("phase0-chores")["sceneState"]


def _drawer_q(sim: WorldSim, entity: str) -> float:
    jid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, sim.build.names[entity].joints["drawer_slide"])
    return float(sim.data.qpos[sim.model.jnt_qposadr[jid]])


def _drive(sim: WorldSim, x: float, y: float, yaw: float, seconds: float) -> None:
    sim.command("w:robot-1", {"base": [x, y, yaw]})
    sim.step_seconds(seconds)


def test_scene_starts_with_items_in_both_drawers_and_tableware_on_the_table(chores_scene):
    sim = WorldSim(chores_scene)
    sim.step_seconds(1.0)
    support = {r["a"]["entity"]: r["b"] for r in sim.world_state()["relations"] if r["type"] == "supported-by"}
    assert support["w:teddy-bear"] == {"entity": "w:sideboard-1", "link": "drawer"}
    assert support["w:pistol"] == {"entity": "w:sideboard-1", "link": "drawer"}
    for item in ("w:toy-airplane", "w:pill-bottle", "w:perfume-bottle"):
        assert support[item] == {"entity": "w:cabinet-1", "link": "drawer"}
    for eid in ("w:bowl-red", "w:bowl-green", "w:bowl-blue", "w:wine-glass", "w:mug"):
        assert support[eid]["entity"] == "w:table-1"
    parts = {cat for ent in sim.template.entities for _, _, cat in sim.perception_parts(ent)}
    assert "drawer" in parts


def test_a_grasped_handle_slides_the_drawer_and_its_contents(chores_scene):
    sim = WorldSim(chores_scene)
    _drive(sim, -1.6, 0.6, 0.0, 6)
    _drive(sim, -1.6, 0.6, math.pi, 4)
    _drive(sim, -2.15, 0.6, math.pi, 5)
    sim.command("w:robot-1", {"lift": 0.706, "reach": 0.10})
    sim.step_seconds(3)
    sim.command("w:robot-1", {"gripper": "close"})
    sim.step_seconds(0.5)
    assert sim.events[-1]["type"] == "grasp/handle"
    assert sim.sensors("w:robot-1")["gripper"]["holdingHandle"]
    bear_x = sim._entity_pose("w:teddy-bear")["p"][0]
    for k in range(1, 31):  # pulled at the skills' drawer speed (0.3 m in about 0.9 s)
        sim.command("w:robot-1", {"base": [-2.15 + 0.01 * k, 0.6, math.pi]})
        sim.step_seconds(0.03)
    sim.step_seconds(2)
    assert _drawer_q(sim, "w:sideboard-1") == pytest.approx(0.30, abs=0.02)
    assert sim._entity_pose("w:teddy-bear")["p"][0] - bear_x == pytest.approx(0.30, abs=0.03)
    sim.command("w:robot-1", {"gripper": "open"})
    sim.step_seconds(0.5)
    assert sim.events[-1]["type"] == "grasp/handle-release"
    assert _drawer_q(sim, "w:sideboard-1") == pytest.approx(0.30, abs=0.02)
    categories = {d["category"] for d in sim.sensors("w:robot-1")["detections"]}
    assert "teddy_bear" in categories, "the open drawer shows what is inside"


def _det(category: str, p: list[float], size: list[float], color: str) -> dict:
    return {"category": category, "pose": {"p": p, "q": [1, 0, 0, 0]}, "size": size, "range": 1.0, "color": color}


def test_nested_bowls_stay_separate_tracks_and_know_what_holds_them():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    frame_odometry = {"x": 0.0, "y": 0.0, "yaw": 0.0, "sigma": {"pos": 0.005, "yaw": 0.005}}
    cam = est.camera_pose()
    inv = np.linalg.inv  # noqa: F841 - poses below are expressed through pose_compose
    from steward.math3d import pose_compose, pose_inverse

    def local(p):
        return pose_compose(pose_inverse(cam), {"p": p, "q": [1, 0, 0, 0]})["p"]

    board_top = 0.85
    world = [
        _det("sideboard", [1.5, 0.0, 0.425], [0.45, 1.4, 0.85], "beige"),
        _det("bowl", [1.5, 0.0, board_top + 0.035], [0.22, 0.22, 0.07], "red"),
        _det("bowl", [1.5, 0.0, board_top + 0.008 + 0.03], [0.17, 0.17, 0.06], "green"),
        _det("bowl", [1.5, 0.0, board_top + 0.016 + 0.025], [0.13, 0.13, 0.05], "blue"),
    ]
    for k in range(5):
        dets = [{**d, "pose": {"p": local(d["pose"]["p"]), "q": [1, 0, 0, 0]}} for d in world]
        est.update({"stamp": 0.1 * (k + 1), "odometry": frame_odometry, "joints": {"lift": 0.6, "reach": 0.1},
                    "gripper": {"state": "open", "holding": False}, "detections": dets})
    bowls = {t.color: t for t in est.tracks.values() if t.category == "bowl"}
    assert set(bowls) == {"red", "green", "blue"}
    board = next(t for t in est.tracks.values() if t.category == "sideboard")
    assert est.support_of(bowls["blue"]) is bowls["green"]
    assert est.support_of(bowls["green"]) is bowls["red"]
    assert est.support_of(bowls["red"]) is board


def test_support_of_nested_bowls_with_rim_biased_boxes():
    """Inner bowls seen only as a high thin rim must still nest: blue⊂green⊂red⊂sideboard."""
    from steward.ego.estimator import Track

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    board_top = 0.85
    # Rim bias: inner bowls are thin and centred near the outer rim, so bottom-bottom delta
    # exceeds the old 0.5×height nesting cap and support used to fall through to None/sideboard.

    def add(oid, cat, color, pos, size):
        est.tracks[oid] = Track(
            id=oid, category_counts={cat: 5.0}, pos=np.array(pos, float), var=np.full(3, 1e-4),
            yaw=0.0, yaw_var=1e-3, size=np.array(size, float), observations=5,
            first_seen=0.0, last_seen=1.0, existence=1.0, color_counts={color: 5.0},
        )

    add("e:board", "sideboard", "beige", [1.5, 0.0, board_top / 2], [0.45, 1.4, board_top])
    add("e:red", "bowl", "red", [1.5, 0.0, board_top + 0.035], [0.22, 0.22, 0.07])
    add("e:green", "bowl", "green", [1.5, 0.0, board_top + 0.065], [0.17, 0.17, 0.02])
    add("e:blue", "bowl", "blue", [1.5, 0.0, board_top + 0.078], [0.13, 0.13, 0.015])
    red, green, blue = est.tracks["e:red"], est.tracks["e:green"], est.tracks["e:blue"]
    board = est.tracks["e:board"]
    assert est.support_of(blue) is green, f"got {getattr(est.support_of(blue), 'id', None)}"
    assert est.support_of(green) is red, f"got {getattr(est.support_of(green), 'id', None)}"
    assert est.support_of(red) is board


def test_drawer_protrusion_is_measured_from_the_furniture_front():
    cabinet = Box2D("cab", 0.0, 0.0, 0.0, 0.25, 0.3)
    closed = Box2D("d", 0.02, 0.0, 0.0, 0.24, 0.25)
    opened = Box2D("d", 0.32, 0.0, 0.0, 0.24, 0.25)
    assert protrusion(closed, [cabinet]) == pytest.approx(0.01, abs=1e-6)
    assert protrusion(opened, [cabinet]) == pytest.approx(0.31, abs=1e-6)
    assert protrusion(opened, []) == 0.0


def test_row_offsets_space_items_by_width_around_the_centre():
    offsets = row_offsets([0.1, 0.2, 0.3], gap=0.05)
    assert offsets == pytest.approx([-0.30, -0.10, 0.20])


def test_a_chair_is_carried_and_put_down_on_the_floor(kitchen_root, tmp_path):
    from steward.net import call

    host = StewardHost(kitchen_root, scenario="phase0-chores", speed=6.0, seed=3, run_dir=tmp_path).start()
    record = AuthoringService(kitchen_root).library.get("chair")
    host.sim.edit_spawn("w:chair-1", record["template"], {"at": {"x": -0.25, "y": -0.9, "yaw": 0.0}}, {"id": "chair"})
    runtime = EgoRuntime(RobotLink(host.robot_servers["w:robot-1"].url), steward_bot().to_dict())
    runtime.start()
    api = ego_api(runtime).start()

    def run(skill: str, params: dict) -> dict:
        receipt = call("POST", f"{api.url}/ego/predict", {"skill": skill, "params": params})
        assert receipt["withinRiskLimit"], receipt["issues"]
        job = call("POST", f"{api.url}/skill/run", {"receipt": receipt["id"]})
        return call("POST", f"{api.url}/skill/wait", {"job": job["id"], "timeoutS": 120}, timeout=130)

    try:
        _explore(api.url, 120)
        chairs = [o for o in call("GET", f"{api.url}/ego/summary")["objects"] if o["category"] == "chair"]
        assert len(chairs) == 1
        _approach_to_observe(api.url, chairs[0])
        assert run("pick", {"target": chairs[0]["id"]})["state"] == "succeeded"
        assert run("place", {"at": [-2.05, 0.05]})["state"] == "succeeded"
        pose = host.sim._entity_pose("w:chair-1")
        assert pose["p"][0] == pytest.approx(-2.05, abs=0.04) and pose["p"][1] == pytest.approx(0.05, abs=0.04)
        assert pose["p"][2] == pytest.approx(0.0, abs=0.01), "the chair stands on the floor"
        chairs = [o for o in call("GET", f"{api.url}/ego/summary")["objects"] if o["category"] == "chair"]
        assert len(chairs) == 1, "carrying the chair must not leave phantom chairs in the ego state"
    finally:
        api.stop()
        runtime.stop()
        host.stop()


def test_picking_a_chair_by_the_wall_leaves_room_to_move_and_it_can_be_put_down_in_place(kitchen_root, tmp_path):
    from steward.net import call

    host = StewardHost(kitchen_root, scenario="phase0-chores", speed=6.0, seed=3, run_dir=tmp_path).start()
    record = AuthoringService(kitchen_root).library.get("chair")
    host.sim.edit_spawn("w:chair-2", record["template"], {"at": {"x": 1.5, "y": -0.9, "yaw": math.pi}}, {"id": "chair"})
    runtime = EgoRuntime(RobotLink(host.robot_servers["w:robot-1"].url), steward_bot().to_dict())
    runtime.start()
    api = ego_api(runtime).start()

    def predict(skill: str, params: dict) -> dict:
        return call("POST", f"{api.url}/ego/predict", {"skill": skill, "params": params})

    def run(receipt: dict) -> dict:
        job = call("POST", f"{api.url}/skill/run", {"receipt": receipt["id"]})
        return call("POST", f"{api.url}/skill/wait", {"job": job["id"], "timeoutS": 120}, timeout=130)

    try:
        _explore(api.url, 120)
        chair = next(o for o in call("GET", f"{api.url}/ego/summary")["objects"] if o["category"] == "chair")
        _approach_to_observe(api.url, chair)
        receipt = predict("pick", {"target": chair["id"]})
        assert receipt["withinRiskLimit"], receipt["issues"]
        assert receipt["plan"]["dock"][0] < 2.2, "the side between the chair and the east wall would box the robot in"
        job = run(receipt)
        assert job["state"] == "succeeded", (job["message"], job["log"])
        assert predict("navigate", {"x": -0.5, "y": 0.6})["withinRiskLimit"], "the robot can still move with the chair"
        held = next(o for o in call("GET", f"{api.url}/ego/summary")["objects"] if o["id"] == chair["id"])
        here = predict("place", {"at": held["position"][:2]})
        assert here["withinRiskLimit"] and here["plan"]["pathLength"] == 0.0, here["issues"]
        before = host.sim._entity_pose("w:chair-2")["p"]
        assert run(here)["state"] == "succeeded"
        after = host.sim._entity_pose("w:chair-2")["p"]
        assert math.dist(before[:2], after[:2]) < 0.04 and after[2] == pytest.approx(0.0, abs=0.01)
    finally:
        api.stop()
        runtime.stop()
        host.stop()


def test_a_stalled_motion_is_detected_within_the_window():
    from steward.ego.skills import _Stall

    stall = _Stall(0.0, 2.0, 0.01)
    assert not any(stall.stuck(t * 0.1, 1.0 - 0.05 * t) for t in range(15))
    assert not any(stall.stuck(1.4 + 0.25 * k, 0.3) for k in range(1, 9)), "still within the window"
    assert stall.stuck(1.4 + 0.25 * 9, 0.3)
    sparse = _Stall(0.0, 2.0, 0.01)
    assert not any(sparse.stuck(float(k), 0.5) for k in range(1, 5)), "a few coarse updates are not a stall"


def test_a_chair_blocking_the_sideboard_is_moved_out_of_the_way_before_the_drawer_swap(kitchen_root, tmp_path):
    host = StewardHost(kitchen_root, scenario="phase0-chores", speed=6.0, seed=3, run_dir=tmp_path).start()
    record = AuthoringService(kitchen_root).library.get("chair")
    host.sim.edit_spawn("w:chair-1", record["template"], {"at": {"x": -2.1, "y": 0.6, "yaw": math.pi}}, {"id": "chair"})
    runtime = EgoRuntime(RobotLink(host.robot_servers["w:robot-1"].url), steward_bot().to_dict())
    runtime.start()
    api = ego_api(runtime).start()
    steps: list[str] = []
    try:
        _explore(api.url, 120)
        before = drawer_contents(host.sim.world_state())  # contents are known once the simulation has stepped
        Chores(api.url, on_step=lambda r: steps.append(r["text"])).swap_drawers()
        ok, detail = check_swap(before, host.sim.world_state())
        assert ok, detail
        assert any(text.startswith("清障完成") for text in steps), steps
        chair = host.sim._entity_pose("w:chair-1")["p"]
        assert chair[0] < -2.3 or abs(chair[1] - 0.6) > 1.0, "the chair no longer stands in front of the sideboard"
    finally:
        api.stop()
        runtime.stop()
        host.stop()


def _piece(oid: str, category: str, xy: tuple[float, float], side: float, on: str | None = "e:1") -> dict:
    return {"id": oid, "category": category, "role": category, "position": [xy[0], xy[1], 0.8],
            "sizeM": [side, side, 0.06], "extentM": [side, side, 0.06], "existence": 1.0, "on": on}


def test_drawer_contents_are_carried_largest_first():
    """A place takes the free spot nearest the middle of the open part of a drawer: the toy airplane
    put there leaves room for a bottle on each side, two bottles put there leave none for it."""
    from steward.chores import _contents

    drawer = {"id": "e:1"}
    summary = {"objects": [_piece("e:2", "pill_bottle", (0.0, 0.0), 0.05), _piece("e:3", "toy_airplane", (0.1, 0.0), 0.22),
                           _piece("e:4", "perfume_bottle", (0.2, 0.0), 0.07), _piece("e:5", "bowl", (0.3, 0.0), 0.3, on="e:9")]}
    assert [o["id"] for o in _contents(summary, drawer)] == ["e:3", "e:4", "e:2"]


def test_a_piece_renumbered_when_it_is_put_down_is_found_again_where_it_was_put(monkeypatch):
    """With a real segmenter the track of a carried bowl is rebuilt under a new id when the gripper
    lets go. The bowl nested in another stands on the same spot as that one: the width tells them apart."""
    import steward.chores as chores_mod

    carried = _piece("e:7", "bowl", (1.0, -1.0), 0.20)
    base, nested, far = (_piece("e:20", "bowl", (-2.7, 0.6), 0.30), _piece("e:21", "bowl", (-2.71, 0.61), 0.21),
                         _piece("e:22", "bowl", (0.5, -1.0), 0.20))
    seen = {"objects": [base, far, _piece("e:23", "mug", (-2.7, 0.6), 0.2)]}
    monkeypatch.setattr(chores_mod, "_summary", lambda api: seen)
    # Only the wider bowl under it is seen there: the piece is reported where it was put, for a later look to find it by.
    assert chores_mod._landed("api", carried, [-2.7, 0.6]) == {**carried, "position": [-2.7, 0.6, 0.8]}
    seen["objects"].append(nested)
    assert chores_mod._landed("api", carried, [-2.7, 0.6])["id"] == "e:21"
    seen["objects"].append({**carried, "position": [-2.69, 0.6, 0.85]})
    assert chores_mod._landed("api", carried, [-2.7, 0.6])["position"] == [-2.69, 0.6, 0.85]


def test_a_repeating_failure_is_recognised_whatever_its_numbers():
    from steward.chores import _signature

    a = _signature("drawers", "cannot open e:10: ['dock pose in front of e:10 collides with e:5'] at 1.23")
    b = _signature("drawers", "cannot open e:12: ['dock pose in front of e:12 collides with e:7'] at -0.5")
    assert a == b
    assert a != _signature("table", "cannot open e:10: ['dock pose in front of e:10 collides with e:5'] at 1.23")


def test_a_round_without_progress_triggers_self_rescue_then_a_diagnosed_operator_alert(monkeypatch):
    import steward.chores as chores_mod
    from steward.mission import MissionFailed

    events: list[str] = []
    reports: list[dict] = []
    messages = iter(["navigate failed: base is blocked: no progress for 2 s (0.24 m from the waypoint)",
                     "navigate failed: turn did not converge (yaw error 0.14 rad)",
                     "open failed: missed the handle of e:30"] * 4)
    summary = {"self": {"x": 0.02, "y": -2.17, "yawDeg": 0.0, "holding": None},
               "objects": [{"id": "e:8", "category": "wall", "existence": 1.0, "position": [0.0, -2.5, 0.76],
                            "extentM": [5.8, 0.06, 1.48], "sizeM": [5.8, 0.1, 1.5], "yawDeg": 0.0,
                            "categoryConfidence": 1.0, "categories": {"wall": 1.0}}]}

    class Stop(Exception):
        pass

    def fail(self):
        self._doing("把红碗 e:9 放到餐桌第 1 位")
        raise MissionFailed(next(messages))

    def alert(api, report, on_wait=None):
        events.append("alert")
        reports.append(report)
        raise Stop(report["message"])

    monkeypatch.setattr(chores_mod, "_explore", lambda api, timeout_s: {"state": "succeeded"})
    monkeypatch.setattr(chores_mod, "_wait_for_control", lambda api, on_wait=None: None)
    monkeypatch.setattr(chores_mod, "_summary", lambda api: summary)
    monkeypatch.setattr(chores_mod, "_report_livelock", alert)
    monkeypatch.setattr(chores_mod.Chores, "publish", lambda self, **changes: events.append(changes.get("state", "")))
    monkeypatch.setattr(chores_mod.Chores, "recover", lambda self, message="": None)
    monkeypatch.setattr(chores_mod.Chores, "escape", lambda self: events.append("escape"))
    for name in ("arrange_on_table", "stack_on_sideboard", "swap_drawers"):
        monkeypatch.setattr(chores_mod.Chores, name, fail)
    with pytest.raises(Stop, match="没有任何进展"):
        chores_mod.chores("http://unused", 0)
    actions = [e for e in events if e in ("escape", "alert", "escaping", "waiting-operator")]
    assert actions == ["escaping", "escape", "waiting-operator", "alert"], events
    assert events.count("recovering") == 6, "three failed phases before the rescue and three after it"
    report = reports[0]
    assert "第 2 轮" in report["goal"] and report["step"] == "把红碗 e:9 放到餐桌第 1 位"
    assert "底盘受阻" in report["diagnosis"] and "6 次失败中 4 次" in report["diagnosis"]
    assert "基本没有移动" in report["diagnosis"] and "墙 e:8 0.30 米" in report["diagnosis"]
    assert "Ego State 里没有餐桌" in report["diagnosis"]


def test_one_chores_round_is_confirmed_by_world_state(kitchen_root, tmp_path):
    host = StewardHost(kitchen_root, scenario="phase0-chores", speed=10.0, seed=5, run_dir=tmp_path).start()
    runtime = EgoRuntime(RobotLink(host.robot_servers["w:robot-1"].url), steward_bot().to_dict())
    runtime.start()
    api = ego_api(runtime).start()
    try:
        def truth():
            return host.sim.world_state(), {g["id"]: g for g in console_state(host)["groundTruth"]}

        runner = Chores(api.url)
        _explore(api.url, 120)
        before = drawer_contents(truth()[0])  # contents are known once the simulation has stepped
        runner.arrange_on_table()
        ok, detail = check_table(*truth())
        assert ok, detail
        runner.stack_on_sideboard()
        ok, detail = check_sideboard(*truth())
        assert ok, detail
        runner.swap_drawers()
        ok, detail = check_swap(before, truth()[0])
        assert ok, detail
        assert PHASES == ("table", "sideboard", "drawers")
        assert host.observer.report()["w:robot-1"]["leaks"] == 0
    finally:
        api.stop()
        runtime.stop()
        host.stop()


def test_a_doubtful_or_missing_role_is_looked_at_from_closer_before_acting(monkeypatch):
    import steward.chores as chores_mod

    def board(confidence, role="sideboard"):
        return {"id": "e:21", "category": "counter", "role": role, "roleConfidence": confidence, "existence": 1.0,
                "categories": {"counter": 0.5, "sideboard": 0.45}, "position": [-2.7, 0.6, 0.43], "color": None}

    views = iter([{"objects": [board(0.45)]}, {"objects": [board(0.72)]}])
    looked: list[str] = []
    monkeypatch.setattr(chores_mod, "_summary", lambda api: next(views))
    monkeypatch.setattr(chores_mod, "_approach_to_observe", lambda api, obj, distance=1.1: looked.append(obj["id"]))
    monkeypatch.setattr(chores_mod, "simtime_sleep", lambda api, seconds: None)
    runner = Chores("http://unused")
    monkeypatch.setattr(runner, "publish", lambda **changes: None)
    assert runner._confirmed("sideboard")["id"] == "e:21" and looked == ["e:21"]
    assert "把握只有 0.45" in runner.attempt

    views = iter([{"objects": [board(0.9, role="counter")]}, {"objects": [board(0.8)]}])
    looked.clear()
    assert runner._confirmed("sideboard")["id"] == "e:21" and looked == ["e:21"], "the likeliest candidate is looked at"
    assert runner.attempt.startswith("寻找餐边柜")


def test_tableware_lying_on_the_floor_is_left_out_of_the_arrangement():
    from steward.chores import tableware

    def item(oid, role, z, extent):
        return {"id": oid, "category": role, "role": role, "existence": 1.0, "position": [0.0, 0.0, z], "extentM": extent}

    summary = {"objects": [item("e:1", "bowl", 0.79, [0.2, 0.2, 0.06]), item("e:2", "mug", 0.8, [0.1, 0.08, 0.1]),
                           item("e:3", "bowl", 0.02, [0.24, 0.24, 0.02])]}
    assert [o["id"] for o in tableware(summary)] == ["e:2", "e:1"], "the flat thing on the floor is not picked up"


def test_a_plate_lying_on_the_floor_blocks_but_is_not_offered_as_movable():
    from steward.ego.esim import Esim
    from steward.ego.estimator import Track

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    for oid, cat, pos, size in (("e:1", "chair", [1.0, 0.0, 0.45], [0.46, 0.46, 0.9]),
                                ("e:2", "plate", [2.0, 0.0, 0.01], [0.24, 0.24, 0.02])):
        est.tracks[oid] = Track(id=oid, category_counts={cat: 5.0}, pos=np.array(pos), var=np.full(3, 1e-4), yaw=0.0,
                                yaw_var=1e-3, size=np.array(size), observations=5, first_seen=0.0, last_seen=0.0,
                                existence=1.0)
    esim = Esim(steward_bot().to_dict())
    esim._blockers = {"e:1", "e:2"}
    blockers = {b["id"]: b for b in esim._describe_blockers(est.to_scene())}
    assert blockers["e:1"]["movable"] and not blockers["e:2"]["movable"] and blockers["e:2"]["parkAt"] == []


def test_a_role_nothing_resembles_is_searched_for_before_any_guess(monkeypatch):
    import steward.chores as chores_mod

    table = {"id": "e:51", "category": "table", "role": "table", "roleConfidence": 0.84, "existence": 1.0,
             "categories": {"table": 0.84}, "position": [0.9, -1.8, 0.39], "color": None}
    cabinet = {"id": "e:10", "category": "cabinet", "role": "cabinet", "roleConfidence": 0.85, "existence": 1.0,
               "categories": {"cabinet": 0.8, "table": 0.1}, "position": [2.7, 0.6, 0.41], "color": None}
    views = iter([{"objects": [cabinet]}, {"objects": [cabinet, table]}, {"objects": [cabinet, table]}])
    searched, looked = [], []
    monkeypatch.setattr(chores_mod, "_summary", lambda api: next(views))
    monkeypatch.setattr(chores_mod, "_approach_to_observe", lambda api, obj, distance=1.1: looked.append(obj["id"]))
    runner = Chores("http://unused")
    monkeypatch.setattr(runner, "publish", lambda **changes: None)
    monkeypatch.setattr(runner, "_search", lambda role: searched.append(role))
    assert runner._confirmed("table")["id"] == "e:51"
    assert searched == ["table"] and looked == [], "a cabinet that is 10 % table-like is not approached as the table"


def test_a_size_ordered_row_counts_from_either_end_of_the_table():
    def ent(eid, cat, x):
        return {"id": eid, "category": {"value": cat}, "pose": {"value": {"p": [x, 0.0, 0.75], "q": [1, 0, 0, 0]}}}

    def scene(xs):
        ents = [ent("w:table-1", "table", 0.0)] + [ent(e, c, x) for (e, c), x in zip(
            [("w:mug", "mug"), ("w:bowl-blue", "bowl"), ("w:bowl-red", "bowl")], xs)]
        return {"entities": ents, "relations": [{"type": "supported-by", "a": {"entity": e["id"]}, "b": {"entity": "w:table-1"}}
                                                for e in ents[1:]]}

    boxes = {"w:mug": {"size": [0.1, 0.08, 0.1]}, "w:bowl-blue": {"size": [0.12, 0.12, 0.05]}, "w:bowl-red": {"size": [0.3, 0.3, 0.08]}}
    assert check_table(scene([-0.3, 0.0, 0.3]), boxes)[0]
    assert check_table(scene([0.3, 0.0, -0.3]), boxes)[0], "smallest at the other end"
    assert not check_table(scene([0.0, -0.3, 0.3]), boxes)[0], "not in size order"


def _stack_bowl(oid: str, pos: list[float], size: list[float], *, color: str, on: str | None,
                role: str = "bowl") -> dict:
    return {
        "id": oid, "category": role, "role": role, "categories": {role: 0.9},
        "position": pos, "sizeM": size, "extentM": size, "color": color,
        "existence": 1.0, "on": on, "yawDeg": 0.0, "lastSeenAgoS": 0.0, "uncertaintyM": 0.02,
    }


def _sideboard_fixture():
    board = {
        "id": "e:board", "category": "sideboard", "role": "sideboard", "existence": 1.0,
        "position": [1.5, 0.0, 0.4], "extentM": [0.5, 1.2, 0.85], "sizeM": [0.5, 1.2, 0.85],
        "yawDeg": 0.0, "categories": {"sideboard": 1.0}, "color": None, "lastSeenAgoS": 0.0,
    }
    red = _stack_bowl("e:r", [1.5, 0.0, 0.88], [0.22, 0.22, 0.07], color="red", on="e:board")
    green = _stack_bowl("e:g", [1.5, 0.0, 0.92], [0.17, 0.17, 0.06], color="green", on="e:r")
    blue = _stack_bowl("e:b", [1.5, 0.2, 0.88], [0.13, 0.13, 0.05], color="blue", on="e:table")
    wine = {
        "id": "e:w", "category": "wine_glass", "role": "wine_glass", "categories": {"wine_glass": 0.9},
        "position": [1.7, 0.0, 0.9], "sizeM": [0.06, 0.06, 0.15], "extentM": [0.06, 0.06, 0.15],
        "color": "clear", "existence": 1.0, "on": "e:table", "yawDeg": 0.0, "lastSeenAgoS": 0.0,
        "uncertaintyM": 0.02,
    }
    return board, red, green, blue, wine


def _patch_sideboard_runner(monkeypatch, chores_mod, state, board, *, move=None, table=None):
    from steward.mission import MissionFailed

    def live(obj):
        return next(o for o in state["objects"] if o["id"] == obj["id"])

    def fake_refresh(api, obj):
        try:
            return dict(live(obj))
        except StopIteration as exc:
            raise MissionFailed(f"{obj['id']} gone") from exc

    monkeypatch.setattr(chores_mod, "_summary", lambda api: state)
    # Unit fixtures already hold a full room; do not burn wall time in ``_settle``.
    monkeypatch.setattr(chores_mod, "_settle", lambda api, seconds=25.0, want=2: state)
    monkeypatch.setattr(chores_mod, "_refresh", fake_refresh)
    monkeypatch.setattr(chores_mod, "_refresh_or_recover", fake_refresh)
    monkeypatch.setattr(chores_mod, "_approach_to_observe", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "simtime_sleep", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "_place_at",
                        lambda api, support, xy: {"on": support["id"], "offset": [0.0, 0.0]})
    if move is None:
        def move(api, obj, place, observe, log, timeout_s, attempts=2):
            return {"state": "succeeded"}
    monkeypatch.setattr(chores_mod, "_move", move)
    runner = chores_mod.Chores("http://unused")

    def confirmed(role):
        if role == "table" and table is not None:
            return table
        return board

    monkeypatch.setattr(runner, "_confirmed", confirmed)
    monkeypatch.setattr(runner, "_doing", lambda *a, **k: None)
    monkeypatch.setattr(runner, "publish", lambda **k: None)
    monkeypatch.setattr(runner, "_clearing", lambda phase, action: action())
    return runner


def test_sideboard_plan_keeps_unsupported_tableware_near_the_table():
    """Scheme 1: missing ``on`` does not drop fresh tableware that sits over the table.

    ``_supported_first`` would keep only the mug once it has ``on``, and omit the wine glass
    whose support is still unknown — then the main loop never moves it."""
    from steward.chores import _sideboard_plan_bowls, _sideboard_plan_others, _sideboard_plan_inventory

    table = {
        "id": "e:table", "category": "table", "role": "table", "existence": 1.0,
        "position": [0.5, -1.0, 0.38], "extentM": [1.2, 0.8, 0.75], "sizeM": [1.2, 0.8, 0.75],
        "yawDeg": 0.0, "categories": {"table": 1.0}, "color": None, "lastSeenAgoS": 0.0, "on": None,
    }
    board, red, green, blue, wine = _sideboard_fixture()
    mug = {
        "id": "e:m", "category": "mug", "role": "mug", "categories": {"mug": 0.9},
        "position": [0.4, -1.0, 0.80], "sizeM": [0.08, 0.08, 0.09], "extentM": [0.08, 0.08, 0.09],
        "color": "black", "existence": 1.0, "on": "e:table", "yawDeg": 0.0, "lastSeenAgoS": 0.0,
        "uncertaintyM": 0.02,
    }
    # Far from every support surface: a wall-corner fragment misread as a cup.
    fragment = {
        "id": "e:frag", "category": "cup", "role": "cup", "categories": {"cup": 0.8},
        "position": [5.0, 5.0, 0.9], "sizeM": [0.07, 0.07, 0.08], "extentM": [0.07, 0.07, 0.08],
        "color": None, "existence": 1.0, "on": None, "yawDeg": 0.0, "lastSeenAgoS": 0.0,
        "uncertaintyM": 0.02,
    }
    red = {**red, "on": "e:table", "position": [0.3, -1.0, 0.78]}
    green = {**green, "on": "e:table", "position": [0.5, -1.0, 0.78]}
    blue = {**blue, "on": None, "position": [0.7, -1.0, 0.78]}
    wine = {**wine, "on": None, "position": [0.9, -1.0, 0.80]}
    summary = {"objects": [table, board, red, green, blue, wine, mug, fragment]}
    assert {b["id"] for b in _sideboard_plan_bowls(summary)} == {"e:r", "e:g", "e:b"}
    assert {o["id"] for o in _sideboard_plan_others(summary)} == {"e:w", "e:m"}
    assert "e:frag" not in {o["id"] for o in _sideboard_plan_others(summary)}
    _bowls, _others, rejected = _sideboard_plan_inventory(summary)
    by_id = {r["id"]: r for r in rejected}
    assert by_id["e:frag"]["reasons"] == ["far"]


def test_doubtful_tableware_includes_stemware_shaped_unknown_and_role_rejects():
    """Rescan targets: unlabeled stemware geometry, and role-rejected tracks with wine mass."""
    from steward.chores import _doubtful_tableware_for_rescan

    table = {
        "id": "e:table", "category": "table", "role": "table", "existence": 1.0,
        "position": [0.5, -1.0, 0.38], "extentM": [1.2, 0.8, 0.75], "sizeM": [1.2, 0.8, 0.75],
        "yawDeg": 0.0, "categories": {"table": 1.0}, "color": None, "lastSeenAgoS": 0.0, "on": None,
    }
    board, red, _green, _blue, wine = _sideboard_fixture()
    red = {**red, "on": "e:table", "position": [0.3, -1.0, 0.78]}
    # Beige stemware with no tableware mass yet (seed4-style): not a plan candidate, still rescanned.
    ghost = {
        "id": "e:ghost", "category": "unknown", "role": "unknown",
        "categories": {"unknown": 1.0},
        "position": [0.49, -1.12, 0.85], "sizeM": [0.08, 0.07, 0.19], "extentM": [0.08, 0.07, 0.19],
        "color": "beige", "existence": 1.0, "on": "e:table", "yawDeg": 0.0, "lastSeenAgoS": 0.0,
    }
    misrole = {
        "id": "e:mis", "category": "wine_glass", "role": "unknown",
        "categories": {"wine_glass": 0.7, "unknown": 0.3},
        "position": [0.7, -1.0, 0.80], "sizeM": [0.06, 0.06, 0.15], "extentM": [0.06, 0.06, 0.15],
        "color": "clear", "existence": 1.0, "on": "e:table", "yawDeg": 0.0, "lastSeenAgoS": 0.0,
    }
    far_unknown = {
        "id": "e:far", "category": "unknown", "role": "unknown", "categories": {"unknown": 1.0},
        "position": [3.0, 3.0, 0.85], "sizeM": [0.08, 0.07, 0.19], "extentM": [0.08, 0.07, 0.19],
        "color": "beige", "existence": 1.0, "on": None, "yawDeg": 0.0, "lastSeenAgoS": 0.0,
    }
    summary = {"objects": [table, board, red, ghost, misrole, far_unknown]}
    ids = {o["id"] for o in _doubtful_tableware_for_rescan(summary)}
    assert ids == {"e:ghost", "e:mis"}


def _stack_rescan_harness(monkeypatch, chores_mod, *, ghost_states: list[dict], board, table,
                          red, green, blue):
    """Drive stack_on_sideboard with a scripted ghost identity across close-look summaries."""
    state = {"i": 0, "looked": []}

    def objects_now():
        ghost = ghost_states[min(state["i"], len(ghost_states) - 1)]
        return [table, board, red, green, blue, ghost]

    def fake_summary(api):
        return {"objects": objects_now()}

    def fake_settle(api, seconds=25.0, want=2):
        return {"objects": objects_now()}

    def fake_approach(api, obj, distance=1.3, prefer_offset=None):
        state["looked"].append({"id": obj["id"], "distance": distance, "prefer_offset": prefer_offset})
        if obj["id"] == "e:ghost" and state["i"] + 1 < len(ghost_states):
            state["i"] += 1

    monkeypatch.setattr(chores_mod, "_summary", fake_summary)
    monkeypatch.setattr(chores_mod, "_settle", fake_settle)
    monkeypatch.setattr(chores_mod, "_refresh_or_recover",
                        lambda api, obj: board if obj.get("role") == "sideboard"
                        else next(o for o in objects_now() if o["id"] == obj["id"]))
    monkeypatch.setattr(chores_mod, "_approach_to_observe", fake_approach)
    monkeypatch.setattr(chores_mod, "simtime_sleep", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "_move", lambda *a, **k: {"state": "succeeded"})
    monkeypatch.setattr(chores_mod, "_place_at",
                        lambda api, support, xy: {"on": support["id"], "offset": [0.0, 0.0]})
    runner = chores_mod.Chores("http://unused")
    monkeypatch.setattr(runner, "_confirmed", lambda role: board)
    monkeypatch.setattr(runner, "_doing", lambda *a, **k: None)
    monkeypatch.setattr(runner, "_step", lambda *a, **k: None)
    monkeypatch.setattr(runner, "_clearing", lambda phase, action: action())
    monkeypatch.setattr(runner, "_with_known_support", lambda obj, expect=None: obj)
    runner.stack_on_sideboard()
    return runner, state["looked"]


def test_stack_rescans_doubtful_unknown_before_planning(monkeypatch):
    """Stemware unknown that becomes plan-ready after the first close look stops early and is planned."""
    import steward.chores as chores_mod

    table = {
        "id": "e:table", "category": "table", "role": "table", "existence": 1.0,
        "position": [0.5, -1.0, 0.38], "extentM": [1.2, 0.8, 0.75], "sizeM": [1.2, 0.8, 0.75],
        "yawDeg": 0.0, "categories": {"table": 1.0}, "color": None, "lastSeenAgoS": 0.0, "on": None,
    }
    board, red, green, blue, wine = _sideboard_fixture()
    red = {**red, "on": "e:table", "position": [0.3, -1.0, 0.78]}
    green = {**green, "on": "e:table", "position": [0.5, -1.0, 0.78]}
    blue = {**blue, "on": "e:table", "position": [0.7, -1.0, 0.78]}
    ghost = {
        "id": "e:ghost", "category": "unknown", "role": "unknown", "categories": {"unknown": 1.0},
        "position": [0.49, -1.12, 0.85], "sizeM": [0.08, 0.07, 0.19], "extentM": [0.08, 0.07, 0.19],
        "color": "beige", "existence": 1.0, "on": "e:table", "yawDeg": 0.0, "lastSeenAgoS": 0.0,
    }
    named = {**wine, "id": "e:ghost", "category": "wine_glass", "role": "wine_glass",
             "categories": {"wine_glass": 0.95}, "on": "e:table",
             "position": [0.49, -1.12, 0.85], "sizeM": [0.08, 0.07, 0.19], "extentM": [0.08, 0.07, 0.19],
             "lastSeenAgoS": 0.0}
    runner, looked = _stack_rescan_harness(
        monkeypatch, chores_mod, ghost_states=[ghost, named], board=board, table=table,
        red=red, green=green, blue=blue)
    ghost_looks = [e for e in looked if e["id"] == "e:ghost"]
    assert len(ghost_looks) == 1
    assert ghost_looks[0]["prefer_offset"] == 0.0
    assert any(e.get("step") == "rescan-object" and e.get("id") == "e:ghost" and e.get("confirmed")
               and e.get("views") == 1 for e in runner.log)
    plan = next(e for e in runner.log if e.get("step") == "plan-sideboard")
    assert any(o["id"] == "e:ghost" and o["role"] == "wine_glass" for o in plan["others"])


def test_stack_rescans_stop_when_any_plan_tableware_role_appears(monkeypatch):
    """Early stop uses plan-ready TABLEWARE (e.g. mug), not a wine_glass-only check."""
    import steward.chores as chores_mod

    table = {
        "id": "e:table", "category": "table", "role": "table", "existence": 1.0,
        "position": [0.5, -1.0, 0.38], "extentM": [1.2, 0.8, 0.75], "sizeM": [1.2, 0.8, 0.75],
        "yawDeg": 0.0, "categories": {"table": 1.0}, "color": None, "lastSeenAgoS": 0.0, "on": None,
    }
    board, red, green, blue, _wine = _sideboard_fixture()
    red = {**red, "on": "e:table", "position": [0.3, -1.0, 0.78]}
    green = {**green, "on": "e:table", "position": [0.5, -1.0, 0.78]}
    blue = {**blue, "on": "e:table", "position": [0.7, -1.0, 0.78]}
    ghost = {
        "id": "e:ghost", "category": "unknown", "role": "unknown", "categories": {"unknown": 1.0},
        "position": [0.49, -1.12, 0.85], "sizeM": [0.08, 0.07, 0.19], "extentM": [0.08, 0.07, 0.19],
        "color": "beige", "existence": 1.0, "on": "e:table", "yawDeg": 0.0, "lastSeenAgoS": 0.0,
    }
    as_mug = {
        "id": "e:ghost", "category": "mug", "role": "mug", "categories": {"mug": 0.9},
        "position": [0.49, -1.12, 0.85], "sizeM": [0.1, 0.08, 0.1], "extentM": [0.1, 0.08, 0.1],
        "color": "black", "existence": 1.0, "on": "e:table", "yawDeg": 0.0, "lastSeenAgoS": 0.0,
    }
    # Still unknown after view 0; mug after view 1.
    runner, looked = _stack_rescan_harness(
        monkeypatch, chores_mod, ghost_states=[ghost, ghost, as_mug], board=board, table=table,
        red=red, green=green, blue=blue)
    ghost_looks = [e for e in looked if e["id"] == "e:ghost"]
    assert len(ghost_looks) == 2
    assert [e["prefer_offset"] for e in ghost_looks] == [0.0, 1.0]
    plan = next(e for e in runner.log if e.get("step") == "plan-sideboard")
    assert any(o["id"] == "e:ghost" and o["role"] == "mug" for o in plan["others"])


def test_stack_leaves_unconfirmed_doubtful_out_of_others_after_max_views(monkeypatch):
    """After RESCAN_MAX_VIEWS without a plan-ready role, the track is not in others."""
    import steward.chores as chores_mod

    table = {
        "id": "e:table", "category": "table", "role": "table", "existence": 1.0,
        "position": [0.5, -1.0, 0.38], "extentM": [1.2, 0.8, 0.75], "sizeM": [1.2, 0.8, 0.75],
        "yawDeg": 0.0, "categories": {"table": 1.0}, "color": None, "lastSeenAgoS": 0.0, "on": None,
    }
    board, red, green, blue, _wine = _sideboard_fixture()
    red = {**red, "on": "e:table", "position": [0.3, -1.0, 0.78]}
    green = {**green, "on": "e:table", "position": [0.5, -1.0, 0.78]}
    blue = {**blue, "on": "e:table", "position": [0.7, -1.0, 0.78]}
    ghost = {
        "id": "e:ghost", "category": "unknown", "role": "unknown", "categories": {"unknown": 1.0},
        "position": [0.49, -1.12, 0.85], "sizeM": [0.08, 0.07, 0.19], "extentM": [0.08, 0.07, 0.19],
        "color": "beige", "existence": 1.0, "on": "e:table", "yawDeg": 0.0, "lastSeenAgoS": 0.0,
    }
    runner, looked = _stack_rescan_harness(
        monkeypatch, chores_mod, ghost_states=[ghost, ghost, ghost, ghost], board=board, table=table,
        red=red, green=green, blue=blue)
    ghost_looks = [e for e in looked if e["id"] == "e:ghost"]
    assert len(ghost_looks) == chores_mod.RESCAN_MAX_VIEWS
    assert [e["prefer_offset"] for e in ghost_looks] == list(chores_mod.RESCAN_VIEW_OFFSETS)
    assert any(e.get("step") == "rescan-object" and e.get("id") == "e:ghost" and not e.get("confirmed")
               and e.get("views") == chores_mod.RESCAN_MAX_VIEWS for e in runner.log)
    plan = next(e for e in runner.log if e.get("step") == "plan-sideboard")
    assert all(o["id"] != "e:ghost" for o in plan["others"])
    rejected = {r["id"]: r for r in plan["rejected"]}
    assert "e:ghost" in rejected
    assert "unconfirmed" in rejected["e:ghost"]["reasons"]
    assert rejected["e:ghost"]["views"] == chores_mod.RESCAN_MAX_VIEWS


def test_move_rebinds_place_on_after_support_renumber(monkeypatch):
    """chores._move refreshes the support and rewrites place.on before predicting place."""
    import steward.chores as chores_mod

    obj = {
        "id": "e:8", "category": "bowl", "role": "bowl", "categories": {"bowl": 1.0},
        "position": [0.0, 0.0, 0.9], "sizeM": [0.12, 0.12, 0.05], "extentM": [0.12, 0.12, 0.05],
        "color": "blue", "existence": 1.0, "on": "e:table",
    }
    stale = {
        "id": "e:31", "category": "bowl", "role": "bowl", "categories": {"bowl": 1.0},
        "position": [-2.7, 0.5, 0.9], "sizeM": [0.19, 0.19, 0.06], "extentM": [0.19, 0.19, 0.06],
        "color": "green", "existence": 1.0, "on": "e:7",
    }
    fresh = {**stale, "id": "e:484"}
    places: list[dict] = []

    monkeypatch.setattr(chores_mod, "_approach_to_observe", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "_refresh", lambda api, o: obj if o["id"] == "e:8" else o)
    monkeypatch.setattr(chores_mod, "_refresh_or_recover",
                        lambda api, o: fresh if o["id"] == "e:31" else o)

    def fake_safe(api, skill, params, observe, log):
        if skill == "pick":
            return {"id": "pick", "feasible": True, "withinRiskLimit": True, "issues": [],
                    "blockers": [], "risk": 0.0, "skill": "pick"}
        places.append(dict(params))
        assert observe["id"] == "e:484" and params["on"] == "e:484"
        return {"id": "place", "feasible": True, "withinRiskLimit": True, "issues": [],
                "blockers": [], "risk": 0.0, "skill": "place", "plan": {"spot": [0, 0]}}

    monkeypatch.setattr(chores_mod, "_safe_receipt", fake_safe)
    monkeypatch.setattr(chores_mod, "_run", lambda api, receipt, timeout_s: {"state": "succeeded", "message": "ok"})
    monkeypatch.setattr(chores_mod, "simtime_sleep", lambda *a, **k: None)
    log: list = []
    chores_mod._move("http://unused", obj, {"on": "e:31"}, stale, log, timeout_s=30.0)
    assert places == [{"on": "e:484"}]


def test_sideboard_plan_rejected_rows_name_stale_role_and_floor():
    """plan-sideboard diagnostics: each omitted tableware-like track carries stable reason codes."""
    from steward.chores import FRESH_S, _sideboard_plan_inventory

    table = {
        "id": "e:table", "category": "table", "role": "table", "existence": 1.0,
        "position": [0.5, -1.0, 0.38], "extentM": [1.2, 0.8, 0.75], "sizeM": [1.2, 0.8, 0.75],
        "yawDeg": 0.0, "categories": {"table": 1.0}, "color": None, "lastSeenAgoS": 0.0, "on": None,
    }
    board, red, _green, _blue, wine = _sideboard_fixture()
    red = {**red, "on": "e:table", "position": [0.3, -1.0, 0.78]}
    stale = {**wine, "id": "e:stale", "on": "e:table", "position": [0.6, -1.0, 0.80],
             "lastSeenAgoS": FRESH_S + 1.0}
    misrole = {
        "id": "e:mis", "category": "wine_glass", "role": "unknown",
        "categories": {"wine_glass": 0.7, "unknown": 0.3},
        "position": [0.7, -1.0, 0.80], "sizeM": [0.06, 0.06, 0.15], "extentM": [0.06, 0.06, 0.15],
        "color": "clear", "existence": 1.0, "on": "e:table", "yawDeg": 0.0, "lastSeenAgoS": 0.0,
    }
    floor = {
        "id": "e:mat", "category": "bowl", "role": "bowl", "categories": {"bowl": 0.9},
        "position": [0.5, -1.0, 0.02], "sizeM": [0.24, 0.24, 0.02], "extentM": [0.24, 0.24, 0.02],
        "color": None, "existence": 1.0, "on": None, "yawDeg": 0.0, "lastSeenAgoS": 0.0,
    }
    summary = {"objects": [table, board, red, stale, misrole, floor]}
    bowls, others, rejected = _sideboard_plan_inventory(summary)
    assert [b["id"] for b in bowls] == ["e:r"] and others == []
    by_id = {r["id"]: r["reasons"] for r in rejected}
    assert by_id["e:stale"] == ["stale"]
    assert by_id["e:mis"] == ["role"]
    assert by_id["e:mat"] == ["floor"]


def test_sideboard_plan_includes_tableware_once_a_close_look_gives_support(monkeypatch):
    """Open-plan after observing the table: pieces that lacked ``on`` are moved via pick-time look."""
    import steward.chores as chores_mod

    table = {
        "id": "e:table", "category": "table", "role": "table", "existence": 1.0,
        "position": [0.5, -1.0, 0.38], "extentM": [1.2, 0.8, 0.75], "sizeM": [1.2, 0.8, 0.75],
        "yawDeg": 0.0, "categories": {"table": 1.0}, "color": None, "lastSeenAgoS": 0.0, "on": None,
    }
    board, red, green, blue, wine = _sideboard_fixture()
    red = {**red, "on": "e:table", "position": [0.3, -1.0, 0.78]}
    green = {**green, "on": "e:table", "position": [0.5, -1.0, 0.78]}
    blue = {**blue, "on": None, "position": [0.7, -1.0, 0.78]}
    wine = {**wine, "on": None, "position": [0.9, -1.0, 0.80]}
    state = {"objects": [table, board, red, green, blue, wine]}
    looked: list[str] = []

    def approach(api, obj, distance=1.3):
        looked.append(obj["id"])
        # Only fill support when looking at the piece itself (pick-time), not the table overview.
        if obj["id"] in ("e:b", "e:w"):
            for o in state["objects"]:
                if o["id"] == obj["id"] and o.get("on") is None:
                    o["on"] = "e:table"

    def move(api, obj, place, observe, log, timeout_s, attempts=2):
        for o in state["objects"]:
            if o["id"] == obj["id"] and place.get("on"):
                o["on"] = place["on"]
        return {"state": "succeeded"}

    runner = _patch_sideboard_runner(monkeypatch, chores_mod, state, board, move=move, table=table)
    monkeypatch.setattr(chores_mod, "_approach_to_observe", approach)
    runner.stack_on_sideboard()
    plan = next(e for e in runner.log if e.get("step") == "plan-sideboard")
    assert looked[0] == "e:table"
    assert {b["id"] for b in plan["bowls"]} == {"e:r", "e:g", "e:b"}
    assert {o["id"] for o in plan["others"]} == {"e:w"}
    # Plan listed wine while its ``on`` was still None; pick-time look then placed it.
    assert any(oid == "e:w" for oid in looked[1:])
    assert next(o for o in state["objects"] if o["id"] == "e:b")["on"] == "e:g"
    assert next(o for o in state["objects"] if o["id"] == "e:w")["on"] == "e:board"


def test_stack_looks_again_when_support_is_unknown(monkeypatch):
    """M5: before moving a bowl with on=None, look closer once so support can appear."""
    import steward.chores as chores_mod

    board = {
        "id": "e:board", "category": "sideboard", "role": "sideboard", "existence": 1.0,
        "position": [1.5, 0.0, 0.4], "extentM": [0.5, 1.2, 0.85], "sizeM": [0.5, 1.2, 0.85],
        "yawDeg": 0.0, "categories": {"sideboard": 1.0}, "color": None,
    }
    red = _stack_bowl("e:1", [1.5, 0.0, 0.88], [0.22, 0.22, 0.07], color="red", on=None)
    red_supported = {**red, "on": "e:board"}
    looked: list[str] = []
    state = {"objects": [board, red]}

    def fake_refresh(api, obj):
        live = next(o for o in state["objects"] if o["id"] == obj["id"])
        return live

    def fake_approach(api, obj, distance=0.9):
        looked.append(obj["id"])
        state["objects"] = [board, red_supported]

    monkeypatch.setattr(chores_mod, "_refresh_or_recover", fake_refresh)
    monkeypatch.setattr(chores_mod, "_approach_to_observe", fake_approach)
    monkeypatch.setattr(chores_mod, "simtime_sleep", lambda *a, **k: None)
    runner = chores_mod.Chores("http://unused")
    monkeypatch.setattr(runner, "_doing", lambda *a, **k: None)
    out = runner._with_known_support(red)
    assert looked == ["e:1"] and out["on"] == "e:board"


def test_refresh_alone_misses_a_bowl_that_drifted_beyond_local_reach(monkeypatch):
    """M2 before: M1 ``_refresh`` only searches ~0.25 m; a renumbered bowl a metre away is lost."""
    from steward import mission as mission_mod
    from steward.mission import MissionFailed, _refresh

    lost = _stack_bowl("e:2", [1.5, 0.0, 0.9], [0.17, 0.17, 0.06], color="green", on="e:1")
    far = _stack_bowl("e:20", [2.5, 0.0, 0.9], [0.17, 0.16, 0.06], color="green", on="e:1")
    monkeypatch.setattr(mission_mod, "call", lambda *a, **k: {"objects": [far]})
    with pytest.raises(MissionFailed, match="e:2"):
        _refresh("http://unused", lost)


def test_refresh_or_recover_finds_a_renumbered_bowl_beyond_local_reach(monkeypatch):
    """M2 after: look again, then match footprint anywhere in the summary."""
    import steward.chores as chores_mod
    from steward import mission as mission_mod

    lost = _stack_bowl("e:2", [1.5, 0.0, 0.9], [0.17, 0.17, 0.06], color="green", on="e:1")
    far = _stack_bowl("e:20", [2.5, 0.0, 0.9], [0.17, 0.16, 0.06], color="green", on="e:1")
    looked = []
    monkeypatch.setattr(mission_mod, "call", lambda *a, **k: {"objects": [far]})
    monkeypatch.setattr(chores_mod, "_approach_to_observe", lambda api, obj, distance=0.9: looked.append(obj["id"]))
    monkeypatch.setattr(chores_mod, "_summary", lambda api: {"objects": [far]})
    found = chores_mod._refresh_or_recover("http://unused", lost)
    assert found["id"] == "e:20" and looked == ["e:2"]


def test_stack_recovers_renumbered_bowl_mid_stack(monkeypatch):
    """Mid-stack lost id: recover the renumbered bowl and continue instead of MissionFailed."""
    import steward.chores as chores_mod
    from steward.mission import MissionFailed

    board = {
        "id": "e:board", "category": "sideboard", "role": "sideboard", "existence": 1.0,
        "position": [1.5, 0.0, 0.4], "extentM": [0.5, 1.2, 0.85], "sizeM": [0.5, 1.2, 0.85],
        "yawDeg": 0.0, "categories": {"sideboard": 1.0}, "color": None,
    }
    red = _stack_bowl("e:1", [1.5, 0.0, 0.88], [0.22, 0.22, 0.07], color="red", on="e:board")
    # Already on the board so the largest bowl needs no move; green is missing until recover.
    green_old = _stack_bowl("e:2", [1.5, 0.2, 0.88], [0.17, 0.17, 0.06], color="green", on="e:board")
    green_new = _stack_bowl("e:20", [2.5, 0.2, 0.88], [0.17, 0.16, 0.06], color="green", on="e:board")
    blue = _stack_bowl("e:3", [1.5, -0.2, 0.88], [0.13, 0.13, 0.05], color="blue", on="e:board")

    # Initial plan sees green_old; refresh of e:2 fails; summary for recover has green_new far away.
    plan_summary = {"objects": [board, red, green_old, blue]}
    recover_summary = {"objects": [board, red, green_new, blue]}
    moves: list[tuple] = []

    def fake_refresh(api, obj):
        if obj["id"] == "e:2":
            raise MissionFailed("e:2 (bowl) is no longer in the ego state")
        live = next((o for o in recover_summary["objects"] if o["id"] == obj["id"]), None)
        if live is None:
            raise MissionFailed(f"{obj['id']} ({obj.get('category')}) is no longer in the ego state")
        return live

    def fake_move(api, obj, place, observe, log, timeout_s, attempts=2):
        moves.append((obj["id"], dict(place)))
        for o in recover_summary["objects"]:
            if o["id"] == obj["id"] and place.get("on"):
                o["on"] = place["on"]
        return {"state": "succeeded"}

    monkeypatch.setattr(chores_mod, "_summary",
                        lambda api: recover_summary if moves else plan_summary)
    monkeypatch.setattr(chores_mod, "_settle",
                        lambda api, seconds=25.0, want=2: recover_summary if moves else plan_summary)
    monkeypatch.setattr(chores_mod, "_refresh", fake_refresh)
    monkeypatch.setattr(chores_mod, "_approach_to_observe", lambda *a, **k: None)
    # After look, recover reads the summary that holds the renumbered track.
    monkeypatch.setattr(chores_mod, "_match_by_footprint",
                        lambda summary, obj: green_new if obj["id"] == "e:2" else None)
    # Force _refresh_or_recover to use our failing refresh + match path.
    def recover(api, obj):
        try:
            return fake_refresh(api, obj)
        except MissionFailed as first:
            found = chores_mod._match_by_footprint(recover_summary, obj)
            if found is not None:
                return found
            raise first

    monkeypatch.setattr(chores_mod, "_refresh_or_recover", recover)
    monkeypatch.setattr(chores_mod, "_move", fake_move)
    monkeypatch.setattr(chores_mod, "simtime_sleep", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "_place_at",
                        lambda api, support, xy: {"on": support["id"], "offset": [0.0, 0.0]})
    runner = chores_mod.Chores("http://unused")
    monkeypatch.setattr(runner, "_confirmed", lambda role: board)
    monkeypatch.setattr(runner, "_doing", lambda *a, **k: None)
    monkeypatch.setattr(runner, "_step", lambda *a, **k: None)
    monkeypatch.setattr(runner, "_clearing", lambda phase, action: action())
    runner.stack_on_sideboard()
    assert any(oid == "e:20" for oid, _ in moves), f"expected move of recovered bowl, got {moves}"
    assert any(place.get("on") == "e:1" for _, place in moves), "blue/green should nest onto the red base"


def test_a_timed_out_request_fails_the_phase_instead_of_ending_the_loop(monkeypatch):
    import steward.chores as chores_mod

    failures: list[str] = []

    class Stop(Exception):
        pass

    def timed_out(self):
        raise TimeoutError("timed out")

    def stop_after_failures(index, phase, message):
        failures.append(message)
        if len(failures) == 3:
            raise Stop

    monkeypatch.setattr(chores_mod, "_explore", lambda api, timeout_s: {"state": "succeeded"})
    monkeypatch.setattr(chores_mod, "_wait_for_control", lambda api, on_wait=None: None)
    monkeypatch.setattr(chores_mod.Chores, "publish", lambda self, **changes: None)
    monkeypatch.setattr(chores_mod.Chores, "recover", lambda self, message="": None)
    monkeypatch.setattr(chores_mod.Chores, "note_failure", lambda self, phase, message: None)
    for name in ("arrange_on_table", "stack_on_sideboard", "swap_drawers"):
        monkeypatch.setattr(chores_mod.Chores, name, timed_out)
    with pytest.raises(Stop):
        chores_mod.chores("http://unused", 0, on_failure=stop_after_failures)
    assert failures == ["timed out"] * 3
