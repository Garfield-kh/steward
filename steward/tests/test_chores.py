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


def test_drawer_contents_drop_tableware_and_leave_lookalikes_uncertain():
    """Wine / mug hard-skip; cup / car_body stay uncertain until a close look (not auto-kept)."""
    from steward.chores import (DRAWER_SWAP_ROLES, _contents, _contents_rejected,
                                _uncertain_drawer_tracks)

    drawer = {"id": "e:10", "position": [0.0, 0.0, 0.55], "sizeM": [0.5, 0.4, 0.22], "extentM": [0.5, 0.4, 0.22]}
    wine = _piece("e:44", "wine_glass", (0.0, 0.05), 0.08, on="e:10")
    wine["position"] = [0.0, 0.05, 0.90]
    wine["sizeM"] = wine["extentM"] = [0.08, 0.08, 0.19]
    mug = _piece("e:1334", "mug", (0.05, 0.0), 0.1, on="e:10")
    mug["position"] = [0.05, 0.0, 0.50]
    car = _piece("e:1080", "car_body", (0.02, 0.02), 0.12, on="e:10")
    car["position"] = [0.02, 0.02, 0.50]
    cup = _piece("e:99", "cup", (-0.05, 0.0), 0.06, on="e:10")
    cup["position"] = [-0.05, 0.0, 0.50]
    plane = _piece("e:1020", "toy_airplane", (0.0, 0.0), 0.22, on="e:10")
    plane["position"] = [0.0, 0.0, 0.50]
    summary = {"objects": [wine, mug, car, cup, plane]}
    assert [o["id"] for o in _contents(summary, drawer)] == ["e:1020"]
    reasons = {r["id"]: r["reasons"] for r in _contents_rejected(summary, drawer)}
    assert "tableware" in reasons["e:44"]
    assert "tableware" in reasons["e:1334"]
    assert "e:1080" not in reasons and "e:99" not in reasons
    assert {o["id"] for o in _uncertain_drawer_tracks(summary, drawer)} == {"e:1080", "e:99"}
    assert set(DRAWER_SWAP_ROLES) >= {"toy_airplane", "teddy_bear", "pistol", "pill_bottle", "perfume_bottle"}


def test_drawer_contents_keep_perfume_when_raw_category_flickers_to_cup():
    """A deployed bottle whose category mass briefly looks like a cup still swaps by role."""
    from steward.chores import _contents

    drawer = {"id": "e:1"}
    perfume = _piece("e:9", "cup", (0.0, 0.0), 0.06, on="e:1")
    perfume["role"] = "perfume_bottle"
    perfume["category"] = "cup"
    summary = {"objects": [perfume]}
    assert [o["id"] for o in _contents(summary, drawer)] == ["e:9"]


def test_confirm_drawer_contents_promotes_cup_after_near_look_and_skips_car(monkeypatch):
    """Uncertain in-drawer tracks: a close look that names perfume keeps it; car_body is skipped."""
    import steward.chores as chores_mod

    drawer = {"id": "e:10", "position": [0.0, 0.0, 0.55], "sizeM": [0.5, 0.4, 0.22], "extentM": [0.5, 0.4, 0.22],
              "role": "drawer", "category": "drawer"}
    plane = _piece("e:1020", "toy_airplane", (0.0, 0.0), 0.22, on="e:10")
    plane["position"] = [0.0, 0.0, 0.50]
    cup = _piece("e:9", "cup", (0.1, 0.0), 0.06, on="e:10")
    cup["position"] = [0.1, 0.0, 0.50]
    car = _piece("e:1080", "car_body", (-0.1, 0.0), 0.12, on="e:10")
    car["position"] = [-0.1, 0.0, 0.50]
    perfume = {**cup, "role": "perfume_bottle", "category": "perfume_bottle"}
    views = {"cup": 0}

    def fake_summary(api):
        objects = [drawer, plane, car]
        if views["cup"] == 0:
            objects.append(cup)
        else:
            objects.append(perfume)
        return {"objects": objects, "self": {"holding": None}}

    def fake_approach(api, obj, distance=0.9, prefer_offset=0.0):
        if obj["id"] == "e:9":
            views["cup"] += 1

    monkeypatch.setattr(chores_mod, "_summary", fake_summary)
    monkeypatch.setattr(chores_mod, "_approach_to_observe", fake_approach)
    monkeypatch.setattr(chores_mod, "_refresh_or_recover", lambda api, obj, **k: obj)
    monkeypatch.setattr(chores_mod, "simtime_sleep", lambda *a, **k: None)

    runner = Chores("api")
    monkeypatch.setattr(runner, "_doing", lambda *a, **k: None)
    monkeypatch.setattr(runner, "publish", lambda **k: None)
    kept = runner._confirm_drawer_contents(drawer, "cabinet")
    assert [o["id"] for o in kept] == ["e:1020", "e:9"]
    assert views["cup"] >= 1
    contents = [e for e in runner.log if e.get("step") == "drawer-contents"][-1]
    rejected_ids = {r["id"] for r in contents["rejected"]}
    assert "e:1080" in rejected_ids
    assert any(r.get("reasons") == ["unconfirmed"] for r in contents["rejected"] if r["id"] == "e:1080")
    confirms = [e for e in runner.log if e.get("step") == "drawer-confirm"]
    assert any(e["id"] == "e:9" and e["confirmed"] for e in confirms)
    assert any(e["id"] == "e:1080" and not e["confirmed"] for e in confirms)


def test_is_drawer_full_detects_no_free_spot():
    from steward.chores import _is_drawer_full
    from steward.mission import MissionFailed

    assert _is_drawer_full(MissionFailed("place infeasible: ['no free spot left in the open part of e:9']"))
    assert not _is_drawer_full(MissionFailed("place infeasible: ['no feasible approach']"))


def test_dedupe_drawer_roles_keeps_largest_per_role():
    """Two similar perfume tracks near each other → keep the larger footprint; ghost dropped."""
    import steward.chores as chores_mod

    big = {"id": "e:400", "stagedRole": "perfume_bottle", "sizeM": [0.08, 0.08, 0.12],
           "extentM": [0.08, 0.08, 0.12], "position": [2.25, 0.58, 0.50]}
    small = {"id": "e:277", "stagedRole": "perfume_bottle", "sizeM": [0.05, 0.05, 0.10],
             "extentM": [0.05, 0.05, 0.10], "position": [2.28, 0.62, 0.50]}
    plane = {"id": "e:383", "stagedRole": "toy_airplane", "sizeM": [0.20, 0.15, 0.08],
             "extentM": [0.20, 0.15, 0.08], "position": [2.30, 0.60, 0.50]}
    unique, dropped = chores_mod._dedupe_drawer_roles([plane, big, small])
    assert [o["id"] for o in unique] == ["e:383", "e:400"]
    assert [o["id"] for o in dropped] == ["e:277"]


def test_dedupe_drawer_roles_keeps_spatially_distinct_same_role():
    """Same-role bottles far apart with strong existence are both real — keep both."""
    import steward.chores as chores_mod

    left = {"id": "e:251", "stagedRole": "pill_bottle", "sizeM": [0.06, 0.06, 0.08],
            "extentM": [0.06, 0.06, 0.08], "position": [2.20, 0.55, 0.50], "existence": 0.95}
    right = {"id": "e:253", "stagedRole": "pill_bottle", "sizeM": [0.05, 0.05, 0.08],
             "extentM": [0.05, 0.05, 0.08], "position": [2.45, 0.70, 0.50], "existence": 0.92}
    unique, dropped = chores_mod._dedupe_drawer_roles([left, right])
    assert [o["id"] for o in unique] == ["e:251", "e:253"]
    assert dropped == []


def test_dedupe_drawer_roles_drops_soft_far_same_role_ghost():
    """Far same-role sibling with soft existence is a drifted ghost — drop it."""
    import steward.chores as chores_mod

    strong = {"id": "e:400", "stagedRole": "perfume_bottle", "sizeM": [0.07, 0.07, 0.10],
              "extentM": [0.07, 0.07, 0.10], "position": [2.25, 0.58, 0.50], "existence": 0.95}
    soft = {"id": "e:277", "stagedRole": "perfume_bottle", "sizeM": [0.06, 0.06, 0.10],
            "extentM": [0.06, 0.06, 0.10], "position": [2.45, 0.72, 0.50], "existence": 0.55}
    unique, dropped = chores_mod._dedupe_drawer_roles([strong, soft])
    assert [o["id"] for o in unique] == ["e:400"]
    assert [o["id"] for o in dropped] == ["e:277"]

def test_dedupe_drawer_roles_keeps_mislabeled_large_sibling():
    """Airplane misread as pill next to a real pill: footprints differ → keep both."""
    import steward.chores as chores_mod

    plane_as_pill = {"id": "e:431", "stagedRole": "pill_bottle", "sizeM": [0.22, 0.22, 0.08],
                     "extentM": [0.22, 0.22, 0.08], "position": [2.30, 0.60, 0.50]}
    real_pill = {"id": "e:288", "stagedRole": "pill_bottle", "sizeM": [0.06, 0.06, 0.08],
                 "extentM": [0.06, 0.06, 0.08], "position": [2.35, 0.55, 0.50]}
    perfume = {"id": "e:289", "stagedRole": "perfume_bottle", "sizeM": [0.07, 0.07, 0.10],
               "extentM": [0.07, 0.07, 0.10], "position": [2.25, 0.58, 0.50]}
    unique, dropped = chores_mod._dedupe_drawer_roles([plane_as_pill, perfume, real_pill])
    assert [o["id"] for o in unique] == ["e:431", "e:289", "e:288"]
    assert dropped == []

def test_confirm_drawer_contents_drops_duplicate_perfume(monkeypatch):
    """Cabinet with two similar perfume tracks keeps one so the swap reaches close."""
    import steward.chores as chores_mod

    drawer = {"id": "e:8", "position": [2.3, 0.6, 0.55], "sizeM": [0.5, 0.4, 0.22],
              "extentM": [0.5, 0.4, 0.22], "role": "drawer", "category": "drawer"}
    plane = _piece("e:383", "toy_airplane", (2.3, 0.6), 0.22, on="e:8")
    plane["position"] = [2.3, 0.6, 0.50]
    pill = _piece("e:275", "pill_bottle", (2.35, 0.55), 0.06, on="e:8")
    pill["position"] = [2.35, 0.55, 0.50]
    perfume_a = _piece("e:400", "perfume_bottle", (2.25, 0.58), 0.07, on="e:8")
    perfume_a["position"] = [2.25, 0.58, 0.50]
    perfume_b = _piece("e:277", "perfume_bottle", (2.28, 0.62), 0.05, on="e:8")
    perfume_b["position"] = [2.28, 0.62, 0.50]

    monkeypatch.setattr(chores_mod, "_summary",
                        lambda api: {"objects": [drawer, plane, pill, perfume_a, perfume_b],
                                     "self": {"holding": None}})
    monkeypatch.setattr(chores_mod, "_refresh_or_recover", lambda api, obj, **k: obj)
    monkeypatch.setattr(chores_mod, "simtime_sleep", lambda *a, **k: None)

    runner = Chores("api")
    monkeypatch.setattr(runner, "_doing", lambda *a, **k: None)
    monkeypatch.setattr(runner, "publish", lambda **k: None)
    kept = runner._confirm_drawer_contents(drawer, "cabinet")
    assert [o["id"] for o in kept] == ["e:383", "e:400", "e:275"]
    assert any(e.get("step") == "drawer-role-dupe" and e.get("id") == "e:277"
               for e in runner.log)
    contents = [e for e in runner.log if e.get("step") == "drawer-contents"][-1]
    assert any(r["id"] == "e:277" and "duplicate-role" in r.get("reasons", [])
               for r in contents["rejected"])


def test_confirm_drawer_contents_extra_look_promotes_large_car_body(monkeypatch):
    """A large unconfirmed in-drawer track gets one extra confirm pass (airplane-scale)."""
    import steward.chores as chores_mod

    drawer = {"id": "e:8", "position": [2.3, 0.6, 0.55], "sizeM": [0.5, 0.4, 0.22],
              "extentM": [0.5, 0.4, 0.22], "role": "drawer", "category": "drawer"}
    car = _piece("e:250", "car_body", (2.3, 0.6), 0.22, on="e:8")
    car["position"] = [2.3, 0.6, 0.50]
    pill = _piece("e:225", "pill_bottle", (2.4, 0.55), 0.06, on="e:8")
    pill["position"] = [2.4, 0.55, 0.50]
    views = {"n": 0}

    def fake_summary(api):
        objects = [drawer, pill]
        if views["n"] >= 3:
            objects.append({**car, "role": "toy_airplane", "category": "toy_airplane"})
        else:
            objects.append(car)
        return {"objects": objects, "self": {"holding": None}}

    def fake_approach(api, obj, distance=0.9, prefer_offset=0.0):
        if obj["id"] == "e:250":
            views["n"] += 1

    monkeypatch.setattr(chores_mod, "_summary", fake_summary)
    monkeypatch.setattr(chores_mod, "_approach_to_observe", fake_approach)
    monkeypatch.setattr(chores_mod, "_refresh_or_recover", lambda api, obj, **k: obj)
    monkeypatch.setattr(chores_mod, "simtime_sleep", lambda *a, **k: None)

    runner = Chores("api")
    monkeypatch.setattr(runner, "_doing", lambda *a, **k: None)
    monkeypatch.setattr(runner, "publish", lambda **k: None)
    kept = runner._confirm_drawer_contents(drawer, "cabinet")
    assert any(o["id"] == "e:250" for o in kept)
    assert any(e.get("step") == "drawer-confirm" and e.get("id") == "e:250" and e.get("extra")
               and e.get("confirmed") for e in runner.log)
    assert views["n"] >= 3


def test_transfer_skips_non_whitelist_without_raising(monkeypatch):
    """Stale cup / meal tracks log a skip and return None instead of failing the drawer phase."""
    import steward.chores as chores_mod

    monkeypatch.setattr(chores_mod, "_refresh_or_recover", lambda api, obj, **k: obj)
    runner = Chores("api")
    monkeypatch.setattr(runner, "publish", lambda **k: None)
    cup = _piece("e:9", "cup", (0.0, 0.0), 0.06)
    dst = _piece("e:2", "drawer", (0.0, 0.0), 0.4, on=None)
    assert runner._transfer(cup, dst, "测试跳过") is None
    assert runner.log[-1]["step"] == "skip" and runner.log[-1]["reason"] == "non-drawer-content"


def test_match_staged_drawer_item_ignores_tableware_near_the_spot():
    """A staged pistol rematch must not adopt a bowl that shares the sideboard top."""
    from steward.chores import _match_staged_drawer_item

    pistol = _piece("e:76", "pistol", (-2.75, 0.86), 0.12)
    pistol["stagedRole"] = "pistol"
    bowl = _piece("e:7", "bowl", (-2.74, 0.87), 0.18, on="e:20")
    renamed = _piece("e:900", "pistol", (-2.76, 0.85), 0.11, on="e:20")
    wine = _piece("e:78", "wine_glass", (-2.75, 0.86), 0.08, on="e:20")
    summary = {"objects": [bowl, wine, renamed]}
    hit = _match_staged_drawer_item(summary, pistol, [-2.75, 0.86])
    assert hit is not None and hit["id"] == "e:900"


def test_match_staged_drawer_item_requires_the_expected_role():
    """A teddy near the pistol stage spot must not be adopted as the staged pistol."""
    from steward.chores import _match_staged_drawer_item

    pistol = _piece("e:76", "pistol", (-2.75, 0.86), 0.12)
    pistol["stagedRole"] = "pistol"
    teddy = _piece("e:78", "teddy_bear", (-2.74, 0.87), 0.14, on="e:20")
    summary = {"objects": [teddy]}
    assert _match_staged_drawer_item(summary, pistol, [-2.75, 0.86]) is None


def test_match_staged_near_widens_radius_when_tight_neighbourhood_empty():
    """A renumbered staged centre just outside 0.35 m is still accepted on the wide pass."""
    from steward.chores import (STAGE_RECOVER_RADIUS_M, _match_staged_drawer_item,
                                _match_staged_near)

    pistol = _piece("e:76", "pistol", (-2.75, 0.86), 0.12)
    pistol["stagedRole"] = "pistol"
    # ~0.42 m from the stage spot — outside the tight radius, inside the wide one.
    live = _piece("e:2006", "pistol", (-2.75 + 0.42, 0.86), 0.11, on="e:21")
    summary = {"objects": [live]}
    xy = [-2.75, 0.86]
    assert _match_staged_drawer_item(summary, pistol, xy, radius=STAGE_RECOVER_RADIUS_M) is None
    hit = _match_staged_near(summary, pistol, xy)
    assert hit is not None and hit["id"] == "e:2006"


def test_recover_drawer_carry_rematches_near_stage_xy(monkeypatch):
    """After place unverified, recover looks at stageXy and keeps a whitelist track only."""
    import steward.chores as chores_mod
    from steward.mission import MissionFailed

    staged = _piece("e:76", "pistol", (-2.75, 0.86), 0.12)
    staged["stageXy"] = [-2.75, 0.86]
    staged["stagedRole"] = "pistol"
    bowl = _piece("e:7", "bowl", (-2.74, 0.87), 0.18, on="e:20")
    live = _piece("e:901", "pistol", (-2.76, 0.85), 0.11, on="e:20")
    protected: list[dict] = []

    def fake_refresh(api, obj):
        if obj["id"] == "e:901":
            return live
        raise MissionFailed(f"object {obj['id']} is no longer in view")

    monkeypatch.setattr(chores_mod, "_refresh", fake_refresh)
    monkeypatch.setattr(chores_mod, "_approach_to_observe", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "simtime_sleep", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "_summary", lambda api: {"objects": [bowl, live]})
    monkeypatch.setattr(chores_mod, "_protect_stage",
                        lambda api, obj, log: protected.append(dict(obj)))
    log: list = []
    out = chores_mod._recover_drawer_carry("api", staged, log)
    assert out["id"] == "e:901"
    assert out.get("stageXy") == [-2.75, 0.86]
    assert any(e.get("reason") == "matched" for e in log if e.get("step") == "drawer-rematch")
    assert any(e.get("step") == "drawer-freshen" for e in log)
    assert protected and protected[0]["id"] == "e:901"


def test_recover_keeps_rematch_when_freshen_loses_track(monkeypatch):
    """After rematch, freshen misses must not abort — keep the whitelist track for _move."""
    import steward.chores as chores_mod
    from steward.mission import MissionFailed

    staged = _piece("e:1796", "pistol", (-2.716, 0.324), 0.12)
    staged["stageXy"] = [-2.716, 0.324]
    staged["stagedRole"] = "pistol"
    live = _piece("e:2006", "pistol", (-2.72, 0.33), 0.11, on="e:21")
    freshen_calls = {"n": 0}

    def fake_refresh(api, obj):
        if obj.get("id") == "e:2006":
            return live
        raise MissionFailed(f"{obj.get('id')} is no longer in the ego state")

    def fake_freshen(api, obj, log, *, prefer_offset=None, distance=None):
        freshen_calls["n"] += 1
        raise MissionFailed(
            f"lost staged drawer item {obj.get('id')} near {obj.get('stageXy')} after freshen look")

    monkeypatch.setattr(chores_mod, "_refresh", fake_refresh)
    monkeypatch.setattr(chores_mod, "_approach_to_observe", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "simtime_sleep", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "_summary", lambda api: {"objects": [live]})
    monkeypatch.setattr(chores_mod, "_protect_stage", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "_freshen_drawer_pick", fake_freshen)

    log: list = []
    out = chores_mod._recover_drawer_carry("api", staged, log)
    assert out["id"] == "e:2006"
    assert out.get("stageXy") == [-2.716, 0.324]
    assert freshen_calls["n"] == 2  # first heading + retry heading
    assert any(e.get("reason") == "soft-keep" for e in log if e.get("step") == "drawer-freshen")


def test_recover_staged_item_ignores_source_drawer_id(monkeypatch):
    """A stamped source drawerId must not beat stageXy (avoids holding the drawer track)."""
    import steward.chores as chores_mod
    from steward.mission import MissionFailed

    staged = _piece("e:2090", "teddy_bear", (-2.814, 0.332), 0.14)
    staged["stageXy"] = [-2.814, 0.332]
    staged["stagedRole"] = "teddy_bear"
    staged["drawerId"] = "e:20"  # source sideboard drawer from confirm
    drawer = _piece("e:20", "drawer", (-2.5, 0.6), 0.45, on=None)
    drawer["role"] = "drawer"
    drawer["position"] = [-2.5, 0.6, 0.55]
    drawer["extentM"] = drawer["sizeM"] = [0.5, 0.4, 0.22]
    # Same role inside the open drawer — must not win over the staged top track.
    in_drawer = _piece("e:363", "teddy_bear", (-2.48, 0.58), 0.14, on="e:20")
    in_drawer["position"] = [-2.48, 0.58, 0.70]
    live = _piece("e:2100", "teddy_bear", (-2.81, 0.33), 0.13, on="e:21")
    live["position"] = [-2.81, 0.33, 0.94]

    monkeypatch.setattr(
        chores_mod, "_refresh",
        lambda api, obj: (_ for _ in ()).throw(
            MissionFailed("e:2090 (teddy_bear) is no longer in the ego state")))
    monkeypatch.setattr(chores_mod, "_approach_to_observe", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "simtime_sleep", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "_summary",
                        lambda api: {"objects": [drawer, in_drawer, live]})
    monkeypatch.setattr(chores_mod, "_summary_object",
                        lambda summary, oid: next(
                            (o for o in summary["objects"] if o["id"] == oid), None))
    monkeypatch.setattr(chores_mod, "_protect_stage", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "_freshen_drawer_pick",
                        lambda api, obj, log, prefer_offset=None: obj)
    log: list = []
    out = chores_mod._recover_drawer_carry("api", staged, log)
    assert out["id"] == "e:2100"
    assert out.get("stageXy") == [-2.814, 0.332]


def test_deployed_footprint_rematch_requires_role_and_neighbourhood():
    """A cabinet airplane must not adopt a staged pistol across the room (seed2 drawers)."""
    from steward.chores import _match_by_footprint

    airplane = _piece("e:395", "toy_airplane", (2.71, 0.60), 0.18)
    staged_pistol = _piece("e:491", "pistol", (-2.74, 0.86), 0.12, on="e:21")
    near_plane = _piece("e:500", "toy_airplane", (2.68, 0.58), 0.17, on="e:3")
    summary = {"objects": [staged_pistol, near_plane]}
    assert _match_by_footprint(summary, airplane, require_deployed=True)["id"] == "e:500"
    # Wrong role alone (even nearby) is refused.
    nearby_gun = _piece("e:492", "pistol", (2.70, 0.59), 0.12, on="e:3")
    assert _match_by_footprint({"objects": [nearby_gun]}, airplane, require_deployed=True) is None
    # Far same-role is refused (would previously steal a staged piece on the opposite top).
    far_plane = _piece("e:501", "toy_airplane", (-2.74, 0.86), 0.17, on="e:21")
    assert _match_by_footprint({"objects": [far_plane]}, airplane, require_deployed=True) is None


def test_recover_drawer_carry_rejects_cross_room_wrong_role(monkeypatch):
    """Cabinet airplane recover must fail closed rather than pick the staged sideboard pistol."""
    import steward.chores as chores_mod
    from steward.mission import MissionFailed

    airplane = _piece("e:395", "toy_airplane", (2.71, 0.60), 0.18)
    car = _piece("e:294", "car_body", (2.70, 0.55), 0.16)
    staged_pistol = _piece("e:491", "pistol", (-2.74, 0.86), 0.12, on="e:21")

    def fake_refresh(api, obj):
        if obj["id"] == "e:395":
            return car  # renumbered into a non-deployed look-alike
        raise MissionFailed(f"object {obj['id']} is no longer in view")

    monkeypatch.setattr(chores_mod, "_refresh", fake_refresh)
    monkeypatch.setattr(chores_mod, "_approach_to_observe", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "simtime_sleep", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "_summary", lambda api: {"objects": [car, staged_pistol]})
    monkeypatch.setattr(chores_mod, "_protect_stage", lambda *a, **k: None)
    log: list = []
    with pytest.raises(MissionFailed, match="lost staged drawer item e:395"):
        chores_mod._recover_drawer_carry("api", airplane, log)
    assert any(e.get("reason") == "refreshed-non-deployed" for e in log if e.get("step") == "drawer-rematch")


def test_move_staged_pick_freshens_from_a_new_heading_after_miss(monkeypatch):
    """A grasp miss on a staged piece retries from another observe heading, not the same pose."""
    import steward.chores as chores_mod
    from steward.mission import MissionFailed

    offsets: list[float | None] = []
    picks = {"n": 0}
    staged = _piece("e:1390", "pistol", (-2.815, 0.865), 0.12)
    staged["stageXy"] = [-2.815, 0.865]
    staged["stagedRole"] = "pistol"
    board = _piece("e:4", "drawer", (0.0, 0.0), 0.4, on=None)

    distances: list[float | None] = []

    def fake_freshen(api, obj, log, *, prefer_offset=None, distance=None):
        offsets.append(prefer_offset)
        distances.append(distance)
        return {**obj, "id": "e:1390"}

    def fake_safe(api, skill, params, observe, log):
        return {"feasible": True, "withinRiskLimit": True, "risk": 0.04, "blockers": [],
                "skill": skill, "plan": {"spot": [-2.815, 0.865]}, "issues": []}

    def fake_run(api, receipt, timeout_s):
        if receipt["skill"] == "pick":
            picks["n"] += 1
            if picks["n"] == 1:
                raise MissionFailed("pick failed: grasp missed e:1390 (7 probes)")
            return {"state": "succeeded", "message": "holding e:1390"}
        return {"state": "succeeded", "message": "placed"}

    retreats = {"n": 0}

    recovers = {"n": 0}

    def fake_recover(api, obj, log):
        recovers["n"] += 1
        return {**obj, "id": "e:1390", "role": "pistol", "category": "pistol"}

    confirms = {"n": 0}

    def fake_confirm(api, obj, stage_xy, log, *, views=2):
        confirms["n"] += 1
        return {**obj, "stageConfirmed": True, "stageXy": list(stage_xy)}

    monkeypatch.setattr(chores_mod, "_freshen_drawer_pick", fake_freshen)
    monkeypatch.setattr(chores_mod, "_recover_drawer_carry", fake_recover)
    monkeypatch.setattr(chores_mod, "_confirm_stage_landing", fake_confirm)
    monkeypatch.setattr(chores_mod, "_protect_stage", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "_align_drawer_place", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "_confirm_drawer_landing",
                        lambda api, obj, drawer, spot, log: {**obj, "position": [spot[0], spot[1], 0.8]})
    monkeypatch.setattr(chores_mod, "_safe_receipt", fake_safe)
    monkeypatch.setattr(chores_mod, "_run", fake_run)
    monkeypatch.setattr(chores_mod, "_refresh_or_recover", lambda api, obj, **k: obj)
    monkeypatch.setattr(chores_mod, "_held_drawer_match", lambda api, obj: {**staged, "id": "e:1390"})
    monkeypatch.setattr(chores_mod, "_retreat_while_holding",
                        lambda *a, **k: retreats.__setitem__("n", retreats["n"] + 1) or True)
    monkeypatch.setattr(chores_mod, "_landed", lambda api, obj, spot: {**obj, "position": [spot[0], spot[1], 0.8]})
    monkeypatch.setattr(chores_mod, "simtime_sleep", lambda *a, **k: None)

    out = chores_mod._move("api", staged, {"on": "e:4"}, board, [], 30.0)
    assert out["id"] == "e:1390"
    assert picks["n"] == 2
    assert offsets == [0.0, 1.0]
    assert distances[0] == chores_mod.STAGE_PICK_DISTANCE
    assert distances[1] == chores_mod.STAGE_PICK_CLOSE_DISTANCE
    assert retreats["n"] == 1  # grasp miss retreats before the next staged heading
    assert recovers["n"] == 1  # grasp miss rematches before the next heading
    assert confirms["n"] == 1  # stage spot reconfirmed after the miss


def test_move_staged_pick_skips_soft_existence_receipt(monkeypatch):
    """Soft-existence pick receipts rematch near stageXy instead of grasping (seed2e teddy)."""
    import steward.chores as chores_mod

    log: list = []
    picks = {"n": 0}
    recovers = {"n": 0}
    staged = _piece("e:1002", "teddy_bear", (-2.809, 0.332), 0.15)
    staged["stageXy"] = [-2.809, 0.332]
    staged["stagedRole"] = "teddy_bear"
    board = _piece("e:3", "drawer", (2.3, 0.6), 0.4, on=None)
    board["role"] = "drawer"

    def fake_freshen(api, obj, log_, *, prefer_offset=None, distance=None):
        return {**obj, "existence": 0.9 if picks["n"] else 0.62}

    def fake_recover(api, item, log_):
        recovers["n"] += 1
        return {**item, "id": "e:1002", "existence": 0.95}

    def fake_safe(api, skill, params, observe, log_):
        if skill == "pick" and picks["n"] == 0:
            return {"feasible": True, "withinRiskLimit": True, "risk": 0.19, "blockers": [],
                    "skill": skill, "plan": {"spot": [-2.809, 0.332]}, "issues": [
                        "target existence probability is only 0.62"]}
        return {"feasible": True, "withinRiskLimit": True, "risk": 0.04, "blockers": [],
                "skill": skill, "plan": {"spot": [2.32, 0.61]}, "issues": []}

    def fake_run(api, receipt, timeout_s):
        if receipt["skill"] == "pick":
            picks["n"] += 1
            return {"state": "succeeded", "message": "holding e:1002"}
        return {"state": "succeeded", "message": "placed"}

    monkeypatch.setattr(chores_mod, "_freshen_drawer_pick", fake_freshen)
    monkeypatch.setattr(chores_mod, "_recover_drawer_carry", fake_recover)
    monkeypatch.setattr(chores_mod, "_align_drawer_place", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "_confirm_drawer_landing",
                        lambda api, obj, drawer, spot, log_: {**obj, "position": [spot[0], spot[1], 0.8]})
    monkeypatch.setattr(chores_mod, "_safe_receipt", fake_safe)
    monkeypatch.setattr(chores_mod, "_run", fake_run)
    monkeypatch.setattr(chores_mod, "_refresh_or_recover", lambda api, obj, **k: obj)
    monkeypatch.setattr(chores_mod, "_held_drawer_match", lambda api, obj: {**staged, "id": "e:1002"})
    monkeypatch.setattr(chores_mod, "_landed",
                        lambda api, obj, spot: {**obj, "position": [spot[0], spot[1], 0.8]})
    monkeypatch.setattr(chores_mod, "simtime_sleep", lambda *a, **k: None)

    out = chores_mod._move("api", staged, {"on": "e:3"}, board, log, 30.0)
    assert out["id"] == "e:1002"
    assert picks["n"] == 1
    assert recovers["n"] >= 1
    assert any(e.get("step") == "pick-retry" and e.get("reason") == "existence" for e in log)


def test_move_staged_pick_grasps_after_one_uncertainty_deferral(monkeypatch):
    """Mild pose-uncertainty receipts defer once, then grasp (seed2h must not burn all tries)."""
    import steward.chores as chores_mod

    log: list = []
    picks = {"n": 0}
    staged = _piece("e:1327", "pistol", (-2.81, 0.332), 0.12)
    staged["stageXy"] = [-2.81, 0.332]
    staged["stagedRole"] = "pistol"
    board = _piece("e:1", "drawer", (2.3, 0.6), 0.4, on=None)
    board["role"] = "drawer"
    issue = "target position uncertainty 2.4 cm gives 12% miss probability; observe the target more closely first"

    def fake_freshen(api, obj, log_, *, prefer_offset=None, distance=None):
        return {**obj, "existence": 1.0}

    def fake_safe(api, skill, params, observe, log_):
        if skill == "pick":
            return {"feasible": True, "withinRiskLimit": True, "risk": 0.12, "blockers": [],
                    "skill": skill, "plan": {"spot": [-2.81, 0.332]}, "issues": [issue]}
        return {"feasible": True, "withinRiskLimit": True, "risk": 0.04, "blockers": [],
                "skill": skill, "plan": {"spot": [2.32, 0.61]}, "issues": []}

    def fake_run(api, receipt, timeout_s):
        if receipt["skill"] == "pick":
            picks["n"] += 1
            return {"state": "succeeded", "message": "holding e:1327"}
        return {"state": "succeeded", "message": "placed"}

    monkeypatch.setattr(chores_mod, "_freshen_drawer_pick", fake_freshen)
    monkeypatch.setattr(chores_mod, "_align_drawer_place", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "_confirm_drawer_landing",
                        lambda api, obj, drawer, spot, log_: {**obj, "position": [spot[0], spot[1], 0.8]})
    monkeypatch.setattr(chores_mod, "_safe_receipt", fake_safe)
    monkeypatch.setattr(chores_mod, "_run", fake_run)
    monkeypatch.setattr(chores_mod, "_refresh_or_recover", lambda api, obj, **k: obj)
    monkeypatch.setattr(chores_mod, "_held_drawer_match", lambda api, obj: {**staged, "id": "e:1327"})
    monkeypatch.setattr(chores_mod, "_landed",
                        lambda api, obj, spot: {**obj, "position": [spot[0], spot[1], 0.8]})
    monkeypatch.setattr(chores_mod, "simtime_sleep", lambda *a, **k: None)

    out = chores_mod._move("api", staged, {"on": "e:1"}, board, log, 30.0)
    assert out["id"] == "e:1327"
    assert picks["n"] == 1
    assert sum(1 for e in log if e.get("step") == "pick-retry" and e.get("reason") == "observe") == 1


def test_move_drawer_carry_defers_pose_uncertainty_once(monkeypatch):
    """In-drawer whitelist picks also defer once on pose-uncertainty (seed2b perfume)."""
    import steward.chores as chores_mod

    log: list = []
    picks = {"n": 0}
    approaches: list[float] = []
    perfume = _piece("e:277", "perfume_bottle", (2.3, 0.6), 0.06)
    perfume["drawerId"] = "e:8"
    perfume["stagedRole"] = "perfume_bottle"
    perfume["existence"] = 0.9
    drawer = _piece("e:19", "drawer", (-2.5, 0.6), 0.4, on=None)
    drawer["role"] = "drawer"
    issue = ("target position uncertainty 4.0 cm gives 46% miss probability; "
             "observe the target more closely first")

    def fake_approach(api, obj, distance=None, prefer_offset=None):
        approaches.append(float(distance))

    def fake_safe(api, skill, params, observe, log_):
        if skill == "pick" and picks["n"] == 0:
            return {"feasible": True, "withinRiskLimit": True, "risk": 0.25, "blockers": [],
                    "skill": skill, "plan": {"spot": [2.3, 0.6]}, "issues": [issue]}
        if skill == "pick":
            return {"feasible": True, "withinRiskLimit": True, "risk": 0.05, "blockers": [],
                    "skill": skill, "plan": {"spot": [2.3, 0.6]}, "issues": []}
        return {"feasible": True, "withinRiskLimit": True, "risk": 0.04, "blockers": [],
                "skill": skill, "plan": {"spot": [-2.4, 0.6]}, "issues": []}

    def fake_run(api, receipt, timeout_s):
        if receipt["skill"] == "pick":
            picks["n"] += 1
            return {"state": "succeeded", "message": "holding e:277"}
        return {"state": "succeeded", "message": "placed"}

    monkeypatch.setattr(chores_mod, "_approach_to_observe", fake_approach)
    monkeypatch.setattr(chores_mod, "_recover_drawer_carry", lambda api, item, log_: item)
    monkeypatch.setattr(chores_mod, "_align_drawer_place", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "_confirm_drawer_landing",
                        lambda api, obj, drawer_, spot, log_: {**obj, "position": [spot[0], spot[1], 0.5]})
    monkeypatch.setattr(chores_mod, "_safe_receipt", fake_safe)
    monkeypatch.setattr(chores_mod, "_run", fake_run)
    monkeypatch.setattr(chores_mod, "_refresh_or_recover", lambda api, obj, **k: obj)
    monkeypatch.setattr(chores_mod, "_held_drawer_match", lambda api, obj: perfume)
    monkeypatch.setattr(chores_mod, "_landed",
                        lambda api, obj, spot: {**obj, "position": [spot[0], spot[1], 0.5]})
    monkeypatch.setattr(chores_mod, "simtime_sleep", lambda *a, **k: None)

    out = chores_mod._move("api", perfume, {"on": "e:19"}, drawer, log, 30.0)
    assert out["id"] == "e:277"
    assert picks["n"] == 1
    assert any(e.get("step") == "pick-retry" and e.get("reason") == "observe" for e in log)
    assert approaches[0] == 0.9
    assert approaches[1] == chores_mod.STAGE_PICK_CLOSE_DISTANCE


def test_move_drawer_carry_skips_soft_existence_after_rematch_budget(monkeypatch):
    """In-drawer soft ghosts skip after rematch budget instead of aborting the swap."""
    import steward.chores as chores_mod
    from steward.mission import MissionFailed

    log: list = []
    perfume = _piece("e:247", "perfume_bottle", (2.3, 0.6), 0.06)
    perfume["drawerId"] = "e:8"
    perfume["stagedRole"] = "perfume_bottle"
    perfume["existence"] = 0.50
    drawer = _piece("e:19", "drawer", (-2.5, 0.6), 0.4, on=None)
    drawer["role"] = "drawer"
    issues = [
        "target position uncertainty 4.0 cm gives 46% miss probability; observe the target more closely first",
        "target existence probability is only 0.50",
    ]

    monkeypatch.setattr(chores_mod, "_approach_to_observe", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "_recover_drawer_carry",
                        lambda api, item, log_: {**item, "existence": 0.50})
    monkeypatch.setattr(chores_mod, "_safe_receipt",
                        lambda api, skill, params, observe, log_: {
                            "feasible": True, "withinRiskLimit": True, "risk": 0.3,
                            "blockers": [], "skill": skill, "plan": {"spot": [2.3, 0.6]},
                            "issues": list(issues)})
    monkeypatch.setattr(chores_mod, "_run",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not grasp")))
    monkeypatch.setattr(chores_mod, "simtime_sleep", lambda *a, **k: None)

    with pytest.raises(MissionFailed, match="soft existence"):
        chores_mod._move("api", perfume, {"on": "e:19"}, drawer, log, 30.0)
    assert sum(1 for e in log if e.get("reason") == "existence") == 2
    assert any(e.get("reason") == "soft-skip" for e in log)
def test_move_staged_pick_limits_soft_existence_observe_defers(monkeypatch):
    """Soft-existence rematch budget is two looks, then grasp on the final attempt."""
    import steward.chores as chores_mod

    log: list = []
    picks = {"n": 0}
    receipts = {"n": 0}
    recovers = {"n": 0}
    staged = _piece("e:1002", "teddy_bear", (-2.809, 0.332), 0.15)
    staged["stageXy"] = [-2.809, 0.332]
    staged["stagedRole"] = "teddy_bear"
    board = _piece("e:3", "drawer", (2.3, 0.6), 0.4, on=None)
    board["role"] = "drawer"

    def fake_freshen(api, obj, log_, *, prefer_offset=None, distance=None):
        return {**obj, "existence": 0.62, "role": "teddy_bear", "category": "teddy_bear"}

    def fake_recover(api, item, log_):
        recovers["n"] += 1
        return {**item, "existence": 0.62}

    def fake_safe(api, skill, params, observe, log_):
        if skill == "pick":
            receipts["n"] += 1
            return {"feasible": True, "withinRiskLimit": True, "risk": 0.19, "blockers": [],
                    "skill": skill, "plan": {"spot": [-2.809, 0.332]}, "issues": [
                        "target existence probability is only 0.62"]}
        return {"feasible": True, "withinRiskLimit": True, "risk": 0.04, "blockers": [],
                "skill": skill, "plan": {"spot": [2.32, 0.61]}, "issues": []}

    def fake_run(api, receipt, timeout_s):
        if receipt["skill"] == "pick":
            picks["n"] += 1
            return {"state": "succeeded", "message": "holding e:1002"}
        return {"state": "succeeded", "message": "placed"}

    monkeypatch.setattr(chores_mod, "_freshen_drawer_pick", fake_freshen)
    monkeypatch.setattr(chores_mod, "_recover_drawer_carry", fake_recover)
    monkeypatch.setattr(chores_mod, "_align_drawer_place", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "_confirm_drawer_landing",
                        lambda api, obj, drawer, spot, log_: {**obj, "position": [spot[0], spot[1], 0.8]})
    monkeypatch.setattr(chores_mod, "_safe_receipt", fake_safe)
    monkeypatch.setattr(chores_mod, "_run", fake_run)
    monkeypatch.setattr(chores_mod, "_refresh_or_recover", lambda api, obj, **k: obj)
    monkeypatch.setattr(chores_mod, "_held_drawer_match", lambda api, obj: {**staged, "id": "e:1002"})
    monkeypatch.setattr(chores_mod, "_landed",
                        lambda api, obj, spot: {**obj, "position": [spot[0], spot[1], 0.8]})
    monkeypatch.setattr(chores_mod, "simtime_sleep", lambda *a, **k: None)

    out = chores_mod._move("api", staged, {"on": "e:3"}, board, log, 30.0)
    assert out["id"] == "e:1002"
    assert picks["n"] == 1
    # Two existence defers (attempts 0,1) + grasp on attempt 2 despite soft existence.
    assert receipts["n"] == 3
    assert recovers["n"] == 2
    assert sum(1 for e in log if e.get("step") == "pick-retry" and e.get("reason") == "existence") == 2


def test_match_staged_drawer_item_prefers_higher_existence():
    """Among role/footprint matches near stageXy, the stronger existence track wins."""
    import steward.chores as chores_mod

    expected = {"stagedRole": "pistol", "sizeM": [0.12, 0.12, 0.06], "extentM": [0.12, 0.12, 0.06]}
    weak = _piece("e:1", "pistol", (-2.75, 0.33), 0.12)
    weak["existence"] = 0.42
    strong = _piece("e:2", "pistol", (-2.74, 0.34), 0.12)
    strong["existence"] = 0.88
    summary = {"objects": [weak, strong]}
    found = chores_mod._match_staged_drawer_item(summary, expected, [-2.749, 0.328])
    assert found is not None and found["id"] == "e:2"


def test_recover_drawer_carry_footprint_centres_on_stage_xy(monkeypatch):
    """Staged footprint rematch uses stageXy, not the pre-stage drawer pose (seed2e)."""
    import steward.chores as chores_mod

    staged = _piece("e:74", "pistol", (2.3, 0.6), 0.12)  # old drawer pose across the room
    staged["stageXy"] = [-2.815, 0.325]
    staged["stagedRole"] = "pistol"
    # Drifted ~0.55 m from the stage spot (outside tight stage radius, inside footprint radius).
    found_live = _piece("e:900", "pistol", (-2.40, 0.55), 0.12)
    found_live["existence"] = 0.9
    log: list = []

    monkeypatch.setattr(chores_mod, "_refresh",
                        lambda api, obj: (_ for _ in ()).throw(
                            chores_mod.MissionFailed("e:74 is no longer in the ego state")))
    monkeypatch.setattr(chores_mod, "_approach_to_observe", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "_summary",
                        lambda api: {"objects": [found_live], "self": {"holding": None}})
    monkeypatch.setattr(chores_mod, "_match_staged_near", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "_confirm_stage_landing",
                        lambda api, obj, xy, log_, *, views=2: {
                            **obj, "stageConfirmed": False, "stageXy": list(xy)})
    monkeypatch.setattr(chores_mod, "_protect_stage", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "_freshen_drawer_pick", lambda api, obj, log_, **k: obj)
    monkeypatch.setattr(chores_mod, "simtime_sleep", lambda *a, **k: None)

    out = chores_mod._recover_drawer_carry("api", staged, log)
    assert out["id"] == "e:900"
    assert out.get("stageXy") == [-2.815, 0.325]
    assert any(e.get("step") == "drawer-rematch" and e.get("to") == "e:900" for e in log)

def test_freshen_drawer_pick_steps_in_on_soft_existence_with_explicit_far_distance(monkeypatch):
    """Explicit far ``distance`` still triggers a close look when existence is soft."""
    import steward.chores as chores_mod

    staged = _piece("e:1962", "teddy_bear", (-2.815, 0.332), 0.15)
    staged["stageXy"] = [-2.815, 0.332]
    staged["stagedRole"] = "teddy_bear"
    staged["existence"] = 0.60
    looks: list[float] = []

    def fake_approach(api, obj, distance=None, prefer_offset=None):
        looks.append(float(distance))

    def fake_refresh(api, obj):
        # Second (close) look raises existence above the pick floor.
        existence = 0.60 if len(looks) <= 1 else 0.92
        return {**obj, "existence": existence, "role": "teddy_bear", "category": "teddy_bear"}

    monkeypatch.setattr(chores_mod, "_approach_to_observe", fake_approach)
    monkeypatch.setattr(chores_mod, "_refresh", fake_refresh)
    monkeypatch.setattr(chores_mod, "_summary", lambda api: {"objects": [staged]})
    monkeypatch.setattr(chores_mod, "simtime_sleep", lambda *a, **k: None)

    log: list = []
    out = chores_mod._freshen_drawer_pick(
        "api", staged, log, prefer_offset=0.0, distance=chores_mod.STAGE_PICK_DISTANCE)
    assert looks == [chores_mod.STAGE_PICK_DISTANCE, chores_mod.STAGE_PICK_CLOSE_DISTANCE]
    assert out["existence"] == 0.92


def test_freshen_drawer_pick_rejects_reclassified_id(monkeypatch):
    """Same track id reclassified away from the staged role rematches a whitelist neighbour."""
    import steward.chores as chores_mod

    staged = _piece("e:98", "pistol", (-2.82, 0.33), 0.12)
    staged["stageXy"] = [-2.82, 0.33]
    staged["stagedRole"] = "pistol"
    ghost = _piece("e:98", "car_body", (-2.82, 0.33), 0.12)
    live = _piece("e:500", "pistol", (-2.81, 0.34), 0.11, on="e:21")
    log: list = []

    monkeypatch.setattr(chores_mod, "_approach_to_observe", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "simtime_sleep", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "_refresh", lambda api, obj: ghost if obj["id"] == "e:98" else obj)
    monkeypatch.setattr(chores_mod, "_summary",
                        lambda api: {"objects": [ghost, live], "self": {}})

    out = chores_mod._freshen_drawer_pick("api", staged, log, prefer_offset=0.0)
    assert out["id"] == "e:500"
    assert out.get("stagedRole") == "pistol"
    assert any(e.get("reason") == "role-mismatch" for e in log if e.get("step") == "drawer-freshen")


def test_move_staged_pick_rematches_after_grasp_miss(monkeypatch):
    """After a staged grasp miss, rematch near stageXy before the next freshen heading."""
    import steward.chores as chores_mod
    from steward.mission import MissionFailed

    recovers = {"n": 0}
    picks = {"n": 0}
    staged = _piece("e:772", "teddy_bear", (-2.81, 0.33), 0.15)
    staged["stageXy"] = [-2.81, 0.33]
    staged["stagedRole"] = "teddy_bear"
    rematched = {**staged, "id": "e:900", "existence": 1.0}
    board = _piece("e:4", "drawer", (2.3, 0.6), 0.4, on=None)

    def fake_freshen(api, obj, log, *, prefer_offset=None, distance=None):
        return {**obj, "existence": 1.0, "role": "teddy_bear", "category": "teddy_bear"}

    def fake_recover(api, obj, log):
        recovers["n"] += 1
        log.append({"step": "drawer-rematch", "reason": "matched", "from": obj.get("id"),
                    "to": rematched["id"]})
        return rematched

    def fake_safe(api, skill, params, observe, log):
        return {"feasible": True, "withinRiskLimit": True, "risk": 0.04, "blockers": [],
                "skill": skill, "plan": {"spot": [-2.81, 0.33]}, "issues": []}

    def fake_run(api, receipt, timeout_s):
        if receipt["skill"] == "pick":
            picks["n"] += 1
            if picks["n"] == 1:
                raise MissionFailed("pick failed: grasp missed e:772 (7 probes)")
            return {"state": "succeeded", "message": f"holding {rematched['id']}"}
        return {"state": "succeeded", "message": "placed"}

    monkeypatch.setattr(chores_mod, "_freshen_drawer_pick", fake_freshen)
    monkeypatch.setattr(chores_mod, "_recover_drawer_carry", fake_recover)
    monkeypatch.setattr(chores_mod, "_confirm_stage_landing",
                        lambda api, obj, xy, log, *, views=2: {
                            **obj, "stageConfirmed": True, "stageXy": list(xy)})
    monkeypatch.setattr(chores_mod, "_protect_stage", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "_align_drawer_place", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "_confirm_drawer_landing",
                        lambda api, obj, drawer, spot, log: {**obj, "position": [spot[0], spot[1], 0.8]})
    monkeypatch.setattr(chores_mod, "_safe_receipt", fake_safe)
    monkeypatch.setattr(chores_mod, "_run", fake_run)
    monkeypatch.setattr(chores_mod, "_refresh_or_recover", lambda api, obj, **k: obj)
    monkeypatch.setattr(chores_mod, "_held_drawer_match", lambda api, obj: rematched)
    monkeypatch.setattr(chores_mod, "_retreat_while_holding", lambda *a, **k: True)
    monkeypatch.setattr(chores_mod, "_landed",
                        lambda api, obj, spot: {**obj, "position": [spot[0], spot[1], 0.8]})
    monkeypatch.setattr(chores_mod, "simtime_sleep", lambda *a, **k: None)

    out = chores_mod._move("api", staged, {"on": "e:4"}, board, [], 30.0)
    assert out["id"] == "e:900"
    assert picks["n"] == 2
    assert recovers["n"] >= 1


def test_transfer_refuses_unconfirmed_stage_before_cabinet_leg(monkeypatch):
    """Unconfirmed stage reconfirms, then rematch; abort only when rematch also fails."""
    import steward.chores as chores_mod
    from steward.mission import MissionFailed

    staged = _piece("e:79", "pistol", (-2.815, 0.865), 0.12)
    staged["stageXy"] = [-2.815, 0.865]
    staged["stagedRole"] = "pistol"
    staged["stageConfirmed"] = False
    dst = _piece("e:6", "drawer", (2.3, 0.6), 0.4, on=None)
    dst["role"] = "drawer"
    confirms = {"n": 0}
    recovers = {"n": 0}

    def fake_confirm(api, obj, stage_xy, log, *, views=2):
        confirms["n"] += 1
        out = {**obj, "stageConfirmed": False, "stageXy": list(stage_xy)}
        log.append({"step": "stage-land-unconfirmed", "from": obj.get("id"),
                    "xy": list(stage_xy), "role": "pistol", "views": views})
        return out

    def fake_recover(api, item, log):
        recovers["n"] += 1
        raise MissionFailed("lost staged drawer item e:79 (expected pistol near [-2.815, 0.865])")

    monkeypatch.setattr(chores_mod, "_confirm_stage_landing", fake_confirm)
    monkeypatch.setattr(chores_mod, "_protect_stage", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "_recover_drawer_carry", fake_recover)
    monkeypatch.setattr(chores_mod, "_move",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("move skipped")))

    runner = chores_mod.Chores("api")
    monkeypatch.setattr(runner, "_doing", lambda *a, **k: None)
    monkeypatch.setattr(runner, "publish", lambda **k: None)
    with pytest.raises(MissionFailed, match="still unconfirmed before transfer"):
        runner._transfer(staged, dst, "从台面移到柜子的抽屉 e:6")
    assert confirms["n"] == 1
    assert recovers["n"] == 1


def test_transfer_soft_keeps_unconfirmed_stage_via_recover(monkeypatch):
    """Unconfirmed stage that rematch soft-keeps still proceeds to the cabinet pick."""
    import steward.chores as chores_mod

    staged = _piece("e:78", "pistol", (-2.749, 0.328), 0.12)
    staged["stageXy"] = [-2.749, 0.328]
    staged["stagedRole"] = "pistol"
    staged["stageConfirmed"] = False
    kept = {**staged, "id": "e:501", "existence": 0.55, "stageConfirmed": False}
    dst = _piece("e:3", "drawer", (2.3, 0.6), 0.4, on=None)
    dst["role"] = "drawer"
    moved: list[str] = []

    monkeypatch.setattr(chores_mod, "_confirm_stage_landing",
                        lambda api, obj, xy, log, *, views=2: {
                            **obj, "stageConfirmed": False, "stageXy": list(xy)})
    monkeypatch.setattr(chores_mod, "_protect_stage", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "_recover_drawer_carry", lambda api, item, log: kept)
    monkeypatch.setattr(chores_mod, "_refresh_or_recover", lambda api, obj, **k: obj)
    monkeypatch.setattr(chores_mod, "_move",
                        lambda api, obj, place, observe, log, timeout_s, **kw: (
                            moved.append(obj["id"]) or {**obj, "on": observe["id"]}))

    runner = chores_mod.Chores("api")
    monkeypatch.setattr(runner, "_doing", lambda *a, **k: None)
    monkeypatch.setattr(runner, "publish", lambda **k: None)
    monkeypatch.setattr(runner, "_clearing", lambda phase, fn: fn())
    out = runner._transfer(staged, dst, "从台面移到柜子的抽屉 e:3")
    assert out is not None and out["id"] == "e:501"
    assert moved == ["e:501"]
    assert any(e.get("step") == "stage-unconfirmed-recover" and e.get("to") == "e:501"
               for e in runner.log)


def test_move_retries_pick_when_target_vanishes_after_grasp(monkeypatch):
    """Whitelist pick identity loss rematches via drawer recover, then picks the new id."""
    import steward.chores as chores_mod
    from steward.mission import MissionFailed

    picks = {"n": 0}
    recovers = {"n": 0}
    bear = _piece("e:80", "teddy_bear", (-2.70, 0.60), 0.15)
    board = _piece("e:4", "drawer", (0.0, 0.0), 0.4, on=None)
    board["role"] = "drawer"

    def fake_safe(api, skill, params, observe, log):
        return {"feasible": True, "withinRiskLimit": True, "risk": 0.04, "blockers": [],
                "skill": skill, "plan": {"spot": [-2.70, 0.60]}, "issues": []}

    def fake_run(api, receipt, timeout_s):
        if receipt["skill"] == "pick":
            picks["n"] += 1
            if picks["n"] == 1:
                raise MissionFailed(
                    "pick failed: e:680 is no longer in the ego state and nothing else is in the gripper")
            return {"state": "succeeded", "message": "holding e:901"}
        return {"state": "succeeded", "message": "placed"}

    def fake_recover(api, item, log):
        recovers["n"] += 1
        return {**item, "id": "e:901"}

    monkeypatch.setattr(chores_mod, "_approach_to_observe", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "_refresh", lambda api, obj: obj)
    monkeypatch.setattr(chores_mod, "_refresh_or_recover", lambda api, obj, **k: obj)
    monkeypatch.setattr(chores_mod, "_recover_drawer_carry", fake_recover)
    monkeypatch.setattr(chores_mod, "_held_drawer_match", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "_align_drawer_place", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "_confirm_drawer_landing",
                        lambda api, obj, drawer, spot, log: {**obj, "position": [spot[0], spot[1], 0.5]})
    monkeypatch.setattr(chores_mod, "_safe_receipt", fake_safe)
    monkeypatch.setattr(chores_mod, "_run", fake_run)
    monkeypatch.setattr(chores_mod, "_retreat_while_holding", lambda *a, **k: True)
    monkeypatch.setattr(chores_mod, "_landed", lambda api, obj, spot: {**obj, "position": [spot[0], spot[1], 0.8]})
    monkeypatch.setattr(chores_mod, "simtime_sleep", lambda *a, **k: None)

    log: list = []
    out = chores_mod._move("api", bear, {"on": "e:4"}, board, log, 30.0)
    assert picks["n"] == 2 and recovers["n"] >= 1
    assert out["id"] == "e:901"
    assert any(e.get("reason") == "grasp" for e in log if e.get("step") == "pick-retry")


def test_move_adopts_held_whitelist_after_identity_vanish(monkeypatch):
    """If jaws closed on a matching whitelist piece while the planned id vanished, skip re-pick."""
    import steward.chores as chores_mod
    from steward.mission import MissionFailed

    picks = {"n": 0}
    bear = _piece("e:80", "teddy_bear", (-2.70, 0.60), 0.15)
    board = _piece("e:4", "drawer", (0.0, 0.0), 0.4, on=None)
    board["role"] = "drawer"
    held = _piece("e:711", "teddy_bear", (-2.70, 0.60), 0.15)

    def fake_safe(api, skill, params, observe, log):
        return {"feasible": True, "withinRiskLimit": True, "risk": 0.04, "blockers": [],
                "skill": skill, "plan": {"spot": [-2.50, 0.60]}, "issues": []}

    def fake_run(api, receipt, timeout_s):
        if receipt["skill"] == "pick":
            picks["n"] += 1
            raise MissionFailed(
                "pick failed: e:680 is no longer in the ego state and nothing else is in the gripper")
        return {"state": "succeeded", "message": "placed"}

    monkeypatch.setattr(chores_mod, "_approach_to_observe", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "_refresh_or_recover", lambda api, obj, **k: obj)
    monkeypatch.setattr(chores_mod, "_recover_drawer_carry", lambda api, obj, log: obj)
    monkeypatch.setattr(chores_mod, "_held_drawer_match", lambda api, expected: held)
    monkeypatch.setattr(chores_mod, "_align_drawer_place", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "_confirm_drawer_landing",
                        lambda api, obj, drawer, spot, log: {**obj, "id": "e:711",
                                                            "drawerLandXy": [spot[0], spot[1]],
                                                            "drawerId": drawer["id"]})
    monkeypatch.setattr(chores_mod, "_safe_receipt", fake_safe)
    monkeypatch.setattr(chores_mod, "_run", fake_run)
    monkeypatch.setattr(chores_mod, "_landed", lambda api, obj, spot: obj)
    monkeypatch.setattr(chores_mod, "simtime_sleep", lambda *a, **k: None)

    log: list = []
    out = chores_mod._move("api", bear, {"on": "e:4"}, board, log, 30.0)
    assert picks["n"] == 1
    assert out["id"] == "e:711"
    assert any(e.get("step") == "pick-held-fallback" for e in log)


def test_confirm_drawer_landing_rebinds_released_track(monkeypatch):
    """After an unverified drawer place, a look into the box adopts the renumbered whitelist id."""
    import steward.chores as chores_mod

    drawer = _piece("e:19", "drawer", (-2.5, 0.6), 0.4, on=None)
    drawer["role"] = "drawer"
    drawer["position"] = [-2.5, 0.6, 0.55]
    drawer["sizeM"] = drawer["extentM"] = [0.5, 0.4, 0.22]
    stub = _piece("e:246", "pill_bottle", (-2.37, 0.50), 0.05, on=None)
    live = _piece("e:900", "pill_bottle", (-2.37, 0.50), 0.05, on="e:19")
    live["position"] = [-2.37, 0.50, 0.50]

    monkeypatch.setattr(chores_mod, "_approach_to_observe", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "simtime_sleep", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "_summary",
                        lambda api: {"objects": [drawer, live], "self": {}})
    monkeypatch.setattr(chores_mod, "_summary_object",
                        lambda summary, oid: next((o for o in summary["objects"] if o["id"] == oid), None))

    log: list = []
    out = chores_mod._confirm_drawer_landing("api", stub, drawer, [-2.37, 0.50], log)
    assert out["id"] == "e:900"
    assert out["drawerLandXy"] == [-2.37, 0.5]
    assert out["drawerId"] == "e:19"
    assert any(e.get("step") == "drawer-land-confirm" for e in log)


def test_confirm_stage_landing_rebinds_unverified_top_release(monkeypatch):
    """After unverified stage place, a look at stageXy adopts the renumbered whitelist id."""
    import steward.chores as chores_mod
    from steward.mission import MissionFailed

    stub = _piece("e:79", "pistol", (-2.815, 0.865), 0.12, on=None)
    stub["stagedRole"] = "pistol"
    stub["drawerId"] = "e:20"  # must be cleared after stage confirm
    live = _piece("e:492", "pistol", (-2.81, 0.87), 0.11, on="e:21")
    live["position"] = [-2.81, 0.87, 0.94]
    offsets: list[float | None] = []

    monkeypatch.setattr(
        chores_mod, "_approach_to_observe",
        lambda api, obj, distance=0.9, prefer_offset=None: offsets.append(prefer_offset))
    monkeypatch.setattr(chores_mod, "simtime_sleep", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "_summary",
                        lambda api: {"objects": [live], "self": {}})
    monkeypatch.setattr(
        chores_mod, "_refresh",
        lambda api, obj: (_ for _ in ()).throw(
            MissionFailed("e:79 (pistol) is no longer in the ego state")))

    log: list = []
    out = chores_mod._confirm_stage_landing("api", stub, [-2.815, 0.865], log)
    assert out["id"] == "e:492"
    assert out.get("stageXy") == [-2.815, 0.865]
    assert out.get("stagedRole") == "pistol"
    assert out.get("stageConfirmed") is True
    assert "drawerId" not in out
    assert offsets and offsets[0] == 0.0
    assert any(e.get("step") == "stage-land-confirm" and e.get("to") == "e:492" for e in log)


def test_confirm_stage_landing_logs_unconfirmed_when_empty(monkeypatch):
    """No whitelist track near stageXy → keep the anchor stub and log unconfirmed."""
    import steward.chores as chores_mod
    from steward.mission import MissionFailed

    stub = _piece("e:79", "pistol", (-2.815, 0.865), 0.12, on=None)
    stub["stagedRole"] = "pistol"
    bowl = _piece("e:7", "bowl", (-2.81, 0.87), 0.18, on="e:21")

    monkeypatch.setattr(chores_mod, "_approach_to_observe", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "simtime_sleep", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "_summary",
                        lambda api: {"objects": [bowl], "self": {}})
    monkeypatch.setattr(
        chores_mod, "_refresh",
        lambda api, obj: (_ for _ in ()).throw(
            MissionFailed("e:79 (pistol) is no longer in the ego state")))

    log: list = []
    out = chores_mod._confirm_stage_landing("api", stub, [-2.815, 0.865], log, views=2)
    assert out.get("stageXy") == [-2.815, 0.865]
    assert out.get("id") == "e:79"
    assert out.get("stageConfirmed") is False
    assert any(e.get("step") == "stage-land-unconfirmed" and e.get("views") == 2 for e in log)


def test_recover_drawer_carry_uses_drawer_land_anchor(monkeypatch):
    """In-drawer land XY + drawer id rematch when the release track id is gone."""
    import steward.chores as chores_mod
    from steward.mission import MissionFailed

    item = _piece("e:246", "pill_bottle", (-2.37, 0.50), 0.05, on=None)
    item["drawerLandXy"] = [-2.37, 0.50]
    item["drawerId"] = "e:19"
    item["stagedRole"] = "pill_bottle"
    drawer = _piece("e:19", "drawer", (-2.5, 0.6), 0.4, on=None)
    drawer["role"] = "drawer"
    drawer["position"] = [-2.5, 0.6, 0.55]
    drawer["sizeM"] = drawer["extentM"] = [0.5, 0.4, 0.22]
    live = _piece("e:901", "pill_bottle", (-2.36, 0.51), 0.05, on="e:19")
    live["position"] = [-2.36, 0.51, 0.50]

    monkeypatch.setattr(chores_mod, "_refresh",
                        lambda api, obj: (_ for _ in ()).throw(
                            MissionFailed(f"{obj['id']} (pill_bottle) is no longer in the ego state")))
    monkeypatch.setattr(chores_mod, "_approach_to_observe", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "simtime_sleep", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "_summary",
                        lambda api: {"objects": [drawer, live], "self": {}})
    monkeypatch.setattr(chores_mod, "_summary_object",
                        lambda summary, oid: next((o for o in summary["objects"] if o["id"] == oid), None))

    log: list = []
    out = chores_mod._recover_drawer_carry("api", item, log)
    assert out["id"] == "e:901"
    assert any(e.get("reason") == "matched" for e in log if e.get("step") == "drawer-rematch")


def test_close_rematches_drawer_on_host_when_open_id_is_gone(monkeypatch):
    """Closing after a long swap re-finds the drawer on the sideboard when e:21 vanished."""
    import steward.chores as chores_mod
    from steward.mission import MissionFailed

    stale = _piece("e:21", "drawer", (-2.5, 0.6), 0.4, on=None)
    stale["role"] = "drawer"
    stale["openingM"] = 0.30
    stale["hostId"] = "e:20"
    board = _piece("e:20", "sideboard", (-2.8, 0.6), 1.2, on=None)
    board["role"] = "sideboard"
    board["yawDeg"] = 0.0
    board["extentM"] = board["sizeM"] = [1.4, 0.5, 0.85]
    live = _piece("e:99", "drawer", (-2.45, 0.6), 0.45, on=None)
    live["role"] = "drawer"
    live["openingM"] = 0.28
    live["position"] = [-2.45, 0.6, 0.55]
    live["extentM"] = live["sizeM"] = [0.5, 0.4, 0.22]
    calls = {"close": 0}

    def fake_refresh(api, obj):
        if obj.get("id") == "e:21":
            raise MissionFailed("e:21 (drawer) is no longer in the ego state")
        if obj.get("id") == "e:20":
            return board
        if obj.get("id") == "e:99":
            return {**live, "openingM": live["openingM"]}
        return obj

    def fake_call(method, url, body=None, **kwargs):
        assert method == "POST" and body["skill"] == "close"
        assert body["params"]["target"] == "e:99"
        calls["close"] += 1
        live["openingM"] = 0.0
        return {"feasible": True, "withinRiskLimit": True, "risk": 0.0, "blockers": [],
                "skill": "close", "plan": {}, "issues": [], "id": "rcpt-1"}

    monkeypatch.setattr(chores_mod, "_refresh", fake_refresh)
    monkeypatch.setattr(chores_mod, "_refresh_or_recover", lambda api, obj, **k: board)
    monkeypatch.setattr(chores_mod, "_approach_to_observe", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "_view", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "_summary",
                        lambda api: {"objects": [board, live], "self": {}})
    monkeypatch.setattr(chores_mod, "_summary_object",
                        lambda summary, oid: next((o for o in summary["objects"] if o["id"] == oid), None))
    monkeypatch.setattr(chores_mod, "call", fake_call)
    monkeypatch.setattr(chores_mod, "_run", lambda api, receipt, timeout_s: {"state": "succeeded"})
    monkeypatch.setattr(chores_mod, "simtime_sleep", lambda *a, **k: None)

    runner = chores_mod.Chores("api")
    monkeypatch.setattr(runner, "_doing", lambda *a, **k: None)
    monkeypatch.setattr(runner, "publish", lambda **k: None)
    runner._close(stale, board)
    assert calls["close"] == 2  # close + seat
    assert any(e.get("reason") == "matched" and e.get("to") == "e:99"
               for e in runner.log if e.get("step") == "drawer-close-rematch")
    assert any(e.get("step") == "drawer-close-seat" for e in runner.log)
    assert any(e.get("step") == "drawer-close-verified" for e in runner.log)


def test_close_skips_when_host_look_finds_no_open_drawer(monkeypatch):
    """If the host look sees no drawer track, treat the face as already shut."""
    import steward.chores as chores_mod
    from steward.mission import MissionFailed

    stale = _piece("e:21", "drawer", (-2.5, 0.6), 0.4, on=None)
    stale["role"] = "drawer"
    stale["openingM"] = 0.30
    board = _piece("e:20", "sideboard", (-2.8, 0.6), 1.2, on=None)
    board["role"] = "sideboard"
    board["yawDeg"] = 0.0
    board["extentM"] = board["sizeM"] = [1.4, 0.5, 0.85]

    monkeypatch.setattr(chores_mod, "_refresh",
                        lambda api, obj: (_ for _ in ()).throw(
                            MissionFailed("e:21 (drawer) is no longer in the ego state")))
    monkeypatch.setattr(chores_mod, "_refresh_or_recover", lambda api, obj, **k: board)
    monkeypatch.setattr(chores_mod, "_approach_to_observe", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "_view", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "_summary", lambda api: {"objects": [board], "self": {}})
    monkeypatch.setattr(chores_mod, "simtime_sleep", lambda *a, **k: None)

    runner = chores_mod.Chores("api")
    monkeypatch.setattr(runner, "_doing", lambda *a, **k: None)
    monkeypatch.setattr(runner, "publish", lambda **k: None)
    runner._close(stale, board)
    assert any(e.get("step") == "drawer-close-skip" and e.get("reason") == "no-drawer"
               for e in runner.log)


def test_close_retries_when_opening_still_ajar(monkeypatch):
    """Close skill success with ajar openingM is retried until under the world gate, then seat."""
    import steward.chores as chores_mod

    drawer = _piece("e:8", "drawer", (2.3, 0.6), 0.4, on=None)
    drawer["role"] = "drawer"
    drawer["openingM"] = 0.30
    cabinet = _piece("e:7", "cabinet", (2.7, 0.6), 0.6, on=None)
    cabinet["role"] = "cabinet"
    # After each close: still above world gate, then at/under the gate; seat is a fourth call.
    openings = [0.30, 0.18, 0.05, 0.015, 0.0]
    calls = {"close": 0}
    distances: list[float | None] = []

    def fake_refresh(api, obj):
        if obj.get("id") == "e:8":
            return {**drawer, "openingM": openings[min(calls["close"], len(openings) - 1)]}
        return obj

    def fake_call(method, url, body=None, **kwargs):
        assert body["skill"] == "close" and body["params"]["target"] == "e:8"
        distances.append(body["params"].get("distance"))
        calls["close"] += 1
        return {"feasible": True, "withinRiskLimit": True, "risk": 0.0, "blockers": [],
                "skill": "close", "plan": {}, "issues": [], "id": f"rcpt-{calls['close']}"}

    monkeypatch.setattr(chores_mod, "_refresh", fake_refresh)
    monkeypatch.setattr(chores_mod, "_approach_to_observe", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "call", fake_call)
    monkeypatch.setattr(chores_mod, "_run", lambda api, receipt, timeout_s: {"state": "succeeded"})
    monkeypatch.setattr(chores_mod, "simtime_sleep", lambda *a, **k: None)

    runner = chores_mod.Chores("api")
    monkeypatch.setattr(runner, "_doing", lambda *a, **k: None)
    monkeypatch.setattr(runner, "publish", lambda **k: None)
    runner._close(drawer, cabinet)
    assert calls["close"] == 4
    assert distances[-1] == pytest.approx(chores_mod.DRAWER_CLOSE_SEAT_M)
    assert any(e.get("step") == "drawer-close-retry" and e.get("reason") == "still-open"
               for e in runner.log)
    assert any(e.get("step") == "drawer-close-seat" for e in runner.log)
    assert any(e.get("step") == "drawer-close-verified" and e.get("reason") == "seat"
               for e in runner.log)


def test_close_seats_when_opening_first_hits_world_gate(monkeypatch):
    """Under-gate openingM still gets one explicit seat push before accept."""
    import steward.chores as chores_mod

    drawer = _piece("e:8", "drawer", (2.3, 0.6), 0.4, on=None)
    drawer["role"] = "drawer"
    drawer["openingM"] = 0.30
    cabinet = _piece("e:7", "cabinet", (2.7, 0.6), 0.6, on=None)
    cabinet["role"] = "cabinet"
    # 0.005 passes drawers_closed on paper; seat still runs for joint residual.
    openings = [0.30, 0.005, 0.0]
    calls = {"close": 0}
    distances: list[float | None] = []

    def fake_refresh(api, obj):
        if obj.get("id") == "e:8":
            return {**drawer, "openingM": openings[min(calls["close"], len(openings) - 1)]}
        return obj

    def fake_call(method, url, body=None, **kwargs):
        distances.append((body or {}).get("params", {}).get("distance"))
        calls["close"] += 1
        return {"feasible": True, "withinRiskLimit": True, "risk": 0.0, "blockers": [],
                "skill": "close", "plan": {}, "issues": [], "id": f"rcpt-{calls['close']}"}

    monkeypatch.setattr(chores_mod, "_refresh", fake_refresh)
    monkeypatch.setattr(chores_mod, "_approach_to_observe", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "call", fake_call)
    monkeypatch.setattr(chores_mod, "_run", lambda api, receipt, timeout_s: {"state": "succeeded"})
    monkeypatch.setattr(chores_mod, "simtime_sleep", lambda *a, **k: None)

    runner = chores_mod.Chores("api")
    monkeypatch.setattr(runner, "_doing", lambda *a, **k: None)
    monkeypatch.setattr(runner, "publish", lambda **k: None)
    runner._close(drawer, cabinet)
    assert calls["close"] == 2
    assert distances[0] is None and distances[1] == pytest.approx(chores_mod.DRAWER_CLOSE_SEAT_M)
    assert any(e.get("step") == "drawer-close-verified" and e.get("reason") == "seat"
               for e in runner.log)


def test_close_seats_when_already_reading_shut(monkeypatch):
    """ES already under the gate still seats once (seed2 joint residual)."""
    import steward.chores as chores_mod

    drawer = _piece("e:8", "drawer", (2.3, 0.6), 0.4, on=None)
    drawer["role"] = "drawer"
    drawer["openingM"] = 0.005
    cabinet = _piece("e:7", "cabinet", (2.7, 0.6), 0.6, on=None)
    cabinet["role"] = "cabinet"
    calls = {"close": 0}

    def fake_call(method, url, body=None, **kwargs):
        calls["close"] += 1
        assert body["params"].get("distance") == pytest.approx(chores_mod.DRAWER_CLOSE_SEAT_M)
        return {"feasible": True, "withinRiskLimit": True, "risk": 0.0, "blockers": [],
                "skill": "close", "plan": {}, "issues": [], "id": "seat"}

    monkeypatch.setattr(chores_mod, "_refresh",
                        lambda api, obj: {**drawer, "openingM": 0.0})
    monkeypatch.setattr(chores_mod, "_approach_to_observe", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "call", fake_call)
    monkeypatch.setattr(chores_mod, "_run", lambda api, receipt, timeout_s: {"state": "succeeded"})
    monkeypatch.setattr(chores_mod, "simtime_sleep", lambda *a, **k: None)

    runner = chores_mod.Chores("api")
    monkeypatch.setattr(runner, "_doing", lambda *a, **k: None)
    monkeypatch.setattr(runner, "publish", lambda **k: None)
    runner._close(drawer, cabinet)
    assert calls["close"] == 1
    assert any(e.get("step") == "drawer-close-under-gate" for e in runner.log)
    assert any(e.get("step") == "drawer-close-verified" and e.get("reason") == "seat"
               for e in runner.log)


def test_close_fails_when_still_open_after_retries(monkeypatch):
    """Exhausted close retries raise when openingM stays above the shut gate."""
    import steward.chores as chores_mod
    from steward.mission import MissionFailed

    drawer = _piece("e:8", "drawer", (2.3, 0.6), 0.4, on=None)
    drawer["role"] = "drawer"
    drawer["openingM"] = 0.30
    cabinet = _piece("e:7", "cabinet", (2.7, 0.6), 0.6, on=None)
    cabinet["role"] = "cabinet"
    calls = {"close": 0}

    def fake_call(method, url, body=None, **kwargs):
        calls["close"] += 1
        return {"feasible": True, "withinRiskLimit": True, "risk": 0.0, "blockers": [],
                "skill": "close", "plan": {}, "issues": [], "id": "rcpt"}

    monkeypatch.setattr(chores_mod, "_refresh",
                        lambda api, obj: {**drawer, "openingM": 0.22})
    monkeypatch.setattr(chores_mod, "_approach_to_observe", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "call", fake_call)
    monkeypatch.setattr(chores_mod, "_run", lambda api, receipt, timeout_s: {"state": "succeeded"})
    monkeypatch.setattr(chores_mod, "simtime_sleep", lambda *a, **k: None)

    runner = chores_mod.Chores("api")
    monkeypatch.setattr(runner, "_doing", lambda *a, **k: None)
    monkeypatch.setattr(runner, "publish", lambda **k: None)
    with pytest.raises(MissionFailed, match="still open 22 cm after 3 close attempts"):
        runner._close(drawer, cabinet)
    assert calls["close"] == 3
    assert sum(1 for e in runner.log if e.get("step") == "drawer-close-retry") == 3


def test_close_accepts_already_closed_when_under_world_gate(monkeypatch):
    """Skill 'already closed' with ES under the world gate seats once, without reopen."""
    import steward.chores as chores_mod

    drawer = _piece("e:22", "drawer", (-2.5, 0.6), 0.4, on=None)
    drawer["role"] = "drawer"
    drawer["openingM"] = 0.30
    board = _piece("e:23", "sideboard", (-2.73, 0.59), 1.2, on=None)
    board["role"] = "sideboard"
    state = {"openingM": 0.30, "reopen": 0}
    calls = {"close": 0}
    distances: list[float | None] = []

    def fake_refresh(api, obj):
        if obj.get("id") == "e:22":
            return {**drawer, "openingM": state["openingM"]}
        return obj

    def fake_recover_opening(api, obj, log, timeout_s):
        state["reopen"] += 1
        state["openingM"] = 0.25
        log.append({"step": "drawer-reopen", "drawer": obj["id"], "openingM": 0.25})
        return {**drawer, "openingM": 0.25}

    def fake_call(method, url, body=None, **kwargs):
        skill = (body or {}).get("skill")
        if skill == "close":
            calls["close"] += 1
            distances.append((body or {}).get("params", {}).get("distance"))
            if calls["close"] == 1:
                return {"feasible": True, "withinRiskLimit": True, "risk": 0.0, "blockers": [],
                        "skill": "close", "plan": {}, "issues": [], "id": "c1"}
            if calls["close"] == 2:
                # Second shove refused: skill thinks shut while ES still shows a tiny residual.
                return {"feasible": False, "withinRiskLimit": False, "risk": 1.0, "blockers": [],
                        "skill": "close", "plan": {}, "issues": ["e:22 is already closed"], "id": "c2"}
            # Seat push with explicit distance.
            return {"feasible": True, "withinRiskLimit": True, "risk": 0.0, "blockers": [],
                    "skill": "close", "plan": {}, "issues": [], "id": "seat"}
        return {"feasible": True, "withinRiskLimit": True, "risk": 0.0, "blockers": [],
                "skill": skill or "navigate", "plan": {}, "issues": [], "id": "nav"}

    def fake_run(api, receipt, timeout_s):
        if receipt.get("skill") == "close":
            # Still above world gate so verify retries; residual appears only on already-closed look.
            state["openingM"] = 0.05
        return {"state": "succeeded"}

    monkeypatch.setattr(chores_mod, "_refresh", fake_refresh)
    monkeypatch.setattr(chores_mod, "_refresh_or_recover", lambda api, obj, **k: fake_refresh(api, obj))
    monkeypatch.setattr(chores_mod, "_recover_drawer_opening", fake_recover_opening)
    monkeypatch.setattr(chores_mod, "_approach_to_observe", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "call", fake_call)
    monkeypatch.setattr(chores_mod, "_run", fake_run)
    monkeypatch.setattr(chores_mod, "simtime_sleep", lambda *a, **k: None)

    def live_then_residual(drawer_obj, furniture=None):
        # After already-closed predict, ES drops under the world gate without reopen.
        if calls["close"] >= 2:
            state["openingM"] = 0.005
        return fake_refresh("api", drawer_obj)

    runner = chores_mod.Chores("api")
    monkeypatch.setattr(runner, "_doing", lambda *a, **k: None)
    monkeypatch.setattr(runner, "publish", lambda **k: None)
    monkeypatch.setattr(runner, "_live_drawer", live_then_residual)
    runner._close(drawer, board)
    assert calls["close"] == 3 and state["reopen"] == 0
    assert distances[2] == pytest.approx(chores_mod.DRAWER_CLOSE_SEAT_M)
    assert any(e.get("step") == "drawer-close-retry" and e.get("reason") == "already-closed"
               for e in runner.log)
    assert any(e.get("step") == "drawer-close-verified" and e.get("reason") == "seat"
               for e in runner.log)


def test_close_rematches_host_after_infeasible_receipt(monkeypatch):
    """Infeasible close on a renumbered drawer rematches via the host (seed2g e:457)."""
    import steward.chores as chores_mod

    stale = _piece("e:457", "drawer", (-2.4, 0.5), 0.4, on=None)
    stale["role"] = "drawer"
    stale["openingM"] = 0.30
    stale["hostId"] = "e:18"
    board = _piece("e:18", "sideboard", (-2.73, 0.59), 1.2, on=None)
    board["role"] = "sideboard"
    good = _piece("e:19", "drawer", (-2.5, 0.6), 0.4, on=None)
    good["role"] = "drawer"
    good["openingM"] = 0.28
    calls = {"close": 0}
    openings = [0.28, 0.0]

    def fake_refresh(api, obj):
        oid = obj.get("id") or ""
        if oid.startswith("stale:") or oid == "e:457":
            raise chores_mod.MissionFailed(f"{oid} is no longer in the ego state")
        if oid == "e:19":
            return {**good, "openingM": openings[min(calls["close"], len(openings) - 1)]}
        if oid == "e:18":
            return board
        return obj

    def fake_call(method, url, body=None, **kwargs):
        skill = (body or {}).get("skill")
        if skill == "close":
            calls["close"] += 1
            if calls["close"] == 1:
                return {"feasible": False, "withinRiskLimit": False, "risk": 1.0,
                        "blockers": [], "skill": "close", "plan": {},
                        "issues": ["end pose in front of e:457 collides with e:18"], "id": "bad"}
            return {"feasible": True, "withinRiskLimit": True, "risk": 0.0, "blockers": [],
                    "skill": "close", "plan": {}, "issues": [], "id": "ok"}
        return {"feasible": True, "withinRiskLimit": True, "risk": 0.0, "blockers": [],
                "skill": skill or "navigate", "plan": {}, "issues": [], "id": "nav"}

    monkeypatch.setattr(chores_mod, "_refresh", fake_refresh)
    monkeypatch.setattr(chores_mod, "_refresh_or_recover", lambda api, obj, **k: fake_refresh(api, obj))
    monkeypatch.setattr(chores_mod, "_drawer_of",
                        lambda summary, furniture: {**good, "openingM": 0.28})
    monkeypatch.setattr(chores_mod, "_summary",
                        lambda api: {"objects": [board, good], "self": {"holding": None}})
    monkeypatch.setattr(chores_mod, "_summary_object",
                        lambda summary, oid: board if oid == "e:18" else good if oid == "e:19" else None)
    monkeypatch.setattr(chores_mod, "_approach_to_observe", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "_view", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "_retreat_while_holding", lambda *a, **k: True)
    monkeypatch.setattr(chores_mod, "call", fake_call)
    monkeypatch.setattr(chores_mod, "_run", lambda api, receipt, timeout_s: {"state": "succeeded"})
    monkeypatch.setattr(chores_mod, "simtime_sleep", lambda *a, **k: None)

    runner = chores_mod.Chores("api")
    monkeypatch.setattr(runner, "_doing", lambda *a, **k: None)
    monkeypatch.setattr(runner, "publish", lambda **k: None)
    runner._close(stale, board)
    assert calls["close"] == 3  # infeasible, rematch close, seat
    assert any(e.get("step") == "drawer-close-retry" and e.get("reason") == "infeasible"
               for e in runner.log)
    assert any(e.get("step") == "drawer-close-seat" for e in runner.log)
    assert any(e.get("step") == "drawer-close-verified" and e.get("openingM") == 0.0
               for e in runner.log)


def test_live_top_support_rematches_when_host_id_became_teddy(monkeypatch):
    """Stage-protect can rename the sideboard track to teddy; place must rebind the furniture."""
    import steward.chores as chores_mod

    stolen = _piece("e:19", "teddy_bear", (-2.815, 0.332), 0.14, on=None)
    stolen["role"] = "teddy_bear"
    stolen["position"] = [-2.815, 0.332, 0.94]
    board = _piece("e:88", "sideboard", (-2.73, 0.59), 1.2, on=None)
    board["role"] = "sideboard"
    board["yawDeg"] = 0.0
    board["position"] = [-2.73, 0.59, 0.42]
    board["extentM"] = board["sizeM"] = [1.42, 0.49, 0.84]
    remembered = {**stolen, "role": "sideboard", "category": "sideboard",
                  "position": [-2.73, 0.59, 0.42]}
    log: list = []

    monkeypatch.setattr(chores_mod, "_refresh", lambda api, obj: stolen)
    monkeypatch.setattr(chores_mod, "_summary",
                        lambda api: {"objects": [stolen, board], "self": {}})
    found = chores_mod._live_top_support("api", remembered, log)
    assert found["id"] == "e:88"
    assert (found.get("role") or found.get("category")) == "sideboard"
    assert any(e.get("step") == "top-support-rematch" and e.get("reason") == "role-lost"
               for e in log)
    assert any(e.get("step") == "top-support-rematch" and e.get("to") == "e:88"
               for e in log)


def test_live_top_support_ignores_nearer_counter_for_sideboard_host(monkeypatch):
    """Sideboard flipped to wall must not stage onto a nearer kitchen counter."""
    import steward.chores as chores_mod

    wall = _piece("e:23", "wall", (-2.73, 0.59), 1.2, on=None)
    wall["role"] = "wall"
    wall["position"] = [-2.73, 0.59, 1.2]
    # Farther than the counter — without STAGE_HOST rematch, min-distance would pick the counter.
    board = _piece("e:88", "sideboard", (2.5, 0.6), 1.2, on=None)
    board["role"] = "sideboard"
    board["position"] = [2.5, 0.6, 0.42]
    board["extentM"] = board["sizeM"] = [1.42, 0.49, 0.84]
    counter = _piece("e:15", "counter", (-2.5, 0.7), 1.5, on=None)
    counter["role"] = "counter"
    counter["position"] = [-2.5, 0.7, 0.9]
    remembered = {"id": "e:23", "role": "sideboard", "category": "sideboard",
                  "position": [-2.73, 0.59, 0.42]}
    log: list = []

    def fake_refresh(api, obj):
        if obj.get("id") == "e:23":
            return wall
        return obj

    monkeypatch.setattr(chores_mod, "_refresh", fake_refresh)
    monkeypatch.setattr(chores_mod, "_summary",
                        lambda api: {"objects": [wall, board, counter], "self": {}})
    found = chores_mod._live_top_support("api", remembered, log)
    assert found["id"] == "e:88"
    assert (found.get("role") or found.get("category")) == "sideboard"
    assert not any(e.get("to") == "e:15" for e in log if e.get("step") == "top-support-rematch")


def test_transfer_skips_lost_source_drawer_item(monkeypatch):
    """A vanished cabinet track mid-swap is skipped; staged pieces still must rematch."""
    import steward.chores as chores_mod
    from steward.mission import MissionFailed

    lost = _piece("e:413", "pill_bottle", (2.28, 0.46), 0.05)
    lost["drawerId"] = "e:3"
    lost["stagedRole"] = "pill_bottle"
    dst = _piece("e:21", "drawer", (-2.5, 0.6), 0.4, on=None)
    dst["role"] = "drawer"

    monkeypatch.setattr(
        chores_mod, "_recover_drawer_carry",
        lambda *a, **k: (_ for _ in ()).throw(
            MissionFailed("lost staged drawer item e:413 (expected pill_bottle near [2.28, 0.46]): "
                          "not in ('teddy_bear', 'pistol', 'toy_airplane', 'pill_bottle', 'perfume_bottle')")))
    monkeypatch.setattr(chores_mod, "_refresh_or_recover", lambda api, obj, **k: dst)
    monkeypatch.setattr(chores_mod, "_move", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("_move must not run for a skipped item")))

    runner = chores_mod.Chores("api")
    monkeypatch.setattr(runner, "_doing", lambda *a, **k: None)
    monkeypatch.setattr(runner, "publish", lambda **k: None)
    assert runner._transfer(lost, dst, "从柜子的抽屉移到餐边柜的抽屉 e:21") is None
    assert any(e.get("step") == "skip" and e.get("reason") == "lost-drawer-item"
               for e in runner.log)

    staged = {**lost, "stageXy": [-2.82, 0.33], "id": "e:78", "stagedRole": "teddy_bear",
              "role": "teddy_bear", "category": "teddy_bear"}
    with pytest.raises(MissionFailed, match="lost staged drawer item"):
        runner._transfer(staged, dst, "从台面移到柜子的抽屉 e:3")


def test_recover_drawer_carry_reconfirms_stage_before_failing(monkeypatch):
    """Staged rematch tries one more stage look before raising lost-item."""
    import steward.chores as chores_mod
    from steward.mission import MissionFailed

    staged = _piece("e:95", "pistol", (-2.82, 0.33), 0.12)
    staged["stageXy"] = [-2.82, 0.33]
    staged["stagedRole"] = "pistol"
    found = {**staged, "id": "e:500"}
    calls = {"confirm": 0}

    def fake_refresh(api, obj):
        raise MissionFailed(f"{obj.get('id')} is no longer in the ego state")

    def fake_confirm(api, obj, stage_xy, log, *, views=2):
        calls["confirm"] += 1
        log.append({"step": "stage-land-confirm", "from": obj.get("id"), "to": found["id"],
                    "xy": stage_xy, "role": "pistol", "views": views})
        return found

    def fake_match(summary, item, xy, radius=0.35):
        return found if calls["confirm"] else None

    monkeypatch.setattr(chores_mod, "_refresh", fake_refresh)
    monkeypatch.setattr(chores_mod, "_approach_to_observe", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "_summary",
                        lambda api: {"objects": [found], "self": {"holding": None}})
    monkeypatch.setattr(chores_mod, "_match_staged_drawer_item", fake_match)
    monkeypatch.setattr(chores_mod, "_match_by_footprint", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "_confirm_stage_landing", fake_confirm)
    monkeypatch.setattr(chores_mod, "_protect_stage", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "_freshen_drawer_pick", lambda api, obj, log, **k: obj)
    monkeypatch.setattr(chores_mod, "simtime_sleep", lambda *a, **k: None)

    log: list = []
    out = chores_mod._recover_drawer_carry("api", staged, log)
    assert out["id"] == "e:500"
    assert calls["confirm"] == 1
    assert any(e.get("reason") == "stage-reconfirm" for e in log if e.get("step") == "drawer-rematch")


def test_move_retries_place_when_drawer_reports_full(monkeypatch):
    """Open-drawer place with no free spot retreats, re-aligns, and retries."""
    import steward.chores as chores_mod
    from steward.mission import MissionFailed

    plane = _piece("e:250", "toy_airplane", (2.3, 0.6), 0.22)
    plane["drawerId"] = "e:9"
    plane["stagedRole"] = "toy_airplane"
    drawer = _piece("e:19", "drawer", (-2.5, 0.6), 0.4, on=None)
    drawer["role"] = "drawer"
    places = {"n": 0}
    log: list = []

    def fake_safe(api, skill, params, observe, log_):
        if skill == "pick":
            return {"feasible": True, "withinRiskLimit": True, "risk": 0.0, "blockers": [],
                    "skill": skill, "plan": {"spot": [2.3, 0.6]}, "issues": []}
        return {"feasible": True, "withinRiskLimit": True, "risk": 0.0, "blockers": [],
                "skill": skill, "plan": {"spot": [-2.4, 0.6]}, "issues": []}

    def fake_run(api, receipt, timeout_s):
        if receipt["skill"] == "pick":
            return {"state": "succeeded", "message": "holding e:250"}
        places["n"] += 1
        if places["n"] == 1:
            raise MissionFailed("place infeasible: ['no free spot left in the open part of e:19']")
        return {"state": "succeeded", "message": "placed"}

    monkeypatch.setattr(chores_mod, "_approach_to_observe", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "_refresh", lambda api, obj: obj)
    monkeypatch.setattr(chores_mod, "_refresh_or_recover", lambda api, obj, **k: obj)
    monkeypatch.setattr(chores_mod, "_recover_drawer_carry", lambda api, item, log_: item)
    monkeypatch.setattr(chores_mod, "_held_drawer_match", lambda api, obj: obj)
    monkeypatch.setattr(chores_mod, "_align_drawer_place", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "_confirm_drawer_landing",
                        lambda api, obj, drawer, spot, log_: {**obj, "position": [spot[0], spot[1], 0.5]})
    monkeypatch.setattr(chores_mod, "_safe_receipt", fake_safe)
    monkeypatch.setattr(chores_mod, "_run", fake_run)
    monkeypatch.setattr(chores_mod, "_retreat_while_holding", lambda *a, **k: True)
    monkeypatch.setattr(chores_mod, "_landed",
                        lambda api, obj, spot: {**obj, "position": [spot[0], spot[1], 0.5]})
    monkeypatch.setattr(chores_mod, "simtime_sleep", lambda *a, **k: None)

    out = chores_mod._move("api", plane, {"on": "e:19"}, drawer, log, 30.0)
    assert out["id"] == "e:250"
    assert places["n"] == 2
    assert any(e.get("step") == "place-retry" and e.get("reason") == "drawer-full" for e in log)


def test_move_drawer_full_clears_one_occupant(monkeypatch):
    """Drawer-full with evacuate_to parks the held piece, clears one occupant, re-grasps."""
    import steward.chores as chores_mod
    from steward.mission import MissionFailed

    plane = _piece("e:250", "toy_airplane", (2.3, 0.6), 0.22)
    plane["drawerId"] = "e:9"
    plane["stagedRole"] = "toy_airplane"
    drawer = _piece("e:19", "drawer", (-2.5, 0.6), 0.4, on="e:2")
    drawer["role"] = "drawer"
    drawer["position"] = [-2.5, 0.6, 0.55]
    drawer["sizeM"] = [0.5, 0.4, 0.22]
    drawer["extentM"] = [0.5, 0.4, 0.22]
    host = _piece("e:2", "cabinet", (-2.5, 0.9), 0.8, on=None)
    host["role"] = "cabinet"
    host["yawDeg"] = 0.0
    host["position"] = [-2.5, 0.9, 0.7]
    host["sizeM"] = [0.8, 0.5, 0.9]
    host["extentM"] = [0.8, 0.5, 0.9]
    other = _piece("e:8", "drawer", (2.3, 0.6), 0.4, on=None)
    other["role"] = "drawer"
    blocker = _piece("e:360", "pill_bottle", (-2.45, 0.55), 0.06, on="e:19")
    blocker["position"] = [-2.45, 0.55, 0.50]
    blocker["existence"] = 1.0
    places = {"n": 0}
    holding = {"id": "e:250"}
    log: list = []

    def fake_summary(api):
        return {"objects": [drawer, host, other, blocker, plane],
                "self": {"holding": holding["id"]}}

    def fake_safe(api, skill, params, observe, log_):
        if skill == "pick":
            holding["id"] = params.get("object") or observe.get("id")
            return {"feasible": True, "withinRiskLimit": True, "risk": 0.0, "blockers": [],
                    "skill": skill, "plan": {"spot": [2.3, 0.6]}, "issues": []}
        return {"feasible": True, "withinRiskLimit": True, "risk": 0.0, "blockers": [],
                "skill": skill, "plan": {"spot": list(params.get("at") or [-2.4, 0.6])},
                "issues": []}

    def fake_run(api, receipt, timeout_s):
        if receipt["skill"] == "pick":
            return {"state": "succeeded", "message": f"holding {holding['id']}"}
        places["n"] += 1
        # First inbound place into the drawer fails full; park + clear + retry succeed.
        if places["n"] == 1:
            raise MissionFailed("place infeasible: ['no free spot left in the open part of e:19']")
        if receipt.get("params", {}).get("on") == host["id"] or (
                isinstance(receipt.get("plan", {}).get("spot"), list)
                and receipt["skill"] == "place" and places["n"] == 2):
            holding["id"] = None
        return {"state": "succeeded", "message": "placed"}

    clear_calls = {"n": 0}

    def fake_park_clear(api, held, dr, evacuate_to, log_, timeout_s):
        clear_calls["n"] += 1
        holding["id"] = held["id"]
        log_.append({"step": "drawer-full-park", "object": held["id"], "on": host["id"],
                     "cleared": blocker["id"]})
        log_.append({"step": "drawer-full-clear", "cleared": blocker["id"],
                     "to": evacuate_to["id"], "drawer": dr["id"]})
        return held, dr

    monkeypatch.setattr(chores_mod, "_summary", fake_summary)
    monkeypatch.setattr(chores_mod, "_approach_to_observe", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "_refresh", lambda api, obj: obj)
    monkeypatch.setattr(chores_mod, "_refresh_or_recover", lambda api, obj, **k: obj)
    monkeypatch.setattr(chores_mod, "_recover_drawer_carry", lambda api, item, log_: item)
    monkeypatch.setattr(chores_mod, "_held_drawer_match", lambda api, obj: obj)
    monkeypatch.setattr(chores_mod, "_align_drawer_place", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "_confirm_drawer_landing",
                        lambda api, obj, drawer_, spot, log_: {**obj, "position": [spot[0], spot[1], 0.5]})
    monkeypatch.setattr(chores_mod, "_safe_receipt", fake_safe)
    monkeypatch.setattr(chores_mod, "_run", fake_run)
    monkeypatch.setattr(chores_mod, "_retreat_while_holding", lambda *a, **k: True)
    monkeypatch.setattr(chores_mod, "_landed",
                        lambda api, obj, spot: {**obj, "position": [spot[0], spot[1], 0.5]})
    monkeypatch.setattr(chores_mod, "simtime_sleep", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "_park_held_clear_drawer_occupant", fake_park_clear)

    out = chores_mod._move("api", plane, {"on": "e:19"}, drawer, log, 30.0,
                           drawer_evacuate_to=other)
    assert out["id"] == "e:250"
    assert clear_calls["n"] == 1
    assert any(e.get("step") == "drawer-full-clear" and e.get("cleared") == "e:360" for e in log)


def test_evacuate_drawer_leftovers_moves_deployed_skips_lookalikes(monkeypatch):
    """Leftover deployed roles leave the drawer; unconfirmed car_body is not force-grasped."""
    import steward.chores as chores_mod

    drawer = _piece("e:19", "drawer", (-2.5, 0.6), 0.4, on="e:2")
    drawer["role"] = "drawer"
    drawer["position"] = [-2.5, 0.6, 0.55]
    drawer["sizeM"] = [0.5, 0.4, 0.22]
    drawer["extentM"] = [0.5, 0.4, 0.22]
    dest = _piece("e:8", "drawer", (2.3, 0.6), 0.4, on=None)
    dest["role"] = "drawer"
    pill = _piece("e:360", "pill_bottle", (-2.45, 0.55), 0.06, on="e:19")
    pill["position"] = [-2.45, 0.55, 0.50]
    pill["existence"] = 1.0
    perfume = _piece("e:376", "perfume_bottle", (-2.55, 0.58), 0.07, on="e:19")
    perfume["position"] = [-2.55, 0.58, 0.50]
    perfume["existence"] = 1.0
    pistol = _piece("e:1022", "pistol", (-2.48, 0.60), 0.10, on="e:19")
    pistol["position"] = [-2.48, 0.60, 0.50]
    pistol["existence"] = 1.0
    car = _piece("e:99", "car_body", (-2.40, 0.62), 0.12, on="e:19")
    car["position"] = [-2.40, 0.62, 0.50]
    car["existence"] = 1.0
    car["role"] = "car_body"
    car["category"] = "car_body"

    transferred: list[str] = []

    monkeypatch.setattr(chores_mod, "_summary",
                        lambda api: {"objects": [drawer, dest, pill, perfume, pistol, car],
                                     "self": {"holding": None}})
    monkeypatch.setattr(chores_mod, "_refresh_or_recover", lambda api, obj, **k: obj)

    runner = chores_mod.Chores("api")
    monkeypatch.setattr(runner, "_doing", lambda *a, **k: None)
    monkeypatch.setattr(runner, "publish", lambda **k: None)

    def fake_transfer(item, dst, how, **kw):
        transferred.append(item["id"])
        return {**item, "on": dst["id"]}

    monkeypatch.setattr(runner, "_transfer", fake_transfer)

    n = runner._evacuate_drawer_leftovers(
        drawer, dest, "清到对面抽屉", skip_roles={"pistol", "teddy_bear"})
    assert n == 2
    assert transferred == ["e:376", "e:360"]  # larger footprint first; staged pistol skipped
    assert any(e.get("step") == "drawer-evacuate" and e.get("id") == "e:360" for e in runner.log)
    assert "e:1022" not in transferred
    assert not any(e.get("id") == "e:99" for e in runner.log)


def test_drawer_host_furniture_falls_back_to_nearest_cabinet(monkeypatch):
    """Rematched drawers without ``on`` still resolve a park host by XY."""
    import steward.chores as chores_mod

    drawer = _piece("e:4", "drawer", (2.3, 0.6), 0.4, on=None)
    drawer["role"] = "drawer"
    drawer["position"] = [2.3, 0.6, 0.55]
    cabinet = _piece("e:3", "cabinet", (2.3, 0.9), 0.8, on=None)
    cabinet["role"] = "cabinet"
    cabinet["position"] = [2.3, 0.9, 0.7]
    cabinet["existence"] = 1.0
    board = _piece("e:20", "sideboard", (-2.5, 0.9), 1.0, on=None)
    board["role"] = "sideboard"
    board["position"] = [-2.5, 0.9, 0.7]
    board["existence"] = 1.0
    monkeypatch.setattr(chores_mod, "_summary",
                        lambda api: {"objects": [drawer, cabinet, board],
                                     "self": {"holding": None}})
    host = chores_mod._drawer_host_furniture("api", drawer)
    assert host is not None and host["id"] == "e:3"


def test_transfer_skips_when_move_loses_source_drawer_item(monkeypatch):
    """Pre-pick recover can succeed, then ``_move``'s second recover loses the track (seed2d)."""
    import steward.chores as chores_mod
    from steward.mission import MissionFailed

    plane = _piece("e:250", "toy_airplane", (2.312, 0.722), 0.22)
    plane["drawerId"] = "e:9"
    plane["stagedRole"] = "toy_airplane"
    dst = _piece("e:19", "drawer", (-2.5, 0.6), 0.4, on=None)
    dst["role"] = "drawer"

    monkeypatch.setattr(chores_mod, "_recover_drawer_carry", lambda *a, **k: plane)
    monkeypatch.setattr(chores_mod, "_refresh_or_recover", lambda api, obj, **k: dst)
    monkeypatch.setattr(
        chores_mod, "_move",
        lambda *a, **k: (_ for _ in ()).throw(
            MissionFailed("lost staged drawer item e:250 (expected toy_airplane near [2.312, 0.722]): "
                          "not in ('teddy_bear', 'pistol', 'toy_airplane', 'pill_bottle', 'perfume_bottle')")))

    runner = chores_mod.Chores("api")
    monkeypatch.setattr(runner, "_doing", lambda *a, **k: None)
    monkeypatch.setattr(runner, "publish", lambda **k: None)
    assert runner._transfer(plane, dst, "从柜子的抽屉移到餐边柜的抽屉 e:19") is None
    assert any(e.get("step") == "skip" and e.get("reason") == "lost-drawer-item"
               and e.get("where") == "move" for e in runner.log)


def test_staged_pick_retries_after_base_blocked(monkeypatch):
    """A dock stall on a staged pick retreats, then freshen/predict from a new heading."""
    import steward.chores as chores_mod
    from steward.mission import MissionFailed

    offsets: list[float | None] = []
    calls = {"picks": 0, "retreat": 0}
    staged = _piece("e:1774", "teddy_bear", (-2.82, 0.332), 0.15)
    staged["stageXy"] = [-2.82, 0.332]
    staged["stagedRole"] = "teddy_bear"
    drawer = _piece("e:8", "drawer", (2.3, 0.6), 0.4, on=None)

    def fake_freshen(api, obj, log, *, prefer_offset=None, distance=None):
        offsets.append(prefer_offset)
        return {**obj, "id": "e:1774"}

    def fake_safe(api, skill, params, observe, log):
        return {"feasible": True, "withinRiskLimit": True, "risk": 0.15, "blockers": [],
                "skill": skill, "plan": {"spot": [2.32, 0.61]}, "issues": []}

    def fake_run(api, receipt, timeout_s):
        if receipt["skill"] == "pick":
            calls["picks"] += 1
            if calls["picks"] == 1:
                raise MissionFailed(
                    "pick failed: base is blocked: no progress for 2 s (0.17 m from the waypoint)")
            return {"state": "succeeded", "message": "holding e:1774"}
        return {"state": "succeeded", "message": "placed"}

    monkeypatch.setattr(chores_mod, "_freshen_drawer_pick", fake_freshen)
    monkeypatch.setattr(chores_mod, "_align_drawer_place", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "_confirm_drawer_landing",
                        lambda api, obj, drawer, spot, log: {**obj, "position": [spot[0], spot[1], 0.8]})
    monkeypatch.setattr(chores_mod, "_safe_receipt", fake_safe)
    monkeypatch.setattr(chores_mod, "_run", fake_run)
    monkeypatch.setattr(chores_mod, "_refresh_or_recover", lambda api, obj, **k: obj)
    monkeypatch.setattr(chores_mod, "_held_drawer_match", lambda api, obj: {**staged, "id": "e:1774"})
    monkeypatch.setattr(chores_mod, "_retreat_while_holding",
                        lambda *a, **k: calls.__setitem__("retreat", calls["retreat"] + 1) or True)
    monkeypatch.setattr(chores_mod, "_landed",
                        lambda api, obj, spot: {**obj, "position": [spot[0], spot[1], 0.8]})
    monkeypatch.setattr(chores_mod, "simtime_sleep", lambda *a, **k: None)

    log: list = []
    out = chores_mod._move("api", staged, {"on": "e:8"}, drawer, log, 30.0)
    assert out["id"] == "e:1774"
    assert calls["picks"] == 2 and calls["retreat"] == 1
    assert offsets == [0.0, 1.0]
    assert any(e.get("reason") == "base-blocked" for e in log if e.get("step") == "pick-retry")


def test_transfer_staged_item_does_not_soft_skip_tableware_rematch(monkeypatch):
    """A staged teddy that briefly refreshes as wine glass is rematched, not skipped as done."""
    import steward.chores as chores_mod

    staged = _piece("e:78", "teddy_bear", (-2.82, 0.33), 0.15)
    staged["stageXy"] = [-2.82, 0.33]
    staged["stagedRole"] = "teddy_bear"
    wine = {**staged, "id": "e:78", "role": "wine_glass", "category": "wine_glass"}
    teddy = _piece("e:902", "teddy_bear", (-2.81, 0.34), 0.14, on="e:20")
    drawer = _piece("e:3", "drawer", (0.0, 0.0), 0.4, on=None)
    drawer["role"] = "drawer"

    def fake_recover(api, item, log):
        assert item.get("stagedRole") == "teddy_bear"
        return teddy

    monkeypatch.setattr(chores_mod, "_recover_drawer_carry", fake_recover)
    monkeypatch.setattr(chores_mod, "_refresh_or_recover", lambda api, obj, **k: obj)
    monkeypatch.setattr(chores_mod, "_move",
                        lambda api, obj, place, observe, log, timeout_s, **kw: {
                            **obj, "on": observe["id"]})
    runner = Chores("api")
    monkeypatch.setattr(runner, "_doing", lambda *a, **k: None)
    monkeypatch.setattr(runner, "publish", lambda **k: None)
    monkeypatch.setattr(runner, "_clearing", lambda phase, action: action())
    out = runner._transfer(staged, drawer, "从台面移到柜子的抽屉")
    assert out is not None and out["id"] == "e:902"
    assert not any(e.get("reason") == "non-drawer-content" for e in runner.log)


def test_staged_drawer_items_sit_toward_the_board_ends_away_from_the_drawer():
    """Temporary staging leaves the drawer face clear: ends of the long axis, opposite the drawer."""
    from steward.chores import STAGE_BACK_M, _local, _stage_top_xy

    board = {"id": "e:22", "position": [0.0, 0.0, 0.8], "sizeM": [1.4, 0.5, 0.85],
             "extentM": [1.4, 0.5, 0.85], "yawDeg": 0.0}
    # Drawer protrudes in +y (short axis) from the board centre.
    drawer = {"id": "e:21", "position": [0.0, 0.35, 0.55], "sizeM": [0.5, 0.4, 0.22]}
    a = _stage_top_xy(board, drawer, 0, 2)
    b = _stage_top_xy(board, drawer, 1, 2)
    assert a[0] < -0.15 and b[0] > 0.15
    assert _local(board, a)[1] < 0 and _local(board, b)[1] < 0
    assert abs(_local(board, a)[1]) <= STAGE_BACK_M + 1e-6
    # Stay inward of the long rim (half-length 0.7 m, edge clear 0.12 m).
    assert abs(a[0]) < 0.7 - 0.11 and abs(b[0]) < 0.7 - 0.11


def test_base_blocked_place_is_retried_after_a_short_retreat(monkeypatch):
    """A dock stall during place retreats while holding, then place is predicted again once."""
    import steward.chores as chores_mod
    from steward.mission import MissionFailed

    calls = {"place_run": 0, "retreat": 0}

    def fake_safe_receipt(api, skill, params, observe, log):
        return {"feasible": True, "withinRiskLimit": True, "risk": 0.0, "blockers": [],
                "skill": skill, "plan": {"spot": [0.0, 0.0]}, "issues": []}

    def fake_run(api, receipt, timeout_s):
        if receipt["skill"] == "pick":
            return {"state": "succeeded", "message": "holding e:1"}
        calls["place_run"] += 1
        if calls["place_run"] == 1:
            raise MissionFailed("place failed: base is blocked: no progress for 2 s (0.61 m from the waypoint)")
        return {"state": "succeeded", "message": "placed"}

    def fake_retreat(api, log, timeout_s):
        calls["retreat"] += 1
        return True

    monkeypatch.setattr(chores_mod, "_safe_receipt", fake_safe_receipt)
    monkeypatch.setattr(chores_mod, "_run", fake_run)
    monkeypatch.setattr(chores_mod, "_retreat_while_holding", fake_retreat)
    monkeypatch.setattr(chores_mod, "_approach_to_observe", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "_refresh", lambda api, obj: obj)
    monkeypatch.setattr(chores_mod, "_refresh_or_recover", lambda api, obj, **k: obj)
    monkeypatch.setattr(chores_mod, "_recover_drawer_carry", lambda api, obj, log: obj)
    monkeypatch.setattr(chores_mod, "_held_drawer_match", lambda api, obj: obj)
    monkeypatch.setattr(chores_mod, "_live_top_support", lambda api, support, log=None: support)
    monkeypatch.setattr(chores_mod, "_landed", lambda api, obj, spot: {**obj, "position": [spot[0], spot[1], 0.5]})
    monkeypatch.setattr(chores_mod, "simtime_sleep", lambda *a, **k: None)

    obj = _piece("e:1", "toy_airplane", (0.0, 0.0), 0.2)
    board = _piece("e:2", "sideboard", (0.0, 0.0), 1.0, on=None)
    out = chores_mod._move("api", obj, {"on": "e:2"}, board, [], 30.0)
    assert out["id"] == "e:1"
    assert calls["place_run"] == 2 and calls["retreat"] == 1


def test_drawer_place_aligns_then_retries_after_dock_infeasible(monkeypatch):
    """Place into an open drawer stands at the mouth first; dock-infeasible predicts retry once."""
    import steward.chores as chores_mod
    from steward.mission import MissionFailed

    calls = {"align": [], "safe": 0, "retreat": 0}

    def fake_align(api, drawer, log, *, prefer_offset=None):
        calls["align"].append(prefer_offset)

    def fake_safe_receipt(api, skill, params, observe, log):
        if skill == "pick":
            return {"feasible": True, "withinRiskLimit": True, "risk": 0.0, "blockers": [],
                    "skill": skill, "plan": {}, "issues": []}
        calls["safe"] += 1
        if calls["safe"] == 1:
            raise MissionFailed(
                "place infeasible: ['no feasible approach: side (1.0, 0.0): dock pose collides with e:20']")
        return {"feasible": True, "withinRiskLimit": True, "risk": 0.0, "blockers": [],
                "skill": skill, "plan": {"spot": [-2.5, 0.6]}, "issues": []}

    def fake_run(api, receipt, timeout_s):
        if receipt["skill"] == "pick":
            return {"state": "succeeded", "message": "holding e:1"}
        return {"state": "succeeded", "message": "placed"}

    monkeypatch.setattr(chores_mod, "_align_drawer_place", fake_align)
    monkeypatch.setattr(chores_mod, "_confirm_drawer_landing",
                        lambda api, obj, drawer, spot, log: {**obj, "position": [spot[0], spot[1], 0.5]})
    monkeypatch.setattr(chores_mod, "_safe_receipt", fake_safe_receipt)
    monkeypatch.setattr(chores_mod, "_run", fake_run)
    monkeypatch.setattr(chores_mod, "_retreat_while_holding",
                        lambda *a, **k: calls.__setitem__("retreat", calls["retreat"] + 1) or True)
    monkeypatch.setattr(chores_mod, "_approach_to_observe", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "_refresh", lambda api, obj: obj)
    monkeypatch.setattr(chores_mod, "_refresh_or_recover", lambda api, obj, **k: obj)
    monkeypatch.setattr(chores_mod, "_recover_drawer_carry", lambda api, obj, log: obj)
    monkeypatch.setattr(chores_mod, "_held_drawer_match", lambda api, obj: obj)
    monkeypatch.setattr(chores_mod, "_landed",
                        lambda api, obj, spot: {**obj, "position": [spot[0], spot[1], 0.5]})
    monkeypatch.setattr(chores_mod, "simtime_sleep", lambda *a, **k: None)

    obj = _piece("e:1", "toy_airplane", (2.5, 0.6), 0.22)
    drawer = _piece("e:19", "drawer", (-2.5, 0.6), 0.4, on=None)
    log: list = []
    out = chores_mod._move("api", obj, {"on": "e:19"}, drawer, log, 30.0)
    assert out["id"] == "e:1"
    assert calls["align"][:2] == [0.0, 1.0]
    assert calls["safe"] == 2 and calls["retreat"] == 1
    assert any(e.get("reason") == "dock-infeasible" for e in log if e.get("step") == "place-retry")


def test_drawer_place_retries_after_false_zero_opening(monkeypatch):
    """Place into a drawer retries when ESim reports open 0 cm / too little; recovers opening once."""
    import steward.chores as chores_mod
    from steward.mission import MissionFailed

    calls = {"safe": 0, "recover": 0, "retreat": 0}

    def fake_safe_receipt(api, skill, params, observe, log):
        if skill == "pick":
            return {"feasible": True, "withinRiskLimit": True, "risk": 0.0, "blockers": [],
                    "skill": skill, "plan": {}, "issues": []}
        calls["safe"] += 1
        if calls["safe"] == 1:
            raise MissionFailed(
                "place infeasible: ['e:19 is open 0 cm, too little for e:1; open it first']")
        return {"feasible": True, "withinRiskLimit": True, "risk": 0.0, "blockers": [],
                "skill": skill, "plan": {"spot": [-2.5, 0.6]}, "issues": []}

    def fake_run(api, receipt, timeout_s):
        if receipt["skill"] == "pick":
            return {"state": "succeeded", "message": "holding e:1"}
        return {"state": "succeeded", "message": "placed"}

    def fake_recover(api, drawer, log, timeout_s):
        calls["recover"] += 1
        return {**drawer, "openingM": 0.30}

    monkeypatch.setattr(chores_mod, "_align_drawer_place", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "_confirm_drawer_landing",
                        lambda api, obj, drawer, spot, log: {**obj, "position": [spot[0], spot[1], 0.5]})
    monkeypatch.setattr(chores_mod, "_safe_receipt", fake_safe_receipt)
    monkeypatch.setattr(chores_mod, "_run", fake_run)
    monkeypatch.setattr(chores_mod, "_recover_drawer_opening", fake_recover)
    monkeypatch.setattr(chores_mod, "_retreat_while_holding",
                        lambda *a, **k: calls.__setitem__("retreat", calls["retreat"] + 1) or True)
    monkeypatch.setattr(chores_mod, "_approach_to_observe", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "_refresh", lambda api, obj: obj)
    monkeypatch.setattr(chores_mod, "_refresh_or_recover", lambda api, obj, **k: obj)
    monkeypatch.setattr(chores_mod, "_recover_drawer_carry", lambda api, obj, log: obj)
    monkeypatch.setattr(chores_mod, "_held_drawer_match", lambda api, obj: obj)
    monkeypatch.setattr(chores_mod, "_landed",
                        lambda api, obj, spot: {**obj, "position": [spot[0], spot[1], 0.5]})
    monkeypatch.setattr(chores_mod, "simtime_sleep", lambda *a, **k: None)

    obj = _piece("e:1", "toy_airplane", (2.5, 0.6), 0.22)
    drawer = _piece("e:19", "drawer", (-2.5, 0.6), 0.4, on=None)
    drawer["openingM"] = 0.0
    log: list = []
    out = chores_mod._move("api", obj, {"on": "e:19"}, drawer, log, 30.0)
    assert out["id"] == "e:1"
    assert calls["safe"] == 2 and calls["recover"] == 1 and calls["retreat"] == 1
    assert any(e.get("reason") == "drawer-too-shut" for e in log if e.get("step") == "place-retry")


def test_drawer_contents_drop_tracks_outside_the_open_box():
    """A whitelist item whose centre lies outside the measured drawer box is not carried."""
    from steward.chores import _contents

    drawer = {"id": "e:1", "position": [0.0, 0.0, 0.55], "sizeM": [0.4, 0.35, 0.22], "extentM": [0.4, 0.35, 0.22]}
    inside = _piece("e:2", "teddy_bear", (0.05, 0.0), 0.11, on="e:1")
    inside["position"] = [0.05, 0.0, 0.50]
    outside = _piece("e:3", "pistol", (1.5, 0.0), 0.15, on="e:1")
    outside["position"] = [1.5, 0.0, 0.50]
    summary = {"objects": [inside, outside]}
    assert [o["id"] for o in _contents(summary, drawer)] == ["e:2"]


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
            # Upstream stack path assigns the return value as the moved object (``_landed``).
            landed = dict(obj)
            if place.get("on") is not None:
                landed["on"] = place["on"]
            return landed
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

    def fake_move(api, obj, place, observe, log, timeout_s, attempts=2):
        landed = dict(obj)
        if place.get("on") is not None:
            landed["on"] = place["on"]
            for o in objects_now():
                if o["id"] == obj["id"]:
                    o["on"] = place["on"]
        return landed

    monkeypatch.setattr(chores_mod, "_summary", fake_summary)
    monkeypatch.setattr(chores_mod, "_settle", fake_settle)
    monkeypatch.setattr(chores_mod, "_refresh_or_recover",
                        lambda api, obj: board if obj.get("role") == "sideboard"
                        else next(o for o in objects_now() if o["id"] == obj["id"]))
    monkeypatch.setattr(chores_mod, "_approach_to_observe", fake_approach)
    monkeypatch.setattr(chores_mod, "simtime_sleep", lambda *a, **k: None)
    monkeypatch.setattr(chores_mod, "_move", fake_move)
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
    monkeypatch.setattr(chores_mod, "_landed", lambda api, o, spot: {**o, "on": "e:484"})
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

    def move(api, obj, place, observe, log, timeout_s, attempts=None):
        for o in state["objects"]:
            if o["id"] == obj["id"] and place.get("on"):
                o["on"] = place["on"]
        landed = dict(obj)
        if place.get("on") is not None:
            landed["on"] = place["on"]
        return landed

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
                return o
        return {**obj, "on": place.get("on")}

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
