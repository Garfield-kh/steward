"""The Cobot Magic driven by the shared Ego Runtime: the kitchen robot's ESim receipts and skills,
with one Piper arm and the torso as its planar arm."""

import dataclasses
import math
import time

import numpy as np
import pytest

from steward.authoring.service import AuthoringService, build_cobot_workstation
from steward.ego.bodies import body_arm, robot_model, runtime_config
from steward.ego.cobot_arm import PiperArm
from steward.ego.runtime import EgoRuntime, RobotLink
from steward.ego.server import ego_api
from steward.net import call
from steward.world.server import StewardHost


def test_the_planar_arm_reports_what_it_was_commanded_once_the_arm_is_there():
    arm = PiperArm(robot_model("cobot-magic"))
    from steward.assets.cobot import HOME, LIFT_HOME

    home = {"lift": LIFT_HOME, **dict(zip(arm.names, HOME, strict=True))}
    arm.virtualize({"joints": home, "odometry": {"x": 0.0, "y": 0.0, "yaw": 0.0}})
    cmd = arm.command({"lift": 0.85, "reach": 0.30, "wrist_yaw": 0.4, "wrist_roll": 0.0, "gripper": "close"})
    assert cmd["gripper"] == {"L_link6": "close"}
    solved = cmd["joints"]
    there = arm.virtualize({"joints": solved, "odometry": {"x": 1.0, "y": 2.0, "yaw": 0.5}})["joints"]
    assert [there[k] for k in ("lift", "reach", "wrist_yaw", "wrist_roll")] == [0.85, 0.30, 0.4, 0.0]
    measured = arm._fk(solved["lift"], [solved[n] for n in arm.names])
    assert measured["lift"] == pytest.approx(0.85, abs=0.006) and measured["reach"] == pytest.approx(0.30, abs=0.006)
    assert measured["wrist_yaw"] == pytest.approx(0.4, abs=0.03)
    assert arm.geometry.lateral == pytest.approx(0.31), "the left arm works on its own line"


@pytest.fixture(scope="module")
def cobot(tmp_path_factory):
    root = tmp_path_factory.mktemp("cobot-ego")
    build_cobot_workstation(AuthoringService(root))
    host = StewardHost(str(root), scenario="cobot-table", speed=math.inf, seed=3,
                       run_dir=tmp_path_factory.mktemp("run")).start()
    model = robot_model("cobot-magic")
    runtime = EgoRuntime(RobotLink(host.robot_servers["w:cobot-1"].url), model, runtime_config("cobot-magic"),
                         arm=body_arm("cobot-magic", model))
    runtime.start()
    api = ego_api(runtime).start()
    try:
        yield host, api.url
    finally:
        api.stop()
        runtime.stop()
        host.stop()


def _run(api: str, skill: str, params: dict) -> dict:
    receipt = call("POST", f"{api}/ego/predict", {"skill": skill, "params": params})
    assert receipt["feasible"] and receipt["withinRiskLimit"], receipt["issues"]
    job = call("POST", f"{api}/skill/run", {"receipt": receipt["id"]})
    return call("POST", f"{api}/skill/wait", {"job": job["id"], "timeoutS": 150}, timeout=160)


def test_the_twin_picks_a_bowl_and_puts_it_down_elsewhere_on_the_table_through_receipts(cobot):
    host, api = cobot
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        objects = call("GET", f"{api}/ego/summary")["objects"]
        bowls = [o for o in objects if o["category"] == "bowl"]
        tables = [o for o in objects if o["category"] == "table"]
        if bowls and tables:
            break
        time.sleep(0.5)
    assert bowls and tables, [o["category"] for o in objects]
    summary = call("GET", f"{api}/ego/summary")
    assert set(summary["self"]["joints"]) == {"lift", "reach", "wrist_yaw", "wrist_roll", "wrist_pitch"}, \
        "the kitchen robot's planar arm, and the tool tilt its gripper lacks"
    picked = _run(api, "pick", {"target": bowls[0]["id"]})
    assert picked["state"] == "succeeded", (picked["message"], picked["log"])
    table = tables[0]
    placed = _run(api, "place", {"on": table["id"], "offset": [-0.30, -0.15]})
    assert placed["state"] == "succeeded", (placed["message"], placed["log"])
    bowl = host.sim._entity_pose("w:bowl-1")["p"]
    real = host.sim._entity_pose("w:table-1")
    w, _, _, z = real["q"]
    yaw = 2 * math.atan2(z, w)
    want = np.asarray(real["p"][:2]) + np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]) @ [-0.30, -0.15]
    assert np.linalg.norm(np.asarray(bowl[:2]) - want) < 0.06, (bowl, want)
    assert bowl[2] > 0.74, "on the table, not on the floor"


def test_the_twins_cameras_are_captured_with_the_pose_of_the_gripper_its_arm_works_with(cobot):
    _, api = cobot
    packet = call("POST", f"{api}/ego/cameras/capture", {})
    assert set(packet["views"]) == {"head_camera", "wide_camera"}
    pose = call("GET", f"{api}/ego/gripper/pose")
    for reported in (packet["gripper_pose"], pose):
        assert (reported["gripper_frame"], reported["gripper_link"]) == ("gripper", "L_link6")
    assert {"lift", "reach"} <= set(pose["joint_positions"]), "the arm's joints as the Ego Harness reads them"


def _home_arm() -> PiperArm:
    from steward.assets.cobot import HOME, LIFT_HOME

    arm = PiperArm(robot_model("cobot-magic"))
    arm.virtualize({"joints": {"lift": LIFT_HOME, **dict(zip(arm.names, HOME, strict=True))},
                    "odometry": {"x": 0.0, "y": 0.0, "yaw": 0.0}})
    return arm


def test_the_piper_tilts_its_tool_forward_to_reach_past_where_it_reaches_pointing_down():
    arm = _home_arm()
    assert not arm.reachable(0.6, 0.62), "the tool straight down reaches 0.50 m ahead of the mount"
    assert arm.reachable(0.6, 0.62, wrist_pitch=1.05)
    solved = arm.command({"lift": 0.9, "reach": 0.62, "wrist_pitch": 1.05})["joints"]
    measured = arm._fk(solved["lift"], [solved[n] for n in arm.names])
    assert measured["reach"] == pytest.approx(0.62, abs=0.006) and measured["lift"] == pytest.approx(0.9, abs=0.006)
    assert measured["wrist_pitch"] == pytest.approx(1.05, abs=0.03) and measured["wrist_roll"] == pytest.approx(0.0, abs=0.03)


def test_what_the_piper_plans_to_reach_it_reaches_from_wherever_its_arm_stands():
    """A local solver seeded only from where the arm stands missed a third of the poses it reaches,
    and which third depended on the posture; planned from the pose alone, the tilt a reach needs
    grows with the reach, and the command it plans is solved from a posture far from the plan's."""
    arm = _home_arm()
    reaches = (0.50, 0.55, 0.60, 0.65)
    pitches = [arm.pitch_for(0.81, r) for r in reaches]
    assert pitches == sorted(pitches) and pitches[0] == 0.0 and pitches[-1] > 0.5
    stretched = PiperArm(robot_model("cobot-magic"))
    stretched.virtualize({"joints": {"lift": 0.4, **dict(zip(stretched.names, [0.0, 2.7, -2.4, 0.0, 0.9, 1.57], strict=True))},
                          "odometry": {"x": 0.0, "y": 0.0, "yaw": 0.0}})
    for reach, pitch in zip(reaches, pitches, strict=True):
        stretched._target = stretched._solved = None
        assert stretched.command({"lift": 0.81, "reach": reach, "wrist_pitch": pitch})["joints"]


def test_the_piper_draws_a_tilted_tool_back_while_its_torso_sinks():
    """Drawing back with the tool tilted, the redundant torso sinks 0.12 m in one 5 cm step while the
    arm joints keep the pinch point on the way: the arm swings nothing it holds, so it is no jump."""
    arm = _home_arm()
    assert arm.reachable_path([{"lift": 0.8, "reach": 0.63 - 0.05 * k, "wrist_pitch": 0.58} for k in range(8)])


def test_the_torso_and_all_six_joints_tip_a_held_box_past_upright_in_one_motion():
    """Solved for the arm alone at a few torso heights, the tip turned the wrist over at 1.4 rad
    (joint4 at its limit); solved with the torso, step after step from the last posture, the arm
    tips the box the 1.9 rad its walls need without a jump."""
    arm = _home_arm()

    def tip(roll: float) -> list[dict]:
        return [{"lift": 0.85 + 0.27 * abs(math.sin(roll * k / 20)), "reach": 0.5, "wrist_roll": roll * k / 20,
                 "wrist_pitch": 1.05} for k in range(21)]

    assert arm.reachable_path(tip(1.9)) and arm.reachable_path(tip(-1.9))
    assert not arm.reachable_path(tip(1.9)[:1] + tip(-1.9)[-1:]), "a half-turn of the wrist in one step is a jump"


def test_the_piper_keeps_its_upper_arm_off_its_own_frame_as_it_tips_a_box_close_in():
    """Tipped past upright close to the body, the sinking torso swung the upper arm into the robot's
    frame: the twin's shoulder stalled against it, the box came down on the table and the arm
    drawing back turned the table over. Every posture solved along such a tip keeps the arm's links
    clear of the frame and chassis."""
    import mujoco

    from steward.ego.ik import _ENT

    arm = _home_arm()
    ik, solved, prior = arm.ik, None, None
    body = ik._collidable({f"{_ENT}/frame", f"{_ENT}/base"})
    links = ik._collidable({f"{_ENT}/L_link{k}" for k in range(2, 6)})
    with arm._lock:
        for k in range(21):
            roll = -1.9 * k / 20
            target = {"lift": 0.81 + 0.14 * abs(math.sin(roll)), "reach": 0.333, "wrist_yaw": 0.0,
                      "wrist_roll": roll, "wrist_pitch": 1.05}
            solved, _ = arm._ik(target, {}, prior)
            if solved is None:
                break
            ik.set_base(0.0, 0.0, 0.0, solved[0])
            q = np.array(ik.data.q)
            q[ik._arms["L"]["qadr"]] = solved[1]
            ik.data.update(q)
            gap = min(mujoco.mj_geomDistance(ik.model, ik.data.data, a, b, 0.5, np.zeros(6)) for a in links for b in body)
            assert gap > 0.009, f"at roll {roll:+.2f} the arm is {gap * 1000:.0f} mm from its frame"
            prior = solved
    assert prior is not None and k > 10, "it still tips the box well over"


def test_the_piper_turns_what_it_holds_over_in_the_air_with_its_tool_tilted_and_lowers_it():
    """With the tool tilted most, a half turn of the wrist about the tool's own axis leaves the pinch
    point near its spot; the arm then lowers it, still turned over, until its forearm or gripper
    housing would come down on the table (beyond the table's edge only: the shoulder hangs below
    the top in front of it). With the tool tilted less the arm cannot hold it turned over there."""
    arm = _home_arm()
    top, edge = 0.753, 0.41
    assert arm.turn_over(0.97, 0.36, 1.05, 0.77, top, math.inf) == pytest.approx(0.89), "the arm alone lowers it so far"
    low = arm.turn_over(0.97, 0.36, 1.05, 0.77, top, edge)
    assert low == pytest.approx(0.93), "over the table the links stop it higher"
    assert arm.turn_over(0.97, 0.36, 0.6, 0.77, top, edge) is None
    arm._last["grippers"] = {arm.geometry.gripper: {"holding": "w:piece"}}
    arm.command({"lift": 0.97, "reach": 0.36, "wrist_pitch": 1.05})
    spot = arm._pinch(*arm._solved)
    arm.command({"wrist_roll": math.pi, "turnOver": True})
    assert float(np.linalg.norm(arm._pinch(*arm._solved) - spot)) < 0.01
    for k in range(1, round((0.97 - low) / 0.02) + 1):
        arm.command({"lift": 0.97 - 0.02 * k})
    measured = arm._fk(arm._solved[0], arm._solved[1])
    assert measured["lift"] == pytest.approx(low, abs=0.006)
    assert abs(measured["wrist_roll"]) == pytest.approx(math.pi, abs=0.05)
    assert arm._lowest(arm._solved, edge) >= top + arm.config.turn_link_clearance - 1e-6


def _pumped(host: StewardHost, work):
    """Run ``work`` on a thread while the main thread serves the host's renders."""
    import threading

    out: dict = {}

    def run() -> None:
        try:
            out["value"] = work()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread
            out["error"] = exc

    worker = threading.Thread(target=run)
    worker.start()
    while worker.is_alive():
        host.pump(0.02)
    if "error" in out:
        raise out["error"]
    return out["value"]


@pytest.fixture
def cobot_in(kitchen_root, tmp_path):
    """The twin in a scenario of the kitchen root (or of ``root``), driven by the shared Ego Runtime."""
    started = []

    def start(scenario: str, root: str | None = None):
        host = StewardHost(root or kitchen_root, scenario=scenario, body="cobot-magic", speed=math.inf, seed=3,
                           run_dir=tmp_path).start()
        rid = next(iter(host.robot_servers))
        model = robot_model("cobot-magic")
        runtime = EgoRuntime(RobotLink(host.robot_servers[rid].url), model, runtime_config("cobot-magic"),
                             arm=body_arm("cobot-magic", model))
        runtime.start()
        api = ego_api(runtime).start()
        started.append((host, runtime, api))
        _pumped(host, lambda: (time.sleep(2), _explore(api.url)))
        return host, api.url

    yield start
    for host, runtime, api in started:
        api.stop()
        runtime.stop()
        host.stop()


def _explore(api: str) -> None:
    job = call("POST", f"{api}/ego/explore", {})
    call("POST", f"{api}/skill/wait", {"job": job["id"], "timeoutS": 120}, timeout=130)


def _receipt(api: str, skill: str, params: dict) -> tuple[dict, dict]:
    # A pour is checked along the whole tip on every side tried: many whole-body solves.
    receipt = call("POST", f"{api}/ego/predict", {"skill": skill, "params": params}, timeout=120)
    assert receipt["feasible"] and receipt["withinRiskLimit"], (skill, receipt["issues"])
    job = call("POST", f"{api}/skill/run", {"receipt": receipt["id"]})
    done = call("POST", f"{api}/skill/wait", {"job": job["id"], "timeoutS": 150}, timeout=160)
    assert done["state"] == "succeeded", (skill, done["message"], done["log"])
    return receipt, done


def test_the_twin_takes_things_out_of_both_drawers_deep_ones_with_its_tool_tilted(cobot_in):
    """The cabinet drawer's bottle lies beyond where the tool reaches pointing down, so the pick tilts
    it; the sideboard drawer's pistol lies below the drawer front, which the wrist would sweep shut
    reaching in at that height, so the arm comes in over the front and goes straight down."""
    host, api = cobot_in("phase0-chores")

    def swap() -> dict:
        objects = call("GET", f"{api}/ego/summary")["objects"]
        drawers = {round(o["position"][0]) > 0: o for o in objects if o["category"] == "drawer"}
        plans = {}
        for cabinet_side, wanted in ((True, "pill_bottle"), (False, "pistol")):
            drawer = drawers[cabinet_side]
            _receipt(api, "open", {"target": drawer["id"]})
            inside = [o for o in call("GET", f"{api}/ego/summary")["objects"]
                      if o.get("on") == drawer["id"] and o["category"] == wanted]
            assert inside, wanted
            picked, _ = _receipt(api, "pick", {"target": inside[0]["id"]})
            plans[wanted] = (picked["plan"], drawer["position"][2] + drawer["sizeM"][2] / 2)
            _receipt(api, "place", {"on": drawer["id"]})
            _receipt(api, "close", {"target": drawer["id"]})
        return plans

    plans = _pumped(host, swap)
    bottle, _ = plans["pill_bottle"]
    assert bottle.get("wristPitch", 0.0) > 0.0, "beyond 0.50 m the tool tilts"
    pistol, front = plans["pistol"]
    assert pistol["approachLift"] > front > pistol["lift"], "in over the drawer front, then down"
    relations = host.sim.world_state()["relations"]
    rest = {r["a"]["entity"]: r["b"] for r in relations if r["type"] == "supported-by"}
    assert rest["w:pill-bottle"]["entity"] == "w:cabinet-1" and rest["w:pistol"]["entity"] == "w:sideboard-1"


def _puzzle_ids(api: str) -> tuple[dict, dict, dict]:
    objects = call("GET", f"{api}/ego/summary")["objects"]
    lid = next(o for o in objects if o["category"] == "lid")
    box = next(o for o in objects if o["category"] in ("box", "picture"))
    table = next(o for o in objects if "table" in o["category"])
    call("POST", f"{api}/ego/annotate", {"entity": lid["id"], "affordances": ["graspable", "flat"]})
    call("POST", f"{api}/ego/annotate", {"entity": box["id"], "label": "box",
                                        "affordances": ["graspable", "container", "receptacle", "pourable"]})
    call("POST", f"{api}/ego/annotate", {"entity": table["id"], "label": "table", "affordances": ["support-surface"]})
    return lid, box, table


@pytest.fixture
def deep_table_root(tmp_path_factory, monkeypatch):
    """A root with the 4 × 4 puzzle on a table deep enough for the twin to pour on (2.4 × 1.15 m): on
    the half-size table the pour spot lies near the middle, 0.32 m out, where the Piper's wrist mostly
    tips the box short of the 1.7 rad its walls need."""
    from steward.authoring import service

    monkeypatch.setitem(service.PUZZLES, "dog", dataclasses.replace(service.PUZZLES["dog"], table_m=(2.4, 1.15), box_from_end_m=0.15))
    root = tmp_path_factory.mktemp("deep-table")
    service.build_default_kitchen(AuthoringService(root))
    return str(root)


def test_the_twin_takes_the_puzzle_box_by_its_rim_tilted_and_pours_it_out_onto_the_table(cobot_in, deep_table_root):
    """The box's walls hold the pieces until it is past upright. The Piper's wrist alone turns over
    at 1.4 rad; the torso and all six joints solved together tip it 1.9 rad in one motion, the
    torso rising as its lower edge swings down."""
    host, api = cobot_in("puzzle-dog", deep_table_root)

    def pour() -> tuple[dict, dict]:
        lid, box, table = _puzzle_ids(api)
        _receipt(api, "pick", {"target": lid["id"]})
        _receipt(api, "place", {"on": table["id"], "offset": [0.75, 0.0]})
        taken, _ = _receipt(api, "pick", {"target": box["id"]})
        poured, _ = _receipt(api, "pour", {"on": table["id"], "offset": [-0.75, -0.14]})
        return taken["plan"], poured["plan"]

    taken, poured = _pumped(host, pour)
    assert taken["wristPitch"] == pytest.approx(1.05), "a container to be poured is taken with the wrist tilted most"
    assert taken["graspDz"] > 0, "on its rim, above its centre"
    assert poured["roll"] >= 1.7
    world = host.sim.world_state()
    rest = {r["a"]["entity"]: r["b"]["entity"] for r in world["relations"] if r["type"] == "supported-by"}
    pieces = {e["id"]: e["pose"]["value"]["p"] for e in world["entities"] if e["id"].startswith("w:piece")}
    still_in = [p for p in pieces if rest.get(p) == "w:box"]
    assert not still_in, f"{still_in} stayed in the box"
    assert sum(p[2] > 0.7 for p in pieces.values()) >= len(pieces) - 2, "the pieces land on the table"


def test_the_twin_turns_a_puzzle_piece_over_in_the_air_and_sets_it_down_on_the_table(cobot_in):
    """Its gripper cannot point up at a surface: picked out of the box with flip the piece is pinched
    with the tool tilted most, turned over by half a turn of the wrist above the table, lowered and
    let go."""
    host, api = cobot_in("puzzle-dog")

    def up(world_id: str) -> float:
        _, x, y, _ = host.sim._entity_pose(world_id)["q"]
        return 1 - 2 * (x * x + y * y)

    def flip() -> tuple[dict, str, float]:
        lid, box, table = _puzzle_ids(api)
        _receipt(api, "pick", {"target": lid["id"]})
        _receipt(api, "place", {"on": table["id"], "offset": [0.35, 0.0]})
        time.sleep(1)
        pieces = [o for o in call("GET", f"{api}/ego/summary")["objects"]
                  if o["category"] == "puzzle_piece" and o.get("on") == box["id"]]
        piece = next(o for o in pieces if call("POST", f"{api}/ego/predict",
                                               {"skill": "pick", "params": {"target": o["id"], "flip": True}})["feasible"])
        picked, _ = _receipt(api, "pick", {"target": piece["id"], "flip": True})
        held = next(r["a"]["entity"] for r in host.sim.world_state()["relations"] if r["type"] == "attached")
        before = up(held)
        placed, _ = _receipt(api, "place", {"on": table["id"], "offset": [-0.2, 0.1], "flip": True})
        time.sleep(1)
        assert picked["plan"]["wristPitch"] == pytest.approx(1.05)
        return placed["plan"], held, before

    placed, held, before = _pumped(host, flip)
    assert placed["turnLift"] > placed["lift"] >= 0.77, "turned over above the table, let go a little above it"
    assert before * up(held) < -0.5, "the piece lies the other side up"
    rest = {r["a"]["entity"]: r["b"]["entity"] for r in host.sim.world_state()["relations"] if r["type"] == "supported-by"}
    assert rest[held] == "w:table-1"


def test_the_twin_lifts_puzzle_pieces_out_of_the_box_over_its_wall(cobot_in):
    host, api = cobot_in("puzzle-cat")

    def lift_one() -> dict:
        lid, box, table = _puzzle_ids(api)
        _receipt(api, "pick", {"target": lid["id"]})
        _receipt(api, "place", {"on": table["id"]})
        time.sleep(1)
        pieces = [o for o in call("GET", f"{api}/ego/summary")["objects"]
                  if o["category"] == "puzzle_piece" and o.get("on") == box["id"]]
        piece = next(o for o in pieces if call("POST", f"{api}/ego/predict",
                                               {"skill": "pick", "params": {"target": o["id"]}})["feasible"])
        lifted, _ = _receipt(api, "pick", {"target": piece["id"]})
        _receipt(api, "place", {"on": table["id"], "offset": [0.15, 0.1]})
        return lifted["plan"]

    lifted = _pumped(host, lift_one)
    assert lifted["approachLift"] > lifted["lift"], "a piece comes out over the box wall"
    assert not [r for r in host.sim.world_state()["relations"] if r["type"] == "attached"]
