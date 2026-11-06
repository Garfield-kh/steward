"""The Cobot Magic driven by the shared Ego Runtime: the kitchen robot's ESim receipts and skills,
with one Piper arm and the torso as its planar arm."""

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
    assert set(summary["self"]["joints"]) == {"lift", "reach", "wrist_yaw", "wrist_roll"}, "the planar arm, as on the kitchen robot"
    picked = _run(api, "pick", {"target": bowls[0]["id"]})
    assert picked["state"] == "succeeded", (picked["message"], picked["log"])
    table = tables[0]
    placed = _run(api, "place", {"on": table["id"], "offset": [-0.30, -0.15]})
    assert placed["state"] == "succeeded", (placed["message"], placed["log"])
    bowl = host.sim._entity_pose("w:bowl-1")["p"]
    yaw = math.radians(table["yawDeg"])
    want = np.asarray(table["position"][:2]) + np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]) @ [-0.30, -0.15]
    assert np.linalg.norm(np.asarray(bowl[:2]) - want) < 0.06, (bowl, want)
    assert bowl[2] > 0.74, "on the table, not on the floor"
