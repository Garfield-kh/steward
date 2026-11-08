"""Embodiments: the registry, the scenario catalog, and `steward up` keeping one Ego Runtime of
the right kind per robot as the console switches scenarios."""

from __future__ import annotations

import math
import shutil
import threading
import types

from steward.authoring.service import AuthoringService, build_cobot_kitchen
from steward.ego.perception import decode_rgbd
from steward.embodiments import EMBODIMENTS, describe, embodiment_of, scenario_embodiments
from steward.net import call
from steward.world.server import StewardHost


def _pump_while(host: StewardHost, work) -> dict:
    result: dict = {}

    def run() -> None:
        try:
            result["value"] = work()
        except Exception as exc:  # noqa: BLE001 - captured for the assertion
            result["error"] = exc

    thread = threading.Thread(target=run)
    thread.start()
    while thread.is_alive():
        host.pump(0.05)
    if "error" in result:
        raise result["error"]
    return result["value"]


def test_the_registry_names_a_body_from_its_scene_entity(kitchen_root, tmp_path) -> None:
    kitchen = AuthoringService(kitchen_root).load_scenario("phase0-chores")
    assert scenario_embodiments(kitchen) == {"w:robot-1": describe("steward-bot")}
    assert describe("steward-bot")["registered"] and describe("steward-bot")["label"] == EMBODIMENTS["steward-bot"]["label"]

    root = tmp_path / "cobot"
    build_cobot_kitchen(AuthoringService(root))
    twin = scenario_embodiments(AuthoringService(root).load_scenario("cobot-kitchen"))["w:cobot-1"]
    assert twin["id"] == "cobot-magic" and twin["registered"]

    stranger = {"id": "w:x", "asset": {"id": "nobody"}, "category": {"value": "nobody"}, "role": "robot"}
    assert embodiment_of(stranger) is None
    shown = describe(embodiment_of(stranger), "nobody")
    assert not shown["registered"] and "nobody" in shown["label"]


def test_the_catalog_offers_builtin_scenarios_of_every_body_before_they_are_built(kitchen_root) -> None:
    catalog = {c["name"]: c for c in AuthoringService(kitchen_root).scenario_catalog()}
    chores = catalog["phase0-chores"]
    assert chores["built"] and chores["robots"]["w:robot-1"]["id"] == "steward-bot" and chores["mission"] is None
    twin = catalog["cobot-kitchen"]
    assert not twin["built"] and twin["mission"] == "nest-bowls"
    assert twin["robots"] == {"w:cobot-1": describe("cobot-magic")}
    assert twin["task"]


def test_loading_a_scenario_of_another_body_builds_it_and_switches_the_console(kitchen_root, tmp_path) -> None:
    root = tmp_path / "root"
    shutil.copytree(kitchen_root, root)
    host = StewardHost(root, scenario="phase0-chores", run_dir=tmp_path / "run").start()
    loaded: list[dict] = []
    host.on_episode_loaded = lambda: loaded.append(dict(host.embodiments))
    try:
        wca = host.wca.url
        listing = call("GET", f"{wca}/scenarios")
        assert "cobot-kitchen" not in listing["scenarios"]
        assert not next(c for c in listing["catalog"] if c["name"] == "cobot-kitchen")["built"]
        state = call("GET", f"{wca}/console/state")
        assert state["embodiments"]["w:robot-1"]["label"] == EMBODIMENTS["steward-bot"]["label"]
        assert state["robots"]["w:robot-1"]["embodiment"]["id"] == "steward-bot"

        result = _pump_while(host, lambda: call("POST", f"{wca}/wca/episode",
                                                {"action": "load", "scenario": "cobot-kitchen"}, timeout=180))
        assert set(result["robots"]) == {"w:cobot-1"}
        assert "cobot-kitchen" in AuthoringService(root).list_scenarios(), "built on first load"
        assert loaded == [{"w:cobot-1": describe("cobot-magic")}], "the runtime keeper hears about the new body"
        state = call("GET", f"{wca}/console/state")
        assert list(state["robots"]) == ["w:cobot-1"]
        assert state["embodiments"]["w:cobot-1"]["id"] == "cobot-magic"
        assert next(c for c in call("GET", f"{wca}/scenarios")["catalog"] if c["name"] == "cobot-kitchen")["built"]

        # The twin's port serves what its runtime needs: encoders without a perception sample,
        # and the overhead camera by name.
        robot = host.robot_servers["w:cobot-1"].url
        encoders = call("GET", f"{robot}/sensors/proprioception")
        assert {"stamp", "joints", "grippers"} <= set(encoders) and "detections" not in encoders
        frame = _pump_while(host, lambda: decode_rgbd(call("GET", f"{robot}/sensors/rgbd?camera=wide_camera")))
        assert frame["camera"] == "wide_camera"
    finally:
        host.stop()


def test_the_fixed_robot_port_goes_to_the_scenarios_first_robot_whatever_its_body(kitchen_root, tmp_path) -> None:
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    root = tmp_path / "root"
    shutil.copytree(kitchen_root, root)
    host = StewardHost(root, scenario="phase0-chores", run_dir=tmp_path / "run", robot_ports={"*": port}).start()
    try:
        assert host.robot_servers["w:robot-1"].url.endswith(f":{port}")
        host.load_episode("cobot-kitchen")
        assert "w:robot-1" not in host.robot_servers, "the robot that left takes its port with it"
        assert host.robot_servers["w:cobot-1"].url.endswith(f":{port}"), \
            "the operator asked for a fixed robot port, not for a robot called w:robot-1"
    finally:
        host.stop()


def test_the_runtime_keeper_runs_one_process_of_the_right_kind_per_robot(kitchen_root, tmp_path, monkeypatch) -> None:
    from steward import cli

    started: list = []
    stopped: list = []

    class FakeProcess:
        def __init__(self, args, robot_url, out_dir, *, embodiment, api_port, task=None,
                     mission_config=None):
            self.embodiment, self.api_port, self.task = embodiment, api_port, task
            self.mission_config = dict(mission_config or {})
            self.url = f"http://ego/{embodiment}/{api_port}"
            started.append(self)

        def stop(self) -> None:
            stopped.append(self)

        def check(self) -> None:
            pass

    monkeypatch.setattr(cli, "_EgoProcess", FakeProcess)
    root = tmp_path / "root"
    shutil.copytree(kitchen_root, root)
    host = StewardHost(root, scenario="phase0-chores", run_dir=tmp_path / "run").start()
    keeper = cli.EgoSupervisor(types.SimpleNamespace(root=str(root), api_port=7420, bind_wait=0.0), tmp_path / "run")
    order: list[str] = []

    def switch(name: str) -> None:
        _pump_while(host, lambda: call("POST", f"{host.wca.url}/wca/episode",
                                       {"action": "load", "scenario": name}, timeout=180))

    try:
        keeper.reconcile(host)
        assert [(p.embodiment, p.api_port, p.task) for p in started] == [("steward-bot", 7420, None)]
        assert host.ego_apis == {"w:robot-1": started[0].url}

        # Through the real load path: departing runtimes stop while the old world is still up,
        # then the world swaps, then the new bodies' runtimes start.
        host.before_episode_load = lambda scenario: (keeper.depart(host, scenario), order.append("depart"))
        host.on_episode_loaded = lambda: (keeper.reconcile(host), order.append("start"))

        switch("cobot-kitchen")
        assert order == ["depart", "start"]
        assert stopped == [started[0]], "the mobile robot's runtime goes with its scenario"
        assert (started[1].embodiment, started[1].api_port, started[1].task) == ("cobot-magic", 7420, None), \
            "the twin's runtime takes over the fixed port; its tasks come from the Ego Harness, as the kitchen robot's"
        assert host.ego_apis == {"w:cobot-1": started[1].url}

        switch("cobot-errand")
        assert len(started) == 2 and stopped == [started[0]], "same robot, same body: the runtime is kept"
        assert host.ego_apis == {"w:cobot-1": started[1].url}

        switch("phase0-shuttle")
        assert stopped[-1] is started[1] and started[-1].embodiment == "steward-bot"

        switch("phase0-kitchen")
        assert len(started) == 3 and len(stopped) == 2, "same robot, same body: the runtime is kept and reboots itself"

        # The console's 重置 replaces the runtime: stopped before the world starts over, then started anew.
        host.before_episode_reset = lambda: (keeper.retire(host), order.append("retire"))
        _pump_while(host, lambda: call("POST", f"{host.wca.url}/wca/episode", {"action": "reset"}, timeout=180))
        assert order[-2:] == ["retire", "start"]
        assert stopped[-1] is started[2] and len(started) == 4 and started[3].embodiment == "steward-bot"
        assert host.ego_apis == {"w:robot-1": started[3].url}
    finally:
        keeper.stop()
        host.stop()
    assert set(stopped) == set(started)


def test_any_body_takes_any_scenario_and_the_host_keeps_the_choice(kitchen_root, tmp_path) -> None:
    host = StewardHost(kitchen_root, scenario="phase0-kitchen", body="cobot-magic", run_dir=tmp_path / "run").start()
    try:
        assert host.embodiments == {"w:robot-1": describe("cobot-magic")}, "the kitchen's robot, the twin's body"
        start = next(e for e in AuthoringService(kitchen_root).load_scenario("phase0-kitchen")["sceneState"]["entities"]
                     if e["id"] == "w:robot-1")["pose"]["value"]["p"]
        robot = next(e for e in host.sim.world_state()["entities"] if e["id"] == "w:robot-1")
        assert robot["category"]["value"] == "cobot-magic"
        assert math.dist(robot["pose"]["value"]["p"][:2], start[:2]) < 1e-3, "at the scenario's start pose"
        frame = _pump_while(host, lambda: decode_rgbd(call("GET", f"{host.robot_servers['w:robot-1'].url}/sensors/rgbd?camera=wide_camera")))
        assert frame["camera"] == "wide_camera"
        catalog = call("GET", f"{host.wca.url}/scenarios")
        assert {b["id"] for b in catalog["bodies"]} == set(EMBODIMENTS) and catalog["body"] == "cobot-magic"
        _pump_while(host, lambda: call("POST", f"{host.wca.url}/wca/episode",
                                       {"action": "load", "scenario": "phase0-shuttle", "body": "steward-bot"}, timeout=180))
        assert host.embodiments == {"w:robot-1": describe("steward-bot")} and host.body == "steward-bot"
    finally:
        host.stop()


def test_a_body_that_does_not_fit_the_start_pose_is_refused_with_what_it_hits(kitchen_root, tmp_path) -> None:
    from steward.embodiments import with_body
    from steward.world.server import start_blocked

    scenario = AuthoringService(kitchen_root).load_scenario("phase0-kitchen")
    robot = next(e for e in scenario["sceneState"]["entities"] if e["id"] == "w:robot-1")
    wall = next(e for e in scenario["sceneState"]["entities"] if e["role"] == "environment")
    assert start_blocked(with_body(scenario, "cobot-magic")) == []
    robot["pose"]["value"]["p"][0] = -2.9  # backed into the kitchen's west wall
    blocked = start_blocked(with_body(scenario, "cobot-magic"))
    assert blocked and all(wall["id"] in b and "w:robot-1" in b for b in blocked)
