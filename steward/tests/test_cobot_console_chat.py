"""``steward cobot`` console chat and plan surfaces match ``steward up``."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from steward.authoring.service import AuthoringService, build_cobot_screw_board
from steward.assets.cobot import cobot_magic
from steward.cli import _cobot_mission_plan, _chat_config
from steward.ego.cobot_server import cobot_ego_api
from steward.ego.cobot_skills import CobotSkills
from steward.ego.estimator import EgoEstimator
from steward.net import call
from steward.world.chat import ChatConfig, HarnessChat
from steward.world.server import StewardHost


def test_cobot_mission_plan_marks_first_step_active():
    plan = _cobot_mission_plan("drive-screws", "拧螺丝", round_n=1)
    assert plan["goal"] == "拧螺丝"
    assert plan["source"] == "mission"
    assert plan["steps"][0]["status"] == "active"
    assert all(st["status"] == "pending" for st in plan["steps"][1:])


def test_cobot_ego_api_publishes_plan_control_and_skill_routes(tmp_path):
    est = EgoEstimator(robot_model=cobot_magic().to_dict())
    skills = CobotSkills(command=lambda cmd: None, state=lambda: {"stamp": 0.0, "odometry": {},
                                                                  "joints": {}, "grippers": {}})
    api = cobot_ego_api(est, port=0, skills=skills)
    try:
        board = api.cobot_view
        record = board.set_plan(_cobot_mission_plan("drive-screws", "拧四颗螺丝", round_n=1,
                                                   current="感知螺丝板与零件"))
        view = call("GET", f"{api.url}/ego/plan")
        assert view["plan"]["goal"] == record["goal"]
        assert view["boot"] == board.boot
        assert view["activity"]
        assert call("GET", f"{api.url}/ego/control") == {"owner": None}
        assert call("POST", f"{api.url}/ego/control", {"action": "acquire", "owner": "robot-1"}) == {
            "owner": "robot-1"}
        assert call("POST", f"{api.url}/ego/control", {"action": "release", "owner": "robot-1"}) == {
            "owner": None}
        chat = HarnessChat(ChatConfig(dsh=Path("/bin/true"), cwd=tmp_path, env_file=tmp_path / ".env",
                                      robot_home=tmp_path / "robot", ego_api=api.url),
                           lambda: "http://127.0.0.1:9")
        state = chat.state()
        assert state["agents"]["robot"]["plan"]["plan"]["goal"] == "拧四颗螺丝"
    finally:
        api.stop()


def test_cobot_host_chat_agents_are_wired_like_up(tmp_path):
    """A cobot host built with the same harness defaults exposes World and robot chat agents."""
    root = tmp_path / "root"
    build_cobot_screw_board(AuthoringService(root))
    args = type("A", (), {
        "root": str(root), "scenario": "cobot-screw-board", "speed": math.inf, "seed": 0,
        "wca_port": 0, "robot_port": 0, "api_port": 0, "phase": "phase0", "freeze_idle": False,
        "bind_wait": 0.0, "body": None, "pause_on_perception_fault": False, "dispatch_task": False,
        "misnames_config": "", "world_harness": str(tmp_path / "world"),
        "robot_harness": str(tmp_path / "robot-1"), "turn_budget_s": 60.0, "ego_api": None,
    })()
    (tmp_path / "world" / "profiles" / "world-task").mkdir(parents=True)
    (tmp_path / "robot-1" / "profiles" / "ego").mkdir(parents=True)
    assert _chat_config(args) is not None
    host = StewardHost(str(root), scenario="cobot-screw-board", speed=math.inf, seed=0, run_dir=tmp_path / "run",
                       wca_port=0, robot_ports={}, chat=_chat_config(args), phase="phase0",
                       fallback_phase="phase0", dispatch_task=False).start()
    try:
        assert set(host.chat.agents) == {"world", "robot"}
        est = EgoEstimator(robot_model=cobot_magic().to_dict())
        rid = host.scenario["robots"][0]
        skills = CobotSkills(command=lambda cmd: host.sim.command(rid, cmd),
                             state=lambda: host.sim.sensors(rid))
        api = cobot_ego_api(est, port=0, live=lambda: host.sim.proprioception(rid), skills=skills)
        try:
            host.attach_ego_api(rid, api.url, skills=True)
            host.chat.config.ego_api = api.url
            api.cobot_view.set_plan(_cobot_mission_plan("drive-screws", host.scenario["task"], round_n=1))
            host.chat.post_note("robot", "脚本任务 drive-screws 第 1 轮开始")
            chat = call("GET", f"{host.wca.url}/console/chat")
            assert "world" in chat["agents"] and "robot" in chat["agents"]
            assert chat["agents"]["robot"]["plan"]["plan"]["source"] == "mission"
            assert any(m["role"] == "system" and "drive-screws" in m["text"] for m in chat["messages"])
        finally:
            api.stop()
    finally:
        host.stop()
