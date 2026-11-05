"""Operator chat: World Harness and Ego Harness turns, and exclusive robot control."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

from steward.assets.generators import steward_bot
from steward.ego.runtime import EgoRuntime, RobotLink
from steward.net import JsonServer
from steward.world.chat import ChatConfig, HarnessChat
from steward.world.server import StewardHost

EVENTS = [
    {"type": "session", "sessionId": "session-1"},
    {"type": "tool_call", "callId": "c1", "tool": "some_tool", "input": {"action": "x"}},
    {"type": "tool_result", "callId": "c1", "status": "completed", "result": "{\"ok\": true}"},
    {"type": "text", "text": "完成"},
    {"type": "final", "sessionId": "session-1", "text": "完成"},
]


def _launcher(tmp_path, body: str) -> Path:
    """An executable stand-in for the DSH launcher that runs the Python ``body``: a sh wrapper, or on
    Windows a cmd wrapper as npm generates there."""
    script = tmp_path / "fake_dsh.py"
    script.write_text("import os, shutil, sys, time\n" + body, encoding="utf-8")
    if os.name == "nt":
        launcher = tmp_path / "dsh.CMD"
        launcher.write_text(f'@"{sys.executable}" "{script}" %*\r\n', encoding="utf-8")
    else:
        launcher = tmp_path / "dsh"
        launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8")
        launcher.chmod(0o755)
    return launcher


def _fake_dsh(tmp_path):
    """A stand-in launcher that records argv, stdin and the patch overlay, then prints run events."""
    lines = "\n".join(json.dumps(e, ensure_ascii=False) for e in EVENTS)
    return _launcher(tmp_path, (
        "home = os.environ['DSH_HOME']\n"
        "args = sys.argv[1:]\n"
        "if '--patch' in args:\n"
        "    shutil.copy(args[args.index('--patch') + 1], os.path.join(home, 'patch.yml'))\n"
        "with open(os.path.join(home, 'argv'), 'a', encoding='utf-8') as fh:\n"
        "    fh.write(''.join(a + '\\n' for a in args) + '\\n')\n"
        "with open(os.path.join(home, 'stdin'), 'w', encoding='utf-8') as fh:\n"
        "    fh.write(sys.stdin.buffer.read().decode('utf-8'))\n"
        "with open(os.path.join(home, 'cwd'), 'w', encoding='utf-8') as fh:\n"
        "    fh.write(os.getcwd())\n"
        f"sys.stdout.buffer.write({lines!r}.encode('utf-8') + b'\\n')\n"))


def _home(tmp_path, name: str, profile: str):
    home = tmp_path / name
    (home / "profiles" / profile).mkdir(parents=True)
    return home


def _wait_idle(chat: HarnessChat, key: str, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while chat.agents[key].busy and time.time() < deadline:
        time.sleep(0.02)
    assert not chat.agents[key].busy


def test_a_robot_ego_api_answering_an_unexpected_shape_does_not_break_the_chat_state(tmp_path):
    """The console polls /console/chat continuously; a malformed robot answer must be absorbed.

    Only "active" is read from /ego/alerts, and any answer without it (an older or different
    Ego API, a proxy error page) raised out of the poll and took the whole chat endpoint down.
    """
    api = JsonServer({("GET", "/ego/alerts"): lambda q, b: {"alerts": []}}).start()
    try:
        robot = _home(tmp_path, "robot-1", "ego")
        chat = HarnessChat(ChatConfig(dsh=_fake_dsh(tmp_path), cwd=tmp_path, env_file=tmp_path / ".env",
                                      robot_home=robot, ego_api=api.url),
                           lambda: "http://127.0.0.1:7999")
        state = chat.state()
        assert state["agents"]["robot"]["alerts"] == []
    finally:
        api.stop()


def test_world_chat_keeps_its_session_and_targets_this_hosts_wca(tmp_path):
    world = _home(tmp_path, "world", "world-task")
    chat = HarnessChat(ChatConfig(dsh=_fake_dsh(tmp_path), cwd=tmp_path, env_file=tmp_path / ".env", world_home=world),
                       lambda: "http://127.0.0.1:7999")
    chat.send("world", "把杯子放到桌子上")
    _wait_idle(chat, "world")
    chat.send("world", "再放回去")
    _wait_idle(chat, "world")
    state = chat.state()
    assert [m["role"] for m in state["messages"]] == ["user", "tool", "assistant", "user", "tool", "assistant"]
    assert {m["agent"] for m in state["messages"]} == {"world"}
    assert state["messages"][1]["status"] == "completed" and state["agents"]["world"]["sessionId"] == "session-1"
    turns = (world / "argv").read_text(encoding="utf-8").split("\n\n")
    assert "--session-id" not in turns[0] and "--session-id\nsession-1" in turns[1]
    assert "http://127.0.0.1:7999" in (world / "patch.yml").read_text(encoding="utf-8")
    assert (world / "stdin").read_text(encoding="utf-8").startswith("操作员请求：")
    assert chat.state(since=state["messages"][-1]["id"])["messages"] == []
    assert not state["agents"]["robot"]["available"]


def test_robot_chat_takes_control_for_the_turn_and_gives_it_back(tmp_path):
    robot = _home(tmp_path, "robot-1", "ego")
    calls = []

    def control(q, b):
        calls.append(b["action"])
        assert b["owner"] == "operator"
        return {"owner": "operator" if b["action"] == "acquire" else None}

    ego = JsonServer({("POST", "/ego/control"): control, ("GET", "/ego/control"): lambda q, b: {"owner": None}}).start()
    try:
        chat = HarnessChat(ChatConfig(dsh=_fake_dsh(tmp_path), cwd=tmp_path, env_file=tmp_path / ".env",
                                      robot_home=robot, ego_api=ego.url), lambda: "unused")
        chat.send("robot", "把杯子放到餐桌上")
        _wait_idle(chat, "robot")
    finally:
        ego.stop()
    assert calls == ["acquire", "release"]
    patch = (robot / "patch.yml").read_text(encoding="utf-8")
    assert "steward-ego-runtime" in patch and ego.url in patch and "owner: operator" in patch
    cwd = Path((robot / "cwd").read_text(encoding="utf-8"))
    assert cwd.parts[-2:] == ("robot-1", "workspace"), "the robot runs in its empty workspace"
    stdin = (robot / "stdin").read_text(encoding="utf-8")
    assert stdin.startswith("你是家用服务机器人") and "World State" not in stdin
    assert [m["agent"] for m in chat.state()["messages"]] == ["robot"] * 3


def test_stopping_a_robot_turn_ends_the_run_and_releases_control(tmp_path):
    robot = _home(tmp_path, "robot-1", "ego")
    script = _launcher(tmp_path, "sys.stdin.read()\nprint('{\"type\": \"session\", \"sessionId\": \"s-1\"}', flush=True)\ntime.sleep(30)\n")
    calls = []

    def control(q, b):
        calls.append(b["action"])
        return {"owner": None}

    ego = JsonServer({("POST", "/ego/control"): control, ("GET", "/ego/control"): lambda q, b: {"owner": None},
                      ("POST", "/skill/stop"): lambda q, b: calls.append("stop") or {"stopped": None}}).start()
    try:
        chat = HarnessChat(ChatConfig(dsh=script, cwd=tmp_path, env_file=tmp_path / ".env", robot_home=robot,
                                      ego_api=ego.url), lambda: "unused")
        chat.send("robot", "去拿椅子")
        deadline = time.time() + 5
        while chat.agents["robot"].session_id is None and time.time() < deadline:
            time.sleep(0.02)
        started = time.time()
        assert chat.cancel("robot") == {"stopped": True}
        _wait_idle(chat, "robot")
        assert time.time() - started < 5
    finally:
        ego.stop()
    assert calls == ["acquire", "stop", "release"]
    roles = [m["role"] for m in chat.state()["messages"]]
    assert "system" in roles and "error" not in roles
    assert chat.cancel("robot") == {"stopped": False}


def test_a_turn_ended_by_the_budget_reports_a_designed_stop_not_a_failure(tmp_path):
    """The wall-clock budget ends long turns by design: ``_terminate`` kills the harness process, and
    on Windows that exit is non-zero with a stderr tail holding only the node startup warning, which
    used to be reported as a run failure and alarmed the operator. A budget timeout (like an operator
    cancel) must read as a designed end: no error entry, no failure alert."""
    script = _launcher(tmp_path, "time.sleep(30)\n")
    calls = []

    def control(q, b):
        calls.append(b["action"])
        return {"owner": None}

    ego = JsonServer({("POST", "/ego/control"): control, ("GET", "/ego/control"): lambda q, b: {"owner": None}}).start()
    try:
        chat = HarnessChat(ChatConfig(dsh=script, cwd=tmp_path, env_file=tmp_path / ".env", robot_home=_home(tmp_path, "robot-1", "ego"),
                                      ego_api=ego.url, timeout_s=0.4), lambda: "unused")
        chat.send("robot", "慢慢来")
        _wait_idle(chat, "robot")
    finally:
        ego.stop()
    roles = [m["role"] for m in chat.state()["messages"]]
    assert "error" not in roles
    assert any("回合预算" in m["text"] for m in chat.state()["messages"])


def test_shuttle_pauses_while_an_operator_controls_the_robot():
    from steward.mission import _wait_for_control

    owners = iter(["operator", "operator", None])
    api = JsonServer({("GET", "/ego/control"): lambda q, b: {"owner": next(owners)}}).start()
    waits = []
    try:
        _wait_for_control(api.url, waits.append, poll_s=0.01)
    finally:
        api.stop()
    assert waits == ["operator"]


def test_ego_runtime_control_blocks_other_clients_until_released(kitchen_root, tmp_path):
    host = StewardHost(kitchen_root, scenario="phase0-kitchen", run_dir=tmp_path)
    runtime = EgoRuntime(RobotLink(host.robot_servers["w:robot-1"].url), steward_bot().to_dict())
    try:
        runtime._poll_once()
        assert runtime.acquire_control("operator") == {"owner": "operator"}
        with pytest.raises(PermissionError, match="controlled by operator"):
            runtime.explore()
        with pytest.raises(PermissionError, match="controlled by operator"):
            runtime.run("rcpt-x")
        with pytest.raises(KeyError):
            runtime.run("rcpt-x", owner="operator")
        with pytest.raises(PermissionError, match="already controlled"):
            runtime.acquire_control("someone-else")
        runtime.release_control("someone-else")
        assert runtime.control() == {"owner": "operator"}
        runtime.release_control("operator")
        assert runtime.control() == {"owner": None}
        with pytest.raises(KeyError):
            runtime.run("rcpt-x")
    finally:
        runtime.executor.cancel()
        host.stop()
