"""The console's live robot thinking, read from the robot's DSH session file."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import zstandard

from steward.net import HttpError
from steward.world.console import console_thinking_text
from steward.world.thinking import session_files, thinking_state, thinking_text


def _row(kind: str, time: int, data: dict | None = None) -> bytes:
    return (json.dumps({"type": kind, "time": time, "data": data or {}}, ensure_ascii=False) + "\n").encode("utf-8")


def test_reasoning_spread_over_several_zstd_frames_is_read_with_its_tool_calls(tmp_path):
    session = tmp_path / "sessions" / "workspace" / "session-1"
    session.mkdir(parents=True)
    message = {"content": [{"type": "reasoning", "text": "先看清桌面上的拼图块"},
                           {"type": "tool-call", "name": "run_code"}]}
    frames = [_row("step/start", 1), _row("assistant/message", 2, {"step": 1, "message": message})]
    compressor = zstandard.ZstdCompressor()
    (session / "session.v4.jsonl.zstd").write_bytes(b"".join(compressor.compress(f) for f in frames))

    state = thinking_state({"robot-1": str(tmp_path)})

    entry = state["robot-1"]["entries"][0]
    assert entry["id"].endswith(":1")
    assert state["robot-1"] == {"entries": [{"id": entry["id"], "step": 1, "active": True, "text": "先看清桌面上的拼图块",
                                             "truncated": False, "totalChars": 10,
                                             "conclusion": ["run_code"]}], "error": None}


def test_a_corrupt_session_file_is_reported_not_raised(tmp_path):
    session = tmp_path / "sessions" / "workspace" / "session-1"
    session.mkdir(parents=True)
    (session / "session.v4.jsonl.zstd").write_bytes(b"not zstd")

    state = thinking_state({"robot-1": str(tmp_path)})

    assert state["robot-1"]["entries"] == [] and "zstd failed to decode" in state["robot-1"]["error"]


def test_the_sessions_of_a_reset_episode_are_no_longer_read(tmp_path):
    session = tmp_path / "sessions" / "workspace" / "session-1"
    session.mkdir(parents=True)
    message = {"content": [{"type": "reasoning", "text": "上一轮的思考"}]}
    (session / "session.v4.jsonl.zstd").write_bytes(
        zstandard.ZstdCompressor().compress(_row("assistant/message", 1, {"step": 1, "message": message})))

    state = thinking_state({"robot-1": str(tmp_path)}, exclude=frozenset(session_files(str(tmp_path))))

    assert state == {}


@pytest.mark.parametrize("length", [1999, 2000, 2001, 8000])
def test_preview_marks_only_truncated_text_and_full_read_retains_the_start(tmp_path, length):
    session = tmp_path / "sessions/workspace/session-1"
    session.mkdir(parents=True)
    text = "开头" + "想" * (length - 4) + "结尾"
    message = {"content": [{"type": "reasoning", "text": text}]}
    path = session / "session.v4.jsonl.zstd"
    path.write_bytes(zstandard.ZstdCompressor().compress(_row("assistant/message", 1, {"step": 1, "message": message})))

    homes = {"robot-1": str(tmp_path)}
    entry = thinking_state(homes)["robot-1"]["entries"][0]
    assert entry["text"] == text[-2000:]
    assert entry["totalChars"] == length
    assert entry["truncated"] == (length > 2000)
    assert "fullText" not in entry
    assert thinking_text(homes, "robot-1", entry["id"]) == {"id": entry["id"], "text": text}

    with pytest.raises(KeyError):
        thinking_text(homes, "robot-1", entry["id"], exclude=frozenset({str(path)}))
    with pytest.raises(KeyError):
        thinking_text(homes, "other-robot", entry["id"])
    with pytest.raises(KeyError):
        thinking_text(homes, "robot-1", "../../arbitrary-file")


def test_full_text_route_rejects_missing_ids_and_excluded_sessions(tmp_path):
    session = tmp_path / "sessions/workspace/session-1"
    session.mkdir(parents=True)
    path = session / "session.v4.jsonl.zstd"
    message = {"content": [{"type": "reasoning", "text": "完整思考"}]}
    path.write_bytes(zstandard.ZstdCompressor().compress(_row("assistant/message", 1, {"step": 1, "message": message})))
    host = SimpleNamespace(chat=SimpleNamespace(config=SimpleNamespace(robot_home=tmp_path)), stale_sessions=frozenset())
    entry = thinking_state({"w:robot-1": str(tmp_path)})["w:robot-1"]["entries"][0]
    assert console_thinking_text(host, "w:robot-1", entry["id"])["text"] == "完整思考"
    with pytest.raises(HttpError) as missing:
        console_thinking_text(host, "", "")
    assert missing.value.status == 400
    host.stale_sessions = frozenset({str(path)})
    with pytest.raises(HttpError) as excluded:
        console_thinking_text(host, "w:robot-1", entry["id"])
    assert excluded.value.status == 404
    host.stale_sessions = frozenset()
    path.write_bytes(b"invalid compressed session")
    with pytest.raises(HttpError) as unreadable:
        console_thinking_text(host, "w:robot-1", entry["id"])
    assert unreadable.value.status == 503


def test_entry_ids_survive_appends_but_do_not_alias_other_sessions(tmp_path):
    homes = {}
    for name in ["first", "second"]:
        home = tmp_path / name
        path = home / "sessions/workspace/session-1/session.v4.jsonl.zstd"
        path.parent.mkdir(parents=True)
        message = {"content": [{"type": "reasoning", "text": name}]}
        path.write_bytes(zstandard.ZstdCompressor().compress(_row("assistant/message", 1, {"step": 1, "message": message})))
        homes[name] = str(home)
    before = thinking_state(homes)
    first_id = before["first"]["entries"][0]["id"]
    second_id = before["second"]["entries"][0]["id"]
    assert first_id != second_id
    with pytest.raises(KeyError):
        thinking_text(homes, "second", first_id)
    path = tmp_path / "first/sessions/workspace/session-1/session.v4.jsonl.zstd"
    with path.open("ab") as stream:
        stream.write(zstandard.ZstdCompressor().compress(_row("step/start", 2, {"step": 2})))
    assert thinking_state(homes)["first"]["entries"][0]["id"] == first_id
    assert thinking_text(homes, "first", first_id)["text"] == "first"
