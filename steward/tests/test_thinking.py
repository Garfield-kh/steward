"""The console's live robot thinking, read from the robot's DSH session file."""

from __future__ import annotations

import json

import zstandard

from steward.world.thinking import thinking_state


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

    assert state["robot-1"] == {"entries": [{"step": 1, "active": True, "text": "先看清桌面上的拼图块",
                                             "conclusion": ["run_code"]}], "error": None}


def test_a_corrupt_session_file_is_reported_not_raised(tmp_path):
    session = tmp_path / "sessions" / "workspace" / "session-1"
    session.mkdir(parents=True)
    (session / "session.v4.jsonl.zstd").write_bytes(b"not zstd")

    state = thinking_state({"robot-1": str(tmp_path)})

    assert state["robot-1"]["entries"] == [] and "zstd failed to decode" in state["robot-1"]["error"]
