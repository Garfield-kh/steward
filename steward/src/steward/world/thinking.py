"""The robot agent's live thinking for the console: the reasoning blocks of the latest DSH
session under the robot's Harness home, read from the newest ``session.v4.jsonl.zstd`` file.

Reads are cached per file: a session file that did not change (same modification time and size)
is not decoded again, and each robot's read is limited to ``_TIME_BUDGET_S`` so a growing session
file cannot hold up the console's state refresh.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import threading
import time

import zstandard

_TEXT_LIMIT = 2000
_TIME_BUDGET_S = 0.5

_cache: dict[str, tuple[float, int, list[dict]]] = {}
_lock = threading.Lock()


def session_files(robot_home: str) -> set[str]:
    """Every DSH session file of the robot's Harness home."""
    return set(glob.glob(f"{robot_home}/sessions/*/*/session.v4.jsonl.zstd"))


def _session_file(robot_home: str, exclude: frozenset[str]) -> str | None:
    """The newest DSH session file of the robot's Harness home that is not in ``exclude``."""
    files = session_files(robot_home) - exclude
    return max(files, key=os.path.getmtime) if files else None


def _decode(path: str) -> list[dict]:
    try:
        with open(path, "rb") as fh:
            data = zstandard.ZstdDecompressor().stream_reader(fh, read_across_frames=True).read()
    except zstandard.ZstdError as exc:
        raise RuntimeError(f"zstd failed to decode {path}: {exc}") from exc
    text = data.decode("utf-8", errors="replace")  # dsh writes UTF-8 whatever the locale
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _rows(path: str, deadline: float) -> list[dict]:
    """The session's rows, from the cache while the file is unchanged."""
    stat = os.stat(path)
    if time.monotonic() > deadline:
        raise TimeoutError(f"reading {path} would exceed the thinking budget")
    with _lock:
        cached = _cache.get(path)
        if cached is not None and cached[0] == stat.st_mtime and cached[1] == stat.st_size:
            return cached[2]
    rows = _decode(path)
    with _lock:
        _cache[path] = (stat.st_mtime, stat.st_size, rows)
        for other in [p for p in _cache if p != path][:20]:
            _cache.pop(other, None)
    return rows


def _entry_id(path: str, index: int) -> str:
    """Identify an append-only session row without exposing its filesystem path."""
    return f"{hashlib.sha256(path.encode()).hexdigest()[:16]}:{index}"


def _reasoning(row: dict) -> str:
    """Reasoning text of an assistant event, or empty text for other events."""
    if row.get("type") != "assistant/message":
        return ""
    return "".join(p["text"] for p in row["data"]["message"]["content"] if p["type"] == "reasoning").strip()


def _thinking_of(rows: list[dict], limit: int, path: str) -> list[dict]:
    """One entry per step that has reasoning: its step number, whether it is the step still being
    thought out (after the latest ``step/start``), its preview and its conclusion (the names of the
    tools it called), oldest first. Each preview includes an opaque full-text id, original character
    count and explicit truncation flag."""
    starts = [r for r in rows if r.get("type") == "step/start"]
    last_start = starts[-1]["time"] if starts else 0
    out = []
    for index, row in enumerate(rows):
        reasoning = _reasoning(row)
        if not reasoning:
            continue
        message = row["data"]["message"]
        tools = [p["name"] for p in message["content"] if p["type"] == "tool-call"]
        out.append({"id": _entry_id(path, index), "step": row["data"]["step"], "active": row["time"] >= last_start,
                    "text": reasoning[-_TEXT_LIMIT:], "truncated": len(reasoning) > _TEXT_LIMIT,
                    "totalChars": len(reasoning), "conclusion": tools})
    return out[-limit:]


def thinking_state(robot_homes: dict[str, str], limit: int = 3, exclude: frozenset[str] = frozenset()) -> dict[str, dict]:
    """Live thinking per robot: its latest reasoning blocks and conclusions.

    :param robot_homes: robot id to its DSH home directory.
    :param limit: reasoning blocks per robot, newest last.
    :param exclude: session files not to read (those of an episode the operator reset).
    :returns: per robot ``{entries, error}`` (one of them None)."""
    deadline = time.monotonic() + _TIME_BUDGET_S
    out: dict[str, dict] = {}
    for rid, home in robot_homes.items():
        path = _session_file(home, exclude)
        if path is None:
            continue
        try:
            out[rid] = {"entries": _thinking_of(_rows(path, deadline), limit, path), "error": None}
        except (OSError, RuntimeError, TimeoutError, json.JSONDecodeError) as exc:
            out[rid] = {"entries": [], "error": f"{type(exc).__name__}: {exc}"}
    return out


def thinking_text(robot_homes: dict[str, str], robot: str, entry: str,
                  exclude: frozenset[str] = frozenset()) -> dict:
    """Read one reasoning block in full from the robot's current non-excluded session.

    :param robot_homes: configured robot ids and Harness homes; callers cannot supply paths.
    :param robot: configured robot id.
    :param entry: opaque id returned by ``thinking_state``.
    :param exclude: sessions hidden after an episode reset.
    :returns: ``{id, text}`` with the untruncated reasoning text.
    :raises KeyError: unknown robot, expired session, or missing reasoning entry.
    :raises OSError: the session cannot be read.
    :raises RuntimeError: compressed session data is invalid.
    :raises TimeoutError: the read deadline expired.
    :raises json.JSONDecodeError: session events are malformed.
    """
    deadline = time.monotonic() + _TIME_BUDGET_S
    home = robot_homes.get(robot)
    path = _session_file(home, exclude) if home is not None else None
    if path is None:
        raise KeyError(entry)
    rows = _rows(path, deadline)
    for index, row in enumerate(rows):
        if _entry_id(path, index) == entry:
            text = _reasoning(row)
            if text:
                return {"id": entry, "text": text}
            break
    raise KeyError(entry)
