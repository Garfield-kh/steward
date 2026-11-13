"""Mission helpers: skill wait budgets, and track renumber recovery."""

from __future__ import annotations

import time

import pytest

from steward import mission
from steward import mission as mission_mod
from steward.mission import MissionFailed, _refresh


class _Robot:
    """An Ego API of one job, whose world clock runs ``speed`` times the wall clock and whose job
    ends at world time ``ends_at`` (never, when None)."""

    def __init__(self, speed: float, ends_at: float | None = None):
        self.speed, self.ends_at = speed, ends_at
        self.started = time.monotonic()
        self.waits: list[float] = []

    def now(self) -> float:
        return (time.monotonic() - self.started) * self.speed

    def call(self, method: str, url: str, body=None, timeout: float = 0.0) -> dict:
        if url.endswith("/clock/sleep"):
            return {"t": self.now()}
        assert url.endswith("/skill/wait"), url
        self.waits.append(body["timeoutS"])
        deadline = time.monotonic() + body["timeoutS"]
        while time.monotonic() < deadline and not self._ended():
            time.sleep(0.005)
        return {"id": body["job"], "state": "succeeded" if self._ended() else "running"}

    def _ended(self) -> bool:
        return self.ends_at is not None and self.now() >= self.ends_at


def test_a_skill_is_waited_for_while_the_world_has_not_had_its_seconds(monkeypatch):
    """A robot that segments its own images stops the world for most of the wall time: a skill of
    0.04 world seconds takes four times the budget on the wall clock, and the mission still gets it."""
    robot = _Robot(speed=0.05, ends_at=0.04)
    monkeypatch.setattr(mission, "call", robot.call)
    job = mission.wait_job("http://robot", "job-1", timeout_s=0.2)
    assert job["state"] == "succeeded"
    assert time.monotonic() - robot.started >= 0.75


def test_a_skill_that_outlasts_its_budget_on_both_clocks_is_returned_running(monkeypatch):
    """At four times real time the world has had the budget after a quarter of it; the mission stays
    for the whole budget on the wall clock, as it does in a world that runs in real time."""
    robot = _Robot(speed=4.0)
    monkeypatch.setattr(mission, "call", robot.call)
    job = mission.wait_job("http://robot", "job-1", timeout_s=0.2)
    assert job["state"] == "running"
    assert 0.2 <= time.monotonic() - robot.started < 0.6
    assert max(robot.waits) <= 0.2


def test_a_job_under_a_world_clock_that_stands_still_is_given_up(monkeypatch):
    robot = _Robot(speed=0.0)
    monkeypatch.setattr(mission, "call", robot.call)
    monkeypatch.setattr(mission, "STALL_S", 0.3)
    job = mission.wait_job("http://robot", "job-1", timeout_s=0.1)
    assert job["state"] == "running"
    assert 0.3 <= time.monotonic() - robot.started < 1.0


def _bowl(oid: str, pos: list[float], size: list[float], *, role: str = "bowl",
          category: str | None = None, color: str = "blue", existence: float = 1.0) -> dict:
    category = category or role
    return {
        "id": oid,
        "category": category,
        "role": role,
        "categories": {category: 0.9},
        "position": pos,
        "sizeM": size,
        "extentM": size,
        "color": color,
        "existence": existence,
        "uncertaintyM": 0.02,
    }


def test_refresh_recovers_a_lost_bowl_by_footprint(monkeypatch):
    """After a merge the id is gone and the rim view may read as plate; footprint still names it."""
    lost = _bowl("e:4", [1.5, 0.0, 0.90], [0.13, 0.13, 0.05], role="bowl", color="blue")
    renumbered = _bowl("e:40", [1.51, 0.01, 0.91], [0.12, 0.12, 0.04],
                       role="plate", category="plate", color="blue")
    neighbour = _bowl("e:5", [1.50, 0.02, 0.88], [0.22, 0.22, 0.07],
                      role="bowl", color="red")

    def fake_call(method, url, *args, **kwargs):
        assert method == "GET" and url.endswith("/ego/summary")
        return {"objects": [renumbered, neighbour]}

    monkeypatch.setattr(mission_mod, "call", fake_call)
    found = _refresh("http://unused", lost)
    assert found["id"] == "e:40"


def test_refresh_prefers_matching_footprint_among_same_role_neighbours(monkeypatch):
    """Stacked bowls share nearly the same xy; size must break the tie, not only distance."""
    lost = _bowl("e:3", [1.5, 0.0, 0.89], [0.17, 0.17, 0.06], role="bowl", color="green")
    small = _bowl("e:30", [1.50, 0.00, 0.92], [0.13, 0.13, 0.05], role="bowl", color="blue")
    medium = _bowl("e:31", [1.505, 0.005, 0.90], [0.17, 0.16, 0.06], role="bowl", color="green")

    monkeypatch.setattr(mission_mod, "call",
                        lambda method, url, *a, **k: {"objects": [small, medium]})
    found = _refresh("http://unused", lost)
    assert found["id"] == "e:31"


def test_refresh_still_fails_when_nothing_nearby_matches(monkeypatch):
    lost = _bowl("e:4", [1.5, 0.0, 0.90], [0.13, 0.13, 0.05])
    far = _bowl("e:9", [3.0, 3.0, 0.9], [0.13, 0.13, 0.05])
    monkeypatch.setattr(mission_mod, "call", lambda *a, **k: {"objects": [far]})
    with pytest.raises(MissionFailed, match="e:4"):
        _refresh("http://unused", lost)


def test_safe_receipt_rematches_unknown_support_id(monkeypatch):
    """Place after a nest renumber: rematch ``on`` like an unknown pick target."""
    from steward.mission import _safe_receipt

    stale = _bowl("e:31", [-2.75, 0.48, 0.89], [0.19, 0.19, 0.06], role="bowl", color="green")
    fresh = _bowl("e:484", [-2.75, 0.48, 0.89], [0.20, 0.20, 0.06], role="bowl", color="green")
    predicts: list[dict] = []
    log: list[dict] = []

    def fake_call(method, url, body=None, **kwargs):
        if method == "GET" and url.endswith("/ego/summary"):
            return {"objects": [fresh]}
        if method == "POST" and url.endswith("/ego/predict"):
            predicts.append(dict(body["params"]))
            on = body["params"].get("on")
            if on == "e:31":
                return {"feasible": False, "withinRiskLimit": False, "issues": ["unknown support 'e:31'"],
                        "blockers": [], "id": "r0", "risk": 1.0, "skill": "place"}
            assert on == "e:484"
            return {"feasible": True, "withinRiskLimit": True, "issues": [], "blockers": [],
                    "id": "r1", "risk": 0.05, "skill": "place", "plan": {"spot": [0.0, 0.0]}}
        raise AssertionError(f"unexpected {method} {url}")

    monkeypatch.setattr(mission_mod, "call", fake_call)
    monkeypatch.setattr(mission_mod, "_approach_to_observe",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no closer look")))
    receipt = _safe_receipt("http://unused", "place", {"on": "e:31"}, stale, log)
    assert receipt["id"] == "r1"
    assert predicts == [{"on": "e:31"}, {"on": "e:484"}]
    assert any(e.get("step") == "support-rematch" and e.get("from") == "e:31" and e.get("to") == "e:484"
               for e in log)
