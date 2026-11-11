"""How long a scripted mission waits for a skill: on the wall clock and on the world's."""

from __future__ import annotations

import time

from steward import mission


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
