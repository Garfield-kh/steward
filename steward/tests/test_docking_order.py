"""Command ordering during docking, without physics stepping or asynchronous workers."""

import json
from pathlib import Path

import pytest

from steward.ego.skills import Job, SkillExecutor, SkillFailed, SkillStopped


def _executor(monkeypatch, holding=False):
    state = {"stamp": 1.0, "odometry": {"x": 0.0, "y": 0.0, "yaw": 0.0},
             "gripper": {"holding": holding}, "joints": {"reach": 0.3}, "armResting": not holding}
    commands = []
    executor = SkillExecutor(commands.append, lambda: state, lambda eid: eid)
    job = Job("job-1", "pick", "receipt-1")
    plan = {"path": [[0.0, 0.0], [1.0, 0.0]], "predock": [1.0, 0.0], "dock": [1.2, 0.0],
            "yaw": 0.0, "retractReach": 0.27}

    def turn(yaw):
        commands.append({"base": [state["odometry"]["x"], state["odometry"]["y"], yaw]})
        state["odometry"]["yaw"] = yaw

    def translate(x, y, yaw, *args, **kwargs):
        commands.append({"base": [x, y, yaw]})
        state["odometry"].update(x=x, y=y, yaw=yaw)

    def set_joint(name, target):
        assert job.log[-1] == "docked"
        assert state["odometry"]["x"] == plan["dock"][0]
        commands.append({name: target})

    monkeypatch.setattr(executor, "_turn_to", turn)
    monkeypatch.setattr(executor, "_translate_to", translate)
    monkeypatch.setattr(executor, "_set_joint", set_joint)
    return executor, job, plan, state, commands


@pytest.mark.parametrize("holding", [False, True])
def test_arm_preparation_follows_verified_docking(monkeypatch, holding):
    executor, job, plan, _, commands = _executor(monkeypatch, holding)
    executor._dock(job, plan)
    expected = json.loads((Path(__file__).parent / "expected/docking-order.json").read_text())
    assert commands == expected["holding" if holding else "empty"]


def test_floor_placement_keeps_carrying_reach_after_docking(monkeypatch):
    executor, job, plan, _, commands = _executor(monkeypatch, holding=True)
    executor._dock(job, plan, retract=False)
    assert all(set(command) == {"base"} for command in commands)
    assert job.log[-1] == "docked"


@pytest.mark.parametrize("failure", ["drive", "unexpected", "missed_dock", "stopped"])
def test_failed_approach_does_not_start_arm_recovery_or_homing(monkeypatch, failure):
    executor, job, plan, _, commands = _executor(monkeypatch)
    monkeypatch.setattr(executor, "_safe_retract", lambda: pytest.fail("arm must stay still before arrival"))
    executor._home_empty = lambda tick: pytest.fail("unsuccessful approach must not home the arms")
    if failure == "missed_dock":
        monkeypatch.setattr(executor, "_translate_to", lambda *args, **kwargs: None)
        monkeypatch.setattr(executor, "_backoff", lambda plan: None)
    else:
        def drive(job, path, yaw):
            raise {"drive": SkillFailed, "unexpected": RuntimeError, "stopped": SkillStopped}[failure]("drive stopped")
        monkeypatch.setattr(executor, "_drive", drive)
    executor._run(job, "pick", {"plan": plan})
    assert job.state == ("stopped" if failure == "stopped" else "failed")
    assert all(set(command) == {"base"} for command in commands)
    assert "docked" not in job.log


def test_failure_after_arrival_still_allows_arm_recovery(monkeypatch):
    executor, job, plan, _, _ = _executor(monkeypatch)
    recovered = []

    def pick(job, plan):
        executor._dock(job, plan)
        raise SkillFailed("grasp missed")

    monkeypatch.setattr(executor, "_pick", pick)
    monkeypatch.setattr(executor, "_safe_retract", lambda: recovered.append(True))
    executor._run(job, "pick", {"plan": plan})
    assert job.state == "failed" and recovered == [True]
