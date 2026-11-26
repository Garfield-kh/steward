"""Receipt arm identity survives intervening predictions and reaches physical commands."""

from types import SimpleNamespace

import pytest

from steward.ego.bodies import robot_model
from steward.ego.cobot_arm import PiperArm
from steward.ego.runtime import EgoRuntime, RobotLink


@pytest.fixture
def runtime():
    model = robot_model("cobot-magic")
    return EgoRuntime(RobotLink("http://127.0.0.1:1"), model, arm=PiperArm(model))


def test_right_arm_receipt_commands_the_right_gripper_after_a_left_prediction(runtime, monkeypatch):
    right = runtime.predict("navigate", {"x": 0, "y": 0, "arm": "R"})
    left = runtime.predict("navigate", {"x": 0, "y": 0, "arm": "L"})
    assert right["feasible"] and left["feasible"]
    assert right["id"] != left["id"]
    commands = []
    monkeypatch.setattr(runtime.link, "command", commands.append)

    def start(skill, receipt):
        runtime.executor._command({"gripper": "close"})
        return SimpleNamespace(to_dict=lambda: {"state": "succeeded"})

    monkeypatch.setattr(runtime.executor, "start", start)
    runtime.run(right["id"])
    assert commands == [{"gripper": {"R_link6": "close"}}]
    assert runtime.arm.geometry.gripper == "R_link6"
    assert runtime.esim.arm is runtime.arm


def test_arm_prediction_does_not_switch_a_running_executor(runtime):
    runtime.executor.current = SimpleNamespace(state="running", skill="pick", id="job-1")
    receipt = runtime.predict("navigate", {"x": 0, "y": 0, "arm": "R"})
    assert runtime.arm.geometry.gripper == "L_link6"
    with pytest.raises(RuntimeError, match="still running"):
        runtime.run(receipt["id"])
    assert runtime.arm.geometry.gripper == "L_link6"
    assert not runtime.receipts[receipt["id"]].get("used")


def test_other_object_load_does_not_block_an_arm_selection(runtime, monkeypatch):
    runtime.estimator.grippers = {"L_link6": {"holding": True}}
    receipt = runtime.predict("navigate", {"x": 0, "y": 0, "arm": "R"})
    assert receipt["feasible"]
    monkeypatch.setattr(runtime.executor, "start", lambda *_: SimpleNamespace(to_dict=lambda: {"state": "succeeded"}))
    runtime.run(receipt["id"])
    assert runtime.arm.geometry.gripper == "R_link6"
    assert runtime.estimator.grippers["L_link6"]["holding"]


@pytest.mark.parametrize("arm", ["r", "other", 1, None])
def test_invalid_arm_is_rejected(runtime, arm):
    with pytest.raises(ValueError, match="arm"):
        runtime.predict("navigate", {"x": 0, "y": 0, "arm": arm})


def test_run_rechecks_a_drawer_handle_acquired_after_prediction(runtime, monkeypatch):
    receipt = runtime.predict("navigate", {"x": 0, "y": 0, "arm": "R"})
    runtime.estimator.grippers = {"L_link6": {"holdingHandle": True}}
    commands = []
    monkeypatch.setattr(runtime.link, "command", commands.append)
    with pytest.raises(PermissionError, match="L_link6"):
        runtime.run(receipt["id"])
    assert commands == []
    assert runtime.arm.geometry.gripper == "L_link6"


def test_single_arm_body_rejects_an_explicit_selection():
    runtime = EgoRuntime(RobotLink("http://127.0.0.1:1"), robot_model("steward-bot"))
    with pytest.raises(ValueError, match="single-arm"):
        runtime.predict("pick", {"target": "e:driver", "arm": "R"})
