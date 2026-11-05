"""The phase0-misnamed level: ideal detections under other names, for debugging reflection."""

import math

import pytest

from steward.cli import MISNAMES_CONFIG
from steward.phases import load_misnames
from steward.world.server import StewardHost


def test_the_shipped_misnames_rename_the_box_and_the_table():
    assert load_misnames(MISNAMES_CONFIG) == {"box": "picture", "table": "dinner table"}


def test_a_misnames_file_without_names_is_rejected(tmp_path):
    bad = tmp_path / "m.toml"
    bad.write_text('[misnames]\nbox = ""\n', encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty names"):
        load_misnames(bad)


def test_the_misnamed_level_reports_renamed_detections_and_phase0_the_true_ones(kitchen_root, tmp_path):
    host = StewardHost(kitchen_root, scenario="puzzle-dog", speed=math.inf, run_dir=tmp_path, phase="phase0-misnamed",
                       misnames={"box": "picture", "table": "dinner table"})
    try:
        host.sim.step_seconds(0.2)
        seen = {d["category"] for d in host.sim.sensors("w:robot-1")["detections"]}
        assert {"picture", "dinner table"} <= seen and not {"box", "table"} & seen
        host.set_phase("phase0")
        seen = {d["category"] for d in host.sim.sensors("w:robot-1")["detections"]}
        assert {"box", "table"} <= seen
    finally:
        host.sim.stop()


def test_the_misnamed_level_needs_misnames(kitchen_root, tmp_path):
    with pytest.raises(ValueError, match="no misnames configured"):
        StewardHost(kitchen_root, scenario="puzzle-dog", speed=math.inf, run_dir=tmp_path, phase="phase0-misnamed")
