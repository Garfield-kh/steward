"""Shared test helpers: configuration documents and fixture paths."""

import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
sys.path.insert(0, str(PACKAGE / "src"))

from steward_rsi.config import parse  # noqa: E402

FIXTURE = HERE / "fixtures" / "airplane_success"
REPO_ROOT = PACKAGE.parents[1]
GOAL = json.loads((PACKAGE / "src" / "steward_rsi" / "data" / "toy_assembly_goal.json").read_text())
SPEC = json.loads((PACKAGE / "src" / "steward_rsi" / "data" / "toy_assembly_spec.json").read_text())


def config(overrides: dict | None = None, path: Path | None = None):
    data = {"advisor": {"provider": "deepseek"}, "paths": {"repo_root": str(REPO_ROOT), "output_dir": "/tmp/rsi-test-runs"}}
    for section, values in (overrides or {}).items():
        data.setdefault(section, {}).update(values)
    return parse(data, path or (PACKAGE / "rsi.toml"))


def load(name: str) -> dict:
    return json.loads((FIXTURE / name).read_text())


def deepcopy(x):
    return copy.deepcopy(x)
