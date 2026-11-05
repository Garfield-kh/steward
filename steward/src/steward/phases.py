"""Phase levels: which implementation each replaceable module runs at.

A later phase replaces modules; the implementations of earlier phases stay in the code base and can
be selected again at any time, also while an episode runs, to compare and debug. The Steward host
applies a level to the world side (what the sensors deliver); the robot's Ego Runtime follows by
inspecting each sensor frame, so the information boundary is unchanged.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

PHASES: dict[str, dict] = {
    "phase0": {
        "label": "Phase 0 · 理想感知",
        "description": "WSim 直接给出带噪声的检测结果（类别与包围盒）；ER 用卡尔曼融合",
        "perception": "ideal",
    },
    "phase0-misnamed": {
        "label": "Phase 0 · 理想感知（名称混淆）",
        "description": "同 Phase 0，但按 config/misnames.toml 把一些类别报成混淆的名字（例如盒子报成 picture），用来调试 Ego Harness 的反思",
        "perception": "ideal",
        "misnamed": True,
    },
    "phase1": {
        "label": "Phase 1 · RGB-D 感知链路",
        "description": "WSim 只给 RGB、深度与匿名实例掩码；ER 自行反投影、分类、多视角积累与形状补全",
        "perception": "rgbd",
        "masks": True,
    },
    "phase1-sam": {
        "label": "Phase 1 · SAM 3.1 分割",
        "description": "WSim 只给 RGB 与深度；ER 在关键帧调用 SAM 3.1 做开放词汇实例分割，类别以分割词为主、形状先验为辅；可选用深度几何聚类补上 SAM 漏掉的区域",
        "perception": "rgbd",
        "masks": False,
    },
}
DEFAULT_PHASE = "phase0"
# The level the command line starts at: the newest phase, whose modules are being matured.
LATEST_PHASE = "phase1"


def phase_modules(level: str) -> dict:
    """The module choices of a phase level; raises ``ValueError`` for an unknown level."""
    if level not in PHASES:
        raise ValueError(f"unknown phase level {level!r}; known: {', '.join(PHASES)}")
    return PHASES[level]


def load_misnames(path: str | Path) -> dict[str, str]:
    """The ``[misnames]`` table of a TOML file (category → reported name); raises ``ValueError``
    when it is missing or holds anything but non-empty names."""
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    table = data.get("misnames")
    if not isinstance(table, dict) or not all(isinstance(k, str) and isinstance(v, str) and k and v.strip()
                                              for k, v in table.items()):
        raise ValueError(f"{path}: [misnames] must map categories to non-empty names")
    return dict(table)


def phase_catalog() -> list[dict]:
    """Every level with its label and description, for the console."""
    return [{"level": k, "label": v["label"], "description": v["description"]} for k, v in PHASES.items()]
