"""Phase levels: which implementation each replaceable module runs at.

A later phase replaces modules; the implementations of earlier phases stay in the code base and can
be selected again at any time, also while an episode runs, to compare and debug. The Steward host
applies a level to the world side (what the sensors deliver); the robot's Ego Runtime follows by
inspecting each sensor frame, so the information boundary is unchanged.
"""

from __future__ import annotations

PHASES: dict[str, dict] = {
    "phase0": {
        "label": "Phase 0 · 理想感知",
        "description": "WSim 直接给出带噪声的检测结果（类别与包围盒）；ER 用卡尔曼融合",
        "perception": "ideal",
    },
    "phase1": {
        "label": "Phase 1 · RGB-D 感知链路",
        "description": "WSim 只给 RGB、深度与匿名实例掩码；ER 自行反投影、分类、多视角积累与形状补全。"
                       "ER 默认 mask_source=geo_sam3（几何∪SAM3，厨房消融召回最高）；"
                       "SAM3 失败或空结果则回退世界侧 oracle。可用 STEWARD_MASK_SOURCE 覆盖。"
                       "WSim 仍默认 oracle。NVS / SLARM / NovelAnything 仅作门控辅助。",
        "perception": "rgbd",
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


def phase_catalog() -> list[dict]:
    """Every level with its label and description, for the console."""
    return [{"level": k, "label": v["label"], "description": v["description"]} for k, v in PHASES.items()]
