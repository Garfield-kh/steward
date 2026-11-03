"""Assemble Phase-1 RGB-D frames with selectable mask / depth sources.

Ablation result (kitchen RGB-D): ``geo_sam3`` (geometric ∪ SAM3 on real RGB) gave the
best recall; NovelAnything / NVS / SLARM help only under strict depth-scale gates and are
not applied blindly to live frames.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import numpy as np
from PIL import Image

from steward.ego.providers.geometric_masks import GeometricMaskConfig, masks_from_depth, replace_frame_mask

MaskSource = Literal["oracle", "geometric", "sam3", "geo_sam3", "keep"]
DepthSource = Literal["sensor", "override"]

MASK_SOURCES = frozenset({"oracle", "geometric", "sam3", "geo_sam3", "keep"})


@dataclass
class FuseConfig:
    mask_source: MaskSource = "geo_sam3"
    geometric: GeometricMaskConfig | None = None
    # If set, multiply override depth by this scale (oracle calibration / plane fit).
    depth_scale: float = 1.0
    sam3_url: str | None = None
    sam3_threshold: float = 0.28
    # When True and SAM3 fails, fall back to geometric (geo_sam3/sam3) or keep (sam3).
    fallback_on_error: bool = True


def env_mask_source(default: str = "geo_sam3") -> str:
    """``STEWARD_MASK_SOURCE`` overrides the default: keep | geometric | sam3 | geo_sam3.

    Unset → ``default``. Invalid values are ignored.
    """
    raw = os.environ.get("STEWARD_MASK_SOURCE")
    if raw is None or not str(raw).strip():
        return default
    v = str(raw).strip().lower()
    return v if v in MASK_SOURCES else default


def resize_depth(depth: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    h, w = out_hw
    src_h, src_w = depth.shape
    ys = np.linspace(0, src_h - 1, h).round().astype(np.int32)
    xs = np.linspace(0, src_w - 1, w).round().astype(np.int32)
    return depth.astype(np.float32)[ys][:, xs]


def resize_rgb(rgb: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    h, w = out_hw
    return np.asarray(Image.fromarray(rgb).resize((w, h), Image.BILINEAR))


def _sam3_mask(rgb: np.ndarray, *, url: str | None, threshold: float, min_pixels: int) -> np.ndarray:
    from steward.ego.providers.sam3_client import Sam3Client, Sam3Config

    cfg = Sam3Config(threshold=threshold, min_pixels=min_pixels)
    if url:
        cfg.base_url = url
    env_url = os.environ.get("STEWARD_SAM3_URL")
    if env_url:
        cfg.base_url = env_url
    return Sam3Client(cfg).masks_from_rgb(rgb)


def prepare_rgbd_frame(
    frame: dict,
    *,
    mask_source: MaskSource = "geometric",
    depth_override: np.ndarray | None = None,
    rgb_override: np.ndarray | None = None,
    depth_scale: float = 1.0,
    geometric: GeometricMaskConfig | None = None,
    up_cam: np.ndarray | None = None,
    sam3_url: str | None = None,
    sam3_threshold: float = 0.28,
    fallback_on_error: bool = True,
) -> dict:
    """Build a frame dict for ``EgoEstimator.update_rgbd``.

    ``mask_source``:
      - ``oracle`` / ``keep``: leave ``frame['mask']`` unchanged
      - ``geometric``: rebuild mask from the (possibly overridden) depth
      - ``sam3``: SAM3 PCS on RGB; on failure / empty result keep sensor oracle mask
      - ``geo_sam3``: geometric ∪ SAM3; on failure keep sensor oracle mask
    """
    out = dict(frame)
    h, w = frame["depth"].shape
    oracle_mask = np.asarray(frame.get("mask", np.zeros((h, w), np.uint8)), dtype=np.uint8)
    if rgb_override is not None:
        out["rgb"] = resize_rgb(rgb_override, (h, w)) if rgb_override.shape[:2] != (h, w) else rgb_override
    if depth_override is not None:
        d = depth_override if depth_override.shape == (h, w) else resize_depth(depth_override, (h, w))
        out["depth"] = (d.astype(np.float32) * float(depth_scale))

    src = mask_source
    min_pix = int((geometric or GeometricMaskConfig()).min_pixels)
    if src == "geometric":
        out["mask"] = masks_from_depth(out["depth"], float(out["fovy"]), geometric, up_cam=up_cam)
    elif src in ("sam3", "geo_sam3"):
        try:
            sam = _sam3_mask(out["rgb"], url=sam3_url, threshold=sam3_threshold, min_pixels=min_pix)
            if int(np.asarray(sam).max()) <= 0:
                raise RuntimeError("SAM3 returned an empty instance mask")
            if src == "geo_sam3":
                from steward.ego.providers.sam3_client import merge_masks
                geo = masks_from_depth(out["depth"], float(out["fovy"]), geometric, up_cam=up_cam)
                out["mask"] = merge_masks(geo, sam)
            else:
                out["mask"] = np.asarray(sam, dtype=np.uint8)
        except Exception:
            if not fallback_on_error:
                raise
            # Prefer the world/sensor oracle mask already on the frame.
            out["mask"] = oracle_mask
    return out


def apply_geometric_masks(frames: list[dict], config: GeometricMaskConfig | None = None) -> list[dict]:
    return [replace_frame_mask(fr, config) for fr in frames]
