"""SAM3 mask helpers and geo∪SAM3 fuse path (mocked HTTP)."""
from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from PIL import Image

from steward.ego.providers.fused_rgbd import prepare_rgbd_frame
from steward.ego.providers.geometric_masks import GeometricMaskConfig
from steward.ego.providers.sam3_client import merge_masks, results_to_mask
from steward.ego.perception import segments


def _png_b64(mask: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray((mask.astype(np.uint8) * 255)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_results_to_mask_and_merge():
    h, w = 40, 50
    a = np.zeros((h, w), bool)
    a[5:15, 5:15] = True
    b = np.zeros((h, w), bool)
    b[20:30, 20:35] = True
    resp = {"results": [{"masks": [_png_b64(a), _png_b64(b)]}]}
    sam = results_to_mask(resp, (h, w), min_pixels=10)
    assert len([i for i in np.unique(sam) if i > 0]) == 2

    geo = np.zeros((h, w), np.uint8)
    geo[8:18, 8:18] = 1  # overlaps first SAM blob
    merged = merge_masks(geo, sam)
    assert merged[10, 10] != 1  # SAM remapped id wins on overlap
    assert merged.max() > 1


def test_prepare_geo_sam3_with_stub_client(monkeypatch):
    h, w = 60, 80
    depth = np.full((h, w), 2.0, np.float32)
    depth[20:35, 25:40] = 1.1
    rgb = np.full((h, w, 3), 160, np.uint8)
    frame = {"rgb": rgb, "depth": depth, "fovy": 58.0, "mask": np.zeros((h, w), np.uint8)}

    def fake_sam3(rgb, *, url, threshold, min_pixels):
        m = np.zeros(rgb.shape[:2], np.uint8)
        m[10:22, 10:22] = 1
        return m

    monkeypatch.setattr("steward.ego.providers.fused_rgbd._sam3_mask", fake_sam3)
    out = prepare_rgbd_frame(
        frame,
        mask_source="geo_sam3",
        geometric=GeometricMaskConfig(min_pixels=15, residual_voxel_m=0.05),
    )
    ids = [i for i in np.unique(out["mask"]) if i > 0]
    assert len(ids) >= 2
    segs = segments(
        {"depth": depth, "mask": out["mask"], "rgb": rgb, "fovy": 58.0},
        {"p": [0.0, 0.0, 1.0], "q": [1.0, 0.0, 0.0, 0.0]},
    )
    assert len(segs) >= 1


def test_sam3_fallback_to_oracle(monkeypatch):
    h, w = 40, 40
    depth = np.full((h, w), 1.5, np.float32)
    depth[10:20, 10:20] = 1.0
    oracle = np.zeros((h, w), np.uint8)
    oracle[12:18, 12:18] = 7
    frame = {
        "rgb": np.zeros((h, w, 3), np.uint8),
        "depth": depth,
        "fovy": 58.0,
        "mask": oracle.copy(),
    }

    def boom(*_a, **_k):
        raise RuntimeError("sam3 down")

    monkeypatch.setattr("steward.ego.providers.fused_rgbd._sam3_mask", boom)
    out = prepare_rgbd_frame(frame, mask_source="sam3", fallback_on_error=True)
    assert np.array_equal(out["mask"], oracle)

    out2 = prepare_rgbd_frame(frame, mask_source="geo_sam3", fallback_on_error=True,
                              geometric=GeometricMaskConfig(min_pixels=10))
    assert np.array_equal(out2["mask"], oracle)

    with pytest.raises(RuntimeError):
        prepare_rgbd_frame(frame, mask_source="sam3", fallback_on_error=False)


def test_sam3_empty_mask_falls_back_to_oracle(monkeypatch):
    h, w = 32, 32
    oracle = np.zeros((h, w), np.uint8)
    oracle[2:8, 2:8] = 4
    frame = {
        "rgb": np.zeros((h, w, 3), np.uint8),
        "depth": np.full((h, w), 1.2, np.float32),
        "fovy": 58.0,
        "mask": oracle.copy(),
    }
    monkeypatch.setattr(
        "steward.ego.providers.fused_rgbd._sam3_mask",
        lambda *_a, **_k: np.zeros((h, w), np.uint8),
    )
    out = prepare_rgbd_frame(frame, mask_source="sam3", fallback_on_error=True)
    assert np.array_equal(out["mask"], oracle)
