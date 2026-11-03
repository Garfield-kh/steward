"""Geometric depth masks produce anonymous instances usable by segments()."""
from __future__ import annotations

import numpy as np

from steward.ego.providers.geometric_masks import GeometricMaskConfig, masks_from_depth
from steward.ego.perception import segments


def test_masks_from_depth_separates_blobs():
    h, w = 120, 160
    depth = np.full((h, w), 2.0, np.float32)
    # two nearer blobs
    depth[30:50, 40:60] = 1.2
    depth[30:50, 100:120] = 1.3
    mask = masks_from_depth(depth, fovy_deg=58.0, config=GeometricMaskConfig(min_pixels=20, residual_voxel_m=0.06))
    ids = [i for i in np.unique(mask) if i > 0]
    assert len(ids) >= 2
    # each blob should dominate one id
    a = mask[30:50, 40:60]
    b = mask[30:50, 100:120]
    assert a.max() > 0 and b.max() > 0
    assert int(np.bincount(a.ravel()).argmax()) != int(np.bincount(b.ravel()).argmax()) or a.max() != b.max()


def test_segments_accept_geometric_masks():
    h, w = 80, 100
    depth = np.full((h, w), 1.5, np.float32)
    depth[20:40, 30:50] = 1.0
    mask = masks_from_depth(depth, 58.0, GeometricMaskConfig(min_pixels=15, residual_voxel_m=0.05))
    rgb = np.full((h, w, 3), 180, np.uint8)
    frame = {"depth": depth, "mask": mask, "rgb": rgb, "fovy": 58.0}
    cam = {"p": [0.0, 0.0, 1.0], "q": [1.0, 0.0, 0.0, 0.0]}
    segs = segments(frame, cam)
    assert len(segs) >= 1
