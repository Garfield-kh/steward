"""Geometric depth masks produce anonymous instances usable by segments()."""
from __future__ import annotations

import numpy as np

from conftest import CAMERA_UP
from steward.ego.providers.geometric_masks import GeometricMaskConfig, masks_from_depth
from steward.ego.perception import segments

FLOOR = ((-1.0, -4.0, -0.1), (8.0, 4.0, 0.0))
TABLE = ((1.6, -0.6, 0.0), (2.4, 0.6, 0.75))
BOWL = ((1.9, -0.1, 0.75), (2.1, 0.1, 0.83))
STOOL = ((1.7, 1.2, 0.0), (2.1, 1.6, 0.45))


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


def test_only_what_stands_on_a_raised_surface_is_kept_when_the_rest_is_not_wanted(box_depth):
    depth, owner = box_depth((240, 320), 60.0, [FLOOR, TABLE, BOWL, STOOL])
    bowl, stool = owner == 2, owner == 3
    everything = masks_from_depth(depth, 60.0, up_cam=CAMERA_UP)
    assert everything[stool].max() > 0, "a stool on the floor is clustered as furniture"
    supported = masks_from_depth(depth, 60.0, up_cam=CAMERA_UP, supports_only=True)
    ids = np.unique(supported[supported > 0])
    assert len(ids) == 1, "the bowl on the table and nothing else"
    assert (supported[bowl] > 0).mean() > 0.8 and (supported[~bowl] > 0).sum() < 0.1 * bowl.sum()


def test_a_frame_of_a_million_points_finds_the_same_bowl_as_a_small_one(box_depth):
    """Plane hypotheses are scored on a sample of a large cloud; the bowl found is the same."""
    shares = []
    for shape in ((240, 320), (960, 1280)):
        depth, owner = box_depth(shape, 60.0, [FLOOR, TABLE, BOWL, STOOL])
        mask = masks_from_depth(depth, 60.0, up_cam=CAMERA_UP, supports_only=True)
        bowl = owner == 2
        assert len(np.unique(mask[mask > 0])) == 1
        shares.append(((mask[bowl] > 0).mean(), (mask[~bowl] > 0).sum() / bowl.sum()))
    assert all(inside > 0.8 and outside < 0.1 for inside, outside in shares), shares


def test_a_cabinet_top_and_the_walls_at_table_height_are_not_part_of_the_table_plane(box_depth):
    """A plane through the table top also passes within centimetres of a cabinet top across the
    room and of a band of each wall. Fitted to all of them it tilts, and the cabinet top then
    stands above it as an object. The plane is the table's own top, and only the cup stands on it."""
    back, side = ((4.0, -4.0, 0.0), (4.1, 4.0, 2.6)), ((-1.0, 2.2, 0.0), (8.0, 2.3, 2.6))
    cabinet, cup = ((2.9, 1.2, 0.0), (3.5, 1.8, 0.80)), ((1.95, -0.05, 0.75), (2.05, 0.05, 0.85))
    for shape in ((240, 320), (480, 640)):
        depth, owner = box_depth(shape, 60.0, [FLOOR, back, side, TABLE, cabinet, cup])
        for seed in range(4):
            noise = np.random.default_rng(seed).normal(0.0, 0.002, depth.shape)
            noisy = np.where(depth > 0, depth + noise, 0.0).astype(np.float32)
            mask = masks_from_depth(noisy, 60.0, up_cam=CAMERA_UP, supports_only=True)
            assert len(np.unique(mask[mask > 0])) == 1, (shape, seed)
            assert (mask[owner == 5] > 0).mean() > 0.8 and not (mask[owner != 5] > 0).any(), (shape, seed)


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
