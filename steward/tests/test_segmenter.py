"""Robot-side SAM 3.1 segmentation (the ``phase1-sam`` level): instance merging, crops, cache,
semantic fusion and keyframe gating, against a stand-in for the service."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from steward.assets.generators import steward_bot
from steward.ego.classify import fuse_semantics
from steward.ego.runtime import EgoRuntime, RobotLink
from steward.ego.segmenter import Segmentation, SegmenterConfig, SegmenterError, Segmenter, load_config
from steward.world.server import StewardHost

VOCAB = {"table": "table", "bowl": "bowl", "plate": "plate", "wall": "wall"}


def _box(shape, y0, y1, x0, x1) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


class FakeClient:
    """Answers each request from ``reply(image, prompts)`` and records what was asked."""

    def __init__(self, reply):
        self.reply = reply
        self.calls: list[tuple[tuple[int, ...], list[str]]] = []

    def segment(self, png, prompts, threshold):
        from io import BytesIO

        from PIL import Image

        image = np.asarray(Image.open(BytesIO(png)))
        self.calls.append((image.shape, list(prompts)))
        return self.reply(image, prompts)


def test_detections_of_one_object_merge_and_a_small_object_keeps_the_pixels_it_shares():
    shape = (60, 80)
    table, bowl = _box(shape, 30, 60, 0, 80), _box(shape, 25, 35, 30, 40)

    def reply(image, prompts):
        return [("table", 0.9, table), ("bowl", 0.7, bowl), ("plate", 0.5, bowl), ("wall", 0.2, ~table)]

    seg = Segmenter(SegmenterConfig(vocabulary=VOCAB, crop_prompts=()), FakeClient(reply)).segment(
        np.zeros((*shape, 3), np.uint8))
    assert sorted(map(sorted, seg.labels.values())) == [["bowl", "plate"], ["table"]], "the 0.2 wall is below threshold"
    bowl_id = next(i for i, l in seg.labels.items() if "bowl" in l)
    assert seg.labels[bowl_id] == {"bowl": 0.7, "plate": 0.5}
    assert (seg.mask == bowl_id).sum() == bowl.sum(), "the bowl is painted over the table"


def test_support_surfaces_are_cropped_enlarged_and_their_small_objects_mapped_back():
    shape = (120, 160)
    table = _box(shape, 80, 120, 20, 140)

    def reply(image, prompts):
        if image.shape[:2] == shape:
            return [("table", 0.9, table)]
        h, w = image.shape[:2]  # the crop: a bowl in its middle
        return [("bowl", 0.8, _box((h, w), h // 3, 2 * h // 3, w // 3, w // 2))]

    client = FakeClient(reply)
    cfg = SegmenterConfig(vocabulary=VOCAB, crop_prompts=("bowl",), crop_target_px=480, crop_max_scale=4.0)
    seg = Segmenter(cfg, client).segment(np.zeros((*shape, 3), np.uint8))
    assert len(client.calls) == 2 and client.calls[1][1] == ["bowl"]
    assert client.calls[1][0][1] > 160 * 2, "the crop is enlarged"
    bowl_id = next(i for i, l in seg.labels.items() if "bowl" in l)
    ys, _ = np.nonzero(seg.mask == bowl_id)
    assert 68 <= ys.min() and ys.max() <= 100, "the bowl lands on the table's upper edge in the full frame"


def test_results_are_cached_by_image_and_settings(tmp_path):
    shape = (20, 20)
    client = FakeClient(lambda image, prompts: [("table", 0.9, _box(shape, 0, 20, 0, 20))])
    cfg = SegmenterConfig(vocabulary=VOCAB, crop_prompts=())
    rgb = np.zeros((*shape, 3), np.uint8)
    first = Segmenter(cfg, client, tmp_path).segment(rgb)
    again = Segmenter(cfg, client, tmp_path).segment(rgb)
    assert not first.cached and again.cached and len(client.calls) == 1
    assert again.labels == first.labels and np.array_equal(again.mask, first.mask)
    Segmenter(SegmenterConfig(vocabulary=VOCAB, crop_prompts=(), threshold=0.5), client, tmp_path).segment(rgb)
    assert len(client.calls) == 2, "other settings are another cache entry"


def test_a_service_failure_is_not_cached(tmp_path):
    def reply(image, prompts):
        raise SegmenterError("sam3: no reachable URL")

    with pytest.raises(SegmenterError):
        Segmenter(SegmenterConfig(vocabulary=VOCAB), FakeClient(reply), tmp_path).segment(np.zeros((8, 8, 3), np.uint8))
    assert not any(tmp_path.rglob("*.npz"))


def test_semantic_scores_lead_and_shape_only_breaks_ties():
    shape = {"bowl": 0.6, "mug": 0.3, "unknown": 0.1}
    fused = fuse_semantics({"mug": 0.8}, shape, weight=0.5, floor=0.03)
    assert max(fused, key=fused.get) == "mug"
    tie = fuse_semantics({"bowl": 0.6, "mug": 0.6}, shape, weight=0.5, floor=0.03)
    assert tie["bowl"] > tie["mug"]
    novel = fuse_semantics({"kettle": 0.7}, shape, weight=0.5, floor=0.03)
    assert max(novel, key=novel.get) == "kettle", "a category without a shape template is named by SAM"
    misfit = fuse_semantics({"teddy_bear": 0.7}, {**shape, "teddy_bear": 0.002}, weight=0.5, floor=0.03)
    assert max(misfit, key=misfit.get) == "teddy_bear", "a confident name outweighs a template that fits badly"
    vetoed = fuse_semantics({"bowl": 0.5}, {"bowl": 1e-5, "counter": 0.6, "unknown": 0.1}, weight=0.5, floor=0.03)
    assert max(vetoed, key=vetoed.get) == "counter", "a template that cannot fit (a two-metre bowl) overrules the name"


def test_the_shipped_settings_load_and_bad_settings_fail_loud(tmp_path):
    cfg = load_config()
    assert cfg.service == "sam3" and set(cfg.crop_prompts) <= set(cfg.vocabulary)
    bad = tmp_path / "s.toml"
    for text, message in [("[vocabulary]\nbowl='bowl'\n", "missing [segmenter]"),
                          ("[segmenter]\nthreshold='x'\n[vocabulary]\nbowl='bowl'\n", "segmenter.threshold"),
                          ("[segmenter]\ntresh=0.4\n[vocabulary]\nbowl='bowl'\n", "unknown segmenter keys"),
                          ("[segmenter]\ncrop_prompts=['cup']\n[vocabulary]\nbowl='bowl'\n", "not in [vocabulary]")]:
        bad.write_text(text)
        with pytest.raises(ValueError, match=message.replace("[", r"\[")):
            load_config(bad)


class StubSegmenter:
    """Names the largest regions of one flat colour a "thing" (a category without a shape
    template); ``fail`` makes it raise instead."""

    def __init__(self):
        self.config = SegmenterConfig(vocabulary=VOCAB)
        self.calls = 0
        self.fail = False

    def segment(self, rgb):
        self.calls += 1
        if self.fail:
            raise SegmenterError("sam3: no reachable URL")
        q = rgb.astype(np.int32) // 16
        key = (q[..., 0] << 8) | (q[..., 1] << 4) | q[..., 2]
        values, counts = np.unique(key, return_counts=True)
        mask = np.zeros(rgb.shape[:2], np.uint8)
        for i, value in enumerate(values[np.argsort(-counts)][:8], start=1):
            mask[key == value] = i
        return Segmentation(mask, {i: {"thing": 0.9} for i in range(1, int(mask.max()) + 1)}, seconds=0.01)


def _pump_until(host, done, timeout=120.0):
    deadline = time.time() + timeout
    while not done() and time.time() < deadline:
        host.pump(0.05)
    assert done(), "timed out"


def test_without_world_masks_the_robot_segments_keyframes_and_alerts_on_failure(kitchen_root, tmp_path: Path):
    host = StewardHost(kitchen_root, scenario="phase0-chores", speed=4.0, seed=3, run_dir=tmp_path).start()
    stub = StubSegmenter()
    runtime = EgoRuntime(RobotLink(host.robot_servers["w:robot-1"].url), steward_bot().to_dict(), segmenter=stub)
    runtime.start()
    try:
        host.set_phase("phase1-sam")
        assert "mask" not in host.sim.sensors("w:robot-1") and host.sim.sensors("w:robot-1")["masks"] is False
        _pump_until(host, lambda: runtime.perception_mode == "rgbd-segmenter" and stub.calls >= 2)
        assert runtime.segmentation["frames"] == stub.calls
        assert any(t.semantic_views for t in runtime.estimator.tracks.values())
        assert "segmentation" in runtime.summary()
        stub.fail = True
        _pump_until(host, lambda: runtime.segmentation["failures"] >= 2)
        assert sum("分割失败" in a["message"] for a in runtime.alerts) == 1, "one alert per run of failures"
    finally:
        runtime.stop()
        host.stop()
