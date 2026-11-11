"""Robot-side open-vocabulary instance segmentation with SAM 3.1 (the ``phase1-sam`` level).

At this level the world sends RGB and depth only; the Ego Runtime segments chosen frames itself.
The frames of one instant (one per camera) are segmented together in two rounds of concurrent
requests: every whole image with every prompt of the vocabulary, then the regions of support
surfaces (tables, counters) that are small in the image, each enlarged and asked again with the
small-object prompts (tiled into one image they are found far less often). A frame that was
reduced for perception has its crops cut from the pixels the camera took. Within a round the
prompts are divided so that every worker of the service has a request. A depth cluster above a
support that is still under ``min_pixels`` is cropped tight and enlarged the same way.

Detections of one object under several prompts are merged into one instance that carries every
label with its score; overlaps are resolved so that the smaller instance (a bowl) keeps the pixels
it shares with a larger one (the table under it). Regions SAM did not find, and that are large
enough, are then filled from depth (:mod:`steward.ego.providers.geometric_masks`) as unnamed
instances, which the shape priors alone classify.

Every result is cached on disk under a key of the image and the settings, so an episode replays
without the service and the same image is never sent twice (steward.md §13.7.4).
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import time
import tomllib
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from steward import perception_services
from steward.ego.providers.geometric_masks import (
    GeometricMaskConfig, _backproject, masks_from_depth, support_speck_boxes,
)

DEFAULT_PATH = Path(__file__).resolve().parents[3] / "config" / "segmentation.toml"


class SegmenterError(Exception):
    """A frame could not be segmented (service unreachable, rejected request, malformed reply)."""


@dataclass(frozen=True)
class SegmenterConfig:
    service: str = "sam3"
    threshold: float = 0.4
    merge_iou: float = 0.6
    min_pixels: int = 25
    crop_min_pixels: int = 8
    """Smallest mask kept from the enlarged crop. A cup three metres away is only a few dozen
    pixels after it is mapped back, below ``min_pixels``, and would otherwise be discarded."""
    crop_categories: tuple[str, ...] = ("table", "counter", "sideboard", "cabinet")
    crop_target_px: int = 480
    crop_max_scale: float = 4.0
    crop_up: float = 0.3
    crop_depth: float = 0.5
    drawer_crop_depth: float = 1.0
    """Share of a drawer or cabinet window kept below its top. The contents sit in the opening,
    not on the top edge, so the tabletop crop is too shallow for them."""
    max_crops: int = 2
    crop_prompts: tuple[str, ...] = ()
    drawer_prompts: tuple[str, ...] = ()
    """Extra crop prompts used only on a drawer's own window. They are not sent with the full image."""
    shape_weight: float = 0.5
    """Exponent of the shape-prior probability in :func:`steward.ego.classify.fuse_semantics`."""
    label_floor: float = 0.03
    """Semantic likelihood of a category SAM did not name."""
    vocabulary: dict[str, str] = field(default_factory=dict)
    """Prompt text → Ego-State category."""
    geometric_fill: bool = False
    """Add depth clusters SAM left unlabelled as unnamed instances (:meth:`Segmenter.fill`).
    The shipped ``segmentation.toml`` turns this on; a constructed config leaves it off."""
    fill_free_share: float = 0.7
    """Share of a depth cluster's pixels SAM must have left unlabelled for it to be added."""
    fill_supports_only: bool = True
    """When the fill is on, add only clusters standing on a support. Residual walls and furniture
    stay out: depth clustering otherwise boxes a corner into a piece of furniture."""

    fill_min_rise_m: float = 0.02
    """Height a cluster that lies wholly on a named support must span to be added: lower, it is
    a ripple of that surface and not an object standing on it. The pistol is flatter than 4 cm
    and still an object."""
    concurrency: int = 1
    """Requests to the service in flight at once: the number of workers it runs (``workers`` in
    its health reply). The prompts of a pass are divided over that many requests, which the
    images of one instant share; more requests than workers only wait in the service's queue."""

    def key(self) -> str:
        """Digest of every setting that changes a SAM result, part of each cache key. The fill
        settings are left out because the fill is computed after the cached result, and
        ``concurrency`` because a prompt is answered the same whichever request carries it."""
        later = ("geometric_fill", "fill_free_share", "fill_supports_only", "fill_min_rise_m", "concurrency")
        sam = {k: v for k, v in asdict(self).items() if k not in later}
        return hashlib.sha256(json.dumps(sam, sort_keys=True).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class DepthClusters:
    """What the depth of one image shows without its colours (:meth:`Segmenter.clusters`): the
    windows of the clusters too small for the fill, and the fill's own clusters. Neither depends
    on what the service answers, so both can be computed while it is asked."""

    specks: list[tuple[int, int, int, int]]
    """Windows (``y0, y1, x0, x1``) of the depth clusters above a support that are smaller than ``min_pixels``."""
    filled: np.ndarray | None
    """(H, W) ids of the depth clusters the fill may add (0 = none); None when the fill is off."""


@dataclass
class Segmentation:
    """One segmented frame: ``mask`` (H, W) uint8 instance ids (0 = none) and, per id, the score of
    each category it was detected as."""

    mask: np.ndarray
    labels: dict[int, dict[str, float]]
    cached: bool = False
    seconds: float = 0.0


def load_config(path: Path = DEFAULT_PATH) -> SegmenterConfig:
    """The segmenter settings in ``path``; raises ``ValueError`` naming the offending key."""
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"{path}: {exc}") from None
    raw, vocab = data.get("segmenter"), data.get("vocabulary")
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: missing [segmenter] table")
    if not isinstance(vocab, dict) or not vocab or not all(isinstance(v, str) and v for v in vocab.values()):
        raise ValueError(f"{path}: [vocabulary] must map prompt texts to category names")
    defaults = SegmenterConfig()
    values = {}
    for key, default in asdict(defaults).items():
        if key == "vocabulary" or key not in raw:
            continue
        value = raw[key]
        if isinstance(default, tuple):
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                raise ValueError(f"{path}: segmenter.{key} must be a list of strings")
            value = tuple(value)
        elif isinstance(default, (int, float)) and not isinstance(default, bool):
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"{path}: segmenter.{key} must be a non-negative number")
            value = type(default)(value)
        elif not isinstance(value, type(default)):
            raise ValueError(f"{path}: segmenter.{key} must be a {type(default).__name__}")
        values[key] = value
    unknown = set(raw) - set(asdict(defaults))
    if unknown:
        raise ValueError(f"{path}: unknown segmenter keys {sorted(unknown)}")
    config = SegmenterConfig(**values, vocabulary=dict(vocab))
    missing = [p for p in (*config.crop_prompts, *config.drawer_prompts) if p not in config.vocabulary]
    if missing:
        raise ValueError(f"{path}: segmenter crop prompt not in [vocabulary]: {missing}")
    if config.concurrency < 1:
        raise ValueError(f"{path}: segmenter.concurrency must be at least 1")
    return config


class Sam3Client:
    """``POST /api/segment`` of a SAM 3.1 service: one image, several text prompts."""

    def __init__(self, service: perception_services.Service):
        self.service = service
        self._base: str = service.urls[0]

    def segment(self, png: bytes, prompts: list[str], threshold: float) -> list[tuple[str, float, np.ndarray]]:
        """Every detection as ``(prompt, score, mask)``, the mask a boolean array of the image's size.

        A connection failure (the blocked domain resets it or answers with a block page) moves on to
        the service's next base URL, starting from the one that answered last; an error status from
        the service itself is not retried."""
        boundary = uuid.uuid4().hex
        options = json.dumps({"prompts": [{"text": p} for p in prompts], "threshold": threshold})
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"frame.png\"\r\n"
                f"Content-Type: image/png\r\n\r\n").encode() + png + \
               (f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"options\"\r\n\r\n{options}\r\n"
                f"--{boundary}--\r\n").encode()
        urls = self.service.urls
        order = [self._base, *(u for u in urls if u != self._base)]
        failures = []
        for base in order:
            method, url = self.service.url(base, "segment")
            request = urllib.request.Request(url, data=body, method=method,
                                             headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
            try:
                with urllib.request.urlopen(request, timeout=self.service.timeout_s) as resp:
                    raw, kind = resp.read(), resp.headers.get("Content-Type", "")
            except urllib.error.HTTPError as exc:
                if exc.headers.get("Content-Type", "").startswith("application/json"):
                    raise SegmenterError(f"{self.service.name}: HTTP {exc.code}: {exc.read()[:300]!r}") from None
                failures.append(f"{base}: HTTP {exc.code} (not the service)")
                continue
            except (urllib.error.URLError, OSError) as exc:
                failures.append(f"{base}: {getattr(exc, 'reason', exc)}")
                continue
            if not kind.startswith("application/json"):
                failures.append(f"{base}: {kind or 'no content type'} reply (not the service)")
                continue
            self._base = base
            break
        else:
            raise SegmenterError(f"{self.service.name}: no reachable URL ({'; '.join(failures)})")
        try:
            reply = json.loads(raw)
        except ValueError as exc:
            raise SegmenterError(f"{self.service.name}: malformed reply ({exc})") from None
        try:
            return [(r["prompt"]["text"], float(score), _decode_mask(png_b64))
                    for r in reply["results"] for score, png_b64 in zip(r["scores"], r["masks"], strict=True)]
        except (KeyError, TypeError, ValueError) as exc:
            raise SegmenterError(f"{self.service.name}: malformed reply ({type(exc).__name__}: {exc})") from None


def _decode_mask(data: str) -> np.ndarray:
    if data.startswith("data:"):
        data = data.split(",", 1)[1]
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(data))).convert("L")) > 127


def _png(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


def _merge_boxes(boxes: list[tuple[int, int, int, int]], inside: float = 0.7) -> list[tuple[int, int, int, int]]:
    """Union ``(y0, y1, x0, x1)`` boxes of which one lies mostly (``inside`` of its area) in the
    other, largest area first. Merely overlapping neighbours stay apart: a union would be enlarged
    less and its small objects missed."""
    merged = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                a, b = merged[i], merged[j]
                common = max(0, min(a[1], b[1]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[2], b[2]))
                smaller = min((a[1] - a[0]) * (a[3] - a[2]), (b[1] - b[0]) * (b[3] - b[2]))
                if smaller and common / smaller >= inside:
                    merged[i] = [min(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3])]
                    del merged[j]
                    changed = True
                    break
            if changed:
                break
    return sorted((tuple(b) for b in merged), key=lambda b: -(b[1] - b[0]) * (b[3] - b[2]))


_SUPPORTS = {"table", "counter", "sideboard", "cabinet"}


def _support_ids(labels: dict[int, dict[str, float]]) -> set[int]:
    """Instances named only as a support surface. A bowl that shares the counter's pixels is not one."""
    return {i for i, lab in labels.items() if lab and set(lab).issubset(_SUPPORTS)}


def _open_count(mask: np.ndarray, box: tuple[int, int, int, int], supports: set[int]) -> int:
    """Pixels of ``box`` that are empty or belong only to a support. A bowl on a counter is covered
    by the counter's mask, and that is still the bowl."""
    y0, y1, x0, x1 = box
    window = mask[y0:y1, x0:x1]
    open_ = (window == 0) | np.isin(window, list(supports))
    return int(open_.sum())


def _crop_background(mask: np.ndarray) -> bool:
    """Whether a crop mask paints most of its window: background mixed in, not an object on the surface."""
    return mask.size > 0 and int(mask.sum()) >= 0.6 * mask.size


def _furniture_slice(category: str, mask: np.ndarray, parents: list[tuple[str, np.ndarray]]) -> bool:
    """Whether a ``box`` detection is the upper body of a cabinet, sideboard or counter.

    Most of its pixels lie on that furniture and it covers at least a quarter of the furniture.
    A box standing on the top is much smaller than the body, so it stays.

    :param category: the detection's category name.
    :param mask: the detection in the full image.
    :param parents: support surfaces from the full-image pass, each ``(category, mask)``.
    """
    if category != "box":
        return False
    area = int(mask.sum())
    if area < 1:
        return False
    for parent_cat, parent in parents:
        if parent_cat not in ("cabinet", "sideboard", "counter"):
            continue
        parent_area = int(parent.sum())
        if parent_area < 1:
            continue
        inside = int(np.logical_and(mask, parent).sum())
        if inside >= 0.7 * area and area >= 0.25 * parent_area:
            return True
    return False


# A bottle in an open drawer is farther than the front panel and a different colour. The handle
# sits on the panel, so it is not farther. A component outside this size is the panel itself.
_PEEL_FROM = ("drawer",)
_PEEL_COLOR = 40.0
_PEEL_FARTHER_M = 0.04
_PEEL_MIN_PX = 12
_PEEL_MAX_PX = 400
_PEEL_GAP_PX = 12
_PEEL_BLACK = 30.0
"""Brightest channel of a black object lying in the drawer, in 0-255. The shadow in the gap
beside the drawer is darker than the wood but not black."""
_PEEL_SAME_HUE = 0.03
"""Chromaticity distance under which a brighter patch is a lit face of the panel's own wood."""
_PEEL_LINE_M = 0.02
"""Second extent under which a part's points form a line rather than a face."""


def _line_like(depth: np.ndarray, fovy_deg: float, part: np.ndarray) -> bool:
    """Whether the points of ``part`` form a line: their second principal extent is under
    :data:`_PEEL_LINE_M`. An object's face has two extents; a sliver in a gap has one."""
    pts, _, _ = _backproject(depth, fovy_deg, part)
    if len(pts) < 3:
        return True
    centred = pts - pts.mean(axis=0)
    axes = np.linalg.svd(centred, full_matrices=False)[2]
    along = centred @ axes.T
    extents = np.sort(np.percentile(along, 98, axis=0) - np.percentile(along, 2, axis=0))[::-1]
    return float(extents[1]) < _PEEL_LINE_M


def _components(mask: np.ndarray, rgb: np.ndarray | None = None, tol: float = 35.0) -> tuple[np.ndarray, int]:
    """4-connected components of a boolean mask, numbered from 1.

    With ``rgb``, a pixel joins a component only when its colour is within ``tol`` of that
    component's first pixel. A blue cabinet and a pink bottle are both unlike the drawer and
    would otherwise be one component.
    """
    h, w = mask.shape
    out = np.zeros((h, w), dtype=np.int32)
    n = 0
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
        if out[y, x]:
            continue
        n += 1
        seed = None if rgb is None else rgb[y, x].astype(np.float32)
        stack = [(y, x)]
        out[y, x] = n
        while stack:
            cy, cx = stack.pop()
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if not (0 <= ny < h and 0 <= nx < w and mask[ny, nx] and out[ny, nx] == 0):
                    continue
                if seed is not None and float(np.linalg.norm(rgb[ny, nx].astype(np.float32) - seed)) > tol:
                    continue
                out[ny, nx] = n
                stack.append((ny, nx))
    return out, n


def _merge_near(parts: list[np.ndarray], gap: int, rgb: np.ndarray, tol: float = 50.0) -> list[np.ndarray]:
    """Union components whose boxes come within ``gap`` pixels and whose colours match.

    One object's wings are separate pixels. A differently coloured bottle beside them is not
    part of that object, and joining every nearby speck would swallow the bottle.
    """
    boxes = []
    for part in parts:
        ys, xs = np.nonzero(part)
        boxes.append((int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max()),
                      rgb[part].mean(axis=0).astype(np.float32), part))
    parent = list(range(len(boxes)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, a in enumerate(boxes):
        for j in range(i + 1, len(boxes)):
            b = boxes[j]
            if float(np.linalg.norm(a[4] - b[4])) > tol:
                continue
            if a[2] - gap <= b[3] and b[2] - gap <= a[3] and a[0] - gap <= b[1] and b[0] - gap <= a[1]:
                parent[find(j)] = find(i)
    groups: dict[int, np.ndarray] = {}
    for i, box in enumerate(boxes):
        root = find(i)
        groups[root] = box[5] if root not in groups else np.logical_or(groups[root], box[5])
    return list(groups.values())


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


class Segmenter:
    """Segments RGB frames with a SAM 3.1 client, caching every result under ``cache_dir``."""

    def __init__(self, config: SegmenterConfig, client, cache_dir: Path | None = None):
        self.config = config
        self.client = client
        self.cache_dir = cache_dir
        self._key = config.key()

    @classmethod
    def from_files(cls, config_path: Path = DEFAULT_PATH, services_path: Path = perception_services.DEFAULT_PATH,
                   cache_dir: Path | None = None) -> Segmenter:
        """The segmenter configured in ``config_path`` with its service from ``services_path``."""
        config = load_config(config_path)
        services = perception_services.load(services_path)
        if config.service not in services:
            raise ValueError(f"{config_path}: segmenter.service {config.service!r} is not in {services_path}")
        return cls(config, Sam3Client(services[config.service]), cache_dir)

    def segment(self, rgb: np.ndarray, detail: np.ndarray | None = None) -> Segmentation:
        """Instances of one RGB image; see :meth:`segment_frames`."""
        return self.segment_frames([(rgb, detail)])[0]

    def segment_frames(self, frames: list[tuple[np.ndarray, np.ndarray | None]]) -> list[Segmentation]:
        """Instances of the RGB images of one instant, asked of the service together.

        Every image is first asked with the whole vocabulary (less the drawer prompts), then its
        support surfaces are asked again with the small-object prompts. Each of the two rounds is
        one batch of concurrent requests (``SegmenterConfig.concurrency``), so two cameras cost
        little more than one.

        :param frames: ``(rgb, detail)`` per image: ``rgb`` (H, W, 3) uint8 is the image the masks
            are for; ``detail`` is the same view at a whole multiple of its size, from which the
            enlarged crops take real pixels, or None to enlarge ``rgb`` itself.
        :returns: one :class:`Segmentation` per image, in order.
        :raises SegmenterError: the service failed for any image; nothing is cached then."""
        cfg = self.config
        started = time.monotonic()
        out: list[Segmentation | None] = []
        paths: list[Path | None] = []
        for rgb, detail in frames:
            digest = hashlib.sha256(rgb.tobytes() + str(rgb.shape).encode() + self._key.encode())
            if detail is not None:
                digest.update(detail.tobytes() + str(detail.shape).encode())
            name = digest.hexdigest()
            path = None if self.cache_dir is None else self.cache_dir / name[:2] / f"{name}.npz"
            paths.append(path)
            if path is not None and path.exists():
                with np.load(path) as z:
                    labels = {int(k): v for k, v in json.loads(str(z["labels"])).items()}
                    out.append(Segmentation(z["mask"], labels, cached=True))
            else:
                out.append(None)
        todo = [i for i, seg in enumerate(out) if seg is None]
        if not todo:
            return out
        whole = [prompt for prompt in cfg.vocabulary if prompt not in cfg.drawer_prompts]
        found = self._ask([(frames[i][0], whole) for i in todo])
        plans = []
        for i, dets in zip(todo, found):
            rgb, detail = frames[i]
            parents = [(category, mask) for category, _score, mask, _from_crop in dets]
            tiles = [(box, self._window_tile(rgb, detail, box, parents)) for box in self._crop_boxes(rgb.shape[:2], dets)]
            plans.append([(box, tile) for box, tile in tiles if tile is not None])
        answers = iter(self._ask([tile for plan in plans for _box, tile in plan]))
        for i, dets, plan in zip(todo, found, plans):
            shape = frames[i][0].shape[:2]
            parents = [(category, mask) for category, _score, mask, _from_crop in dets]
            for box, _tile in plan:
                dets = dets + self._from_crop(next(answers), box, shape, parents)
            mask, labels = self._instances(dets, shape)
            if paths[i] is not None:
                paths[i].parent.mkdir(parents=True, exist_ok=True)
                tmp = paths[i].with_suffix(".tmp.npz")
                np.savez_compressed(tmp, mask=mask, labels=np.array(json.dumps(labels)))
                tmp.replace(paths[i])
            out[i] = Segmentation(mask, labels)
        seconds = time.monotonic() - started
        for i in todo:
            out[i].seconds = seconds  # the images were asked together: each waited for the whole batch
        return out

    def clusters(self, depth: np.ndarray, fovy_deg: float, up_cam: np.ndarray) -> DepthClusters:
        """The depth clusters :meth:`zoom_frames` and :meth:`fill` work from. It reads only the
        depth, so a caller with several images runs it beside :meth:`segment_frames` and hands the
        result to both; they compute it themselves when it is not given.

        :param depth: (H, W) metres of the segmented image, with the robot's own body cleared.
        :param up_cam: the world's up direction in the camera frame (x forward, z up)."""
        cfg = self.config
        specks = []
        if cfg.crop_prompts and cfg.crop_target_px > 0:
            specks = support_speck_boxes(depth, fovy_deg, up_cam, min_pixels=cfg.crop_min_pixels, max_pixels=cfg.min_pixels)
        filled = None
        if cfg.geometric_fill:
            filled = masks_from_depth(depth, fovy_deg, GeometricMaskConfig(min_pixels=cfg.min_pixels), up_cam=up_cam,
                                      supports_only=cfg.fill_supports_only)
        return DepthClusters(specks, filled)

    def fill(self, seg: Segmentation, depth: np.ndarray, fovy_deg: float, up_cam: np.ndarray,
             clusters: DepthClusters | None = None) -> Segmentation:
        """``seg`` with every depth cluster of which at least ``fill_free_share`` of the pixels SAM
        left unlabelled — a pixel named only as a support counts as unlabelled — added, on those
        pixels, as an instance without labels.

        :param depth: (H, W) metres of the segmented image.
        :param up_cam: the world's up direction in the camera frame (x forward, z up).
        :param clusters: :meth:`clusters` of the same depth, when the caller already has them.
        :returns: ``seg`` itself when the fill is off."""
        cfg = self.config
        if not cfg.geometric_fill:
            return seg
        geo = (clusters or self.clusters(depth, fovy_deg, up_cam)).filled
        mask, labels = seg.mask.copy(), dict(seg.labels)
        supports = _support_ids(labels)
        free = (mask == 0) | np.isin(mask, list(supports))
        next_id = int(mask.max()) + 1
        for instance in np.unique(geo):
            if instance == 0 or next_id > 255:
                continue
            region = geo == instance
            own = region & free
            if int(own.sum()) < cfg.min_pixels or own.sum() < cfg.fill_free_share * region.sum():
                continue
            # A cluster that lies entirely on a support is the object standing on it only when it
            # rises clear of that surface. A two-centimetre smear of the tabletop is not a plate.
            # The pistol is flatter than 4 cm and still an object.
            if int((own & (mask == 0)).sum()) == 0:
                pts, _, _ = _backproject(depth, fovy_deg, own)
                up = np.asarray(up_cam, dtype=np.float64)
                up = up / (np.linalg.norm(up) + 1e-12)
                if len(pts) == 0 or float((pts @ up).max() - (pts @ up).min()) < cfg.fill_min_rise_m:
                    continue
            mask[own] = next_id
            next_id += 1
        return Segmentation(mask, labels, cached=seg.cached, seconds=seg.seconds)

    def peel(self, seg: Segmentation, rgb: np.ndarray, depth: np.ndarray,
             fovy_deg: float | None = None) -> Segmentation:
        """Split a small object out of a drawer when SAM named the opening and the object as one drawer.

        The object is a different colour from the furniture and sits farther from the camera than
        the front panel. Those pixels become their own instance, with no label. A handle on the
        panel is at the panel's depth and stays. Pixels already named as their own object stay.

        :param rgb: (H, W, 3) uint8 of the segmented image.
        :param depth: (H, W) metres.
        :param fovy_deg: vertical field of view; with it, a part whose points form a line
            narrower than :data:`_PEEL_LINE_M` (the shadow in the gap beside a pulled-out drawer)
            is not peeled.
        :returns: ``seg`` itself when nothing peels off.
        """
        mask = seg.mask
        labels = seg.labels
        if mask.size == 0 or not labels:
            return seg
        h, w = mask.shape
        rgb_f = rgb.astype(np.float32)
        depth_f = np.asarray(depth, dtype=np.float32)
        valid_d = np.isfinite(depth_f) & (depth_f > 0.05)
        out = mask.copy()
        next_id = int(out.max()) + 1
        changed = False
        for i, lab in labels.items():
            if next_id > 255 or not lab or set(lab).isdisjoint(_PEEL_FROM):
                continue
            region = mask == i
            if int(region.sum()) < 20 or not bool((region & valid_d).any()):
                continue
            ys, xs = np.nonzero(region)
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            height = max(1, y1 - y0)
            wy0, wy1 = max(0, y0 - height), min(h, y1)
            wx0, wx1 = max(0, x0 - 4), min(w, x1 + 4)
            window = np.zeros((h, w), dtype=bool)
            window[wy0:wy1, wx0:wx1] = True
            pool = window & ((out == i) | (out == 0))
            med = np.median(rgb_f[region], axis=0)
            # The front panel is the near part of the mask. The cavity, and a bottle SAM painted
            # as the drawer, are farther or simply a different colour. The median is the cavity
            # once the opening fills the mask, so it cannot be the reference.
            front = depth_f[region & valid_d]
            dfront = float(np.percentile(front, 15))
            dist = np.linalg.norm(rgb_f - med, axis=2)
            colored = dist > _PEEL_COLOR
            inside = pool & (out == i) & colored
            outside = pool & (out == 0) & colored & valid_d & (depth_f > dfront + _PEEL_FARTHER_M)
            odd = inside | outside
            _labeled, n = _components(odd, rgb)
            parts = []
            wood = med / max(float(med.sum()), 1.0)
            for k in range(1, n + 1):
                part = _labeled == k
                count = int(part.sum())
                if not (_PEEL_MIN_PX <= count <= _PEEL_MAX_PX):
                    continue
                # Wood and shadow differ from the front panel too. A content is coloured, a
                # small white bottle, or a black object lying in the cavity. A tan smear is not,
                # nor is a lit face of the same wood: brighter, with the panel's own hue.
                mean = rgb[part].mean(axis=0)
                white = float(mean.min()) > 170.0 and count <= 80
                in_cavity = part & valid_d & (depth_f > dfront + _PEEL_FARTHER_M)
                black = float(mean.max()) < _PEEL_BLACK and int(in_cavity.sum()) >= count // 2
                if float(mean.max() - mean.min()) < 70.0 and not (white or black):
                    continue
                hue = mean / max(float(mean.sum()), 1.0)
                if not (white or black) and float(np.linalg.norm(hue - wood)) < _PEEL_SAME_HUE:
                    continue
                if fovy_deg is not None and _line_like(depth_f, fovy_deg, part & valid_d):
                    continue
                parts.append(part)
            for part in _merge_near(parts, _PEEL_GAP_PX, rgb, tol=60.0):
                if int(part.sum()) > _PEEL_MAX_PX or next_id > 255:
                    continue
                out[part] = next_id
                next_id += 1
                changed = True
        if not changed:
            return seg
        return Segmentation(out, labels, cached=seg.cached, seconds=seg.seconds)

    def zoom(self, seg: Segmentation, rgb: np.ndarray, depth: np.ndarray, fovy_deg: float,
             up_cam: np.ndarray, detail: np.ndarray | None = None) -> Segmentation:
        """``seg`` plus the small objects of one image; see :meth:`zoom_frames`."""
        return self.zoom_frames([(seg, rgb, depth, fovy_deg, up_cam, detail)])[0]

    def zoom_frames(self, frames: list[tuple[Segmentation, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray | None]],
                    clusters: list[DepthClusters] | None = None) -> list[Segmentation]:
        """Each image's segmentation plus any small object SAM names inside a tight crop of a
        sub-threshold depth cluster, the crops of all the images asked of the service together.

        Geometric fill ignores a cluster under ``min_pixels``. The window is only that cluster, so
        the enlargement makes a few dozen original pixels a large part of the request. The crop
        result is cached under the image, the configuration and the windows asked, so a replay does
        not call the service. The crops of the images of one instant are one batch of concurrent
        requests (``SegmenterConfig.concurrency``): two cameras wait once, not one after the other.

        :param frames: ``(seg, rgb, depth, fovy_deg, up_cam, detail)`` per image: its segmentation,
            the (H, W, 3) uint8 image, its (H, W) depth in metres with the robot's own body already
            cleared, the vertical field of view, the world's up direction in the camera frame (x
            forward, z up), and the same view at a whole multiple of ``rgb``'s size to enlarge
            from, or None.
        :param clusters: :meth:`clusters` of each image's depth, in order, when the caller already has them.
        :returns: one :class:`Segmentation` per image, in order; an image with nothing small
            enough to ask about keeps its own.
        :raises SegmenterError: the service failed; nothing from this call is cached then.
        """
        cfg = self.config
        out = [seg for seg, *_rest in frames]
        if not cfg.crop_prompts or cfg.crop_target_px <= 0:
            return out
        asked: list[tuple[int, Path | None, list[tuple[str, np.ndarray]], list[tuple[tuple[int, int, int, int], tuple]]]] = []
        for i, (seg, rgb, depth, fovy_deg, up_cam, detail) in enumerate(frames):
            boxes = (clusters[i] if clusters is not None else self.clusters(depth, fovy_deg, up_cam)).specks
            supports = _support_ids(seg.labels)
            boxes = [box for box in boxes if _open_count(seg.mask, box, supports) >= cfg.crop_min_pixels][:cfg.max_crops]
            if not boxes:
                continue
            digest = hashlib.sha256(rgb.tobytes() + str(rgb.shape).encode() + self._key.encode()
                                    + b"zoom-support" + json.dumps(boxes).encode()).hexdigest()
            path = None if self.cache_dir is None else self.cache_dir / digest[:2] / f"{digest}.npz"
            if path is not None and path.exists():
                with np.load(path) as z:
                    labels = {int(k): v for k, v in json.loads(str(z["labels"])).items()}
                    out[i] = Segmentation(z["mask"], labels, cached=True, seconds=seg.seconds)
                continue
            parents = [(cat, seg.mask == k) for k, labels in seg.labels.items() for cat in labels]
            tiles = [(box, self._window_tile(rgb, detail, box, parents)) for box in boxes]
            asked.append((i, path, parents, [(box, tile) for box, tile in tiles if tile is not None]))
        started = time.monotonic()
        answers = iter(self._ask([tile for _i, _path, _parents, plan in asked for _box, tile in plan]))
        for i, path, parents, plan in asked:
            seg = frames[i][0]
            found: list[tuple[str, float, np.ndarray, bool]] = []
            for box, _tile in plan:
                found.extend(self._from_crop(next(answers), box, seg.mask.shape, parents))
            if found:
                seg = self._paint(seg, found)
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".tmp.npz")
                np.savez_compressed(tmp, mask=seg.mask, labels=np.array(json.dumps(seg.labels)))
                tmp.replace(path)
            out[i] = Segmentation(seg.mask, seg.labels, cached=seg.cached, seconds=seg.seconds)
        waited = time.monotonic() - started
        for i, _path, _parents, _plan in asked:
            out[i].seconds += waited  # the images were asked together: each waited for the whole batch
        return out

    def _window_tile(self, rgb: np.ndarray, detail: np.ndarray | None, box: tuple[int, int, int, int],
                     parents: list[tuple[str, np.ndarray]]) -> tuple[np.ndarray, list[str]] | None:
        """The image and prompts to ask one window (``y0, y1, x0, x1`` of ``rgb``) with, or None.

        A window shorter than ``crop_target_px`` is enlarged to it, at most ``crop_max_scale``
        times. A longer window is skipped, except one that overlaps a drawer or a sideboard: that
        window is asked at its own size, and it gets the drawer prompts. The pistol sits in the
        sideboard even when that face was not named a drawer. With ``detail`` the tile is cut from
        the pixels the camera took, so the enlargement shows what the camera saw and not an
        interpolation of ``rgb``.
        """
        cfg = self.config
        y0, y1, x0, x1 = box
        scale = min(cfg.crop_max_scale, cfg.crop_target_px / max(y1 - y0, x1 - x0, 1))
        overlaps = any(bool(mask[y0:y1, x0:x1].any()) for category, mask in parents if category in ("drawer", "sideboard"))
        if scale <= 1.0 and not overlaps:
            return None
        scale = max(scale, 1.0)
        step = 1 if detail is None else detail.shape[0] // rgb.shape[0]
        window = (rgb if detail is None else detail)[y0 * step:y1 * step, x0 * step:x1 * step]
        size = (max(1, round((x1 - x0) * scale)), max(1, round((y1 - y0) * scale)))
        tile = np.asarray(Image.fromarray(np.ascontiguousarray(window)).resize(size, Image.BICUBIC))
        prompts = list(cfg.crop_prompts)
        if overlaps:
            prompts.extend(prompt for prompt in cfg.drawer_prompts if prompt not in prompts)
        return tile, prompts

    def _from_crop(self, detections: list[tuple[str, float, np.ndarray, bool]], box: tuple[int, int, int, int],
                   shape: tuple[int, int], parents: list[tuple[str, np.ndarray]],
                   ) -> list[tuple[str, float, np.ndarray, bool]]:
        """A tile's detections as masks of the whole image, without those that are the tile's
        background or the body of the furniture it was cut from."""
        y0, y1, x0, x1 = box
        out = []
        for category, score, mask, _from_crop in detections:
            small = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((x1 - x0, y1 - y0), Image.NEAREST)) > 127
            if _crop_background(small):
                continue  # the enlarged window's background came back as part of the mask
            full = np.zeros(shape, dtype=bool)
            full[y0:y1, x0:x1] = small
            if _furniture_slice(category, full, parents):
                continue  # the upper body of the cabinet, not a box standing on it
            if full.any():
                out.append((category, score, full, True))
        return out

    def _paint(self, seg: Segmentation, found: list[tuple[str, float, np.ndarray, bool]]) -> Segmentation:
        """Paint ``found`` onto pixels ``seg`` left empty. A mask that lands on something already
        named keeps only its free pixels, and is dropped when fewer than ``crop_min_pixels`` remain."""
        extra, labels = self._instances(found, seg.mask.shape)
        mask, out = seg.mask.copy(), dict(seg.labels)
        supports = _support_ids(out)
        for i, lab in labels.items():
            if set(lab).issubset(_SUPPORTS):
                continue
            region = (extra == i) & ((mask == 0) | np.isin(mask, list(supports)))
            if int(region.sum()) < self.config.crop_min_pixels or (max(out) if out else 0) >= 255:
                continue
            new_id = (max(out) if out else 0) + 1
            mask[region] = new_id
            out[new_id] = lab
        return Segmentation(mask, out, cached=seg.cached, seconds=seg.seconds)

    # ------------------------------------------------------------------ passes
    def _ask(self, jobs: list[tuple[np.ndarray, list[str]]]) -> list[list[tuple[str, float, np.ndarray, bool]]]:
        """Detections ``(category, score, mask, from_crop=False)`` of each ``(image, prompts)`` job.

        The jobs go out together. While fewer requests than ``concurrency`` are planned, the job
        whose requests carry the most prompts is divided once more, so the service's workers are
        all busy and none is asked for more than its share. Detections come back in the order of
        the job's prompts, whichever request carried them."""
        cfg = self.config
        shares = [1 if prompts else 0 for _image, prompts in jobs]
        while 0 < sum(shares) < cfg.concurrency:
            widest = max(range(len(jobs)), key=lambda j: len(jobs[j][1]) / shares[j] if shares[j] else 0.0)
            if len(jobs[widest][1]) <= shares[widest]:
                break
            shares[widest] += 1
        requests = [(j, _png(image), prompts[k::share])
                    for j, ((image, prompts), share) in enumerate(zip(jobs, shares)) for k in range(share)]
        if len(requests) <= 1:
            replies = [self.client.segment(png, prompts, cfg.threshold) for _j, png, prompts in requests]
        else:
            with ThreadPoolExecutor(max_workers=cfg.concurrency, thread_name_prefix="segment") as pool:
                replies = list(pool.map(lambda r: self.client.segment(r[1], r[2], cfg.threshold), requests))
        out: list[list[tuple[str, float, np.ndarray, bool]]] = [[] for _ in jobs]
        for (j, _png_bytes, _prompts), reply in zip(requests, replies):
            shape = jobs[j][0].shape[:2]
            out[j] += [(prompt, score, mask, False) for prompt, score, mask in reply
                       if score >= cfg.threshold and prompt in cfg.vocabulary and mask.shape == shape]
        for j, (_image, prompts) in enumerate(jobs):
            order = {prompt: n for n, prompt in enumerate(prompts)}
            out[j] = [(cfg.vocabulary[prompt], score, mask, from_crop)
                      for prompt, score, mask, from_crop in sorted(out[j], key=lambda d: order.get(d[0], len(order)))]
        return out

    def _crop_boxes(self, shape: tuple[int, int], found: list[tuple[str, float, np.ndarray, bool]],
                    ) -> list[tuple[int, int, int, int]]:
        """The windows (``y0, y1, x0, x1``) of the support surfaces among ``found`` to ask again,
        at most ``max_crops``, largest first."""
        cfg = self.config
        if not cfg.crop_prompts or cfg.crop_target_px <= 0:
            return []
        h, w = shape
        boxes = []
        for category, _score, mask, _from_crop in found:
            if category not in cfg.crop_categories:
                continue
            ys, xs = np.nonzero(mask)
            y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
            height = y1 - y0
            # A table's objects stand on its top. A drawer's bottles stand on the floor of the
            # opening, which is the lower part of the mask.
            depth = cfg.drawer_crop_depth if category in ("drawer", "cabinet") else cfg.crop_depth
            boxes.append((max(0, y0 - int(cfg.crop_up * height)), min(h, y0 + max(1, int(depth * height))),
                          max(0, x0 - 4), min(w, x1 + 4)))
        return _merge_boxes(boxes)[:cfg.max_crops]

    def _instances(self, detections, shape) -> tuple[np.ndarray, dict[int, dict[str, float]]]:
        """Merge detections of one object and paint instances largest first, so smaller ones keep
        the pixels they share with what they stand on.

        A detection found only in an enlarged crop is kept down to ``crop_min_pixels``. The full
        image still drops anything under ``min_pixels``."""
        cfg = self.config
        groups: list[tuple[np.ndarray, dict[str, float], bool]] = []
        for category, score, mask, from_crop in sorted(detections, key=lambda d: -d[1]):
            for i, (gmask, glabels, crop_only) in enumerate(groups):
                if _iou(gmask, mask) >= cfg.merge_iou:
                    glabels[category] = max(glabels.get(category, 0.0), score)
                    groups[i] = (gmask | mask, glabels, crop_only and from_crop)
                    break
            else:
                groups.append((mask.copy(), {category: score}, from_crop))
        groups.sort(key=lambda g: -int(g[0].sum()))
        canvas = np.zeros(shape, dtype=np.int32)
        for i, (gmask, _, _) in enumerate(groups, start=1):
            canvas[gmask] = i
        mask = np.zeros(shape, dtype=np.uint8)
        labels: dict[int, dict[str, float]] = {}
        for i, (_, glabels, crop_only) in enumerate(groups, start=1):
            visible = canvas == i
            floor = cfg.crop_min_pixels if crop_only else cfg.min_pixels
            if int(visible.sum()) < floor or len(labels) >= 255:
                continue
            new_id = len(labels) + 1
            mask[visible] = new_id
            labels[new_id] = {k: round(v, 4) for k, v in glabels.items()}
        return mask, labels
