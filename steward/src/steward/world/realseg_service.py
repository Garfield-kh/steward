"""Class-agnostic instance segmentation API over SAM3, for Steward's real-perception mode.

POST /segment_instances  {image_base64, points?: [[u, v], ...], boxes?: [[x0, y0, x1, y1], ...],
                          min_score?, box_min_score?, max_instances?}
    -> {mask_png_base64, shape, instances: [{id, box, score}]}

Points are interactive prompts; SAM3 answers each with up to three granularity candidates
(multimask), of which the finest confidently-scoring one is kept. Boxes (pixel xyxy) ground a
full detection pass each — they localize small foreground objects that point prompts miss,
because points on a small object over a large surface mostly score the surface. Near-duplicate
masks are merged by IoU and the survivors are painted into one uint8 label map, largest first
so finer masks claim their pixels. Without ``points`` a 6x4 grid is prompted. This complements
the text-prompt ``/segment`` endpoint of the cap-harness SAM3 service on :8114.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import functools
import io
import logging
from typing import Any

from fastapi import FastAPI, HTTPException
import numpy as np
from PIL import Image
from pydantic import BaseModel
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model
import torch
import uvicorn

LOGGER = logging.getLogger(__name__)
app = FastAPI(title="steward perception: SAM3 instance segmentation")

GRANULARITY_MARGIN = 0.15  # score slack when picking the finest multimask candidate
MIN_SOLIDITY = 0.45  # share of its bounding box a candidate must fill to count as a whole object

_MODEL: Any | None = None
_PROCESSOR: Any | None = None
_DEVICE = "cuda"
_GPU_SEMAPHORE = asyncio.Semaphore(1)


class SegmentTextRequest(BaseModel):
    image_base64: str
    text_prompt: str
    min_score: float = 0.5
    max_instances: int = 5


class SegmentInstancesRequest(BaseModel):
    image_base64: str
    points: list[list[float]] | None = None
    boxes: list[list[float]] | None = None  # pixel [x0, y0, x1, y1] prompts
    min_score: float = 0.5
    box_min_score: float = 0.3
    max_instances: int = 24


class InstanceInfo(BaseModel):
    id: int
    box: list[float]
    score: float


class SegmentInstancesResponse(BaseModel):
    mask_png_base64: str
    shape: list[int]
    instances: list[InstanceInfo]


def _release_cached_gpu_memory() -> None:
    if _DEVICE.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _decode_image(payload: str) -> Image.Image:
    try:
        raw = base64.b64decode(payload, validate=True)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {exc}") from exc


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if getattr(value, "dtype", None) == torch.bfloat16:
        value = value.float()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _default_grid(width: int, height: int, cols: int = 6, rows: int = 4) -> list[list[float]]:
    return [[width * (i + 0.5) / cols, height * (j + 0.5) / rows] for j in range(rows) for i in range(cols)]


def _solidity(mask: np.ndarray) -> float:
    """Share of a mask's bounding box that the mask fills (a disk ~0.78, a thin arc ~0.2)."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return 0.0
    box = float(ys.max() - ys.min() + 1) * float(xs.max() - xs.min() + 1)
    return float(len(ys)) / box


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(a, b).sum()) / float(union)


def _segment_instances(image: Image.Image, points: list[list[float]], boxes: list[list[float]],
                       min_score: float, box_min_score: float,
                       max_instances: int) -> SegmentInstancesResponse:
    device_type = "cuda" if "cuda" in _DEVICE else "cpu"
    with torch.autocast(device_type, dtype=torch.bfloat16):
        state = _PROCESSOR.set_image(image)
        kept: list[tuple[float, np.ndarray]] = []
        for point in points:
            masks, scores, _ = _MODEL.predict_inst(
                state,
                point_coords=np.asarray([[float(point[0]), float(point[1])]], dtype=np.float32),
                point_labels=np.ones(1, dtype=np.int64),
                multimask_output=True,
            )
            mask_array = np.asarray(masks)
            score_array = np.asarray(scores).reshape(-1)
            if mask_array.size == 0 or score_array.size == 0:
                continue
            # SAM's multimask candidates differ in granularity; argmax favors the surrounding
            # context (whole wall, whole scene) and swallows the prompted object. For an
            # instance sweep we want the finest candidate that is still confidently the thing
            # pointed at: smallest area whose score is within GRANULARITY_MARGIN of the best —
            # but not a sub-part. A point on a wide bowl's rim offers the rim arc as the finest
            # candidate; an arc fills little of its own bounding box, a whole object fills most
            # of it, so thin candidates only count when nothing solid is offered.
            best = float(score_array.max())
            if best < min_score:
                continue
            eligible = [i for i in range(len(score_array))
                        if float(score_array[i]) >= max(min_score, best - GRANULARITY_MARGIN)]
            solid = [i for i in eligible if _solidity(mask_array[i] > 0) >= MIN_SOLIDITY]
            chosen = min(solid or eligible, key=lambda i: int(mask_array[i].sum()))
            candidate = mask_array[chosen] > 0
            if candidate.sum() < 25:
                continue
            if any(_iou(candidate, existing) > 0.5 for _, existing in kept):
                continue
            kept.append((float(score_array[chosen]), candidate))
            if len(kept) >= max_instances:
                break
        # Box prompts localize small foreground objects that point prompts miss (they prefer the
        # large surface behind). Pixel xyxy in, normalized cxcywh out, one grounding pass each.
        for box in boxes[:8]:
            x0, y0, x1, y1 = [float(v) for v in box]
            x0, x1 = sorted((max(0.0, x0), min(image.width - 1.0, x1)))
            y0, y1 = sorted((max(0.0, y0), min(image.height - 1.0, y1)))
            if x1 - x0 < 6 or y1 - y0 < 6:
                continue
            cx, cy = (x0 + x1) / 2 / image.width, (y0 + y1) / 2 / image.height
            bw, bh = (x1 - x0) / image.width, (y1 - y0) / image.height
            _PROCESSOR.reset_all_prompts(state)
            out = _PROCESSOR.add_geometric_prompt(box=[cx, cy, bw, bh], label=True, state=state)
            masks = out.get("masks")
            scores = out.get("scores")
            if masks is None or scores is None:
                continue
            mask_array = _to_numpy(masks)
            if mask_array.ndim == 4 and mask_array.shape[1] == 1:
                mask_array = mask_array[:, 0]
            score_array = _to_numpy(scores).reshape(-1)
            if mask_array.size == 0 or score_array.size == 0:
                continue
            best = int(np.argmax(score_array))
            if float(score_array[best]) < box_min_score:
                continue
            candidate = np.asarray(mask_array[best]) > 0
            if candidate.sum() < 25:
                continue
            if any(_iou(candidate, existing) > 0.5 for _, existing in kept):
                continue
            kept.append((float(score_array[best]), candidate))
            if len(kept) >= max_instances:
                break

    height, width = image.height, image.width
    png, infos = _paint(kept, height, width)
    return SegmentInstancesResponse(mask_png_base64=png, shape=[height, width], instances=infos)


def _paint(kept: list[tuple[float, "np.ndarray"]], height: int, width: int
           ) -> tuple[str, list[InstanceInfo]]:
    # largest first: fine masks claim their pixels later and win overlaps
    kept.sort(key=lambda item: item[1].sum(), reverse=True)
    labels = np.zeros((height, width), dtype=np.uint8)
    infos: list[InstanceInfo] = []
    for instance_id, (score, mask) in enumerate(kept, start=1):
        labels[mask] = instance_id
        ys, xs = np.nonzero(mask)
        infos.append(InstanceInfo(id=instance_id,
                                  box=[float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
                                  score=score))
    buf = io.BytesIO()
    Image.fromarray(labels, mode="L").save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii"), infos


@app.post("/segment_text", response_model=SegmentInstancesResponse)
async def segment_text(request: SegmentTextRequest) -> SegmentInstancesResponse:
    """Class-prompted segmentation: the whole instances a name matches, best first."""
    if _MODEL is None or _PROCESSOR is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    try:
        image = _decode_image(request.image_base64)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {exc}") from exc

    device_type = "cuda" if "cuda" in _DEVICE else "cpu"
    with torch.autocast(device_type, dtype=torch.bfloat16):
        state = _PROCESSOR.set_image(image)
        output = _PROCESSOR.set_text_prompt(state=state, prompt=request.text_prompt)
    masks = output.get("masks")
    scores = output.get("scores")
    if masks is None or scores is None:
        return SegmentInstancesResponse(mask_png_base64="", shape=[image.height, image.width], instances=[])
    mask_array = _to_numpy(masks)
    if mask_array.ndim == 4 and mask_array.shape[1] == 1:
        mask_array = mask_array[:, 0]
    score_array = _to_numpy(scores).reshape(-1)
    kept: list[tuple[float, np.ndarray]] = []
    order = np.argsort(score_array)[::-1]
    for i in order:
        if float(score_array[i]) < request.min_score:
            continue
        candidate = mask_array[i] > 0
        if candidate.sum() < 25:
            continue
        if any(_iou(candidate, existing) > 0.5 for _, existing in kept):
            continue
        kept.append((float(score_array[i]), candidate))
        if len(kept) >= request.max_instances:
            break
    png, infos = _paint(kept, image.height, image.width)
    return SegmentInstancesResponse(mask_png_base64=png, shape=[image.height, image.width], instances=infos)


@app.get("/health")
async def health() -> dict:
    return {"ok": _MODEL is not None and _PROCESSOR is not None}


@app.post("/segment_instances", response_model=SegmentInstancesResponse)
async def segment_instances(request: SegmentInstancesRequest) -> SegmentInstancesResponse:
    if _MODEL is None or _PROCESSOR is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    try:
        image = _decode_image(request.image_base64)
        points = request.points or _default_grid(image.width, image.height)
        if not points or any(len(p) != 2 for p in points):
            raise HTTPException(status_code=400, detail="points must be [x, y] pairs")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Bad request: {exc}") from exc
    loop = asyncio.get_running_loop()
    try:
        async with _GPU_SEMAPHORE:
            return await loop.run_in_executor(
                None, functools.partial(_segment_instances, image, points, request.boxes or [],
                                        request.min_score, request.box_min_score,
                                        request.max_instances))
    finally:
        _release_cached_gpu_memory()


def main() -> None:
    parser = argparse.ArgumentParser(description="SAM3 class-agnostic instance segmentation service")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8117)
    args = parser.parse_args()

    global _DEVICE, _MODEL, _PROCESSOR
    _DEVICE = args.device
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if "cuda" in args.device:
            torch.cuda.set_device(int(args.device.split(":")[-1]) if ":" in args.device else 0)
    LOGGER.info("Loading SAM3 model on %s", args.device)
    _MODEL = build_sam3_image_model(enable_inst_interactivity=True).to(args.device)
    _PROCESSOR = Sam3Processor(_MODEL, device=args.device, confidence_threshold=0.0)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()


# ---------------------------------------------------------------------------
# Deployment (not part of the steward package; nothing imports this file):
#   scp src/steward/world/realseg_service.py spark:/home/lilin/steward_perception_service.py
#   ssh spark /home/lilin/restart_perception.sh   # serves :8117, SAM3 weights
# The restart script on spark runs it with venv-sam3 and the offline HF_HOME
# layout; the client side is steward.world.realseg via STEWARD_SEGMENTER_URL.
# ---------------------------------------------------------------------------
