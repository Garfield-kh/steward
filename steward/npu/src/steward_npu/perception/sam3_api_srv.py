"""Optional SAM 3 text-prompt image service on CUDA or Ascend NPU.

GET /api/health; POST /api/segment accepts multipart file and JSON options.
Model initialization and accelerator selection occur only in main().
"""

import argparse
import base64
import collections
import hashlib
import io
import json
import os
import queue
import sys
import threading
import time
import types
from email.parser import BytesParser
from email.policy import HTTP
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
from PIL import Image


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18230)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument(
        "--device-type",
        choices=["cuda", "npu"],
        default="npu" if os.environ.get("STEWARD_USE_NPU") == "1" else "cuda",
    )
    ap.add_argument(
        "--devices",
        "--gpus",
        dest="gpus",
        default="0",
        help="Comma-separated logical device indices after visibility filtering; one replica each",
    )
    ap.add_argument("--ckpt", required=True, help="Local SAM 3 checkpoint file")
    ap.add_argument(
        "--memory-fraction",
        type=float,
        default=0.45,
        help="NPU allocator limit per replica; (0, 1]",
    )
    ap.add_argument("--log", default=None, help="JSONL request log")
    ap.add_argument(
        "--save-dir",
        default=None,
        help="save the first --save-n request images here (probe use)",
    )
    ap.add_argument("--save-n", type=int, default=0)
    ap.add_argument(
        "--npu-text-cache",
        type=int,
        default=0,
        help="NPU-only bounded text-feature cache; 0 disables (baseline)",
    )
    ap.add_argument(
        "--profile-stages",
        action="store_true",
        help="NPU-only synchronized stage timings; disable for throughput tests",
    )
    ap.add_argument(
        "--npu-empty-box-fastpath",
        action="store_true",
        help="Skip geometry encoding for zero box prompts on NPU",
    )
    ap.add_argument(
        "--npu-batched-postprocess",
        action="store_true",
        help="Gather detections and transfer masks in bounded batches",
    )
    ap.add_argument(
        "--npu-empty-geometry",
        action="store_true",
        help="Skip empty prompt preparation while retaining geometry CLS attention",
    )
    ap.add_argument(
        "--npu-shared-geometry",
        action="store_true",
        help="Compute text-only geometric CLS once per shared image",
    )
    ap.add_argument(
        "--npu-graphs",
        action="store_true",
        help="Use bounded NPU grounding graph replay",
    )
    ap.add_argument("--npu-graph-cache", type=int, default=4)
    A = ap.parse_args(argv)
    if A.npu_text_cache < 0 or A.npu_text_cache > 32:
        ap.error("--npu-text-cache must be between 0 and 32")
    if A.npu_shared_geometry and not A.npu_empty_geometry:
        ap.error("--npu-shared-geometry requires --npu-empty-geometry")
    if A.npu_graph_cache < 1 or A.npu_graph_cache > 4:
        ap.error("--npu-graph-cache must be between 1 and 4")
    if A.npu_graphs and not A.npu_empty_geometry:
        ap.error("--npu-graphs requires --npu-empty-geometry")
    if A.device_type != "npu" and (
        A.npu_text_cache
        or A.profile_stages
        or A.npu_empty_box_fastpath
        or A.npu_batched_postprocess
        or A.npu_empty_geometry
        or A.npu_shared_geometry
        or A.npu_graphs
    ):
        ap.error("NPU optimization flags require --device-type npu")

    if not os.path.isfile(A.ckpt):
        ap.error(f"sam3.pt not found: {A.ckpt}")

    if not 0 < A.memory_fraction <= 1:
        ap.error("--memory-fraction must be in (0, 1]")
    try:
        devices = [int(x) for x in A.gpus.split(",")]
        if not devices or min(devices) < 0 or len(set(devices)) != len(devices):
            raise ValueError
    except ValueError:
        ap.error("--devices must contain distinct non-negative indices")
    return A


def initialize_runtime(config):
    global A, ACCEL, box_ops, FindStage, interpolate, Sam3Processor, build_sam3_image_model
    A = config

    if A.device_type == "npu":
        import torch_npu
    ACCEL = getattr(torch, A.device_type)

    from sam3.model import box_ops
    from sam3.model.data_misc import FindStage, interpolate
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


LOG_LOCK = threading.Lock()
SAVED = [0]


class Replica:
    def __init__(self, gpu: int):
        self.device = f"{A.device_type}:{gpu}"
        ACCEL.set_device(gpu)
        if A.device_type == "npu":
            ACCEL.set_per_process_memory_fraction(A.memory_fraction, gpu)
        model = build_sam3_image_model(
            checkpoint_path=A.ckpt, load_from_HF=False, device=self.device
        )
        self.model = model.to(self.device).eval()
        if A.npu_empty_box_fastpath:
            encoder = self.model.geometry_encoder
            original = encoder._encode_boxes

            def encode_boxes(instance, boxes, boxes_mask, boxes_labels, img_feats):
                # Text-only calls have zero geometry tokens. ROIAlign otherwise
                # copies the full feature map to CPU even though its output is empty.
                if boxes.shape[0] == 0 and not instance.training:
                    empty = instance.label_embed.weight.new_empty(
                        (0, boxes.shape[1], instance.d_model)
                    )
                    return empty, boxes_mask
                return original(boxes, boxes_mask, boxes_labels, img_feats)

            encoder._encode_boxes = types.MethodType(encode_boxes, encoder)
        if A.npu_empty_geometry:
            from sam3.model.act_ckpt_utils import activation_ckpt_wrapper

            encoder = self.model.geometry_encoder
            original_forward = encoder.forward

            def empty_geometry(
                instance, geo_prompt, img_feats, img_sizes, img_pos_embeds=None
            ):
                points, boxes = geo_prompt.point_embeddings, geo_prompt.box_embeddings
                if (
                    instance.training
                    or points is None
                    or boxes is None
                    or points.shape[0]
                    or boxes.shape[0]
                    or geo_prompt.mask_embeddings is not None
                    or instance.cls_embed is None
                ):
                    return original_forward(
                        geo_prompt, img_feats, img_sizes, img_pos_embeds
                    )
                # The original empty point/box concatenation leaves just this learned CLS token.
                full_bs = boxes.shape[1]
                shared = A.npu_shared_geometry and getattr(
                    self.model, "_single_image_text_request", False
                )
                bs = 1 if shared else full_bs
                embeds = instance.cls_embed.weight.view(1, 1, instance.d_model).repeat(
                    1, bs, 1
                )
                mask = torch.zeros(
                    bs, 1, dtype=geo_prompt.box_mask.dtype, device=embeds.device
                )
                if instance.final_proj is not None:
                    embeds = instance.norm(instance.final_proj(embeds))
                if instance.encode is not None:
                    memory = img_feats[-1][:, :bs]
                    pos = (
                        img_pos_embeds[-1][:, :bs]
                        if img_pos_embeds is not None
                        else torch.zeros_like(memory)
                    )
                    for lay in instance.encode:
                        embeds = activation_ckpt_wrapper(lay)(
                            tgt=embeds,
                            memory=memory,
                            tgt_key_padding_mask=mask,
                            pos=pos,
                            act_ckpt_enable=False,
                        )
                    embeds = instance.encode_norm(embeds)
                return (
                    (embeds.repeat(1, full_bs, 1), mask.repeat(full_bs, 1))
                    if shared
                    else (embeds, mask)
                )

            encoder.forward = types.MethodType(empty_geometry, encoder)
        self.graphs = None
        if A.npu_graphs:
            from .graph_accel import GroundingGraphs

            self.graphs = GroundingGraphs(self.model, max_entries=A.npu_graph_cache)
            self.model.forward_grounding = self.graphs
        self.proc = Sam3Processor(
            self.model, device=self.device, confidence_threshold=0.0
        )
        self.cache = collections.OrderedDict()  # image digest -> backbone_out (LRU)
        self.forward_calls = 0
        self.cache_hits = 0
        self.gpu = gpu
        self.text_cache = collections.OrderedDict()
        self.text_cache_hits = 0
        self.stage_times = {}

    def text_features(self, prompts):
        # One eval model/device and fixed BF16 autocast per replica. Preserve order
        # and duplicate prompts: changing either can change positional outputs.
        key = tuple(prompts)
        if A.npu_text_cache and key in self.text_cache:
            self.text_cache.move_to_end(key)
            self.text_cache_hits += 1
            return self.text_cache[key]
        features = self.model.backbone.forward_text(prompts, device=self.device)
        if A.npu_text_cache and len(prompts) <= 128:
            self.text_cache[key] = features
            while len(self.text_cache) > A.npu_text_cache:
                self.text_cache.popitem(last=False)
        return features

    @torch.inference_mode()
    def backbone(self, digest: str, im: Image.Image):
        if digest in self.cache:
            self.cache.move_to_end(digest)
            self.cache_hits += 1
            return self.cache[digest]
        st = self.proc.set_image(im)
        self.forward_calls += 1
        self.cache[digest] = st
        while len(self.cache) > 4:
            self.cache.popitem(last=False)
        return st

    @torch.inference_mode()
    def postprocess_batch(self, out, probs, threshold, w, h):
        # Threshold on the original device/dtype to preserve BF16 comparison semantics.
        keep = (probs > threshold).cpu().numpy()
        counts = keep.sum(axis=1).tolist()
        if not keep.any():
            return [
                (np.zeros(0), np.zeros((0, 4)), np.zeros((0, h, w), bool))
                for _ in counts
            ]
        indices = np.flatnonzero(keep.reshape(-1))
        index = torch.from_numpy(indices).to(self.device)
        scores = probs.float().cpu().numpy().reshape(-1)[indices]
        boxes = out["pred_boxes"].reshape(-1, 4).index_select(0, index).float()
        scale = torch.tensor([w, h, w, h], device=self.device, dtype=torch.float32)
        boxes = (box_ops.box_cxcywh_to_xyxy(boxes) * scale[None]).cpu().numpy()
        source = out["pred_masks"].flatten(0, 1)
        chunks = []
        # Bound temporary full-resolution float masks, including high-detection requests.
        for begin in range(0, len(indices), 16):
            masks = (
                source.index_select(0, index[begin : begin + 16]).float().unsqueeze(1)
            )
            masks = (
                interpolate(
                    masks, (h, w), mode="bilinear", align_corners=False
                ).sigmoid()[:, 0]
                > 0.5
            )
            chunks.append(masks.cpu().numpy())
        masks = np.concatenate(chunks, axis=0)
        result = []
        begin = 0
        for count in counts:
            end = begin + count
            if count:
                result.append((scores[begin:end], boxes[begin:end], masks[begin:end]))
            else:
                result.append(
                    (np.zeros(0), np.zeros((0, 4)), np.zeros((0, h, w), bool))
                )
            begin = end
        return result

    @torch.inference_mode()
    def segment(self, digest: str, im: Image.Image, prompts: list, threshold: float):
        """Per prompt: (probs[k], boxes_xyxy[k,4] pixel, masks[k,H,W] bool) as numpy."""
        w, h = im.size
        if A.device_type == "npu":
            ACCEL.set_device(
                self.device
            )  # CANN request threads need their own device context
        stages = {}
        if A.profile_stages:
            ACCEL.synchronize(self.device)
        last = time.perf_counter()

        def mark(name):
            nonlocal last
            if A.profile_stages:
                ACCEL.synchronize(self.device)
                now = time.perf_counter()
                stages[name] = now - last
                last = now

        with torch.autocast(A.device_type, dtype=torch.bfloat16):
            st = self.backbone(digest, im)
            mark("image_s")
            n = len(prompts)
            bb = dict(st["backbone_out"])
            bb.update(self.text_features(prompts))
            mark("text_s")
            find = FindStage(
                img_ids=torch.zeros(n, device=self.device, dtype=torch.long),
                text_ids=torch.arange(n, device=self.device, dtype=torch.long),
                input_boxes=None,
                input_boxes_mask=None,
                input_boxes_label=None,
                input_points=None,
                input_points_mask=None,
            )
            # Scope sharing to this known single-image, text-only request.
            self.model._single_image_text_request = True
            try:
                out = self.model.forward_grounding(
                    backbone_out=bb,
                    find_input=find,
                    find_target=None,
                    geometric_prompt=self.model._get_dummy_prompt(num_prompts=n),
                )
            finally:
                self.model._single_image_text_request = False
            mark("grounding_s")
            probs = (
                out["pred_logits"].sigmoid()
                * out["presence_logit_dec"].sigmoid().unsqueeze(1)
            ).squeeze(
                -1
            )  # (n,Q)
            if A.npu_batched_postprocess:
                res = self.postprocess_batch(out, probs, threshold, w, h)
                mark("postprocess_s")
                self.stage_times = stages
                return res
            res = []
            scale = torch.tensor([w, h, w, h], device=self.device, dtype=torch.float32)
            for i in range(n):
                keep = probs[i] > threshold
                p = probs[i][keep].float()
                if p.numel() == 0:
                    res.append(
                        (np.zeros(0), np.zeros((0, 4)), np.zeros((0, h, w), bool))
                    )
                    continue
                boxes = (
                    box_ops.box_cxcywh_to_xyxy(out["pred_boxes"][i][keep].float())
                    * scale[None]
                )
                m = (
                    interpolate(
                        out["pred_masks"][i][keep].float().unsqueeze(1),
                        (h, w),
                        mode="bilinear",
                        align_corners=False,
                    ).sigmoid()[:, 0]
                    > 0.5
                )
                res.append((p.cpu().numpy(), boxes.cpu().numpy(), m.cpu().numpy()))
            mark("postprocess_s")
        self.stage_times = stages
        return res


REPLICAS = queue.Queue()
ALL = []


def _png_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", compress_level=1)
    return base64.b64encode(buf.getvalue()).decode()


def _overlay(im: Image.Image, masks: np.ndarray) -> str:
    a = np.asarray(im).astype(np.float32)
    rng = np.random.default_rng(0)
    for m in masks:
        a[m] = 0.5 * a[m] + 0.5 * rng.integers(0, 255, 3)
    return _png_b64(a.astype(np.uint8))


def parse_multipart(ctype: str, body: bytes) -> dict:
    msg = BytesParser(policy=HTTP).parsebytes(
        b"Content-Type: " + ctype.encode() + b"\r\n\r\n" + body
    )
    fields = {}
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        fields[name] = part.get_payload(decode=True)
    return fields


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code: int, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.rstrip("/") in ("/api/health", "/health"):
            self._send(
                200,
                {
                    "status": "ready",
                    "model": "SAM 3 (facebook/sam3 sam3.pt)",
                    "device": ", ".join(
                        f"{r.device} {ACCEL.get_device_name(r.gpu)}" for r in ALL
                    ),
                    "precision": "bf16 autocast",
                    "workers": len(ALL),
                    "sessions": 0,
                    "features": ["text", "multi-prompt image (one batched pass)"],
                    "mask_input_supported": False,
                    "ckpt": A.ckpt,
                },
            )
        elif self.path.rstrip("/") in ("", "/"):
            self._send(
                200,
                {
                    "service": "sam3 self-hosted",
                    "endpoints": ["GET /api/health", "POST /api/segment"],
                },
            )
        else:
            self._send(404, {"detail": "Not Found"})

    def do_POST(self):
        if self.path.rstrip("/") != "/api/segment":
            self._send(404, {"detail": "Not Found"})
            return
        t_in = time.monotonic()
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n)
        try:
            fields = parse_multipart(self.headers.get("Content-Type", ""), body)
            raw = fields.get("file")
            if not raw:
                raise ValueError("missing multipart field 'file'")
            opts = json.loads((fields.get("options") or b"{}").decode())
            prompts = opts.get("prompts") or []
            texts = []
            for p in prompts:
                if (
                    not isinstance(p, dict)
                    or not isinstance(p.get("text"), str)
                    or set(p) - {"text"}
                ):
                    raise ValueError(
                        f"only text prompts are supported by this service: {p!r}"
                    )
                texts.append(p["text"])
            if not texts:
                raise ValueError("no prompts")
            threshold = float(opts.get("threshold", 0.5))
            want_overlay = bool(opts.get("overlay", False))
            im = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            self._send(422, {"detail": f"{type(exc).__name__}: {exc}"})
            return
        digest = hashlib.sha256(raw).hexdigest()
        if A.save_dir and SAVED[0] < A.save_n:
            with LOG_LOCK:
                if SAVED[0] < A.save_n:
                    SAVED[0] += 1
                    os.makedirs(A.save_dir, exist_ok=True)
                    with open(
                        os.path.join(A.save_dir, f"{SAVED[0]:03d}_{digest[:10]}.png"),
                        "wb",
                    ) as f:
                        f.write(raw)
                    with open(
                        os.path.join(A.save_dir, f"{SAVED[0]:03d}_{digest[:10]}.json"),
                        "w",
                    ) as f:
                        json.dump(opts, f)
        rep = REPLICAS.get()
        t_q = time.monotonic()
        try:
            t0 = time.monotonic()
            res = rep.segment(digest, im, texts, threshold)
            ACCEL.synchronize(rep.device)
            t_gpu = time.monotonic() - t0
            fc, ch = rep.forward_calls, rep.cache_hits
            stages = dict(rep.stage_times)
            text_hits = rep.text_cache_hits
        except Exception as exc:  # noqa: BLE001
            self._send(500, {"detail": f"{type(exc).__name__}: {exc}"})
            return
        finally:
            REPLICAS.put(rep)
        w, h = im.size
        results = []
        for text, (p, boxes, masks) in zip(texts, res):
            r = {
                "prompt": {"text": text},
                "frame_index": 0,
                "out_obj_ids": list(range(len(p))),
                "out_probs": [float(x) for x in p],
                "out_boxes_xywh": [
                    [
                        float(b[0] / w),
                        float(b[1] / h),
                        float((b[2] - b[0]) / w),
                        float((b[3] - b[1]) / h),
                    ]
                    for b in boxes
                ],
                "frame_stats": {"num_obj_tracked": int(len(p)), "num_obj_dropped": 0},
                "masks": [_png_b64(m.astype(np.uint8) * 255) for m in masks],
            }
            if want_overlay:
                r["overlay"] = _overlay(im, masks)
            r["boxes"] = [[float(v) for v in b] for b in boxes]
            r["scores"] = [float(x) for x in p]
            results.append(r)
        secs = time.monotonic() - t_q
        self._send(
            200,
            {
                "results": results,
                "width": w,
                "height": h,
                "seconds": round(secs, 3),
                "backbone": {"forward_calls": fc, "cache_hits": ch},
            },
        )
        if A.log:
            rec = {
                "t": round(time.time(), 3),
                "w": w,
                "h": h,
                "n_prompts": len(texts),
                "n_masks": int(sum(len(x[0]) for x in res)),
                "queue_s": round(t_q - t_in, 4),
                "gpu_s": round(t_gpu, 4),
                "total_s": round(time.monotonic() - t_in, 4),
                "gpu": rep.gpu,
                "first_prompt": texts[0],
                "bytes_in": n,
            }
            if rep.graphs is not None:
                rec.update(
                    graph_entries=len(rep.graphs.cache),
                    graph_replays=rep.graphs.replays,
                    graph_fallbacks=rep.graphs.fallbacks,
                )
            if A.device_type == "npu":
                rec.update(
                    stages=stages,
                    text_cache_hits=text_hits,
                    text_cache_capacity=A.npu_text_cache,
                )
            with LOG_LOCK, open(A.log, "a") as f:
                f.write(json.dumps(rec) + "\n")


def main(argv=None):
    initialize_runtime(parse_args(argv))
    for g in [int(x) for x in A.gpus.split(",") if x != ""]:
        r = Replica(g)
        # warm-up
        r.segment(
            "warm", Image.new("RGB", (640, 480), (128, 128, 128)), ["cup", "table"], 0.4
        )
        r.cache.clear()
        ALL.append(r)
        REPLICAS.put(r)
        print(f"replica {A.device_type}:{g} ready", flush=True)
    srv = ThreadingHTTPServer((A.host, A.port), Handler)
    srv.daemon_threads = True
    print(f"listening http://{A.host}:{A.port} pid={os.getpid()}", flush=True)
    try:
        srv.serve_forever()
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
