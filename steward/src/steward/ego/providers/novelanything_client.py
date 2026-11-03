"""Thin client for LinxLab NovelAnything (relative depth / novel views).

Use as a *gated* Phase-1 assist only: accept predicted views when a depth scale gate
passes and SAM3 on the predicted RGB is non-empty. Do not treat relative depth as metric
without a scale estimate against the sensor depth.
"""
from __future__ import annotations

import io
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore


@dataclass
class NovelAnythingConfig:
    base_url: str = "https://linxlab-research.com/novelanything"
    timeout_s: float = 600.0
    poll_s: float = 2.0
    # Keep a novel depth only when median(sensor/pred) is in this range.
    scale_lo: float = 0.7
    scale_hi: float = 1.35


class NovelAnythingClient:
    def __init__(self, config: NovelAnythingConfig | None = None):
        if requests is None:
            raise RuntimeError("requests is required for NovelAnythingClient")
        self.config = config or NovelAnythingConfig()

    def ready(self) -> bool:
        r = requests.get(f"{self.config.base_url.rstrip('/')}/health/ready", timeout=15)
        return r.ok and (r.json().get("status") in ("ready", "ok", True) or r.ok)

    def submit(self, payload: dict[str, Any]) -> str:
        """Submit a job; ``payload`` follows the service's novel-view schema."""
        r = requests.post(
            f"{self.config.base_url.rstrip('/')}/api/jobs",
            json=payload,
            timeout=60,
        )
        r.raise_for_status()
        body = r.json()
        job_id = body.get("job_id") or body.get("id")
        if not job_id:
            raise RuntimeError(f"NovelAnything submit missing job id: {body}")
        return str(job_id)

    def wait(self, job_id: str) -> dict:
        cfg = self.config
        deadline = time.time() + cfg.timeout_s
        while time.time() < deadline:
            r = requests.get(f"{cfg.base_url.rstrip('/')}/api/jobs/{job_id}", timeout=30)
            r.raise_for_status()
            body = r.json()
            status = str(body.get("status", "")).lower()
            if status in ("succeeded", "success", "done", "completed"):
                return body
            if status in ("failed", "error", "cancelled"):
                raise RuntimeError(f"NovelAnything job {job_id} {status}: {body}")
            time.sleep(cfg.poll_s)
        raise TimeoutError(f"NovelAnything job {job_id} timed out after {cfg.timeout_s}s")

    @staticmethod
    def depth_scale(sensor_m: np.ndarray, pred_m: np.ndarray, valid: np.ndarray | None = None) -> float:
        """Robust median scale sensor ≈ scale * pred."""
        s = np.asarray(sensor_m, dtype=np.float32)
        p = np.asarray(pred_m, dtype=np.float32)
        if valid is None:
            valid = (s > 0.05) & (p > 1e-4) & np.isfinite(s) & np.isfinite(p)
        if int(valid.sum()) < 64:
            return float("nan")
        ratio = s[valid] / p[valid]
        return float(np.median(ratio))

    def scale_ok(self, scale: float) -> bool:
        return self.config.scale_lo <= float(scale) <= self.config.scale_hi

    @staticmethod
    def encode_png(rgb: np.ndarray) -> bytes:
        buf = io.BytesIO()
        Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(buf, format="PNG")
        return buf.getvalue()
