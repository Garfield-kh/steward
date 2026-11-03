"""Thin client for linxlab rgbd_videogen (NVS) API."""
from __future__ import annotations

import base64
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore


@dataclass
class NvsConfig:
    base_url: str = "http://linxlab-research.com/nvs"
    num_source_views: int = 6
    max_target_views: int = 6
    num_inference_steps: int = 30
    seed: int = 1234
    output_fps: int = 4
    timeout_s: float = 900.0


class NvsClient:
    def __init__(self, config: NvsConfig | None = None):
        if requests is None:
            raise RuntimeError("requests is required for NvsClient")
        self.config = config or NvsConfig()

    def ready(self) -> bool:
        r = requests.get(f"{self.config.base_url}/health/ready", timeout=20)
        return r.ok and r.json().get("status") == "ready"

    def submit_scene(self, scene: dict) -> str:
        """Submit ``rgbd_videogen_scene_v1`` with data-URI images; return job_id."""
        cfg = self.config
        # Embed file paths as data URIs if needed
        scene = json.loads(json.dumps(scene))  # deep copy via json
        for fr in scene.get("frames") or []:
            for view in (fr.get("views") or {}).values():
                img = view.get("image") or ""
                if img.startswith("data:"):
                    continue
                path = Path(img)
                if path.is_file():
                    data = path.read_bytes()
                    view["image"] = "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")
        body = {
            "scene": scene,
            "num_inference_steps": cfg.num_inference_steps,
            "num_source_views": cfg.num_source_views,
            "max_target_views": cfg.max_target_views,
            "seed": cfg.seed,
            "output_fps": cfg.output_fps,
        }
        r = requests.post(
            f"{cfg.base_url}/v1/nvs/jobs",
            headers={"Content-Type": "application/json", "Idempotency-Key": str(uuid.uuid4())},
            data=json.dumps(body),
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["job_id"]

    def wait(self, job_id: str) -> dict:
        t0 = time.time()
        while time.time() - t0 < self.config.timeout_s:
            j = requests.get(f"{self.config.base_url}/v1/nvs/jobs/{job_id}", timeout=30).json()
            if j.get("state") in ("succeeded", "failed", "cancelled"):
                if j.get("state") != "succeeded":
                    raise RuntimeError(json.dumps(j, ensure_ascii=False))
                return j
            time.sleep(3)
        raise TimeoutError(f"NVS job {job_id} timeout")

    def result(self, job_id: str) -> dict:
        r = requests.get(f"{self.config.base_url}/v1/nvs/jobs/{job_id}/result", timeout=120)
        r.raise_for_status()
        return r.json()

    def download_artifacts(self, result: dict, out_dir: Path) -> dict[str, Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        saved: dict[str, Path] = {}
        base = self.config.base_url
        for kind, path in (result.get("artifact_urls") or {}).items():
            url = path if str(path).startswith("http") else base + path
            r = requests.get(url, timeout=180)
            r.raise_for_status()
            name = str(kind).replace("/", "_")
            dest = out_dir / name
            dest.write_bytes(r.content)
            saved[name] = dest
        return saved

    @staticmethod
    def load_depths(depth_npz: Path) -> dict[int, np.ndarray]:
        z = np.load(depth_npz)
        out: dict[int, np.ndarray] = {}
        for k in z.files:
            if k.startswith("v") and k[1:].isdigit():
                out[int(k[1:])] = z[k].astype(np.float32)
        return out
