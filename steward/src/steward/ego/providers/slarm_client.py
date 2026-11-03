"""Thin client for linxlab SLARM reconstruction API (slarm_scene_v1)."""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore


@dataclass
class SlarmConfig:
    base_url: str = "http://linxlab-research.com/slarm"
    outputs: list[str] = field(default_factory=lambda: ["depth", "pointcloud", "semantic"])
    timeout_s: float = 1200.0


class SlarmClient:
    def __init__(self, config: SlarmConfig | None = None):
        if requests is None:
            raise RuntimeError("requests is required for SlarmClient")
        self.config = config or SlarmConfig()

    def ready(self) -> bool:
        r = requests.get(f"{self.config.base_url}/health/ready", timeout=20)
        return r.ok and r.json().get("status") == "ready"

    @staticmethod
    def sha256_file(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def upload_zip(self, zip_path: Path, client_ref: str = "steward-phase1") -> str:
        digest = self.sha256_file(zip_path)
        with zip_path.open("rb") as f:
            r = requests.post(
                f"{self.config.base_url}/v1/datasets",
                headers={"Idempotency-Key": str(uuid.uuid4())},
                data={
                    "archive_sha256": digest,
                    "format": "slarm_scene_v1",
                    "client_ref": client_ref,
                },
                files={"archive": (zip_path.name, f, "application/zip")},
                timeout=600,
            )
        r.raise_for_status()
        return r.json()["id"]

    def wait_dataset(self, dataset_id: str, timeout_s: float = 300.0) -> dict:
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            d = requests.get(f"{self.config.base_url}/v1/datasets/{dataset_id}", timeout=30).json()
            if d.get("status") == "ready":
                return d
            if d.get("status") == "invalid":
                raise RuntimeError(json.dumps(d, ensure_ascii=False))
            time.sleep(2)
        raise TimeoutError("SLARM dataset timeout")

    def create_reconstruction(self, dataset_id: str, frame_end: int, client_ref: str = "steward-phase1") -> str:
        body = {
            "dataset_id": dataset_id,
            "frame_range": {"start": 0, "end": frame_end},
            "outputs": list(self.config.outputs),
            "evaluate": False,
            "client_ref": client_ref,
        }
        r = requests.post(
            f"{self.config.base_url}/v1/reconstructions",
            headers={"Idempotency-Key": str(uuid.uuid4()), "Content-Type": "application/json"},
            json=body,
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["id"]

    def wait_reconstruction(self, rec_id: str) -> dict:
        t0 = time.time()
        while time.time() - t0 < self.config.timeout_s:
            j = requests.get(f"{self.config.base_url}/v1/reconstructions/{rec_id}", timeout=30).json()
            if j.get("status") in ("succeeded", "failed", "cancelled", "expired"):
                if j.get("status") != "succeeded":
                    raise RuntimeError(json.dumps(j, ensure_ascii=False))
                return j
            time.sleep(3)
        raise TimeoutError("SLARM reconstruction timeout")

    def result(self, rec_id: str) -> dict:
        r = requests.get(f"{self.config.base_url}/v1/reconstructions/{rec_id}/result", timeout=120)
        r.raise_for_status()
        return r.json()

    def download_artifacts(self, result: dict, out_dir: Path, kinds: set[str] | None = None) -> dict[str, Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        kinds = kinds or {"pointcloud", "render_arrays", "cameras", "manifest"}
        saved: dict[str, Path] = {}
        base = self.config.base_url
        for a in result.get("artifacts") or []:
            if a.get("kind") not in kinds:
                continue
            dest = out_dir / a["filename"]
            with requests.get(f"{base}{a['download_path']}", stream=True, timeout=600) as resp:
                resp.raise_for_status()
                with dest.open("wb") as f:
                    for chunk in resp.iter_content(1 << 20):
                        if chunk:
                            f.write(chunk)
            saved[a["filename"]] = dest
        return saved

    @staticmethod
    def center_depths_from_renders(art_dir: Path, cam_index: int = 0) -> dict[int, np.ndarray]:
        """Map frame_index → depth (H,W) from renders_*.npz."""
        out: dict[int, np.ndarray] = {}
        for rp in sorted(Path(art_dir).glob("renders_*.npz")):
            z = np.load(rp)
            d = z["depth_z_m"]
            v = z["valid_mask"] if "valid_mask" in z.files else np.isfinite(d)
            idxs = z["frame_indices"].tolist() if "frame_indices" in z.files else list(range(d.shape[0]))
            for i, t in enumerate(idxs):
                di = d[i, cam_index] if d.ndim == 4 else d[i]
                vi = v[i, cam_index] if v.ndim == 4 else v[i]
                arr = di.astype(np.float32)
                arr = np.where(vi.astype(bool), arr, 0.0)
                out[int(t)] = arr
        return out
