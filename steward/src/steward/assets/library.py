"""Content-addressed asset library.

Layout under the library root::

    index.json                 {"assets": {name: {"latest": hash, "versions": [record, ...]}}}
    objects/<hash>.json        {"name", "category", "template", "meta", "validation"}
    previews/<hash>-<view>.png

An asset version is identified by the SHA-256 of its canonical template JSON, so publishing the
same template twice yields the same version. Templates are World State entities with id
``w:template`` (see :mod:`steward.assets.generators`).
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path

from steward.assets.generators import TEMPLATE_ID
from steward.math3d import quat_normalize
from steward.scene.schema import Entity, exact
from steward.scene.validate import validate_world


def template_hash(template: dict) -> str:
    canonical = json.dumps(template, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


class AssetLibrary:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        (self.root / "objects").mkdir(parents=True, exist_ok=True)
        (self.root / "previews").mkdir(parents=True, exist_ok=True)
        self._index_path = self.root / "index.json"
        if not self._index_path.exists():
            self._write_index({"assets": {}})

    def _read_index(self) -> dict:
        return json.loads(self._index_path.read_text(encoding="utf-8"))

    def _write_index(self, index: dict) -> None:
        tmp = self._index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self._index_path)

    def publish(self, name: str, template: dict, meta: dict | None = None, validation: dict | None = None) -> str:
        """Store a validated template under ``name`` and return its version hash."""
        if template.get("id") != TEMPLATE_ID:
            raise ValueError(f"asset template id must be {TEMPLATE_ID!r}")
        errors = validate_world({"mode": "world", "entities": [template]})
        if errors:
            raise ValueError(f"asset {name!r} is not a valid World State entity: {errors}")
        if validation is not None and not validation.get("ok", False):
            raise ValueError(f"asset {name!r} failed validation: {validation.get('errors')}")
        digest = template_hash(template)
        record = {
            "name": name,
            "category": template["category"]["value"],
            "template": template,
            "meta": meta or {},
            "validation": validation,
        }
        (self.root / "objects" / f"{digest}.json").write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
        index = self._read_index()
        entry = index["assets"].setdefault(name, {"latest": digest, "versions": []})
        if all(v["hash"] != digest for v in entry["versions"]):
            entry["versions"].append({"hash": digest, "published": time.time(), "category": record["category"],
                                      "meta": record["meta"]})
        entry["latest"] = digest
        self._write_index(index)
        return digest

    def get(self, name: str, version: str | None = None) -> dict:
        index = self._read_index()
        if name not in index["assets"]:
            raise KeyError(f"unknown asset {name!r}")
        digest = version or index["assets"][name]["latest"]
        path = self.root / "objects" / f"{digest}.json"
        if not path.exists():
            raise KeyError(f"unknown version {digest!r} of asset {name!r}")
        return json.loads(path.read_text(encoding="utf-8")) | {"version": digest}

    def search(self, query: str | None = None, category: str | None = None) -> list[dict]:
        out = []
        for name, entry in sorted(self._read_index()["assets"].items()):
            latest = next(v for v in entry["versions"] if v["hash"] == entry["latest"])
            if category and latest["category"] != category:
                continue
            if query and query.lower() not in name.lower() and query.lower() not in latest["category"].lower():
                continue
            out.append({"name": name, "category": latest["category"], "version": entry["latest"],
                        "versions": len(entry["versions"]), "meta": latest.get("meta", {})})
        return out

    def preview_path(self, digest: str, view: str) -> Path:
        return self.root / "previews" / f"{digest}-{view}.png"


def instantiate(template: dict, entity_id: str, at: dict, asset: dict | None = None) -> Entity:
    """Create a World State entity from a template at pose ``at`` ({"p", "q"})."""
    data = copy.deepcopy(template)
    data["id"] = entity_id
    ent = Entity.from_dict(data)
    ent.pose = exact({"p": [float(x) for x in at["p"]], "q": quat_normalize(at.get("q", [1, 0, 0, 0]))})
    ent.asset = asset
    return ent
