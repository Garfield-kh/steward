"""Install the bundled static Merom house into a Steward data root.

Mesh names are content hashes. Published scenes refer to the extracted files in the data root,
so loading a built scene does not depend on the source USD or the checkout's asset directory.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING
from zipfile import ZipFile

if TYPE_CHECKING:
    from steward.authoring.service import AuthoringService


BUNDLE = Path(__file__).resolve().parents[1] / "assets" / "merom_house"


def build_merom_house(service: AuthoringService) -> dict:
    """Install the robot-free house with metre-scale geometry and separate collision pieces.

    :param service: Authoring service whose data root receives the scene and meshes.
    :returns: Published scene name and scenario record.
    :raises ValueError: A bundled mesh does not match its content-hash filename.
    """
    scene = json.loads((BUNDLE / "scene.json").read_text(encoding="utf-8"))
    scenario = json.loads((BUNDLE / "scenario.json").read_text(encoding="utf-8"))
    geometries = [g for ent in scene["entities"] for link in ent["body"]["links"]
                  for group in link["geometry"].values() for g in group if g["kind"] == "mesh"]
    with ZipFile(BUNDLE / "meshes.zip") as archive:
        meshes = {ref: archive.read(ref) for ref in {g["ref"] for g in geometries}}
    for ref, content in meshes.items():
        if ref != hashlib.sha256(content).hexdigest() + ".msh":
            raise ValueError(f"Merom mesh {ref!r} does not match its content hash")

    mesh_dir = service.root.resolve() / "imports" / "merom-house"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    for ref, content in meshes.items():
        (mesh_dir / ref).write_bytes(content)
    for geom in geometries:
        geom["ref"] = str(mesh_dir / geom["ref"])
    scene_path = service.root / "scenes" / f"{scenario['scene']}.json"
    scene_path.write_text(json.dumps(scene, ensure_ascii=False) + "\n", encoding="utf-8")
    published = service.save_scenario(scenario)
    return {"scene": scenario["scene"], "scenario": published}
