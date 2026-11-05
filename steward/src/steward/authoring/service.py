"""Asset and scene authoring operations used by the World Harness and the CLI.

Scenes are composed from published assets. A placement either gives a floor pose
(``at``: x, y, yaw) or puts the asset on top of an already placed support entity (``on`` plus an
``offset`` in the support frame). A composed scene is validated by compiling it, rejecting
initial interpenetration between entities and letting it settle for one second.
"""

from __future__ import annotations

import base64
import copy
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from steward.assets.generators import GENERATORS
from steward.assets.library import AssetLibrary, instantiate
from steward.assets.preview import render_scene, render_template
from steward.assets.validator import validate_template
from steward.embodiments import EMBODIMENTS, describe, scenario_embodiments
from steward.math3d import pose_compose, quat_from_yaw, yaw_from_quat
from steward.scene.extent import entity_obb
from steward.scene.mjcf import build_mjcf
from steward.scene.schema import SceneState, SpatialLayers, pose
from steward.scene.validate import validate_world


class AuthoringError(ValueError):
    pass


class AuthoringService:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.library = AssetLibrary(self.root / "assets")
        (self.root / "scenes").mkdir(parents=True, exist_ok=True)
        (self.root / "scenarios").mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------- assets
    def generators(self) -> list[str]:
        return sorted(GENERATORS)

    def create_asset(self, generator: str | None = None, params: dict | None = None,
                     template: dict | None = None) -> dict:
        """Build a template from a generator or accept an authored template, then validate it."""
        if (generator is None) == (template is None):
            raise AuthoringError("give exactly one of generator or template")
        if generator is not None:
            if generator not in GENERATORS:
                raise AuthoringError(f"unknown generator {generator!r}; known: {', '.join(sorted(GENERATORS))}")
            try:
                template = GENERATORS[generator](**(params or {})).to_dict()
            except TypeError as exc:
                raise AuthoringError(f"invalid parameters for {generator}: {exc}") from None
        return {"template": template, "validation": validate_template(template)}

    def preview_asset(self, template: dict, stem: str) -> list[str]:
        out = self.root / "previews"
        return [str(p) for p in render_template(template, out, stem)]

    def publish_asset(self, name: str, template: dict, meta: dict | None = None) -> dict:
        validation = validate_template(template)
        if not validation["ok"]:
            raise AuthoringError(f"asset {name!r} failed validation: {validation['errors']}")
        version = self.library.publish(name, template, meta=meta, validation=validation)
        previews = render_template(template, self.library.root / "previews", version[:16])
        return {"name": name, "version": version, "validation": validation, "previews": [str(p) for p in previews]}

    # ----------------------------------------------------------------- scenes
    def compose_scene(self, spec: dict) -> dict:
        name = spec["name"]
        scene = SceneState(mode="world", layers=SpatialLayers(regions=copy.deepcopy(spec.get("regions", []))))
        for placement in spec["entities"]:
            record = self.library.get(placement["asset"], placement.get("version"))
            at = self._placement_pose(scene, placement)
            ent = instantiate(record["template"], placement["id"], at,
                              asset={"id": placement["asset"], "version": record["version"]})
            if scene.find(ent.id) is not None:
                raise AuthoringError(f"duplicate entity id {ent.id!r}")
            scene.entities.append(ent)
        scene_dict = scene.to_dict()
        errors = validate_world(scene_dict)
        if errors:
            raise AuthoringError(f"composed scene is not a valid World State: {errors[:5]}")
        report = self.check_scene(scene_dict)
        if not report["ok"]:
            raise AuthoringError(f"scene {name!r} failed physical checks: {report['errors']}")
        path = self.root / "scenes" / f"{name}.json"
        path.write_text(json.dumps(scene_dict, indent=2), encoding="utf-8")
        return {"name": name, "path": str(path), "report": report, "entities": len(scene.entities)}

    def _placement_pose(self, scene: SceneState, placement: dict) -> dict:
        if "at" in placement:
            at = placement["at"]
            return pose([at["x"], at["y"], at.get("z", 0.0)], quat_from_yaw(at.get("yaw", 0.0)))
        if "on" in placement:
            support = scene.find(placement["on"])
            if support is None:
                raise AuthoringError(f"support {placement['on']!r} must be placed before {placement['id']!r}")
            probe = SceneState(mode="world", entities=[support])
            build = build_mjcf(probe)
            model = build.compile()
            data = mujoco.MjData(model)
            build.apply_initial(model, data)
            obb = entity_obb(model, data, build.names[support.id])
            top = obb.center[2] + obb.size[2] / 2
            dx, dy = placement.get("offset", [0.0, 0.0])
            sp = support.pose.value
            local = pose_compose(sp, pose([dx, dy, 0.0]))
            yaw = placement.get("yaw", 0.0)
            return pose([local["p"][0], local["p"][1], top + 0.001], quat_from_yaw(yaw))
        if "in" in placement:
            return self._inside_pose(scene, placement)
        raise AuthoringError(f"placement {placement['id']!r} needs 'at', 'on' or 'in'")

    def _inside_pose(self, scene: SceneState, placement: dict) -> dict:
        """Pose on the floor of a container link (a drawer), ``offset`` in the container frame."""
        holder = scene.find(placement["in"])
        if holder is None:
            raise AuthoringError(f"container {placement['in']!r} must be placed before {placement['id']!r}")
        links = [link.id for link in holder.rigid.links if any(a.value == "container" for a in link.affordances)]
        link = placement.get("link") or (links[0] if links else None)
        if link not in links:
            raise AuthoringError(f"{holder.id} has no container link {link!r} (container links: {links})")
        probe = SceneState(mode="world", entities=[holder])
        build = build_mjcf(probe)
        model = build.compile()
        data = mujoco.MjData(model)
        build.apply_initial(model, data)
        obb = entity_obb(model, data, build.names[holder.id], [link])
        floor = _inner_floor(model, data, build.names[holder.id].bodies[link], obb)
        dx, dy = placement.get("offset", [0.0, 0.0])
        yaw0 = yaw_from_quat(obb.quat)
        c, s = math.cos(yaw0), math.sin(yaw0)
        x = obb.center[0] + c * dx - s * dy
        y = obb.center[1] + s * dx + c * dy
        return pose([x, y, floor + 0.001], quat_from_yaw(yaw0 + placement.get("yaw", 0.0)))

    def check_scene(self, scene_dict: dict, settle_seconds: float = 1.0) -> dict:
        report = {"ok": False, "errors": [], "metrics": {}}
        scene = SceneState.from_dict(scene_dict)
        build = build_mjcf(scene)
        model = build.compile()
        data = mujoco.MjData(model)
        build.apply_initial(model, data)
        body_entity = {}
        for eid, names in build.names.items():
            for body in names.bodies.values():
                body_entity[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)] = eid
        for c in range(data.ncon):
            con = data.contact[c]
            e1 = body_entity.get(int(model.geom_bodyid[con.geom1]))
            e2 = body_entity.get(int(model.geom_bodyid[con.geom2]))
            if e1 != e2 and con.dist < -0.01:
                report["errors"].append(f"{e1} and {e2} interpenetrate by {-con.dist:.3f} m")
        free = [e.id for e in scene.entities if e.rigid.root.value == "free"]
        start = {eid: np.array(entity_obb(model, data, build.names[eid]).center) for eid in free}
        for _ in range(int(settle_seconds / model.opt.timestep)):
            mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)):
            report["errors"].append("simulation diverged while settling")
            return report
        moved = {}
        for eid in free:
            d = float(np.linalg.norm(np.array(entity_obb(model, data, build.names[eid]).center) - start[eid]))
            moved[eid] = round(d, 4)
            if d > 0.03:
                report["errors"].append(f"{eid} moved {d:.3f} m while settling")
        report["metrics"]["settle_displacement_m"] = moved
        report["ok"] = not report["errors"]
        return report

    def render_scene_views(self, scene_name: str) -> list[str]:
        scene = SceneState.from_dict(self.load_scene(scene_name))
        width, depth, cx, cy = 6.0, 5.0, 0.0, 0.0
        for ent in scene.entities:
            if ent.category.value == "building":
                floor = ent.rigid.links[0].geometry["collision"][0]["size"]
                width, depth = floor[0], floor[1]
                cx, cy = ent.pose.value["p"][0], ent.pose.value["p"][1]
        span = max(width, depth)
        cams = [
            {"name": "preview_overview", "pos": [cx, cy - depth * 0.35, span * 1.1], "target": [cx, cy + depth * 0.05, 0.0],
             "fovy": 60},
            {"name": "preview_corner", "pos": [cx + width / 2 - 0.3, cy - depth / 2 + 0.3, 2.0],
             "target": [cx - width * 0.15, cy + depth * 0.15, 0.6], "fovy": 75},
        ]
        return [str(p) for p in render_scene(scene, cams, self.root / "previews", f"scene-{scene_name}")]

    def load_scene(self, name: str) -> dict:
        path = self.root / "scenes" / f"{name}.json"
        if not path.exists():
            raise AuthoringError(f"unknown scene {name!r}")
        return json.loads(path.read_text(encoding="utf-8"))

    def list_scenes(self) -> list[str]:
        return sorted(p.stem for p in (self.root / "scenes").glob("*.json"))

    def scenario_catalog(self) -> list[dict]:
        """Every scenario the host can load, with its task and the body of each robot.

        Built scenarios are read from disk; builtin scenarios not yet built in this root are
        listed from their specs (``built: false``) so the console can offer them and
        :meth:`ensure_scenario` builds them on first load.
        """
        out: dict[str, dict] = {}
        for name in self.list_scenarios():
            scenario = self.load_scenario(name)
            out[name] = {"name": name, "task": scenario["task"], "mission": scenario.get("mission"),
                         "robots": scenario_embodiments(scenario), "built": True}
        for name, builtin in BUILTIN_SCENARIOS.items():
            if name in out:
                continue
            robots = {e["id"]: describe(e["asset"]) for e in builtin["spec"]()["entities"]
                      if e["asset"] in EMBODIMENTS}
            out[name] = {"name": name, "task": builtin["task"], "mission": builtin.get("mission"),
                         "robots": robots, "built": False}
        return [out[name] for name in sorted(out)]

    def ensure_scenario(self, name: str) -> None:
        """Build a builtin scenario that is not in this root yet; anything else is left to
        :meth:`load_scenario` to find or report."""
        if name in BUILTIN_SCENARIOS and name not in self.list_scenarios():
            BUILTIN_SCENARIOS[name]["build"](self)

    # -------------------------------------------------------------- scenarios
    def save_scenario(self, scenario: dict) -> dict:
        for key in ("name", "scene", "robots", "task"):
            if key not in scenario:
                raise AuthoringError(f"scenario needs {key!r}")
        scene = self.load_scene(scenario["scene"])
        ids = {e["id"] for e in scene["entities"]}
        for rid in scenario["robots"]:
            if rid not in ids:
                raise AuthoringError(f"scenario robot {rid!r} is not in scene {scenario['scene']!r}")
        record = {"perturbationPolicy": {"allowed": [], "budget": 0}, **scenario}
        path = self.root / "scenarios" / f"{scenario['name']}.json"
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return {"name": scenario["name"], "path": str(path)}

    def load_scenario(self, name: str) -> dict:
        path = self.root / "scenarios" / f"{name}.json"
        if not path.exists():
            raise AuthoringError(f"unknown scenario {name!r}")
        scenario = json.loads(path.read_text(encoding="utf-8"))
        return scenario | {"sceneState": self.load_scene(scenario["scene"])}

    def list_scenarios(self) -> list[str]:
        return sorted(p.stem for p in (self.root / "scenarios").glob("*.json"))


def _inner_floor(model: mujoco.MjModel, data: mujoco.MjData, body_name: str, obb) -> float:
    """Height of a container's inner floor: the highest top, below the container's middle, of a
    box geom of its link that covers the container's centre."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    centre = np.asarray(obb.center, float)
    floor = obb.center[2] - obb.size[2] / 2
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != bid or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX:
            continue
        rot = data.geom_xmat[g].reshape(3, 3)
        local = rot.T @ (centre - data.geom_xpos[g])
        half = model.geom_size[g]
        top = float(data.geom_xpos[g][2] + np.abs(rot[2]) @ half)
        if abs(local[0]) <= half[0] and abs(local[1]) <= half[1] and top < centre[2]:
            floor = max(floor, top)
    return floor


def kitchen_spec() -> dict:
    """The Phase 0 kitchen: counter with a cup and a bowl, a dining table with a plate, a drawer
    cabinet, and one robot."""
    return {
        "name": "phase0-kitchen",
        "entities": [
            {"id": "w:room", "asset": "kitchen-room", "at": {"x": 0.0, "y": 0.0}},
            {"id": "w:counter-1", "asset": "counter", "at": {"x": 0.0, "y": 2.18}},
            {"id": "w:cabinet-1", "asset": "drawer-cabinet", "at": {"x": 2.70, "y": 0.6, "yaw": math.pi}},
            {"id": "w:table-1", "asset": "dining-table", "at": {"x": 0.6, "y": -0.9}},
            {"id": "w:cup-1", "asset": "cup", "on": "w:counter-1", "offset": [-0.5, 0.0]},
            {"id": "w:bowl-1", "asset": "bowl", "on": "w:counter-1", "offset": [0.45, 0.02]},
            {"id": "w:plate-1", "asset": "plate", "on": "w:table-1", "offset": [-0.3, 0.1]},
            {"id": "w:robot-1", "asset": "steward-bot", "at": {"x": -1.8, "y": -1.2, "yaw": 0.0}},
        ],
        "regions": [{"id": "kitchen", "label": "kitchen", "polygon": [[-3, -2.5], [3, -2.5], [3, 2.5], [-3, 2.5]]}],
    }


SHUTTLE_TASK = ("反复在餐桌、柜子和餐边柜之间搬运物品，循环进行：物品按「餐桌 → 柜子 → 餐边柜 → 餐桌」的顺序轮转，"
                "每次把下一站空着的那件物品搬过去，任何时候都不要在同一件家具上堆放两件被搬运的物品。")


def shuttle_spec() -> dict:
    """Kitchen for the shuttle task: a cup on the table, a bowl on the cabinet and an empty
    sideboard along the west wall, so one of the three surfaces is always free."""
    return {
        "name": "phase0-shuttle",
        "entities": [
            {"id": "w:room", "asset": "kitchen-room", "at": {"x": 0.0, "y": 0.0}},
            {"id": "w:counter-1", "asset": "counter", "at": {"x": 0.0, "y": 2.18}},
            {"id": "w:cabinet-1", "asset": "drawer-cabinet", "at": {"x": 2.70, "y": 0.6, "yaw": math.pi}},
            {"id": "w:sideboard-1", "asset": "sideboard", "at": {"x": -2.755, "y": 0.6}},
            {"id": "w:table-1", "asset": "dining-table", "at": {"x": 0.6, "y": -0.9}},
            {"id": "w:plate-1", "asset": "plate", "on": "w:counter-1", "offset": [0.35, 0.0]},
            {"id": "w:cup-1", "asset": "cup", "on": "w:table-1", "offset": [-0.25, 0.1]},
            {"id": "w:bowl-1", "asset": "bowl", "on": "w:cabinet-1", "offset": [0.0, 0.0]},
            {"id": "w:robot-1", "asset": "steward-bot", "at": {"x": -1.6, "y": -1.3, "yaw": 0.0}},
        ],
        "regions": [{"id": "kitchen", "label": "kitchen", "polygon": [[-3, -2.5], [3, -2.5], [3, 2.5], [-3, 2.5]]}],
    }


CHORES_TASK = ("餐桌上的三个颜色、大小不同的碗、一个透明高脚酒杯和一个黑色咖啡杯：先在餐桌上按从小到大的顺序排成一排；"
               "再搬到餐边柜台面上，能叠的碗叠起来，整齐摆放；然后把餐边柜抽屉里的东西和柜子抽屉里的东西对调。"
               "以上任务循环重复。")


def chores_spec() -> dict:
    """Kitchen for the chores task: five tableware items scattered on the far half of the table, a
    teddy bear and a pistol in the sideboard drawer, and a toy airplane, a pill bottle and a perfume
    bottle in the cabinet drawer."""
    return {
        "name": "phase0-chores",
        "entities": [
            {"id": "w:room", "asset": "kitchen-room", "at": {"x": 0.0, "y": 0.0}},
            {"id": "w:counter-1", "asset": "counter", "at": {"x": 0.0, "y": 2.18}},
            {"id": "w:cabinet-1", "asset": "tall-drawer-cabinet", "at": {"x": 2.70, "y": 0.6, "yaw": math.pi}},
            {"id": "w:sideboard-1", "asset": "drawer-sideboard", "at": {"x": -2.755, "y": 0.6}},
            {"id": "w:table-1", "asset": "dining-table", "at": {"x": 0.6, "y": -0.9}},
            {"id": "w:bowl-red", "asset": "bowl-red-large", "on": "w:table-1", "offset": [-0.36, -0.17]},
            {"id": "w:wine-glass", "asset": "wine-glass", "on": "w:table-1", "offset": [-0.10, -0.22]},
            {"id": "w:bowl-green", "asset": "bowl-green-medium", "on": "w:table-1", "offset": [0.12, -0.18]},
            {"id": "w:mug", "asset": "black-mug", "on": "w:table-1", "offset": [0.33, -0.24], "yaw": 0.8},
            {"id": "w:bowl-blue", "asset": "bowl-blue-small", "on": "w:table-1", "offset": [0.48, -0.12]},
            {"id": "w:teddy-bear", "asset": "teddy-bear", "in": "w:sideboard-1", "offset": [-0.02, -0.12]},
            {"id": "w:pistol", "asset": "pistol", "in": "w:sideboard-1", "offset": [0.0, 0.13]},
            {"id": "w:toy-airplane", "asset": "toy-airplane", "in": "w:cabinet-1", "offset": [0.05, -0.12]},
            {"id": "w:pill-bottle", "asset": "pill-bottle", "in": "w:cabinet-1", "offset": [-0.1, 0.14]},
            {"id": "w:perfume-bottle", "asset": "perfume-bottle", "in": "w:cabinet-1", "offset": [0.1, 0.14]},
            {"id": "w:robot-1", "asset": "steward-bot", "at": {"x": -1.6, "y": -1.3, "yaw": 0.0}},
        ],
        "regions": [{"id": "kitchen", "label": "kitchen", "polygon": [[-3, -2.5], [3, -2.5], [3, 2.5], [-3, 2.5]]}],
    }


KITCHEN_ASSETS = {
    "kitchen-room": ("building", {"width": 6.0, "depth": 5.0}),
    "counter": ("counter", {}),
    "sideboard": ("sideboard", {}),
    "drawer-cabinet": ("cabinet_with_drawer", {}),
    "dining-table": ("dining_table", {}),
    "cup": ("cup", {}),
    "bowl": ("bowl", {}),
    "plate": ("plate", {}),
    "steward-bot": ("steward_bot", {}),
    "drawer-sideboard": ("sideboard", {"drawer_width": 0.6, "drawer_height": 0.22}),
    "tall-drawer-cabinet": ("cabinet_with_drawer", {"drawer_height": 0.24}),
    # Diameters 30, 20 and 12 cm: nested, each bowl stands well clear of the wall of the one around
    # it, so every bowl can be seen and grasped by itself.
    "bowl-red-large": ("nesting_bowl", {"radius": 0.15, "height": 0.075, "mass": 0.6, "color": "red"}),
    "bowl-green-medium": ("nesting_bowl", {"radius": 0.10, "height": 0.06, "mass": 0.35, "color": "green"}),
    "bowl-blue-small": ("nesting_bowl", {"radius": 0.06, "height": 0.045, "mass": 0.18, "color": "blue"}),
    "wine-glass": ("wine_glass", {}),
    "black-mug": ("mug", {"color": "black"}),
    "teddy-bear": ("teddy_bear", {}),
    "scissors": ("scissors", {}),
    "toy-airplane": ("toy_airplane", {}),
    "pistol": ("pistol", {}),
    "pill-bottle": ("pill_bottle", {}),
    "perfume-bottle": ("perfume_bottle", {}),
    "chair": ("chair", {}),
}


IMAGES = Path(__file__).resolve().parent.parent / "assets" / "images"


@dataclass(frozen=True)
class Puzzle:
    """One picture puzzle: its photo, the subject named in the task, pieces per side and the table it lies on."""

    image: str
    subject: str
    grid: int
    table_m: tuple[float, float] = (2.0, 1.0)


# A pour spot keeps the box half side + the robot's 0.25 m pour spill from every table edge, so a 4 × 4
# box needs a table at least 1.04 m deep; the robot reaches 0.59 m in from the edge it docks at, so a
# table up to 1.18 m deep is reachable everywhere from its two long sides.
PUZZLES = {
    "cat": Puzzle("cat.jpg", "猫", 3),
    "greatwall": Puzzle("great-wall.jpg", "长城", 3),
    "dog": Puzzle("dog.jpg", "狗", 4, (2.4, 1.15)),
}
PUZZLE_PIECE_M = 0.115
PUZZLE_TASK = ("餐桌上有一盒拼图，盒盖上印着拼好后的完整图案（{subject}）。盒子上写着：共 {count} 片，"
               "拼好后 {size_cm:.1f} 厘米见方，和盒盖上印的图一样大。打开盒子，把盒子里的拼图块倒在桌上，"
               "再把这些拼图块在桌上拼成和盒盖上一样的图案，拼时每块的图案面都要朝上。")
PUZZLE_LID_BORDER_RGB = (236, 232, 224)


def png_b64(image, size: int) -> str:
    """A PIL image as a base64 PNG of ``size`` × ``size`` pixels, for a printed texture."""
    from PIL import Image

    buf = io.BytesIO()
    image.convert("RGB").resize((size, size), Image.LANCZOS).save(buf, "PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def puzzle_textures(image_file: str, lid_m: float, grid: int) -> tuple[str, list[list[str]]]:
    """The lid print and the ``grid`` × ``grid`` pieces (row 0 at the top) as textures. The lid
    shows the picture as large as the finished puzzle, centred on a plain border across ``lid_m``."""
    from PIL import Image

    image = Image.open(IMAGES / image_file).convert("RGB")
    side = min(image.size)
    step = side // grid
    pieces = [[png_b64(image.crop((c * step, r * step, (c + 1) * step, (r + 1) * step)), 128) for c in range(grid)]
              for r in range(grid)]
    lid_px = 256
    picture_px = round(lid_px * grid * PUZZLE_PIECE_M / lid_m)
    lid = Image.new("RGB", (lid_px, lid_px), PUZZLE_LID_BORDER_RGB)
    margin = (lid_px - picture_px) // 2
    lid.paste(image.crop((0, 0, grid * step, grid * step)).resize((picture_px, picture_px), Image.LANCZOS), (margin, margin))
    return png_b64(lid, lid_px), pieces


def puzzle_spec(name: str, image: str, seed: int) -> tuple[list[tuple[str, str, dict]], dict, dict]:
    """Assets, scene spec and goal for a picture puzzle: an open box with its printed lid on a table,
    the pieces lying shuffled and turned by quarter turns inside it.

    The goal states where every piece lies relative to its right and lower neighbour when the
    picture is complete (offset in the piece's frame, upright, same heading)."""
    puzzle = PUZZLES[image]
    n = puzzle.grid
    table_w, table_d = puzzle.table_m
    table = f"puzzle-table-{round(table_w * 100)}x{round(table_d * 100)}"
    box_w = n * PUZZLE_PIECE_M + (n + 1) * 0.012 + 0.02
    lid_w = box_w + 0.02
    full, pieces = puzzle_textures(puzzle.image, lid_w, n)
    assets = [
        (table, "dining_table", {"width": table_w, "depth": table_d}),
        (f"{name}-box", "open_box", {"width": box_w, "depth": box_w}),
        (f"{name}-lid", "printed_board", {"width": lid_w, "depth": lid_w, "thickness": 0.008,
                                          "texture": full, "mass": 0.08, "category": "lid"}),
    ]
    for r in range(n):
        for c in range(n):
            assets.append((f"{name}-piece-{r}{c}", "printed_board",
                           {"width": PUZZLE_PIECE_M, "depth": PUZZLE_PIECE_M, "texture": pieces[r][c],
                            "category": "puzzle_piece"}))
    rng = np.random.default_rng(seed)
    order = rng.permutation(n * n)
    turns = rng.integers(0, 4, n * n)
    pitch = PUZZLE_PIECE_M + 0.012
    middle = (n - 1) / 2
    entities = [
        {"id": "w:room", "asset": "kitchen-room", "at": {"x": 0.0, "y": 0.0}},
        {"id": "w:table-1", "asset": table, "at": {"x": 0.5, "y": -0.6}},
        {"id": "w:box", "asset": f"{name}-box", "on": "w:table-1", "offset": [0.3 - table_w / 2, 0.15]},
    ]
    for slot, k in enumerate(order):
        r, c = divmod(int(k), n)
        sr, sc = divmod(slot, n)
        entities.append({"id": f"w:piece-{r}{c}", "asset": f"{name}-piece-{r}{c}", "in": "w:box",
                         "offset": [(sc - middle) * pitch, (middle - sr) * pitch],
                         "yaw": float(turns[slot]) * math.pi / 2})
    entities += [
        {"id": "w:lid", "asset": f"{name}-lid", "on": "w:box", "offset": [0.0, 0.0]},
        {"id": "w:robot-1", "asset": "steward-bot", "at": {"x": -1.6, "y": -1.3, "yaw": 0.0}},
    ]
    constraints = []
    for r in range(n):
        for c in range(n):
            if c + 1 < n:
                constraints.append({"a": f"w:piece-{r}{c}", "b": f"w:piece-{r}{c + 1}", "offset": [PUZZLE_PIECE_M, 0.0]})
            if r + 1 < n:
                constraints.append({"a": f"w:piece-{r}{c}", "b": f"w:piece-{r + 1}{c}", "offset": [0.0, -PUZZLE_PIECE_M]})
    goal = {"kind": "relative-poses", "entities": [f"w:piece-{r}{c}" for r in range(n) for c in range(n)],
            "constraints": constraints, "tolerance": {"positionM": 0.03, "yawDeg": 15.0}, "upright": True}
    spec = {"name": name, "entities": entities,
            "regions": [{"id": "room", "label": "room", "polygon": [[-3, -2.5], [3, -2.5], [3, 2.5], [-3, 2.5]]}]}
    return assets, spec, goal


def build_puzzle_scenarios(service: AuthoringService) -> list[str]:
    """Publish the picture-puzzle assets and save one ``puzzle-<name>`` scenario per entry of ``PUZZLES``."""
    names = []
    for seed, (image, puzzle) in enumerate(PUZZLES.items()):
        name = f"puzzle-{image}"
        assets, spec, goal = puzzle_spec(name, image, seed)
        for asset, generator, params in assets:
            created = service.create_asset(generator=generator, params=params)
            if not created["validation"]["ok"]:
                raise AuthoringError(f"generated asset {asset} failed validation: {created['validation']['errors']}")
            service.publish_asset(asset, created["template"], meta={"generator": generator})
        service.compose_scene(spec)
        service.save_scenario({"name": name, "scene": name, "robots": ["w:robot-1"],
                               "task": PUZZLE_TASK.format(subject=puzzle.subject, count=puzzle.grid**2,
                                                          size_cm=puzzle.grid * PUZZLE_PIECE_M * 100),
                               "goal": goal, "phase": "phase0"})
        names.append(name)
    return names


def build_default_kitchen(service: AuthoringService) -> dict:
    """Generate, validate and publish the kitchen assets, compose the scene and save the scenario."""
    published = {}
    for name, (generator, params) in KITCHEN_ASSETS.items():
        created = service.create_asset(generator=generator, params=params)
        if not created["validation"]["ok"]:
            raise AuthoringError(f"generated asset {name} failed validation: {created['validation']['errors']}")
        published[name] = service.publish_asset(name, created["template"], meta={"generator": generator, "params": params})
    scene = service.compose_scene(kitchen_spec())
    scenario = service.save_scenario({
        "name": "phase0-kitchen",
        "scene": "phase0-kitchen",
        "robots": ["w:robot-1"],
        "task": "把杯子放到桌上。",
    })
    service.compose_scene(shuttle_spec())
    service.save_scenario({
        "name": "phase0-shuttle",
        "scene": "phase0-shuttle",
        "robots": ["w:robot-1"],
        "task": SHUTTLE_TASK,
        "shuttle": {"supports": ["table", "cabinet", "sideboard"], "objects": ["cup", "bowl"]},
    })
    service.compose_scene(chores_spec())
    service.save_scenario({
        "name": "phase0-chores",
        "scene": "phase0-chores",
        "robots": ["w:robot-1"],
        "task": CHORES_TASK,
    })
    puzzles = build_puzzle_scenarios(service)
    return {"assets": {k: v["version"][:12] for k, v in published.items()}, "scene": scene, "scenario": scenario,
            "puzzles": puzzles}


def cobot_spec() -> dict:
    """The Cobot Magic workstation: the twin parked at a tabletop, a bowl and a plate on it.

    The table's near edge sits at x = 0.40, just in front of the chassis (half-length 0.34), so
    the twin works the band its arms can reach over the edge (measured: x 0.40-0.60 at y +-0.45,
    strictly vertical only in the nearest part of it). The bowl and the plate stand on the same
    arm's side: the errand is one arm's job, pick and place both.
    """
    return {
        "name": "cobot-table",
        "entities": [
            {"id": "w:room", "asset": "cobot-room", "at": {"x": 0.0, "y": 0.0}},
            {"id": "w:table-1", "asset": "cobot-table", "at": {"x": 0.85, "y": 0.0}},
            {"id": "w:cobot-1", "asset": "cobot-magic", "at": {"x": 0.0, "y": 0.0}},
            # Table centre is (0.85, 0); both pieces stand on the left arm's side (the mission is
            # one arm's errand), fully on the tabletop and clear of each other's footprint --
            # verified against the solver, see tests/test_cobot_table.py.
            {"id": "w:bowl-1", "asset": "bowl", "on": "w:table-1", "offset": [-0.35, 0.20]},
            {"id": "w:plate-1", "asset": "plate", "on": "w:table-1", "offset": [-0.31, 0.42]},
        ],
        "regions": [{"id": "workcell", "label": "workcell", "polygon": [[-2, -2], [2, -2], [2, 2], [-2, 2]]}],
    }


COBOT_TASK = "把碗搬到盘子上；一次只用一个夹爪，搬完再回到停放姿态。"


def cobot_kitchen_spec() -> dict:
    """The chores kitchen's tableware on one table the twin is parked at.

    The same three nesting bowls (30, 20 and 12 cm), wine glass and black mug as the chores
    kitchen. The table's near edge sits at x = 0.38, just in front of the chassis; every piece
    stands in the strip the arms reach over that edge (measured: x 0.38-0.60, and y +-0.25 puts a
    piece in front of one arm rather than between both). The largest bowl stands on the left
    arm's line because stacking happens along that line, the other two on the arms that carry
    them.
    """
    return {
        "name": "cobot-kitchen",
        "entities": [
            {"id": "w:room", "asset": "kitchen-room", "at": {"x": 0.0, "y": 0.0}},
            {"id": "w:cobot-1", "asset": "cobot-magic", "at": {"x": 0.0, "y": 0.0}},
            {"id": "w:table-1", "asset": "cobot-kitchen-table", "at": {"x": 0.90, "y": 0.0}},
            # Every piece is fully on the tabletop (near edge x = 0.35) and inside the strip the
            # arms reach over it (x 0.40-0.60): a cylinder resting on one contact point tips.
            # The largest bowl is the stack point and both arms reach it; the others sit on their
            # own arm's side, spaced by their radii.
            {"id": "w:bowl-red", "asset": "bowl-red-large", "on": "w:table-1", "offset": [-0.42, 0.00]},
            {"id": "w:bowl-green", "asset": "bowl-green-medium", "on": "w:table-1", "offset": [-0.45, 0.38]},
            {"id": "w:bowl-blue", "asset": "bowl-blue-small", "on": "w:table-1", "offset": [-0.44, -0.28]},
            # The glass and the mug are left alone by this task: they stand on the far half, out of
            # the arms' strip but inside the overhead camera's view.
            {"id": "w:wine-glass", "asset": "wine-glass", "on": "w:table-1", "offset": [0.20, -0.30]},
            {"id": "w:mug", "asset": "black-mug", "on": "w:table-1", "offset": [0.20, 0.30]},
        ],
        "regions": [{"id": "kitchen", "label": "kitchen", "polygon": [[-3, -2.5], [3, -2.5], [3, 2.5], [-3, 2.5]]}],
    }


COBOT_KITCHEN_TASK = "把桌上三个颜色不同的碗按大小叠起来；高脚杯和咖啡杯留在桌上。"


def _publish_assets(service: AuthoringService, assets: dict[str, tuple[str, dict]]) -> dict[str, str]:
    """Generate, validate and publish a set of generated assets."""
    published = {}
    for name, (generator, params) in assets.items():
        created = service.create_asset(generator=generator, params=params)
        if not created["validation"]["ok"]:
            raise AuthoringError(f"generated asset {name} failed validation: {created['validation']['errors']}")
        published[name] = service.publish_asset(name, created["template"],
                                                meta={"generator": generator, "params": params})
    return published


KITCHEN_TASK = "把杯子放到桌上。"

def build_cobot_workstation(service: AuthoringService) -> dict:
    """Publish the Cobot Magic twin's assets, compose its workcell and save the cobot-table scenario."""
    from steward.assets.cobot import cobot_magic

    # The twin's generator lives in its own module and imports the shared generators' constants,
    # so it is registered here rather than at import time (that would be a cycle).
    GENERATORS.setdefault("cobot_magic", cobot_magic)
    assets = {
        "cobot-room": ("building", {"width": 4.0, "depth": 4.0, "height": 2.6}),
        "cobot-table": ("dining_table", {"width": 1.0, "depth": 0.9}),
        "cobot-magic": ("cobot_magic", {}),
        "bowl": KITCHEN_ASSETS["bowl"],
        "plate": KITCHEN_ASSETS["plate"],
    }
    published = _publish_assets(service, assets)
    scene = service.compose_scene(cobot_spec())
    scenario = service.save_scenario({
        "name": "cobot-table",
        "scene": "cobot-table",
        "robots": ["w:cobot-1"],
        "task": COBOT_TASK,
        # The scripted mission the twin's Ego Runtime runs in this scene (`steward cobot-ego --task`).
        "mission": "bowl-on-plate",
        # Frame the console on the workstation: both arms and both pieces of tableware fill the
        # view, which is the only way a three-centimetre grasp is visible on the Ego-State panel.
        "view": {"lookat": [0.30, 0.14, 0.80], "distance": 1.9, "azimuth": 90.0, "elevation": -35.0},
    })
    return {"assets": {k: v["version"][:12] for k, v in published.items()}, "scene": scene, "scenario": scenario}


def build_cobot_kitchen(service: AuthoringService) -> dict:
    """Put the twin in the chores kitchen: the coloured bowls on one table within its reach."""
    from steward.assets.cobot import cobot_magic

    GENERATORS.setdefault("cobot_magic", cobot_magic)
    assets = {
        "kitchen-room": KITCHEN_ASSETS["kitchen-room"],
        "cobot-magic": ("cobot_magic", {}),
        # 0.8 x 0.8 under the overhead camera's cone: the whole top has to be inside the frame,
        # or the table never becomes a track (see assets/cobot.py's top_camera note).
        "cobot-kitchen-table": ("dining_table", {"width": 1.2, "depth": 1.0}),
        "bowl-red-large": KITCHEN_ASSETS["bowl-red-large"],
        "bowl-green-medium": KITCHEN_ASSETS["bowl-green-medium"],
        "bowl-blue-small": KITCHEN_ASSETS["bowl-blue-small"],
        "wine-glass": KITCHEN_ASSETS["wine-glass"],
        "black-mug": KITCHEN_ASSETS["black-mug"],
    }
    published = _publish_assets(service, assets)
    scene = service.compose_scene(cobot_kitchen_spec())
    scenario = service.save_scenario({
        "name": "cobot-kitchen",
        "scene": "cobot-kitchen",
        "robots": ["w:cobot-1"],
        "task": COBOT_KITCHEN_TASK,
        "mission": "nest-bowls",
        # Frame the console on the twin and its table: the arms are 0.5 m tall and the grasp is
        # 3 cm, which is invisible from the room-sized default.
        "view": {"lookat": [0.32, 0.10, 0.80], "distance": 1.9, "azimuth": 90.0, "elevation": -35.0},
    })
    return {"assets": {k: v["version"][:12] for k, v in published.items()}, "scene": scene, "scenario": scenario}


# The errand's map: where the twin stands to work each station, the route it drives between them,
# and where the bowl goes. The route leaves the table's near edge and comes back to the counter
# along the counter's own centre line -- the last turn is at x = 2.0, where the chassis's swept
# corner (0.45 m) still clears the counter's front at 2.70, and the last 0.30 m is a straight
# drive into the working pose, which is how a real base parks against a counter.
COBOT_ERRAND_MISSION = {
    "drive": {"at": [2.30, 0.0, 0.0],
              "path": [[-0.20, -1.10], [2.00, -1.10], [2.00, 0.0], [2.30, 0.0]]},
    # The counter's top (0.90) plus the bowl's half height and the 3 cm it is released above it.
    "place": [2.80, 0.0, 0.96],
    # The counter is 15 cm taller than the table: the torso rises before the place.
    "lift": 0.78,
    "grip": "L_link6",
}


def cobot_errand_spec() -> dict:
    """Two stations the twin has to drive between: the dining table and a kitchen counter.

    Nothing here is within reach of the other: the table's bowl is 0.46 m from the start pose,
    the counter's drop point is 2.8 m away and 15 cm higher, so the mission can only succeed by
    carrying the bowl across the room and raising the torso.
    """
    return {
        "name": "cobot-errand",
        "entities": [
            {"id": "w:room", "asset": "cobot-errand-room", "at": {"x": 0.0, "y": 0.0}},
            {"id": "w:cobot-1", "asset": "cobot-magic", "at": {"x": 0.0, "y": 0.0}},
            {"id": "w:table-1", "asset": "cobot-errand-table", "at": {"x": 0.90, "y": 0.0}},
            # The bowl stands in the same reachable band as the kitchen: 0.46 m ahead, in front of
            # the left arm, fully on the tabletop.
            {"id": "w:bowl-1", "asset": "bowl", "on": "w:table-1", "offset": [-0.44, 0.20]},
            # Turned a quarter turn so its 1.8 m front faces the robot: you park along a counter,
            # not at its end.
            {"id": "w:counter-1", "asset": "cobot-errand-counter",
             "at": {"x": 3.0, "y": 0.0, "yaw": 1.5707963267948966}},
        ],
        "regions": [{"id": "flat", "label": "flat", "polygon": [[-4, -2.5], [4, -2.5], [4, 2.5], [-4, 2.5]]}],
    }


COBOT_ERRAND_TASK = "把桌上的碗搬到对面的台面上：自己开过去，到了再把升降柱升起来。"


def build_cobot_errand(service: AuthoringService) -> dict:
    """Publish the two stations and save the cobot-errand scenario (one bowl, two stations)."""
    from steward.assets.cobot import cobot_magic

    GENERATORS.setdefault("cobot_magic", cobot_magic)
    assets = {
        # Long enough for a 2.3 m drive around the table with room to spare at both ends.
        "cobot-errand-room": ("building", {"width": 8.0, "depth": 5.0, "height": 2.6}),
        "cobot-magic": ("cobot_magic", {}),
        "cobot-errand-table": ("dining_table", {"width": 1.2, "depth": 1.0}),
        "cobot-errand-counter": ("counter", {"width": 1.8, "depth": 0.6, "height": 0.9}),
        "bowl": KITCHEN_ASSETS["bowl"],
    }
    published = _publish_assets(service, assets)
    scene = service.compose_scene(cobot_errand_spec())
    scenario = service.save_scenario({
        "name": "cobot-errand",
        "scene": "cobot-errand",
        "robots": ["w:cobot-1"],
        "task": COBOT_ERRAND_TASK,
        "mission": "carry-to-counter",
        # The map the mission is scripted against; stations belong to the deployment, not the code.
        "missionConfig": COBOT_ERRAND_MISSION,
        # From behind the start pose: the table, the robot and the route to the counter in one view.
        "view": {"lookat": [1.4, 0.0, 0.70], "distance": 3.6, "azimuth": 90.0, "elevation": -40.0},
    })
    return {"assets": {k: v["version"][:12] for k, v in published.items()}, "scene": scene, "scenario": scenario}


# The scenarios the CLI and the console can build on demand, with what the catalog says about
# them before they exist in a root. ``build`` may write several scenarios at once (the kitchen
# builder saves all three Phase 0 scenarios), so the table names each one it produces.
BUILTIN_SCENARIOS: dict[str, dict] = {
    "phase0-kitchen": {"build": build_default_kitchen, "spec": kitchen_spec, "task": KITCHEN_TASK},
    "phase0-shuttle": {"build": build_default_kitchen, "spec": shuttle_spec, "task": SHUTTLE_TASK},
    "phase0-chores": {"build": build_default_kitchen, "spec": chores_spec, "task": CHORES_TASK},
    "cobot-table": {"build": build_cobot_workstation, "spec": cobot_spec, "task": COBOT_TASK,
                    "mission": "bowl-on-plate"},
    "cobot-kitchen": {"build": build_cobot_kitchen, "spec": cobot_kitchen_spec, "task": COBOT_KITCHEN_TASK,
                      "mission": "nest-bowls"},
    "cobot-errand": {"build": build_cobot_errand, "spec": cobot_errand_spec, "task": COBOT_ERRAND_TASK,
                     "mission": "carry-to-counter"},
}
