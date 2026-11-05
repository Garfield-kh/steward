"""Observer: compares each robot's Ego-State with the World State.

Telemetry arrives one-way from robots. Each snapshot is first checked with ``validate_ego``; a
snapshot carrying World State identifiers is recorded as a leak and not scored. Scoring
associates ES entities to World State entities (or environment links) by nearest oriented
bounding box center within a gate, ignoring category so that category accuracy can be measured,
then reports recall, false positives, position error, NEES and self-localisation error. Results
never flow back to robots.

Gross perception faults are watched for on every snapshot: a large ES object with no counterpart
in the world (a phantom, such as a concave wall corner filled in as a solid box) and the robot
standing inside an ES object. A fault seen in ``ANOMALY_REPEATS`` snapshots in a row is reported
once to ``on_anomaly`` with an analysis of what the world really holds there.
"""

from __future__ import annotations

import json
import math
import threading
from pathlib import Path

import numpy as np

from steward.math3d import yaw_from_quat
from steward.scene.validate import validate_ego


PHANTOM_FOOTPRINT_M = 1.0
STALE_S = 5.0  # an ES object not seen for this long is remembered, not perceived: a phantom only counts
# while the robot keeps seeing it (an object moved behind its back is stale memory, not a fault)
ANOMALY_REPEATS = 3
NAMES_ZH = {"wall": "墙", "table": "餐桌", "counter": "操作台", "cabinet": "柜子", "sideboard": "餐边柜", "chair": "椅子",
            "bowl": "碗", "mug": "咖啡杯", "wine_glass": "高脚杯", "plate": "盘子", "drawer": "抽屉", "plant": "绿植",
            "unknown": "未知物体"}


class Observer:
    def __init__(self, sim_provider, out_dir: Path | None = None, gate_m: float = 0.35, phase=None, on_anomaly=None):
        self._sim = sim_provider
        self.on_anomaly = on_anomaly
        self._streaks: dict[str, int] = {}
        self._reported: set[str] = set()
        self._phase = phase
        self.out_dir = out_dir
        self.gate_m = gate_m
        self._lock = threading.Lock()
        self.latest: dict[str, dict] = {}
        self.history: dict[str, list[dict]] = {}
        self.latest_ego: dict[str, dict] = {}
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)

    def _ground_truth(self, robot_id: str) -> tuple[list[dict], dict]:
        sim = self._sim()
        gt = []
        robot_pose = None
        for ent in sim.template.entities:
            if ent.id == robot_id:
                ws = sim.world_state()
                robot_pose = next(e for e in ws["entities"] if e["id"] == robot_id)["pose"]["value"]
                continue
            if ent.role == "robot":
                continue
            for part, links, category in sim.perception_parts(ent):
                obb = sim.entity_obb(ent.id, links)
                gt.append({"id": part, "category": category, "center": obb.center, "size": obb.size,
                           "env": category == "wall"})
        return gt, robot_pose

    def ingest(self, robot_id: str, ego: dict) -> dict:
        errors = validate_ego(ego)
        sim = self._sim()
        record: dict = {"robot": robot_id, "t": sim.time, "esVersion": ego.get("version")}
        if self._phase is not None:
            record["phase"] = self._phase()
        if errors:
            record["leak"] = errors[:5]
            self._store(robot_id, record)
            return record
        with self._lock:
            self.latest_ego[robot_id] = ego
        gt, robot_pose = self._ground_truth(robot_id)
        es_objs = [e for e in ego["entities"] if e["id"] != ego.get("self")]
        pairs = []
        for i, e in enumerate(es_objs):
            for j, g in enumerate(gt):
                d = float(np.linalg.norm(np.asarray(e["pose"]["value"]["p"]) - np.asarray(g["center"])))
                if d <= (self.gate_m * 2 if g["env"] else self.gate_m):
                    mismatch = 0.0 if e["category"]["value"] == g["category"] else 0.15
                    pairs.append((d + mismatch, i, j))
        pairs.sort()
        used_e, used_g, matches = set(), set(), []
        for _, i, j in pairs:
            if i in used_e or j in used_g:
                continue
            used_e.add(i)
            used_g.add(j)
            d = float(np.linalg.norm(np.asarray(es_objs[i]["pose"]["value"]["p"]) - np.asarray(gt[j]["center"])))
            matches.append((es_objs[i], gt[j], d))
        obj_matches = [m for m in matches if not m[1]["env"]]
        gt_objects = [g for g in gt if not g["env"]]
        es_objects = [e for e in es_objs if e["role"] != "environment"]
        nees = []
        for e, g, _ in obj_matches:
            cov = e["pose"].get("unc", {}).get("cov")
            if cov:
                err = np.asarray(e["pose"]["value"]["p"]) - np.asarray(g["center"])
                var = np.array([cov[0], cov[7], cov[14]])
                nees.append(float(np.sum(err**2 / np.maximum(var, 1e-12))))
        pos_errors = [d for _, _, d in obj_matches]
        size_errors = [_size_error(e, g) for e, g, _ in obj_matches]
        record.update({
            "gtObjects": len(gt_objects),
            "esObjects": len(es_objects),
            "matchedObjects": len(obj_matches),
            "recall": round(len(obj_matches) / len(gt_objects), 4) if gt_objects else None,
            "falsePositives": len(es_objects) - len(obj_matches),
            "meanPositionErrorM": round(float(np.mean(pos_errors)), 4) if pos_errors else None,
            "maxPositionErrorM": round(float(np.max(pos_errors)), 4) if pos_errors else None,
            "meanSizeErrorM": round(float(np.mean(size_errors)), 4) if size_errors else None,
            "meanNees3": round(float(np.mean(nees)), 3) if nees else None,
            "categoryAccuracy": round(sum(e["category"]["value"] == g["category"] for e, g, _ in obj_matches)
                                      / len(obj_matches), 4) if obj_matches else None,
            "wallsMatched": len(matches) - len(obj_matches),
            "matches": {e["id"]: g["id"] for e, g, _ in matches},
        })
        anomalies = _anomalies(es_objs, {e["id"] for e, _, _ in matches}, gt, robot_pose, ego.get("stamp"))
        if anomalies:
            record["anomalies"] = [{k: a[k] for k in ("key", "kind", "entity")} for a in anomalies]
        self._watch(robot_id, anomalies)
        me = next((e for e in ego["entities"] if e["id"] == ego.get("self")), None)
        if me is not None and robot_pose is not None:
            p, q = me["pose"]["value"]["p"], me["pose"]["value"]["q"]
            record["selfPositionErrorM"] = round(math.dist(p[:2], robot_pose["p"][:2]), 4)
            dyaw = (yaw_from_quat(q) - yaw_from_quat(robot_pose["q"]) + math.pi) % (2 * math.pi) - math.pi
            record["selfYawErrorRad"] = round(abs(dyaw), 4)
        self._store(robot_id, record)
        return record

    def _watch(self, robot_id: str, anomalies: list[dict]) -> None:
        """Count consecutive snapshots per fault and report each fault once when it persists."""
        seen = {f"{robot_id}:{a['key']}": a for a in anomalies}
        with self._lock:
            self._streaks = {k: self._streaks.get(k, 0) + 1 for k in seen}
            due = [a for k, a in seen.items() if self._streaks[k] >= ANOMALY_REPEATS and k not in self._reported]
            self._reported.update(f"{robot_id}:{a['key']}" for a in due)
        if self.on_anomaly is not None:
            for anomaly in due:
                self.on_anomaly(robot_id, anomaly)

    def _store(self, robot_id: str, record: dict) -> None:
        with self._lock:
            self.latest[robot_id] = record
            self.history.setdefault(robot_id, []).append(record)
            if self.out_dir is not None:
                with (self.out_dir / "observer.jsonl").open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record) + "\n")

    def reset(self) -> None:
        """Forget all records, for example when a new episode starts."""
        with self._lock:
            self.latest.clear()
            self.history.clear()
            self.latest_ego.clear()
            self._streaks.clear()
            self._reported.clear()

    def report(self) -> dict:
        with self._lock:
            out = {}
            for rid, hist in self.history.items():
                scored = [h for h in hist if "recall" in h]
                first = next((h for h in scored if h.get("meanPositionErrorM") is not None), None)
                out[rid] = {
                    "snapshots": len(hist),
                    "leaks": sum(1 for h in hist if "leak" in h),
                    "first": _brief(first),
                    "latest": _brief(scored[-1] if scored else None),
                    "maxRecall": max((h["recall"] or 0) for h in scored) if scored else None,
                }
            return out


def _box(ent: dict) -> tuple[list[float], list[float], float]:
    size = ent["body"]["links"][0]["geometry"]["collision"][0]["size"]
    return ent["pose"]["value"]["p"], size, yaw_from_quat(ent["pose"]["value"]["q"])


def _within(centre: list[float], size: list[float], yaw: float, xy: list[float], margin: float = 0.0) -> bool:
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = xy[0] - centre[0], xy[1] - centre[1]
    return abs(c * dx + s * dy) <= size[0] / 2 + margin and abs(-s * dx + c * dy) <= size[1] / 2 + margin


def _anomalies(es_objs: list[dict], matched: set[str], gt: list[dict], robot_pose: dict | None,
               stamp: float | None = None) -> list[dict]:
    """Gross faults of one Ego-State against the world, each with an analysis for the operator. A
    phantom counts only while the robot still sees it (last seen within ``STALE_S`` of ``stamp``)."""
    out = []
    for ent in es_objs:
        if ent.get("role") == "environment":
            continue
        centre, size, yaw = _box(ent)
        cat = ent["category"]["value"]
        name = f"{NAMES_ZH.get(cat, cat)} {ent['id']}"
        footprint = max(size[0], size[1])
        if robot_pose is not None and _within(centre, size, yaw, robot_pose["p"], -0.05):
            out.append({"key": f"inside:{ent['id']}", "kind": "robot-inside", "entity": ent["id"],
                        "message": f"机器人真实位置 ({robot_pose['p'][0]:.2f}, {robot_pose['p'][1]:.2f}) 落在 ES 的{name}内部",
                        "analysis": f"机器人以为自己站在{name}（{size[0]:.2f}×{size[1]:.2f}×{size[2]:.2f} 米）里面，"
                                    "导航与停靠规划会全部失败；" + _contents_analysis(centre, size, gt)})
        seen = ent["pose"].get("t")
        fresh = stamp is None or seen is None or stamp - seen <= STALE_S
        if ent["id"] not in matched and footprint >= PHANTOM_FOOTPRINT_M and fresh:
            out.append({"key": f"phantom:{ent['id']}", "kind": "phantom", "entity": ent["id"],
                        "message": f"ES 的{name}（{size[0]:.2f}×{size[1]:.2f}×{size[2]:.2f} 米，在 ({centre[0]:.2f}, {centre[1]:.2f})）"
                                   "在世界中没有对应物体",
                        "analysis": _contents_analysis(centre, size, gt)})
    return out


def _contents_analysis(centre: list[float], size: list[float], gt: list[dict]) -> str:
    """What the world really holds where an ES box is, and the likely cause of the box."""
    inside = [g for g in gt if abs(g["center"][0] - centre[0]) <= (size[0] + g["size"][0]) / 2
              and abs(g["center"][1] - centre[1]) <= (size[1] + g["size"][1]) / 2]
    walls = [g for g in inside if g["env"]]
    objects = [g for g in inside if not g["env"]]
    names = "、".join(g["id"].removeprefix("w:") for g in inside) or "什么也没有"
    if len(walls) >= 2:
        cause = "两面墙在墙角的点云被并成了一个实心盒子：凹进去的墙角被当成了凸出的立方体"
    elif walls and objects:
        cause = "墙面的点云和旁边物体的点云被并成了一个盒子"
    elif len(objects) >= 2:
        cause = "几个相邻物体的点云被并成了一个盒子"
    elif not inside:
        cause = "深度噪声或错误关联生成了不存在的物体"
    else:
        cause = "这个物体的尺寸被严重高估"
    return f"该范围内实际有：{names}。可能原因：{cause}。"


def _brief(rec: dict | None) -> dict | None:
    if rec is None:
        return None
    keys = ("t", "recall", "falsePositives", "meanPositionErrorM", "meanSizeErrorM", "meanNees3", "categoryAccuracy",
            "selfPositionErrorM")
    return {k: rec.get(k) for k in keys}


def _size_error(es_entity: dict, gt: dict) -> float:
    """Mean absolute extent error (metres) between an ES box and the true box, axes matched by
    size order. The ES box carries a 2-sigma margin on x and y, which is removed first."""
    size = list(es_entity["body"]["links"][0]["geometry"]["collision"][0]["size"])
    cov = (es_entity["pose"].get("unc") or {}).get("cov")
    if cov:
        margin = 2.0 * math.sqrt(max((cov[0] + cov[7]) / 2, 0.0))
        size[0], size[1] = max(size[0] - margin, 0.0), max(size[1] - margin, 0.0)
    return float(np.mean(np.abs(np.sort(size) - np.sort(np.asarray(gt["size"], float)))))
