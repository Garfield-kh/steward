"""Real remote segmentation for WSim's RGB-D sensor (opt-in, Phase 1).

``STEWARD_SEGMENTER_URL`` points WSim at a class-agnostic instance-segmentation API (the
steward_perception service on spark: SAM3 prompted with a grid of points). While enabled, a
worker thread replaces the synthetic anonymous masks: it renders the robot's head-camera
frame, prompts the API with a uniform grid plus depth-guided points, drops ground-plane and
camera-near instances (the synthetic channel excludes the floor and the robot itself), and
publishes the frame as the next sensor snapshot. ``/sensors/rgbd`` long-polls: every frame is
served exactly once, so the robot's perception loop runs at the segmenter's throughput (about
a second per frame) instead of 10 Hz — the honest behavior of a robot whose real segmenter is
slow. Until the first real frame exists, or whenever the API call fails, the freshly rendered
frame keeps its synthetic mask (each failure is logged to stderr).
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import sys
import threading
import time

import numpy as np
from PIL import Image

from steward.math3d import matrix_from_quat
from steward.net import call

GRID_COLS, GRID_ROWS = 8, 6  # uniform prompt grid over the image
DEPTH_POINTS = 8  # extra prompts on surfaces at distinct depths
DEPTH_GAP_M = 0.25  # depth separation for two prompts to count as distinct surfaces
LAYER_M = 0.05  # depth bin for foreground proposals
PROTRUSION_M = 0.06  # how much nearer than its surrounding ring a foreground region must be
MAX_BOXES = 8
MIN_PIXELS = 25  # matches the synthetic channel's smallest instance
NEAR_RANGE_M = 0.30  # closer than this nothing is a reachable object, whatever the segmenter says
SELF_SHARE = 0.5  # share of an instance's pixels inside the robot's own projected silhouette
GROUND_Z_M = 0.06  # world height under which a large flat instance is the floor
GROUND_SHARE = 0.05  # floor candidates must also cover this share of the image
PLANE_SLAB_M = 0.06  # half-thickness of the world-z slab a surface's points concentrate in
PLANE_FLAT_SHARE = 0.6  # share of an instance's points that must sit in that slab
PLANE_SPAN_M = 0.30  # horizontal span that makes a flat instance furniture-sized
PLANE_MINOR_M = 0.35  # ... and it must be wide across BOTH horizontal axes: a bowl rim is
#     0.3 m across at most, so a round flat thing never counts as furniture
PLANE_MIN_Z_M = 0.15  # below this a flat piece is floor, which _filter drops anyway
PLANE_LEVEL_M = 0.05  # height tolerance for two pieces of one surface
PLANE_GAP_M = 0.25  # world-XY gap under which two level pieces are one surface
CARVE_LIFT_M = 0.03  # depth noise is ~1 cm; anything this far above a surface's level stands on it
HOST_MAX_FOOTPRINT_M = 0.35  # only tableware-sized regions can receive carved pixels
CARVE_FLAT_SHARE = 0.45  # a surface mixed with what stands on it is flatter than a wall (0.1-0.2)
FACE_MIN_HEIGHT_M = 0.30  # a region this tall is a front panel, not tableware (a wine glass is 0.19)
FACE_MIN_WIDTH_M = 0.40  # ... and this wide along the surface's edge is furniture, not a leg or a glass
FACE_MAX_THICK_M = 0.25  # ... and this thin across one horizontal axis is a panel, not a box
FACE_TOP_TOL_M = 0.10  # how far a panel's top may sit from the surface level it belongs to
FACE_GAP_M = 0.15  # how close the panel must stand to the surface's footprint
# Geometric surface extraction from the depth image itself (independent of the segmenter).
GEO_HORIZONTAL_NZ = 0.85  # |normal_z| above which a pixel belongs to a horizontal surface
GEO_VERTICAL_NZ = 0.35  # |normal_z| below which a pixel belongs to a vertical panel
GEO_JOIN_M = 0.035  # 3D gap across which two neighbouring pixels are one surface
GEO_FACE_JOIN_M = 0.10  # looser for panels: their depth is noisy at grazing angles
GEO_MIN_AREA_PX = 1200  # smallest surface worth publishing (a tabletop seen from afar)
GEO_MIN_SPAN_M = 0.50  # a surface must be this wide ...
GEO_MIN_MINOR_M = 0.30  # ... and this deep to be furniture rather than a rim
GEO_FACE_MIN_HEIGHT_M = 0.25  # a panel must be this tall ...
GEO_FACE_MIN_WIDTH_M = 0.40  # ... and this wide ...
GEO_FACE_TOP_TOL_M = 0.10  # ... and its top must meet the surface level this closely
GEO_FACE_GAP_M = 0.15  # ... and it must stand this close to the surface's footprint
GEO_FREE_ID = 100  # ids from here up belong to geometry-made instances
LEG_MAX_SPAN_M = 0.30  # a thing under a tabletop no wider than this is one of its legs
SLIVER_MAX_MINOR_M = 0.035  # a real thing is never this thin across while much longer


class RealSegmenter:
    """Serves a robot's RGB-D frames with masks from a remote class-agnostic segmenter."""

    def __init__(self, sim, robot_id: str, url: str):
        self._sim = sim
        self._robot = robot_id
        self._url = url.rstrip("/")
        self._stop = threading.Event()
        self._cond = threading.Condition()
        self._frame: bytes | None = None  # npz bytes to serve next
        self._stamp = -1.0  # sim time stamp of _frame
        self._served = -1.0  # stamp already handed out (long-poll dedupe)
        self._failures = 0
        self._thread: threading.Thread | None = None
        self._estimator = None  # built on first frame: the robot's kinematic self-model
        self._camera_id: str | None = None

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name=f"realseg-{self._robot}",
                                            daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()

    # ------------------------------------------------------------------ serving

    def get(self, timeout: float = 8.0) -> bytes | None:
        """Next unpublished frame, waiting up to ``timeout``; None before the first frame."""
        with self._cond:
            if self._frame is None:
                return None
            deadline = time.monotonic() + timeout
            while self._stamp <= self._served + 1e-9 and not self._stop.is_set():
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                self._cond.wait(min(left, 0.5))
            self._served = self._stamp
            return self._frame

    # ------------------------------------------------------------------ worker

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                blob = self._sim._rgbd_render(self._robot)
                frame = np.load(io.BytesIO(blob))
                rgb, depth_mm, fovy = frame["rgb"], frame["depth"], float(frame["fovy"])
                header = json.loads(str(frame["header"]))
                points = self._points(depth_mm)
                boxes = self._boxes(depth_mm)
                labels, instances = self._segment(rgb, points, boxes)
                with self._sim.lock:
                    pose, _sensor = self._sim.camera_pose(self._robot)
                labels, instances = self._merge_furniture(labels, instances, depth_mm, fovy, pose)
                labels, instances = self._fuse_geometry(labels, instances, depth_mm, fovy, pose)
                labels = self._carve_standing(labels, instances, depth_mm, fovy, pose)
                mask = self._filter(labels, instances, depth_mm, fovy, pose)
                out = io.BytesIO()
                np.savez_compressed(out, rgb=rgb, depth=depth_mm, mask=mask,
                                    fovy=np.float32(fovy), header=np.array(json.dumps(header)))
                self._publish(out.getvalue(), float(header["stamp"]))
                if self._failures:
                    print(f"realseg: segmenter recovered after {self._failures} failures",
                          file=sys.stderr, flush=True)
                    self._failures = 0
            except Exception as exc:
                self._failures += 1
                print(f"realseg: segmenter call failed ({self._failures}): {exc}",
                      file=sys.stderr, flush=True)
                # Keep the robot perceiving: publish the frame with its synthetic mask.
                try:
                    self._publish(blob, float(header["stamp"]))
                except Exception:
                    pass
                self._stop.wait(1.0)

    def _publish(self, blob: bytes, stamp: float) -> None:
        with self._cond:
            if stamp < self._stamp - 1e-6:
                # Episode reset: the sim clock went backwards, serve the new episode afresh.
                self._served = -1.0
            self._frame, self._stamp = blob, stamp
            self._cond.notify_all()

    # ------------------------------------------------------------------ prompts

    def _points(self, depth_mm: np.ndarray) -> list[list[float]]:
        """Uniform grid plus one prompt per depth-distinct surface."""
        h, w = depth_mm.shape
        points = [[w * (i + 0.5) / GRID_COLS, h * (j + 0.5) / GRID_ROWS]
                  for j in range(GRID_ROWS) for i in range(GRID_COLS)]
        d = depth_mm.astype(np.float32) / 1000.0
        cells = []
        for j in range(GRID_ROWS * 2):
            for i in range(GRID_COLS * 2):
                ys, xs = np.mgrid[j * h // (GRID_ROWS * 2):(j + 1) * h // (GRID_ROWS * 2),
                                  i * w // (GRID_COLS * 2):(i + 1) * w // (GRID_COLS * 2)]
                cd = d[ys, xs]
                valid = (cd > 0.05) & (cd < 6.0)
                if int(valid.sum()) < 40:
                    continue
                cells.append((float(np.median(cd[valid])), int(xs[valid].mean()),
                              int(ys[valid].mean())))
        picked: list[tuple[float, int, int]] = []
        for med, vx, vy in sorted(cells):
            if all(abs(med - m) > DEPTH_GAP_M for m, _, _ in picked):
                picked.append((med, vx, vy))
            if len(picked) >= DEPTH_POINTS:
                break
        points.extend([[float(vx), float(vy)] for _, vx, vy in picked])
        return points

    def _boxes(self, depth_mm: np.ndarray) -> list[list[float]]:
        """Boxes around depth protrusions: small regions of one depth layer whose surroundings
        sit distinctly farther — an object resting on (or attached to) a surface behind it."""
        from collections import deque

        h, w = depth_mm.shape
        d = depth_mm.astype(np.float32) / 1000.0
        valid = (d > 0.05) & (d < 6.0)
        if int(valid.sum()) < 200:
            return []
        layer = np.where(valid, (d / LAYER_M).astype(np.int32), -1)
        boxes: list[list[float]] = []
        for value in np.unique(layer):
            if value < 0:
                continue
            comp_mask = layer == value
            if int(comp_mask.sum()) > 0.05 * h * w:
                continue  # large surfaces (walls, table tops) are not proposals
            visited = np.zeros((h, w), dtype=bool)
            for y0, x0 in zip(*np.nonzero(comp_mask)):
                if visited[y0, x0]:
                    continue
                pixels: list[tuple[int, int]] = []
                pending = deque([(int(y0), int(x0))])
                visited[y0, x0] = True
                while pending:
                    y, x = pending.popleft()
                    pixels.append((y, x))
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx < w and comp_mask[ny, nx] \
                                    and not visited[ny, nx]:
                                visited[ny, nx] = True
                                pending.append((ny, nx))
                if len(pixels) < 12:
                    continue
                ys = np.array([p[0] for p in pixels])
                xs = np.array([p[1] for p in pixels])
                ring = np.zeros((h, w), dtype=bool)
                y0b, y1b = max(0, ys.min() - 3), min(h - 1, ys.max() + 3)
                x0b, x1b = max(0, xs.min() - 3), min(w - 1, xs.max() + 3)
                ring[y0b:y1b + 1, x0b:x1b + 1] = True
                ring[ys, xs] = False
                ring &= valid
                if int(ring.sum()) < 20:
                    continue
                if float(np.median(d[ring])) <= float(np.median(d[ys, xs])) + PROTRUSION_M:
                    continue  # nothing farther behind: same surface, not an object on it
                box = [float(max(0, xs.min() - 2)), float(max(0, ys.min() - 2)),
                       float(min(w - 1, xs.max() + 2)), float(min(h - 1, ys.max() + 2))]
                boxes.append(box)
                if len(boxes) >= MAX_BOXES:
                    return boxes
        return boxes

    # ------------------------------------------------------------------ segmenter API

    def _segment(self, rgb: np.ndarray, points: list[list[float]],
                 boxes: list[list[float]]) -> tuple[np.ndarray, list[dict]]:
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="PNG")
        body = call("POST", f"{self._url}/segment_instances",
                    {"image_base64": base64.b64encode(buf.getvalue()).decode("ascii"),
                     "points": points, "boxes": boxes, "min_score": 0.5}, timeout=25.0)
        labels = np.array(Image.open(io.BytesIO(base64.b64decode(body["mask_png_base64"]))))
        return labels, body.get("instances", [])

    # ------------------------------------------------------------------ filtering

    def _fuse_geometry(self, labels: np.ndarray, instances: list[dict], depth_mm: np.ndarray,
                       fovy: float, cam_pose: dict) -> tuple[np.ndarray, list[dict]]:
        """Publish furniture as its own instance, built from the depth image's planes.

        The segmenter answers "which pixels look like one thing", and a tabletop plus its front
        panel rarely do. The estimator, though, needs a furniture *body*: given only a thin top
        slab it completes the box from the category prototype, and the completed table swallows
        the bowls standing on it. Depth knows better: a horizontal plane's height and extent are
        exact, and the vertical panel whose top meets that plane is the same piece of furniture.
        Each such surface (with its panels) becomes one instance that supersedes whatever regions
        the segmenter had put there.
        """
        surfaces, faces, world, _valid = self._geometric_surfaces(depth_mm, fovy, cam_pose)
        if not surfaces:
            return labels, instances
        out = labels.copy()
        next_id = GEO_FREE_ID
        made = []
        for surface in surfaces:
            group = [surface]
            for face in faces:
                if abs(face["z"][1] - surface["z"]) > GEO_FACE_TOP_TOL_M:
                    continue
                gap = max(surface["x0"] - face["x1"], face["x0"] - surface["x1"],
                          surface["y0"] - face["y1"], face["y0"] - surface["y1"], 0.0)
                if gap <= GEO_FACE_GAP_M:
                    group.append(face)
            ys = np.concatenate([g["ys"] for g in group])
            xs = np.concatenate([g["xs"] for g in group])
            out[ys, xs] = next_id
            bodied = len(group) > 1
            surface["id"] = next_id
            made.append({"id": next_id, "score": 0.5, "bodied": bodied})
            next_id += 1
        kept = []
        for info in instances:
            inst = int(info["id"])
            before = int((labels == inst).sum())
            if before == 0:
                continue
            leg_of = self._leg_of(labels == inst, surfaces, world)
            if leg_of is not None:
                # A thin thing wholly under a tabletop and inside its footprint is a leg. Left as
                # an object it reads as a toy or a bottle standing where the robot wants to dock,
                # and the table becomes unapproachable from every side.
                out[labels == inst] = leg_of
                continue
            remaining = labels == inst
            remaining &= out == inst
            after = int(remaining.sum())
            if after < max(MIN_PIXELS, 0.3 * before):
                # The geometry owns this region now. Whatever the segmenter had lumped into it
                # (a rim of table around the glass standing there) is dropped rather than left
                # as a wide sliver: a sliver that wide is read as a piece of the furniture and
                # merges back into it, carrying the glass's height along.
                out[remaining] = 0
                continue
            kept.append(info)
        kept.extend(made)
        if surfaces:
            # Whatever still glues furniture to something standing on it is a leftover blob:
            # it has pixels at the surface's own level *and* pixels above it, which no real
            # tableware has (a bowl's lowest pixels are already above the surface). Clip the
            # part above the level so the estimator never sees a tabletop with a glass built in.
            geo_ids = {int(item["id"]) for item in made}
            for info in kept:
                inst = int(info["id"])
                if inst in geo_ids:
                    continue
                ys, xs = np.nonzero(out == inst)
                if len(ys) < 200:
                    continue
                pts = world[ys, xs]
                ok = np.isfinite(pts[:, 0])
                if ok.sum() < 100:
                    continue
                extent = max(np.percentile(pts[ok, 0], 98) - np.percentile(pts[ok, 0], 2),
                             np.percentile(pts[ok, 1], 98) - np.percentile(pts[ok, 1], 2))
                if extent < HOST_MAX_FOOTPRINT_M:
                    continue  # tableware: its base is at the surface too, and it is whole as it is
                for surface in surfaces:
                    inside = ok & (pts[:, 0] > surface["x0"] - 0.05) & (pts[:, 0] < surface["x1"] + 0.05) \
                        & (pts[:, 1] > surface["y0"] - 0.05) & (pts[:, 1] < surface["y1"] + 0.05)
                    if int(inside.sum()) < 100:
                        continue
                    z = pts[inside, 2]
                    below = float(np.mean(z <= surface["z"] + CARVE_LIFT_M))
                    high = inside & (np.nan_to_num(pts[:, 2], nan=-1.0) > surface["z"] + CARVE_LIFT_M)
                    if below >= 0.15 and int(high.sum()) >= 100:
                        out[ys[high], xs[high]] = 0
                    break
            # A furniture-sized region lying on a known surface *is* that surface: the segmenter
            # answered the same question with a worse boundary. Leaving it publishes a second,
            # inflated table (the bowls on it are inside its box), and the planner picks either.
            bodied = [item for item in made if item.get("bodied")]
            bodied_xy = []
            for item in bodied:
                ys, xs = np.nonzero(out == int(item["id"]))
                pts = world[ys, xs]
                ok = np.isfinite(pts[:, 0])
                if ok.sum() < 100:
                    continue
                bodied_xy.append((float(np.median(pts[ok, 0])), float(np.median(pts[ok, 1]))))
            for info in kept:
                inst = int(info["id"])
                if inst in geo_ids:
                    continue
                ys, xs = np.nonzero(out == inst)
                if len(ys) < 300:
                    continue
                pts = world[ys, xs]
                ok = np.isfinite(pts[:, 0])
                if ok.sum() < 200:
                    continue
                span = max(np.percentile(pts[ok, 0], 98) - np.percentile(pts[ok, 0], 2),
                           np.percentile(pts[ok, 1], 98) - np.percentile(pts[ok, 1], 2))
                if span < 0.5:
                    continue
                if not bodied_xy:
                    continue  # no furniture body was built from depth here: keep the segmenter's
                cx, cy = np.median(pts[ok, 0]), np.median(pts[ok, 1])
                for surface in surfaces:
                    if not (surface["x0"] - 0.1 <= cx <= surface["x1"] + 0.1
                            and surface["y0"] - 0.1 <= cy <= surface["y1"] + 0.1):
                        continue
                    if float(np.percentile(pts[ok, 2], 10)) > surface["z"] + 0.30:
                        continue  # a tall thing standing here, not the furniture itself
                    out[ys, xs] = 0
                    break
        if os.environ.get("STEWARD_SEGMENTER_DEBUG"):
            world, valid = self._world_grid(depth_mm, fovy, cam_pose)
            spills = []
            for info in kept:
                inst = int(info["id"])
                ys, xs = np.nonzero(out == inst)
                if len(ys) < 200:
                    continue
                pts = world[ys, xs]
                pts = pts[np.isfinite(pts[:, 0])]
                if len(pts) == 0:
                    continue
                for surface in surfaces:
                    inside = ((pts[:, 0] > surface["x0"] - 0.05) & (pts[:, 0] < surface["x1"] + 0.05)
                              & (pts[:, 1] > surface["y0"] - 0.05) & (pts[:, 1] < surface["y1"] + 0.05))
                    if not inside.any():
                        continue
                    peak = float(np.percentile(pts[inside, 2], 98))
                    if peak > surface["z"] + 0.10:
                        spills.append(f"{inst}@{peak:.2f}(surf {surface['z']:.2f})")
                    break
            if spills:
                print(f"realseg dbg: furniture spill {spills}", file=sys.stderr, flush=True)
        return out, kept

    @staticmethod
    def _leg_of(region: np.ndarray, surfaces: list[dict], world: np.ndarray) -> int | None:
        """The id of the surface a region is a leg of, or None."""
        ys, xs = np.nonzero(region)
        if len(ys) < MIN_PIXELS:
            return None
        pts = world[ys, xs]
        pts = pts[np.isfinite(pts[:, 0])]
        if len(pts) < MIN_PIXELS:
            return None
        span = max(np.percentile(pts[:, 0], 98) - np.percentile(pts[:, 0], 2),
                   np.percentile(pts[:, 1], 98) - np.percentile(pts[:, 1], 2))
        if span > LEG_MAX_SPAN_M:
            return None
        cx, cy = float(np.median(pts[:, 0])), float(np.median(pts[:, 1]))
        for surface in surfaces:
            if "id" not in surface:
                continue
            if not (surface["x0"] - 0.08 <= cx <= surface["x1"] + 0.08
                    and surface["y0"] - 0.08 <= cy <= surface["y1"] + 0.08):
                continue
            if float(np.percentile(pts[:, 2], 95)) <= surface["z"] - 0.02:
                return int(surface["id"])
        return None

    def _geometric_surfaces(self, depth_mm: np.ndarray, fovy: float, cam_pose: dict
                            ) -> tuple[list[dict], list[dict], np.ndarray, np.ndarray]:
        """Horizontal surfaces and vertical panels, as pixel lists with world extents."""
        world, valid = self._world_grid(depth_mm, fovy, cam_pose)
        dx = np.zeros_like(world)
        dy = np.zeros_like(world)
        dx[:, 1:-1] = world[:, 2:] - world[:, :-2]
        dy[1:-1, :] = world[2:, :] - world[:-2, :]
        normal = np.cross(dx, dy)
        norm = np.linalg.norm(normal, axis=-1)
        usable = valid & (norm > 1e-9)
        with np.errstate(invalid="ignore"):
            nz = np.abs(normal[..., 2]) / np.maximum(norm, 1e-12)
        horizontal = usable & (nz > GEO_HORIZONTAL_NZ)
        vertical = usable & (nz < GEO_VERTICAL_NZ)
        surfaces = []
        for ys, xs in self._grow(horizontal, world, GEO_JOIN_M):
            if len(ys) < GEO_MIN_AREA_PX:
                continue
            pts = world[ys, xs]
            z = float(np.median(pts[:, 2]))
            if z < PLANE_MIN_Z_M:
                continue  # the floor
            x0, x1 = np.percentile(pts[:, 0], [2, 98])
            y0, y1 = np.percentile(pts[:, 1], [2, 98])
            if max(x1 - x0, y1 - y0) < GEO_MIN_SPAN_M or min(x1 - x0, y1 - y0) < GEO_MIN_MINOR_M:
                continue
            surfaces.append({"ys": ys, "xs": xs, "z": z, "x0": x0, "x1": x1, "y0": y0, "y1": y1})
        faces = []
        for ys, xs in self._grow(vertical, world, GEO_FACE_JOIN_M):
            if len(ys) < 400:
                continue
            pts = world[ys, xs]
            z0, z1 = np.percentile(pts[:, 2], [2, 98])
            if z1 - z0 < GEO_FACE_MIN_HEIGHT_M:
                continue
            x0, x1 = np.percentile(pts[:, 0], [2, 98])
            y0, y1 = np.percentile(pts[:, 1], [2, 98])
            if max(x1 - x0, y1 - y0) < GEO_FACE_MIN_WIDTH_M:
                continue
            if min(x1 - x0, y1 - y0) > FACE_MAX_THICK_M:
                continue
            faces.append({"ys": ys, "xs": xs, "z": (float(z0), float(z1)),
                          "x0": x0, "x1": x1, "y0": y0, "y1": y1})
        return surfaces, faces, world, valid

    def _world_grid(self, depth_mm: np.ndarray, fovy: float,
                    cam_pose: dict) -> tuple[np.ndarray, np.ndarray]:
        """Per-pixel world position of the depth image (NaN where there is no reading)."""
        h, w = depth_mm.shape
        d = depth_mm.astype(np.float32) / 1000.0
        f = (h / 2.0) / math.tan(math.radians(fovy) / 2.0)
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        valid = (d > 0.05) & (d < 6.0)
        u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        cam = np.stack([d, -(u - cx) * d / f, -(v - cy) * d / f], axis=-1)
        rot = matrix_from_quat(cam_pose["q"])
        world = cam @ rot.T + np.asarray(cam_pose["p"], float)
        world[~valid] = np.nan
        return world, valid

    @staticmethod
    def _grow(sel: np.ndarray, world: np.ndarray, tol: float) -> list[tuple[np.ndarray, np.ndarray]]:
        """4-connected regions of ``sel`` whose neighbouring pixels stay within ``tol`` in 3D."""
        h, w = sel.shape
        seen = np.zeros((h, w), dtype=bool)
        out = []
        for y0, x0 in zip(*np.nonzero(sel)):
            if seen[y0, x0]:
                continue
            seen[y0, x0] = True
            stack = [(int(y0), int(x0))]
            ys: list[int] = []
            xs: list[int] = []
            while stack:
                y, x = stack.pop()
                ys.append(y)
                xs.append(x)
                here = world[y, x]
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < h and 0 <= nx < w and sel[ny, nx] and not seen[ny, nx] \
                            and float(np.linalg.norm(world[ny, nx] - here)) <= tol:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            out.append((np.array(ys), np.array(xs)))
        return out

    def _merge_furniture(self, labels: np.ndarray, instances: list[dict], depth_mm: np.ndarray,
                         fovy: float, cam_pose: dict) -> tuple[np.ndarray, list[dict]]:
        """Fuse the pieces one piece of furniture arrives in: coplanar top fragments, and the
        vertical front face (or legs) that belongs to the same top.

        The segmenter routinely returns one tabletop as several stubs and its front panel as a
        separate region. The estimator then sees a thin top-only slab, and its category prototype
        completes the box upward instead of downward — a 0.75 m table ends up 0.95 m tall with the
        bowls it carries buried inside it, so every pick is refused for lack of a support surface.
        Two large flat instances whose tops sit level and close together are one surface; a tall
        thin panel whose top meets a surface level and whose face stands along that surface's edge
        is that surface's front. Compact instances never qualify on either count — the span
        threshold is several times a bowl's width — so objects standing on the surface keep their
        own masks.
        """
        h, w = labels.shape
        d = depth_mm.astype(np.float32) / 1000.0
        f = (h / 2.0) / math.tan(math.radians(fovy) / 2.0)
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        rot = matrix_from_quat(cam_pose["q"])
        origin = np.asarray(cam_pose["p"], float)
        boxes = []  # per instance: pixels, id, z level, z0, z1, footprint x0..y1
        for info in instances:
            ys, xs = np.nonzero(labels == int(info["id"]))
            if len(ys) < 4 * MIN_PIXELS:
                continue
            pick = np.linspace(0, len(ys) - 1, min(4000, len(ys))).astype(int)
            dd = d[ys[pick], xs[pick]]
            ok = (dd > 0.05) & (dd < 6.0)
            if int(ok.sum()) < 100:
                continue
            px, py, dv = xs[pick][ok], ys[pick][ok], dd[ok]
            cam = np.stack([dv, -(px - cx) * dv / f, -(py - cy) * dv / f], axis=1)
            world = cam @ rot.T + origin
            z_ref = float(np.percentile(world[:, 2], 75))
            slab = np.abs(world[:, 2] - z_ref) < PLANE_SLAB_M
            x0, x1 = np.percentile(world[:, 0], [2, 98])
            y0, y1 = np.percentile(world[:, 1], [2, 98])
            boxes.append({"px": len(ys), "id": int(info["id"]), "z_ref": z_ref,
                          "flat": float(slab.mean()), "z": world[:, 2],
                          "x0": x0, "x1": x1, "y0": y0, "y1": y1})
        surfaces = [b for b in boxes
                    if b["flat"] >= PLANE_FLAT_SHARE and b["z_ref"] >= PLANE_MIN_Z_M
                    and max(b["x1"] - b["x0"], b["y1"] - b["y0"]) >= PLANE_SPAN_M
                    and min(b["x1"] - b["x0"], b["y1"] - b["y0"]) >= PLANE_MINOR_M]
        surfaces.sort(key=lambda b: -b["px"])
        absorb: dict[int, int] = {}  # instance id -> survivor instance id

        def gap(a: dict, b: dict) -> float:
            return max(a["x0"] - b["x1"], b["x0"] - a["x1"],
                       a["y0"] - b["y1"], b["y0"] - a["y1"], 0.0)

        def union_gap(group: list[dict], face: dict) -> float:
            x0 = min(b["x0"] for b in group)
            x1 = max(b["x1"] for b in group)
            y0 = min(b["y0"] for b in group)
            y1 = max(b["y1"] for b in group)
            return gap({"x0": x0, "x1": x1, "y0": y0, "y1": y1}, face)

        groups: list[list[dict]] = []
        for surface in surfaces:
            for group in groups:
                top = max(b["z_ref"] for b in group)
                if abs(top - surface["z_ref"]) < PLANE_LEVEL_M and union_gap(group, surface) < PLANE_GAP_M:
                    group.append(surface)
                    break
            else:
                groups.append([surface])
        for group in groups:
            if len(group) < 2:
                continue
            survivor = group[0]["id"]  # surface list is pixel-sorted, so this is the largest
            for member in group[1:]:
                absorb[member["id"]] = survivor
            # The group's front: a tall thin region whose top levels with the surface and whose
            # face runs along the group's edge. Legs are thin in both horizontal directions.
            z_top = max(b["z_ref"] for b in group) + PLANE_SLAB_M
            for face in boxes:
                if face in group or face["id"] in absorb:
                    continue
                face_height = float(np.percentile(face["z"], 98) - np.percentile(face["z"], 2))
                width = max(face["x1"] - face["x0"], face["y1"] - face["y0"])
                thickness = min(face["x1"] - face["x0"], face["y1"] - face["y0"])
                if face_height < FACE_MIN_HEIGHT_M or thickness > FACE_MAX_THICK_M:
                    continue
                if width < FACE_MIN_WIDTH_M:
                    continue
                if abs(float(np.percentile(face["z"], 98)) - z_top) > FACE_TOP_TOL_M:
                    continue
                if union_gap(group, face) >= FACE_GAP_M:
                    continue
                absorb[face["id"]] = survivor
        if not absorb:
            return labels, instances  # nothing arrived in pieces
        for old, new in absorb.items():
            if old != new:
                labels[labels == old] = new
        kept = [info for info in instances if int(info["id"]) not in absorb]
        return labels, kept

    def _carve_standing(self, labels: np.ndarray, instances: list[dict], depth_mm: np.ndarray,
                        fovy: float, cam_pose: dict) -> np.ndarray:
        """Cut from a large horizontal surface's mask whatever stands above its level.

        Seen from a few metres away a table and the glass on it arrive as one region: the glass
        spans too few pixels to be its own instance, and the surface swallows it. The estimator
        then sizes the table by the glass's top (a 0.75 m table becomes 0.95 m) and every object
        on it ends up buried inside the furniture box, so no pick can find a support surface.
        Anything more than a slab's thickness above a surface's own level is not part of that
        surface: hand those pixels back to the instance standing in the same place (the glass's
        own region when it exists), or make them their own instance.
        """
        h, w = labels.shape
        d = depth_mm.astype(np.float32) / 1000.0
        f = (h / 2.0) / math.tan(math.radians(fovy) / 2.0)
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        rot = matrix_from_quat(cam_pose["q"])
        origin = np.asarray(cam_pose["p"], float)
        out = labels.copy()
        next_id = max([int(v) for v in np.unique(labels)] + [int(i["id"]) for i in instances] + [0]) + 1
        stats = []  # centroid and z range per non-surface instance, for handing pixels back
        for info in instances:
            m = labels == int(info["id"])
            ys, xs = np.nonzero(m)
            if len(ys) < MIN_PIXELS:
                continue
            dd = d[ys, xs]
            ok = (dd > 0.05) & (dd < 6.0)
            if int(ok.sum()) < 20:
                continue
            cam = np.stack([dd[ok], -(xs[ok] - cx) * dd[ok] / f, -(ys[ok] - cy) * dd[ok] / f], axis=1)
            world = cam @ rot.T + origin
            extent = max(np.percentile(world[:, 0], 98) - np.percentile(world[:, 0], 2),
                         np.percentile(world[:, 1], 98) - np.percentile(world[:, 1], 2))
            stats.append({"id": int(info["id"]), "xy": np.median(world[:, :2], axis=0),
                          "z": (float(np.percentile(world[:, 2], 2)), float(np.percentile(world[:, 2], 98))),
                          "px": len(ys), "footprint": extent})
        for info in instances:
            inst = int(info["id"])
            m = labels == inst
            ys, xs = np.nonzero(m)
            if len(ys) < 4 * MIN_PIXELS:
                continue
            dd = d[ys, xs]
            ok = (dd > 0.05) & (dd < 6.0)
            if int(ok.sum()) < 100:
                continue
            px, py, dv = xs[ok], ys[ok], dd[ok]
            cam = np.stack([dv, -(px - cx) * dv / f, -(py - cy) * dv / f], axis=1)
            world = cam @ rot.T + origin
            level = float(np.percentile(world[:, 2], 30))
            flat = float(np.mean(np.abs(world[:, 2] - level) < PLANE_SLAB_M))
            span = max(np.percentile(world[:, 0], 98) - np.percentile(world[:, 0], 2),
                       np.percentile(world[:, 1], 98) - np.percentile(world[:, 1], 2))
            minor = min(np.percentile(world[:, 0], 98) - np.percentile(world[:, 0], 2),
                        np.percentile(world[:, 1], 98) - np.percentile(world[:, 1], 2))
            if span < PLANE_SPAN_M or minor < PLANE_MINOR_M or level < PLANE_MIN_Z_M \
                    or flat < CARVE_FLAT_SHARE:
                continue  # not a big horizontal surface (a bowl rim is round and small)
            above = world[:, 2] > level + CARVE_LIFT_M
            if int(above.sum()) < MIN_PIXELS:
                continue
            grid = np.zeros((h, w), dtype=bool)
            grid[py[above], px[above]] = True
            if os.environ.get("STEWARD_SEGMENTER_DEBUG"):
                print(f"realseg dbg: surface {inst} px={len(ys)} level={level:.3f} flat={flat:.2f} "
                      f"span={span:.2f} above={int(above.sum())}", file=sys.stderr, flush=True)
            for comp in self._components(grid):
                if len(comp) < MIN_PIXELS:
                    continue
                cys = np.array([p[0] for p in comp])
                cxs = np.array([p[1] for p in comp])
                cdd = d[cys, cxs]
                cok = (cdd > 0.05) & (cdd < 6.0)
                if int(cok.sum()) < 10:
                    continue
                cw = np.stack([cdd[cok], -(cxs[cok] - cx) * cdd[cok] / f,
                               -(cys[cok] - cy) * cdd[cok] / f], axis=1) @ rot.T + origin
                cxy = np.median(cw[:, :2], axis=0)
                cz = (float(np.percentile(cw[:, 2], 2)), float(np.percentile(cw[:, 2], 98)))
                host = None
                best = 0.30  # metres in XY
                for other in stats:
                    if other["id"] == inst:
                        continue
                    if other["footprint"] > HOST_MAX_FOOTPRINT_M:
                        continue  # never hand tableware back to the furniture it stands on
                    if other["z"][0] < level - CARVE_LIFT_M:
                        continue  # a piece of furniture below, not something standing here
                    dist = float(np.linalg.norm(other["xy"] - cxy))
                    overlap = min(cz[1], other["z"][1]) - max(cz[0], other["z"][0])
                    if dist < best and overlap > 0.5 * (cz[1] - cz[0]):
                        host, best = other["id"], dist
                if host is None:
                    host = next_id
                    next_id += 1
                    instances.append({"id": host, "score": float(info.get("score", 0.5)),
                                      "box": [float(cxs.min()), float(cys.min()),
                                              float(cxs.max()), float(cys.max())]})
                    footprint = max(float(np.percentile(cw[:, 0], 98) - np.percentile(cw[:, 0], 2)),
                                    float(np.percentile(cw[:, 1], 98) - np.percentile(cw[:, 1], 2)))
                    stats.append({"id": host, "xy": cxy, "z": cz, "px": len(comp), "footprint": footprint})
                out[cys, cxs] = host
        return out

    @staticmethod
    def _components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
        """8-connected components of a boolean image, as lists of (y, x) pixels."""
        from collections import deque

        h, w = grid.shape
        seen = np.zeros((h, w), dtype=bool)
        out = []
        for y0, x0 in zip(*np.nonzero(grid)):
            if seen[y0, x0]:
                continue
            seen[y0, x0] = True
            pixels = []
            pending = deque([(int(y0), int(x0))])
            while pending:
                y, x = pending.popleft()
                pixels.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and grid[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            pending.append((ny, nx))
            out.append(pixels)
        return out

    def _slivers(self, labels: np.ndarray, instances: list[dict], depth_mm: np.ndarray,
                 fovy: float, cam_pose: dict) -> set[int]:
        """Ids of thin fragments that lie inside a whole instance of similar size and height.

        A point prompted on a wide bowl's rim can return the rim alone; it becomes a track of its
        own a dozen centimetres from the bowl's centre, and a pick aimed at it always misses. A
        real thing is never centimetres thin while tens of centimetres long; anything that is, and
        that a rounder, fuller neighbour covers, is that neighbour's part."""
        h, w = labels.shape
        d = depth_mm.astype(np.float32) / 1000.0
        f = (h / 2.0) / math.tan(math.radians(fovy) / 2.0)
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        rot = matrix_from_quat(cam_pose["q"])
        origin = np.asarray(cam_pose["p"], float)
        stats = []
        for info in instances:
            ys, xs = np.nonzero(labels == int(info["id"]))
            if len(ys) < 50:
                continue
            dd = d[ys, xs]
            ok = (dd > 0.05) & (dd < 6.0)
            if int(ok.sum()) < 30:
                continue
            cam = np.stack([dd[ok], -(xs[ok] - cx) * dd[ok] / f, -(ys[ok] - cy) * dd[ok] / f], axis=1)
            world = cam @ rot.T + origin
            x0, x1 = np.percentile(world[:, 0], [2, 98])
            y0, y1 = np.percentile(world[:, 1], [2, 98])
            z0, z1 = np.percentile(world[:, 2], [2, 98])
            stats.append({"id": int(info["id"]), "major": max(x1 - x0, y1 - y0),
                          "minor": min(x1 - x0, y1 - y0), "x0": x0, "x1": x1, "y0": y0, "y1": y1,
                          "z": (float(z0), float(z1)), "px": len(ys)})
        out = set()
        for sliver in stats:
            if sliver["minor"] > SLIVER_MAX_MINOR_M or sliver["major"] < 2.0 * sliver["minor"]:
                continue
            for other in stats:
                if other["id"] == sliver["id"] or other["minor"] < 2.5 * sliver["minor"] \
                        or other["minor"] > HOST_MAX_FOOTPRINT_M:
                    continue
                cx = (sliver["x0"] + sliver["x1"]) / 2
                cy = (sliver["y0"] + sliver["y1"]) / 2
                if other["x0"] - 0.02 <= cx <= other["x1"] + 0.02 \
                        and other["y0"] - 0.02 <= cy <= other["y1"] + 0.02 \
                        and sliver["z"][1] > other["z"][0] and sliver["z"][0] < other["z"][1]:
                    out.add(sliver["id"])
                    break
        return out

    def _filter(self, labels: np.ndarray, instances: list[dict], depth_mm: np.ndarray,
                fovy: float, cam_pose: dict) -> np.ndarray:
        """Drop the floor and camera-near instances; renumber the rest to shuffled 1..N."""
        h, w = labels.shape
        d = depth_mm.astype(np.float32) / 1000.0
        f = (h / 2.0) / math.tan(math.radians(fovy) / 2.0)
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        rot = matrix_from_quat(cam_pose["q"])
        origin = np.asarray(cam_pose["p"], float)
        rng_shuffle = np.random.default_rng()
        out = np.zeros((h, w), dtype=np.uint8)
        kept = 0
        ids = rng_shuffle.permutation(np.arange(1, 256))[: len(instances)]
        slivers = self._slivers(labels, instances, depth_mm, fovy, cam_pose)
        self_px = self._self_mask(labels.shape, cam_pose)
        for info, new_id in zip(instances, ids, strict=False):
            inst = int(info["id"])
            if inst in slivers:
                continue  # a rim arc of a bowl that is already an instance of its own
            ys, xs = np.nonzero(labels == inst)
            if len(ys) < MIN_PIXELS:
                continue
            dd = d[ys, xs]
            valid = (dd > 0.05) & (dd < 6.0)
            if int(valid.sum()) < MIN_PIXELS:
                continue
            if self_px is not None and float(self_px[ys, xs].mean()) > SELF_SHARE:
                continue  # the robot's own arm or gripper, projected from its kinematics
            if float(np.median(dd[valid])) < NEAR_RANGE_M:
                continue  # something pressed against the lens: not a reachable object
            cam = np.stack([dd[valid],
                            -(xs[valid] - cx) * dd[valid] / f,
                            -(ys[valid] - cy) * dd[valid] / f], axis=1)
            world_z = cam @ rot.T[:, 2] + origin[2]
            if (len(ys) > GROUND_SHARE * h * w
                    and float(np.percentile(world_z, 90)) < GROUND_Z_M):
                continue  # the floor
            out[ys, xs] = int(new_id)
            kept += 1
        return out

    def _self_mask(self, shape: tuple[int, int], cam_pose: dict) -> np.ndarray | None:
        """Pixels the robot's own body occupies, from its kinematics; None when unavailable.

        The synthetic channel got this for free (the renderer excludes the observing robot); a
        real segmenter has no idea what the robot looks like, so the robot projects itself."""
        try:
            from steward.ego.estimator import EgoEstimator
            from steward.ego.selfmask import self_pixels

            if self._estimator is None:
                entity = self._sim.template.entity(self._robot)
                self._estimator = EgoEstimator(robot_model=entity.to_dict())
                self._camera_id = next(
                    sensor["id"] for link in entity.rigid.links for sensor in link.sensors
                    if sensor.get("kind") in ("rgb", "depth", "rgbd"))
            frame = self._sim.sensors(self._robot)
            self._estimator.joints = frame["joints"]
            self._estimator.odometry = frame["odometry"]
            _, sensor = self._sim.camera_pose(self._robot, self._camera_id)
            return self_pixels(self._estimator.robot_model, self._estimator.link_pose,
                               cam_pose, sensor["intrinsics"])
        except Exception:  # noqa: BLE001 - a missing self mask must not stop perception
            return None
