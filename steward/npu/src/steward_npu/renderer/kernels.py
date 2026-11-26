"""Experimental analytic ray renderer; isolated from Steward's production renderer.

Boxes/cylinders only. Intersection, depth, IDs and simple headlight shading are
computed on the selected tensor device. NO OpenGL rendering or CPU pixel fallback.
Adds cubemap textures, directional lights and analytic hard shadows.
AA, mipmapping, exact OpenGL shading and general meshes are not implemented.
Consequently this is a feasibility experiment, NOT a drop-in sensor renderer.
"""

import itertools
from types import SimpleNamespace

import numpy as np
import torch


class AnalyticRenderer:
    def __init__(self, scene, device="cpu", chunk=65536):
        self.device = torch.device(device)
        self.chunk = chunk
        self.s = scene
        types = np.asarray(scene["type"])
        if not set(types.tolist()) <= {5, 6}:
            raise ValueError("Prototype supports only cylinders and boxes")
        if np.any(scene["rgba"][:, 3] < 1):
            raise ValueError("Transparent geometry is unsupported")
        if bool(scene["camera_orthographic"]):
            raise ValueError("Only perspective cameras supported")
        cylinders = types == 5
        if np.any(
            np.abs(scene["size"][cylinders, 0] - scene["size"][cylinders, 1]) > 1e-7
        ):
            raise ValueError("Elliptic cylinders are unsupported")
        # mjvScene uses (radius_x, radius_y, half_height), unlike mjModel.geom_size.
        self.w = int(scene["width"])
        self.h = int(scene["height"])
        self.near = float(scene["camera_frustum_near"])
        self.far = float(scene["camera_frustum_far"])
        self.rot = self.tensor(scene["mat"].reshape(-1, 3, 3))
        self.pos = self.tensor(scene["pos"])
        self.size = self.tensor(scene["size"])
        self.colors = self.tensor(scene["rgba"][:, :3])
        self.box = self.tensor(types == 6, dtype=torch.bool)
        self.ids = self.tensor(scene["objid"], dtype=torch.int64)
        self.objtypes = self.tensor(scene["objtype"], dtype=torch.int64)
        self.origin = self.tensor(scene["camera_pos"])
        self.forward = self.tensor(scene["camera_forward"])
        self.up = self.tensor(scene["camera_up"])
        right = torch.linalg.cross(self.forward, self.up)
        right = right / torch.linalg.vector_norm(right)
        yy, xx = torch.meshgrid(
            torch.arange(self.h, device=self.device, dtype=torch.float32),
            torch.arange(self.w, device=self.device, dtype=torch.float32),
            indexing="ij",
        )
        top = float(scene["camera_frustum_top"])
        bottom = float(scene["camera_frustum_bottom"])
        halfwidth = (top - bottom) * self.w / self.h / 2
        cx = float(scene["camera_frustum_center"])
        self.rays = (
            self.forward
            + (
                ((xx + float(scene.get("sample_x", 0.5))) / self.w * 2 - 1) * halfwidth
                + cx
            )[..., None]
            / self.near
            * right
            + (
                top - (yy + float(scene.get("sample_y", 0.5))) / self.h * (top - bottom)
            )[..., None]
            / self.near
            * self.up
        ).reshape(-1, 3)
        self.local_origin = torch.sum(
            (self.origin - self.pos)[:, :, None] * self.rot, dim=1
        )
        self.rot_flat = self.rot.permute(1, 0, 2).reshape(3, -1)
        if not np.all(scene["light_type"] == 1):
            raise ValueError("Only directional lights supported")
        self.lights = [
            {
                key: self.tensor(scene["light_" + key][i])
                for key in ["dir", "ambient", "diffuse", "specular"]
            }
            for i in range(len(scene["light_type"]))
        ]
        self.shininess = self.tensor(scene["shininess"]) * 128
        self.specular = self.tensor(scene["specular"])
        self.emission = self.tensor(scene.get("emission", np.zeros(len(types))))
        self.shadows = bool(scene.get("shadows", True))
        self.light_shadows = np.asarray(
            scene.get("light_castshadow", 1 - scene["light_headlight"]), dtype=bool
        )
        if not np.all(scene["tex_nchannel"] == 3) or not np.all(scene["tex_type"] == 1):
            raise ValueError("Only RGB cubemap textures supported")
        if np.any(scene["mat_texuniform"]):
            raise ValueError("Uniform texture mapping unsupported")
        self.texture = self.tensor(scene["tex_data"].reshape(-1, 3), dtype=torch.uint8)
        self.texwidth = self.tensor(scene["tex_width"], dtype=torch.int64)
        self.texadr = self.tensor(scene["tex_adr"] // 3, dtype=torch.int64)
        self.geom_mat = self.tensor(scene["matid"], dtype=torch.int64)
        self.mat_tex = self.tensor(scene["mat_texid"][:, 1], dtype=torch.int64)
        self.geom_translation = torch.sum(self.pos[:, :, None] * self.rot, dim=1)

    def texture_color(self, local, gid):
        """Bilinear cubemap lookup in local geometry coordinates (no mipmaps)."""
        coord = local / torch.clamp(self.size[gid], min=1e-10)
        axis = torch.abs(coord).argmax(dim=-1)
        x, y, z = coord.unbind(-1)
        primary = coord.gather(1, axis[:, None]).squeeze(1)
        sign = torch.where(primary >= 0, 1.0, -1.0)
        face = axis * 2 + (primary < 0).to(torch.int64)
        u = torch.where(axis == 0, -z * sign, torch.where(axis == 1, x, x * sign))
        v = torch.where(axis == 0, -y, torch.where(axis == 1, z * sign, -y))
        scale = torch.clamp(torch.abs(primary), min=1e-10)
        mat = self.geom_mat[gid]
        texid = self.mat_tex[torch.clamp(mat, min=0)]
        safeid = torch.clamp(texid, min=0)
        width = self.texwidth[safeid]
        tx = torch.clamp((u / scale + 1) * 0.5 * width - 0.5, min=0)
        ty = torch.clamp((v / scale + 1) * 0.5 * width - 0.5, min=0)
        tx = torch.minimum(tx, (width - 1).float())
        ty = torch.minimum(ty, (width - 1).float())
        x0 = tx.floor().long()
        y0 = ty.floor().long()
        x1 = torch.minimum(x0 + 1, width - 1)
        y1 = torch.minimum(y0 + 1, width - 1)
        base = self.texadr[safeid] + face * width * width

        def fetch(ix, iy):
            return self.texture[base + iy * width + ix].float() / 255

        fx = (tx - x0)[:, None]
        fy = (ty - y0)[:, None]
        color = (fetch(x0, y0) * (1 - fx) + fetch(x1, y0) * fx) * (1 - fy) + (
            fetch(x0, y1) * (1 - fx) + fetch(x1, y1) * fx
        ) * fy
        return torch.where(((mat >= 0) & (texid >= 0))[:, None], color, 1.0)

    def hits(self, d, o, near, far, cull=False):
        """Nearest box/cylinder intersections; ray parameters encode camera depth."""
        inf = 1e20
        eps = 1e-10
        s = self.size[None, :, :]
        safe = torch.where(torch.abs(d) > eps, d, torch.where(d >= 0, eps, -eps))
        t1 = (-s - o) / safe
        t2 = (s - o) / safe
        enter = torch.minimum(t1, t2).amax(dim=-1)
        leave = torch.maximum(t1, t2).amin(dim=-1)
        bt = torch.where(
            (leave >= enter) & (leave > near),
            torch.where(enter > near, enter, leave),
            inf,
        )
        if cull:
            bt = torch.where(enter > near, bt, inf)
        dx, dy, dz = d.unbind(-1)
        ox, oy, oz = o.unbind(-1)
        qa = dx * dx + dy * dy
        qb = ox * dx + oy * dy
        den = torch.clamp(qa, min=eps)
        closest_t = -qb / den
        clearance = (
            s[:, :, 0] ** 2 - (ox + closest_t * dx) ** 2 - (oy + closest_t * dy) ** 2
        )
        discr = clearance
        root = torch.sqrt(torch.clamp(clearance / den, min=0))
        ts1 = closest_t - root
        ts2 = closest_t + root
        ok1 = (
            (discr >= 0)
            & (qa > eps)
            & (ts1 > near)
            & (torch.abs(oz + ts1 * dz) <= s[:, :, 2])
        )
        ok2 = (
            (discr >= 0)
            & (qa > eps)
            & (ts2 > near)
            & (torch.abs(oz + ts2 * dz) <= s[:, :, 2])
        )
        if cull:
            ok2 = torch.zeros_like(ok2)
        sides = torch.minimum(torch.where(ok1, ts1, inf), torch.where(ok2, ts2, inf))
        zden = torch.where(torch.abs(dz) > eps, dz, eps)
        ct1 = (-s[:, :, 2] - oz) / zden
        ct2 = (s[:, :, 2] - oz) / zden

        def cap(t):
            return (
                (t > near)
                & (torch.abs(dz) > eps)
                & ((ox + t * dx) ** 2 + (oy + t * dy) ** 2 <= s[:, :, 0] ** 2)
            )

        cap1 = cap(ct1)
        cap2 = cap(ct2)
        if cull:
            cap1 = cap1 & (dz > 0)
            cap2 = cap2 & (dz < 0)
        caps = torch.minimum(torch.where(cap1, ct1, inf), torch.where(cap2, ct2, inf))
        hits = torch.where(self.box[None, :], bt, torch.minimum(sides, caps))
        return torch.where(hits < far, hits, inf)

    def tensor(self, x, dtype=torch.float32):
        return torch.as_tensor(np.asarray(x).copy(), dtype=dtype, device=self.device)

    def sync(self):
        if self.device.type == "npu":
            torch.npu.synchronize()
        elif self.device.type == "cuda":
            torch.cuda.synchronize()


class TiledRenderer(AnalyticRenderer):
    """Conservative CPU bounds select candidates; tensors compute all pixels."""

    def __init__(self, scene, device="cpu", tile=512):
        super().__init__(scene, device)
        self.tile = tile
        self.tiles = []
        rotations = np.asarray(scene["mat"]).reshape(-1, 3, 3)
        corners = np.asarray(list(itertools.product([-1, 1], repeat=3)), np.float32)
        world = np.asarray(scene["pos"])[:, None, :] + np.einsum(
            "gij,gkj->gki",
            rotations,
            corners[None, :, :] * np.asarray(scene["size"])[:, None, :],
        )
        forward = np.asarray(scene["camera_forward"])
        up = np.asarray(scene["camera_up"])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        delta = world - np.asarray(scene["camera_pos"])
        px = delta @ right
        py = delta @ up
        pz = delta @ forward
        top = float(scene["camera_frustum_top"])
        bottom = float(scene["camera_frustum_bottom"])
        half = (top - bottom) * self.w / self.h / 2
        center = float(scene["camera_frustum_center"])
        for y in range(0, self.h, tile):
            for x in range(0, self.w, tile):
                height = min(tile, self.h - y)
                width = min(tile, self.w - x)
                left = (center + (2 * x / self.w - 1) * half) / self.near
                right_edge = (
                    center + (2 * (x + width) / self.w - 1) * half
                ) / self.near
                upper = (top - y / self.h * (top - bottom)) / self.near
                lower = (top - (y + height) / self.h * (top - bottom)) / self.near
                # A convex bounding box wholly outside any plane cannot hit this tile.
                margin = 1e-4
                excluded = (
                    (px - left * pz < -margin).all(1)
                    | (px - right_edge * pz > margin).all(1)
                    | (py - lower * pz < -margin).all(1)
                    | (py - upper * pz > margin).all(1)
                    | (pz < self.near - margin).all(1)
                    | (pz > self.far + margin).all(1)
                )
                ids = np.flatnonzero(~excluded)
                # Include one harmless candidate for an empty tile; far clipping yields background.
                if not len(ids):
                    ids = np.array([0])
                ix = self.tensor(ids, dtype=torch.int64)
                self.tiles.append(
                    dict(
                        x=x,
                        y=y,
                        w=width,
                        h=height,
                        ids=ix,
                        rot=self.rot[ix].permute(1, 0, 2).reshape(3, -1),
                        origin=self.local_origin[ix][None, :, :],
                        geom=SimpleNamespace(
                            size=self.size[ix],
                            box=self.box[ix],
                            box_only=bool(np.all(np.asarray(scene["type"])[ids] == 6)),
                        ),
                        textured=bool(np.any(np.asarray(scene["matid"])[ids] >= 0)),
                        count=len(ids),
                    )
                )
        for tile_info in self.tiles:
            x, y, w, h = (tile_info[k] for k in ("x", "y", "w", "h"))
            ray = self.rays.reshape(self.h, self.w, 3)[y : y + h, x : x + w].reshape(
                -1, 3
            )
            tile_info["rays"] = ray
            tile_info["view"] = -ray / torch.linalg.vector_norm(
                ray, dim=-1, keepdim=True
            )
        self.light_directions = [
            -light["dir"] / torch.linalg.vector_norm(light["dir"])
            for light in self.lights
        ]
        self.shadow_frames = {}
        self.shadow_cache = {}
        for li, light in enumerate(scene["light_dir"]):
            direction = -np.asarray(light, dtype=np.float32)
            direction /= np.linalg.norm(direction)
            axis = np.eye(3, dtype=np.float32)[np.argmin(np.abs(direction))]
            u = np.cross(direction, axis)
            u /= np.linalg.norm(u)
            v = np.cross(direction, u)
            basis = np.stack([u, v, direction], axis=1)
            projected = world @ basis
            self.shadow_frames[li] = (
                self.tensor(basis),
                projected.min(1),
                projected.max(1),
            )
        self.stats = {
            "tiles": len(self.tiles),
            "primary_candidates_mean": float(np.mean([t["count"] for t in self.tiles])),
            "primary_candidates_max": max(t["count"] for t in self.tiles),
            "total_geoms": len(scene["type"]),
            "textured_tiles": sum(t["textured"] for t in self.tiles),
        }

    @staticmethod
    def selected_hits(geometry, d, o, near, far, cull=False):
        if not geometry.box_only:
            return AnalyticRenderer.hits(geometry, d, o, near, far, cull=cull)
        # Preserve baseline slab arithmetic exactly, without computing unused cylinder roots.
        eps = 1e-10
        inf = 1e20
        s = geometry.size[None, :, :]
        safe = torch.where(torch.abs(d) > eps, d, torch.where(d >= 0, eps, -eps))
        t1 = (-s - o) / safe
        t2 = (s - o) / safe
        enter = torch.minimum(t1, t2).amax(dim=-1)
        leave = torch.maximum(t1, t2).amin(dim=-1)
        bt = torch.where(
            (leave >= enter) & (leave > near),
            torch.where(enter > near, enter, leave),
            inf,
        )
        if cull:
            bt = torch.where(enter > near, bt, inf)
        return torch.where(bt < far, bt, inf)

    @torch.inference_mode()
    def render(self):
        """Return RGB uint8, depth float32, and (object ID, type) int32 arrays."""
        rgb_out = torch.empty(
            (self.h, self.w, 3), device=self.device, dtype=torch.uint8
        )
        depth_out = torch.empty(
            (self.h, self.w), device=self.device, dtype=torch.float32
        )
        seg_out = torch.empty(
            (self.h, self.w, 2), device=self.device, dtype=torch.int32
        )
        shadow_counts = []
        records = []
        receiver_bounds = {
            li: []
            for li in self.shadow_frames
            if self.shadows and self.light_shadows[li]
        }
        for tile in self.tiles:
            x, y, w, h = (tile[k] for k in ("x", "y", "w", "h"))
            rays = tile["rays"]
            n = len(rays)
            d = (rays @ tile["rot"]).reshape(n, -1, 3)
            hits = self.selected_hits(
                tile["geom"], d, tile["origin"], self.near, self.far, cull=True
            )
            depth, local_gid = hits.min(1)
            gid = tile["ids"][local_gid]
            valid = depth < self.far
            safe_depth = torch.where(valid, depth, 0)
            local = (
                self.local_origin[gid]
                + d[torch.arange(n, device=self.device), local_gid]
                * safe_depth[:, None]
            )
            size = self.size[gid]
            axis = (torch.abs(local) / torch.clamp(size, min=1e-10)).argmax(-1)
            normal_box = torch.nn.functional.one_hot(axis, 3).float() * torch.sign(
                local
            )
            if tile["geom"].box_only:
                normal_local = normal_box
            else:
                zero = torch.zeros(n, device=self.device)
                cyl_side = torch.cat(
                    [
                        local[:, :2] / torch.clamp(size[:, 0:1], min=1e-10),
                        zero[:, None],
                    ],
                    1,
                )
                cyl_cap = torch.stack([zero, zero, torch.sign(local[:, 2])], 1)
                normal_cyl = torch.where(
                    (torch.abs(torch.abs(local[:, 2]) - size[:, 2]) < 1e-5)[:, None],
                    cyl_cap,
                    cyl_side,
                )
                normal_local = torch.where(self.box[gid, None], normal_box, normal_cyl)
            normal = torch.sum(self.rot[gid] * normal_local[:, None, :], dim=-1)
            view = tile["view"]
            albedo = self.colors[gid]
            world = self.origin + rays * safe_depth[:, None]
            records.append(
                (tile, rays, gid, depth, valid, normal, view, albedo, world, local)
            )
            for li in receiver_bounds:
                basis = self.shadow_frames[li][0]
                receiver = (world + normal * 1e-4) @ basis
                receiver_bounds[li].append(
                    torch.stack([receiver.amin(0), receiver.amax(0)])
                )
        # One small metadata readback per shadow light, rather than one synchronization per tile.
        bounds_host = {
            li: torch.stack(bounds).cpu().numpy()
            for li, bounds in receiver_bounds.items()
        }
        for ri, (
            tile,
            rays,
            gid,
            depth,
            valid,
            normal,
            view,
            albedo,
            world,
            local,
        ) in enumerate(records):
            x, y, w, h = (tile[k] for k in ("x", "y", "w", "h"))
            n = len(rays)
            rgb = albedo * self.emission[gid, None]
            for li, light in enumerate(self.lights):
                direction = self.light_directions[li]
                diffuse = torch.clamp((normal * direction).sum(-1), min=0)
                half = direction + view
                half = half / torch.linalg.vector_norm(half, dim=-1, keepdim=True)
                spec = (
                    torch.clamp((normal * half).sum(-1), min=0) ** self.shininess[gid]
                )
                visibility = torch.ones(n, device=self.device)
                if self.shadows and self.light_shadows[li]:
                    origins = world + normal * 1e-4
                    basis, occluder_min, occluder_max = self.shadow_frames[li]
                    lo, hi = bounds_host[li][ri]
                    pad = 1e-4
                    possible = (
                        (occluder_max[:, 0] >= lo[0] - pad)
                        & (occluder_min[:, 0] <= hi[0] + pad)
                        & (occluder_max[:, 1] >= lo[1] - pad)
                        & (occluder_min[:, 1] <= hi[1] + pad)
                        & (occluder_max[:, 2] >= lo[2] - pad)
                    )
                    ids = np.flatnonzero(possible)
                    shadow_counts.append(len(ids))
                    if len(ids):
                        key = (li, tuple(ids.tolist()))
                        if key not in self.shadow_cache:
                            ix = self.tensor(ids, dtype=torch.int64)
                            self.shadow_cache[key] = (
                                self.rot[ix].permute(1, 0, 2).reshape(3, -1),
                                self.geom_translation[ix],
                                SimpleNamespace(
                                    size=self.size[ix],
                                    box=self.box[ix],
                                    box_only=bool(
                                        np.all(np.asarray(self.s["type"])[ids] == 6)
                                    ),
                                ),
                            )
                        rot, translation, geometry = self.shadow_cache[key]
                        shadow_o = (origins @ rot).reshape(n, -1, 3) - translation[
                            None, :, :
                        ]
                        shadow_d = (direction @ rot).reshape(1, -1, 3)
                        blockers = self.selected_hits(
                            geometry, shadow_d, shadow_o, 1e-5, 1e10
                        )
                        visibility = (blockers.min(1).values >= 1e10).float()
                rgb += (
                    albedo * light["ambient"]
                    + (
                        albedo * light["diffuse"] * diffuse[:, None]
                        + self.specular[gid, None]
                        * light["specular"]
                        * spec[:, None]
                        * (diffuse > 0)[:, None]
                    )
                    * visibility[:, None]
                )
            rgb = torch.clamp(rgb, 0, 1)
            if tile["textured"]:
                rgb = rgb * self.texture_color(local, gid)
            rgb = torch.where(valid[:, None], rgb, 0)
            rgb_out[y : y + h, x : x + w] = (
                (rgb * 255).round().to(torch.uint8).reshape(h, w, 3)
            )
            depth_out[y : y + h, x : x + w] = torch.where(
                valid, depth, self.far
            ).reshape(h, w)
            seg_out[y : y + h, x : x + w] = (
                torch.stack(
                    [
                        torch.where(valid, self.ids[gid], -1),
                        torch.where(valid, self.objtypes[gid], -1),
                    ],
                    -1,
                )
                .to(torch.int32)
                .reshape(h, w, 2)
            )
        self.stats["shadow_candidates_mean"] = (
            float(np.mean(shadow_counts)) if shadow_counts else 0.0
        )
        self.stats["shadow_candidates_max"] = max(shadow_counts, default=0)
        return rgb_out.cpu().numpy(), depth_out.cpu().numpy(), seg_out.cpu().numpy()

    @torch.inference_mode()
    def render_depth_at(self, sample_x, sample_y):
        """Depth-only pass at a measured GPU resolve sample; same conservative tile bounds."""
        right = torch.linalg.cross(self.forward, self.up)
        right = right / torch.linalg.vector_norm(right)
        span = float(self.s["camera_frustum_top"]) - float(
            self.s["camera_frustum_bottom"]
        )
        shift = (
            sample_x - 0.5
        ) * span * self.w / self.h / self.w / self.near * right - (
            sample_y - 0.5
        ) * span / self.h / self.near * self.up
        output = torch.empty((self.h, self.w), device=self.device, dtype=torch.float32)
        for tile in self.tiles:
            x, y, w, h = (tile[k] for k in ("x", "y", "w", "h"))
            rays = tile["rays"] + shift
            directions = (rays @ tile["rot"]).reshape(len(rays), -1, 3)
            hits = self.selected_hits(
                tile["geom"], directions, tile["origin"], self.near, self.far, cull=True
            )
            depth = hits.min(1).values
            output[y : y + h, x : x + w] = torch.where(
                depth < self.far, depth, self.far
            ).reshape(h, w)
        return output.cpu().numpy()
