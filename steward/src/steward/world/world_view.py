"""World State view for the Steward console, rendered off the simulation thread.

The renderer keeps a private copy of the WSim model and data. Each frame copies the simulation
state under the WSim lock and renders the copy outside it, so the view never holds up physics or
sensor rendering. A changed entity set (the WSim recompiled its model) replaces the copy. Every
method must run on one thread, the view loop's, which owns the OpenGL renderers.
"""

from __future__ import annotations

from steward.assembly_profile import enabled as _assembly_enabled


import copy
from collections.abc import Callable

import mujoco

from steward.glrender import close_renderer, encode_image, free_camera
from steward.viewloop import ViewSettings
if _assembly_enabled():
    from steward.world.sim import HIDDEN_GROUP, WorldSim, shown_fingers
else:
    from steward.world.sim import HIDDEN_GROUP, WorldSim


class WorldViewRenderer:
    """Renders the World State from a free camera, walls on the camera side cut away."""

    def __init__(self, sim: Callable[[], WorldSim | None]):
        self._sim = sim
        self._source: mujoco.MjModel | None = None
        self._model: mujoco.MjModel | None = None
        self._data: mujoco.MjData | None = None
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}

    def render(self, settings: ViewSettings) -> tuple[bytes, str, dict]:
        """Render one JPEG frame of the current World State.

        :param settings: camera and image size.
        :returns: the image, its content type and ``{"t": sim time}``.
        """
        sim = self._sim()
        if sim is None:
            raise RuntimeError("no episode is loaded")
        if _assembly_enabled():
            with sim.lock:
                if self._source is not sim.model:
                    self.close()
                    self._model = copy.deepcopy(sim.model)
                    self._data = mujoco.MjData(self._model)
                    self._source = sim.model
                mujoco.mj_copyData(self._data, self._model, sim.data)
                hidden = sim.cutaway_walls(settings.camera)
                fingers = sim.finger_display()
                t = sim.time
        else:
            with sim.lock:
                if self._source is not sim.model:
                    self.close()
                    self._model = copy.deepcopy(sim.model)
                    self._data = mujoco.MjData(self._model)
                    self._source = sim.model
                mujoco.mj_copyData(self._data, self._model, sim.data)
                hidden = sim.cutaway_walls(settings.camera)
                t = sim.time
        key = (settings.width, settings.height)
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(self._model, settings.height, settings.width)
        saved = {gid: int(self._model.geom_group[gid]) for gid in hidden}
        if _assembly_enabled():
            try:
                for gid in hidden:
                    self._model.geom_group[gid] = HIDDEN_GROUP
                with shown_fingers(self._model, self._data, fingers):
                    self._renderers[key].update_scene(self._data, camera=free_camera(settings.camera))
            finally:
                for gid, group in saved.items():
                    self._model.geom_group[gid] = group
        else:
            try:
                for gid in hidden:
                    self._model.geom_group[gid] = HIDDEN_GROUP
                self._renderers[key].update_scene(self._data, camera=free_camera(settings.camera))
            finally:
                for gid, group in saved.items():
                    self._model.geom_group[gid] = group
        return encode_image(self._renderers[key].render(), "jpeg"), "image/jpeg", {"t": round(t, 3)}

    def close(self) -> None:
        for renderer in self._renderers.values():
            close_renderer(renderer)
        self._renderers.clear()
        self._source = self._model = self._data = None
