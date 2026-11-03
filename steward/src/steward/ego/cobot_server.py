"""A minimal Ego API for the Cobot Magic twin.

``steward cobot`` drives the skills from a mission thread, so there is no Ego Runtime process to
publish the robot's belief. Without one the console's Ego State panel stays empty -- and that
panel is the whole point of watching a twin. This server publishes what the mission's estimator
already knows: the rendered Ego-State and its object list.

It deliberately implements only the read side. Skills are issued by the mission, so the routes
that would let an operator drive the arm (``/skill/*``, ``/ego/control``) are absent rather than
present and failing.
"""

from __future__ import annotations

import math
import threading
from typing import Callable

from steward.ego.estimator import EgoEstimator
from steward.ego.view import EgoViewRenderer
from steward.net import NO_CONTENT, HttpError, JsonServer
from steward.viewloop import ViewLoop, ViewSettings, frame_response


class CobotEgoView:
    """Publishes one estimator's belief over HTTP for the console.

    ``live`` supplies the robot's current sensor frame. The mission looks at the table a few times
    per round, so without it the panel would show the arm frozen where the last look left it --
    while the arm is in fact mid-motion. Proprioception is continuous; only exteroception is
    look-by-look.
    """

    def __init__(self, estimator: EgoEstimator, live: Callable[[], dict] | None = None) -> None:
        self.estimator = estimator
        self.live = live
        self._lock = threading.Lock()
        renderer = EgoViewRenderer()
        self.view = ViewLoop("ego", lambda settings: self._render(renderer, settings), renderer.close).start()

    def _scene(self):
        frame = None if self.live is None else self.live()
        with self._lock:
            return self.estimator.to_scene(frame)

    def _render(self, renderer: EgoViewRenderer, settings: ViewSettings):
        scene = self._scene().to_dict()
        return renderer.render(scene, settings), "image/jpeg", {"stamp": self.estimator.stamp}

    # ------------------------------------------------------------------ routes
    def routes(self) -> dict:
        def view_frame(q, b):
            frame = self.view.latest(int(q["after"]) if q.get("after") else None)
            if frame is not None:
                return frame_response(frame, self.view.status())
            status = self.view.status()
            if status["frame"] is None and status["settings"] is None:
                raise HttpError(409, "the ego view is not configured; POST /ego/view first")
            if status["error"] and status["frame"] is None:
                raise HttpError(503, f"the ego view cannot render: {status['error']}")
            return NO_CONTENT

        return {
            ("GET", "/ego/view"): lambda q, b: self.view.status(),
            ("POST", "/ego/view"): lambda q, b: self.view.configure(ViewSettings.parse(b)),
            ("GET", "/ego/view/frame"): view_frame,
            ("GET", "/ego/summary"): lambda q, b: self.summary(),
            ("GET", "/ego/state"): lambda q, b: self._scene().to_dict(),
            # The twin raises no alerts of its own yet, but the shape is the contract: the World
            # Harness reads ["active"], and a missing key took its whole chat endpoint down.
            ("GET", "/ego/alerts"): lambda q, b: {"alerts": [], "active": []},
        }

    def summary(self) -> dict:
        """The object list behind the Ego State view, in the shape the console already reads."""
        with self._lock:
            scene = self.estimator.to_scene(None if self.live is None else self.live())
            stamp = self.estimator.stamp
            held = self.estimator.held_id
        objects = []
        for ent in scene.entities:
            if ent.role == "robot":
                continue
            try:
                size = ent.rigid.links[0].geometry["collision"][0]["size"]
            except (AttributeError, KeyError, IndexError):
                size = []
            cov = (ent.pose.unc or {}).get("cov") if ent.pose.unc else None
            objects.append({
                "id": ent.id,
                "category": ent.category.value,
                "position": [round(float(v), 3) for v in ent.pose.value["p"]],
                "sizeM": [round(float(v), 3) for v in size],
                "uncertaintyM": round(math.sqrt((cov[0] + cov[7]) / 2), 4) if cov else None,
                "existence": float(ent.existence),
                "held": ent.id == held,
            })
        return {"stamp": stamp, "mode": "cobot", "objects": objects}

    def stop(self) -> None:
        self.view.stop()


def cobot_ego_api(estimator: EgoEstimator, port: int = 0, bind_wait_s: float = 0.0,
                  live: Callable[[], dict] | None = None) -> JsonServer:
    """Serve one estimator's belief on ``port`` (0 picks a free one)."""
    view = CobotEgoView(estimator, live=live)
    server = JsonServer(view.routes(), port=port, name="ego", bind_wait_s=bind_wait_s)
    server.start()

    original_stop = server.stop

    def stop() -> None:
        view.stop()
        original_stop()

    server.stop = stop  # type: ignore[method-assign]
    return server
