"""Place into an open drawer docks at the drawer mouth, not the host furniture body."""

from __future__ import annotations

import math

import numpy as np
import pytest

from steward.assets.generators import steward_bot
from steward.ego.esim import Esim
from steward.ego.estimator import EgoEstimator, Track
from steward.ego.planner import entity_box


def _track(est: EgoEstimator, eid: str, category: str, pos: list[float], size: list[float],
           yaw: float = 0.0) -> Track:
    track = Track(id=eid, category_counts={category: 5.0}, pos=np.array(pos, dtype=float),
                  var=np.full(3, 1e-4), yaw=yaw, yaw_var=1e-3, size=np.array(size, dtype=float),
                  observations=10, first_seen=0.0, last_seen=0.0, existence=1.0)
    est.tracks[eid] = track
    return track


def _drawer_place_scene(*, drawer_yaw: float = 0.0, held: bool = True):
    """Open drawer facing the room (+x); host sideboard behind the mouth with positive opening.

    A far wall puts the room centre in +x so ``_front`` points the drawer outward even when the
    drawer track's own yaw is wrong (``drawer_yaw``).
    """
    model = steward_bot().to_dict()
    est = EgoEstimator(robot_model=model)
    est.odometry = {"x": 1.35, "y": 0.0, "yaw": math.pi, "sigma": {"pos": 0.0, "yaw": 0.0}}
    est.joints = {"lift": 0.85, "reach": 0.15, "wrist_yaw": 0.0, "wrist_roll": 0.0}
    _track(est, "e:wall", "wall", [3.0, 0.0, 1.2], [0.2, 4.0, 2.4])
    # Host front near x≈0.2; drawer sticks out to ≈0.55 so opening stays positive.
    _track(est, "e:22", "sideboard", [-0.15, 0.0, 0.4], [0.70, 1.0, 0.8])
    # size [along-yaw-x, along-yaw-y, z]; with yaw=π/2 the long extent still covers the pull-out.
    _track(est, "e:21", "drawer", [0.30, 0.0, 0.55], [0.46, 0.50, 0.18], yaw=drawer_yaw)
    if held:
        teddy = _track(est, "e:77", "teddy_bear", [1.20, 0.05, 0.95], [0.12, 0.10, 0.18])
        est.held_id, teddy.held = teddy.id, True
    return model, est.to_scene()


def test_drawer_mouth_clear_includes_the_host_sideboard():
    model, scene = _drawer_place_scene()
    esim = Esim(model)
    drawer = scene.find("e:21")
    assert esim._drawer_mouth_clear(scene, drawer) == {"e:21", "e:22"}


def test_close_keeps_handle_on_open_drawer_front():
    """Close plans the handle on the pulled-out front, not the host carcass face.

    Clamping to the face then pushing ``opening`` parks the close end pose inside the sideboard
    (seed2 drawers: end pose collides with host).
    """
    model, scene = _drawer_place_scene(held=False)
    esim = Esim(model)
    drawer = scene.find("e:21")
    host = scene.find("e:22")
    front = esim._front(scene, drawer)
    hbox = esim._front(scene, host)
    n = (math.cos(front.yaw), math.sin(front.yaw))
    mouth = front.cx * n[0] + front.cy * n[1] + front.hx
    face = hbox.cx * n[0] + hbox.cy * n[1] + hbox.hx + 0.02
    assert mouth > face + 0.03

    receipt = esim.predict(scene, "close", {"target": "e:21"})
    assert receipt["feasible"], receipt["issues"]
    handle = receipt["plan"]["handle"]
    handle_along = handle[0] * n[0] + handle[1] * n[1]
    # Stay on the open front (near mouth), not snapped back to the carcass face.
    assert handle_along > face + 0.05
    assert abs(handle_along - (mouth - 0.015)) < 0.04
    end = receipt["plan"]["end"]
    end_along = end[0] * n[0] + end[1] * n[1]
    # End stays outside the host slab (half-extent along the outward axis).
    assert end_along > hbox.cx * n[0] + hbox.cy * n[1] + hbox.hx


def test_open_still_clamps_handle_to_host_face_when_drawer_is_swollen():
    """Open keeps the face clamp so a proud swollen box does not grasp empty air."""
    model = steward_bot().to_dict()
    est = EgoEstimator(robot_model=model)
    est.odometry = {"x": 1.35, "y": 0.0, "yaw": math.pi, "sigma": {"pos": 0.0, "yaw": 0.0}}
    est.joints = {"lift": 0.85, "reach": 0.15, "wrist_yaw": 0.0, "wrist_roll": 0.0}
    _track(est, "e:wall", "wall", [3.0, 0.0, 1.2], [0.2, 4.0, 2.4])
    # Host face near x≈0.35; drawer centre inside, but hx swollen so cx+hx is proud of the face.
    _track(est, "e:22", "sideboard", [0.0, 0.0, 0.4], [0.70, 1.0, 0.8])
    _track(est, "e:21", "drawer", [0.05, 0.0, 0.55], [0.90, 0.50, 0.18])
    scene = est.to_scene()
    esim = Esim(model)
    drawer = scene.find("e:21")
    host = scene.find("e:22")
    front = esim._front(scene, drawer)
    hbox = esim._front(scene, host)
    n = (math.cos(front.yaw), math.sin(front.yaw))
    mouth = front.cx * n[0] + front.cy * n[1] + front.hx
    face = hbox.cx * n[0] + hbox.cy * n[1] + hbox.hx + 0.02
    assert mouth > face + 0.03
    assert esim._opening(scene, drawer) <= 0.2

    receipt = esim.predict(scene, "open", {"target": "e:21"})
    assert receipt["feasible"], receipt["issues"]
    handle = receipt["plan"]["handle"]
    handle_along = handle[0] * n[0] + handle[1] * n[1]
    assert handle_along <= face + 0.005


def test_pulled_open_memory_floors_opening_when_host_is_missing():
    """A robot-pulled drawer keeps a usable opening after its host support box disappears."""
    from steward.scene.schema import Est

    model, scene = _drawer_place_scene()
    esim = Esim(model)
    # Drop the host so geometric protrusion is 0 (same failure mode as a lost sideboard track).
    scene.entities = [e for e in scene.entities if e.id != "e:22"]
    drawer = scene.find("e:21")
    assert esim._opening(scene, drawer) == 0.0

    drawer.state_vars["pulledOpen"] = Est(value=True, src="inferred")
    assert esim._opening(scene, drawer) == pytest.approx(esim.config.drawer_pull)

    receipt = esim.predict(scene, "place", {"on": "e:21"})
    assert receipt["feasible"], receipt["issues"]


def test_estimator_drawer_opening_floors_while_pulled_open():
    """Summary openingM stays above a false 0 cm while the robot's pull is still open."""
    from steward.ego.estimator import DRAWER_REMEMBERED_OPEN_M

    model = steward_bot().to_dict()
    est = EgoEstimator(robot_model=model)
    _track(est, "e:21", "drawer", [0.30, 0.0, 0.55], [0.46, 0.50, 0.18])
    # No furniture host → protrusion 0.
    assert est.drawer_opening(est.tracks["e:21"]) == 0.0
    est._pulled_open.add("e:21")
    assert est.drawer_opening(est.tracks["e:21"]) == pytest.approx(DRAWER_REMEMBERED_OPEN_M)
    scene = est.to_scene()
    assert scene.find("e:21").state_vars["pulledOpen"].value is True


def test_place_into_open_drawer_docks_past_the_mouth():
    model, scene = _drawer_place_scene()
    esim = Esim(model)
    drawer = scene.find("e:21")
    front = esim._front(scene, drawer)
    assert esim._opening(scene, drawer) > 0.15

    receipt = esim.predict(scene, "place", {"on": "e:21"})
    assert receipt["feasible"], receipt["issues"]
    dock = receipt["plan"]["dock"]
    outward = (math.cos(front.yaw), math.sin(front.yaw))
    mouth = front.cx * outward[0] + front.cy * outward[1] + front.hx
    dock_along = dock[0] * outward[0] + dock[1] * outward[1]
    assert dock_along > mouth
    # Base faces into the drawer (opposite the outward normal).
    assert abs(math.cos(receipt["plan"]["yaw"] - front.yaw) + 1.0) < 0.2


def test_mouth_only_dock_uses_room_front_when_drawer_yaw_is_wrong():
    """``mouth_only`` reorients via ``_front``; a mis-yawed drawer track still docks on +x."""
    model, scene = _drawer_place_scene(drawer_yaw=math.pi / 2)
    esim = Esim(model)
    drawer = scene.find("e:21")
    front = esim._front(scene, drawer)
    assert abs(front.yaw) < 0.2  # room-facing +x, not the track's π/2

    approach = esim._approach(
        scene, drawer, (0.35, 0.0), 0.70, {"e:77"},
        dock_clear=esim._drawer_mouth_clear(scene, drawer), mouth_only=True)
    assert approach["dock"][0] > front.cx + front.hx
    assert abs(abs(approach["yaw"]) - math.pi) < 0.35

    receipt = esim.predict(scene, "place", {"on": "e:21"})
    assert receipt["feasible"], receipt["issues"]


def test_dock_clear_allows_a_mouth_dock_inside_a_swollen_host_footprint():
    """Host box covers every reachable mouth dock; clearing it for footprint makes the side legal."""
    from steward.math3d import quat_from_yaw

    model, scene = _drawer_place_scene()
    esim = Esim(model)
    # Cap planner reach so backoff cannot step past the swollen host without dock_clear.
    esim.reach_max = 0.40
    drawer = scene.find("e:21")
    host = scene.find("e:22")
    host.pose.value["p"][0] = 0.35
    geom = host.rigid.links[0].geometry["collision"][0]
    geom["size"] = [1.80, 1.0, 0.8]
    front = esim._front(scene, drawer)
    mouth = front.cx + front.hx
    # Already facing the mouth inside predock range so nav need not cross the host slab.
    stand = mouth + esim.half_x + esim.config.dock_gap + 0.10
    me = scene.entity(scene.self_id)
    me.pose.value["p"] = [stand, 0.0, me.pose.value["p"][2]]
    me.pose.value["q"] = list(quat_from_yaw(math.pi))
    hbox = entity_box(host)[0]
    assert hbox.cx + hbox.hx > mouth + esim.reach_max

    with pytest.raises(Exception) as blocked:
        esim._approach(scene, drawer, (front.cx + 0.05, front.cy), 0.70, {"e:77"}, mouth_only=True)
    assert "dock pose collides" in str(blocked.value)

    approach = esim._approach(
        scene, drawer, (front.cx + 0.05, front.cy), 0.70, {"e:77"},
        dock_clear=esim._drawer_mouth_clear(scene, drawer), mouth_only=True)
    assert approach["dock"][0] > mouth


def _interface_plane_in_drawer_scene():
    """Held fuselage after a π base turn: restoring graspWorldYawRad needs a ±π wrist at the mouth.

    Grasp heading −π/2 with base yaw 0 → relative −π/2. After driving to face the drawer (base π)
    the held world yaw is π/2; mouth dock_yaw is π, so transport yaw −π/2 demands wrist −π.
    """
    model = steward_bot().to_dict()
    est = EgoEstimator(robot_model=model)
    est.odometry = {"x": 1.35, "y": 0.0, "yaw": math.pi, "sigma": {"pos": 0.0, "yaw": 0.0}}
    est.joints = {"lift": 0.85, "reach": 0.15, "wrist_yaw": 0.0, "wrist_roll": 0.0}
    _track(est, "e:wall", "wall", [3.0, 0.0, 1.2], [0.2, 4.0, 2.4])
    _track(est, "e:22", "sideboard", [-0.15, 0.0, 0.4], [0.70, 1.0, 0.8])
    _track(est, "e:21", "drawer", [0.30, 0.0, 0.55], [0.46, 0.50, 0.18])
    _track(est, "e:table", "table", [1.35, -1.2, 0.375], [0.8, 0.8, 0.75])
    held = _track(est, "e:plane", "airplane_fuselage", [1.20, 0.05, 0.95], [0.18, 0.045, 0.04],
                  yaw=math.pi / 2)
    held.grasp_world_yaw = -math.pi / 2
    est.held_id, held.held = held.id, True
    return model, est.to_scene()


def test_drawer_place_skips_transport_yaw_that_would_need_a_half_turn_wrist():
    model, scene = _interface_plane_in_drawer_scene()
    esim = Esim(model)
    held = scene.entity("e:plane")
    orient = (held.state_vars.get("orientation").value
              if held.state_vars.get("orientation") is not None else {})
    assert orient.get("graspWorldYawRad") == pytest.approx(-math.pi / 2)
    assert esim._screw_holes(held)

    # Surface place still restores transport yaw (assembly intent).
    on_table = esim.predict(scene, "place", {"on": "e:table", "offset": [0.0, 0.0]})
    assert any("RSI_TRANSPORT_YAW" in i for i in on_table["issues"])

    receipt = esim.predict(scene, "place", {"on": "e:21"})
    assert receipt["feasible"], receipt["issues"]
    assert not any("RSI_TRANSPORT_YAW" in i for i in receipt["issues"])
    assert not any("wrist turn" in i and "beyond" in i for i in receipt["issues"])
