from __future__ import annotations

import math
import threading
import time

import pytest

from steward.world.sim import WorldSim


def test_running_clock_restarts_after_reset_and_does_not_jump_on_speed_change(kitchen_scene):
    sim = WorldSim(kitchen_scene, speed=4.0)
    sim.start()
    try:
        time.sleep(0.6)
        assert sim.time > 1.5
        sim.reset()
        time.sleep(0.3)
        assert sim.time < 1.6, "a reset must restart the clock instead of fast-forwarding to the old time"
        sim.speed = 1.0
        time.sleep(0.5)
        before = sim.time
        sim.speed = 8.0
        time.sleep(0.25)
        assert sim.time - before < 3.0, "raising the speed must not replay elapsed wall time"
    finally:
        sim.stop()


def test_free_camera_view_hides_walls_on_the_camera_side_only(kitchen_scene):
    sim = WorldSim(kitchen_scene)
    south = sim.cutaway_walls({"azimuth": 90, "elevation": -50})
    southeast = sim.cutaway_walls({"azimuth": 135, "elevation": -45})
    assert len(south) == 1 and len(southeast) == 2
    assert sim.cutaway_walls({"azimuth": 90, "elevation": -89}) == []
    groups = sim.model.geom_group.copy()
    png = sim.render({"lookat": [0, 0, 0.5], "distance": 8, "azimuth": 90, "elevation": -50}, 160, 120)
    assert png.startswith(b"\x89PNG")
    assert (sim.model.geom_group == groups).all(), "rendering must restore the wall geoms"


def test_ego_view_rebuilds_do_not_corrupt_world_renders(kitchen_scene):
    from steward.assets.generators import steward_bot
    from steward.ego.estimator import EgoEstimator
    from steward.ego.view import EgoViewRenderer
    from steward.viewloop import ViewSettings

    sim = WorldSim(kitchen_scene, seed=1)
    camera = {"lookat": [0, 0, 0.5], "distance": 8, "azimuth": 90, "elevation": -50}
    before = sim.render(camera, 160, 120)
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    view = EgoViewRenderer()
    for _ in range(3):
        est.update(sim.sensors("w:robot-1"))
        est.tracks.pop(next(iter(est.tracks)), None)  # change the entity set to force a rebuild
        assert view.render(est.to_scene().to_dict(), ViewSettings(camera, 10, 160, 120)).startswith(b"\xff\xd8")
    view.close()
    assert sim.render(camera, 160, 120) == before
    sim.stop()


def test_render_from_worker_thread_is_served_by_main_thread_pump(kitchen_scene):
    sim = WorldSim(kitchen_scene)
    result: dict = {}

    def worker() -> None:
        result["png"] = sim.render(sim.robot_camera_name("w:robot-1"), 160, 120)

    thread = threading.Thread(target=worker)
    thread.start()
    while thread.is_alive():
        sim.pump(0.05)
    thread.join()
    assert result["png"].startswith(b"\x89PNG")


def test_render_without_main_thread_pump_times_out_without_blocking_physics(kitchen_scene):
    sim = WorldSim(kitchen_scene)
    errors: list[Exception] = []

    def worker() -> None:
        try:
            sim.render("overview", 160, 120, timeout=0.2)
        except Exception as exc:  # noqa: BLE001 - captured for the assertion
            errors.append(exc)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    t0 = sim.time
    sim.step_seconds(0.1)
    assert isinstance(errors[0], TimeoutError)
    assert sim.time == pytest.approx(t0 + 0.1, abs=0.01)


def test_a_woken_control_loop_holds_the_clock_until_it_waits_again(kitchen_scene):
    sim = WorldSim(kitchen_scene, speed=math.inf)
    sim.start()
    try:
        woke = sim.wait_until(0.5, hold="loop")
        assert 0.5 <= woke < 0.5 + sim.timestep * 1.5, "the world stops at the loop's wake-up time"
        time.sleep(0.2)
        assert sim.time == woke, "the world does not step while the woken loop runs"
        sim.release("loop")
        time.sleep(0.2)
        assert sim.time > woke + 1.0, "released, the world runs on as fast as it can"
    finally:
        sim.stop()


def test_a_woken_loop_that_renews_its_hold_keeps_the_clock_past_the_lease(kitchen_scene, monkeypatch):
    from steward.world import sim as sim_module

    monkeypatch.setattr(sim_module, "HOLD_LEASE_S", 0.3)
    sim = WorldSim(kitchen_scene, speed=math.inf)
    sim.start()
    try:
        woke = sim.wait_until(0.5, hold="perception")
        for _ in range(6):  # a remote segmentation request of 0.6 s
            time.sleep(0.1)
            sim.renew("perception")
        assert sim.time == woke, "renewed, the hold outlives its lease"
        time.sleep(0.5)
        assert sim.time > woke, "not renewed, the lease runs out and the world runs on"
    finally:
        sim.stop()


def test_with_freeze_idle_the_world_steps_only_while_a_caller_waits_for_time_to_pass(kitchen_scene):
    sim = WorldSim(kitchen_scene, speed=math.inf, freeze_idle=True)
    sim.start()
    try:
        def perceive_once():
            sim.wait_until(0.5, hold="perception", pulls=False)
            sim.release("perception")

        threading.Thread(target=perceive_once, daemon=True).start()
        time.sleep(0.2)
        assert sim.time == 0.0, "a loop that only perceives does not make the world step"
        assert sim.wait_until(1.0) >= 1.0, "a caller waiting for a later time makes it step"
        t = sim.time
        time.sleep(0.2)
        assert sim.time == t, "then it stands still again"
    finally:
        sim.stop()


def test_console_box_says_whether_an_object_lies_upright():
    """A face-down object's yaw reads as the mirrored heading, so the console GT summary also says
    whether the entity's +z axis still points up - same threshold as the relative-poses goal."""
    from types import SimpleNamespace

    from steward.math3d import quat_mul
    from steward.world.console import _box

    c, s = math.cos(math.pi / 4), math.sin(math.pi / 4)
    up = SimpleNamespace(center=[0.1, 0.2, 0.77], size=[0.115, 0.115, 0.006], quat=[c, 0.0, 0.0, s])
    down = SimpleNamespace(center=[0.1, 0.2, 0.77], size=[0.115, 0.115, 0.006], quat=quat_mul([0.0, 1.0, 0.0, 0.0], [c, 0.0, 0.0, s]))
    assert _box("w:x", "puzzle_piece", "object", up)["upright"] is True
    assert _box("w:y", "puzzle_piece", "object", down)["upright"] is False
