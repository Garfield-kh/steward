"""Shuttle task: objects circulate table → cabinet → sideboard without ever sharing a support."""

from __future__ import annotations

import threading
import time

from steward.assets.generators import steward_bot
from steward.ego.runtime import EgoRuntime, RobotLink
from steward.ego.server import ego_api
from steward.mission import shuttle
from steward.world.server import StewardHost

SUPPORTS = {"w:table-1": "table", "w:cabinet-1": "cabinet", "w:sideboard-1": "sideboard"}


def _truth(host: StewardHost) -> dict[str, str | None]:
    out: dict[str, str | None] = {"w:cup-1": None, "w:bowl-1": None}
    for rel in host.sim.world_state()["relations"]:
        if rel["type"] == "supported-by" and rel["a"]["entity"] in out and rel["b"]["entity"] in SUPPORTS:
            out[rel["a"]["entity"]] = SUPPORTS[rel["b"]["entity"]]
    return out


def _settled(host: StewardHost, timeout: float = 15.0) -> None:
    """Wait until the world has stepped once: support relations exist only after a settle."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and None in _truth(host).values():
        time.sleep(0.05)


def test_shuttle_moves_objects_around_the_cycle(kitchen_root, tmp_path):
    host = StewardHost(kitchen_root, scenario="phase0-shuttle", speed=6.0, seed=2, run_dir=tmp_path).start()
    runtime = EgoRuntime(RobotLink(host.robot_servers["w:robot-1"].url), steward_bot().to_dict())
    runtime.start()
    api = ego_api(runtime).start()
    snapshots: list[dict] = []
    result: dict = {}

    def run() -> None:
        result["value"] = shuttle(api.url, 3, on_step=lambda i, m: snapshots.append({"move": m, "truth": _truth(host)}))

    try:
        _settled(host)
        assert _truth(host) == {"w:cup-1": "table", "w:bowl-1": "cabinet"}
        thread = threading.Thread(target=run)
        thread.start()
        while thread.is_alive():
            host.pump(0.05)
        moves = [(s["move"]["object"], s["move"]["from"], s["move"]["to"]) for s in snapshots]
        assert moves == [("bowl", "cabinet", "sideboard"), ("cup", "table", "cabinet"), ("bowl", "sideboard", "table")]
        for snap in snapshots:
            truth = snap["truth"]
            assert None not in truth.values() and len(set(truth.values())) == 2, truth
    finally:
        api.stop()
        runtime.stop()
        host.stop()
