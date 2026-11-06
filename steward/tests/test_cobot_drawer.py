"""The twin at a cabinet drawer: drive up, grasp the handle, pull it open, and check World State.

The cabinet's drawer slides toward the robot, so a mobile manipulator pulls it by backing away
while it holds the handle. Here the twin has a fixed standoff (the chassis would hit the carcass
closer in) and pulls with the arm: the world makes a grasped drawer follow the pinch point on a PD
force along its slide, which is what makes an arm pull a drawer at all.
"""

import pytest

from steward.assets.cobot import ARM_MOUNT_X, cobot_magic
from steward.authoring.service import AuthoringService
from steward.ego.cobot_skills import CobotSkills, MotionConfig
from steward.world.sim import WorldSim

# The cabinet's own front face is at +depth/2 = +0.25 in its frame, and the handle sticks out to
# 0.27 (generators.cabinet_with_drawer: the 0.03 m steel bar at the drawer's front). Turned a half
# turn (yaw = pi) its drawer faces the twin: front face at x = 1.2 - 0.25, handle at 1.2 - 0.27.
CABINET_AT = 1.2
CABINET_YAW = 3.141592653589793
HANDLE = [CABINET_AT - 0.27, 0.0, 0.676]
AXIS = [-1.0, 0.0, 0.0]  # opening direction: toward the robot
STANDOFF = 0.58 - ARM_MOUNT_X  # the arm mounts 0.58 m from the origin, the carcass's face at 0.95
LIFT = 0.60              # shoulders at 0.755: the wrist reaches 0.811 with the tool down
TRAVEL = 0.18            # a real opening, and the pinch stays well inside the reachable band


@pytest.fixture(scope="module")
def rig(tmp_path_factory):
    root = tmp_path_factory.mktemp("cobot-drawer")
    service = AuthoringService(root)
    from steward.assets.generators import GENERATORS

    GENERATORS.setdefault("cobot_magic", cobot_magic)

    def publish(name: str, generator: str, params: dict) -> None:
        created = service.create_asset(generator=generator, params=params)
        service.publish_asset(name, created["template"], meta={"generator": generator, "params": params})

    publish("cobot-drawer-room", "building", {"width": 5.0, "depth": 4.0, "height": 2.6})
    publish("cobot-drawer-cabinet", "cabinet_with_drawer",
            {"width": 0.6, "depth": 0.5, "height": 0.8, "drawer_height": 0.2})
    publish("cobot-magic", "cobot_magic", {})
    spec = {
        "name": "cobot-drawer",
        "entities": [
            {"id": "w:room", "asset": "cobot-drawer-room", "at": {"x": 0.0, "y": 0.0}},
            {"id": "w:cobot-1", "asset": "cobot-magic", "at": {"x": 0.0, "y": 0.0}},
            {"id": "w:cabinet-1", "asset": "cobot-drawer-cabinet",
             "at": {"x": CABINET_AT, "y": 0.0, "yaw": CABINET_YAW}},
        ],
        "regions": [{"id": "flat", "label": "flat", "polygon": [[-2.5, -2], [2.5, -2], [2.5, 2], [-2.5, 2]]}],
    }
    service.compose_scene(spec)
    scene = service.load_scene("cobot-drawer")
    sim = WorldSim(scene, speed=3.0)
    sim.start()
    robot = next(e["id"] for e in scene["entities"] if e["role"] == "robot")
    skills = CobotSkills(command=lambda cmd: sim.command(robot, cmd),
                         state=lambda: sim.sensors(robot), config=MotionConfig(tick_s=0.01))
    try:
        yield sim, robot, skills
    finally:
        sim.stop()


def _drawer_q(sim: WorldSim) -> float:
    """The drawer's own slide position, as World State reports it (not the mission's log)."""
    cabinet = next(e for e in sim.world_state()["entities"] if e["id"] == "w:cabinet-1")
    return float(next(j for j in cabinet["body"]["joints"] if j["id"] == "drawer_slide")["q"]["value"][0])


def _run(skills: CobotSkills, name: str, plan: dict):
    job = skills.start(name, {"id": f"test-{name}", "plan": plan})
    job = skills.wait(job.id, 180.0)
    return job


def test_the_twin_drives_up_and_pulls_the_drawer_open(rig) -> None:
    sim, robot, skills = rig
    assert _drawer_q(sim) == pytest.approx(0.0, abs=0.005)

    job = _run(skills, "navigate", {"at": [STANDOFF, 0.0, 0.0], "path": [[STANDOFF, 0.0]]})
    assert job.state == "succeeded", job.message
    job = _run(skills, "lift", {"height": LIFT})
    assert job.state == "succeeded", job.message
    job = _run(skills, "drawer", {"handle": HANDLE, "axis": AXIS, "travel": TRAVEL,
                                  "gripper": "R_link6"})
    assert job.state == "succeeded", f"{job.message}\n" + "\n".join(job.log)

    # The drawer itself moved: the handle was grasped, and the pull dragged the slide joint.
    assert _drawer_q(sim) > 0.10, f"the drawer opened only {_drawer_q(sim) * 100:.0f} cm"
    # And it was let go: the gripper no longer holds the handle, or the drawer would be locked
    # to the hand forever.
    grip = sim.sensors(robot)["grippers"]["R_link6"]
    assert grip["holding"] is False
    assert any(e["type"] == "grasp/handle-release" for e in sim.events), sim.events[-5:]
