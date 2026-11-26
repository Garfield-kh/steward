"""Each Cobot gripper owns its load through sensing, planning, release and reboot."""

import numpy as np

from steward.assets.cobot import cobot_magic
from steward.ego.estimator import EgoEstimator, Track
from steward.ego.esim import Esim
from steward.ego.cobot_arm import PiperArm, PiperArmConfig


def test_two_loads_follow_their_own_encoders_and_release_independently():
    model = cobot_magic().to_dict()
    est = EgoEstimator(robot_model=model)
    est.joints = {j["id"]: j["q"]["value"][0] for j in model["body"]["joints"] if j["type"]["value"] != "fixed"}
    for side, category in [("R", "electric_screwdriver"), ("L", "toy_screw")]:
        grip = f"{side}_link6"
        pos = est.gripper_position(grip)
        t = Track(id=f"e:{side}", category_counts={category: 1.0}, pos=pos.copy(), var=np.full(3, 1e-4),
                  yaw=0., yaw_var=1e-4, size=np.array([.02, .02, .03]), observations=3,
                  first_seen=0., last_seen=0.)
        est.tracks[t.id] = t
        est.mark_held(t.id, gripper=grip, at=pos)
    assert est.tracks["e:R"].held and est.tracks["e:L"].held
    frame = {"stamp": 1., "odometry": est.odometry, "joints": {**est.joints, "R_joint1": .2, "L_joint1": -.2},
             "grippers": {f"{s}_link6": {"state": "closed", "holding": True} for s in ("L", "R")}}
    est.update_body(frame)
    scene = est.to_scene(frame)
    for side in ("L", "R"):
        np.testing.assert_allclose(scene.entity(f"e:{side}").pose.value["p"], est.gripper_position(f"{side}_link6"))
        planner = Esim(model, arm=PiperArm(model, PiperArmConfig(arm=side)))
        assert planner._held(scene).id == f"e:{side}"
    est.select_gripper("L_link6")
    est.release_held()
    assert not est.tracks["e:L"].held and est.tracks["e:R"].held
    frame["stamp"] = 2.
    frame["grippers"]["L_link6"] = {"state": "open", "holding": False}
    est.update_body(frame)
    assert est.tracks["e:R"].held
    scene = est.to_scene(frame)
    assert [(r.a["entity"], r.b["link"]) for r in scene.relations if r.type == "attached"] == [("e:R", "R_link6")]
    est.reboot()
    assert not any(t.held for t in est.tracks.values())
