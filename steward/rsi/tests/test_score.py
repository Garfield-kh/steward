import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

from helpers import FIXTURE, GOAL, SPEC, REPO_ROOT, deepcopy, load

from steward_rsi.scene import World, derive_kits, goal_from_service, select_kit, spec_from_service
from steward_rsi.score import funnel, markdown, score_episode, score_runs

S = math.sqrt(0.5)


def car_assembled(lateral_x_mm=0.0, screws=("w:car-screw-0", "w:car-screw-1", "w:car-screw-2", "w:car-screw-3"),
                  skip=()):
    """The fixture's initial world with the car body upright and four wheels fastened on its sides."""
    state = deepcopy(load("world_initial.json"))
    w = World(state)
    body = w.entities["w:car-body"]
    origin = list(body["pose"]["value"]["p"])
    body["pose"]["value"]["q"] = [1.0, 0.0, 0.0, 0.0]
    holes = w.holes("w:car-body")
    wheels = ["w:car-wheel-fl", "w:car-wheel-fr", "w:car-wheel-rl", "w:car-wheel-rr"]
    for k, (wheel, screw) in enumerate(zip(wheels, screws)):
        if k in skip:
            continue
        hole = holes[f"wheel_{k}"]
        side = 1.0 if hole["axis"][1] > 0 else -1.0
        # Wheel z (hub axis) along the body-hole axis; the wheel's flat face on the body side.
        q = [S, -S, 0.0, 0.0] if side > 0 else [S, S, 0.0, 0.0]
        p = [origin[0] + hole["p"][0] + lateral_x_mm / 1000.0, origin[1] + side * 0.05, origin[2] + hole["p"][2]]
        w.entities[wheel]["pose"]["value"] = {"p": p, "q": q}
        w.entities[screw]["stateVars"] = {"fastener": {"value": {"holes": [
            {"entity": wheel, "link": "body", "hole": "hub"},
            {"entity": "w:car-body", "link": "body", "hole": f"wheel_{k}"}]}}}
    return state


def write_run(root: Path, name: str, final: dict, initial: dict | None = None) -> Path:
    run = root / name
    run.mkdir(parents=True)
    (run / "world_initial.json").write_text(json.dumps(initial or load("world_initial.json")))
    (run / "world_state.json").write_text(json.dumps(final))
    (run / "episode.json").write_text(json.dumps({"seed": 1, "endReason": "turn-ended"}))
    return run


class SceneTest(unittest.TestCase):
    def test_bundled_goal_matches_repository(self):
        service = REPO_ROOT / "steward/src/steward/authoring/service.py"
        if not service.is_file():
            self.skipTest("repository source not present")
        self.assertEqual(goal_from_service(service), GOAL)

    def test_bundled_spec_matches_repository(self):
        service = REPO_ROOT / "steward/src/steward/authoring/service.py"
        if not service.is_file():
            self.skipTest("repository source not present")
        self.assertEqual(spec_from_service(service), SPEC)

    def test_spec_reader_rejects_code(self):
        path = Path(tempfile.mkdtemp()) / "service.py"
        try:
            path.write_text("def toy_assembly_spec():\n    return {'entities': open('x').read()}\n")
            with self.assertRaises(ValueError):
                spec_from_service(path)
        finally:
            shutil.rmtree(path.parent)

    def test_kits_from_mid_assembly_state(self):
        # The final state of a finished airplane as the start state: fastened screws no longer rest
        # on the table and the fuselage rests on its parts, but table and screws come from the spec.
        kits = {k.body: k for k in derive_kits(GOAL, SPEC, World(load("world_state.json")))}
        plane = kits["w:plane-fuselage"]
        self.assertEqual(plane.table, "w:table-plane")
        self.assertEqual(plane.screws, [f"w:plane-screw-{i}" for i in range(4)])
        self.assertEqual(kits["w:car-body"].table, "w:table-car")

    def test_kits_from_goal_and_initial_state(self):
        initial = World(load("world_initial.json"))
        kits = {k.body: k for k in derive_kits(GOAL, SPEC, initial)}
        self.assertEqual(set(kits), {"w:plane-fuselage", "w:car-body"})
        car = kits["w:car-body"]
        self.assertEqual(car.table, "w:table-car")
        self.assertEqual(car.screws, [f"w:car-screw-{i}" for i in range(4)])
        self.assertEqual(car.hole_asset, {f"wheel_{i}": "car-wheel" for i in range(4)})
        plane = kits["w:plane-fuselage"]
        self.assertEqual(plane.hole_asset, {"wing_0": "airplane-wing", "wing_1": "airplane-wing",
                                            "gear_0": "airplane-landing-gear", "gear_1": "airplane-landing-gear"})
        self.assertEqual(select_kit(list(kits.values()), initial, "car").body, "w:car-body")
        self.assertEqual(select_kit(list(kits.values()), initial, "airplane").body, "w:plane-fuselage")
        with self.assertRaises(ValueError):
            select_kit(list(kits.values()), initial, "boat")


class AirplaneFixtureTest(unittest.TestCase):
    def test_success_episode(self):
        r = score_episode(FIXTURE, GOAL, SPEC, "airplane", 20.0, 6.0)
        self.assertTrue(r["valid"])
        self.assertEqual(r["counts"], {"placed": 4, "aligned": 4, "seated": 4, "total": 4})
        self.assertAlmostEqual(r["maxLateralMm"], 16.555, places=2)
        self.assertTrue(r["complete"])
        self.assertFalse(r["completeStrict"])
        self.assertTrue(r["supportedByKitTable"])
        self.assertEqual(r["kitHeldByRobot"], [])
        self.assertEqual(r["foreignMoved"], {})
        self.assertEqual(r["foreignScrewsUsed"], [])
        self.assertEqual(r["seatedMilestonesWallS"], {"1": 1145.2, "2": 1698.1, "3": 3781.6, "4": 4902.3})
        self.assertEqual(r["samplesRead"], 6)
        self.assertEqual(r["holeFirstSeatedWallS"], {"wing_0": 1145.2, "wing_1": 1698.1, "gear_1": 3781.6,
                                                     "gear_0": 4902.3})
        self.assertEqual(r["seed"], 800628)
        summary = funnel([r])
        self.assertEqual(summary["worstStage"], "complete_strict")  # every 20 mm stage converts fully

    def test_same_episode_scored_as_car(self):
        r = score_episode(FIXTURE, GOAL, SPEC, "car", 20.0, 6.0)
        self.assertEqual(r["counts"]["seated"], 0)
        self.assertFalse(r["complete"])
        self.assertIn("w:plane-fuselage", r["foreignMoved"])  # building the airplane disturbs the other kit

    def test_success_scored_from_its_own_final_state(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            final = load("world_state.json")
            run = write_run(tmp, "resumed", final, initial=final)
            r = score_episode(run, GOAL, SPEC, "airplane", 20.0, 6.0)
            self.assertEqual(r["kit"]["table"], "w:table-plane")
            self.assertEqual(r["counts"], {"placed": 4, "aligned": 4, "seated": 4, "total": 4})
            self.assertTrue(all(row["screwAuthorized"] for row in r["interfaces"]))
            self.assertTrue(r["complete"])
        finally:
            shutil.rmtree(tmp)

    def test_threshold_is_configurable(self):
        self.assertFalse(score_episode(FIXTURE, GOAL, SPEC, "airplane", 15.0, 6.0)["complete"])


class CarSyntheticTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_complete_car(self):
        run = write_run(self.tmp, "ok", car_assembled(lateral_x_mm=3.0))
        r = score_episode(run, GOAL, SPEC, "car", 20.0, 6.0)
        self.assertEqual(r["counts"], {"placed": 4, "aligned": 4, "seated": 4, "total": 4})
        self.assertAlmostEqual(r["maxLateralMm"], 3.0, places=3)
        self.assertTrue(r["complete"])
        self.assertTrue(r["completeStrict"])

    def test_lateral_error_between_thresholds(self):
        r = score_episode(write_run(self.tmp, "x", car_assembled(lateral_x_mm=10.0)), GOAL, SPEC, "car", 20.0, 6.0)
        self.assertTrue(r["complete"])
        self.assertFalse(r["completeStrict"])

    def test_wrong_kit_screw(self):
        final = car_assembled(screws=("w:car-screw-0", "w:plane-screw-0", "w:car-screw-2", "w:car-screw-3"))
        r = score_episode(write_run(self.tmp, "y", final), GOAL, SPEC, "car", 20.0, 6.0)
        self.assertEqual(r["foreignScrewsUsed"], ["w:plane-screw-0"])
        self.assertFalse(r["complete"])

    def test_partial_assembly_funnel(self):
        write_run(self.tmp, "a", car_assembled())
        write_run(self.tmp, "b", car_assembled(skip=(2, 3)))
        write_run(self.tmp, "c", load("world_initial.json"))
        (self.tmp / "d").mkdir()
        (self.tmp / "d" / "episode.json").write_text("{}")  # no snapshot: infrastructure failure
        results, summary = score_runs([self.tmp], GOAL, SPEC, "car", 20.0, 6.0, out_dir=self.tmp / "score")
        self.assertEqual(summary["episodes"], 4)
        self.assertEqual(summary["validEpisodes"], 3)
        self.assertEqual(summary["interfaces"], 12)
        stages = {s["stage"]: s for s in summary["stages"]}
        self.assertEqual((stages["placed"]["reached"], stages["placed"]["of"]), (6, 12))
        self.assertEqual((stages["seated"]["reached"], stages["seated"]["of"]), (6, 6))
        self.assertEqual((stages["complete"]["reached"], stages["complete"]["of"]), (1, 1))
        self.assertEqual(summary["worstStage"], "placed")
        self.assertEqual(stages["complete_strict"]["of"], 1)
        self.assertTrue((self.tmp / "score" / "funnel.md").is_file())
        text = markdown(results, summary)
        self.assertIn("| placed | interface | 6 | 12 | 50% |", text)
        self.assertIn("invalid", text)

    def _with_wheel(self, change):
        final = car_assembled()
        change(World(final).entities["w:car-wheel-fl"]["pose"]["value"])
        return score_episode(write_run(self.tmp, "bad", final), GOAL, SPEC, "car", 20.0, 6.0)

    def test_fastener_record_does_not_replace_geometry(self):
        def turned(pose):
            pose["q"] = [1.0, 0.0, 0.0, 0.0]  # hub axis vertical, 90 degrees off the body-hole axis

        def pulled_out(pose):
            pose["p"][1] += 0.10  # 10 cm out along the hole axis

        def flipped(pose):
            pose["q"] = [S, S, 0.0, 0.0]  # hub axis opposite to the body-hole axis

        for change in (turned, pulled_out, flipped):
            with self.subTest(change=change.__name__):
                r = self._with_wheel(change)
                row = next(i for i in r["interfaces"] if i["hole"] == "wheel_0")
                self.assertIsNotNone(row["screw"])
                self.assertFalse(row["placed"])
                self.assertFalse(row["seated"])
                self.assertEqual(r["counts"]["seated"], 3)
                self.assertFalse(r["complete"])
                self.assertFalse(r["completeStrict"])
                shutil.rmtree(self.tmp / "bad")

    def test_unseated_but_placed_wheel(self):
        final = car_assembled()
        w = World(final)
        del w.entities["w:car-screw-3"]["stateVars"]
        r = score_episode(write_run(self.tmp, "z", final), GOAL, SPEC, "car", 20.0, 6.0)
        row = next(i for i in r["interfaces"] if i["hole"] == "wheel_3")
        self.assertTrue(row["placed"] and row["aligned"])
        self.assertFalse(row["seated"])
        self.assertFalse(r["complete"])


if __name__ == "__main__":
    unittest.main()
