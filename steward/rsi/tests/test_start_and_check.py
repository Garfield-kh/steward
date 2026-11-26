import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from helpers import FIXTURE, PACKAGE, config, load

from steward_rsi import __main__ as cli
from steward_rsi.start_state import save_start, start_poses


class StartStateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_final_state_export(self):
        written, warnings, source = save_start(FIXTURE, self.tmp / "s", [3, 4])
        self.assertEqual([p.name for p in written], ["3.json", "4.json"])
        self.assertEqual(source, "world_state.json")
        poses = json.loads(written[0].read_text())
        self.assertIn("w:plane-fuselage", poses)
        self.assertIn("w:car-wheel-fl", poses)
        self.assertNotIn("w:table-plane", poses)  # not graspable
        self.assertNotIn("w:robot-1", poses)
        self.assertNotIn("__robot__", poses)
        self.assertEqual(len(poses["w:plane-wing"]), 7)
        self.assertEqual(warnings, [])
        self.assertEqual(json.loads((self.tmp / "s" / "SOURCE.json").read_text())["seeds"], [3, 4])

    def test_sample_at_time(self):
        _, _, source = save_start(FIXTURE, self.tmp / "s", [1], at_s=2000.0)
        self.assertEqual(source, "truth_samples.jsonl@1698.143s")
        with self.assertRaises(LookupError):
            save_start(FIXTURE, self.tmp / "t", [1], at_s=1.0)

    def test_robot_entry_and_held_warning(self):
        state = load("world_state.json")
        robot = next(e for e in state["entities"] if e["role"] == "robot")
        robot["pose"]["value"] = {"p": [0.1, -1.8, 0.0], "q": [0.7071068, 0, 0, 0.7071068]}
        robot["body"]["joints"] = [{"id": "lift", "type": {"value": "prismatic"}, "q": {"value": [0.84]}},
                                   {"id": "head_mount", "type": {"value": "fixed"}, "q": {"value": []}}]
        state["relations"].append({"type": "attached", "a": {"entity": "w:plane-screw-0"}, "b": {"entity": "w:robot-1"}})
        poses, warnings = start_poses(state, with_robot=True)
        self.assertEqual(poses["__robot__"]["id"], "w:robot-1")
        self.assertAlmostEqual(poses["__robot__"]["base"][2], 1.5707963, places=5)
        self.assertEqual(poses["__robot__"]["joints"], {"lift": 0.84})
        self.assertTrue(any("holds w:plane-screw-0" in w for w in warnings))


class ControllerCheckTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "steward").mkdir()
        self.profile = self.tmp / "steward" / "var" / "dsh" / "robot-1" / "profiles" / "ego"
        self.profile.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def cfg(self):
        return config({"paths": {"repo_root": str(self.tmp)}})

    def test_key_in_env_file_and_model_override(self):
        (self.tmp / "steward" / ".env").write_text("DEEPSEEK_API_KEY=sk-controller-secret\n")
        (self.profile / "cordis.patch.yml").write_text(
            "- id: agent-default-model\n  config:\n    provider: deepseek-official\n    model: deepseek-flash\n")
        rows = cli.controller_checks(self.cfg(), environ={})
        self.assertEqual([r[0] for r in rows], ["OK", "OK"])
        self.assertIn("model deepseek-flash", rows[1][2])
        self.assertNotIn("sk-controller-secret", repr(rows))

    def test_missing_key_and_profile(self):
        shutil.rmtree(self.profile)
        rows = cli.controller_checks(self.cfg(), environ={})
        self.assertEqual([r[0] for r in rows], ["FAIL", "FAIL"])

    def test_check_offline_prints_no_key(self):
        cfg_file = self.tmp / "rsi.toml"
        cfg_file.write_text(f'[advisor]\nprovider = "claude"\n[claude]\napi_key = "sk-ant-advisor-secret"\n'
                            f'[paths]\nrepo_root = "{self.tmp}"\n[machine]\npython = "/nonexistent/python"\n')
        out = io.StringIO()
        with redirect_stdout(out):
            code = cli.main(["--config", str(cfg_file), "check", "--offline"])
        text = out.getvalue()
        self.assertEqual(code, 1)  # the fake repository has no Steward source
        self.assertIn("OK   advisor key: claude key read from rsi.toml", text)
        self.assertNotIn("sk-ant-advisor-secret", text)
        self.assertNotIn("advisor request", text)

    def test_dry_run_cli(self):
        out = io.StringIO()
        with redirect_stdout(out):
            code = cli.main(["--config", str(PACKAGE / "rsi.example.toml"), "run", "--dry-run", "--name", "t"])
        self.assertEqual(code, 0)
        self.assertIn("build-toy-assembly", out.getvalue())
        self.assertFalse((PACKAGE / "runs").exists())


if __name__ == "__main__":
    unittest.main()
