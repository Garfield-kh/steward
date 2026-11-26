import unittest
from pathlib import Path

from helpers import config, PACKAGE

from steward_rsi.config import ConfigError, Secret, load, parse, resolve_api_key


class ConfigTest(unittest.TestCase):
    def test_defaults(self):
        cfg = config()
        self.assertEqual(cfg.advisor.name, "deepseek")
        self.assertEqual(cfg.advisor.model, "deepseek-flash")
        self.assertEqual(cfg.machine.backend, "gpu")
        self.assertEqual(cfg.machine.devices, [0, 1, 2, 3])
        self.assertEqual(cfg.task.threshold_mm, 20.0)
        self.assertEqual(cfg.task.strict_mm, 6.0)
        self.assertIsNone(cfg.task.start_state_dir)
        self.assertTrue(cfg.task.task_file.name == "airplane.md" and cfg.task.task_file.is_file())

    def test_example_file_parses(self):
        cfg = load(PACKAGE / "rsi.example.toml")
        self.assertEqual(cfg.advisor.name, "deepseek")
        self.assertEqual(cfg.repo_root, PACKAGE.parents[1])
        self.assertEqual(cfg.output_dir, PACKAGE / "runs")
        self.assertEqual(cfg.task.task_file, PACKAGE / "src" / "steward_rsi" / "tasks" / "airplane.md")
        self.assertTrue(cfg.task.task_file.is_file())

    def test_relative_paths_follow_the_file(self):
        cfg = parse({"task": {"start_state_dir": "states"}, "paths": {"repo_root": "repo"}}, Path("/x/y/rsi.toml"))
        self.assertEqual(cfg.task.start_state_dir, Path("/x/y/states"))
        self.assertEqual(cfg.repo_root, Path("/x/y/repo"))
        self.assertEqual(cfg.machine.python, "/x/y/repo/steward/.venv/bin/python")

    def test_invalid_values(self):
        bad = [
            {"advisor": {"provider": "gemini"}},
            {"machine": {"backend": "tpu"}},
            {"machine": {"devices": []}},
            {"machine": {"devices": [0, 0]}},
            {"machine": {"devices": [0, "1"]}},
            {"machine": {"episodes_per_device": 0}},
            {"machine": {"wall_minutes": 0}},
            {"machine": {"segmentation_configs": ["a.toml"]}},
            {"task": {"name": "boat"}},
            {"task": {"seeds": []}},
            {"task": {"threshold_mm": "20"}},
            {"deepseek": {"send_images": "yes"}},
            {"advisor": {"provider": "claude"}, "claude": {"thinking": "enabled"}},
        ]
        for doc in bad:
            with self.subTest(doc=doc), self.assertRaises(ConfigError):
                parse(doc, Path("/tmp/rsi.toml"))

    def test_missing_file(self):
        with self.assertRaises(ConfigError) as ctx:
            load("/nonexistent/rsi.toml")
        self.assertIn("rsi.example.toml", str(ctx.exception))


class KeyTest(unittest.TestCase):
    def test_file_key_wins_over_environment(self):
        cfg = config({"deepseek": {"api_key": "sk-from-file"}})
        key, origin = resolve_api_key(cfg.advisor, {"DEEPSEEK_API_KEY": "sk-from-env"})
        self.assertEqual(key.reveal(), "sk-from-file")
        self.assertEqual(origin, "rsi.toml")

    def test_environment_key(self):
        cfg = config({"advisor": {"provider": "claude"}, "claude": {"api_key_env": "MY_CLAUDE"}})
        key, origin = resolve_api_key(cfg.advisor, {"MY_CLAUDE": "sk-ant-env"})
        self.assertEqual(key.reveal(), "sk-ant-env")
        self.assertEqual(origin, "env:MY_CLAUDE")

    def test_missing_key_names_variable_only(self):
        cfg = config({"advisor": {"provider": "openai"}})
        with self.assertRaises(ConfigError) as ctx:
            resolve_api_key(cfg.advisor, {"OTHER": "sk-unrelated"})
        self.assertIn("OPENAI_API_KEY", str(ctx.exception))
        self.assertNotIn("sk-unrelated", str(ctx.exception))

    def test_secret_never_prints(self):
        cfg = config({"deepseek": {"api_key": "sk-hidden-123"}})
        for text in (repr(cfg), str(cfg.advisor), repr(Secret("sk-hidden-123")), f"{cfg.advisor.api_key}"):
            self.assertNotIn("sk-hidden-123", text)


if __name__ == "__main__":
    unittest.main()
