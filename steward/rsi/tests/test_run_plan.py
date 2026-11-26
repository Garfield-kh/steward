import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

from helpers import config, load

from steward_rsi import __main__ as cli
from steward_rsi import runner
from steward_rsi.runner import (Episode, Recorder, RoundExistsError, copy_harness, dry_run_text, execute, plan,
                                stop_group)

ROUND = Path("/tmp/rsi-plan-test/round-1")


class PlanTest(unittest.TestCase):
    def test_gpu_assignment_ports_and_env(self):
        cfg = config({"machine": {"devices": [2, 3, 5, 6], "episodes_per_device": 2, "base_port": 32000,
                                  "wall_minutes": 90},
                      "task": {"seeds": list(range(100, 110))},
                      "deepseek": {"api_key": "sk-must-not-appear"}})
        plans = plan(cfg, ROUND)
        self.assertEqual(len(plans), 10)
        self.assertEqual([p.slot for p in plans], [0, 1, 2, 3, 4, 5, 6, 7, 0, 1])
        self.assertEqual([p.device for p in plans], [2, 2, 3, 3, 5, 5, 6, 6, 2, 2])
        self.assertEqual(plans[3].ports, {"wca": 32030, "robot": 32031, "api": 32032})
        self.assertEqual(plans[8].ports, plans[0].ports)  # the same slot reuses its ports, one after another
        all_ports = [port for p in plans[:8] for port in p.ports.values()]
        self.assertEqual(len(all_ports), len(set(all_ports)))
        p = plans[4]
        self.assertEqual(p.seed, 104)
        self.assertEqual(p.dir, ROUND / "ep004-seed104")
        self.assertEqual(p.env_set["CUDA_VISIBLE_DEVICES"], "5")
        self.assertEqual(p.env_set["MUJOCO_EGL_DEVICE_ID"], "5")
        self.assertEqual(p.env_set["STEWARD_ASSEMBLY_PROFILE"], "toy")
        self.assertEqual(p.env_set["STEWARD_FASTEN_MODE"], "seat")
        self.assertEqual(p.env_set["STEWARD_NOSLIP"], "5")
        self.assertEqual(p.env_set["STEWARD_RECEIPT_LOG_DIR"], str(p.dir / "receipts"))
        self.assertEqual(p.env_unset, ["STEWARD_START_STATE_DIR"])
        self.assertEqual(p.up_cmd[1:3], ["-m", "steward.cli"])
        self.assertIn("--no-dispatch-task", p.up_cmd)
        self.assertIn("--no-pause-on-perception-fault", p.up_cmd)
        self.assertEqual(p.up_cmd[p.up_cmd.index("--seed") + 1], "104")
        self.assertEqual(p.up_cmd[p.up_cmd.index("--wca-port") + 1], "32040")
        self.assertEqual(p.up_cmd[p.up_cmd.index("--turn-budget-s") + 1], "5400")
        self.assertEqual(p.up_cmd[p.up_cmd.index("--robot-harness") + 1], str(p.dir / "dsh" / "robot-1"))
        self.assertEqual(p.up_cmd[p.up_cmd.index("--root") + 1], str(p.dir / "root"))
        self.assertEqual(p.build_cmd[-1], "build-toy-assembly")
        text = dry_run_text(cfg, plans)
        self.assertNotIn("sk-must-not-appear", text)
        self.assertIn("export CUDA_VISIBLE_DEVICES=5", text)
        self.assertIn("unset STEWARD_START_STATE_DIR", text)

    def test_npu_start_state_and_segmentation(self):
        cfg = config({"machine": {"backend": "npu", "devices": [0, 1], "env": {"LD_LIBRARY_PATH": "/opt/x"},
                                  "segmentation_configs": ["/cfg/seg0.toml", "/cfg/seg1.toml"]},
                      "task": {"seeds": [7, 8, 9], "start_state_dir": "/states/before-gear"},
                      "run": {"isolate_harness": False, "robot_harness_template": "/h/robot-1"}})
        plans = plan(cfg, ROUND)
        p = plans[1]
        self.assertEqual(p.device, 1)
        self.assertEqual(p.env_set["NPU_DEVICE"], "1")
        self.assertNotIn("CUDA_VISIBLE_DEVICES", p.env_set)
        self.assertEqual(p.env_set["LD_LIBRARY_PATH"], "/opt/x")
        self.assertEqual(p.env_set["STEWARD_START_STATE_DIR"], "/states/before-gear")
        self.assertEqual(p.env_unset, [])
        self.assertEqual(p.up_cmd[1:3], ["-m", "steward_npu"])
        self.assertEqual(p.build_cmd[1:3], ["-m", "steward_npu"])
        self.assertEqual(p.up_cmd[p.up_cmd.index("--segmentation-config") + 1], "/cfg/seg1.toml")
        self.assertEqual(p.up_cmd[p.up_cmd.index("--robot-harness") + 1], "/h/robot-1")
        self.assertEqual(plans[2].device, 0)


class TempDirTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp)


class SecretEnvTest(TempDirTest):
    ENV = {"DEEPSEEK_API_KEY": "sk-controller-secret", "HF_TOKEN": "hf-secret", "DB_PASSWORD": "pw-secret",
           "client_secret": "cs-secret", "LD_LIBRARY_PATH": "/opt/x"}

    def test_dry_run_and_plan_redact_secret_names(self):
        cfg = config({"machine": {"devices": [0], "env": self.ENV}, "task": {"seeds": [0]}})
        plans = plan(cfg, self.tmp / "r")
        text = dry_run_text(cfg, plans) + json.dumps(plans[0].to_dict())
        for value in ("sk-controller-secret", "hf-secret", "pw-secret", "cs-secret"):
            self.assertNotIn(value, text)
        self.assertIn("export DEEPSEEK_API_KEY='<redacted>'", text)
        self.assertIn("export LD_LIBRARY_PATH=/opt/x", text)
        self.assertEqual(plans[0].to_dict()["env_set"]["HF_TOKEN"], "<redacted>")
        # The episode processes receive the real values.
        env = Episode(cfg, plans[0], Recorder(self.tmp / "pids.jsonl"), threading.Event()).env()
        self.assertEqual(env["DEEPSEEK_API_KEY"], "sk-controller-secret")
        self.assertEqual(env["HF_TOKEN"], "hf-secret")


class CopyHarnessTest(TempDirTest):
    def test_outside_relative_links_become_absolute(self):
        template = self.tmp / "steward" / "var" / "dsh" / "robot-1"
        (template / "profiles" / "ego").mkdir(parents=True)
        (template / "profiles" / "ego" / "cordis.yml").write_text("x")
        (template / "sessions").mkdir()
        (template / "sessions" / "old.jsonl").write_text("old")
        outside = self.tmp / "steward" / "dsh" / "packages" / "bundle"
        outside.mkdir(parents=True)
        (outside / "index.js").write_text("js")
        (template / "node_modules").mkdir()
        os.symlink(os.path.relpath(outside, template / "node_modules"), template / "node_modules" / "bundle")
        os.symlink("profiles/ego", template / "ego-link")
        os.symlink(str(outside / "index.js"), template / "abs-link")
        (template / ".env").write_text("DEEPSEEK_API_KEY=x\n")

        dest = self.tmp / "runs" / "round" / "ep000-seed0" / "dsh" / "robot-1"
        copy_harness(template, dest)

        bundle = dest / "node_modules" / "bundle"
        self.assertTrue(bundle.is_symlink())
        self.assertEqual(os.readlink(bundle), str(outside))
        self.assertEqual((bundle / "index.js").read_text(), "js")
        inner = dest / "ego-link"
        self.assertEqual(os.readlink(inner), "profiles/ego")
        self.assertEqual(inner.resolve(), (dest / "profiles" / "ego").resolve())
        self.assertEqual(os.readlink(dest / "abs-link"), str(outside / "index.js"))
        self.assertFalse((dest / "sessions").exists())
        self.assertTrue((dest / ".env").is_file())


class RoundDirTest(TempDirTest):
    def test_existing_round_with_episodes_is_refused(self):
        round_dir = self.tmp / "runs" / "baseline"
        (round_dir / "ep000-seed0").mkdir(parents=True)
        (round_dir / "ep000-seed0" / "episode.json").write_text('{"status": "finished"}')
        cfg = config({"task": {"seeds": [0]}, "paths": {"output_dir": str(self.tmp / "runs")}})
        with self.assertRaises(RoundExistsError):
            execute(cfg, plan(cfg, round_dir), round_dir)
        self.assertFalse((round_dir / "plan.json").exists())

        cfg_file = self.tmp / "rsi.toml"
        cfg_file.write_text(f'[paths]\nrepo_root = "{cfg.repo_root}"\noutput_dir = "{self.tmp / "runs"}"\n'
                            f'[task]\nseeds = [0]\n')
        for command in ("run", "round"):
            err = io.StringIO()
            with redirect_stderr(err):
                code = cli.main(["--config", str(cfg_file), command, "--name", "baseline"])
            self.assertEqual(code, 1)
            self.assertIn("choose another --name", err.getvalue())

    def test_empty_round_dir_is_accepted(self):
        (self.tmp / "fresh").mkdir()
        runner.check_round_dir(self.tmp / "fresh")
        runner.check_round_dir(self.tmp / "missing")


class FakeProc:
    pid = 0
    returncode = None

    def poll(self):
        return None


class FakeWorld:
    """Answers the runner's World HTTP calls; ``turn`` lists the chat replies after the task POST."""

    def __init__(self, turn, reject=None):
        self.turn = list(turn)
        self.reject = reject
        self.posted = False
        self.urls = []

    def get_json(self, url, timeout=30.0):
        self.urls.append(url)
        if "/wca/scene" in url:
            return load("world_initial.json")
        if "/console/chat" in url:
            if not self.posted:
                return {"agents": {"robot": {"available": True, "busy": False}}, "messages": []}
            return self.turn.pop(0) if len(self.turn) > 1 else self.turn[0]
        raise AssertionError(url)

    def http(self, method, url, body=None, timeout=30.0):
        if method == "POST":
            if self.reject:
                raise urllib.error.HTTPError(url, self.reject, "busy", {}, io.BytesIO(b'{"error": "still working"}'))
            self.posted = True
            return json.dumps({"accepted": True, "id": 7}).encode()
        return b"jpeg"


class DriveTest(TempDirTest):
    def episode(self, world, **run):
        cfg = config({"task": {"seeds": [0]}, "machine": {"wall_minutes": 0.05},
                      "run": {"frame_every_min": 0, **run}})
        p = plan(cfg, self.tmp / "round")[0]
        (p.dir / "logs").mkdir(parents=True)
        episode = Episode(cfg, p, Recorder(self.tmp / "pids.jsonl"), threading.Event())
        patches = [mock.patch.object(runner, "_get_json", world.get_json), mock.patch.object(runner, "_http", world.http),
                   mock.patch.object(runner, "POLL_S", 0.01), mock.patch.object(runner, "STARTUP_POLL_S", 0.01)]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        return episode, p

    def test_turn_that_ends_before_the_first_poll(self):
        error = {"id": 8, "agent": "robot", "role": "error", "text": "robot-1 failed: no API key", "t": 1.0}
        world = FakeWorld([{"agents": {"robot": {"available": True, "busy": False}},
                            "messages": [{"id": 7, "agent": "robot", "role": "user", "text": "task"}, error]}])
        episode, p = self.episode(world)
        episode._drive(FakeProc())
        meta = json.loads((p.dir / "episode.json").read_text())
        self.assertEqual(meta["endReason"], "turn-ended")
        self.assertEqual(meta["endMessages"], [{"role": "error", "text": "robot-1 failed: no API key", "t": 1.0}])
        self.assertTrue(any(u.endswith("/console/chat?since=7") for u in world.urls))
        self.assertTrue((p.dir / "world_state.json").is_file())

    def test_busy_then_idle_with_budget_note(self):
        busy = {"agents": {"robot": {"busy": True}}, "messages": []}
        note = {"id": 9, "agent": "robot", "role": "system", "text": "turn budget reached", "t": 2.0}
        idle = {"agents": {"robot": {"busy": False}}, "messages": [note]}
        episode, p = self.episode(FakeWorld([busy, busy, idle]))
        episode._drive(FakeProc())
        meta = json.loads((p.dir / "episode.json").read_text())
        self.assertEqual(meta["endReason"], "turn-ended")
        self.assertEqual([m["role"] for m in meta["endMessages"]], ["system"])

    def test_rejected_task(self):
        episode, p = self.episode(FakeWorld([{}], reject=409))
        episode._drive(FakeProc())
        meta = json.loads((p.dir / "episode.json").read_text())
        self.assertEqual(meta["status"], "error")
        self.assertEqual(meta["endReason"], "task-rejected")
        self.assertIn("409", meta["error"])
        self.assertIn("still working", meta["error"])


class BuildStopTest(TempDirTest):
    def test_stop_interrupts_the_asset_build(self):
        cfg = config({"task": {"seeds": [0]}, "run": {"build_timeout_s": 600}})
        p = plan(cfg, self.tmp / "round")[0]
        p.build_cmd = [sys.executable, "-c", "import time; time.sleep(60)"]
        (p.dir / "logs").mkdir(parents=True)
        stop = threading.Event()
        episode = Episode(cfg, p, Recorder(self.tmp / "pids.jsonl"), stop)
        threading.Timer(0.3, stop.set).start()
        t0 = time.monotonic()
        self.assertFalse(episode._build())
        self.assertLess(time.monotonic() - t0, 15.0)
        meta = json.loads((p.dir / "episode.json").read_text())
        self.assertEqual(meta["error"], "interrupted during the asset build")
        self.assertTrue(meta["buildStop"].startswith("SIGTERM"))


class StopGroupTest(unittest.TestCase):
    def test_signals_only_its_own_group(self):
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
        time.sleep(0.2)
        self.assertEqual(stop_group(proc, grace_s=10), f"SIGTERM rc={proc.returncode}")
        self.assertTrue(stop_group(proc).startswith("already exited"))

    def test_escalates_to_sigkill(self):
        code = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('ready', flush=True); time.sleep(60)"
        proc = subprocess.Popen([sys.executable, "-c", code], start_new_session=True, stdout=subprocess.PIPE, text=True)
        proc.stdout.readline()
        self.assertEqual(stop_group(proc, grace_s=0.5), "SIGKILL")
        proc.stdout.close()


if __name__ == "__main__":
    unittest.main()
