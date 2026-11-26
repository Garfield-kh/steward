import json
import shutil
import tempfile
import unittest
from pathlib import Path

from helpers import FIXTURE, GOAL, SPEC, config

from steward_rsi import providers
from steward_rsi.config import Secret
from steward_rsi.review import QUESTIONS, build_prompt, choose_episodes, receipt_digest, review
from steward_rsi.score import score_runs


def write_receipts(folder: Path):
    folder.mkdir(parents=True)
    rows = [
        {"event": "capability_configuration"},
        {"jobId": "job-1", "skill": "look_around", "receipt": {"state": "failed", "message": "timed out"}},
        {"jobId": "job-2", "skill": "navigate", "receipt": {"state": "succeeded", "message": "ok"}},
    ]
    (folder / "receipts-1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    events = [
        {"event": "start", "job": "job-4", "decision": "running", "simTime": 27.6},
        {"event": "heartbeat", "job": "job-4", "decision": "running", "simTime": 28.0},
        {"event": "recovery", "job": "job-4", "decision": "recovering", "reason": "changed_evidence_or_pending_holes"},
        {"event": "exit", "job": "job-4", "decision": "executed_unverified", "reason": None, "simTime": 243.09,
         "holeOutcomes": [{"hole": "wing_0", "state": "observed_not_seated", "released": True, "observedAt": 79.0}]},
    ]
    (folder / "assembly-sessions-1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in events) + "{truncated")


class FakeProvider(providers.DeepSeekProvider):
    def __init__(self, cfg):
        super().__init__(cfg, Secret("sk-x"), transport=self._fake_http, sleep=lambda s: None)
        self.sent = None

    def _fake_http(self, url, headers, body, timeout):
        self.sent = json.loads(body)
        return 200, json.dumps({"choices": [{"message": {"content": "## 1. Worst stage\nseated"},
                                             "finish_reason": "stop"}], "model": "fake"}).encode()


class ReviewTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.run = self.tmp / "ep000-seed800628"
        shutil.copytree(FIXTURE, self.run)
        write_receipts(self.run / "receipts")
        (self.run / "frames").mkdir()
        (self.run / "frames" / "final-kit.jpg").write_bytes(b"\xff\xd8fakejpeg")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_receipt_digest(self):
        d = receipt_digest(self.run / "receipts")
        self.assertEqual(d["skillOutcomes"], {"look_around:failed": 1, "navigate:succeeded": 1})
        self.assertEqual(d["recentFailures"][0]["message"], "timed out")
        self.assertEqual(d["assemblyEvents"], {"start": 1, "recovery": 1, "exit": 1})
        self.assertEqual(d["recoveryReasons"], {"changed_evidence_or_pending_holes": 1})
        self.assertEqual(d["lastExits"][0]["decision"], "executed_unverified")
        self.assertEqual(d["lastHoleOutcomes"], [{"hole": "wing_0", "state": "observed_not_seated", "released": True}])
        self.assertEqual(receipt_digest(self.tmp / "none"), {"available": False})

    def test_prompt_contains_question_sheet(self):
        results, summary = score_runs([self.tmp], GOAL, SPEC, "airplane", 20.0, 6.0)
        prompt = build_prompt(summary, results, {"ep000-seed800628": {"available": False}}, "TASK TEXT",
                              "FUNNEL TABLE", "Chinese", ["ep0 final-kit"])
        for needle in ("TASK TEXT", "FUNNEL TABLE", "Worst stage by conversion:", "in Chinese",
                       "1. Worst stage", "2. Most likely cause", "3. One change", "4. Verify from an intermediate state",
                       "save-start", "ep0 final-kit", '"maxLateralMm": 16.555'):
            self.assertIn(needle, prompt)
        self.assertEqual(QUESTIONS.count("\n1. "), 1)

    def test_choose_episodes_prefers_least_progress(self):
        a = {"run": "a", "valid": True, "complete": False, "counts": {"seated": 3, "aligned": 4, "placed": 4}}
        b = {"run": "b", "valid": True, "complete": False, "counts": {"seated": 0, "aligned": 1, "placed": 2}}
        c = {"run": "c", "valid": True, "complete": True, "counts": {"seated": 4, "aligned": 4, "placed": 4}}
        self.assertEqual([r["run"] for r in choose_episodes([a, b, c], limit=2)], ["b", "c"])

    def test_review_round_trip(self):
        score_dir = self.tmp / "score"
        score_runs([self.tmp], GOAL, SPEC, "airplane", 20.0, 6.0, out_dir=score_dir)
        cfg = config({"deepseek": {"send_images": True, "max_images": 2}})
        provider = FakeProvider(cfg.advisor)
        advice = review(score_dir / "score.json", {self.run.name: self.run}, provider, "TASK", "English", score_dir)
        self.assertIn("Worst stage", advice.read_text())
        self.assertIn("model=fake", advice.read_text())
        prompt = (score_dir / "review_prompt.md").read_text()
        self.assertIn("executed_unverified", prompt)
        parts = provider.sent["messages"][1]["content"]
        self.assertEqual(sum(1 for p in parts if p["type"] == "image_url"), 1)


if __name__ == "__main__":
    unittest.main()
