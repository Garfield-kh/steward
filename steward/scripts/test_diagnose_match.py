"""Diagnostic HTTP tests; run with python3 -m unittest discover -s steward/scripts -p test_diagnose_match.py."""

import json
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from diagnose_match import discover, trial


class DiagnosticTests(unittest.TestCase):
    """Exercise client disconnects against a server that completes only after observing EOF."""

    def test_discovery_rejects_ambiguous_lids_and_accepts_override(self):
        objects = [{"id": "piece", "category": "puzzle_piece", "existence": 1},
                   {"id": "lid1", "category": "lid", "existence": 1},
                   {"id": "lid2", "category": "lid", "existence": 1}]
        with self.assertRaisesRegex(ValueError, "2 个参考图"):
            discover({"objects": objects}, None, None)
        self.assertEqual(discover({"objects": objects}, "lid2", None),
                         {"entities": ["piece"], "reference": "lid2"})
        with self.assertRaises(ValueError):
            discover({"objects": objects}, "lid2", ["missing"])

    def exercise_http(self, mode, early_error=False, hold_after_disconnect=False):
        records = []
        lock = threading.Lock()
        disconnected = threading.Event()
        pending = threading.Condition()
        pending_summaries = []
        release = threading.Event()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass  # Test HTTP traffic has no operator-facing logging.

            def reply(self, status, body):
                raw = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                if self.path == "/ego/inputs":
                    self.reply(200, {"inputs": list(records)})
                else:
                    with pending:
                        pending_summaries.append(True)
                        pending.notify_all()
                    with lock:
                        self.reply(200, {"objects": []})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if early_error:
                    self.reply(409, {"error": "face not observed"})
                    return
                with lock:
                    self.connection.settimeout(5)
                    if self.rfile.read(1) == b"":
                        disconnected.set()
                    if hold_after_disconnect:
                        release.wait(5)
                    records.append({"path": self.path, "body": body})

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = False
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(trial, f"http://127.0.0.1:{server.server_port}",
                                         {"entities": ["piece"], "reference": "lid", "px": 24},
                                         deadline=0.1, recovery=5, baseline=0.001, threshold=0.02, mode=mode)
                try:
                    if hold_after_disconnect:
                        self.assertTrue(disconnected.wait(5))
                        with pending:
                            self.assertTrue(pending.wait_for(lambda: len(pending_summaries) == 2, timeout=5))
                        # Both probes are queued before the measured blocking interval begins.
                        time.sleep(0.06)
                finally:
                    release.set()
                result = future.result(timeout=10)
            self.assertTrue(result["recovered"])
            self.assertIn("unknown", result["cpu_still_computing"])
            if early_error:
                self.assertEqual(result["match"]["http"], 409)
                self.assertFalse(result["valid_matches_received"])
                self.assertFalse(result["handler_continued_after_disconnect_evidence"])
            else:
                self.assertTrue(disconnected.is_set())
                self.assertEqual(result["input_records_after"]["count"], 1)
                self.assertEqual(result["match"]["outcome"],
                                 "client_timeout" if mode == "timeout" else "client_cancelled")
                if hold_after_disconnect:
                    self.assertTrue(result["state_blocking_observed"])
                    self.assertTrue(result["handler_continued_after_disconnect_evidence"])
        finally:
            server.shutdown()
            thread.join()
            server.server_close()

    def test_timeout_disconnects_the_server_socket(self):
        self.exercise_http("timeout")

    def test_cancel_disconnects_the_server_socket(self):
        self.exercise_http("cancel")

    def test_missing_observation_is_not_reported_as_cancellation(self):
        self.exercise_http("cancel", early_error=True)

    def test_delayed_handler_completion_after_cancel_is_reported(self):
        self.exercise_http("cancel", hold_after_disconnect=True)


if __name__ == "__main__":
    unittest.main()
