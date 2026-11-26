"""CPU-only contract tests; no checkpoint, server or accelerator is initialized."""

import contextlib
import io
import tempfile
import unittest
from types import SimpleNamespace

import torch
from steward_npu.perception import sam3_api_srv as service
from steward_npu.perception.graph_accel import clone, signature, update
from steward_npu.renderer.kernels import TiledRenderer
import numpy as np


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.checkpoint = tempfile.NamedTemporaryFile()
        self.args = ["--ckpt", self.checkpoint.name]

    def tearDown(self):
        self.checkpoint.close()

    def test_import_does_not_initialize_runtime(self):
        self.assertFalse(hasattr(service, "ACCEL"))
        self.assertEqual(service.ALL, [])

    def test_explicit_checkpoint_devices_and_memory(self):
        args = service.parse_args(
            self.args
            + ["--device-type", "npu", "--devices", "0,2", "--memory-fraction", ".3"]
        )
        self.assertEqual(args.gpus, "0,2")
        self.assertEqual(args.memory_fraction, 0.3)

    def test_invalid_flags_fail_before_model_loading(self):
        cases = [
            [],
            self.args + ["--devices", "0,0"],
            self.args + ["--devices", "-1"],
            self.args + ["--memory-fraction", "0"],
            self.args + ["--device-type", "cuda", "--npu-graphs"],
            self.args + ["--device-type", "npu", "--npu-shared-geometry"],
            self.args + ["--device-type", "npu", "--npu-text-cache", "33"],
        ]
        for args in cases:
            with (
                self.subTest(args=args),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                service.parse_args(args)

    def test_multipart_preserves_binary_image_and_options(self):
        body = b'--b\r\nContent-Disposition: form-data; name="file"\r\n\r\n\x00\xff\r\n--b\r\nContent-Disposition: form-data; name="options"\r\n\r\n{}\r\n--b--\r\n'
        self.assertEqual(
            service.parse_multipart("multipart/form-data; boundary=b", body),
            {"file": b"\x00\xff", "options": b"{}"},
        )


class GraphInputTests(unittest.TestCase):
    def test_static_graph_buffers_keep_addresses_while_values_change(self):
        source = {
            "input": SimpleNamespace(x=torch.tensor([1.0, 2.0])),
            "names": ["a", "b"],
        }
        static = clone(source)
        pointer = static["input"].x.data_ptr()
        self.assertNotEqual(pointer, source["input"].x.data_ptr())
        source["input"].x.add_(4)
        self.assertEqual(signature(source), signature(static))
        update(static, source)
        self.assertEqual(static["input"].x.data_ptr(), pointer)
        self.assertTrue(torch.equal(static["input"].x, source["input"].x))
        self.assertNotEqual(signature(source), signature({"input": torch.zeros(3)}))


class RendererRejectionTests(unittest.TestCase):
    def test_unsupported_geometry_is_rejected_instead_of_cpu_fallback(self):
        with self.assertRaisesRegex(ValueError, "cylinders and boxes"):
            TiledRenderer({"type": np.array([7])})

    def test_transparent_geometry_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Transparent"):
            TiledRenderer(
                {"type": np.array([6]), "rgba": np.array([[1.0, 1.0, 1.0, 0.5]])}
            )


class AdapterLifecycleTests(unittest.TestCase):
    def setUp(self):
        import importlib.util
        import sys
        from pathlib import Path
        from unittest import mock
        import steward_npu

        self.mock = mock
        fake_mujoco = SimpleNamespace(Renderer=type("Renderer", (), {}))
        path = Path(steward_npu.__file__).parent / "renderer/dynamic_renderer.py"
        spec = importlib.util.spec_from_file_location("adapter_under_test", path)
        self.adapter = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {"mujoco": fake_mujoco}):
            spec.loader.exec_module(self.adapter)

    def tearDown(self):
        self.adapter.shutdown_worker()

    def renderer(self):
        renderer = object.__new__(self.adapter.DynamicNpuRenderer)
        renderer._closed = False
        renderer._buffers = None
        renderer._mode = 0
        renderer._model = None
        renderer._scene = None
        renderer.width = renderer.height = 1
        return renderer

    def test_closed_renderer_cannot_restart_worker(self):
        renderer = self.renderer()
        renderer.close()
        renderer.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            renderer.render()
        self.assertIsNone(self.adapter._worker)

    def test_one_frame_is_reused_for_rgb_depth_and_ids(self):
        import pickle

        buffers = (
            np.zeros((1, 1, 3), np.uint8),
            np.ones((1, 1), np.float32),
            np.ones((1, 1, 2), np.int32),
        )
        worker = self.mock.Mock()
        worker.poll.return_value = None
        worker.stdin = io.BytesIO()
        worker.stdout = io.BytesIO(pickle.dumps({"buffers": buffers, "timing": {}}))
        self.adapter._worker = worker
        renderer = self.renderer()
        with (
            self.mock.patch.object(self.adapter, "snapshot", return_value={}),
            self.mock.patch.dict("os.environ", {}, clear=True),
        ):
            self.assertTrue(np.array_equal(renderer.render(), buffers[0]))
            renderer.enable_depth_rendering()
            self.assertTrue(np.array_equal(renderer.render(), buffers[1]))
            renderer.enable_segmentation_rendering()
            self.assertTrue(np.array_equal(renderer.render(), buffers[2]))

    def test_worker_eof_reports_error_and_cleans_up(self):
        worker = self.mock.Mock()
        worker.poll.return_value = None
        worker.stdin = io.BytesIO()
        worker.stdout = io.BytesIO()
        self.adapter._worker = worker
        with (
            self.mock.patch.object(self.adapter, "snapshot", return_value={}),
            self.assertRaisesRegex(RuntimeError, "exited before returning"),
        ):
            self.renderer().render()
        self.assertIsNone(self.adapter._worker)
        self.assertTrue(worker.stdin.closed)
        self.assertTrue(worker.stdout.closed)


if __name__ == "__main__":
    unittest.main()
