# Copyright 2025 Rebellions Inc. All rights reserved.

import asyncio
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from vllm_rbln.runtime_markers import (
    RuntimeMarkerSink,
    correlation_id_from_request_id,
)


class FakeClock:
    def __init__(self):
        self.value = 1_000

    def __call__(self):
        self.value += 10
        return self.value


class RuntimeMarkerSinkTest(unittest.TestCase):
    def test_completion_internal_request_suffix_is_removed_for_correlation(self):
        self.assertEqual(
            correlation_id_from_request_id(
                "cmpl-d1cef74a-9f6a-4b7f-8ecd-04477d7fb032-0-a3cc9d69"
            ),
            "d1cef74a-9f6a-4b7f-8ecd-04477d7fb032",
        )
        self.assertEqual(
            correlation_id_from_request_id("cmpl-customer-0-nothex"),
            "customer-0-nothex",
        )
        self.assertEqual(
            correlation_id_from_request_id("chatcmpl-request-1"),
            "request-1",
        )

    def test_disabled_sink_creates_nothing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            sink = RuntimeMarkerSink(output_dir="", register_atexit=False)
            self.assertFalse(
                sink.emit(
                    "request_received",
                    "request-1",
                    phase="request",
                    source="test.source",
                    process_role="test",
                )
            )
            sink.finalize()
            self.assertEqual(list(Path(temp_dir).iterdir()), [])

    def test_common_contract_redaction_duplicates_and_stats(self):
        clock = FakeClock()
        with tempfile.TemporaryDirectory() as temp_dir:
            sink = RuntimeMarkerSink(
                temp_dir,
                host_id="host-a",
                clock_domain_id="clock-a",
                monotonic_ns=clock,
                register_atexit=False,
            )
            emitted = sink.emit(
                "decode_step_start",
                "chatcmpl-request-1",
                phase="decode",
                source="RBLNModelRunner.model_executable",
                process_role="npu_model_runner",
                correlation_id=correlation_id_from_request_id(
                    "chatcmpl-request-1"
                ),
                attributes={
                    "decode.step_index": 0,
                    "runtime.note": "safe but 0xDEADBEEF is not",
                    "runtime.pointer": "0x1234",
                    "invalid": "not namespaced",
                },
            )
            self.assertTrue(emitted)
            self.assertFalse(
                sink.emit(
                    "decode_step_start",
                    "chatcmpl-request-1",
                    phase="decode",
                    source="RBLNModelRunner.model_executable",
                    process_role="npu_model_runner",
                    attributes={"decode.step_index": 0},
                )
            )
            self.assertTrue(
                sink.emit(
                    "decode_step_start",
                    "chatcmpl-request-1",
                    phase="decode",
                    source="RBLNModelRunner.model_executable",
                    process_role="npu_model_runner",
                    attributes={"decode.step_index": 1},
                )
            )

            marker_path = sink.path
            self.assertIsNotNone(marker_path)
            records = [
                json.loads(line)
                for line in marker_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(records), 2)
            required = {
                "schema_version",
                "event_name",
                "timestamp_ns",
                "host_id",
                "clock_domain_id",
                "process_role",
                "pid",
                "thread_id",
                "request_id",
                "phase",
                "source",
                "attributes",
            }
            self.assertTrue(required.issubset(records[0]))
            self.assertEqual(records[0]["correlation_id"], "request-1")
            self.assertEqual(records[0]["attributes"]["decode.step_index"], 0)
            serialized = json.dumps(records)
            self.assertNotIn("0xDEADBEEF", serialized)
            self.assertNotIn("0x1234", serialized)
            self.assertNotIn("runtime.pointer", serialized)
            self.assertNotIn('"invalid"', serialized)

            sink.finalize()
            stats_path = Path(temp_dir) / (
                f"runtime-markers-{records[0]['pid']}.stats.json"
            )
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            self.assertEqual(stats["records"], 2)
            self.assertGreater(stats["bytes"], 0)
            self.assertGreaterEqual(stats["average_write_ns"], 0)
            self.assertGreaterEqual(stats["max_write_ns"], 0)
            self.assertEqual(stats["dropped"], 0)
            self.assertEqual(stats["duplicates"], 1)
            stats_bytes = stats_path.read_bytes()
            sink.finalize()
            self.assertEqual(stats_path.read_bytes(), stats_bytes)

    def test_write_failure_is_isolated(self):
        sink = RuntimeMarkerSink(
            "/dev/null/runtime-markers",
            register_atexit=False,
        )
        self.assertFalse(
            sink.emit(
                "request_received",
                "request-1",
                phase="request",
                source="test.source",
                process_role="test",
            )
        )
        self.assertEqual(sink.stats()["dropped"], 1)

    def test_request_suffix_filter(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            sink = RuntimeMarkerSink(
                temp_dir,
                request_suffix_filter="selected",
                register_atexit=False,
            )
            self.assertFalse(
                sink.emit(
                    "request_received",
                    "request-other",
                    phase="request",
                    source="test.source",
                    process_role="test",
                )
            )
            self.assertTrue(
                sink.emit(
                    "request_received",
                    "request-selected",
                    phase="request",
                    source="test.source",
                    process_role="test",
                )
            )

    def test_proxy_lifespan_finalizes_stats_when_client_close_fails(self):
        proxy_path = (
            Path(__file__).parent
            / "torch_compile/e2e/v1/kv_connector/nixl_integration"
            / "toy_proxy_server.py"
        )
        spec = importlib.util.spec_from_file_location(
            "phase4b_runtime_marker_proxy",
            proxy_path,
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        proxy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(proxy)

        class FailingClient:
            async def aclose(self):
                raise RuntimeError("close failed")

        class MarkerSink:
            def __init__(self):
                self.finalize_calls = 0

            def finalize(self):
                self.finalize_calls += 1

        marker_sink = MarkerSink()
        proxy.runtime_markers = marker_sink
        proxy.global_args = SimpleNamespace(
            prefiller_instances=[],
            decoder_instances=[],
        )
        proxy_app = SimpleNamespace(state=SimpleNamespace())

        async def exercise_lifespan():
            context = proxy.lifespan(proxy_app)
            await context.__aenter__()
            proxy_app.state.prefill_clients.append({"client": FailingClient()})
            await context.__aexit__(None, None, None)

        with self.assertRaisesRegex(RuntimeError, "close failed"):
            asyncio.run(exercise_lifespan())
        self.assertEqual(marker_sink.finalize_calls, 1)


if __name__ == "__main__":
    unittest.main()
