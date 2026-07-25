# Copyright 2025 Rebellions Inc. All rights reserved.

import json
import unittest
from collections import defaultdict, deque
from types import SimpleNamespace
from unittest.mock import patch

from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import (
    NixlConnectorWorker,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus

from vllm_rbln.distributed.kv_transfer.kv_connector.v1.nixl_read_diagnostic import (
    DiagnosticContext,
    NixlReadDiagnostic,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl_connector import (
    RblnNixlConnectorWorker,
)
from vllm_rbln.v1.core.rbln_scheduler import RBLNScheduler
from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner


class FakeLogger:
    def __init__(self):
        self.lines = []

    def warning(self, message, *args):
        self.lines.append(message % args)

    @property
    def records(self):
        return [json.loads(line.split(" ", 1)[1]) for line in self.lines]


class FakeClock:
    def __init__(self):
        self.now_ns = 1_000_000

    def __call__(self):
        return self.now_ns

    def advance(self, nanoseconds):
        self.now_ns += nanoseconds


class FakeTimer:
    def __init__(self, delay, callback):
        self.delay = delay
        self.callback = callback
        self.daemon = False
        self.started = False
        self.cancelled = False

    def start(self):
        self.started = True

    def cancel(self):
        self.cancelled = True

    def fire(self):
        if not self.cancelled:
            self.callback()


class FakeTimerFactory:
    def __init__(self):
        self.timers = []

    def __call__(self, delay, callback):
        timer = FakeTimer(delay, callback)
        self.timers.append(timer)
        return timer


class FakeKVTopology:
    @staticmethod
    def block_size_ratio_from_engine_id(_engine_id):
        return 1


class FakeTransferStats:
    def __init__(self):
        self.telemetry = []

    def record_transfer(self, telemetry):
        self.telemetry.append(telemetry)

    def record_failed_transfer(self):
        pass


class FakeHandle:
    def __repr__(self):
        return "<FakeHandle at 0xDEADBEEF>"


class FakeNixlWrapper:
    def __init__(self, *, states=("PROC", "DONE"), fail_at=None):
        self.states = deque(states)
        self.fail_at = fail_at
        self.calls = []
        self.handle = FakeHandle()

    def make_prepped_xfer(self, *args, **kwargs):
        self.calls.append("make_prepped_xfer")
        if self.fail_at == "make_prepped_xfer":
            raise RuntimeError("make failed at 0xDEADBEEF")
        return self.handle

    def transfer(self, handle):
        self.calls.append("transfer")
        if self.fail_at == "transfer":
            raise RuntimeError("transfer failed at 0xDEADBEEF")
        return "PROC"

    def check_xfer_state(self, handle):
        self.calls.append("check_xfer_state")
        if self.fail_at == "check_xfer_state":
            raise RuntimeError("poll failed at 0xDEADBEEF")
        return self.states.popleft()

    def get_xfer_telemetry(self, handle):
        self.calls.append("get_xfer_telemetry")
        return {"descCount": 2, "totalBytes": 2048}

    def release_xfer_handle(self, handle):
        self.calls.append("release_xfer_handle")


class FakeRuntimeMarkerSink:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.records = []

    def enabled_for(self, _request_id):
        return self.enabled

    def emit(self, event_name, request_id, **fields):
        if not self.enabled:
            return False
        self.records.append(
            {
                "event_name": event_name,
                "request_id": request_id,
                **fields,
            }
        )
        return True


def fake_base_read_blocks(
    worker,
    local_block_ids,
    remote_block_ids,
    dst_engine_id,
    request_id,
    remote_request_id,
    remote_rank,
    local_xfer_side_handle,
    remote_xfer_side_handle,
):
    try:
        handle = worker.nixl_wrapper.make_prepped_xfer(
            "READ",
            local_xfer_side_handle,
            [1, 4],
            remote_xfer_side_handle,
            [1, 4],
            notif_msg=b"notification",
        )
        worker.nixl_wrapper.transfer(handle)
        worker._recving_transfers[request_id].append(handle)
    except Exception:
        worker._failed_recv_reqs.add(request_id)


class NixlReadDiagnosticTest(unittest.TestCase):
    def setUp(self):
        self.logger = FakeLogger()
        self.clock = FakeClock()
        self.timer_factory = FakeTimerFactory()
        self.dumps = []
        self.diagnostic = NixlReadDiagnostic(
            self.logger,
            enabled=True,
            monotonic_ns=self.clock,
            timer_factory=self.timer_factory,
            traceback_dumper=lambda **kwargs: self.dumps.append(kwargs),
        )
        self.context = DiagnosticContext(
            request_id="request-secret-123456789abc",
            remote_engine_id="engine-secret-abcdef123456",
            logical_block_count=1,
            descriptor_count=56,
            total_bytes=58_720_256,
        )

    def make_worker(self, wrapper=None, *, diagnostic=None, runtime_markers=None):
        worker = object.__new__(RblnNixlConnectorWorker)
        worker._phase4b_diagnostic = diagnostic or self.diagnostic
        worker._runtime_marker_sink = runtime_markers or FakeRuntimeMarkerSink()
        worker._phase4b_diagnostic_contexts = {}
        worker._phase4b_remote_descriptor_meta = {
            ("remote-engine", 0): tuple((1024, 0) for _ in range(6))
        }
        worker._runtime_transfer_start_ns = {}
        worker._runtime_transfer_fields = {}
        worker._runtime_transfer_poll_counts = {}
        worker.nixl_wrapper = wrapper or FakeNixlWrapper()
        worker.kv_topo = FakeKVTopology()
        worker.engine_id = "local-engine"
        worker.num_regions = 2
        worker.dst_num_blocks = {"local-engine": 3, "remote-engine": 3}
        worker.src_blocks_data = tuple((1000 + i * 1024, 1024, 0) for i in range(6))
        worker._has_mamba = False
        worker._recving_transfers = defaultdict(list)
        worker._recving_metadata = {}
        worker._failed_recv_reqs = set()
        worker._invalid_block_ids = set()
        worker._is_hma_required = False
        worker._log_failure = lambda **kwargs: None
        worker.xfer_stats = FakeTransferStats()
        return worker

    def run_read(self, worker):
        with patch.object(
            NixlConnectorWorker,
            "_read_blocks",
            fake_base_read_blocks,
        ):
            worker._read_blocks(
                [[1]],
                [[1]],
                "remote-engine",
                "diagnostic-request",
                "remote-request",
                0,
                101,
                202,
            )

    def markers(self):
        return [record["marker"] for record in self.logger.records]

    def test_common_fields_and_raw_pointer_redaction(self):
        self.diagnostic.emit(
            "make_prepped_xfer_exception",
            self.context,
            status="exception",
            exception=RuntimeError("failed at 0xDEADBEEF"),
            unsafe="native object 0x1234",
        )

        record = self.logger.records[0]
        required = {
            "diagnostic_marker",
            "marker",
            "monotonic_ns",
            "request_suffix",
            "remote_engine_suffix",
            "logical_block_count",
            "descriptor_count",
            "total_bytes",
            "thread_id",
            "pid",
            "elapsed_ns",
            "status",
            "exception_type",
            "exception_message",
        }
        self.assertTrue(required.issubset(record))
        self.assertEqual(record["request_suffix"], "123456789abc")
        self.assertNotIn("request-secret", json.dumps(record))
        self.assertNotIn("0xDEADBEEF", json.dumps(record))
        self.assertNotIn("0x1234", json.dumps(record))

    def test_request_filter_and_disabled_mode(self):
        filtered_logger = FakeLogger()
        filtered = NixlReadDiagnostic(
            filtered_logger,
            enabled=True,
            request_suffix_filter="selected",
            monotonic_ns=self.clock,
        )
        filtered.emit("transfer_enter", self.context)
        self.assertEqual(filtered_logger.lines, [])
        selected_context = DiagnosticContext(request_id="request-selected")
        filtered.emit("transfer_enter", selected_context)
        self.assertEqual(len(filtered_logger.lines), 1)

        disabled_logger = FakeLogger()
        disabled = NixlReadDiagnostic(
            disabled_logger,
            enabled=False,
            monotonic_ns=self.clock,
        )
        disabled.emit("transfer_enter", self.context)
        self.assertEqual(disabled_logger.lines, [])

    def test_make_transfer_and_handle_append_boundaries(self):
        worker = self.make_worker()
        self.run_read(worker)

        markers = self.markers()
        self.assertLess(
            markers.index("make_prepped_xfer_enter"),
            markers.index("make_prepped_xfer_return"),
        )
        self.assertLess(
            markers.index("transfer_enter"), markers.index("transfer_return")
        )
        self.assertIn("handle_append", markers)
        append = next(
            record
            for record in self.logger.records
            if record["marker"] == "handle_append"
        )
        self.assertTrue(append["handle_appended"])
        self.assertEqual(append["container_size_before"], 0)
        self.assertEqual(append["container_size_after"], 1)
        transfer_return = next(
            record
            for record in self.logger.records
            if record["marker"] == "transfer_return"
        )
        self.assertEqual(transfer_return["return_status"], "PROC")
        self.assertEqual(transfer_return["descriptor_count"], 2)
        self.assertEqual(transfer_return["total_bytes"], 2048)
        self.assertNotIn("FakeHandle at", "\n".join(self.logger.lines))

    def test_proc_proc_done_polling_rate_limit_and_completion(self):
        wrapper = FakeNixlWrapper(states=("PROC", "PROC", "DONE"))
        worker = self.make_worker(wrapper)
        self.run_read(worker)

        self.assertEqual(worker._pop_done_transfers(worker._recving_transfers), set())
        self.clock.advance(100_000_000)
        self.assertEqual(worker._pop_done_transfers(worker._recving_transfers), set())
        self.clock.advance(100_000_000)
        self.assertEqual(
            worker._pop_done_transfers(worker._recving_transfers),
            {"diagnostic-request"},
        )

        poll_records = [
            record
            for record in self.logger.records
            if record["marker"] == "status_poll"
        ]
        self.assertEqual(
            [record["status"] for record in poll_records], ["PROC", "DONE"]
        )
        self.assertEqual(poll_records[-1]["poll_sequence"], 3)
        changes = [
            record["status"]
            for record in self.logger.records
            if record["marker"] == "status_change"
        ]
        self.assertEqual(changes, ["PROC", "DONE"])
        markers = self.markers()
        self.assertLess(
            markers.index("done_observed"), markers.index("telemetry_enter")
        )
        self.assertLess(
            markers.index("telemetry_return"),
            markers.index("handle_release_enter"),
        )
        self.assertLess(
            markers.index("handle_release_return"),
            markers.index("finished_recving_add"),
        )
        self.assertNotIn("diagnostic-request", worker._recving_transfers)
        self.assertTrue(all(timer.cancelled for timer in self.timer_factory.timers))

    def test_runtime_transfer_end_is_emitted_only_after_done(self):
        runtime_markers = FakeRuntimeMarkerSink()
        wrapper = FakeNixlWrapper(states=("PROC", "DONE"))
        worker = self.make_worker(
            wrapper,
            runtime_markers=runtime_markers,
        )
        self.run_read(worker)

        self.assertEqual(
            [record["event_name"] for record in runtime_markers.records],
            ["kv_transfer_start"],
        )
        self.assertEqual(worker._pop_done_transfers(worker._recving_transfers), set())
        self.assertEqual(
            [record["event_name"] for record in runtime_markers.records],
            ["kv_transfer_start"],
        )
        self.assertEqual(
            worker._pop_done_transfers(worker._recving_transfers),
            {"diagnostic-request"},
        )
        self.assertEqual(
            [record["event_name"] for record in runtime_markers.records],
            ["kv_transfer_start", "kv_transfer_end"],
        )
        end_attributes = runtime_markers.records[-1]["attributes"]
        self.assertEqual(end_attributes["kv.logical_block_count"], 1)
        self.assertEqual(end_attributes["kv.descriptor_count"], 2)
        self.assertEqual(end_attributes["kv.transfer_bytes"], 2048)
        self.assertEqual(end_attributes["kv.poll_count"], 2)
        self.assertGreaterEqual(end_attributes["kv.duration_ns"], 0)

    def test_poll_is_logged_again_after_one_second(self):
        context = self.context
        first = self.diagnostic.begin_poll(context, handle_present=True)
        self.diagnostic.finish_poll(context, first, "PROC", handle_present=True)
        self.clock.advance(500_000_000)
        second = self.diagnostic.begin_poll(context, handle_present=True)
        self.diagnostic.finish_poll(context, second, "PROC", handle_present=True)
        self.clock.advance(600_000_000)
        third = self.diagnostic.begin_poll(context, handle_present=True)
        self.diagnostic.finish_poll(context, third, "PROC", handle_present=True)

        polls = [
            record
            for record in self.logger.records
            if record["marker"] == "status_poll"
        ]
        self.assertEqual([record["poll_sequence"] for record in polls], [1, 3])

    def test_make_prepped_xfer_exception_is_logged_without_append(self):
        worker = self.make_worker(FakeNixlWrapper(fail_at="make_prepped_xfer"))
        self.run_read(worker)

        self.assertIn("make_prepped_xfer_exception", self.markers())
        self.assertNotIn("transfer_enter", self.markers())
        append = next(
            record
            for record in self.logger.records
            if record["marker"] == "handle_append"
        )
        self.assertFalse(append["handle_appended"])
        self.assertTrue(all(timer.cancelled for timer in self.timer_factory.timers))
        self.assertNotIn("0xDEADBEEF", "\n".join(self.logger.lines))

    def test_transfer_exception_is_logged_without_append(self):
        worker = self.make_worker(FakeNixlWrapper(fail_at="transfer"))
        self.run_read(worker)

        self.assertIn("make_prepped_xfer_return", self.markers())
        self.assertIn("transfer_exception", self.markers())
        append = next(
            record
            for record in self.logger.records
            if record["marker"] == "handle_append"
        )
        self.assertFalse(append["handle_appended"])
        self.assertTrue(all(timer.cancelled for timer in self.timer_factory.timers))

    def test_first_status_poll_exception_is_logged(self):
        worker = self.make_worker(FakeNixlWrapper(fail_at="check_xfer_state"))
        self.run_read(worker)
        done = worker._pop_done_transfers(worker._recving_transfers)

        self.assertEqual(done, {"diagnostic-request"})
        self.assertIn("status_poll_enter", self.markers())
        self.assertIn("status_poll_exception", self.markers())

    def test_watchdog_uses_three_request_scoped_timers_and_cancels(self):
        self.diagnostic.start_watchdog(self.context)

        self.assertEqual(
            [timer.delay for timer in self.timer_factory.timers], [5.0, 15.0, 30.0]
        )
        self.assertTrue(all(timer.started for timer in self.timer_factory.timers))
        for timer in self.timer_factory.timers:
            timer.fire()
        self.assertEqual(len(self.dumps), 3)
        self.assertEqual(self.markers().count("watchdog_stack"), 3)

        self.diagnostic.cancel_watchdog(self.context.request_id)
        for timer in self.timer_factory.timers:
            timer.fire()
        self.assertEqual(len(self.dumps), 3)

    def test_disabled_diagnostic_preserves_base_flow(self):
        disabled_logger = FakeLogger()
        disabled = NixlReadDiagnostic(disabled_logger, enabled=False)
        wrapper = FakeNixlWrapper(states=("DONE",))
        worker = self.make_worker(wrapper, diagnostic=disabled)
        self.run_read(worker)

        self.assertEqual(
            wrapper.calls[:2],
            ["make_prepped_xfer", "transfer"],
        )
        self.assertEqual(disabled_logger.lines, [])

    def test_finished_recving_is_consumed_by_scheduler(self):
        scheduler = object.__new__(RBLNScheduler)
        scheduler._phase4b_diagnostic = self.diagnostic
        request = SimpleNamespace(
            request_id=self.context.request_id,
            status=RequestStatus.WAITING_FOR_REMOTE_KVS,
        )
        scheduler.requests = {request.request_id: request}
        scheduler.finished_recving_kv_req_ids = set()
        output = SimpleNamespace(finished_recving={request.request_id})

        def consume(base_scheduler, connector_output):
            base_scheduler.finished_recving_kv_req_ids.update(
                connector_output.finished_recving
            )

        with patch.object(Scheduler, "_update_from_kv_xfer_finished", consume):
            scheduler._update_from_kv_xfer_finished(output)

        self.assertIn(request.request_id, scheduler.finished_recving_kv_req_ids)
        self.assertEqual(
            self.markers(),
            ["finished_recving_scheduler_input", "finished_recving_scheduler_add"],
        )

    def test_scheduler_transition_is_logged(self):
        scheduler = object.__new__(RBLNScheduler)
        scheduler._phase4b_diagnostic = self.diagnostic
        scheduler._runtime_marker_sink = FakeRuntimeMarkerSink()
        request = SimpleNamespace(
            request_id=self.context.request_id,
            status=RequestStatus.WAITING_FOR_REMOTE_KVS,
        )

        def promote(_scheduler, blocked_request):
            blocked_request.status = RequestStatus.WAITING
            return True

        with patch.object(Scheduler, "_try_promote_blocked_waiting_request", promote):
            promoted = scheduler._try_promote_blocked_waiting_request(request)

        self.assertTrue(promoted)
        transition = self.logger.records[-1]
        self.assertEqual(transition["marker"], "scheduler_transition")
        self.assertEqual(
            transition["request_status_before"], "WAITING_FOR_REMOTE_KVS"
        )
        self.assertEqual(transition["request_status_after"], "WAITING")
        self.assertEqual(
            scheduler._runtime_marker_sink.records[0]["event_name"],
            "decode_loop_start",
        )

    def test_repeated_blocked_scheduler_transition_is_rate_limited(self):
        scheduler = object.__new__(RBLNScheduler)
        scheduler._phase4b_diagnostic = self.diagnostic
        scheduler._runtime_marker_sink = FakeRuntimeMarkerSink()
        request = SimpleNamespace(
            request_id=self.context.request_id,
            status=RequestStatus.WAITING_FOR_REMOTE_KVS,
        )

        with patch.object(
            Scheduler,
            "_try_promote_blocked_waiting_request",
            return_value=False,
        ):
            for _ in range(4):
                self.assertFalse(
                    scheduler._try_promote_blocked_waiting_request(request)
                )
            transitions = [
                record
                for record in self.logger.records
                if record["marker"] == "scheduler_transition"
            ]
            self.assertEqual(len(transitions), 1)

            self.clock.advance(10_000_000_000)
            self.assertFalse(
                scheduler._try_promote_blocked_waiting_request(request)
            )
            transitions = [
                record
                for record in self.logger.records
                if record["marker"] == "scheduler_transition"
            ]
            self.assertEqual(len(transitions), 2)
        self.assertEqual(scheduler._runtime_marker_sink.records, [])

    def test_model_marker_steps_skip_dummy_and_advance_after_sampling(self):
        runner = object.__new__(RBLNModelRunner)
        runner._runtime_marker_sink = FakeRuntimeMarkerSink()
        runner._runtime_marker_step_indices = {}
        runner._runtime_marker_active_requests = set()
        dummy_output = SimpleNamespace(
            kv_connector_metadata=None,
            num_scheduled_tokens={"chatcmpl-request-1": 1},
        )
        self.assertEqual(runner._runtime_marker_steps(dummy_output), [])

        real_output = SimpleNamespace(
            kv_connector_metadata=object(),
            num_scheduled_tokens={"chatcmpl-request-1": 1},
        )
        steps = runner._runtime_marker_steps(real_output)
        self.assertEqual(steps, [("chatcmpl-request-1", 0)])
        runner._emit_runtime_step_markers(
            "decode_step_start",
            steps,
            phase="decode",
            source="RBLNModelRunner.model_executable",
        )
        runner._emit_runtime_step_markers(
            "sampling_end",
            steps,
            phase="sampling",
            source="RBLNModelRunner._sample",
        )
        runner._advance_runtime_marker_steps(steps)

        self.assertEqual(
            [
                record["attributes"]["decode.step_index"]
                for record in runner._runtime_marker_sink.records
            ],
            [0, 0],
        )
        self.assertEqual(
            runner._runtime_marker_steps(real_output),
            [("chatcmpl-request-1", 1)],
        )
        continued_output = SimpleNamespace(
            kv_connector_metadata=None,
            num_scheduled_tokens={"chatcmpl-request-1": 1},
        )
        self.assertEqual(
            runner._runtime_marker_steps(continued_output),
            [("chatcmpl-request-1", 1)],
        )


if __name__ == "__main__":
    unittest.main()
