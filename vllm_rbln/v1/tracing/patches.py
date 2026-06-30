# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runtime monkey-patching for per-request Perfetto tracing.

Tested against vLLM 0.18. Patches `EngineCore`, `AsyncLLM`, and the OpenAI
API server so that per-request timestamps (arrival, first scheduled,
prefill, decode, finish) are recorded as Chrome Trace JSON events and the
trace lifecycle is controlled via `/v1/trace/start` and `/v1/trace/stop`.

Patches are applied automatically when this module is imported. Importing
is wired into `vllm_rbln.register_ops()`, so a normal `vllm serve` invocation
exposes the trace endpoints without any extra wrapper.

When tracing is not active, the patched code paths add only a single
attribute check before falling through to the original implementation.
"""

import contextlib
import datetime
import functools
import os
import time

from vllm_rbln.logger import init_logger
from vllm_rbln.v1.tracing.perfetto_writer import PerfettoTraceWriter

logger = init_logger(__name__)

_PATCHED = False


def patch_all() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    _patch_engine_core()
    _patch_async_llm()
    _patch_api_server()
    _patch_bench_serve()

    logger.info("vllm_rbln.tracing: all patches applied")


# ---------------------------------------------------------------------------
# EngineCore
# ---------------------------------------------------------------------------


def _patch_engine_core() -> None:
    from vllm.v1.engine.core import EngineCore

    wall_to_mono = time.monotonic() - time.time()

    def _ensure_perfetto_attrs(self) -> None:
        """Idempotently initialize tracing state.

        Required because subclasses (EngineCoreProc, DPEngineCoreProc,
        DPMoEEngineCoreActor, etc.) may bypass or override EngineCore.__init__,
        leaving instances without the attributes our patched __init__ would set.
        Called from every entry point that touches tracing state.
        """
        if not hasattr(self, "_perfetto_enabled"):
            self._perfetto = None
            self._perfetto_phases = {}
            self._perfetto_arrival = {}
            self._perfetto_enabled = False

    # --- __init__: add tracing attrs ---
    _orig_init = EngineCore.__init__

    @functools.wraps(_orig_init)
    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        _ensure_perfetto_attrs(self)

    EngineCore.__init__ = _patched_init

    # --- add_request: record arrival time ---
    _orig_add_request = EngineCore.add_request

    @functools.wraps(_orig_add_request)
    def _patched_add_request(self, request, *args, **kwargs):
        _ensure_perfetto_attrs(self)
        if self._perfetto_enabled and hasattr(request, "arrival_time"):
            self._perfetto_arrival[request.request_id] = (
                request.arrival_time + wall_to_mono
            )
        return _orig_add_request(self, request, *args, **kwargs)

    EngineCore.add_request = _patched_add_request

    # --- start/stop trace ---
    def start_perfetto_trace(self, output_dir: str = "") -> str:
        _ensure_perfetto_attrs(self)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        pid = os.getpid()
        path = os.path.join(output_dir or ".", f"trace_{ts}_pid{pid}.json")
        self._perfetto = PerfettoTraceWriter(path)
        self._perfetto_phases.clear()
        self._perfetto_enabled = True
        # Force simple step() to avoid batch_queue timing issues
        if getattr(self, "batch_queue", None) is not None:
            self._perfetto_orig_step_fn = self.step_fn
            self.step_fn = self.step
        logger.info("Perfetto trace started: %s", path)
        return path

    def stop_perfetto_trace(self) -> tuple[str, int]:
        _ensure_perfetto_attrs(self)
        if self._perfetto is None:
            return ("", 0)
        self._perfetto_enabled = False
        if hasattr(self, "_perfetto_orig_step_fn"):
            self.step_fn = self._perfetto_orig_step_fn
            del self._perfetto_orig_step_fn
        t_end = time.monotonic()
        for rid in list(self._perfetto_phases):
            self._perfetto.emit_async_end("request", rid, t_end, cat="lifecycle")
        count = self._perfetto.flush()
        path = str(self._perfetto._path)
        self._perfetto = None
        self._perfetto_phases.clear()
        logger.info("Perfetto trace stopped: %s (%d events)", path, count)
        return (path, count)

    # --- _emit_perfetto_step ---
    def _emit_perfetto_step(self, sched_out, engine_core_outputs, t0, t1, t2, t3):
        _ensure_perfetto_attrs(self)
        tr = self._perfetto
        if tr is None:
            return

        if sched_out.total_num_scheduled_tokens == 0:
            return

        # New requests — always prefill
        for req_data in sched_out.scheduled_new_reqs:
            rid = req_data.req_id
            ntok = sched_out.num_scheduled_tokens.get(rid, 0)
            arrival = self._perfetto_arrival.pop(rid, None)
            req_start = arrival if arrival is not None else t0
            if rid not in self._perfetto_phases:
                tr.emit_async_begin("request", rid, req_start, cat="lifecycle")
            if arrival is not None and t0 > arrival:
                tr.emit_duration(
                    "queuing",
                    arrival,
                    t0 - arrival,
                    tid=rid,
                    cname="thread_state_sleeping",
                )
            self._perfetto_phases[rid] = "prefill"
            tr.emit_duration(
                "prefill",
                t1,
                t2 - t1,
                tid=rid,
                args={"request_id": rid, "num_tokens": ntok},
            )

        # Cached requests
        cached = sched_out.scheduled_cached_reqs
        for i, rid in enumerate(cached.req_ids):
            ntok = sched_out.num_scheduled_tokens.get(rid, 0)
            if rid not in self._perfetto_phases:
                arrival = self._perfetto_arrival.pop(rid, None)
                tr.emit_async_begin("request", rid, arrival or t0, cat="lifecycle")
                if arrival is not None and t0 > arrival:
                    tr.emit_duration(
                        "queuing",
                        arrival,
                        t0 - arrival,
                        tid=rid,
                        cname="thread_state_sleeping",
                    )
            if hasattr(cached, "num_output_tokens"):
                phase = "prefill" if cached.num_output_tokens[i] == 0 else "decode"
            else:
                phase = "prefill" if ntok > 1 else "decode"
            self._perfetto_phases[rid] = phase
            tr.emit_duration(
                phase,
                t1,
                t2 - t1,
                tid=rid,
                args={"request_id": rid, "num_tokens": ntok},
            )

        # Finished requests
        finished_rids = set(sched_out.finished_req_ids)
        if engine_core_outputs:
            for eco in engine_core_outputs.values():
                for out in eco.outputs:
                    if out.finished and out.request_id in self._perfetto_phases:
                        finished_rids.add(out.request_id)
        for rid in finished_rids:
            if rid in self._perfetto_phases:
                tr.emit_async_end("request", rid, t3, cat="lifecycle")
                del self._perfetto_phases[rid]

    EngineCore.start_perfetto_trace = start_perfetto_trace
    EngineCore.stop_perfetto_trace = stop_perfetto_trace
    EngineCore._emit_perfetto_step = _emit_perfetto_step

    # --- step(): copy original logic with timing ---
    _orig_step = EngineCore.step

    @functools.wraps(_orig_step)
    def _patched_step(self):
        if not getattr(self, "_perfetto_enabled", False):
            return _orig_step(self)

        if not self.scheduler.has_requests():
            return {}, False

        t0 = time.monotonic()
        scheduler_output = self.scheduler.schedule()
        t1 = time.monotonic()

        future = self.model_executor.execute_model(scheduler_output, non_block=True)
        grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
        with (
            self.log_error_detail(scheduler_output),
            self.log_iteration_details(scheduler_output),
        ):
            model_output = future.result()
            if model_output is None:
                model_output = self.model_executor.sample_tokens(grammar_output)
        t2 = time.monotonic()

        # Before processing the model output, process any aborts that happened
        # during the model execution.
        self._process_aborts_queue()
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )
        t3 = time.monotonic()

        self._emit_perfetto_step(scheduler_output, engine_core_outputs, t0, t1, t2, t3)

        return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0

    EngineCore.step = _patched_step

    # --- shutdown: flush pending traces ---
    _orig_shutdown = EngineCore.shutdown

    @functools.wraps(_orig_shutdown)
    def _patched_shutdown(self):
        if getattr(self, "_perfetto_enabled", False):
            try:
                self.stop_perfetto_trace()
            except Exception:
                logger.exception("Error flushing perfetto trace on shutdown")
        _orig_shutdown(self)

    EngineCore.shutdown = _patched_shutdown
    logger.info("vllm_rbln.tracing: EngineCore patched")


# ---------------------------------------------------------------------------
# AsyncLLM
# ---------------------------------------------------------------------------


def _patch_async_llm() -> None:
    from vllm.v1.engine.async_llm import AsyncLLM

    if hasattr(AsyncLLM, "start_perfetto_trace"):
        return

    async def start_perfetto_trace(self, output_dir: str = "") -> str:
        return await self.engine_core.call_utility_async(
            "start_perfetto_trace", output_dir
        )

    async def stop_perfetto_trace(self) -> tuple[str, int]:
        return await self.engine_core.call_utility_async("stop_perfetto_trace")

    AsyncLLM.start_perfetto_trace = start_perfetto_trace
    AsyncLLM.stop_perfetto_trace = stop_perfetto_trace
    logger.info("vllm_rbln.tracing: AsyncLLM patched")


# ---------------------------------------------------------------------------
# API server
# ---------------------------------------------------------------------------


def _patch_api_server() -> None:
    """Hook build_app to register /v1/trace/start and /v1/trace/stop.

    vLLM 0.18 removed the module-level `engine_client` helper and `router`
    object that earlier versions exposed. Routes are now attached via
    `register_*_api_routers(app)` calls inside `build_app`, and the engine
    client lives at `app.state.engine_client`. We wrap `build_app` so the
    trace routes are added on every freshly-built app instance.
    """
    from fastapi import Request
    from fastapi.responses import JSONResponse
    from vllm.entrypoints.openai import api_server

    if getattr(api_server, "_perfetto_trace_patched", False):
        return
    api_server._perfetto_trace_patched = True

    _orig_build_app = api_server.build_app

    @functools.wraps(_orig_build_app)
    def _patched_build_app(*args, **kwargs):
        app = _orig_build_app(*args, **kwargs)

        async def start_trace(raw_request: Request):
            client = raw_request.app.state.engine_client
            path = await client.start_perfetto_trace()
            return JSONResponse({"status": "started", "file": path})

        async def stop_trace(raw_request: Request):
            client = raw_request.app.state.engine_client
            path, count = await client.stop_perfetto_trace()
            return JSONResponse({"status": "stopped", "file": path, "events": count})

        app.add_api_route("/v1/trace/start", start_trace, methods=["POST"])
        app.add_api_route("/v1/trace/stop", stop_trace, methods=["POST"])
        return app

    api_server.build_app = _patched_build_app
    logger.info("vllm_rbln.tracing: API routes registered via build_app hook")


# ---------------------------------------------------------------------------
# vllm bench serve --trace flag
# ---------------------------------------------------------------------------


def _patch_bench_serve() -> None:
    """Add `--trace` flag to `vllm bench serve` that brackets the bench run
    with /v1/trace/start and /v1/trace/stop, plus auto-merge per-pid trace
    files. Mirrors the existing `--profile` flag pattern.

    NOTE: `vllm bench serve` is a pure HTTP client and does NOT load vllm
    plugins automatically. For this patch to take effect, vllm_rbln must be
    imported before the bench CLI parses arguments (e.g. via a small bootstrap
    that does `import vllm_rbln; vllm_rbln.register_ops()` before invoking
    `vllm bench serve`).
    """
    try:
        from vllm.benchmarks import serve as serve_mod
    except ImportError:
        return  # bench module not available; safe no-op (server-only env)

    if getattr(serve_mod, "_perfetto_trace_patched", False):
        return
    serve_mod._perfetto_trace_patched = True

    # 1. Add --trace argparse argument
    _orig_add_cli = serve_mod.add_cli_args

    @functools.wraps(_orig_add_cli)
    def _patched_add_cli(parser):
        _orig_add_cli(parser)
        parser.add_argument(
            "--trace",
            action="store_true",
            help=(
                "Enable per-request Perfetto trace via /v1/trace/start and "
                "/v1/trace/stop (requires the server to have vllm_rbln "
                "tracing patches loaded). Per-pid trace files are "
                "auto-merged into trace_<ts>_merged.json."
            ),
        )

    serve_mod.add_cli_args = _patched_add_cli

    # 2. Wrap main_async to bracket the bench with trace HTTP calls + merge
    import glob as _glob
    import json as _json
    import re as _re
    import urllib.error
    import urllib.request

    _orig_main_async = serve_mod.main_async

    @functools.wraps(_orig_main_async)
    async def _patched_main_async(args):
        if not getattr(args, "trace", False):
            return await _orig_main_async(args)

        # Mirror main_async's own base_url derivation (serve.py:1640-1646)
        if getattr(args, "base_url", None):
            base_url = args.base_url
        else:
            base_url = f"http://{args.host}:{args.port}"

        ts = None
        server_cwd = None

        # --- trace start ---
        try:
            req = urllib.request.Request(f"{base_url}/v1/trace/start", method="POST")
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = _json.loads(resp.read().decode())
                logger.info("perfetto trace started: %s", data)
                fpath = data.get("file", "")
                if fpath:
                    m = _re.search(r"\d{8}_\d{6}", os.path.basename(fpath))
                    if m:
                        ts = m.group(0)
                    server_cwd = os.path.dirname(fpath)
        except (urllib.error.URLError, OSError, ValueError) as e:
            logger.warning(
                "perfetto trace start failed (%s). Is the server running "
                "with vllm_rbln tracing patches?",
                e,
            )

        # --- run benchmark ---
        try:
            result = await _orig_main_async(args)
        finally:
            # --- trace stop ---
            try:
                req = urllib.request.Request(f"{base_url}/v1/trace/stop", method="POST")
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = _json.loads(resp.read().decode())
                    logger.info("perfetto trace stopped: %s", data)
            except (urllib.error.URLError, OSError, ValueError) as e:
                logger.warning("perfetto trace stop failed: %s", e)

            # --- merge per-pid trace files (same-filesystem assumption) ---
            if ts and server_cwd:
                pattern = os.path.join(server_cwd, f"trace_{ts}_pid*.json")
                files = sorted(_glob.glob(pattern))
                if files:
                    all_events: list[dict] = []
                    for f in files:
                        try:
                            with open(f) as fh:
                                ev = _json.load(fh).get("traceEvents", [])
                            all_events.extend(ev)
                        except (OSError, ValueError) as e:
                            logger.warning("skipping %s during merge: %s", f, e)
                    all_events.sort(key=lambda e: e.get("ts", 0))
                    merged_path = os.path.join(server_cwd, f"trace_{ts}_merged.json")
                    try:
                        with open(merged_path, "w") as fh:
                            _json.dump({"traceEvents": all_events}, fh, indent=0)
                        logger.info(
                            "perfetto traces merged: %s (%d events from %d files)",
                            merged_path,
                            len(all_events),
                            len(files),
                        )
                        for f in files:
                            with contextlib.suppress(OSError):
                                os.unlink(f)
                    except OSError as e:
                        logger.warning("merge write failed: %s", e)

                    # --- analyze merged trace (TTFT + decode breakdown) ---
                    try:
                        from vllm_rbln.v1.tracing.analyze import (
                            analyze_merged_trace,
                        )

                        analyze_merged_trace(merged_path)
                    except Exception as e:  # noqa: BLE001
                        logger.warning("trace analysis failed: %s", e)

        return result

    serve_mod.main_async = _patched_main_async
    logger.info("vllm_rbln.tracing: vllm bench serve --trace flag registered")


# Apply patches at import time (idempotent via _PATCHED guard).
patch_all()
