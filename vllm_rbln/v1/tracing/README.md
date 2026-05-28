# Perfetto Tracing for vLLM-RBLN

Per-request server-side tracing built on top of vLLM. Emits Chrome Trace
JSON events with `arrival`, `queuing`, `prefill`, and `decode` phases per
request, and produces an automatic TTFT + decode breakdown report at the
end of a bench run.

## What you get

- **`trace_<ts>_merged.json`** — Perfetto-viewable trace
  (drop into <https://ui.perfetto.dev/>).
- **`bench.json`** — standard `vllm bench serve` output (unchanged).
- **stdout report** — TTFT + queue + service + decode percentile breakdown
  with one-line interpretation.

## Setup

```bash
cd vllm-rbln
pip install -e .
```

This registers the `vllm-rbln` console script, which auto-loads the tracing
patches before invoking the vLLM CLI (needed because `vllm bench serve`
does not load plugins on its own).

## Usage

### 1. Start the server

```bash
vllm-rbln serve <model> --port 8000 ...
```

Same arguments as `vllm serve`. The tracing patches register
`/v1/trace/start` and `/v1/trace/stop` HTTP endpoints automatically.
Verify in the server log:

```
[vllm-rbln] INFO ... vllm_rbln.tracing: EngineCore patched
[vllm-rbln] INFO ... vllm_rbln.tracing: AsyncLLM patched
[vllm-rbln] INFO ... vllm_rbln.tracing: API routes registered via build_app hook
[vllm-rbln] INFO ... vllm_rbln.tracing: vllm bench serve --trace flag registered
[vllm-rbln] INFO ... vllm_rbln.tracing: all patches applied
```

### 2. Run a bench with tracing

```bash
vllm-rbln bench serve --trace \
    --backend vllm --model <M> \
    --max-concurrency 32 --request-rate 4 \
    --dataset-name random --random-input-len 1024 --random-output-len 256 \
    --num-prompts 640 \
    --save-detailed --result-filename bench.json
```

`--trace` is the only new flag. It brackets the bench with HTTP calls to
`/v1/trace/start` and `/v1/trace/stop`, then merges per-PID trace files
into `trace_<ts>_merged.json`, and prints a TTFT + decode analysis to
stdout.

### 3. (Optional) Re-analyze an existing merged trace

```bash
python tools/analyze_trace.py /path/to/trace_<ts>_merged.json
```

## Report structure

```
Distribution (independent percentiles):
metric          mean   p1   p5  p10  p50  p90  p95  p99  max
ttft (ms)        ...
queue (ms)       ...
service (ms)     ...
dec_total (ms)   ...
dec_avg (ms)     ...
dec_step (ms)    ...

Per-TTFT-percentile breakdown (request at each TTFT percentile):
  pct    ttft  queue  service  queue%  service%  dec_total  dec_steps  avg_step
  p1     ...
  ...

Decode-step duration distribution (across all decode events):
  step (ms)  ...
  hint: median ≈ baseline pure decode; p99/max suggest prefill-induced stalls

Cumulative (N requests):
  total TTFT    = ...
  total queue   = ... (XX% of TTFT)
  total service = ... (XX% of TTFT)
  total decode  = ...

Interpretation: queue/TTFT mean = X% → <verdict>
```

`<verdict>` is one of `COMPUTE-BOUNDED`, `QUEUE-BOUNDED`,
or `QUEUE-DOMINATED`.

## Background: dispatch bubble at `max_concurrency = DP_size`

A counter-intuitive observation when `max_concurrency` is set equal to
data-parallel size (so at most one in-flight request per replica):
residual queue time of roughly **one chunk latency** persists, regardless
of total prefill length.

### Observation

| Setup | service | queue | queue / chunk_latency |
| --- | --- | --- | --- |
| ISL=1024 (2 chunks of ~200 ms) | 400 ms | ~190 ms | ≈ 1 |
| ISL=512  (1 chunk of ~200 ms)  | 200 ms | ~180 ms | ≈ 1 |

Queue tracks **one chunk latency**, independent of total service time.
Reducing prefill length does not reduce queue — only reducing chunk size
does.

### Mechanism — dispatch / availability sync mismatch

```
time  →   t_N            t_N+1          t_N+2
          │              │              │
          ▼              ▼              ▼
replica:  ┌──iter N────┐┌──iter N+1───┐┌──iter N+2───┐
          │ prefill A  ││ (empty step ││ prefill B   │
          │ (last)     ││  = bubble)  ││             │
          └────────────┘└─────────────┘└─────────────┘
                         ▲     ▲
                  A done │     │ B arrives at LB
                         (stats publish lag + dispatch latency)
                         → iter N+1's schedule() already passed
                         → B picked up only at iter N+2
                         → queue(B) ≈ 1 chunk latency
```

When prefill A completes, the replica becomes idle but the LB dispatcher
does not observe this synchronously — `DPCoordinator` publishes engine
counts on a fixed cycle (default 100 ms, see
`min_stats_update_interval_ms` in `vllm/v1/engine/coordinator.py`). The
next scheduler iteration on that replica therefore proceeds **empty
(a bubble)** because no new request was visible to `schedule()`. When a
new request finally arrives during the bubble iteration, it cannot be
picked up until that iteration completes — adding one chunk latency to
its TTFT.

### Implications

- The residual queue is **not an LB algorithm bug**; it is the
  unavoidable cost of asynchronous dispatch given iteration-atomic
  prefill processing.
- Queue scales with **chunk size**, not total prefill length. Reducing
  `--max-num-batched-tokens` (the per-step chunk budget) shrinks the
  residual queue proportionally.
- `max_concurrency = 1` (full serialization) drives queue to zero by
  ensuring arrivals are always timed to just-completed iterations, at
  the cost of `1/DP` throughput.
- A truly synchronous dispatch path (replica → LB completion callback,
  rather than periodic stats publish) would eliminate the bubble without
  giving up throughput.

## Fallback wrapper (no `vllm-rbln` install required)

If you want to keep the server unchanged or skip the `vllm-rbln` CLI shim,
the same bench-side orchestration is available as a standalone script:

```bash
python tools/run_bench_serve_with_trace.py <bench serve args>
```

This wrapper does the same trace start/stop + merge + analysis externally,
calling vanilla `vllm bench serve` as a subprocess. The server must still
have the tracing patches loaded (i.e. started via `vllm-rbln serve` or
with `vllm_rbln.register_ops()` invoked before `vllm.entrypoints.cli.main`).

Wrapper-only flags (consumed locally, not forwarded):

| Flag | Default | Purpose |
| --- | --- | --- |
| `--server-url URL` | `http://localhost:8000` | bench client target |
| `--server-cwd DIR` | auto from `/v1/trace/start` response | where per-PID traces are written |
| `--trace-output FILE` | `trace_<ts>_merged.json` | merged trace filename |
| `--no-trace` | off | skip trace start/stop/merge |
| `--no-sanity` | off | skip `/openapi.json` endpoint check |
| `--keep-pid-traces` | off | keep per-PID trace files after merge |
| `--wrapper-help` | — | show this help block |

## Files

```
vllm_rbln/v1/tracing/
  __init__.py            module docs
  patches.py             EngineCore / AsyncLLM / api_server / bench_serve monkey-patches
  perfetto_writer.py     Chrome Trace JSON writer (called from EngineCore.step)
  analyze.py             merged trace → TTFT + decode percentile report
  README.md              this file

vllm_rbln/
  cli.py                 vllm-rbln console-script shim (force plugin load)

tools/
  merge_traces.py        per-PID trace JSON merger CLI
  run_bench_serve_with_trace.py   bench wrapper (alternative to --trace)
  analyze_trace.py       standalone analysis CLI for existing merged traces
```

## What the patches do

`patch_all()` (called from `register_ops()`) installs:

- `EngineCore.add_request` — records `arrival_time` per request.
- `EngineCore.step` — measures `t0..t3` per scheduler iteration and emits
  per-request `queuing`, `prefill`, `decode` Chrome Trace events.
- `EngineCore.start_perfetto_trace` / `stop_perfetto_trace` — control
  methods invoked through the OpenAI API server.
- `AsyncLLM.start_perfetto_trace` / `stop_perfetto_trace` — async wrappers
  that route via `call_utility_async` to all EngineCore processes.
- `api_server.build_app` — adds `/v1/trace/start` and `/v1/trace/stop`
  routes to the FastAPI app at startup.
- `vllm.benchmarks.serve.add_cli_args` — adds the `--trace` flag to
  `vllm bench serve`.
- `vllm.benchmarks.serve.main_async` — brackets the bench run with trace
  HTTP calls and auto-merges per-PID trace files.

All patches are idempotent (guarded by module-level flags) and add only a
single attribute check on hot paths when tracing is not active.
