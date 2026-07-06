# Async-forward overlap (gpt-oss EP+DP4 decode)

## Goal

In gpt-oss-120b **EP + DP4** decode, hide the per-step DP `num_tokens` `all_reduce`
(host gloo) behind the NPU `forward`, to cut ~10% off step latency — while keeping
**parity** and **async scheduling** on.

## Mechanism

`RBLN_DYNAMO_ASYNC=defer` routes the compiled forward / `compute_logits` through the
async runtime (non-blocking dispatch). `_prepare_inputs` computes the DP all_reduce
first (pure host, so it overlaps the *previous* step's still-running forward), then
drains that forward before issuing any device work.

## Current state

- **Parity: DONE.** `defer` produces token-identical output to sync (16/16).
- **Overlap: NOT yet active.** To keep parity, a drain is currently done *before* the
  sampler, which serializes the step — so `defer` currently runs at ~sync throughput.

### Why the overlap isn't on yet

The greedy sampler is `torch.ops.rbln.argmax`, a **host op** (no device primitive), so
it eagerly reads `logits` on the main thread inside `sample_tokens`. For the overlap,
the forward must stay in-flight past the next all_reduce; but then the eager sampler
reads *uncomputed* logits → garbage. The safe-but-non-overlapping fix in place drains
the forward before the sampler.

### Remaining work for the ~10%

Defer the eager sampler (`_sample` + `_bookkeeping_sync`) from `sample_tokens(N)` to the
**next step's `_prepare_inputs` drain** (i.e. after the forward completes, and after the
all_reduce it overlapped). This keeps the forward overlapping the all_reduce *and* lets
the sampler read correct logits. It requires snapshotting the `input_batch` state that
`_bookkeeping_sync` reads (step N+1's `_update_states` overwrites it first). Profiling
shows the overlap is worth ~8-11% on the step-period median/steady-state; this
deferred-sampler is the enabler.

## Committed work

- **rebel_compiler** — branch `async-overlap`
  - async runtime: process-wide shared FIFO worker, `RunIO` device/vmem resolve path,
    idempotent `AsyncResultMap` (a run-id may be awaited by more than one consumer).
- **vllm-rbln** — branch `async-overlap-prototype`
  - overlap prototype: `_prepare_inputs` hoists the all_reduce above the device H2D ops
    and drains the prior forward at the device boundary; deferred output copy; warm-cache
    shim disabled under async.
  - parity fix: drain forward before the eager sampler.

## How to run the gpt-oss test

Prereqs: build rebel_compiler (`rebel-compiler-build` skill) and
`uv pip install -e ~/codebase/rebel_compiler/python` in the vllm-executor venv.

```bash
cd ~/codebase/vllm-executor && source .venv/bin/activate

# Shared env. Use RBLN_DEVICES=4,5,6,7 if 0-3 are taken by another user (check rbln-smi).
export VLLM_RBLN_USE_DEVICE_TENSOR=1 TORCH_RBLN_DISABLE_FALLBACK=compile_error \
  VLLM_RBLN_AUTO_PORT=1 RBLN_WEIGHT_FREE=1 VLLM_RBLN_BATCH_ATTN_OPT=1 \
  VLLM_RBLN_SORT_BATCH=1 VLLM_RBLN_MOE_REDUCE_SCATTER=1 \
  SPDLOG_LEVEL=warning RBLN_VERBOSE=warning VLLM_LOGGING_LEVEL=INFO RBLN_DEVICES=0,1,2,3

# Run. Prepend RBLN_DYNAMO_ASYNC=defer for the async-overlap path; omit it for sync.
RBLN_DYNAMO_ASYNC=defer VLLM_RBLN_OPTIMISTIC_SCHED=1 \
python3 -m vllm_rbln_exec.parity_runner --task r --model gpt-oss-120b --ep --dp 4 --rsd 1 \
  --max-model-len 131072 --block-size 1024 --max-num-batched-tokens 512 --batch 1 \
  --num-hidden-layers 18 --max-num-blocks 129 --max-tokens 16 --num-prompts 4
```

Flags: `RBLN_DYNAMO_ASYNC=defer` = async overlap path (omit = sync). `=1` = async
runtime but await-immediately (functionally sync; useful to isolate the async I/O path).
`VLLM_RBLN_OPTIMISTIC_SCHED=1` turns on async scheduling (batch-queue depth 2).

### Parity check (sync vs defer)

`--task r` compares RBLN vs a CPU/GPU golden if cached; with none, use rbln-vs-rbln:
sampling is greedy (temperature 0), so **identical token_ids ⇒ parity**.

```bash
CJ=~/.cache/vllm-rbln-exec/rbln_results_openai_gpt-oss-120b_L18_T16_LP16_S131072_KV1024_EP_TP1_RSD1_DP4_PP1_P16_*.json
# 1) sync, cache result
rm -f $CJ; <env> VLLM_RBLN_OPTIMISTIC_SCHED=1 python3 -m vllm_rbln_exec.parity_runner ... --cache-results
cp $CJ /tmp/sync.json
# 2) defer, cache result (delete first so it re-runs instead of loading the cache)
rm -f $CJ; <env> RBLN_DYNAMO_ASYNC=defer VLLM_RBLN_OPTIMISTIC_SCHED=1 python3 -m vllm_rbln_exec.parity_runner ... --cache-results
cp $CJ /tmp/defer.json
# 3) diff token_ids per prompt (identical ⇒ parity)
python3 -c "import json;a=json.load(open('/tmp/sync.json'));b=json.load(open('/tmp/defer.json'));print('PARITY' if all(x.get('token_ids')==y.get('token_ids') for x,y in zip(a,b)) else 'MISMATCH')"
```

### Profiling (see the overlap)

Add `--profile`; torch traces land in `./profile/rbln_openai_gpt-oss-120b_dp{0-3}/`
(`*.pt.trace.json.gz`, open in ui.perfetto.dev). Useful signals: `get_dp_padding`
(main-thread all_reduce wait), `gloo:all_reduce` (on the `pt_gloo_runloop` threads),
`await_pending` (forward drain). torch trace timestamps are CLOCK_MONOTONIC, so the
per-rank traces are comparable across processes (for cross-rank arrival-skew analysis).

### Cleanup between runs

```bash
for p in $(ps -eo uid,pid,args | awk '$1=='$(id -u)'' | grep -iE "VLLM::|parity_runner" | grep -v grep | awk '{print $2}'); do kill -9 $p; done
find /dev/shm -maxdepth 1 -uid $(id -u) -delete 2>/dev/null   # only your own; leave other users'
```
