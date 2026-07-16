# Request — continue async-overlap PERF work (504 already fixed)

> Paste this whole file as the first message of a new session (or say "read
> docs/async_overlap_perf_handoff.md and do it"). It is self-contained. Also load context from
> `docs/async_overlap.md` (§1 configs, §2 the 504 fix, §3 perf method) and the auto-memory
> `async-overlap-p0-aborts.md`.

## What is already DONE (do not redo)
The async-overlap **504 abort is fixed** — root cause: `AsyncDynamoRuntime.run` allocated a fresh
device output tensor every forward (`torch.empty`); under `RBLN_DYNAMO_ASYNC=defer` that per-step
`AllocBlock → context_.Submit()` (page-table commit) raced the worker's in-flight `rcclAllToAllX`
(page fault → `SYS_TASK_ABORTED`/504). Fix = **approach A**: reuse a ring of persistent device
output buffers (`rebel_compiler/rebel/python/rebel/sync_runtime.py`, `_out_ring` /
`_acquire_output_buffer`, env `RBLN_ASYNC_OUTPUT_RING` default 8, keyed by (output_index, shape,
dtype)). Validated DP4+EP: decode `AllocBlock` 17→0, 504=0, `repro ✅ PASS` (bit-identical).
Fence approaches were tried and ABANDONED (deadlock — see docs §2; do not revisit).

## Your task: PERFORMANCE
**Goal:** config **#3** (async: `RBLN_DYNAMO_ASYNC=defer VLLM_RBLN_OPTIMISTIC_SCHED=1`) must be
**faster** than config **#1** (sync: `VLLM_RBLN_DISABLE_ASYNC=1`) — that is the whole point of
async-overlap (it hides the per-step CPU DP `num_tokens` all_reduce behind the NPU forward).

### Step 1 — measure cleanly and logically
- **Quiet machine is mandatory.** This is a SHARED box; a neighbour's DP job spikes 1-min load to
  ~150 and starves decode (~32→~10 tok/s), making numbers meaningless. Check `/proc/loadavg`; only
  trust runs where load1 stayed low (< ~25) for the whole run. Pick a free NPU set: `rbln-smi`
  (e.g. devices 0-3 if the neighbour is on 4-7). Ready-made autonomous helper:
  **`~/perf_auto.sh`** — waits for load1<25, runs sync + async (`--repro-run 12` ≈ 13 samples each),
  auto-retries if the sync median looks contention-starved (≤25), writes `~/perf_auto.out`
  (`PA cfg=... median=...`, `PA CLEAN RESULT`). Re-create it if missing (it's not committed).
- **Amortize init:** weights are streamed every launch (`RBLN_WEIGHT_FREE=1`, several min;
  compilation IS cached by `--use-cached-models`). Use `--repro-run N` for N+1 samples per single
  load — never relaunch per sample. `--cache-ignore` only bypasses the parity *result* cache
  (forces a real run); keep it.
- Report medians (variation is large; ≥10 samples/config). Exact flags: docs §1 / §3.

### Step 2 — if #3 is NOT faster, optimize (autonomously)
Root-cause first, then fix. Leads already established (docs §3): the deferred forward *does*
overlap; the drain is dominated by `await(forward)`; the leading hypothesis is that async desyncs
the DP ranks and the EP all-to-all (`rcclAllToAllX`, a barrier) inside the forward absorbs the skew
and inflates `forward(N)`, cancelling the hidden all_reduce. Get **per-rank forward / CCL timing**
on a quiet machine to confirm before changing anything.

## HARD PRECONDITION on every perf change (non-negotiable)
Functionality must never regress. After EVERY change, re-validate on DP4+EP and confirm BOTH:
1. **No 504** — `grep -c SYS_TASK_ABORTED` == 0, run completes end-to-end.
2. **repro ✅ PASS** — bit-identical across `--repro-run` (the ring-reuse / any change introduces
   no output corruption).
If a perf change breaks either, **revert it** — a faster-but-wrong async is worthless (the user
has stated this repeatedly). Prefer the smallest change that works; measure before/after.

## Constraints (from the user — apply strictly)
- **General**, not tied to this one workload/config. No **workaround**, no **pre-reservation /
  prewarm** (guessing sizes is banned), no host **gloo** in the runtime — CCL/device sync must be
  device-level (`rccl`/`rbln`); the user is the runtime team.
- Follow the `karpathy-guidelines` skill: surgical, minimal, no speculative abstraction; state
  assumptions; define verifiable success criteria and loop until met.
- Build via the `rebel-compiler-build` skill, but note: build with the **vllm-executor venv**
  (`~/codebase/vllm-executor/.venv`, has rebel-compiler editable + numpy) — plain `~/.venv` skips
  the `rebel._C` rebuild. Python-only changes to `sync_runtime.py` need no rebuild (editable;
  the git source `rebel_compiler/rebel/python/rebel/...` is hardlinked to the loaded
  `rebel_compiler/python/rebel/...`).

## Pointers
- `docs/async_overlap.md` — §1 the 3 configs + exact run commands, §2 the 504 fix (done), §3 perf
  method + hypotheses, "How to continue (perf)".
- auto-memory `async-overlap-p0-aborts.md` — one-screen summary + ruled-out approaches.
- The A fix: `rebel_compiler/.../sync_runtime.py` `AsyncDynamoRuntime`.
- Abandoned fence attempt: `git stash list` in `rebel_compiler` (stash@{0}) — reference only.
- Model/scenario: `gpt-oss-120b --ep --dp 4 --rsd 1 --batch 1 --max-tokens 128 --num-prompts 4`,
  `--mode 0` (generation; mode 1 = compiler-func fixes max_tokens=5, not for perf).
