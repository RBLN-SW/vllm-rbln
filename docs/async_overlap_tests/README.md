# async-overlap: determinism & perf test handoff

Portable procedure to reproduce the **determinism** and **throughput** tests for
the async-overlap work on any RBLN server (incl. a dedicated measurement box).
All scripts here are self-contained; paths are configurable via env vars.

---

## 0. What "async" means here

Two independent knobs, both ON = "full async":

| Knob | Env | Layer |
|---|---|---|
| vLLM async (optimistic) scheduling | `VLLM_RBLN_OPTIMISTIC_SCHED=1` | selects `RBLNAsyncScheduler` (vllm-rbln) |
| rebel dynamo async runtime (deferred forward) | `RBLN_DYNAMO_ASYNC=defer` | rebel_compiler AsyncRuntime |

Baseline "sync" = `VLLM_RBLN_DISABLE_ASYNC=1` (blocking scheduling). Sync is the
determinism reference. Do **not** confuse `RBLN_DYNAMO_ASYNC` (dynamo async) with
`RBLN_RUNTIME_FORCE_SYNC` (device-level knob — not used here).

---

## 1. Required code state

### vllm-rbln @ `async-overlap-prototype`
- `ca061d54` — **determinism fix**: cold-start quiesce gate inside
  `RBLNAsyncScheduler` (`vllm_rbln/v1/core/rbln_scheduler.py`). This is the whole
  determinism mechanism. Always-on, async-only, no env flag.
- `f6b3cf4c` — revert of the misattributed unconditional force_sync
  (`def1bc9b`); determinism is owned by the scheduler, not the sampler.

### rebel_compiler @ `async_rebase`
- `9463c95766` — 504 fix via output-buffer reuse (**KEEP** — complementary).
- `41a2a9869e` — 504 fix via stream-ordered device memory (commit-deferral +
  free-side quarantine).
- `4ff56e8742` — Stream thread-safety hardening.

The venv that runs `parity_runner` **must load a rebel_compiler build containing
these commits** (see §2).

### upstream vLLM — MUST BE PRISTINE ⚠️
The determinism fix lives **entirely in vllm-rbln**. Upstream vLLM
(`vllm/v1/engine/core.py`, `vllm/envs.py`, …) must have **no RBLN patch** — in
particular **no `RBLN_DETERMINISTIC_ADMIT`** (an old diagnostic hack that patched
`_process_input_queue`; it has been removed). Verify:

```bash
python - <<'PY'
import vllm, os, subprocess
d = os.path.dirname(vllm.__file__)
r = subprocess.run(["grep","-rn","RBLN_DETERMINISTIC_ADMIT",d], capture_output=True, text=True)
print("CONTAMINATED:" , r.stdout or "(clean — no RBLN_DETERMINISTIC_ADMIT)")
PY
```
(The only legitimate `rbln` mentions upstream are the Ray-accelerator refs in
`vllm/envs.py`, e.g. `RAY_EXPERIMENTAL_NOSET_RBLN_RT_VISIBLE_DEVICES`.)

### vllm-executor (parity_runner) — perf instrumentation
Perf uses an env-gated timing print in `_run_llm`. Apply once (copy the patch to
the vllm-executor repo, or apply by path):
```bash
cd <vllm-executor>
git apply <vllm-rbln>/docs/async_overlap_tests/parity_runner_perf_timing.patch
# if line numbers drifted, hand-edit _run_llm per the patch's +side.
```
It has **no effect** unless `RBLN_PERF_TIMING=1`, so it's safe to leave applied.

---

## 2. Build rebel_compiler into the test venv

The test venv = the one that runs `parity_runner` (it dlopen's rebel's `_C.so`).

```bash
source <test-venv>/bin/activate          # e.g. ~/codebase/vllm-executor/.venv
export PATH="$PATH:$HOME/.venv/bin"       # conan must be reachable
cd ~/codebase/rebel_compiler
./rebel_install.sh -a -n                  # -a full, -n ninja; drop -n on ninja error
```
Verify the freshly built `_C.so` is what the test venv imports (timestamp check),
and that the three rebel commits above are present (`git -C ~/codebase/rebel_compiler log --oneline -3`).

---

## 3. Environment / model

- **Devices**: 4 RBLN devices, DP4 + EP. Set `RBLN_DEVICES` (default `4,5,6,7`).
- **Model**: `gpt-oss-120b` (MoE, mxfp4), cached offline
  (`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`).
- **Compiled-graph cache**: `--use-cached-models`. First run per process may
  compile (cold); warm runs reuse the cached graph.
- Common env is baked into the scripts (`COMMON=...`).

---

## 4. Determinism test

`./det_check.sh` runs one config N times in a single process and checks that
**every run's per-input output *text* is bit-identical** (greedy temp=0, seed 42;
`parity_runner` compares `outputs[i].text`, with `i` aligned to input order —
vLLM returns `sorted(outputs, key=request_id)` so position i = same input).

```bash
OUTDIR=./_test_out \
NAME=full_async_b4 \
CFGENV="VLLM_RBLN_OPTIMISTIC_SCHED=1 RBLN_DYNAMO_ASYNC=defer" \
BATCH=4 REPRO=12 \
./det_check.sh
# -> prints: DET_RESULT ... repro=PASS|FAIL abort=<n> mismatches=<n>
```
- `repro=PASS` + `abort=0` + `mismatches=0` = deterministic.
- Run sync (`CFGENV="VLLM_RBLN_DISABLE_ASYNC=1"`) as the reference.

## 5. Perf test (throughput)

`./perf_metric.sh` + `perf_parse.py`. **Metric = total_output_tokens / generate
wall-clock, warm runs only** (the cold/compile generate — call 1 — is dropped).
Batch semantics: per-rank batch `B = max_num_seqs`, `num_prompts = B*dp`, so each
DP rank runs `B` concurrent sequences; global tok/s is summed across ranks.

```bash
OUTDIR=./_test_out BATCHES="1 4 8 16" REPRO=2 ./perf_metric.sh
# -> table: batch | async tok/s | sync tok/s | async/sync ratio | detail
```

---

## 6. New-server session quickstart (TL;DR)

```bash
# 1. sync the three repos at the branches/commits in §1
#    vllm-rbln @ async-overlap-prototype  (>= f6b3cf4c)
#    rebel_compiler @ async_rebase        (>= 4ff56e8742)
#    vllm-executor                        (+ parity_runner_perf_timing.patch)

# 2. confirm upstream vLLM is pristine (no RBLN_DETERMINISTIC_ADMIT) — §1 snippet

# 3. build rebel into the test venv — §2
source <test-venv>/bin/activate
export PATH="$PATH:$HOME/.venv/bin"
( cd ~/codebase/rebel_compiler && ./rebel_install.sh -a -n )

# 4. copy this dir's scripts somewhere writable, set devices
export RBLN_DEVICES=0,1,2,3      # this box's 4 devices

# 5. determinism (async vs sync)
OUTDIR=./_out NAME=async_b4 CFGENV="VLLM_RBLN_OPTIMISTIC_SCHED=1 RBLN_DYNAMO_ASYNC=defer" BATCH=4 REPRO=12 ./det_check.sh
OUTDIR=./_out NAME=sync_b4  CFGENV="VLLM_RBLN_DISABLE_ASYNC=1"                              BATCH=4 REPRO=12 ./det_check.sh

# 6. perf sweep (async vs sync, batches 1/4/8/16)
OUTDIR=./_out BATCHES="1 4 8 16" ./perf_metric.sh
```

---

## 7. Findings on the origin server (to confirm / compare after migration)

- **504 (SYS_TASK_ABORTED)**: fixed — 0 aborts across the full batch sweep.
- **Determinism**: async PASS at batch **1/4/8**; **FAIL at batch 16**.
  - Judged a **token-scheduling** issue, **not** a functional/operational bug:
    sync b16 is fully deterministic (same test), so the model/kernel/allocator
    are correct at b16. The async divergence comes from non-blocking
    prefill/decode **interleaving** (prefill batch size is limited to 1, so many
    prefills interleave with decodes; async timing varies the per-step DP batch
    composition, and EP+DP MoE is not batch-composition-invariant). The
    cold-start quiesce fixes only the first-step composition, enough for ≤8.
  - `eraase_count <= 0` runtime warning appears in async batch>1 (incl. PASSING
    b4/b8) — pre-existing non-fatal `TryMergeBlocks` check (rebel commit
    `1d3f8eaced`, not ours); **uncorrelated with the determinism failure**.
- **Perf** (corrected metric §5: total_output_tokens / generate wall, warm-only,
  global summed over DP ranks; DP4+EP, gpt-oss-120b, max_tokens=128):

  | per-rank batch | async tok/s | sync tok/s | async/sync |
  |---:|---:|---:|---:|
  | 1  | 145.4 | 167.1  | 0.87x |
  | 4  | 360.7 | 513.4  | 0.70x |
  | 8  | 612.6 | 1246.0 | 0.49x |
  | 16 | 984.4 | 1947.9 | 0.51x |

  **The async-overlap prototype is a throughput LOSS at every batch, worsening
  with batch (~2x slower at b8/b16)** — the opposite of the overlap premise.
  Likely cause: the single process-wide FIFO async worker (serializes all async
  runtimes) + per-step deferred-forward drain overhead, which dominates as device
  work grows. Implication for the rewrite: do **not** assume async helps at scale;
  the overlap mechanism itself needs redesign. (Earlier median-of-per-step-rate
  numbers were unreliable — startup-0 contamination — and are superseded by this.)

Migration trigger: if determinism/perf can't be run cleanly here, move to the
dedicated measurement box and repeat §6, then fill in a results table below.
