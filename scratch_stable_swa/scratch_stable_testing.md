# Scratch+Stable SWA — Testing & Results

This doc has everything related to testing and the performance numbers I got for scratch stable SWA.
---

## 1. Environment

### Versions (pin these — other builds have the slow v2v-copy bug)

The versions I used:

| package | version |
|---|---|
| `torch` | `2.10.0+cpu` |
| `torch-rbln` | `0.2.2.dev7+g57e4b35` |
| `rebel-compiler` | `0.10.5.dev353+g93d36eac` |


### Activate the venv

```
source .venv/bin/activate
```

After this, every command below is run verbatim. (Optional: prefix with `RBLN_RT_LOG_LEVEL=error` to quiet the runtime log spam.) I'm sure anyone reading this doc is competent enough but I'm just putting this here just in case lol.

---

## 2. Unit tests (CPU only — no NPU, run in seconds)

These validate the index math and the sub-block/KV-cache bookkeeping the scratch+stable path
relies on.

```
python scratch_stable_swa/tests/test_scratch_stable_metadata.py
python -m pytest tests/torch_compile/unit/v1/core/test_sub_block_prefix_caching.py -q
python -m pytest tests/torch_compile/unit/v1/worker/test_rbln_model_runner_kv_cache.py -q
```

What each covers:
- **`test_scratch_stable_metadata.py`** — the `scratch_stable_index_tensors` math
  (`cache_seq_lens`, `local_seq_idx`, `stable_end`, the stable-block gather) for decode, prefill
  chunk offset, and batched requests. Prints `PASS …` per case.
- **`test_sub_block_prefix_caching.py`** — sub-block hashing / chaining / matching (the layer
  scratch+stable composes with).
- **`test_rbln_model_runner_kv_cache.py`** — the runner-side KV-cache + copy-op bookkeeping.

Expected (current tree):

```
all scratch_stable_index_tensors tests passed      # metadata
84 passed                                           # sub_block_prefix_caching
25 passed                                           # rbln_model_runner_kv_cache
```

---

## 3. End-to-end benchmark + correctness (on NPU)

Note that I used gemma-3-1b-it through the real model and compares three
configs. **One flag selects the implementation**, so the same harness covers correctness *and*
performance.

| config | kernel | prefix caching | what it isolates |
|---|---|---|---|
| **A** | original single-block SWA | off | reference |
| **B** | scratch+stable | off | A→B = the extra-store overhead |
| **C** | scratch+stable | on | B→C = the cache benefit |

The harness sets all the required env (`VLLM_RBLN_USE_VLLM_MODEL`, `COMPILE_MODEL`,
`CUSTOM_KERNEL`, and the per-config `SWA_SCRATCH_STABLE` / `SUB_BLOCK_CACHE`) itself. With no
`--out`, each run **prints everything to the terminal** (the built prompts, the generated
answers, and the metrics) — nothing is written to a file.

Run the six commands (3 configs × 2 workloads):

```
python scratch_stable_swa/tests/_swa_impl_benchmark.py --worker --config A --workload share --native --prefix-blocks 3 --max-tokens 32
python scratch_stable_swa/tests/_swa_impl_benchmark.py --worker --config B --workload share --native --prefix-blocks 3 --max-tokens 32
python scratch_stable_swa/tests/_swa_impl_benchmark.py --worker --config C --workload share --native --prefix-blocks 3 --max-tokens 32
python scratch_stable_swa/tests/_swa_impl_benchmark.py --worker --config A --workload noshare --native --prefix-blocks 3 --max-tokens 32
python scratch_stable_swa/tests/_swa_impl_benchmark.py --worker --config B --workload noshare --native --prefix-blocks 3 --max-tokens 32
python scratch_stable_swa/tests/_swa_impl_benchmark.py --worker --config C --workload noshare --native --prefix-blocks 3 --max-tokens 32
```

- `share` = one large shared prefix (`--prefix-blocks 3` → 1536-token prefix) across 4 questions.
- `noshare` = distinct prefixes (the guardrail: caching must not help/regress here).

**Correctness check (by eye):** all three configs (A, B, C) for a given workload print
**identical** generated answers — that is the proof scratch+stable (B/C) matches the original
kernel (A), and that cache-on (C) matches cache-off (B). For `share` the answers are:

```
[0] 'The text was written by Edgar Allan Poe.\nThe title is "The Raven."'
[1] 'The main theme of the poem is ... unending sorrow ... the memory of Lenore ...'
[2] 'The opening creates a sense of mystery, dread, and melancholy. ...'
[3] 'The image is the raven's response "Nevermore."'
```

### Device-side NPU profiling (optional)

Add `--npu-profile` to any command above to also print a per-op **device cycle** table
(self-contained — junk dumps are auto-pruned):

```
python scratch_stable_swa/tests/_swa_impl_benchmark.py --worker --config C --workload share --native --prefix-blocks 3 --max-tokens 32 --npu-profile
```

Add `--profile --scopes` instead to write a Perfetto trace (`*.pt.trace.json.gz`) — the command
prints the exact path to upload to https://ui.perfetto.dev/.

---

## 4. Performance numbers

All numbers: gemma-3-1b-it, native window `W=512` (`block_size=512`), 3-block / 1536-token shared
prefix, `--max-tokens 32`. Correctness gate (greedy token ids A==B==C) **PASSES** both workloads.

### 4.1 Headline comparison

**`share` workload** (long shared prefix — caching applies):

| metric | A original | B ss no-cache | C ss cache |
|---|---|---|---|
| TTFT mean (s) | 0.574 | 0.616 | **0.208** |
| TPOT mean (s) | 0.0204 | 0.0224 | 0.0155 |
| e2e mean (s) | 0.957 | 1.037 | **0.498** |
| wall (s) | 1.542 | 1.665 | **0.812** |
| TPS (tok/s) | 62.9 | 58.2 | **119.4** |
| cached tokens | 0 | 0 | 6144 |

- **A→B (extra-store overhead):** TPOT **+9.9%**, TTFT **+7.3%**
- **B→C (prefix-cache benefit):** TTFT **−66.2%**, wall **−51.2%**
- **A→C (net):** TTFT **−63.7%**, wall **−47.3%**

**`noshare` workload** (distinct prefixes — guardrail):

| metric | A original | B ss no-cache | C ss cache |
|---|---|---|---|
| TTFT mean (s) | 0.564 | 0.575 | 0.597 |
| TPOT mean (s) | 0.0246 | 0.0263 | 0.0263 |
| wall (s) | 1.321 | 1.383 | 1.405 |
| TPS (tok/s) | 48.5 | 46.3 | 45.6 |
| cached tokens | 0 | 0 | 0 |

- A→B store overhead TPOT **+7.1%**; B→C ≈ noise (caching correctly idle); A→C net ≈ **+6%**
  (the store tax with no offsetting cache win — the expected trade-off on non-shared traffic).

**Headline:** scratch+stable costs ~7–10% per token for the extra stable-block store and pays it
back ~6–7× whenever there is prefix reuse.

### 4.2 Per-request stage times (vLLM metrics, config C cached, share)

```
prefill = 0.050 s    decode = 0.251 s    inference = 0.301 s    queue = 0.114 s
```

Decode dominates once caching collapses prefill — so per-token (TPOT) cost is where the
implementation matters most.

### 4.3 NPU device cycles (config C, `--npu-profile`)

The scratch+stable tax is overwhelmingly **host-side, not NPU**. On the device:

| op | cycles | % | note |
|---|---|---|---|
| `dense_fp16` | 15,656,924 | 76.7% | the model's matmuls (inherent) |
| `dynamic_matmul_fp16` | 1,172,904 | 5.7% | attention matmuls |
| `dynamic_concat` | 570,768 | 2.8% | SWA window |
| `dynamic_slice` | 510,048 | 2.5% | SWA window |
| `gelu` | 463,476 | 2.3% | |
| `dynamic_window_softmax` | 153,472 | 0.8% | SWA softmax |
| `dynamic_insert` | 67,520 | **0.3%** | **the stable-block store** |

The new stable store (`dynamic_insert`) is **0.3%** of device cycles, and the scratch+stable
decode kernel is only ~6% slower than the original on the NPU — the real cost is host metadata.

### 4.4 Host-side optimizations (2026-06-22) — deterministic op-count reductions

Two kernel-free host optimizations, both verified output-identical (token gate A==B==C +
on==off + byte-identical answers all pass):

| optimization | effect (ops inside `Preprocess`) |
|---|---|
| coalesce the 4 `ss_*` H2D transfers into 1 | `aten::copy_` 16964 → 16082 (−882, = 3×49 steps×6 SWA groups) |
| hoist group-invariant work out of the per-group loop (`build()` runs 7× per step) | `aten::repeat` **707 → 101**, `aten::arange` 3416 → 992, ~8500 fewer `to`/`_to_copy`/`copy_` |

Together they remove roughly **~500–550 µs/step of host work** (~15–25% of `Preprocess`). Wall-time
/ TPOT gains are real but below single-run noise at `max_num_seqs=2` (decode/queue swing
0.30–0.44 s run-to-run); the op-count drops are the reliable evidence.

---

## 5. Optional: standalone caching-equivalence test

A separate canonical equivalence check (also covers the sub-block "Case 1" path). Unlike the
benchmark, it does NOT set its own env, so export the flags first:

```
export VLLM_RBLN_USE_VLLM_MODEL=1 VLLM_RBLN_SWA_SCRATCH_STABLE=1 VLLM_RBLN_SUB_BLOCK_CACHE=1 VLLM_RBLN_COMPILE_MODEL=1 RBLN_USE_CUSTOM_KERNEL=1
python scratch_stable_swa/tests/_prefix_cache_swa.py --mode off --native --short
python scratch_stable_swa/tests/_prefix_cache_swa.py --mode on  --native --short
```

The two runs print their generated token ids; they must be **identical** (the `on` run reports
non-zero `prefix_cache_hit_tokens`, confirming caching actually fired). Result: **PASS** —
on==off, byte-identical, on this tree.
