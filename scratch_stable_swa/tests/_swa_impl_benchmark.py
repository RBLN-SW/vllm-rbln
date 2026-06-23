# Copyright 2025 Rebellions Inc. All rights reserved.
# ruff: noqa
"""A/B/C comparison harness: original SWA vs scratch+stable SWA (prefix caching).

For the shipped scratch+stable path (see scratch_stable_swa/scratch_stable_testing.md
and scratch_stable_swa/scratch_stable_code_walkthrough.MD).
It quantifies, on a real gemma-3 run, the two things that matter:

  A -> B : the EXTRA-STORE OVERHEAD of scratch+stable (the append into the stable
           block on every step), measured with caching held OFF -> pure kernel tax.
  B -> C : the PREFIX-CACHE BENEFIT that scratch+stable unlocks (the whole point).
  A -> C : the end-to-end net effect.

Config matrix (one flag selects the implementation: VLLM_RBLN_SWA_SCRATCH_STABLE):

  | id | kernel               | prefix caching | flags                                  |
  | A  | original single-block| off (can't)    | SWA_SCRATCH_STABLE=0                    |
  | B  | scratch+stable       | off            | SWA_SCRATCH_STABLE=1, caching off       |
  | C  | scratch+stable       | on             | SWA_SCRATCH_STABLE=1, caching on        |

Metrics (read from llm.get_metrics()):
  - TTFT      vllm:time_to_first_token_seconds   (mean = sum/count)
  - TPOT/ITL  vllm:inter_token_latency_seconds   (mean = sum/count) -> the decode tax
  - e2e       vllm:e2e_request_latency_seconds
  - TPS       vllm:generation_tokens / wall-clock of the timed generate()
  - hit rate  vllm:prefix_cache_hits / vllm:prefix_cache_queries
  - cached    vllm:prompt_tokens_cached          (tokens served from cache)
NOTE: vllm:prefix_cache_hits reads 0 for the hybrid SWA group even on a real hit
(design doc 10.7). The TRUSTWORTHY hit signals are prompt_tokens_cached and the
token-id equivalence gate; the raw hits counter is reported for completeness only.

HARD CONSTRAINT: two LLMs in one process crash the NPU ("Bad address"). So each
config runs in its OWN subprocess. This file is both the orchestrator (default)
and the per-config worker (--worker).

Usage (orchestrate all configs/workloads, print the comparison table):
  VLLM_RBLN_USE_VLLM_MODEL=1 VLLM_RBLN_SUB_BLOCK_CACHE=1 \
  VLLM_RBLN_COMPILE_MODEL=1 RBLN_USE_CUSTOM_KERNEL=1 \
  python scratch_stable_swa/tests/_swa_impl_benchmark.py --native

The orchestrator sets the per-config flags (SWA_SCRATCH_STABLE / caching) itself;
keep the common recipe flags above in the environment.
"""

import argparse
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
from time import perf_counter

# ---------------------------------------------------------------------------
# Reuse the prompts + helpers from the existing prefix-cache driver so the
# workloads are identical to the validated caching-equivalence test.
# ---------------------------------------------------------------------------
_HERE = pathlib.Path(__file__).parent
_spec = importlib.util.spec_from_file_location(
    "_pcswa", _HERE / "_prefix_cache_swa.py"
)
_pcswa = importlib.util.module_from_spec(_spec)
# Defer executing the sibling module until we are actually in worker mode (it
# imports vllm at import time, which we don't want in the lightweight orchestrator).


# Recipe flags common to ALL configs (design doc 10.7). gemma-3 only loads via vLLM's
# native model path (VLLM_RBLN_USE_VLLM_MODEL=1); the optimum-rbln path rejects
# Gemma3ForCausalLM. Set as defaults so the harness runs without an env prefix; an
# explicit value in the environment still wins (setdefault).
_RECIPE_ENV = {
    "VLLM_RBLN_USE_VLLM_MODEL": "1",
    "VLLM_RBLN_COMPILE_MODEL": "1",
    "RBLN_USE_CUSTOM_KERNEL": "1",
}


def _config_env(config: str) -> dict:
    """Implementation-selection flags that DEFINE each config (hard-set, so a stray
    shell value can't break them). SUB_BLOCK_CACHE is tied to scratch+stable: gemma is
    a multi-group hybrid, and sub-block caching on a multi-group cache is only allowed
    WITH scratch+stable (rbln_model_runner.initialize_kv_cache guard). So A (original
    kernel) must have it OFF; B/C (scratch+stable) have it ON. B and C then differ ONLY
    in enable_prefix_caching, keeping A->B (kernel/store cost) and B->C (caching) clean."""
    on = config in ("B", "C")
    return {
        "VLLM_RBLN_SWA_SCRATCH_STABLE": "1" if on else "0",
        "VLLM_RBLN_SUB_BLOCK_CACHE": "1" if on else "0",
    }


CONFIGS = ("A", "B", "C")
CONFIG_LABEL = {
    "A": "original (no-cache)",
    "B": "scratch+stable (no-cache)",
    "C": "scratch+stable (cache)",
}
WORKLOADS = ("share", "noshare")

# Distinct, non-overlapping prefixes -> the no-sharing guardrail workload. Caching
# must NOT help here; B/C must stay within noise of A on this one.
_NOSHARE_PREFIXES = [
    "A brief field report on alpine botany. The edelweiss grows above the treeline "
    "and has adapted dense white hairs to reflect ultraviolet light and retain heat. ",
    "Notes on deep-sea hydrothermal vents. Chemosynthetic bacteria at black smokers "
    "convert hydrogen sulfide into energy, forming the base of an ecosystem without sun. ",
    "A short summary of medieval cartography. The mappae mundi placed Jerusalem at the "
    "center and oriented east to the top, mixing geography with theology and legend. ",
    "An overview of jazz improvisation. A soloist outlines the chord changes while "
    "displacing rhythm and quoting motifs, trading phrases with the rest of the combo. ",
]
_NOSHARE_QUESTIONS = [
    "\nQuestion: Summarize the passage in one sentence.\nAnswer:\n",
    "\nQuestion: What is the central mechanism described?\nAnswer:\n",
    "\nQuestion: Name one detail from the text.\nAnswer:\n",
    "\nQuestion: What field does this passage belong to?\nAnswer:\n",
]


# ===========================================================================
# Worker: build ONE LLM for ONE config, run the workload, dump a result JSON.
# ===========================================================================
def _hist_mean(metrics, name):
    """Mean of a histogram metric (sum/count), or None if unobserved."""
    from vllm.v1.metrics.reader import Histogram

    total_sum = 0.0
    total_count = 0
    for m in metrics:
        if isinstance(m, Histogram) and m.name == name:
            total_sum += m.sum
            total_count += m.count
    return (total_sum / total_count) if total_count else None


def _counter(metrics, name):
    from vllm.v1.metrics.reader import Counter

    return sum(m.value for m in metrics if isinstance(m, Counter) and m.name == name)


_SHARE_QUESTIONS = [
    "\nQuestion: Who wrote this text and what's the title?\nAnswer:\n",
    "\nQuestion: What is the main theme?\nAnswer:\n",
    "\nQuestion: What feeling does the opening create?\nAnswer:\n",
    "\nQuestion: Name one image from the text.\nAnswer:\n",
]


def _grow_to_tokens(tok, base: str, target: int) -> str:
    """Repeat `base` until it tokenizes to >= `target` tokens, then trim to exactly
    `target` so the prefix is a clean multiple of the block size (maximizes
    full-block prefix-cache hits)."""
    n_base = len(tok(base, add_special_tokens=False).input_ids)
    reps = max(1, target // max(1, n_base) + 1)
    ids = tok((base + " ") * reps, add_special_tokens=False).input_ids[:target]
    return tok.decode(ids)


def _build_workload(workload: str, tok, target_tokens: int):
    """Return (prompts, shared_prefix_token_count). For `share`, ONE shared prefix
    of ~target_tokens + 4 questions. For `noshare`, 4 DISTINCT prefixes each grown
    to the same length (so the decode-tax comparison is at equal context length)."""
    if workload == "share":
        shared = _grow_to_tokens(tok, _pcswa.prefix, target_tokens)
        n = len(tok(shared, add_special_tokens=False).input_ids)
        return [shared + q for q in _SHARE_QUESTIONS], n
    else:  # noshare: distinct prefixes, no cross-request reuse
        prompts = []
        for i in range(4):
            pfx = _grow_to_tokens(tok, _NOSHARE_PREFIXES[i], target_tokens)
            prompts.append(pfx + _NOSHARE_QUESTIONS[i])
        return prompts, 0  # no shared prefix in this workload


def _print_prompts(prompts, tok, full=False, edge=240):
    """Show the built prompts. By default truncates each to head+tail (the tail holds
    the question suffix) with the token count; --full-prompts prints them verbatim."""
    print(f"\n--- built {len(prompts)} prompt(s) ---")
    for i, p in enumerate(prompts):
        n = len(tok(p, add_special_tokens=False).input_ids)
        print(f"[prompt {i}] {n} tokens:")
        if full or len(p) <= 2 * edge:
            print(p)
        else:
            print(p[:edge] + f"\n    ...<{len(p) - 2 * edge} chars elided>...\n"
                  + p[-edge:])
        print("-" * 60)


def _prune_npu_junk(npu_dir):
    """The RBLN device profiler scatters hundreds of MB of debug dumps into CWD
    (tvm_debug.log, rank_0/, kernel-compile temp dirs). Keep only the command-stream
    profiles + runtime summary; delete everything else so repeated runs can't fill
    the disk."""
    import shutil
    for entry in os.listdir(npu_dir):
        if entry.startswith("0_cs_") or entry == "runtime.log":
            continue
        p = os.path.join(npu_dir, entry)
        try:
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
        except OSError:
            pass


def _npu_cycles_table(npu_dir, top=20):
    """Aggregate device compute-cycles by op from the RBLN command-stream logs
    (0_cs_*.log: per-entry `comp_name(...), comp_cycle(N)`)."""
    import glob
    import re
    pat = re.compile(r"comp_name\(([^)]*)\), comp_cycle\((\d+)\)")
    cyc, cnt = {}, {}
    for f in glob.glob(os.path.join(npu_dir, "0_cs_*.log")):
        with open(f) as fh:
            for name, c in pat.findall(fh.read()):
                if not name:
                    continue
                cyc[name] = cyc.get(name, 0) + int(c)
                cnt[name] = cnt.get(name, 0) + 1
    if not cyc:
        print(f"[npu-profile] no command-stream cycles found in {npu_dir}")
        return
    total = sum(cyc.values())
    print(f"\n--- NPU device cycles by op  (total={total:,} cycles) ---")
    print(f"{'op':<26}{'cycles':>14}{'%':>7}{'calls':>9}")
    for name in sorted(cyc, key=cyc.get, reverse=True)[:top]:
        print(f"{name:<26}{cyc[name]:>14,}{100 * cyc[name] / total:>6.1f}%{cnt[name]:>9}")


def run_worker(config: str, workload: str, out: str, native: bool, max_tokens: int,
               prefix_blocks: int, show_full_prompts: bool = False,
               profile: bool = False, profile_dir: str = None,
               scopes: bool = False, npu_profile: bool = False):
    # Set the recipe + implementation-selection env BEFORE importing vllm (these are
    # read at LLM-construction time). config is the single source of truth for which
    # SWA implementation runs, so a direct `--worker` invocation is self-consistent.
    for k, v in _RECIPE_ENV.items():
        os.environ.setdefault(k, v)
    os.environ.update(_config_env(config))
    if scopes:
        # Emit the named host scopes (Forward/Preprocess/Postprocess/Sample/Bookkeep)
        # in the torch trace. They use record_function_or_nullcontext, which is a no-op
        # unless this is set — that's why a default trace shows only raw aten ops.
        os.environ["VLLM_CUSTOM_SCOPES_FOR_PROFILING"] = "1"
    if npu_profile:
        # Activate the RBLN device profiler (rebel flags.RBLN_PROFILER) for true
        # device-side NPU timings. The torch profiler is CPU-only and cannot see inside
        # the NPU. Not supported with model parallel; fine here (TP=1, single device).
        os.environ["RBLN_PROFILER"] = "1"
    # Execute the sibling module now (imports vllm) to reuse its prompts/helpers.
    _spec.loader.exec_module(_pcswa)
    from vllm import LLM, SamplingParams
    from vllm.transformers_utils.config import get_hf_text_config

    enable_caching = config == "C"
    W = 512 if native else 1024
    target_tokens = prefix_blocks * W
    # room for the question (~40 tok) + generation + chunk slack on top of the prefix.
    max_model_len = target_tokens + 512 + max_tokens

    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)

    def _sw_override(hf_config):
        get_hf_text_config(hf_config).update({"sliding_window": 1024})
        return hf_config

    # Resolve output paths to ABSOLUTE now — the NPU profiler forces a chdir below, and
    # _HERE / a relative --out would otherwise break once cwd changes.
    out = os.path.abspath(out) if out else None
    base_profile_dir = os.path.abspath(profile_dir or str(
        _HERE / "_swa_bench_out" / "profile" / f"{config}_{workload}"))

    # Torch profiler -> per-config Perfetto trace dir (see examples/experimental/
    # offline_inference_profile.py). Each config writes its own dir so A/B/C traces
    # can be loaded side by side at https://ui.perfetto.dev/.
    llm_kwargs = {}
    if profile:
        os.makedirs(base_profile_dir, exist_ok=True)
        llm_kwargs["profiler_config"] = {
            "profiler": "torch", "torch_profiler_dir": base_profile_dir}
        print(f"[profile] torch traces -> {base_profile_dir}")

    # The RBLN device profiler dumps 0_cs_*.log (per-op device cycles) PLUS ~500MB of
    # debug junk into CWD. Contain it: run inside <profile_dir>/npu and prune the junk
    # afterward (finally), so repeated runs can't fill the disk.
    npu_dir = None
    prev_cwd = os.getcwd()
    if npu_profile:
        npu_dir = os.path.join(base_profile_dir, "npu")
        os.makedirs(npu_dir, exist_ok=True)
        os.chdir(npu_dir)
        print(f"[npu-profile] device logs -> {npu_dir}")

    try:
        llm = LLM(
            model="google/gemma-3-1b-it",
            max_num_seqs=2,
            max_model_len=max_model_len,
            enable_chunked_prefill=True,
            max_num_batched_tokens=128,
            hf_overrides=None if native else _sw_override,
            trust_remote_code=True,
            enable_prefix_caching=enable_caching,
            block_size=W,
            num_gpu_blocks_override=512,
            disable_log_stats=False,  # required: get_metrics() asserts stat logging on
            **llm_kwargs,
        )

        tok = llm.get_tokenizer()
        prompts, shared_prefix_tokens = _build_workload(workload, tok, target_tokens)
        _print_prompts(prompts, tok, full=show_full_prompts)

        # Tiny warmup so the first compiled step's cost is not folded into the timed run.
        llm.generate(".", SamplingParams(temperature=0.0, max_tokens=2))
        # For the cache config, prime the shared prefix so the timed run actually hits.
        if enable_caching and workload == "share":
            llm.generate(prompts[0], sampling_params)

        before = llm.get_metrics()
        hits_before = _counter(before, "vllm:prefix_cache_hits")
        queries_before = _counter(before, "vllm:prefix_cache_queries")
        cached_before = _counter(before, "vllm:prompt_tokens_cached")
        gen_before = _counter(before, "vllm:generation_tokens")

        # Profile the timed generate (warmup/prime excluded, as in the example). NOTE:
        # profiler overhead inflates wall/TTFT/TPOT, so treat a profiling run as a
        # trace-collection run, not a headline-perf run.
        if profile:
            llm.start_profile()
        start = perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        wall = perf_counter() - start
        if profile:
            llm.stop_profile()
            import glob
            traces = sorted(glob.glob(
                os.path.join(base_profile_dir, "*.pt.trace.json.gz")))
            if traces:
                print(f"[profile] Perfetto trace -> {traces[-1]}  "
                      "(upload this to ui.perfetto.dev)")

        after = llm.get_metrics()
        gen_tokens = _counter(after, "vllm:generation_tokens") - gen_before

        result = {
            "config": config,
            "workload": workload,
            "prefix_blocks": prefix_blocks,
            "shared_prefix_tokens": shared_prefix_tokens,
            "wall_s": wall,
            "ttft_mean_s": _hist_mean(after, "vllm:time_to_first_token_seconds"),
            "tpot_mean_s": _hist_mean(after, "vllm:inter_token_latency_seconds"),
            "e2e_mean_s": _hist_mean(after, "vllm:e2e_request_latency_seconds"),
            # Per-request STAGE times (the decode-stage timing the manager asked for):
            "prefill_time_mean_s": _hist_mean(after, "vllm:request_prefill_time_seconds"),
            "decode_time_mean_s": _hist_mean(after, "vllm:request_decode_time_seconds"),
            "inference_time_mean_s": _hist_mean(
                after, "vllm:request_inference_time_seconds"),
            "queue_time_mean_s": _hist_mean(after, "vllm:request_queue_time_seconds"),
            "gen_tokens": gen_tokens,
            "tps": (gen_tokens / wall) if wall else None,
            "hits": _counter(after, "vllm:prefix_cache_hits") - hits_before,
            "queries": _counter(after, "vllm:prefix_cache_queries") - queries_before,
            "cached_tokens": _counter(after, "vllm:prompt_tokens_cached") - cached_before,
            "token_ids": [list(o.outputs[0].token_ids) for o in outputs],
            "answers": [o.outputs[0].text for o in outputs],
        }
        if out:
            with open(out, "w") as f:
                json.dump(result, f)

        # Generated answers (the distinguishing question suffix is in the prompt tail).
        print(f"\n--- generated answers ({len(outputs)}) ---")
        for i, text in enumerate(result["answers"]):
            print(f"[answer {i}] {text!r}")
            print("-" * 60)

        # Readable summary so a standalone (--out-less) run is useful.
        print(f"\n[worker {config}/{workload}] "
              f"prefix_blocks={prefix_blocks} shared_prefix_tokens={shared_prefix_tokens}")
        print(f"  TTFT={result['ttft_mean_s']}  TPOT={result['tpot_mean_s']}  "
              f"e2e={result['e2e_mean_s']}")
        print(f"  STAGE (per req): prefill={result['prefill_time_mean_s']}  "
              f"decode={result['decode_time_mean_s']}  "
              f"inference={result['inference_time_mean_s']}  "
              f"queue={result['queue_time_mean_s']}")
        print(f"  wall={wall:.3f}s  gen_tokens={gen_tokens}  tps={result['tps']:.1f}  "
              f"hits={result['hits']}  cached_tokens={result['cached_tokens']}"
              + (f"  -> {out}" if out else ""))

        if npu_profile:
            _npu_cycles_table(npu_dir)
    finally:
        if npu_profile:
            os.chdir(prev_cwd)
            _prune_npu_junk(npu_dir)
            # A few empty profiler stubs are touched at import time (before the chdir),
            # so they land in prev_cwd; remove them to keep the repo clean.
            import shutil
            for stub in ("tvm_debug.log", "profiler_debug.log",
                         "profiler_counter_invalid.log", "rank_0"):
                p = os.path.join(prev_cwd, stub)
                try:
                    shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
                except OSError:
                    pass
            print(f"[npu-profile] kept 0_cs_*.log + runtime.log in {npu_dir} "
                  "(junk pruned)")


# ===========================================================================
# Orchestrator: spawn each config in its own subprocess, then compare.
# ===========================================================================
def _env_for(config: str) -> dict:
    env = dict(os.environ)
    for k, v in _RECIPE_ENV.items():
        env.setdefault(k, v)
    env.update(_config_env(config))
    return env


def _fmt(v, prec=4):
    if isinstance(v, bool) or v is None:
        return "  n/a "
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return f"{v:.{prec}f}"
    return "  n/a "


def _pct_delta(base, new):
    """Percent change new vs base (positive = slower/more)."""
    if not base or not new:
        return None
    return (new - base) / base * 100.0


def orchestrate(native: bool, max_tokens: int, prefix_blocks: int, workloads,
                scratch_dir: pathlib.Path, profile: bool = False):
    scratch_dir.mkdir(parents=True, exist_ok=True)
    results = {}  # (config, workload) -> dict | None
    for workload in workloads:
        for config in CONFIGS:
            out = scratch_dir / f"swa_bench_{config}_{workload}.json"
            cmd = [
                sys.executable, str(_HERE / "_swa_impl_benchmark.py"),
                "--worker", "--config", config, "--workload", workload,
                "--out", str(out), "--max-tokens", str(max_tokens),
                "--prefix-blocks", str(prefix_blocks),
            ]
            if native:
                cmd.append("--native")
            if profile:
                cmd += ["--profile", "--profile-dir",
                        str(scratch_dir / "profile" / f"{config}_{workload}")]
            print(f"\n>>> running config {config} / workload {workload} ...")
            proc = subprocess.run(cmd, env=_env_for(config))
            if proc.returncode == 0 and out.exists():
                with open(out) as f:
                    results[(config, workload)] = json.load(f)
            else:
                print(f"!!! config {config}/{workload} FAILED (rc={proc.returncode})")
                results[(config, workload)] = None

    _report(results, workloads)


def _report(results, workloads):
    print("\n" + "=" * 78)
    print("SWA IMPLEMENTATION COMPARISON  (gemma-3-1b-it)")
    print("=" * 78)

    for workload in workloads:
        a = results.get(("A", workload))
        b = results.get(("B", workload))
        c = results.get(("C", workload))
        present0 = next((r for r in (a, b, c) if r), None)
        size = ""
        if present0:
            size = (f"  (prefix_blocks={present0['prefix_blocks']}, "
                    f"shared_prefix_tokens={present0['shared_prefix_tokens']})")
        print(f"\n### workload = {workload}{size}")

        # --- correctness gate: greedy token ids must match across A/B/C ---
        gate = "n/a"
        present = [r for r in (a, b, c) if r]
        if len(present) >= 2:
            ref = present[0]["token_ids"]
            gate = "PASS" if all(r["token_ids"] == ref for r in present) else "FAIL"
        print(f"correctness gate (token-id A==B==C): {gate}"
              + ("   <-- perf below is UNTRUSTWORTHY" if gate == "FAIL" else ""))

        rows = [
            ("TTFT mean (s)", "ttft_mean_s"),
            ("TPOT mean (s)", "tpot_mean_s"),
            ("e2e mean (s)", "e2e_mean_s"),
            ("wall (s)", "wall_s"),
            ("TPS (tok/s)", "tps"),
            ("gen tokens", "gen_tokens"),
            ("cache hits", "hits"),
            ("cache queries", "queries"),
            ("cached tokens", "cached_tokens"),
        ]
        hdr = f"{'metric':<16}|{'A original':>14}|{'B ss no-cache':>16}|{'C ss cache':>14}"
        print(hdr)
        print("-" * len(hdr))
        for label, key in rows:
            va = _fmt(a[key]) if a else "  n/a "
            vb = _fmt(b[key]) if b else "  n/a "
            vc = _fmt(c[key]) if c else "  n/a "
            print(f"{label:<16}|{va:>14}|{vb:>16}|{vc:>14}")

        # --- derived deltas ---
        print("derived:")
        if a and b:
            print(f"  A->B extra-store overhead  TPOT {_signed(_pct_delta(a['tpot_mean_s'], b['tpot_mean_s']))}"
                  f"  TTFT {_signed(_pct_delta(a['ttft_mean_s'], b['ttft_mean_s']))}")
        if b and c:
            print(f"  B->C prefix-cache benefit  TTFT {_signed(_pct_delta(b['ttft_mean_s'], c['ttft_mean_s']))}"
                  f"  wall {_signed(_pct_delta(b['wall_s'], c['wall_s']))}")
        if a and c:
            print(f"  A->C net                   TTFT {_signed(_pct_delta(a['ttft_mean_s'], c['ttft_mean_s']))}"
                  f"  wall {_signed(_pct_delta(a['wall_s'], c['wall_s']))}")
    print("\n" + "=" * 78)


def _signed(p):
    if p is None:
        return " n/a"
    return f"{p:+.1f}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true",
                        help="internal: run a single config (used by the orchestrator)")
    parser.add_argument("--config", choices=CONFIGS)
    parser.add_argument("--workload", choices=WORKLOADS)
    parser.add_argument("--out", type=str)
    parser.add_argument("--native", action="store_true",
                        help="gemma native window (W=512, block_size=512); no 1024 override")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--prefix-blocks", type=int, default=3,
                        help="shared-prefix size in blocks (W tokens each); the prefix "
                             "is built to exactly this many full blocks")
    parser.add_argument("--workload-only", choices=WORKLOADS, default=None,
                        help="orchestrate just one workload")
    parser.add_argument("--full-prompts", action="store_true",
                        help="print built prompts verbatim (default: head+tail truncated)")
    parser.add_argument("--profile", action="store_true",
                        help="collect a torch/Perfetto trace of the timed generate into "
                             "<scratch-dir>/profile/<config>_<workload>/ (view at "
                             "ui.perfetto.dev); inflates timings, so use for traces only")
    parser.add_argument("--profile-dir", type=str, default=None,
                        help="override the trace output dir (worker mode)")
    parser.add_argument("--scopes", action="store_true",
                        help="emit named host scopes (Forward/Preprocess/Sample/...) in "
                             "the torch trace (sets VLLM_CUSTOM_SCOPES_FOR_PROFILING=1); "
                             "pair with --profile")
    parser.add_argument("--npu-profile", action="store_true",
                        help="activate the RBLN device profiler (RBLN_PROFILER=1) for "
                             "true NPU-side timings; the torch profiler is CPU-only")
    parser.add_argument("--scratch-dir", type=str,
                        default=str(_HERE / "_swa_bench_out"))
    args = parser.parse_args()

    if args.worker:
        run_worker(args.config, args.workload, args.out, args.native, args.max_tokens,
                   args.prefix_blocks, show_full_prompts=args.full_prompts,
                   profile=args.profile, profile_dir=args.profile_dir,
                   scopes=args.scopes, npu_profile=args.npu_profile)
    else:
        workloads = (args.workload_only,) if args.workload_only else WORKLOADS
        orchestrate(args.native, args.max_tokens, args.prefix_blocks, workloads,
                    pathlib.Path(args.scratch_dir), profile=args.profile)


if __name__ == "__main__":
    main()
