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

# ruff: noqa

import argparse
import hashlib
import json
import os
from functools import partial
from multiprocessing import get_context

import numpy as np
import torch

# ========= Tunables =========
EPSILON = 1e-1 * 5
PRINT_VECT_SNIPPET = True  # show a short head/tail snippet under "just print logits"
SNIPPET_ELEMS = 6  # how many head/tail elements to show (total = SNIPPET_ELEMS*2)
# ============================

DEFAULT_PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# ---------- CLI ----------
parser = argparse.ArgumentParser(
    description="CPU vs RBLN parity runner (separate processes; clean envs)."
)
parser.add_argument(
    "--model",
    type=str,
    default="llama3.2-1b",
    choices=[
        "llama3.2-1b",
        "llama3-8b",
        "qwen3-1.7b",
        "qwen1.5-moe-15b",
        "qwen3-moe-30b",
        "qwen3-moe-235b",
        "deepseek-v2",
        "llama4-maverick",
    ],
)
parser.add_argument("--batch", type=int, default=1, help="Batch size.")
parser.add_argument("--tp", type=int, default=1)
parser.add_argument("--pp", type=int, default=1)
parser.add_argument("--dp", type=int, default=1)
parser.add_argument("--ep", action="store_true")
parser.add_argument("--profile", action="store_true")
parser.add_argument("--max-model-len", type=int, default=40 * 1024)
parser.add_argument("--block-size-cpu", type=int, default=128)
parser.add_argument("--block-size-rbln", type=int, default=8 * 1024)
parser.add_argument("--max-batched", type=int, default=128)
parser.add_argument(
    "--prompts",
    type=str,
    nargs="*",
    default=None,
    help="Explicit list of prompts. If omitted, DEFAULT_PROMPTS are used/cycled.",
)
parser.add_argument(
    "--num-prompts",
    type=int,
    default=None,
    help="Number of prompts to use (cycles DEFAULT_PROMPTS if not enough). "
    "If --prompts is provided, trims to first N.",
)
parser.add_argument("--trust-remote-code", action="store_true")
parser.add_argument(
    "--use-cache", action="store_true", help="Use cached CPU results if available"
)
parser.add_argument(
    "--num-hidden-layers",
    type=int,
    default=None,
    help="Override model hidden layer count",
)
parser.add_argument(
    "--max-tokens",
    type=int,
    default=256,
    help="Number of tokens to generate per prompt",
)
parser.add_argument(
    "--logprobs",
    type=int,
    default=1024,
    help="Per-token logprobs: 0=off; N=top-N; -1=full vocab",
)
parser.add_argument(
    "--max-logprobs-cap",
    type=int,
    default=128256,
    help="Engine-wide cap for logprobs; must be >= requested logprobs",
)

# Visualization toggles
parser.add_argument(
    "--inspect-logits",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Enable/disable logits-like inspection (default: on). "
    "Use --no-inspect-logits to disable.",
)
parser.add_argument(
    "--topk",
    type=int,
    default=5,
    help="Top-K for argmax summary when --inspect-logits is on.",
)
parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors.")
parser.add_argument(
    "--no-snippet", action="store_true", help="Hide the head/tail logits snippets."
)
parser.add_argument(
    "--snippet-elems",
    type=int,
    default=SNIPPET_ELEMS,
    help="How many elems to show in head/tail snippet when enabled.",
)

args = parser.parse_args()


# ANSI color helpers
class C:
    if args.no_color:
        R = G = Y = B = M = C = W = BOLD = DIM = RESET = ""
    else:
        R = "\x1b[31m"
        G = "\x1b[32m"
        Y = "\x1b[33m"
        B = "\x1b[34m"
        M = "\x1b[35m"
        C_ = "\x1b[36m"
        W = "\x1b[37m"
        BOLD = "\x1b[1m"
        DIM = "\x1b[2m"
        RESET = "\x1b[0m"


C.C = C.C_  # alias

# ---------- Model map ----------
MODELS = {
    "llama3.2-1b": ("meta-llama/Llama-3.2-1B", False),
    "llama3-8b": ("meta-llama/Meta-Llama-3-8B", False),
    "qwen3-1.7b": ("Qwen/Qwen3-1.7B", False),
    "qwen1.5-moe-15b": ("Qwen/Qwen1.5-MoE-A2.7B", True),
    "qwen3-moe-30b": ("Qwen/Qwen3-30B-A3B", True),
    "qwen3-moe-235b": ("Qwen/Qwen3-235B-A22B", True),
    "deepseek-v2": ("deepseek-ai/DeepSeek-V2-Lite", True),
    "llama4-maverick": ("meta-llama/Llama-4-Maverick-17B-128E", True),
}

assert args.model in MODELS, "Invalid model name"
model_id, should_ep = MODELS[args.model]
if should_ep:
    assert args.ep, f"{args.model} requires --ep"
else:
    assert not args.ep, f"{args.model} should not use --ep"


# ---------- Prompt resolution ----------
def resolve_prompts():
    if args.prompts:
        if args.num_prompts is not None:
            return args.prompts[: args.num_prompts]
        return args.prompts
    target_n = (
        args.num_prompts if args.num_prompts is not None else len(DEFAULT_PROMPTS)
    )
    if target_n <= len(DEFAULT_PROMPTS):
        return DEFAULT_PROMPTS[:target_n]
    reps = (target_n + len(DEFAULT_PROMPTS) - 1) // len(DEFAULT_PROMPTS)
    return (DEFAULT_PROMPTS * reps)[:target_n]


prompts = resolve_prompts()
print(
    f"Model = {model_id}, EP={args.ep}, TP={args.tp}, PP={args.pp}, DP={args.dp}, "
    f"MaxTokens={args.max_tokens}, Logprobs={args.logprobs}, Prompts={len(prompts)}"
)


# ---------- HF override (picklable) ----------
def hf_override_num_layers(hf_config, num_hidden_layers: int):
    from vllm.transformers_utils.config import get_hf_text_config

    if hasattr(hf_config, "text_config"):
        txt = get_hf_text_config(hf_config)
        txt.update({"num_hidden_layers": num_hidden_layers})
    else:
        hf_config.update({"num_hidden_layers": num_hidden_layers})
    return hf_config


# ---------- Helpers ----------
def generate_llm_args(device: str):
    llm_args = {
        "model": model_id,
        "max_model_len": args.max_model_len,
        "enable_chunked_prefill": True,
        "max_num_seqs": args.batch,
        "max_num_batched_tokens": args.max_batched,
        "trust_remote_code": args.trust_remote_code,
        "max_logprobs": args.max_logprobs_cap,
    }
    if args.profile:
        profile_dir = f"./profile/{device}_{model_id.replace('/', '_')}"
        os.makedirs(profile_dir, exist_ok=True)
        llm_args["profiler_config"] = {
            "profiler": "torch",
            "torch_profiler_dir": profile_dir,
        }
    if device == "rbln":
        llm_args.update(
            {
                "tensor_parallel_size": args.tp,
                "pipeline_parallel_size": args.pp,
                "data_parallel_size": args.dp,
                "enable_expert_parallel": args.ep,
            }
        )
    else:
        llm_args.update(
            {
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "data_parallel_size": 1,
                "enable_expert_parallel": False,
            }
        )
    if args.num_hidden_layers is not None:
        llm_args["hf_overrides"] = partial(
            hf_override_num_layers, num_hidden_layers=args.num_hidden_layers
        )
    llm_args["block_size"] = (
        args.block_size_cpu if device == "cpu" else args.block_size_rbln
    )
    return llm_args


def set_env_for_device(device: str):
    for k in [
        "VLLM_PLUGINS",
        "RBLN_KERNEL_MODE",
        "USE_VLLM_MODEL",
        "VLLM_DISABLE_COMPILE_CACHE",
    ]:
        os.environ.pop(k, None)

    if device == "cpu":
        os.environ["VLLM_PLUGINS"] = "cpu"
    elif device == "rbln":
        os.environ["RBLN_KERNEL_MODE"] = "triton"
        os.environ["USE_VLLM_MODEL"] = "1"
        os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    else:
        raise ValueError(f"Unknown device: {device}")

    if args.profile:
        profile_dir = f"./profile/{device}_{model_id.replace('/', '_')}"
        os.makedirs(profile_dir, exist_ok=True)
        os.environ["VLLM_TORCH_PROFILER_DIR"] = profile_dir


def cache_path():
    os.makedirs("./cache", exist_ok=True)
    key_parts = [
        args.model.replace("/", "_"),
        f"L{args.num_hidden_layers}" if args.num_hidden_layers else "",
        f"T{args.max_tokens}",
    ]

    if args.logprobs == -1:
        key_parts.append("LPvocab")
    elif args.logprobs > 0:
        key_parts.append(f"LP{args.logprobs}")

    key_parts.append(f"P{len(prompts)}")
    joined_prompts = "\n".join(prompts)
    prompt_hash = hashlib.md5(joined_prompts.encode("utf-8")).hexdigest()[:8]
    key_parts.append(f"H{prompt_hash}")

    key = "_".join(filter(None, key_parts))
    return f"./cache/cpu_results_{key}.json"


def _extract_logprobs_dict(request_output) -> dict:
    """Return {token_id(str): logprob(float)} for position 0 if available."""
    lp = getattr(request_output.outputs[0], "logprobs", None)
    if isinstance(lp, list) and lp and isinstance(lp[0], dict):
        try:
            return {str(k): v.logprob for k, v in lp[0].items()}
        except Exception:
            return {}
    return {}


def _pack_outputs(outputs):
    """Convert vLLM RequestOutput[] -> serializable list of dicts."""
    packed = []
    for o in outputs:
        packed.append(
            {
                "prompt": o.prompt,
                "text": o.outputs[0].text,
                "logprobs": _extract_logprobs_dict(o),
            }
        )
    return packed


def save_cpu_results(packed_outputs):
    path = cache_path()
    with open(path, "w") as f:
        json.dump(packed_outputs, f, indent=2)
    print(f"✅ Cached CPU results to {path}")


def load_cpu_results():
    path = cache_path()
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    print(f"💾 Loaded cached CPU results from {path}")
    return data


# ======== Logits-style inspection helpers ========
def _vocab_len_from_logprob_maps(*maps: dict) -> int:
    max_id = -1
    for m in maps:
        for k in m.keys():
            try:
                tid = int(k)
                if tid > max_id:
                    max_id = tid
            except Exception:
                continue
    return max_id + 1 if max_id >= 0 else 0


def _vectorize_logprobs(cpu_map: dict, rbln_map: dict, vocab_hint: int = None):
    V = vocab_hint or max(_vocab_len_from_logprob_maps(cpu_map, rbln_map), 0)
    if V == 0:
        return np.array([]), np.array([])

    cpu_vec = np.full((V,), -np.inf, dtype=np.float64)
    rbln_vec = np.full((V,), -np.inf, dtype=np.float64)

    for k, v in cpu_map.items():
        try:
            cpu_vec[int(k)] = float(v)
        except Exception:
            pass
    for k, v in rbln_map.items():
        try:
            rbln_vec[int(k)] = float(v)
        except Exception:
            pass
    return cpu_vec, rbln_vec


def _pearson_safe(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    if np.all(~np.isfinite(a)) or np.all(~np.isfinite(b)):
        return float("nan")
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return float("nan")
    a_m = a[mask]
    b_m = b[mask]
    if np.std(a_m) == 0 or np.std(b_m) == 0:
        return float("nan")
    return float(np.corrcoef(a_m, b_m)[0, 1])


def _topk_indices_values(vec: np.ndarray, k: int):
    if vec.size == 0:
        return np.array([], dtype=int), np.array([])
    k = min(k, vec.size)
    idx = np.argpartition(-vec, k - 1)[:k]
    order = np.argsort(-vec[idx])
    idx = idx[order]
    vals = vec[idx]
    return idx, vals


def _fmt_snippet(vec: np.ndarray, elems: int) -> str:
    if vec.size == 0:
        return "[]"
    if vec.size <= elems * 2:
        return np.array2string(
            vec, precision=4, suppress_small=False, max_line_width=10**9
        )
    head = vec[:elems]
    tail = vec[-elems:]
    h = np.array2string(head, precision=4, max_line_width=10**9)
    t = np.array2string(tail, precision=4, max_line_width=10**9)
    return f"{h[:-1]}, ..., {t[1:]}"


def _color_by_value(val: float, good=0.999, warn=0.99) -> str:
    if np.isnan(val):
        return f"{C.Y}{val}{C.RESET}"
    if val >= good:
        return f"{C.G}{val:.6f}{C.RESET}"
    if val >= warn:
        return f"{C.Y}{val:.6f}{C.RESET}"
    return f"{C.R}{val:.6f}{C.RESET}"


def _hdr(title: str):
    bar = f"{C.DIM}{'─' * 100}{C.RESET}"
    print(f"\n{bar}")
    print(f"{C.BOLD}{title}{C.RESET}")
    print(bar)


def _kv(key: str, val: str, pad=18):
    print(f"{C.DIM}{key.rjust(pad)}{C.RESET}: {val}")


def _topk_table(r_idx, r_vals, c_idx, c_vals, k):
    print(f"{C.DIM}Top-{k} (logprob) argmax — RBLN vs GOLD{C.RESET}")
    print(f"{C.DIM}{'-' * 56}{C.RESET}")
    print(f"{'Rank':>4}  {'R.idx':>8} {'R.val':>10}    {'G.idx':>8} {'G.val':>10}")
    rows = max(len(r_idx), len(c_idx))
    for i in range(rows):
        ri = r_idx[i] if i < len(r_idx) else -1
        rv = r_vals[i] if i < len(r_vals) else float("nan")
        ci = c_idx[i] if i < len(c_idx) else -1
        cv = c_vals[i] if i < len(c_vals) else float("nan")
        print(f"{i + 1:>4}  {ri:>8} {rv:>10.4f}    {ci:>8} {cv:>10.4f}")


# ----------------- Compare & print -----------------
def compare_and_print(cpu_packed, rbln_packed):
    need_logits = args.inspect_logits
    if need_logits and args.logprobs != -1:
        print(
            f"{C.Y}⚠️  --inspect-logits works best with --logprobs -1 (full vocab). Continuing…{C.RESET}"
        )

    for i, (cpu_out, rbln_out) in enumerate(zip(cpu_packed, rbln_packed)):
        prompt = cpu_out.get("prompt", "<unknown>")
        cpu_text = cpu_out.get("text", "") or ""
        rbln_text = rbln_out.get("text", "") or ""
        cpu_lp = cpu_out.get("logprobs", {}) or {}
        rbln_lp = rbln_out.get("logprobs", {}) or {}

        # Outlier count from original logic
        num_outlier = 0
        for token_id_str, cpu_lp_val in cpu_lp.items():
            try:
                token_id = int(token_id_str)
            except ValueError:
                continue
            r_val = rbln_lp.get(str(token_id))
            if r_val is None and isinstance(next(iter(rbln_lp.values()), None), float):
                r_val = rbln_lp.get(token_id)
            if r_val is None:
                continue
            if abs(float(cpu_lp_val) - float(r_val)) >= EPSILON:
                num_outlier += 1

        # Header
        _hdr(f"Prompt[{i}]: {prompt}")
        _kv("Generated text (CPU)", f"len={len(cpu_text)}")
        _kv("Generated text (RBLN)", f"len={len(rbln_text)}")
        _kv("Outliers (absΔ ≥ EPS)", f"{num_outlier}  {C.DIM}(EPS={EPSILON}){C.RESET}")

        if not need_logits:
            continue

        # Build aligned vectors (logits-like = logprobs)
        c_vec, r_vec = _vectorize_logprobs(cpu_lp, rbln_lp, vocab_hint=None)

        # Metrics (finite overlap only)
        r_t = torch.tensor(r_vec, dtype=torch.float64)
        c_t = torch.tensor(c_vec, dtype=torch.float64)
        finite = torch.isfinite(r_t) & torch.isfinite(c_t)
        overlap = int(finite.sum().item())
        if overlap > 0:
            diffs = torch.abs(r_t[finite] - c_t[finite])
            max_l1 = float(torch.max(diffs).item())
            mean_l1 = float(torch.mean(diffs).item())
        else:
            max_l1 = float("nan")
            mean_l1 = float("nan")
        pear = _pearson_safe(r_vec, c_vec)

        # Summary line
        vocab = max(len(r_vec), len(c_vec))
        _kv("Vocab size", f"{vocab}")
        _kv("Finite overlap", f"{overlap}")
        _kv("max|Δ|", f"{max_l1:.6f}" if np.isfinite(max_l1) else str(max_l1))
        _kv("mean|Δ|", f"{mean_l1:.6f}" if np.isfinite(mean_l1) else str(mean_l1))
        _kv("pearson", _color_by_value(pear))

        # Optional vector snippets instead of huge dumps
        if PRINT_VECT_SNIPPET and not args.no_snippet:
            se = max(1, args.snippet_elems)
            print(f"\n{C.DIM}Logits-like (logprob) snippet — head…tail{C.RESET}")
            print(f"  rbln  : {_fmt_snippet(r_vec, se)}")
            print(f"  golden: {_fmt_snippet(c_vec, se)}")

        # Top-K table
        r_idx, r_vals = _topk_indices_values(r_vec, args.topk)
        c_idx, c_vals = _topk_indices_values(c_vec, args.topk)
        print()
        _topk_table(r_idx, r_vals, c_idx, c_vals, args.topk)


# ----------------- Worker -----------------
def _worker(device: str, q, prompts_local, logprobs_flag, max_tokens_local):
    set_env_for_device(device)
    from vllm import LLM, SamplingParams

    llm_args = generate_llm_args(device)
    llm = LLM(**llm_args)

    # Warmup (compile)
    llm.generate(".", SamplingParams(temperature=0.0, max_tokens=2))

    # Resolve logprobs count
    if logprobs_flag == 0:
        lp_count = None
    elif logprobs_flag == -1:
        lp_count = None
        try:
            tok = llm.get_tokenizer()
            lp_count = getattr(tok, "vocab_size", None) or len(tok)
        except Exception:
            lp_count = None
    else:
        lp_count = logprobs_flag

    sampling_params = SamplingParams(
        temperature=0.0,
        ignore_eos=True,
        max_tokens=max_tokens_local,
        logprobs=lp_count,
    )

    print(f"[{device}] VLLM_PLUGINS = {os.environ.get('VLLM_PLUGINS', '<unset>')}")
    outputs = llm.generate(prompts_local, sampling_params)
    q.put(_pack_outputs(outputs))


# ----------------- Main -----------------
def main():
    # CPU phase
    if args.use_cache:
        cpu_packed = load_cpu_results()
        if cpu_packed is None:
            ctx = get_context("spawn")
            q = ctx.Queue()
            print("\n[main] Launching CPU worker…")
            p1 = ctx.Process(
                target=_worker, args=("cpu", q, prompts, args.logprobs, args.max_tokens)
            )
            p1.start()
            cpu_packed = q.get()
            p1.join()
            if p1.exitcode != 0:
                raise SystemExit("CPU process failed.")
            save_cpu_results(cpu_packed)
    else:
        ctx = get_context("spawn")
        q = ctx.Queue()
        print("\n[main] Launching CPU worker…")
        p1 = ctx.Process(
            target=_worker, args=("cpu", q, prompts, args.logprobs, args.max_tokens)
        )
        p1.start()
        cpu_packed = q.get()
        p1.join()
        if p1.exitcode != 0:
            raise SystemExit("CPU process failed.")
        save_cpu_results(cpu_packed)

    # RBLN phase
    ctx = get_context("spawn")
    q = ctx.Queue()
    print("\n[main] Launching RBLN worker…")
    p2 = ctx.Process(
        target=_worker, args=("rbln", q, prompts, args.logprobs, args.max_tokens)
    )
    p2.start()
    rbln_packed = q.get()
    p2.join()
    if p2.exitcode != 0:
        raise SystemExit("RBLN process failed.")

    # Compare
    compare_and_print(cpu_packed, rbln_packed)


if __name__ == "__main__":
    main()
