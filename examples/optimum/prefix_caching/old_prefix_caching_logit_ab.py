# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Logit A/B for sub-block prefix caching.

Two engines from one config, differing only in enable_prefix_caching, run the
same prompts: the False engine gives the reference logits (and, run twice, the
noise floor), the True engine gives the signal. max_tokens=1 keeps every
request on an identical context, so greedy divergence is impossible and the
only difference is how the KV for the prompt was assembled.

The prefix is sized so the cache hit lands mid-block: with
max_num_batched_tokens == block_size, any off0 > 0 makes the chunk span two
blocks, which is the multi-block store path under test.

Env: PROBE_BLOCK_SIZE, PROBE_MAX_BATCHED, PROBE_PREFIX_TOKENS, PROBE_NUM_PROBES
"""

import os

os.environ.setdefault("VLLM_RBLN_USE_VLLM_MODEL", "1")

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402

MODEL = "meta-llama/Llama-3.2-1B"
BLOCK_SIZE = int(os.environ.get("PROBE_BLOCK_SIZE", 1024))
MAX_BATCHED = int(os.environ.get("PROBE_MAX_BATCHED", BLOCK_SIZE))
PREFIX_TOKENS = int(os.environ.get("PROBE_PREFIX_TOKENS", 3000))
NUM_PROBES = int(os.environ.get("PROBE_NUM_PROBES", 8))

SAMPLING_PARAMS = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)

TAILS = [
    "Question: what follows from the passage above? Answer:",
    "Summarise the argument in one sentence:",
    "List the three most important claims:",
    "Explain why the paradigms differ:",
    "Which assumption is the weakest, and why?",
    "Translate the key idea into plain language:",
    "Name one counterexample to the passage:",
    "What would falsify the claim above?",
]

FILLER = (
    "Artificial intelligence research has produced a long sequence of "
    "distinct paradigms, each with its own assumptions about representation "
    "and search. "
)


def make_llm(enable_prefix_caching: bool) -> LLM:
    return LLM(
        model=MODEL,
        block_size=BLOCK_SIZE,
        max_num_batched_tokens=MAX_BATCHED,
        max_model_len=8192,
        max_num_seqs=4,
        enable_prefix_caching=enable_prefix_caching,
        tensor_parallel_size=1,
    )


def analyze(a, b, n_show=10):
    """Per-rank comparison of two {token_id: Logprob} maps.

    Aligns two ways: by rank (did the ordering move?) and by token (did this
    token's value move?).

    A rank swap is judged against the local spacing: if |delta| is smaller than
    the gap to the next candidate, that delta could not have reordered anything.
    """
    ra = sorted(a, key=lambda t: a[t].rank if a[t].rank else 1 << 30)
    rb = sorted(b, key=lambda t: b[t].rank if b[t].rank else 1 << 30)

    rows = []
    for i in range(min(n_show, len(ra))):
        tok = ra[i]
        lp = a[tok].logprob
        gap = (lp - a[ra[i + 1]].logprob) if i + 1 < len(ra) else None
        in_b = tok in b
        d = (lp - b[tok].logprob) if in_b else None
        rows.append({
            "rank": i + 1,
            "a_tok": tok,
            "a_text": a[tok].decoded_token,
            "a_lp": lp,
            "b_tok": rb[i] if i < len(rb) else None,
            "b_lp": b[tok].logprob if in_b else None,
            "b_rank": b[tok].rank if in_b else None,
            "delta": d,
            "gap": gap,
            "ratio": (abs(d) / gap) if (d is not None and gap and gap > 0) else None,
            "swapped": i < len(rb) and ra[i] != rb[i],
        })

    deltas_top5 = [abs(r["delta"]) for r in rows[:5] if r["delta"] is not None]
    deltas_all  = [abs(r["delta"]) for r in rows     if r["delta"] is not None]
    margin_a = (a[ra[0]].logprob - a[ra[1]].logprob) if len(ra) > 1 else float("inf")
    margin_b = (b[rb[0]].logprob - b[rb[1]].logprob) if len(rb) > 1 else float("inf")

    return {
        "rows": rows,
        "top1_same": ra[0] == rb[0],
        "top1": (ra[0], rb[0]),
        "margin_a": margin_a,
        "margin_b": margin_b,
        "max_d_top5": max(deltas_top5, default=0.0),
        "max_d_all":  max(deltas_all,  default=0.0),
        "censored":   sum(1 for r in rows if r["delta"] is None),
        "swaps":      sum(1 for r in rows if r["swapped"]),
        "unexplained": [r["rank"] for r in rows if r["ratio"] is not None and r["ratio"] > 1.0],
    }


def show(label, m):
    print(f"\n--- {label} ---")
    print(
        "  rank  A token                 A logprob   B logprob      delta   "
        "d/gap  B rank"
    )
    for r in m["rows"]:
        tag   = f"{r['a_tok']} {r['a_text']!r}"[:22]
        b_lp  = "      --" if r["b_lp"]  is None else f"{r['b_lp']:9.4f}"
        d     = "  (censored)" if r["delta"] is None else f"{r['delta']:11.3e}"
        ratio = "    --" if r["ratio"] is None else f"{r['ratio']:6.2f}"
        b_rank = "  --" if r["b_rank"] is None else f"{r['b_rank']:4d}"
        mark  = " <-swap" if r["swapped"] else ""
        print(
            f"  {r['rank']:4d}  {tag:<22} {r['a_lp']:9.4f}  {b_lp}  {d}  "
            f"{ratio}  {b_rank}{mark}"
        )

    safety = m["margin_a"] / m["max_d_top5"] if m["max_d_top5"] > 0 else float("inf")
    print(
        f"  top-1: {'SAME' if m['top1_same'] else 'DIFFERENT'} {m['top1']}   "
        f"margin A={m['margin_a']:.4f} B={m['margin_b']:.4f}   "
        f"safety(margin/max|d|@top5)={safety:.1f}x"
    )
    print(
        f"  max|d|@top5={m['max_d_top5']:.3e}  max|d|@shown={m['max_d_all']:.3e}  "
        f"swaps={m['swaps']}  censored={m['censored']}  "
        f"unexplained(d/gap>1) at ranks={m['unexplained'] or 'none'}"
    )


def main():
    llm = make_llm(enable_prefix_caching=False)
    tok = llm.get_tokenizer()

    # Build prompts as token ids only — decode()/encode() round-trips are not
    # identity here (BOS gets doubled), so we never go through text.
    body = tok.encode(FILLER * (PREFIX_TOKENS // 20 + 8), add_special_tokens=False)
    prefix_ids = [tok.bos_token_id] + body[: PREFIX_TOKENS - 1]

    warm_ids = prefix_ids + tok.encode("\n\nSummary:", add_special_tokens=False)
    probe_ids = [
        prefix_ids + tok.encode("\n\n" + t, add_special_tokens=False)
        for t in TAILS[:NUM_PROBES]
    ]
    probes     = [TokensPrompt(prompt_token_ids=ids) for ids in probe_ids]
    probe_lens = [len(ids) for ids in probe_ids]

    # 1. Noise floor: each probe run twice on the no-cache engine.
    n_p = len(probes)
    ref  = llm.generate(probes + probes, SAMPLING_PARAMS)
    base  = [ref[i].outputs[0].logprobs[0] for i in range(n_p)]
    base2 = [ref[i].outputs[0].logprobs[0] for i in range(n_p, 2 * n_p)]
    assert max(getattr(o, "num_cached_tokens", 0) or 0 for o in ref) == 0, \
        "baseline engine reported a cache hit"

    # 2. Signal: same probes on the prefix-caching engine, after warm-up.
    del llm
    llm = make_llm(enable_prefix_caching=True)
    llm.generate([TokensPrompt(prompt_token_ids=warm_ids)], SAMPLING_PARAMS)
    outs   = llm.generate(probes, SAMPLING_PARAMS)
    cached = [o.outputs[0].logprobs[0] for o in outs]
    hits   = [getattr(o, "num_cached_tokens", 0) or 0 for o in outs]

    print(
        f"\n  warm={len(warm_ids)} tok  probes={probe_lens[0]}..{probe_lens[-1]} tok  "
        f"block={BLOCK_SIZE}  chunk={MAX_BATCHED}"
    )

    # Analysis
    mets     = [analyze(cached[i], base[i]) for i in range(n_p)]
    nz       = [analyze(base[i], base2[i])  for i in range(n_p)]
    safeties = [
        m["margin_a"] / m["max_d_top5"] if m["max_d_top5"] > 0 else float("inf")
        for m in mets
    ]
    worst_i  = min(range(n_p), key=lambda i: safeties[i])
    agree    = sum(bool(m["top1_same"]) for m in mets)

    # Per-probe table
    print("\nper-probe: hit geometry and cached-vs-nocache verdict")
    print(
        "   #    hit   off0  spill  top1      margin  max|d|@top5   safety  "
        "swaps  unexplained"
    )
    exercised = 0
    for i, m in enumerate(mets):
        off0  = hits[i] % BLOCK_SIZE
        chunk = min(MAX_BATCHED, probe_lens[i] - hits[i])
        spill = off0 + chunk > BLOCK_SIZE
        exercised += bool(off0)
        print(
            f"  {i:2d}  {hits[i]:5d}  {off0:5d}  {'yes' if spill else ' no':>5}  "
            f"{'same' if m['top1_same'] else 'DIFF':>4}  {m['margin_a']:10.4f}  "
            f"{m['max_d_top5']:11.3e}  {safeties[i]:6.1f}x  {m['swaps']:5d}  "
            f"{m['unexplained'] or 'none'}"
        )

    # Summary
    print(
        f"\nsummary: top-1 agree {agree}/{n_p}   "
        f"worst safety {safeties[worst_i]:.1f}x (probe {worst_i})"
    )
    print(
        f"  max|d|@top5   signal={max(m['max_d_top5'] for m in mets):.3e}   "
        f"noise={max((m['max_d_top5'] for m in nz), default=0.0):.3e}"
    )
    unexp = {i: m["unexplained"] for i, m in enumerate(mets) if m["unexplained"]}
    print(f"  unexplained swaps (d/gap>1): {unexp or 'none'}")

    if exercised == 0:
        print(
            "  WARNING: every probe has off0 == 0, so the recomputed chunk has the "
            "same shape and position in both arms and the KV is bit-identical by "
            "construction -- a zero delta here tests nothing.  Make the hit land "
            "mid-block (sub-block match, or a prefix that is not a multiple of "
            "the block size)."
        )

    for i, m in enumerate(mets):
        marker = "   <== worst safety" if i == worst_i else ""
        show(f"probe #{i}: cached vs nocache{marker}", m)
    print("=" * 78)


if __name__ == "__main__":
    main()