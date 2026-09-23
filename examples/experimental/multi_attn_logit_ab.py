# SPDX-License-Identifier: Apache-2.0
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

"""Logit A/B/C for the sliding-window kernel and the cache view it is handed.

  0  RBLN_USE_MULTI_ATTN=0: RBLNSlidingWindowSpec, the shift kernel
  1  RBLN_USE_MULTI_ATTN=1: sliding_window_attention_v1 on window-sized blocks,
     KV (..., sliding_window, D)
  2  1 plus RBLN_SWA_FULL_BLOCK=1: the same kernel on the manager block,
     KV (..., BLOCK_SIZE, D)

Arm 0 is the reference every other arm is compared against.  Prefix caching is
off throughout, so the arms differ only in the SWA kernel and its cache view.
Each arm runs in its own process with its own VLLM_CACHE_ROOT: the env vars
are read when the KV cache is built, and they are not part of the mega-cache
key, so a shared cache could hand one arm another's compiled graphs.

Each prompt is the passage, cycled to length, and then one question about it,
so the arms decode real sentences instead of repeating filler.

Per request:

  pos0    max|d| against arm 0 at the first generated position, the prefill
  decode  max|d| against arm 0 over later positions, up to the first divergence
  div     index of the first generated token that differs from arm 0, '-' if none
  noise   max|d| between two runs of the same arm, over all positions

Usage:
  python multi_attn_logit_ab.py              # run both arms, then compare
  python multi_attn_logit_ab.py --skip-run   # compare the saved results
"""

import argparse
import json
import os
import subprocess
import sys
from itertools import cycle, islice

MODEL = "openai/gpt-oss-20b"
BLOCK_SIZE = 4096
MAX_BATCHED = 512
MAX_NUM_SEQS = 1
TP = 1
PROMPT_TOKENS = 7000
MAX_TOKENS = 32
TOPK = 10

HERE = os.path.dirname(os.path.abspath(__file__))
ARM_ENV = {
    0: {"RBLN_USE_MULTI_ATTN": "0", "RBLN_SWA_FULL_BLOCK": "0"},
    1: {"RBLN_USE_MULTI_ATTN": "1", "RBLN_SWA_FULL_BLOCK": "0"},
    2: {"RBLN_USE_MULTI_ATTN": "1", "RBLN_SWA_FULL_BLOCK": "1"},
}
ARMS = tuple(ARM_ENV)
OTHERS = ARMS[1:]

SYSTEM = (
    "You are a meticulous research assistant. Read the passage that follows and "
    "answer only from it. Quote exact wording where it matters, say plainly when "
    "the passage does not settle a question, and never introduce facts it does "
    "not contain. Keep answers short unless the question asks you to expand."
    "\n\n"
)

# Long and varied on purpose: a short string cycled hundreds of times puts the
# model in a degenerate state where the top token sits at p ~ 0.999, and the
# top-1 column stops being informative.
PASSAGE = """Symbolic artificial intelligence dominated the field for its first
three decades. Programs manipulated discrete structures -- lists, trees, logical
formulae -- and researchers described intelligence as search through a space of
such structures. The General Problem Solver, written in the late nineteen
fifties, made the claim explicit: give the machine a goal, a set of operators,
and a way to measure distance to the goal, and it would find its way there.

The approach worked wherever the space could be written down. Chess, theorem
proving, and blocks-world planning all yielded to it. Perception did not. A
program that could prove a theorem in propositional logic could not reliably
tell a cup from a bowl, and the reason was not a shortage of computing power.
The knowledge needed to see was not the kind anyone knew how to write as rules.

Statistical methods reversed the emphasis. Rather than specifying the structure,
a model estimated parameters from examples and tolerated ambiguity by assigning
probabilities instead of truth values. Speech recognition moved first, then
machine translation, then vision. What made the shift possible was not a single
algorithm but a change in what counted as an answer: a ranked list of hypotheses
with scores, rather than one derivation with a proof.

Neural networks began as one statistical family among several and ended as the
dominant one. Their advantage was compositional. A layer that learned edges
could feed a layer that learned corners, and nobody had to decide in advance
what an edge was. Depth turned out to matter more than any particular choice of
nonlinearity, and hardware that multiplied matrices quickly turned out to matter
more than depth.

Each paradigm inherited the previous one's unsolved problems under a new name.
Symbolic systems were brittle at the edges of their rules; statistical systems
are brittle at the edges of their training distribution, and the two failures
are closer than the vocabulary suggests. What changed is the cost of being wrong
in a new way. A rule can be read and edited; a weight cannot. So the question of
why a system produced the output it did has moved from a debugging concern to a
research programme of its own.
"""


# One request per question; the questions alone make the prompts differ, and
# each ends inside the last window, where both kernels read it.
QUESTIONS = (
    # "What did the General Problem Solver make explicit?",
    "Why could symbolic programs not tell a cup from a bowl?",
    # "Which field moved first to statistical methods?",
    # "What counted as an answer after the statistical shift?",
    # "What was the advantage of neural networks?",
    # "What turned out to matter more than depth?",
    # "How are the failures of symbolic and statistical systems alike?",
    # "Why has explaining a system's output become a research programme?",
)


def result_path(arm: int) -> str:
    return os.path.join(HERE, f"multi_attn_{arm}.json")


def build_prompts(tok) -> list[list[int]]:
    """One prompt of PROMPT_TOKENS ids per question.

    Built from ids, never round-tripped through text: decode() re-emits BOS as
    a literal string, which the next encode() would prefix with a second BOS.
    """

    def enc(text):
        return tok.encode(text, add_special_tokens=False)

    def fill(ids, n):
        """The last n ids of `ids` cycled, so the passage ends whole."""
        return list(islice(cycle(ids), -n % len(ids), -n % len(ids) + n))

    head, body = enc(SYSTEM), enc(PASSAGE)
    prompts = []
    for q in QUESTIONS:
        ask = enc(f"\nQuestion: {q}\nAnswer:")
        n_body = PROMPT_TOKENS - 1 - len(head) - len(ask)
        prompts.append([tok.bos_token_id] + head + fill(body, n_body) + ask)
    assert all(len(ids) == PROMPT_TOKENS for ids in prompts)
    return prompts


def run_arm(arm: int) -> None:
    """Child process: generate every request twice and save the logprobs."""
    for name, value in ARM_ENV[arm].items():
        assert os.environ.get(name) == value, (name, os.environ.get(name))

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(
        model=MODEL,
        # patches/ applies only on the vllm model path.
        model_impl="vllm",
        block_size=BLOCK_SIZE,
        max_num_batched_tokens=MAX_BATCHED,
        max_model_len=8192,
        max_num_seqs=MAX_NUM_SEQS,
        enable_prefix_caching=False,
        tensor_parallel_size=TP,
    )
    prompts = [
        TokensPrompt(prompt_token_ids=ids) for ids in build_prompts(llm.get_tokenizer())
    ]
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, logprobs=TOPK)
    outs = llm.generate(prompts + prompts, sp)

    runs = []
    for out in outs:
        o = out.outputs[0]
        runs.append(
            {
                "text": o.text,
                "token_ids": list(o.token_ids),
                "logprobs": [
                    {str(t): [lp.logprob, lp.decoded_token] for t, lp in pos.items()}
                    for pos in o.logprobs
                ],
            }
        )
    n = len(prompts)
    with open(result_path(arm), "w") as f:
        json.dump({"first": runs[:n], "second": runs[n:]}, f)


def spawn(arm: int) -> None:
    env = dict(os.environ)
    env.update(ARM_ENV[arm])
    # A size with prefix caching off is refused by sub_block_size_in_use(), and
    # before #1158 the flag alone refuses a multi-group KV cache.
    env.pop("VLLM_RBLN_SUB_BLOCK_SIZE", None)
    env["VLLM_RBLN_SUB_BLOCK_CACHE"] = "0"
    root = env.get("VLLM_CACHE_ROOT", os.path.expanduser("~/.cache/vllm"))
    env["VLLM_CACHE_ROOT"] = os.path.join(root, f"multi_attn_{arm}")
    subprocess.run([sys.executable, __file__, "--arm", str(arm)], env=env, check=True)


def max_delta(a: dict, b: dict) -> float:
    """Largest logprob move among the tokens both positions report."""
    return max((abs(a[t][0] - b[t][0]) for t in a if t in b), default=0.0)


def first_divergence(a: list[int], b: list[int]) -> int | None:
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)


def comparable(r1: dict, r2: dict) -> int:
    """Positions two runs share: up to and including their first divergence.

    Past it the two condition on different tokens.
    """
    div = first_divergence(r1["token_ids"], r2["token_ids"])
    return min(len(r1["logprobs"]), len(r2["logprobs"])) if div is None else div + 1


def step_deltas(r1: dict, r2: dict) -> list[float]:
    end = comparable(r1, r2)
    return [max_delta(x, y) for x, y in zip(r1["logprobs"][:end], r2["logprobs"][:end])]


def dump(i: int, step: int, runs: dict, deltas: dict) -> None:
    """Top-TOPK logprobs at one position for every arm, arm 0's ranking.

    An arm already past its divergence from arm 0 prints blank.
    """
    ref = runs[0]["logprobs"][step]
    live = [arm for arm in OTHERS if step < len(deltas[arm])]
    picked = []
    for arm in ARMS:
        if arm == 0 or arm in live:
            tid = str(runs[arm]["token_ids"][step])
            picked.append(f"{arm}:{runs[arm]['logprobs'][step][tid][1]!r}")
    moved = "  ".join(f"max|d|{arm}={deltas[arm][step]:.3e}" for arm in live)
    print(f"\n--- request #{i}  step {step}  token {' '.join(picked)}  {moved} ---")
    print(
        "  rank  token                  arm 0"
        + "".join(f"      arm {arm}    delta {arm}" for arm in OTHERS)
    )
    for rank, t in enumerate(sorted(ref, key=lambda t: -ref[t][0]), 1):
        a, text = ref[t]
        row = f"  {rank:4d}  {f'{t} {text!r}'[:20]:<20} {a:9.4f}"
        for arm in OTHERS:
            other = runs[arm]["logprobs"][step] if arm in live else {}
            if t in other:
                b = other[t][0]
                row += f"  {b:9.4f}  {a - b:10.3e}"
            else:
                row += f"  {'--':>9}  {'--':>10}"
        print(row)


def compare() -> None:
    res = {}
    for arm in ARMS:
        with open(result_path(arm)) as f:
            res[arm] = json.load(f)
    n = len(res[0]["first"])
    runs = [{arm: res[arm]["first"][i] for arm in ARMS} for i in range(n)]
    deltas = [{arm: step_deltas(r[0], r[arm]) for arm in OTHERS} for r in runs]

    print(
        f"\nblock={BLOCK_SIZE}  chunk={MAX_BATCHED}  tp={TP}  "
        f"prompt={PROMPT_TOKENS} tok  max_tokens={MAX_TOKENS}"
    )
    print(
        "\n   # arm   top1       pos0     decode   div      noise"
        "   (against arm 0; arm 0's noise on its own row)"
    )
    for i, r in enumerate(runs):
        noise0 = max(step_deltas(r[0], res[0]["second"][i]))
        print(f"  {i:2d}   0  {'':>5}  {'':>9}  {'':>9}  {'':>4}  {noise0:9.3e}")
        for arm in OTHERS:
            d = deltas[i][arm]
            div = first_divergence(r[0]["token_ids"], r[arm]["token_ids"])
            same = r[0]["token_ids"][0] == r[arm]["token_ids"][0]
            decode = f"{max(d[1:]):9.3e}" if len(d) > 1 else f"{'--':>9}"
            noise = max(step_deltas(r[arm], res[arm]["second"][i]))
            print(
                f"  {'':2}   {arm}  {'same' if same else 'DIFF':>5}  {d[0]:9.3e}  "
                f"{decode}  {'-' if div is None else div:>4}  {noise:9.3e}"
            )

    # Step 0 is the prefill; each later step is one decode.  A row stops at the
    # first divergence from arm 0.
    print("\nmax|d| against arm 0 per step")
    print("   # arm  " + " ".join(f"{k:5d}" for k in range(MAX_TOKENS)))
    for i, d in enumerate(deltas):
        for arm in OTHERS:
            label = f"{i:2d}" if arm == OTHERS[0] else "  "
            print(f"  {label}   {arm}  " + " ".join(f"{x:5.2f}" for x in d[arm]))

    for i, r in enumerate(runs):
        print(f"\n=== request #{i}  {QUESTIONS[i]}")
        for arm in ARMS:
            print(f"  arm {arm}: {r[arm]['text']!r}")
        for step in range(max(len(d) for d in deltas[i].values())):
            dump(i, step, r, deltas[i])


def main():
    os.environ.setdefault("VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS", "60")
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=int, choices=ARMS)
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args()

    if args.arm is not None:
        run_arm(args.arm)
        return
    if not args.skip_run:
        for arm in ARMS:
            spawn(arm)
    compare()


if __name__ == "__main__":
    main()
