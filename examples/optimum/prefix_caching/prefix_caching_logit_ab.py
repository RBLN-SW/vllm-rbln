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

"""Logit A/B for sub-block prefix caching.

Two engines from one config, differing only in enable_prefix_caching, run the
same requests.  max_tokens=1 keeps them all on an identical context, so
greedy divergence is impossible and the only thing that can move the logits is
how the KV for the prompt was assembled.

Four numbers per request:

  hit     prompt tokens served from the cache
  off0    where the recomputed tail starts inside its block (hit % BLOCK_SIZE)
  spill   the recomputed chunk crosses a block boundary, which is the
          multi-block store path under test.  Set by SPILL; 0 makes the chunk
          end exactly on the boundary, which is the control.
  max|d|  largest logprob move against the no-cache reference, readable only
          next to `noise` -- the same measurement between two runs on the same
          engine.  noise == 0 and max|d| == 0 together mean the assembled KV is
          bit-exact; a max|d| above `noise` is a real difference.
"""

import os

os.environ.setdefault("VLLM_RBLN_USE_VLLM_MODEL", "1")

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402

MODEL = "meta-llama/Llama-3.2-1B"
BLOCK_SIZE = 1024
MAX_BATCHED = 1024
SUB_BLOCK = 128  # must match VLLM_RBLN_SUB_BLOCK_SIZE
TP = 1
NUM_REQUESTS = 8

# The two knobs of the experiment.
FULL_BLOCKS = 2  # full blocks before the sub-block hit, so the copy's destination
                 # block index is not 0 and an off-by-one there cannot hide
SPILL = 4  # tokens the recomputed chunk pushes into the NEXT block; 0 = no spill

# Both lengths follow from those, so there is one number per effect and no slack:
# the hit is FULL_BLOCKS full blocks plus 7 of the 8 sub-blocks of the next one,
# which is the largest hit that still leaves room inside a block.  The prefix is
# exactly the hit, so everything past it is recomputed and chunk is exactly the
# rest of that block plus SPILL.
PREFIX_TOKENS = FULL_BLOCKS * BLOCK_SIZE + (BLOCK_SIZE - SUB_BLOCK)
PROMPT_TOKENS = PREFIX_TOKENS + SUB_BLOCK + SPILL
assert SUB_BLOCK + SPILL <= MAX_BATCHED, "chunk would be split across steps"


def make_llm(enable_prefix_caching: bool) -> LLM:
    return LLM(
        model=MODEL,
        block_size=BLOCK_SIZE,
        max_num_batched_tokens=MAX_BATCHED,
        max_model_len=8192,
        max_num_seqs=4,
        enable_prefix_caching=enable_prefix_caching,
        tensor_parallel_size=TP,
    )


TOPK = 10


def sp() -> SamplingParams:
    return SamplingParams(temperature=0.0, max_tokens=1, logprobs=TOPK)


def logits_of(out):
    """{token_id: Logprob} for the single generated position."""
    lp = out.outputs[0].logprobs
    assert lp, "no logprobs returned -- is logprobs=N set?"
    return lp[0]


def hit_of(out) -> int:
    """Prompt tokens served from the prefix cache, 0 on engines without it."""
    return getattr(out, "num_cached_tokens", 0) or 0


def top1(d) -> int:
    return max(d, key=lambda t: d[t].logprob)


def max_delta(a, b) -> float:
    """Largest logprob move among the tokens both maps report."""
    return max((abs(a[t].logprob - b[t].logprob) for t in a if t in b), default=0.0)


def dump(i, ref, cached, geometry):
    """Side-by-side top-TOPK logprobs, ordered by the no-cache ranking.

    A token in one arm's top-TOPK need not be in the other's, so a missing
    counterpart prints as `--` rather than being silently dropped.
    """
    print(f"\n--- request #{i}  {geometry} ---")
    print("  rank  token                  nocache     cached       delta")
    for rank, t in enumerate(sorted(ref, key=lambda x: -ref[x].logprob), 1):
        a = ref[t].logprob
        tag = f"{t} {ref[t].decoded_token!r}"[:20]
        if t in cached:
            b = cached[t].logprob
            print(f"  {rank:4d}  {tag:<20} {a:9.4f}  {b:9.4f}  {a - b:10.3e}")
        else:
            print(f"  {rank:4d}  {tag:<20} {a:9.4f}         --          --")


def build_prompts():
    """(warm, requests) as raw token ids.  No tokenizer, no text.

    Only the geometry matters, so the ids are synthesized rather than tokenized:

      - every request shares exactly PREFIX_TOKENS ids -> the same hit
      - each diverges at index PREFIX_TOKENS          -> the match stops there,
        and no two requests are equal, so they cannot hit each other's cache
      - each is exactly PROMPT_TOKENS long            -> the same chunk

    warm shares the prefix and then diverges too, so its full blocks and the
    sub-blocks of its trailing partial block are what the requests match against.
    Ids stay well below 128000 so none of them is a special token.
    """
    prefix = list(range(1, PREFIX_TOKENS + 1))
    tail_len = PROMPT_TOKENS - PREFIX_TOKENS
    warm = prefix + [50_000]
    requests = [
        prefix + [60_000 + i] + list(range(1, tail_len))
        for i in range(NUM_REQUESTS)
    ]
    assert all(len(ids) == PROMPT_TOKENS for ids in requests)
    return (
        TokensPrompt(prompt_token_ids=warm),
        [TokensPrompt(prompt_token_ids=ids) for ids in requests],
    )


def main():
    llm = make_llm(enable_prefix_caching=False)
    warm, requests = build_prompts()
    n = len(requests)

    # Reference, plus the same requests a second time for the noise floor.
    ref = llm.generate(requests + requests, sp())
    base = [logits_of(o) for o in ref[:n]]
    base2 = [logits_of(o) for o in ref[n:]]
    stray = max(hit_of(o) for o in ref)
    assert stray == 0, f"baseline engine reported a {stray}-token cache hit"

    # Signal.  Only one engine can hold the devices, so the baseline goes first.
    del llm
    llm = make_llm(enable_prefix_caching=True)
    llm.generate([warm], sp())  # populate the shared prefix
    outs = llm.generate(requests, sp())

    print(
        f"\nblock={BLOCK_SIZE}  chunk={MAX_BATCHED}  tp={TP}  "
        f"prompt={PROMPT_TOKENS} tok"
    )
    print("   #    hit   off0  chunk  spill   top1      max|d|       noise")
    agree = spilled = 0
    geometries = []
    for i, out in enumerate(outs):
        hit = hit_of(out)
        off0 = hit % BLOCK_SIZE
        chunk = min(MAX_BATCHED, PROMPT_TOKENS - hit)
        spill = off0 + chunk > BLOCK_SIZE
        cached = logits_of(out)
        same = top1(cached) == top1(base[i])
        agree += same
        spilled += spill
        geometries.append(
            f"hit={hit} off0={off0} chunk={chunk} "
            f"spill={'yes' if spill else 'no'}  top1={'same' if same else 'DIFF'}"
        )
        print(
            f"  {i:2d}  {hit:5d}  {off0:5d}  {chunk:5d}  "
            f"{'yes' if spill else ' no':>5}  {'same' if same else 'DIFF':>5}  "
            f"{max_delta(cached, base[i]):9.3e}  "
            f"{max_delta(base[i], base2[i]):9.3e}"
        )

    print(f"\ntop-1 agree {agree}/{n}   spill {spilled}/{n}")
    if spilled == 0:
        print(
            "WARNING: no request spilled across a block boundary -- the "
            "multi-block store path never ran.  Raise SPILL."
        )

    for i, out in enumerate(outs):
        dump(i, base[i], logits_of(out), geometries[i])


if __name__ == "__main__":
    main()
