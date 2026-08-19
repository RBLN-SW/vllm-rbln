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
from itertools import cycle, islice

os.environ.setdefault("VLLM_RBLN_USE_VLLM_MODEL", "1")

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402

MODEL = "meta-llama/Llama-3.2-1B"
BLOCK_SIZE = 1024
MAX_BATCHED = 1024
MAX_NUM_SEQS = 4
SUB_BLOCK = 128  # must match VLLM_RBLN_SUB_BLOCK_SIZE
TP = 1
NUM_REQUESTS = 8

# The shared prefix is a role setup followed by a document, which is the shape
# prefix caching exists for.  SYSTEM leads it once; PASSAGE fills the rest.
SYSTEM = (
    "You are a meticulous research assistant. Read the passage that follows and "
    "answer only from it. Quote exact wording where it matters, say plainly when "
    "the passage does not settle a question, and never introduce facts it does "
    "not contain. Keep answers short unless the question asks you to expand."
    "\n\n"
)

# The rest of the prefix, and the recomputed tail, are both cut from this
# passage.  It is long and varied on purpose: a short string cycled hundreds of
# times puts the model in a degenerate state where the top token sits at
# p ~ 0.999, and then even a large logprob move cannot change the argmax and the
# top-1 column stops being informative.  The words themselves do not matter --
# the cache sees ids.
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

# The two knobs of the experiment.
FULL_BLOCKS = 2  # full blocks before the sub-block hit, so the copy's destination
                 # block index is not 0 and an off-by-one there cannot hide
SPILL = 125  # tokens the recomputed chunk pushes into the NEXT block; 0 = no spill

# Both lengths follow from those, so there is one number per effect and no slack:
# the hit is FULL_BLOCKS full blocks plus 7 of the 8 sub-blocks of the next one,
# which is the largest hit that still leaves room inside a block.  The prefix is
# exactly the hit, so everything past it is recomputed and chunk is exactly the
# rest of that block plus SPILL.
PREFIX_TOKENS = FULL_BLOCKS * BLOCK_SIZE + (BLOCK_SIZE - SUB_BLOCK)
PROMPT_TOKENS = PREFIX_TOKENS + SUB_BLOCK + SPILL
# `chunk` is the one number RequestOutput does not expose, so the table derives
# it.  That derivation is only exact while the recompute of every request the
# scheduler can run at once fits in one step's token budget -- MAX_BATCHED is a
# per-step total shared across requests, not a per-request cap.  If this fails,
# a request's first chunk is smaller than the table claims and `spill` becomes an
# upper bound rather than a fact.
assert MAX_NUM_SEQS * (SUB_BLOCK + SPILL) <= MAX_BATCHED, (
    "the recomputed chunk can be split across steps; lower MAX_NUM_SEQS or SPILL"
)


def make_llm(enable_prefix_caching: bool) -> LLM:
    return LLM(
        model=MODEL,
        block_size=BLOCK_SIZE,
        max_num_batched_tokens=MAX_BATCHED,
        max_model_len=8192,
        max_num_seqs=MAX_NUM_SEQS,
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


def build_prompts(tok):
    """(warm, requests) as token ids.

    Each prompt is the same shared prefix, then one token of its own, then filler:

        prefix (PREFIX_TOKENS)                | mark | tail
        bos | SYSTEM | PASSAGE, cycled        | own  | PASSAGE again
        ------------------------------------- +------+---------------
        identical in every prompt             | own  | identical again

    which is all the geometry needs.  The shared prefix fixes the hit.  The one
    differing token -- `mark` -- stops the match at PREFIX_TOKENS and keeps any
    two requests from sharing a longer prefix; if they did, the second would hit
    the first's whole cached block and the spill would silently vanish.  The
    total length fixes the chunk.

    Built from ids, never round-tripped through text: decode() re-emits BOS as
    the literal "<|begin_of_text|>", which the next encode() would prefix with a
    second BOS.
    """

    def enc(text):
        return tok.encode(text, add_special_tokens=False)

    def fill(ids, n):
        """Exactly n ids, cycling through `ids`."""
        return list(islice(cycle(ids), n))

    head, body = enc(SYSTEM), enc(PASSAGE)
    assert len(head) < PREFIX_TOKENS - 1, "SYSTEM alone fills the prefix"
    prefix = [tok.bos_token_id] + head + fill(body, PREFIX_TOKENS - 1 - len(head))
    tail = fill(body, PROMPT_TOKENS - PREFIX_TOKENS - 1)

    # One distinct id per prompt is all the divergence needs, so take them from
    # the sentence rather than inventing a second word list.
    marks = sorted(set(body))[: NUM_REQUESTS + 1]
    assert len(marks) == NUM_REQUESTS + 1, "PASSAGE has too few distinct tokens"

    warm = prefix + marks[-1:]
    requests = [prefix + [mark] + tail for mark in marks[:-1]]
    assert all(len(ids) == PROMPT_TOKENS for ids in requests)
    return (
        TokensPrompt(prompt_token_ids=warm),
        [TokensPrompt(prompt_token_ids=ids) for ids in requests],
    )


def main():
    llm = make_llm(enable_prefix_caching=False)
    warm, requests = build_prompts(llm.get_tokenizer())
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
    print("   #  prompt    hit   off0  chunk  spill   top1      max|d|       noise")
    agree = spilled = 0
    geometries, wrong = [], []
    for i, out in enumerate(outs):
        # prompt length and hit are read back from the engine; off0 and chunk
        # follow from them, so pinning these two pins the whole row.
        n_prompt = len(out.prompt_token_ids)
        hit = hit_of(out)
        off0 = hit % BLOCK_SIZE
        chunk = min(MAX_BATCHED, n_prompt - hit)
        spill = off0 + chunk - BLOCK_SIZE
        if n_prompt != PROMPT_TOKENS:
            wrong.append(f"request {i}: prompt {n_prompt}, built {PROMPT_TOKENS}")
        if hit != PREFIX_TOKENS:
            wrong.append(f"request {i}: hit {hit}, expected {PREFIX_TOKENS}")
        if spill != SPILL:
            wrong.append(f"request {i}: spill {spill}, configured {SPILL}")

        cached = logits_of(out)
        same = top1(cached) == top1(base[i])
        agree += same
        spilled += spill > 0
        geometries.append(
            f"hit={hit} off0={off0} chunk={chunk} spill={spill}  "
            f"top1={'same' if same else 'DIFF'}"
        )
        print(
            f"  {i:2d}  {n_prompt:6d}  {hit:5d}  {off0:5d}  {chunk:5d}  "
            f"{spill:5d}  {'same' if same else 'DIFF':>5}  "
            f"{max_delta(cached, base[i]):9.3e}  "
            f"{max_delta(base[i], base2[i]):9.3e}"
        )

    # After the table, so a bad setup is still shown before it aborts.  A hit that
    # is not PREFIX_TOKENS usually means sub-block caching is off, or
    # VLLM_RBLN_SUB_BLOCK_SIZE differs from SUB_BLOCK, or warm never landed.
    assert not wrong, "geometry is not what was configured:\n  " + "\n  ".join(wrong)

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
