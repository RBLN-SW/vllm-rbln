# Copyright 2026 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Sparse repetition / frequency / presence penalties and logit bias.

The host keeps, per request, the 64-wide vocab chunks of the tokens it has seen
or biased, with three lane parameters (add, mul, sub). The sampling graph
gathers only those chunks, penalizes them, and scatters them back.

Invariants the scatter relies on:
- A lane no token touched holds (0, 1, 0), so rewriting it is a no-op.
- Slots past a row's last used chunk repeat slot 0, so duplicate scatter
  targets always carry identical values.

See docs/sparse_penalty.md.
"""

import dataclasses
from collections.abc import Callable, Sequence
from typing import NamedTuple

import numpy as np
import torch
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.logits_processor.builtin import LogitBiasLogitsProcessor
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.worker.gpu_input_batch import InputBatch

CHUNK = 64
K_MIN = 256
ADD, MUL, SUB = 0, 1, 2

_IDENTITY = np.zeros((3, CHUNK), np.float16)
_IDENTITY[MUL] = 1.0


class SparsePlan(NamedTuple):
    rows: int
    row_vocab: int
    k: int
    cidx: torch.Tensor  # [rows, k] int32 chunk ids
    lanes: torch.Tensor  # [rows, 3, k, CHUNK] fp16 (add, mul, sub)


def supports_vocab(vocab_size: int) -> bool:
    # A batch of one is split into two half-vocab rows.
    return vocab_size % (2 * CHUNK) == 0


def next_k(k: int) -> int:
    """Chunk-slot bucket after `k`: +64 up to 512, then doubling."""
    return k + CHUNK if k < 512 else k * 2


def uses_argmax_invariant_processors(sampling_metadata: SamplingMetadata) -> bool:
    """Whether a random-sampling step must run min_p and other argmax-invariant
    processors on the logits before sampling."""
    return any(
        getattr(p, "min_p_count", 1)
        for p in sampling_metadata.logitsprocs.argmax_invariant
    )


def without_deferred(sampling_metadata: SamplingMetadata) -> SamplingMetadata:
    """The metadata upstream `apply_logits_processors` should see when penalties
    and logit bias run in the sampling graph instead."""
    return dataclasses.replace(
        sampling_metadata,
        no_penalties=True,
        logitsprocs=LogitsProcessors(
            p
            for p in sampling_metadata.logitsprocs.all
            if not isinstance(p, LogitBiasLogitsProcessor)
        ),
    )


class _GrowK(Exception):
    pass


class _Request:
    """Append-only lane store of one request. Slots fill in first-seen order."""

    __slots__ = (
        "split",
        "half",
        "rep",
        "freq",
        "pres",
        "active",
        "slots",
        "seen",
        "counts",
        "done",
        "cidx",
        "lanes",
    )

    def __init__(
        self, split: bool, vocab_size: int, k: int, rep: float, freq: float, pres: float
    ):
        self.split = split
        self.half = vocab_size // 2
        self.rep = np.float16(rep)
        self.freq = float(freq)
        self.pres = float(pres)
        self.active = rep != 1.0 or self.freq != 0.0 or self.pres != 0.0
        rows = 2 if split else 1
        self.slots: list[dict[int, int]] = [{} for _ in range(rows)]
        self.seen: set[int] = set()
        self.counts: dict[int, int] = {}
        self.done = 0
        self.cidx = np.zeros((rows, k), np.int32)
        self.lanes = np.empty((rows, 3, k, CHUNK), np.float16)

    def _lane(self, token: int) -> tuple[int, int, int]:
        """(row, slot, lane) of `token`, taking a new slot on first touch."""
        row, t = (
            (1, token - self.half) if self.split and token >= self.half else (0, token)
        )
        slots = self.slots[row]
        chunk = t // CHUNK
        s = slots.get(chunk)
        if s is None:
            s = len(slots)
            if s >= self.cidx.shape[1]:
                raise _GrowK
            slots[chunk] = s
            self.cidx[row, s] = chunk
            self.lanes[row, :, s] = _IDENTITY
        return row, s, t % CHUNK

    def add_bias(self, token: int, value: float) -> None:
        row, s, lane = self._lane(token)
        self.lanes[row, ADD, s, lane] = value

    def add_prompt(self, token: int) -> None:
        if token not in self.seen:
            self.seen.add(token)
            row, s, lane = self._lane(token)
            self.lanes[row, MUL, s, lane] = self.rep

    def add_output(self, token: int) -> None:
        self.add_prompt(token)
        if self.freq or self.pres:
            c = self.counts[token] = self.counts.get(token, 0) + 1
            row, s, lane = self._lane(token)
            self.lanes[row, SUB, s, lane] = self.freq * c + self.pres


class SparsePenaltyState:
    """Per-request host state, updated with only the tokens added each step."""

    def __init__(self, k_min: int = K_MIN):
        self.k = k_min
        self._reqs: dict[str, _Request] = {}
        self._cidx: np.ndarray | None = None
        self._lanes: np.ndarray | None = None

    def plan(
        self,
        sampling_metadata: SamplingMetadata,
        input_batch: InputBatch,
        logits_shape: torch.Size,
        device: torch.device,
    ) -> SparsePlan | None:
        """The plan for this step, or None when the step keeps the upstream path."""
        num_rows, vocab_size = logits_shape
        holder = sampling_metadata.thinking_budget_state_holder
        if not supports_vocab(vocab_size) or (
            holder is not None and holder.has_tracked_requests()
        ):
            return None
        biases: dict[int, dict[int, float]] = next(
            (
                p.biases
                for p in sampling_metadata.logitsprocs.non_argmax_invariant
                if isinstance(p, LogitBiasLogitsProcessor)
            ),
            {},
        )
        if sampling_metadata.no_penalties and not biases:
            return None

        n = input_batch.num_reqs
        if sampling_metadata.no_penalties:
            rep = np.ones(n, np.float32)
            freq = pres = np.zeros(n, np.float32)
            outputs: Sequence[Sequence[int]] = []
        else:
            rep = input_batch.repetition_penalties_cpu[:n]
            freq = input_batch.frequency_penalties_cpu[:n]
            pres = input_batch.presence_penalties_cpu[:n]
            outputs = sampling_metadata.output_token_ids

        def prompt(i: int) -> np.ndarray:
            return input_batch.token_ids_cpu[i, : input_batch.num_prompt_tokens[i]]

        rows, row_vocab, k, cidx, lanes = self.update(
            input_batch.req_ids[:n],
            prompt,
            outputs,
            rep,
            freq,
            pres,
            biases,
            vocab_size,
            num_rows,
        )
        return SparsePlan(rows, row_vocab, k, cidx.to(device), lanes.to(device))

    def identity_plan(
        self, logits_shape: torch.Size, device: torch.device
    ) -> SparsePlan:
        """A plan that changes nothing, at the current bucket. Used for warm-up."""
        num_rows, vocab_size = logits_shape
        rows, row_vocab, k, cidx, lanes = self._copy_out([], vocab_size, num_rows)
        return SparsePlan(rows, row_vocab, k, cidx.to(device), lanes.to(device))

    def update(
        self,
        req_ids: Sequence[str],
        prompt_fn: Callable[[int], np.ndarray],
        output_token_ids: Sequence[Sequence[int]],
        rep: np.ndarray,
        freq: np.ndarray,
        pres: np.ndarray,
        biases: dict[int, dict[int, float]],
        vocab_size: int,
        num_rows: int,
    ) -> tuple[int, int, int, torch.Tensor, torch.Tensor]:
        """Fold this step's tokens in and return (rows, row_vocab, k, cidx, lanes).

        `num_rows` is the padded batch. Rows past `len(req_ids)` are identity.
        When a request outgrows `k` chunk slots, `k` moves to `next_k` and every
        live request is rebuilt from its full history.
        """
        while True:
            try:
                return self._update(
                    req_ids,
                    prompt_fn,
                    output_token_ids,
                    rep,
                    freq,
                    pres,
                    biases,
                    vocab_size,
                    num_rows,
                )
            except _GrowK:
                self.k = next_k(self.k)
                self._reqs.clear()

    def _update(
        self,
        req_ids,
        prompt_fn,
        output_token_ids,
        rep,
        freq,
        pres,
        biases,
        vocab_size,
        num_rows,
    ):
        split = num_rows == 1
        live = set(req_ids)
        for rid in [r for r in self._reqs if r not in live]:
            del self._reqs[rid]

        for i, rid in enumerate(req_ids):
            outs = output_token_ids[i] if i < len(output_token_ids) else ()
            n_out = len(outs)
            for j, t in enumerate(outs):
                if t < 0:  # async placeholder
                    n_out = j
                    break
            req = self._reqs.get(rid)
            if req is not None and (req.split != split or n_out < req.done):
                req = None
            if req is None:
                req = self._reqs[rid] = _Request(
                    split,
                    vocab_size,
                    self.k,
                    float(rep[i]),
                    float(freq[i]),
                    float(pres[i]),
                )
                for t, v in biases.get(i, {}).items():
                    req.add_bias(int(t), v)
                if req.active:
                    for t in prompt_fn(i):
                        if 0 <= t < vocab_size:
                            req.add_prompt(int(t))
                    new_outputs = outs[:n_out]
                else:
                    new_outputs = ()
            else:
                new_outputs = outs[req.done : n_out] if req.active else ()
            for t in new_outputs:
                if 0 <= t < vocab_size:
                    req.add_output(int(t))
            req.done = n_out

        return self._copy_out(req_ids, vocab_size, num_rows)

    def _copy_out(self, req_ids, vocab_size, num_rows):
        per_req = 2 if num_rows == 1 else 1
        rows = num_rows * per_req
        if (
            self._cidx is None
            or self._lanes is None
            or self._cidx.shape[0] < rows
            or self._cidx.shape[1] != self.k
        ):
            self._cidx = np.zeros((rows, self.k), np.int32)
            self._lanes = np.empty((rows, 3, self.k, CHUNK), np.float16)
        cidx, lanes = self._cidx[:rows], self._lanes[:rows]
        for out in range(rows):
            i, r = divmod(out, per_req)
            req = self._reqs[req_ids[i]] if i < len(req_ids) else None
            n = len(req.slots[r]) if req is not None else 0
            if n == 0:
                cidx[out] = 0
                lanes[out] = _IDENTITY[:, None, :]
                continue
            assert req is not None
            cidx[out, :n] = req.cidx[r, :n]
            cidx[out, n:] = req.cidx[r, 0]
            lanes[out, :, :n] = req.lanes[r, :, :n]
            lanes[out, :, n:] = req.lanes[r, :, :1]
        return (
            rows,
            vocab_size // per_req,
            self.k,
            torch.from_numpy(cidx),
            torch.from_numpy(lanes),
        )


def apply_sparse_penalty(
    logits: torch.Tensor,
    cidx: torch.Tensor,
    lanes: torch.Tensor,
    rows: int,
    row_vocab: int,
    k: int,
) -> torch.Tensor:
    l3 = logits.view(rows, row_vocab // CHUNK, CHUNK)
    i3 = cidx.view(rows, k, 1).expand(rows, k, CHUNK)
    v = torch.gather(l3, 1, i3) + lanes[:, ADD]
    mul = lanes[:, MUL]
    v = (torch.where(v > 0, v / mul, v * mul) - lanes[:, SUB]).to(l3.dtype)
    return l3.scatter(1, i3, v).view(logits.shape)
