# Copyright 2026 Rebellions Inc. All rights reserved.
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
"""Repetition / frequency / presence penalties and logit bias, fused into the
sampling graph.

The host keeps, per request, the 64-wide vocab chunks of the tokens it has seen
or biased, with three lane parameters (add, mul, sub). The sampling graph
gathers only those chunks, penalizes them, and scatters them back.

Invariants the scatter relies on:
- A lane no token touched holds (0, 1, 0), so rewriting it is a no-op.
- Slots past a row's last used chunk repeat slot 0, so duplicate scatter
  targets always carry identical values.

See docs/fused_sampling.md.
"""

import dataclasses
from collections.abc import Callable, Iterable, Sequence
from typing import NamedTuple

import numpy as np
import torch
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.logits_processor.builtin import (
    LogitBiasLogitsProcessor,
    MinTokensLogitsProcessor,
)
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.worker.gpu_input_batch import InputBatch

CHUNK = 64
K_BUCKETS = (256, 512, 1024)
ADD, MUL, SUB = 0, 1, 2

# Lanes are fp16: a value outside its finite normal range keeps the request on
# the existing path rather than turning into inf or 0.
_FP16_MAX = float(np.finfo(np.float16).max)
_FP16_TINY = float(np.finfo(np.float16).tiny)

_IDENTITY = np.zeros((3, CHUNK), np.float16)
_IDENTITY[MUL] = 1.0


class FusedPenaltyPlan(NamedTuple):
    rows: int
    row_vocab: int
    k: int
    cidx: torch.Tensor  # [rows, k] int32 chunk ids
    lanes: torch.Tensor  # [rows, 3, k, CHUNK] fp16 (add, mul, sub)


def row_layout(num_rows: int, vocab_size: int) -> tuple[int, int]:
    """(rows, row_vocab) of the fused graph for a batch of `num_rows`."""
    # A batch of one is split into two half-vocab rows.
    # TODO: drop the split once the pinned compiler fixes batch-1 chunk gather.
    per_req = 2 if num_rows == 1 else 1
    return num_rows * per_req, vocab_size // per_req


def k_cap(row_vocab: int) -> int:
    """Most chunk slots a row can use. A graph with more fails to compile."""
    # TODO: use the plain chunk count once the pinned compiler pads gather and
    # scatter indices itself (rebel_compiler#15631).
    return row_vocab // CHUNK // CHUNK * CHUNK


def k_buckets(row_vocab: int, base: Sequence[int] = K_BUCKETS) -> tuple[int, ...]:
    """The chunk-slot counts a row of `row_vocab` compiles graphs for."""
    cap = k_cap(row_vocab)
    return tuple(sorted({b for b in base if b < cap} | {cap}))


def supports_vocab(vocab_size: int) -> bool:
    return vocab_size % (2 * CHUNK) == 0 and k_cap(vocab_size // 2) > 0


def uses_argmax_invariant_processors(sampling_metadata: SamplingMetadata) -> bool:
    """Whether a random-sampling step must run min_p and other argmax-invariant
    processors on the logits before sampling."""
    return any(
        getattr(p, "min_p_count", 1)
        for p in sampling_metadata.logitsprocs.argmax_invariant
    )


def without_penalties_and_logit_bias(
    sampling_metadata: SamplingMetadata,
) -> SamplingMetadata:
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


def keeps_logit_bias_order(sampling_metadata: SamplingMetadata) -> bool:
    """Whether running logit bias in the graph, after the eager processors, keeps
    upstream's order. Upstream applies it before custom non-argmax-invariant
    processors, which may read the biased logits."""
    return all(
        isinstance(p, (LogitBiasLogitsProcessor, MinTokensLogitsProcessor))
        for p in sampling_metadata.logitsprocs.non_argmax_invariant
    )


class _Request:
    """Append-only lane store of one request. Slots fill in first-seen order."""

    __slots__ = (
        "is_vocab_split",
        "half_vocab_size",
        "repetition_penalty",
        "frequency_penalty",
        "presence_penalty",
        "has_penalties",
        "has_repetition_penalty",
        "fits",
        "chunk_to_slot_by_row",
        "seen_token_ids",
        "output_token_counts",
        "num_processed_output_tokens",
        "chunk_indices",
        "lanes",
    )

    def __init__(
        self,
        split: bool,
        vocab_size: int,
        capacity: int,
        rep: float,
        freq: float,
        pres: float,
    ):
        self.is_vocab_split = split
        self.half_vocab_size = vocab_size // 2
        self.repetition_penalty = np.float16(rep)
        self.frequency_penalty = float(freq)
        self.presence_penalty = float(pres)
        self.has_penalties = (
            rep != 1.0 or self.frequency_penalty != 0.0 or self.presence_penalty != 0.0
        )
        self.has_repetition_penalty = rep != 1.0
        self.fits = not self.has_repetition_penalty or _FP16_TINY <= rep <= _FP16_MAX
        rows = 2 if split else 1
        self.chunk_to_slot_by_row: list[dict[int, int]] = [{} for _ in range(rows)]
        self.seen_token_ids: set[int] = set()
        self.output_token_counts: dict[int, int] = {}
        self.num_processed_output_tokens = 0
        self.chunk_indices = np.zeros((rows, capacity), np.int32)
        self.lanes = np.empty((rows, 3, capacity, CHUNK), np.float16)

    @property
    def used(self) -> int:
        return max(len(s) for s in self.chunk_to_slot_by_row)

    def _grow(self) -> None:
        n = self.chunk_indices.shape[1]
        cidx = np.zeros((self.chunk_indices.shape[0], 2 * n), np.int32)
        lanes = np.empty((self.lanes.shape[0], 3, 2 * n, CHUNK), np.float16)
        cidx[:, :n] = self.chunk_indices
        lanes[:, :, :n] = self.lanes
        self.chunk_indices, self.lanes = cidx, lanes

    def _lane(self, token: int) -> tuple[int, int, int]:
        """(row, slot, lane) of `token`, taking a new slot on first touch."""
        row, t = (
            (1, token - self.half_vocab_size)
            if self.is_vocab_split and token >= self.half_vocab_size
            else (0, token)
        )
        slots = self.chunk_to_slot_by_row[row]
        chunk = t // CHUNK
        s = slots.get(chunk)
        if s is None:
            s = len(slots)
            if s >= self.chunk_indices.shape[1]:
                self._grow()
            slots[chunk] = s
            self.chunk_indices[row, s] = chunk
            self.lanes[row, :, s] = _IDENTITY
        return row, s, t % CHUNK

    def add_bias(self, token: int, value: float) -> None:
        if abs(value) > _FP16_MAX:
            self.fits = False
            return
        row, s, lane = self._lane(token)
        self.lanes[row, ADD, s, lane] = value

    def add_prompt(self, token: int) -> None:
        if token not in self.seen_token_ids:
            self.seen_token_ids.add(token)
            row, s, lane = self._lane(token)
            self.lanes[row, MUL, s, lane] = self.repetition_penalty

    def add_output(self, token: int) -> None:
        if self.has_repetition_penalty:
            self.add_prompt(token)
        if self.frequency_penalty or self.presence_penalty:
            c = self.output_token_counts[token] = (
                self.output_token_counts.get(token, 0) + 1
            )
            sub = self.frequency_penalty * c + self.presence_penalty
            if abs(sub) > _FP16_MAX:
                self.fits = False
                return
            row, s, lane = self._lane(token)
            self.lanes[row, SUB, s, lane] = sub


class FusedPenaltyState:
    """Per-request host state, updated with only the tokens added each step."""

    def __init__(self, base_buckets: Sequence[int] = K_BUCKETS):
        self.base_buckets = tuple(base_buckets)
        self._reqs: dict[str, _Request] = {}
        self._cidx: np.ndarray | None = None
        self._lanes: np.ndarray | None = None

    def plan(
        self,
        sampling_metadata: SamplingMetadata,
        input_batch: InputBatch,
        logits_shape: torch.Size,
        device: torch.device,
    ) -> FusedPenaltyPlan | None:
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
        if biases and not keeps_logit_bias_order(sampling_metadata):
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

        step = self.update(
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
        if step is None:
            return None
        rows, row_vocab, k, cidx, lanes = step
        return FusedPenaltyPlan(rows, row_vocab, k, cidx.to(device), lanes.to(device))

    def forget(self, req_ids: Iterable[str]) -> None:
        """Drop finished requests. A step lists only the requests it schedules,
        so absence from one step does not mean a request is gone."""
        for rid in req_ids:
            self._reqs.pop(rid, None)

    def k_buckets(self, logits_shape: torch.Size) -> tuple[int, ...]:
        """Every `k` a plan for `logits_shape` can take."""
        return k_buckets(row_layout(*logits_shape)[1], self.base_buckets)

    def identity_plan(
        self, logits_shape: torch.Size, device: torch.device, k: int
    ) -> FusedPenaltyPlan:
        """A plan that changes nothing, at chunk-slot count `k`. Used for warm-up."""
        num_rows, vocab_size = logits_shape
        rows, row_vocab, k, cidx, lanes = self._copy_out([], vocab_size, num_rows, k)
        return FusedPenaltyPlan(rows, row_vocab, k, cidx.to(device), lanes.to(device))

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
    ) -> tuple[int, int, int, torch.Tensor, torch.Tensor] | None:
        """Write this step's new tokens into the lanes. Return
        (rows, row_vocab, k, cidx, lanes).

        `num_rows` is the padded batch. Rows past `len(req_ids)` are identity.
        `k` is the smallest bucket that fits the live requests. None when one of
        them uses more chunks than any bucket holds.
        """
        split = num_rows == 1
        for i, rid in enumerate(req_ids):
            outs = output_token_ids[i] if i < len(output_token_ids) else ()
            n_out = len(outs)
            req = self._reqs.get(rid)
            if req is not None and (
                req.is_vocab_split != split or n_out < req.num_processed_output_tokens
            ):
                req = None
            # Tokens already written are all resolved ids.
            start = req.num_processed_output_tokens if req is not None else 0
            for j in range(start, n_out):
                if outs[j] < 0:  # async placeholder
                    n_out = j
                    break
            if req is None:
                req = self._reqs[rid] = _Request(
                    split,
                    vocab_size,
                    self.base_buckets[0],
                    float(rep[i]),
                    float(freq[i]),
                    float(pres[i]),
                )
                for t, v in biases.get(i, {}).items():
                    req.add_bias(int(t), v)
                if req.has_penalties:
                    # Frequency and presence only read output tokens.
                    if req.has_repetition_penalty:
                        for t in prompt_fn(i):
                            if 0 <= t < vocab_size:
                                req.add_prompt(int(t))
                    new_outputs = outs[:n_out]
                else:
                    new_outputs = ()
            else:
                new_outputs = (
                    outs[req.num_processed_output_tokens : n_out]
                    if req.has_penalties
                    else ()
                )
            for t in new_outputs:
                if 0 <= t < vocab_size:
                    req.add_output(int(t))
            req.num_processed_output_tokens = n_out

        if not all(self._reqs[rid].fits for rid in req_ids):
            return None
        need = max((self._reqs[rid].used for rid in req_ids), default=0)
        row_vocab = row_layout(num_rows, vocab_size)[1]
        k = next((b for b in k_buckets(row_vocab, self.base_buckets) if b >= need), 0)
        if not k:
            return None
        return self._copy_out(req_ids, vocab_size, num_rows, k)

    def _copy_out(self, req_ids, vocab_size, num_rows, k):
        rows, row_vocab = row_layout(num_rows, vocab_size)
        per_req = rows // num_rows
        if (
            self._cidx is None
            or self._lanes is None
            or self._cidx.shape[0] < rows
            or self._cidx.shape[1] != k
        ):
            self._cidx = np.zeros((rows, k), np.int32)
            self._lanes = np.empty((rows, 3, k, CHUNK), np.float16)
        cidx, lanes = self._cidx[:rows], self._lanes[:rows]
        for out in range(rows):
            i, r = divmod(out, per_req)
            req = self._reqs[req_ids[i]] if i < len(req_ids) else None
            n = len(req.chunk_to_slot_by_row[r]) if req is not None else 0
            if n == 0:
                cidx[out] = 0
                lanes[out] = _IDENTITY[:, None, :]
                continue
            assert req is not None
            cidx[out, :n] = req.chunk_indices[r, :n]
            cidx[out, n:] = req.chunk_indices[r, 0]
            lanes[out, :, :n] = req.lanes[r, :, :n]
            lanes[out, :, n:] = req.lanes[r, :, :1]
        return (
            rows,
            row_vocab,
            k,
            torch.from_numpy(cidx),
            torch.from_numpy(lanes),
        )


def apply_fused_penalty(
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
