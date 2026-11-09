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

from collections.abc import Callable, Sequence

import numpy as np
import torch

CHUNK = 64
K_MIN = 256
ADD, MUL, SUB = 0, 1, 2

_IDENTITY = np.zeros((3, CHUNK), np.float16)
_IDENTITY[MUL] = 1.0


def supports_vocab(vocab_size: int) -> bool:
    # A batch of one is split into two half-vocab rows.
    return vocab_size % (2 * CHUNK) == 0


class _GrowK(Exception):
    pass


class _Request:
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
        self.lanes[:] = _IDENTITY[None, :, None, :]

    def locate(self, token: int) -> tuple[int, int]:
        if self.split and token >= self.half:
            return 1, token - self.half
        return 0, token

    def slot(self, row: int, chunk: int) -> int:
        slots = self.slots[row]
        s = slots.get(chunk)
        if s is None:
            s = len(slots)
            if s >= self.cidx.shape[1]:
                raise _GrowK
            slots[chunk] = s
            self.cidx[row, s] = chunk
            self.lanes[row, :, s, :] = _IDENTITY
            if s == 0:
                self.cidx[row, 1:] = chunk
        return s

    def mirror(self, row: int) -> None:
        # Unused slots repeat slot 0, so the scatter writes the same values twice.
        n = len(self.slots[row])
        if 0 < n < self.cidx.shape[1]:
            self.cidx[row, n:] = self.cidx[row, 0]
            self.lanes[row, :, n:, :] = self.lanes[row, :, :1, :]

    def touch(
        self, token: int, *, penalize: bool, output: bool, bias: float | None = None
    ) -> tuple[int, int]:
        row, t = self.locate(token)
        s = self.slot(row, t // CHUNK)
        lane = t % CHUNK
        if bias is not None:
            self.lanes[row, ADD, s, lane] = bias
        if penalize and token not in self.seen:
            self.seen.add(token)
            self.lanes[row, MUL, s, lane] = self.rep
        if output and (self.freq or self.pres):
            c = self.counts.get(token, 0) + 1
            self.counts[token] = c
            self.lanes[row, SUB, s, lane] = self.freq * c + self.pres
        return row, s


class SparsePenaltyState:
    """Per-request host state, updated with only the tokens added each step."""

    def __init__(self, k_min: int = K_MIN):
        self.k = k_min
        self._reqs: dict[str, _Request] = {}
        self._cidx: np.ndarray | None = None
        self._lanes: np.ndarray | None = None

    def build(
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
        """Returns (rows, row_vocab, k, cidx [rows, k], lanes [rows, 3, k, 64]).

        `num_rows` is the padded batch; rows past `len(req_ids)` are identity.
        """
        while True:
            try:
                return self._build(
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
                self.k = self.k + CHUNK if self.k < 512 else self.k * 2
                self._reqs.clear()

    def _build(
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
                req = _Request(
                    split,
                    vocab_size,
                    self.k,
                    float(rep[i]),
                    float(freq[i]),
                    float(pres[i]),
                )
                self._reqs[rid] = req
                for t, v in biases.get(i, {}).items():
                    req.touch(int(t), penalize=False, output=False, bias=v)
                if req.active:
                    for t in prompt_fn(i):
                        if 0 <= t < vocab_size:
                            req.touch(int(t), penalize=True, output=False)
                    for t in outs[:n_out]:
                        if 0 <= t < vocab_size:
                            req.touch(int(t), penalize=True, output=True)
                for r in range(len(req.slots)):
                    req.mirror(r)
            elif n_out > req.done and req.active:
                rows_to_mirror = set()
                for t in outs[req.done : n_out]:
                    if 0 <= t < vocab_size:
                        r, s = req.touch(int(t), penalize=True, output=True)
                        if s == 0:
                            rows_to_mirror.add(r)
                for r in rows_to_mirror:
                    req.mirror(r)
            req.done = n_out

        per_req = 2 if split else 1
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
        for i, rid in enumerate(req_ids):
            req = self._reqs[rid]
            cidx[i * per_req : (i + 1) * per_req] = req.cidx
            lanes[i * per_req : (i + 1) * per_req] = req.lanes
        live_rows = len(req_ids) * per_req
        if live_rows < rows:
            cidx[live_rows:] = 0
            lanes[live_rows:] = _IDENTITY[None, :, None, :]
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
