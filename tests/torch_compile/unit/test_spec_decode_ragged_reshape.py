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

"""Regression tests for the ragged decode reshape crash under spec decode.

Reported symptom (execute_model): ``input_ids.view(num_reqs, -1)`` ->
``RuntimeError: shape '[7, -1]' is invalid for input of size 19``.

The reshape needs a rectangular batch. The per-request padding that makes it so
runs only when ``max_spec_decode_len is not None``, and the old gate inferred
that from two proxies (``max_tokens_per_req_across_dp``, nulled when any DP peer
is in prefill; ``scheduled_spec_decode_tokens``, empty on a cold/boundary-trimmed
ngram step). When both read "no spec" on a still-ragged step, padding was
skipped and the reshape crashed.

Fix: ``resolve_spec_decode_pad_len`` keys off ``spec_decode_max_query_len``
(= num_spec_tokens+1, cross-DP consistent), plus a defensive assert before the
reshape for any residual raggedness.
"""

import pytest
import torch

from vllm_rbln.utils import pad_speculative_draft_tokens
from vllm_rbln.v1.worker.rbln_model_runner import resolve_spec_decode_pad_len

_NUM_SPEC_TOKENS = 3
_MAX_SPEC_DECODE_LEN = _NUM_SPEC_TOKENS + 1  # 4


def _build_flat_input(per_req_lengths):
    """Flat 1D input_ids in the natural [req0_tokens, req1_tokens, ...] order
    the runner produces when the slide-pad step is skipped. Non-zero, distinct
    values so padding (which inserts zeros) is observable."""
    total = int(sum(per_req_lengths))
    return torch.arange(1, total + 1, dtype=torch.int32)


# ---------------------------------------------------------------------------
# 1) Symptom reproduction: when padding is skipped the reshape runs on a ragged
#    buffer and raises the reported error.
# ---------------------------------------------------------------------------


class TestRaggedReshapeSymptom:
    def test_unpadded_ragged_reshape_raises_reported_runtimeerror(self):
        """With ``max_spec_decode_len is None`` the runner skips padding and
        runs ``input_ids.view(num_reqs, -1)`` directly. Reproduces the exact
        reported failure: 7 reqs, 19 flat tokens."""
        num_reqs = 7
        per_req_lengths = [4, 4, 4, 3, 2, 1, 1]  # ragged, sums to 19
        assert sum(per_req_lengths) == 19
        input_ids = _build_flat_input(per_req_lengths)
        assert input_ids.numel() == 19

        with pytest.raises(RuntimeError, match=r"\[7, -1\].*size 19"):
            input_ids.view(num_reqs, -1)


# ---------------------------------------------------------------------------
# 2) The fix: resolve_spec_decode_pad_len keys off spec_decode_max_query_len.
# ---------------------------------------------------------------------------


class TestResolveSpecDecodePadLen:
    def test_decode_spec_step_resolves_to_query_len(self):
        """The headline fix: the exact bug scenario (decode phase, spec active)
        now resolves to num_spec+1 regardless of the proxies that misled the
        legacy gate."""
        assert (
            resolve_spec_decode_pad_len(
                is_prefill_phase=False,
                spec_decode_max_query_len=_MAX_SPEC_DECODE_LEN,
            )
            == _MAX_SPEC_DECODE_LEN
        )

    @pytest.mark.parametrize(
        "is_prefill_phase,spec_decode_max_query_len,expected",
        [
            # Decode + spec active -> pad to num_spec+1.
            (False, _MAX_SPEC_DECODE_LEN, _MAX_SPEC_DECODE_LEN),
            # Cross-DP no-spec scrub forced query_len==1 -> batch already
            # rectangular at 1, no padding.
            (False, 1, None),
            # Spec disabled (num_spec_tokens == 0) -> spec_decode_max_query_len
            # is None -> no padding.
            (False, None, None),
            # Prefill phase -> the spec sliding-window scheme does not apply;
            # never pad (matches the pre-fix prefill behavior).
            (True, _MAX_SPEC_DECODE_LEN, None),
            (True, 1, None),
            (True, None, None),
        ],
    )
    def test_matrix(self, is_prefill_phase, spec_decode_max_query_len, expected):
        assert (
            resolve_spec_decode_pad_len(
                is_prefill_phase=is_prefill_phase,
                spec_decode_max_query_len=spec_decode_max_query_len,
            )
            == expected
        )


# ---------------------------------------------------------------------------
# 3) End-to-end chain: resolved length -> pad -> rectangular reshape.
# ---------------------------------------------------------------------------


class TestRaggedReshapeFixed:
    def test_resolved_len_pads_ragged_batch_into_rectangular(self):
        """The full chain the runner takes after the fix: resolve the pad
        length, pad per-request, then reshape. The same ragged buffer that
        crashed the unpadded reshape now reshapes cleanly to
        (num_reqs, num_spec+1)."""
        num_reqs = 7
        per_req_lengths = [4, 4, 4, 3, 2, 1, 1]  # the size-19 ragged batch
        input_ids = _build_flat_input(per_req_lengths)

        max_spec_decode_len = resolve_spec_decode_pad_len(
            is_prefill_phase=False,
            spec_decode_max_query_len=_MAX_SPEC_DECODE_LEN,
        )
        assert max_spec_decode_len == _MAX_SPEC_DECODE_LEN

        num_scheduled = torch.tensor(per_req_lengths, dtype=torch.int32)
        padded = pad_speculative_draft_tokens(
            input_ids, num_scheduled, max_spec_decode_len
        )

        # Now rectangular: num_reqs * (num_spec+1).
        assert padded.numel() == num_reqs * _MAX_SPEC_DECODE_LEN
        assert padded.numel() % num_reqs == 0
        reshaped = padded.view(num_reqs, -1)  # would have raised before the fix
        assert reshaped.shape == (num_reqs, _MAX_SPEC_DECODE_LEN)

        # First request had a full window: row preserved verbatim.
        assert reshaped[0].tolist() == [1, 2, 3, 4]
        # A short request (length 1) is zero-padded on the right.
        last_token = int(input_ids[-1].item())
        assert reshaped[-1].tolist() == [last_token, 0, 0, 0]

    def test_over_length_request_is_rejected_loudly(self):
        """Defense in depth: if a request ever exceeds the resolved window
        (so no single pad length can rectangular-ize the batch),
        pad_speculative_draft_tokens fails loudly rather than silently
        corrupting the layout — the same spirit as the reshape guard."""
        # One request with 6 flat tokens > max_spec_decode_len (4).
        per_req_lengths = [6, 1, 1]
        input_ids = _build_flat_input(per_req_lengths)
        num_scheduled = torch.tensor(per_req_lengths, dtype=torch.int32)
        with pytest.raises(ValueError, match="max_len"):
            pad_speculative_draft_tokens(input_ids, num_scheduled, _MAX_SPEC_DECODE_LEN)


# ---------------------------------------------------------------------------
# 4) Defensive reshape guard: a still-ragged buffer reaching the reshape must
#    fail with batch context, not a bare shape error. Mirrors the assert added
#    in front of input_ids.view(num_reqs, -1).
# ---------------------------------------------------------------------------


class TestReshapeGuard:
    @staticmethod
    def _guarded_reshape(input_ids, num_reqs, *, max_spec_decode_len):
        # Mirror of the defensive assert + reshape in execute_model.
        assert input_ids.numel() % num_reqs == 0, (
            f"ragged batch reached decode reshape: numel={input_ids.numel()} "
            f"num_reqs={num_reqs} max_spec_decode_len={max_spec_decode_len}"
        )
        return input_ids.view(num_reqs, -1)

    def test_guard_raises_contextful_assertion_on_ragged(self):
        input_ids = _build_flat_input([4, 4, 4, 3, 2, 1, 1])  # size 19
        with pytest.raises(AssertionError, match="ragged batch reached decode reshape"):
            self._guarded_reshape(input_ids, 7, max_spec_decode_len=None)

    def test_guard_passes_on_rectangular(self):
        input_ids = _build_flat_input([4] * 7)  # size 28
        out = self._guarded_reshape(input_ids, 7, max_spec_decode_len=4)
        assert out.shape == (7, 4)
