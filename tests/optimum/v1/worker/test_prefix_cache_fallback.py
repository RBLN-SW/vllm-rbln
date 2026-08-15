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

"""Tests for the prefix-cache KV copy fallback.

When copying prefix-cached KV blocks fails (e.g. device OOM), the runner must
not kill the engine: it rebuilds the prefill inputs without the cached-prefix
trim and runs a full prefill instead.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_rbln.model_executor.models.optimum.model_base import (
    KVCacheCopyError,
    RBLNOptimumDecoderMixin,
)

from .utils import (
    MockModelWrapper,
    _schedule_cached_reqs,
    _schedule_new_request,
    create_model_runner,
)

PROMPT_TOKEN_IDS = [1, 2, 3, 4, 5, 6, 7, 8]
CACHED_BLOCK_TABLE = [3]
CACHED_LENGTH = [4]


class MockDecoderModelWrapper(MockModelWrapper, RBLNOptimumDecoderMixin):
    """Decoder-mixin mock so the runner takes the KV-copy path.

    Records copy attempts and optionally fails them with KVCacheCopyError.
    """

    def __init__(self, fail_copy: bool = False):
        super().__init__()
        self.fail_copy = fail_copy
        self.copy_calls: list[tuple[list[int], list[int]]] = []

    def copy_cached_kv_blocks(
        self,
        cached_block_tables: list[int],
        cached_lengths: list[int],
        block_tables: torch.Tensor,
    ) -> None:
        if not cached_block_tables:
            return
        self.copy_calls.append((cached_block_tables, cached_lengths))
        if self.fail_copy:
            raise KVCacheCopyError("Failed to copy KV cache: device OOM")


def _make_runner_with_mock(fail_copy: bool):
    runner = create_model_runner()
    model = MockDecoderModelWrapper(fail_copy=fail_copy)
    forward_inputs = []

    def fake_forward(model_input, **kwargs) -> torch.Tensor:
        forward_inputs.append(model_input)
        num_reqs = runner.input_batch.num_reqs
        vocab_size = runner.model_config.get_vocab_size()
        return torch.randn(
            (num_reqs, 1, vocab_size), dtype=torch.float32, device=runner.device
        )

    model.forward = fake_forward
    runner.model = model
    runner.use_optimum_lora = False
    return runner, model, forward_inputs


def _run_prefill_with_cache_hit(runner):
    scheduler_output = _schedule_new_request(
        "req_0",
        block_ids=([0],),
        outer_block_ids=[0],
        token_ids=PROMPT_TOKEN_IDS,
        cached_block_table=CACHED_BLOCK_TABLE,
        cached_length=CACHED_LENGTH,
    )
    runner.execute_model(scheduler_output)
    return runner.sample_tokens(None)


def test_prefill_with_kv_copy_success():
    """Baseline: on a cache hit the prefill input is trimmed by the cached
    length and positions start at the cache boundary."""
    runner, model, forward_inputs = _make_runner_with_mock(fail_copy=False)

    model_output = _run_prefill_with_cache_hit(runner)

    assert model.copy_calls == [(CACHED_BLOCK_TABLE, CACHED_LENGTH)]
    assert len(forward_inputs) == 1
    model_input = forward_inputs[0]
    num_cached = sum(CACHED_LENGTH)
    assert model_input.input_tokens.tolist() == [PROMPT_TOKEN_IDS[num_cached:]]
    assert model_input.input_positions.tolist() == [
        list(range(num_cached, len(PROMPT_TOKEN_IDS)))
    ]
    assert model_output is not None
    assert model_output.req_ids == ["req_0"]


def test_prefill_falls_back_to_full_prefill_on_kv_copy_failure():
    """A failed KV copy must not propagate: the runner recomputes the full
    prompt (no trim, positions from 0) and still produces a token."""
    runner, model, forward_inputs = _make_runner_with_mock(fail_copy=True)

    model_output = _run_prefill_with_cache_hit(runner)

    # The copy was attempted exactly once, then abandoned.
    assert model.copy_calls == [(CACHED_BLOCK_TABLE, CACHED_LENGTH)]
    # The forward ran once, with the full untrimmed prompt.
    assert len(forward_inputs) == 1
    model_input = forward_inputs[0]
    assert model_input.input_tokens.tolist() == [PROMPT_TOKEN_IDS]
    assert model_input.input_positions.tolist() == [list(range(len(PROMPT_TOKEN_IDS)))]
    assert model_input.partial_prefix is None
    assert model_output is not None
    assert model_output.req_ids == ["req_0"]


def test_decode_skips_kv_copy_after_fallback():
    """Decode steps after a fallback prefill run normally without copies."""
    runner, model, forward_inputs = _make_runner_with_mock(fail_copy=True)

    _run_prefill_with_cache_hit(runner)
    assert len(model.copy_calls) == 1

    # Decode step: no cached blocks are scheduled, so no copy attempt.
    # _schedule_cached_reqs only reads these three request attributes.
    req = SimpleNamespace(
        request_id="req_0",
        num_computed_tokens=len(PROMPT_TOKEN_IDS),
        output_token_ids=[9],
    )
    scheduler_output = _schedule_cached_reqs([req], new_block_ids=[None])
    runner.execute_model(scheduler_output)
    model_output = runner.sample_tokens(None)

    assert len(model.copy_calls) == 1
    assert len(forward_inputs) == 2
    assert not forward_inputs[1].is_prompt
    assert model_output is not None
    assert model_output.req_ids == ["req_0"]


def test_non_copy_errors_still_propagate():
    """Only KVCacheCopyError triggers the fallback; other model errors must
    surface unchanged."""
    runner, model, forward_inputs = _make_runner_with_mock(fail_copy=False)

    def broken_copy(*args, **kwargs):
        raise ValueError("scheduler/runner invariant violated")

    model.copy_cached_kv_blocks = broken_copy

    with pytest.raises(ValueError, match="invariant"):
        _run_prefill_with_cache_hit(runner)
    assert not forward_inputs
