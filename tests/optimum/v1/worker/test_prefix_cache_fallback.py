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

"""Tests for the prefix-cache KV copy fallback decision."""

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from vllm_rbln.model_executor.models.optimum.model_base import (
    KVCacheCopyError,
    RBLNOptimumDecoderMixin,
)
from vllm_rbln.v1.worker.optimum_model_runner import RBLNOptimumModelRunner

from .utils import MockModelWrapper

CACHED_BLOCK_TABLE = [3]
CACHED_LENGTH = [4]


class MockDecoderModelWrapper(MockModelWrapper, RBLNOptimumDecoderMixin):
    def __init__(self, fail_copy: bool = False):
        super().__init__(max_num_seqs=1)
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


class MockUnexpectedErrorDecoderModelWrapper(MockDecoderModelWrapper):
    def copy_cached_kv_blocks(
        self,
        cached_block_tables: list[int],
        cached_lengths: list[int],
        block_tables: torch.Tensor,
    ) -> None:
        raise ValueError("scheduler/runner invariant violated")


def _try_copy_prefix_cached_kv(
    *, fail_copy: bool = False, is_prompt: bool = True
) -> tuple[bool, MockDecoderModelWrapper]:
    model = MockDecoderModelWrapper(fail_copy=fail_copy)
    runner = cast(Any, SimpleNamespace(model=model))
    model_input = cast(
        Any,
        SimpleNamespace(
            is_prompt=is_prompt,
            block_tables=torch.tensor([[0]]),
            running_requests_ids=["req_0"],
        ),
    )
    scheduler_output = cast(
        Any,
        SimpleNamespace(
            cached_block_table=CACHED_BLOCK_TABLE,
            cached_length=CACHED_LENGTH,
        ),
    )
    copied = RBLNOptimumModelRunner.try_copy_prefix_cached_kv(
        runner, model_input, scheduler_output
    )
    return copied, model


def test_prefill_copies_prefix_cached_kv():
    copied, model = _try_copy_prefix_cached_kv()

    assert copied
    assert model.copy_calls == [(CACHED_BLOCK_TABLE, CACHED_LENGTH)]


def test_prefill_requests_fallback_on_kv_copy_failure():
    copied, model = _try_copy_prefix_cached_kv(fail_copy=True)

    assert not copied
    assert model.copy_calls == [(CACHED_BLOCK_TABLE, CACHED_LENGTH)]


def test_decode_skips_prefix_cached_kv_copy():
    copied, model = _try_copy_prefix_cached_kv(is_prompt=False)

    assert copied
    assert not model.copy_calls


def test_non_copy_errors_still_propagate():
    model = MockUnexpectedErrorDecoderModelWrapper()
    runner = cast(Any, SimpleNamespace(model=model))
    model_input = cast(
        Any,
        SimpleNamespace(
            is_prompt=True,
            block_tables=torch.tensor([[0]]),
            running_requests_ids=["req_0"],
        ),
    )
    scheduler_output = cast(
        Any,
        SimpleNamespace(
            cached_block_table=CACHED_BLOCK_TABLE,
            cached_length=CACHED_LENGTH,
        ),
    )

    with pytest.raises(ValueError, match="invariant"):
        RBLNOptimumModelRunner.try_copy_prefix_cached_kv(
            runner, model_input, scheduler_output
        )
