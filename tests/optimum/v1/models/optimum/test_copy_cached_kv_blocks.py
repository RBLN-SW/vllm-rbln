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

"""The prefix-cache copy picks the runtime KV cache API by the compiled cache layout."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm_rbln.model_executor.models.optimum import model_base
from vllm_rbln.model_executor.models.optimum.model_base import (
    KVCacheCopyError,
    RBLNOptimumDecoderMixin,
)

# Each layer's K and V in one tensor, and K and V as separate tensors.
PACKED_SHAPE = [4, 2, 2, 1, 256, 64]
SPLIT_SHAPE = [4, 2, 256, 64]


class SplitOnlyRuntime:
    """A rebel runtime from before the packed KV cache API."""

    def __init__(self) -> None:
        self.copies: list[tuple[str, int, int, int]] = []

    def _copy_kv_cache(self, src: int, dst: int, size: int) -> None:
        self.copies.append(("split", src, dst, size))


class Runtime(SplitOnlyRuntime):
    def _copy_packed_kv_cache(self, src: int, dst: int, size: int) -> None:
        self.copies.append(("packed", src, dst, size))


class DecoderModel(nn.Module, RBLNOptimumDecoderMixin):
    def __init__(self, cache_shape: list[int], runtime: SplitOnlyRuntime) -> None:
        super().__init__()
        self.model = SimpleNamespace(
            rbln_config=SimpleNamespace(
                cache_metas=[SimpleNamespace(shape=cache_shape)]
            ),
            prefill_decoder=SimpleNamespace(runtime=runtime),
        )
        self.setup_decoder_mixin(
            attn_impl=None,
            vocab_size=8,
            use_multiple_decoder=False,
            default_batch_size=1,
            decoder_batch_sizes=[1],
        )


@pytest.fixture(autouse=True)
def no_logits_processor(monkeypatch):
    # The real one needs a current vLLM config, and these tests never compute logits.
    monkeypatch.setattr(model_base, "LogitsProcessor", lambda *args, **kwargs: None)


@pytest.mark.parametrize(
    ("cache_shape", "api"), [(SPLIT_SHAPE, "split"), (PACKED_SHAPE, "packed")]
)
def test_copies_through_the_api_of_the_cache_layout(cache_shape, api):
    runtime = Runtime()
    model = DecoderModel(cache_shape, runtime)

    model.copy_cached_kv_blocks([3, 5], [16, 7], torch.tensor([1, 2]))

    assert runtime.copies == [(api, 3, 1, 16), (api, 5, 2, 7)]


def test_split_cache_copies_on_a_runtime_without_the_packed_api():
    runtime = SplitOnlyRuntime()
    model = DecoderModel(SPLIT_SHAPE, runtime)

    model.copy_cached_kv_blocks([3], [16], torch.tensor([1]))

    assert runtime.copies == [("split", 3, 1, 16)]


def test_packed_cache_on_a_runtime_without_the_packed_api_fails_the_copy():
    runtime = SplitOnlyRuntime()
    model = DecoderModel(PACKED_SHAPE, runtime)

    # The runner falls back to a full prefill on this error.
    with pytest.raises(KVCacheCopyError, match="_copy_packed_kv_cache"):
        model.copy_cached_kv_blocks([3], [16], torch.tensor([1]))
    assert runtime.copies == []
