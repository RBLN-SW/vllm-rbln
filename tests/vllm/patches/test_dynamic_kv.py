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

"""The engine-side halves of the dynamic-KV handoff: reducing the per-rank
answers, and re-checking that the resized pool can hold one request."""

from types import SimpleNamespace

import pytest

import vllm_rbln.patches.dynamic_kv as dk
from vllm_rbln.patches.dynamic_kv import (
    assert_kv_cache_minimum,
    resolve_rank_num_blocks,
)
from vllm_rbln.v1.worker.utils import minimum_kv_blocks


class TestOverrideBranch:
    """`--num-gpu-blocks-override`, an unsupported configuration and an explicit
    off all keep the pool without asking the workers."""

    @staticmethod
    def _engine(calls):
        return SimpleNamespace(
            model_executor=SimpleNamespace(
                collective_rpc=lambda method, args=(): calls.append(method) or [None]
            )
        )

    @staticmethod
    def _config(dynamic=None):
        return SimpleNamespace(
            cache_config=SimpleNamespace(num_gpu_blocks_override=26, num_gpu_blocks=26),
            additional_config=SimpleNamespace(use_dynamic_kv_cache=dynamic),
        )

    def test_the_override_skips_the_workers(self, monkeypatch):
        calls: list = []
        kv_cache_config = SimpleNamespace(num_blocks=26)
        monkeypatch.setattr(
            dk,
            "engine_core_original_initialize_kv_caches",
            lambda self, cfg: kv_cache_config,
        )
        monkeypatch.setattr(dk, "dynamic_kv_unsupported_reason", lambda cfg: None)
        out = dk.patched_initialize_kv_caches(self._engine(calls), self._config())
        assert out is kv_cache_config
        assert out.num_blocks == 26
        assert calls == []

    def test_an_unsupported_config_skips_the_workers(self, monkeypatch):
        calls: list = []
        kv_cache_config = SimpleNamespace(num_blocks=26)
        monkeypatch.setattr(
            dk,
            "engine_core_original_initialize_kv_caches",
            lambda self, cfg: kv_cache_config,
        )
        monkeypatch.setattr(
            dk, "dynamic_kv_unsupported_reason", lambda cfg: "the optimum path"
        )
        config = SimpleNamespace(
            cache_config=SimpleNamespace(
                num_gpu_blocks_override=None, num_gpu_blocks=26
            ),
            additional_config=SimpleNamespace(use_dynamic_kv_cache=None),
        )
        assert dk.patched_initialize_kv_caches(self._engine(calls), config) is (
            kv_cache_config
        )
        assert calls == []

    def test_off_skips_the_workers_without_a_reason(self, monkeypatch, caplog):
        calls: list = []
        kv_cache_config = SimpleNamespace(num_blocks=26)
        monkeypatch.setattr(
            dk,
            "engine_core_original_initialize_kv_caches",
            lambda self, cfg: kv_cache_config,
        )
        monkeypatch.setattr(dk, "dynamic_kv_unsupported_reason", lambda cfg: None)
        with caplog.at_level("WARNING"):
            out = dk.patched_initialize_kv_caches(
                self._engine(calls), self._config(dynamic=False)
            )
        assert out is kv_cache_config
        assert calls == []
        assert caplog.records == []


class TestResolveRankNumBlocks:
    def test_all_none_means_the_path_is_not_in_play(self):
        assert resolve_rank_num_blocks([None, None]) is None

    def test_the_minimum_across_ranks_wins(self):
        assert resolve_rank_num_blocks([304, 274, 280, 274]) == 274

    def test_a_mixed_answer_is_one_ranks_failure(self):
        with pytest.raises(RuntimeError, match="some ranks"):
            resolve_rank_num_blocks([274, None])


def _cdiv(a, b):
    return -(-a // b)


def _config(block_size, max_model_len, max_num_seqs=1, max_num_batched_tokens=512):
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block_size),
        model_config=SimpleNamespace(max_model_len=max_model_len),
        scheduler_config=SimpleNamespace(
            max_num_seqs=max_num_seqs, max_num_batched_tokens=max_num_batched_tokens
        ),
    )


def _full_spec(block_size, page=1 << 20):
    return SimpleNamespace(
        page_size_bytes=page,
        max_memory_usage_bytes=lambda cfg: (
            _cdiv(cfg.model_config.max_model_len, block_size) * page
        ),
    )


def _swa_spec(block_size, window, page=1 << 20):
    def admission(max_num_batched_tokens, max_model_len):
        return (
            _cdiv(min(window - 1 + max_num_batched_tokens, max_model_len), block_size)
            + 1
        )

    return SimpleNamespace(
        page_size_bytes=page,
        max_admission_blocks_per_request=admission,
        max_memory_usage_bytes=lambda cfg: (
            admission(
                cfg.scheduler_config.max_num_batched_tokens,
                cfg.model_config.max_model_len,
            )
            * page
        ),
    )


def _kv(num_blocks, *specs):
    return SimpleNamespace(
        num_blocks=num_blocks,
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=s) for s in specs],
    )


@pytest.mark.parametrize(
    ("block_size", "max_model_len", "num_blocks"),
    [
        (1024, 32768, 33),  # exactly one request plus the null block
        (1024, 32768, 34),  # one to spare
        (8192, 32768, 5),  # larger blocks
        (1024, 32768, 1548),  # a real measured answer
        (128, 1000, 9),  # cdiv rounds up: 1000/128 -> 8, +1 null
    ],
)
def test_accepts_a_pool_that_fits(block_size, max_model_len, num_blocks):
    assert_kv_cache_minimum(
        _config(block_size, max_model_len), _kv(num_blocks, _full_spec(block_size))
    )


@pytest.mark.parametrize(
    ("block_size", "max_model_len", "num_blocks", "needed"),
    [
        (1024, 32768, 32, 33),  # forgets the null block
        (1024, 32768, 1, 33),
        (8192, 32768, 4, 5),
        (128, 1000, 8, 9),  # the rounded-up block is required
    ],
)
def test_rejects_a_pool_that_cannot_hold_one_request(
    block_size, max_model_len, num_blocks, needed
):
    with pytest.raises(ValueError, match=f"needs {needed}") as excinfo:
        assert_kv_cache_minimum(
            _config(block_size, max_model_len), _kv(num_blocks, _full_spec(block_size))
        )
    # The message has to be actionable, like the upstream one it restores.
    assert str(num_blocks) in str(excinfo.value)
    assert "max_model_len" in str(excinfo.value)


def test_a_decode_batch_does_not_raise_the_minimum():
    # Like upstream, a pool short of max_num_seqs sequences only caps
    # concurrency; the scheduler preempts instead of failing.
    cfg = _config(8192, 32768, max_num_seqs=64)
    minimum = minimum_kv_blocks(cfg, _kv(0, _full_spec(8192)))
    assert (minimum.one_request, minimum.needed) == (4, 5)
    assert_kv_cache_minimum(cfg, _kv(5, _full_spec(8192)))


def test_groups_sharing_the_pool_are_summed():
    # gpt-oss shape: a full group and a 128-token sliding window group at 8192.
    cfg = _config(8192, 32768, max_num_seqs=128, max_num_batched_tokens=512)
    minimum = minimum_kv_blocks(cfg, _kv(0, _full_spec(8192), _swa_spec(8192, 128)))
    # full: 4; sliding: cdiv(127 + 512, 8192) + 1 = 2 -> 6 for one request.
    assert (minimum.one_request, minimum.needed) == (6, 7)
