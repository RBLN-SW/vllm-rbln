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
    check_enough_kv_cache_blocks_after_resize,
    resolve_rank_num_blocks,
)


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


def _config(
    block_size,
    max_model_len,
    max_num_seqs=1,
    max_num_batched_tokens=512,
    max_concurrent_batches=2,
):
    return SimpleNamespace(
        max_in_flight_tokens=max_concurrent_batches * max_num_batched_tokens,
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
    def max_memory_usage_bytes(cfg):
        held = min(
            window - 1 + cfg.max_in_flight_tokens,
            cfg.model_config.max_model_len,
        )
        return (_cdiv(held, block_size) + 1) * page

    return SimpleNamespace(
        page_size_bytes=page, max_memory_usage_bytes=max_memory_usage_bytes
    )


def _kv(num_blocks, *specs):
    return SimpleNamespace(
        num_blocks=num_blocks,
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=s) for s in specs],
    )


@pytest.mark.parametrize(
    ("block_size", "max_model_len", "num_blocks", "max_num_seqs"),
    [
        (1024, 32768, 33, 1),  # exactly one request plus the null block
        (1024, 32768, 34, 1),  # one to spare
        (8192, 32768, 5, 1),  # larger blocks
        (8192, 32768, 5, 64),  # a decode batch only caps concurrency
        (1024, 32768, 1548, 1),  # a real measured answer
        (128, 1000, 9, 1),  # cdiv rounds up: 1000/128 -> 8, +1 null
    ],
)
def test_accepts_a_pool_that_fits(block_size, max_model_len, num_blocks, max_num_seqs):
    check_enough_kv_cache_blocks_after_resize(
        _config(block_size, max_model_len, max_num_seqs=max_num_seqs),
        _kv(num_blocks, _full_spec(block_size)),
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
        check_enough_kv_cache_blocks_after_resize(
            _config(block_size, max_model_len), _kv(num_blocks, _full_spec(block_size))
        )
    # The message has to be actionable, like the upstream one it restores.
    assert str(num_blocks) in str(excinfo.value)
    assert "max_model_len" in str(excinfo.value)


def test_groups_sharing_the_pool_are_summed():
    # gpt-oss shape: a full group and a 128-token sliding window group at 8192.
    # full: 4; sliding: cdiv(127 + 2 * 512, 8192) + 1 = 2 -> 6, +1 null.
    cfg = _config(8192, 32768)
    specs = (_full_spec(8192), _swa_spec(8192, 128))
    check_enough_kv_cache_blocks_after_resize(cfg, _kv(7, *specs))
    with pytest.raises(ValueError, match="needs 7"):
        check_enough_kv_cache_blocks_after_resize(cfg, _kv(6, *specs))
