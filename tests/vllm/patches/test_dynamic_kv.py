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
    """`--num-gpu-blocks-override` keeps the pool, but a dry run still asks the
    workers for their report."""

    @staticmethod
    def _engine(calls):
        return SimpleNamespace(
            model_executor=SimpleNamespace(
                collective_rpc=lambda method, args=(): calls.append(method) or [None]
            )
        )

    @staticmethod
    def _config():
        return SimpleNamespace(
            cache_config=SimpleNamespace(num_gpu_blocks_override=26, num_gpu_blocks=26)
        )

    def _run(self, monkeypatch, *, dry_run):
        calls: list = []
        kv_cache_config = SimpleNamespace(num_blocks=26)
        monkeypatch.setattr(
            dk,
            "engine_core_original_initialize_kv_caches",
            lambda self, cfg: kv_cache_config,
        )
        monkeypatch.setattr(dk.envs, "VLLM_RBLN_DYNAMIC_KV_CACHE_DRY_RUN", dry_run)
        out = dk.patched_initialize_kv_caches(self._engine(calls), self._config())
        assert out is kv_cache_config
        assert out.num_blocks == 26
        return calls

    def test_the_override_alone_skips_the_workers(self, monkeypatch):
        assert self._run(monkeypatch, dry_run=False) == []

    def test_a_dry_run_still_collects_the_report(self, monkeypatch):
        assert self._run(monkeypatch, dry_run=True) == ["compute_dynamic_kv_num_blocks"]


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


def test_a_decode_batch_can_need_more_than_one_request():
    # 64 sequences at one block each beat the 4 blocks a single request takes.
    minimum = minimum_kv_blocks(
        _config(8192, 32768, max_num_seqs=64), _kv(0, _full_spec(8192))
    )
    assert (minimum.one_request, minimum.decode_batch, minimum.needed) == (4, 64, 65)


def test_groups_sharing_the_pool_are_summed():
    # gpt-oss shape: a full group and a 128-token sliding window group at 8192.
    cfg = _config(8192, 32768, max_num_seqs=1, max_num_batched_tokens=512)
    minimum = minimum_kv_blocks(cfg, _kv(0, _full_spec(8192), _swa_spec(8192, 128)))
    # full: 4; sliding: cdiv(127 + 512, 8192) + 1 = 2 -> 6 for one request;
    # a decode step: 1 + (cdiv(128, 8192) + 1) = 3.
    assert (minimum.one_request, minimum.decode_batch, minimum.needed) == (6, 3, 7)


def test_a_window_spec_without_the_admission_method_is_loud():
    # A vLLM rename must not degrade the per-sequence term to one block.
    from vllm.v1.kv_cache_interface import SlidingWindowSpec

    class _Renamed(SlidingWindowSpec):
        max_admission_blocks_per_request = None  # type: ignore[assignment]

        def __init__(self):
            pass

        @property
        def page_size_bytes(self):
            return 1 << 20

        def max_memory_usage_bytes(self, cfg):
            return 4 << 20

    with pytest.raises(AttributeError, match="max_admission_blocks_per_request"):
        minimum_kv_blocks(_config(8192, 32768), _kv(0, _Renamed()))


def test_a_uniform_type_group_keeps_the_window_admission_cap():
    # An all-sliding-window model whose layers differ in hidden size reaches the
    # sizer wrapped. The wrapper is not a SlidingWindowSpec, so the guard above
    # cannot see it and the per-sequence term would drop to one block.
    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

    cfg = _config(128, 2048, max_num_seqs=2, max_num_batched_tokens=128)
    wrapped = UniformTypeKVCacheSpecs(
        block_size=128, kv_cache_specs={"layers.0.attn": _swa_spec(128, 128)}
    )
    minimum = minimum_kv_blocks(cfg, _kv(0, wrapped))
    # sliding: cdiv(127 + 128, 128) + 1 = 3 per sequence, over 2 sequences.
    assert (minimum.one_request, minimum.decode_batch) == (3, 6)
