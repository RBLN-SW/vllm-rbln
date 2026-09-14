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
    assert_kv_cache_fits_one_request,
    resolve_rank_num_blocks,
)


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


def _config(block_size, max_model_len):
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block_size),
        model_config=SimpleNamespace(max_model_len=max_model_len),
    )


def _kv(num_blocks):
    return SimpleNamespace(num_blocks=num_blocks)


@pytest.mark.parametrize(
    ("block_size", "max_model_len", "num_blocks"),
    [
        (1024, 32768, 32),  # exactly one request
        (1024, 32768, 33),  # one to spare
        (8192, 32768, 4),  # exactly one request, larger blocks
        (1024, 32768, 1548),  # a real measured answer
        (128, 1000, 8),  # cdiv rounds up: 1000/128 -> 8
    ],
)
def test_accepts_a_pool_that_fits(block_size, max_model_len, num_blocks):
    assert_kv_cache_fits_one_request(
        _config(block_size, max_model_len), _kv(num_blocks)
    )


@pytest.mark.parametrize(
    ("block_size", "max_model_len", "num_blocks", "needed"),
    [
        (1024, 32768, 31, 32),  # one block short
        (1024, 32768, 1, 32),
        (8192, 32768, 3, 4),
        (128, 1000, 7, 8),  # the rounded-up block is required
    ],
)
def test_rejects_a_pool_that_cannot_hold_one_request(
    block_size, max_model_len, num_blocks, needed
):
    with pytest.raises(ValueError, match=f"needs {needed}") as excinfo:
        assert_kv_cache_fits_one_request(
            _config(block_size, max_model_len), _kv(num_blocks)
        )
    # The message has to be actionable, like the upstream one it restores.
    assert str(num_blocks) in str(excinfo.value)
    assert "max_model_len" in str(excinfo.value)
