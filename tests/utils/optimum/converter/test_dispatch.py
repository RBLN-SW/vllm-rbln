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

import os
from unittest.mock import patch

import pytest

from vllm_rbln.utils.optimum.converter import dispatch
from vllm_rbln.utils.optimum.converter.dispatch import (
    _generate_model_path_name,
    _strip_runtime_only_keys,
    sync_vllm_and_optimum,
)


class TestStripRuntimeOnlyKeys:
    def test_removes_runtime_keys_at_top_level(self):
        cfg = {
            "create_runtimes": True,
            "devices": [0, 1],
            "kvcache_num_blocks": 32,
            "batch_size": 4,
            "max_seq_len": 1024,
        }
        assert _strip_runtime_only_keys(cfg) == {
            "batch_size": 4,
            "max_seq_len": 1024,
        }

    def test_removes_recursively_inside_nested_dicts(self):
        cfg = {
            "language_model": {
                "create_runtimes": True,
                "devices": [0],
                "kvcache_num_blocks": 16,
                "batch_size": 2,
            },
            "vision_model": {
                "devices": [1],
                "max_seq_len": 256,
            },
        }
        assert _strip_runtime_only_keys(cfg) == {
            "language_model": {"batch_size": 2},
            "vision_model": {"max_seq_len": 256},
        }

    def test_removes_recursively_inside_lists_of_dicts(self):
        cfg = {
            "experts": [
                {"devices": [0], "rank": 0},
                {"devices": [1], "rank": 1},
            ]
        }
        assert _strip_runtime_only_keys(cfg) == {
            "experts": [{"rank": 0}, {"rank": 1}]
        }

    def test_leaves_unrelated_keys_untouched(self):
        cfg = {"batch_size": 4, "tensor_parallel_size": 2}
        assert _strip_runtime_only_keys(cfg) == cfg

    def test_handles_non_container_values(self):
        # Scalars / strings are returned as-is when traversed inside lists.
        cfg = {"x": [1, 2, "three", {"devices": [0], "keep": True}]}
        assert _strip_runtime_only_keys(cfg) == {
            "x": [1, 2, "three", {"keep": True}]
        }


class TestGenerateModelPathName:
    @pytest.fixture(autouse=True)
    def _stub_tp_size(self, monkeypatch):
        # Pin tp_size so tests don't depend on env state.
        monkeypatch.setattr(dispatch.envs, "VLLM_RBLN_TP_SIZE", 1)

    def test_deterministic(self, vllm_config_factory):
        cfg = vllm_config_factory(
            model="meta-llama/Llama-3-8B",
            max_model_len=2048,
            block_size=128,
            max_num_seqs=4,
        )
        assert _generate_model_path_name(cfg) == _generate_model_path_name(cfg)

    def test_runtime_only_keys_do_not_affect_hash(self, vllm_config_factory):
        base = vllm_config_factory(
            additional_config={"rbln_config": {"batch_size": 4}}
        )
        with_runtime = vllm_config_factory(
            additional_config={
                "rbln_config": {
                    "batch_size": 4,
                    "create_runtimes": False,
                    "devices": [0, 1],
                    "kvcache_num_blocks": 64,
                }
            }
        )
        assert _generate_model_path_name(base) == _generate_model_path_name(
            with_runtime
        )

    @pytest.mark.parametrize(
        "field, value",
        [
            ("model", "meta-llama/Llama-3-70B"),
            ("max_model_len", 4096),
            ("block_size", 256),
            ("max_num_seqs", 16),
        ],
    )
    def test_user_visible_fields_change_hash(
        self, vllm_config_factory, field, value
    ):
        a = vllm_config_factory()
        b_kwargs = {field: value}
        # remap the schema-level fields to the factory kwargs.
        if field == "max_num_seqs":
            b = vllm_config_factory(max_num_seqs=value)
        elif field == "block_size":
            b = vllm_config_factory(block_size=value)
        elif field == "max_model_len":
            b = vllm_config_factory(max_model_len=value)
        else:
            b = vllm_config_factory(**b_kwargs)
        assert _generate_model_path_name(a) != _generate_model_path_name(b)

    def test_tp_size_changes_hash(self, vllm_config_factory, monkeypatch):
        cfg = vllm_config_factory()
        before = _generate_model_path_name(cfg)
        monkeypatch.setattr(dispatch.envs, "VLLM_RBLN_TP_SIZE", 4)
        after = _generate_model_path_name(cfg)
        assert before != after

    def test_non_runtime_rbln_config_changes_hash(self, vllm_config_factory):
        a = vllm_config_factory(additional_config={"rbln_config": {"attn_impl": "eager"}})
        b = vllm_config_factory(
            additional_config={"rbln_config": {"attn_impl": "flash_attn"}}
        )
        assert _generate_model_path_name(a) != _generate_model_path_name(b)

    def test_model_name_is_sanitized(self, vllm_config_factory):
        cfg = vllm_config_factory(model="org/name:revision")
        path = _generate_model_path_name(cfg)
        assert "/" not in path
        assert ":" not in path
        # sanitized prefix preserved
        assert path.startswith("org_name_revision_")


class TestSyncVllmAndOptimum:
    """Orchestration: precompiled / cache-hit / cache-miss branches.

    We patch :func:`load_compiled_rbln_config` (the only filesystem touch
    inside :func:`_resolve_rbln_config`) and the two sync helpers.
    """

    def test_precompiled_dispatches_to_optimum(self, vllm_config_factory):
        vllm_config = vllm_config_factory(
            model="/precompiled/model",
            additional_config={"rbln_config": {"devices": [0]}},
        )
        compiled_rbln_config = {"batch_size": 4, "max_seq_len": 1024}
        with (
            patch.object(
                dispatch,
                "load_compiled_rbln_config",
                return_value=compiled_rbln_config,
            ) as load_mock,
            patch.object(dispatch, "sync_from_optimum") as opt_mock,
            patch.object(dispatch, "sync_from_vllm") as vllm_mock,
        ):
            sync_vllm_and_optimum(vllm_config)

        load_mock.assert_called_once_with(vllm_config)
        opt_mock.assert_called_once_with(vllm_config, compiled_rbln_config)
        vllm_mock.assert_not_called()
        # cached_model_path is not staged for the precompiled branch.
        assert "cached_model_path" not in vllm_config.additional_config

    def test_cache_hit_rewrites_model_and_dispatches_to_optimum(
        self, vllm_config_factory, tmp_path, monkeypatch
    ):
        cache_root = tmp_path / "cache"
        monkeypatch.setattr(dispatch.envs, "VLLM_CACHE_ROOT", str(cache_root))
        monkeypatch.setattr(dispatch.envs, "VLLM_RBLN_TP_SIZE", 1)

        vllm_config = vllm_config_factory(
            model="meta-llama/Llama-3-8B",
            additional_config={},
        )
        # Pre-create the rbln_config.json under the hashed cache path.
        expected_dir = os.path.join(
            str(cache_root),
            "compiled_models",
            _generate_model_path_name(vllm_config),
        )
        os.makedirs(expected_dir, exist_ok=True)
        rbln_path = os.path.join(expected_dir, "rbln_config.json")
        open(rbln_path, "w").close()

        compiled_rbln_config = {"batch_size": 4, "max_seq_len": 1024}

        with (
            # First call (initial precompiled probe) returns None (no
            # rbln_config in user-supplied model dir); second call (cache
            # hit) returns the loaded config.
            patch.object(
                dispatch,
                "load_compiled_rbln_config",
                side_effect=[None, compiled_rbln_config],
            ) as load_mock,
            patch.object(dispatch, "sync_from_optimum") as opt_mock,
            patch.object(dispatch, "sync_from_vllm") as vllm_mock,
        ):
            sync_vllm_and_optimum(vllm_config)

        assert load_mock.call_count == 2
        opt_mock.assert_called_once_with(vllm_config, compiled_rbln_config)
        vllm_mock.assert_not_called()
        assert vllm_config.additional_config["cached_model_path"] == expected_dir
        # cache-hit rewrites the model path so optimum loads from cache.
        assert vllm_config.model_config.model == expected_dir

    def test_cache_miss_stages_path_and_dispatches_to_vllm(
        self, vllm_config_factory, tmp_path, monkeypatch
    ):
        cache_root = tmp_path / "cache"
        monkeypatch.setattr(dispatch.envs, "VLLM_CACHE_ROOT", str(cache_root))
        monkeypatch.setattr(dispatch.envs, "VLLM_RBLN_TP_SIZE", 1)

        vllm_config = vllm_config_factory(
            model="meta-llama/Llama-3-8B",
            additional_config={},
        )
        expected_dir = os.path.join(
            str(cache_root),
            "compiled_models",
            _generate_model_path_name(vllm_config),
        )

        with (
            patch.object(
                dispatch, "load_compiled_rbln_config", return_value=None
            ) as load_mock,
            patch.object(dispatch, "sync_from_optimum") as opt_mock,
            patch.object(dispatch, "sync_from_vllm") as vllm_mock,
        ):
            sync_vllm_and_optimum(vllm_config)

        load_mock.assert_called_once_with(vllm_config)
        vllm_mock.assert_called_once_with(vllm_config)
        opt_mock.assert_not_called()
        # Cache miss stages the path but does not rewrite model_config.model
        # (compilation will write artefacts there later).
        assert vllm_config.additional_config["cached_model_path"] == expected_dir
        assert vllm_config.model_config.model == "meta-llama/Llama-3-8B"

    def test_load_failure_is_wrapped(self, vllm_config_factory):
        vllm_config = vllm_config_factory()
        with patch.object(
            dispatch, "load_compiled_rbln_config", side_effect=OSError("boom")
        ):
            with pytest.raises(RuntimeError):
                sync_vllm_and_optimum(vllm_config)
