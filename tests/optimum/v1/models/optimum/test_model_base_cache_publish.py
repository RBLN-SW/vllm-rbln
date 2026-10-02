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
"""On a cache miss the compiled model is exported straight into a staging
directory beside the cache entry and published by one rename.

The alternative -- optimum's default ``TemporaryDirectory`` under ``$TMPDIR``
followed by ``save_pretrained`` into the cache -- holds the artifact twice on
disk and leaks the ``$TMPDIR`` copy whenever the server is stopped by a signal.
"""

import os
import types

import pytest
import torch
from transformers import LlamaConfig

from vllm_rbln.config import OptimumRBLNConfig
from vllm_rbln.model_executor.models.optimum import model_base
from vllm_rbln.model_executor.models.optimum.model_base import RBLNOptimumModelBase
from vllm_rbln.utils.optimum.paths import RBLN_CONFIG_FILE


def _fake_export(passed, *, write_config=True):
    """A model class whose export writes into ``model_save_dir`` like optimum
    does, and whose ``save_pretrained`` must not be reached."""

    class FakeRBLNModel:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            passed.update(kwargs)
            save_dir = kwargs["model_save_dir"]
            os.makedirs(save_dir, exist_ok=True)
            if write_config:
                with open(os.path.join(save_dir, RBLN_CONFIG_FILE), "w") as f:
                    f.write("{}")
                with open(os.path.join(save_dir, "prefill.rbln"), "wb") as f:
                    f.write(b"rbln")

            def save_pretrained(_path):
                raise AssertionError("the export is published by rename, not copied")

            return types.SimpleNamespace(
                rbln_config=types.SimpleNamespace(),
                save_pretrained=save_pretrained,
            )

    return FakeRBLNModel


def _init_model(monkeypatch, cached_model_path, model_cls) -> types.SimpleNamespace:
    monkeypatch.setattr(
        model_base.RBLNCompileSpec,
        "for_architecture",
        classmethod(
            lambda cls, *a, **k: types.SimpleNamespace(
                model_cls=model_cls, rbln_config={}
            )
        ),
    )
    monkeypatch.setattr(model_base, "get_attn_block_size", lambda cfg: 4096)

    hf_config = LlamaConfig(num_hidden_layers=2)
    hf_config.architectures = ["LlamaForCausalLM"]
    obj = RBLNOptimumModelBase.__new__(RBLNOptimumModelBase)
    obj.model_config = types.SimpleNamespace(
        hf_config=hf_config, model="repo", max_model_len=4096, dtype=torch.float16
    )
    obj.scheduler_config = types.SimpleNamespace(
        max_num_seqs=1, max_num_batched_tokens=128
    )
    obj.vllm_config = types.SimpleNamespace(
        additional_config=OptimumRBLNConfig(cached_model_path=cached_model_path),
        model_config=obj.model_config,
        scheduler_config=obj.scheduler_config,
        cache_config=types.SimpleNamespace(gpu_memory_utilization=0.9),
        ec_transfer_config=None,
    )
    obj.init_model()
    return obj


def test_export_lands_in_the_cache_without_a_temp_copy(monkeypatch, tmp_path):
    cache = str(tmp_path / "compiled_models" / "repo_abcd")
    passed = {}

    obj = _init_model(monkeypatch, cache, _fake_export(passed))

    # optimum was told where to write, and it was next to the cache entry --
    # the same filesystem, never $TMPDIR.
    assert passed["model_save_dir"] == cache + model_base._EXPORT_SUFFIX
    # The staging directory became the cache entry; nothing is left behind.
    assert model_base.is_compiled_dir(cache)
    assert not os.path.exists(cache + model_base._EXPORT_SUFFIX)
    assert obj.vllm_config.model_config.model == cache


def test_a_partial_cache_entry_from_a_dead_export_is_replaced(monkeypatch, tmp_path):
    cache = str(tmp_path / "repo_abcd")
    # A previous run died mid-export: the cache path exists but is not a
    # compiled dir (no rbln_config.json), and its staging dir is still there.
    os.makedirs(cache)
    open(os.path.join(cache, "prefill.rbln"), "wb").close()
    stale = cache + model_base._EXPORT_SUFFIX
    os.makedirs(stale)
    open(os.path.join(stale, "half.rbln"), "wb").close()

    _init_model(monkeypatch, cache, _fake_export({}))

    assert model_base.is_compiled_dir(cache)
    assert not os.path.exists(os.path.join(cache, "half.rbln"))
    assert not os.path.exists(stale)


def test_an_export_that_wrote_no_compiled_model_is_not_published(monkeypatch, tmp_path):
    cache = str(tmp_path / "repo_abcd")

    with pytest.raises(RuntimeError, match="left no compiled model"):
        _init_model(monkeypatch, cache, _fake_export({}, write_config=False))

    assert not model_base.is_compiled_dir(cache)
