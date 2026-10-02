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
"""Fixtures for driving `RBLNOptimumModelBase` through a cache miss.

The config is a real `VllmConfig`, so the platform hook, the block-size sync
and the compile spec all run as in a server; only the optimum-rbln export is
replaced, by a class that writes into `model_save_dir` the way optimum does.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import optimum.rbln
import pytest
import torch
from optimum.rbln import RBLNAutoModelForCausalLM
from transformers import PretrainedConfig
from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig

from vllm_rbln.model_executor.models.optimum.model_base import RBLNOptimumModelBase
from vllm_rbln.utils.optimum.paths import RBLN_CONFIG_FILE
from vllm_rbln.utils.optimum.registry import get_rbln_model_info


@pytest.fixture(autouse=True)
def set_npu_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    # The platform resolves the NPU name from the env on a host without one.
    monkeypatch.setenv("RBLN_FORCE_NPU_NAME", "RBLN-CA25")


@pytest.fixture
def vllm_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> VllmConfig:
    """A cache-miss config: the compiled-model cache lives under `tmp_path`
    and holds nothing, so building the config stages `cached_model_path`."""
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path / "vllm-cache"))
    return VllmConfig(
        model_config=ModelConfig(
            model="facebook/opt-125m", dtype=torch.float, max_model_len=2048
        ),
        scheduler_config=SchedulerConfig(
            max_num_seqs=1,
            max_num_batched_tokens=128,
            max_model_len=2048,
            is_encoder_decoder=False,
        ),
        cache_config=CacheConfig(block_size=16, cache_dtype="auto"),
        additional_config={
            "sub_block_size": 16,
            "optimum_overrides": {"prefill_chunk_size": 16},
        },
    )


class ExportedModel:
    """What the export hands back: enough for the model base to finish
    construction. It has no `save_pretrained`; the artifact is published by
    rename, so a copy attempt fails loudly."""

    rbln_config = object()


class FakeExport:
    """Stands in for the optimum-rbln model class on the cache-miss path.

    `from_pretrained` records its kwargs and writes a compiled directory into
    `model_save_dir`, as optimum does. Tests tune it through the class
    attributes: `writes_compiled_dir`, and `after_export`, which runs after the
    files are written and before the publish, where a sibling process can
    interleave.
    """

    calls: list[dict[str, Any]] = []
    writes_compiled_dir = True
    after_export: Callable[[], None] = staticmethod(lambda: None)

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: Any) -> ExportedModel:
        cls.calls.append(kwargs)
        if cls.writes_compiled_dir:
            write_compiled_dir(Path(kwargs["model_save_dir"]))
        cls.after_export()
        return ExportedModel()


@pytest.fixture
def fake_export(monkeypatch: pytest.MonkeyPatch) -> type[FakeExport]:
    """`FakeExport` with fresh state, installed where decoder architectures
    compile; other architectures are looked up on `optimum.rbln` by name, see
    `export_kwargs`."""
    monkeypatch.setattr(FakeExport, "calls", [])
    monkeypatch.setattr(FakeExport, "writes_compiled_dir", True)
    monkeypatch.setattr(FakeExport, "after_export", staticmethod(lambda: None))
    monkeypatch.setattr(
        RBLNAutoModelForCausalLM, "from_pretrained", FakeExport.from_pretrained
    )
    return FakeExport


def write_compiled_dir(path: Path, marker: str = "prefill.rbln") -> None:
    """Lay down what makes a directory a compiled model to `is_compiled_dir`."""
    path.mkdir(parents=True, exist_ok=True)
    (path / RBLN_CONFIG_FILE).write_text("{}")
    (path / marker).write_bytes(b"rbln")


@pytest.fixture
def export_kwargs(
    vllm_config: VllmConfig,
    fake_export: type[FakeExport],
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[PretrainedConfig], dict[str, Any]]:
    """Compile `hf_config` on a cache miss and return the kwargs the export
    received."""

    def compile_with(hf_config: PretrainedConfig) -> dict[str, Any]:
        _, model_cls_name = get_rbln_model_info(hf_config)
        monkeypatch.setattr(optimum.rbln, model_cls_name, fake_export)
        vllm_config.model_config.hf_config = hf_config
        RBLNOptimumModelBase(vllm_config)
        (kwargs,) = fake_export.calls
        return kwargs

    return compile_with
