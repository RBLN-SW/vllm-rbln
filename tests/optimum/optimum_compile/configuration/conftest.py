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
import pytest


@pytest.fixture
def set_npu_env_var(monkeypatch):
    monkeypatch.setenv("RBLN_FORCE_NPU_NAME", "RBLN-CA25")


@pytest.fixture
def compile_environment(monkeypatch, tmp_path, set_npu_env_var):
    from vllm_rbln.utils.optimum.configuration import sync

    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.setattr(sync, "load_compiled_rbln_config", lambda path: None)
    monkeypatch.setattr(sync, "resolve_compiled_model_path", lambda model, cache: None)
    monkeypatch.setattr(
        "vllm_rbln.utils.optimum.configuration.compile_config.default_prefill_chunk_size",
        lambda: 128,
    )
    monkeypatch.setattr(
        "vllm_rbln.platform.RblnPlatform.get_device_name", lambda: "RBLN-CA25"
    )


@pytest.fixture
def vllm_config_factory(compile_environment):
    from types import SimpleNamespace

    import torch
    from transformers import LlamaConfig

    from vllm_rbln.config import build_optimum_rbln_config
    from vllm_rbln.utils.optimum.configuration import sync

    def make_config(
        prefill_chunk_size=None,
        dtype=torch.bfloat16,
        *,
        max_num_batched_tokens=2048,
        max_num_scheduled_tokens=None,
        sync_config=True,
        prefix_caching=False,
        optimum_overrides=None,
        num_devices_per_local_rank=1,
        mamba_block_size=None,
        hf_config=None,
    ):
        additional_config = {
            "optimum_overrides": optimum_overrides or {},
            "num_devices_per_local_rank": num_devices_per_local_rank,
        }
        if prefill_chunk_size is not None:
            additional_config["prefill_chunk_size"] = prefill_chunk_size
        cfg = SimpleNamespace(
            model_config=SimpleNamespace(
                model="meta-llama/Llama-3.1-8B",
                max_model_len=8192,
                dtype=dtype,
                hf_config=(
                    hf_config
                    if hf_config is not None
                    else LlamaConfig(architectures=["LlamaForCausalLM"])
                ),
            ),
            scheduler_config=SimpleNamespace(
                max_num_seqs=4,
                max_num_batched_tokens=max_num_batched_tokens,
                max_num_scheduled_tokens=max_num_scheduled_tokens,
            ),
            cache_config=SimpleNamespace(
                block_size=8192,
                gpu_memory_utilization=0.9,
                user_specified_block_size=True,
                enable_prefix_caching=prefix_caching,
                num_gpu_blocks_override=None,
                user_specified_mamba_block_size=False,
                mamba_block_size=mamba_block_size,
            ),
            additional_config=build_optimum_rbln_config(additional_config),
        )

        if sync_config:
            sync.sync_vllm_and_optimum(cfg)
        return cfg

    return make_config
