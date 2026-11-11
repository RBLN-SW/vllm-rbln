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

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, PaliGemmaConfig, Qwen3Config, WhisperConfig

from vllm_rbln.config import OptimumRBLNCompileConfig
from vllm_rbln.utils.optimum.configuration import sync
from vllm_rbln.utils.optimum.configuration.compile_config import compile_cache_name
from vllm_rbln.utils.optimum.configuration.readers import (
    load_compiled_rbln_config,
)
from vllm_rbln.utils.optimum.path import (
    get_compile_cache_path,
    resolve_compiled_model_path,
)

pytestmark = pytest.mark.usefixtures("compile_environment")


class TestSync:
    @pytest.mark.parametrize("prefix_caching", [False, True])
    def test_paligemma_uses_final_chunk_before_cache_lookup(
        self, monkeypatch, prefix_caching, vllm_config_factory
    ):
        observed = []

        def generate_name(inputs):
            observed.append(
                inputs.rbln_config.get("language_model", inputs.rbln_config)[
                    "prefill_chunk_size"
                ]
            )
            return compile_cache_name(inputs)

        monkeypatch.setattr(sync, "compile_cache_name", generate_name)
        cfg = vllm_config_factory(
            prefix_caching=prefix_caching,
            hf_config=PaliGemmaConfig(
                architectures=["PaliGemmaForConditionalGeneration"]
            ),
        )
        inputs = cfg.additional_config.compile_config.rbln_config["language_model"]
        assert observed == [inputs["prefill_chunk_size"]] == [8192]
        assert cfg.cache_config.block_size == 8192
        sync.sync_vllm_and_optimum(cfg)
        assert cfg.scheduler_config.max_num_batched_tokens == 8192

    def test_paligemma_rejects_user_chunk_that_differs_from_final_inputs(
        self, vllm_config_factory
    ):
        with pytest.raises(ValueError, match="prefill_chunk_size"):
            vllm_config_factory(
                512,
                hf_config=PaliGemmaConfig(
                    architectures=["PaliGemmaForConditionalGeneration"]
                ),
            )

    def test_whisper_runtime_uses_final_compile_lengths(self, vllm_config_factory):
        cfg = vllm_config_factory(
            hf_config=WhisperConfig(
                architectures=["WhisperForConditionalGeneration"],
                max_target_positions=8192,
            )
        )
        inputs = cfg.additional_config.compile_config.rbln_config
        assert inputs["dec_max_seq_len"] == cfg.model_config.max_model_len == 8192
        assert inputs["batch_size"] == cfg.scheduler_config.max_num_seqs == 4

    def test_paligemma_cache_hit_preserves_the_final_vllm_config_updates(
        self, monkeypatch, vllm_config_factory
    ):
        hf_config = PaliGemmaConfig(architectures=["PaliGemmaForConditionalGeneration"])
        cfg = vllm_config_factory(hf_config=hf_config)
        cache_path = Path(cfg.additional_config.artifact_path)
        cache_path.mkdir(parents=True)
        metadata = cfg.additional_config.compile_config.rbln_config
        metadata = json.loads(json.dumps(metadata))
        metadata["language_model"].update(
            kvcache_block_size=8192, kvcache_num_blocks=4, dtype="bfloat16"
        )
        (cache_path / "rbln_config.json").write_text(json.dumps(metadata))
        monkeypatch.setattr(
            sync, "load_compiled_rbln_config", load_compiled_rbln_config
        )
        monkeypatch.setattr(
            sync,
            "resolve_compiled_model_path",
            lambda model, cache: str(cache_path) if cache is not None else None,
        )
        cached = vllm_config_factory(hf_config=hf_config)
        assert cached.additional_config.artifact_path == str(cache_path)
        assert cached.additional_config.precompiled
        assert (
            cached.scheduler_config.max_num_batched_tokens
            == cfg.scheduler_config.max_num_batched_tokens
            == 8192
        )
        assert cached.cache_config.block_size == cfg.cache_config.block_size == 8192

    @pytest.mark.parametrize(
        ("runner", "reranker", "expected_arch", "expected_cls", "expected_tokens"),
        [
            ("pooling", False, "Qwen3Model", "RBLNQwen3Model", 1024),
            ("pooling", True, "Qwen3ForCausalLM", "RBLNAutoModelForCausalLM", 128),
            ("generate", False, "Qwen3ForCausalLM", "RBLNAutoModelForCausalLM", 128),
        ],
    )
    def test_qwen3_architecture_is_resolved_before_cache_lookup(
        self,
        monkeypatch,
        runner,
        reranker,
        expected_arch,
        expected_cls,
        expected_tokens,
    ):
        from vllm_rbln.platform.optimum_impl import check_and_update

        hf_config = Qwen3Config(
            architectures=[
                "Qwen3ForSequenceClassification" if reranker else "Qwen3ForCausalLM"
            ],
            is_original_qwen3_reranker=reranker,
            classifier_from_token=["no", "yes"] if reranker else None,
        )
        cfg = SimpleNamespace(
            model_config=SimpleNamespace(
                model="Qwen/Qwen3-0.6B",
                architectures=hf_config.architectures,
                runner_type=runner,
                hf_config=hf_config,
                hf_text_config=hf_config,
                multimodal_config=None,
                max_model_len=1024,
                dtype=torch.bfloat16,
            ),
            parallel_config=SimpleNamespace(
                worker_cls="auto", tensor_parallel_size=1, pipeline_parallel_size=1
            ),
            scheduler_config=SimpleNamespace(
                max_num_seqs=1, max_num_batched_tokens=2048, async_scheduling=False
            ),
            cache_config=SimpleNamespace(
                block_size=1024,
                gpu_memory_utilization=0.9,
                user_specified_block_size=True,
                enable_prefix_caching=False,
                num_gpu_blocks_override=None,
            ),
            additional_config={},
            speculative_config=None,
        )
        observed = []

        def generate_name(inputs):
            observed.append(
                (
                    list(inputs.hf_config.architectures),
                    inputs.model_cls,
                    inputs.rbln_config.get("prefill_chunk_size", 1024),
                )
            )
            return compile_cache_name(inputs)

        monkeypatch.setattr(sync, "compile_cache_name", generate_name)
        check_and_update(cfg)
        assert observed == [([expected_arch], expected_cls, expected_tokens)]

    def test_repeated_sync_preserves_the_compile_config(self, vllm_config_factory):
        cfg = vllm_config_factory()
        inputs = cfg.additional_config.compile_config
        path = cfg.additional_config.artifact_path
        cfg.additional_config.precompiled = True
        sync.sync_vllm_and_optimum(cfg)
        assert cfg.additional_config.compile_config is inputs
        assert cfg.additional_config.artifact_path == path
        assert not cfg.additional_config.precompiled

    @pytest.mark.parametrize(
        ("user_tokens", "overrides", "expected"),
        [
            (None, {}, 128),
            (None, {"prefill_chunk_size": 256}, 256),
            (512, {}, 512),
            (512, {"prefill_chunk_size": 512}, 512),
        ],
    )
    def test_chunk_size_is_resolved_before_cache_lookup(
        self, monkeypatch, user_tokens, overrides, expected, vllm_config_factory
    ):
        observed = []

        def generate_name(inputs):
            observed.append(
                inputs.rbln_config.get("language_model", inputs.rbln_config)[
                    "prefill_chunk_size"
                ]
            )
            return compile_cache_name(inputs)

        monkeypatch.setattr(sync, "compile_cache_name", generate_name)
        cfg = vllm_config_factory(user_tokens, optimum_overrides=overrides)
        assert observed == [expected]
        assert cfg.scheduler_config.max_num_batched_tokens == expected
        assert (
            cfg.additional_config.compile_config.rbln_config["prefill_chunk_size"]
            == expected
        )

    @pytest.mark.parametrize(
        ("dtype", "devices", "device_overrides", "expected_devices"),
        [
            (torch.bfloat16, 2, {}, 2),
            (torch.bfloat16, 1, {"num_devices": 4}, 4),
            (torch.float32, 3, {}, 3),
        ],
    )
    def test_compile_config_is_resolved_before_cache_lookup(
        self,
        monkeypatch,
        dtype,
        devices,
        device_overrides,
        expected_devices,
        vllm_config_factory,
    ):
        observed = []

        def generate_name(inputs):
            observed.append(
                (
                    inputs.rbln_config["batch_size"],
                    inputs.rbln_config["max_seq_len"],
                    inputs.dtype,
                    inputs.rbln_config["num_devices"],
                )
            )
            return compile_cache_name(inputs)

        monkeypatch.setattr(sync, "compile_cache_name", generate_name)
        cfg = vllm_config_factory(
            dtype=dtype,
            num_devices_per_local_rank=devices,
            optimum_overrides={
                "batch_size": 2,
                "max_seq_len": 4096,
                "kvcache_block_size": 4096,
                **device_overrides,
            },
        )
        assert observed == [(2, 4096, dtype, expected_devices)]
        inputs = cfg.additional_config.compile_config
        assert inputs.dtype == dtype
        assert (
            inputs.rbln_config["batch_size"] == cfg.scheduler_config.max_num_seqs == 2
        )
        assert (
            inputs.rbln_config["max_seq_len"] == cfg.model_config.max_model_len == 4096
        )
        assert inputs.rbln_config["num_devices"] == expected_devices
        assert cfg.additional_config.num_devices_per_local_rank == expected_devices

    def test_repeated_sync_preserves_the_attention_block_size(
        self, vllm_config_factory
    ):
        cfg = vllm_config_factory(prefix_caching=True)
        sync.sync_vllm_and_optimum(cfg)
        assert cfg.additional_config.attn_block_size == 8192
        assert cfg.cache_config.block_size == 128

    @pytest.mark.parametrize("location", ["model", "cache", "generated"])
    def test_cache_metadata_is_loaded_without_replacing_the_model_id(
        self, monkeypatch, vllm_config_factory, location
    ):
        cfg = vllm_config_factory()
        cache_path = Path(cfg.additional_config.artifact_path)
        cache_path.mkdir(parents=True)
        metadata = dict(cfg.additional_config.compile_config.rbln_config)
        metadata.update(prefill_chunk_size=256, kvcache_num_blocks=4)
        (cache_path / "rbln_config.json").write_text(json.dumps(metadata))
        if location == "model":
            cfg.model_config.model = str(cache_path)
        if location != "cache":
            cfg.additional_config.artifact_path = None
        source_model = cfg.model_config.model
        monkeypatch.setattr(
            sync, "load_compiled_rbln_config", load_compiled_rbln_config
        )
        monkeypatch.setattr(
            sync,
            "resolve_compiled_model_path",
            resolve_compiled_model_path,
        )
        with monkeypatch.context() as scoped:
            scoped.setattr(
                sync,
                "load_compiled_rbln_config",
                lambda path: pytest.fail(
                    "path resolution must not read rbln_config.json"
                ),
            )
            assert get_compile_cache_path(
                compile_cache_name(cfg.additional_config.compile_config),
                str(cache_path.parent.parent),
            ) == str(cache_path)
        assert cfg.model_config.model == source_model

        synced = []
        monkeypatch.setattr(
            sync,
            "apply_runtime_config",
            lambda cfg, data, **kw: synced.append(data),
        )
        monkeypatch.setattr(
            sync,
            "prepare_compile_config",
            lambda cfg: pytest.fail(
                "cached metadata must bypass compile config preparation"
            ),
        )
        sync.sync_vllm_and_optimum(cfg)
        assert len(synced) == 1
        assert synced[0].prefill_chunk_size == metadata["prefill_chunk_size"]
        assert cfg.model_config.model == source_model
        assert cfg.additional_config.artifact_path == str(cache_path)
        assert cfg.additional_config.precompiled


def test_initial_cache_hit_applies_vllm_config_updates_once(
    monkeypatch, vllm_config_factory
):
    from vllm_rbln.utils.optimum.configuration import runtime_config as runtime_module

    calls = []
    selected_configs = []
    apply = sync.apply_runtime_config

    def record_apply(cfg, data, **kwargs):
        selected_configs.append(data)
        return apply(cfg, data, **kwargs)

    monkeypatch.setattr(sync, "apply_runtime_config", record_apply)
    original = runtime_module.update_block_size

    def record(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime_module, "update_block_size", record)
    metadata = {
        "batch_size": 2,
        "max_seq_len": 4096,
        "kvcache_block_size": 4096,
        "kvcache_num_blocks": 2,
        "prefill_chunk_size": 256,
        "dtype": "float32",
        "num_devices": 2,
    }
    monkeypatch.setattr(sync, "load_compiled_rbln_config", lambda path: metadata)
    cfg = vllm_config_factory(
        mamba_block_size=8192,
        optimum_overrides={"device": [0], "create_runtimes": False},
    )
    assert len(selected_configs) == 1
    assert selected_configs[0].max_num_seqs == metadata["batch_size"]
    assert selected_configs[0].num_devices_per_local_rank == metadata["num_devices"]
    assert len(calls) == 1
    assert cfg.scheduler_config.max_num_seqs == 2
    assert cfg.scheduler_config.max_num_batched_tokens == 256
    assert cfg.model_config.max_model_len == 4096
    assert cfg.model_config.dtype == torch.float32
    assert cfg.additional_config.num_devices_per_local_rank == 2
    assert cfg.additional_config.optimum_overrides == {"device": [0]}
    assert cfg.additional_config.artifact_path is not None
    assert cfg.additional_config.precompiled
    assert cfg.cache_config.mamba_block_size == 4096
    assert cfg.cache_config.num_gpu_blocks == 3
    sync.sync_vllm_and_optimum(cfg)
    assert cfg.cache_config.num_gpu_blocks == 3


def test_builder_does_not_apply_vllm_config_updates(vllm_config_factory):
    cfg = vllm_config_factory()
    cfg.additional_config.compile_config = None
    cfg.additional_config.optimum_overrides = {"batch_size": 2, "max_seq_len": 4096}
    from vllm_rbln.utils.optimum.configuration.compile_config import (
        prepare_compile_config,
    )

    inputs = prepare_compile_config(cfg)
    assert cfg.scheduler_config.max_num_seqs == 4
    assert cfg.model_config.max_model_len == 8192
    assert cfg.scheduler_config.max_num_batched_tokens == 128
    assert cfg.additional_config.compile_config is None
    assert inputs.rbln_config["batch_size"] == 2


def test_compile_config_does_not_share_mutable_overrides(vllm_config_factory):
    overrides = {"custom_options": {"values": [1, 2]}}
    cfg = vllm_config_factory(optimum_overrides=overrides)
    overrides["custom_options"]["values"].append(3)
    assert cfg.additional_config.compile_config.rbln_config["custom_options"] == {
        "values": [1, 2]
    }


@pytest.mark.parametrize(
    "hf_config",
    [
        LlamaConfig(architectures=["LlamaForCausalLM"]),
        PaliGemmaConfig(architectures=["PaliGemmaForConditionalGeneration"]),
        WhisperConfig(
            architectures=["WhisperForConditionalGeneration"],
            max_target_positions=8192,
        ),
        Qwen3Config(architectures=["Qwen3Model"]),
    ],
)
def test_vllm_config_is_synced_after_cache_lookup(
    monkeypatch, hf_config, vllm_config_factory
):
    events = []
    selected = []
    generate = sync.compile_cache_name
    apply = sync.apply_runtime_config

    def record_lookup(inputs):
        events.append("lookup")
        return generate(inputs)

    def record_apply(cfg, data, **kwargs):
        events.append("apply")
        selected.append(data)
        return apply(cfg, data, **kwargs)

    monkeypatch.setattr(sync, "compile_cache_name", record_lookup)
    monkeypatch.setattr(sync, "apply_runtime_config", record_apply)
    cfg = vllm_config_factory(hf_config=hf_config)
    assert events == ["lookup", "apply"]
    assert selected[0].max_num_seqs == 4
    assert selected[0].max_model_len == 8192
    assert cfg.scheduler_config.max_num_seqs == 4
    assert cfg.model_config.max_model_len == 8192


@pytest.mark.parametrize("precompiled", [False, True])
def test_compile_config_survives_worker_serialization(
    monkeypatch, vllm_config_factory, precompiled
):
    import pickle

    cfg = vllm_config_factory(
        hf_config=PaliGemmaConfig(architectures=["PaliGemmaForConditionalGeneration"])
    )
    if precompiled:
        metadata = json.loads(
            json.dumps(cfg.additional_config.compile_config.rbln_config)
        )
        metadata["language_model"]["kvcache_num_blocks"] = 4
        monkeypatch.setattr(
            sync,
            "load_compiled_rbln_config",
            lambda path: metadata,
        )
        sync.sync_vllm_and_optimum(cfg)
    original_path = cfg.additional_config.artifact_path
    cfg.additional_config = pickle.loads(pickle.dumps(cfg.additional_config))
    inputs = cfg.additional_config.compile_config
    assert isinstance(inputs, OptimumRBLNCompileConfig)
    assert inputs.rbln_config["language_model"]["prefill_chunk_size"] == 8192
    assert cfg.additional_config.artifact_path == original_path
    assert cfg.additional_config.precompiled is precompiled
    sync.sync_vllm_and_optimum(cfg)
    assert cfg.additional_config.artifact_path == original_path
    assert cfg.additional_config.precompiled is precompiled
    assert cfg.scheduler_config.max_num_batched_tokens == 8192


def test_vllm_config_updates_include_submodule_devices_and_image_buckets(
    vllm_config_factory,
):
    cfg = vllm_config_factory(
        hf_config=PaliGemmaConfig(architectures=["PaliGemmaForConditionalGeneration"]),
        optimum_overrides={
            "language_model": {"device": [0, 1], "image_prefill_chunk_size": 256}
        },
    )
    inputs = cfg.additional_config.compile_config
    language_model = inputs.rbln_config["language_model"]
    assert language_model["num_devices"] == 2
    assert cfg.additional_config.num_devices_per_local_rank == 2
    assert cfg.additional_config.image_prefill_chunk_size == [256]


def test_sync_stores_the_builder_inputs_without_repacking(
    monkeypatch, vllm_config_factory
):
    built_inputs = []
    build = sync.prepare_compile_config

    def record(*args, **kwargs):
        inputs = build(*args, **kwargs)
        built_inputs.append(inputs)
        return inputs

    monkeypatch.setattr(sync, "prepare_compile_config", record)
    cfg = vllm_config_factory()
    assert len(built_inputs) == 1
    assert isinstance(cfg.additional_config.compile_config, OptimumRBLNCompileConfig)
    assert cfg.additional_config.compile_config is built_inputs[0]


def test_vllm_config_updates_are_derived_from_compile_config(vllm_config_factory):
    cfg = vllm_config_factory()
    cfg.additional_config.compile_config.rbln_config["prefill_chunk_size"] = 256
    sync.sync_vllm_and_optimum(cfg)
    assert cfg.scheduler_config.max_num_batched_tokens == 256


def test_image_buckets_do_not_mutate_the_compile_config(vllm_config_factory):
    cfg = vllm_config_factory(
        hf_config=PaliGemmaConfig(architectures=["PaliGemmaForConditionalGeneration"]),
        optimum_overrides={"language_model": {"image_prefill_chunk_size": [512, 256]}},
    )
    original_key = compile_cache_name(cfg.additional_config.compile_config)
    cfg.additional_config.image_prefill_chunk_size.append(128)
    assert cfg.additional_config.compile_config.rbln_config["language_model"][
        "image_prefill_chunk_size"
    ] == [512, 256]
    assert compile_cache_name(cfg.additional_config.compile_config) == original_key
