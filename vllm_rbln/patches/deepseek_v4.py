# Copyright 2026 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

import functools

from vllm_rbln.patches.registry import add_registration, register_patch

_MODULE = "vllm_rbln.model_executor.models.deepseek_v4"


@add_registration(
    reason=(
        "Upstream's DeepSeek-V4 binds FlashMLA / FlashInfer / DeepGEMM / tilelang kernels at "
        "import, so the RBLN copy (V4 attention custom ops, RBLN MoE runner) replaces it."
    )
)
def register_deepseek_v4() -> None:
    from vllm.model_executor.models import ModelRegistry

    ModelRegistry.register_model(
        "DeepseekV4ForCausalLM", f"{_MODULE}:RBLNDeepseekV4ForCausalLM"
    )


@add_registration(
    reason=(
        "Override the built-in deepseek_v4_fp8 config so the block-FP8 linears keep their "
        "fp8 weights with a folded bf16 scale (W8A16) and the MXFP4 experts run through the "
        "RBLN group-dequantise MoE op instead of the CUDA backends."
    )
)
def register_rbln_deepseek_v4_fp8_config() -> None:
    from vllm.model_executor.layers.quantization import register_quantization_config

    from vllm_rbln.model_executor.layers.quantization.deepseek_v4 import (
        RBLNDeepseekV4FP8Config,
    )

    register_quantization_config("deepseek_v4_fp8")(RBLNDeepseekV4FP8Config)


@add_registration(
    reason=(
        "The RBLN DeepSeek-V4 keeps its per-request rings (SWA ring, compressor states) as "
        "MambaSpec states (one block a request). Mapping the architecture to a config that also "
        "runs MambaModelConfig gives them vLLM's Mamba cache setup (mamba_block_size, mode), on "
        "top of upstream's deepseek_v4_fp8 quant-method rewrite."
    )
)
def register_deepseek_v4_mamba_config() -> None:
    from vllm.model_executor.models.config import (
        MODELS_CONFIG_MAP,
        DeepseekV4ForCausalLMConfig,
        MambaModelConfig,
    )

    class RBLNDeepseekV4Config(DeepseekV4ForCausalLMConfig, MambaModelConfig):
        pass

    MODELS_CONFIG_MAP["DeepseekV4ForCausalLM"] = RBLNDeepseekV4Config


def _is_deepseek_v4(vllm_config) -> bool:
    model_config = getattr(vllm_config, "model_config", None)
    if model_config is None:
        return False
    return getattr(model_config.hf_text_config, "model_type", None) == "deepseek_v4"


def deepseek_v4_kv_cache_groups(vllm_config, kv_cache_spec):
    """The RBLN DeepSeek-V4 groups, or None for any other spec set.

    An attention group a block size holds the paged caches (MLAAttentionSpec: the compressed KV
    of the CSA / HCA layers and the CSA indexer keys, `tokens_per_state` r): one block table, a
    block of `block_size` tokens storing `block_size / r` rows a layer (the HCA caches take
    larger blocks). Each distinct MambaSpec (the SWA
    ring, the CSA / HCA compressor state, the CSA indexer's compressor state) is a group of its
    own: one block a request, each layer its own tensor.
    """
    from vllm.v1.kv_cache_interface import (
        KVCacheGroupSpec,
        MambaSpec,
        MLAAttentionSpec,
        UniformTypeKVCacheSpecs,
    )

    if not _is_deepseek_v4(vllm_config):
        return None
    attn = {n: s for n, s in kv_cache_spec.items() if type(s) is MLAAttentionSpec}
    states = {n: s for n, s in kv_cache_spec.items() if isinstance(s, MambaSpec)}
    if not attn or len(attn) + len(states) != len(kv_cache_spec):
        return None
    # one attention group a block size (the HCA caches take larger blocks than the CSA ones)
    groups = []
    by_block: dict = {}
    for name, spec in attn.items():
        by_block.setdefault(spec.block_size, {})[name] = spec
    for _, specs in sorted(by_block.items()):
        uniform = UniformTypeKVCacheSpecs.from_specs(specs)
        assert uniform is not None, f"DeepSeek-V4 paged caches not uniform: {specs}"
        groups.append(KVCacheGroupSpec(list(specs), uniform))
    by_spec: dict = {}
    for name, spec in states.items():
        by_spec.setdefault(spec, []).append(name)
    groups.extend(KVCacheGroupSpec(names, spec) for spec, names in by_spec.items())
    return groups


def _deepseek_v4_pages(kv_cache_groups):
    """[(layer, page bytes)] of every layer, in group order; None off the V4 grouping."""
    from vllm.v1.kv_cache_interface import (
        MambaSpec,
        MLAAttentionSpec,
        UniformTypeKVCacheSpecs,
    )

    pages = []
    for group in kv_cache_groups:
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            if not all(type(s) is MLAAttentionSpec for s in spec.kv_cache_specs.values()):
                return None
            pages.extend((n, spec.kv_cache_specs[n].page_size_bytes) for n in group.layer_names)
        elif isinstance(spec, MambaSpec):
            pages.extend((n, spec.page_size_bytes) for n in group.layer_names)
        else:
            return None
    return pages or None


@register_patch(
    target="vllm.v1.core.kv_cache_utils.get_kv_cache_groups",
    reason=(
        "Upstream groups DeepSeek-V4 for its packed block-outermost layout (FlashMLA). The RBLN "
        "kernels take each cache as its own whole tensor: one attention group for the paged "
        "caches and one MambaSpec group a per-request ring."
    ),
    build=True,
)
def _build_deepseek_v4_kv_cache_groups():
    from vllm.v1.core import kv_cache_utils

    original = kv_cache_utils.get_kv_cache_groups

    @functools.wraps(original)
    def get_kv_cache_groups(vllm_config, kv_cache_spec):
        groups = deepseek_v4_kv_cache_groups(vllm_config, kv_cache_spec)
        return groups if groups is not None else original(vllm_config, kv_cache_spec)

    return get_kv_cache_groups


@register_patch(
    target="vllm.v1.core.kv_cache_utils.get_kv_cache_config_from_groups",
    reason=(
        "The RBLN DeepSeek-V4 groups (deepseek_v4_kv_cache_groups) give every layer its own "
        "layer-outermost region: the paged caches' and the rings' pages differ in size and in "
        "how the compiler cuts them over the chiplets, so no two layers alias."
    ),
    build=True,
)
def _build_deepseek_v4_kv_cache_config():
    from vllm.v1.core import kv_cache_utils
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheTensor

    original = kv_cache_utils.get_kv_cache_config_from_groups

    @functools.wraps(original)
    def get_kv_cache_config_from_groups(vllm_config, kv_cache_groups, available_memory):
        pages = _deepseek_v4_pages(kv_cache_groups) if _is_deepseek_v4(vllm_config) else None
        if pages is None:
            return original(vllm_config, kv_cache_groups, available_memory)
        bytes_per_block = sum(page for _, page in pages)
        num_blocks = kv_cache_utils.may_override_num_blocks(
            vllm_config, available_memory // bytes_per_block
        )
        size = bytes_per_block * num_blocks
        tensors, offset = [], 0
        for name, page in pages:
            tensors.append(
                KVCacheTensor(
                    size=size,
                    layers=[name],
                    layer_stride=page * num_blocks,
                    block_stride=page,
                    offset=offset,
                )
            )
            offset += page * num_blocks
        return KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=tensors,
            kv_cache_groups=kv_cache_groups,
            prefix_cache_retention_interval=(
                vllm_config.cache_config.prefix_cache_retention_interval
            ),
        )

    return get_kv_cache_config_from_groups
