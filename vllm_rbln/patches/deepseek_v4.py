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

import functools

from vllm_rbln.patches.registry import add_registration, register_patch

_MODULE = "vllm_rbln.model_executor.models.deepseek_v4"


@add_registration(
    reason=(
        "Upstream's DeepSeek-V4 binds FlashMLA / FlashInfer / DeepGEMM / "
        "tilelang kernels at import, so the RBLN copy replaces it."
    )
)
def register_deepseek_v4() -> None:
    from vllm.model_executor.models import ModelRegistry

    ModelRegistry.register_model(
        "DeepseekV4ForCausalLM", f"{_MODULE}:RBLNDeepseekV4ForCausalLM"
    )
    ModelRegistry.register_model("DeepSeekV4MTPModel", f"{_MODULE}:RBLNDeepseekV4MTP")


@add_registration(
    reason=(
        "Override the built-in deepseek_v4_fp8 config so the block-FP8 "
        "linears keep their "
        "fp8 weights with a folded bf16 scale (W8A16) and the MXFP4 "
        "experts run through the RBLN op."
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
        "RBLN's circular buffers retain whole-block prompt checkpoints only, "
        "do not support KV transfer, and limit speculative lookahead."
    )
)
def register_deepseek_v4_config() -> None:
    from vllm.model_executor.models.config import (
        MODELS_CONFIG_MAP,
        DeepseekV4ForCausalLMConfig,
    )

    class RBLNDeepseekV4Config(DeepseekV4ForCausalLMConfig):
        @classmethod
        def verify_and_update_config(cls, vllm_config) -> None:
            cache = vllm_config.cache_config
            if cache.enable_prefix_caching:
                from vllm_rbln.platform import use_device_tensor

                if not use_device_tensor():
                    raise ValueError(
                        "DeepSeek-V4 prefix caching requires device tensors."
                    )
                if cache.prefix_cache_retention_interval != 0:
                    raise ValueError(
                        "DeepSeek-V4 prefix caching on RBLN supports "
                        "prefix_cache_retention_interval=0 only."
                    )
                if (
                    cache.prefix_match_unit is not None
                    and cache.prefix_match_unit < cache.block_size
                ):
                    raise ValueError(
                        "DeepSeek-V4 prefix caching on RBLN uses whole blocks; "
                        "prefix_match_unit must not be smaller than block_size."
                    )
            speculative = vllm_config.speculative_config
            if cache.enable_prefix_caching and speculative is not None:
                speculative.disable_eagle_block_drop = True
            if speculative is not None and speculative.num_speculative_tokens > 3:
                raise ValueError(
                    "DeepSeek-V4 on RBLN supports at most 3 speculative tokens; "
                    "longer verification needs unaligned multi-block compression."
                )
            if vllm_config.kv_transfer_config is not None:
                raise ValueError(
                    "DeepSeek-V4 KV transfer requires separate capacities and slot "
                    "IDs for paged caches and ring states; this is not supported "
                    "by the RBLN connectors yet."
                )
            super().verify_and_update_config(vllm_config)

    MODELS_CONFIG_MAP["DeepseekV4ForCausalLM"] = RBLNDeepseekV4Config


def _is_deepseek_v4(vllm_config) -> bool:
    model_config = getattr(vllm_config, "model_config", None)
    if model_config is None:
        return False
    return getattr(model_config.hf_text_config, "model_type", None) == "deepseek_v4"


def deepseek_v4_kv_cache_groups(vllm_config, kv_cache_spec):
    from vllm.v1.kv_cache_interface import (
        KVCacheGroupSpec,
        MLAAttentionSpec,
        UniformTypeKVCacheSpecs,
    )

    if not _is_deepseek_v4(vllm_config):
        return None
    from vllm_rbln.v1.attention.backends.deepseek_v4 import RBLNDeepseekV4StateSpec

    attn = {n: s for n, s in kv_cache_spec.items() if type(s) is MLAAttentionSpec}
    states = {
        n: s for n, s in kv_cache_spec.items() if isinstance(s, RBLNDeepseekV4StateSpec)
    }
    if not kv_cache_spec or len(attn) + len(states) != len(kv_cache_spec):
        return None
    # HCA and CSA use different token block sizes and therefore different tables.
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


@register_patch(
    target="vllm.v1.core.kv_cache_manager.get_kv_cache_coordinator",
    reason=(
        "vLLM 0.31 has no coordinator registration or lookahead-hash replay "
        "boundaries. Use them for RBLN rings. Remove this backport when upgrading "
        "past vllm-project/vllm@da8a505684e6."
    ),
    build=True,
)
def _build_deepseek_v4_cache_coordinator():
    from vllm.v1.core import kv_cache_manager

    from vllm_rbln.v1.attention.backends.deepseek_v4 import RBLNDeepseekV4StateSpec
    from vllm_rbln.v1.kv_cache import RBLNDeepseekV4CacheCoordinator

    original = kv_cache_manager.get_kv_cache_coordinator

    @functools.wraps(original)
    def get_kv_cache_coordinator(kv_cache_config, *args, **kwargs):
        if kwargs["enable_caching"] and any(
            isinstance(g.kv_cache_spec, RBLNDeepseekV4StateSpec)
            for g in kv_cache_config.kv_cache_groups
        ):
            return RBLNDeepseekV4CacheCoordinator(kv_cache_config, *args, **kwargs)
        return original(kv_cache_config, *args, **kwargs)

    return get_kv_cache_coordinator


@register_patch(
    target="vllm.v1.engine.core.EngineCore.__init__",
    reason=(
        "vLLM 0.31 cannot configure the Request hasher's MTP lookahead. Install "
        "it before requests are created for the ring coordinator. Remove this "
        "backport when upgrading past vllm-project/vllm@da8a505684e6."
    ),
    build=True,
)
def _build_deepseek_v4_request_hasher():
    from vllm.utils.hashing import get_hash_fn_by_name
    from vllm.v1.engine.core import EngineCore

    from vllm_rbln.v1.core.rbln_kv_cache_manager import get_request_block_hasher
    from vllm_rbln.v1.kv_cache import RBLNDeepseekV4CacheCoordinator

    original = EngineCore.__init__

    @functools.wraps(original)
    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        coordinator = self.scheduler.kv_cache_manager.coordinator
        if isinstance(coordinator, RBLNDeepseekV4CacheCoordinator):
            self.request_block_hasher = get_request_block_hasher(
                self.scheduler.hash_block_size,
                get_hash_fn_by_name(
                    self.vllm_config.cache_config.prefix_caching_hash_algo
                ),
                coordinator.num_hash_lookahead_tokens,
            )

    return initialize


def _deepseek_v4_pages(kv_cache_groups):
    """Layer, page bytes and fixed slots (None for paged caches); None off V4."""
    from vllm.v1.kv_cache_interface import (
        MLAAttentionSpec,
        UniformTypeKVCacheSpecs,
    )

    from vllm_rbln.v1.attention.backends.deepseek_v4 import RBLNDeepseekV4StateSpec

    if not any(
        isinstance(g.kv_cache_spec, RBLNDeepseekV4StateSpec) for g in kv_cache_groups
    ):
        return None
    pages: list[tuple[str, int, int | None]] = []
    for group in kv_cache_groups:
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            if not all(
                type(s) is MLAAttentionSpec for s in spec.kv_cache_specs.values()
            ):
                return None
            pages.extend(
                (n, spec.kv_cache_specs[n].page_size_bytes, None)
                for n in group.layer_names
            )
        elif isinstance(spec, RBLNDeepseekV4StateSpec):
            pages.extend(
                (n, spec.page_size_bytes, spec.num_state_slots)
                for n in group.layer_names
            )
        else:
            return None
    return pages


@register_patch(
    target="vllm.v1.core.kv_cache_utils.get_kv_cache_groups",
    reason=(
        "Upstream groups DeepSeek-V4 for FlashMLA's packed block-outermost "
        "layout. RBLN kernels require whole per-layer tensors, grouped by token "
        "block size, with separate circular-buffer pools for per-request rings."
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
    target="vllm.v1.engine.core.resolve_kv_cache_block_sizes",
    reason=(
        "CircularBufferSpec.block_size describes ring capacity, not a token "
        "checkpoint interval. Including it in the hash GCD would hash every "
        "few tokens and reject valid whole-block prefix_match_unit settings."
    ),
    build=True,
)
def _build_deepseek_v4_prefix_block_sizes():
    from dataclasses import replace
    from math import lcm

    from vllm.v1.core import kv_cache_utils
    from vllm.v1.engine import core

    from vllm_rbln.v1.attention.backends.deepseek_v4 import RBLNDeepseekV4StateSpec

    original = core.resolve_kv_cache_block_sizes

    @functools.wraps(original)
    def resolve_kv_cache_block_sizes(kv_cache_config, vllm_config):
        groups = kv_cache_config.kv_cache_groups
        if vllm_config.cache_config.enable_prefix_caching and any(
            isinstance(g.kv_cache_spec, RBLNDeepseekV4StateSpec) for g in groups
        ):
            interval = lcm(
                vllm_config.cache_config.block_size,
                *(
                    kv_cache_utils.resolve_dcp_kv_block_size(
                        g.kv_cache_spec,
                        vllm_config.parallel_config.decode_context_parallel_size,
                    )
                    for g in groups
                ),
            )
            # A temporary logical view for alignment/hash resolution only;
            # allocation and block tables keep the physical ring capacities.
            kv_cache_config = replace(
                kv_cache_config,
                kv_cache_groups=[
                    replace(
                        g,
                        kv_cache_spec=g.kv_cache_spec.copy_with_new_block_size(
                            interval
                        ),
                    )
                    if isinstance(g.kv_cache_spec, RBLNDeepseekV4StateSpec)
                    else g
                    for g in groups
                ],
            )
        return original(kv_cache_config, vllm_config)

    return resolve_kv_cache_block_sizes


@register_patch(
    target="vllm.v1.core.kv_cache_utils._get_kv_cache_bytes_per_block",
    reason=(
        "Upstream's largest-group estimate cannot represent DeepSeek-V4's "
        "separate per-layer regions. Fixed ring slots do not grow with the "
        "paged block pool."
    ),
    build=True,
)
def _build_deepseek_v4_bytes_per_block():
    from vllm.v1.core import kv_cache_utils

    original = kv_cache_utils._get_kv_cache_bytes_per_block

    @functools.wraps(original)
    def _get_kv_cache_bytes_per_block(kv_cache_groups):
        pages = _deepseek_v4_pages(kv_cache_groups)
        if pages is None:
            return original(kv_cache_groups)
        return sum(page for _, page, slots in pages if slots is None)

    return _get_kv_cache_bytes_per_block


@register_patch(
    target="vllm.v1.core.kv_cache_utils.get_kv_cache_config_from_groups",
    reason=(
        "The RBLN DeepSeek-V4 groups (deepseek_v4_kv_cache_groups) give "
        "every layer its own "
        "layer-outermost region: the paged caches' and the rings' pages "
        "differ in size and in "
        "how the compiler cuts them over the chiplets, so no two layers alias."
    ),
    build=True,
)
def _build_deepseek_v4_kv_cache_config():
    from vllm.v1.core import kv_cache_utils
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheTensor

    from vllm_rbln.v1.kv_cache import RBLNFixedKVCacheTensor

    original = kv_cache_utils.get_kv_cache_config_from_groups

    @functools.wraps(original)
    def get_kv_cache_config_from_groups(vllm_config, kv_cache_groups, available_memory):
        pages = (
            _deepseek_v4_pages(kv_cache_groups)
            if _is_deepseek_v4(vllm_config)
            else None
        )
        if pages is None:
            return original(vllm_config, kv_cache_groups, available_memory)
        fixed_bytes = sum(page * slots for _, page, slots in pages if slots is not None)
        bytes_per_block = sum(page for _, page, slots in pages if slots is None)
        # A ring-only PP stage does not constrain the shared paged pool's count.
        num_blocks = kv_cache_utils.may_override_num_blocks(
            vllm_config,
            (available_memory - fixed_bytes) // bytes_per_block
            if bytes_per_block
            else 1,
        )
        size = fixed_bytes + bytes_per_block * num_blocks
        if vllm_config.cache_config.num_gpu_blocks_override is None and (
            num_blocks < 1 or size > available_memory
        ):
            raise ValueError(
                "DeepSeek-V4 fixed state slots exceed the available KV budget; "
                "reduce max_num_seqs or increase the KV memory budget."
            )
        tensors, offset = [], 0
        for name, page, slots in pages:
            extent = page * (num_blocks if slots is None else slots)
            tensor_type = KVCacheTensor if slots is None else RBLNFixedKVCacheTensor
            tensors.append(
                tensor_type(
                    size=size,
                    layers=[name],
                    layer_stride=extent,
                    block_stride=page,
                    offset=offset,
                )
            )
            offset += extent
        return KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=tensors,
            kv_cache_groups=kv_cache_groups,
            prefix_cache_retention_interval=(
                vllm_config.cache_config.prefix_cache_retention_interval
            ),
        )

    return get_kv_cache_config_from_groups


@register_patch(
    target="vllm.v1.core.kv_cache_utils._max_memory_usage_bytes_from_groups",
    reason=(
        "Upstream charges every device group to one shared block pool. RBLN V4 "
        "rings have fixed, separate pools, including their own null slots."
    ),
    build=True,
)
def _build_deepseek_v4_memory_usage():
    from vllm.utils.math_utils import cdiv
    from vllm.v1.core import kv_cache_utils

    from vllm_rbln.v1.attention.backends.deepseek_v4 import RBLNDeepseekV4StateSpec

    original = kv_cache_utils._max_memory_usage_bytes_from_groups

    @functools.wraps(original)
    def max_memory_usage(vllm_config, groups):
        pages = _deepseek_v4_pages(groups)
        if pages is None:
            return original(vllm_config, groups)
        fixed = sum(page * slots for _, page, slots in pages if slots is not None)
        page = sum(page for _, page, slots in pages if slots is None)
        # Empty PP groups still consume IDs in the scheduler's global pool.
        blocks = sum(
            cdiv(
                g.kv_cache_spec.max_memory_usage_bytes(vllm_config),
                g.kv_cache_spec.page_size_bytes,
            )
            for g in groups
            if not isinstance(g.kv_cache_spec, RBLNDeepseekV4StateSpec)
        )
        return fixed + page * blocks

    return max_memory_usage


@register_patch(
    target="vllm.v1.engine.core.get_kv_cache_configs",
    reason=(
        "Upstream reconciles PP ranks by replanning with N * page_bytes. RBLN V4 "
        "requires fixed_ring_bytes + N * page_bytes; auto-fit, overrides and PP "
        "resizing must preserve the independent ring pools."
    ),
    build=True,
)
def _build_deepseek_v4_kv_cache_configs():
    from vllm.v1.core import kv_cache_utils
    from vllm.v1.engine import core as engine_core
    from vllm.v1.kv_cache_interface import KVCacheSpec, KVCacheSpecRegistry

    from vllm_rbln.v1.worker.utils import rescale_kv_cache_config

    original = engine_core.get_kv_cache_configs

    @functools.wraps(original)
    def get_kv_cache_configs(vllm_config, kv_cache_specs, available_memory):
        if not _is_deepseek_v4(vllm_config):
            return original(vllm_config, kv_cache_specs, available_memory)
        merged: dict[str, KVCacheSpec] = {}
        for specs in kv_cache_specs:
            for name, spec in specs.items():
                assert name not in merged or merged[name] == spec
                merged[name] = spec
        KVCacheSpecRegistry.check_kv_cache_spec_registry(merged)
        global_groups = deepseek_v4_kv_cache_groups(vllm_config, merged)
        assert global_groups is not None
        projected = [
            kv_cache_utils._project_kv_cache_groups_to_worker(global_groups, specs)
            for specs in kv_cache_specs
        ]
        page_bytes = [kv_cache_utils._pool_bytes_per_block(g) for g in projected]
        override = vllm_config.cache_config.num_gpu_blocks_override
        if override is not None:
            available_memory = [
                override * page
                + sum(
                    size * slots
                    for _, size, slots in _deepseek_v4_pages(groups)
                    if slots is not None
                )
                for groups, page in zip(projected, page_bytes)
            ]
        check_memory = [
            memory - page for memory, page in zip(available_memory, page_bytes)
        ]
        if vllm_config.model_config.original_max_model_len == -1:
            kv_cache_utils._auto_fit_max_model_len(vllm_config, projected, check_memory)
        for groups, memory in zip(projected, check_memory):
            kv_cache_utils._check_enough_kv_cache_memory(
                memory,
                functools.partial(
                    kv_cache_utils._max_memory_usage_bytes_from_groups,
                    vllm_config,
                    groups,
                ),
                vllm_config.model_config.max_model_len,
                functools.partial(
                    kv_cache_utils._estimate_max_model_len_from_groups,
                    vllm_config,
                    groups,
                ),
            )
        configs = [
            kv_cache_utils.get_kv_cache_config_from_groups(vllm_config, groups, memory)
            for groups, memory in zip(projected, available_memory)
        ]
        num_blocks = min(
            (cfg.num_blocks for cfg, page in zip(configs, page_bytes) if page),
            default=1,
        )
        for cfg in configs:
            rescale_kv_cache_config(cfg, num_blocks)
            cfg.kv_tp_replicas = kv_cache_utils.kv_cache_groups_tp_replicas(
                cfg.kv_cache_groups,
                vllm_config.parallel_config.tensor_parallel_size,
                vllm_config.parallel_config.decode_context_parallel_size,
            )
        return configs

    return get_kv_cache_configs


@register_patch(
    target="vllm.v1.core.kv_cache_utils.get_max_concurrency_for_kv_cache_config",
    reason=(
        "RBLN V4 ring IDs come from bounded pools separate from paged blocks. "
        "Upstream's single-pool concurrency estimate counts them as paged blocks "
        "and does not enforce their independent capacity."
    ),
    build=True,
)
def _build_deepseek_v4_max_concurrency():
    from vllm.utils.math_utils import cdiv
    from vllm.v1.core import kv_cache_utils

    from vllm_rbln.v1.attention.backends.deepseek_v4 import RBLNDeepseekV4StateSpec

    original = kv_cache_utils.get_max_concurrency_for_kv_cache_config

    @functools.wraps(original)
    def get_max_concurrency(vllm_config, cfg):
        rings = [
            g.kv_cache_spec
            for g in cfg.kv_cache_groups
            if isinstance(g.kv_cache_spec, RBLNDeepseekV4StateSpec)
        ]
        if not rings:
            return original(vllm_config, cfg)
        limits = [float(spec.num_state_slots - 1) for spec in rings]
        blocks = sum(
            cdiv(
                g.kv_cache_spec.max_memory_usage_bytes(vllm_config),
                g.kv_cache_spec.page_size_bytes,
            )
            for g in cfg.kv_cache_groups
            if not isinstance(g.kv_cache_spec, RBLNDeepseekV4StateSpec)
        )
        if blocks:
            limits.append((cfg.num_blocks - 1) / blocks)
        return min(limits)

    return get_max_concurrency
