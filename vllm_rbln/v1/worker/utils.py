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
"""Utilities for RBLN worker. (CPU affinity, batch reorder, ...)"""

import math
import os
import platform
from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import torch
from vllm.config import ModelConfig, ParallelConfig
from vllm.platforms import CpuArchEnum, current_platform
from vllm.utils.cpu_resource_utils import (
    LogicalCPUInfo,
    get_allowed_cpu_list,
    get_visible_memory_node,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    EncoderOnlyAttentionSpec,
    KVCacheConfig,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.utils import AttentionGroup, select_common_block_size

from vllm_rbln import envs
from vllm_rbln.logger import init_logger
from vllm_rbln.v1.kv_cache import RBLNSlidingWindowSpec

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_input_batch import InputBatch

logger = init_logger(__name__)

RBLN_SYSFS_CLASS_DIR = "/sys/class/rebellions"
# sysfs lists every card on the host, /dev only ours; reading sysfs by the raw
# RBLN_DEVICES entry would charge a neighbouring container's workload to us.
RBLN_DEV_DIR = "/dev"

# 144 GiB of quad-chiplet DRAM minus the 4 GiB system region
REBEL_DRAM_NBYTES = 144 * 2**30 - 4 * 2**30


def extract_layer_index(layer_name: str, num_attn_module: int = 1) -> int:
    int_vals: list[int] = []
    for subname in layer_name.split("."):
        try:
            int_vals.append(int(subname))
        except ValueError:
            continue
    if num_attn_module <= 1 or "attn" not in layer_name:
        assert len(int_vals) == 1, (
            f"layer name {layer_name} should only contain one integer"
        )
        return int_vals[0]
    assert int_vals, f"layer name {layer_name} has no integer layer index"
    base = int_vals[0]
    if len(int_vals) >= 2:
        sub = int_vals[1]
    elif "scale" in layer_name:
        sub = 2
    elif "indexer" in layer_name:
        sub = 1
    else:
        sub = 0
    return base * num_attn_module + sub


def pipeline_adjusted_layer_index(
    layer_name: str,
    model_config,
    parallel_config,
    num_attn_module: int,
) -> int:
    raw_layer_index = extract_layer_index(layer_name, num_attn_module)
    if model_config is None:
        return raw_layer_index

    start, end = model_config.get_layers_start_end_indices(parallel_config)
    total_num_hidden_layers = model_config.get_total_num_hidden_layers()
    if raw_layer_index >= total_num_hidden_layers * num_attn_module:
        # MTP/nextn layers are named past the target's layer count
        # (mtp_start_layer_idx == num_hidden_layers), so their KV cache sits
        # right after the target layers in the compacted per-rank cache list.
        return (end - start) * num_attn_module + (
            raw_layer_index - total_num_hidden_layers * num_attn_module
        )
    return raw_layer_index - start * num_attn_module


def num_attn_module(model_config, cache_dtype) -> int:
    hf_config = model_config.hf_config
    if getattr(hf_config, "model_type", None) == "longcat_flash":
        return 2
    text_config = getattr(model_config, "hf_text_config", hf_config)
    # A DSA model puts the lightning-indexer key cache next to MLA: 2 modules,
    # or 3 when the indexer cache is fp8 (a companion fp16 scale cache).
    if hasattr(text_config, "index_topk") or hasattr(hf_config, "index_topk"):
        is_fp8 = bool(cache_dtype) and cache_dtype.startswith("fp8")
        return 3 if is_fp8 else 2
    return 1


def get_rbln_visible_card_indices() -> list[int]:
    """Card indices this process may use, from `RBLN_DEVICES`.

    Unset or empty means every card under /sys/class/rebellions. Prefer
    `get_rbln_owned_card_indices`; this one reads each entry as a sysfs card
    name and remains only as the fallback for hosts with no device nodes.
    """
    raw = os.environ.get("RBLN_DEVICES", "")
    if raw.strip():
        return sorted(
            {int(token) for token in raw.replace(",", " ").split() if token.strip()}
        )
    if not os.path.isdir(RBLN_SYSFS_CLASS_DIR):
        return []
    found: list[int] = []
    for name in os.listdir(RBLN_SYSFS_CLASS_DIR):
        if name.startswith("rbln") and name[4:].isdigit():
            found.append(int(name[4:]))
    return sorted(found)


def _rbln_present_card_indices() -> list[int]:
    """Physical card indices this process can open, from /dev/rbln*.

    Device nodes keep their host numbering. [] means callers should fall back.
    """
    try:
        names = os.listdir(RBLN_DEV_DIR)
    except OSError:
        return []
    return sorted(
        int(name[4:])
        for name in names
        if name.startswith("rbln") and name[4:].isdigit()
    )


def get_rbln_owned_card_indices() -> list[int]:
    """sysfs card indices this process owns, with `RBLN_DEVICES` resolved.

    Entry `i` selects the `i`-th present device; a physical name is accepted too,
    but the positional reading wins when both are possible.
    """
    present = _rbln_present_card_indices()
    if not present:
        # No device nodes (unit tests, host without the driver): nothing better
        # is knowable, so keep the previous behaviour exactly.
        return get_rbln_visible_card_indices()
    raw = os.environ.get("RBLN_DEVICES", "")
    if not raw.strip():
        return present
    owned: list[int] = []
    for token in raw.replace(",", " ").split():
        if not token.strip():
            continue
        index = int(token)
        if 0 <= index < len(present):
            owned.append(present[index])
        elif index in present:
            owned.append(index)
    return sorted(set(owned)) or present


def _read_card_attr_int(card_index: int, attr: str) -> int | None:
    path = os.path.join(RBLN_SYSFS_CLASS_DIR, f"rbln{card_index}", attr)
    try:
        with open(path) as handle:
            return int(handle.read().strip())
    except (OSError, ValueError):
        return None


def read_rbln_card_dram_total_bytes() -> int | None:
    """Per-card DRAM capacity in bytes, or None when sysfs is unavailable.

    Reads only the cards this process owns; a heterogeneous set is rejected
    rather than averaged. Clamped to `REBEL_DRAM_NBYTES` here rather than at the
    call sites, because both of them size real allocations from this number.
    """
    values: dict[int, int] = {}
    for card_index in get_rbln_owned_card_indices():
        value = _read_card_attr_int(card_index, "dram_total")
        if value is not None and value > 0:
            values[card_index] = value
    if not values:
        return None
    distinct = set(values.values())
    if len(distinct) != 1:
        raise RuntimeError(
            "visible RBLN cards report different dram_total values "
            f"({values}); a single per-chiplet DRAM budget cannot be derived."
        )
    dram_total = next(iter(distinct))
    if dram_total > REBEL_DRAM_NBYTES:
        logger.warning(
            "sysfs reports %d bytes of card DRAM, more than the %d this build "
            "assumes usable; clamping. Raise REBEL_DRAM_NBYTES if the card is "
            "genuinely larger.",
            dram_total,
            REBEL_DRAM_NBYTES,
        )
        return REBEL_DRAM_NBYTES
    return dram_total


def read_rbln_card_dram_used_bytes() -> int:
    """Largest `dram_used` across the cards we own (0 when sysfs is absent).

    Sampled before this process allocates, so it is what other tenants hold.
    Card-scope: sysfs has no per-chiplet breakdown, so the caller must scale it
    before subtracting it from a per-chiplet budget. Cards we do not own are
    excluded: charging a neighbour's usage cost one run 1024 blocks -> 308.
    """
    used = [
        value
        for card_index in get_rbln_owned_card_indices()
        if (value := _read_card_attr_int(card_index, "dram_used")) is not None
    ]
    return max(used, default=0)


def rescale_kv_cache_config(cfg: KVCacheConfig, num_blocks: int) -> None:
    """Retarget `cfg` at `num_blocks`, in place.

    `KVCacheTensor.size` is `num_blocks * page_size_bytes` and consumers read it
    rather than recomputing, so it has to move with `num_blocks`.
    """
    old_num_blocks = cfg.num_blocks
    if old_num_blocks <= 0:
        raise ValueError(f"cannot rescale a KV cache config of {old_num_blocks} blocks")
    cfg.num_blocks = num_blocks
    for kv_tensor in cfg.kv_cache_tensors:
        kv_tensor.size = (kv_tensor.size * num_blocks) // old_num_blocks


def chiplet_replication_factor(num_key_value_heads: int, rsd_size: int) -> float:
    """How many times the KV cache is physically replicated across chiplets.

    Each chiplet rounds up to `ceil(kvh / rsd_size)` KV heads, so the ratio is
    > 1 exactly when `kvh` is not a multiple of `rsd_size`. Examples at
    rsd_size=4: kvh=2 -> 2.0, kvh=4 -> 1.0, kvh=10 -> 1.2.
    """
    if num_key_value_heads <= 0 or rsd_size <= 0:
        return 1.0
    return rsd_size * math.ceil(num_key_value_heads / rsd_size) / num_key_value_heads


def divide_by_chiplet_replication(
    nbytes: int, num_key_value_heads: int, rsd_size: int
) -> int:
    """`nbytes // chiplet_replication_factor(...)` in exact integer arithmetic."""
    if num_key_value_heads <= 0 or rsd_size <= 0:
        return nbytes
    return (
        nbytes
        * num_key_value_heads
        // (rsd_size * math.ceil(num_key_value_heads / rsd_size))
    )


def estimate_model_kernel_size(
    model_config: ModelConfig,
    parallel_config: ParallelConfig,
    *,
    nbits_per_param: int | None = None,
    n_model_params: int | float | None = None,
    n_model_bytes: int | float | None = None,
    default_bits_per_param: int | None = None,
) -> int:
    def align(x: int | float, nbytes: int) -> int:
        return int(math.ceil(x / nbytes) * nbytes)

    def align_2MB(x: int | float) -> int:
        return align(x, 2**21)

    num_layers = model_config.get_num_layers(parallel_config)
    vocab_size = model_config.get_vocab_size()
    hidden_size = model_config.get_hidden_size()
    tp_size = parallel_config.tensor_parallel_size

    if default_bits_per_param is None:
        device_name = current_platform.get_device_name().lower()
        assert "rbln" in device_name
        if "ca" in device_name or "cr" in device_name:
            default_bits_per_param = 16
        else:
            raise ValueError(
                "invalid RBLN architecture, candidates = [ATOM(ca), REBEL(cr)]"
            )

    if n_model_params is None and n_model_bytes is None:
        raise ValueError(
            "Either `n_model_params` or `n_model_bytes` should be specified "
            "to estimate the kernel memory."
        )
    if n_model_params is not None and n_model_bytes is not None:
        raise ValueError(
            "Only one of `n_model_params` or `n_model_bytes` may be specified."
        )

    lm_heads_params = align(vocab_size, 64) * hidden_size
    lm_heads_nbytes = (
        align_2MB(lm_heads_params * default_bits_per_param // 8 / tp_size) * tp_size
    )
    if n_model_bytes is not None:
        lm_heads_bytes = lm_heads_params * default_bits_per_param // 8
        word_embedding_bytes = lm_heads_bytes
        layer_bytes = n_model_bytes - lm_heads_bytes - word_embedding_bytes
        layer_nbytes = align_2MB(layer_bytes / num_layers) * num_layers
    else:
        if nbits_per_param is None:
            raise ValueError(
                "`nbits_per_param` should be specified when using `n_model_params` "
                "to estimate the kernel memory."
            )
        word_embedding_params = lm_heads_params
        params = n_model_params - lm_heads_params - word_embedding_params
        layer_nbytes = (
            align_2MB(params * nbits_per_param // 8 / num_layers) * num_layers
        )

    return layer_nbytes + lm_heads_nbytes


# NOTE: This function comes from optimum-rbln. Keep in sync.
def estimate_available_memory(
    model_config: ModelConfig,
    parallel_config: ParallelConfig,
    nbits_per_param: int | None = None,
    n_model_params: int | None = None,
    n_model_bytes: int | None = None,
    kernel_size: int | None = None,
    buffer: int | None = None,
    num_runtimes: int = 2,
    gpu_memory_utilization: float = 0.9,
) -> int:
    # We are finding max_num_blocks(x) that satisfies the following equation:

    # available_dram - kernel_size - buffer
    #     - num_layers * 2 * tensor_parallel_size
    #     * align_2MB(
    #         x
    #         * block_size
    #         * align_64(head_dim)
    #         * math.ceil(num_key_value_heads / tensor_parallel_size)
    #         * 2
    #     ) > 0

    # This inequality can be rewritten as follows:

    # a - c * align_2MB(b * x) > 0
    # where
    #    a = available_dram - kernel_size - buffer
    #    b = block_size
    #         * align_64(head_dim)
    #         * math.ceil(num_key_value_heads / tensor_parallel_size) * 2
    #    c = num_layers * 2 * tensor_parallel_size

    # We can rewrite the inequality as follows:
    # k > align_2MB(b*x)
    # where
    #    k = a / c

    # After that, we can derive the following equation:
    # x = floor(2**21 / b * floor((k - 1) / 2**21))

    num_key_value_heads = model_config.get_num_kv_heads(parallel_config)

    device_name = current_platform.get_device_name().lower()
    assert "rbln" in device_name
    if "ca" in device_name:
        # ATOM - RBLN-CA[xxx]
        # ATOM DRAM - 16GB (single chip)
        ATOM_DRAM_NBYTES = 16 * 2**30
        ATOM_SYS_DRAM_NBYTES = 288 * 2**20
        # consider RSD size for ATOM
        rsd_size = envs.VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK
        available_dram_bytes = rsd_size * (ATOM_DRAM_NBYTES - ATOM_SYS_DRAM_NBYTES)
        # ATOM - basic data type fp16
        default_bits_per_param = 16
    elif "cr" in device_name:
        assert envs.VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK == 1
        # REBEL - RBLN-CR[xxx]
        REBEL_CHIPLET_SIZE = 4
        # single device == Quad chiplet
        rsd_size = REBEL_CHIPLET_SIZE
        available_dram_bytes = REBEL_DRAM_NBYTES
        if envs.VLLM_RBLN_USE_DYNAMIC_KV_CACHE:
            # Flag-gated: reading the driver would tie the default path's KV
            # size to a driver release, which is not this feature's to decide.
            try:
                sysfs_dram_total = read_rbln_card_dram_total_bytes()
            except RuntimeError as exc:
                # The reader refuses on heterogeneous cards; this estimate only
                # wants one card's capacity, so fall back rather than fail.
                logger.warning(
                    "%s; falling back to the built-in %d byte DRAM capacity.",
                    exc,
                    REBEL_DRAM_NBYTES,
                )
                sysfs_dram_total = None
            if sysfs_dram_total is None:
                logger.debug(
                    "sysfs %s is unavailable; falling back to the built-in %d "
                    "byte DRAM capacity.",
                    RBLN_SYSFS_CLASS_DIR,
                    REBEL_DRAM_NBYTES,
                )
            else:
                available_dram_bytes = sysfs_dram_total
        # FIXME(RBLN) - basic data type fp8 for REBEL, for now fp16
        default_bits_per_param = 16
    else:
        raise ValueError(
            "invalid RBLN architecture, candidates = [ATOM(ca), REBEL(cr)]"
        )

    num_runtimes = num_runtimes * rsd_size
    available_dram_bytes = int(available_dram_bytes * gpu_memory_utilization)

    def check_oom(available_dram_bytes: int) -> None:
        if available_dram_bytes <= 0:
            raise MemoryError(
                "Insufficient DRAM during block calculation. "
                "Try reducing gpu_memory_utilization."
            )

    if kernel_size is None:
        if n_model_params is None and n_model_bytes is None:
            raise ValueError(
                "Either `n_model_params` or `n_model_bytes` should be specified "
                "to estimate the kernel memory."
            )
        if n_model_params is not None and n_model_bytes is not None:
            raise ValueError(
                "Only one of `n_model_params` or `n_model_bytes` may be specified."
            )
        if n_model_params is not None and nbits_per_param is None:
            raise ValueError(
                "`nbits_per_param` should be specified when using `n_model_params` "
                "to estimate the kernel memory."
            )
        kernel_size = estimate_model_kernel_size(
            model_config=model_config,
            parallel_config=parallel_config,
            nbits_per_param=nbits_per_param,
            n_model_params=n_model_params,
            n_model_bytes=n_model_bytes,
            default_bits_per_param=default_bits_per_param,
        )
    elif n_model_params is not None or n_model_bytes is not None:
        raise ValueError(
            "`n_model_params`/`n_model_bytes` and `kernel_size` cannot both be "
            "specified."
        )

    available_dram_bytes -= kernel_size

    if buffer is None:
        # TODO: Accurate buffer estimation
        buffer_per_runtime_per_core = 2**28  # 256MB per runtime
        # 1 for prefill, 1 for decoder
        buffer = buffer_per_runtime_per_core * num_runtimes
    available_dram_bytes -= buffer

    # NOTE(RBLN): `max(1, rsd_size // kvh)` is optimistic unless kvh divides or
    # is a multiple of rsd_size. The exact factor is always >= it, so switching
    # under the flag can only shrink this estimate, never grow it into an OOM.
    rsd_replicas = max(1, rsd_size // num_key_value_heads)
    released_dram_bytes = available_dram_bytes // rsd_replicas
    exact_dram_bytes = divide_by_chiplet_replication(
        available_dram_bytes, num_key_value_heads, rsd_size
    )
    if released_dram_bytes != exact_dram_bytes:
        # NOTE(RBLN): only the flag path acts on this, so only it warns. The
        # default path keeps the released number and says so at debug level:
        # warning about an estimate this function is not changing reads as a
        # fault where there is none.
        disagreement = (
            "KV cache replication: num_key_value_heads=%d over %d chiplets is "
            "replicated %.4gx (%d heads per chiplet), but the released estimate "
            "divides by %d: %.3f GiB released vs %.3f GiB exact.%s"
        )
        args = (
            num_key_value_heads,
            rsd_size,
            chiplet_replication_factor(num_key_value_heads, rsd_size),
            math.ceil(num_key_value_heads / rsd_size),
            rsd_replicas,
            released_dram_bytes / 2**30,
            exact_dram_bytes / 2**30,
        )
        # One format string, two tails: `logger.*(a + b)` trips ruff G003.
        if envs.VLLM_RBLN_USE_DYNAMIC_KV_CACHE:
            logger.warning(disagreement, *args, " Using the exact figure.")
        else:
            logger.debug(disagreement, *args, " Keeping the released figure.")
    available_dram_bytes = (
        exact_dram_bytes if envs.VLLM_RBLN_USE_DYNAMIC_KV_CACHE else released_dram_bytes
    )

    check_oom(available_dram_bytes)

    return available_dram_bytes


def get_autobind_cpu_ids(
    rank: int,
    local_rank: int,
    parallel_config: ParallelConfig,
    cpu_selector: Callable[[list[LogicalCPUInfo]], list[LogicalCPUInfo]],
) -> str:
    """Get CPU IDs for automatic thread binding based on NUMA nodes.

    Args:
        rank: Global rank of the worker.
        local_rank: Local rank of the worker.
        parallel_config: Parallel configuration.
        cpu_selector: Function to select CPUs from each physical core.

    Returns:
        Comma-separated string of CPU IDs, or "all" or "nobind".
    """
    allowed_numa_nodes = get_visible_memory_node()
    logical_cpu_list = get_allowed_cpu_list()

    # Calculate rank_across_dp for CPU binding
    # This ensures different DP groups get different CPU allocations
    world_size = parallel_config.world_size
    if parallel_config.data_parallel_size > 1:
        world_size_across_dp = parallel_config.world_size_across_dp
        dp_rank = parallel_config.data_parallel_rank
        rank_across_dp = dp_rank * world_size + local_rank
    else:
        world_size_across_dp = world_size
        rank_across_dp = rank

    # Group CPUs by NUMA node
    numa_node_to_cpus: dict[int, list[LogicalCPUInfo]] = {}
    for cpu_info in logical_cpu_list:
        numa_node = cpu_info.numa_node
        if numa_node not in numa_node_to_cpus:
            numa_node_to_cpus[numa_node] = []
        numa_node_to_cpus[numa_node].append(cpu_info)

    # Filter to only allowed NUMA nodes
    available_numa_nodes = [n for n in allowed_numa_nodes if n in numa_node_to_cpus]

    if not available_numa_nodes:
        logger.error(
            "Auto thread-binding failed: no available NUMA nodes "
            "with allowed CPUs. Please try to bind threads manually."
        )
        return "all"

    numa_node_idx = rank_across_dp % len(available_numa_nodes)
    selected_numa_node = available_numa_nodes[numa_node_idx]
    numa_node_cpu_list = numa_node_to_cpus[selected_numa_node]
    ranks_in_same_numa = [
        r
        for r in range(world_size_across_dp)
        if r % len(available_numa_nodes) == numa_node_idx
    ]

    # Select CPUs from each physical core via cpu_selector
    core_to_cpus: dict[int, list[LogicalCPUInfo]] = {}
    for cpu_info in numa_node_cpu_list:
        if cpu_info.physical_core not in core_to_cpus:
            core_to_cpus[cpu_info.physical_core] = []
        core_to_cpus[cpu_info.physical_core].append(cpu_info)
    selected_cpu_list = []
    for cpu_list in core_to_cpus.values():
        cpu_list = sorted(cpu_list, key=lambda x: x.id)
        selected_cpu_list.extend(cpu_selector(cpu_list))
    selected_cpu_list = sorted(selected_cpu_list, key=lambda x: x.id)

    # Always divide CPUs among ranks in the same NUMA node
    # for exclusive allocation
    if len(ranks_in_same_numa) > 1:
        cpus_per_rank = len(selected_cpu_list) // len(ranks_in_same_numa)
        remainder = len(selected_cpu_list) % len(ranks_in_same_numa)

        rank_position = ranks_in_same_numa.index(rank_across_dp)
        start_idx = rank_position * cpus_per_rank + min(rank_position, remainder)
        end_idx = start_idx + cpus_per_rank + (1 if rank_position < remainder else 0)
        logical_cpu_list = selected_cpu_list[start_idx:end_idx]
    else:
        logical_cpu_list = selected_cpu_list

    if not logical_cpu_list:
        logger.warning(
            "Auto thread-binding: no CPUs allocated for rank %d "
            "(rank_across_dp %d). Falling back to default.",
            rank,
            rank_across_dp,
        )
        return "all"

    # Log binding information
    if len(ranks_in_same_numa) > 1:
        logger.debug(
            "auto thread-binding: rank %d (rank_across_dp %d) "
            "-> NUMA node %d, CPUs: %s (exclusive allocation, "
            "shared NUMA node with ranks %s, id, physical core): %s",
            rank,
            rank_across_dp,
            selected_numa_node,
            ",".join(str(x.id) for x in logical_cpu_list),
            [r for r in ranks_in_same_numa if r != rank_across_dp],
            [(x.id, x.physical_core) for x in logical_cpu_list],
        )
    else:
        logger.debug(
            "auto thread-binding: rank %d (rank_across_dp %d) "
            "-> NUMA node %d, CPUs: %s (exclusive allocation, "
            "id, physical core): %s",
            rank,
            rank_across_dp,
            selected_numa_node,
            ",".join(str(x.id) for x in logical_cpu_list),
            [(x.id, x.physical_core) for x in logical_cpu_list],
        )

    return ",".join([str(x.id) for x in logical_cpu_list])


def compute_rbln_local_omp_cpuid(
    rank: int,
    local_rank: int,
    parallel_config: ParallelConfig,
) -> str:
    """CPU set string that ``set_cpu_affinity`` will use (comma list, ``all``, or
    ``nobind``)."""
    if envs.VLLM_RBLN_NUMA and platform.system() == "Linux":
        cpu_arch = current_platform.get_cpu_architecture()
        if cpu_arch in (CpuArchEnum.POWERPC, CpuArchEnum.S390X):
            # For S390X/POWERPC SMT-8/4/2
            return get_autobind_cpu_ids(
                rank,
                local_rank,
                parallel_config,
                lambda cpus: [cpu for cpu in cpus if cpu.id % 8 < 4],
            )
        if cpu_arch == CpuArchEnum.X86:
            # For x86 SMT-2, use 1 CPU per core
            return get_autobind_cpu_ids(
                rank, local_rank, parallel_config, lambda cpus: cpus[:1]
            )
        return "nobind"
    return "nobind"


def get_rbln_planned_affinity_cpu_count(
    rank: int,
    local_rank: int,
    parallel_config: ParallelConfig,
) -> int:
    """Logical CPU count this rank will pin to after NUMA split (before
    ``sched_setaffinity``).

    Use this to size ``torch``/OpenMP threads before affinity is applied so thread
    counts match the post-bind CPU mask. If binding is ``nobind``/``all``, uses the
    current ``sched_getaffinity`` mask.
    """
    local_omp_cpuid = compute_rbln_local_omp_cpuid(rank, local_rank, parallel_config)
    if local_omp_cpuid not in ("all", "nobind"):
        cpu_ids = [int(x.strip()) for x in local_omp_cpuid.split(",") if x.strip()]
        return max(1, len(cpu_ids))
    return max(1, len(os.sched_getaffinity(0)))


def set_cpu_affinity(
    rank: int,
    local_rank: int,
    parallel_config: ParallelConfig,
) -> None:
    """Setup thread affinity based on NUMA nodes.

    Args:
        rank: Global rank of the worker.
        local_rank: Local rank of the worker.
        parallel_config: Parallel configuration.
    """
    local_omp_cpuid = compute_rbln_local_omp_cpuid(rank, local_rank, parallel_config)

    if local_omp_cpuid not in ("all", "nobind"):
        # Parse CPU IDs from string (e.g., "0,1,2,3" -> [0, 1, 2, 3])
        cpu_ids = [int(cpu_id.strip()) for cpu_id in local_omp_cpuid.split(",")]
        # Set CPU affinity for current process
        try:
            os.sched_setaffinity(0, cpu_ids)
            # Verify CPU affinity was set correctly
            actual_cpu_ids = sorted(os.sched_getaffinity(0))
            expected_cpu_ids = sorted(cpu_ids)
            if actual_cpu_ids != expected_cpu_ids:
                logger.warning(
                    "CPU affinity mismatch for rank %d (local_rank %d): "
                    "expected %s, but got %s",
                    rank,
                    local_rank,
                    expected_cpu_ids,
                    actual_cpu_ids,
                )
            else:
                logger.debug(
                    "Set CPU affinity for rank %d (local_rank %d): CPUs %s",
                    rank,
                    local_rank,
                    local_omp_cpuid,
                )
        except OSError as e:
            logger.error(
                "Failed to set CPU affinity for rank %d (local_rank %d): %s",
                rank,
                local_rank,
                str(e),
            )
            raise
    elif local_omp_cpuid == "nobind":
        logger.debug(
            "Skipping CPU affinity binding for rank %d (local_rank %d): nobind",
            rank,
            local_rank,
        )


def set_omp_num_threads(
    rank: int,
    local_rank: int,
    default_num_threads: int = 2,
) -> None:
    """Set the number of threads for intra-op parallelism in this process.

    This function sets the thread count using torch.set_num_threads(),
    which directly controls the OpenMP/MKL thread pool for the current
    process only, regardless of when it's called.

    Args:
        rank: Global rank of the worker.
        local_rank: Local rank of the worker.
        default_num_threads: Number of threads to use if RBLN_NUM_THREADS
            is not set. Defaults to 2.
    """

    # Determine the number of threads to use
    if "RBLN_NUM_THREADS" in os.environ:
        num_threads = int(os.environ["RBLN_NUM_THREADS"])
    else:
        num_threads = default_num_threads
        # Set env var for any future subprocesses
        os.environ["RBLN_NUM_THREADS"] = str(num_threads)

    # Directly set PyTorch's thread count for this process
    torch.set_num_threads(num_threads)

    logger.debug(
        "Set torch.num_threads to %d for rank %d (local_rank %d)",
        num_threads,
        rank,
        local_rank,
    )


def prepare_kernel_block_sizes(
    kv_cache_config: KVCacheConfig, attn_groups: list[list[AttentionGroup]]
) -> list[int]:
    """
    Generate kernel_block_sizes that matches each block_size.

    For attention backends that support virtual block splitting,
    use the supported block sizes from the backend.
    For other backends (like Mamba), use the same block size (no splitting).

    Args:
        kv_cache_config: The KV cache configuration.
        attn_groups: Attention groups indexed by KV cache group id.

    Returns:
        List of kernel block sizes for each cache group.
    """
    kernel_block_sizes = []
    for kv_cache_gid, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
        kv_cache_spec = kv_cache_group.kv_cache_spec
        if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
            # All layers in the UniformTypeKVCacheSpecs have the same type,
            # pick an arbitrary one to dispatch.
            kv_cache_spec = next(iter(kv_cache_spec.kv_cache_specs.values()))
        if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
            continue
        if isinstance(kv_cache_spec, RBLNSlidingWindowSpec):
            kernel_block_sizes.append(kv_cache_spec.sliding_window)
        elif isinstance(kv_cache_spec, AttentionSpec):
            # This is an attention backend that supports virtual block splitting.
            kv_manager_block_size = kv_cache_group.kv_cache_spec.block_size
            group_backends = [g.backend for g in attn_groups[kv_cache_gid]]
            selected_kernel_size = select_common_block_size(
                kv_manager_block_size, group_backends
            )
            kernel_block_sizes.append(selected_kernel_size)
        elif isinstance(kv_cache_spec, MambaSpec):
            # This is likely Mamba or other non-attention cache, no splitting.
            kernel_block_sizes.append(kv_cache_spec.block_size)
        else:
            raise NotImplementedError(
                f"unknown kv cache spec {kv_cache_group.kv_cache_spec}"
            )
    return kernel_block_sizes


def reorder_input_batch(input_batch: "InputBatch", perm: np.ndarray) -> None:
    """Permute every per-request field of ``input_batch`` in one vectorized
    pass (new slot ``k`` takes old index ``perm[k]``).

    Mirrors upstream ``InputBatch.swap_states`` (vllm 0.22.0) but reindexes each
    field once instead of via N-1 pairwise swaps; the caller emits the
    logits-processor move records. Keep the field set in sync with
    ``swap_states`` on vLLM bumps -- ``test_reorder_matches_swap_states`` guards
    equivalence.
    """
    ib = input_batch
    n = len(perm)
    # numpy index: valid for advanced indexing of both numpy arrays and tensors.
    p = np.asarray(perm)

    # token_ids_cpu / is_token_ids rows are max_model_len wide but only
    # [:num_tokens + spec] is meaningful, so reindex just those columns. valid_w
    # is permutation-invariant (max over the same values), so read it first.
    max_spec = max((len(s) for s in ib.spec_token_ids[:n]), default=0)
    valid_w = min(
        int(ib.num_tokens_no_spec[:n].max()) + max_spec,
        ib.token_ids_cpu.shape[1],
    )

    # request id / token bookkeeping (python lists + index dict)
    ib._req_ids[:n] = [ib._req_ids[i] for i in p]
    ib.req_output_token_ids[:n] = [ib.req_output_token_ids[i] for i in p]
    ib.spec_token_ids[:n] = [ib.spec_token_ids[i] for i in p]
    for k in range(n):
        rid = ib._req_ids[k]
        if rid is not None:
            ib.req_id_to_index[rid] = k

    # per-request scalars; RHS fancy-index copies, so in-place assign is alias-safe
    for arr in (
        ib.num_tokens_no_spec,
        ib.num_prompt_tokens,
        ib.num_computed_tokens_cpu,
    ):
        arr[:n] = arr[p]

    ib.token_ids_cpu[:n, :valid_w] = ib.token_ids_cpu[p, :valid_w]
    ib.is_token_ids[:n, :valid_w] = ib.is_token_ids[p, :valid_w]

    if ib.req_prompt_embeds:
        ib.req_prompt_embeds = {
            k: ib.req_prompt_embeds[int(p[k])]
            for k in range(n)
            if int(p[k]) in ib.req_prompt_embeds
        }

    # block-table CPU rows + counts (device copy re-synced downstream)
    for bt in ib.block_table.block_tables:
        bt.num_blocks_per_row[:n] = bt.num_blocks_per_row[p]
        bt.block_table.np[:n] = bt.block_table.np[p]

    ib.request_lora_mapping[:n] = ib.request_lora_mapping[p]

    # Pooling models carry no sampling / logits state.
    if ib.is_pooling_model:
        return

    for arr in (
        ib.temperature_cpu,
        ib.top_p_cpu,
        ib.top_k_cpu,
        ib.frequency_penalties_cpu,
        ib.presence_penalties_cpu,
        ib.repetition_penalties_cpu,
        ib.num_accepted_tokens_cpu,
    ):
        arr[:n] = arr[p]

    # index-keyed dicts: new slot k inherits old slot p[k]'s entry
    ib.generators = {
        k: ib.generators[int(p[k])] for k in range(n) if int(p[k]) in ib.generators
    }
    ib.bad_words_token_ids = {
        k: ib.bad_words_token_ids[int(p[k])]
        for k in range(n)
        if int(p[k]) in ib.bad_words_token_ids
    }

    if ib.allowed_token_ids_mask_cpu_tensor is not None:
        ib.allowed_token_ids_mask_cpu_tensor[:n] = ib.allowed_token_ids_mask_cpu_tensor[
            p
        ]


def get_kv_cache_names(
    kv_caches: dict[str, torch.Tensor],
    num_attn_module: int = 1,
) -> list[str]:
    """Return KV cache layer names ordered by layer index.

    A deterministic, hash-seed-independent ordering is required by the KV
    connector: NIXL assigns transfer region indices in iteration order, so the
    P/D region <-> layer agreement breaks if the order varies between runs.
    Adapted from ``vllm.v1.worker.utils.bind_kv_cache``.
    """
    index2name: dict[int, list[str]] = defaultdict(list)
    for layer_name in kv_caches:
        index2name[extract_layer_index(layer_name, num_attn_module)].append(layer_name)

    kv_cache_names: list[str] = []
    for layer_index in sorted(index2name.keys()):
        layer_names = index2name[layer_index]
        if len(layer_names) > 1 and not (
            current_platform.is_cuda_alike()
            or current_platform.is_xpu()
            or current_platform.is_cpu()
        ):
            # Multiple layers sharing one index (e.g. encoder-decoder cross +
            # self attention) is only known-safe on GPU/CPU runners.
            raise NotImplementedError
        kv_cache_names.extend(layer_names)
    return kv_cache_names
