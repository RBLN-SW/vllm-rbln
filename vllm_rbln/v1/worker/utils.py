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
"""CPU affinity utilities for RBLN worker."""

import math
import os
import platform
from collections.abc import Callable

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

logger = init_logger(__name__)


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
        # REBEL DRAM - 144GB (quad chips, chiplet) - system(4G) = 140GB
        REBEL_DRAM_NBYTES = 144 * 2**30
        REBEL_SYS_DRAM_NBYTES = 4 * 2**30
        REBEL_DRAM_NBYTES -= REBEL_SYS_DRAM_NBYTES
        REBEL_CHIPLET_SIZE = 4
        # single device == Quad chiplet
        rsd_size = REBEL_CHIPLET_SIZE
        available_dram_bytes = REBEL_DRAM_NBYTES
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

    rsd_replicas = (rsd_size // num_key_value_heads) or 1 if "ca" in device_name else 1
    available_dram_bytes = available_dram_bytes // rsd_replicas

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
