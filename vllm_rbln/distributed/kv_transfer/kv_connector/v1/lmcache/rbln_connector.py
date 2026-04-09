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

"""RBLN NPU connector for LMCache.

Handles KV cache transfers between RBLN NPU device memory and CPU memory
using the rebel runtime API:
- D2H: runtime.fetch_kv_cache(data_ptr, block_idx, offset, block_size, kv_name)
- H2D: runtime.update_kv_cache(data_ptr, block_idx, offset, block_size, kv_name)

Performance optimizations over the naive per-layer/per-token approach:
1. All-layer DMA: ``layer_name=None`` transfers all layers in a single
   DMA call instead of one call per layer.
2. Vectorized scatter/gather: torch advanced indexing replaces
   per-token Python loops.
3. Full-block H2D skip RMW: when all slots in a block are written,
   the read-modify-write fetch is skipped.
"""

import time
from collections import defaultdict
from typing import List, Optional, Union

from typing_extensions import override

import torch
from lmcache.v1.gpu_connector.gpu_connectors import GPUConnectorInterface
from lmcache.v1.memory_management import MemoryObj
from rebel.kv_cache import aligned_tensor

from vllm_rbln.logger import init_logger

from .transfer_stats import TransferStats

logger = init_logger(__name__)


class RBLNConnector(GPUConnectorInterface):
    """RBLN NPU connector for KV cache transfer.

    Uses the rebel runtime's block-level fetch_kv_cache / update_kv_cache
    APIs with slot_mapping to transfer data between the paged NPU KV cache
    and CPU memory objects.

    The runtime is obtained lazily from ``runtime_holder``, a mutable list
    that the RBLN compile backend populates after model compilation.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int,
        dtype: torch.dtype,
        kv_cache_names: Optional[List[str]] = None,
        runtime_holder: Optional[list] = None,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.hidden_dim_size = num_kv_heads * head_dim
        self.block_size = block_size
        self.dtype = dtype
        self.kv_cache_names = kv_cache_names or []
        self._runtime_holder: list = (
            runtime_holder if runtime_holder is not None else []
        )

        self._xfer_buffer_all: Optional[torch.Tensor] = None
        self._xfer_layer_views: List[torch.Tensor] = []
        self._xfer_stacked: Optional[torch.Tensor] = None

        self._stats_d2h = TransferStats("D2H")
        self._stats_h2d = TransferStats("H2D")

    @property
    def runtime(self):
        """Lazily resolve the low-level rebel runtime from the holder.

        ``runtime_holder[0]`` may be a ``DynamoRuntime`` wrapper produced
        by ``torch.compile(backend='rbln')``.  The actual KV-cache API
        lives on its inner ``_runtime_handle`` (a ``PyRblnSyncRuntime``).
        """
        if not self._runtime_holder:
            return None
        rt = self._runtime_holder[0]
        try:
            from rebel.sync_runtime import DynamoRuntime

            if isinstance(rt, DynamoRuntime):
                return rt._runtime_handle
        except ImportError:
            pass
        return rt

    def set_runtime_holder(self, runtime_holder: list) -> None:
        self._runtime_holder = runtime_holder

    def set_runtime(self, runtime) -> None:
        self._runtime_holder = [runtime]

    def set_kv_cache_names(self, kv_cache_names: List[str]) -> None:
        self.kv_cache_names = kv_cache_names

    def initialize_xfer_buffers(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Create a single 4096-byte aligned transfer buffer for all-layer DMA.

        ``aligned_tensor()`` allocates as float16 (numpy has no bfloat16).
        ``.view(kv_dtype)`` reinterprets to the actual KV dtype — both are
        2 bytes so this is zero-copy and preserves alignment.
        """
        first_tensor = next(iter(kv_caches.values()))
        kv_dtype = first_tensor.dtype

        total_numel = sum(kv.numel() for kv in kv_caches.values())
        self._xfer_buffer_all = aligned_tensor(total_numel).view(kv_dtype)

        self._xfer_layer_views = []
        offset = 0
        for kv in kv_caches.values():
            layer_numel = kv.numel()
            view = self._xfer_buffer_all[offset : offset + layer_numel]
            self._xfer_layer_views.append(view.reshape(kv.shape))
            offset += layer_numel

        first_shape = first_tensor.shape
        self._xfer_stacked = self._xfer_buffer_all.reshape(len(kv_caches), *first_shape)

        logger.info(
            "Initialized xfer buffers: %d layers, %d elements, dtype=%s",
            len(kv_caches),
            total_numel,
            kv_dtype,
        )

    def _group_slots_by_block(
        self, slot_mapping: torch.Tensor
    ) -> dict[int, list[tuple[int, int]]]:
        block_indices = slot_mapping // self.block_size
        offsets = slot_mapping % self.block_size

        groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for token_idx, (blk, off) in enumerate(
            zip(block_indices.tolist(), offsets.tolist(), strict=True)
        ):
            groups[blk].append((off, token_idx))
        return groups

    @override
    def initialize_kvcaches_ptr(self, **kwargs):
        if "runtime_holder" in kwargs:
            self._runtime_holder = kwargs["runtime_holder"]
        elif "runtime" in kwargs:
            self._runtime_holder = [kwargs["runtime"]]
        if "kv_cache_names" in kwargs:
            self.kv_cache_names = kwargs["kv_cache_names"]

    # ------------------------------------------------------------------
    # D2H: NPU → CPU
    # ------------------------------------------------------------------

    @override
    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        assert memory_obj.tensor is not None
        slot_mapping = kwargs.get("slot_mapping")
        assert slot_mapping is not None, (
            "slot_mapping is required for RBLN D2H transfer"
        )
        assert self.runtime is not None, (
            "rebel runtime is required for RBLN D2H transfer"
        )
        assert len(self.kv_cache_names) > 0, (
            "kv_cache_names must be set before D2H transfer"
        )
        assert self._xfer_buffer_all is not None, (
            "xfer_buffers not initialized — call initialize_xfer_buffers first"
        )

        num_tokens = end - start
        hidden = self.hidden_dim_size
        dst = memory_obj.tensor.reshape(2, self.num_layers, num_tokens, hidden)
        needs_cast = dst.dtype != self._xfer_buffer_all.dtype

        chunk_slots = slot_mapping[start:end]
        block_groups = self._group_slots_by_block(chunk_slots)
        rt = self.runtime
        stacked = self._xfer_stacked
        num_layers = self.num_layers

        n_full = sum(
            1 for slots in block_groups.values() if len(slots) == self.block_size
        )
        logger.debug(
            "[D2H] tokens=%d→%d (%d) blocks=%d (full=%d partial=%d)",
            start,
            end,
            num_tokens,
            len(block_groups),
            n_full,
            len(block_groups) - n_full,
        )

        t_start = time.perf_counter()
        t_dma = 0.0
        t_sg = 0.0

        for block_idx, slots in block_groups.items():
            t0 = time.perf_counter()
            rt.fetch_kv_cache(
                self._xfer_buffer_all.data_ptr(),
                block_idx,
                0,
                self.block_size,
                None,
            )
            t_dma += time.perf_counter() - t0

            off_list = [s[0] for s in slots]
            tok_list = [s[1] for s in slots]
            n_slots = len(slots)

            # 6D layout: [layers, 2, num_blocks, num_kv_heads, 1, block_size, head_dim]
            all_k = stacked[:, 0, block_idx, :, 0, :, :]
            all_v = stacked[:, 1, block_idx, :, 0, :, :]

            t1 = time.perf_counter()
            if n_slots == self.block_size:
                sorted_pairs = sorted(
                    zip(off_list, tok_list, strict=True),
                    key=lambda x: x[0],
                )
                sorted_tok = [p[1] for p in sorted_pairs]

                flat_k = all_k.permute(0, 2, 1, 3).reshape(
                    num_layers, self.block_size, -1
                )
                flat_v = all_v.permute(0, 2, 1, 3).reshape(
                    num_layers, self.block_size, -1
                )

                tok_start = sorted_tok[0]
                is_contiguous = sorted_tok[-1] == tok_start + self.block_size - 1
                if is_contiguous:
                    tok_end = tok_start + self.block_size
                    if needs_cast:
                        dst[0, :, tok_start:tok_end] = flat_k.to(dst.dtype)
                        dst[1, :, tok_start:tok_end] = flat_v.to(dst.dtype)
                    else:
                        dst[0, :, tok_start:tok_end] = flat_k
                        dst[1, :, tok_start:tok_end] = flat_v
                else:
                    tok_idx = torch.tensor(sorted_tok, dtype=torch.long)
                    if needs_cast:
                        dst[0, :, tok_idx] = flat_k.to(dst.dtype)
                        dst[1, :, tok_idx] = flat_v.to(dst.dtype)
                    else:
                        dst[0, :, tok_idx] = flat_k
                        dst[1, :, tok_idx] = flat_v
            else:
                off_t = torch.tensor(off_list, dtype=torch.long)
                tok_t = torch.tensor(tok_list, dtype=torch.long)

                sel_k = all_k[:, :, off_t, :]
                sel_v = all_v[:, :, off_t, :]
                flat_k = sel_k.permute(0, 2, 1, 3).reshape(num_layers, n_slots, -1)
                flat_v = sel_v.permute(0, 2, 1, 3).reshape(num_layers, n_slots, -1)

                dst[0, :, tok_t] = flat_k.to(dst.dtype) if needs_cast else flat_k
                dst[1, :, tok_t] = flat_v.to(dst.dtype) if needs_cast else flat_v
            t_sg += time.perf_counter() - t1

        t_total = time.perf_counter() - t_start
        nbytes = num_tokens * hidden * 2 * num_layers * dst.element_size()
        self._stats_d2h.record(nbytes, t_total, t_dma, t_sg)
        logger.debug(
            "[D2H] done %.2fms (dma=%.2fms sg=%.2fms) %.2fMB %.2fGB/s",
            t_total * 1000,
            t_dma * 1000,
            t_sg * 1000,
            nbytes / 1e6,
            nbytes / t_total / 1e9 if t_total > 0 else 0.0,
        )

    # ------------------------------------------------------------------
    # H2D: CPU → NPU
    # ------------------------------------------------------------------

    @override
    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        assert memory_obj.tensor is not None
        slot_mapping = kwargs.get("slot_mapping")
        assert slot_mapping is not None, (
            "slot_mapping is required for RBLN H2D transfer"
        )
        assert self.runtime is not None, (
            "rebel runtime is required for RBLN H2D transfer"
        )
        assert len(self.kv_cache_names) > 0, (
            "kv_cache_names must be set before H2D transfer"
        )
        assert self._xfer_buffer_all is not None, (
            "xfer_buffers not initialized — call initialize_xfer_buffers first"
        )

        num_tokens = end - start
        hidden = self.hidden_dim_size
        src = memory_obj.tensor.reshape(2, self.num_layers, num_tokens, hidden)
        needs_cast = src.dtype != self._xfer_buffer_all.dtype

        chunk_slots = slot_mapping[start:end]
        block_groups = self._group_slots_by_block(chunk_slots)
        rt = self.runtime
        stacked = self._xfer_stacked
        num_layers = self.num_layers

        n_full = sum(
            1 for slots in block_groups.values() if len(slots) == self.block_size
        )
        logger.debug(
            "[H2D] tokens=%d→%d (%d) blocks=%d (full=%d partial=%d)",
            start,
            end,
            num_tokens,
            len(block_groups),
            n_full,
            len(block_groups) - n_full,
        )

        t_start = time.perf_counter()
        t_dma = 0.0
        t_sg = 0.0

        for block_idx, slots in block_groups.items():
            is_full_block = len(slots) == self.block_size

            if not is_full_block:
                t0 = time.perf_counter()
                rt.fetch_kv_cache(
                    self._xfer_buffer_all.data_ptr(),
                    block_idx,
                    0,
                    self.block_size,
                    None,
                )
                t_dma += time.perf_counter() - t0

            off_list = [s[0] for s in slots]
            tok_list = [s[1] for s in slots]
            n_slots = len(slots)

            t1 = time.perf_counter()
            if is_full_block:
                sorted_pairs = sorted(
                    zip(off_list, tok_list, strict=True),
                    key=lambda x: x[0],
                )
                sorted_tok = [p[1] for p in sorted_pairs]

                tok_start = sorted_tok[0]
                is_contiguous = sorted_tok[-1] == tok_start + self.block_size - 1
                if is_contiguous:
                    tok_end = tok_start + self.block_size
                    src_k = src[0, :, tok_start:tok_end]
                    src_v = src[1, :, tok_start:tok_end]
                else:
                    tok_idx = torch.tensor(sorted_tok, dtype=torch.long)
                    src_k = src[0, :, tok_idx]
                    src_v = src[1, :, tok_idx]

                if needs_cast:
                    buf_dtype = self._xfer_buffer_all.dtype
                    src_k = src_k.to(buf_dtype)
                    src_v = src_v.to(buf_dtype)

                reshaped_k = src_k.reshape(
                    num_layers, self.block_size, self.num_kv_heads, self.head_dim
                ).permute(0, 2, 1, 3)
                reshaped_v = src_v.reshape(
                    num_layers, self.block_size, self.num_kv_heads, self.head_dim
                ).permute(0, 2, 1, 3)
                stacked[:, 0, block_idx, :, 0, :, :] = reshaped_k
                stacked[:, 1, block_idx, :, 0, :, :] = reshaped_v
            else:
                off_t = torch.tensor(off_list, dtype=torch.long)
                tok_t = torch.tensor(tok_list, dtype=torch.long)

                src_k = src[0, :, tok_t]
                src_v = src[1, :, tok_t]

                if needs_cast:
                    buf_dtype = self._xfer_buffer_all.dtype
                    src_k = src_k.to(buf_dtype)
                    src_v = src_v.to(buf_dtype)

                reshaped_k = src_k.reshape(
                    num_layers, n_slots, self.num_kv_heads, self.head_dim
                ).permute(0, 2, 1, 3)
                reshaped_v = src_v.reshape(
                    num_layers, n_slots, self.num_kv_heads, self.head_dim
                ).permute(0, 2, 1, 3)
                stacked[:, 0, block_idx, :, 0, off_t, :] = reshaped_k
                stacked[:, 1, block_idx, :, 0, off_t, :] = reshaped_v
            t_sg += time.perf_counter() - t1

            t0 = time.perf_counter()
            rt.update_kv_cache(
                self._xfer_buffer_all.data_ptr(),
                block_idx,
                0,
                self.block_size,
                None,
            )
            t_dma += time.perf_counter() - t0

        t_total = time.perf_counter() - t_start
        nbytes = num_tokens * hidden * 2 * num_layers * src.element_size()
        self._stats_h2d.record(nbytes, t_total, t_dma, t_sg)
        logger.debug(
            "[H2D] done %.2fms (dma=%.2fms sg=%.2fms) %.2fMB %.2fGB/s",
            t_total * 1000,
            t_dma * 1000,
            t_sg * 1000,
            nbytes / 1e6,
            nbytes / t_total / 1e9 if t_total > 0 else 0.0,
        )

    # ------------------------------------------------------------------
    # Batched operations
    # ------------------------------------------------------------------

    @override
    def batched_from_gpu(
        self,
        memory_objs: Union[List[List[MemoryObj]], List[MemoryObj]],
        starts: List[int],
        ends: List[int],
        **kwargs,
    ):
        for memory_obj, s, e in zip(memory_objs, starts, ends, strict=True):
            self.from_gpu(memory_obj, s, e, **kwargs)

    @override
    def batched_to_gpu(
        self,
        memory_objs: Union[
            List[List[MemoryObj]], List[MemoryObj], List[int], None
        ] = None,
        starts: Optional[List[int]] = None,
        ends: Optional[List[int]] = None,
        **kwargs,
    ):
        if memory_objs is None or starts is None or ends is None:
            return
        for memory_obj, s, e in zip(memory_objs, starts, ends, strict=True):
            self.to_gpu(memory_obj, s, e, **kwargs)

    @override
    def get_shape(self, num_tokens: int) -> torch.Size:
        return torch.Size([2, self.num_layers, num_tokens, self.hidden_dim_size])

    def get_transfer_stats(self) -> dict:
        return {
            "d2h": {
                "calls": self._stats_d2h.calls,
                "avg_total_ms": self._stats_d2h.get_avg_total_ms(),
                "avg_dma_ms": self._stats_d2h.get_avg_dma_ms(),
                "avg_scatter_gather_ms": (self._stats_d2h.get_avg_scatter_gather_ms()),
                "throughput_gbps": self._stats_d2h.get_throughput_gbps(),
                "total_mb": self._stats_d2h.get_total_mb(),
            },
            "h2d": {
                "calls": self._stats_h2d.calls,
                "avg_total_ms": self._stats_h2d.get_avg_total_ms(),
                "avg_dma_ms": self._stats_h2d.get_avg_dma_ms(),
                "avg_scatter_gather_ms": (self._stats_h2d.get_avg_scatter_gather_ms()),
                "throughput_gbps": self._stats_h2d.get_throughput_gbps(),
                "total_mb": self._stats_h2d.get_total_mb(),
            },
        }
