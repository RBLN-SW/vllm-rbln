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
"""Attention layer with FlashAttention."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import rebel  # noqa: F401 — registers rbln custom ops (rebel.ops.torch_custom_ops)
import torch
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadataBuilder,
    AttentionType,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
from vllm.v1.attention.backends.utils import (
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import InputBatch

import vllm_rbln.rbln_envs as envs
import vllm_rbln.utils as rbln_utils
from vllm_rbln.logger import init_logger
from vllm_rbln.v1.attention.kv_cache_bindings import KVCacheViewInfo

logger = init_logger(__name__)


@register_backend(AttentionBackendEnum.FLASH_ATTN)
class RBLNAttentionBackend(AttentionBackend):
    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [32, 64, 80, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        # Must match AttentionBackendEnum (see Attention.__init__ in v0.18+). This
        # backend is registered via @register_backend(AttentionBackendEnum.FLASH_ATTN).
        return "FLASH_ATTN"

    @staticmethod
    def get_impl_cls() -> type["RBLNFlashAttentionImpl"]:
        return RBLNFlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["RBLNFlashAttentionMetadataBuilder"]:
        return RBLNFlashAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        """kv cache shape
        # B - num_blocks == num_partitions
        # S - block_size == partition_size
        # H - num_kv_heads
        # G - num_heads / num_kv_heads = 32/8 = 4
        # D - head_size
        # L - q_len
        list of kv cache = [num_layer][kv=2]
        kv_cache_shape= [B, H, 1, S, D]
        query_shape   = [1, H, G, L, D]
        """
        return (2, num_blocks, num_kv_heads, 1, block_size, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: dict[int, int],
    ) -> None:
        raise RuntimeError("swap_blocks is not used for the RBLN backend.")

    @staticmethod
    def copy_blocks(
        kv_caches: list[torch.Tensor],
        src_to_dists: dict[int, list[int]],
    ) -> None:
        raise RuntimeError("swap_blocks is not used for the RBLN backend.")


@dataclass
class RBLNFlashAttentionMetadata:
    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_tables: torch.Tensor
    slot_mapping: torch.Tensor

    # For cascade attention.
    use_cascade: bool | None
    common_prefix_len: int | None
    cu_prefix_query_lens: torch.Tensor | None
    prefix_kv_lens: torch.Tensor | None
    suffix_kv_lens: torch.Tensor | None

    # Optional aot scheduling
    scheduler_metadata: torch.Tensor | None = None
    prefix_scheduler_metadata: torch.Tensor | None = None

    # To distinguish prefill and decode
    is_prefill: bool = True

    # For RBLN Attention
    attn_masks: torch.Tensor | None = None
    kv_caches: list[torch.Tensor] | None = None
    kv_cache_view_infos: list[KVCacheViewInfo] | None = None
    # for sliding window attention
    cache_seq_lens: torch.Tensor | None = None
    cache_offsets: torch.Tensor | None = None
    local_block_tables: torch.Tensor | None = None
    swa_attn_masks: torch.Tensor | None = None


class RBLNFlashAttentionMetadataBuilder(
    AttentionMetadataBuilder[RBLNFlashAttentionMetadata]
):
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.kv_cache_spec = kv_cache_spec
        self.layer_names = layer_names
        self.vllm_config = vllm_config
        self.device = device

        self.model_config = vllm_config.model_config
        self.parallel_config = vllm_config.parallel_config
        self.cache_config = vllm_config.cache_config
        self.compilation_config = vllm_config.compilation_config
        self.scheduler_config = vllm_config.scheduler_config

        # self.runner = runner
        # self.input_batch = runner.input_batch
        self.num_heads_q = self.model_config.get_num_attention_heads(
            self.parallel_config
        )
        self.num_heads_kv = self.model_config.get_num_kv_heads(self.parallel_config)
        self.kv_cache_dtype = kv_cache_spec.dtype
        self.headdim = self.model_config.get_head_size()
        self.block_size = kv_cache_spec.block_size

        self.chunked_prefill = (
            self.scheduler_config.enable_chunked_prefill
            or self.cache_config.enable_prefix_caching
        )
        self.chunked_prefill_size = self.scheduler_config.max_num_batched_tokens

        self.enforce_eager = get_current_vllm_config().model_config.enforce_eager

        self.is_causal = envs.VLLM_RBLN_FLASH_CAUSAL_ATTN

        self._swa_cache_seq_lens_buf: torch.Tensor | None = None
        self._swa_cache_offsets_buf: torch.Tensor | None = None

    def _to_device_inplace(
        self, cpu_tensor: torch.Tensor, attr_name: str
    ) -> torch.Tensor:
        buf: torch.Tensor | None = getattr(self, attr_name)
        if (
            buf is None
            or buf.shape != cpu_tensor.shape
            or buf.dtype != cpu_tensor.dtype
        ):
            buf = torch.empty(
                cpu_tensor.shape, dtype=cpu_tensor.dtype, device=self.device
            )
            setattr(self, attr_name, buf)
        buf.copy_(cpu_tensor)
        return buf

    def reorder_batch(
        self, input_batch: "InputBatch", scheduler_output: "SchedulerOutput"
    ) -> bool:
        return False

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        positions=None,
        batch_pad=None,
        is_prefill=False,
    ) -> RBLNFlashAttentionMetadata:
        use_dt = envs.VLLM_RBLN_USE_DEVICE_TENSOR
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len
        query_max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_tables_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping
        if use_dt:
            # Prefer the pre-existing CPU copy to avoid an extra D2H sync;
            # arithmetic stays on CPU until .to(self.device) in the constructor.
            query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
            _seq_lens_cpu = common_attn_metadata._seq_lens_cpu
            seq_lens_cpu = (
                _seq_lens_cpu[:num_reqs]
                if _seq_lens_cpu is not None
                else seq_lens[:num_reqs].cpu()
            )
            query_seq_lens = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
            num_computed_tokens_cpu = seq_lens_cpu - query_seq_lens
            seq_idx = positions[query_start_loc_cpu[:num_reqs]].view(-1, 1)
        else:
            query_seq_lens = query_start_loc[1:] - query_start_loc[:-1]
            num_computed_tokens = seq_lens - query_seq_lens
            seq_idx = positions[query_start_loc[:num_reqs]].view(-1, 1)

        # custom (triton) kernel's to_dynamic_index rejects int64 seq_lens
        seq_idx = seq_idx.to(torch.int32)

        cu_prefix_query_lens = None
        prefix_kv_lens = None
        suffix_kv_lens = None
        prefix_scheduler_metadata = None

        max_seq_len = self.model_config.max_model_len

        assert batch_pad is not None, "batch_pad is required for RBLN Attention Backend"

        attn_masks = None
        if is_prefill:
            # NOTE(jiwoo.park) prefill's block_tables must be a 1D tensor.
            block_tables_tensor = block_tables_tensor[0]
            if not self.is_causal:
                prefill_chunk_size = self.chunked_prefill_size
                chunked_attention_mask = torch.zeros(
                    1,
                    1,
                    1,
                    prefill_chunk_size,
                    max_seq_len,
                    dtype=torch.float16 if self.enforce_eager else torch.float32,
                )
                causal_mask = 1 - torch.triu(
                    torch.ones(1, 1, prefill_chunk_size, prefill_chunk_size),
                    diagonal=1,
                )
                step = seq_idx[0]
                if step >= prefill_chunk_size:
                    chunked_attention_mask[:, :, :, :, :step] = 1
                chunked_attention_mask[:, :, :, :, step : step + prefill_chunk_size] = (
                    causal_mask
                )
                attn_masks = chunked_attention_mask
                attn_masks = attn_masks.to(self.device)
        else:
            # batch padding
            seq_idx = rbln_utils.pad(seq_idx, 0, batch_pad)
            block_tables_tensor = rbln_utils.pad(block_tables_tensor, 0, batch_pad)
            if not self.is_causal:
                decode_attention_mask = torch.zeros(
                    batch_pad,
                    1,
                    1,
                    1,
                    max_seq_len,
                    dtype=torch.float16 if self.enforce_eager else torch.float32,
                )
                for batch_index, batch_step in enumerate(
                    seq_lens_cpu if use_dt else seq_lens
                ):
                    decode_attention_mask[batch_index, :, :, :, : batch_step + 1] = 1
                attn_masks = decode_attention_mask
                attn_masks = attn_masks.to(self.device)

        cache_seq_lens = None
        cache_offsets = None
        local_block_tables = None
        swa_attn_masks = None
        if sliding_window := getattr(self.kv_cache_spec, "sliding_window", None):
            nct_src = (
                num_computed_tokens_cpu if use_dt else num_computed_tokens[:num_reqs]
            )
            sl_src = seq_lens_cpu if use_dt else seq_lens[:num_reqs]
            num_computed_tokens = nct_src.view(-1, 1)
            seq_lens = sl_src.view(-1, 1)
            query_lens = seq_lens - num_computed_tokens
            cache_seq_lens = torch.clamp(num_computed_tokens, max=sliding_window)
            cache_offsets = cache_seq_lens + query_lens
            if not is_prefill:
                cache_seq_lens = rbln_utils.pad(cache_seq_lens, 0, batch_pad)
                cache_offsets = rbln_utils.pad(cache_offsets, 0, batch_pad)
                # Generate sliding window attention mask for decode
                # mask[b, s] = 1.0 if s <= cache_seq_lens[b] else 0.0
                positions = torch.arange(sliding_window)[None, :]
                swa_attn_masks = torch.where(positions - cache_seq_lens > 0, 0.0, 1.0)[
                    :, None, None, :
                ]

            local_block_tables = block_tables_tensor[..., :1]

        attn_metadata = RBLNFlashAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=query_max_seq_len,
            seq_lens=seq_idx.to(self.device),
            block_tables=block_tables_tensor.to(self.device),
            slot_mapping=slot_mapping,
            use_cascade=False,
            common_prefix_len=common_prefix_len,
            scheduler_metadata=None,
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
            prefix_scheduler_metadata=prefix_scheduler_metadata,
            is_prefill=is_prefill,
            attn_masks=attn_masks,
            cache_seq_lens=self._to_device_inplace(
                cache_seq_lens, "_swa_cache_seq_lens_buf"
            )
            if cache_seq_lens is not None
            else None,
            cache_offsets=self._to_device_inplace(
                cache_offsets, "_swa_cache_offsets_buf"
            )
            if cache_offsets is not None
            else None,
            local_block_tables=local_block_tables.to(self.device)
            if local_block_tables is not None
            else None,
            swa_attn_masks=swa_attn_masks.to(self.device)
            if swa_attn_masks is not None
            else None,
        )

        return attn_metadata

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        return False


class RBLNFlashAttentionImpl(AttentionImpl[RBLNFlashAttentionMetadata]):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        vllm_config = get_current_vllm_config()
        self.enforce_eager = vllm_config.model_config.enforce_eager
        self.device = vllm_config.device_config.device
        self.block_size = vllm_config.cache_config.block_size
        self.max_model_len = vllm_config.model_config.max_model_len

        if kv_sharing_target_layer_name is not None:
            raise NotImplementedError("KV sharing is not supported in RBLN.")
        if logits_soft_cap is not None:
            logger.warning_once(
                "RBLN Attention Backend does not support logits soft cap. "
                "Outputs may be slightly off."
            )
            logits_soft_cap = None

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = torch.tensor(scale, device=self.device)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        # unused?
        self.need_mask = (
            self.alibi_slopes is not None or self.sliding_window is not None
        )

        supported_head_sizes = RBLNAttentionBackend.get_supported_head_sizes()
        if head_size not in supported_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by PagedAttention. "
                f"Supported head sizes are: {supported_head_sizes}."
            )
        if kv_cache_dtype != "auto":
            raise NotImplementedError(
                "Torch SDPA backend does not support FP8 KV cache. "
                "Please use xFormers backend instead."
            )
        self.attn_type = attn_type

        # TODO(RBLN): We need to apply sinks attn kernel.
        self.sinks = sinks
        if self.sinks is not None:
            assert self.sinks.shape[0] == num_heads, (
                "Sinks must have the same number of heads as the number of "
                "heads in the layer"
            )
            if len(self.sinks.size()) == 1:
                self.sinks = self.sinks[:, None]

        self.is_causal = envs.VLLM_RBLN_FLASH_CAUSAL_ATTN
        self.is_batch_attention_opt = envs.VLLM_RBLN_BATCH_ATTN_OPT
        self.is_normal = (self.block_size == self.max_model_len) and (
            self.sinks is None
        )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: RBLNFlashAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with xFormers and PagedAttention.

        Args:
            query:  shape = [num_tokens, num_heads, head_size]
            key:    shape = [num_tokens, num_kv_heads, head_size]
            value:  shape = [num_tokens, num_kv_heads, head_size]
            kv_cache shape= [2, num_blocks, num_kv_heads, 1,
                             block_size, head_size]

        Shape that we expect:
            kv_cache  = [2, num_blocks, num_kv_heads, 1, block_size, head_size]
            key       = [batch, num_kv_heads, 1, query_len, head_size]
            query     = [batch, num_kv_heads, num_queries_per_kv,
                         query_len, head_size]
            key_t     = [batch, num_kv_heads, 1, head_size, block_size]
        Returns:
            attn_out  = [num_tokens, num_heads, head_size] if output is given,
                        otherwise [batch, query_len, num_heads * head_size]

            hidden_size = num_heads * head_size
        """
        # B - num_blocks == num_partitions
        # S - block_size == partition_size
        # H - num_kv_heads
        # G - num_heads / num_kv_heads = 4
        # D - head_size
        # L - query length
        # C - max_seq_len
        # NB- num batch
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for "
                "RBLNFlashAttentionImpl"
            )

        # NOTE(RBLN): vLLM passes q/k/v as [num_tokens, heads, head_size].
        # Convert that single contract to RBLN's [batch, kv_heads, groups, len, dim].
        assert query.dim() == 3
        b_size = 1 if attn_metadata.is_prefill else attn_metadata.block_tables.size(0)
        q_len = query.size(0) // b_size
        query = query.view(b_size, q_len, self.num_heads, self.head_size)
        key = key.view(b_size, q_len, self.num_kv_heads, self.head_size)
        value = value.view(b_size, q_len, self.num_kv_heads, self.head_size)
        query = query.transpose(1, 2)
        query = query.view(
            b_size, self.num_kv_heads, self.num_queries_per_kv, q_len, self.head_size
        )
        key = key.transpose(1, 2)
        key = key.view(b_size, self.num_kv_heads, 1, q_len, self.head_size)
        value = value.transpose(1, 2)
        value = value.view(b_size, self.num_kv_heads, 1, q_len, self.head_size)

        # NOTE - for cache update,
        # slot mapping will be necessary from sequence index
        # slot_mapping = [block_number, block_offset]

        # flash_attention_naive extended to have cache update
        # cache update is included into flash attention
        # but not within partition loop
        # input = {q, k, v, kv_cache, mask, scalar_scale,
        # seq_lens, block_table, slot_mapping}
        # output = {attn_output}
        # q, k, v = [batch,H,G,L,D]
        # key/value cache = [B,H,1,S,D]
        # mask  = [1,1,1,L,C]
        # o = [batch,H,G,L,D]

        # build attention mask within [0, 1]
        # - attention mask SHOULD be causal mask based on query length
        # - attention mask is used for masked softmax not actual value
        # if there is not positional embedding,
        # it can be merged into attention mask
        # attn_masks = _make_alibi_bias(alibi_slopes, dtype, seq_lens)
        # seq_lens (B, 1)
        # block_tables tensor (1, num_blocks = 256)
        # ex) tensor[block0 : 0, block1 : 100,
        #  block2: 10, block3: 5, ...]
        # attn_output = [batch,H,4,L,D]
        assert kv_cache is not None

        if self.sliding_window is not None:
            assert self.sliding_window == kv_cache.size(-2), (
                "SWA kernel_block_size must match window_size"
            )
            assert attn_metadata.cache_seq_lens is not None
            assert attn_metadata.cache_offsets is not None
            if envs.VLLM_RBLN_COMPILE_MODEL:
                if envs.VLLM_RBLN_USE_CUSTOM_KERNEL:
                    sliding_window_attention_naive_prefill = (
                        torch.ops.rbln_triton_ops.sliding_window_attention_naive_prefill
                    )
                    sliding_window_attention_naive_decode = (
                        torch.ops.rbln_triton_ops.sliding_window_attention_naive_decode
                    )
                else:
                    sliding_window_attention_naive_prefill = (
                        torch.ops.rbln_custom_ops.sliding_window_attention_naive_prefill
                    )
                    sliding_window_attention_naive_decode = (
                        torch.ops.rbln_custom_ops.sliding_window_attention_naive_decode
                    )
            else:
                sliding_window_attention_naive_prefill = (
                    torch.ops.rbln_custom_ops.sliding_window_attention_naive_prefill
                )
                sliding_window_attention_naive_decode = (
                    torch.ops.rbln_custom_ops.sliding_window_attention_naive_decode
                )

            if not attn_metadata.is_prefill:
                decode_args = [
                    query,
                    key,
                    value,
                    kv_cache,
                    attn_metadata.cache_seq_lens,
                    attn_metadata.cache_offsets,
                    self.scale,
                    attn_metadata.local_block_tables,
                    self.scale,  # dummy
                ]
                if not envs.VLLM_RBLN_USE_CUSTOM_KERNEL:
                    use_swa_mask = self.is_batch_attention_opt and b_size > 1
                    decode_args.append(
                        attn_metadata.swa_attn_masks if use_swa_mask else None
                    )
                    decode_args.append(self.sinks)
                attn_output = sliding_window_attention_naive_decode(  # noqa: E501
                    *decode_args,
                )
            else:
                prefill_args = [
                    query,
                    key,
                    value,
                    kv_cache,
                    attn_metadata.cache_seq_lens,
                    attn_metadata.cache_offsets,
                    self.scale,
                    attn_metadata.local_block_tables,
                    self.scale,  # dummy
                ]
                if not envs.VLLM_RBLN_USE_CUSTOM_KERNEL:
                    prefill_args.append(self.sinks)
                attn_output = sliding_window_attention_naive_prefill(  # noqa: E501
                    *prefill_args
                )
        # actually non-flash paged attention DOES NOT use slot_mapping
        elif self.is_causal:
            if self.is_normal:
                assert attn_metadata.seq_lens is not None
                assert attn_metadata.block_tables is not None

                if envs.VLLM_RBLN_COMPILE_MODEL:
                    if envs.VLLM_RBLN_USE_CUSTOM_KERNEL:
                        causal_attention_naive_prefill = (
                            torch.ops.rbln_triton_ops.causal_attention_naive_prefill
                        )
                        causal_attention_naive_decode = (
                            torch.ops.rbln_triton_ops.causal_attention_naive_decode
                        )
                    else:
                        causal_attention_naive_prefill = (
                            torch.ops.rbln_custom_ops.causal_attention_naive_prefill
                        )
                        causal_attention_naive_decode = (
                            torch.ops.rbln_custom_ops.causal_attention_naive_decode
                        )

                if not attn_metadata.is_prefill:
                    decode_args = [
                        query,
                        key,
                        value,
                        kv_cache,
                        attn_metadata.seq_lens,
                        self.scale,
                        attn_metadata.block_tables,
                        self.scale,  # dummy (required by rbln_triton_ops signature)
                    ]
                    if not envs.VLLM_RBLN_USE_CUSTOM_KERNEL:
                        decode_args.append(self.sinks)
                    attn_output = causal_attention_naive_decode(  # noqa: E501
                        *decode_args,
                    )
                else:
                    prefill_args = [
                        query,
                        key,
                        value,
                        kv_cache,
                        attn_metadata.seq_lens,
                        self.scale,
                        attn_metadata.block_tables,
                        self.scale,  # dummy (required by rbln_triton_ops signature)
                    ]
                    if not envs.VLLM_RBLN_USE_CUSTOM_KERNEL:
                        prefill_args.append(self.sinks)
                    attn_output = causal_attention_naive_prefill(  # noqa: E501
                        *prefill_args,
                    )
            else:
                if envs.VLLM_RBLN_COMPILE_MODEL:
                    if envs.VLLM_RBLN_USE_CUSTOM_KERNEL:
                        flash_causal_attention_naive_prefill = (  # noqa: E501
                            torch.ops.rbln_triton_ops.flash_causal_attention_naive_prefill
                        )
                        flash_causal_attention_naive_decode = (  # noqa: E501
                            torch.ops.rbln_triton_ops.flash_causal_attention_naive_decode
                        )
                    else:
                        flash_causal_attention_naive_prefill = (  # noqa: E501
                            torch.ops.rbln_custom_ops.flash_causal_attention_naive_prefill
                        )
                        flash_causal_attention_naive_decode = (  # noqa: E501
                            torch.ops.rbln_custom_ops.flash_causal_attention_naive_decode
                        )
                else:
                    flash_causal_attention_naive_prefill = (
                        torch.ops.rbln_custom_ops.flash_causal_attention_naive_prefill
                    )
                    flash_causal_attention_naive_decode = (
                        torch.ops.rbln_custom_ops.flash_causal_attention_naive_decode
                    )

                if not attn_metadata.is_prefill:
                    decode_args = [
                        query,
                        key,
                        value,
                        kv_cache,
                        self.scale,
                        attn_metadata.seq_lens,
                        attn_metadata.block_tables,
                        self.scale,  # dummy
                    ]
                    if not envs.VLLM_RBLN_USE_CUSTOM_KERNEL:
                        decode_args.append(self.sinks)
                    attn_output = flash_causal_attention_naive_decode(  # noqa: E501
                        *decode_args,
                    )
                else:
                    prefill_args = [
                        query,
                        key,
                        value,
                        kv_cache,
                        self.scale,
                        attn_metadata.seq_lens,
                        attn_metadata.block_tables,
                        self.scale,  # dummy
                    ]
                    if not envs.VLLM_RBLN_USE_CUSTOM_KERNEL:
                        prefill_args.append(self.sinks)
                    attn_output = flash_causal_attention_naive_prefill(  # noqa: E501
                        *prefill_args,
                    )
        else:
            if self.is_normal:
                assert attn_metadata.attn_masks is not None
                assert attn_metadata.seq_lens is not None
                assert attn_metadata.block_tables is not None

                if envs.VLLM_RBLN_COMPILE_MODEL:
                    if envs.VLLM_RBLN_USE_CUSTOM_KERNEL:
                        attention_naive_prefill = (
                            torch.ops.rbln_triton_ops.attention_naive_prefill
                        )
                        attention_naive_decode = (
                            torch.ops.rbln_triton_ops.attention_naive_decode
                        )
                    else:
                        attention_naive_prefill = (
                            torch.ops.rbln_custom_ops.attention_naive_prefill
                        )
                        attention_naive_decode = (
                            torch.ops.rbln_custom_ops.attention_naive_decode
                        )

                if not attn_metadata.is_prefill:
                    decode_args = [
                        query,
                        key,
                        value,
                        kv_cache,
                        attn_metadata.attn_masks,
                        attn_metadata.seq_lens,
                        self.scale,
                        attn_metadata.block_tables,
                        self.scale,  # dummy (required by rbln_triton_ops signature)
                    ]
                    if not envs.VLLM_RBLN_USE_CUSTOM_KERNEL:
                        decode_args.append(self.sinks)
                    attn_output = attention_naive_decode(  # noqa: E501
                        *decode_args,
                    )
                else:
                    prefill_args = [
                        query,
                        key,
                        value,
                        kv_cache,
                        attn_metadata.attn_masks,
                        attn_metadata.seq_lens,
                        self.scale,
                        attn_metadata.block_tables,
                        self.scale,  # dummy (required by rbln_triton_ops signature)
                    ]
                    if not envs.VLLM_RBLN_USE_CUSTOM_KERNEL:
                        prefill_args.append(self.sinks)
                    attn_output = attention_naive_prefill(  # noqa: E501
                        *prefill_args,
                    )
            else:
                if envs.VLLM_RBLN_COMPILE_MODEL:
                    if envs.VLLM_RBLN_USE_CUSTOM_KERNEL:
                        flash_attention_naive_prefill = (
                            torch.ops.rbln_triton_ops.flash_attention_naive_prefill
                        )
                        flash_attention_naive_decode = (
                            torch.ops.rbln_triton_ops.flash_attention_naive_decode
                        )
                    else:
                        flash_attention_naive_prefill = (
                            torch.ops.rbln_custom_ops.flash_attention_naive_prefill
                        )
                        flash_attention_naive_decode = (
                            torch.ops.rbln_custom_ops.flash_attention_naive_decode
                        )
                else:
                    flash_attention_naive_prefill = (
                        torch.ops.rbln_custom_ops.flash_attention_naive_prefill
                    )
                    flash_attention_naive_decode = (
                        torch.ops.rbln_custom_ops.flash_attention_naive_decode
                    )

                if not attn_metadata.is_prefill:
                    decode_args = [
                        query,
                        key,
                        value,
                        kv_cache,
                        attn_metadata.attn_masks,
                        self.scale,
                        attn_metadata.seq_lens,
                        attn_metadata.block_tables,
                        self.scale,  # dummy
                    ]
                    if not envs.VLLM_RBLN_USE_CUSTOM_KERNEL:
                        decode_args.append(self.sinks)
                    attn_output = flash_attention_naive_decode(  # noqa: E501
                        *decode_args,
                    )
                else:
                    prefill_args = [
                        query,
                        key,
                        value,
                        kv_cache,
                        attn_metadata.attn_masks,
                        self.scale,
                        attn_metadata.seq_lens,
                        attn_metadata.block_tables,
                        self.scale,  # dummy
                    ]
                    if not envs.VLLM_RBLN_USE_CUSTOM_KERNEL:
                        prefill_args.append(self.sinks)
                    attn_output = flash_attention_naive_prefill(  # noqa: E501
                        *prefill_args,
                    )

        # 2. attention output reshape for attention backend return
        # attn_output = [batch,H*4,L,D] -> [batch,L,H*4,D] -> [batch,L,H*4*D]
        if self.enforce_eager or not envs.VLLM_RBLN_COMPILE_MODEL:
            attn_output = attn_output.reshape(
                b_size, self.num_heads, q_len, self.head_size
            ).transpose(1, 2)
            attn_output = attn_output.reshape(
                b_size, q_len, self.num_heads * self.head_size
            )
        else:
            attn_output = attn_output.view(
                b_size, self.num_heads, q_len, self.head_size
            ).transpose(1, 2)
            attn_output = attn_output.view(
                b_size, q_len, self.num_heads * self.head_size
            )
        # attn_output = [batch,L,H*4*D]
        if output is not None:
            attn_output = attn_output.view(
                b_size * q_len, self.num_heads, self.head_size
            )
            output.copy_(attn_output)
            return output
        return attn_output
