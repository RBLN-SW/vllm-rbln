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

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import torch
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.utils.torch_utils import is_quantized_kv_cache
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
from vllm.v1.kv_cache_interface import AttentionSpec

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import InputBatch

import vllm_rbln.envs as envs
import vllm_rbln.utils as rbln_utils
from vllm_rbln.logger import init_logger
from vllm_rbln.v1.attention.kv_cache_bindings import KVCacheViewInfo

from ..ops.attention_naive import (
    attention_naive_decode,
    attention_naive_prefill,
)
from ..ops.causal_attention_naive import (
    causal_attention_naive_decode,
    causal_attention_naive_prefill,
)
from ..ops.flash_attention_naive import (
    flash_attention_naive_decode,
    flash_attention_naive_prefill,
)
from ..ops.flash_causal_attention_naive import (
    flash_causal_attention_naive_decode,
    flash_causal_attention_naive_prefill,
)
from ..ops.sliding_window_attention import sliding_window_attention_v1

logger = init_logger(__name__)


def _fp8_cache_dtype(kv_cache_dtype: str) -> torch.dtype | None:
    """Real element dtype the uint8 fp8 KV-cache container holds, or None on
    the non-fp8 "auto" path (the cache tensor's own dtype is real). Upstream's
    kv_cache_dtype_str_to_dtype gives the uint8 byte-container dtype instead,
    so it cannot be used here. "fp8" is an alias of e4m3 upstream."""
    return {
        "fp8": torch.float8_e4m3fn,
        "fp8_e4m3": torch.float8_e4m3fn,
        "fp8_e5m2": torch.float8_e5m2,
    }.get(kv_cache_dtype)


@register_backend(AttentionBackendEnum.FLASH_ATTN)
class RBLNFlashAttentionBackend(AttentionBackend):
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
    ]

    @staticmethod
    def get_name() -> str:
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

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [32, 64, 80, 96, 128, 160, 192, 224, 256]


@dataclass
class RBLNFlashAttentionMetadata:
    seq_lens: torch.Tensor
    block_tables: torch.Tensor

    # For RBLN Attention
    is_prefill: bool
    attn_masks: torch.Tensor | None = None
    kv_caches: list[torch.Tensor] | None = None
    kv_cache_view_infos: list[KVCacheViewInfo] | None = None

    def __post_init__(self):
        # FIXME(RBLN): to_dynamic_index does not accept int64 inputs.Thus in the
        # VLLM_RBLN_USE_CUSTOM_KERNEL=0 path, rebel-compiler automatically converts
        # integer tensor inputs to a supported dtype.
        # However, this preprocessing is somewhat missing in the triton-rbln kernel
        # path(VLLM_RBLN_USE_CUSTOM_KERNEL=1), so we explicitly cast the input to a
        # supported dtype here. This can be removed when the triton-rbln kernel path
        # performs the same dtype conversion.

        if not envs.VLLM_RBLN_USE_CUSTOM_KERNEL:
            return

        self.seq_lens = self.seq_lens.to(torch.int32)


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
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.scheduler_config = vllm_config.scheduler_config

        self.block_size = kv_cache_spec.block_size
        self.chunked_prefill_size = self.scheduler_config.max_num_batched_tokens
        self.enforce_eager = get_current_vllm_config().model_config.enforce_eager
        # Causality is per model, not per process: each model builds its
        # attention under its own config, and upstream sets `use_non_causal` on
        # the draft config only, so a non-causal drafter and a causal target
        # coexist in one process.
        self.is_causal = (
            envs.VLLM_RBLN_FLASH_CAUSAL_ATTN
            and not vllm_config.attention_config.use_non_causal
        )

        self._staged: dict[tuple, torch.Tensor] = {}

    def _stage(self, t: torch.Tensor | None, slot: str) -> torch.Tensor | None:
        """Copy a host tensor into this builder's persistent device buffer.

        `slot` must be unique per logical tensor: two tensors that share a
        shape and a dtype would otherwise silently share one buffer.

        The returned tensor is overwritten by the next build() call, so the
        caller must run the forward pass before building again. That holds
        because each AttentionGroup owns its own builder instance and every
        build() is immediately followed by a forward.
        """
        if t is None:
            return None
        key = (slot, t.shape, t.dtype)
        if (buf := self._staged.get(key)) is None:
            buf = torch.empty(t.shape, dtype=t.dtype, device=self.device)
            self._staged[key] = buf
        buf.copy_(t)
        return buf

    def reorder_batch(
        self, input_batch: "InputBatch", scheduler_output: "SchedulerOutput"
    ) -> bool:
        return False

    def build(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        positions: torch.Tensor,
        batch_pad: int,
        is_prefill: bool,
        skip_attn_masks: bool = False,
    ) -> RBLNFlashAttentionMetadata:
        num_reqs = common_attn_metadata.num_reqs
        # NOTE(RBLN): vllm-rbln keeps attention metadata on the host and copies
        # to the device only when constructing RBLNFlashAttentionMetadata below.
        # See RBLNModelRunner._build_attention_metadata.
        query_start_loc_cpu = common_attn_metadata.query_start_loc
        seq_lens_cpu = common_attn_metadata.seq_lens
        block_tables_tensor = common_attn_metadata.block_table_tensor

        seq_idx = positions[query_start_loc_cpu[:num_reqs]].view(-1, 1)
        max_seq_len = self.model_config.max_model_len

        # Masks are host-built at `max_model_len` width and staged every step,
        # so a caller that replaces them opts out with `skip_attn_masks`.
        attn_masks = None
        if is_prefill:
            # NOTE(RBLN): block_tables_tensor for prefill must be a 1D tensor.
            block_tables_tensor = block_tables_tensor[0]
            if not self.is_causal and not skip_attn_masks:
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
        else:
            seq_idx = rbln_utils.pad(seq_idx, 0, batch_pad)
            block_tables_tensor = rbln_utils.pad(block_tables_tensor, 0, batch_pad)
            if not self.is_causal and not skip_attn_masks:
                decode_attention_mask = torch.zeros(
                    batch_pad,
                    1,
                    1,
                    1,
                    max_seq_len,
                    dtype=torch.float16 if self.enforce_eager else torch.float32,
                )
                for batch_index, batch_step in enumerate(seq_lens_cpu):
                    decode_attention_mask[batch_index, :, :, :, : batch_step + 1] = 1
                attn_masks = decode_attention_mask

        attn_metadata = RBLNFlashAttentionMetadata(
            seq_lens=self._stage(seq_idx, "seq_idx"),
            block_tables=self._stage(block_tables_tensor, "block_tables"),
            is_prefill=is_prefill,
            attn_masks=self._stage(attn_masks, "attn_masks"),
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

        supported_head_sizes = RBLNFlashAttentionBackend.get_supported_head_sizes()
        if head_size not in supported_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by RBLNFlashAttention. "
                f"Supported head sizes are: {supported_head_sizes}."
            )
        # supported_kv_cache_dtypes is not enforced upstream for out-of-tree
        # platforms (validate_configuration is never called), so this is the
        # gate.
        if is_quantized_kv_cache(self.kv_cache_dtype) and (
            self.kv_cache_dtype
            not in RBLNFlashAttentionBackend.supported_kv_cache_dtypes
        ):
            raise NotImplementedError(
                "RBLNFlashAttention does not support "
                f"kv_cache_dtype={self.kv_cache_dtype!r}; supported: "
                f"{RBLNFlashAttentionBackend.supported_kv_cache_dtypes}"
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

        self.is_causal = (
            envs.VLLM_RBLN_FLASH_CAUSAL_ATTN
            and not vllm_config.attention_config.use_non_causal
        )
        self.is_normal = (self.block_size == self.max_model_len) and (
            self.sinks is None
        )

        if self.sliding_window is not None and envs.VLLM_RBLN_USE_CUSTOM_KERNEL:
            raise NotImplementedError(
                "Sliding window attention is not supported with "
                "VLLM_RBLN_USE_CUSTOM_KERNEL=1: rbln_triton_ops has no "
                "sliding_window_attention_v1 kernel."
            )

        # forward() dispatches on (sliding_window, is_causal, is_normal), and
        # only the flash causal target hands the fp8 dequant scales and the
        # real cache dtype to the compiled op -- and only on the
        # rbln_custom_ops path (the rbln_triton_ops kernels take neither).
        # Every other target would read the uint8 byte container as raw bytes
        # with no scales, so reject those combinations here rather than
        # producing garbage. All the dispatch inputs are fixed by __init__.
        if self.kv_cache_dtype.startswith("fp8"):
            if envs.VLLM_RBLN_USE_CUSTOM_KERNEL:
                raise NotImplementedError(
                    "fp8 KV cache is not supported with "
                    "VLLM_RBLN_USE_CUSTOM_KERNEL=1: the rbln_triton_ops "
                    "attention kernels take no dequant scales."
                )
            if self.sliding_window is not None or not self.is_causal or self.is_normal:
                raise NotImplementedError(
                    "fp8 KV cache is only supported by the flash causal "
                    "attention path (VLLM_RBLN_FLASH_CAUSAL_ATTN=1, "
                    "block_size != max_model_len, no sliding window); got "
                    f"sliding_window={self.sliding_window}, "
                    f"is_causal={self.is_causal}, is_normal={self.is_normal}."
                )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: RBLNFlashAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> None:
        """Forward pass with RBLNFlashAttention.

        Args:
            query:  shape = [num_tokens, num_heads, head_size]
            key:    shape = [num_tokens, num_kv_heads, head_size]
            value:  shape = [num_tokens, num_kv_heads, head_size]
            kv_cache shape= [2, num_blocks, num_kv_heads, 1,
                                block_size, head_size]

        Shape that we expect:
            kv_cache  = [2, num_blocks, num_kv_heads, 1, block_size, head_size]
            key       = [1, num_kv_heads, 1, block_size, head_size]
            query     = [1, num_kv_heads, 4, query_len, head_size]
            key_t     = [1, num_kv_heads, 1, head_size, block_size]

        Returns:
            attn_out  = [num_tokens, num_heads, head_size]

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
            raise RuntimeError(
                "Fused output quantization is not yet supported for RBLNFlashAttention."
            )

        # 1. query reshape for custom operation
        num_tokens = query.shape[0]
        b_size = attn_metadata.seq_lens.shape[0]
        q_len = num_tokens // b_size
        query = query.view(b_size, q_len, self.num_heads, self.head_size).transpose(
            1, 2
        )
        query = query.view(
            b_size, self.num_kv_heads, self.num_queries_per_kv, q_len, self.head_size
        )
        key = key.view(b_size, q_len, self.num_kv_heads, self.head_size).transpose(1, 2)
        key = key.view(b_size, self.num_kv_heads, 1, q_len, self.head_size)
        value = value.view(b_size, q_len, self.num_kv_heads, self.head_size).transpose(
            1, 2
        )
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
        if self.sliding_window is not None:
            # One op for both phases. `seq_lens` is the absolute position the
            # chunk starts at, uncapped: it names the slot every token is
            # written to, and the blocks the window covers are resolved from it
            # and the whole block table inside the op.
            attn_output = sliding_window_attention_v1(
                query,
                key,
                value,
                kv_cache,
                attn_metadata.seq_lens,
                self.scale,
                attn_metadata.block_tables,
                self.sliding_window,
                self.sinks,
            )

        elif self.is_causal:
            if self.is_normal:
                if attn_metadata.is_prefill:
                    attn_output = causal_attention_naive_prefill(
                        query,
                        key,
                        value,
                        kv_cache,
                        attn_metadata.seq_lens,
                        self.scale,
                        attn_metadata.block_tables,
                        self.sinks,
                    )
                else:
                    attn_output = causal_attention_naive_decode(
                        query,
                        key,
                        value,
                        kv_cache,
                        attn_metadata.seq_lens,
                        self.scale,
                        attn_metadata.block_tables,
                        self.sinks,
                    )
            else:
                # * batched attention - seq_lens[B, 1] == seq_idx,
                #   original sequence index
                # * otherwise         - seq_lens[B, P] == seq_lens_tensor,
                #   dynamic size for each partition
                k_quantize_scale = layer._k_scale
                v_quantize_scale = layer._v_scale
                cache_dtype = _fp8_cache_dtype(self.kv_cache_dtype)
                if attn_metadata.is_prefill:
                    attn_output = flash_causal_attention_naive_prefill(
                        query,
                        key,
                        value,
                        kv_cache,
                        self.scale,
                        attn_metadata.seq_lens,
                        attn_metadata.block_tables,
                        self.sinks,
                        k_quantize_scale,
                        v_quantize_scale,
                        cache_dtype,
                    )
                else:
                    attn_output = flash_causal_attention_naive_decode(
                        query,
                        key,
                        value,
                        kv_cache,
                        self.scale,
                        attn_metadata.seq_lens,
                        attn_metadata.block_tables,
                        self.sinks,
                        k_quantize_scale,
                        v_quantize_scale,
                        cache_dtype,
                    )
        else:
            if self.is_normal:
                if attn_metadata.is_prefill:
                    attn_output = attention_naive_prefill(
                        query,
                        key,
                        value,
                        kv_cache,
                        attn_metadata.attn_masks,
                        attn_metadata.seq_lens,
                        self.scale,
                        attn_metadata.block_tables,
                        self.sinks,
                    )
                else:
                    attn_output = attention_naive_decode(
                        query,
                        key,
                        value,
                        kv_cache,
                        attn_metadata.attn_masks,
                        attn_metadata.seq_lens,
                        self.scale,
                        attn_metadata.block_tables,
                        self.sinks,
                    )
            else:
                if attn_metadata.is_prefill:
                    attn_output = flash_attention_naive_prefill(
                        query,
                        key,
                        value,
                        kv_cache,
                        attn_metadata.attn_masks,
                        self.scale,
                        attn_metadata.seq_lens,
                        attn_metadata.block_tables,
                        self.sinks,
                    )
                else:
                    attn_output = flash_attention_naive_decode(
                        query,
                        key,
                        value,
                        kv_cache,
                        attn_metadata.attn_masks,
                        self.scale,
                        attn_metadata.seq_lens,
                        attn_metadata.block_tables,
                        self.sinks,
                    )

        # 2. attention output reshape for attention backend return
        # attn_output = [batch,H*4,L,D] -> [batch,L,H*4,D] -> [batch*L,H*4,D]
        if self.enforce_eager or not envs.VLLM_RBLN_COMPILE_MODEL:
            attn_output = attn_output.reshape(
                b_size, self.num_heads, q_len, self.head_size
            ).transpose(1, 2)
            attn_output = attn_output.reshape(
                b_size * q_len, self.num_heads, self.head_size
            )
        else:
            attn_output = attn_output.view(
                b_size, self.num_heads, q_len, self.head_size
            ).transpose(1, 2)
            attn_output = attn_output.view(
                b_size * q_len, self.num_heads, self.head_size
            )

        output.copy_(attn_output)
