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
"""DFlash speculative decoding for RBLN.

DFlash drafts all speculative tokens in one drafter forward per step:

1. Context-KV insert: target hidden states (concatenated aux layers) are
   projected into per-layer K/V and written into the drafter's KV cache.
   The drafter never runs over the prompt itself.
2. Block forward: one pass over ``[bonus_token, mask * K]`` per request with
   non-causal attention (bidirectional inside the block, full attention to
   the cached context); argmax at the K mask positions yields all drafts.

Compiled graphs, all static shapes captured during warmup:

* context KV insert -- prefill ``[1, max_num_batched_tokens]``, decode
  ``[bucket, 1]`` (no-spec steps) and ``[bucket, 1 + K]`` (verify steps)
* block forward     -- ``[bucket, 1 + K]``

The context KV write reuses the attention op's built-in cache update (k/v
are written at ``seq_idx .. seq_idx + L - 1`` through the block table) with
a dummy query whose attention output is discarded.
TODO(RBLN): replace with a dedicated KV-cache-update op when one is
available.
"""

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

import vllm_rbln.envs as envs
from vllm_rbln.compilation import build_process_group_dict, compile
from vllm_rbln.forward_context import set_forward_context
from vllm_rbln.logger import init_logger
from vllm_rbln.patches.attention import _resolve_kv_cache
from vllm_rbln.platform import USE_DEVICE_TENSOR
from vllm_rbln.v1.attention.backends.flash_attention import (
    RBLNFlashAttentionMetadata,
)
from vllm_rbln.v1.attention.kv_cache_bindings import (
    attach_kv_cache_bindings,
    build_kv_cache_forward_context_kwargs,
)
from vllm_rbln.v1.spec_decode.eagle import RBLNEagleProposer

if TYPE_CHECKING:
    from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

logger = init_logger(__name__)


def build_block_inputs_cpu(
    anchor_positions: torch.Tensor,
    num_reqs_padded: int,
    block_len: int,
    max_model_len: int,
    mask_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build drafter block positions and attention mask on the host.

    ``anchor_positions`` is the bonus-token position per request (right
    after the last accepted token). The mask allows columns
    ``[0, anchor + block_len)``: the block's K/V are written at
    anchor..anchor+K before attention runs, overwriting any rejected-token
    K/V in that range, so the visible extent is contiguous. Padded requests
    attend ``[0, block_len)`` to keep softmax rows non-empty.

    Returns ``positions [num_reqs_padded, block_len]`` and
    ``mask [num_reqs_padded, 1, 1, 1, max_model_len]``.
    """
    num_reqs = anchor_positions.shape[0]
    anchor_padded = torch.zeros(num_reqs_padded, dtype=torch.int64)
    anchor_padded[:num_reqs] = anchor_positions

    offsets = torch.arange(block_len, dtype=torch.int64)
    positions = anchor_padded.unsqueeze(1) + offsets.unsqueeze(0)
    positions.clamp_(max=max_model_len - 1)

    attend_end = (anchor_padded + block_len).clamp(max=max_model_len)
    cols = torch.arange(max_model_len, dtype=torch.int64)
    mask = (cols.unsqueeze(0) < attend_end.unsqueeze(1)).to(mask_dtype)
    mask = mask.view(num_reqs_padded, 1, 1, 1, max_model_len)

    return positions, mask


class RBLNDFlashProposer(RBLNEagleProposer):
    """DFlash proposer for RBLN NPUs.

    Reuses the RBLN EAGLE proposer's runner integration
    (``prepare_next_token_ids_padded`` / ``prepare_inputs_padded`` /
    ``_determine_draft_batch_padding``) and replaces the drafting itself
    with DFlash's single-pass block drafting.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner: "RBLNModelRunner",
    ):
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.method == "dflash"
        super().__init__(vllm_config, device, runner)

        # DFlash embeds the mask token via embed_tokens; checkpoints carry
        # no mask_hidden tensor, so the base-class load path must not look
        # for one.
        self.parallel_drafting_hidden_state_tensor = None

        draft_hf_config = self.draft_model_config.hf_config
        drafter_config = dict(getattr(draft_hf_config, "eagle_config", {}) or {})
        drafter_config.update(getattr(draft_hf_config, "dflash_config", {}) or {})

        if drafter_config.get("causal"):
            raise NotImplementedError(
                "vllm-rbln does not support causal DFlash variants yet."
            )
        if drafter_config.get("use_swa") or drafter_config.get("layer_types"):
            raise NotImplementedError(
                "vllm-rbln does not support DFlash drafters with sliding "
                "window attention yet."
            )
        if drafter_config.get("sample_from_anchor"):
            raise NotImplementedError(
                "sample_from_anchor is not supported (DSpark-style drafters)."
            )

        self.block_len = 1 + self.num_speculative_tokens
        self.use_aux_hidden_state = drafter_config.get("use_aux_hidden_state", True)
        if self.use_aux_hidden_state:
            num_features = draft_hf_config.num_hidden_layers
            layer_ids = drafter_config.get("target_layer_ids") or drafter_config.get(
                "layer_ids"
            )
            if layer_ids:
                num_features = len(layer_ids)
            target_hidden_size = getattr(
                draft_hf_config,
                "target_hidden_size",
                draft_hf_config.hidden_size,
            )
            self.context_feature_size = target_hidden_size * num_features
        else:
            self.context_feature_size = vllm_config.model_config.get_hidden_size()

        # Device buffers are allocated in load_model(): the runner's
        # bucketing manager does not exist yet at drafter construction.
        self.context_features: torch.Tensor | None = None
        self.context_positions: torch.Tensor | None = None
        self.block_input_ids: torch.Tensor | None = None
        self.block_positions: torch.Tensor | None = None

        self._staged: dict[tuple, torch.Tensor] = {}
        self.mask_dtype = (
            torch.float16 if vllm_config.model_config.enforce_eager else torch.float32
        )

    def _get_eagle3_use_aux_hidden_state_from_config(self) -> bool:
        dflash_config = getattr(
            self.draft_model_config.hf_config, "dflash_config", None
        )
        if dflash_config is not None:
            return dflash_config.get("use_aux_hidden_state", True)
        return True

    def _stage(self, t: torch.Tensor, slot: str) -> torch.Tensor:
        """Copy a host tensor into a persistent per-(slot, shape, dtype)
        device buffer, mirroring RBLNFlashAttentionMetadataBuilder._stage."""
        key = (slot, t.shape, t.dtype)
        if (buf := self._staged.get(key)) is None:
            buf = torch.empty(t.shape, dtype=t.dtype, device=self.device)
            self._staged[key] = buf
        buf.copy_(t)
        return buf

    def load_model(self, target_model: torch.nn.Module) -> None:
        # Bypass RBLNEagleProposer.load_model: it compiles the EAGLE
        # wrapper, which does not apply to DFlash.
        SpecDecodeBaseProposer.load_model(self, target_model)

        max_bucket = self.runner.bucketing_manager.decode_batch_buckets[-1]
        self.max_block_tokens = max_bucket * self.block_len
        max_ctx_tokens = max(self.max_num_tokens, self.max_block_tokens)

        self.context_features = torch.zeros(
            (max_ctx_tokens, self.context_feature_size),
            dtype=self.dtype,
            device=self.device,
        )
        self.context_positions = torch.zeros(
            max_ctx_tokens, dtype=torch.int64, device=self.device
        )
        self.block_input_ids = torch.zeros(
            self.max_block_tokens, dtype=torch.int32, device=self.device
        )
        self.block_positions = torch.zeros(
            self.max_block_tokens, dtype=torch.int64, device=self.device
        )

        def context_kv_wrapper(
            context_features: torch.Tensor,  # [B, L, feature_size]
            context_positions: torch.Tensor,  # [B, L] int64
        ):
            """Project target features into per-layer K/V and write them
            into the drafter KV cache via the attention op's cache update."""
            model = self.model.model
            features = self.model.combine_hidden_states(context_features)
            normed = model.hidden_norm(features)

            attn_metadata_dict = get_forward_context().attn_metadata
            batch_size, ctx_len = context_positions.shape
            num_tokens = batch_size * ctx_len

            out = None
            for decoder_layer in model.layers:
                attn_module = decoder_layer.self_attn
                attn = attn_module.attn
                head_dim = attn_module.head_dim
                kv_size = attn_module.kv_size

                kv_weight = attn_module.qkv_proj.weight[attn_module.q_size :]
                kv_bias = (
                    attn_module.qkv_proj.bias[attn_module.q_size :]
                    if attn_module.qkv_proj.bias is not None
                    else None
                )
                kv = F.linear(normed, kv_weight, kv_bias)
                k, v = kv.split([kv_size, kv_size], dim=-1)

                k_shape = k.shape
                k = attn_module.k_norm(
                    k.view(*k_shape[:-1], k_shape[-1] // head_dim, head_dim)
                ).view(k_shape)

                # Only K is roped; the patched RoPE signature requires a
                # query operand, so pass a minimal zero tensor.
                zero_q = k.new_zeros(batch_size, ctx_len, head_dim)
                _, k = attn_module.rotary_emb(context_positions, zero_q, k)

                attn_metadata = attn_metadata_dict[attn.layer_name]
                kv_cache = _resolve_kv_cache(attn_metadata, attn.layer_index)

                q_dummy = k.new_zeros(num_tokens, attn_module.num_heads * head_dim)
                out = k.new_empty(num_tokens, attn_module.num_heads, head_dim)
                attn.impl.forward(
                    attn,
                    q_dummy,
                    k.reshape(num_tokens, kv_size),
                    v.reshape(num_tokens, kv_size),
                    kv_cache,
                    attn_metadata,
                    output=out,
                )

            # The traced graph needs an output; the real effect is the
            # in-place KV cache update above.
            assert out is not None
            return out[:1, :1, :1]

        def block_wrapper(
            input_ids: torch.Tensor,  # [B, block_len] int32
            positions: torch.Tensor,  # [B, block_len] int64
        ):
            """One drafter forward over [bonus, mask * K]; returns argmax'd
            draft token ids (target vocab) at the K mask positions."""
            batch_size, block_len = input_ids.shape
            hidden_states = self.model(input_ids, positions, None)
            hidden_states = hidden_states.view(batch_size, block_len, -1)
            sample_hidden = hidden_states[:, 1:, :].reshape(
                batch_size * (block_len - 1), -1
            )
            logits = self.model.logits_processor(self.model.lm_head, sample_hidden)
            # int32 cast is required under compile: argmax returns int64.
            draft_ids = logits.argmax(dim=-1).to(torch.int32)
            d2t = self.model.draft_id_to_target_id
            if d2t is not None:
                draft_ids = draft_ids + d2t.index_select(
                    0, draft_ids.to(torch.int64)
                ).to(torch.int32)
            return draft_ids.view(batch_size, block_len - 1)

        if (
            self.vllm_config.speculative_config.enforce_eager
            or not envs.VLLM_RBLN_COMPILE_MODEL
        ):
            self.context_kv_executable = context_kv_wrapper
            self.model_executable = block_wrapper
        else:
            compile_kwargs = dict(
                dynamic=False,
                fullgraph=True,
                compile_context=self.runner.compile_context,
                num_devices=envs.VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK,
                model_trace_method="export" if USE_DEVICE_TENSOR else "",
                process_group_dict=build_process_group_dict(),
                guard_filter_fn=torch.compiler.keep_tensor_guards_unsafe,
                mode="strict" if envs.VLLM_RBLN_COMPILE_STRICT_MODE else "",
            )
            self.context_kv_executable = compile(context_kv_wrapper, **compile_kwargs)
            self.model_executable = compile(block_wrapper, **compile_kwargs)

        logger.info(
            "DFlash drafter loaded: block_len=%d (1 bonus + %d drafts), "
            "mask_token_id=%d, context_feature_size=%d",
            self.block_len,
            self.num_speculative_tokens,
            self.parallel_drafting_token_id,
            self.context_feature_size,
        )

    def _build_draft_attn_metadata(
        self,
        seq_idx_cpu: torch.Tensor,  # [num_reqs] int64
        block_tables_cpu: torch.Tensor,
        num_reqs: int,
        num_reqs_padded: int,
        is_prefill: bool,
        mask_cpu: torch.Tensor | None,
        slot_prefix: str,
    ) -> dict[str, object]:
        """Per-layer attention metadata for a drafter graph.

        With a mask, is_causal=False selects the explicit-mask op family;
        without one, is_causal=True selects the causal family, which the
        context-KV insert uses purely for its cache-update side effect.
        """
        if is_prefill:
            # The prefill op family expects a 1D block table.
            block_tables = block_tables_cpu[0]
            seq_idx = seq_idx_cpu.view(1, 1).to(torch.int32)
        else:
            seq_idx = torch.zeros((num_reqs_padded, 1), dtype=torch.int32)
            seq_idx[:num_reqs, 0] = seq_idx_cpu.to(torch.int32)
            # Padded rows keep block 0 (the reserved null block), so their
            # KV writes land in scratch space.
            block_tables = torch.zeros(
                (num_reqs_padded, block_tables_cpu.shape[1]),
                dtype=block_tables_cpu.dtype,
            )
            block_tables[:num_reqs] = block_tables_cpu[:num_reqs]

        per_layer_attn_metadata: dict[str, object] = {}
        for attn_group in self.draft_attn_groups:
            attn_metadata = RBLNFlashAttentionMetadata(
                seq_lens=self._stage(seq_idx, f"{slot_prefix}_seq_idx"),
                block_tables=self._stage(block_tables, f"{slot_prefix}_block_tables"),
                is_prefill=is_prefill,
                attn_masks=(
                    self._stage(mask_cpu, f"{slot_prefix}_mask")
                    if mask_cpu is not None
                    else None
                ),
                is_causal=mask_cpu is None,
            )
            attach_kv_cache_bindings(
                attn_metadata,
                self.runner.kv_caches,
                self.runner.kv_cache_bases,
                self.runner.kv_cache_view_infos,
            )
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata
        return per_layer_attn_metadata

    def _insert_context_kv(
        self,
        context_features: torch.Tensor,  # [num_ctx, feature_size]
        context_positions: torch.Tensor,  # [num_ctx]
        ctx_start_cpu: torch.Tensor,  # [num_reqs] int64
        block_tables_cpu: torch.Tensor,
        num_reqs: int,
        num_reqs_padded: int,
        ctx_len: int,
        is_prefill: bool,
        num_tokens_across_dp: torch.Tensor | None,
        num_padded_tokens: int | None,
    ) -> None:
        assert self.context_features is not None
        num_ctx = context_features.shape[0]
        if is_prefill:
            num_input_tokens = self.max_num_tokens
            batch_size = 1
        else:
            num_input_tokens = num_reqs_padded * ctx_len
            batch_size = num_reqs_padded

        # Rows beyond num_ctx keep stale buffer contents. Their K/V writes
        # are harmless: prefill padding is overwritten before it can be
        # attended, and padded decode requests write to the null block.
        self.context_features[:num_ctx] = context_features
        self.context_positions[:num_ctx] = context_positions.to(
            self.context_positions.dtype
        )

        per_layer_attn_metadata = self._build_draft_attn_metadata(
            seq_idx_cpu=ctx_start_cpu,
            block_tables_cpu=block_tables_cpu,
            num_reqs=num_reqs,
            num_reqs_padded=num_reqs_padded,
            is_prefill=is_prefill,
            mask_cpu=None,
            slot_prefix="dflash_ctx",
        )

        with set_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            num_padded_tokens=num_padded_tokens,
            **build_kv_cache_forward_context_kwargs(self.runner.kv_cache_bases),
        ):
            self.context_kv_executable(
                self.context_features[:num_input_tokens].view(
                    batch_size, -1, self.context_feature_size
                ),
                self.context_positions[:num_input_tokens].view(batch_size, -1),
            )

    def propose(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        num_rejected_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.runner is not None
        is_prefill = self.runner.is_prefill
        num_reqs = self.runner.input_batch.num_reqs
        cad = common_attn_metadata

        seq_lens_cpu = cad.seq_lens[:num_reqs].cpu().to(torch.int64)
        num_rejected_cpu = (
            num_rejected_tokens[:num_reqs].cpu().to(torch.int64)
            if num_rejected_tokens is not None
            else torch.zeros(num_reqs, dtype=torch.int64)
        )
        block_tables_cpu = cad.block_table_tensor.cpu()

        # The context is the window the target just processed. RBLN forces
        # uniform per-request query lengths, so it splits evenly.
        num_ctx = target_token_ids.shape[0]
        assert num_ctx % num_reqs == 0, (
            f"non-uniform target query lengths are not supported: "
            f"{num_ctx=} {num_reqs=}"
        )
        ctx_len = num_ctx // num_reqs
        ctx_start_cpu = seq_lens_cpu - ctx_len

        # Bonus-token position: right after the last accepted token.
        anchor_cpu = seq_lens_cpu - num_rejected_cpu

        context_features = target_hidden_states.reshape(
            -1, target_hidden_states.shape[-1]
        )[:num_ctx]
        assert context_features.shape[-1] == self.context_feature_size, (
            f"context feature width mismatch: got "
            f"{context_features.shape[-1]}, expected {self.context_feature_size}"
        )

        num_reqs_padded, num_padded_tokens, num_tokens_across_dp = (
            self._determine_draft_batch_padding(num_reqs, num_ctx, is_prefill)
        )

        self._insert_context_kv(
            context_features=context_features,
            context_positions=target_positions.reshape(-1)[:num_ctx],
            ctx_start_cpu=ctx_start_cpu,
            block_tables_cpu=block_tables_cpu,
            num_reqs=num_reqs,
            num_reqs_padded=num_reqs_padded,
            ctx_len=ctx_len,
            is_prefill=is_prefill,
            num_tokens_across_dp=num_tokens_across_dp,
            num_padded_tokens=num_padded_tokens,
        )

        return self._run_block_forward(
            next_token_ids=next_token_ids,
            anchor_cpu=anchor_cpu,
            block_tables_cpu=block_tables_cpu,
            num_reqs=num_reqs,
        )

    def _run_block_forward(
        self,
        next_token_ids: torch.Tensor,
        anchor_cpu: torch.Tensor,
        block_tables_cpu: torch.Tensor,
        num_reqs: int,
    ) -> torch.Tensor:
        assert self.block_input_ids is not None
        block_len = self.block_len
        num_reqs_padded, num_padded_tokens, num_tokens_across_dp = (
            self._determine_draft_batch_padding(num_reqs, num_reqs * block_len, False)
        )
        num_block_tokens = num_reqs_padded * block_len

        positions_cpu, mask_cpu = build_block_inputs_cpu(
            anchor_positions=anchor_cpu,
            num_reqs_padded=num_reqs_padded,
            block_len=block_len,
            max_model_len=self.max_model_len,
            mask_dtype=self.mask_dtype,
        )

        input_ids = self.block_input_ids[:num_block_tokens]
        input_ids.fill_(self.parallel_drafting_token_id)
        input_ids[0 : num_reqs * block_len : block_len] = next_token_ids[
            :num_reqs
        ].int()
        self.block_positions[:num_block_tokens].copy_(positions_cpu.view(-1))

        per_layer_attn_metadata = self._build_draft_attn_metadata(
            seq_idx_cpu=anchor_cpu,
            block_tables_cpu=block_tables_cpu,
            num_reqs=num_reqs,
            num_reqs_padded=num_reqs_padded,
            is_prefill=False,
            mask_cpu=mask_cpu,
            slot_prefix="dflash_blk",
        )

        with set_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
            num_tokens=num_block_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            num_padded_tokens=num_padded_tokens,
            **build_kv_cache_forward_context_kwargs(self.runner.kv_cache_bases),
        ):
            draft_token_ids = self.model_executable(
                input_ids.view(num_reqs_padded, block_len),
                self.block_positions[:num_block_tokens].view(
                    num_reqs_padded, block_len
                ),
            )

        return draft_token_ids[:num_reqs]

    @torch.inference_mode()
    def dummy_run(
        self,
        num_reqs: int,
        num_tokens_per_req: int,
        is_prefill: bool,
        *,
        num_padded_tokens: int | None = None,
    ) -> None:
        num_ctx = num_reqs * num_tokens_per_req
        assert num_ctx <= self.max_num_tokens

        num_reqs_padded, dp_padded, num_tokens_across_dp = (
            self._determine_draft_batch_padding(num_reqs, num_ctx, is_prefill)
        )
        num_padded_tokens = num_padded_tokens or dp_padded

        block_tables_cpu = self.runner.input_batch.block_table[0].get_cpu_tensor()[
            :num_reqs
        ]
        zeros = torch.zeros(num_reqs, dtype=torch.int64)

        self._insert_context_kv(
            context_features=self.context_features[:num_ctx],
            context_positions=self.context_positions[:num_ctx],
            ctx_start_cpu=zeros,
            block_tables_cpu=block_tables_cpu,
            num_reqs=num_reqs,
            num_reqs_padded=num_reqs_padded,
            ctx_len=num_tokens_per_req,
            is_prefill=is_prefill,
            num_tokens_across_dp=num_tokens_across_dp,
            num_padded_tokens=num_padded_tokens,
        )

        next_token_ids = torch.zeros(num_reqs, dtype=torch.int32, device=self.device)
        self._run_block_forward(
            next_token_ids=next_token_ids,
            anchor_cpu=zeros,
            block_tables_cpu=block_tables_cpu,
            num_reqs=num_reqs,
        )
