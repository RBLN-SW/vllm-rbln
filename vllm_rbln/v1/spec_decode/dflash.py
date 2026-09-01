# Copyright 2026 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DFlash drafter on RBLN.

One forward drafts every speculative token: the query is the sampled token
followed by `num_speculative_tokens` mask tokens, each predicting one draft
token. The drafter never runs over the context -- every draft layer's context
K/V is projected from the target's hidden states instead.

Upstream's `DFlashQwen3Model.precompute_and_store_context_kv` cannot do that
here (`ops.rms_norm` and `ops.rotary_embedding` are CUDA extensions, and no
RBLN attention impl has `do_kv_cache_update`), and the drafter's mixed
attention -- four sliding layers then one full -- is never declared to vLLM,
so the window lives in the mask.
"""

from copy import copy
from itertools import chain

import torch
import torch.nn.functional as F
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.spec_decode.dflash import DFlashProposer

import vllm_rbln.envs as envs
import vllm_rbln.utils as rbln_utils
from vllm_rbln.compilation import build_process_group_dict, compile
from vllm_rbln.forward_context import set_forward_context
from vllm_rbln.platform import USE_DEVICE_TENSOR
from vllm_rbln.v1.attention.kv_cache_bindings import (
    attach_kv_cache_bindings,
    build_kv_cache_forward_context_kwargs,
)
from vllm_rbln.v1.spec_decode.eagle import RBLNEagleProposer


class RBLNDFlashProposer(DFlashProposer):
    vllm_config: VllmConfig

    prepare_next_token_ids_padded = RBLNEagleProposer.prepare_next_token_ids_padded
    prepare_inputs_padded = RBLNEagleProposer.prepare_inputs_padded
    _determine_batch_execution_and_padding = (
        RBLNEagleProposer._determine_batch_execution_and_padding
    )

    @staticmethod
    def _require_single_sequence(scheduler_config) -> None:
        """DFlash loses acceptance on a wider decode batch. Cause unidentified."""
        if scheduler_config.max_num_seqs > 1:
            raise NotImplementedError(
                "DFlash speculative decoding requires --max-num-seqs 1; got "
                f"{scheduler_config.max_num_seqs}. A wider decode batch "
                "silently collapses acceptance to below the no-speculation "
                "baseline."
            )

    def __init__(self, vllm_config, device: torch.device, runner=None):
        # Checked before the base class does any work.
        self._require_single_sequence(vllm_config.scheduler_config)
        if (
            vllm_config.speculative_config.enforce_eager
            or not envs.VLLM_RBLN_COMPILE_MODEL
        ):
            # The attention ops are pattern stubs the compiler replaces, so an
            # eager context write would silently write nothing.
            raise NotImplementedError(
                "The DFlash drafter cannot run eager on RBLN: its context K/V "
                "write goes through an attention op that only exists as a "
                "compiled kernel."
            )
        if not USE_DEVICE_TENSOR:
            # The cache is allocated on the meta device then, and the context
            # write is a host copy -- which meta accepts and discards.
            raise NotImplementedError(
                "The DFlash drafter requires VLLM_RBLN_USE_DEVICE_TENSOR=1 on "
                "RBLN: without it the KV cache has no host-visible storage and "
                "the drafter's context K/V write is silently dropped."
            )
        super().__init__(vllm_config=vllm_config, device=device, runner=runner)
        if self.dflash_causal:
            # `_create_draft_vllm_config` turns this into
            # `use_non_causal=False`, which resolves the drafter onto the causal
            # kernel family -- and that family takes no mask, which is the only
            # place the draft block and its window are expressed.
            raise NotImplementedError(
                "The DFlash drafter must be non-causal on RBLN: the draft pass "
                "carries its block geometry in an explicit mask, which the "
                "causal kernels do not take. Unset `causal` in the draft "
                "model's dflash config."
            )
        self.runner = runner
        # `self.arange` is on the device. The RBLN metadata builder reads
        # `query_start_loc` and `seq_lens` on the host and stages them itself.
        self.arange_cpu = torch.arange(self.arange.shape[0], dtype=torch.int32)

        # The drafter is a mixed stack: sliding-window layers then one full
        # attention layer. The window is carried by the mask alone -- see
        # `_build_draft_attn_metadata`.
        hf_config = vllm_config.speculative_config.draft_model_config.hf_config
        layer_types = getattr(hf_config, "layer_types", None) or []
        self.sliding_window = (
            getattr(hf_config, "sliding_window", None)
            if "sliding_attention" in layer_types
            else None
        )
        self.sliding_layer_indices = frozenset(
            index
            for index, kind in enumerate(layer_types)
            if kind == "sliding_attention"
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
    ) -> torch.Tensor | list[list[int]]:
        """Draft `num_speculative_tokens` tokens in one forward.

        Overrides upstream's `propose`, which cannot run here: it opens a second
        DP all-reduce over a step the runner has already synchronized, and it
        enters `vllm.forward_context`, which carries neither the padded token
        count nor the KV cache bases RBLN attention resolves its caches from.
        """
        assert self.runner is not None
        assert not self.supports_mm_inputs
        num_reqs = self.runner.input_batch.num_reqs
        num_query_per_req = 1 + self.num_speculative_tokens
        num_query_total = num_reqs * num_query_per_req

        # The runner already reduced the aux states with the drafter's own
        # projection, fused into the target graph.
        assert target_hidden_states.shape[-1] == self.hidden_size

        num_context, ctx_starts, valid_ctx_lens = self._fill_first_pass_inputs(
            next_token_ids,
            target_positions,
            common_attn_metadata,
            num_rejected_tokens,
            num_query_per_req,
        )

        self._write_context_kv(
            target_hidden_states,
            common_attn_metadata,
            num_reqs,
            num_context,
            valid_ctx_lens,
        )

        draft_ids = self._run_query_pass(
            common_attn_metadata,
            num_reqs,
            num_query_per_req,
            num_query_total,
            ctx_starts,
            valid_ctx_lens,
        )
        # One draft id per mask position, so the flat result holds
        # `num_reqs * num_speculative_tokens` ids -- not one per request.
        draft_ids = draft_ids[: num_reqs * self.num_speculative_tokens].view(
            num_reqs, self.num_speculative_tokens
        )
        # `d2t` holds offsets, not absolute ids, so the target id is
        # `draft_id + d2t[draft_id]`. Outside the graph on purpose: the index is
        # the argmax result, which the compiled subgraph above cannot const-fold.
        if (d2t := self.model.draft_id_to_target_id) is not None:
            draft_ids = draft_ids + d2t[draft_ids]
        if bool(self._dropped_rows.any()):
            # An empty list, never zeros: the scheduler reads zeros as real
            # token ids. The target-only step still advances the request to the
            # aligned page, where speculation resumes.
            rows = draft_ids.tolist()
            return [
                [] if bool(self._dropped_rows[i]) else rows[i] for i in range(num_reqs)
            ]
        return draft_ids

    def initialize_attn_backend(self, kv_cache_config, kernel_block_sizes=None) -> None:
        """Build the draft metadata builders under the draft config.

        Impl and builder both read causality from the config they are built
        under, so they have to be built under the same one. Upstream uses the
        target's `vllm_config` here, which leaves the drafter's impl non-causal
        and its builder causal -- emitting no mask, for a kernel that needs one.
        """
        draft_vllm_config = self._create_draft_vllm_config()
        with set_current_vllm_config(draft_vllm_config):
            original, self.vllm_config = self.vllm_config, draft_vllm_config
            try:
                super().initialize_attn_backend(kv_cache_config, kernel_block_sizes)
            finally:
                self.vllm_config = original

    def load_model(self, target_model) -> None:
        super().load_model(target_model)
        self._check_rope_style(target_model)

        # Projection inputs are padded to one of two shapes so the region
        # compiles twice: one draft block per request in decode, one prefill
        # chunk otherwise.
        self._proj_buckets = (1 + self.num_speculative_tokens, self.max_num_tokens)
        self._proj_states = torch.zeros(
            self.max_num_tokens,
            self.hidden_size,
            dtype=self.model.model.layers[0].self_attn.qkv_proj.weight.dtype,
            device=self.device,
        )
        self._proj_positions = torch.zeros(
            1, self.max_num_tokens, dtype=torch.int64, device=self.device
        )

        # The projection has to be compiled: eager dispatch pulls its inputs
        # to the host and `rms_norm` faults moving the norm weight across.
        # The cache write has to stay out -- see `_write_context_kv`.
        def project_context_kv(states: torch.Tensor, positions: torch.Tensor):
            model = self.model.model
            num_tokens = states.shape[0]
            normed = model.hidden_norm(states)
            keys: list[torch.Tensor] = []
            values: list[torch.Tensor] = []
            for layer in model.layers:
                self_attn = layer.self_attn
                head_dim = self_attn.head_dim
                bias = self_attn.qkv_proj.bias
                kv = F.linear(
                    normed,
                    self_attn.qkv_proj.weight[self_attn.q_size :],
                    None if bias is None else bias[self_attn.q_size :],
                )
                key, value = kv.split([self_attn.kv_size, self_attn.kv_size], dim=-1)
                k_shape = key.shape
                key = self_attn.k_norm(
                    key.view(*k_shape[:-1], k_shape[-1] // head_dim, head_dim)
                ).view(k_shape)
                flat = key.view(1, num_tokens, -1)
                key, _ = self_attn.rotary_emb(positions, flat, flat)
                # Head-major: the cache write below is a copy per layer and
                # head, and only that slice is contiguous on both sides.
                keys.append(key.view(num_tokens, -1, head_dim).transpose(0, 1))
                values.append(value.view(num_tokens, -1, head_dim).transpose(0, 1))
            return (
                torch.stack(keys).contiguous(),
                torch.stack(values).contiguous(),
            )

        def model_wrapper(
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            token_indices_to_sample: torch.Tensor,
        ):
            hidden_states = self.model(
                input_ids=input_ids, positions=positions, inputs_embeds=None
            )
            if self.model_returns_tuple():
                hidden_states = hidden_states[0]
            hidden_states = hidden_states.view(-1, self.hidden_size)[
                token_indices_to_sample
            ]
            if self.model.draft_id_to_target_id is not None:
                # Upstream widens the draft logits to the target vocabulary
                # by scattering at `arange(draft_vocab) + d2t`. That index is
                # input-independent, so it const-folds to an anonymous constant
                # weight-free apply cannot fill, and the write goes out of
                # bounds. Stay in draft-vocab space and map after the argmax.
                logits = self.model.logits_processor(self.model.lm_head, hidden_states)
            else:
                logits = self.model.compute_logits(hidden_states)
            # NOTE(RBLN): the greedy pick belongs in the graph.
            return torch.ops.rbln.argmax(logits)

        compile_kwargs = dict(
            dynamic=False,
            fullgraph=True,
            compile_context=self.runner.compile_context,
            num_devices=envs.VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK,
            model_trace_method="export" if USE_DEVICE_TENSOR else "",
            process_group_dict=build_process_group_dict(),
            guard_filter_fn=torch.compiler.keep_tensor_guards_unsafe,
            runtime_holder=self.runner.runtime_holder,
            mode="strict" if envs.VLLM_RBLN_COMPILE_STRICT_MODE else "",
            use_static_output=True,
        )
        self.model_executable = compile(model_wrapper, **compile_kwargs)
        self._project_context_kv = compile(project_context_kv, **compile_kwargs)
        # Whether a draft forward joins the DP all-gather; an idle rank may only
        # skip drafting when it does not.
        self.draft_has_moe = any(
            isinstance(module, MoERunner) for module in self.model.modules()
        )

    def _check_rope_style(self, target_model) -> None:
        """Refuse a drafter whose RoPE style differs from the target's.

        The context K is projected from the target's hidden states, so the
        rotation must be the one the drafter was distilled against. A mismatch
        is inert on the pinned release and surfaces only as collapsed
        acceptance. `rotary_dim` may differ -- the drafter rotates the full
        head, the target half of it -- so only the style is compared.
        """
        target = next(
            (
                module
                for module in target_model.modules()
                if isinstance(module, RotaryEmbedding)
            ),
            None,
        )
        assert target is not None, "the target model has no rotary embedding"
        draft = self.model.model.layers[0].self_attn.rotary_emb
        if draft.is_neox_style != target.is_neox_style:
            raise ValueError(
                "DFlash drafter and target disagree on RoPE style: target is "
                f"{'neox' if target.is_neox_style else 'gptj'}-style, drafter "
                f"is {'neox' if draft.is_neox_style else 'gptj'}-style. The "
                "context K/V is projected from the target's hidden states, so "
                "the rotation has to match or acceptance collapses silently."
            )

    _build_dummy_attn_metadata = RBLNEagleProposer._build_dummy_attn_metadata

    @torch.inference_mode()
    def dummy_run(
        self,
        num_reqs: int,
        num_tokens_per_req: int,
        is_prefill: bool = False,
        *,
        num_padded_tokens: int | None = None,
    ) -> None:
        """Compile every drafter graph here, not on the first real step.

        `is_prefill` is unused: the projection's shape comes from its bucket and
        the query pass always runs one draft block per request.
        """
        status = self.runner.dp_status
        if (
            status is not None
            and status.is_idle[self.dp_rank]
            and not self.draft_has_moe
        ):
            # Same condition as the chained drafter: what this pass produces is
            # discarded, and without fused MoE no busy rank waits on it.
            return

        # 1) The projection. Both buckets, straight from the staging buffers --
        # a pure region, so no metadata and no cache write.
        for bucket in self._proj_buckets:
            self._project_context_kv(
                self._proj_states[:bucket], self._proj_positions[:, :bucket]
            )

        # 2) The query pass, at the only shape it ever runs: one draft block
        # (`1 + num_speculative_tokens` queries) per request.
        num_query_per_req = 1 + self.num_speculative_tokens
        num_query_total = num_reqs * num_query_per_req
        if num_query_total > self.max_num_tokens:
            return
        cad = self._build_dummy_attn_metadata(num_reqs, num_query_per_req)
        # Start every request at block 0 so no row crosses a page and the
        # redirect path stays out of the warmup.
        ctx_starts = torch.zeros(num_reqs, dtype=torch.int32)
        valid_ctx_lens = torch.zeros(num_reqs, dtype=torch.int32)
        self._run_query_pass(
            cad,
            num_reqs,
            num_query_per_req,
            num_query_total,
            ctx_starts,
            valid_ctx_lens,
        )

    def _fill_first_pass_inputs(
        self,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        cad: CommonAttentionMetadata,
        num_rejected_tokens: torch.Tensor | None,
        num_query_per_req: int,
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        """What upstream's `copy_and_expand_dflash_inputs_kernel` does, in torch.

        Triton has no driver here, and the work is a few gathers. The runner
        keeps `positions` and `query_start_loc` on the host, so the arithmetic
        stays there and only the finished buffers cross.

        Returns the flat context length, each request's first context position,
        and its accepted context length.
        """
        batch_size = cad.batch_size()
        qsl = cad.query_start_loc[: batch_size + 1].cpu()
        ctx_start_tok, ctx_end_tok = qsl[:-1], qsl[1:]
        target_positions = target_positions.cpu()
        offsets = torch.arange(num_query_per_req)

        # Rejected tokens sit at the tail of the target's query, so the accepted
        # context is the run that ends before them.
        valid_ctx_end = ctx_end_tok
        if num_rejected_tokens is not None:
            valid_ctx_end = ctx_end_tok - num_rejected_tokens.cpu()
        valid_ctx_lens = valid_ctx_end - ctx_start_tok
        num_context = int(valid_ctx_lens.sum())

        # Query positions continue from the last accepted context token.
        last_pos = target_positions[valid_ctx_end - 1]
        query_pos = last_pos[:, None] + 1 + offsets[None, :]
        num_query_total = batch_size * num_query_per_req
        self.positions[:num_query_total] = query_pos.reshape(-1).to(
            self.positions.device
        )

        # Query ids are the sampled token followed by mask tokens.
        query_ids = torch.full(
            (batch_size, num_query_per_req),
            self.parallel_drafting_token_id,
            dtype=self.input_ids.dtype,
        )
        query_ids[:, 0] = next_token_ids[:batch_size].cpu()
        self.input_ids[:num_query_total] = query_ids.reshape(-1).to(
            self.input_ids.device
        )

        # Compact the accepted context: the rejected tail leaves it
        # non-contiguous in the target's token order.
        ctx_gather = torch.cat(
            [
                torch.arange(int(start), int(start) + int(count))
                for start, count in zip(ctx_start_tok, valid_ctx_lens)
            ]
        )
        self._ctx_gather = ctx_gather.to(self.device)
        # Keep the host copy: reading the device buffer back in the write step
        # is a device-to-host copy on the dispatch path, and that is what faults
        # (`rbln_memcpy_v2h failed`) once the context grows.
        self._ctx_positions_cpu = target_positions[ctx_gather]
        self._context_positions_buffer[:num_context] = self._ctx_positions_cpu.to(
            self._context_positions_buffer.device
        )

        ctx_starts = target_positions[ctx_start_tok]
        return num_context, ctx_starts, valid_ctx_lens

    def _build_draft_attn_metadata(
        self,
        cad: CommonAttentionMetadata,
        positions: torch.Tensor,
        num_reqs: int,
        num_reqs_padded: int,
    ) -> dict[str, object]:
        per_layer: dict[str, object] = {}
        for attn_group in self.draft_attn_groups:
            builder = attn_group.get_metadata_builder()
            attn_metadata = builder.build(
                common_attn_metadata=cad,
                positions=positions,
                is_prefill=False,
                batch_pad=num_reqs_padded,
                # Every mask here is replaced below, so the builder's own
                # `max_model_len`-wide one would be host work and a staged
                # buffer for nothing.
                skip_attn_masks=True,
            )
            attach_kv_cache_bindings(
                attn_metadata,
                self.runner.kv_caches,
                self.runner.kv_cache_bases,
                self.runner.kv_cache_view_infos,
            )
            # `initialize_attn_backend` builds under the draft config, so a
            # causal builder here means that scoping came undone.
            assert not builder.is_causal, (
                "the draft metadata builder was built causal; it takes no mask"
            )
            # What the builder would have produced, without producing it.
            max_seq_len = builder.model_config.max_model_len
            mask_dtype = torch.float16 if builder.enforce_eager else torch.float32
            mask_device = builder.device

            # One attention group covers the whole drafter -- groups are keyed by
            # backend class -- so the two attention types are separated by the
            # mask here, not by the grouping.
            sliding = self._sliding_layer_names(attn_group)
            full_metadata = copy(attn_metadata)
            full_metadata.attn_masks = (
                self._draft_block_mask(
                    cad.seq_lens, num_reqs, num_reqs_padded, max_seq_len, None
                )
                .to(mask_dtype)
                .contiguous()
                .to(mask_device)
            )
            for layer_name in attn_group.layer_names:
                per_layer[layer_name] = full_metadata
            if not sliding:
                continue
            # The window travels in `attn_masks`, not `swa_attn_masks`:
            # `DFlashQwen3Attention` never declares `layer_types`, so every
            # draft layer gets a `FullAttentionSpec` and a kernel that ignores
            # the SWA fields. Per-layer `seq_lens` would need a second dynamic
            # index on an already-indexed partition, which the compiler
            # refuses with a segfault rather than a diagnostic.
            window_metadata = copy(attn_metadata)
            window_metadata.attn_masks = (
                self._draft_block_mask(
                    cad.seq_lens,
                    num_reqs,
                    num_reqs_padded,
                    max_seq_len,
                    self.sliding_window,
                )
                .to(mask_dtype)
                .contiguous()
                .to(mask_device)
            )
            for layer_name in sliding:
                per_layer[layer_name] = window_metadata
        return per_layer

    def _sliding_layer_names(self, attn_group) -> list[str]:
        """The drafter's sliding-window layers inside this attention group."""
        if self.sliding_window is None:
            return []
        # Draft layer names continue the target's numbering, so the drafter's
        # own index is the offset from its first layer.
        first = min(extract_layer_index(name) for name in self._draft_attn_layer_names)
        return [
            layer_name
            for layer_name in attn_group.layer_names
            if extract_layer_index(layer_name) - first in self.sliding_layer_indices
        ]

    def _draft_block_mask(
        self,
        seq_lens: torch.Tensor,
        num_reqs: int,
        num_reqs_padded: int,
        max_seq_len: int,
        sliding_window: int | None,
    ) -> torch.Tensor:
        """A mask for a whole draft block, optionally bounded by the window.

        Full layers see the whole context plus the block's own keys; sliding
        layers see the window ending at their own position. Built on the host:
        each eager device op here would compile its own graph.
        """
        num_query_per_req = 1 + self.num_speculative_tokens
        key_pos = torch.arange(max_seq_len)
        query_pos = seq_lens[:num_reqs].view(-1, 1) + torch.arange(
            num_query_per_req
        ).view(1, -1)
        if sliding_window is None:
            # Every query in the block sees the same keys, so one row is built
            # and expanded rather than repeated per query.
            key_end = query_pos.max(dim=1, keepdim=True).values
            valid = key_pos.view(1, -1) <= key_end
            mask = valid.view(num_reqs, 1, 1, 1, max_seq_len).expand(
                num_reqs, 1, 1, num_query_per_req, max_seq_len
            )
        else:
            # Causal against the context, but not inside the draft block:
            # every mask slot sees every other one. A block that only looked
            # backwards is what `flash_causal_attention_naive_*` already
            # gives, and this model would not need the mask-taking family at
            # all.
            distance = query_pos.unsqueeze(-1) - key_pos.view(1, 1, -1)
            block_start = seq_lens[:num_reqs].view(-1, 1, 1)
            in_block = (key_pos.view(1, 1, -1) >= block_start) & (
                key_pos.view(1, 1, -1) < block_start + num_query_per_req
            )
            valid = ((distance >= 0) | in_block) & (distance < sliding_window)
            mask = valid.view(num_reqs, 1, 1, num_query_per_req, max_seq_len)
        if num_reqs_padded > num_reqs:
            mask = rbln_utils.pad(mask, 0, num_reqs_padded)
        return mask

    def _write_context_kv(
        self,
        target_hidden_states: torch.Tensor,
        cad: CommonAttentionMetadata,
        num_reqs: int,
        num_context: int,
        valid_ctx_lens: torch.Tensor,
    ) -> None:
        """Project the target's hidden states into every draft layer's KV cache.

        The projection is a compiled region; the cache write is not. A compiled
        stateful store would give the cache input its own physical-view
        configuration while the drafter's forward graph assigns another, and
        alternating the two rematerialises every cache view per proposal.
        """
        model = self.model.model
        block_size = self.block_size
        positions = self._ctx_positions_cpu
        block_table = cad.block_table_tensor.cpu()

        # Per-token destination, then maximal runs: the copy target has to be a
        # contiguous span inside one physical block.
        req_of_token = torch.repeat_interleave(
            torch.arange(num_reqs), valid_ctx_lens.to(torch.int64)
        )
        blocks = block_table[req_of_token, positions // block_size].tolist()
        offsets = (positions % block_size).tolist()
        reqs = req_of_token.tolist()
        runs: list[tuple[int, int, int, int]] = []
        start = 0
        for i in range(1, num_context + 1):
            if (
                i < num_context
                and reqs[i] == reqs[i - 1]
                and blocks[i] == blocks[i - 1]
                and offsets[i] == offsets[i - 1] + 1
            ):
                continue
            runs.append((start, i - start, blocks[start], offsets[start]))
            start = i

        # One call, not a loop: `_project_context_kv` is compiled with static
        # output buffers, so two calls would hand back views of the same storage
        # and the second would clobber the first. The scheduler caps a step's
        # tokens at `max_num_batched_tokens`, which is what `max_num_tokens`
        # holds, so one call always covers the context.
        assert num_context <= self.max_num_tokens, (
            f"context of {num_context} tokens exceeds the projection's "
            f"{self.max_num_tokens}-token buffers"
        )
        states = target_hidden_states[self._ctx_gather]
        bucket = next(size for size in self._proj_buckets if size >= num_context)
        self._proj_states[:num_context].copy_(states)
        self._proj_positions[0, :num_context].copy_(
            self._context_positions_buffer[:num_context]
        )
        keys, values = self._project_context_kv(
            self._proj_states[:bucket], self._proj_positions[:, :bucket]
        )
        keys = keys[:, :, :num_context]
        values = values[:, :, :num_context]

        # One copy per layer and head: all heads at once is contiguous on
        # neither side, and the runtime stages that strided pair through host
        # memory, which faults as `rbln_memcpy_v2h failed`. `unbind` builds the
        # same views in one call per layer instead of one index at a time.
        destinations: list[torch.Tensor] = []
        sources: list[torch.Tensor] = []
        for layer_index, layer in enumerate(model.layers):
            cache = layer.self_attn.attn.kv_cache
            k_layer = keys[layer_index]
            v_layer = values[layer_index]
            for token_start, count, block, offset in runs:
                token_slice = slice(token_start, token_start + count)
                cache_slice = slice(offset, offset + count)
                dst_k = cache[0, block, :, 0, cache_slice, :].unbind(0)
                dst_v = cache[1, block, :, 0, cache_slice, :].unbind(0)
                src_k = k_layer[:, token_slice, :].unbind(0)
                src_v = v_layer[:, token_slice, :].unbind(0)
                # Interleave key/value per head to keep the original order.
                destinations.extend(chain.from_iterable(zip(dst_k, dst_v)))
                sources.extend(chain.from_iterable(zip(src_k, src_v)))
        torch._foreach_copy_(destinations, sources)

    def _run_query_pass(
        self,
        cad: CommonAttentionMetadata,
        num_reqs: int,
        num_query_per_req: int,
        num_query_total: int,
        ctx_starts: torch.Tensor,
        valid_ctx_lens: torch.Tensor,
    ) -> torch.Tensor:
        # The context length BEFORE the query block: the kernel scatters the
        # block's own K/V at that offset. It reads the offset out of
        # `positions`, not out of this field, so this is only what the draft
        # mask is built from -- and the mask has to name the same slots the
        # block really occupies.
        seq_lens = ctx_starts + valid_ctx_lens
        query_cad = CommonAttentionMetadata(
            query_start_loc=self.arange_cpu[: num_reqs + 1] * num_query_per_req,
            seq_lens=seq_lens,
            query_start_loc_cpu=self.arange_cpu[: num_reqs + 1] * num_query_per_req,
            _seq_lens_cpu=None,
            _num_computed_tokens_cpu=None,
            seq_lens_cpu_upper_bound=cad.seq_lens_cpu_upper_bound,
            num_reqs=num_reqs,
            num_actual_tokens=num_query_total,
            max_query_len=num_query_per_req,
            max_seq_len=cad.max_seq_len + num_query_per_req,
            block_table_tensor=cad.block_table_tensor,
            slot_mapping=torch.tensor(0),  # dummy
            causal=self.dflash_causal,
        )
        # One dynamic offset per partition scatters the whole block, so a block
        # cannot straddle two pages -- the last `num_query_per_req - 1` offsets
        # would write into another request's. Redirect those rows to their next
        # page start and drop their drafts; model-input positions stay true.
        crossing = (seq_lens % self.block_size) + num_query_per_req > self.block_size
        if bool(crossing.any()):
            next_page = (seq_lens // self.block_size + 1) * self.block_size
            pages = (next_page // self.block_size).to(torch.int64)
            if int(pages.max()) >= cad.block_table_tensor.shape[-1]:
                # Past the context ceiling, so the whole step gives up its
                # drafts: returning here is what keeps the query graph -- and
                # its scatter -- from running at all.
                self._dropped_rows = torch.ones_like(crossing)
                return self.positions.new_zeros(num_query_total)
            rows = torch.arange(pages.shape[0])
            assert not bool(
                (cad.block_table_tensor.cpu()[rows, pages][crossing] == 0).any()
            ), "the scheduler's lookahead reservation left a crossing row on block 0"
            seq_lens = torch.where(crossing, next_page, seq_lens)
        self._dropped_rows = crossing

        # The builder reads one thing out of the positions it is given --
        # `positions[query_start_loc_cpu[:num_reqs]]` -- and that becomes the
        # kernel's block write offset. Real query positions would leave no way
        # to redirect a crossing row, so the block start travels in its own
        # host-side probe and the model keeps the true positions for RoPE.
        query_cad.seq_lens = seq_lens
        block_starts = torch.zeros(num_query_total, dtype=torch.int64)
        block_starts[self.arange_cpu[:num_reqs] * num_query_per_req] = seq_lens.to(
            torch.int64
        )

        per_layer = self._build_draft_attn_metadata(
            query_cad,
            block_starts,
            num_reqs,
            num_reqs,
        )
        # `RBLNDPMetadata.make` requires this to be None off the DP path.
        _, num_tokens_across_dp = self._determine_batch_execution_and_padding(
            num_reqs, num_query_total, False, first_pass=False
        )
        with set_forward_context(
            per_layer,
            self.vllm_config,
            num_tokens=num_query_total,
            num_tokens_across_dp=num_tokens_across_dp,
            num_padded_tokens=(
                num_query_total if num_tokens_across_dp is not None else None
            ),
            **build_kv_cache_forward_context_kwargs(self.runner.kv_cache_bases),
        ):
            return self.model_executable(
                input_ids=self.input_ids[:num_query_total].view(num_reqs, -1),
                positions=self.positions[:num_query_total].view(num_reqs, -1),
                token_indices_to_sample=self._sample_indices(
                    num_reqs, num_query_per_req
                ),
            )

    def _sample_indices(self, num_reqs: int, num_query_per_req: int) -> torch.Tensor:
        """The mask positions -- the ones the target verifies."""
        rows = torch.arange(num_reqs)[:, None]
        cols = torch.arange(self.num_speculative_tokens)[None, :]
        return (rows * num_query_per_req + 1 + cols).reshape(-1).to(self.device)
