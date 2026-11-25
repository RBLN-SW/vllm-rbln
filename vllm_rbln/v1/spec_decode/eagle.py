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
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.models.deepseek_eagle3 import Eagle3DeepseekV2ForCausalLM
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

import vllm_rbln.envs as envs
from vllm_rbln.compilation import (
    build_process_group_dict,
    compile,
)
from vllm_rbln.config import RBLNConfig
from vllm_rbln.forward_context import set_forward_context
from vllm_rbln.logger import init_logger
from vllm_rbln.model_executor.models.interfaces import (
    supports_rbln_multimodal_embeddings,
)
from vllm_rbln.platform import use_device_tensor
from vllm_rbln.v1.spec_decode import pass_state
from vllm_rbln.v1.spec_decode.utils import (
    eagle_prepare_inputs_padded,
    eagle_prepare_next_token_padded,
)
from vllm_rbln.v1.worker.dp_utils import (
    BatchDescriptor,
    determine_draft_batch_execution_and_padding,
)
from vllm_rbln.v1.worker.input_stager import InputLayout, InputStager

if TYPE_CHECKING:
    from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

logger = init_logger(__name__)


@dataclass
class _TailInputs:
    attn_metadata: list[dict[str, object]]
    positions: torch.Tensor
    num_reqs_padded: int
    num_padded_tokens: int | None
    num_tokens_across_dp: torch.Tensor | None


class RBLNEagleProposer(EagleProposer):
    # A producer's warm-up stops after the first pass, which is prefill-shaped.
    warms_up_decode_graphs_on_a_producer = False

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner: "RBLNModelRunner",
    ):
        # The parent's ctor gets "cpu", so its bookkeeping -- input_ids,
        # positions, hidden_states, backup ids -- lives on the host: torch-rbln
        # runs int/bool eager ops through a CPU fallback. Whatever reads
        # `self.device` afterwards (attention metadata builders, input stager)
        # still gets the NPU, where the draft model itself keeps running.
        super().__init__(vllm_config, torch.device("cpu"), runner)
        self.device = device
        if self.uses_mrope:
            # Cache addresses follow token offsets independently of M-RoPE.
            self.positions = torch.zeros(self.max_positions, dtype=torch.int64)

        if self.needs_extra_input_slots:
            raise NotImplementedError(
                "vllm-rbln does not support EAGLE extra input slots required for "
                "parallel drafting or draft-model speculative decoding yet."
            )
        draft_sample_method = self.speculative_config.draft_sample_method
        if draft_sample_method != "greedy":
            raise NotImplementedError(
                f"draft_sample_method={draft_sample_method!r} is not implemented yet."
            )

        self.runner = runner
        self.input_stager = InputStager(device)
        # Populated from the draft model in `load_model`. None means the draft
        # head shares the target vocabulary, so `propose()` maps no ids.
        self.draft_id_to_target_id: torch.Tensor | None = None
        # Whether a draft forward joins the DP all-gather; see `load_model`.
        self.draft_has_moe = False
        self.tail_executable: Callable[..., torch.Tensor] | None = None
        self._tail_buffers: dict[tuple, torch.Tensor] = {}

    def propose(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        mm_embed_inputs: tuple[list[tuple[int, int, torch.Tensor]], torch.Tensor]
        | None = None,
        num_rejected_tokens: torch.Tensor | None = None,
        target_mrope_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # NOTE(RBLN): combine_hidden_states in eagle3 is fused
        # into the target model graph.
        assert target_hidden_states.shape[-1] == self.hidden_size

        num_tokens = target_token_ids.shape[0]
        if token_indices_to_sample is None:
            token_indices_to_sample = common_attn_metadata.query_start_loc[1:] - 1

        assert self.runner is not None
        is_prefill = self.runner.is_prefill

        # Build attention metadata
        num_reqs = self.runner.input_batch.num_reqs
        batch_desc, num_tokens_across_dp = self._determine_batch_execution_and_padding(
            num_reqs,
            num_tokens,
            is_prefill,
        )
        num_reqs_padded = batch_desc.num_reqs_padded
        num_padded_tokens = batch_desc.num_tokens_padded
        per_layer_attn_metadata: dict[str, object] = {}
        for attn_group in self.draft_attn_groups:
            attn_metadata = attn_group.get_metadata_builder().build(
                common_attn_metadata=common_attn_metadata,
                positions=target_positions,
                is_prefill=is_prefill,
                batch_pad=num_reqs_padded,
                back_pad=self.runner.decode_back_pad,
                is_prompt_sorted=self.runner.sort_batch_by_length,
            )
            self.runner.bind_kv_cache_to_metadata(attn_metadata)
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata

        input_ids, positions, hidden_states, token_indices_to_sample_padded = (
            self._preprocess(
                num_reqs,
                num_reqs_padded,
                num_tokens,
                target_positions,
                target_hidden_states,
                is_prefill=is_prefill,
                token_indices_to_sample=token_indices_to_sample,
                target_token_ids=target_token_ids,
                next_token_ids=next_token_ids,
                mrope_positions=target_mrope_positions,
            )
        )
        inputs_embeds = None

        # A strict producer never verifies a draft, and has no decoding peer to
        # wait for. An intermediate chunk's drafts are discarded, but under DP a
        # MoE drafter still runs every pass for possible decoding peers.
        run_tail = (
            self.num_speculative_tokens > 1
            and not self.runner.is_strict_kv_producer
            and not (
                self.runner.is_intermediate_chunked_prefill
                and not (
                    self.vllm_config.parallel_config.data_parallel_size > 1
                    and self.draft_has_moe
                )
            )
        )
        if (
            run_tail
            and self.allowed_attn_types is not None
            and not isinstance(attn_metadata, self.allowed_attn_types)
        ):
            raise ValueError(
                f"Unsupported attention metadata type for speculative "
                "decoding with num_speculative_tokens > 1: "
                f"{type(attn_metadata)}. Supported types are: "
                f"{self.allowed_attn_types}"
            )
        # Everything the tail reads from the host is staged before the first
        # pass launches, so nothing between the two launches waits on it.
        tail = (
            self._prepare_tail(
                common_attn_metadata,
                target_positions[token_indices_to_sample],
                num_reqs,
                num_rejected_tokens,
                mrope_positions=(
                    target_mrope_positions[:, token_indices_to_sample]
                    if target_mrope_positions is not None
                    else None
                ),
            )
            if run_tail
            else None
        )

        with set_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            num_padded_tokens=num_padded_tokens,
            **self.runner.kv_cache_forward_context_kwargs(),
        ):
            hidden_states, draft_ids = self.model_executable(
                input_ids=input_ids,
                positions=positions,
                hidden_states=hidden_states,
                inputs_embeds=inputs_embeds,
                token_indices_to_sample=token_indices_to_sample_padded,
                **self._stage_mm_inputs(input_ids, is_prefill, mm_embed_inputs),
            )

        if self.runner.is_strict_kv_producer:
            return draft_ids.new_zeros((num_reqs, self.num_speculative_tokens))
        if self.num_speculative_tokens == 1:
            draft_tokens_ids = self._to_target_token_ids(draft_ids[:num_reqs])
            return draft_tokens_ids.view(-1, 1)
        if tail is None:
            return draft_ids.new_zeros((num_reqs, self.num_speculative_tokens))
        return self._run_tail(tail, draft_ids, hidden_states, num_reqs)

    def _prepare_tail(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        positions: torch.Tensor,
        num_reqs: int,
        num_rejected_tokens: torch.Tensor | None,
        pinned_num_tokens_padded: int | None = None,
        mrope_positions: torch.Tensor | None = None,
    ) -> _TailInputs:
        """Per-pass attention metadata and the first tail position, staged."""
        # it mutates seq_lens in place; detach from the runner's persistent buffer.
        common_attn_metadata.seq_lens = common_attn_metadata.seq_lens.clone()
        common_attn_metadata.num_actual_tokens = num_reqs
        common_attn_metadata.max_query_len = 1
        common_attn_metadata.query_start_loc = self.arange[: num_reqs + 1]
        common_attn_metadata.query_start_loc_cpu = self.arange[: num_reqs + 1]

        # In padded drafter batch, we need to adjust the sequence lengths
        # to remove the "padding" (i.e. rejected tokens).
        if num_rejected_tokens is not None:
            common_attn_metadata.seq_lens -= num_rejected_tokens

        batch_desc, num_tokens_across_dp = self._determine_batch_execution_and_padding(
            num_reqs,
            num_reqs,
            False,
            first_pass=False,
            pinned_num_tokens_padded=pinned_num_tokens_padded,
        )
        num_reqs_padded = batch_desc.num_reqs_padded

        positions = positions.view(-1)
        first_positions = positions + 1
        per_pass: list[dict[str, object]] = []
        for pass_idx in range(self.num_speculative_tokens - 1):
            positions = positions + 1
            exceeds_max_model_len = positions[:num_reqs] >= self.max_model_len
            common_attn_metadata.seq_lens += 1
            common_attn_metadata.seq_lens.masked_fill_(exceeds_max_model_len, 1)
            per_pass.append(
                self._build_attn_metadata(
                    common_attn_metadata,
                    positions,
                    is_prefill=False,
                    batch_pad=num_reqs_padded,
                    staging_slot=pass_idx + 1,
                )
            )

        if mrope_positions is None:
            host = torch.zeros((num_reqs_padded, 1), dtype=first_positions.dtype)
            host[:num_reqs, 0] = first_positions[:num_reqs]
        else:
            first_mrope = mrope_positions[:1, :num_reqs] + 1
            host = torch.zeros((3, num_reqs_padded, 1), dtype=first_mrope.dtype)
            host[:, :num_reqs, 0] = first_mrope
        staged_positions = self._tail_buffer("positions", host.shape, host.dtype)
        staged_positions.copy_(host)
        return _TailInputs(
            attn_metadata=per_pass,
            positions=staged_positions,
            num_reqs_padded=num_reqs_padded,
            num_padded_tokens=batch_desc.num_tokens_padded,
            num_tokens_across_dp=num_tokens_across_dp,
        )

    def _run_tail(
        self,
        tail: _TailInputs,
        draft_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        num_reqs: int,
    ) -> torch.Tensor:
        """Passes 2..k in one region, fed by the first pass's device outputs."""
        rows = tail.num_reqs_padded
        if draft_ids.shape[0] != rows:
            # The first pass ran a different batch (prefill, or a DP peer's
            # bucket), so its outputs are re-padded to the tail's.
            n = min(draft_ids.shape[0], rows)
            ids = self._tail_buffer("ids", (rows,), draft_ids.dtype)
            ids.zero_()
            ids[:n].copy_(draft_ids[:n])
            hidden = self._tail_buffer(
                "hidden", (rows, self.hidden_size), hidden_states.dtype
            )
            hidden.zero_()
            hidden[:n].copy_(hidden_states.view(-1, self.hidden_size)[:n])
            draft_ids, hidden_states = ids, hidden

        assert self.tail_executable is not None
        with set_forward_context(
            tail.attn_metadata,
            self.vllm_config,
            num_tokens=num_reqs,
            num_tokens_across_dp=tail.num_tokens_across_dp,
            num_padded_tokens=tail.num_padded_tokens,
            **self.runner.kv_cache_forward_context_kwargs(),
        ):
            all_ids = self.tail_executable(
                draft_ids=draft_ids,
                hidden_states=hidden_states,
                positions=tail.positions,
            )
        # [batch_size, num_speculative_tokens], already in target space: the one
        # D2H of the whole draft.
        return all_ids[:num_reqs].cpu().long()

    def _tail_buffer(
        self, name: str, shape: tuple[int, ...], dtype: torch.dtype
    ) -> torch.Tensor:
        key = (name, tuple(shape), dtype)
        if (buf := self._tail_buffers.get(key)) is None:
            buf = torch.empty(shape, dtype=dtype, device=self.device)
            self._tail_buffers[key] = buf
        return buf

    def _build_attn_metadata(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        positions: torch.Tensor,
        *,
        is_prefill: bool,
        batch_pad: int,
        staging_slot: int = 0,
    ) -> dict[str, object]:
        per_layer: dict[str, object] = {}
        for attn_group in self.draft_attn_groups:
            attn_metadata = attn_group.get_metadata_builder().build(
                common_attn_metadata=common_attn_metadata,
                positions=positions,
                is_prefill=is_prefill,
                batch_pad=batch_pad,
                is_prompt_sorted=self.runner.sort_batch_by_length,
                staging_slot=staging_slot,
            )
            self.runner.bind_kv_cache_to_metadata(attn_metadata)
            for layer_name in attn_group.layer_names:
                per_layer[layer_name] = attn_metadata
        return per_layer

    def _to_target_token_ids(self, draft_token_ids: torch.Tensor) -> torch.Tensor:
        """Map draft-vocabulary ids to target-vocabulary ids.

        `d2t` holds offsets, not absolute ids -- upstream scatters at
        `arange(draft_vocab_size) + d2t` -- hence `id + d2t[id]`. None means the
        draft head already predicts the target vocabulary, so ids pass through.

        Equivalent to upstream's scatter-then-argmax for a monotonic mapping:
        both pick the same winner, and on an exact tie both pick the lowest id.
        """
        # A graph output: the one D2H of a draft pass, the rest stays on the host.
        draft_token_ids = draft_token_ids.cpu()
        d2t = self.draft_id_to_target_id
        if d2t is None:
            return draft_token_ids.long().clone()
        return draft_token_ids + d2t[draft_token_ids]

    def prepare_next_token_ids_padded(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        sampled_token_ids: torch.Tensor,
        requests: dict[str, CachedRequestState],
        gpu_input_batch: InputBatch,
        discard_request_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_reqs = gpu_input_batch.num_reqs
        self.backup_next_token_ids.np[:num_reqs] = np.array(
            [
                requests[gpu_input_batch.req_ids[i]].get_token_id(
                    common_attn_metadata.seq_lens[i].item()
                )
                for i in range(num_reqs)
            ],
            dtype=np.int32,
        )
        self.backup_next_token_ids.copy_to_gpu(num_reqs)
        backup_tokens = self.backup_next_token_ids.gpu

        assert discard_request_mask.dtype == torch.bool
        assert backup_tokens.dtype == torch.int32

        batch_size = sampled_token_ids.shape[0]
        sampled_token_ids = sampled_token_ids.to(backup_tokens.device)
        return eagle_prepare_next_token_padded(
            sampled_token_ids,
            discard_request_mask[:batch_size],
            backup_tokens[:batch_size],
            gpu_input_batch.vocab_size,
        )

    def prepare_inputs_padded(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        spec_decode_metadata: SpecDecodeMetadata,
        valid_sampled_tokens_count: torch.Tensor,
    ) -> tuple[CommonAttentionMetadata, torch.Tensor, torch.Tensor]:
        """
        This function is used to prepare the inputs for speculative decoding
        It updates the common_attn_metadata for speculative decoding,
        but does not consider the rejected tokens. Instead, all tokens
        are included as inputs to the speculator, with the rejected tokens
        used as padding and filtered out later by `token_indices_to_sample`.
        """
        token_indices_to_sample, num_rejected_tokens = eagle_prepare_inputs_padded(
            spec_decode_metadata.cu_num_draft_tokens,
            valid_sampled_tokens_count,
            common_attn_metadata.query_start_loc,
        )

        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        new_query_len_per_req = query_start_loc[1:] - query_start_loc[:-1]
        total_num_tokens = query_start_loc[-1].item()

        spec_common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            query_start_loc_cpu=query_start_loc,
            num_reqs=common_attn_metadata.num_reqs,
            num_actual_tokens=total_num_tokens,
            max_query_len=new_query_len_per_req.max().item(),
            max_seq_len=seq_lens.max().item(),
            block_table_tensor=common_attn_metadata.block_table_tensor,
            slot_mapping=torch.tensor(0),  # dummy,
            causal=True,
            dcp_local_seq_lens=common_attn_metadata.dcp_local_seq_lens,
        )

        return (
            spec_common_attn_metadata,
            token_indices_to_sample,
            num_rejected_tokens,
        )

    def get_model_name(self, model: nn.Module) -> str:
        # Upstream's load_model picks the target's image-token field by class
        # name and lists upstream classes only; an RBLN subclass of one goes by
        # its parent's name.
        return super().get_model_name(model).removeprefix("RBLN")

    def load_model(self, target_model: nn.Module) -> None:
        super().load_model(target_model)
        if self.supports_mm_inputs and not supports_rbln_multimodal_embeddings(
            self.model
        ):
            raise NotImplementedError(
                "The draft model must declare supports_rbln_multimodal_embeddings=True "
                "to use external multimodal embeddings on RBLN."
            )
        if (
            self.supports_mm_inputs
            and self.vllm_config.parallel_config.pipeline_parallel_size != 1
        ):
            raise NotImplementedError(
                "Multimodal drafting requires pipeline_parallel_size=1."
            )

        self.draft_id_to_target_id = getattr(self.model, "draft_id_to_target_id", None)
        if self.draft_id_to_target_id is not None:
            self.draft_id_to_target_id = self.draft_id_to_target_id.cpu()

        # Fused MoE is the only reader of the step's padded token dimension, so a
        # draft without it runs no collective of its own -- which is what lets an
        # idle rank skip drafting entirely rather than draft a discarded result.
        self.draft_has_moe = any(
            isinstance(module, MoERunner) for module in self.model.modules()
        )

        def model_wrapper(
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            token_indices_to_sample: torch.Tensor | None = None,
            inputs_embeds: torch.Tensor | None = None,
            mm_embeds: torch.Tensor | None = None,
            mm_mask: torch.Tensor | None = None,
        ):
            if mm_embeds is not None:
                assert mm_mask is not None
                inputs_embeds = torch.where(
                    mm_mask > 0.5, mm_embeds, self.model.embed_input_ids(input_ids)
                )
                input_ids = None
            ret_hidden_states = self.model(
                input_ids=input_ids,
                positions=positions,
                hidden_states=hidden_states,
                inputs_embeds=inputs_embeds,
            )
            if not self.model_returns_tuple():
                last_hidden_states = ret_hidden_states
                hidden_states = last_hidden_states
                # Only a tuple-returning drafter (Gemma4 MTP) samples narrower.
                assert last_hidden_states.shape[-1] == self.hidden_size
            else:
                last_hidden_states, hidden_states = ret_hidden_states

            hidden_states = hidden_states.view(-1, self.hidden_size)
            # Gemma4 MTP samples from its own, narrower hidden states.
            sample_hidden_states = last_hidden_states.view(
                -1, last_hidden_states.shape[-1]
            )

            if token_indices_to_sample is not None:
                hidden_states = hidden_states[token_indices_to_sample]
                sample_hidden_states = sample_hidden_states[token_indices_to_sample]

            logits = compute_draft_logits(sample_hidden_states)

            # NOTE(RBLN): the greedy pick belongs in the graph.
            # To support probabilistic sampling, we need to return
            # the logits too.
            return hidden_states, torch.ops.rbln.argmax(logits)

        def compute_draft_logits(sample_hidden_states: torch.Tensor) -> torch.Tensor:
            if self.draft_id_to_target_id is not None:
                # NOTE(RBLN): upstream's `compute_logits` widens draft-vocab
                # logits to the target vocabulary by scattering into an `-inf`
                # row. Its index is input-independent, so the subgraph folds
                # into an anonymous constant that weight-free apply cannot
                # resolve by name; it then executes on placeholder indices and
                # that out-of-bounds write is the SIGSEGV in KV warmup. Stay in
                # draft-vocab space and map after the argmax instead
                # (`_to_target_token_ids`) -- no per-model patch needed.
                assert isinstance(
                    self.model, (Eagle3LlamaForCausalLM, Eagle3DeepseekV2ForCausalLM)
                )
                return self.model.logits_processor(
                    self.model.lm_head, sample_hidden_states
                )
            return self.model.compute_logits(sample_hidden_states)

        def tail_wrapper(
            draft_ids: torch.Tensor,
            hidden_states: torch.Tensor,
            positions: torch.Tensor,
        ):
            # d2t is reached through the module so it stays a named weight in the
            # graph: weight-free apply cannot fill an anonymous constant.
            d2t = (
                None
                if self.draft_id_to_target_id is None
                else self.model.draft_id_to_target_id
            )

            def to_target(ids: torch.Tensor) -> torch.Tensor:
                ids = ids.reshape(-1)
                if d2t is not None:
                    ids = ids + d2t.index_select(0, ids)
                return ids.to(torch.int32)

            rows = draft_ids.shape[0]
            # The draft head's input embedding is in target space.
            ids = to_target(draft_ids)
            hidden = hidden_states.view(rows, 1, self.hidden_size)
            outs = [ids]
            for pass_idx in range(self.num_speculative_tokens - 1):
                pass_state.PASS_IDX = pass_idx
                ret = self.model(
                    input_ids=ids.view(rows, 1),
                    positions=positions + pass_idx,
                    hidden_states=hidden,
                    inputs_embeds=None,
                )
                if not self.model_returns_tuple():
                    last_hidden_states = next_hidden_states = ret
                else:
                    last_hidden_states, next_hidden_states = ret
                ids = to_target(
                    torch.ops.rbln.argmax(
                        compute_draft_logits(
                            last_hidden_states.view(-1, self.hidden_size)
                        )
                    )
                )
                outs.append(ids)
                hidden = next_hidden_states.view(rows, 1, self.hidden_size)
            pass_state.PASS_IDX = 0
            return torch.stack(outs, dim=1)

        self.model_executable = self._compile_drafter(model_wrapper)
        if self.num_speculative_tokens > 1:
            self.tail_executable = self._compile_drafter(tail_wrapper)

    def _compile_drafter(self, fn):
        rbln_config: RBLNConfig = self.vllm_config.additional_config
        return compile(
            fn,
            dynamic=False,
            fullgraph=True,
            compile_context=self.runner.compile_context,
            num_devices=1,
            model_trace_method="export" if use_device_tensor() else "",
            process_group_dict=build_process_group_dict(),
            guard_filter_fn=torch.compiler.keep_tensor_guards_unsafe,
            runtime_holder=self.runner.runtime_holder,
            mode="strict" if envs.VLLM_RBLN_COMPILE_STRICT_MODE else "",
            use_static_output=True,
            use_direct_dispatch=True,
            dtype=rbln_config.compile_dtype,
        )

    def _build_dummy_attn_metadata(
        self,
        num_reqs: int,
        num_tokens_per_req: int,
    ) -> CommonAttentionMetadata:
        num_tokens = num_tokens_per_req * num_reqs
        assert num_tokens <= self.max_num_tokens

        num_scheduled_tokens = np.array([num_tokens_per_req] * num_reqs, dtype=np.int32)
        seq_lens = torch.from_numpy(num_scheduled_tokens)

        cum_num_tokens, _ = self.runner._get_cumsum_and_arange(num_scheduled_tokens)
        query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32)
        query_start_loc[1 : num_reqs + 1] = torch.from_numpy(cum_num_tokens)

        return CommonAttentionMetadata(
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc,
            seq_lens=seq_lens,
            num_reqs=num_reqs,
            num_actual_tokens=num_tokens,
            max_query_len=num_tokens_per_req,
            max_seq_len=seq_lens.max().item(),
            block_table_tensor=self.runner.input_batch.block_table[
                self.kv_cache_gid
            ].get_cpu_tensor()[:num_reqs],
            slot_mapping=torch.tensor(0),  # dummy
            causal=True,
        )

    @torch.inference_mode()
    def dummy_run(
        self,
        num_reqs: int,
        num_tokens_per_req: int,
        is_prefill: bool,
        *,
        num_padded_tokens: int | None = None,
    ) -> None:
        status = self.runner.dp_status
        if (
            status is not None
            and status.is_idle[self.dp_rank]
            and not self.draft_has_moe
        ):
            # This rank has no work, so what a draft pass produces here is
            # discarded, and without fused MoE no busy rank is waiting inside a
            # collective for it.
            return

        num_tokens = num_tokens_per_req * num_reqs
        assert num_tokens <= self.max_num_tokens
        override_padded = num_padded_tokens

        common_attn_metadata = self._build_dummy_attn_metadata(
            num_reqs, num_tokens_per_req
        )
        batch_desc, num_tokens_across_dp = self._determine_batch_execution_and_padding(
            num_reqs,
            num_tokens,
            is_prefill,
            pinned_num_tokens_padded=override_padded,
        )
        num_reqs_padded = batch_desc.num_reqs_padded
        num_padded_tokens = batch_desc.num_tokens_padded

        per_layer_attn_metadata: dict[str, object] = {}
        for attn_group in self.draft_attn_groups:
            attn_metadata = attn_group.get_metadata_builder().build(
                common_attn_metadata=common_attn_metadata,
                positions=self.positions[:num_tokens],
                is_prefill=is_prefill,
                batch_pad=num_reqs_padded,
                is_prompt_sorted=self.runner.sort_batch_by_length,
            )
            self.runner.bind_kv_cache_to_metadata(attn_metadata)
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata

        token_indices_to_sample = (
            torch.arange(num_reqs, dtype=torch.int32) * num_tokens_per_req
        )
        input_ids, positions, hidden_states, token_indices_to_sample_padded = (
            self._preprocess(
                num_reqs,
                num_reqs_padded,
                num_tokens,
                self.positions[:num_tokens],
                self.hidden_states[:num_tokens],
                is_prefill=is_prefill,
                token_indices_to_sample=token_indices_to_sample,
                mrope_positions=(
                    self.mrope_positions[:, :num_tokens]
                    if self.supports_mm_inputs and self.uses_mrope
                    else None
                ),
            )
        )
        inputs_embeds = None

        with set_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            num_padded_tokens=num_padded_tokens,
            **self.runner.kv_cache_forward_context_kwargs(),
        ):
            hidden_states, draft_ids = self.model_executable(
                input_ids=input_ids,
                positions=positions,
                hidden_states=hidden_states,
                inputs_embeds=inputs_embeds,
                token_indices_to_sample=token_indices_to_sample_padded,
                **self._stage_mm_inputs(input_ids, is_prefill),
            )

        if self.num_speculative_tokens == 1 or self.runner.is_strict_kv_producer:
            return

        tail = self._prepare_tail(
            common_attn_metadata,
            self.positions[:num_reqs],
            num_reqs,
            None,
            pinned_num_tokens_padded=override_padded,
            mrope_positions=(
                self.mrope_positions[:, :num_reqs]
                if self.supports_mm_inputs and self.uses_mrope
                else None
            ),
        )
        self._run_tail(tail, draft_ids, hidden_states, num_reqs)

    def _determine_batch_execution_and_padding(
        self,
        num_reqs: int,
        num_tokens: int,
        is_prefill: bool,
        *,
        first_pass: bool = True,
        pinned_num_tokens_padded: int | None = None,
    ) -> tuple[BatchDescriptor, torch.Tensor | None]:
        """This pass's padded batch (see v1/worker/dp_utils.py).

        Under DP the draft decides from the status the step published, so a step
        that has not published one is a caller in the wrong place rather than a
        rank on its own; on a single rank there is no group to read.
        """
        if self.vllm_config.parallel_config.data_parallel_size == 1:
            status = None
        else:
            status = self.runner.dp_status
            assert status is not None, (
                "the step published no status for the draft to decide from"
            )
        return determine_draft_batch_execution_and_padding(
            cfg=self.runner.shape_config,
            status=status,
            dp_rank=self.dp_rank,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            is_prefill=is_prefill,
            draft_has_moe=self.draft_has_moe,
            first_pass=first_pass,
            pinned_num_tokens_padded=pinned_num_tokens_padded,
        )

    def _preprocess(
        self,
        num_reqs: int,
        num_reqs_padded: int,
        num_input_tokens: int,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        *,
        is_prefill: bool,
        token_indices_to_sample: torch.Tensor | None = None,
        target_token_ids: torch.Tensor | None = None,
        next_token_ids: torch.Tensor | None = None,
        mrope_positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if target_token_ids is not None:
            assert next_token_ids is not None
            assert token_indices_to_sample is not None
            assert num_input_tokens == target_token_ids.shape[0]

            self.input_ids[: num_input_tokens - 1] = target_token_ids[1:]
            self.input_ids[token_indices_to_sample] = next_token_ids

        input_ids = self.input_ids[:num_input_tokens].view(num_reqs, -1)
        positions = (
            positions.view(-1)[:num_input_tokens].view(num_reqs, -1)
            if mrope_positions is None
            else mrope_positions[:, :num_input_tokens].view(3, num_reqs, -1)
        )
        hidden_states = hidden_states.view(-1, self.hidden_size)[
            :num_input_tokens
        ].view(num_reqs, -1, self.hidden_size)

        layout = InputLayout(
            num_reqs=num_reqs,
            num_reqs_padded=num_reqs if is_prefill else num_reqs_padded,
            query_len=input_ids.shape[1],
            query_len_padded=(
                self.vllm_config.additional_config.max_prefill_chunk_size
                if is_prefill
                else input_ids.shape[1]
            ),
        )
        staged = self.input_stager.stage(
            input_ids=input_ids,
            positions=positions,
            hidden_states=hidden_states,
            token_indices=token_indices_to_sample,
            layout=layout,
        )
        assert staged.hidden_states is not None

        return (
            staged.input_ids,
            staged.positions,
            staged.hidden_states,
            staged.token_indices,
        )

    def _stage_mm_inputs(
        self,
        input_ids: torch.Tensor,
        is_prefill: bool,
        mm_embed_inputs: tuple[list[tuple[int, int, torch.Tensor]], torch.Tensor]
        | None = None,
    ) -> dict[str, torch.Tensor]:
        if not (self.supports_mm_inputs and is_prefill):
            return {}
        if mm_embed_inputs is None:
            rows: list[tuple[int, int, torch.Tensor]] = []
            mask = torch.zeros(*input_ids.shape, 1, dtype=self.dtype)
        else:
            rows, mask = mm_embed_inputs
            mask = mask.view(input_ids.shape[0], -1, 1)
        layout = InputLayout(
            num_reqs=input_ids.shape[0],
            num_reqs_padded=input_ids.shape[0],
            query_len=mask.shape[1],
            query_len_padded=input_ids.shape[1],
        )
        embeds, mask = self.input_stager.stage_mm_embeds(
            rows, mask, self.inputs_embeds_size, layout
        )
        return {"mm_embeds": embeds, "mm_mask": mask}
