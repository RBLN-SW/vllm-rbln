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
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.model_executor.models.deepseek_eagle3 import Eagle3DeepseekV2ForCausalLM
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

import vllm_rbln.envs as envs
from vllm_rbln.compilation import (
    build_process_group_dict,
    compile,
)
from vllm_rbln.forward_context import RBLNDPMetadata, set_forward_context
from vllm_rbln.logger import init_logger
from vllm_rbln.platform import USE_DEVICE_TENSOR
from vllm_rbln.utils import pad
from vllm_rbln.v1.attention.kv_cache_bindings import (
    attach_kv_cache_bindings,
    build_kv_cache_forward_context_kwargs,
)
from vllm_rbln.v1.spec_decode.utils import (
    eagle_prepare_inputs_padded,
    eagle_prepare_next_token_padded,
)

if TYPE_CHECKING:
    from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

logger = init_logger(__name__)


class RBLNEagleProposer(EagleProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner: "RBLNModelRunner",
    ):
        super().__init__(vllm_config, device, runner)

        if self.supports_mm_inputs:
            raise NotImplementedError

        self.runner = runner
        # Set in load_model when eagle3 + compilation are both on.
        self._compiled_combine = None

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
        # The step loop below is instrumented, but everything before it -- the
        # aux projection, buffer fill, DP padding, first metadata build and the
        # first forward -- was not, and it is the larger half: under
        # RBLN_RUNTIME_FORCE_SYNC=1 the whole `draft` phase measured 45.87 ms
        # per step while the two loop iterations accounted for only 6.4 ms.
        # These scopes split the remainder. All are no-ops unless
        # VLLM_CUSTOM_SCOPES_FOR_PROFILING=1.
        if self.method == "eagle3":
            assert isinstance(
                self.model, (Eagle3LlamaForCausalLM, Eagle3DeepseekV2ForCausalLM)
            )
            with record_function_or_nullcontext("drafter/first: combine"):
                target_hidden_states = self._combine_hidden_states(target_hidden_states)
            assert target_hidden_states.shape[-1] == self.hidden_size

        with record_function_or_nullcontext("drafter/first: set_inputs"):
            num_tokens, token_indices_to_sample = self.set_inputs_first_pass(
                target_token_ids=target_token_ids,
                next_token_ids=next_token_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                token_indices_to_sample=token_indices_to_sample,
                cad=common_attn_metadata,
            )

        assert self.runner is not None
        is_prefill = self.runner.is_prefill

        # Build attention metadata
        num_reqs = self.runner.input_batch.num_reqs
        with record_function_or_nullcontext("drafter/first: dp_padding"):
            num_reqs_padded, num_padded_tokens, num_tokens_across_dp = (
                self._determine_draft_batch_padding(num_reqs, num_tokens, is_prefill)
            )
        per_layer_attn_metadata: dict[str, object] = {}
        with record_function_or_nullcontext("drafter/first: attn_meta"):
            for attn_group in self.draft_attn_groups:
                attn_metadata = attn_group.get_metadata_builder().build(
                    common_attn_metadata=common_attn_metadata,
                    positions=target_positions,
                    is_prefill=is_prefill,
                    batch_pad=num_reqs_padded,
                )
                attach_kv_cache_bindings(
                    attn_metadata,
                    self.runner.kv_caches,
                    self.runner.kv_cache_bases,
                    self.runner.kv_cache_view_infos,
                )
                for layer_name in attn_group.layer_names:
                    per_layer_attn_metadata[layer_name] = attn_metadata

        with record_function_or_nullcontext("drafter/first: preprocess"):
            input_ids, positions, hidden_states, token_indices_to_sample_padded = (
                self._preprocess(
                    num_reqs,
                    num_reqs_padded,
                    num_tokens,
                    token_indices_to_sample,
                    is_prefill,
                )
            )
        inputs_embeds = None

        with (
            record_function_or_nullcontext("drafter/first: forward"),
            set_forward_context(
                per_layer_attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                num_padded_tokens=num_padded_tokens,
                **build_kv_cache_forward_context_kwargs(self.runner.kv_cache_bases),
            ),
        ):
            hidden_states, logits = self.model_executable(
                input_ids=input_ids,
                positions=positions,
                hidden_states=hidden_states,
                inputs_embeds=inputs_embeds,
                last_token_indices=token_indices_to_sample_padded,
            )

        # Early exit if there is only one draft token to be generated.
        if self.num_speculative_tokens == 1:
            draft_tokens_ids = logits[:num_reqs].argmax(dim=-1)
            return draft_tokens_ids.view(-1, 1)

        # Gathers plus the first argmax. Grouped so the `draft` remainder can be
        # attributed: argmax.out is a registered host fallback and measures
        # 0.195 ms per call standalone, essentially all of it the crossing.
        _post = record_function_or_nullcontext("drafter/first: sample")
        _post.__enter__()
        assert token_indices_to_sample_padded is not None
        positions = target_positions[
            token_indices_to_sample_padded.to(target_positions.device)
        ]

        # Upstream gathers the fed-back hidden states to the sampled positions
        # immediately after the `positions` gather above (llm_base_proposer.py:
        # `hidden_states = hidden_states[token_indices_to_sample]`). This
        # override kept the `positions` line and dropped this one.
        #
        # `model_wrapper` returns `hidden_states` UNGATHERED -- only
        # `sample_hidden_states` is indexed by `last_token_indices`, and that
        # goes to `compute_logits`. So this tensor is (num_reqs * (1 +
        # num_spec), hidden_size) on the first pass, while the loop below feeds
        # it back through `self.hidden_states[:num_reqs]`. Without the gather
        # each request reads the leading request's rows instead of its own last
        # token.
        #
        # It never crashes: `input_ids` stays per-request correct, so the
        # drafter emits plausible tokens conditioned on the wrong hidden state.
        # That is the step in per-position conditional acceptance -- 63% on the
        # first draft, which uses the target's hidden states, then 36% flat once
        # the drafter's own output is fed back.
        # index_select rather than advanced indexing: this is a 1-D row
        # selection, so the two are equivalent, but only index_select takes the
        # backend's native path (measured 0.011 ms vs 4.03 ms per call for
        # aten::index). Adding the gather as advanced indexing cost ~100 ms per
        # step -- far more than the op itself -- which points at a host fallback
        # draining the async dispatch queue mid-step rather than at the indexing
        # arithmetic.
        hidden_states = hidden_states.index_select(
            0, token_indices_to_sample_padded.to(hidden_states.device)
        )

        draft_token_ids = logits[:num_reqs].argmax(dim=-1)

        if self.allowed_attn_types is not None and not isinstance(
            attn_metadata, self.allowed_attn_types
        ):
            raise ValueError(
                f"Unsupported attention metadata type for speculative "
                "decoding with num_speculative_tokens > 1: "
                f"{type(attn_metadata)}. Supported types are: "
                f"{self.allowed_attn_types}"
            )

        _post.__exit__(None, None, None)

        # Generate the remaining draft tokens.
        draft_token_ids_list = [draft_token_ids]

        common_attn_metadata.num_actual_tokens = num_reqs
        common_attn_metadata.max_query_len = 1
        common_attn_metadata.query_start_loc = self.arange[: num_reqs + 1]
        common_attn_metadata.query_start_loc_cpu = self.arange[: num_reqs + 1].cpu()

        # In padded drafter batch, we need to adjust the sequence lengths
        # to remove the "padding" (i.e. rejected tokens).
        # Only apply this adjustment when we have rejected tokens
        # (i.e., not the first proposal).
        if self.num_speculative_tokens > 1 and num_rejected_tokens is not None:
            common_attn_metadata.seq_lens -= num_rejected_tokens
            # Same reasoning as in the step loop below: the flash-attention
            # builder reads the HOST shadow, not `seq_lens`, so mirror the
            # adjustment there. Invalidating it instead (the commented-out lines
            # this replaces) would force a D2H sync every step; applying the
            # same arithmetic on the host is free.
            _slc0 = common_attn_metadata._seq_lens_cpu
            if _slc0 is not None:
                _slc0 -= num_rejected_tokens.to(_slc0.device, _slc0.dtype)

        num_reqs_padded, num_padded_tokens, num_tokens_across_dp = (
            self._determine_draft_batch_padding(num_reqs, num_reqs, False)
        )
        for token_index in range(self.num_speculative_tokens - 1):
            _upd = record_function_or_nullcontext("drafter: loop_update")
            _upd.__enter__()
            # Update the inputs
            # cast to int32 is crucial when eagle model is compiled.
            # tensor.argmax returns int64 by default.
            self.input_ids[:num_reqs] = draft_token_ids_list[-1].int()
            positions = positions.view(-1) + 1
            self.positions[:num_reqs] = positions[:num_reqs]
            self.hidden_states[: hidden_states.shape[0]] = hidden_states

            exceeds_max_model_len = positions[:num_reqs] >= self.max_model_len
            common_attn_metadata.seq_lens += 1
            common_attn_metadata.seq_lens.masked_fill_(exceeds_max_model_len, 1)
            # Keep the HOST shadow in step with the device tensor. The
            # flash-attention builder reads `_seq_lens_cpu`, not `seq_lens`
            # (`num_computed_tokens = seq_lens - query_seq_lens_cpu`), so a
            # shadow that is never updated leaves every step after the first
            # attending with a stale sequence length. That does not crash -- it
            # silently degrades the drafts, i.e. exactly the acceptance rate
            # this path exists to produce.
            #
            # Mirror the device op exactly: unsliced, same mask.
            # `exceeds_max_model_len` derives from `positions`, a host tensor.
            _slc = common_attn_metadata._seq_lens_cpu
            if _slc is not None:
                _slc += 1
                _slc.masked_fill_(exceeds_max_model_len.to(_slc.device), 1)

            _upd.__exit__(None, None, None)

            # Split the drafter iteration into host metadata build vs device
            # forward. The enclosing "rbln_model_runner: draft" phase is by far
            # the most expensive part of a spec-decode step here, and the two
            # candidates -- host attention metadata construction and the device
            # forward -- need very different fixes, so the aggregate number is
            # not actionable on its own.
            #
            # Read these only under RBLN_RUNTIME_FORCE_SYNC=1. With the default
            # async dispatch, device time lands in whichever scope happens to
            # block, which is how an earlier profile blamed `postprocess` (one
            # line of indexing) for 32 ms.
            #
            # Both are no-ops unless VLLM_CUSTOM_SCOPES_FOR_PROFILING=1.
            _attn_scope = record_function_or_nullcontext("drafter: attn_meta")
            _attn_scope.__enter__()

            # Rebuild attention metadata
            per_layer_attn_metadata.clear()
            for attn_group in self.draft_attn_groups:
                attn_metadata = attn_group.get_metadata_builder().build(
                    common_attn_metadata=common_attn_metadata,
                    positions=positions,
                    is_prefill=False,
                    batch_pad=num_reqs_padded,
                )
                attach_kv_cache_bindings(
                    attn_metadata,
                    self.runner.kv_caches,
                    self.runner.kv_cache_bases,
                    self.runner.kv_cache_view_infos,
                )
                for layer_name in attn_group.layer_names:
                    per_layer_attn_metadata[layer_name] = attn_metadata

            input_ids, positions, hidden_states, _ = self._preprocess(
                num_reqs, num_reqs_padded, num_reqs, None, False
            )

            # Run the model.
            _attn_scope.__exit__(None, None, None)

            with (
                record_function_or_nullcontext("drafter: forward"),
                set_forward_context(
                    per_layer_attn_metadata,
                    self.vllm_config,
                    num_tokens=num_reqs,
                    num_tokens_across_dp=num_tokens_across_dp,
                    num_padded_tokens=num_padded_tokens,
                    **build_kv_cache_forward_context_kwargs(self.runner.kv_cache_bases),
                ),
            ):
                hidden_states, logits = self.model_executable(
                    input_ids=input_ids,
                    positions=positions,
                    hidden_states=hidden_states,
                    inputs_embeds=inputs_embeds,
                    last_token_indices=None,
                )
            draft_token_ids = logits[:num_reqs].argmax(dim=-1)
            draft_token_ids_list.append(draft_token_ids)

        # [batch_size, num_speculative_tokens]
        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
        return draft_token_ids

    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
    ) -> tuple[int, torch.Tensor]:
        if self.needs_extra_input_slots:
            raise NotImplementedError(
                "vllm-rbln does not support EAGLE extra input slots required for "
                "parallel drafting or draft-model speculative decoding yet."
            )

        if token_indices_to_sample is None:
            token_indices_to_sample = cad.query_start_loc[1:] - 1

        num_tokens = target_token_ids.shape[0]
        self.input_ids[: num_tokens - 1] = target_token_ids[1:]
        self.input_ids[token_indices_to_sample] = next_token_ids

        self._set_positions(num_tokens, target_positions)

        self.hidden_states[:num_tokens] = target_hidden_states.view(
            -1, self.hidden_size
        )[:num_tokens]

        return num_tokens, token_indices_to_sample

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
        backup_tokens_gpu = self.backup_next_token_ids.gpu

        assert discard_request_mask.dtype == torch.bool
        assert backup_tokens_gpu.dtype == torch.int32

        batch_size = sampled_token_ids.shape[0]
        return eagle_prepare_next_token_padded(
            sampled_token_ids,
            discard_request_mask[:batch_size],
            backup_tokens_gpu[:batch_size],
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
            common_attn_metadata.query_start_loc_cpu,
        )

        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        seq_lens_cpu = (
            common_attn_metadata._seq_lens_cpu
            if common_attn_metadata._seq_lens_cpu is not None
            else common_attn_metadata.seq_lens.cpu()
        )
        new_query_len_per_req = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]

        total_num_tokens = query_start_loc_cpu[-1].item()

        spec_common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=common_attn_metadata.query_start_loc,
            seq_lens=common_attn_metadata.seq_lens,
            query_start_loc_cpu=query_start_loc_cpu,
            _seq_lens_cpu=common_attn_metadata._seq_lens_cpu,
            _num_computed_tokens_cpu=common_attn_metadata._num_computed_tokens_cpu,
            num_reqs=common_attn_metadata.num_reqs,
            num_actual_tokens=total_num_tokens,
            max_query_len=new_query_len_per_req.max().item(),
            max_seq_len=seq_lens_cpu.max().item(),
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

    def load_model(self, target_model: nn.Module) -> None:
        super().load_model(target_model)

        def model_wrapper(
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            last_token_indices: torch.Tensor | None = None,
            inputs_embeds: torch.Tensor | None = None,
        ):
            ret_hidden_states = self.model(
                input_ids=input_ids,
                positions=positions,
                hidden_states=hidden_states,
                inputs_embeds=inputs_embeds,
            )
            if not self.model_returns_tuple():
                last_hidden_states = ret_hidden_states
                hidden_states = last_hidden_states
            else:
                last_hidden_states, hidden_states = ret_hidden_states

            hidden_states = hidden_states.view(-1, self.hidden_size)
            last_hidden_states = last_hidden_states.view(-1, self.hidden_size)
            sample_hidden_states = (
                last_hidden_states[last_token_indices]
                if last_token_indices is not None
                else last_hidden_states
            )
            logits = self.model.compute_logits(sample_hidden_states)

            return hidden_states, logits

        if (
            self.vllm_config.speculative_config.enforce_eager
            or not envs.VLLM_RBLN_COMPILE_MODEL
        ):
            self.model_executable = model_wrapper
        else:
            self.model_executable = compile(
                model_wrapper,
                dynamic=False,
                fullgraph=True,
                compile_context=self.runner.compile_context,
                num_devices=envs.VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK,
                model_trace_method="export" if USE_DEVICE_TENSOR else "",
                process_group_dict=build_process_group_dict(),
                guard_filter_fn=torch.compiler.keep_tensor_guards_unsafe,
                mode="strict" if envs.VLLM_RBLN_COMPILE_STRICT_MODE else "",
            )
            # Separate graph for the aux-state projection, which runs outside
            # `model_wrapper` (propose() calls it before the drafter's own
            # inputs exist). See `_combine_hidden_states` for why it matters:
            # eager it is the largest non-forward item in the step.
            if self.method == "eagle3":
                self._compiled_combine = compile(
                    self.model.combine_hidden_states,
                    dynamic=False,
                    compile_context=self.runner.compile_context,
                    num_devices=envs.VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK,
                    model_trace_method="export" if USE_DEVICE_TENSOR else "",
                    process_group_dict=build_process_group_dict(),
                    guard_filter_fn=torch.compiler.keep_tensor_guards_unsafe,
                    mode="strict" if envs.VLLM_RBLN_COMPILE_STRICT_MODE else "",
                )

    def _combine_hidden_states(
        self, target_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """EAGLE3's aux-state projection, compiled when the shape is a bucket.

        Run eagerly this single `Linear(num_aux * target_hidden -> hidden)` was
        the largest non-forward item in a spec-decode step: 19.6 ms/step,
        measured with RBLN_RUNTIME_FORCE_SYNC=1 and torch profiler. It is not
        compute -- standalone it costs the same for 4 and for 16 tokens, and
        scales exactly with the WEIGHT size (56.6 MB -> 7.5 ms, 18.9 MB ->
        2.5 ms, i.e. 7.5 GB/s both times), so the weight is materialised on
        every call. Compiled, the same op is 0.30 ms because the weight stays
        device-resident. 26x.

        Decode has a single shape once the token count is padded to the batch
        bucket, so the graph compiles once. Prefill token counts are unbounded;
        those stay eager rather than risk one recompile per shape.
        """
        combine = getattr(self, "_compiled_combine", None)
        if combine is None or self.runner.is_prefill:
            return self.model.combine_hidden_states(target_hidden_states)

        num_tokens = target_hidden_states.shape[0]
        num_reqs_padded = self.runner.bucketing_manager.find_decode_batch_bucket(
            self.runner.input_batch.num_reqs
        )
        padded = num_reqs_padded * (1 + self.num_speculative_tokens)
        if num_tokens > padded:
            # Shape outside the bucket we compiled for -- do not trigger a
            # recompile just to save a few ms.
            return self.model.combine_hidden_states(target_hidden_states)

        out = combine(pad(target_hidden_states, 0, padded))
        return out[:num_tokens] if num_tokens < padded else out

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
            query_start_loc=query_start_loc.to(self.device),
            query_start_loc_cpu=query_start_loc,
            seq_lens=seq_lens.to(self.device),
            _seq_lens_cpu=seq_lens,
            num_reqs=num_reqs,
            num_actual_tokens=num_tokens,
            max_query_len=num_tokens_per_req,
            max_seq_len=seq_lens.max().item(),
            block_table_tensor=self.runner.input_batch.block_table[0].get_cpu_tensor()[
                :num_reqs
            ],
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
        num_tokens = num_tokens_per_req * num_reqs
        assert num_tokens <= self.max_num_tokens
        override_padded = num_padded_tokens

        common_attn_metadata = self._build_dummy_attn_metadata(
            num_reqs, num_tokens_per_req
        )
        num_reqs_padded, dp_padded, num_tokens_across_dp = (
            self._determine_draft_batch_padding(num_reqs, num_tokens, is_prefill)
        )
        num_padded_tokens = override_padded or dp_padded

        per_layer_attn_metadata: dict[str, object] = {}
        for attn_group in self.draft_attn_groups:
            attn_metadata = attn_group.get_metadata_builder().build(
                common_attn_metadata=common_attn_metadata,
                positions=self.positions[:num_tokens],
                is_prefill=is_prefill,
                batch_pad=num_reqs_padded,
            )
            attach_kv_cache_bindings(
                attn_metadata,
                self.runner.kv_caches,
                self.runner.kv_cache_bases,
                self.runner.kv_cache_view_infos,
            )
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata

        token_indices_to_sample = (
            torch.arange(num_reqs, device=self.device, dtype=torch.int32)
            * num_tokens_per_req
        )
        input_ids, positions, hidden_states, token_indices_to_sample_padded = (
            self._preprocess(
                num_reqs,
                num_reqs_padded,
                num_tokens,
                token_indices_to_sample,
                is_prefill,
            )
        )
        inputs_embeds = None

        with set_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            num_padded_tokens=num_padded_tokens,
            **build_kv_cache_forward_context_kwargs(self.runner.kv_cache_bases),
        ):
            _, _ = self.model_executable(
                input_ids=input_ids,
                positions=positions,
                hidden_states=hidden_states,
                inputs_embeds=inputs_embeds,
                last_token_indices=token_indices_to_sample_padded,
            )

        if self.num_speculative_tokens == 1:
            return

        common_attn_metadata.num_actual_tokens = num_reqs
        common_attn_metadata.max_query_len = 1
        common_attn_metadata.query_start_loc = self.arange[: num_reqs + 1]
        common_attn_metadata.query_start_loc_cpu = (
            common_attn_metadata.query_start_loc.cpu()
        )
        common_attn_metadata._seq_lens_cpu += 1
        common_attn_metadata.seq_lens = common_attn_metadata._seq_lens_cpu.to(
            self.device
        )

        num_reqs_padded, dp_padded, num_tokens_across_dp = (
            self._determine_draft_batch_padding(num_reqs, num_reqs, False)
        )
        num_padded_tokens = override_padded or dp_padded
        per_layer_attn_metadata.clear()
        for attn_group in self.draft_attn_groups:
            attn_metadata = attn_group.get_metadata_builder().build(
                common_attn_metadata=common_attn_metadata,
                positions=self.positions[:num_reqs],
                is_prefill=False,
                batch_pad=num_reqs_padded,
            )
            attach_kv_cache_bindings(
                attn_metadata,
                self.runner.kv_caches,
                self.runner.kv_cache_bases,
                self.runner.kv_cache_view_infos,
            )
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata

        input_ids, positions, hidden_states, _ = self._preprocess(
            num_reqs, num_reqs_padded, num_reqs, None, False
        )

        for _ in range(self.num_speculative_tokens - 1):
            with set_forward_context(
                per_layer_attn_metadata,
                self.vllm_config,
                num_tokens=num_reqs,
                num_tokens_across_dp=num_tokens_across_dp,
                num_padded_tokens=num_padded_tokens,
                **build_kv_cache_forward_context_kwargs(self.runner.kv_cache_bases),
            ):
                _, _ = self.model_executable(
                    input_ids=input_ids,
                    positions=positions,
                    hidden_states=hidden_states,
                    inputs_embeds=inputs_embeds,
                    last_token_indices=None,
                )

    def _preprocess(
        self,
        num_reqs: int,
        num_reqs_padded: int,
        num_input_tokens: int,
        token_indices_to_sample: torch.Tensor | None,
        is_prefill: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if is_prefill:
            input_ids = self.input_ids.view(num_reqs, -1)
            positions = self.positions.view(num_reqs, -1)
            target_hidden_states = self.hidden_states
        else:
            input_ids = self.input_ids[:num_input_tokens].view(num_reqs, -1)
            positions = self.positions[:num_input_tokens].view(num_reqs, -1)
            target_hidden_states = self.hidden_states[:num_input_tokens].view(
                num_reqs, -1, self.hidden_size
            )
            input_ids = pad(input_ids, 0, num_reqs_padded)
            positions = pad(positions, 0, num_reqs_padded)
            target_hidden_states = pad(target_hidden_states, 0, num_reqs_padded)

        target_hidden_states = target_hidden_states.view(
            *input_ids.shape, -1
        )  # [B, L, H]
        token_indices_to_sample_padded = (
            pad(token_indices_to_sample, 0, num_reqs_padded)
            if token_indices_to_sample is not None
            else None
        )

        return (
            input_ids,
            positions,
            target_hidden_states,
            token_indices_to_sample_padded,
        )

    def _determine_draft_batch_padding(
        self,
        num_reqs: int,
        num_tokens: int,
        is_prefill: bool,
    ) -> tuple[int, int | None, torch.Tensor | None]:
        num_reqs_padded = (
            self.runner.bucketing_manager.find_decode_batch_bucket(num_reqs)
            if not is_prefill
            else num_reqs
        )
        dp_size = self.vllm_config.parallel_config.data_parallel_size
        if dp_size == 1:
            return num_reqs_padded, None, None

        num_tokens_across_dp, num_reqs_across_dp = (
            RBLNDPMetadata.num_tokens_and_reqs_across_dp(
                num_tokens, num_reqs, dp_size, self.dp_rank, is_prefill
            )
        )
        num_tokens_padded = self.max_num_tokens
        if self.runner.specialized_moe_decode and not is_prefill:
            if num_reqs_across_dp is None:
                num_reqs_padded = self.runner.bucketing_manager.decode_batch_buckets[-1]
            else:
                num_reqs_padded = (
                    self.runner.bucketing_manager.find_decode_batch_bucket(
                        int(num_reqs_across_dp.max())
                    )
                )
                max_tokens_per_req = int(
                    (num_tokens_across_dp // num_reqs_across_dp).max()
                )
                num_tokens_padded = num_reqs_padded * max_tokens_per_req
        return num_reqs_padded, num_tokens_padded, num_tokens_across_dp
