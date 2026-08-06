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
        self.arange_cpu = torch.arange(self.arange.shape[0], dtype=torch.int32)
        # Populated from the draft model in `load_model`. None means the draft
        # head shares the target vocabulary, so `propose()` maps no ids.
        self.draft_id_to_target_id: torch.Tensor | None = None

    def _fold_combine(self) -> bool:
        """Whether the aux-state projection runs inside the drafter's graph.

        Both the decode and the prefill drafter graphs take a fixed-shape
        `hidden_states`, so the projection can be folded into either without
        adding a shape variant: decode pads to the batch bucket, prefill takes
        the whole buffer. That leaves the projection's weight device-resident
        instead of materialised on every eager call.

        `is_prefill` must be passed wherever the caller already knows it.
        `dummy_run` compiles prefill and decode shapes in the same pass while
        `runner.is_prefill` does not move between them, so reading the runner
        there folds the prefill graph as well and the first real prefill step
        recompiles -- the exact failure this fold was added to remove, moved to
        the other path.
        """
        return self.method == "eagle3"

    def _aux_width(self) -> int:
        """Width the folded projection expects at its input.

        `combine_hidden_states` takes `num_aux_hidden_states` states concatenated
        on the last dim, so the drafter graphs see that width rather than
        `hidden_size` once the projection is folded in. The warmup has to build
        its dummy inputs at the same width, otherwise it compiles the narrow
        shape and serving triggers a runtime recompile -- which surfaces as
        `code=201 INIT_INTERNAL (Seed address mismatch)` rather than as anything
        that names the shape.
        """
        return self.hidden_size * getattr(self.model.model, "num_aux_hidden_states", 3)

    def _draft_ids(self, out: torch.Tensor, num_reqs: int) -> torch.Tensor:
        """Target-vocabulary ids for this drafter step.

        `model_wrapper` reduces inside the compiled region, so `out` is already
        draft ids rather than logits: `torch.ops.rbln.argmax` lowers to
        contrib_top_k_top_p_sample(k=1, p=0) and runs on the device, where an
        eager `argmax` is a registered host fallback and costs a round trip per
        drafter pass. Only the draft->target map is left.
        """
        return self._to_target_token_ids(out[:num_reqs])

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
        fold = self._fold_combine()
        if self.method == "eagle3":
            assert isinstance(
                self.model, (Eagle3LlamaForCausalLM, Eagle3DeepseekV2ForCausalLM)
            )
            if not fold:
                # Eager on purpose. The projection used to get its own compiled
                # graph because torch-rbln's non-deploy mode scans every eager
                # op's inputs for NaN/Inf, which walked this 56.6 MB weight on
                # every call (19.6 ms/step). With TORCH_RBLN_DEPLOY=ON the scan
                # is gone and eager is 0.011 ms -- faster than the 0.035 ms
                # graph -- so the graph was pure overhead: compiled on every
                # start, never called under the default FUSE_FIRST_FORWARD=1,
                # and its weight load is what fails `rbln_memcpy_v2h` when
                # FUSE=0 (56,623,104 bytes = 3072 x 9216 x 2 exactly).
                # This path now REQUIRES TORCH_RBLN_DEPLOY=ON to be fast.
                with record_function_or_nullcontext("drafter/first: combine"):
                    target_hidden_states = self.model.combine_hidden_states(
                        target_hidden_states
                    )
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
            if fold:
                # The projection happens inside the graph now, so hand it the
                # wide aux states rather than the (narrow) drafter buffer. Both
                # branches reproduce exactly what `_preprocess` would have
                # produced from the buffer, at num_aux * target_hidden width.
                w = target_hidden_states.shape[-1]
                flat = target_hidden_states.reshape(-1, w)
                if is_prefill:
                    # `_preprocess` hands the whole buffer to the prefill graph
                    # rather than a slice, which is what keeps that graph's shape
                    # constant across chunk sizes. Pad to the same row count so
                    # the folded projection inherits that property.  Upstream
                    # called prefill token counts unbounded, but they are bounded
                    # by max_num_batched_tokens and `pad` short-circuits on the
                    # full chunks that dominate.
                    hidden_states = pad(flat, 0, self.hidden_states.shape[0]).view(
                        num_reqs, -1, w
                    )
                else:
                    hidden_states = pad(
                        flat[:num_tokens].view(num_reqs, -1, w), 0, num_reqs_padded
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
                token_indices_to_sample=token_indices_to_sample_padded,
            )

        # Early exit if there is only one draft token to be generated.
        if self.num_speculative_tokens == 1:
            draft_tokens_ids = self._draft_ids(logits, num_reqs)
            return draft_tokens_ids.view(-1, 1)

        with record_function_or_nullcontext("drafter/first: sample"):
            assert token_indices_to_sample_padded is not None
            positions = target_positions[
                token_indices_to_sample_padded.to(target_positions.device)
            ]

            # `hidden_states` is deliberately not gathered here -- #821 moved
            # that gather inside `model_wrapper`. Doing it again would index an
            # already (num_reqs_padded, hidden_size) tensor with token-space
            # indices and raise on the first decode.
            draft_token_ids = self._draft_ids(logits, num_reqs)

        if self.allowed_attn_types is not None and not isinstance(
            attn_metadata, self.allowed_attn_types
        ):
            raise ValueError(
                f"Unsupported attention metadata type for speculative "
                "decoding with num_speculative_tokens > 1: "
                f"{type(attn_metadata)}. Supported types are: "
                f"{self.allowed_attn_types}"
            )

        # Generate the remaining draft tokens.
        draft_token_ids_list = [draft_token_ids]

        # it mutates seq_lens in place; detach from the runner's persistent buffer.
        common_attn_metadata.seq_lens = common_attn_metadata.seq_lens.clone()
        common_attn_metadata.num_actual_tokens = num_reqs
        common_attn_metadata.max_query_len = 1
        common_attn_metadata.query_start_loc = self.arange_cpu[: num_reqs + 1]
        common_attn_metadata.query_start_loc_cpu = self.arange_cpu[: num_reqs + 1]

        # In padded drafter batch, we need to adjust the sequence lengths
        # to remove the "padding" (i.e. rejected tokens).
        # Only apply this adjustment when we have rejected tokens
        # (i.e., not the first proposal).
        _adj = record_function_or_nullcontext("drafter/first: rejected_adjust")
        _adj.__enter__()
        if self.num_speculative_tokens > 1 and num_rejected_tokens is not None:
            common_attn_metadata.seq_lens -= num_rejected_tokens

        _adj.__exit__(None, None, None)

        with record_function_or_nullcontext("drafter/first: batch_padding"):
            num_reqs_padded, num_padded_tokens, num_tokens_across_dp = (
                self._determine_draft_batch_padding(num_reqs, num_reqs, False)
            )
        for token_index in range(self.num_speculative_tokens - 1):
            with record_function_or_nullcontext("drafter: loop_update"):
                self.input_ids[:num_reqs] = draft_token_ids_list[-1].int()
                positions = positions.view(-1) + 1
                self.positions[:num_reqs] = positions[:num_reqs]
                self.hidden_states[: hidden_states.shape[0]] = hidden_states

                exceeds_max_model_len = positions[:num_reqs] >= self.max_model_len
                common_attn_metadata.seq_lens += 1
                common_attn_metadata.seq_lens.masked_fill_(exceeds_max_model_len, 1)

            with record_function_or_nullcontext("drafter: attn_meta"):
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
                    token_indices_to_sample=None,
                )
            with record_function_or_nullcontext("drafter: sample"):
                draft_token_ids = self._draft_ids(logits, num_reqs)
                draft_token_ids_list.append(draft_token_ids)

        with record_function_or_nullcontext("drafter: stack"):
            # [batch_size, num_speculative_tokens]
            draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
        return draft_token_ids

    def _to_target_token_ids(self, draft_token_ids: torch.Tensor) -> torch.Tensor:
        """Map draft-vocabulary ids to target-vocabulary ids.

        `d2t` holds offsets, not absolute ids -- upstream scatters at
        `arange(draft_vocab_size) + d2t` -- hence `id + d2t[id]`. None means the
        draft head already predicts the target vocabulary, so ids pass through.

        Equivalent to upstream's scatter-then-argmax for a monotonic mapping:
        both pick the same winner, and on an exact tie both pick the lowest id.
        """
        d2t = self.draft_id_to_target_id
        if d2t is None:
            return draft_token_ids
        # `index_select` rather than `d2t[ids]`: equivalent for a 1-D row
        # selection, but only `index_select` takes the backend's native path --
        # advanced indexing is the largest single eager op in the step trace.
        # Cast back so the caller keeps the int32 the compiled region produced.
        flat = draft_token_ids.reshape(-1)
        mapped = flat + d2t.index_select(0, flat)
        return mapped.to(draft_token_ids.dtype).view_as(draft_token_ids)

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
            token_indices_to_sample = (cad.query_start_loc[1:] - 1).to(self.device)

        num_tokens = target_token_ids.shape[0]
        self.input_ids[: num_tokens - 1] = target_token_ids[1:]
        self.input_ids[token_indices_to_sample] = next_token_ids

        self._set_positions(num_tokens, target_positions)

        if not self._fold_combine():
            self.hidden_states[:num_tokens] = target_hidden_states.view(
                -1, self.hidden_size
            )[:num_tokens]
        # With FUSE_FIRST_FORWARD the aux states are still at
        # num_aux * target_hidden width, so they do not fit this buffer and the
        # caller pads the tensor it already holds instead. Nothing is dropped:
        # `propose` overrides the hidden_states that `_preprocess` returns.

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

    def load_model(self, target_model: nn.Module) -> None:
        super().load_model(target_model)
        self._probe_dp_rendezvous_need()

        self.draft_id_to_target_id = getattr(self.model, "draft_id_to_target_id", None)

        def model_wrapper(
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            token_indices_to_sample: torch.Tensor | None = None,
            inputs_embeds: torch.Tensor | None = None,
        ):
            if hidden_states.shape[-1] != self.hidden_size:
                # Region 3/0 folded in: `hidden_states` arrived at
                # num_aux * target_hidden width, so project it here instead of in
                # its own compiled graph. Loop iterations feed the drafter's own
                # output, which is already hidden_size wide, so the width check
                # picks the first pass without a second graph variant.
                hidden_states = self.model.combine_hidden_states(hidden_states)
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
            sample_hidden_states = last_hidden_states.view(-1, self.hidden_size)

            if token_indices_to_sample is not None:
                # Advanced indexing rather than `index_select` here, unlike the
                # rejection sampler's two gathers. `token_indices_to_sample`
                # arrives on CPU while `hidden_states` is on rbln:0, and
                # `index_select` requires both on one device -- dynamo fails the
                # trace with "Unhandled FakeTensor Device Propagation for
                # aten.index_select.default, found two different devices". A
                # `.to(device)` would fix the trace and add a host-to-device copy
                # every step, which is the opposite of the point. Advanced
                # indexing tolerates the CPU index, and this tensor is
                # hidden_size wide (3072) rather than vocabulary wide, so the
                # native-path argument that motivates the sampler-side change
                # carries much less here.
                hidden_states = hidden_states[token_indices_to_sample]
                sample_hidden_states = sample_hidden_states[token_indices_to_sample]

            if self.draft_id_to_target_id is not None:
                # NOTE(RBLN): upstream's `compute_logits` widens draft-vocab
                # logits to the target vocabulary by scattering into an `-inf`
                # row. Its index is input-independent, so the subgraph folds
                # into an anonymous constant that weight-free apply cannot
                # resolve by name; it then executes on placeholder indices and
                # that out-of-bounds write is the SIGSEGV in KV warmup. Stay in
                # draft-vocab space and map after the argmax instead
                # (`_to_target_token_ids`) -- no per-model patch needed.
                #
                # Narrowed like the eagle3 branch in `propose()`: only these two
                # carry `draft_id_to_target_id`, and the assert is what tells
                # mypy `logits_processor` is callable rather than a Tensor.
                assert isinstance(
                    self.model, (Eagle3LlamaForCausalLM, Eagle3DeepseekV2ForCausalLM)
                )
                logits = self.model.logits_processor(
                    self.model.lm_head, sample_hidden_states
                )
            else:
                logits = self.model.compute_logits(sample_hidden_states)

            # Reduce inside the region: traced here it is a device op, while an
            # eager `argmax` is a registered host fallback. int32 here too, so the
            # `.int()` in `loop_update` becomes a no-op rather than a round trip.
            # The draft->target map stays outside -- `_to_target_token_ids` needs
            # `d2t`, which is not a graph input.
            ids = torch.ops.rbln.argmax(logits)
            return hidden_states, ids.to(torch.int32)

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
        if self._fold_combine():
            # Compile the shape serving will actually hand in. `propose` replaces
            # the narrow buffer view with the wide aux states when the projection
            # is folded, so the warmup has to do the same or the first real step
            # recompiles.
            hidden_states = torch.zeros(
                (*input_ids.shape, self._aux_width()),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
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
                token_indices_to_sample=token_indices_to_sample_padded,
            )

        if self.num_speculative_tokens == 1:
            return

        common_attn_metadata.num_actual_tokens = num_reqs
        common_attn_metadata.max_query_len = 1
        common_attn_metadata.query_start_loc = self.arange_cpu[: num_reqs + 1]
        common_attn_metadata.query_start_loc_cpu = self.arange_cpu[: num_reqs + 1]
        common_attn_metadata.seq_lens += 1

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
                    token_indices_to_sample=None,
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

    # Conservative defaults: an instance that never ran the probe keeps the
    # collective.
    _draft_has_moe: bool = True
    _single_decode_bucket: bool = False

    def _probe_dp_rendezvous_need(self) -> None:
        """Decide whether the drafter's DP shape rendezvous can be skipped.

        The drafter calls `num_tokens_and_reqs_across_dp` twice per step. Each
        call is a CPU all_reduce that every DP rank must reach before any can
        proceed, so its cost is rank skew rather than collective bandwidth.
        Measured on MiniMax-M2.5 with DP4+EP, `max_num_seqs 1`: 1.5 ms/step on
        decode-only steps, 13.8 ms/step on steps where this rank prefills.

        Three values come out of the collective:

          num_tokens_across_dp, num_padded_tokens
              Reach the compiled model only through the forward context's
              `RBLNDPMetadata`, which is read exclusively by
              `model_executor/layers/fused_moe`. A draft model with no
              `RBLNFusedMoE` module cannot observe either value.
          num_reqs_padded
              Real: it sets the attention metadata's `batch_pad` and the input
              padding width. But on a prefill step it is fixed to `num_reqs`
              before the collective runs, and on a decode step it is
              `find_decode_batch_bucket(max num_reqs across ranks)`, which is
              constant when only one decode bucket exists.

        So the skip is sound exactly when the draft model carries no MoE layer
        and either this is a prefill step or there is a single decode bucket.
        Both are verified here rather than assumed -- a draft model that does
        carry MoE layers keeps the collective, and so does a configuration with
        more than one decode bucket.
        """
        # vllm-rbln dropped its `RBLNFusedMoE` subclass in the 0.24 bump
        # (c5228c5d), and upstream turned `FusedMoE` into a factory that composes
        # a router, `RoutedExperts` and a runner -- so the module to look for is
        # `RoutedExperts`, which is what holds the expert weights.
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

        moe_layers = [
            name
            for name, mod in self.model.named_modules()
            if isinstance(mod, RoutedExperts)
        ]
        self._draft_has_moe = bool(moe_layers)
        buckets = self.runner.bucketing_manager.decode_batch_buckets
        self._single_decode_bucket = len(buckets) == 1
        logger.info(
            "EAGLE3 drafter DP rendezvous: draft_moe_layers=%d "
            "decode_buckets=%s -> skip on decode=%s, on prefill=%s",
            len(moe_layers),
            buckets,
            self._skip_dp_rendezvous(False),
            self._skip_dp_rendezvous(True),
        )

    def _skip_dp_rendezvous(self, is_prefill: bool) -> bool:
        if self._draft_has_moe:
            return False
        return is_prefill or self._single_decode_bucket

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

        if self._skip_dp_rendezvous(is_prefill):
            # Same values, no collective. See `_probe_dp_rendezvous_need`.
            return (
                num_reqs_padded,
                self.max_num_tokens,
                torch.full((dp_size,), num_tokens, dtype=torch.int32),
            )

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
