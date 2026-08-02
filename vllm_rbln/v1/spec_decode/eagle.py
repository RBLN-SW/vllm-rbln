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
import os
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
from vllm_rbln.patches.attention import (
    reset_draft_unroll_index,
    set_draft_unroll_index,
)
from vllm_rbln.platform import USE_DEVICE_TENSOR
from vllm_rbln.utils import pad
from vllm_rbln.v1.attention.kv_cache_bindings import (
    attach_kv_cache_bindings,
    build_kv_cache_forward_context_kwargs,
)
from vllm_rbln.v1.spec_decode.utils import (
    DRAFT_ID_LOG_STEPS,
    PREFILL_SHAPE_LOG,
    FUSE_FIRST_FORWARD,
    FUSE_PREFILL,
    NARROW_LOGITS,
    SKIP_DP_RENDEZVOUS,
    UNROLL_DRAFTER,
    WARMUP_SKIP_FOLD,
    eagle_prepare_inputs_padded,
    eagle_prepare_next_token_padded,
)

if TYPE_CHECKING:
    from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

logger = init_logger(__name__)

# Pick the drafter's token inside the compiled graph instead of on the host.
#
# `torch.ops.rbln.argmax` lowers to contrib_top_k_top_p_sample(k=1, p=0) and runs
# on the device, but only when it is traced into a compiled region -- calling it
# eagerly lands on the host implementation. The target sampler already does it
# this way, via compile_sampler(rbln_greedy_sample).
#
# Measured on Qwen3-1.7B + AngelSlim/Qwen3-1.7B_eagle3, DP4, num_spec 3, paired
# A/B over two server instances, no profiler attached:
#
#   concurrency 1   22.72 -> 18.93 ms/step   (-3.79, -16.7%)   TPOT 11.57 -> 9.64
#   concurrency 4   24.61 -> 21.51 ms/step   (-3.10, -12.6%)
#   concurrency 8   41.04 -> 36.68 ms/step   (-4.36, -10.6%)
#
# Acceptance (0.3212 / 0.3432) and tokens-per-step (1.9635 / 2.0297) are
# bit-identical between arms, as they must be: this only moves where the
# reduction runs.
_DEVICE_ARGMAX = os.getenv("VLLM_RBLN_DRAFT_DEVICE_ARGMAX", "1") == "1"


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
        self._compiled_unrolled = None
        self._draft_id_logged = 0
        self._prefill_shape_logged = 0

    def _fold_combine(self, is_prefill: bool | None = None) -> bool:
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
        if not (FUSE_FIRST_FORWARD and self.method == "eagle3"):
            return False
        if is_prefill is None:
            is_prefill = self.runner.is_prefill
        return FUSE_PREFILL or not is_prefill

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
        """Token ids for this drafter step.

        Two independent flags feed this:

        `_DEVICE_ARGMAX`  the reduction already ran inside the compiled region,
                          so `out` is ids, not logits.
        `NARROW_LOGITS`   `compute_logits` left the logits at draft-vocabulary
                          width, so the id is a DRAFT id and still needs the
                          draft->target map. `target_ids` is `arange + d2t`, so
                          one gather finishes it. Skipping this map would emit
                          ids from the wrong vocabulary and acceptance would
                          collapse to ~zero.

        `index_select` rather than advanced indexing for the same reason as the
        other gathers here -- equivalent for a 1-D row selection, but only
        `index_select` takes the backend's native path.
        """
        if _DEVICE_ARGMAX:
            # `model_wrapper` already reduced and, under NARROW_LOGITS, already
            # mapped draft->target inside the region. Nothing left to do here.
            return out[:num_reqs]
        ids = out[:num_reqs].argmax(dim=-1)
        if NARROW_LOGITS:
            ids = self.model.target_ids.index_select(0, ids)
        return ids


    def _build_loop_metadata(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        positions: torch.Tensor,
        num_reqs: int,
        num_reqs_padded: int,
    ) -> tuple[dict[str, object], torch.Tensor]:
        """Advance one drafter iteration on the host and build its metadata.

        Mirrors the `loop_update` + `attn_meta` pair in `propose`, minus the
        buffer writes: the unrolled graph chains through graph values, so
        `self.input_ids` / `positions` / `hidden_states` are not used between
        iterations.

        The `_seq_lens_cpu` shadow has to move with the device tensor. The
        flash-attention builder reads the shadow, not `seq_lens`, so leaving it
        stale makes every iteration after the first attend at the wrong length --
        which does not crash and does not collapse acceptance.
        """
        positions = positions.view(-1) + 1
        exceeds = positions[:num_reqs] >= self.max_model_len
        common_attn_metadata.seq_lens += 1
        common_attn_metadata.seq_lens.masked_fill_(exceeds, 1)
        _slc = common_attn_metadata._seq_lens_cpu
        if _slc is not None:
            _slc += 1
            _slc.masked_fill_(exceeds.to(_slc.device), 1)

        per_layer: dict[str, object] = {}
        for attn_group in self.draft_attn_groups:
            am = attn_group.get_metadata_builder().build(
                common_attn_metadata=common_attn_metadata,
                positions=positions,
                is_prefill=False,
                batch_pad=num_reqs_padded,
            )
            attach_kv_cache_bindings(
                am,
                self.runner.kv_caches,
                self.runner.kv_cache_bases,
                self.runner.kv_cache_view_infos,
            )
            for layer_name in attn_group.layer_names:
                per_layer[layer_name] = am
        return per_layer, positions


    def _propose_unrolled(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        per_layer_attn_metadata: dict[str, object],
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        token_indices_to_sample_padded: torch.Tensor | None,
        target_positions: torch.Tensor,
        num_reqs: int,
        num_tokens: int,
        num_reqs_padded: int,
        num_padded_tokens: int | None,
        num_tokens_across_dp: torch.Tensor | None,
        num_rejected_tokens: torch.Tensor | None,
    ) -> torch.Tensor:
        """The drafter's whole chain in one compiled call.

        Replaces the first forward plus the `num_spec - 1` loop iterations, which
        are three graphs today (`1/3`, `1/1` twice) with host work between them.

        Everything the iterations need that differs is `seq_lens`, so this builds
        one metadata dict per iteration on the host -- the builder reads the
        `_seq_lens_cpu` shadow, which cannot move into the graph -- and hands the
        list to `set_forward_context`. `patched_get_attention_context` indexes it
        by `draft_unroll_index`, which the unrolled body advances; each copy is
        traced with a constant index, so the three iterations read three different
        tensors even though attention takes its metadata from the forward context
        rather than as an argument.

        `block_tables` is loop-invariant: `num_lookahead_tokens = num_spec` is
        passed to `allocate_slots`, so the slots for every draft position exist
        before the loop starts. The masks are None under
        VLLM_RBLN_FLASH_CAUSAL_ATTN and the sliding-window fields are None, and
        the KV write derives its slot from `seq_lens` inside the kernel.

        Verify with `equiv_check.sh`, not with acceptance.
        """
        metas = [per_layer_attn_metadata]

        # The loop iterations see a one-token query, and the rejected-token
        # adjustment lands before the first of them. Same order as `propose`.
        common_attn_metadata.num_actual_tokens = num_reqs
        common_attn_metadata.max_query_len = 1
        common_attn_metadata.query_start_loc = self.arange[: num_reqs + 1]
        common_attn_metadata.query_start_loc_cpu = self.arange[: num_reqs + 1].cpu()
        if num_rejected_tokens is not None:
            common_attn_metadata.seq_lens -= num_rejected_tokens
            _slc0 = common_attn_metadata._seq_lens_cpu
            if _slc0 is not None:
                _slc0 -= num_rejected_tokens.to(_slc0.device, _slc0.dtype)

        with record_function_or_nullcontext("drafter/unroll: batch_padding"):
            num_reqs_padded, num_padded_tokens, num_tokens_across_dp = (
                self._determine_draft_batch_padding(num_reqs, num_reqs, False)
            )

        assert token_indices_to_sample_padded is not None
        loop_positions = target_positions[
            token_indices_to_sample_padded.to(target_positions.device)
        ]
        with record_function_or_nullcontext("drafter/unroll: attn_meta"):
            for _ in range(self.num_speculative_tokens - 1):
                meta, loop_positions = self._build_loop_metadata(
                    common_attn_metadata, loop_positions, num_reqs, num_reqs_padded
                )
                metas.append(meta)

        with (
            record_function_or_nullcontext("drafter/unroll: forward"),
            set_forward_context(
                metas,
                self.vllm_config,
                num_tokens=num_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                num_padded_tokens=num_padded_tokens,
                **build_kv_cache_forward_context_kwargs(self.runner.kv_cache_bases),
            ),
        ):
            draft_token_ids = self._compiled_unrolled(
                input_ids,
                positions,
                hidden_states,
                token_indices_to_sample_padded,
            )

        if DRAFT_ID_LOG_STEPS and self._draft_id_logged < DRAFT_ID_LOG_STEPS:
            self._draft_id_logged += 1
            logger.info(
                "DRAFT_IDS step=%d %s",
                self._draft_id_logged,
                draft_token_ids.reshape(-1).tolist(),
            )
        return draft_token_ids

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
                with record_function_or_nullcontext("drafter/first: combine"):
                    target_hidden_states = self._combine_hidden_states(
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
                    # the folded projection inherits that property -- the
                    # docstring on `_combine_hidden_states` calls prefill token
                    # counts unbounded, but they are bounded by
                    # max_num_batched_tokens and `pad` short-circuits on the full
                    # chunks that dominate.
                    hidden_states = pad(flat, 0, self.hidden_states.shape[0]).view(
                        num_reqs, -1, w
                    )
                else:
                    hidden_states = pad(
                        flat[:num_tokens].view(num_reqs, -1, w), 0, num_reqs_padded
                    )
        inputs_embeds = None

        if (
            UNROLL_DRAFTER
            and self._compiled_unrolled is not None
            and not is_prefill
            and self.num_speculative_tokens > 1
        ):
            return self._propose_unrolled(
                common_attn_metadata=common_attn_metadata,
                per_layer_attn_metadata=per_layer_attn_metadata,
                input_ids=input_ids,
                positions=positions,
                hidden_states=hidden_states,
                token_indices_to_sample_padded=token_indices_to_sample_padded,
                target_positions=target_positions,
                num_reqs=num_reqs,
                num_tokens=num_tokens,
                num_reqs_padded=num_reqs_padded,
                num_padded_tokens=num_padded_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                num_rejected_tokens=num_rejected_tokens,
            )

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

        # Gathers plus the first argmax. Grouped so the `draft` remainder can be
        # attributed: argmax.out is a registered host fallback and measures
        # 0.195 ms per call standalone, essentially all of it the crossing.
        _post = record_function_or_nullcontext("drafter/first: sample")
        _post.__enter__()
        assert token_indices_to_sample_padded is not None
        positions = target_positions[
            token_indices_to_sample_padded.to(target_positions.device)
        ]

        # No gather of `hidden_states` here: `model_wrapper` already applies
        # `token_indices_to_sample` to it (see load_model). This override used to
        # gather again, which was correct against the pre-#821 wrapper -- that one
        # indexed only `sample_hidden_states` -- but #821 moved the fed-back
        # tensor's gather inside, so a second one indexes an already
        # (num_reqs_padded, hidden_size) tensor with token-space indices and
        # raises "index out of bounds for dimension 0 with size 1" on the first
        # decode.

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
        _adj = record_function_or_nullcontext("drafter/first: rejected_adjust")
        _adj.__enter__()
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

        _adj.__exit__(None, None, None)

        with record_function_or_nullcontext("drafter/first: batch_padding"):
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
                    token_indices_to_sample=None,
                )
            with record_function_or_nullcontext("drafter: sample"):
                draft_token_ids = self._draft_ids(logits, num_reqs)
                draft_token_ids_list.append(draft_token_ids)

        with record_function_or_nullcontext("drafter: stack"):
            # [batch_size, num_speculative_tokens]
            draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
        if DRAFT_ID_LOG_STEPS and self._draft_id_logged < DRAFT_ID_LOG_STEPS:
            self._draft_id_logged += 1
            logger.info(
                "DRAFT_IDS step=%d %s",
                self._draft_id_logged,
                draft_token_ids.reshape(-1).tolist(),
            )
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
        self._probe_dp_rendezvous_need()

        def model_wrapper(
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            token_indices_to_sample: torch.Tensor | None = None,
            inputs_embeds: torch.Tensor | None = None,
        ):
            if FUSE_FIRST_FORWARD and hidden_states.shape[-1] != self.hidden_size:
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

            logits = self.model.compute_logits(sample_hidden_states)

            if _DEVICE_ARGMAX:
                # Traced into this region, so the reduction is a device op.
                ids = torch.ops.rbln.argmax(logits)
                if NARROW_LOGITS:
                    # The draft->target map has to be inside the region too.
                    # Left on the host it both costs a gather and forces the
                    # result back to `target_ids`' int64, which would undo the
                    # int32 cast below and put the `.int()` in `loop_update`
                    # back on the critical path.
                    ids = self.model.target_ids.index_select(0, ids.reshape(-1))
                # int32 here rather than on the host: the op yields i64 and the
                # caller needs i32 for the compiled drafter's input_ids, so the
                # cast belongs inside the region. `.int()` in `loop_update` then
                # becomes a no-op instead of a device round trip.
                return hidden_states, ids.to(torch.int32)

            return hidden_states, logits

        def unrolled_wrapper(
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            token_indices_to_sample: torch.Tensor | None = None,
        ):
            """The drafter's whole chain in one region.

            The caller puts a list of per-iteration metadata dicts in the forward
            context; `set_draft_unroll_index` below picks which one each unrolled
            copy reads. The index is a Python int read during tracing, so it bakes
            a constant per copy and nothing survives into the graph.

            Iterations chain through graph values rather than through
            `self.input_ids` / `positions` / `hidden_states`, so those buffers are
            not written here. Nothing downstream reads them for this step -- the
            next step's `set_inputs_first_pass` overwrites them.
            """
            reset_draft_unroll_index()
            # Fold the aux projection here rather than letting `model_wrapper` do
            # it. That wrapper decides by width -- `hidden_states.shape[-1] !=
            # self.hidden_size` -- and inside one unrolled region only the first
            # copy is handed a wide tensor, so the three copies traced different
            # bodies and acceptance collapsed to exactly zero. Folding once up
            # front hands every copy a hidden_size-wide input, so the width test
            # is uniformly false and the three copies agree.
            #
            # Same computation, same place in the graph; only the branch moves.
            if FUSE_FIRST_FORWARD and hidden_states.shape[-1] != self.hidden_size:
                hidden_states = self.model.combine_hidden_states(hidden_states)
            h, ids = model_wrapper(
                input_ids, positions, hidden_states, token_indices_to_sample
            )
            flat = ids.reshape(-1)
            out = [flat]
            pos = positions.reshape(-1)[: flat.shape[0]]
            for step in range(1, self.num_speculative_tokens):
                set_draft_unroll_index(step)
                pos = pos + 1
                # `model_wrapper` returns hidden states as (num_tokens, hidden),
                # but the model expects the `[B, L, H]` layout `_preprocess`
                # produces. The loop iterations feed one token per request, so
                # that is (num_reqs, 1, hidden).
                nxt = out[-1].reshape(-1, 1)
                h, ids = model_wrapper(
                    nxt,
                    pos.reshape(-1, 1),
                    h.reshape(nxt.shape[0], -1, self.hidden_size),
                )
                out.append(ids.reshape(-1))
            reset_draft_unroll_index()
            return torch.stack(out, dim=1)

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
            if UNROLL_DRAFTER and self.method == "eagle3":
                self._compiled_unrolled = compile(
                    unrolled_wrapper,
                    dynamic=False,
                    fullgraph=True,
                    compile_context=self.runner.compile_context,
                    num_devices=envs.VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK,
                    model_trace_method="export" if USE_DEVICE_TENSOR else "",
                    process_group_dict=build_process_group_dict(),
                    guard_filter_fn=torch.compiler.keep_tensor_guards_unsafe,
                    mode="strict" if envs.VLLM_RBLN_COMPILE_STRICT_MODE else "",
                )
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
        if self._fold_combine(is_prefill) and not WARMUP_SKIP_FOLD:
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
        if PREFILL_SHAPE_LOG and is_prefill and self._prefill_shape_logged < PREFILL_SHAPE_LOG:
            self._prefill_shape_logged += 1
            ragged = num_reqs > 1 and num_input_tokens % num_reqs != 0
            logger.info(
                "PREFILL_SHAPE n=%d num_reqs=%d num_input_tokens=%d "
                "buffer_rows=%d ragged=%s%s",
                self._prefill_shape_logged,
                num_reqs,
                num_input_tokens,
                self.hidden_states.shape[0],
                ragged,
                "  <- view(num_reqs, -1) is wrong here" if ragged else "",
            )

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
        from vllm_rbln.model_executor.layers.fused_moe.layer import RBLNFusedMoE

        moe_layers = [
            name
            for name, mod in self.model.named_modules()
            if isinstance(mod, RBLNFusedMoE)
        ]
        self._draft_has_moe = bool(moe_layers)
        buckets = self.runner.bucketing_manager.decode_batch_buckets
        self._single_decode_bucket = len(buckets) == 1
        logger.info(
            "EAGLE3 drafter DP rendezvous: requested=%s draft_moe_layers=%d "
            "decode_buckets=%s -> skip on decode=%s, on prefill=%s",
            SKIP_DP_RENDEZVOUS,
            len(moe_layers),
            buckets,
            self._skip_dp_rendezvous(False),
            self._skip_dp_rendezvous(True),
        )

    def _skip_dp_rendezvous(self, is_prefill: bool) -> bool:
        if not SKIP_DP_RENDEZVOUS or self._draft_has_moe:
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
