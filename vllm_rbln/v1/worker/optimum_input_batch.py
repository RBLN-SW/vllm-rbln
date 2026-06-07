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

from typing import cast

import torch
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.utils import copy_slice
from vllm.v1.worker.gpu_input_batch import InputBatch


class RBLNInputBatch(InputBatch):
    """
    Input batch for RBLN sampler.
    To pad sampling metadata for RBLN sampler, provide bucket_sizes.
    """

    def __init__(self, *args, **kwargs):
        use_rbln_sampler = kwargs.pop("use_rbln_sampler")
        super().__init__(*args, **kwargs)
        if use_rbln_sampler:
            # Overwrite sampling_metadata with RBLN sampling metadata
            self.sampling_metadata = self._make_sampling_metadata_rbln(self.num_reqs)
            # Default top_k to vocab_size to guard
            # against runtime errors in top_k/top_p ops:
            # an unset top_k is still used as an index
            # in the fused kernel, so vocab_size
            # acts as "no filtering" while staying in a valid range.
            #
            # Refs:
            #   - upstream default: https://github.com/vllm-project/vllm/blob/01efc7ef781391e744ed08c3292817a773d654e6/vllm/v1/worker/gpu_input_batch.py#L348
            #   - failure site:    https://github.com/vllm-project/vllm/blob/01efc7ef781391e744ed08c3292817a773d654e6/vllm/v1/sample/ops/topk_topp_sampler.py#L151
            self.top_k.fill_(self.vocab_size)
            self.top_k_cpu_tensor.fill_(self.vocab_size)
            # Default top_p to 1.0
            self.top_p.fill_(1.0)
            self.top_p_cpu_tensor.fill_(1.0)
            # Default temperature to 1.0 to guard against NaN logits.
            #
            # Why: top_k / top_p applied to unscaled logits can produce NaNs,
            # which propagate into the sampled token ids as out-of-vocab values.
            # Those ids are later used as indices in torch.gather (e.g. for logprobs),
            # triggering an "index out of bounds" RuntimeError in the CPU kernel.
            self.temperature.fill_(1.0)
            self.temperature_cpu_tensor.fill_(1.0)
            # Default penalties to no-ops (frequency/presence=0, repetition=1).
            #
            # Why: apply_penalties does `logits -= penalty * mask` and
            # `logits = logits / repetition_penalty` over the full padded
            # batch. If a pad row's penalty is NaN (from torch.empty), then
            # NaN * 0 = NaN contaminates every position; if repetition is 0,
            # the division yields inf. Either way the pad row leaves
            # apply_penalties unusable for the rest of the sampler pipeline.
            self.frequency_penalties_cpu_tensor.fill_(0.0)
            self.presence_penalties_cpu_tensor.fill_(0.0)
            self.repetition_penalties_cpu_tensor.fill_(1.0)

            self.frequency_penalties.fill_(0.0)
            self.presence_penalties.fill_(0.0)
            self.repetition_penalties.fill_(1.0)

    def refresh_metadata_rbln(self, bucket_size: int):
        """Apply any batch updates to sampling metadata."""
        # NOTE(eunji.lee):
        # Pooling model doesn't use RBLN sampler
        if self.is_pooling_model:
            batch_changed = self.batch_update_builder.reset()
            if batch_changed:
                self.sampling_metadata = self._make_sampling_metadata()
            return

        # For non-pooling models - generate and apply logitsprocs update;
        # reset batch update tracking.
        # Update sampling metadata if batch state is changed.
        batch_update = self.batch_update_builder.get_and_reset(self.num_reqs)
        for logit_proc in self.logitsprocs.all:
            logit_proc.update_state(batch_update)
        if batch_update:
            self.sampling_metadata = self._make_sampling_metadata_rbln(bucket_size)

    def _make_sampling_metadata_rbln(self, bucket_size: int) -> SamplingMetadata:
        # NOTE(eunji.lee):
        # Use bucket_size instead of num_reqs
        # to pad sampling metadata for RBLN sampler.
        num_reqs = bucket_size

        if not self.all_greedy:
            temperature = copy_slice(
                self.temperature_cpu_tensor, self.temperature, num_reqs
            )
        else:
            temperature = None
        if not self.no_top_p:
            copy_slice(self.top_p_cpu_tensor, self.top_p, num_reqs)
        if not self.no_top_k:
            copy_slice(self.top_k_cpu_tensor, self.top_k, num_reqs)

        if not self.no_penalties:
            # Since syncing these tensors is expensive only copy them
            # if necessary i.e. if there are requests which require
            # penalties to be applied during sampling.
            copy_slice(
                self.frequency_penalties_cpu_tensor, self.frequency_penalties, num_reqs
            )
            copy_slice(
                self.presence_penalties_cpu_tensor, self.presence_penalties, num_reqs
            )
            copy_slice(
                self.repetition_penalties_cpu_tensor,
                self.repetition_penalties,
                num_reqs,
            )

        needs_prompt_token_ids = (
            not self.no_penalties
            or self.logits_processing_needs_token_ids[:num_reqs].any()
        )
        if needs_prompt_token_ids:
            # NOTE(0.18): _make_prompt_token_ids_tensor was renamed to
            # _make_prompt_token_ids_cpu_tensor in 0.19 and now returns a CPU
            # tensor; the caller is responsible for the device upload.
            # When 0.18 support is dropped, replace this block with a direct
            # call to _cpu_tensor.
            if hasattr(self, "_make_prompt_token_ids_cpu_tensor"):
                prompt_token_ids = self._make_prompt_token_ids_cpu_tensor().to(
                    device=self.device, non_blocking=True
                )
            else:
                prompt_token_ids = self._make_prompt_token_ids_tensor()
        else:
            prompt_token_ids = None

        # Only set output_token_ids if required by the current requests'
        # sampling parameters.
        needs_output_token_ids = (
            not self.no_penalties
            or bool(self.bad_words_token_ids)
            or self.logitsprocs_need_output_token_ids
        )
        output_token_ids = (
            cast(list[list[int]], self.req_output_token_ids)
            if needs_output_token_ids
            else []
        )

        allowed_token_ids_mask: torch.Tensor | None = None
        if not self.no_allowed_token_ids:
            assert self.allowed_token_ids_mask is not None
            copy_slice(
                self.allowed_token_ids_mask_cpu_tensor,
                self.allowed_token_ids_mask,
                num_reqs,
            )
            allowed_token_ids_mask = self.allowed_token_ids_mask[:num_reqs]

        return SamplingMetadata(
            temperature=temperature,
            all_greedy=self.all_greedy,
            all_random=self.all_random,
            top_p=None if self.no_top_p else self.top_p[:num_reqs],
            top_k=None if self.no_top_k else self.top_k[:num_reqs],
            generators=self.generators,
            max_num_logprobs=self.max_num_logprobs,
            prompt_token_ids=prompt_token_ids,
            frequency_penalties=self.frequency_penalties[:num_reqs],
            presence_penalties=self.presence_penalties[:num_reqs],
            repetition_penalties=self.repetition_penalties[:num_reqs],
            output_token_ids=output_token_ids,
            no_penalties=self.no_penalties,
            allowed_token_ids_mask=allowed_token_ids_mask,
            bad_words_token_ids=self.bad_words_token_ids,
            logitsprocs=self.logitsprocs,
        )
