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

# Copied from vllm.v1.sample.rejection_sampler: https://github.com/vllm-project/vllm/blob/v0.13.0/vllm/v1/sample/rejection_sampler.py
# Search for NOTE(RBLN) or TODO(RBLN) for details

from dataclasses import replace

import torch
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.sample.rejection_sampler import RejectionSampler, generate_uniform_probs
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

from vllm_rbln.logger import init_logger

logger = init_logger(__name__)

PLACEHOLDER_TOKEN_ID = -1
GREEDY_TEMPERATURE = 0
# Maximum number of speculative draft tokens allowed per request in a single
# step. This value is chosen to be large enough to handle typical use cases.
MAX_SPEC_LEN = 128


# TODO(RBLN): Enable RBLNSampler for
# - apply_bad_words_with_drafts
# - apply_all_penalties
# - apply_top_k_top_p
class RBLNRejectionSampler(RejectionSampler):
    # NOTE(RBLN): This class simply overrides forward by copying the upstream
    # implementation verbatim, so that it uses the functions defined in this
    # file. There are no behavioral changes.
    def forward(
        self,
        metadata: SpecDecodeMetadata,
        # [num_tokens, vocab_size]
        draft_probs: torch.Tensor | None,
        # [num_tokens + batch_size, vocab_size]
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        """
        Args:
            metadata:
                Metadata for spec decoding.
            draft_probs (Optional[torch.Tensor]):
                Probability distribution for the draft tokens. Shape is
                [num_tokens, vocab_size]. Can be None if probabilities are
                not provided, which is the case for ngram spec decode.
            logits (torch.Tensor):
                Target model's logits probability distribution.
                Shape is [num_tokens + batch_size, vocab_size]. Here,
                probabilities from different requests are flattened into a
                single tensor because this is the shape of the output logits.
                NOTE: `logits` can be updated in place to save memory.
            sampling_metadata (vllm.v1.sample.metadata.SamplingMetadata):
                Additional metadata needed for sampling, such as temperature,
                top-k/top-p parameters, or other relevant information.
        Returns:
            SamplerOutput:
                Contains the final output token IDs and their logprobs if
                requested.
        """
        assert metadata.max_spec_len <= MAX_SPEC_LEN

        bonus_logits_indices = metadata.bonus_logits_indices
        target_logits_indices = metadata.target_logits_indices

        # When indexing with a tensor (bonus_logits_indices), PyTorch
        # creates a new tensor with separate storage from the original
        # logits tensor. This means any in-place operations on bonus_logits
        # won't affect the original logits tensor.
        assert logits is not None
        bonus_logits = logits[bonus_logits_indices]
        bonus_sampler_output = self.sampler(
            logits=bonus_logits,
            sampling_metadata=replace(
                sampling_metadata,
                max_num_logprobs=-1,
            ),
            predict_bonus_token=True,
            # Override the logprobs mode to return logits because they are
            # needed later to compute the accepted token logprobs.
            logprobs_mode_override="processed_logits"
            if self.is_processed_logprobs_mode
            else "raw_logits",
        )
        bonus_token_ids = bonus_sampler_output.sampled_token_ids

        # Just like `bonus_logits`, `target_logits` is a new tensor with
        # separate storage from the original `logits` tensor. Therefore,
        # it is safe to update `target_logits` in place.
        raw_target_logits = logits[target_logits_indices]
        # Use float32 for the target_logits.
        raw_target_logits = raw_target_logits.to(torch.float32)
        target_logits = self.apply_logits_processors(
            raw_target_logits, sampling_metadata, metadata
        )
        # [num_tokens, vocab_size]
        # NOTE(woosuk): `target_logits` can be updated in place inside the
        # `apply_sampling_constraints` function.
        target_logits = apply_sampling_constraints(
            target_logits,
            metadata.cu_num_draft_tokens,
            sampling_metadata,
        )
        # Compute probability distribution from target logits.
        target_probs = target_logits.softmax(dim=-1, dtype=torch.float32)

        output_token_ids = rejection_sample(
            metadata.draft_token_ids,
            metadata.num_draft_tokens,
            metadata.max_spec_len,
            metadata.cu_num_draft_tokens,
            draft_probs,
            target_probs,
            bonus_token_ids,
            sampling_metadata,
        )

        logprobs_tensors = None
        if sampling_metadata.max_num_logprobs is not None:
            logprobs_tensors = self._get_logprobs_tensors(
                sampling_metadata.max_num_logprobs,
                metadata,
                logits,
                target_logits if self.is_processed_logprobs_mode else raw_target_logits,
                bonus_sampler_output.logprobs_tensors.logprobs,
                output_token_ids,
            )

        return SamplerOutput(
            sampled_token_ids=output_token_ids,
            logprobs_tensors=logprobs_tensors,
        )


def rejection_sample(
    # [num_tokens]
    draft_token_ids: torch.Tensor,
    # [batch_size]
    num_draft_tokens: list[int],
    max_spec_len: int,
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor,
    # [num_tokens, vocab_size]
    draft_probs: torch.Tensor | None,
    # [num_tokens, vocab_size]
    target_probs: torch.Tensor,
    # [batch_size, 1]
    bonus_token_ids: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    assert draft_token_ids.ndim == 1
    assert draft_probs is None or draft_probs.ndim == 2
    assert cu_num_draft_tokens.ndim == 1
    assert target_probs.ndim == 2

    batch_size = len(num_draft_tokens)
    num_tokens = draft_token_ids.shape[0]
    vocab_size = target_probs.shape[-1]
    device = target_probs.device
    assert draft_token_ids.is_contiguous()
    assert draft_probs is None or draft_probs.is_contiguous()
    assert target_probs.is_contiguous()
    assert bonus_token_ids.is_contiguous()
    assert target_probs.shape == (num_tokens, vocab_size)

    # Create output buffer.
    output_token_ids = torch.full(
        (batch_size, max_spec_len + 1),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,  # Consistent with SamplerOutput.sampled_token_ids.
        device=device,
    )

    if sampling_metadata.all_greedy:
        is_greedy = None
    else:
        is_greedy = sampling_metadata.temperature == GREEDY_TEMPERATURE
    if not sampling_metadata.all_random:
        # Rejection sampling for greedy sampling requests.
        target_argmax = target_probs.argmax(dim=-1)

        # NOTE(RBLN): Call torch_rejection_greedy_sample_kernel instead of
        # rejection_greedy_sample_kernel
        torch_rejection_greedy_sample_kernel(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            target_argmax,
            bonus_token_ids,
            is_greedy,
            batch_size,
            device,
        )
        if sampling_metadata.all_greedy:
            return output_token_ids

    # Generate uniform probabilities for rejection sampling.
    # [num_tokens]
    uniform_probs = generate_uniform_probs(
        num_tokens,
        num_draft_tokens,
        sampling_metadata.generators,
        device,
    )

    # Sample recovered tokens for each position.
    # [num_tokens]
    recovered_token_ids = sample_recovered_tokens(
        max_spec_len,
        num_draft_tokens,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        sampling_metadata,
        device,
    )

    # NOTE(RBLN): Call torch_rejection_random_sample_kernel instead of
    # rejection_random_sample_kernel
    torch_rejection_random_sample_kernel(
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        bonus_token_ids,
        recovered_token_ids,
        uniform_probs,
        is_greedy,
        batch_size,
        device,
    )

    return output_token_ids


# NOTE(RBLN): This function was copied without modification to replace
# expand_batch_to_tokens it calls with the PyTorch native implementations
# defined in this file.
def apply_sampling_constraints(
    logits: torch.Tensor,  # [num_tokens, vocab_size]
    cu_num_draft_tokens: torch.Tensor,  # [batch_size]
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    """Process logits based on sampling metadata.

    This function applies temperature scaling to the logits,
    as well as top-k and top-p. For greedy decoding, it returns
    the original logits.

    Args:
        logits: Input logits tensor to be processed.
        cu_num_draft_tokens: Cumulative number of draft tokens.
        sampling_metadata: Metadata containing sampling parameters such as
            temperature and whether greedy sampling is used.

    Returns:
        torch.Tensor: Processed logits if non-greedy sampling is used,
        otherwise returns the original logits.
    """
    assert logits.ndim == 2
    assert cu_num_draft_tokens.ndim == 1
    if sampling_metadata.all_greedy:
        return logits

    num_tokens = logits.shape[0]
    temperature = expand_batch_to_tokens(
        sampling_metadata.temperature,
        cu_num_draft_tokens,
        num_tokens,
        replace_from=GREEDY_TEMPERATURE,
        replace_to=1,
    )
    # NOTE(woosuk): Update `logits` in place to avoid allocating a new tensor.
    logits.div_(temperature.unsqueeze(-1))

    # Get expanded top_k and top_p tensors.
    top_k = None
    if sampling_metadata.top_k is not None:
        top_k = expand_batch_to_tokens(
            sampling_metadata.top_k,
            cu_num_draft_tokens,
            num_tokens,
        )
    top_p = None
    if sampling_metadata.top_p is not None:
        top_p = expand_batch_to_tokens(
            sampling_metadata.top_p,
            cu_num_draft_tokens,
            num_tokens,
        )

    # NOTE(woosuk): `apply_top_k_top_p` uses sorting to calculate the mask,
    # which is slow for large vocab sizes. This may cause performance issues.
    return apply_top_k_top_p(logits, top_k, top_p)


def expand_batch_to_tokens(
    x: torch.Tensor,  # [batch_size]
    cu_num_tokens: torch.Tensor,  # [batch_size]
    num_tokens: int,
    replace_from: int = 0,
    replace_to: int = 0,
) -> torch.Tensor:
    """Expand [batch_size] tensor to [num_tokens] tensor based on the number of
    tokens per batch in cu_num_tokens.

    For example, if x = [a, b, c] and cu_num_tokens = [2, 5, 6], then
    num_tokens = 6, and expanded_x = [a, a, b, b, b, c].

    Args:
        x: [batch_size] tensor to expand.
        cu_num_tokens: [batch_size] tensor containing the cumulative number of
            tokens per batch. Each element represents the total number of
            tokens up to and including that batch.
        num_tokens: Total number of tokens.
        replace_from: int = 0
            Value to be replaced if it is found in x.
        replace_to: int = 0
            Value to replace with when replace_from is found.
    Returns:
        expanded_x: [num_tokens] tensor.
    """
    batch_size = x.shape[0]
    assert cu_num_tokens.shape[0] == batch_size
    # NOTE(RBLN): Call torch_expand_kernel instead of expand_kernel
    expanded_x = torch_expand_kernel(
        x, cu_num_tokens, num_tokens, replace_from, replace_to
    )
    return expanded_x


# NOTE(RBLN): Note that max_spec_len is not used, but kept to match with the
# upstream code and prevent confusions.
def sample_recovered_tokens(
    max_spec_len: int,
    num_draft_tokens: list[int],
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor,
    # [num_tokens]
    draft_token_ids: torch.Tensor,
    # [num_tokens, vocab_size]
    draft_probs: torch.Tensor | None,
    # [num_tokens, vocab_size]
    target_probs: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    device: torch.device,
) -> torch.Tensor:
    # NOTE(woosuk): Create only one distribution for each request.
    batch_size = len(num_draft_tokens)
    vocab_size = target_probs.shape[-1]
    q = torch.empty(
        (batch_size, vocab_size),
        dtype=torch.float32,
        device=device,
    )
    q.exponential_()
    for i, generator in sampling_metadata.generators.items():
        # Do not generate random numbers for requests with no draft tokens.
        # This can be important for reproducibility.
        if num_draft_tokens[i] > 0:
            q[i].exponential_(generator=generator)

    # NOTE(RBLN): Call torch_sample_recovered_tokens_kernel instead of
    # sample_recovered_tokens_kernel
    recovered_token_ids = torch_sample_recovered_tokens_kernel(
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        q,
        batch_size,
        device,
    )
    return recovered_token_ids


# NOTE(RBLN): PyTorch native replacement of rejection_greedy_sample_kernel
def torch_rejection_greedy_sample_kernel(
    output_token_ids: torch.Tensor,
    cu_num_draft_tokens: torch.Tensor,
    draft_token_ids: torch.Tensor,
    target_argmax: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    is_greedy: torch.Tensor | None,
    batch_size: int,
    device: torch.device,
) -> None:
    if is_greedy is None:
        is_greedy_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
    else:
        is_greedy_mask = is_greedy.to(device=device, dtype=torch.bool)

    cu = cu_num_draft_tokens.to(device=device, dtype=torch.int64)
    start = torch.zeros_like(cu)
    start[1:] = cu[:-1]
    end = cu
    lens = (end - start).to(torch.int64)

    for req_idx in range(batch_size):
        if not bool(is_greedy_mask[req_idx]):
            continue

        n = int(lens[req_idx].item())

        if n == 0:
            output_token_ids[req_idx, 0] = bonus_token_ids[req_idx].to(torch.int32)
            continue

        s = int(start[req_idx].item())
        e = s + n

        d = draft_token_ids[s:e]
        t = target_argmax[s:e]

        mismatch = d != t
        if mismatch.any():
            k = int(mismatch.to(torch.int64).argmax().item())
            out_len = k + 1
            output_token_ids[req_idx, :out_len] = t[:out_len].to(torch.int32)
        else:
            output_token_ids[req_idx, :n] = t.to(torch.int32)
            output_token_ids[req_idx, n] = bonus_token_ids[req_idx].to(torch.int32)


# NOTE(RBLN): PyTorch native replacement of rejection_random_sample_kernel
def torch_rejection_random_sample_kernel(
    output_token_ids: torch.Tensor,
    cu_num_draft_tokens: torch.Tensor,
    draft_token_ids: torch.Tensor,
    draft_probs: torch.Tensor | None,
    target_probs: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    recovered_token_ids: torch.Tensor,
    uniform_probs: torch.Tensor,
    is_greedy: torch.Tensor | None,
    batch_size: int,
    device: torch.device,
) -> None:
    if is_greedy is None:
        is_greedy_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    else:
        is_greedy_mask = is_greedy.to(device=device, dtype=torch.bool)

    cu = cu_num_draft_tokens.to(device=device, dtype=torch.int64)
    start = torch.zeros_like(cu)
    start[1:] = cu[:-1]
    end = cu
    lens = (end - start).to(torch.int64)

    for req_idx in range(batch_size):
        if bool(is_greedy_mask[req_idx]):
            continue

        n = int(lens[req_idx].item())

        if n == 0:
            output_token_ids[req_idx, 0] = bonus_token_ids[req_idx].to(torch.int32)
            continue

        s = int(start[req_idx].item())
        e = s + n

        d_ids = draft_token_ids[s:e].to(torch.int64)
        u = uniform_probs[s:e].to(torch.float64)

        t_prob = (
            target_probs[s:e].gather(1, d_ids.unsqueeze(1)).squeeze(1).to(torch.float64)
        )

        if draft_probs is None:
            accept = t_prob >= u
        else:
            d_prob = (
                draft_probs[s:e]
                .gather(1, d_ids.unsqueeze(1))
                .squeeze(1)
                .to(torch.float64)
            )
            accept = (d_prob > 0) & ((t_prob / d_prob) >= u)

        if (~accept).any():
            k = int((~accept).to(torch.int64).argmax().item())
            if k > 0:
                output_token_ids[req_idx, :k] = draft_token_ids[s : s + k].to(
                    torch.int32
                )
            output_token_ids[req_idx, k] = recovered_token_ids[s + k].to(torch.int32)
        else:
            output_token_ids[req_idx, :n] = draft_token_ids[s:e].to(torch.int32)
            output_token_ids[req_idx, n] = bonus_token_ids[req_idx].to(torch.int32)


# NOTE(RBLN): PyTorch native replacement of expand_kernel
def torch_expand_kernel(
    input: torch.Tensor,
    cu_num_tokens: torch.Tensor,
    num_tokens: int,
    replace_from: int = 0,
    replace_to: int = 0,
) -> torch.Tensor:
    prev = torch.zeros_like(cu_num_tokens)
    prev[1:] = cu_num_tokens[:-1]
    counts = (cu_num_tokens - prev).to(torch.int64)

    expanded_x = input.repeat_interleave(counts)

    if replace_from != replace_to:
        expanded_x = torch.where(
            expanded_x == replace_from,
            expanded_x.new_tensor(replace_to),
            expanded_x,
        )

    if expanded_x.numel() != num_tokens:
        if expanded_x.numel() > num_tokens:
            expanded_x = expanded_x[:num_tokens]
        else:
            pad = expanded_x.new_full((num_tokens - expanded_x.numel(),), replace_to)
            expanded_x = torch.cat([expanded_x, pad], dim=0)

    return expanded_x


# NOTE(RBLN): PyTorch native replacement of sample_recovered_tokens_kernel
def torch_sample_recovered_tokens_kernel(
    cu_num_draft_tokens: torch.Tensor,
    draft_token_ids: torch.Tensor,
    draft_probs: torch.Tensor | None,
    target_probs: torch.Tensor,
    q: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    recovered_token_ids = torch.empty_like(draft_token_ids)

    cu = cu_num_draft_tokens.to(device=device, dtype=torch.int64)
    start = torch.zeros_like(cu)
    start[1:] = cu[:-1]
    end = cu
    lens = (end - start).to(torch.int64)

    for req_idx in range(batch_size):
        n = int(lens[req_idx].item())
        if n <= 0:
            continue
        s = int(start[req_idx].item())
        e = s + n

        q_req = q[req_idx].to(torch.float32)

        if draft_probs is None:
            prob = target_probs[s:e].to(torch.float32)
            d_ids = draft_token_ids[s:e].to(torch.int64)
            prob = prob.clone()
            prob.scatter_(1, d_ids.unsqueeze(1), 0.0)
        else:
            prob = torch.maximum(
                target_probs[s:e].to(torch.float32)
                - draft_probs[s:e].to(torch.float32),
                torch.zeros((), device=device, dtype=torch.float32),
            )

        scores = prob / q_req.unsqueeze(0)
        recovered_token_ids[s:e] = scores.argmax(dim=-1).to(recovered_token_ids.dtype)

    return recovered_token_ids
