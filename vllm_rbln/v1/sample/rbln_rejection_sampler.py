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

import inspect
from dataclasses import replace

import rebel

try:
    import torch.rbln

    has_torch_rbln = True
except ImportError:
    has_torch_rbln = False
import torch
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.sample.rejection_sampler import RejectionSampler
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

import vllm_rbln.rbln_envs as envs
from vllm_rbln.logger import init_logger
from vllm_rbln.v1.sample.rbln_sampler import random_sample

logger = init_logger(__name__)

PLACEHOLDER_TOKEN_ID = -1
GREEDY_TEMPERATURE = 0
GREEDY_EPS = 1e-3
# Maximum number of speculative draft tokens allowed per request in a single
# step. Bounded to [1, 32] by the rbln::rejection_sample NPU primitive.
MAX_SPEC_LEN = 32


@torch.library.custom_op("rbln::rejection_sample", mutates_args=())
def rejection_sample(
    draft_token_ids: torch.Tensor,
    target_probs: torch.Tensor,
    cu_num_draft_tokens: torch.Tensor,
    top_k: torch.Tensor | None,
    top_p: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference CPU implementation of the rbln::rejection_sample NPU primitive.

    Inputs (packed-then-padded layout):
      draft_token_ids     : (B*K,)   int32
      target_probs        : (B*K, V) f16 / f32  (one-hot for greedy, scaled
                                                 logits otherwise)
      cu_num_draft_tokens : (B,)     int  — cumulative end offsets, cu[-1] = N
      top_k               : (B,)     int  | None
      top_p               : (B,)     f32  | None

    Outputs (per-batch padded layout):
      out_tokens   : (B, K) int32  — recovered token at the first-reject slot.
                                     Other positions are filler; the caller
                                     masks them by num_accepted.
      num_accepted : (B,)   int32  — count of consecutive accepts before the
                                     first reject (== n_i when all accepted).

    Per-row sampling delegates to `apply_top_k_top_p`. Per-batch top_k / top_p
    is expanded to per-row via repeat_interleave so the existing
    apply_top_k_top_p (which expects (N,) k/p) can be reused without changes.
    """
    B = cu_num_draft_tokens.shape[0]
    K = draft_token_ids.shape[0] // B
    device = draft_token_ids.device

    # Expand per-batch top_k / top_p to per-row (B*K,).
    k_row = top_k.repeat_interleave(K) if top_k is not None else None
    p_row = top_p.repeat_interleave(K) if top_p is not None else None

    target_probs = apply_top_k_top_p(
        target_probs.to(torch.float32), k_row, p_row
    )  # (B*K, V)
    sampled = random_sample(target_probs, {})  # (B*K, )
    sampled = sampled.view(B, K).to(torch.int32)
    drafts = draft_token_ids.view(B, K).to(torch.int32)

    out_tokens = torch.zeros((B, K), dtype=torch.int32, device=device)
    num_accepted = torch.zeros((B,), dtype=torch.int32, device=device)

    cu = cu_num_draft_tokens.to(torch.int64).tolist()
    for i in range(B):
        start = 0 if i == 0 else cu[i - 1]
        n_i = cu[i] - start
        for j in range(n_i):
            if sampled[i, j].item() == drafts[i, j].item():
                num_accepted[i] += 1
            else:
                out_tokens[i, j] = sampled[i, j]
                break

    return out_tokens, num_accepted


@rejection_sample.register_fake
def rejection_sample_fake(
    draft_token_ids: torch.Tensor,
    target_probs: torch.Tensor,
    cu_num_draft_tokens: torch.Tensor,
    top_k: torch.Tensor | None,
    top_p: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FakeTensor body — only output shape/dtype matters for torch.compile."""
    B = cu_num_draft_tokens.shape[0]
    K = draft_token_ids.shape[0] // B
    device = draft_token_ids.device
    out_tokens = torch.empty((B, K), dtype=torch.int32, device=device)
    num_accepted = torch.empty((B,), dtype=torch.int32, device=device)
    return out_tokens, num_accepted


def rbln_random_sample(
    draft_token_ids: torch.Tensor,
    target_probs: torch.Tensor,
    cu_num_draft_tokens: torch.Tensor,
    top_k: torch.Tensor | None,
    top_p: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    output_tokens, acceptance_rate = torch.ops.rbln.rejection_sample(
        draft_token_ids,
        target_probs,
        cu_num_draft_tokens,
        top_k,
        top_p,
    )
    return output_tokens, acceptance_rate


# TODO(RBLN): Enable RBLNSampler for
# - apply_bad_words_with_drafts
# - apply_all_penalties
class RBLNRejectionSampler(RejectionSampler):
    def __init__(self, *args, **kwargs):
        seed = kwargs.pop("seed", None)
        compile_context = kwargs.pop("compile_context", None)
        super().__init__(*args, **kwargs)
        rebel.manual_seed(seed)
        options: dict = {}
        use_dt = envs.VLLM_RBLN_USE_DEVICE_TENSOR
        if compile_context is None:
            if "use_global_ctx" in inspect.signature(rebel.CompileContext).parameters:
                compile_context = rebel.CompileContext(use_global_ctx=True)
            else:
                compile_context = rebel.CompileContext()
        if not use_dt:
            options["compile_context"] = compile_context
        if envs.VLLM_RBLN_COMPILE_STRICT_MODE:
            options["mode"] = "strict"

        if has_torch_rbln or use_dt:
            options["tensor_parallel_size"] = 1
            if not use_dt:
                options["use_global_ctx"] = True
                options["global_device_id"] = 0
        self.compiled_rejection_sample = torch.compile(
            rbln_random_sample,
            dynamic=False,
            fullgraph=True,
            backend="rbln",
            options=options,
        )

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
        # target_probs = target_logits.softmax(dim=-1, dtype=torch.float32)
        target_probs = target_logits.to(torch.float32)

        output_token_ids = self.rejection_sample(
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
        self,
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
        assert draft_token_ids.is_contiguous()
        assert draft_probs is None or draft_probs.is_contiguous()
        assert target_probs.is_contiguous()
        assert bonus_token_ids.is_contiguous()
        assert target_probs.shape == (num_tokens, vocab_size)

        # Output buffer (batch space). Unwritten slots stay as PLACEHOLDER.
        output_token_ids = torch.full(
            (batch_size, max_spec_len + 1),
            PLACEHOLDER_TOKEN_ID,
            dtype=torch.int64,  # Consistent with SamplerOutput.sampled_token_ids.
        )

        # `active_mask` is in batch space: True for rows with any draft.
        active_mask = torch.tensor(
            [n > 0 for n in num_draft_tokens],
            device=output_token_ids.device,
            dtype=torch.bool,
        )  # [batch_size]

        # ------------------------------------------------------------------
        # 1) Build NPU primitive inputs (packed-then-padded layout).
        # NPU expects the first N = sum(num_draft_tokens) rows to be the
        # concat of valid drafts/probs across batches and the remaining
        # B*K - N rows to be tail padding (zeros). `draft_token_ids` and
        # `target_probs` come in already concatenated, so we just copy into
        # the front of the B*K buffer.
        # ------------------------------------------------------------------
        N = num_tokens  # = sum(num_draft_tokens)
        reshaped_draft_token_ids = torch.zeros(
            batch_size * max_spec_len, dtype=torch.int32
        )
        reshaped_target_probs = torch.zeros(
            batch_size * max_spec_len, vocab_size, dtype=torch.float16
        )
        reshaped_draft_token_ids[:N] = draft_token_ids.to(torch.int32)
        # FIXME float16
        reshaped_target_probs[:N] = target_probs.to(torch.float16)

        # Per-batch padded view of drafts for the scatter in section 3a. NPU's
        # input is packed-then-padded, but `output_token_ids` is per-batch
        # padded, so we materialize a (B, K) view that aligns row-by-row with
        # `selected_token_ids` and `output_token_ids`.
        draft_per_batch = torch.full(
            (batch_size, max_spec_len),
            PLACEHOLDER_TOKEN_ID,
            dtype=torch.int64,
            device=output_token_ids.device,
        )
        src_offset = 0
        for i, n in enumerate(num_draft_tokens):
            if n == 0:
                continue
            draft_per_batch[i, :n] = draft_token_ids[src_offset : src_offset + n].to(
                torch.int64
            )
            src_offset += n

        # NPU primitive expects cu_num_draft_tokens in int32; vLLM hands it
        # over as int64 (inherited from num_draft_tokens cumsum).
        cu_num_draft_tokens_i32 = cu_num_draft_tokens.to(torch.int32)

        # ------------------------------------------------------------------
        # 2) Call the NPU primitive.
        # Returns:
        #   selected_token_ids : (B, K) int — per-batch padded recovered tokens.
        #   num_accepted       : (B,)   int — per-batch number of accepted draft
        #                                     tokens (in [0, num_draft_tokens[i]]).
        # ------------------------------------------------------------------
        selected_token_ids, num_accepted = self.compiled_rejection_sample(
            reshaped_draft_token_ids,
            reshaped_target_probs,
            cu_num_draft_tokens_i32,
            sampling_metadata.top_k,
            sampling_metadata.top_p,
        )

        # ------------------------------------------------------------------
        # 3) Compose per-position output for the first K columns:
        #      j < num_accepted[i]          -> draft token (accepted as-is)
        #      j == num_accepted[i] (active) -> NPU-recovered token from target
        #      j > num_accepted[i]          -> PLACEHOLDER (left untouched)
        # ------------------------------------------------------------------
        num_accepted_per_batch = num_accepted.reshape(batch_size).to(torch.int64)
        positions = torch.arange(
            max_spec_len, device=output_token_ids.device
        ).unsqueeze(0)  # (1, K)
        all_accepted_active = (num_accepted_per_batch == max_spec_len) & active_mask

        # 3a) Accepted positions: write the draft token unchanged.
        accepted_pos_mask = positions < num_accepted_per_batch.unsqueeze(1)  # (B, K)
        output_token_ids[:, :max_spec_len] = torch.where(
            accepted_pos_mask,
            draft_per_batch,
            output_token_ids[:, :max_spec_len],
        )

        # 3b) First-reject position: write the NPU-recovered token.
        recovered_pos_mask = (
            (positions == num_accepted_per_batch.unsqueeze(1))
            & active_mask.unsqueeze(1)  # To skip inactive row (num_draft_tokens == 0)
            & ~all_accepted_active.unsqueeze(1)  # all-accept -> no recovery
        )  # (B, K)
        output_token_ids[:, :max_spec_len] = torch.where(
            recovered_pos_mask,
            selected_token_ids.to(torch.int64),
            output_token_ids[:, :max_spec_len],
        )

        # ------------------------------------------------------------------
        # 4) Scatter the bonus token into `output_token_ids`.
        # ------------------------------------------------------------------
        # [batch_size, 1] -> [batch_size]
        bonus = bonus_token_ids.squeeze(-1).to(torch.int64)

        # 4a) Fully-accepted active rows: emit the bonus token at the last col.
        output_token_ids[all_accepted_active, -1] = bonus[all_accepted_active]

        # 4b) Inactive rows (no drafts): only the bonus token at col 0.
        output_token_ids[~active_mask, 0] = bonus[~active_mask]
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
        # Make One-hot target distribution for the rejection sampler.
        _, max_idx = logits.max(dim=-1, keepdim=True)
        logits = torch.zeros_like(logits).scatter_(-1, max_idx, 1.0)
        return logits

    num_tokens = logits.shape[0]
    # NOTE(eunji.lee):
    # Upstream vLLM treats any temperature below _SAMPLING_EPS as greedy, sets it to 0,
    # and then overrides it to 1 right before the sampling op.
    # In rbln_rejection_sampler, random sampling is faster than the greedy path, so we
    # only treat temperature == GREEDY_TEMPERATURE (0) as greedy decoding.
    temperature = expand_batch_to_tokens(
        sampling_metadata.temperature,
        cu_num_draft_tokens,
        num_tokens,
        replace_from=GREEDY_TEMPERATURE,
        replace_to=GREEDY_EPS,
    )
    # NOTE(woosuk): Update `logits` in place to avoid allocating a new tensor.
    logits.div_(temperature.unsqueeze(-1))

    # NOTE(eunji.lee): top_k and top_p are applied together during rejection sampling.
    return logits


def expand_batch_to_tokens(
    x: torch.Tensor,  # [batch_size]
    cu_num_tokens: torch.Tensor,  # [batch_size]
    num_tokens: int,
    replace_from: int = 0,
    replace_to: int | float = 0,
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


# NOTE(RBLN): PyTorch native replacement of expand_kernel
def torch_expand_kernel(
    input: torch.Tensor,
    cu_num_tokens: torch.Tensor,
    num_tokens: int,
    replace_from: int = 0,
    replace_to: int | float = 0,
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
