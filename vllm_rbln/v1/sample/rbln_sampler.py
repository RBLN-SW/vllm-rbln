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
from collections.abc import Callable
from typing import Any

import numpy as np
import rebel
import torch
import torch.nn as nn
from vllm.config.model import LogprobsMode
from vllm.sampling_params import _SAMPLING_EPS
from vllm.v1.outputs import LogprobsTensors, SamplerOutput
from vllm.v1.sample.logits_processor.builtin import LogitBiasLogitsProcessor
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.bad_words import apply_bad_words
from vllm.v1.sample.ops.logprobs import batched_count_greater_than
from vllm.v1.sample.sampler import Sampler as VLLMSampler

import vllm_rbln.envs as envs
from vllm_rbln.compilation import compile, create_compile_context
from vllm_rbln.logger import init_logger
from vllm_rbln.platform import use_device_tensor
from vllm_rbln.v1.sample.ops.sparse_penalty import (
    SparsePenaltyState,
    apply_sparse_penalty,
    supports_vocab,
)
from vllm_rbln.v1.sample.ops.top_k_top_p import build_op_top_k_top_p

logger = init_logger(__name__)


# TODO(yunseong): move this to the runner
def _stage_into(owner: Any, out: torch.Tensor) -> torch.Tensor:
    """Snapshot `out` into a buffer belonging to `owner`, alternating two slots.

    `out` is the sampling graph's own output, which the runtime recycles on the
    next launch. Async holds the sampled tokens past the step boundary, so it
    gets this copy instead. Enqueued here, next to the launch, and not by the
    caller: a non_blocking copy is only safe while the source cannot have been
    recycled yet, and every device op between the two widens that window.
    """
    ring = owner._sampled_token_ring
    if (
        not ring
        or ring[0].shape != out.shape
        or ring[0].dtype != out.dtype
        or ring[0].device != out.device
    ):
        ring = [torch.empty_like(out) for _ in range(2)]
        owner._sampled_token_ring = ring
        owner._ring_slot = 0
    buf = ring[owner._ring_slot]
    owner._ring_slot ^= 1
    # non_blocking, or the copy waits on the sampling graph.
    buf.copy_(out, non_blocking=True)
    return buf


def rbln_top_k_top_p_sample(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
) -> torch.Tensor:
    """
    Implementation of RBLN top-k top-p sampling with temperature scaling.
    To avoid self parameter issues when torch.compile is used,
    we define this as a static method.
    """
    # Apply temperature.
    logits = logits.div_(temperature.to(logits.dtype).unsqueeze(dim=1))

    # Apply top-k top-p sampling using RBLN custom op.
    # It requires softmax prior to calling the op.
    probs = torch.nn.functional.softmax(logits, dim=-1)
    sampled = torch.ops.rbln.top_k_top_p(probs, k, p)
    return sampled


def rbln_greedy_sample(logits: torch.Tensor) -> torch.Tensor:
    """Implementation of RBLN greedy sampling.

    To avoid self parameter issues when torch.compile is used,
    we define this as a static method.
    """
    # NOTE(RBLN): argmax op is registered in the compiler
    return torch.ops.rbln.argmax(logits)


def _sparse_graph_fn(kind: str, rows: int, row_vocab: int, k: int) -> Callable:
    def penalize(logits, cidx, lanes):
        return apply_sparse_penalty(logits, cidx, lanes, rows, row_vocab, k)

    if kind == "greedy":
        return lambda logits, cidx, lanes: rbln_greedy_sample(
            penalize(logits, cidx, lanes)
        )
    if kind == "top_k_top_p":
        return lambda logits, cidx, lanes, temperature, top_k, top_p: (
            rbln_top_k_top_p_sample(
                penalize(logits, cidx, lanes), temperature, top_k, top_p
            )
        )
    return penalize


def compile_sampler(
    op: Callable[..., torch.Tensor],
    compile_context: rebel.CompileContext | None,
) -> Callable[..., torch.Tensor]:
    compile_context = (
        compile_context
        or create_compile_context(
            use_global_ctx=True,
        )
        if not use_device_tensor()
        else None
    )
    return compile(
        op,
        dynamic=False,
        fullgraph=True,
        compile_context=compile_context,
        num_devices=1 if use_device_tensor() else None,
        model_trace_method="export" if use_device_tensor() else "",
        mode="strict" if envs.VLLM_RBLN_COMPILE_STRICT_MODE else "",
        use_global_ctx=True if not use_device_tensor() else None,
        global_device_id=0 if not use_device_tensor() else None,
        # FIXME: Currently, sampler ops do not support caching.
        # Reusing seed buffer is not supported when the compiled sampler is loaded.
        use_cache=False,
    )


class RBLNTopKTopPSampler(nn.Module):
    def __init__(
        self,
        logprobs_mode: LogprobsMode = "raw_logprobs",
        compile_context: rebel.CompileContext | None = None,
    ):
        # TODO(rbln): Merge more ops to rbln context.
        #       Currently, we only have softmax in rbln context.
        super().__init__()
        self.logprobs_mode = logprobs_mode

        assert self.logprobs_mode not in ("processed_logits", "processed_logprobs"), (
            "RBLN Sampling does not support returning logits/logprobs"
        )

        self._compiled_rbln_topk_topp_sampler = compile_sampler(
            rbln_top_k_top_p_sample, compile_context
        )

    def forward(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        temperature: torch.Tensor,
        k: torch.Tensor | None,
        p: torch.Tensor | None,
        staging_owner: Any = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """More optimized implementation for top-k and top-p sampling.

        Unlike upstream `TopKTopPSampler`, `temperature` is applied here.
        """
        if generators:
            logger.debug_once(
                "RBLN Sampling does not support "
                "per-request generators. Ignoring generators."
            )

        out = self._compiled_rbln_topk_topp_sampler(logits, temperature, k, p)
        if staging_owner is not None:
            out = _stage_into(staging_owner, out)
        return out, None


class RBLNSampler(VLLMSampler):
    def __init__(
        self,
        logprobs_mode: LogprobsMode = "raw_logprobs",
        use_fp64_gumbel: bool = False,
        compile_context: rebel.CompileContext | None = None,
    ):
        super().__init__(logprobs_mode=logprobs_mode, use_fp64_gumbel=use_fp64_gumbel)

        compile_context = (
            compile_context
            or create_compile_context(
                use_global_ctx=True,
            )
            if not use_device_tensor()
            else None
        )
        if logprobs_mode in ("raw_logprobs", "raw_logits"):
            self.topk_topp_sampler = RBLNTopKTopPSampler(
                logprobs_mode=logprobs_mode, compile_context=compile_context
            )
        else:
            logger.warning_once(
                f"RBLN Sampling does not support logprobs_mode: {logprobs_mode}. "
                "Using the upstream sampler instead."
            )

        self._compiled_greedy_sample = compile_sampler(
            rbln_greedy_sample, compile_context
        )

        self._compile_context = compile_context
        self._sparse_state = (
            SparsePenaltyState()
            if envs.VLLM_RBLN_SPARSE_PENALTY
            and logprobs_mode in ("raw_logprobs", "raw_logits")
            else None
        )
        self._sparse_graphs: dict[tuple, Callable[..., torch.Tensor]] = {}
        self._sparse_pending: tuple | None = None

    def _sparse_graph(self, kind: str, rows: int, row_vocab: int, k: int):
        key = (kind, rows, row_vocab, k)
        fn = self._sparse_graphs.get(key)
        if fn is None:
            fn = self._sparse_graphs[key] = compile_sampler(
                _sparse_graph_fn(*key), self._compile_context
            )
        return fn

    def _use_sparse_penalty(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        predict_bonus_token: bool,
        input_batch: Any,
    ) -> bool:
        if self._sparse_state is None or input_batch is None or predict_bonus_token:
            return False
        if not supports_vocab(logits.shape[-1]):
            return False
        holder = sampling_metadata.thinking_budget_state_holder
        if holder is not None and holder.has_tracked_requests():
            return False
        has_bias = any(
            isinstance(p, LogitBiasLogitsProcessor) and p.biases
            for p in sampling_metadata.logitsprocs.non_argmax_invariant
        )
        return has_bias or not sampling_metadata.no_penalties

    def _apply_logits_processors_sparse(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        input_batch: Any,
    ) -> torch.Tensor:
        """Upstream `apply_logits_processors`, with logit bias and penalties
        deferred to the sampling graph."""
        if sampling_metadata.allowed_token_ids_mask is not None:
            logits.masked_fill_(sampling_metadata.allowed_token_ids_mask, float("-inf"))
        if sampling_metadata.bad_words_token_ids:
            apply_bad_words(
                logits,
                sampling_metadata.bad_words_token_ids,
                sampling_metadata.output_token_ids,
            )
        biases: dict[int, dict[int, float]] = {}
        for processor in sampling_metadata.logitsprocs.non_argmax_invariant:
            if isinstance(processor, LogitBiasLogitsProcessor):
                biases = processor.biases
                continue
            logits = processor.apply(logits)

        num_reqs = input_batch.num_reqs
        vocab_size = logits.shape[-1]
        if sampling_metadata.no_penalties:
            rep = np.ones(num_reqs, np.float32)
            freq = pres = np.zeros(num_reqs, np.float32)
            outputs: list[list[int]] = []
        else:
            rep = input_batch.repetition_penalties_cpu[:num_reqs]
            freq = input_batch.frequency_penalties_cpu[:num_reqs]
            pres = input_batch.presence_penalties_cpu[:num_reqs]
            outputs = sampling_metadata.output_token_ids

        def prompt(i: int) -> np.ndarray:
            return input_batch.token_ids_cpu[i, : input_batch.num_prompt_tokens[i]]

        assert self._sparse_state is not None
        rows, row_vocab, k, cidx, lanes = self._sparse_state.build(
            input_batch.req_ids[:num_reqs],
            prompt,
            outputs,
            rep,
            freq,
            pres,
            biases,
            vocab_size,
            logits.shape[0],
        )
        self._sparse_pending = (
            (rows, row_vocab, k),
            cidx.to(logits.device),
            lanes.to(logits.device),
        )
        return logits

    def warm_up_sparse_penalty(
        self, logits: torch.Tensor, sampling_metadata: SamplingMetadata
    ) -> None:
        """Compile the fused graphs for the first chunk bucket."""
        if self._sparse_state is None or not supports_vocab(logits.shape[-1]):
            return
        num_rows, vocab_size = logits.shape
        rows = 2 if num_rows == 1 else num_rows
        k = self._sparse_state.k
        lanes = torch.zeros(rows, 3, k, 64, dtype=torch.float16)
        lanes[:, 1] = 1.0
        self._sparse_pending = (
            (rows, vocab_size * num_rows // rows, k),
            torch.zeros(rows, k, dtype=torch.int32).to(logits.device),
            lanes.to(logits.device),
        )
        self.sample(logits, sampling_metadata)

    def greedy_sample(
        self, logits: torch.Tensor, staging_owner: Any = None
    ) -> torch.Tensor:
        out = self._compiled_greedy_sample(logits)
        return out if staging_owner is None else _stage_into(staging_owner, out)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        logprobs_mode_override: LogprobsMode | None = None,
        staging_owner: Any = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Sample logits based on sampling metadata.

        The various logits processing functions called in this method
        may update the logits tensor in-place.
        """

        logprobs_mode = logprobs_mode_override or self.logprobs_mode
        assert not (sampling_metadata.all_greedy and sampling_metadata.all_random)
        sparse, self._sparse_pending = self._sparse_pending, None
        if sparse is not None:
            wants_logprobs = logprobs_mode in (
                "processed_logits",
                "processed_logprobs",
            ) and (
                sampling_metadata.max_num_logprobs is not None
                or bool(sampling_metadata.logprob_token_ids)
            )
            uses_min_p = not sampling_metadata.all_greedy and any(
                getattr(p, "min_p_count", 1)
                for p in sampling_metadata.logitsprocs.argmax_invariant
            )
            if wants_logprobs or uses_min_p:
                # These need the penalized logits themselves.
                (rows, row_vocab, k), cidx, lanes = sparse
                logits = self._sparse_graph("penalize", rows, row_vocab, k)(
                    logits, cidx, lanes
                )
                sparse = None
        if sampling_metadata.all_greedy:
            # Upstream vLLM keeps this result to merge with the random one via
            # `torch.where`. vLLM RBLN has no merge step: a mixed batch sends its
            # greedy rows through the random-sampling path with top_k=1, so the op
            # can only draw their argmax.
            processed_logprobs = None
            if (
                sampling_metadata.max_num_logprobs is not None
                or sampling_metadata.logprob_token_ids
            ):
                if logprobs_mode == "processed_logits":
                    processed_logprobs = logits
                elif logprobs_mode == "processed_logprobs":
                    processed_logprobs = self.compute_logprobs(logits)
            if sparse is not None:
                (rows, row_vocab, k), cidx, lanes = sparse
                out = self._sparse_graph("greedy", rows, row_vocab, k)(
                    logits, cidx, lanes
                )
                if staging_owner is not None:
                    out = _stage_into(staging_owner, out)
                return out, processed_logprobs
            return self.greedy_sample(logits, staging_owner), processed_logprobs

        assert sampling_metadata.temperature is not None

        temperature = sampling_metadata.temperature
        if not sampling_metadata.all_random:
            temperature = torch.where(temperature < _SAMPLING_EPS, 1.0, temperature)

        argmax_invariant = sampling_metadata.logitsprocs.argmax_invariant
        # if argmax_invariant processors are active, apply temperature scaling
        # before applying them.
        if any(getattr(p, "min_p_count", 1) for p in argmax_invariant):
            # Divide in place, as upstream does: allocating a second logits-sized
            # tensor here costs more than the division itself. Rows past num_reqs of
            # the padded buffer must therefore carry temperature 1.0 -- see
            # RBLNInputBatch._make_sampling_metadata_rbln.
            logits = logits.div_(temperature.to(logits.dtype).unsqueeze(dim=1))
            temperature = torch.ones_like(temperature)

        # Apply logits processors that only apply to random sampling
        # (argmax invariant)
        for processor in argmax_invariant:
            logits = processor.apply(logits)

        k, p = build_op_top_k_top_p(
            sampling_metadata,
            logits.shape[0],
            logits.shape[-1],
            logits.device,
        )
        if sparse is not None:
            (rows, row_vocab, k_bucket), cidx, lanes = sparse
            out = self._sparse_graph("top_k_top_p", rows, row_vocab, k_bucket)(
                logits, cidx, lanes, temperature, k, p
            )
            if staging_owner is not None:
                out = _stage_into(staging_owner, out)
            return out, None

        # Apply temperature and top_k and/or top_p.
        random_sampled, processed_logprobs = self.topk_topp_sampler(
            logits,
            sampling_metadata.generators,
            temperature,
            k,
            p,
            staging_owner,
        )

        return random_sampled, processed_logprobs

    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        predict_bonus_token: bool = False,
        logprobs_mode_override: LogprobsMode | None = None,
        staging_owner: Any = None,
        input_batch: Any = None,
    ) -> SamplerOutput:
        logprobs_mode = logprobs_mode_override or self.logprobs_mode
        # NOTE(woosuk): Use the original logits (before any penalties or
        # temperature scaling) for the top-k logprobs.
        # This is different from the V0 sampler, which uses the logits that
        # is used for sampling (after penalties and temperature scaling).
        num_logprobs = sampling_metadata.max_num_logprobs
        raw_logprobs: torch.Tensor | None = None
        if num_logprobs is not None or sampling_metadata.logprob_token_ids:
            if logprobs_mode == "raw_logprobs":
                raw_logprobs = self.compute_logprobs(logits)
            elif logprobs_mode == "raw_logits":
                if logits.dtype == torch.float32:
                    raw_logprobs = logits.clone()
                else:
                    raw_logprobs = logits.to(torch.float32)

        # NOTE(eunji.lee) To reduce the copy overhead, we turned off type casting.
        # Use float32 for the logits.
        # logits = logits.to(torch.float32)

        self._sparse_pending = None
        if self._use_sparse_penalty(
            logits, sampling_metadata, predict_bonus_token, input_batch
        ):
            logits = self._apply_logits_processors_sparse(
                logits, sampling_metadata, input_batch
            )
        else:
            logits = self.apply_logits_processors(
                logits, sampling_metadata, predict_bonus_token
            )
        # Sample the next token.
        sampled, processed_logprobs = self.sample(
            logits, sampling_metadata, staging_owner=staging_owner
        )
        if processed_logprobs is not None:
            raw_logprobs = processed_logprobs

        logprob_token_ids_tensors = None
        if sampling_metadata.logprob_token_ids:
            assert raw_logprobs is not None
            logprob_token_ids_tensors = self.gather_specific_token_logprobs(
                raw_logprobs, sampling_metadata.logprob_token_ids, sampled.long()
            )

        if num_logprobs is None:
            logprobs_tensors = logprob_token_ids_tensors
        elif num_logprobs == -1:
            # Return the full unsorted and unranked logprobs.
            logprobs_tensors = LogprobsTensors(
                torch.empty(0), raw_logprobs, torch.empty(0)
            )
        else:
            # Gather the logprobs and ranks of the topk and sampled token.
            logprobs_tensors = self.gather_logprobs(
                raw_logprobs, num_logprobs, token_ids=sampled.long()
            )

        # If we have both num_logprobs and logprob_token_ids, prefer
        # logprob_token_ids as it's more specific
        if logprob_token_ids_tensors is not None and num_logprobs is not None:
            logprobs_tensors = logprob_token_ids_tensors

        # These are GPU tensors.
        sampler_output = SamplerOutput(
            # The sampled tokens are expanded to 2D tensor with shape
            # [num_requests, 1], where each row represents one generated
            # token per request.
            sampled_token_ids=sampled.unsqueeze(-1),
            logprobs_tensors=logprobs_tensors,
        )
        return sampler_output

    @staticmethod
    def gather_logprobs(
        logprobs: torch.Tensor,
        num_logprobs: int,
        token_ids: torch.Tensor,
    ) -> LogprobsTensors:
        """
        Gather logprobs for topk and sampled/prompt token.

        Args:
          logprobs: (num tokens) x (vocab) tensor
          num_logprobs: maximum number of logprobs to
                        retain per token
          token_ids: prompt tokens (if prompt logprobs)
                     or sampled tokens (if sampled
                     logprobs); 1D token ID tensor
                     with (num tokens) elements
                     Must be int64.

        Returns:
          Top-k int indices tensor, (num tokens) x (num_logprobs + 1)
          Top-k float logprobs tensor, (num tokens) x (num_logprobs + 1)
          Sampled token rank tensor, (num tokens)
        """
        assert token_ids.dtype == torch.int64
        # Find the topK values.
        topk_logprobs, topk_indices = torch.topk(logprobs, num_logprobs, dim=-1)

        # Get with the logprob of the prompt or sampled token.
        token_ids = token_ids.unsqueeze(-1)
        token_logprobs = logprobs.gather(-1, token_ids)

        # Compute the ranks of the actual token.
        token_ranks = batched_count_greater_than(logprobs, token_logprobs)

        # Concatenate together with the topk.
        indices = torch.cat((token_ids, topk_indices), dim=1)
        logprobs = torch.cat((token_logprobs, topk_logprobs), dim=1)

        # Use int32 to reduce the tensor size.
        indices = indices.to(torch.int32)

        return LogprobsTensors(indices, logprobs, token_ranks)


# Each config is designed to trigger exactly one dynamo specialization, so the
# number of compiled sampler graphs per batch size equals len(WARM_UP_CONFIGS).
# `dynamic=False` makes dynamo specialize on whether `k` and `p` are None, so
# each row below is its own graph:
#
#   compiled function          inputs                       warm-up config
#   rbln_greedy_sample         logits                       greedy
#   rbln_top_k_top_p_sample    logits, temperature          multinomial
#   rbln_top_k_top_p_sample    logits, temperature, p       topp
#   rbln_top_k_top_p_sample    logits, temperature, k       topk
#   rbln_top_k_top_p_sample    logits, temperature, k, p    topp_topk
WARM_UP_CONFIGS: list[dict[str, Any]] = [
    {
        "name": "greedy",
        "no_penalties": True,
        "all_greedy": True,
        "all_random": False,
        "temperature": 0.0,
    },
    {
        "name": "random",
        "no_penalties": True,
        "all_greedy": False,
        "all_random": True,
        "temperature": 0.5,
    },
    {
        "name": "topp",
        "no_penalties": True,
        "all_greedy": False,
        "all_random": True,
        "top_p": 0.9,
        "temperature": 0.5,
    },
    {
        "name": "topk",
        "no_penalties": True,
        "all_greedy": False,
        "all_random": True,
        "top_k": 1.0,
        "temperature": 0.5,
    },
    {
        "name": "topp_topk",
        "no_penalties": True,
        "all_greedy": False,
        "all_random": True,
        "top_p": 0.9,
        "top_k": 1.0,
        "temperature": 0.5,
    },
]
