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
# isort: off
import torch
import torch.nn as nn
from vllm_rbln.logger import init_logger
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler as VLLMSampler
import rebel
from vllm.config.model import LogprobsMode
from vllm.v1.outputs import LogprobsTensors, SamplerOutput
from vllm_rbln.v1.sample.ops.penalties import (
    apply_all_penalties as rbln_apply_all_penalties,
)
import vllm_rbln.rbln_envs as envs

logger = init_logger(__name__)

_SAMPLING_EPS = 1e-5


def random_sample(
    probs: torch.Tensor,
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    """Randomly sample from the probabilities.

    We use this function instead of torch.multinomial because torch.multinomial
    causes CPU-GPU synchronization.
    """
    q = torch.empty_like(probs)
    # NOTE(woosuk): To batch-process the requests without their own seeds,
    # which is the common case, we first assume that every request does
    # not have its own seed. Then, we overwrite the values for the requests
    # that have their own seeds.
    if len(generators) != probs.shape[0]:
        q.exponential_()
    if generators:
        # TODO(woosuk): This can be slow because we handle each request
        # one by one. Optimize this.
        for i, generator in generators.items():
            q[i].exponential_(generator=generator)
    return probs.div_(q).argmax(dim=-1).view(-1)


def apply_top_k_top_p(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
) -> torch.Tensor:
    """
    Mock implementation of `top_k_top_p`
    used for torch ops registration.
    This function currently performs standard top-p, top-k (nucleus)
    sampling that includes sorting the probabilities.
    It serves as a placeholder implementation — in the actual version,
    a dual-pivot algorithm is implemented in rebel and
    it will be used to avoid the sorting step and improve efficiency.
    """
    # NOTE This function is used when falling back to eager execution
    # due to RBLN compilation failure.
    # If both k and p are None, return the argmax (greedy sampling).
    if k is None and p is None:
        return logits.argmax(dim=-1).view(-1)

    logits_sort, logits_idx = logits.sort(dim=-1, descending=False, stable=True)
    if k is not None:
        # Apply top-k.
        top_k_mask = logits_sort.size(1) - k.to(torch.long)  # shape: B
        # Get all the top_k values.
        top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
        top_k_mask = logits_sort < top_k_mask
        logits_sort.masked_fill_(top_k_mask, -float("inf"))

    if p is not None:
        # Apply top-p.
        probs_sort = logits_sort.softmax(dim=-1)
        probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
        top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
        # at least one
        top_p_mask[:, -1] = False
        logits_sort.masked_fill_(top_p_mask, -float("inf"))

    # Re-sort the probabilities.
    logits = logits_sort.scatter(dim=-1, index=logits_idx, src=logits_sort)
    # Select the sampled token.
    return random_sample(logits, {})


@torch.library.custom_op("rbln::top_k_top_p", mutates_args=())
def top_k_top_p(
    logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None
) -> torch.Tensor:
    return apply_top_k_top_p(logits, k, p)


@top_k_top_p.register_fake
def top_k_top_p_fake(
    logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None
) -> torch.Tensor:
    return apply_top_k_top_p(logits, k, p)


def rbln_top_k_top_p_sample(
    logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None
) -> torch.Tensor:
    """
    Implementation of RBLN top-k top-p sampling.
    To avoid self parameter issues when torch.compile is used,
    we define this as a static method.
    """
    # Apply top-k top-p sampling using RBLN custom op.
    # It requires softmax prior to calling the op.
    probs = torch.nn.functional.softmax(logits, dim=-1)
    sampled = torch.ops.rbln.top_k_top_p(probs, k, p)
    return sampled


class RBLNTopKTopPSampler(nn.Module):
    def __init__(
        self,
        logprobs_mode: LogprobsMode = "raw_logprobs",
        seed: int = 42,
        compile_context: rebel.CompileContext = None,
    ):
        # TODO(rbln): Merge more ops to rbln context.
        #       Currently, we only have softmax in rbln context.
        super().__init__()
        self.logprobs_mode = logprobs_mode

        assert self.logprobs_mode not in ("processed_logits", "processed_logprobs"), (
            "RBLN Sampling does not support returning logits/logprobs"
        )

        rebel.manual_seed(seed)
        options = {
            "compile_context": compile_context
            if compile_context
            else rebel.CompileContext()
        }
        if envs.VLLM_RBLN_COMPILE_STRICT_MODE:
            options["mode"] = "strict"
        self._compiled_rbln_topk_topp_sampler = torch.compile(
            rbln_top_k_top_p_sample,
            dynamic=False,
            fullgraph=True,
            backend="rbln",
            options=options,
        )
        self.forward = self.forward_rbln

    @torch.compiler.disable
    def top_k_top_p_sample(
        self, logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None
    ) -> torch.Tensor:
        return self._compiled_rbln_topk_topp_sampler(logits, k, p)

    def forward_rbln(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """More optimized implementation for top-k and top-p sampling."""
        if generators:
            logger.debug_once(
                "RBLN Sampling does not support "
                "per-request generators. Ignoring generators."
            )

        return self.top_k_top_p_sample(logits, k, p), None


class RBLNSampler(VLLMSampler):
    def __init__(
        self,
        logprobs_mode: LogprobsMode = "raw_logprobs",
        seed: int = 42,
        compile_context: rebel.CompileContext = None,
    ):
        super().__init__()
        if logprobs_mode in ("raw_logprobs", "raw_logits"):
            self.topk_topp_sampler = RBLNTopKTopPSampler(
                logprobs_mode=logprobs_mode, seed=seed, compile_context=compile_context
            )
        else:
            logger.warning_once(
                f"RBLN Sampling does not support logprobs_mode: {logprobs_mode}. "
                "Using native sampler instead."
            )

    def apply_penalties(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        output_token_ids: list[list[int]],
    ) -> torch.Tensor:
        if not sampling_metadata.no_penalties:
            assert sampling_metadata.prompt_token_ids is not None
            logits = rbln_apply_all_penalties(
                logits,
                sampling_metadata.prompt_token_ids,
                sampling_metadata.presence_penalties,
                sampling_metadata.frequency_penalties,
                sampling_metadata.repetition_penalties,
                output_token_ids,
            )
        return logits

    def greedy_sample(self, logits: torch.Tensor) -> torch.Tensor:
        # NOTE:
        # If there are any parameters for top-k or top-p,
        # this function works as a greedy sampler.
        sampled, _ = self.topk_topp_sampler(logits, dict(), None, None)
        return sampled

    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        predict_bonus_token: bool = False,
        logprobs_mode_override: LogprobsMode | None = None,
    ) -> SamplerOutput:
        logprobs_mode = logprobs_mode_override or self.logprobs_mode
        # NOTE(woosuk): Use the original logits (before any penalties or
        # temperature scaling) for the top-k logprobs.
        # This is different from the V0 sampler, which uses the logits that
        # is used for sampling (after penalties and temperature scaling).
        num_logprobs = sampling_metadata.max_num_logprobs
        if num_logprobs is not None:
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

        logits = self.apply_logits_processors(
            logits, sampling_metadata, predict_bonus_token
        )
        # Sample the next token.
        sampled, processed_logprobs = self.sample(logits, sampling_metadata)
        if processed_logprobs is not None:
            raw_logprobs = processed_logprobs
        # Convert sampled token ids to int64 (long) type to ensure compatibility
        # with subsequent operations that may use these values as indices.
        # This conversion is necessary because FlashInfer sampling operations
        # return int32 (while PyTorch argmax and topk return int64).
        sampled = sampled.long()

        if num_logprobs is None:
            logprobs_tensors = None
        elif num_logprobs == -1:
            # Return the full unsorted and unranked logprobs.
            logprobs_tensors = LogprobsTensors(
                torch.empty(0), raw_logprobs, torch.empty(0)
            )
        else:
            # Gather the logprobs and ranks of the topk and sampled token.
            logprobs_tensors = self.gather_logprobs(
                raw_logprobs, num_logprobs, token_ids=sampled
            )

        # Use int32 to reduce the tensor size.
        sampled = sampled.to(torch.int32)

        # These are GPU tensors.
        sampler_output = SamplerOutput(
            # The sampled tokens are expanded to 2D tensor with shape
            # [num_requests, 1], where each row represents one generated
            # token per request.
            sampled_token_ids=sampled.unsqueeze(-1),
            logprobs_tensors=logprobs_tensors,
        )
        return sampler_output

    def apply_temperature(
        self,
        logits: torch.Tensor,
        temp: torch.Tensor,
        all_random: bool,
    ) -> torch.Tensor:
        # NOTE:
        # in-place division triggers buffer key error
        # in torchinductor
        if not all_random:
            temp = torch.where(temp < _SAMPLING_EPS, 1.0, temp)
        return logits.div(temp.unsqueeze(dim=1))


WARM_UP_CONFIGS = [
    {
        "name": "no_penalty_greedy",
        "no_penalties": True,
        "all_greedy": True,
        "all_random": False,
        "temperature": 0.0,
    },
    {
        "name": "no_penalty_topp",
        "no_penalties": True,
        "all_greedy": False,
        "all_random": True,
        "top_p": 0.9,
        "temperature": 0.5,
    },
    {
        "name": "no_penalty_topk",
        "no_penalties": True,
        "all_greedy": False,
        "all_random": True,
        "top_k": 1.0,
        "temperature": 0.5,
    },
    {
        "name": "no_penalty_topp_topk",
        "no_penalties": True,
        "all_greedy": False,
        "all_random": True,
        "top_p": 0.9,
        "top_k": 1.0,
        "temperature": 0.5,
    },
    {
        "name": "penalty_greedy",
        "no_penalties": False,
        "frequency_penalties": 0.1,
        "presence_penalties": 0.1,
        "repetition_penalties": 1.0,
        "all_greedy": True,
        "all_random": False,
        "temperature": 0.0,
    },
    {
        "name": "penalty_topp",
        "no_penalties": False,
        "frequency_penalties": 0.1,
        "presence_penalties": 0.1,
        "repetition_penalties": 1.0,
        "all_greedy": False,
        "all_random": True,
        "top_p": 0.9,
        "temperature": 0.5,
    },
    {
        "name": "penalty_topk",
        "no_penalties": False,
        "frequency_penalties": 0.1,
        "presence_penalties": 0.1,
        "repetition_penalties": 1.0,
        "all_greedy": False,
        "all_random": True,
        "top_k": 1.0,
        "temperature": 0.5,
    },
    {
        "name": "penalty_topp_topk",
        "no_penalties": False,
        "frequency_penalties": 0.1,
        "presence_penalties": 0.1,
        "repetition_penalties": 1.0,
        "all_greedy": False,
        "all_random": True,
        "top_p": 0.9,
        "top_k": 1.0,
        "temperature": 0.5,
    },
]
