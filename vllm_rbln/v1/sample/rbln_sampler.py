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
from typing import Optional
from vllm_rbln.logger import init_logger
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler as VLLMSampler
import rebel
from vllm.config.model import LogprobsMode
from vllm_rbln.v1.sample.ops.penalties import (apply_all_penalties as
                                               rbln_apply_all_penalties)
import vllm_rbln.rbln_envs as envs

logger = init_logger(__name__)

_SAMPLING_EPS = 1e-5


def apply_top_k_top_p(
    logits: torch.Tensor,
    k: Optional[torch.Tensor],
    p: Optional[torch.Tensor],
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
    if p is None and k is None:
        # Custom operators in torch.compile must not return aliases of inputs.
        # We return a clone to satisfy this strict aliasing constraint.
        return logits.clone()

    logits_sort, logits_idx = logits.sort(dim=-1,
                                          descending=False,
                                          stable=True)

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
    return logits


@torch.library.custom_op("rbln::top_k_top_p", mutates_args=())
def top_k_top_p(logits: torch.Tensor, k: Optional[torch.Tensor],
                p: Optional[torch.Tensor]) -> torch.Tensor:
    return apply_top_k_top_p(logits, k, p)


@top_k_top_p.register_fake
def top_k_top_p_fake(logits: torch.Tensor, k: Optional[torch.Tensor],
                     p: Optional[torch.Tensor]) -> torch.Tensor:
    return apply_top_k_top_p(logits, k, p)


class RBLNSampler(VLLMSampler):

    def __init__(self,
                 logprobs_mode: LogprobsMode = "raw_logprobs",
                 seed: int = 42):
        super().__init__()
        rebel.manual_seed(seed)

        options = {"compile_context": rebel.CompileContext()}
        if envs.VLLM_RBLN_COMPILE_STRICT_MODE:
            options["mode"] = "strict"
        self._compiled_rbln_topk_topp_sampler = torch.compile(
            self._rbln_topk_topp_sampler_impl,
            dynamic=False,
            fullgraph=True,
            backend="rbln",
            options=options,
        )
        self.logprobs_mode = logprobs_mode

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

    def apply_topk_topp_sampler(
        self, logits: torch.Tensor, top_k: Optional[torch.Tensor],
        top_p: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        sampled = self.rbln_topk_topp_sampler(logits, top_k, top_p)
        logits_to_return = None
        if self.logprobs_mode == "processed_logits":
            logits_to_return = logits
        elif self.logprobs_mode == "processed_logprobs":
            logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)
        return sampled, logits_to_return

    @staticmethod
    def _rbln_topk_topp_sampler_impl(logits: torch.Tensor, top_k: torch.Tensor,
                                     top_p: torch.Tensor) -> torch.Tensor:
        """
        Implementation of RBLN top-k top-p sampling.
        To avoid self parameter issues when torch.compile is used,
        we define this as a static method.
        """
        # Apply top-k top-p sampling using RBLN custom op.
        # It requires softmax prior to calling the op.
        probs = torch.nn.functional.softmax(logits, dim=-1)
        sampled = torch.ops.rbln.top_k_top_p(probs, top_k, top_p)
        return sampled

    @torch.compiler.disable
    def rbln_topk_topp_sampler(self, logits: torch.Tensor, top_k: torch.Tensor,
                               top_p: torch.Tensor) -> torch.Tensor:
        """
        Wrapper for the compiled RBLN top-p sampler.
        To avoid recompile on runtime, we decorate this method with
        `torch.compiler.disable` and call the pre-compiled function.
        """
        return self._compiled_rbln_topk_topp_sampler(logits, top_k, top_p)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        logprobs_mode_override: LogprobsMode | None = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Sample logits based on sampling metadata.

        The various logits processing functions called in this method
        may update the logits tensor in-place.
        """
        logprobs_mode = logprobs_mode_override or self.logprobs_mode
        assert not (sampling_metadata.all_greedy
                    and sampling_metadata.all_random)
        if sampling_metadata.all_random:
            greedy_sampled = None
        else:
            greedy_sampled = self.rbln_topk_topp_sampler(logits, None, None)
            if sampling_metadata.all_greedy:
                processed_logprobs = None
                if sampling_metadata.max_num_logprobs is not None:
                    if logprobs_mode == "processed_logits":
                        processed_logprobs = logits
                    elif logprobs_mode == "processed_logprobs":
                        processed_logprobs = self.compute_logprobs(logits)
                return greedy_sampled, processed_logprobs

        assert sampling_metadata.temperature is not None

        # Apply temperature.
        logits = self.apply_temperature(logits, sampling_metadata.temperature,
                                        sampling_metadata.all_random)
        # Apply logits processors that only apply to random sampling
        # (argmax invariant)
        for processor in sampling_metadata.logitsprocs.argmax_invariant:
            logits = processor.apply(logits)

        random_sampled, processed_logprobs = self.apply_topk_topp_sampler(
            logits, sampling_metadata.top_k, sampling_metadata.top_p)

        if greedy_sampled is None:
            return random_sampled, processed_logprobs

        sampled = torch.where(
            sampling_metadata.temperature < _SAMPLING_EPS,
            greedy_sampled,
            random_sampled,
            out=greedy_sampled,  # Reuse tensor
        )
        return sampled, processed_logprobs

    @staticmethod
    def apply_penalties(
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        output_token_ids: list[list[int]],
    ) -> torch.Tensor:
        if sampling_metadata.no_penalties:
            return logits
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

    @staticmethod
    def get_bucket_sizes(max_num_seqs: int) -> list[int]:
        """Get the bucket sizes for the sampler.
        Args:
            max_num_seqs (int): The maximum number of sequences.
        Returns:
            list[int]: The bucket sizes.
        [1, 2, 4] + list(range(8, 256, 8)) + list(
            range(256, max_num_seqs + 1, 16))
        """
        # FIXME(eunji.lee)
        # Not used. To be removed.
        bucket_sizes = [i for i in [1, 2, 4] if i <= max_num_seqs]
        if max_num_seqs >= 8:
            # Step size 8 for small batch sizes, up to 256(not included)
            bucket_sizes += list(range(8, min(max_num_seqs + 1, 256), 8))
        if max_num_seqs >= 256:
            # Step size 16 for larger batch sizes
            bucket_sizes += list(range(256, max_num_seqs + 1, 16))
        return bucket_sizes


WARM_UP_CONFIGS = [
    {
        "name": "no_penalty_greedy",
        "no_penalties": True,
        "all_greedy": True,
        "all_random": False,
        "temperature": 0.0
    },
    {
        "name": "no_penalty_topp",
        "no_penalties": True,
        "all_greedy": False,
        "all_random": True,
        "top_p": 0.9,
        "temperature": 0.5
    },
    {
        "name": "no_penalty_topk",
        "no_penalties": True,
        "all_greedy": False,
        "all_random": True,
        "top_k": 1.0,
        "temperature": 0.5
    },
    {
        "name": "no_penalty_topp_topk",
        "no_penalties": True,
        "all_greedy": False,
        "all_random": True,
        "top_p": 0.9,
        "top_k": 1.0,
        "temperature": 0.5
    },
    {
        "name": "penalty_greedy",
        "no_penalties": False,
        "frequency_penalties": 0.1,
        "presence_penalties": 0.1,
        "repetition_penalties": 1.0,
        "all_greedy": True,
        "all_random": False,
        "temperature": 0.0
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
        "temperature": 0.5
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
        "temperature": 0.5
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
        "temperature": 0.5
    },
]
