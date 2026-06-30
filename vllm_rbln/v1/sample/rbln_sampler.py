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
import inspect
import torch
import torch.nn as nn
from vllm.sampling_params import _SAMPLING_EPS
from vllm_rbln.v1.sample.ops.logprobs import batched_count_greater_than

try:
    import torch.rbln

    has_torch_rbln = True
except ImportError:
    has_torch_rbln = False

from vllm_rbln.logger import init_logger
from vllm_rbln.torch_compile_backend import logged_rbln_backend
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


def resolve_compile_context(
    compile_context: rebel.CompileContext | None,
) -> rebel.CompileContext:
    """Return a default CompileContext when one is not provided.

    Used when running through the device tensor path in rbln_model_runner or
    when triggered by optimum_model_runner.
    """
    if compile_context is not None:
        return compile_context
    if "use_global_ctx" in inspect.signature(rebel.CompileContext).parameters:
        return rebel.CompileContext(use_global_ctx=True)
    return rebel.CompileContext()


def build_compile_options(compile_context: rebel.CompileContext) -> dict:
    """Build the torch.compile ``options`` dict shared by the RBLN samplers."""
    use_dt = envs.VLLM_RBLN_USE_DEVICE_TENSOR
    options: dict = {}
    if use_dt:
        options["model_trace_method"] = "export"
    if not use_dt:
        options["compile_context"] = compile_context
    if envs.VLLM_RBLN_COMPILE_STRICT_MODE:
        options["mode"] = "strict"
    if has_torch_rbln or use_dt:
        options["tensor_parallel_size"] = 1
        if not use_dt:
            options["use_global_ctx"] = True
            options["global_device_id"] = 0
    return options


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


def rbln_greedy_sample(logits: torch.Tensor) -> torch.Tensor:
    """
    Implementation of RBLN greedy sampling.
    To avoid self parameter issues when torch.compile is used,
    we define this as a static method.
    """
    sampled = torch.ops.rbln.argmax(logits)
    return sampled


class RBLNTopKTopPSampler(nn.Module):
    def __init__(
        self,
        logprobs_mode: LogprobsMode = "raw_logprobs",
        compile_context: rebel.CompileContext = None,
    ):
        # TODO(rbln): Merge more ops to rbln context.
        #       Currently, we only have softmax in rbln context.
        super().__init__()
        self.logprobs_mode = logprobs_mode

        assert self.logprobs_mode not in ("processed_logits", "processed_logprobs"), (
            "RBLN Sampling does not support returning logits/logprobs"
        )

        options = build_compile_options(compile_context)
        if envs.VLLM_RBLN_USE_DEVICE_TENSOR:
            options["model_trace_method"] = "export"

        self._compiled_rbln_topk_topp_sampler = torch.compile(
            rbln_top_k_top_p_sample,
            dynamic=False,
            fullgraph=True,
            backend=logged_rbln_backend,
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
        compile_context: rebel.CompileContext = None,
    ):
        super().__init__()
        # If using device tensor in rbln_model_runner
        # or triggered by optimum_model_runner
        compile_context = resolve_compile_context(compile_context)
        if logprobs_mode in ("raw_logprobs", "raw_logits"):
            self.topk_topp_sampler = RBLNTopKTopPSampler(
                logprobs_mode=logprobs_mode,
                compile_context=compile_context,
            )
        else:
            logger.warning_once(
                f"RBLN Sampling does not support logprobs_mode: {logprobs_mode}. "
                "Using native sampler instead."
            )
        options = build_compile_options(compile_context)
        # FIXME compiling both greedy and top-k top-p sampling
        # causes some issues in torchinductor.
        self._compiled_greedy_sample = torch.compile(
            rbln_greedy_sample,
            dynamic=False,
            fullgraph=True,
            backend=logged_rbln_backend,
            options=options,
        )

    @torch.compiler.disable
    def greedy_sample(self, logits: torch.Tensor) -> torch.Tensor:
        return self._compiled_greedy_sample(logits)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        logprobs_mode_override: LogprobsMode | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Sample logits based on sampling metadata.

        The various logits processing functions called in this method
        may update the logits tensor in-place.
        """

        logprobs_mode = logprobs_mode_override or self.logprobs_mode
        assert not (sampling_metadata.all_greedy and sampling_metadata.all_random)
        if not sampling_metadata.all_greedy:
            greedy_sampled = None
        else:
            # It runs only all_greedy is True
            greedy_sampled = self.greedy_sample(logits)
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
        logits = self.apply_temperature(
            logits, sampling_metadata.temperature, sampling_metadata.all_random
        )

        # Apply logits processors that only apply to random sampling
        # (argmax invariant)
        for processor in sampling_metadata.logitsprocs.argmax_invariant:
            logits = processor.apply(logits)

        # Apply top_k and/or top_p.
        random_sampled, processed_logprobs = self.topk_topp_sampler(
            logits,
            sampling_metadata.generators,
            sampling_metadata.top_k,
            sampling_metadata.top_p,
        )

        assert greedy_sampled is None, (
            "Upstream vLLM runs greedy and random sampling "
            "separately and merges the results, "
            "but vLLM RBLN processes greedy and random requests together: "
            "greedy requests are routed through the random-sampling path "
            "with a very small temperature value."
        )
        return random_sampled, processed_logprobs

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
        temperature: torch.Tensor,
        all_random: bool,
    ) -> torch.Tensor:
        # NOTE:
        # in-place division triggers buffer key error
        # in torchinductor
        # NOTE:
        # Greedy requests use a small temperature (1e-3) so softmax collapses
        # to a near one-hot at argmax. _SAMPLING_EPS (1e-5) is too small here —
        # it pushes logits past softmax's safe exp range and overflows.
        _GREEDY_EPS = 1e-3
        if not all_random:
            temperature = torch.where(
                temperature < _SAMPLING_EPS, _GREEDY_EPS, temperature
            )
        temperature = temperature.to(logits.dtype)
        return logits.div(temperature.unsqueeze(dim=1))

    # NOTE(eunji.lee):
    # mark_unbacked torch method should be called outside of torch.compile
    @staticmethod
    @torch.compiler.disable
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
        # Avoid 0/1 specialization recompile on the batch dimension
        # of the compiled batched_count_greater_than. mark_unbacked makes
        # the size fully symbolic so dynamo doesn't specialize when
        # batch_size transitions from 1 to >=2.
        # torch._dynamo.decorators.mark_unbacked(logprobs, 0)
        # torch._dynamo.decorators.mark_unbacked(token_logprobs, 0)
        token_ranks = batched_count_greater_than(logprobs, token_logprobs)

        # Concatenate together with the topk.
        indices = torch.cat((token_ids, topk_indices), dim=1)
        logprobs = torch.cat((token_logprobs, topk_logprobs), dim=1)

        # Use int32 to reduce the tensor size.
        indices = indices.to(torch.int32)

        return LogprobsTensors(indices, logprobs, token_ranks)


WARM_UP_CONFIGS = [
    {
        "name": "no_penalty_greedy",
        "no_penalties": True,
        "all_greedy": True,
        "all_random": False,
        "temperature": 0.0,
    },
    {
        "name": "no_penalty_random",
        "no_penalties": True,
        "all_greedy": False,
        "all_random": True,
        "temperature": 0.5,
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
