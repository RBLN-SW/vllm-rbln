# Copyright 2026 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HF and vLLM runners for the native suite. Imports vllm at module scope, so
import lazily from a fixture (after pytest_configure sets the env)."""

from __future__ import annotations

import gc
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

# RBLN requires an explicit block_size; the rest are the known-good native config
# for these small models (chunked prefill, small batch/token budget).
_RBLN_RUNNER_DEFAULTS = dict(
    block_size=1024,
    max_model_len=8192,
    max_num_batched_tokens=128,
    enable_chunked_prefill=True,
    max_num_seqs=1,
    disable_log_stats=False,
)


class VllmRunner:
    """System under test: ``vllm.LLM`` with the native RBLN config; kwargs override."""

    def __init__(self, model: str, **kwargs) -> None:
        self.llm = LLM(model=model, **{**_RBLN_RUNNER_DEFAULTS, **kwargs})

    def generate_greedy(
        self, prompts: list[str], max_tokens: int
    ) -> list[tuple[list[int], str]]:
        params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        outputs = self.llm.generate(prompts, params)
        results = []
        for request_output in outputs:
            completion = request_output.outputs[0]
            results.append((list(completion.token_ids), completion.text))
        return results

    def generate_greedy_logprobs(
        self, prompts: list[str], max_tokens: int, num_logprobs: int
    ) -> list[tuple[list[int], str, list[dict[int, float]]]]:
        params = SamplingParams(
            temperature=0.0, max_tokens=max_tokens, logprobs=num_logprobs
        )
        outputs = self.llm.generate(prompts, params)
        results = []
        for request_output in outputs:
            completion = request_output.outputs[0]
            logprobs = [
                {tid: lp.logprob for tid, lp in step.items()}
                for step in (completion.logprobs or [])
            ]
            results.append((list(completion.token_ids), completion.text, logprobs))
        return results

    def spec_decode_accepted_tokens(self) -> int:
        """Cumulative accepted draft tokens across the engine's lifetime; 0 when
        speculative decoding is off. Requires log_stats enabled (the default)."""
        for metric in self.llm.get_metrics():
            if metric.name == "vllm:spec_decode_num_accepted_tokens":
                return int(metric.value)
        return 0

    def __enter__(self) -> VllmRunner:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        # Kill EngineCore explicitly so the driver reclaims the NPU before the
        # next engine is built; vLLM otherwise tears it down lazily. Not
        # suppressed, since a failed shutdown would leak the device silently.
        self.llm.llm_engine.engine_core.shutdown()
        del self.llm
        torch._dynamo.reset()
        cleanup_dist_env_and_memory()


class HfRunner:
    """Reference oracle: the same model via HuggingFace transformers on CPU."""

    def __init__(self, model: str, dtype: str = "auto") -> None:
        self.model: Any = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=dtype, trust_remote_code=True
        )
        self.model.eval()
        self.tokenizer: Any = AutoTokenizer.from_pretrained(
            model, trust_remote_code=True
        )

    def generate_greedy(
        self, prompts: list[str], max_tokens: int
    ) -> list[tuple[list[int], str]]:
        results = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            output_ids = self.model.generate(
                **inputs, do_sample=False, max_new_tokens=max_tokens
            )
            # Keep only the newly generated tokens (drop the prompt prefix).
            new_ids = output_ids[0][inputs["input_ids"].shape[1] :]
            text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
            results.append((new_ids.tolist(), text))
        return results

    def generate_greedy_logprobs(
        self, prompts: list[str], max_tokens: int, num_logprobs: int
    ) -> list[tuple[list[int], str, list[dict[int, float]]]]:
        results = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            output = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_tokens,
                return_dict_in_generate=True,
                output_scores=True,
            )
            new_ids = output.sequences[0][inputs["input_ids"].shape[1] :]
            text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
            logprobs = []
            for step_scores in output.scores:
                step_logprobs = torch.log_softmax(step_scores[0].float(), dim=-1)
                top = torch.topk(step_logprobs, num_logprobs)
                logprobs.append(
                    {int(t): float(v) for v, t in zip(top.values, top.indices)}
                )
            results.append((new_ids.tolist(), text, logprobs))
        return results

    def __enter__(self) -> HfRunner:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del self.model
        gc.collect()
