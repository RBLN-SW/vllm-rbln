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

import asyncio
import functools
import gc
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

from tests.native.vllm_config import local_weights_path

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


@functools.cache
def kv_blocks_per_request(model: str, max_model_len: int, block_size: int) -> int:
    """Blocks one request needs across every KV cache group.

    vLLM sizes the shared pool in units of ``group_size * page_size``, so the
    requirement is the sum over groups: a full-attention group holds the whole
    context, an RBLN sliding-window group exactly one page. Plain
    ``cdiv(max_model_len, block_size) + 1`` is the special case of at most one
    sliding group -- true for a 1:1 pattern (gpt-oss), false for gemma3's 5:1.

    Does not model PP stages, Mamba/linear attention, KV sharing, or mixed
    window sizes; pin num_gpu_blocks_override on the spec for those.
    """
    from vllm_rbln import envs

    full_blocks = math.ceil(max_model_len / block_size)
    try:
        config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    except OSError:
        # Only a hybrid model needs the config; a name that cannot be resolved
        # here fails at model load anyway, so assume the single-group answer.
        return full_blocks
    config = config.get_text_config()
    num_layers = config.num_hidden_layers
    if envs.VLLM_RBLN_NUM_HIDDEN_LAYERS > 0:
        num_layers = min(num_layers, envs.VLLM_RBLN_NUM_HIDDEN_LAYERS)
    layer_types = (
        getattr(config, "layer_types", None) or ["full_attention"] * num_layers
    )
    counts = Counter(layer_types[:num_layers])

    if len(counts) == 1:
        group_size = num_layers
    else:
        fewest, most = min(counts.values()), max(counts.values())
        # vLLM's heuristic: the "1" of an n:1 pattern, unless padding to the
        # larger count wastes less.
        group_size = most if most < fewest * 1.5 else fewest
    return sum(
        math.ceil(count / group_size) * (full_blocks if kind == "full_attention" else 1)
        for kind, count in counts.items()
    )


def rbln_engine_args(model: str, **kwargs) -> dict:
    merged = {**_RBLN_RUNNER_DEFAULTS, **kwargs}
    merged.setdefault(
        "num_gpu_blocks_override",
        kv_blocks_per_request(model, merged["max_model_len"], merged["block_size"]) + 1,
    )
    return merged


class VllmRunner:
    """System under test: ``vllm.LLM`` with the native RBLN config; kwargs override."""

    def __init__(self, model: str, **kwargs) -> None:
        model = local_weights_path(model)
        self.llm = LLM(model=model, **rbln_engine_args(model, **kwargs))

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


@dataclass(frozen=True)
class DPRequest:
    """One request; ``dp_rank=None`` leaves placement to the engine's load
    balancer, an int pins it to that rank so a test's load is a known quantity."""

    prompt: str
    max_tokens: int
    dp_rank: int | None = None


async def _build_async_engine(args: AsyncEngineArgs) -> AsyncLLM:
    # Synchronous work in a coroutine on purpose: AsyncLLM starts its output
    # handler eagerly only when a loop is already running, and it has to be this
    # runner's loop.
    return AsyncLLM.from_engine_args(args)


class AsyncVllmRunner:
    """System under test for data parallel: ``AsyncLLM`` with the native RBLN
    config; kwargs override, same as VllmRunner.

    Not VllmRunner because the sync ``LLM`` rejects ``data_parallel_size > 1`` --
    the internal-load-balancing engine client only exists on the async side
    (``make_async_mp_client`` -> ``DPLBAsyncMPClient``). vLLM then owns rank
    assignment, the DP master port, the coordinator and the liveness monitor.
    """

    def __init__(
        self, model: str, *, request_timeout_s: float = 600.0, **kwargs
    ) -> None:
        self.request_timeout_s = request_timeout_s
        model = local_weights_path(model)
        args = AsyncEngineArgs(model=model, **rbln_engine_args(model, **kwargs))
        # One loop for the runner's lifetime; a fresh asyncio.run() per call would
        # orphan the engine's output handler on a closed loop.
        self._loop = asyncio.new_event_loop()
        try:
            self.engine = self._loop.run_until_complete(_build_async_engine(args))
        except BaseException:
            # __exit__ never runs when __init__ raises, and an unclosed loop adds a
            # ResourceWarning next to the real failure. BaseException so a Ctrl-C
            # during a long compile is covered too.
            self._loop.close()
            raise

    def generate_greedy(self, requests: list[DPRequest]) -> list[tuple[list[int], str]]:
        """Run every request concurrently, returning results in the order given.
        Concurrency is the point: DP ranks only interact while more than one of
        them has -- or pointedly does not have -- work."""
        return self._loop.run_until_complete(self._generate_all(requests))

    def generate_greedy_logprobs(
        self, requests: list[DPRequest], num_logprobs: int
    ) -> list[tuple[list[int], str, list[dict[int, float]]]]:
        """generate_greedy plus per-step top-k logprobs, for the tolerant
        comparison: greedy flips on near-tied logits, so two runs that agree
        mathematically can still pick different tokens."""
        return self._loop.run_until_complete(self._generate_all(requests, num_logprobs))

    async def _generate_all(
        self, requests: list[DPRequest], num_logprobs: int | None = None
    ) -> list[Any]:
        # gather() must be built inside the loop, or its futures attach to
        # whatever loop asyncio considers current.
        return list(
            await asyncio.gather(
                *(
                    self._generate_one(i, req, num_logprobs)
                    for i, req in enumerate(requests)
                )
            )
        )

    async def _generate_one(
        self, index: int, request: DPRequest, num_logprobs: int | None = None
    ) -> tuple[list[int], str] | tuple[list[int], str, list[dict[int, float]]]:
        params = SamplingParams(
            temperature=0.0, max_tokens=request.max_tokens, logprobs=num_logprobs
        )
        stream = self.engine.generate(
            request.prompt,
            params,
            request_id=f"dp-req-{index}",
            data_parallel_rank=request.dp_rank,
        )

        async def drain() -> Any:
            final = None
            async for output in stream:
                final = output
            return final

        # A bound, not a budget: a rank stalled in a collective would otherwise
        # hang the session. Startup has its own (VLLM_ENGINE_READY_TIMEOUT_S).
        try:
            final = await asyncio.wait_for(drain(), timeout=self.request_timeout_s)
        except TimeoutError as exc:
            raise TimeoutError(
                f"request {index} (dp_rank={request.dp_rank}) did not finish "
                f"within {self.request_timeout_s}s"
            ) from exc

        assert final is not None, (
            f"request {index} (dp_rank={request.dp_rank}): the engine closed the "
            f"stream without producing any output"
        )
        completion = final.outputs[0]
        if num_logprobs is None:
            return list(completion.token_ids), completion.text
        logprobs = [
            {tid: lp.logprob for tid, lp in step.items()}
            for step in (completion.logprobs or [])
        ]
        return list(completion.token_ids), completion.text, logprobs

    def __enter__(self) -> AsyncVllmRunner:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        # Reaps every DP engine core, not just rank 0's. Not suppressed: a failed
        # shutdown would leak the whole DP group of devices silently.
        try:
            self.engine.shutdown()
            # Let the cancelled output handler settle, or it is "destroyed but
            # pending".
            self._loop.run_until_complete(asyncio.sleep(0))
        finally:
            self._loop.close()
        del self.engine
        torch._dynamo.reset()
        cleanup_dist_env_and_memory()


class HfRunner:
    """Reference oracle: the same model via HuggingFace transformers on CPU."""

    def __init__(self, model: str, dtype: str = "auto") -> None:
        from vllm_rbln import envs

        model = local_weights_path(model)
        config = AutoConfig.from_pretrained(model, trust_remote_code=True)
        # Mirror the engine's truncation, or the oracle is a different model.
        if envs.VLLM_RBLN_NUM_HIDDEN_LAYERS > 0:
            text_config = config.get_text_config()
            text_config.num_hidden_layers = min(
                text_config.num_hidden_layers, envs.VLLM_RBLN_NUM_HIDDEN_LAYERS
            )
        self.model: Any = AutoModelForCausalLM.from_pretrained(
            model, config=config, torch_dtype=dtype, trust_remote_code=True
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
