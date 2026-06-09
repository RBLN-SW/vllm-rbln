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

"""End-to-end regression for the spec-decode + device-tensor sampling crash.

With ``VLLM_RBLN_USE_DEVICE_TENSOR=1`` and ngram speculative decoding, the
target logits are ``[num_tokens + batch_size, vocab]``, which always exceeds
``num_reqs``. ``RBLNModelRunner._sample`` used to pad ``SamplingMetadata`` to
that length and ``RBLNRejectionSampler`` then reused it for the
``[batch_size]``-row bonus logits, so the temperature broadcast crashed as soon
as two or more requests reached the sampler in the same step. A single request
broadcast and hid the bug.

This test drives >= 2 concurrent requests (``max_num_seqs >= 2``) with repeated
n-grams (so the ngram proposer actually drafts) and asserts the engine no
longer crashes.
"""

from __future__ import annotations

import pytest
from vllm import SamplingParams

from ...utils import managed_llm

MODEL_ID = "Qwen/Qwen3-0.6B"
BLOCK_SIZE = 1024

# Prompts with repeated n-grams so the ngram proposer finds prompt-lookup
# matches and emits draft tokens (otherwise the rejection sampler bonus path is
# never exercised). Distinct content across requests keeps the batch
# heterogeneous, which is what triggered the temperature broadcast mismatch.
PROMPTS = [
    "The quick brown fox jumps over the lazy dog. The quick brown fox jumps "
    "over the lazy dog. The quick brown fox",
    "Paris is the capital of France. Paris is the capital of France. Paris is "
    "the capital of",
    "one two three four one two three four one two three four one two three",
    "to be or not to be, that is the question. to be or not to be, that is the",
]


def _llm_kwargs() -> dict:
    return dict(
        model=MODEL_ID,
        # >= 2 so the four prompts batch together and reach the sampler in the
        # same step — the necessary condition for the crash.
        max_num_seqs=4,
        max_model_len=2048,
        enable_chunked_prefill=True,
        max_num_batched_tokens=128,
        # Keep the device KV budget tiny; the prompts are short so a handful of
        # blocks is plenty, and it keeps the engine within the 15.7GiB card.
        num_gpu_blocks_override=8,
        speculative_config={
            "method": "ngram",
            "num_speculative_tokens": 3,
            "prompt_lookup_max": 5,
            "prompt_lookup_min": 2,
        },
        seed=0,
        disable_log_stats=False,
        # block_size is validated in the EngineArgs ctor; pass it directly to
        # LLM so this regression can exercise the RBLN block size.
        block_size=BLOCK_SIZE,
    )


def test_ngram_device_tensor_multi_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    # Random (non-greedy) sampling so the temperature tensor is actually
    # consumed in the bonus path — this is where the broadcast used to fail.
    sampling_params = SamplingParams(temperature=0.8, top_p=0.9, max_tokens=32)

    env = {"VLLM_RBLN_USE_DEVICE_TENSOR": "1"}
    with managed_llm(monkeypatch, env, **_llm_kwargs()) as llm:
        outputs = llm.generate(PROMPTS, sampling_params=sampling_params)

        assert len(outputs) == len(PROMPTS)
        for output in outputs:
            assert output.prompt in PROMPTS
            assert len(output.outputs) == 1
            assert output.outputs[0].text.strip()

        metrics = llm.get_metrics()
        metric_names = {metric.name for metric in metrics}
        assert "vllm:spec_decode_num_drafts" in metric_names
        assert "vllm:spec_decode_num_draft_tokens" in metric_names
        assert "vllm:spec_decode_num_accepted_tokens" in metric_names
