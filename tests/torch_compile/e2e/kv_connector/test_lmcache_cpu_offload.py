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

"""Feature test: LMCache CPU offloading store/retrieve via vLLM offline inference.

Verifies that:
1. Prefill KV caches are stored to CPU via RBLNConnector (D2H)
2. Subsequent requests with the same prefix hit the cache (retrieve / H2D)
3. External cache hit count is > 0

Requires:
- RBLN NPU hardware (rebel runtime)

Usage:
    pytest tests/torch_compile/e2e/kv_connector/test_lmcache_cpu_offload.py -v -s
"""

import os

import pytest

MODEL = os.environ.get("TEST_MODEL", "Qwen/Qwen3-1.7B")
BLOCK_SIZE = 128
CHUNK_SIZE = 64
MAX_NUM_BATCHED_TOKENS = 128
MAX_MODEL_LEN = 512
MAX_NUM_SEQS = 8
GPU_MEM_UTIL = 0.9

SHARED_PREFIX = (
    "The following is a comprehensive overview of machine learning. "
    "Machine learning is a subset of artificial intelligence that focuses "
    "on building systems that learn from data. There are three main types: "
    "supervised learning, unsupervised learning, and reinforcement learning. "
) * 4

SUFFIXES = [
    "Summarize the key points above.",
    "What are the main types mentioned?",
    "Explain supervised learning in detail.",
    "What is reinforcement learning?",
    "How does unsupervised learning work?",
    "Give an example of supervised learning.",
    "What is the role of data in ML?",
    "Compare supervised and unsupervised learning.",
]

CORRECTNESS_PREFIX = (
    "The following is an introduction to computer networks. "
    "A computer network is a set of computers sharing resources located on or "
    "provided by network nodes. Computers use common communication protocols "
    "over digital interconnections to communicate with each other. "
) * 4


def _make_lmcache_env():
    return {
        "LMCACHE_CHUNK_SIZE": str(CHUNK_SIZE),
        "LMCACHE_LOCAL_CPU": "True",
        "LMCACHE_MAX_LOCAL_CPU_SIZE": "5",
        # Scheduler must have its own engine to look up cached tokens.
        # Without this, get_num_new_matched_tokens always returns 0
        # and external_prefix_cache metrics stay at 0.
        "LMCACHE_ENABLE_SCHEDULER_BYPASS_LOOKUP": "True",
        "LMCACHE_LOG_LEVEL": "WARNING",
    }


def _kv_transfer_config():
    from vllm.config import KVTransferConfig

    return KVTransferConfig(
        kv_connector="RBLNLMCacheConnectorV1",
        kv_role="kv_both",
    )


def _get_external_cache_hits(llm) -> tuple[int, int]:
    """Return (hits, queries) from vllm external prefix cache metrics.

    These counters are incremented by the scheduler when it calls
    ``connector.get_num_new_matched_tokens()`` for each new request:
    - **queries**: total tokens the scheduler asked the KV connector
      to look up (prompt tokens minus locally-cached tokens).
    - **hits**: tokens the connector actually found in the external
      cache (LMCache CPU store).

    If both are 0, the connector's scheduler-side is not participating
    in prefix lookup — likely an init failure (check for "degraded mode"
    in engine logs).
    """
    from vllm.v1.metrics.reader import Counter

    hits = 0
    queries = 0
    for metric in llm.get_metrics():
        if metric.name == "vllm:external_prefix_cache_hits" and isinstance(
            metric, Counter
        ):
            hits += metric.value
        elif metric.name == "vllm:external_prefix_cache_queries" and isinstance(
            metric, Counter
        ):
            queries += metric.value
    return hits, queries


@pytest.fixture(scope="module")
def lmcache_llm(monkeypatch_module):
    for k, v in _make_lmcache_env().items():
        monkeypatch_module.setenv(k, v)

    monkeypatch_module.setenv("VLLM_RBLN_USE_VLLM_MODEL", "1")
    monkeypatch_module.setenv("VLLM_RBLN_COMPILE_STRICT_MODE", "1")
    monkeypatch_module.setenv("VLLM_DISABLE_COMPILE_CACHE", "0")
    monkeypatch_module.setenv("VLLM_RBLN_SAMPLER", "0")
    monkeypatch_module.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    monkeypatch_module.setenv("VLLM_RBLN_SUB_BLOCK_CACHE", "0")

    from vllm import LLM

    return LLM(
        model=MODEL,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        block_size=BLOCK_SIZE,
        gpu_memory_utilization=GPU_MEM_UTIL,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        kv_transfer_config=_kv_transfer_config(),
        disable_log_stats=False,
    )


@pytest.fixture
def sampling_params():
    from vllm import SamplingParams

    return SamplingParams(max_tokens=16, temperature=0.0)


class TestCPUOffloadCorrectness:
    def test_output_unchanged_after_cpu_retrieval(
        self, lmcache_llm, sampling_params
    ):
        prompt = CORRECTNESS_PREFIX + SUFFIXES[0]

        out_first = lmcache_llm.generate([prompt], sampling_params)
        text_first = out_first[0].outputs[0].text

        out_second = lmcache_llm.generate([prompt], sampling_params)
        text_second = out_second[0].outputs[0].text

        assert len(text_first) > 0
        assert len(text_second) > 0
        assert text_first == text_second

    def test_multiple_suffixes_unchanged_after_cpu_retrieval(
        self, lmcache_llm, sampling_params
    ):
        prompts = [CORRECTNESS_PREFIX + s for s in SUFFIXES[:3]]

        first_outputs = [
            r.outputs[0].text
            for r in lmcache_llm.generate(prompts, sampling_params)
        ]
        second_outputs = [
            r.outputs[0].text
            for r in lmcache_llm.generate(prompts, sampling_params)
        ]

        for i, (first, second) in enumerate(zip(first_outputs, second_outputs)):
            assert len(first) > 0
            assert first == second

    def test_batch8_outputs_unchanged_after_cpu_retrieval(
        self, lmcache_llm, sampling_params
    ):
        prompts = [CORRECTNESS_PREFIX + s for s in SUFFIXES]

        first_outputs = [
            r.outputs[0].text
            for r in lmcache_llm.generate(prompts, sampling_params)
        ]
        second_outputs = [
            r.outputs[0].text
            for r in lmcache_llm.generate(prompts, sampling_params)
        ]

        for i, (first, second) in enumerate(zip(first_outputs, second_outputs)):
            assert len(first) > 0
            assert first == second


class TestCPUOffloadStoreRetrieve:
    def test_store_and_lookup(self, lmcache_llm, sampling_params):
        prompt_a = SHARED_PREFIX + SUFFIXES[0]
        prompt_b = SHARED_PREFIX + SUFFIXES[1]

        lmcache_llm.generate([prompt_a], sampling_params)
        hits_before, queries_before = _get_external_cache_hits(lmcache_llm)

        lmcache_llm.generate([prompt_b], sampling_params)
        hits_after, queries_after = _get_external_cache_hits(lmcache_llm)

        assert queries_after - queries_before > 0
        assert hits_after - hits_before > 0

    def test_batch_with_shared_prefix(self, lmcache_llm, sampling_params):
        prompts = [SHARED_PREFIX + s for s in SUFFIXES]

        hits_before, _ = _get_external_cache_hits(lmcache_llm)
        outputs = lmcache_llm.generate(prompts, sampling_params)
        hits_after, _ = _get_external_cache_hits(lmcache_llm)

        assert len(outputs) == len(prompts)
        for output in outputs:
            assert len(output.outputs[0].text) > 0

        assert hits_after - hits_before > 0
