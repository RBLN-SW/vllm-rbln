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

"""Check prefix cache metrics for two Qwen3-VL embedding requests."""

import fire
from qwen3_vl_embedding_offline import (
    default_instruction,
    image_placeholder,
    image_url,
    text,
)
from vllm import LLM
from vllm.multimodal.utils import fetch_image
from vllm.v1.metrics.reader import Counter

PREFIX_CACHE_QUERIES = "vllm:prefix_cache_queries"
PREFIX_CACHE_HITS = "vllm:prefix_cache_hits"


def read_prefix_cache_counters(llm: LLM) -> tuple[int, int]:
    counters = {PREFIX_CACHE_QUERIES: 0, PREFIX_CACHE_HITS: 0}
    seen = set()
    for metric in llm.get_metrics():
        if isinstance(metric, Counter) and metric.name in counters:
            counters[metric.name] += metric.value
            seen.add(metric.name)
    if missing := counters.keys() - seen:
        names = ", ".join(sorted(missing))
        raise RuntimeError(f"Missing prefix cache counters: {names}")
    return counters[PREFIX_CACHE_QUERIES], counters[PREFIX_CACHE_HITS]


def main(
    max_model_len: int = 8192,
    vit_max_seq_len: int = 16384,
    model: str = "Qwen/Qwen3-VL-Embedding-2B",
    prefix_repeats: int = 128,
):
    """Report cold and warm prefix cache metrics for a shared image and text."""
    if prefix_repeats < 1:
        raise ValueError("prefix_repeats must be positive.")

    multi_modal_data = {"image": fetch_image(image_url)}
    llm = LLM(
        model=model,
        model_impl="optimum",
        runner="pooling",
        block_size=4096,
        max_model_len=max_model_len,
        max_num_seqs=2,
        limit_mm_per_prompt={"image": 1},
        enable_prefix_caching=True,
        disable_log_stats=False,
        additional_config={
            "optimum_overrides": {"visual": {"max_seq_len": vit_max_seq_len}}
        },
    )
    if not llm.llm_engine.vllm_config.cache_config.enable_prefix_caching:
        raise RuntimeError(f"Prefix caching is disabled for {model}.")

    # The shared prefix must span at least one cache sub-block.
    prefix = (
        f"<|im_start|>system\n{default_instruction}<|im_end|>\n"
        f"<|im_start|>user\n{image_placeholder}" + f"{text} " * prefix_repeats
    )
    queries, hits = read_prefix_cache_counters(llm)
    for phase, suffix in (
        ("cold", "Describe the animal."),
        ("warm", "Describe the weather."),
    ):
        output = llm.embed(
            {
                "prompt": f"{prefix}{suffix}<|im_end|>\n<|im_start|>assistant\n",
                "multi_modal_data": multi_modal_data,
            },
            use_tqdm=False,
        )[0]
        new_queries, new_hits = read_prefix_cache_counters(llm)
        query_delta = new_queries - queries
        hit_delta = new_hits - hits
        print(
            f"{phase}: prompt_tokens={len(output.prompt_token_ids)} "
            f"num_cached_tokens={output.num_cached_tokens} "
            f"prefix_cache_queries={query_delta} prefix_cache_hits={hit_delta}"
        )
        queries, hits = new_queries, new_hits

        if query_delta <= 0:
            raise RuntimeError(f"The {phase} request recorded no prefix cache queries.")
        if phase == "cold" and (hit_delta != 0 or output.num_cached_tokens != 0):
            raise RuntimeError("The cold request unexpectedly hit the prefix cache.")
        if phase == "warm" and (hit_delta <= 0 or output.num_cached_tokens <= 0):
            raise RuntimeError(
                "The warm request did not reuse a cached prefix. Check the cache "
                "capacity and ensure the shared prefix spans a cache sub-block."
            )


if __name__ == "__main__":
    fire.Fire(main)
