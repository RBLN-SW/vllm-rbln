# Copyright 2026 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reset the prefix cache during generation using LLMEngine on RBLN.

At step 10, reset_running_requests=True preempts running requests and clears
prefix cache entries. Requests then resume by recomputing their KV cache;
the reset invalidates cache reuse rather than zeroing KV tensor memory.
A completed-request comparison checks cold, warm, and reset cache hits.
See docs/sub_block_prefix_caching.md for sub-block caching device requirements.
"""

import argparse

from vllm import EngineArgs, LLMEngine, SamplingParams, TokensPrompt

from vllm_rbln.config import RBLNConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    args = parser.parse_args()

    block_size = 1024
    sub_block_size = 128
    engine = LLMEngine.from_engine_args(
        EngineArgs(
            model=args.model,
            model_impl="vllm",
            max_num_seqs=4,
            block_size=block_size,
            max_num_batched_tokens=512,
            enable_prefix_caching=True,
            additional_config=RBLNConfig(
                enable_sub_block_cache=True, sub_block_size=sub_block_size
            ),
        )
    )
    prompts = [
        "A robot may not injure a human being " * 50,
        "A robot may not injure a human being " * 50,
        "To be or not to be,",
        "What is the meaning of life?",
    ]

    # Keep requests active long enough to reach the reset at step 10.
    sampling_params = SamplingParams(temperature=0.0, min_tokens=16, max_tokens=16)
    step_id = 0
    request_id = 0
    cache_reset = False
    finished_requests: set[str] = set()
    while request_id < len(prompts) or engine.has_unfinished_requests():
        if request_id < len(prompts):
            engine.add_request(str(request_id), prompts[request_id], sampling_params)
            request_id += 1

        if step_id == 10:
            cache_reset = engine.reset_prefix_cache(reset_running_requests=True)
            if not cache_reset:
                raise RuntimeError("Failed to reset the prefix cache")

        for output in engine.step():
            if output.finished:
                finished_requests.add(output.request_id)
        step_id += 1

    if not cache_reset:
        raise RuntimeError("All requests finished before the cache reset at step 10")
    assert len(finished_requests) == len(prompts), "Every request must finish."
    print("Prefix cache reset at step 10: PASS; all requests finished.")

    tokenizer = engine.get_tokenizer()
    probe_token_ids = tokenizer.encode(prompts[0])
    assert sub_block_size < len(probe_token_ids) < block_size, (
        "The cache probe must span sub-blocks while fitting within one KV block."
    )
    probe = TokensPrompt(prompt_token_ids=probe_token_ids)
    probe_params = SamplingParams(temperature=0.0, max_tokens=1)
    cached_tokens = []
    print("\nCached Prompt Tokens:")
    for phase in ("cold", "warm", "after_reset"):
        if phase != "warm" and not engine.reset_prefix_cache():
            raise RuntimeError("Failed to clear the prefix cache for the cache probe")
        engine.add_request(f"cache_{phase}", probe, probe_params)
        final_output = None
        while engine.has_unfinished_requests():
            for output in engine.step():
                if output.finished:
                    final_output = output
        assert final_output is not None, "The cache probe must finish."
        cached_tokens.append(final_output.num_cached_tokens)
        print(f"  - {phase}: {final_output.num_cached_tokens}")

    assert cached_tokens[0] == 0, "The cold probe must not reuse cached tokens."
    assert (
        cached_tokens[1] >= sub_block_size and cached_tokens[1] % sub_block_size == 0
    ), "The warm probe must reuse complete sub-blocks."
    assert cached_tokens[2] == 0, "The probe after reset must not reuse cached tokens."
    print("Prefix cache invalidation: PASS; cold=0, warm>0, after_reset=0.")


if __name__ == "__main__":
    main()
