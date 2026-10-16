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

"""Pause and resume a streaming request using AsyncLLM on RBLN.

keep mode retains the request; clear_cache=False also preserves its KV cache.
Concurrent tasks check that token delivery stops during the pause and that
the request completes after resuming.
"""

import argparse
import asyncio
import time

from vllm import RequestOutput, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

PAUSE_DURATION = 3.0


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    args = parser.parse_args()

    engine = AsyncLLM.from_engine_args(
        AsyncEngineArgs(
            model=args.model,
            model_impl="vllm",
            block_size=1024,
            max_num_batched_tokens=512,
            max_num_seqs=1,
        )
    )
    prompt = "Write a story about a dragon. Once upon a time"
    sampling_params = SamplingParams(max_tokens=30, ignore_eos=True)
    token_times: list[tuple[int, float]] = []
    ready_to_pause = asyncio.Event()
    pause_token_idx = 0

    async def generator_task() -> RequestOutput:
        async for output in engine.generate(
            request_id="pause-resume",
            prompt=prompt,
            sampling_params=sampling_params,
        ):
            token_count = len(output.outputs[0].token_ids)
            token_times.append((token_count, time.monotonic()))
            if token_count >= 5:
                ready_to_pause.set()
        return output

    async def controller_task() -> None:
        nonlocal pause_token_idx
        await ready_to_pause.wait()
        await engine.pause_generation(mode="keep", clear_cache=False)
        assert await engine.is_paused(), "Engine should be paused."
        pause_token_idx = len(token_times)
        assert token_times[-1][0] < 30, "Request finished before the pause."
        print(f"Paused at token {token_times[-1][0]} for {PAUSE_DURATION}s.")

        await asyncio.sleep(PAUSE_DURATION)
        assert len(token_times) == pause_token_idx, "Tokens arrived during the pause."

        await engine.resume_generation()
        assert not await engine.is_paused(), "Engine should have resumed."
        print("Resumed generation.")

    try:
        final_output, _ = await asyncio.gather(generator_task(), controller_task())
        assert len(token_times) > pause_token_idx, "No output arrived after resuming."
        pause_gap = (
            token_times[pause_token_idx][1] - token_times[pause_token_idx - 1][1]
        )
        assert pause_gap >= PAUSE_DURATION * 0.9, "Token delivery did not pause."
        assert final_output.finished, "Request should have finished."
        assert len(final_output.outputs[0].token_ids) == 30, (
            "Expected 30 output tokens."
        )

        print("-" * 50)
        print(f"Prompt: {prompt!r}")
        print(f"Generated text: {final_output.outputs[0].text!r}")
        print("PASS: Generation paused and resumed; the request completed.")
    finally:
        engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
