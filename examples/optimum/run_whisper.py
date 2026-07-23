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
import asyncio

import fire
from transformers import WhisperProcessor
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

VALID_TASKS = {"transcribe", "translate"}


def generate_prompts(
    batch_size: int,
    model_id: str,
    task: str,
    language: str,
):
    from datasets import load_dataset

    dataset = load_dataset(
        "distil-whisper/librispeech_asr-noise",
        "test-pub-noise",
        streaming=True,
        split="40",
    )
    dataset = dataset.take(batch_size)
    messages = []
    processor = WhisperProcessor.from_pretrained(model_id)
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=language,
        task=task,
    )
    forced_decoder_ids = [idx for _, idx in forced_decoder_ids]
    for item in dataset:
        messages.append(
            {
                "prompt_token_ids": forced_decoder_ids,
                "multi_modal_data": {
                    "audio": (item["audio"]["array"], item["audio"]["sampling_rate"])
                },
            }
        )

    return messages


async def generate(engine: AsyncLLMEngine, request_id, request):
    results_generator = engine.generate(
        request,
        SamplingParams(
            temperature=0,
            ignore_eos=False,
            skip_special_tokens=True,
            max_tokens=448,
        ),
        str(request_id),
    )

    final_output = None
    async for request_output in results_generator:
        final_output = request_output
    return final_output


async def main(
    inputs: list,
    model_id: str,
    max_num_seqs: int,
):
    engine_args = AsyncEngineArgs(
        model=model_id,
        limit_mm_per_prompt={"audio": 1},
        max_num_seqs=max_num_seqs,
    )

    engine = AsyncLLMEngine.from_engine_args(engine_args)

    futures = []
    for request_id, request in enumerate(inputs):
        futures.append(asyncio.create_task(generate(engine, request_id, request)))

    results = await asyncio.gather(*futures)

    for i, result in enumerate(results):
        output = result.outputs[0].text
        print(f"===================== Output {i} ==============================")
        print(output)
        print("===============================================================\n")


def entry_point(
    num_input_prompt: int = 1,
    model_id: str = "openai/whisper-base",
    max_num_seqs: int = 1,
    task: str = "transcribe",
    language: str = "en",
):
    if task not in VALID_TASKS:
        raise ValueError(
            f"Invalid task {task!r}. Whisper supports: {sorted(VALID_TASKS)}"
        )
    inputs = generate_prompts(num_input_prompt, model_id, task, language)
    asyncio.run(
        main(
            inputs=inputs,
            model_id=model_id,
            max_num_seqs=max_num_seqs,
        )
    )


if __name__ == "__main__":
    fire.Fire(entry_point)
