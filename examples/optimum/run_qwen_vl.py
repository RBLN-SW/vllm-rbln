# SPDX-License-Identifier: Apache-2.0
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
from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoTokenizer
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

# If the video is too long
# set `VLLM_ENGINE_ITERATION_TIMEOUT_S` to a higher timeout value.
VIDEO_URLS = [
    "https://duguang-labelling.oss-cn-shanghai.aliyuncs.com/qiansun/video_ocr/videos/50221078283.mp4",
    "https://cdn.pixabay.com/video/2022/04/18/114413-701051082_large.mp4",
    "https://videos.pexels.com/video-files/855282/855282-hd_1280_720_25fps.mp4",
]


def generate_prompts_video(batch_size: int, model_id: str):
    processor = AutoProcessor.from_pretrained(model_id, padding_side="left")
    messages = [[
        {
            "role":
            "user",
            "content": [
                {
                    "type": "video",
                    "video": VIDEO_URLS[i],
                },
                {
                    "type": "text",
                    "text": "Describe this video."
                },
            ],
        },
    ] for i in range(batch_size)]

    texts = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    arr_image_inputs = []
    arr_video_inputs = []
    arr_video_kwargs = []
    for i in range(batch_size):
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages[i], return_video_kwargs=True)
        arr_image_inputs.append(image_inputs)
        arr_video_inputs.append(video_inputs)
        arr_video_kwargs.append(video_kwargs)

    return [{
        "prompt": text,
        "multi_modal_data": {
            "video": video_inputs,
        },
        "mm_processor_kwargs": {
            "min_pixels": 1024 * 14 * 14,
            "max_pixels": 5120 * 14 * 14,
            **video_kwargs,
        },
    } for text, image_inputs, video_inputs, video_kwargs in zip(
        texts, arr_image_inputs, arr_video_inputs, arr_video_kwargs)]


def generate_prompts_image(batch_size: int, model_id: str):
    dataset = load_dataset("lmms-lab/llava-bench-in-the-wild",
                           split="train").shuffle(seed=42)
    processor = AutoProcessor.from_pretrained(model_id, padding_side="left")
    messages = [[
        {
            "role":
            "system",
            "content": [{
                "type":
                "text",
                "text":
                "You are a helpful assistant."
                "Answer the each question based on the image.",
            }],
        },
        {
            "role":
            "user",
            "content": [
                {
                    "type": "image",
                    "image": dataset[i]["image"]
                },
                {
                    "type": "text",
                    "text": dataset[i]["question"]
                },
            ],
        },
    ] for i in range(batch_size)]

    texts = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    arr_image_inputs = []
    arr_video_inputs = []
    arr_video_kwargs = []

    for i in range(batch_size):
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages[i], return_video_kwargs=True)
        arr_image_inputs.append(image_inputs)
        arr_video_inputs.append(video_inputs)
        arr_video_kwargs.append(video_kwargs)

    return [
        {
            "prompt": text,
            "multi_modal_data": {
                # "video": video_inputs,
                "image": image_inputs
            },
            "mm_processor_kwargs": {
                "min_pixels": 1024 * 14 * 14,
                "max_pixels": 5120 * 14 * 14,
                "padding": True,
                **video_kwargs,
            },
        } for text, image_inputs, video_inputs, video_kwargs in zip(
            texts, arr_image_inputs, arr_video_inputs, arr_video_kwargs)
    ]


def generate_prompts_wo_processing(batch_size: int, model_id: str):
    dataset = load_dataset("lmms-lab/llava-bench-in-the-wild",
                           split="train").shuffle(seed=42)
    processor = AutoProcessor.from_pretrained(model_id, padding_side="left")
    messages = [[
        {
            "role":
            "system",
            "content": [{
                "type":
                "text",
                "text":
                "You are a helpful assistant."
                "Answer the each question based on the image.",
            }],
        },
        {
            "role":
            "user",
            "content": [
                {
                    "type": "image",
                    "image": dataset[i]["image"]
                },
                {
                    "type": "text",
                    "text": dataset[i]["question"]
                },
            ],
        },
    ] for i in range(batch_size)]
    images = [[dataset[i]["image"]] for i in range(batch_size)]

    texts = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    return [{
        "prompt": text,
        "multi_modal_data": {
            "image": image
        },
        "mm_processor_kwargs": {
            "min_pixels": 1024 * 14 * 14,
            "max_pixels": 5120 * 14 * 14,
        }
    } for text, image in zip(texts, images)]


async def generate(engine: AsyncLLMEngine, tokenizer, request_id, request):
    results_generator = engine.generate(
        request,
        SamplingParams(temperature=0,
                       ignore_eos=False,
                       skip_special_tokens=True,
                       stop_token_ids=[tokenizer.eos_token_id],
                       max_tokens=200),
        str(request_id),
    )

    final_output = None
    async for request_output in results_generator:
        final_output = request_output
    return final_output


async def main(
    num_input_prompt: int,
    model_id: str,
):
    engine_args = AsyncEngineArgs(model=model_id)

    engine = AsyncLLMEngine.from_engine_args(engine_args)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    inputs = generate_prompts_image(num_input_prompt, model_id)
    # inputs = generate_prompts_video(num_input_prompt, model_id)
    # inputs = generate_prompts_wo_processing(num_input_prompt, model_id)

    futures = []
    for request_id, request in enumerate(inputs):
        futures.append(
            asyncio.create_task(
                generate(engine, tokenizer, request_id, request)))

    results = await asyncio.gather(*futures)

    for i, result in enumerate(results):
        output = result.outputs[0].text
        print(
            f"===================== Output {i} ==============================")
        print(output)
        print(
            "===============================================================\n"
        )


def entry_point(
    num_input_prompt: int = 1,
    model_id: str = "/qwen2_5-vl-7b-32k-b4-kv16k",
):
    asyncio.run(main(
        num_input_prompt=num_input_prompt,
        model_id=model_id,
    ))


if __name__ == "__main__":
    fire.Fire(entry_point)