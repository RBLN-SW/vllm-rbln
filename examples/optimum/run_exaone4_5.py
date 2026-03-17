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
import os
from pathlib import Path

import fire
from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoTokenizer


def _ensure_sitecustomize_for_subprocess() -> None:
    """Expose local sitecustomize.py to vLLM registry subprocesses."""
    this_dir = str(Path(__file__).resolve().parent)
    current = os.environ.get("PYTHONPATH")

    if not current:
        os.environ["PYTHONPATH"] = this_dir
        return

    paths = current.split(os.pathsep)
    if this_dir not in paths:
        os.environ["PYTHONPATH"] = os.pathsep.join([this_dir, *paths])


_ensure_sitecustomize_for_subprocess()
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
    conversations = [
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": VIDEO_URLS[i],
                    },
                    {"type": "text", "text": "Describe this video."},
                ],
            },
        ]
        for i in range(batch_size)
    ]

    texts = [
        processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        for conv in conversations
    ]
    _, video_inputs, video_kwargs = process_vision_info(
        conversations, return_video_kwargs=True
    )

    if "fps" in video_kwargs and isinstance(video_kwargs["fps"], list):
        if len(video_kwargs["fps"]) == 1:
            video_kwargs["fps"] = video_kwargs["fps"][0]
        else:
            video_kwargs.pop("fps", None)

    return [
        {
            "prompt": text,
            "multi_modal_data": {
                "video": video_inputs,
            },
            "mm_processor_kwargs": {
                "min_pixels": 256 * 28 * 28,
                "max_pixels": 1280 * 28 * 28,
                **video_kwargs,
            },
        }
        for text, video_inputs in zip(texts, video_inputs)
    ]


def generate_prompts_image(batch_size: int, model_id: str):
    dataset = load_dataset("lmms-lab/llava-bench-in-the-wild", split="train").shuffle(
        seed=42
    )
    processor = AutoProcessor.from_pretrained(model_id, padding_side="left")
    conversations = [
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What’s shown in this image?",
                    }
                    ,
                    {"type": "image"},
                ],
            },
        ]
        for i in range(batch_size)
    ]
    image_inputs = [dataset[i]["image"] for i in range(batch_size)]
    texts = [
        processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        for conv in conversations
    ]

    return [
        {
            "prompt": text,
            "multi_modal_data": {
                "image": [image_inputs]
            },
            "mm_processor_kwargs": {
                "min_pixels": 256 * 28 * 28,
                "max_pixels": 1280 * 28 * 28,
                "padding": True,
            },
        }
        for text, image_inputs in zip(texts, image_inputs)
    ]


def generate_prompts_wo_processing(batch_size: int, model_id: str):
    dataset = load_dataset("lmms-lab/llava-bench-in-the-wild", split="train").shuffle(
        seed=42
    )
    processor = AutoProcessor.from_pretrained(model_id, padding_side="left")
    messages = [
        [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful assistant."
                        "Answer the each question based on the image.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": dataset[i]["image"]},
                    {"type": "text", "text": dataset[i]["question"]},
                ],
            },
        ]
        for i in range(batch_size)
    ]
    images = [[dataset[i]["image"]] for i in range(batch_size)]

    texts = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    return [
        {
            "prompt": text,
            "multi_modal_data": {"image": image},
            "mm_processor_kwargs": {
                "min_pixels": 1024 * 14 * 14,
                "max_pixels": 5120 * 14 * 14,
            },
        }
        for text, image in zip(texts, images)
    ]


async def generate(engine: AsyncLLMEngine, tokenizer, request_id, request):
    results_generator = engine.generate(
        request,
        SamplingParams(
            temperature=0,
            ignore_eos=False,
            skip_special_tokens=True,
            stop_token_ids=[tokenizer.eos_token_id],
            max_tokens=200,
        ),
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
            asyncio.create_task(generate(engine, tokenizer, request_id, request))
        )

    results = await asyncio.gather(*futures)

    for i, result in enumerate(results):
        output = result.outputs[0].text
        print(f"===================== Output {i} ==============================")
        print(output)
        print("===============================================================\n")


def entry_point(
    num_input_prompt: int = 4,
    model_id: str = "/home/kblee/.cache/rbln-exec/compile_results/optimum-exaone4-5/model_id__exaone4.5-32b#batch_size__4#max_seq_len__128000#n_layers__64#tensor_parallel_size__16#vit_seq_lens__16384#kvcache_partition_len__5120#use_attn_mask__False/model",
):
    asyncio.run(
        main(
            num_input_prompt=num_input_prompt,
            model_id=model_id,
        )
    )


if __name__ == "__main__":
    fire.Fire(entry_point)
