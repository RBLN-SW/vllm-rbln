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

import os

import fire
from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM, SamplingParams

# If the video is too long
# set `VLLM_ENGINE_ITERATION_TIMEOUT_S` to a higher timeout value.
VIDEO_URLS = [
    "https://duguang-labelling.oss-cn-shanghai.aliyuncs.com/qiansun/video_ocr/videos/50221078283.mp4",
    "https://cdn.pixabay.com/video/2022/04/18/114413-701051082_large.mp4",
    "https://videos.pexels.com/video-files/855282/855282-hd_1280_720_25fps.mp4",
]


def generate_prompts_video(batch_size: int, model: str):
    processor = AutoProcessor.from_pretrained(model, padding_side="left")
    messages = [
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
            messages[i],
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        arr_image_inputs.append(image_inputs)
        arr_video_inputs.append(video_inputs)
        arr_video_kwargs.append(video_kwargs)

    return [
        {
            "prompt": text,
            "multi_modal_data": {
                "video": video_inputs,
            },
            "mm_processor_kwargs": {
                "min_pixels": 1024 * 14 * 14,
                "max_pixels": 5120 * 14 * 14,
                **video_kwargs,
            },
        }
        for text, image_inputs, video_inputs, video_kwargs in zip(
            texts, arr_image_inputs, arr_video_inputs, arr_video_kwargs
        )
    ]


def generate_prompts_image(batch_size: int, model: str):
    dataset = load_dataset("lmms-lab/llava-bench-in-the-wild", split="train").shuffle(
        seed=42
    )
    processor = AutoProcessor.from_pretrained(model, padding_side="left")
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

    texts = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    arr_image_inputs = []

    for i in range(batch_size):
        image_inputs, _ = process_vision_info(
            messages[i],
        )
        arr_image_inputs.append(image_inputs)

    return [
        {
            "prompt": text,
            "multi_modal_data": {"image": image_inputs},
            "mm_processor_kwargs": {
                "min_pixels": 1024 * 14 * 14,
                "max_pixels": 5120 * 14 * 14,
                "padding": True,
            },
        }
        for text, image_inputs in zip(texts, arr_image_inputs)
    ]


def generate_prompts_wo_processing(batch_size: int, model: str):
    dataset = load_dataset("lmms-lab/llava-bench-in-the-wild", split="train").shuffle(
        seed=42
    )
    processor = AutoProcessor.from_pretrained(model, padding_side="left")
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


def main(
    num_input_prompt: int = 1,
    # NOTE: This example supports Qwen2-VL, Qwen2.5-VL, Qwen3-VL and Qwen3.5
    model: str = "Qwen/Qwen3-VL-2B-Instruct",
    max_num_seqs: int = 1,
    max_model_len: int = 8192,
    num_devices: int = 1,
    block_size: int = None,  # if None, will be set to max_model_len
    vision_max_seq_len: int = 2048,
):
    # Compile shape follows the rebel_compiler multi-modal compile.py pattern:
    # num_devices → env var, batch_size / max_seq_len → additional_config
    # ["rbln_config"]; block_size stays a top-level vLLM kwarg. The previous
    # hard-coded 17-device `device` / `visual.device` pinning was dropped so this
    # runs on any box; for a specific layout, add those keys back into rbln_config.
    os.environ["VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK"] = str(num_devices)
    if block_size is None:
        block_size = max_model_len
    rbln_config = {
        "visual": {"max_seq_len": vision_max_seq_len},
    }
    llm = LLM(
        model=model,
        block_size=block_size,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        additional_config={"rbln_config": rbln_config},
    )
    tokenizer = AutoTokenizer.from_pretrained(model)
    inputs = generate_prompts_image(num_input_prompt, model)
    # inputs = generate_prompts_video(num_input_prompt, model)
    # inputs = generate_prompts_wo_processing(num_input_prompt, model)

    sampling_params = SamplingParams(
        temperature=0,
        ignore_eos=False,
        skip_special_tokens=True,
        stop_token_ids=[tokenizer.eos_token_id],
        max_tokens=200,
    )

    results = llm.generate(inputs, sampling_params)

    for i, result in enumerate(results):
        output = result.outputs[0].text
        print(f"===================== Output {i} ==============================")
        print(output)
        print("===============================================================\n")


if __name__ == "__main__":
    fire.Fire(main)
