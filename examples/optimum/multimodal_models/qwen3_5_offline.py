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

# ---------------------------------------------------------------------------
# Qwen3.5 (vision-language) offline example.
#
# Qwen3.5 is a HYBRID decoder: most layers are `linear_attention` (GatedDeltaNet,
# carrying conv_state + recurrent_state) and a minority are gated `full_attention`
# (paged KV cache). It is essentially "Qwen3-VL without deepstack".
#
# Notes specific to Qwen3.5 (vs the shared qwen_vl_offline.py):
#   * FLASH attention is forced at compile time (see
#     vllm_rbln/.../compilation/multimodal/qwen.py::get_param_qwen3_5): the hybrid
#     backbone's multi-window prefill is mis-lowered by the eager op. Flash requires
#     `max_model_len >= 4096` and `block_size (kvcache_partition_len) >= 4096`, which
#     is why the defaults below are >= 4096.
#   * The linear-attention conv/recurrent state caches and their control masks are
#     handled entirely inside optimum-rbln's RBLNQwen3_5RuntimeModel — nothing extra
#     is passed here.
#   * END-TO-END COMPILE currently needs the fused GatedDeltaNet device op in
#     optimum-rbln/rebel. Until that lands, `LLM(...)` will fail during compilation
#     (cache miss). Once it lands this example runs unchanged; you can also point
#     `--model` at a pre-compiled artifact directory to skip compilation.
# ---------------------------------------------------------------------------

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

    arr_video_inputs = []
    arr_video_kwargs = []
    for i in range(batch_size):
        _, video_inputs, video_kwargs = process_vision_info(
            messages[i],
            return_video_kwargs=True,
            return_video_metadata=True,
        )
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
        for text, video_inputs, video_kwargs in zip(
            texts, arr_video_inputs, arr_video_kwargs
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
        image_inputs, _ = process_vision_info(messages[i])
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


def main(
    num_input_prompt: int = 1,
    # A native Qwen3.5 VL checkpoint (HF id or a local path). Qwen3.5 is supported
    # natively by transformers (no trust_remote_code needed). You can also pass a
    # pre-compiled artifact directory to skip compilation.
    model: str = "Qwen/Qwen3.5-4B-Instruct",
    modality: str = "image",  # "image" | "video"
):
    # number of devices per local rank for main module
    os.environ["VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK"] = "16"
    llm = LLM(
        model=model,
        # >= 4096 is required for the forced flash-attention path (Qwen3.5 hybrid).
        block_size=4096,
        max_model_len=8192,
        max_num_seqs=1,
        additional_config={
            "rbln_config": {
                "device": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                "visual": {
                    "device": [16],
                },
            }
        },
    )
    tokenizer = AutoTokenizer.from_pretrained(model)

    if modality == "video":
        inputs = generate_prompts_video(num_input_prompt, model)
    else:
        inputs = generate_prompts_image(num_input_prompt, model)

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
