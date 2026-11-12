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

import fire
import torch
from transformers import AutoProcessor
from vllm import LLM
from vllm.multimodal.utils import fetch_image

IMAGE_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
TEXTS = [
    "A woman playing with her dog on a beach at sunset.",
    "City skyline view from a high-rise building at night.",
]


def build_input(processor, text=None, image=None):
    content = []
    if image is not None:
        content.append({"type": "image"})
    if text is not None:
        content.append({"type": "text", "text": text})
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "Represent the user's input."}],
        },
        {"role": "user", "content": content},
    ]
    prompt = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    if image is None:
        return prompt
    return {"prompt": prompt, "multi_modal_data": {"image": image}}


def main(
    max_model_len: int = 8192,
    vit_max_seq_len: int = 16384,
    model: str = "Qwen/Qwen3-VL-Embedding-2B",
):
    llm = LLM(
        model=model,
        runner="pooling",
        block_size=4096,
        max_model_len=max_model_len,
        # optimum-rbln requires the vision encoder's patch budget per image.
        additional_config={
            "optimum_overrides": {"visual": {"max_seq_len": vit_max_seq_len}}
        },
    )
    processor = AutoProcessor.from_pretrained(model)

    inputs = [build_input(processor, text=text) for text in TEXTS]
    inputs.append(build_input(processor, image=fetch_image(IMAGE_URL)))
    outputs = llm.embed(inputs)

    embeddings = torch.stack([torch.tensor(o.outputs.embedding) for o in outputs])
    scores = embeddings[:-1] @ embeddings[-1]
    print(f"text-image scores: {scores.tolist()}")


if __name__ == "__main__":
    fire.Fire(main)
