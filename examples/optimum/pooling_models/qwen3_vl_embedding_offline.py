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

# The inputs and embed() calls follow upstream vLLM's Qwen3-VL-Embedding example:
# https://github.com/vllm-project/vllm/blob/main/examples/pooling/embed/vision_embedding_offline.py

import fire
from vllm import LLM
from vllm.multimodal.utils import fetch_image
from vllm.utils.print_utils import print_embeddings

image_url = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/cat_snow.jpg"
text = "A cat standing in the snow."

default_instruction = "Represent the user's input."
image_placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
prompt_text = (
    f"<|im_start|>system\n{default_instruction}<|im_end|>\n"
    f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
)
prompt_image = (
    f"<|im_start|>system\n{default_instruction}<|im_end|>\n"
    f"<|im_start|>user\n{image_placeholder}<|im_end|>\n<|im_start|>assistant\n"
)
prompt_image_text = (
    f"<|im_start|>system\n{default_instruction}<|im_end|>\n"
    f"<|im_start|>user\n{image_placeholder}{text}<|im_end|>\n<|im_start|>assistant\n"
)


def main(
    max_model_len: int = 8192,
    vit_max_seq_len: int = 16384,
    model: str = "Qwen/Qwen3-VL-Embedding-2B",
):
    multi_modal_data = {"image": fetch_image(image_url)}
    # Unlike upstream, images are not pre-resized with qwen-vl-utils: the
    # processor's own max_pixels keeps each image within vit_max_seq_len.
    llm = LLM(
        model=model,
        runner="pooling",
        block_size=4096,
        max_model_len=max_model_len,
        limit_mm_per_prompt={"image": 1},
        # optimum-rbln requires the vision encoder's patch budget per image.
        additional_config={
            "optimum_overrides": {"visual": {"max_seq_len": vit_max_seq_len}}
        },
    )

    print("Text embedding output:")
    outputs = llm.embed(prompt_text, use_tqdm=False)
    print_embeddings(outputs[0].outputs.embedding)

    print("Image embedding output:")
    outputs = llm.embed(
        {
            "prompt": prompt_image,
            "multi_modal_data": multi_modal_data,
        },
        use_tqdm=False,
    )
    print_embeddings(outputs[0].outputs.embedding)

    print("Image+Text embedding output:")
    outputs = llm.embed(
        {
            "prompt": prompt_image_text,
            "multi_modal_data": multi_modal_data,
        },
        use_tqdm=False,
    )
    print_embeddings(outputs[0].outputs.embedding)


if __name__ == "__main__":
    fire.Fire(main)
