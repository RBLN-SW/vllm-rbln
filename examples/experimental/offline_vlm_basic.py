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

import argparse
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--block-size", type=int, default=1024)
    return parser.parse_args()


def main():
    args = parse_args()

    image_dir = Path(__file__).with_name("images")
    image_paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    images = [(path, Image.open(path).convert("RGB")) for path in image_paths]
    processor = AutoProcessor.from_pretrained(args.model)

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        block_size=args.block_size,
        enable_chunked_prefill=True,
        max_num_batched_tokens=128,
        gpu_memory_utilization=0.9,
        profiler_config={
            "profiler": "torch",
            "torch_profiler_dir": "./vllm_profile_0624",
        },
    )

    requests = []
    for _, image in images:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe this image in one sentence."},
                ],
            }
        ]
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        requests.append(
            {
                "prompt": prompt,
                "multi_modal_data": {"image": image},
                "mm_processor_kwargs": {
                    "min_pixels": 1024 * 14 * 14,
                    "max_pixels": 5120 * 14 * 14,
                },
            }
        )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=64)

    warmup_outputs = llm.generate([requests[0]], sampling_params)
    print(f"{images[0][0].name} (warmup): {warmup_outputs[0].outputs[0].text}")

    if len(requests) == 1:
        return

    # requests = requests[-4:]
    llm.start_profile()
    outputs = llm.generate(requests[-4:], sampling_params)
    llm.stop_profile()

    for (image_path, _), output in zip(images[1:], outputs):
        print(f"{image_path.name}: {output.outputs[0].text}")


if __name__ == "__main__":
    main()
