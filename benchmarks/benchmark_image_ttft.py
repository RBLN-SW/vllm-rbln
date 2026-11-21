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
"""Measure single-request image TTFT on the RBLN vllm model path."""

import argparse
import json
import os
from pathlib import Path
from statistics import median

IMAGE_BUCKETS = [128, 256, 512, 1024, 2048, 4096]
PRESETS = {
    "qwen3_5": {
        "model": "Qwen/Qwen3.5-2B",
        "cases": [(352, 121), (704, 484), (2048, 4096)],
        "processor": {"max_pixels": 2048**2},
    },
    "glm5_next": {
        "model": "zai-org/GLM-5.3-Flash",
        "cases": [(308, 121), (616, 484), (1792, 4096)],
        "processor": {"max_image_tokens": 4096},
        "engine": {"kv_cache_dtype": "fp8", "block_size": 1024, "max_model_len": 16384},
    },
    "minimax_m3": {
        "model": "nvidia/MiniMax-M3-NVFP4",
        "cases": [(308, 121), (616, 484), (1792, 4096)],
        "processor": {"max_pixels": 1792**2},
        "engine": {"block_size": 512},
    },
    "gemma4": {
        "model": "google/gemma-4-26B-A4B-it",
        "cases": [(1024, 70), (1024, 280), (1024, 1120)],
        "processor": {"max_soft_tokens": 1120},
        "engine": {"block_size": 2048, "max_num_batched_tokens": 1152},
        "rbln": {
            "encoder_token_budgets": [70, 140, 280, 560, 1120],
            "prefill_chunk_size": [256, 1152],
        },
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(PRESETS), required=True)
    parser.add_argument(
        "--num-hidden-layers", type=int, default=4, help="Decoder layers (0 for all)"
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--out", type=Path, help="Write results to a new JSON file")
    args = parser.parse_args()
    if args.warmup < 0 or args.num_hidden_layers < 0 or args.runs < 1:
        parser.error("warmup and decoder layers must be nonnegative, and runs positive")
    if args.out and args.out.exists():
        parser.error(f"output file already exists: {args.out}")

    os.environ["VLLM_RBLN_NUM_HIDDEN_LAYERS"] = str(args.num_hidden_layers)

    import numpy as np
    from PIL import Image
    from vllm import LLM, SamplingParams

    from vllm_rbln.config import RBLNConfig

    preset = PRESETS[args.preset]
    engine = {
        "block_size": 256,
        "max_model_len": 8192,
        "max_num_batched_tokens": 256,
    } | preset.get("engine", {})
    llm = LLM(
        model=preset["model"],
        model_impl="vllm",
        tensor_parallel_size=1,
        max_num_seqs=1,
        enable_prefix_caching=False,
        async_scheduling=False,
        limit_mm_per_prompt={"image": 1, "video": 0, "audio": 0},
        mm_processor_kwargs=preset["processor"],
        mm_processor_cache_gb=0,
        disable_log_stats=False,
        additional_config=RBLNConfig(
            **preset.get("rbln", {"encoder_token_budgets": IMAGE_BUCKETS})
        ),
        **engine,
    )
    sampling = SamplingParams(temperature=0, max_tokens=8, ignore_eos=True)
    results = []
    for side, image_tokens in preset["cases"]:
        rng = np.random.default_rng(0)
        image = Image.fromarray(rng.integers(0, 256, (side, side, 3), dtype=np.uint8))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_pil", "image_pil": image},
                    {"type": "text", "text": "What is in this image?"},
                ],
            }
        ]
        # Gemma 4 selects the image token budget through the processor.
        processor = (
            {"max_soft_tokens": image_tokens} if args.preset == "gemma4" else None
        )
        ttft_ms = []
        for request in range(args.warmup + args.runs):
            llm.llm_engine.reset_encoder_cache()
            (output,) = llm.chat(
                messages, sampling, use_tqdm=False, mm_processor_kwargs=processor
            )
            if request >= args.warmup:
                metrics = output.metrics
                if metrics is None or metrics.first_token_latency <= 0:
                    raise RuntimeError(
                        f"Missing TTFT for {side}px / {image_tokens} tokens"
                    )
                ttft_ms.append(metrics.first_token_latency * 1000)

        median_ttft_ms = median(ttft_ms)
        print(f"{image_tokens:4d} image tokens | median TTFT {median_ttft_ms:.2f} ms")
        results.append(
            {
                "image_side_px": side,
                "image_tokens": image_tokens,
                "median_ttft_ms": median_ttft_ms,
            }
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x") as output_file:
            json.dump(
                {
                    "model": preset["model"],
                    "num_hidden_layers": args.num_hidden_layers,
                    "warmup": args.warmup,
                    "runs": args.runs,
                    "results": results,
                },
                output_file,
                indent=2,
            )
            output_file.write("\n")


if __name__ == "__main__":
    main()
