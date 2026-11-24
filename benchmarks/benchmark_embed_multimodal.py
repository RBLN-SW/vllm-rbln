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

"""Measure image encoder latency from input preparation to embedding readiness."""

import argparse
import json
import os
from functools import wraps
from pathlib import Path
from statistics import median
from time import perf_counter

from benchmark_image_ttft import (
    PRESETS,
    create_image_llm,
    gemma_image_tokens,
    image_request,
)


class EncoderE2EWorkerExtension:
    """Time embedding calls between the start and finish RPCs."""

    def start_encoder_e2e_measurement(self) -> None:
        import torch

        model = self.model_runner.model
        embed_multimodal = model.embed_multimodal
        samples_ms: list[float] = []
        self._encoder_e2e_original = embed_multimodal
        self._encoder_e2e_samples_ms = samples_ms

        @wraps(embed_multimodal)
        def measure(*args, **kwargs):
            start = perf_counter()
            embeddings = embed_multimodal(*args, **kwargs)
            tensors = (
                (embeddings,) if isinstance(embeddings, torch.Tensor) else embeddings
            )
            for device in {tensor.device for tensor in tensors}:
                if device.type == "rbln":
                    torch.rbln.synchronize(device)
            samples_ms.append((perf_counter() - start) * 1000)
            return embeddings

        model.embed_multimodal = measure

    def finish_encoder_e2e_measurement(self) -> list[float]:
        self.model_runner.model.embed_multimodal = self._encoder_e2e_original
        samples_ms = self._encoder_e2e_samples_ms
        del self._encoder_e2e_original, self._encoder_e2e_samples_ms
        return samples_ms


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
    os.environ["VLLM_RBLN_METRICS"] = "0"

    from vllm import SamplingParams

    preset = PRESETS[args.preset]
    llm = create_image_llm(
        preset,
        worker_extension_cls="benchmark_embed_multimodal.EncoderE2EWorkerExtension",
    )
    sampling = SamplingParams(temperature=0, max_tokens=1)
    results = []
    for side, token_setting in preset["cases"]:
        max_soft_tokens = token_setting if args.preset == "gemma4" else None
        messages, processor = image_request(side, max_soft_tokens)
        for request in range(args.warmup + args.runs):
            if request == args.warmup:
                llm.collective_rpc("start_encoder_e2e_measurement")
            llm.llm_engine.reset_encoder_cache()
            (output,) = llm.chat(
                messages, sampling, use_tqdm=False, mm_processor_kwargs=processor
            )
        (samples,) = llm.collective_rpc("finish_encoder_e2e_measurement")
        if len(samples) != args.runs:
            raise RuntimeError(
                f"Expected {args.runs} encoder E2E samples for case {token_setting}, "
                f"got {len(samples)}"
            )

        median_ms = median(samples)
        image_tokens = token_setting
        budget_label = ""
        result = {
            "image_side_px": side,
            "median_encoder_e2e_ms": median_ms,
            "encoder_e2e_samples_ms": samples,
        }
        if max_soft_tokens is not None:
            image_tokens = gemma_image_tokens(llm, output, max_soft_tokens)
            result["max_soft_tokens"] = max_soft_tokens
            budget_label = f" (max_soft_tokens={max_soft_tokens})"
        result["image_tokens"] = image_tokens
        print(
            f"{image_tokens:4d} image tokens{budget_label} | "
            f"ENCODER E2E {median_ms:.2f} ms"
        )
        results.append(result)

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
