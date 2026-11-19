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

"""Replay image traffic with short and long outputs using vLLM's benchmark client.

Prepare one trace, then replay it against integrated and disaggregated endpoints.
Each endpoint gets an all-short control (--all-short) and a mixed-output run.
Restart servers between runs to avoid encoder-cache reuse across replays.
Synthetic tiled images exercise encoder load, not answer quality. Warm-up uses
separate images. --ignore-eos fixes output lengths for this controlled experiment.
Raw results belong outside the public repository.

    uv run python benchmark_interference.py prepare --trace /tmp/ec/trace.json
    uv run python benchmark_interference.py run --trace /tmp/ec/trace.json \
        --base-url http://localhost:8200 --result /tmp/ec/integrated-mixed.json
"""

import argparse
import asyncio
import base64
import hashlib
import json
import random
from pathlib import Path


def prepare(args) -> None:
    from PIL import Image

    if args.trace.exists():
        raise FileExistsError(args.trace)
    if args.requests < 2 or args.rate <= 0 or args.image_size < 1:
        raise ValueError("requests >= 2, rate > 0 and image-size > 0 are required")
    if not 0 < args.long_fraction < 1 or not 1 < args.short_tokens < args.long_tokens:
        raise ValueError(
            "Require 0 < long-fraction < 1 and 1 < short-tokens < long-tokens"
        )
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    image_dir = args.trace.parent / f"{args.trace.stem}_images"
    image_dir.mkdir()
    rng = random.Random(args.seed)
    background_count = round(args.requests * args.long_fraction)
    if not 0 < background_count < args.requests:
        raise ValueError("The trace must contain both short and background requests")
    cohorts = ["background"] * background_count + ["short"] * (
        args.requests - background_count
    )
    rng.shuffle(cohorts)
    requests = []
    timestamp = 0.0
    for index, cohort in enumerate(cohorts + ["warmup-short", "warmup-long"]):
        pixels = rng.randbytes(14 * 14 * 3)
        image = Image.frombytes("RGB", (14, 14), pixels).resize(
            (args.image_size, args.image_size), Image.Resampling.NEAREST
        )
        path = (image_dir / f"{index:06d}.png").resolve()
        image.save(path)
        if index and index < args.requests:
            timestamp += rng.expovariate(args.rate)
        requests.append(
            {
                "id": f"image-{index:06d}",
                "image": str(path),
                "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "cohort": cohort,
                "timestamp": timestamp,
                "output_tokens": args.long_tokens
                if cohort in ("background", "warmup-long")
                else args.short_tokens,
            }
        )
    trace = {
        "seed": args.seed,
        "rate": args.rate,
        "short_tokens": args.short_tokens,
        "long_tokens": args.long_tokens,
        "image_size": args.image_size,
        "long_fraction": args.long_fraction,
        "prompt": (
            "Describe the visible colors, shapes, arrangement and patterns "
            "in this image in detail."
        ),
        "requests": requests[: args.requests],
        "warmup": requests[args.requests :],
    }
    with args.trace.open("x") as stream:
        json.dump(trace, stream, indent=2)
    print(f"Prepared {args.requests} distinct images; arrival span {timestamp:.1f}s")


def summarize(trace: dict, result: dict, all_short: bool) -> dict:
    import numpy as np

    rows = trace["requests"]
    successful = [
        i
        for i, error in enumerate(result["errors"])
        if not error and result["output_lens"][i] > 0
    ]
    backgrounds = [i for i in successful if rows[i]["cohort"] == "background"]
    overlapping = [
        i
        for i in range(len(rows))
        if rows[i]["cohort"] == "short"
        and any(
            result["start_times"][j] + result["ttfts"][j]
            <= result["start_times"][i]
            < result["start_times"][j] + result["latencies"][j]
            for j in backgrounds
        )
    ]
    groups = {
        "all": list(range(len(rows))),
        "short": [i for i, r in enumerate(rows) if r["cohort"] == "short"],
        "background": [i for i, r in enumerate(rows) if r["cohort"] == "background"],
        "short_during_background_decode": overlapping,
    }
    summary = {}
    for name, indices in groups.items():
        ok = [i for i in indices if i in successful]
        values = {
            "e2e_ms": [result["latencies"][i] * 1000 for i in ok],
            "ttft_ms": [result["ttfts"][i] * 1000 for i in ok],
            "tpot_ms": [
                (result["latencies"][i] - result["ttfts"][i])
                * 1000
                / (result["output_lens"][i] - 1)
                for i in ok
                if result["output_lens"][i] > 1
            ],
        }
        summary[name] = {
            "requests": len(indices),
            "completed": len(ok),
            "failed": len(indices) - len(ok),
            "output_length_mismatches": sum(
                result["output_lens"][i]
                != (trace["short_tokens"] if all_short else rows[i]["output_tokens"])
                for i in ok
            ),
            **{
                metric: {
                    f"p{p}": float(np.percentile(samples, p)) for p in (50, 95, 99)
                }
                if samples
                else None
                for metric, samples in values.items()
            },
        }
    return summary


async def run(args) -> None:
    from vllm.benchmarks.datasets import SampleRequest
    from vllm.benchmarks.serve import TaskType, benchmark

    if args.result.exists():
        raise FileExistsError(args.result)
    trace_bytes = args.trace.read_bytes()
    trace = json.loads(trace_bytes)
    base_url = args.base_url.rstrip("/")

    def samples(rows):
        requests = []
        for row in rows:
            data = Path(row["image"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != row["image_sha256"]:
                raise ValueError(
                    f"Image changed since trace preparation: {row['image']}"
                )
            url = "data:image/png;base64," + base64.b64encode(data).decode()
            length = trace["short_tokens"] if args.all_short else row["output_tokens"]
            requests.append(
                SampleRequest(
                    prompt=trace["prompt"],
                    prompt_len=0,
                    expected_output_len=length,
                    multi_modal_data=[
                        {
                            "type": "image_url",
                            "image_url": {"url": url},
                            "uuid": row["image_sha256"],
                        }
                    ],
                    request_id=row["id"],
                    timestamp=row["timestamp"],
                )
            )
        return requests

    requests = samples(trace["requests"])
    warmup = samples(trace["warmup"])
    common = dict(
        task_type=TaskType.GENERATION,
        endpoint_type="openai-chat",
        api_url=base_url + "/v1/chat/completions",
        base_url=base_url,
        model_id=args.model,
        model_name=args.model,
        tokenizer=None,
        logprobs=None,
        request_rate=trace["rate"],
        burstiness=1.0,
        disable_tqdm=True,
        num_warmups=0,
        profile=False,
        selected_percentile_metrics=["e2el", "ttft", "tpot"],
        selected_percentiles=[50, 95, 99],
        ignore_eos=True,
        goodput_config_dict={},
        max_concurrency=None,
        lora_modules=None,
        extra_headers=None,
        extra_body={"temperature": 0},
    )
    warmed = await benchmark(
        input_requests=warmup, self_timed=False, ready_check_timeout_sec=60, **common
    )
    if warmed["failed"]:
        raise RuntimeError(
            "Warm-up failed; inspect the endpoint errors before measuring"
        )
    result = await benchmark(
        input_requests=requests, self_timed=True, ready_check_timeout_sec=0, **common
    )
    payload = {
        "trace_sha256": hashlib.sha256(trace_bytes).hexdigest(),
        "base_url": base_url,
        "model": args.model,
        "all_short": args.all_short,
        "summary": summarize(trace, result, args.all_short),
        "vllm_benchmark": result,
        "requests": [
            {k: row[k] for k in ("id", "cohort", "timestamp", "output_tokens")}
            for row in trace["requests"]
        ],
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    with args.result.open("x") as stream:
        json.dump(payload, stream, indent=2)
    print(json.dumps(payload["summary"], indent=2))
    print(f"Saved result to {args.result}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser(
        "prepare", help="Generate images and a replay trace"
    )
    prepare_parser.add_argument("--trace", type=Path, required=True)
    prepare_parser.add_argument("--requests", type=int, default=300)
    prepare_parser.add_argument("--rate", type=float, default=0.5)
    prepare_parser.add_argument("--seed", type=int, default=0)
    prepare_parser.add_argument("--long-fraction", type=float, default=0.3)
    prepare_parser.add_argument("--short-tokens", type=int, default=32)
    prepare_parser.add_argument("--long-tokens", type=int, default=256)
    prepare_parser.add_argument("--image-size", type=int, default=448)
    run_parser = commands.add_parser(
        "run", help="Replay using vLLM E2E/TTFT/TPOT metrics"
    )
    run_parser.add_argument("--trace", type=Path, required=True)
    run_parser.add_argument(
        "--base-url", required=True, help="Server root, without /v1"
    )
    run_parser.add_argument("--model", default="google/gemma-4-26B-A4B")
    run_parser.add_argument(
        "--all-short",
        action="store_true",
        help="Control: cap every output at the short length",
    )
    run_parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        asyncio.run(run(args))


if __name__ == "__main__":
    main()
