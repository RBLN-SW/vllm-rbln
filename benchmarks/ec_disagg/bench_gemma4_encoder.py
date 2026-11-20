#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
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

"""Sweep online multimodal load against one already-running topology."""

import argparse
import csv
import hashlib
import itertools
import json
import math
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path


def run_vllm_bench():
    from vllm.benchmarks.datasets import datasets
    from vllm.entrypoints.cli.main import main as vllm_main

    process_image = datasets.process_image

    def process_image_with_uuid(*args, **kwargs):
        item = process_image(*args, **kwargs)
        # Match the EPD proxy's content ID so repeated images reuse the same key.
        image_url = item["image_url"]["url"]
        return {**item, "uuid": hashlib.sha256(image_url.encode("utf-8")).hexdigest()}

    datasets.process_image = process_image_with_uuid
    sys.argv = [sys.argv[0], "bench", "serve", *sys.argv[2:]]
    vllm_main()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--topology",
        required=True,
        choices=[
            "baseline",
            "1e1pd",
            "1e3pd",
            "2e2pd",
        ],
    )
    parser.add_argument("--base-url", required=True, help="Service origin, without /v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="google/gemma-4-26B-A4B")
    parser.add_argument(
        "--served-model-name", help="API alias if different from --model"
    )
    parser.add_argument("--tokenizer", help="Tokenizer ID or local directory")
    parser.add_argument("--qps", nargs="+", type=float, default=[1, 2, 4, 8])
    parser.add_argument("--images", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--input-tokens", nargs="+", type=int, default=[400])
    parser.add_argument("--output-tokens", type=int, default=150)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--min-prompts", type=int, default=1000)
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--warmups", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--case-timeout", type=float, default=1800)
    parser.add_argument("--ttft-ms", type=float, default=20_000)
    parser.add_argument("--tpot-ms", type=float, default=100)
    parser.add_argument("--max-failure-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    positive = (
        *args.qps,
        *args.images,
        *args.input_tokens,
        args.image_size,
        args.min_prompts,
        args.seconds,
        args.repeats,
        args.case_timeout,
        args.ttft_ms,
        args.tpot_ms,
    )
    if any(not math.isfinite(value) or value <= 0 for value in positive):
        parser.error("rates, dimensions, durations, and repetitions must be positive")
    if args.output_tokens < 2 or args.warmups < 0 or not 0 <= args.max_failure_rate < 1:
        parser.error(
            "use at least two output tokens, nonnegative warmups, "
            "and a failure rate in [0, 1)"
        )

    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    settings = vars(args).copy()
    for key in ("topology", "base_url", "output_dir"):
        del settings[key]
    settings["vllm_version"] = version("vllm")
    settings["image_uuid_scheme"] = "sha256-image-url"
    manifest = root / "workload.json"
    if manifest.exists() and json.loads(manifest.read_text()) != settings:
        parser.error("workload differs from the existing run. Use a new --output-dir.")
    manifest.write_text(json.dumps(settings, indent=2) + "\n")

    topology_dir = root / args.topology
    topology_dir.mkdir(exist_ok=True)
    origin = args.base_url.rstrip("/")
    endpoint = topology_dir / "endpoint.txt"
    if endpoint.exists() and endpoint.read_text().strip() != origin:
        parser.error(
            "topology URL differs from the existing run. Use a new --output-dir."
        )
    endpoint.write_text(origin + "\n")

    cases = itertools.product(
        sorted(set(args.input_tokens)),
        sorted(set(args.images)),
        sorted(set(args.qps)),
        range(1, args.repeats + 1),
    )
    for input_tokens, images, qps, repeat in cases:
        case = f"in{input_tokens}-img{images}-qps{qps:.16g}-r{repeat}"
        final_path = topology_dir / f"{case}.json"
        if final_path.exists():
            continue
        pending_path = topology_dir / f"{case}.pending.json"
        log_path = topology_dir / f"{case}.log"
        prompts = max(args.min_prompts, math.ceil(args.seconds * qps))
        digest = hashlib.sha256(f"{args.seed}/{case}".encode()).digest()
        seed = int.from_bytes(digest[:4], "big")
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--run-vllm-bench",
            "--base-url",
            origin,
            "--endpoint",
            "/v1/chat/completions",
            "--backend",
            "openai-chat",
            "--model",
            args.model,
            "--tokenizer",
            args.tokenizer or args.model,
            "--dataset-name",
            "random-mm",
            "--random-input-len",
            str(input_tokens),
            "--random-output-len",
            str(args.output_tokens),
            "--random-range-ratio",
            "0",
            "--random-mm-base-items-per-request",
            str(images),
            "--random-mm-num-mm-items-range-ratio",
            "0",
            "--random-mm-limit-mm-per-prompt",
            json.dumps({"image": images, "video": 0}),
            "--random-mm-bucket-config",
            f"{{({args.image_size}, {args.image_size}, 1): 1.0}}",
            "--request-rate",
            str(qps),
            "--num-prompts",
            str(prompts),
            "--seed",
            str(seed),
            "--ready-check-timeout-sec",
            "60",
            "--num-warmups",
            str(args.warmups),
            "--ignore-eos",
            "--disable-tqdm",
            "--percentile-metrics",
            "ttft,tpot,itl,e2el",
            "--metric-percentiles",
            "50,95,99",
            "--goodput",
            f"ttft:{args.ttft_ms}",
            f"tpot:{args.tpot_ms}",
            "--save-result",
            "--save-detailed",
            "--result-dir",
            str(topology_dir),
            "--result-filename",
            pending_path.name,
            "--metadata",
            f"topology={args.topology}",
            f"images={images}",
            f"text_input_tokens={input_tokens}",
            f"repeat={repeat}",
            f"seed={seed}",
        ]
        if args.served_model_name:
            command.extend(["--served-model-name", args.served_model_name])
        print(f"{args.topology} {case} ({prompts} requests)", flush=True)
        try:
            with log_path.open("w") as log:
                subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=args.case_timeout,
                )
            pending_path.replace(final_path)
        except (subprocess.SubprocessError, FileNotFoundError) as exc:
            parser.exit(1, f"{case} failed: {exc}\nSee {log_path}\n")

        write_summary(root, args)

    write_summary(root, args)
    print(f"Summary: {root / 'summary.csv'}")


def write_summary(root, args):
    rows = []
    for path in sorted(root.glob("*/in*-img*-qps*-r*.json")):
        if path.name.endswith((".pending.json", ".pytorch.json")):
            continue
        result = json.loads(path.read_text())
        completed = result["completed"]
        failed = result["failed"]
        failure_rate = failed / result["num_prompts"]
        output_length_mismatches = completed - result["output_lens"].count(
            args.output_tokens
        )
        ttft = result["p99_ttft_ms"]
        tpot = result["p99_tpot_ms"]
        rows.append(
            {
                "topology": result["topology"],
                "text_input_tokens": int(result["text_input_tokens"]),
                "images": int(result["images"]),
                "target_qps": result["request_rate"],
                "repeat": int(result["repeat"]),
                "completed": completed,
                "failed": failed,
                "failure_rate": failure_rate,
                "output_length_mismatches": output_length_mismatches,
                "achieved_qps": result["request_throughput"],
                "request_goodput_qps": result["request_goodput"],
                "p99_ttft_ms": ttft,
                "p99_tpot_ms": tpot,
                "p99_itl_ms": result["p99_itl_ms"],
                "p99_e2el_ms": result["p99_e2el_ms"],
                "slo_pass": (
                    completed > 0
                    and output_length_mismatches == 0
                    and completed + failed == result["num_prompts"]
                    and failure_rate <= args.max_failure_rate
                    and 0 < ttft <= args.ttft_ms
                    and 0 < tpot <= args.tpot_ms
                ),
                "result_json": str(path.relative_to(root)),
            }
        )
    sort_keys = ("topology", "text_input_tokens", "images", "target_qps", "repeat")
    rows.sort(key=lambda row: tuple(row[key] for key in sort_keys))
    with (root / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--run-vllm-bench"]:
        run_vllm_bench()
    else:
        main()
