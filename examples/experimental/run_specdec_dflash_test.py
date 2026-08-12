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
"""DFlash speculative decoding smoke test.

DFlash drafts all speculative tokens in one forward pass of a small
block-diffusion drafter conditioned on target hidden states.
num_speculative_tokens must be the checkpoint's draft block_size - 1
(e.g. 15 for -b16 checkpoints, 7 for block_size 8); it is read from the
checkpoint config when not given explicitly.

Example:
    python run_specdec_dflash_test.py \
        --base-model Qwen/Qwen3-8B \
        --dflash-model z-lab/Qwen3-8B-DFlash-b16
"""

import argparse
import json
import os

os.environ["VLLM_RBLN_USE_VLLM_MODEL"] = "1"
os.environ["VLLM_RBLN_COMPILE_STRICT_MODE"] = "1"
os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
os.environ["VLLM_RBLN_ENABLE_WARM_UP"] = "1"

from huggingface_hub import hf_hub_download
from vllm import LLM, SamplingParams
from vllm.v1.metrics.reader import Counter, Vector

DEFAULT_BASE_MODEL_ID = "Qwen/Qwen3-8B"
DEFAULT_DFLASH_MODEL_ID = "z-lab/Qwen3-8B-DFlash-b16"
DEFAULT_PROMPTS = [
    "A robot may not injure a human being",
    "The capital of France is",
]


def _resolve_num_speculative_tokens(model_id: str) -> int:
    """Read block_size - 1 from the checkpoint config (top level or
    dflash_config, depending on the checkpoint)."""
    config_path = hf_hub_download(model_id, "config.json")
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    block_size = config.get("block_size") or (config.get("dflash_config") or {}).get(
        "block_size"
    )
    if not block_size:
        raise ValueError(
            f"Could not find block_size in {model_id} config.json; "
            "pass --num-speculative-tokens explicitly."
        )
    return block_size - 1


def _summarize_metrics(metrics, num_spec_tokens: int):
    num_drafts = 0
    num_draft_tokens = 0
    num_accepted_tokens = 0
    acceptance_counts = [0] * num_spec_tokens
    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_draft_tokens":
            assert isinstance(metric, Counter)
            num_draft_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens":
            assert isinstance(metric, Counter)
            num_accepted_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            for pos in range(len(metric.values)):
                acceptance_counts[pos] += metric.values[pos]
    return num_drafts, num_draft_tokens, num_accepted_tokens, acceptance_counts


def _print_outputs(outputs) -> None:
    print("-" * 50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"prompt: {prompt!r}")
        print(f"generated text: {generated_text!r}")
        print("-" * 50)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL_ID)
    parser.add_argument("--dflash-model", default=DEFAULT_DFLASH_MODEL_ID)
    parser.add_argument(
        "--num-speculative-tokens",
        type=int,
        default=None,
        help="Draft block_size - 1. Read from the checkpoint config when not given.",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument(
        "--num-gpu-blocks-override",
        type=int,
        default=None,
        help="Cap the KV cache block count. Without it, default KV sizing "
        "can consume the whole per-device memory budget and warmup OOMs "
        "on large targets.",
    )
    args = parser.parse_args()

    num_speculative_tokens = (
        args.num_speculative_tokens
        if args.num_speculative_tokens is not None
        else _resolve_num_speculative_tokens(args.dflash_model)
    )
    print(f"num_speculative_tokens: {num_speculative_tokens}")

    llm = LLM(
        model=args.base_model,
        max_model_len=args.max_model_len,
        block_size=args.block_size,
        enable_chunked_prefill=True,
        max_num_batched_tokens=256,
        max_num_seqs=4,
        num_gpu_blocks_override=args.num_gpu_blocks_override,
        speculative_config={
            "method": "dflash",
            "model": args.dflash_model,
            "num_speculative_tokens": num_speculative_tokens,
        },
        disable_log_stats=False,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    sampling_params = SamplingParams(temperature=0.1, top_p=0.9, max_tokens=128)
    outputs = llm.generate(DEFAULT_PROMPTS, sampling_params=sampling_params)
    _print_outputs(outputs)

    try:
        metrics = llm.get_metrics()
    except AssertionError:
        print("Failed to load metrics.")
        return

    total_num_output_tokens = sum(
        len(output.outputs[0].token_ids) for output in outputs
    )
    num_drafts, num_draft_tokens, num_accepted_tokens, acceptance_counts = (
        _summarize_metrics(metrics, num_speculative_tokens)
    )

    print("-" * 50)
    print(f"total_num_output_tokens: {total_num_output_tokens}")
    print(f"num_drafts: {num_drafts}")
    print(f"num_draft_tokens: {num_draft_tokens}")
    print(f"num_accepted_tokens: {num_accepted_tokens}")
    acceptance_length = 1 + (num_accepted_tokens / num_drafts) if num_drafts > 0 else 1
    print(f"mean acceptance length: {acceptance_length:.2f}")
    print("-" * 50)

    for pos, accepted_count in enumerate(acceptance_counts):
        acceptance_rate = accepted_count / num_drafts if num_drafts > 0 else 0
        print(f"acceptance at token {pos}: {acceptance_rate:.2f}")


if __name__ == "__main__":
    main()
