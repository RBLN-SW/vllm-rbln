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
import json
import os
import re
from pathlib import Path
from typing import Any

os.environ["RBLN_USE_CUSTOM_KERNEL"] = "1"
os.environ["VLLM_RBLN_USE_VLLM_MODEL"] = "1"
os.environ["VLLM_RBLN_COMPILE_STRICT_MODE"] = "1"
os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
os.environ["VLLM_RBLN_ENABLE_WARM_UP"] = "1"
os.environ["VLLM_RBLN_SAMPLER"] = "0"

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file
from transformers import AutoConfig
from vllm import LLM, SamplingParams
from vllm.v1.metrics.reader import Counter, Vector

DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-1B"
DEFAULT_MEDUSA_MODEL_ID = "MegaLearner/medusa_llama_3_2_1b_3_heads"
DEFAULT_PROMPTS = [
    "A robot may not injure a human being",
    "The capital of France is",
]

_VLLM_BLOCK_WEIGHT_RE = re.compile(r"^blocks\.\d+\.layers\.0\.weight$")
_EXT_WEIGHT_RE = re.compile(r"^(\d+)\.0\.linear\.weight$")
_EXT_BIAS_RE = re.compile(r"^(\d+)\.0\.linear\.bias$")
_EXT_LM_HEAD_RE = re.compile(r"^(\d+)\.1\.weight$")


def _load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _is_vllm_medusa_config(config_dict: dict) -> bool:
    return config_dict.get("model_type") == "medusa" and isinstance(
        config_dict.get("num_heads"), int
    )


def _is_vllm_medusa_state_dict(state_dict: dict[str, torch.Tensor]) -> bool:
    return any(_VLLM_BLOCK_WEIGHT_RE.match(key) for key in state_dict)


def _map_external_medusa_state_dict(
    source_state_dict: dict[str, torch.Tensor],
    num_heads: int,
) -> dict[str, torch.Tensor]:
    converted: dict[str, torch.Tensor] = {}

    for key, value in source_state_dict.items():
        if (m := _EXT_WEIGHT_RE.match(key)) is not None:
            head_idx = int(m.group(1))
            converted[f"blocks.{head_idx}.layers.0.weight"] = value
            continue
        if (m := _EXT_BIAS_RE.match(key)) is not None:
            head_idx = int(m.group(1))
            converted[f"blocks.{head_idx}.layers.0.bias"] = value
            continue
        if (m := _EXT_LM_HEAD_RE.match(key)) is not None:
            head_idx = int(m.group(1))
            converted[f"lm_heads.{head_idx}.weight"] = value
            continue

    for head_idx in range(num_heads):
        required = [
            f"blocks.{head_idx}.layers.0.weight",
            f"blocks.{head_idx}.layers.0.bias",
            f"lm_heads.{head_idx}.weight",
        ]
        missing = [k for k in required if k not in converted]
        if missing:
            raise ValueError(
                f"Missing converted tensors for head {head_idx}: {missing}"
            )

    return converted


def ensure_converted_medusa_adapter(
    *,
    medusa_model_id: str,
    base_model_id: str,
) -> tuple[str, int]:
    config_path = hf_hub_download(medusa_model_id, "config.json")
    source_config = _load_json(config_path)
    if _is_vllm_medusa_config(source_config):
        return medusa_model_id, int(source_config["num_heads"])

    if (
        "medusa_num_heads" not in source_config
        or "medusa_num_layers" not in source_config
    ):
        raise ValueError(
            "Unsupported Medusa adapter config. Expected either vLLM Medusa config "
            "or external config containing medusa_num_heads/medusa_num_layers."
        )

    num_heads = int(source_config["medusa_num_heads"])
    num_layers = int(source_config["medusa_num_layers"])

    if num_layers != 1:
        raise ValueError(
            f"Only medusa_num_layers=1 is supported for conversion, got {num_layers}"
        )

    if "model_type" in source_config and source_config["model_type"] == "medusa":
        return medusa_model_id, num_heads

    lm_head_path = hf_hub_download(medusa_model_id, "medusa_lm_head.pt")
    snapshot_dir = Path(lm_head_path).parent
    source_state = torch.load(lm_head_path, map_location="cpu")
    if not isinstance(source_state, dict):
        raise ValueError("Expected dict-like state_dict in medusa_lm_head.pt")

    if _is_vllm_medusa_state_dict(source_state):
        return medusa_model_id, num_heads

    out_dir = snapshot_dir / "vllm_converted_medusa"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_config_path = out_dir / "config.json"
    out_weights_path = out_dir / "model.safetensors"

    base_config = AutoConfig.from_pretrained(base_model_id)
    hidden_size = int(base_config.hidden_size)
    vocab_size = int(base_config.vocab_size)

    converted_state = _map_external_medusa_state_dict(source_state, num_heads=num_heads)

    out_config = {
        "model_type": "medusa",
        "architectures": ["MedusaModel"],
        "dtype": "bfloat16",
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
        "num_heads": num_heads,
        "num_hidden_layers": num_layers,
        "truncated_vocab_size": vocab_size,
        "original_lm_head": False,
        "medusa_fc_bias": any("bias" in k for k in converted_state),
    }

    if out_config_path.exists() and out_weights_path.exists():
        existing = _load_json(out_config_path)
        if existing == out_config:
            return str(out_dir), num_heads

    save_file(converted_state, str(out_weights_path))
    out_config_path.write_text(json.dumps(out_config, indent=2), encoding="utf-8")
    return str(out_dir), num_heads


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Medusa speculative decoding test on vLLM-RBLN."
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--medusa-model-id", default=DEFAULT_MEDUSA_MODEL_ID)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    return parser.parse_args()


def _print_outputs(outputs: list[Any]) -> None:
    for output in outputs:
        print("-" * 50)
        print(f"prompt: {output.prompt}")
        print(f"generated text: {output.outputs[0].text}")
        print("-" * 50)


def _summarize_metrics(
    metrics: list[Any], num_speculative_tokens: int
) -> tuple[int, int, int, list[int]]:
    num_drafts = 0
    num_draft_tokens = 0
    num_accepted_tokens = 0
    acceptance_counts = [0] * num_speculative_tokens

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
            num_positions = min(len(metric.values), len(acceptance_counts))
            for pos in range(num_positions):
                acceptance_counts[pos] += metric.values[pos]

    return num_drafts, num_draft_tokens, num_accepted_tokens, acceptance_counts


def main() -> None:
    args = parse_args()
    medusa_model_id, num_speculative_tokens = ensure_converted_medusa_adapter(
        medusa_model_id=args.medusa_model_id,
        base_model_id=args.model_id,
    )

    llm = LLM(
        model=args.model_id,
        max_model_len=2048,
        block_size=1024,
        enable_chunked_prefill=True,
        max_num_batched_tokens=256,
        max_num_seqs=4,
        speculative_config={
            "method": "medusa",
            "model": medusa_model_id,
            "num_speculative_tokens": num_speculative_tokens,
        },
        disable_log_stats=False,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=0.8,
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
