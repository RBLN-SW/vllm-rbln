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
# vLLM(v0.10.2) bug: speculative decoding works only in multi-processing.
# os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoConfig
from vllm import LLM, SamplingParams
from vllm.v1.metrics.reader import Counter, Vector

DEFAULT_METHOD = "eagle"
DEFAULT_EAGLE_BASEMODEL_ID = "meta-llama/Llama-3.2-1B"
DEFAULT_EAGLE_MODEL_ID = "JKroller/llama3.2-1b-eagle"
DEFAULT_EAGLE3_BASEMODEL_ID = "Qwen/Qwen3-1.7B"
DEFAULT_EAGLE3_MODEL_ID = "AngelSlim/Qwen3-1.7B_eagle3"
NUM_SPECULATIVE_TOKENS = 3
DEFAULT_PROMPTS = [
    "A robot may not injure a human being",
    "The capital of France is",
]


def _load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _remote_has_config(model_id: str) -> bool:
    try:
        hf_hub_download(model_id, "config.json")
        return True
    except Exception:
        return False


def _infer_num_draft_layers(weight_keys: list[str]) -> int:
    layer_indices: set[int] = set()
    layer_pattern = re.compile(r"^(?:model\.)?layers\.(\d+)\.")
    for key in weight_keys:
        if (m := layer_pattern.match(key)) is not None:
            layer_indices.add(int(m.group(1)))
    if not layer_indices:
        raise ValueError("Could not infer Eagle draft layer count from weights.")
    return max(layer_indices) + 1


def _map_external_eagle_state_dict(
    source_state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    converted: dict[str, torch.Tensor] = {}
    for key, value in source_state_dict.items():
        new_key = key
        if new_key.startswith("model."):
            new_key = new_key[len("model.") :]
        if new_key == "fusion.fc.bias":
            # vLLM Eagle Llama uses fc without bias.
            continue
        if new_key.startswith("fusion.fc."):
            new_key = f"fc.{new_key[len('fusion.fc.') :]}"
        converted[new_key] = value
    return converted


def _validate_converted_state_dict(
    state_dict: dict[str, torch.Tensor],
    num_draft_layers: int,
) -> None:
    required_keys = {"fc.weight"}
    required_layer_suffixes = (
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
        "post_attention_layernorm.weight",
    )
    for layer_idx in range(num_draft_layers):
        for suffix in required_layer_suffixes:
            required_keys.add(f"layers.{layer_idx}.{suffix}")

    missing = [key for key in sorted(required_keys) if key not in state_dict]
    if missing:
        raise ValueError(f"Missing required Eagle tensors after conversion: {missing}")


def ensure_converted_eagle_adapter(
    *,
    eagle_model_id: str,
    base_model_id: str,
) -> str:
    model_path = Path(eagle_model_id)
    if model_path.exists() and (model_path / "config.json").is_file():
        return str(model_path)
    if _remote_has_config(eagle_model_id):
        return eagle_model_id

    source_weights_path = hf_hub_download(eagle_model_id, "model.safetensors")
    snapshot_dir = Path(source_weights_path).parent
    out_dir = snapshot_dir / "vllm_converted_eagle"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_config_path = out_dir / "config.json"
    out_weights_path = out_dir / "model.safetensors"

    with safe_open(source_weights_path, framework="pt", device="cpu") as f:
        source_keys = list(f.keys())
    num_draft_layers = _infer_num_draft_layers(source_keys)

    base_config = AutoConfig.from_pretrained(base_model_id)
    out_config = base_config.to_dict()
    out_config["num_hidden_layers"] = num_draft_layers
    out_config["draft_vocab_size"] = int(base_config.vocab_size)
    out_config["dtype"] = "bfloat16"
    out_config["torch_dtype"] = "bfloat16"

    if out_config_path.exists() and out_weights_path.exists():
        existing = _load_json(out_config_path)
        if existing == out_config:
            return str(out_dir)

    source_state_dict = load_file(source_weights_path)
    converted_state_dict = _map_external_eagle_state_dict(source_state_dict)
    _validate_converted_state_dict(converted_state_dict, num_draft_layers)
    save_file(converted_state_dict, str(out_weights_path))
    out_config_path.write_text(json.dumps(out_config, indent=2), encoding="utf-8")
    return str(out_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Medusa speculative decoding test on vLLM-RBLN."
    )
    parser.add_argument("--method", choices=["eagle", "eagle3"], default="eagle")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--eagle-model-id", default=None)
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


def main():
    args = parse_args()

    method = args.method
    base_model_id = args.model_id or (
        DEFAULT_EAGLE_BASEMODEL_ID if method == "eagle" else DEFAULT_EAGLE3_BASEMODEL_ID
    )
    eagle_model_id = args.eagle_model_id or (
        DEFAULT_EAGLE_MODEL_ID if method == "eagle" else DEFAULT_EAGLE3_MODEL_ID
    )
    eagle_model_id = ensure_converted_eagle_adapter(
        eagle_model_id=eagle_model_id,
        base_model_id=base_model_id,
    )

    # Create an LLM.
    llm = LLM(
        model=base_model_id,
        max_model_len=2048,
        block_size=1024,
        enable_chunked_prefill=True,
        max_num_batched_tokens=256,
        max_num_seqs=4,
        speculative_config={
            "method": method,
            "model": eagle_model_id,
            "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
        },
        disable_log_stats=False,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=0.5,
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
        _summarize_metrics(metrics, NUM_SPECULATIVE_TOKENS)
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
