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

"""Test-only helpers for speculative decoding e2e coverage."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file
from transformers import AutoConfig

DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-1B"
DEFAULT_MEDUSA_MODEL_ID = "MegaLearner/medusa_llama_3_2_1b_3_heads"

_VLLM_BLOCK_WEIGHT_RE = re.compile(r"^blocks\.\d+\.layers\.0\.weight$")
_EXT_WEIGHT_RE = re.compile(r"^(\d+)\.0\.linear\.weight$")
_EXT_BIAS_RE = re.compile(r"^(\d+)\.0\.linear\.bias$")
_EXT_LM_HEAD_RE = re.compile(r"^(\d+)\.1\.weight$")


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _is_vllm_medusa_config(config_dict: dict[str, Any]) -> bool:
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
        if (match := _EXT_WEIGHT_RE.match(key)) is not None:
            head_idx = int(match.group(1))
            converted[f"blocks.{head_idx}.layers.0.weight"] = value
            continue
        if (match := _EXT_BIAS_RE.match(key)) is not None:
            head_idx = int(match.group(1))
            converted[f"blocks.{head_idx}.layers.0.bias"] = value
            continue
        if (match := _EXT_LM_HEAD_RE.match(key)) is not None:
            head_idx = int(match.group(1))
            converted[f"lm_heads.{head_idx}.weight"] = value
            continue

    for head_idx in range(num_heads):
        required_keys = [
            f"blocks.{head_idx}.layers.0.weight",
            f"blocks.{head_idx}.layers.0.bias",
            f"lm_heads.{head_idx}.weight",
        ]
        missing = [key for key in required_keys if key not in converted]
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
            "Unsupported Medusa adapter config. Expected either vLLM Medusa "
            "config or external config containing medusa_num_heads/"
            "medusa_num_layers."
        )

    num_heads = int(source_config["medusa_num_heads"])
    num_layers = int(source_config["medusa_num_layers"])

    if num_layers != 1:
        raise ValueError(
            f"Only medusa_num_layers=1 is supported for conversion, got {num_layers}"
        )

    if source_config.get("model_type") == "medusa":
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
    converted_state = _map_external_medusa_state_dict(
        source_state,
        num_heads=num_heads,
    )

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
        "medusa_fc_bias": any("bias" in key for key in converted_state),
    }

    if out_config_path.exists() and out_weights_path.exists():
        existing = _load_json(out_config_path)
        if existing == out_config:
            return str(out_dir), num_heads

    save_file(converted_state, str(out_weights_path))
    out_config_path.write_text(json.dumps(out_config, indent=2), encoding="utf-8")
    return str(out_dir), num_heads


__all__ = [
    "DEFAULT_MEDUSA_MODEL_ID",
    "DEFAULT_MODEL_ID",
    "ensure_converted_medusa_adapter",
]
