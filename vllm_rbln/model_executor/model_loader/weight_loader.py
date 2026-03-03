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

from collections.abc import Iterable

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.models import (deepseek_v2, llama, llama4, qwen2,
                                        qwen2_moe, qwen3_moe, minimax_m2,
                                        utils)

logger = init_logger(__name__)

# Following isort, docstring requires a dummy line
"""
[RBLN] This is only used to set the number of layers to load using
the `hf_override` configuration.
Everything else is the same as the original vLLM code.
This code will probably be deprecated in the future.
Therefore, do not implement any logic here.
"""


def load_llama_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    stacked_params_mapping = [
        # (param_name, shard_name, shard_id)
        (".qkv_proj", ".q_proj", "q"),
        (".qkv_proj", ".k_proj", "k"),
        (".qkv_proj", ".v_proj", "v"),
        (".gate_up_proj", ".gate_proj", 0),
        (".gate_up_proj", ".up_proj", 1),
    ]
    params_dict = dict(self.named_parameters())
    loaded_params: set[str] = set()
    for name, loaded_weight in weights:
        """
        [RBLN] Skips loading of layers greater than `num_hidden_layers`.
        This must be modified to more graceful code in the future.
        """
        if name.startswith("layers"):
            layer_idx = int(name.split(".")[1])
            if layer_idx >= self.config.num_hidden_layers:
                continue
        #######
        if "rotary_emb.inv_freq" in name:
            continue
        if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
            # Models trained using ColossalAI may include these tensors in
            # the checkpoint. Skip them.
            continue
        if self.quant_config is not None and (
            scale_name := self.quant_config.get_cache_scale(name)
        ):
            # Loading kv cache quantization scales
            param = params_dict[scale_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            loaded_weight = (
                loaded_weight if loaded_weight.dim() == 0 else loaded_weight[0]
            )
            weight_loader(param, loaded_weight)
            loaded_params.add(scale_name)
            continue
        if "scale" in name:
            # Remapping the name of FP8 kv-scale.
            name = maybe_remap_kv_scale_name(name, params_dict)
            if name is None:
                continue
        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name not in name:
                continue
            name = name.replace(weight_name, param_name)
            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue

            if utils.is_pp_missing_parameter(name, self):
                continue

            param = params_dict[name]
            weight_loader = param.weight_loader
            weight_loader(param, loaded_weight, shard_id)
            break
        else:
            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue

            if utils.is_pp_missing_parameter(name, self):
                continue

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
        loaded_params.add(name)
    return loaded_params


def load_qwen2_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    stacked_params_mapping = [
        # (param_name, shard_name, shard_id)
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]
    params_dict = dict(self.named_parameters(remove_duplicate=False))
    loaded_params: set[str] = set()
    for name, loaded_weight in weights:
        """
        [RBLN] Skips loading of layers greater than `num_hidden_layers`.
        This must be modified to more graceful code in the future.
        """
        if name.startswith("layers"):
            layer_idx = int(name.split(".")[1])
            if layer_idx >= self.config.num_hidden_layers:
                continue
        #######

        if "rotary_emb.inv_freq" in name:
            continue
        if self.quant_config is not None and (
            scale_name := self.quant_config.get_cache_scale(name)
        ):
            # Loading kv cache quantization scales
            param = params_dict[scale_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            loaded_weight = (
                loaded_weight if loaded_weight.dim() == 0 else loaded_weight[0]
            )
            weight_loader(param, loaded_weight)
            loaded_params.add(scale_name)
            continue
        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name not in name:
                continue
            name = name.replace(weight_name, param_name)
            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue
            if utils.is_pp_missing_parameter(name, self):
                continue
            param = params_dict[name]
            weight_loader = param.weight_loader
            weight_loader(param, loaded_weight, shard_id)
            break
        else:
            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue
            # Remapping the name of FP8 kv-scale.
            name = maybe_remap_kv_scale_name(name, params_dict)
            if name is None:
                continue
            if utils.is_pp_missing_parameter(name, self):
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
        loaded_params.add(name)
    return loaded_params


def load_qwen3moe_weights(
    self, weights: Iterable[tuple[str, torch.Tensor]]
) -> set[str]:
    stacked_params_mapping = [
        # (param_name, shard_name, shard_id)
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]

    # Params for weights, fp8 weight scales, fp8 activation scales
    # (param_name, weight_name, expert_id, shard_id)
    expert_params_mapping = FusedMoE.make_expert_params_mapping(
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
        num_experts=self.config.num_experts,
    )

    params_dict = dict(self.named_parameters())
    loaded_params: set[str] = set()
    for name, loaded_weight in weights:
        """
        [RBLN] Skips loading of layers greater than `num_hidden_layers`.
        This must be modified to more graceful code in the future.
        """
        if name.startswith("layers"):
            layer_idx = int(name.split(".")[1])
            if layer_idx >= self.config.num_hidden_layers:
                continue
        #######

        for param_name, weight_name, shard_id in stacked_params_mapping:
            # Skip non-stacked layers and experts (experts handled below).
            if weight_name not in name:
                continue
            # We have mlp.experts[0].gate_proj in the checkpoint.
            # Since we handle the experts below in expert_params_mapping,
            # we need to skip here BEFORE we update the name, otherwise
            # name will be updated to mlp.experts[0].gate_up_proj, which
            # will then be updated below in expert_params_mapping
            # for mlp.experts[0].gate_gate_up_proj, which breaks load.
            if "mlp.experts" in name:
                continue
            name = name.replace(weight_name, param_name)
            # Skip loading extra bias for GPTQ models.
            if (
                name.endswith(".bias") or name.endswith("_bias")
            ) and name not in params_dict:
                continue
            # Skip layers on other devices.
            if utils.is_pp_missing_parameter(name, self):
                continue
            if name not in params_dict:
                continue

            param = params_dict[name]
            weight_loader = param.weight_loader
            weight_loader(param, loaded_weight, shard_id)
            break
        else:
            for mapping in expert_params_mapping:
                param_name, weight_name, expert_id, shard_id = mapping
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip layers on other devices.
                if utils.is_pp_missing_parameter(name, self):
                    continue
                # Skip loading extra bias for GPTQ models.
                if (
                    name.endswith(".bias") or name.endswith("_bias")
                ) and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(
                    param, loaded_weight, name, shard_id=shard_id, expert_id=expert_id
                )
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if (
                    name.endswith(".bias") or name.endswith("_bias")
                ) and name not in params_dict:
                    continue
                # Skip layers on other devices.
                if utils.is_pp_missing_parameter(name, self):
                    continue
                # Remapping the name of FP8 kv-scale.
                if name.endswith("kv_scale"):
                    remapped_kv_scale_name = name.replace(".kv_scale", ".attn.kv_scale")
                    if remapped_kv_scale_name not in params_dict:
                        logger.warning_once(
                            "Found kv scale in the checkpoint "
                            f"(e.g. {name}), but not found the expected "
                            f"name in the model "
                            f"(e.g. {remapped_kv_scale_name}). "
                            "kv-scale is not loaded."
                        )
                        continue
                    else:
                        name = remapped_kv_scale_name
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
        loaded_params.add(name)
    return loaded_params


def load_qwen2moe_weights(
    self, weights: Iterable[tuple[str, torch.Tensor]]
) -> set[str]:
    stacked_params_mapping = [
        # (param_name, shard_name, shard_id)
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]

    # Params for weights, fp8 weight scales, fp8 activation scales
    # (param_name, weight_name, expert_id, shard_id)
    expert_params_mapping = FusedMoE.make_expert_params_mapping(
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
        num_experts=self.config.num_experts,
    )

    params_dict = dict(self.named_parameters())
    loaded_params: set[str] = set()
    for name, loaded_weight in weights:
        """
        [RBLN] Skips loading of layers greater than `num_hidden_layers`.
        This must be modified to more graceful code in the future.
        """
        if name.startswith("layers"):
            layer_idx = int(name.split(".")[1])
            if layer_idx >= self.config.num_hidden_layers:
                continue
        #######

        for param_name, weight_name, shard_id in stacked_params_mapping:
            # Skip non-stacked layers and experts (experts handled below).
            if weight_name not in name:
                continue
            # We have mlp.experts[0].gate_proj in the checkpoint.
            # Since we handle the experts below in expert_params_mapping,
            # we need to skip here BEFORE we update the name, otherwise
            # name will be updated to mlp.experts[0].gate_up_proj, which
            # will then be updated below in expert_params_mapping
            # for mlp.experts[0].gate_gate_up_proj, which breaks load.
            if "mlp.experts" in name:
                continue
            name = name.replace(weight_name, param_name)
            # Skip loading extra bias for GPTQ models.
            if (
                name.endswith(".bias") or name.endswith("_bias")
            ) and name not in params_dict:
                continue
            # Skip layers on other devices.
            if utils.is_pp_missing_parameter(name, self):
                continue
            if name not in params_dict:
                continue

            param = params_dict[name]
            weight_loader = param.weight_loader
            weight_loader(param, loaded_weight, shard_id)
            break
        else:
            for mapping in expert_params_mapping:
                param_name, weight_name, expert_id, shard_id = mapping
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip layers on other devices.
                if utils.is_pp_missing_parameter(name, self):
                    continue
                # Skip loading extra bias for GPTQ models.
                if (
                    name.endswith(".bias") or name.endswith("_bias")
                ) and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(
                    param, loaded_weight, name, shard_id=shard_id, expert_id=expert_id
                )
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if (
                    name.endswith(".bias") or name.endswith("_bias")
                ) and name not in params_dict:
                    continue
                # Skip layers on other devices.
                if utils.is_pp_missing_parameter(name, self):
                    continue
                # Remapping the name of FP8 kv-scale.
                if name.endswith("kv_scale"):
                    remapped_kv_scale_name = name.replace(".kv_scale", ".attn.kv_scale")
                    if remapped_kv_scale_name not in params_dict:
                        logger.warning_once(
                            "Found kv scale in the checkpoint "
                            f"(e.g. {name}), but not found the expected "
                            f"name in the model "
                            f"(e.g. {remapped_kv_scale_name}). "
                            "kv-scale is not loaded."
                        )
                        continue
                    else:
                        name = remapped_kv_scale_name
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
        loaded_params.add(name)
    return loaded_params


def load_deepseek_v2_weights(
    self, weights: Iterable[tuple[str, torch.Tensor]]
) -> set[str]:
    stacked_params_mapping = [
        # (param_name, shard_name, shard_id)
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]

    # Params for weights, fp8 weight scales, fp8 activation scales
    # (param_name, weight_name, expert_id, shard_id)
    expert_params_mapping = FusedMoE.make_expert_params_mapping(
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
        num_experts=self.config.n_routed_experts,
    )

    params_dict = dict(self.named_parameters())
    loaded_params: set[str] = set()
    for name, loaded_weight in weights:
        """
        [RBLN] Skips loading of layers greater than `num_hidden_layers`.
        This must be modified to more graceful code in the future.
        """
        if name.startswith("model.layers"):
            layer_idx = int(name.split(".")[2])
            if layer_idx >= self.config.num_hidden_layers:
                continue
        #######
        if "rotary_emb.inv_freq" in name:
            continue

        spec_layer = deepseek_v2.get_spec_layer_idx_from_weight_name(self.config, name)
        if spec_layer is not None:
            continue  # skip spec decode layers for main model

        for param_name, weight_name, shard_id in stacked_params_mapping:
            # Skip non-stacked layers and experts (experts handled below).
            if weight_name not in name:
                continue
            # We have mlp.experts[0].gate_proj in the checkpoint.
            # Since we handle the experts below in expert_params_mapping,
            # we need to skip here BEFORE we update the name, otherwise
            # name will be updated to mlp.experts[0].gate_up_proj, which
            # will then be updated below in expert_params_mapping
            # for mlp.experts[0].gate_gate_up_proj, which breaks load.
            if ("mlp.experts." in name) and name not in params_dict:
                continue
            name = name.replace(weight_name, param_name)
            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue

            if utils.is_pp_missing_parameter(name, self):
                continue

            param = params_dict[name]
            weight_loader = param.weight_loader
            weight_loader(param, loaded_weight, shard_id)
            break
        else:
            for mapping in expert_params_mapping:
                param_name, weight_name, expert_id, shard_id = mapping
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                if utils.is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(
                    param, loaded_weight, name, shard_id=shard_id, expert_id=expert_id
                )
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

                if utils.is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
        loaded_params.add(name)
    return loaded_params


def load_llama4_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    stacked_params_mapping = [
        # (param_name, shard_name, shard_id)
        (".qkv_proj", ".q_proj", "q"),
        (".qkv_proj", ".k_proj", "k"),
        (".qkv_proj", ".v_proj", "v"),
        (".gate_up_proj", ".gate_proj", 0),
        (".gate_up_proj", ".up_proj", 1),
    ]
    fused_experts_params = False
    expert_params_mapping = FusedMoE.make_expert_params_mapping(
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
        num_experts=self.num_experts,
    )
    expert_params_mapping_fused = FusedMoE.make_expert_params_mapping(
        ckpt_gate_proj_name="gate_up_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="gate_up_proj",
        num_experts=1,
    )
    params_dict = dict(self.named_parameters())
    loaded_params: set[str] = set()
    for name, loaded_weight in weights:
        """
        [RBLN] Skips loading of layers greater than `num_hidden_layers`.
        This must be modified to more graceful code in the future.
        """
        if name.startswith("layers"):
            layer_idx = int(name.split(".")[1])
            if layer_idx >= self.config.num_hidden_layers:
                continue
        #######

        if "experts.gate_up_proj" in name or "experts.down_proj" in name:
            fused_experts_params = True
            expert_params_mapping = expert_params_mapping_fused
        if self.quant_config is not None and (
            scale_name := self.quant_config.get_cache_scale(name)
        ):
            # Loading kv cache quantization scales
            param = params_dict[scale_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            loaded_weight = (
                loaded_weight if loaded_weight.dim() == 0 else loaded_weight[0]
            )
            weight_loader(param, loaded_weight)
            loaded_params.add(scale_name)
            continue
        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name not in name or "experts" in name:
                continue
            name = name.replace(weight_name, param_name)
            if utils.is_pp_missing_parameter(name, self):
                continue
            param = params_dict[name]
            weight_loader = param.weight_loader
            weight_loader(param, loaded_weight, shard_id)
            loaded_params.add(name)
            break
        else:
            moe_loaded = self.load_moe_expert_weights(
                name,
                loaded_weight,
                params_dict,
                loaded_params,
                expert_params_mapping,
                fused=fused_experts_params,
            )

            if not moe_loaded:
                if utils.is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)
    return loaded_params


def load_minimax_m2_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    stacked_params_mapping = [
        # (param_name, shard_name, shard_id)
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
    ]

    # Params for weights, fp8 weight scales, fp8 activation scales
    # (param_name, weight_name, expert_id, shard_id)
    expert_params_mapping = self.get_expert_mapping()

    params_dict = dict(self.named_parameters())
    loaded_params: set[str] = set()

    load_last_n_layers = getattr(self.config, "_rbln_load_last_n_layers", False)

    original_num_layers = 0
    if load_last_n_layers:
        weights_list = list(weights)

        for name, _ in weights_list:
            if name.startswith("layers."):
                layer_idx = int(name.split(".")[1])
                original_num_layers = max(original_num_layers, layer_idx + 1)

        layer_offset = original_num_layers - self.config.num_hidden_layers
        logger.info(
            f"[RBLN] o_proj only: Loading o_proj from last {self.config.num_hidden_layers} layers "
            f"(layers {layer_offset} to {original_num_layers - 1}, reversed: {original_num_layers - 1} -> 0, ..., {layer_offset} -> {self.config.num_hidden_layers - 1}), "
            f"other weights from first {self.config.num_hidden_layers} layers"
        )
        weights = weights_list

    for name, loaded_weight in weights:

        if name.startswith("layers."):
            layer_idx = int(name.split(".")[1])
            if load_last_n_layers:
                is_o_proj = "o_proj" in name
                if is_o_proj:
                    if layer_idx < layer_offset:
                        continue
                    if layer_idx >= original_num_layers:
                        continue
                    new_layer_idx = (original_num_layers - 1) - layer_idx
                    name = name.replace(f"layers.{layer_idx}.", f"layers.{new_layer_idx}.")
                else:
                    if layer_idx >= self.config.num_hidden_layers:
                        continue
            else:
                if layer_idx >= self.config.num_hidden_layers:
                    continue

        if "rotary_emb.inv_freq" in name:
            continue

        spec_layer = minimax_m2.get_spec_layer_idx_from_weight_name(
            self.config, name)
        if spec_layer is not None:
            continue  # skip spec decode layers for main model

        for param_name, weight_name, shard_id in stacked_params_mapping:
            # Skip non-stacked layers and experts (experts handled below).
            if weight_name not in name:
                continue
            # We have mlp.experts[0].gate_proj in the checkpoint.
            # Since we handle the experts below in expert_params_mapping,
            # we need to skip here BEFORE we update the name, otherwise
            # name will be updated to mlp.experts[0].gate_up_proj, which
            # will then be updated below in expert_params_mapping
            # for mlp.experts[0].gate_gate_up_proj, which breaks load.
            if ("mlp.experts." in name) and name not in params_dict:
                continue
            name = name.replace(weight_name, param_name)
            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue

            if utils.is_pp_missing_parameter(name, self):
                continue

            param = params_dict[name]
            weight_loader = param.weight_loader
            weight_loader(param, loaded_weight, shard_id)
            break
        else:
            for mapping in expert_params_mapping:
                param_name, weight_name, expert_id, shard_id = mapping
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                if utils.is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(
                    param,
                    loaded_weight,
                    name,
                    shard_id=shard_id,
                    expert_id=expert_id,
                )
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

                if utils.is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = getattr(
                    param, "weight_loader", default_weight_loader
                )
                weight_loader(param, loaded_weight)
        loaded_params.add(name)
    return loaded_params


llama.LlamaModel.load_weights = load_llama_weights
llama4.Llama4Model.load_weights = load_llama4_weights

qwen2.Qwen2Model.load_weights = load_qwen2_weights
qwen2_moe.Qwen2MoeModel.load_weights = load_qwen2moe_weights
qwen3_moe.Qwen3MoeModel.load_weights = load_qwen3moe_weights
deepseek_v2.DeepseekV2ForCausalLM.load_weights = load_deepseek_v2_weights
minimax_m2.MiniMaxM2Model.load_weights = load_minimax_m2_weights
