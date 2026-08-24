# Copyright 2025 Rebellions Inc. All rights reserved.
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

from collections.abc import Iterable

import torch
from vllm.distributed import (
    get_ep_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.models.gpt_oss import GptOssModel, MLPBlock
from vllm.version import __version_tuple__ as VLLM_VERSION

from vllm_rbln.patches import register_patch


@register_patch(
    target="vllm.model_executor.models.gpt_oss.MLPBlock.forward",
    reason=(
        "Adapt GPT-OSS MLPBlock.forward to the RBLNMoERunner interface: pass "
        "the rounter callable instead of precomputed router logits so routing "
        "runs after RBLN DP multicast, then explicitly all-reduce TP outputs."
    ),
)
def patched_gptoss_mlp_forward(
    self: MLPBlock, hidden_states: torch.Tensor
) -> torch.Tensor:
    final_hidden_states = self.experts(hidden_states=hidden_states, router=self.router)
    if get_tensor_model_parallel_world_size() > 1:
        final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

    return final_hidden_states


# Fails loudly on the upgrade that makes this backport redundant.
assert VLLM_VERSION < (0, 25), (
    f"vLLM {VLLM_VERSION} already ships vllm#46441; delete "
    "patched_gptoss_load_weights and this assertion."
)


@register_patch(
    target="vllm.model_executor.models.gpt_oss.GptOssModel.load_weights",
    reason=(
        "vLLM 0.24.0 derives the expert shard bounds from the EP group's "
        "global rank instead of its rank within the group, so every rank "
        "outside the first EP group slices past the end of the expert "
        "dimension and silently loads nothing whenever the EP group is "
        "smaller than the world, i.e. under pipeline parallelism. Backports "
        "vllm#46441 (901a3b0). TODO(vllm>=0.25.0): delete."
    ),
)
def patched_gptoss_load_weights(
    self: GptOssModel, weights: Iterable[tuple[str, torch.Tensor]]
) -> set[str]:
    stacked_params_mapping = [
        # (param_name, shard_name, shard_id)
        (".qkv_proj", ".q_proj", "q"),
        (".qkv_proj", ".k_proj", "k"),
        (".qkv_proj", ".v_proj", "v"),
    ]

    tp_rank = get_tensor_model_parallel_rank()
    tp_size = get_tensor_model_parallel_world_size()

    # Attention heads per rank
    heads_per_rank = self.config.num_attention_heads // tp_size
    head_start = tp_rank * heads_per_rank

    ep_size = get_ep_group().world_size
    # The one line this patch exists for; upstream 0.24.0 reads `.rank`.
    ep_rank = get_ep_group().rank_in_group
    num_experts = self.config.num_local_experts
    experts_per_rank = num_experts // ep_size
    ep_rank_start = ep_rank * experts_per_rank
    ep_rank_end = (ep_rank + 1) * experts_per_rank

    quant_method = (
        self.config.quantization_config["quant_method"]
        if hasattr(self.config, "quantization_config")
        else None
    )
    # Normalize the checkpoint's quant_method to the internal name.
    # Note: there are three places where "mxfp4" -> "gpt_oss_mxfp4"
    # normalization occurs, each serving a different data path:
    #   1. GptOssMxfp4Config.override_quantization_method() — sets
    #      ModelConfig.quantization (used to select the QuantizationConfig
    #      class at model init time), reading from model_arch_config which
    #      is a snapshot taken before verify_and_update_model_config runs.
    #   2. GptOssForCausalLMConfig.verify_and_update_model_config() —
    #      patches hf_config.quantization_config in-place (a separate copy
    #      of the dict from model_arch_config) for later hf_config lookups.
    #   3. Here — reads directly from self.config (the raw HF config) which
    #      may still carry the original "mxfp4" string from the checkpoint.
    if quant_method == "mxfp4":
        quant_method = "gpt_oss_mxfp4"

    if quant_method == "gpt_oss_mxfp4":
        return self._load_weights_mxfp4(
            ep_rank_end,
            ep_rank_start,
            heads_per_rank,
            head_start,
            weights,
            stacked_params_mapping,
        )
    elif quant_method == "quark":
        return self._load_weights_quark(
            ep_rank_end,
            ep_rank_start,
            heads_per_rank,
            head_start,
            weights,
            stacked_params_mapping,
        )
    else:
        return self._load_weights_other(
            ep_rank_end,
            ep_rank_start,
            heads_per_rank,
            head_start,
            weights,
            stacked_params_mapping,
        )
