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

import os
import re
from collections.abc import Iterable
from contextlib import suppress
from itertools import islice
from typing import Any

import torch
from torch import nn
from transformers import AutoConfig, PretrainedConfig
from transformers.tokenization_utils_base import AddedToken, PreTrainedTokenizerBase
from vllm import ModelRegistry
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.activation import GeluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.config import MODELS_CONFIG_MAP
from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils import config as hf_config_utils
from vllm.transformers_utils.model_arch_config_convertor import (
    MODEL_ARCH_CONFIG_CONVERTORS,
    ModelArchConfigConvertorBase,
)


class Gemma4TextConfig(PretrainedConfig):
    model_type = "gemma4_text"
    architectures = ["Gemma4ForCausalLM"]

    def __init__(
        self,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        attention_k_eq_v: bool = False,
        bos_token_id: int = 2,
        dtype: str | None = None,
        enable_moe_block: bool = False,
        eos_token_id: int | list[int] = 1,
        final_logit_softcapping: float | None = None,
        global_head_dim: int | None = None,
        head_dim: int | None = None,
        hidden_activation: str = "gelu_pytorch_tanh",
        hidden_size: int = 0,
        hidden_size_per_layer_input: int = 0,
        initializer_range: float = 0.02,
        intermediate_size: int = 0,
        layer_types: list[str] | None = None,
        max_position_embeddings: int = 0,
        model_type: str = "gemma4_text",
        num_attention_heads: int = 0,
        num_experts: int | None = None,
        num_global_key_value_heads: int | None = None,
        num_hidden_layers: int = 0,
        num_key_value_heads: int = 0,
        num_kv_shared_layers: int = 0,
        pad_token_id: int = 0,
        rms_norm_eps: float = 1e-6,
        rope_parameters: dict[str, Any] | None = None,
        sliding_window: int | None = None,
        tie_word_embeddings: bool = True,
        top_k_experts: int | None = None,
        use_bidirectional_attention: bool | str = False,
        use_cache: bool = True,
        use_double_wide_mlp: bool = False,
        vocab_size: int = 0,
        vocab_size_per_layer_input: int | None = None,
        **kwargs,
    ) -> None:
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.attention_k_eq_v = attention_k_eq_v
        self.dtype = dtype
        self.enable_moe_block = enable_moe_block
        self.final_logit_softcapping = final_logit_softcapping
        self.global_head_dim = global_head_dim
        self.head_dim = head_dim
        self.hidden_activation = hidden_activation
        self.hidden_size = hidden_size
        self.hidden_size_per_layer_input = hidden_size_per_layer_input
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.layer_types = layer_types or []
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_experts = num_experts
        self.num_global_key_value_heads = num_global_key_value_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.num_kv_shared_layers = num_kv_shared_layers
        self.rms_norm_eps = rms_norm_eps
        self.rope_parameters = rope_parameters or {"rope_type": "default"}
        self.sliding_window = sliding_window
        self.top_k_experts = top_k_experts
        self.use_bidirectional_attention = use_bidirectional_attention
        self.use_cache = use_cache
        self.use_double_wide_mlp = use_double_wide_mlp
        self.vocab_size = vocab_size
        self.vocab_size_per_layer_input = (
            vocab_size
            if vocab_size_per_layer_input is None
            else vocab_size_per_layer_input
        )

        super().__init__(
            bos_token_id=bos_token_id,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class Gemma4Config(PretrainedConfig):
    model_type = "gemma4"
    sub_configs = {"text_config": Gemma4TextConfig}

    def __init__(
        self,
        text_config: Gemma4TextConfig | dict[str, Any] | None = None,
        vision_config: dict[str, Any] | PretrainedConfig | None = None,
        audio_config: dict[str, Any] | PretrainedConfig | None = None,
        tie_word_embeddings: bool = True,
        **kwargs,
    ) -> None:
        if text_config is None:
            text_config = Gemma4TextConfig()
        elif isinstance(text_config, dict):
            text_config = Gemma4TextConfig(**text_config)
        self.text_config = text_config
        self.vision_config = vision_config
        self.audio_config = audio_config
        self.hidden_size = text_config.hidden_size

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


def _patch_gemma4_tokenizer_compat() -> None:
    original = PreTrainedTokenizerBase._set_model_specific_special_tokens
    if getattr(original, "_vllm_rbln_gemma4_patched", False):
        return

    def patched(self, special_tokens):
        if isinstance(special_tokens, list):
            special_tokens = {
                f"extra_special_token_{idx}": value
                for idx, value in enumerate(special_tokens)
            }

        self.SPECIAL_TOKENS_ATTRIBUTES = self.SPECIAL_TOKENS_ATTRIBUTES + list(
            special_tokens.keys()
        )
        for key, value in special_tokens.items():
            if isinstance(value, (str, AddedToken)):
                self._special_tokens_map[key] = value
            else:
                raise TypeError(
                    f"Special token {key} has to be either str or AddedToken "
                    f"but got: {type(value)}"
                )

    patched._vllm_rbln_gemma4_patched = True
    PreTrainedTokenizerBase._set_model_specific_special_tokens = patched


class Gemma4ConfigVerifier:
    @staticmethod
    def verify_and_update_config(vllm_config: VllmConfig) -> None:
        # Upstream vLLM forces a GPU Triton backend here to avoid mixed
        # backend execution for heterogeneous Gemma4 head sizes. RBLN uses
        # its own single attention backend selection path, so only the
        # head-size convertor backport is needed on this plugin.
        return

    @staticmethod
    def verify_and_update_model_config(model_config: Any) -> None:
        return


def _get_text_config(config: PretrainedConfig) -> PretrainedConfig:
    if hasattr(config, "text_config"):
        return config.text_config
    return config


class Gemma4MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_activation: str,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_activation != "gelu_pytorch_tanh":
            raise ValueError(
                "Gemma4 uses `gelu_pytorch_tanh` as the hidden activation "
                "function. Please set `hidden_act` and `hidden_activation` to "
                "`gelu_pytorch_tanh`."
            )
        self.act_fn = GeluAndMul(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Gemma4Attention(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position_embeddings: int,
        proj_head_dim: int | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim
        # NOTE(RBLN): This is a workaround to avoid compilation failure
        # when the value of num_kv_heads differs between full-attention and
        # sliding-attention layers.
        # proj_head_dim is the head_size of qkv_proj output. When >= head_dim,
        # the trailing per-head dim is sliced off after the QKV split so that
        # downstream ops (norms, rotary, self.attn, o_proj) see the layer's
        # actual head_dim. Defaults to head_dim for the no-padding path.
        self.proj_head_dim = proj_head_dim if proj_head_dim is not None else head_dim
        self.proj_q_size = self.num_heads * self.proj_head_dim
        self.proj_kv_size = self.num_kv_heads * self.proj_head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = 1.0

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.proj_head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, has_weight=False)

        layer_idx = extract_layer_index(prefix)
        layer_type = config.layer_types[layer_idx]
        self.is_sliding = layer_type == "sliding_attention"
        sliding_window = config.sliding_window if self.is_sliding else None

        if layer_type in config.rope_parameters:
            rope_parameters = dict(config.rope_parameters[layer_type])
        else:
            rope_parameters = dict(config.rope_parameters)

        kv_sharing_target_layer_name = None
        self.is_kv_shared_layer = False
        num_kv_shared_layers = getattr(config, "num_kv_shared_layers", 0)
        if num_kv_shared_layers > 0:
            first_kv_shared_layer_idx = config.num_hidden_layers - num_kv_shared_layers
            if layer_idx >= first_kv_shared_layer_idx:
                self.is_kv_shared_layer = True
                prev_layers = config.layer_types[:first_kv_shared_layer_idx]
                current_layer_type = config.layer_types[layer_idx]
                kv_shared_layer_index = (
                    len(prev_layers) - 1 - prev_layers[::-1].index(current_layer_type)
                )
                prefix_before_layers = prefix.split(".layers.")[0]
                kv_sharing_target_layer_name = (
                    f"{prefix_before_layers}.layers."
                    f"{kv_shared_layer_index}.self_attn.attn"
                )

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters=rope_parameters,
            is_neox_style=True,
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            kv_sharing_target_layer_name=kv_sharing_target_layer_name,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split(
            [self.proj_q_size, self.proj_kv_size, self.proj_kv_size], dim=-1
        )

        q = q.unflatten(-1, (self.num_heads, self.proj_head_dim))
        if self.proj_head_dim != self.head_dim:
            q = q[..., : self.head_dim]
        q = self.q_norm(q)
        q = q.flatten(-2, -1)

        if not self.is_kv_shared_layer:
            k = k.unflatten(-1, (self.num_kv_heads, self.proj_head_dim))
            if self.proj_head_dim != self.head_dim:
                k = k[..., : self.head_dim]
            k = self.k_norm(k)
            k = k.flatten(-2, -1)
            q, k = self.rotary_emb(positions, q, k)

            v = v.unflatten(-1, (self.num_kv_heads, self.proj_head_dim))
            if self.proj_head_dim != self.head_dim:
                v = v[..., : self.head_dim]
            v = self.v_norm(v)
            v = v.flatten(-2, -1)
        else:
            q = self.rotary_emb(positions, q, k)[0]

        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class Gemma4DecoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        layer_idx = extract_layer_index(prefix)
        layer_type = config.layer_types[layer_idx]
        is_full_attention = layer_type == "full_attention"
        head_dim = (
            getattr(config, "global_head_dim", config.head_dim)
            if is_full_attention
            else config.head_dim
        )

        use_k_eq_v = is_full_attention and getattr(config, "attention_k_eq_v", False)
        if use_k_eq_v:
            num_kv_heads = getattr(
                config, "num_global_key_value_heads", config.num_key_value_heads
            )
        else:
            num_kv_heads = config.num_key_value_heads

        if os.environ.get("PAD_GEMMA4_QKV_PROJ", "False").lower() in ("true", "1"):
            # NOTE(RBLN): This is a workaround to avoid compilation failure
            # when the value of num_kv_heads differs between full-attention and
            # sliding-attention layers.
            # qkv_proj output is uniform (max) across layers and num_kv_heads
            # is uniform via weight replication. Q/K/V are sliced from
            # proj_head_dim → head_dim after the split, and downstream ops
            # run at the layer's actual head_dim.
            proj_head_dim = max(
                getattr(config, "global_head_dim", 0) or 0, config.head_dim
            )
            num_kv_heads = max(
                getattr(config, "num_global_key_value_heads", 0) or 0,
                config.num_key_value_heads,
            )
        else:
            proj_head_dim = head_dim

        self.self_attn = Gemma4Attention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            proj_head_dim=proj_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )

        layer_intermediate_size = config.intermediate_size
        self.mlp = Gemma4MLP(
            hidden_size=self.hidden_size,
            intermediate_size=layer_intermediate_size,
            hidden_activation=config.hidden_activation,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = (hidden_states + residual) * self.layer_scalar
        return hidden_states, None


@support_torch_compile
class Gemma4Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = _get_text_config(vllm_config.model_config.hf_config)
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Gemma4DecoderLayer(
                config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.register_buffer(
            "normalizer",
            torch.tensor(config.hidden_size**0.5),
            persistent=False,
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids) * self.normalizer

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                **kwargs,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        if residual is None:
            return self.norm(hidden_states)

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        params_dict.update(dict(self.named_buffers()))
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            if self.quant_config is not None and (
                scale_name := self.quant_config.get_cache_scale(name)
            ):
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loaded_weight = loaded_weight[0]
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue

            if name.endswith((".k_scale", ".v_scale", ".q_scale", ".prob_scale")):
                remapped_name = maybe_remap_kv_scale_name(name, params_dict)
                if remapped_name is not None and remapped_name in params_dict:
                    param = params_dict[remapped_name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(remapped_name)
                    continue

            for param_name, shard_name, shard_id in stacked_params_mapping:
                if shard_name not in name:
                    continue
                stacked_name = name.replace(shard_name, param_name)
                if stacked_name not in params_dict:
                    continue
                if is_pp_missing_parameter(stacked_name, self):
                    continue
                param = params_dict[stacked_name]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(stacked_name)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None or is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        return loaded_params


class Gemma4ForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = _get_text_config(vllm_config.model_config.hf_config)
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config
        self.model = Gemma4Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)

        self.logits_processor = LogitsProcessor(
            config.vocab_size,
            soft_cap=getattr(config, "final_logit_softcapping", None),
        )
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            **kwargs,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        config = self.config
        use_k_eq_v = getattr(config, "attention_k_eq_v", False)
        k_eq_v_layer_indices: set[int] = set()
        if use_k_eq_v:
            for idx, layer_type in enumerate(config.layer_types):
                if layer_type == "full_attention":
                    k_eq_v_layer_indices.add(idx)

        pad_qkv = os.environ.get("PAD_GEMMA4_QKV_PROJ", "False").lower() in ("true", "1")
        padded_head_dim = max(
            getattr(config, "global_head_dim", 0) or 0, config.head_dim
        )
        padded_num_kv_heads = max(
            getattr(config, "num_global_key_value_heads", 0) or 0,
            config.num_key_value_heads,
        )
        num_attention_heads = config.num_attention_heads

        def actual_dims_for_layer(layer_idx: int) -> tuple[int, int]:
            layer_type = config.layer_types[layer_idx]
            is_full = layer_type == "full_attention"
            head_dim = (
                (getattr(config, "global_head_dim", None) or config.head_dim)
                if is_full
                else config.head_dim
            )
            if is_full and use_k_eq_v:
                num_kv = getattr(
                    config,
                    "num_global_key_value_heads",
                    config.num_key_value_heads,
                )
            else:
                num_kv = config.num_key_value_heads
            return head_dim, num_kv

        def pad_for_layer(name: str, weight: torch.Tensor) -> torch.Tensor:
            if not pad_qkv:
                return weight
            m = re.search(r"layers\.(\d+)\.", name)
            if not m:
                return weight
            layer_idx = int(m.group(1))
            actual_head_dim, actual_num_kv = actual_dims_for_layer(layer_idx)
            head_dim_diff = padded_head_dim - actual_head_dim
            if head_dim_diff == 0 and actual_num_kv == padded_num_kv_heads:
                return weight

            if "self_attn.q_proj.weight" in name:
                # [num_heads * actual_head_dim, hidden_size] →
                # [num_heads * padded_head_dim, hidden_size] (zero-pad
                # head_dim). Q's head count doesn't change.
                w = weight.view(num_attention_heads, actual_head_dim, -1)
                if head_dim_diff > 0:
                    w = torch.nn.functional.pad(w, (0, 0, 0, head_dim_diff))
                return w.reshape(num_attention_heads * padded_head_dim, -1)
            if (
                "self_attn.k_proj.weight" in name
                or "self_attn.v_proj.weight" in name
            ):
                # K/V: zero-pad head_dim → padded_head_dim, then *replicate*
                # each kv head padded_num_kv_heads/actual_num_kv times so
                # GQA grouping at padded_num_kv_heads reproduces the
                # original grouping.
                if padded_num_kv_heads % actual_num_kv != 0:
                    raise ValueError(
                        "padded_num_kv_heads must be a multiple of "
                        "actual_num_kv for replication"
                    )
                rep = padded_num_kv_heads // actual_num_kv
                w = weight.view(actual_num_kv, actual_head_dim, -1)
                if head_dim_diff > 0:
                    w = torch.nn.functional.pad(w, (0, 0, 0, head_dim_diff))
                # [actual_num_kv, padded_head_dim, hidden_size] →
                # [actual_num_kv, rep, padded_head_dim, hidden_size] →
                # [padded_num_kv_heads, padded_head_dim, hidden_size]
                if rep > 1:
                    w = (
                        w.unsqueeze(1)
                        .expand(actual_num_kv, rep, padded_head_dim, w.shape[-1])
                        .reshape(padded_num_kv_heads, padded_head_dim, w.shape[-1])
                    )
                return w.reshape(padded_num_kv_heads * padded_head_dim, -1)
            return weight

        def iter_weights() -> Iterable[tuple[str, torch.Tensor]]:
            for name, weight in weights:
                name = name.replace("language_model.", "")

                if "self_attn.k_proj" in name and k_eq_v_layer_indices:
                    match = re.search(r"layers\.(\d+)\.", name)
                    if match and int(match.group(1)) in k_eq_v_layer_indices:
                        yield name, pad_for_layer(name, weight)
                        v_name = name.replace("k_proj", "v_proj")
                        yield v_name, pad_for_layer(v_name, weight.clone())
                        continue

                yield name, pad_for_layer(name, weight)

        skip = [
            "audio_tower.",
            "vision_tower.",
            "embed_audio.",
            "embed_vision.",
        ]
        if self.config.tie_word_embeddings:
            skip.append("lm_head.")

        loader = AutoWeightsLoader(self, skip_substrs=skip)
        return loader.load_weights(iter_weights())


class Gemma4ForConditionalGeneration(Gemma4ForCausalLM):
    pass


class Gemma4ModelArchConfigConvertor(ModelArchConfigConvertorBase):
    def get_head_size(self) -> int:
        head_dim = getattr(self.hf_text_config, "head_dim", 0)
        global_head_dim = getattr(self.hf_text_config, "global_head_dim", 0)
        return max(head_dim, global_head_dim) or super().get_head_size()


def _register_auto_config(model_type: str, config_cls: type[PretrainedConfig]) -> None:
    with suppress(ValueError):
        AutoConfig.register(model_type, config_cls, exist_ok=True)


def _register_gemma4_support() -> None:
    _patch_gemma4_tokenizer_compat()

    _register_auto_config("gemma4", Gemma4Config)
    _register_auto_config("gemma4_text", Gemma4TextConfig)

    hf_config_utils._CONFIG_REGISTRY["gemma4"] = Gemma4Config
    hf_config_utils._CONFIG_REGISTRY["gemma4_text"] = Gemma4TextConfig

    MODELS_CONFIG_MAP["Gemma4ForCausalLM"] = Gemma4ConfigVerifier
    MODELS_CONFIG_MAP["Gemma4ForConditionalGeneration"] = Gemma4ConfigVerifier

    MODEL_ARCH_CONFIG_CONVERTORS["gemma4"] = Gemma4ModelArchConfigConvertor
    MODEL_ARCH_CONFIG_CONVERTORS["gemma4_text"] = Gemma4ModelArchConfigConvertor

    ModelRegistry.register_model(
        "Gemma4ForCausalLM",
        "vllm_rbln.models.gemma4:Gemma4ForCausalLM",
    )
    ModelRegistry.register_model(
        "Gemma4ForConditionalGeneration",
        "vllm_rbln.models.gemma4:Gemma4ForConditionalGeneration",
    )


_register_gemma4_support()
