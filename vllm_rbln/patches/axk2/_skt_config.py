# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from transformers import PretrainedConfig


class AXK2Config(PretrainedConfig):
    r"""
    Configuration class for the AXK2 model.
    Extends AXK1 with gated normalization and attention output gating.
    """

    model_type = "axk2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 163840,
        hidden_size: int = 2048,
        intermediate_size: int = 5120,
        moe_intermediate_size: int = 512,
        num_hidden_layers: int = 48,
        num_nextn_predict_layers: int | None = None,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 32,
        n_shared_experts: int | None = 1,
        n_routed_experts: int | None = 128,
        ep_size: int | None = 1,
        routed_scaling_factor: float | None = 2.5,
        kv_lora_rank: int | None = 128,
        q_lora_rank: int | None = 384,
        qk_rope_head_dim: int | None = 32,
        v_head_dim: int | None = 64,
        qk_nope_head_dim: int | None = 64,
        topk_method: str | None = "noaux_tc",
        n_group: int | None = None,
        topk_group: int | None = None,
        num_experts_per_tok: int | None = 8,
        moe_layer_freq: int | None = 1,
        first_k_dense_replace: int = 1,
        norm_topk_prob: bool = True,
        scoring_func: str | None = "sigmoid",
        aux_loss_alpha: float | None = 0.0001,
        seq_aux: float | None = True,
        hidden_act: str | None = "silu",
        max_position_embeddings: int | None = 4096,
        initializer_range: float | None = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool | None = True,
        pad_token_id: int | None = None,
        bos_token_id: int | None = 163691,
        eos_token_id: int | None = 163691,
        pretraining_tp: int | None = 1,
        tie_word_embeddings: bool | None = False,
        rope_theta: float | None = 1000000.0,
        rope_scaling: dict[str, Any] | None = None,
        rope_parameters: dict[str, Any] | None = None,
        attention_bias: bool | None = False,
        attention_dropout: float | None = 0.0,
        attention_output_gate: bool = False,
        gated_norm: bool = False,
        gated_norm_rank: int = 16,
        # DSA (Dynamic Sparse Attention) parameters
        index_n_heads: int | None = None,
        index_head_dim: int | None = None,
        index_topk: int | None = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.num_attention_heads = num_attention_heads
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.ep_size = ep_size
        self.routed_scaling_factor = routed_scaling_factor
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.topk_method = topk_method
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_layer_freq = moe_layer_freq
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        if rope_parameters is None and rope_scaling is not None:
            rope_parameters = dict(rope_scaling)
            if "type" in rope_parameters and "rope_type" not in rope_parameters:
                rope_parameters["rope_type"] = rope_parameters.pop("type")
            rope_parameters.setdefault("rope_theta", rope_theta)
        self.rope_parameters = rope_parameters
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.attention_output_gate = attention_output_gate
        self.gated_norm = gated_norm
        self.gated_norm_rank = gated_norm_rank
        # DSA — only attach when provided. vLLM uses `hasattr(config,
        # "index_topk")` as the is_v32 sentinel (see deepseek_v2,
        # glm4_moe_lite, deepseek_mtp, etc.), so unconditionally setting
        # these to None makes the sentinel fire in non-DSA mode and crashes
        # AXK2Model.__init__ when it tries to allocate topk_indices_buffer.
        if index_n_heads is not None:
            self.index_n_heads = index_n_heads
        if index_head_dim is not None:
            self.index_head_dim = index_head_dim
        if index_topk is not None:
            self.index_topk = index_topk

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
