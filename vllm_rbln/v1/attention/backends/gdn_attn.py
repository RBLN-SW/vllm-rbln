# Copyright 2026 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

import torch
from vllm.config import VllmConfig
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.registry import (
    MambaAttentionBackendEnum,
    register_backend,
)
from vllm.v1.kv_cache_interface import MambaSpec

import vllm_rbln.utils as rbln_utils


@register_backend(MambaAttentionBackendEnum.GDN_ATTN, is_mamba=True)
class RBLNGDNAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "GDN_ATTN"

    @staticmethod
    def get_builder_cls() -> type["RBLNGDNAttentionMetadataBuilder"]:
        return RBLNGDNAttentionMetadataBuilder

    @classmethod
    def is_ssm(cls) -> bool:
        return True


@dataclass
class RBLNGDNAttentionMetadata:
    is_prefill: bool
    # Each request's row in the state caches: its Mamba block. Padded decode
    # rows use the null block, which no request owns.
    state_slots: torch.Tensor
    # Host-side rows of the requests that start in this step; the runner zeroes
    # them before the step reads them.
    fresh_slots: list[int]
    # Prefill only: 1 for the real tokens of the padded chunk.
    valid_mask: torch.Tensor | None = None
    # Prefill only: the window positions of the last K - 1 real conv inputs.
    conv_state_index: torch.Tensor | None = None
    # Graph inputs, attached by the runner like the attention KV caches.
    conv_states: dict[str, torch.Tensor] | None = None
    ssm_states: dict[str, torch.Tensor] | None = None
    # The layers' rbln_cache_update results. The compiler drops an update
    # whose result no graph output reaches, so the runner returns these.
    state_updates: list[torch.Tensor] = field(default_factory=list)


class RBLNGDNAttentionMetadataBuilder(
    AttentionMetadataBuilder[RBLNGDNAttentionMetadata]
):
    def __init__(
        self,
        kv_cache_spec: MambaSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.chunked_prefill_size = vllm_config.scheduler_config.max_num_batched_tokens
        # The conv state shape is (K - 1, conv_dim).
        self.conv_state_len = kv_cache_spec.shapes[0][0]
        # Matches RBLNQwenGatedDeltaNetAttention.keeps_state.
        self.keeps_state = vllm_config.model_config.runner_type != "pooling"
        self._staged: dict[tuple, torch.Tensor] = {}

    def _stage(self, t: torch.Tensor | None, slot: str) -> torch.Tensor | None:
        if t is None:
            return None
        key = (slot, t.shape, t.dtype)
        if (buf := self._staged.get(key)) is None:
            buf = torch.empty(t.shape, dtype=t.dtype, device=self.device)
            self._staged[key] = buf
        buf.copy_(t)
        return buf

    def build(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        positions: torch.Tensor,
        batch_pad: int,
        is_prefill: bool,
        back_pad: torch.Tensor | None = None,
        is_prompt_sorted: bool = False,
    ) -> RBLNGDNAttentionMetadata:
        num_reqs = common_attn_metadata.num_reqs
        query_start_loc = common_attn_metadata.query_start_loc
        query_lens = query_start_loc[1 : num_reqs + 1] - query_start_loc[:num_reqs]
        seq_lens = common_attn_metadata.seq_lens[:num_reqs]
        num_computed_tokens = seq_lens - query_lens
        if not self.keeps_state and bool((num_computed_tokens > 0).any()):
            raise ValueError(
                "A pooling model keeps no Gated DeltaNet state, so it reads a "
                "prompt in one prefill: raise max_num_batched_tokens to at "
                "least the prompt length."
            )

        state_slots = common_attn_metadata.block_table_tensor[:num_reqs, 0].to(
            torch.int32
        )
        fresh_slots = state_slots[num_computed_tokens == 0].tolist()

        valid_mask = None
        conv_state_index = None
        if is_prefill:
            assert num_reqs == 1
            num_valid = int(query_lens[0])
            valid_mask = (torch.arange(self.chunked_prefill_size) < num_valid).to(
                torch.float32
            )[None]
            conv_state_index = num_valid + torch.arange(
                self.conv_state_len, dtype=torch.int32
            )
        else:
            state_slots = rbln_utils.pad(state_slots, 0, batch_pad)

        return RBLNGDNAttentionMetadata(
            is_prefill=is_prefill,
            state_slots=self._stage(state_slots, "state_slots"),
            fresh_slots=fresh_slots,
            valid_mask=self._stage(valid_mask, "valid_mask"),
            conv_state_index=self._stage(conv_state_index, "conv_state_index"),
        )
