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

import types
from dataclasses import replace

import pytest
import torch

from vllm_rbln.model_executor.models.optimum.base import ModelInputForRBLN
from vllm_rbln.model_executor.models.optimum.llama4 import (
    RBLNOptimumLlama4ForConditionalGeneration as Llama4,
)

HIDDEN = 4
IMAGE_TOKEN = 7


def _rows(n, base=0.0):
    return torch.arange(n, dtype=torch.float32).unsqueeze(1).expand(n, HIDDEN) + base


def _prefill_input(tokens, mm_embeds, is_mm_embed):
    return ModelInputForRBLN(
        input_tokens=torch.tensor([tokens]),
        input_positions=torch.arange(len(tokens), dtype=torch.int32).unsqueeze(0),
        block_tables=torch.tensor([0], dtype=torch.int16),
        running_requests_ids=["r0"],
        padded_batch_size=1,
        batch_rows=slice(0, 1),
        is_prompt=True,
        mm_embeds=mm_embeds,
        is_mm_embed=torch.tensor([is_mm_embed]),
    )


class TestLlama4:
    def test_image_tiles_are_grouped_per_image(self):
        model = Llama4.__new__(Llama4)
        pixels = torch.zeros(3, 3, 4, 4)
        features = _rows(6).reshape(3, 2, HIDDEN)
        model.model = types.SimpleNamespace(
            get_image_features=lambda px: types.SimpleNamespace(
                last_hidden_state=features if px is pixels else None
            )
        )
        items = model.embed_multimodal(
            pixel_values=pixels,
            patches_per_image=torch.tensor([2, 1]),
            aspect_ratios=torch.tensor([[1, 2], [1, 1]]),
        )
        assert [item.shape for item in items] == [(4, HIDDEN), (2, HIDDEN)]
        torch.testing.assert_close(items[0], features[:2].flatten(0, 1))
        torch.testing.assert_close(items[1], features[2])
        assert model.embed_multimodal() == []

    @pytest.mark.parametrize("is_prompt", [True, False])
    def test_forward_uses_embeddings_and_block_tables(self, is_prompt):
        model = Llama4.__new__(Llama4)
        calls = []

        def decoder(**kwargs):
            calls.append(kwargs)
            return types.SimpleNamespace(logits=torch.ones(1, 1, 8))

        model.model = types.SimpleNamespace(
            language_model=types.SimpleNamespace(
                prefill_decoder=decoder, decoders={1: decoder}
            ),
            config=types.SimpleNamespace(image_token_id=IMAGE_TOKEN),
            get_input_embeddings=lambda: lambda ids: torch.ones(*ids.shape, HIDDEN),
        )
        model_input = _prefill_input(
            [1, IMAGE_TOKEN, 2], [_rows(1)], [False, True, False]
        )
        model_input = model.build_prefill_forward_inputs(model_input)
        if not is_prompt:
            model_input = replace(
                model_input,
                is_prompt=False,
                input_tokens=torch.tensor([[2]]),
                input_positions=torch.tensor([[3]], dtype=torch.int32),
                block_tables=model_input.block_tables.unsqueeze(0),
                inputs_embeds=None,
            )
        result = model.forward(model_input)
        assert result.shape == (1, 1, 8)
        assert "input_ids" not in calls[0]
        assert calls[0]["block_tables"] is model_input.block_tables
        assert calls[0]["cache_position"] is model_input.input_positions
        if is_prompt:
            assert calls[0]["inputs_embeds"] is model_input.inputs_embeds
            torch.testing.assert_close(model_input.inputs_embeds[0, 1], _rows(1)[0])
        else:
            torch.testing.assert_close(
                calls[0]["inputs_embeds"], torch.ones(1, 1, HIDDEN)
            )
