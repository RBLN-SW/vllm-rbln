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
"""Unit tests for Qwen3-ASR's audio-feature bridging and decoder wiring.

vLLM delivers mel features concatenated along time; optimum-rbln's compiled
audio tower wants them batched and right-padded to whole chunks with a
validity mask. The decoder graphs are plain Qwen3 (no MRoPE), so forward()
must not pass a position embed. No hardware needed.
"""

import types

import pytest
import torch

from vllm_rbln.model_executor.models.optimum.qwen3_asr import (
    RBLNOptimumQwen3ASRForConditionalGeneration as Qwen3ASR,
)

N_MELS = 4
HIDDEN = 8
# 537 mel frames -> 70 audio tokens, 300 -> 39 (three stride-2 convs per
# 100-frame chunk, 13 tokens per full chunk).
LENGTHS = [537, 300]
TOKENS = [70, 39]


class _FakeAudioTower:
    chunk_len = 100

    def __init__(self, dtype: torch.dtype):
        self.dtype = dtype
        self.seen: dict = {}

    def __call__(self, features, mask):
        self.seen["features"] = features
        self.seen["mask"] = mask
        return torch.arange(sum(TOKENS) * HIDDEN, dtype=self.dtype).view(-1, HIDDEN)


def _bare_qwen3_asr(tower_dtype=torch.float32) -> tuple[Qwen3ASR, dict]:
    obj = Qwen3ASR.__new__(Qwen3ASR)
    tower = _FakeAudioTower(tower_dtype)
    obj.model = types.SimpleNamespace(
        audio_tower=tower,
        rbln_config=types.SimpleNamespace(dtype=torch.float32),
        get_input_embeddings=lambda: types.SimpleNamespace(
            weight=torch.zeros(1, dtype=torch.float32)
        ),
    )
    return obj, tower.seen


def _audio_input():
    features = torch.randn(N_MELS, sum(LENGTHS))
    return {
        "input_features": features,
        "audio_feature_lengths": torch.tensor(LENGTHS),
    }


def test_process_audio_input_pads_to_chunks_and_splits_per_audio():
    obj, seen = _bare_qwen3_asr()
    audio_input = _audio_input()

    embeds = obj._process_audio_input(audio_input)

    # Both audios padded to the longest one, rounded up to a 100-frame chunk.
    assert seen["features"].shape == (2, N_MELS, 600)
    assert seen["mask"].shape == (2, 600)
    assert seen["mask"].sum(dim=-1).tolist() == LENGTHS
    features = audio_input["input_features"]
    assert torch.equal(seen["features"][0, :, :537], features[:, :537])
    assert torch.equal(seen["features"][1, :, :300], features[:, 537:])
    assert not seen["features"][1, :, 300:].any()
    # The packed tower output is split back into one tensor per audio.
    assert [e.shape[0] for e in embeds] == TOKENS


def test_process_audio_input_rejects_dtype_mismatch_with_text_embeds():
    obj, _ = _bare_qwen3_asr(tower_dtype=torch.float16)

    with pytest.raises(AssertionError, match="dtype"):
        obj._process_audio_input(_audio_input())


def _bare_decoder(max_batch_size: int) -> Qwen3ASR:
    obj = Qwen3ASR.__new__(Qwen3ASR)
    obj.decoder_batch_size = max_batch_size
    obj.use_multiple_decoder = False
    obj.available_blocks = torch.arange(50, 60, dtype=torch.int16)
    return obj


def test_forward_prefill_passes_inputs_embeds_without_position_embed():
    obj = _bare_decoder(max_batch_size=2)
    passed = {}

    def fake_prefill(**kw):
        passed.update(kw)
        return types.SimpleNamespace(logits=torch.zeros(1, 1))

    obj.model = types.SimpleNamespace(prefill_decoder=fake_prefill)
    inputs_embeds = torch.zeros(1, 3, HIDDEN)
    model_input = types.SimpleNamespace(
        is_prompt=True,
        running_requests_ids=["A"],
        input_tokens=torch.tensor([[11, 12, 13]]),
        input_positions=torch.tensor([[0, 1, 2]]),
        block_tables=torch.tensor([[10]], dtype=torch.int16),
        inputs_embeds=inputs_embeds,
    )

    obj.forward(model_input)

    assert set(passed) == {"inputs_embeds", "block_tables", "cache_position"}
    assert passed["inputs_embeds"] is inputs_embeds
    assert torch.equal(passed["block_tables"], torch.tensor([10], dtype=torch.int16))
    assert torch.equal(
        passed["cache_position"], torch.tensor([[0, 1, 2]], dtype=torch.int32)
    )


def test_forward_decode_pads_batch_and_trims_logits():
    obj = _bare_decoder(max_batch_size=2)
    recorded = {}

    def fake_decoder(**kw):
        recorded.update(kw)
        return types.SimpleNamespace(logits=torch.arange(2.0).view(2, 1))

    obj.model = types.SimpleNamespace(
        decoders={2: fake_decoder},
        embed_tokens=lambda ids: ids.to(torch.float32).unsqueeze(-1),
    )
    model_input = types.SimpleNamespace(
        is_prompt=False,
        running_requests_ids=["A"],
        input_tokens=torch.tensor([[201]]),
        input_positions=torch.tensor([[3]]),
        block_tables=torch.tensor([[10]], dtype=torch.int16),
    )

    logits = obj.forward(model_input)

    assert set(recorded) == {"inputs_embeds", "cache_position", "block_tables"}
    # One request padded to the decoder batch of two; the pad row gets a free block.
    assert recorded["inputs_embeds"].shape == (2, 1, 1)
    assert recorded["inputs_embeds"][0].item() == 201
    assert torch.equal(
        recorded["cache_position"], torch.tensor([[3], [0]], dtype=torch.int32)
    )
    assert recorded["block_tables"][0].tolist() == [10]
    assert recorded["block_tables"][1].item() in range(50, 60)
    assert logits.shape == (1, 1)
