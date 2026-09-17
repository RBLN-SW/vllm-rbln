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

from vllm_rbln.model_executor.models.optimum import qwen3_asr
from vllm_rbln.model_executor.models.optimum.base import ModelInputForRBLN
from vllm_rbln.model_executor.models.optimum.model_base import RBLNOptimumModelBase
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

    def __init__(self):
        self.seen: dict = {}

    def __call__(self, features, mask):
        self.seen["features"] = features
        self.seen["mask"] = mask
        return torch.arange(sum(TOKENS) * HIDDEN, dtype=torch.float32).view(-1, HIDDEN)


def _bare_qwen3_asr() -> tuple[Qwen3ASR, dict]:
    obj = Qwen3ASR.__new__(Qwen3ASR)
    tower = _FakeAudioTower()
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


class _ReachedModelInit(Exception):
    """Raised by the faked base __init__ to show the checkpoint guard passed."""


def _construct_with_config_json(monkeypatch, config_json: dict) -> None:
    monkeypatch.setattr(
        qwen3_asr, "get_hf_file_to_dict", lambda name, model, revision: config_json
    )

    def stop(self, vllm_config):
        raise _ReachedModelInit

    monkeypatch.setattr(RBLNOptimumModelBase, "__init__", stop)
    vllm_config = types.SimpleNamespace(
        model_config=types.SimpleNamespace(model="Qwen/Qwen3-ASR-0.6B", revision=None)
    )
    Qwen3ASR(vllm_config)


def test_original_checkpoint_layout_is_rejected_with_hf_hint(monkeypatch):
    with pytest.raises(ValueError, match="-hf"):
        _construct_with_config_json(
            monkeypatch, {"model_type": "qwen3_asr", "thinker_config": {}}
        )


def test_transformers_native_checkpoint_layout_passes_guard(monkeypatch):
    with pytest.raises(_ReachedModelInit):
        _construct_with_config_json(
            monkeypatch, {"model_type": "qwen3_asr", "audio_config": {}}
        )


def test_forward_prefill_passes_inputs_embeds_without_position_embed():
    obj = Qwen3ASR.__new__(Qwen3ASR)
    passed = {}

    def fake_prefill(**kw):
        passed.update(kw)
        return types.SimpleNamespace(logits=torch.zeros(1, 1))

    obj.model = types.SimpleNamespace(prefill_decoder=fake_prefill)
    model_input = ModelInputForRBLN(
        input_tokens=torch.tensor([[11, 12, 13]]),
        input_positions=torch.tensor([[0, 1, 2]], dtype=torch.int32),
        block_tables=torch.tensor([10], dtype=torch.int16),
        running_requests_ids=["A"],
        padded_batch_size=1,
        batch_rows=slice(0, 1),
        is_prompt=True,
        inputs_embeds=torch.zeros(1, 3, HIDDEN),
    )

    obj.forward(model_input)

    assert set(passed) == {"inputs_embeds", "block_tables", "cache_position"}
    assert passed["inputs_embeds"] is model_input.inputs_embeds
    assert passed["block_tables"] is model_input.block_tables
    assert passed["cache_position"] is model_input.input_positions


def test_forward_decode_picks_the_decoder_of_the_padded_batch():
    obj = Qwen3ASR.__new__(Qwen3ASR)
    recorded = {}

    def fake_decoder(**kw):
        recorded.update(kw)
        return types.SimpleNamespace(logits=torch.arange(2.0).view(2, 1))

    obj.model = types.SimpleNamespace(decoders={1: None, 2: fake_decoder})
    model_input = ModelInputForRBLN(
        input_tokens=torch.tensor([[201], [0]]),
        input_positions=torch.tensor([[3], [0]], dtype=torch.int32),
        block_tables=torch.tensor([[10], [50]], dtype=torch.int16),
        running_requests_ids=["A"],
        padded_batch_size=2,
        batch_rows=slice(0, 1),
    )

    logits = obj.forward(model_input)

    assert obj.model.decoder is fake_decoder
    assert set(recorded) == {"input_ids", "cache_position", "block_tables"}
    assert recorded["input_ids"] is model_input.input_tokens
    assert recorded["cache_position"] is model_input.input_positions
    assert recorded["block_tables"] is model_input.block_tables
    # The runner owns the padding rows, so forward returns every row it got.
    assert logits.shape == (2, 1)
