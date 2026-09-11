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

import json
import types

import numpy as np
import pytest
import torch

from vllm_rbln.v1.worker import kv_dump

NUM_BLOCKS, NUM_HEADS, BLOCK_SIZE, HEAD_DIM = 33, 8, 64, 128
VALID_LEN, NUM_LAYERS = 5, 3


class FlashBackend:
    """MiniMax, gpt-oss, qwen: K and V share one tensor, blocks on axis 1."""

    axes = (1, 4)

    @staticmethod
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, **kwargs):
        return (2, num_blocks, num_kv_heads, 1, block_size, head_size)


class MLABackend:
    """DeepSeek: one latent vector per token, no K/V axis, blocks on axis 0."""

    axes = (0, 1)

    @staticmethod
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, **kwargs):
        return (num_blocks, block_size, head_size)


class IndexerBackend:
    """DeepSeek-V3.2 indexer: rejects a shape query for more than one KV head."""

    axes = (0, 1)

    @staticmethod
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, **kwargs):
        assert num_kv_heads == 1, "indexer cache stores a single latent vector"
        return (num_blocks, block_size, head_size)


def _runner(block_id, backend=FlashBackend, spec_block_size=BLOCK_SIZE):
    """A stand-in for RBLNModelRunner holding one request in a prefill step."""
    rows = np.zeros((4, 4), dtype=np.int32)
    rows[0, 0] = block_id
    names = [f"layers.{i}.attn" for i in range(NUM_LAYERS)]
    block_axis, token_axis = backend.axes

    kv_caches = []
    for layer in range(NUM_LAYERS):
        cache = torch.zeros(
            backend.get_kv_cache_shape(NUM_BLOCKS, BLOCK_SIZE, NUM_HEADS, HEAD_DIM),
            dtype=torch.int8,
        )
        written = [slice(None)] * cache.dim()
        written[block_axis] = block_id
        written[token_axis] = slice(0, VALID_LEN)
        cache[tuple(written)] = layer + 1
        kv_caches.append(cache)

    group = types.SimpleNamespace(
        backend=backend,
        layer_names=names,
        kv_cache_spec=types.SimpleNamespace(block_size=spec_block_size),
    )
    return types.SimpleNamespace(
        is_prefill=True,
        input_batch=types.SimpleNamespace(
            num_reqs=1,
            req_ids=["req-0"],
            block_table=[
                types.SimpleNamespace(block_table=types.SimpleNamespace(np=rows))
            ],
        ),
        seq_lens=torch.tensor([VALID_LEN, 0, 0, 0], dtype=torch.int32),
        kv_cache_names=names,
        kv_caches=kv_caches,
        attn_groups=[[group]],
    )


def _step(total_num_scheduled_tokens):
    return types.SimpleNamespace(total_num_scheduled_tokens=total_num_scheduled_tokens)


@pytest.fixture
def dump_dir(tmp_path, monkeypatch):
    import vllm.distributed.parallel_state as parallel_state

    monkeypatch.setattr(
        parallel_state,
        "get_dp_group",
        lambda: types.SimpleNamespace(rank_in_group=0),
    )
    monkeypatch.setenv("VLLM_RBLN_KV_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_RBLN_KV_DUMP_MAX", "64")
    monkeypatch.delenv("VLLM_RBLN_KV_DUMP_LAYERS", raising=False)
    kv_dump._state.update(armed=True, wave=0, dumped=0, off=None)
    yield tmp_path
    kv_dump._state.update(armed=True, wave=0, dumped=0, off=None)


def _dumps(directory):
    return sorted(path.name for path in directory.glob("*.npz"))


@pytest.mark.parametrize("backend", [FlashBackend, MLABackend, IndexerBackend])
def test_axes_come_from_the_backend(backend):
    assert kv_dump.block_and_token_axes(backend) == backend.axes


def test_dumps_once_per_wave(dump_dir):
    kv_dump.maybe_dump(_runner(14))
    kv_dump.maybe_dump(_runner(14))
    assert len(_dumps(dump_dir)) == 1

    kv_dump.note_step(_step(0))
    kv_dump.maybe_dump(_runner(18))
    assert len(_dumps(dump_dir)) == 2


def test_block_follows_the_request_not_the_index(dump_dir):
    kv_dump.maybe_dump(_runner(14))
    kv_dump.note_step(_step(0))
    kv_dump.maybe_dump(_runner(18))
    assert _dumps(dump_dir) == ["kv_r0_w001_blk14.npz", "kv_r0_w002_blk18.npz"]


def test_a_busy_step_does_not_rearm(dump_dir):
    kv_dump.maybe_dump(_runner(14))
    kv_dump.note_step(_step(64))
    kv_dump.maybe_dump(_runner(9))
    assert len(_dumps(dump_dir)) == 1


def test_decode_steps_never_dump(dump_dir):
    runner = _runner(14)
    runner.is_prefill = False
    kv_dump.maybe_dump(runner)
    assert _dumps(dump_dir) == []


@pytest.mark.parametrize(
    "backend,expected_shape",
    [
        (FlashBackend, (2, NUM_HEADS, 1, VALID_LEN, HEAD_DIM)),
        (MLABackend, (VALID_LEN, HEAD_DIM)),
    ],
)
def test_slice_matches_the_backend_layout(dump_dir, backend, expected_shape):
    kv_dump.maybe_dump(_runner(18, backend=backend))
    dumped = np.load(dump_dir / "kv_r0_w001_blk18.npz")
    assert sorted(dumped.files) == [f"layers.{i}.attn" for i in range(NUM_LAYERS)]
    assert dumped["layers.1.attn"].shape == expected_shape
    assert (dumped["layers.1.attn"] == 2).all()


def test_all_layers_by_default(dump_dir):
    kv_dump.maybe_dump(_runner(18))
    dumped = np.load(dump_dir / "kv_r0_w001_blk18.npz")
    assert len(dumped.files) == NUM_LAYERS


@pytest.mark.parametrize(
    "selection,expected",
    [
        ("0,2", ["layers.0.attn", "layers.2.attn"]),
        ("-1", ["layers.2.attn"]),
        ("1", ["layers.1.attn"]),
    ],
)
def test_layer_selection(dump_dir, monkeypatch, selection, expected):
    monkeypatch.setenv("VLLM_RBLN_KV_DUMP_LAYERS", selection)
    kv_dump.maybe_dump(_runner(18))
    dumped = np.load(dump_dir / "kv_r0_w001_blk18.npz")
    assert sorted(dumped.files) == sorted(expected)
    meta = json.loads((dump_dir / "kv_r0_w001_blk18.json").read_text())
    assert meta["selected_layers"] == [int(i) for i in selection.split(",")]


def test_a_layer_index_out_of_range_is_an_error(dump_dir, monkeypatch):
    monkeypatch.setenv("VLLM_RBLN_KV_DUMP_LAYERS", str(NUM_LAYERS))
    with pytest.raises(AssertionError, match="outside the"):
        kv_dump.maybe_dump(_runner(18))


def test_hybrid_block_layers_are_skipped_not_misaddressed(dump_dir):
    kv_dump.maybe_dump(_runner(18, spec_block_size=BLOCK_SIZE * 2))
    meta = json.loads((dump_dir / "kv_r0_w001_blk18.json").read_text())
    assert meta["layer_names"] == []
    assert len(meta["hybrid_layers_skipped"]) == NUM_LAYERS


def test_metadata_sidecar(dump_dir):
    kv_dump.maybe_dump(_runner(18))
    meta = json.loads((dump_dir / "kv_r0_w001_blk18.json").read_text())
    assert meta["block_id"] == 18
    assert meta["valid_len"] == VALID_LEN
    assert meta["wave"] == 1
    assert meta["req_id"] == "req-0"
    assert meta["hybrid_layers_skipped"] == {}


def test_max_caps_the_dumps(dump_dir):
    kv_dump._state["dumped"] = 64
    kv_dump.maybe_dump(_runner(14))
    assert _dumps(dump_dir) == []


def test_unset_dir_is_inert(tmp_path, monkeypatch):
    monkeypatch.delenv("VLLM_RBLN_KV_DUMP_DIR", raising=False)
    kv_dump._state.update(armed=True, wave=0, dumped=0, off=None)
    kv_dump.note_step(_step(0))
    kv_dump.maybe_dump(_runner(14))
    assert list(tmp_path.iterdir()) == []


def test_a_request_with_no_block_is_an_error(dump_dir):
    runner = _runner(14)
    runner.input_batch.block_table[0].block_table.np[0, 0] = 0
    with pytest.raises(AssertionError, match="holding no block"):
        kv_dump.maybe_dump(runner)
