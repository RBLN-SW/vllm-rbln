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

"""Dump one KV cache block per generation wave, for the first request of the wave.

Turned on by ``VLLM_RBLN_KV_DUMP_DIR``; inert without it.

A dump is that request's block across every layer, cut to the tokens the prefill
wrote. A block holds ``block_size`` slots and a prompt fills a few dozen, so the
whole block would be mostly padding.

The block is followed by request, not by index: the allocator hands the same
request a different block on each pass, so a fixed index dumps a different
request each wave.

The axis layout comes from the backend rather than a constant here. FlashAttention
is ``(2, num_blocks, num_kv_heads, 1, block_size, head_size)`` and MLA is
``(num_blocks, block_size, head_size)``, so either constant mis-slices the other.
"""

import json
import os
import sys
import time

import numpy as np
import torch

import vllm_rbln.envs as envs

_state: dict = {"armed": True, "wave": 0, "dumped": 0, "off": None}


def enabled() -> bool:
    if _state["off"] is None:
        _state["off"] = not envs.VLLM_RBLN_KV_DUMP_DIR
    return not _state["off"]


def note_step(scheduler_output) -> None:
    """Re-arm on a step that scheduled nothing, which is a wave boundary.

    The determinism gate's repro loop drains to zero between iterations, so this
    yields one dump per iteration without the module knowing the loop exists.
    """
    if enabled() and scheduler_output.total_num_scheduled_tokens == 0:
        _state["armed"] = True


def maybe_dump(runner) -> None:
    """Dump if this is the first prefill step of a wave. Call from sample_tokens.

    Not from execute_model: that only dispatches the target forward, and with
    spec decode the draft model runs inside sample_tokens, so its KV layer is
    still unwritten there. Sampling already reads a token back to the host, so
    dumping after it rides a fence the step pays anyway instead of adding one
    into the host's run-ahead window.
    """
    if not (enabled() and _state["armed"] and runner.is_prefill):
        return
    if _state["dumped"] >= envs.VLLM_RBLN_KV_DUMP_MAX:
        return
    _state["armed"] = False
    _state["wave"] += 1
    _dump(runner, _state["wave"])
    _state["dumped"] += 1


def block_and_token_axes(backend) -> tuple[int, int]:
    """Where the block and token axes sit in this backend's KV cache shape.

    Read off the backend's own shape function with sentinel sizes, so a new
    backend needs no entry here. One KV head keeps the indexer backend, which
    asserts that, answering too.
    """
    shape = backend.get_kv_cache_shape(
        num_blocks=101, block_size=103, num_kv_heads=1, head_size=109
    )
    return shape.index(101), shape.index(103)


def _dump(runner, wave: int) -> None:
    from vllm.distributed.parallel_state import get_dp_group

    req_index = 0
    req_id = runner.input_batch.req_ids[req_index]
    row = runner.input_batch.block_table[0].block_table.np[req_index]
    held = row[row != 0]
    assert held.size, f"request {req_id} reached the forward holding no block"
    block_id = int(held[0])
    valid_len = int(runner.seq_lens[req_index])

    spec_by_layer = {}
    for groups in runner.attn_groups:
        for group in groups:
            for layer_name in group.layer_names:
                spec_by_layer[layer_name] = (group.backend, group.kv_cache_spec)

    layers = list(zip(runner.kv_cache_names, runner.kv_caches))
    wanted = envs.VLLM_RBLN_KV_DUMP_LAYERS
    if wanted:
        total = len(layers)
        for index in wanted:
            assert -total <= index < total, (
                f"VLLM_RBLN_KV_DUMP_LAYERS has {index}, outside the {total} KV layers"
            )
        layers = [layers[index] for index in wanted]

    arrays: dict[str, np.ndarray] = {}
    hybrid: dict[str, str] = {}
    for layer_name, kv in layers:
        backend, spec = spec_by_layer[layer_name]
        block_axis, token_axis = block_and_token_axes(backend)
        # A hybrid pool splits one logical block into several kernel blocks, so
        # the scheduler's block id does not index this tensor.
        if kv.shape[token_axis] != spec.block_size:
            hybrid[layer_name] = f"kernel {kv.shape[token_axis]} != {spec.block_size}"
            continue
        selector: list = [slice(None)] * kv.dim()
        selector[block_axis] = block_id
        selector[token_axis] = slice(0, valid_len)
        host = kv[tuple(selector)].to("cpu")
        if host.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            host = host.view(torch.uint8)
        arrays[layer_name] = host.numpy()

    out_dir = envs.VLLM_RBLN_KV_DUMP_DIR
    os.makedirs(out_dir, exist_ok=True)
    rank = get_dp_group().rank_in_group
    stem = f"kv_r{rank}_w{wave:03d}_blk{block_id}"
    np.savez(os.path.join(out_dir, f"{stem}.npz"), **arrays)
    with open(os.path.join(out_dir, f"{stem}.json"), "w") as f:
        json.dump(
            {
                "wave": wave,
                "rank": rank,
                "pid": os.getpid(),
                "req_id": req_id,
                "block_id": block_id,
                "valid_len": valid_len,
                "layer_names": list(arrays),
                "selected_layers": wanted,
                "hybrid_layers_skipped": hybrid,
                "slice_shape": [list(a.shape) for a in list(arrays.values())[:1]],
                "kv_dtype": str(runner.kv_caches[0].dtype),
                "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            f,
            indent=2,
        )
    print(
        f"[kv-dump] wave {wave} r{rank} req={req_id} block={block_id} "
        f"len={valid_len} layers={len(arrays)} skipped={len(hybrid)} -> {stem}.npz",
        flush=True,
        file=sys.stderr,
    )
