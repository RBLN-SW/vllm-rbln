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
"""The krt runner: gpt-oss-120b on the vLLM-native path, its decoder layers
run as kcc-built KPL kernels through krt.

Selected with `--worker-cls vllm_rbln.v1.worker.krt_model_runner.KrtWorker`.
Everything but the decoder layers is `RBLNModelRunner`'s and the vLLM
model's own, run uncompiled on the host: the embedding, the final norm, the
LM head, input preparation, the KV cache manager and sampling.
"""

import os
from collections import Counter

import torch
from torch import nn
from vllm.forward_context import get_forward_context

import vllm_rbln.envs as envs
from vllm_rbln.logger import init_logger
from vllm_rbln.platform import USE_DEVICE_TENSOR
from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner
from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

logger = init_logger(__name__)


class KrtClient:
    """The connection to the krt runner's kernel side (`krt_runner.server`
    in codesigner, which holds kcc, the backends and the KV pool): it runs in
    an environment of its own, so a routed layer's call crosses a socket.

    `KRT_SERVER` names the socket and `KRT_AUTHKEY` its key. Every request is
    answered, and a failure on the kernel side is raised here."""

    def __init__(self, address: str):
        from multiprocessing.connection import Client

        key = os.environ.get("KRT_AUTHKEY", "krt").encode()
        self.conn = Client(address, family="AF_UNIX", authkey=key)
        self.opened = False
        #: layer 1's output from the pair kernel, run at layer 0's
        #: call and handed back at layer 1's
        self.pending: torch.Tensor | None = None

    def call(self, *request):
        self.conn.send(request)
        status, value = self.conn.recv()
        if status != "ok":
            raise RuntimeError(f"krt server: {request[0]} failed:\n{value}")
        return value


def _np(t: torch.Tensor):
    """A host tensor as numpy for the kernel side: floats as float32 (bf16
    has no numpy type), integers as they are."""
    t = t.detach().cpu()
    return (t.float() if t.is_floating_point() else t).numpy()


#: the attention metadata fields a layer kernel's step is made from
_META = ("seq_lens", "block_tables", "cache_seq_lens", "cache_offsets",
         "local_block_tables")


class KrtDecoderLayer(nn.Module):
    """One gpt-oss decoder layer, called as vLLM's `TransformerBlock` is.

    The layer contract between vLLM and a layer kernel: vLLM hands a layer
    `(hidden_states, residual)` and takes `(hidden_states, residual)` back, and
    the next layer's input norm (or the model's final norm) adds the two before
    it normalizes. What crosses a layer boundary is therefore their sum, and
    that sum is what a layer kernel takes as `x` and returns as `out`
    (`kernel_input`, `kernel_output`).

    A routed layer (`KRT_ROUTE`, with `KRT_SERVER`) sends each call to the
    kernel side (`routed`) and does not run its torch layer; the others run
    the torch layer they wrap. Every call is counted by the two things that
    pick a kernel build, the step's kind (prefill or decode) and its token
    count. With
    `KRT_LAYER_CONTRACT=1` the torch layer's result is handed back through the
    kernel contract, so a run checks that contract against vLLM's own.
    """

    contract = os.environ.get("KRT_LAYER_CONTRACT") == "1"
    #: with a routed layer, also run the torch layer and log how far the
    #: kernel's result is from it
    compare = os.environ.get("KRT_COMPARE") == "1"
    #: with layers 0 and 1 routed, run each step of theirs (prefill chunk and
    #: decode batch) as one kernel (the sliding layer then the full one,
    #: `krt_runner.server`'s "pair"): layer 0's call sends both, layer 1's
    #: hands back the result.
    pair = os.environ.get("KRT_PAIR") == "1"

    def __init__(self, layer: nn.Module, client: KrtClient | None = None,
                 model: str = "", block_size: int = 0,
                 max_model_len: int = 0):
        super().__init__()
        self.layer = layer
        self.client, self.model, self.block_size = client, model, block_size
        self.max_model_len = max_model_len
        self.calls: Counter[tuple[str, int]] = Counter()
        #: layer 1's wrapper, on layer 0's when the two run as a pair
        self.partner: KrtDecoderLayer | None = None

    @staticmethod
    def kernel_input(
        hidden_states: torch.Tensor, residual: torch.Tensor | None
    ) -> torch.Tensor:
        """The residual stream a layer kernel takes as `x`: the sum the input
        norm would form (vLLM's fused add-norm adds in float32, then rounds)."""
        if residual is None:
            return hidden_states
        return (hidden_states.float() + residual.float()).to(hidden_states.dtype)

    @staticmethod
    def kernel_output(out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """A layer kernel's `out` as vLLM's `(hidden_states, residual)`: the
        whole sum as the hidden state and a zero residual, so the next add
        returns `out` itself."""
        return out, torch.zeros_like(out)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        meta = get_forward_context().attn_metadata[self.layer.attn.attn.layer_name]
        key = ("prefill" if meta.is_prefill else "decode", hidden_states.shape[:-1].numel())
        self.calls[key] += 1
        if self.calls[key] == 1:
            logger.info("krt: layer %d first %s call, %d tokens",
                        self.layer.layer_idx, *key)
        if self.client is not None and self.pair:
            if self.partner is not None:
                return self.pair_first(hidden_states, positions, residual,
                                       meta)
            if self.layer.layer_idx == 1:
                return self.pair_second(hidden_states, positions, residual,
                                        meta)
        if self.client is not None:
            return self.routed(hidden_states, positions, residual, meta)
        if not self.contract:
            return self.layer(hidden_states, positions, residual)
        mlp_out, after_attn = self.layer(hidden_states, positions, residual)
        return self.kernel_output(self.kernel_input(mlp_out, after_attn))


    def open(self) -> None:
        """Open the kernel side's session at the first routed call, when the
        KV cache exists: its pool is vLLM's, in vLLM's blocks
        (`KRT_POOL_BLOCKS` caps it; a block id past it is refused there)."""
        kv = self.layer.attn.attn.kv_cache
        # the sliding view `[2, kernel blocks, NKV, 1, window, DH]` of the
        # shared tensor: its memory is `kernel blocks * window` slots a head
        blocks = kv.shape[1] * kv.shape[-2] // self.block_size
        blocks = min(blocks, int(os.environ.get("KRT_POOL_BLOCKS", blocks)))
        got = self.client.call("open", {"model": self.model,
                                        "block_size": self.block_size,
                                        "blocks": blocks,
                                        "max_model_len": self.max_model_len})
        self.client.opened = True
        logger.info("krt: kernel session open, pool %s, full layers read %d "
                    "blocks", got["pool"], got["pages"])

    def call_of(self, hidden_states, positions, residual, meta) -> dict:
        """A layer call as the kernel side takes it (`krt_runner.adapter`)."""
        rotary = self.layer.attn.rotary_emb
        return {
            "hidden_states": _np(hidden_states),
            "residual": None if residual is None else _np(residual),
            "positions": _np(positions),
            "cos_sin": _np(rotary.cos_sin_cache.index_select(
                0, positions.flatten())),
            "meta": self.meta_of(meta),
        }

    @staticmethod
    def meta_of(meta) -> dict:
        return {"is_prefill": bool(meta.is_prefill),
                **{n: _np(getattr(meta, n)) for n in _META
                   if getattr(meta, n) is not None}}

    def pair_first(self, hidden_states, positions, residual, meta):
        """Layer 0's call, run with layer 1's as one kernel: layer 1's output
        is kept for its call. What this returns is layer 1's input,
        which the kernel does not take: the torch layer's under
        `KRT_COMPARE` (so layer 1's torch result is vLLM's own to compare
        against), the input as it came otherwise."""
        if not self.client.opened:
            self.open()
        second = self.partner.layer.attn.attn.layer_name
        meta1 = get_forward_context().attn_metadata[second]
        call1 = {"positions": _np(positions), "meta": self.meta_of(meta1)}
        out = self.client.call(
            "pair", self.call_of(hidden_states, positions, residual, meta),
            call1)
        self.client.pending = torch.from_numpy(out).reshape(
            hidden_states.shape).to(hidden_states.dtype)
        if self.compare:
            return self.layer(hidden_states, positions, residual)
        return self.kernel_output(self.kernel_input(hidden_states, residual))

    def pair_second(self, hidden_states, positions, residual, meta):
        """Layer 1's call: the pair kernel's output layer 0's call kept."""
        out, self.client.pending = self.client.pending, None
        if out is None:
            raise RuntimeError("layer 1's call without layer 0's pair")
        self.check(out, hidden_states, positions, residual, meta)
        return self.kernel_output(out)

    def check(self, out, hidden_states, positions, residual, meta):
        """Under `KRT_COMPARE`, the torch layer's result beside the kernel's:
        logged, and with `KRT_DUMP` saved for a look offline."""
        if not self.compare:
            return
        want = self.kernel_input(*self.layer(hidden_states, positions,
                                             residual)).float()
        err = (out.float() - want).abs()
        logger.info(
            "krt: layer %d %s, kernel vs torch: max abs %.3e, "
            "max/|torch|max %.3e", self.layer.layer_idx,
            "prefill" if meta.is_prefill else "decode", err.max(),
            err.max() / want.abs().max())
        dump = os.environ.get("KRT_DUMP")
        if dump:
            import numpy as np
            key = "prefill" if meta.is_prefill else "decode"
            np.savez(f"{dump}/layer{self.layer.layer_idx}_{key}"
                     f"{self.calls[(key, positions.numel())]}.npz",
                     kernel=_np(out), torch=_np(want),
                     positions=_np(positions))

    def routed(self, hidden_states, positions, residual, meta):
        """The layer as its kernel: the call vLLM made, sent to the kernel
        side, the layer's output back through the layer contract."""
        if not self.client.opened:
            self.open()
        out = self.client.call(
            "step", self.layer.layer_idx,
            self.call_of(hidden_states, positions, residual, meta))
        out = torch.from_numpy(out).reshape(hidden_states.shape).to(
            hidden_states.dtype)
        self.check(out, hidden_states, positions, residual, meta)
        return self.kernel_output(out)


class KrtModelRunner(RBLNModelRunner):
    def load_model(self) -> None:
        if USE_DEVICE_TENSOR or envs.VLLM_RBLN_COMPILE_MODEL:
            raise ValueError(
                "the krt runner takes host tensors and an uncompiled model: "
                "set VLLM_RBLN_USE_DEVICE_TENSOR=0 and VLLM_RBLN_COMPILE_MODEL=0"
            )
        super().load_model()
        arch = type(self.model).__name__
        if arch != "GptOssForCausalLM":
            raise ValueError(f"the krt runner runs gpt-oss only, not {arch}")
        decoder = self.model.model
        # `KRT_SERVER`: the kernel side's socket; `KRT_ROUTE`: the decoder
        # layers it runs (the others stay torch layers on the host)
        address = os.environ.get("KRT_SERVER")
        route = {int(i) for i in os.environ.get("KRT_ROUTE", "").split(",")
                 if i}
        client = KrtClient(address) if address and route else None
        for i in range(decoder.start_layer, decoder.end_layer):
            routed = client is not None and i in route
            layer = decoder.layers[i]
            if routed:
                client.call("layer", i, {n: _np(p) for n, p in
                                         layer.named_parameters()})
            decoder.layers[i] = KrtDecoderLayer(
                layer, client if routed else None,
                self.model_config.model, self.cache_config.block_size,
                self.model_config.max_model_len)
        if KrtDecoderLayer.pair and client is not None:
            if not {0, 1} <= route:
                raise ValueError("KRT_PAIR runs layers 0 and 1 together: "
                                 "route both (KRT_ROUTE=0,1)")
            decoder.layers[0].partner = decoder.layers[1]
        logger.info("krt: decoder layers %d..%d wrapped; routed to kernels: "
                    "%s", decoder.start_layer, decoder.end_layer - 1,
                    sorted(route) if client else "none")


class KrtWorker(RBLNWorker):
    model_runner_cls = KrtModelRunner
