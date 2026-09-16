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


"""RBLN's variant of upstream's NIXL transfer topology.

It lives beside the connector rather than inside it so that the patch module
that substitutes it upstream can import it without pulling the whole connector
package in at plugin load.
"""

import torch
from vllm.distributed.kv_transfer.kv_connector.utils import (
    EngineId,
    EngineTransferInfo,
    TransferTopology,
)
from vllm.v1.kv_cache_interface import KVCacheSpec


class RblnTransferTopology(TransferTopology):
    """Upstream's topology, taught RBLN's two attention KV layouts.

    The rbln_triton_ops cache keeps K and V as separate spans, so each becomes
    its own NIXL region -- the layout upstream standardized away in 0.26. The
    rbln_custom_ops cache puts them inside one block, which the descriptor path
    has no second region to name, so that one is refused at handshake time
    rather than transferring halves of a block. MLA and Mamba caches are
    upstream's own shapes and stay on its path.

    ``__post_init__`` reimplements upstream's rather than extending it, since
    upstream's asserts a blocks-first 4-dim shape RBLN never produces. So a
    field upstream sets there has to be set here too, and leaving one out
    fails nowhere until the first handshake reads it.
    """

    def __post_init__(self) -> None:
        self.local_physical_heads = max(1, self.total_num_kv_heads // self.tp_size)
        self._engines: dict[tuple[EngineId, int], EngineTransferInfo] = {}
        self._cross_layers_blocks = False
        if self.is_mamba:
            # Upstream skips the shape for the same reason: a Mamba cache is a
            # (conv, ssm) pair, and the connector hands it no tensor shape.
            return
        backend = self.attn_backends[0]
        # A num_blocks no other extent can collide with, so the shape itself
        # says which axis carries it. Upstream's get_kv_cache_block_dim reads
        # it the same way, but only off an AttentionBackend subclass.
        mock_blocks = 1234567
        shape = backend.get_kv_cache_shape(
            num_blocks=mock_blocks, block_size=16, num_kv_heads=1, head_size=1
        )
        block_dim = shape.index(mock_blocks)
        if self.is_mla:
            assert block_dim == 0, (
                "RBLN NIXL descriptors assume a (num_blocks, ...) MLA cache, "
                f"got {shape} from {backend.__name__}."
            )
        elif block_dim != 1:
            raise NotImplementedError(
                "RBLN NIXL cuts K and V out of separate regions, which the "
                f"blocks-first attention cache {shape} from {backend.__name__} "
                "interleaves inside each block. Disaggregated serving needs "
                "the rbln_triton_ops kernels until the descriptor path moves "
                "to upstream's interleaved-KV split."
            )
        self._cross_layers_blocks = (
            self.tensor_shape is not None and len(self.tensor_shape) == len(shape) + 1
        )

    def get_transfer_cache_regions(
        self, cache: torch.Tensor, layer_spec: KVCacheSpec
    ) -> list[torch.Tensor] | torch.Tensor:
        if self.is_mla or self.is_mamba or self._cross_layers_blocks:
            return super().get_transfer_cache_regions(cache, layer_spec)
        # Only a K/V-first cache gets this far -- __post_init__ refuses the
        # blocks-first one. Iterating the tensor yields K and V; the caller
        # divides the page size by how many come back.
        return cache
