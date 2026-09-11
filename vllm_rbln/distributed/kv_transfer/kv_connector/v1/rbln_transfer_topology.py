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
    """Upstream's topology with the KV layout it standardized on in 0.26 undone.

    ``__post_init__`` reimplements upstream's rather than extending it, since
    upstream's asserts the very layout this class exists to decline. So a
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
        shape = self.attn_backends[0].get_kv_cache_shape(
            num_blocks=1, block_size=16, num_kv_heads=1, head_size=1
        )
        leading = (1,) if self.is_mla else (2, 1)
        assert shape[: len(leading)] == leading, (
            "RBLN NIXL descriptors assume a (2, num_blocks, ...) attention "
            f"cache or a (num_blocks, ...) MLA cache, got {shape} from "
            f"{self.attn_backends[0].__name__}."
        )
        self._cross_layers_blocks = (
            self.tensor_shape is not None and len(self.tensor_shape) == len(shape) + 1
        )

    def get_transfer_cache_regions(
        self, cache: torch.Tensor, layer_spec: KVCacheSpec
    ) -> list[torch.Tensor] | torch.Tensor:
        if self.is_mla or self.is_mamba or self._cross_layers_blocks:
            return super().get_transfer_cache_regions(cache, layer_spec)
        # Iterating the tensor yields K and V; the caller divides the page size
        # by how many come back.
        return cache
