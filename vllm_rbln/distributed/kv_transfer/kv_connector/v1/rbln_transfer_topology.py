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

from vllm.distributed.kv_transfer.kv_connector.utils import (
    EngineId,
    EngineTransferInfo,
    TransferTopology,
)


class RblnTransferTopology(TransferTopology):
    """Upstream's topology, refusing the attention cache RBLN now allocates.

    The RBLN attention cache is ``(num_blocks, 2, H, 1, block_size, D)``, so K
    and V sit inside the same block and cannot become two NIXL regions. The
    descriptor path has not moved to upstream's interleaved-KV split yet, so
    this refuses that cache at handshake time rather than transferring halves
    of itself. MLA and Mamba caches are unaffected and stay on upstream's path.

    ``__post_init__`` reimplements upstream's rather than extending it, since
    upstream's derives the layout from a 5-dim shape RBLN never produces. So a
    field upstream sets there has to be set here too, and leaving one out
    fails nowhere until the first handshake reads it.
    """

    def __post_init__(self) -> None:
        self.local_physical_heads = max(1, self.total_num_kv_heads // self.tp_size)
        self._engines: dict[tuple[EngineId, int], EngineTransferInfo] = {}
        self._cross_layers_blocks = False
        self._is_kv_layout_blocks_first = self.is_mamba
        if self.is_mamba:
            # Upstream skips the shape for the same reason: a Mamba cache is a
            # (conv, ssm) pair, and the connector hands it no tensor shape.
            return
        shape = self.attn_backends[0].get_kv_cache_shape(
            num_blocks=1, block_size=16, num_kv_heads=1, head_size=1
        )
        if not self.is_mla:
            raise NotImplementedError(
                "RBLN NIXL cuts K and V out of separate regions, which the "
                f"blocks-first attention cache {shape} from "
                f"{self.attn_backends[0].__name__} interleaves inside each "
                "block. Disaggregated serving on RBLN is limited to MLA until "
                "the descriptor path moves to upstream's interleaved-KV split."
            )
        assert shape[:1] == (1,), (
            "RBLN NIXL descriptors assume a (num_blocks, ...) MLA cache, got "
            f"{shape} from {self.attn_backends[0].__name__}."
        )
        self._cross_layers_blocks = (
            self.tensor_shape is not None and len(self.tensor_shape) == len(shape) + 1
        )
