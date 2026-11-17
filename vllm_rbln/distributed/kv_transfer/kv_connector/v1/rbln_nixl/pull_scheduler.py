# Copyright 2025 Rebellions Inc. All rights reserved.
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

from typing import TYPE_CHECKING, Any, ClassVar

from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
    NixlPullConnectorScheduler,
)
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.base_scheduler import (
    RblnNixlSchedulerBase,
)
from vllm_rbln.v1.kv_cache import (
    REMOTE_STATE_BOUNDARY,
    prompt_boundary_block,
    state_boundary_is_reachable,
)

if TYPE_CHECKING:
    from vllm.v1.request import Request


class RblnNixlPullConnectorScheduler(RblnNixlSchedulerBase, NixlPullConnectorScheduler):
    """Scheduler side of the read path."""

    sends_state_boundary: ClassVar[bool] = True

    def get_exchange_clipped_blocks(
        self, block_ids: BlockIds, clip_ssm: bool = True
    ) -> BlockIds:
        """Also keep a single-state SSM group's block-boundary state.

        NOTE(RBLN): upstream keeps only the running state, so the consumer can
        decode but never caches the prompt. `RBLNMambaManager` holds the state at
        the prompt's last block boundary in the slot right before the running
        one -- kept by a producer, allocated by a consumer expecting it -- so a
        live block there goes along. Both ends clip the same way, and a consumer
        holding only the running state pairs with the tail of a producer's two.
        """
        clipped = super().get_exchange_clipped_blocks(block_ids, clip_ssm)
        if (
            not clip_ssm
            or not clipped
            or not self._is_hma_required
            or self._ssm_state_slots_are_positional
            or not state_boundary_is_reachable(self.vllm_config)
        ):
            return clipped
        selected = self.kv_cache_config.select_transfer_block_ids(block_ids)
        out = list(clipped)
        for i, blocks in enumerate(selected):
            n_spec = self._ssm_spec_blocks[i]
            if n_spec is None or len(blocks) < 2:
                continue
            if n_spec := min(n_spec, len(blocks) - 1):
                blocks = blocks[:-n_spec]
            if len(blocks) >= 2 and blocks[-2] != NULL_BLOCK_ID:
                out[i] = list(blocks[-2:])
        return tuple(out)

    def request_finished(
        self,
        request: "Request",
        block_ids: BlockIds,
    ) -> tuple[bool, dict[str, Any] | None]:
        delay_free_blocks, params = super().request_finished(request, block_ids)
        if (
            not delay_free_blocks
            or params is None
            or not params["do_remote_prefill"]
            or not self._has_mamba
        ):
            return delay_free_blocks, params
        groups = self.kv_cache_config.transfer_groups
        remote_block_ids = list(params["remote_block_ids"])
        for i, blocks in enumerate(remote_block_ids):
            if self._ssm_spec_blocks[i] is None or len(blocks) != 2:
                continue
            block_size = groups[i].kv_cache_spec.block_size
            if prompt_boundary_block(request.num_computed_tokens, block_size) is None:
                # A block-aligned prompt's running state is its boundary state;
                # the slot before it is a superseded one not yet freed.
                remote_block_ids[i] = blocks[-1:]
            else:
                params[REMOTE_STATE_BOUNDARY] = (
                    request.num_computed_tokens // block_size * block_size
                )
        params["remote_block_ids"] = tuple(remote_block_ids)
        return delay_free_blocks, params
