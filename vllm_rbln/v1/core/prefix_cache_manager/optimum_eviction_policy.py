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
from collections import OrderedDict

from vllm_rbln.logger import init_logger

from .optimum_block_mapping_manager import BlockMappingManager

logger = init_logger(__name__)


class LRUEvictionPolicy:
    """
    LRU (Least Recently Used) eviction policy implementation.

    Recency comes from `touch`. The prefix cache manager touches a
    sequence's blocks in reverse order (front of the prefix last), so
    within one sequence the tail is always evicted before the front.
    Matching stops at the first missing block, so evicting the front
    would invalidate the whole cached prefix while evicting the tail
    only shortens the hit.
    """

    def __init__(self):
        self._access_order: OrderedDict[int, bool] = OrderedDict()

    def touch(self, block_id: int) -> None:
        """Mark a block as recently accessed"""
        # Move to the end to mark as most recently used
        self._access_order.move_to_end(block_id)

    def register_block(self, block_id: int) -> None:
        assert block_id not in self._access_order
        self._access_order[block_id] = True

    def unregister_block(self, block_id: int) -> None:
        self._access_order.pop(block_id, None)

    def select_blocks_for_eviction(
        self, mapping_manager: BlockMappingManager, count: int
    ) -> list[int]:
        inactive_block_ids = {
            mapping.outer_block_id
            for mapping in mapping_manager.get_inactive_mappings()
        }

        evictable_blocks = [
            block_id
            for block_id in self._access_order
            if block_id in inactive_block_ids
        ]
        if len(evictable_blocks) < count:
            return []
        return evictable_blocks[:count]
