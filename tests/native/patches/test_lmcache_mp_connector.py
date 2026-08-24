# Copyright 2026 Rebellions Inc. All rights reserved.
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

# ``lmcache`` is an optional runtime dependency of vllm-rbln, so the patch
# target module cannot be imported for real here. Stub it in sys.modules
# before importing the patch so `patched_update_state_after_alloc`'s internal
# `from vllm...lmcache_mp_connector import ...` resolves to the stub.
import enum
import sys
import types
from types import SimpleNamespace
from typing import Any

_TARGET_MODULE = "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_mp_connector"


class _RequestState(enum.Enum):
    PREFETCHING = "prefetching"
    WAITING_FOR_LOAD = "waiting_for_load"
    READY = "ready"


_stub: Any = types.ModuleType(_TARGET_MODULE)
_stub.LMCacheMPRequestState = _RequestState
_stub.reformat_block_ids = lambda block_ids: block_ids[0]
_stub.logger = SimpleNamespace(debug=lambda *args, **kwargs: None)
sys.modules[_TARGET_MODULE] = _stub

from vllm_rbln.patches.lmcache_mp_connector import (  # noqa: E402
    patched_update_state_after_alloc,
)


class _Tracker:
    def __init__(self, num_lmcache_hit_blocks: int, num_vllm_hit_blocks: int) -> None:
        self.allocated_block_ids: list[int] = []
        self.num_lmcache_hit_blocks = num_lmcache_hit_blocks
        self.num_vllm_hit_blocks = num_vllm_hit_blocks
        self.state = _RequestState.PREFETCHING
        self.all_token_ids = list(range(64))

    def append_block_ids(self, block_ids: list[int]) -> None:
        self.allocated_block_ids.extend(block_ids)

    def needs_retrieve(self) -> bool:
        return self.num_lmcache_hit_blocks > self.num_vllm_hit_blocks


class _SchedulerAdapter:
    def __init__(self) -> None:
        self.cleaned_up: list[str] = []
        self.freed: list[dict[str, object]] = []

    def cleanup_lookup_result(self, request_id: str) -> None:
        self.cleaned_up.append(request_id)

    def free_lookup_locks(self, token_ids, start, end, request_id) -> None:
        self.freed.append({"start": start, "end": end, "request_id": request_id})


def _connector(tracker: _Tracker) -> SimpleNamespace:
    return SimpleNamespace(
        _get_request_tracker=lambda request_id: tracker,
        scheduler_adapter=_SchedulerAdapter(),
        vllm_block_size=16,
    )


def test_non_chosen_connector_transitions_to_ready_and_frees_all_locks():
    tracker = _Tracker(num_lmcache_hit_blocks=4, num_vllm_hit_blocks=1)
    connector = _connector(tracker)
    blocks = SimpleNamespace(get_block_ids=lambda: ([1, 2, 3, 4],))

    patched_update_state_after_alloc(
        connector, SimpleNamespace(request_id="req-0"), blocks, 0
    )

    assert tracker.state == _RequestState.READY
    assert connector.scheduler_adapter.freed == [
        {"start": 0, "end": 4 * 16, "request_id": "req-0"}
    ]


def test_chosen_connector_waits_for_load_and_frees_only_vllm_hit_locks():
    tracker = _Tracker(num_lmcache_hit_blocks=4, num_vllm_hit_blocks=1)
    connector = _connector(tracker)
    blocks = SimpleNamespace(get_block_ids=lambda: ([1, 2, 3, 4],))

    patched_update_state_after_alloc(
        connector, SimpleNamespace(request_id="req-1"), blocks, 100
    )

    assert tracker.state == _RequestState.WAITING_FOR_LOAD
    assert connector.scheduler_adapter.freed == [
        {"start": 0, "end": 1 * 16, "request_id": "req-1"}
    ]
