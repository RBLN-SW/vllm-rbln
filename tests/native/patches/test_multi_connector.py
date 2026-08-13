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

from types import SimpleNamespace

from vllm_rbln.patches.multi_connector import (
    _upstream_still_blanks_blocks,
    patched_update_state_after_alloc,
)


class _RecordingConnector:
    def __init__(self) -> None:
        self.calls: list[tuple[object, int]] = []

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        self.calls.append((blocks, num_external_tokens))


class _Blocks:
    def __init__(self, tag: str) -> None:
        self.tag = tag

    def new_empty(self) -> "_Blocks":
        return _Blocks("empty")


def _multi(*connectors: _RecordingConnector, chosen: dict[str, int] | None = None):
    return SimpleNamespace(
        _connectors=list(connectors),
        _requests_to_connector=chosen if chosen is not None else {},
    )


def test_non_chosen_connector_gets_real_blocks_with_zero_external():
    winner, loser = _RecordingConnector(), _RecordingConnector()
    multi = _multi(winner, loser, chosen={"req-0": 0})
    blocks = _Blocks("real")

    patched_update_state_after_alloc(
        multi, SimpleNamespace(request_id="req-0"), blocks, 128
    )

    assert winner.calls == [(blocks, 128)]
    assert loser.calls == [(blocks, 0)]


def test_all_connectors_get_real_blocks_when_nobody_is_chosen():
    # The deployed path: a cold offload cache means no child returns a non-zero
    # lookup, so chosen stays -1.
    first, second = _RecordingConnector(), _RecordingConnector()
    multi = _multi(first, second)
    blocks = _Blocks("real")

    patched_update_state_after_alloc(
        multi, SimpleNamespace(request_id="req-cold"), blocks, 0
    )

    for connector in (first, second):
        assert connector.calls == [(blocks, 0)]


def test_condition_is_a_real_check():
    assert isinstance(_upstream_still_blanks_blocks(), bool)
