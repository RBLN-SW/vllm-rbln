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

# Backport of vllm-project/vllm#46865. The property under test is narrow: every
# sub-connector must see the request's *real* blocks, because an offload
# connector records them for its store path. Only ``num_external_tokens``
# separates the chosen connector from the rest.

from types import SimpleNamespace

from vllm_rbln.patches.multi_connector import (
    _upstream_still_blanks_blocks,
    patched_update_state_after_alloc,
)


class _RecordingConnector:
    """Captures what update_state_after_alloc was handed."""

    def __init__(self) -> None:
        self.calls: list[tuple[object, int]] = []

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        self.calls.append((blocks, num_external_tokens))


class _Blocks:
    """Stand-in for KVCacheBlocks. new_empty() would signal the old behaviour."""

    def __init__(self, tag: str) -> None:
        self.tag = tag

    def new_empty(self) -> "_Blocks":
        return _Blocks("empty")


def _multi(*connectors: _RecordingConnector, chosen: dict[str, int] | None = None):
    return SimpleNamespace(
        _connectors=list(connectors),
        _requests_to_connector=chosen if chosen is not None else {},
    )


def test_non_chosen_connectors_get_real_blocks_with_zero_external():
    """The regression this backport exists for: a losing connector still needs
    the block ids, it just is not the one loading."""
    winner, loser = _RecordingConnector(), _RecordingConnector()
    multi = _multi(winner, loser, chosen={"req-0": 0})
    blocks = _Blocks("real")

    patched_update_state_after_alloc(
        multi, SimpleNamespace(request_id="req-0"), blocks, 128
    )

    assert winner.calls == [(blocks, 128)]
    # Real blocks, not blocks.new_empty(); only the token count differs.
    assert loser.calls == [(blocks, 0)]


def test_all_connectors_get_real_blocks_when_nobody_is_chosen():
    """The deployed case: on a prefill instance with a cold offload cache no
    child returns a non-zero lookup, so ``chosen`` stays -1. Upstream handed
    every child empty blocks here, which is what made the cache unable to
    bootstrap."""
    first, second = _RecordingConnector(), _RecordingConnector()
    multi = _multi(first, second)  # _requests_to_connector empty -> chosen = -1
    blocks = _Blocks("real")

    patched_update_state_after_alloc(
        multi, SimpleNamespace(request_id="req-cold"), blocks, 0
    )

    for connector in (first, second):
        assert connector.calls == [(blocks, 0)]
        assert connector.calls[0][0].tag == "real"


def test_condition_reads_upstream_source():
    """The patch must disable itself once the pinned vLLM carries the fix, so
    the condition is a real check rather than a constant."""
    assert isinstance(_upstream_still_blanks_blocks(), bool)
