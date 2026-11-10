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
"""Read MiniMax-M3's reasoning state from the generation prompt only.

Backport of vllm-project/vllm#63492. The M3 chat template writes
``<mm:think></mm:think>`` and ``</mm:think>`` into the system message, so
v0.31.0's ``is_reasoning_end`` finds a closed reasoning block in every prompt.
The streaming path then sends the whole output, ``<mm:think>`` included, as
``content``. Both bodies are #63492's: v0.31.0's plus the one line that cuts
``input_ids`` at the last message-begin token, with its ``_last_message_ids``
method as a module function. They are kept identical so that this module is
deleted, not merged.

Like every registry patch, it applies on the vllm model path only.

TODO(vllm-project/vllm#63492): delete this module and its test once the pinned
vllm release carries the fix; the assert below fails on that release.
"""

from collections.abc import Iterable, Sequence

from vllm.reasoning.minimax_m3_reasoning_parser import MiniMaxM3ReasoningParser

from vllm_rbln.patches import register_patch

assert not hasattr(MiniMaxM3ReasoningParser, "_last_message_ids"), (
    "This vllm carries vllm-project/vllm#63492. Delete "
    "vllm_rbln/patches/minimax_m3_reasoning.py, its import in "
    "vllm_rbln/patches/__init__.py and "
    "tests/vllm/patches/test_minimax_m3_reasoning.py."
)

# The chat template opens every message, including the generation prompt, with
# this token.
MESSAGE_BEGIN = "]~b]"


def last_message_ids(
    parser: MiniMaxM3ReasoningParser, input_ids: Sequence[int]
) -> Sequence[int]:
    """The ids after the last message-begin token; output-only ids come back whole.

    Args:
        parser (MiniMaxM3ReasoningParser): the parser, whose cached vocab names
            the message-begin token.
        input_ids (Sequence[int]): prompt ids, optionally followed by output ids.
    """
    begin = parser.vocab.get(MESSAGE_BEGIN)
    for i in range(len(input_ids) - 1, -1, -1):
        if input_ids[i] == begin:
            return input_ids[i + 1 :]
    return input_ids


@register_patch(
    target=(
        "vllm.reasoning.minimax_m3_reasoning_parser."
        "MiniMaxM3ReasoningParser.is_reasoning_end"
    ),
    reason=(
        "v0.31.0 reads the reasoning markers the M3 chat template writes into "
        "the system message as a finished reasoning block, so a streamed "
        "response carries <mm:think> in content. vllm-project/vllm#63492 looks "
        "at the generation prompt only; no vllm release carries it yet."
    ),
    key="vllm_rbln.patches.minimax_m3_reasoning.is_reasoning_end",
    owner_module="vllm_rbln.patches.minimax_m3_reasoning",
)
def patched_is_reasoning_end(
    self: MiniMaxM3ReasoningParser, input_ids: Sequence[int]
) -> bool:
    input_ids = last_message_ids(self, input_ids)
    start_index = self._rfind_token_sequence(input_ids, self._start_token_ids)
    end_index = self._rfind_token_sequence(input_ids, self._end_token_ids)
    if end_index < 0:
        return False
    if start_index < 0:
        return True
    return end_index > start_index


@register_patch(
    target=(
        "vllm.reasoning.minimax_m3_reasoning_parser."
        "MiniMaxM3ReasoningParser.is_reasoning_end_streaming"
    ),
    reason=(
        "is_reasoning_end_streaming reads the same template markers out of the "
        "prompt part of input_ids. vllm-project/vllm#63492 scopes it the same way."
    ),
    key="vllm_rbln.patches.minimax_m3_reasoning.is_reasoning_end_streaming",
    owner_module="vllm_rbln.patches.minimax_m3_reasoning",
)
def patched_is_reasoning_end_streaming(
    self: MiniMaxM3ReasoningParser,
    input_ids: Sequence[int],
    delta_ids: Iterable[int],
) -> bool:
    if self._reasoning_ended_streaming:
        return True

    if self._reasoning_active_streaming or self._pending_marker_streaming:
        return False

    delta_ids = tuple(delta_ids)
    input_ids = last_message_ids(self, input_ids)
    if self._contains_token_sequence(delta_ids, self._end_token_ids):
        return True
    if self._contains_token_sequence(input_ids, self._end_token_ids):
        return True
    if self._initial_in_reasoning:
        return False
    if self._ends_with_token_sequence_prefix(input_ids, self._start_token_ids):
        return False
    if self._ends_with_token_sequence_prefix(input_ids, self._end_token_ids):
        return False
    if not self._contains_token_sequence(input_ids, self._start_token_ids):
        return bool(input_ids)
    return False
