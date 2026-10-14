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

"""Tokens the Harmony parser rejects are dropped instead of failing the chunk.

A bare parser: ``process_chunk`` only drives the openai_harmony
``StreamableParser`` it creates from the harmony encoding, so no reasoning
parser, tool parser or tokenizer is needed.
"""

import pytest
from vllm.entrypoints.openai.parser.harmony_utils import get_encoding
from vllm.parser.harmony import ChunkResult, HarmonyParser

from vllm_rbln.patches.harmony import patched_process_chunk

_ANSWER = (
    "<|channel|>analysis<|message|>think<|end|>"
    "<|start|>assistant<|channel|>final<|message|>answer<|return|>"
)
_TOOL_CALL = (
    "<|channel|>analysis<|message|>think<|end|>"
    "<|start|>assistant<|channel|>commentary to=functions.get_weather"
    '<|message|>{"city":"Seoul"}<|call|>'
)


def _encode(harmony_str: str) -> list[int]:
    return get_encoding().encode(harmony_str, allowed_special="all")


def _deltas(result: ChunkResult) -> str:
    return "".join(segment.delta for segment in result.segments if segment.delta)


@pytest.fixture
def harmony_parser() -> HarmonyParser:
    # Only flush()'s recovery path reads the tokenizer; no test here reaches it.
    return HarmonyParser(tokenizer=None)


def test_the_patch_is_the_one_installed():
    assert HarmonyParser.process_chunk is patched_process_chunk


def test_a_stop_token_where_start_is_due_does_not_raise(harmony_parser):
    # vllm-project/vllm#59254's own case.
    malformed = _encode(
        "<|channel|>analysis<|message|>think<|end|><|return|>"
        "<|start|>assistant<|channel|>final<|message|>answer<|return|>"
    )

    result = harmony_parser.process_chunk(malformed)

    assert _deltas(result) == "thinkanswer"


@pytest.mark.parametrize(
    ("message", "deltas"),
    [
        (_ANSWER, "thinkanswer"),
        (_TOOL_CALL, 'think{"city":"Seoul"}'),
    ],
)
def test_sampling_past_the_stop_token_keeps_what_was_parsed(
    harmony_parser, message, deltas
):
    # What ignore_eos produces: the message ends, and plain text keeps coming.
    # Every token after the stop token is dropped: no segment, no reasoning
    # count, and the parsed message is untouched.
    parsed = _encode(message)
    trailing = _encode("\n\nand more text after the end")

    with_tail = harmony_parser.process_chunk(parsed + trailing)
    alone = HarmonyParser(tokenizer=None).process_chunk(parsed)

    assert _deltas(with_tail) == deltas
    assert len(with_tail.segments) == len(parsed)
    assert with_tail.reasoning_token_count == alone.reasoning_token_count


def test_the_stream_ends_cleanly_after_the_skipped_tokens(harmony_parser):
    # The end of a streamed response: flush() closes the parser without taking
    # its raw-output recovery, so the skipped tail does not come back as text.
    harmony_parser.process_chunk(_encode(_ANSWER + "\n\nand more text"))

    flushed = harmony_parser.flush()

    assert all(not segment.delta for segment in flushed)
