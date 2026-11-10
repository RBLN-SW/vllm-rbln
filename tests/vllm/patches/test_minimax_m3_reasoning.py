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

"""MiniMax-M3's reasoning state comes from the generation prompt alone.

The tokenizer and the cases are vllm-project/vllm#63492's
(`tests/reasoning/test_minimax_m3_reasoning_parser.py`), cut to what the
patch changes.
"""

import string
from collections.abc import Sequence

import pytest
from vllm.reasoning.minimax_m3_reasoning_parser import MiniMaxM3ReasoningParser

from vllm_rbln.patches.minimax_m3_reasoning import (
    patched_is_reasoning_end,
    patched_is_reasoning_end_streaming,
)


class MiniMaxM3Tokenizer:
    """Character tokenizer with M3's reasoning markers and message-begin token."""

    special_tokens = ("<mm:think>", "</mm:think>", "]~b]")

    def __init__(self):
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}
        for token in self.special_tokens:
            self._add_token(token)
        for char in string.printable:
            self._add_token(char)

    def _add_token(self, token: str) -> int:
        token_id = self._token_to_id.get(token)
        if token_id is None:
            token_id = len(self._token_to_id) + 1
            self._token_to_id[token] = token_id
            self._id_to_token[token_id] = token
        return token_id

    def get_vocab(self) -> dict[str, int]:
        return dict(self._token_to_id)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [self._add_token(token) for token in self.tokenize(text)]

    def decode(
        self, ids: Sequence[int] | int, skip_special_tokens: bool = False
    ) -> str:
        if isinstance(ids, int):
            ids = [ids]
        return "".join(self._id_to_token[token_id] for token_id in ids)

    def tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        pos = 0
        while pos < len(text):
            for special_token in self.special_tokens:
                if text.startswith(special_token, pos):
                    tokens.append(special_token)
                    pos += len(special_token)
                    break
            else:
                tokens.append(text[pos])
                pos += 1
        return tokens

    def convert_ids_to_tokens(
        self, ids: Sequence[int], skip_special_tokens: bool = False
    ) -> list[str]:
        return [self._id_to_token[token_id] for token_id in ids]

    def convert_tokens_to_ids(self, tokens: str | list[str]) -> int | list[int]:
        if isinstance(tokens, str):
            return self._add_token(tokens)
        return [self._add_token(token) for token in tokens]

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return "".join(tokens)


# Rendered like the M3 chat template: the system message spells out both
# markers, and a prior assistant turn without reasoning starts with the closer.
PROMPT = (
    "]~b]system\nWrap your reasoning in <mm:think></mm:think> tags, or begin "
    "directly after the </mm:think> prefix.[e~[\n"
    "]~b]user\nCapital of France?[e~[\n"
    "]~b]ai\n</mm:think>Paris.[e~[\n"
    "]~b]user\nAnd Germany?[e~[\n"
    "]~b]ai\n"
)


@pytest.fixture
def parser_and_tokenizer() -> tuple[MiniMaxM3ReasoningParser, MiniMaxM3Tokenizer]:
    tokenizer = MiniMaxM3Tokenizer()
    return MiniMaxM3ReasoningParser(tokenizer), tokenizer


def test_the_patch_is_the_one_installed():
    assert MiniMaxM3ReasoningParser.is_reasoning_end is patched_is_reasoning_end
    assert (
        MiniMaxM3ReasoningParser.is_reasoning_end_streaming
        is patched_is_reasoning_end_streaming
    )


def test_prompt_markers_do_not_end_reasoning(parser_and_tokenizer):
    parser, tokenizer = parser_and_tokenizer

    assert not parser.is_reasoning_end(tokenizer.encode(PROMPT))
    assert not parser.is_reasoning_end(tokenizer.encode(PROMPT + "<mm:think>ab"))
    assert parser.is_reasoning_end(tokenizer.encode(PROMPT + "<mm:think>a</mm:think>"))
    assert parser.is_reasoning_end(tokenizer.encode(PROMPT + "</mm:think>Berlin"))


def test_streaming_end_check_ignores_prompt_markers(parser_and_tokenizer):
    parser, tokenizer = parser_and_tokenizer
    prompt_ids = tokenizer.encode(PROMPT)
    open_ids = tokenizer.encode("<mm:think>ab")
    closed_ids = tokenizer.encode("</mm:think>")

    assert not parser.is_reasoning_end_streaming(prompt_ids + open_ids, open_ids[-1:])
    assert parser.is_reasoning_end_streaming(
        prompt_ids + open_ids + closed_ids, closed_ids
    )


def test_output_without_a_message_begin_token_is_read_whole(parser_and_tokenizer):
    parser, tokenizer = parser_and_tokenizer

    assert parser.is_reasoning_end(tokenizer.encode("<mm:think>a</mm:think>b"))
    assert not parser.is_reasoning_end(tokenizer.encode("<mm:think>a"))
