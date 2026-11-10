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

The tokenizer and the prompt are vllm-project/vllm#63492's
(`tests/reasoning/test_minimax_m3_reasoning_parser.py`). Every case but the
last fails without the registration.
"""

import string
from collections.abc import Sequence

import pytest
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.parser_manager import ParserManager
from vllm.reasoning import ReasoningParserManager
from vllm.reasoning.minimax_m3_reasoning_parser import MiniMaxM3ReasoningParser

from vllm_rbln.patches.minimax_m3_reasoning import (
    PARSER_NAME,
    register_minimax_m3_reasoning_parser,
)
from vllm_rbln.reasoning.minimax_m3_reasoning_parser import (
    RBLNMiniMaxM3ReasoningParser,
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


@pytest.fixture(autouse=True)
def registered() -> None:
    register_minimax_m3_reasoning_parser()


@pytest.fixture
def tokenizer() -> MiniMaxM3Tokenizer:
    return MiniMaxM3Tokenizer()


@pytest.fixture
def parser(tokenizer) -> MiniMaxM3ReasoningParser:
    return ReasoningParserManager.get_reasoning_parser(PARSER_NAME)(tokenizer)


def test_the_name_resolves_to_the_rbln_parser():
    parser_cls = ReasoningParserManager.get_reasoning_parser(PARSER_NAME)

    assert parser_cls is RBLNMiniMaxM3ReasoningParser


def test_prompt_markers_do_not_end_reasoning(parser, tokenizer):
    assert not parser.is_reasoning_end(tokenizer.encode(PROMPT))


def test_streaming_end_check_ignores_prompt_markers(parser, tokenizer):
    open_ids = tokenizer.encode("<mm:think>ab")

    assert not parser.is_reasoning_end_streaming(
        tokenizer.encode(PROMPT) + open_ids, open_ids[-1:]
    )


def test_a_streamed_response_keeps_its_reasoning_out_of_content(tokenizer):
    # The serving path: Parser.parse_delta asks is_reasoning_end(prompt) on the
    # first delta and, on True, sends every later delta to content.
    parser_cls = ParserManager.get_parser(reasoning_parser_name=PARSER_NAME)
    parser = parser_cls(tokenizer)
    request = ChatCompletionRequest(
        model="minimax-m3", messages=[{"role": "user", "content": "And Germany?"}]
    )
    prompt_ids = tokenizer.encode(PROMPT)

    reasoning, content = [], []
    for char in "<mm:think>ab</mm:think>Berlin":
        delta = parser.parse_delta(
            char, tokenizer.encode(char), request, prompt_ids, finished=False
        )
        if delta is not None:
            reasoning.append(delta.reasoning or "")
            content.append(delta.content or "")

    assert "".join(reasoning) == "ab"
    assert "".join(content) == "Berlin"


def test_upstream_does_not_carry_the_fix():
    # The removal condition: once this fails, delete the parser, its
    # registration and this file.
    assert not hasattr(MiniMaxM3ReasoningParser, "_last_message_ids")
