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
"""MiniMax-M3 reasoning parser that reads the reasoning state from the last message.

Backport of vllm-project/vllm#63492. The M3 chat template writes
``<mm:think></mm:think>`` and ``</mm:think>`` into the system message, so
v0.31.0's parser finds a closed reasoning block in every prompt and the
streaming path sends the whole output as ``content``. Only the generation
prompt says whether generation starts inside reasoning.

TODO(vllm>0.33.0): delete with its registration once the pinned vLLM carries
vllm-project/vllm#63492; ``test_upstream_does_not_carry_the_fix`` fails then.
"""

from collections.abc import Iterable, Sequence

from vllm.reasoning.minimax_m3_reasoning_parser import MiniMaxM3ReasoningParser

from vllm_rbln.logger import init_logger

logger = init_logger(__name__)

# The chat template opens every message, including the generation prompt, with
# this token.
MESSAGE_BEGIN = "]~b]"


class RBLNMiniMaxM3ReasoningParser(MiniMaxM3ReasoningParser):
    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self._message_begin_token_id = self.vocab.get(MESSAGE_BEGIN)
        if self._message_begin_token_id is None:
            logger.warning(
                "%r is not a single token in this tokenizer, so the MiniMax-M3 "
                "reasoning parser reads markers from the whole prompt, and a "
                "streamed response may carry its reasoning in content.",
                MESSAGE_BEGIN,
            )

    def _last_message_ids(self, input_ids: Sequence[int]) -> Sequence[int]:
        # Output-only ids carry no message-begin token and come back whole.
        if self._message_begin_token_id is None:
            return input_ids
        for i in range(len(input_ids) - 1, -1, -1):
            if input_ids[i] == self._message_begin_token_id:
                return input_ids[i + 1 :]
        return input_ids

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        return super().is_reasoning_end(self._last_message_ids(input_ids))

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        return super().is_reasoning_end_streaming(
            self._last_message_ids(input_ids), delta_ids
        )
