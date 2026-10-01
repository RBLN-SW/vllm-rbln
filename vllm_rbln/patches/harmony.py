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
"""Skip the tokens the Harmony parser rejects instead of failing the request.

Backport of vllm-project/vllm#59254 (issue vllm-project/vllm#50690). A request
that turns the stop tokens off -- ``ignore_eos`` -- keeps sampling after
``<|return|>`` or ``<|call|>``, and the next token is not the ``<|start|>`` the
parser expects. ``process_chunk`` let that ``HarmonyError`` escape, so a
streaming chat completion on gpt-oss answered HTTP 500 and lost the whole
response. Upstream now drops the offending token and carries on: the segments
already parsed keep their content and reasoning, and ``usage`` still counts
every generated token.

The body is upstream's, unchanged from v0.26.0 through v0.30.0, plus the
``try``/``except`` around ``process``.

TODO(vllm>=0.31.0): delete -- vllm-project/vllm#59254 merged to main after
v0.30.0, so the first release that carries it removes the need.
"""

from collections.abc import Sequence

from openai_harmony import HarmonyError
from vllm.parser.harmony import ChunkResult, HarmonyParser, Segment

from vllm_rbln.patches import register_patch


@register_patch(
    target="vllm.parser.harmony.HarmonyParser.process_chunk",
    reason=(
        "Upstream's process_chunk lets a HarmonyError escape when a token "
        "follows <|return|> or <|call|>, which a request with ignore_eos always "
        "produces, and the streaming chat completion fails with HTTP 500. "
        "vllm-project/vllm#59254 skips the token instead, and no release the "
        "pin can take carries it yet."
    ),
    key="vllm_rbln.patches.harmony.process_chunk",
    owner_module="vllm_rbln.patches.harmony",
)
def patched_process_chunk(self: HarmonyParser, token_ids: Sequence[int]) -> ChunkResult:
    if not token_ids:
        return ChunkResult(segments=[], reasoning_token_count=0)

    segments: list[Segment] = []
    reasoning_token_count = 0
    for token_id in token_ids:
        try:
            self._harmony_parser.process(token_id)
        except HarmonyError:
            continue
        channel = self._harmony_parser.current_channel
        recipient = self._normalize_recipient(self._harmony_parser.current_recipient)
        delta = self._harmony_parser.last_content_delta or ""
        completed_message = self._poll_completed_message()

        if completed_message is not None:
            self._current_message_tokens.clear()
        else:
            self._current_message_tokens.append(token_id)

        if channel == "analysis" or (channel == "commentary" and recipient is not None):
            reasoning_token_count += 1

        segments.append(
            Segment(
                channel=channel,
                recipient=recipient,
                delta=delta,
                completed_message=completed_message,
            )
        )

    return ChunkResult(
        segments=segments,
        reasoning_token_count=reasoning_token_count,
    )
