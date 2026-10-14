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

Backport of vllm-project/vllm#59254 (issue vllm-project/vllm#50690). With
``ignore_eos`` a request keeps sampling after ``<|return|>`` or ``<|call|>``,
and the next token is not the ``<|start|>`` the parser expects.
``process_chunk`` let that ``HarmonyError`` escape, and the chat completion
answered HTTP 500. The body is upstream's, unchanged from v0.26.0 through
v0.30.0, plus upstream's ``try``/``except`` around ``process``. It is kept
identical so that this module is deleted, not merged.

Like every registry patch, it applies on the vllm model path only.

TODO(vllm>=0.31.0): delete this module, its test and the assert below.
"""

from collections.abc import Sequence
from importlib.metadata import version

from openai_harmony import HarmonyError
from packaging.version import Version
from vllm.parser.harmony import ChunkResult, HarmonyParser, Segment

from vllm_rbln.patches import register_patch

assert Version(version("vllm")) < Version("0.31.0"), (
    "vllm-project/vllm#59254 merged after v0.30.0. Once this vllm carries it, "
    "delete vllm_rbln/patches/harmony.py, its import in "
    "vllm_rbln/patches/__init__.py and tests/vllm/patches/test_harmony.py."
)


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
