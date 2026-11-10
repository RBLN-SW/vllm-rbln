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

from vllm_rbln.patches.registry import add_registration

PARSER_NAME = "minimax_m3"


@add_registration(
    reason=(
        "v0.31.0's minimax_m3 reasoning parser reads the markers the M3 chat "
        "template writes into the system message as a finished reasoning block, "
        "so a streamed response carries <mm:think> in content. "
        "vllm-project/vllm#63492 fixes it; no vLLM release carries it yet."
    )
)
def register_minimax_m3_reasoning_parser() -> None:
    from vllm.reasoning import ReasoningParserManager

    # Lazy, so only a process that builds the parser imports it. The frontend
    # and structured output both resolve the parser by this name.
    ReasoningParserManager.register_lazy_module(
        name=PARSER_NAME,
        module_path="vllm_rbln.reasoning.minimax_m3_reasoning_parser",
        class_name="RBLNMiniMaxM3ReasoningParser",
    )
    # A lookup before this point cached upstream's class, which wins over lazy.
    ReasoningParserManager.reasoning_parsers.pop(PARSER_NAME, None)
