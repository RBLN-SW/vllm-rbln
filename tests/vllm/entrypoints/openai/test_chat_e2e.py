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

"""Chat requests in the shape most harnesses send: no sampler fields, so the
model's generation_config.json picks a random sampler, and a required tool call,
so a grammar bounds every token. None depends on output quality."""

import asyncio
import json

import openai
import pytest
import pytest_asyncio

from tests.utils import RemoteOpenAIServer
from tests.vllm.runners import rbln_engine_args
from tests.vllm.vllm_config import local_weights_path

pytestmark = pytest.mark.model_compile

MODEL = "Qwen/Qwen3-0.6B"
MESSAGES = [{"role": "user", "content": "What is the weather in Paris?"}]
# A strict tool holds the arguments to the schema. A truncated model may still
# fill the JSON whitespace up to the cap, so only a completion that closed must
# parse. Every request must finish without an error, which a token outside the
# grammar causes.
CITIES = ["Paris", "Seoul"]
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather in a city.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "enum": CITIES}},
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    }
]
MAX_COMPLETION_TOKENS = 64
NUM_CONCURRENT = 4


@pytest.fixture(scope="module")
def server():
    path = local_weights_path(MODEL)
    engine_args = rbln_engine_args(path, max_num_seqs=NUM_CONCURRENT)
    args = ["--model-impl", "vllm", "--served-model-name", MODEL]
    args += ["--enable-auto-tool-choice", "--tool-call-parser", "hermes"]
    for key, value in engine_args.items():
        flag = "--" + key.replace("_", "-")
        if value is True:
            args.append(flag)
        elif value is not False:
            args += [flag, str(value)]
    with RemoteOpenAIServer(path, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


def assert_tool_calls(
    finish_reason: str | None, completion_tokens: int, calls: list[tuple[str, str]]
) -> None:
    if finish_reason == "length":
        assert completion_tokens == MAX_COMPLETION_TOKENS
        return
    assert calls
    for name, arguments in calls:
        assert name == "get_weather"
        assert json.loads(arguments)["city"] in CITIES


@pytest.mark.asyncio
async def test_required_tool_call_without_sampler_fields(client: openai.AsyncOpenAI):
    responses = await asyncio.gather(
        *[
            client.chat.completions.create(
                model=MODEL,
                messages=MESSAGES,
                tools=TOOLS,
                tool_choice="required",
                max_completion_tokens=MAX_COMPLETION_TOKENS,
            )
            for _ in range(NUM_CONCURRENT)
        ]
    )

    for response in responses:
        (choice,) = response.choices
        assert_tool_calls(
            choice.finish_reason,
            response.usage.completion_tokens,
            [
                (call.function.name, call.function.arguments)
                for call in choice.message.tool_calls or []
            ],
        )


@pytest.mark.asyncio
async def test_streamed_tool_call_arguments_join_into_json(
    client: openai.AsyncOpenAI,
):
    stream = await client.chat.completions.create(
        model=MODEL,
        messages=MESSAGES,
        tools=TOOLS,
        tool_choice="required",
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        stream=True,
        stream_options={"include_usage": True},
    )
    chunks = [chunk async for chunk in stream]

    # include_usage adds one last chunk that carries the usage and no choice.
    *deltas, usage = chunks
    assert usage.choices == []
    names: dict[int, str] = {}
    arguments: dict[int, str] = {}
    for chunk in deltas:
        for call in chunk.choices[0].delta.tool_calls or []:
            if call.function.name:
                names[call.index] = call.function.name
            arguments[call.index] = arguments.get(call.index, "") + (
                call.function.arguments or ""
            )
    assert_tool_calls(
        deltas[-1].choices[0].finish_reason,
        usage.usage.completion_tokens,
        [(name, arguments.get(index, "")) for index, name in names.items()],
    )


@pytest.mark.asyncio
async def test_max_completion_tokens_caps_a_sampled_completion(
    client: openai.AsyncOpenAI,
):
    response = await client.chat.completions.create(
        model=MODEL, messages=MESSAGES, max_completion_tokens=5
    )

    (choice,) = response.choices
    assert response.usage.completion_tokens <= 5
    if choice.finish_reason == "length":
        assert response.usage.completion_tokens == 5
