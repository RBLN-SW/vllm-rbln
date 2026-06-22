# Copyright 2025 Rebellions Inc. All rights reserved.
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

import asyncio
import os

import openai
import pytest
import pytest_asyncio
from utils import RemoteOpenAIServer

MODEL_DIR = os.getenv("REBEL_VLLM_PRE_COMPILED_DIR", "./")
MODEL_NAME = MODEL_DIR + "/opt_125m_batch2"
MAX_TOKENS = 1


@pytest.fixture(scope="module")
def server_args(request: pytest.FixtureRequest) -> list[str]:
    """Provide extra arguments to the server via indirect parametrization

    Usage:

    >>> @pytest.mark.parametrize(
    >>>     "server_args",
    >>>     [
    >>>         [
    >>>             "--model=NousResearch/Hermes-3-Llama-3.1-70B",
    >>>             "--enable-auto-tool-choice",
    >>>         ],
    >>>     ],
    >>>     indirect=True,
    >>> )
    >>> def test_foo(server, client):
    >>>     ...

    This will run `test_foo` with a server with:
    - `--model=NousResearch/Hermes-3-Llama-3.1-70B --enable-auto-tool-choice`.

    """
    if not hasattr(request, "param"):
        return []

    val = request.param

    if isinstance(val, str):
        return [val]

    return request.param


@pytest.fixture(scope="module")
def server(server_args):
    args = [
        *server_args,
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
async def test_request_cancellation(server: RemoteOpenAIServer):
    # clunky test: send an ungodly amount of load in with short timeouts
    # then ensure that it still responds quickly afterwards

    prompt = "Write a long story"
    client = server.get_async_client(timeout=0.1)
    tasks = []
    # To make timeout while generating tokens
    for _ in range(10):
        task = asyncio.create_task(
            client.completions.create(
                prompt=prompt,
                model=MODEL_NAME,
                max_tokens=100,
                extra_body={"min_tokens": 100},
            )
        )
        tasks.append(task)

    done, pending = await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)

    # Make sure all requests were sent to the server and timed out
    # (We don't want to hide other errors like 400s that would invalidate this
    # test)
    assert len(pending) == 0
    for d in done:
        with pytest.raises(openai.APITimeoutError):
            d.result()

    # If the server had not cancelled all the other requests, then it would not
    # be able to respond to this one within the timeout
    client = server.get_async_client(timeout=200)
    response = await client.completions.create(
        prompt=prompt, model=MODEL_NAME, max_tokens=MAX_TOKENS
    )

    assert len(response.choices) == 1


@pytest.mark.asyncio
async def test_request_wrong_content_type(server: RemoteOpenAIServer):
    prompt = "Write a long story"
    client = server.get_async_client()

    with pytest.raises(openai.APIStatusError):
        await client.completions.create(
            prompt=prompt,
            model=MODEL_NAME,
            max_tokens=MAX_TOKENS,
            extra_headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
