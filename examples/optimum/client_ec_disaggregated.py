# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Client for EC disaggregated serving (producer + consumer).

Prerequisites:
  1. Start producer:  bash serve_ec_producer.sh <model_id> 8000
  2. Start consumer:  bash serve_ec_consumer.sh <model_id> 8001

Usage:
  python client_ec_disaggregated.py --num-requests 3

Flow per request:
  1. Send request to producer (fire-and-forget, triggers vision encoding)
  2. Send same request to consumer (waits for encoder cache via NIXL, decodes)
  3. Collect result from consumer
"""

import asyncio
import base64
import random
from io import BytesIO

import fire
import httpx
from datasets import load_dataset


def _make_chat_request(image_b64: str, question: str, max_tokens: int = 200):
    """Build an OpenAI-compatible chat completion request body."""
    return {
        "model": "Qwen2-VL-7B-Instruct",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant. "
                "Answer each question based on the image.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}",
                        },
                    },
                    {"type": "text", "text": question},
                ],
            },
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
    }


async def _send_request(
    client: httpx.AsyncClient,
    url: str,
    body: dict,
    timeout: float = 120.0,
) -> httpx.Response:
    return await client.post(
        url,
        json=body,
        timeout=timeout,
    )


async def process_request(
    client: httpx.AsyncClient,
    producer_url: str,
    consumer_url: str,
    body: dict,
    request_id: int,
) -> str:
    """Send the same request to producer (fire-and-forget) then consumer."""
    # Fire request to producer — triggers vision encoding + NIXL registration.
    # We don't need the generated text from producer, but must wait for it
    # to finish so the encoder cache is registered before consumer pulls.
    producer_task = asyncio.create_task(
        _send_request(client, producer_url, body)
    )

    # Wait for producer to finish encoding, then send to consumer.
    await producer_task

    # Consumer pulls encoder cache via NIXL and runs decoder.
    resp = await _send_request(client, consumer_url, body)
    resp.raise_for_status()
    result = resp.json()
    return result["choices"][0]["message"]["content"]


async def main(
    num_requests: int = 3,
    producer_port: int = 8000,
    consumer_port: int = 8001,
    host: str = "127.0.0.1",
    max_tokens: int = 200,
):
    producer_url = f"http://{host}:{producer_port}/v1/chat/completions"
    consumer_url = f"http://{host}:{consumer_port}/v1/chat/completions"

    # Load sample images
    seed = random.randint(0, 2**32 - 1)
    print(f"Using random seed: {seed}")
    dataset = load_dataset(
        "lmms-lab/llava-bench-in-the-wild", split="train"
    ).shuffle(seed=seed)

    # Encode images to base64
    requests = []
    for i in range(num_requests):
        buf = BytesIO()
        dataset[i]["image"].save(buf, format="PNG")
        image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        body = _make_chat_request(
            image_b64, dataset[i]["question"], max_tokens
        )
        requests.append(body)

    print(f"Sending {num_requests} requests to producer={producer_url}, "
          f"consumer={consumer_url}\n")

    async with httpx.AsyncClient() as client:
        tasks = [
            process_request(client, producer_url, consumer_url, body, i)
            for i, body in enumerate(requests)
        ]
        results = await asyncio.gather(*tasks)

    for i, text in enumerate(results):
        print(f"==================== Output {i} ==============================")
        print(text)
        print("===============================================================\n")


if __name__ == "__main__":
    fire.Fire(main)
