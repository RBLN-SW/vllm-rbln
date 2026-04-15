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
Client for EC disaggregated serving (multi-producer + consumer).

Prerequisites:
  1. Start producers:  NUM_PRODUCERS=6 bash serve_ec_producer.sh
  2. Start consumer:   NUM_PRODUCERS=6 bash serve_ec_consumer.sh

Usage:
  # Quick test (3 requests, all at once):
  python client_ec_disaggregated.py --num-requests 3

  # Dynamic load test (50 requests, 5 req/s, 6 producers):
  python client_ec_disaggregated.py \
      --num-requests 50 \
      --request-rate 5.0 \
      --num-producers 6 \
      --producer-base-port 8000 \
      --consumer-port 9000

  # Burst test (all requests at once, max 16 concurrent):
  python client_ec_disaggregated.py \
      --num-requests 100 \
      --max-concurrency 16 \
      --num-producers 6

  # Baseline (no disaggregation, single server on port 8000):
  python client_ec_disaggregated.py \
      --baseline \
      --baseline-port 8000 \
      --num-requests 50

Flow per request:
  1. Pick a producer (round-robin)
  2. Send request to producer (triggers vision encoding)
  3. Wait for producer to finish
  4. Send same request to consumer (pulls cache via NIXL, decodes)
  5. Collect result from consumer
"""

import asyncio
import base64
import time
from io import BytesIO

import fire
import httpx
import numpy as np
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
    timeout: float = 6000.0,
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
    timeout: float = 6000.0,
) -> dict:
    """Send the same request to producer then consumer."""
    t_start = time.perf_counter()

    # Send to producer — triggers vision encoding + NIXL registration.
    t_producer_start = time.perf_counter()
    producer_resp = await _send_request(client, producer_url, body, timeout)
    t_producer_end = time.perf_counter()

    # Consumer pulls encoder cache via NIXL and runs decoder.
    t_consumer_start = time.perf_counter()
    resp = await _send_request(client, consumer_url, body, timeout)
    resp.raise_for_status()
    t_consumer_end = time.perf_counter()

    result = resp.json()
    usage = result.get("usage", {})
    return {
        "request_id": request_id,
        "text": result["choices"][0]["message"]["content"],
        "producer_time": t_producer_end - t_producer_start,
        "consumer_time": t_consumer_end - t_consumer_start,
        "total_time": t_consumer_end - t_start,
        "completion_tokens": usage.get("completion_tokens", 0),
    }


async def process_request_baseline(
    client: httpx.AsyncClient,
    server_url: str,
    body: dict,
    request_id: int,
    timeout: float = 6000.0,
) -> dict:
    """Baseline: send request to a single server (no disaggregation)."""
    t_start = time.perf_counter()
    resp = await _send_request(client, server_url, body, timeout)
    resp.raise_for_status()
    t_end = time.perf_counter()

    result = resp.json()
    usage = result.get("usage", {})
    return {
        "request_id": request_id,
        "text": result["choices"][0]["message"]["content"],
        "producer_time": 0.0,
        "consumer_time": t_end - t_start,
        "total_time": t_end - t_start,
        "completion_tokens": usage.get("completion_tokens", 0),
    }


async def _rate_limited_worker(
    semaphore: asyncio.Semaphore,
    client: httpx.AsyncClient,
    producer_url: str,
    consumer_url: str,
    body: dict,
    request_id: int,
    timeout: float = 6000.0,
) -> dict:
    """Wrap process_request with a concurrency-limiting semaphore."""
    async with semaphore:
        return await process_request(
            client, producer_url, consumer_url, body, request_id, timeout
        )


async def _rate_limited_worker_baseline(
    semaphore: asyncio.Semaphore,
    client: httpx.AsyncClient,
    server_url: str,
    body: dict,
    request_id: int,
    timeout: float = 6000.0,
) -> dict:
    """Wrap process_request_baseline with a concurrency-limiting semaphore."""
    async with semaphore:
        return await process_request_baseline(
            client, server_url, body, request_id, timeout
        )


def _print_summary(results: list[dict], is_baseline: bool = False) -> None:
    """Print per-request results and aggregate metrics."""
    producer_times = []
    consumer_times = []
    total_times = []
    total_tokens = 0

    mode = "BASELINE" if is_baseline else "DISAGGREGATED"
    print(f"\n{'='*60}")
    print(f"  Mode: {mode}")
    print(f"{'='*60}")

    # Per-request metrics
    for r in results:
        tokens = r["completion_tokens"]
        total_tokens += tokens
        producer_times.append(r["producer_time"])
        consumer_times.append(r["consumer_time"])
        total_times.append(r["total_time"])
        tok_s = tokens / r["consumer_time"] if r["consumer_time"] > 0 else 0
        if is_baseline:
            print(f"[req {r['request_id']:3d}] "
                  f"time={r['total_time']:.2f}s  "
                  f"{tokens} tokens  {tok_s:.1f} tok/s")
        else:
            print(f"[req {r['request_id']:3d}] "
                  f"producer={r['producer_time']:.2f}s  "
                  f"consumer={r['consumer_time']:.2f}s  "
                  f"total={r['total_time']:.2f}s  "
                  f"{tokens} tokens  {tok_s:.1f} tok/s")

    # Per-request response text
    print(f"\n{'='*60}")
    print("Response texts")
    print(f"{'='*60}")
    for r in sorted(results, key=lambda x: x["request_id"]):
        print(f"\n--- [req {r['request_id']}] ---")
        print(r["text"])

    # Aggregate metrics
    wall_time = max(total_times) if total_times else 0
    n = len(results)
    print(f"\n{'='*60}")
    print(f"  {mode} Summary")
    print(f"{'='*60}")
    print(f"Completed: {n} requests")
    print(f"Total tokens: {total_tokens}")
    print(f"Wall time: {wall_time:.2f}s")
    if wall_time > 0:
        print(f"Throughput: {n / wall_time:.2f} req/s, "
              f"{total_tokens / wall_time:.1f} tok/s")
    if not is_baseline:
        print(f"Avg producer: {np.mean(producer_times):.2f}s")
        print(f"Avg consumer: {np.mean(consumer_times):.2f}s")
    print(f"Avg total:    {np.mean(total_times):.2f}s")
    print(f"P50 total:    {np.percentile(total_times, 50):.2f}s")
    print(f"P99 total:    {np.percentile(total_times, 99):.2f}s")
    print(f"{'='*60}")


async def main(
    num_requests: int = 3,
    num_producers: int = 1,
    producer_base_port: int = 8000,
    consumer_port: int = 8001,
    host: str = "127.0.0.1",
    max_tokens: int = 200,
    request_rate: float = 0.0,
    max_concurrency: int = 0,
    timeout: float = 6000.0,
    baseline: bool = False,
    baseline_port: int = 8000,
):
    """Run disaggregated EC benchmark (or baseline comparison).

    Args:
        num_requests: Total number of requests to send.
        num_producers: Number of producer instances (ports base..base+N-1).
        producer_base_port: First producer's HTTP port.
        consumer_port: Consumer's HTTP port.
        host: Server host address.
        max_tokens: Max tokens per response.
        request_rate: Requests per second (0 = send all at once).
        max_concurrency: Max concurrent requests (0 = unlimited).
        timeout: HTTP request timeout in seconds.
        baseline: If True, send to a single server (no disaggregation).
        baseline_port: Server port for baseline mode.
    """
    # Load sample images
    seed = 2442233655
    # seed = random.randint(0, 2**32 - 1)
    print(f"Using random seed: {seed}")
    dataset = load_dataset(
        "lmms-lab/llava-bench-in-the-wild", split="train"
    ).shuffle(seed=seed)

    # Encode images to base64
    requests_data = []
    for i in range(num_requests):
        idx = i % len(dataset)
        buf = BytesIO()
        dataset[idx]["image"].save(buf, format="PNG")
        image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        body = _make_chat_request(
            image_b64, dataset[idx]["question"], max_tokens
        )
        requests_data.append(body)

    sem = asyncio.Semaphore(
        max_concurrency if max_concurrency > 0 else num_requests
    )

    if baseline:
        server_url = f"http://{host}:{baseline_port}/v1/chat/completions"
        print("Mode: BASELINE (no disaggregation)")
        print(f"Config: {num_requests} requests, "
              f"rate={'unlimited' if request_rate <= 0 else f'{request_rate} req/s'}, "
              f"max_concurrency={max_concurrency}")
        print(f"Server: {server_url}\n")

        async with httpx.AsyncClient() as client:
            tasks = []
            for i, body in enumerate(requests_data):
                tasks.append(
                    _rate_limited_worker_baseline(
                        sem, client, server_url, body, i, timeout
                    )
                )
                if request_rate > 0 and i < len(requests_data) - 1:
                    await asyncio.sleep(1.0 / request_rate)

            results = await asyncio.gather(*tasks)

        _print_summary(results, is_baseline=True)

    else:
        producer_urls = [
            f"http://{host}:{producer_base_port + i}/v1/chat/completions"
            for i in range(num_producers)
        ]
        consumer_url = f"http://{host}:{consumer_port}/v1/chat/completions"

        print(f"Mode: DISAGGREGATED ({num_producers} producers)")
        print(f"Config: {num_requests} requests, "
              f"rate={'unlimited' if request_rate <= 0 else f'{request_rate} req/s'}, "
              f"max_concurrency={max_concurrency}")
        print(f"Producers: {producer_urls}")
        print(f"Consumer:  {consumer_url}\n")

        async with httpx.AsyncClient() as client:
            tasks = []
            for i, body in enumerate(requests_data):
                p_url = producer_urls[i % num_producers]
                tasks.append(
                    _rate_limited_worker(
                        sem, client, p_url, consumer_url, body, i, timeout
                    )
                )
                if request_rate > 0 and i < len(requests_data) - 1:
                    await asyncio.sleep(1.0 / request_rate)

            results = await asyncio.gather(*tasks)

        _print_summary(results, is_baseline=False)


if __name__ == "__main__":
    fire.Fire(main)
