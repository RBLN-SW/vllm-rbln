#!/usr/bin/env python3
"""
Single-request EC disaggregation latency test.

Sends the same image request N times sequentially to measure
consistent per-request latency. No concurrency, no randomness.

Usage:
  python test_ec_single_request.py --proxy-url http://127.0.0.1:1950 --num-requests 5
"""

import argparse
import base64
import io
import time

import requests
from PIL import Image


def _make_test_image_b64() -> str:
    """Generate a simple deterministic test image as base64 data URI."""
    img = Image.new("RGB", (224, 224), color=(100, 150, 200))
    # Draw a simple pattern for visual encoder to process
    for x in range(0, 224, 32):
        for y in range(0, 224, 32):
            img.putpixel((x, y), (255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def make_request(model: str) -> dict:
    """Build a deterministic chat completion request with a local image."""
    image_url = _make_test_image_b64()
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                    {
                        "type": "text",
                        "text": "Describe this image in one sentence.",
                    },
                ],
            }
        ],
        "max_tokens": 50,
        "stream": False,
        "temperature": 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy-url", default="http://127.0.0.1:1950")
    parser.add_argument("--model", default="Qwen2-VL-7B-Instruct")
    parser.add_argument("--num-requests", type=int, default=5)
    args = parser.parse_args()

    endpoint = f"{args.proxy_url}/v1/chat/completions"
    req_data = make_request(args.model)

    print(f"Sending {args.num_requests} sequential requests to {endpoint}")
    print(f"Model: {args.model}")
    print("=" * 60)

    latencies = []
    for i in range(args.num_requests):
        t0 = time.perf_counter()
        resp = requests.post(endpoint, json=req_data, timeout=120)
        elapsed = time.perf_counter() - t0

        if resp.status_code != 200:
            print(f"  [{i}] FAILED status={resp.status_code}: {resp.text[:200]}")
            continue

        data = resp.json()
        text = data["choices"][0]["message"]["content"][:80]
        tokens = data["usage"]["completion_tokens"]
        latencies.append(elapsed)

        print(f"  [{i}] {elapsed:.3f}s | {tokens} tokens | {text}")

    if latencies:
        print("=" * 60)
        print(f"Requests:  {len(latencies)} / {args.num_requests}")
        print(f"Mean:      {sum(latencies)/len(latencies):.3f}s")
        print(f"Min:       {min(latencies):.3f}s")
        print(f"Max:       {max(latencies):.3f}s")
        if len(latencies) > 2:
            # Drop first (cold) and compute mean of rest
            warm = latencies[1:]
            print(f"Mean (warm): {sum(warm)/len(warm):.3f}s")


if __name__ == "__main__":
    main()
