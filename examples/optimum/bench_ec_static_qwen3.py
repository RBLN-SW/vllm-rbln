#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Rebellions Inc. All rights reserved.
"""
Static EC-disaggregated performance probe for Qwen3-VL (batch=2).

Each round sends exactly 2 concurrent chat-completions requests to the
proxy, waits for both to finish, then sleeps gap_sec before the next
round. Unique deterministic image per request so mm_hash never repeats
(forcing full encoder + prefill every round).

Designed to be run against services launched with
  BENCH_SKIP=1 bash examples/optimum/run_ec_push_benchmark.sh

Server-side timings live in the consumer/producer logs under
`[EC-PERF]` lines; this driver only captures per-request client-side
TTFT and end-to-end latency plus token counts.

Usage:
  python bench_ec_static_qwen3.py --proxy-url http://127.0.0.1:1900 \
    --model Qwen3-VL-8B-Instruct --rounds 10 --gap-sec 20
"""

import argparse
import asyncio
import base64
import io
import json
import math
import time

import aiohttp
from PIL import Image


def _make_image_b64(seed: int, size: int = 448) -> str:
    """Deterministic distinct image per seed."""
    # Mix seed into base color so each seed is visibly different — also
    # avoids identical mm_hash across rounds.
    r = (seed * 37) % 256
    g = (seed * 61) % 256
    b = (seed * 89) % 256
    img = Image.new("RGB", (size, size), color=(r, g, b))
    # Overlay a pattern so the encoder has real content to process.
    stride = 32
    for x in range(0, size, stride):
        for y in range(0, size, stride):
            off = ((x + y + seed) // stride) % 3
            px = [(255, 0, 0), (0, 255, 0), (0, 0, 255)][off]
            img.putpixel((x, y), px)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def build_request(model: str, seed: int, max_tokens: int) -> dict:
    image_url = _make_image_b64(seed)
    prompt = (
        f"(request #{seed}) Describe this image in one short sentence, "
        "mentioning the dominant color."
    )
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


async def send_streaming(
    session: aiohttp.ClientSession,
    endpoint: str,
    payload: dict,
    label: str,
) -> dict:
    """Send a streaming chat request and return timing + token counts."""
    start = time.perf_counter()
    ttft: float | None = None
    first_content_ts: float | None = None
    gen_tokens = 0
    prompt_tokens = 0
    finish_reason = None
    error = None
    try:
        async with session.post(
            endpoint,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=600),
        ) as resp:
            if resp.status != 200:
                error = f"status={resp.status} body={(await resp.text())[:200]}"
                return {
                    "label": label,
                    "error": error,
                    "e2e_s": time.perf_counter() - start,
                }
            async for raw in resp.content:
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line.startswith("data:"):
                    continue
                body = line[len("data:") :].strip()
                if body == "[DONE]":
                    break
                try:
                    obj = json.loads(body)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content")
                    if content:
                        if first_content_ts is None:
                            first_content_ts = time.perf_counter()
                            ttft = first_content_ts - start
                    fr = choices[0].get("finish_reason")
                    if fr:
                        finish_reason = fr
                usage = obj.get("usage")
                if usage:
                    gen_tokens = usage.get("completion_tokens", gen_tokens)
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
    except Exception as exc:  # noqa: BLE001
        error = repr(exc)
    end = time.perf_counter()
    return {
        "label": label,
        "error": error,
        "ttft_s": ttft,
        "e2e_s": end - start,
        "first_content_ts": first_content_ts,
        "start_ts": start,
        "end_ts": end,
        "gen_tokens": gen_tokens,
        "prompt_tokens": prompt_tokens,
        "finish_reason": finish_reason,
    }


async def run_round(
    session: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    round_idx: int,
    max_tokens: int,
) -> list[dict]:
    seed_base = round_idx * 2
    payloads = [
        build_request(model, seed=seed_base + 0, max_tokens=max_tokens),
        build_request(model, seed=seed_base + 1, max_tokens=max_tokens),
    ]
    labels = [f"r{round_idx:02d}-a", f"r{round_idx:02d}-b"]
    tasks = [
        asyncio.create_task(send_streaming(session, endpoint, p, lbl))
        for p, lbl in zip(payloads, labels)
    ]
    return await asyncio.gather(*tasks)


def _fmt_ms(x: float | None) -> str:
    if x is None:
        return "     n/a"
    return f"{x * 1000:8.1f}"


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy-url", default="http://127.0.0.1:1900")
    parser.add_argument("--model", default="Qwen3-VL-8B-Instruct")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--gap-sec", type=float, default=20.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write per-request results as JSON.",
    )
    args = parser.parse_args()

    endpoint = f"{args.proxy_url}/v1/chat/completions"
    print("=" * 78)
    print(f"Static EC perf probe — model={args.model}")
    print(
        f"Rounds={args.rounds}, batch=2/round, gap={args.gap_sec}s, "
        f"max_tokens={args.max_tokens}"
    )
    print(f"Proxy={endpoint}")
    print("=" * 78)
    print(
        f"{'label':>10} | {'ttft_ms':>8} | {'e2e_ms':>8} | "
        f"{'prompt_tok':>10} | {'gen_tok':>7} | "
        f"{'decode_tpot_ms':>15} | finish"
    )
    print("-" * 78)

    all_results: list[dict] = []
    wall0 = time.perf_counter()
    async with aiohttp.ClientSession() as session:
        for r in range(args.rounds):
            results = await run_round(
                session, endpoint, args.model, r, args.max_tokens
            )
            for res in results:
                if res.get("error"):
                    print(
                        f"{res['label']:>10} | "
                        f"{'ERR':>8} | {res['e2e_s']*1000:8.1f} | "
                        f"{'n/a':>10} | {'n/a':>7} | "
                        f"{'n/a':>15} | {res['error'][:40]}"
                    )
                else:
                    tpot_ms = None
                    if (
                        res["ttft_s"] is not None
                        and res["gen_tokens"] > 1
                    ):
                        # Client-side per-token latency excluding first.
                        tpot_ms = (
                            (res["e2e_s"] - res["ttft_s"])
                            / (res["gen_tokens"] - 1)
                            * 1000.0
                        )
                    print(
                        f"{res['label']:>10} | "
                        f"{_fmt_ms(res['ttft_s'])} | "
                        f"{_fmt_ms(res['e2e_s'])} | "
                        f"{res['prompt_tokens']:>10} | "
                        f"{res['gen_tokens']:>7} | "
                        f"{(f'{tpot_ms:.2f}' if tpot_ms is not None else 'n/a'):>15} | "
                        f"{res['finish_reason']}"
                    )
                all_results.append(res)
            if r < args.rounds - 1:
                print(f"  — sleeping {args.gap_sec}s before next round —")
                await asyncio.sleep(args.gap_sec)

    wall = time.perf_counter() - wall0

    # Aggregate — drop the first round as warmup.
    warm = [
        r for r in all_results
        if not r.get("error") and not r["label"].startswith("r00")
    ]
    print("=" * 78)
    print(f"Total wall: {wall:.1f}s  ({len(all_results)} requests)")
    if warm:
        ttfts = [r["ttft_s"] * 1000 for r in warm if r["ttft_s"] is not None]
        e2es = [r["e2e_s"] * 1000 for r in warm]
        tpots = []
        for r in warm:
            if r["ttft_s"] is not None and r["gen_tokens"] > 1:
                tpots.append(
                    (r["e2e_s"] - r["ttft_s"])
                    / (r["gen_tokens"] - 1)
                    * 1000
                )

        def stats(name: str, xs: list[float], unit: str = "ms") -> None:
            if not xs:
                print(f"  {name}: (no samples)")
                return
            xs_sorted = sorted(xs)
            mean = sum(xs) / len(xs)
            p50 = xs_sorted[len(xs) // 2]
            p99 = xs_sorted[int(math.ceil(len(xs) * 0.99)) - 1]
            print(
                f"  {name:<20} mean={mean:8.2f}{unit}  "
                f"p50={p50:8.2f}{unit}  p99={p99:8.2f}{unit}  "
                f"min={min(xs):8.2f}{unit}  max={max(xs):8.2f}{unit}  "
                f"n={len(xs)}"
            )

        print("Warm (rounds 1..N-1):")
        stats("TTFT", ttfts)
        stats("E2E", e2es)
        stats("TPOT (client)", tpots)
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"Wrote per-request JSON → {args.output_json}")


if __name__ == "__main__":
    asyncio.run(main())
