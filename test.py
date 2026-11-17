#!/usr/bin/env python3
# pip install requests
import argparse
import time

import requests

MODEL = "MiniMaxAI/MiniMax-M2.7"
PROMPT = "Hi are you conscious?"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8800/v1/chat/completions",
                        help="proxy (disagg.py) chat endpoint")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--dp-rank", type=int, default=0,
                        help="decode 서버 DP rank (X-data-parallel-rank)")
    args = parser.parse_args()

    # /v1/chat/completions 로 보내면 서버가 모델의 chat template 을 적용함
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "stream": False,
    }
    headers = {"X-data-parallel-rank": str(args.dp_rank)}

    t0 = time.perf_counter()
    resp = requests.post(args.url, json=payload, headers=headers, timeout=None)
    if not resp.ok:
        print(f"HTTP {resp.status_code}: {resp.text[:1000]}")
        resp.raise_for_status()
    data = resp.json()
    elapsed = time.perf_counter() - t0

    choice = data["choices"][0]
    message = choice["message"]
    usage = data.get("usage", {})

    print(f"[prompt] {PROMPT}")
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if reasoning:
        print(f"[reasoning]\n{reasoning}")
    print(f"[output]\n{message.get('content')}")
    print(f"[finish_reason={choice.get('finish_reason')} "
          f"in={usage.get('prompt_tokens')} out={usage.get('completion_tokens')} "
          f"e2e={elapsed:.1f}s]")


if __name__ == "__main__":
    main()
