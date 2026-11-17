#!/usr/bin/env python3
# pip install requests
import argparse
import random
import threading
import time

import requests

MODEL = "MiniMaxAI/MiniMax-M2.7"
VOCAB_SIZE = 200064


def parse_lanes(text):
    """'single_lane:0,multi_lane' -> [{'xfer_mode': 'single_lane', 'xfer_rail': 0},
    {'xfer_mode': 'multi_lane'}]"""
    lanes = []
    for item in text.split(","):
        mode, _, rail = item.strip().partition(":")
        if mode not in ("single_lane", "multi_lane"):
            raise argparse.ArgumentTypeError(f"unknown xfer_mode {mode!r}")
        lane = {"xfer_mode": mode}
        if rail:
            lane["xfer_rail"] = int(rail)
        lanes.append(lane)
    return lanes


def worker(idx, url, token_ids, args, barrier):
    dp_rank = idx % args.dp_size
    payload = {
        "model": MODEL,
        "prompt": token_ids,
        "max_tokens": args.output_len,
        "temperature": 0,
        "ignore_eos": True,
        "stream": False,
    }
    lane = {k: v for k, v in (("xfer_mode", args.xfer_mode),
                              ("xfer_rail", args.xfer_rail)) if v is not None}
    if args.overlap_lanes:
        lane["overlap_lanes"] = args.overlap_lanes
    if lane:
        payload["kv_transfer_params"] = lane
    headers = {"X-data-parallel-rank": str(dp_rank)}
    barrier.wait()  # 전원 준비 후 동시에 발사
    t0 = time.perf_counter()
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=None)
        resp.raise_for_status()
        usage = resp.json().get("usage", {})
        print(f"[{idx}] dp={dp_rank} in={usage.get('prompt_tokens')} "
              f"out={usage.get('completion_tokens')} "
              f"e2e={time.perf_counter() - t0:.1f}s")
    except Exception as e:
        body = getattr(getattr(e, "response", None), "text", "")
        print(f"[{idx}] dp={dp_rank} error: {e} {body[:300]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8800/v1/completions",
                        help="proxy (disagg.py) endpoint")
    parser.add_argument("--input-len", type=int, default=50 * 1024)
    parser.add_argument("--output-len", type=int, default=16)
    parser.add_argument("--num-requests", type=int, default=1)
    parser.add_argument("--dp-size", type=int, default=4,
                        help="decode 서버 --data-parallel-size")
    parser.add_argument("--xfer-mode", choices=["single_lane", "multi_lane"],
                        default=None,
                        help="KV pull transfer mode (생략 시 서버 설정)")
    parser.add_argument("--xfer-rail", type=int, default=None,
                        help="single_lane 에서 쓸 rail (0..rail_count-1, -1=unpin)")
    parser.add_argument("--overlap-lanes", type=parse_lanes, default=None,
                        help="proxy 가 prefill 후 lane 마다 decode 를 동시에 보냄. "
                             "예: single_lane:0,single_lane:1 / multi_lane,single_lane:1")
    args = parser.parse_args()

    barrier = threading.Barrier(args.num_requests)
    prompts = [
        [random.randint(100, VOCAB_SIZE - 1) for _ in range(args.input_len)]
        for _ in range(args.num_requests)
    ]
    threads = [threading.Thread(target=worker,
                                args=(i, args.url, p, args, barrier))
               for i, p in enumerate(prompts)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
