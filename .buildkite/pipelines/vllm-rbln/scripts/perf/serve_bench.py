#!/usr/bin/env python3
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

"""Run serving benchmarks with separate warmup inputs and optional DP routing.

Accepts `vllm bench serve` flags plus `--dp-size`. Warmup uses a different seed
and does not write result files. Only the measured run reaches sweep results.
"""

from __future__ import annotations

import argparse
import itertools
import sys

from vllm.benchmarks import serve
from vllm.benchmarks.lib import endpoint_request_func as erf
from vllm.utils.argparse_utils import FlexibleArgumentParser

_RANK_HEADER = "X-data-parallel-rank"


def patch_round_robin(dp_size: int) -> None:
    original = erf._update_headers_common
    ranks = itertools.cycle(range(dp_size))

    def round_robin(headers, request_func_input) -> None:
        original(headers, request_func_input)
        headers[_RANK_HEADER] = str(next(ranks))

    # The call sites are all in that module and resolve it as a global, so
    # replacing the attribute reaches every one of them.
    erf._update_headers_common = round_robin


def main() -> int:
    parser = FlexibleArgumentParser(description=__doc__)
    parser.add_argument("--dp-size", type=int, default=1)
    serve.add_cli_args(parser)
    args = parser.parse_args()
    if args.dp_size < 1 or args.num_warmups < 0:
        parser.error("dp-size must be positive and num-warmups must be non-negative")

    if args.dp_size > 1:
        patch_round_robin(args.dp_size)

    if args.num_warmups:
        warmup = argparse.Namespace(**vars(args))
        warmup.seed = args.seed ^ (1 << 30)
        warmup.num_prompts = args.num_warmups
        warmup.num_warmups = 0
        warmup.request_rate = float("inf")
        warmup.profile = False
        warmup.save_result = False
        warmup.append_result = False
        warmup.plot_timeline = False
        print(f"Running separate warmup: seed={warmup.seed}", flush=True)
        result = serve.main(warmup)
        if result["completed"] != warmup.num_prompts or result["failed"]:
            raise RuntimeError("Warmup did not complete all requests successfully")

        # Upstream's readiness request also replays the first measured input.
        args.ready_check_timeout_sec = 0

    args.num_warmups = 0
    print(f"Running measured benchmark: seed={args.seed}", flush=True)
    serve.main(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
