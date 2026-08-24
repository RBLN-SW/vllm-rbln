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

"""`vllm bench serve` with the data parallel rank cycled per request.

    dp_round_robin_bench.py --dp-size N vllm bench serve ...

The server takes a rank per request via a header, but the bench client sets one
static set of headers for a whole run -- so a single client cannot spread its
requests across ranks, and the server's own balancer does not divide them evenly
enough to measure a DP configuration. Every backend assembles its headers through
one function, so cycling the rank there covers all of them.
"""

from __future__ import annotations

import itertools
import sys

from vllm.benchmarks.lib import endpoint_request_func as erf

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
    argv = sys.argv[1:]
    if len(argv) < 3 or argv[0] != "--dp-size":
        print(f"usage: {sys.argv[0]} --dp-size N vllm bench serve ...", file=sys.stderr)
        return 2

    patch_round_robin(int(argv[1]))

    from vllm.entrypoints.cli.main import main as vllm_main

    sys.argv = argv[2:]
    vllm_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
