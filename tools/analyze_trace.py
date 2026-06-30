#!/usr/bin/env python3
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

"""Print a TTFT + decode breakdown report for an existing merged Perfetto
trace JSON. Equivalent to the analysis that
`vllm-rbln bench serve --trace` runs automatically at the end of a bench.

Usage:
    python tools/analyze_trace.py path/to/trace_<ts>_merged.json
"""

import argparse
import sys

from vllm_rbln.v1.tracing.analyze import analyze_merged_trace


def main() -> int:
    p = argparse.ArgumentParser(
        description="Analyze a merged Perfetto trace and print TTFT + decode breakdown."
    )
    p.add_argument("trace_json", help="Path to a merged Perfetto trace JSON file.")
    args = p.parse_args()
    result = analyze_merged_trace(args.trace_json)
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())
