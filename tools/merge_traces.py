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

"""
Merge multiple Chrome Trace JSON files into a single file.

Usage:
    python merge_traces.py trace_*.json -o merged.json
    python merge_traces.py trace_1.json trace_2.json vllm_trace.json -o merged.json
"""

import argparse
import json
import sys
from pathlib import Path


def merge_traces(input_files: list[str], output_file: str) -> int:
    all_events: list[dict] = []
    for fpath in input_files:
        p = Path(fpath)
        if not p.exists():
            print(f"WARNING: {fpath} not found, skipping", file=sys.stderr)
            continue
        with open(p) as f:
            data = json.load(f)
        events = data.get("traceEvents", [])
        all_events.extend(events)
        print(f"  {fpath}: {len(events)} events")

    if not all_events:
        print("No events found.", file=sys.stderr)
        return 1

    # Sort by timestamp for cleaner Perfetto display
    all_events.sort(key=lambda e: e.get("ts", 0))

    payload = {"traceEvents": all_events}
    with open(output_file, "w") as f:
        json.dump(payload, f, indent=0)
    print(f"Merged {len(all_events)} events -> {output_file}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Merge Chrome Trace JSON files")
    parser.add_argument("inputs", nargs="+", help="Input trace JSON files")
    parser.add_argument(
        "-o",
        "--output",
        default="merged_trace.json",
        help="Output file (default: merged_trace.json)",
    )
    args = parser.parse_args()
    sys.exit(merge_traces(args.inputs, args.output))


if __name__ == "__main__":
    main()
