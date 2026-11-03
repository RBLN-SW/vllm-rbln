# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Byte-diff two greedy probes. Any difference in the cached rounds is a bug."""

import json
import os
import pathlib
import sys

SP = pathlib.Path(os.environ.get("OUT_DIR", "results"))


def main() -> None:
    a_name, b_name = sys.argv[1], sys.argv[2]
    with open(SP / f"probe_{a_name}.json") as handle:
        a = json.load(handle)
    with open(SP / f"probe_{b_name}.json") as handle:
        b = json.load(handle)

    bad = 0
    for round_name in a:
        for i, (ta, tb) in enumerate(zip(a[round_name], b[round_name])):
            same = ta == tb
            bad += not same
            print(f"{'OK  ' if same else 'DIFF'} {round_name}/{i}")
            if not same:
                print(f"  {a_name}: {ta[:200]!r}")
                print(f"  {b_name}: {tb[:200]!r}")
    print(f"\n{bad} differing completions")


if __name__ == "__main__":
    main()
