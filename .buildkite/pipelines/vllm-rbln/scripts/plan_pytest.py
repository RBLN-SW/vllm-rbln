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

"""Generate pytest jobs from the NPU requirements of the collected E2E tests."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from collections import defaultdict
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[4]
GROUP_PATHS = {
    "compile": ROOT / "tests/vllm/compile",
    "specdec": ROOT / "tests/vllm/v1/spec_decode",
}


class E2EPlan:
    def __init__(self, chips: dict[str, str]) -> None:
        self.chips = chips
        self.nodeids: dict[str, dict[int, list[str]]] = {
            group: defaultdict(list) for group in GROUP_PATHS
        }

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        from tests.vllm.conftest import _item_spec
        from tests.vllm.utils import devices_needed

        for item in session.items:
            group = next(
                group
                for group, path in GROUP_PATHS.items()
                if item.path.is_relative_to(path)
            )
            spec = _item_spec(item)
            if spec is not None:
                if self.chips[group] not in spec.chips:
                    continue
                count = devices_needed(spec.engine_kwargs)
            else:
                mark = item.get_closest_marker("model_compile")
                assert mark is not None
                count = mark.kwargs.get("npu_count")

            if type(count) is not int or count < 1:
                raise pytest.UsageError(f"{item.nodeid} needs a positive NPU count")
            self.nodeids[group][count].append(item.nodeid)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    template_path = Path(__file__).resolve().parent.parent / "pipeline-pytest.yml"
    pipeline = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    templates = pipeline["_planned_e2e"]
    plan = E2EPlan(
        {
            group: templates[group]["resources"]["npu"]["product"]
            for group in GROUP_PATHS
        }
    )

    sys.path.insert(0, str(ROOT))
    e2e_args = shlex.split(os.environ.get("PYTEST_E2E_ARGS", ""))
    status = pytest.main(
        [
            "--collect-only",
            "-q",
            "-m",
            "model_compile",
            *e2e_args,
            *map(str, GROUP_PATHS.values()),
        ],
        plugins=[plan],
    )
    if status != pytest.ExitCode.OK:
        return int(status)

    e2e = next(
        step for step in pipeline["steps"] if step["key"] == "vllm-rbln-pytest-e2e-cr13"
    )
    runner = Path(__file__).with_name("run-e2e.sh").resolve().relative_to(ROOT)
    for group, buckets in plan.nodeids.items():
        if not buckets:
            parser.error(f"No {group} tests collected for {plan.chips[group]}")
        template = templates[group]
        for count, nodeids in sorted(buckets.items()):
            command = shlex.join(
                ["bash", str(runner), "--group", group, "--", *nodeids]
            )
            e2e["steps"].append(
                template
                | {
                    "key": f"{template['key']}-{count}npu",
                    "label": f"{group}-{count}npu",
                    "resources": {
                        "npu": {"count": count, "product": plan.chips[group]}
                    },
                    "command": [
                        *template["command"],
                        command + " ${PYTEST_E2E_ARGS:-}",
                    ],
                }
            )

    rendered = {
        key: value for key, value in pipeline.items() if not key.startswith("_")
    }
    args.output.write_text(json.dumps(rendered, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
