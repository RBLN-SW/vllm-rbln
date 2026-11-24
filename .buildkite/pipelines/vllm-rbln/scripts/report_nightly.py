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

"""Collect nightly artifacts and publish a Slack report, or preview it locally."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

import yaml
from lane import (
    agent_bin,
    case_status,
    load_targets,
    perf_cases,
    restore_ci_targets,
)

ROOT = Path(__file__).resolve().parents[4]


def command_steps(steps):
    for step in steps:
        if "steps" in step:
            yield from command_steps(step["steps"])
        elif "key" in step:
            yield step


def counts_text(counts):
    return " / ".join(
        f"{counts.get(label, 0):,} {label}"
        for label in ("passed", "failed", "errors", "skipped", "xfailed", "xpassed")
        if counts.get(label) or label in ("passed", "failed")
    )


def build_report(results, *, run_id, chip, build_url, branch, commit):
    plan_path = results / "pytest-plan.json"
    nightly = yaml.safe_load((ROOT / ".buildkite/vllm-rbln-nightly.yml").read_text())
    degraded = not plan_path.is_file()
    tests = []
    if plan_path.is_file():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        test_steps = list(command_steps(plan["steps"]))
        for label, prefix in (
            ("unit", "vllm-rbln-pytest-"),
            ("e2e", "vllm-rbln-e2e-"),
        ):
            counts = Counter()
            missing = incomplete = unsuccessful = 0
            for step in (step for step in test_steps if step["key"].startswith(prefix)):
                key = step["key"]
                # Downloads omit retried jobs; job directories contain current shards.
                paths = list((results / "pytest-results" / key).glob("*/result.json"))
                missing += max(int(step.get("parallelism", 1)) - len(paths), 0)
                for path in paths:
                    result = json.loads(path.read_text())
                    if result["exit_code"] is None:
                        incomplete += 1
                        continue
                    counts.update(result["counts"])
                    unsuccessful += result["exit_code"] != 0
            text = counts_text(counts)
            for count, status in (
                (missing, "missing"),
                (incomplete, "incomplete"),
                (unsuccessful, "unsuccessful"),
            ):
                if count:
                    text += f" / {count} jobs {status}"
            degraded |= bool(missing or incomplete or unsuccessful)
            tests.append(f"- {label}: {text}")
    else:
        tests.append("- pytest: missing job plan")

    counts = Counter()
    missing = 0
    for step in command_steps(nightly["steps"]):
        if not step["key"].startswith("nightly-examples-vllm-"):
            continue
        group = step["key"].removeprefix("nightly-examples-vllm-")
        path = results / "example-results" / group / "results.json"
        if not path.exists():
            missing += 1
            continue
        for case in json.loads(path.read_text()):
            assert case["status"] in ("PASS", "FAIL")
            counts["passed" if case["status"] == "PASS" else "failed"] += 1
    examples = "- " + counts_text(counts)
    if missing:
        examples += f" / {missing} groups missing"
    degraded |= bool(counts["failed"] or missing)

    titles = [
        "No",
        "Model",
        "Case",
        "out tok/s",
        "total tok/s",
        "TTFT mean (s)",
        "TPOT mean (ms)",
    ]
    rows = [[{"type": "raw_text", "text": title} for title in titles]]
    markdown = ["| " + " | ".join(titles) + " |", "|" + "---|" * len(titles)]
    completed = planned = 0
    snapshot = results / "ci-targets" / run_id
    targets = {}
    source = None
    if (snapshot / "snapshot.tar.gz").exists():
        source = restore_ci_targets(snapshot)
        targets = load_targets(snapshot / "perf", chip, "perf")
    for name, target in targets.items():
        for launch, bench in sorted(perf_cases(target)):
            planned += 1
            directory = results / "perf-results" / run_id / chip / name / launch / bench
            result_file = directory / "result.json"
            result = (
                json.loads(result_file.read_text()) if result_file.exists() else None
            )
            status = case_status(result)
            measurements = result["rows"] if result else []
            case_ok = status == "ok"
            metrics = (
                ("output_throughput", 1),
                ("total_token_throughput", 1),
                ("mean_ttft_ms", 1000),
                ("mean_tpot_ms", 1),
            )
            for row in measurements:
                case_ok &= (
                    int(row["completed"]) == int(row["num_prompts"])
                    and int(row["failed"]) == 0
                    and all(
                        row.get(key) is not None and math.isfinite(float(row[key]))
                        for key, _ in metrics
                    )
                )
            completed += case_ok
            degraded |= not case_ok
            url = (result or {}).get("job_url") or build_url
            parts = urlsplit(url)
            if parts.fragment:
                query = dict(parse_qsl(parts.query)) | {"job": parts.fragment}
                url = urlunsplit(parts._replace(query=urlencode(query), fragment=""))
            number = str(planned)
            link = {
                "type": "rich_text",
                "elements": [
                    {
                        "type": "rich_text_section",
                        "elements": [
                            {"type": "link", "text": number, "url": url},
                        ],
                    },
                ],
            }
            if result is None:
                failure = "MISSING"
            elif result["exit_code"] is None:
                failure = "INCOMPLETE"
            else:
                failure = "FAILED"
            for row in measurements or [{}]:
                case = f"{launch} / {bench}"
                if not case_ok:
                    case = f"❌ {case} ({failure})"
                cells = [target["model"].rsplit("/", 1)[-1], case]
                for key, scale in metrics:
                    value = row.get(key)
                    cells.append(
                        f"{float(value) / scale:.1f}" if value is not None else "—"
                    )
                rows.append(
                    [link, *({"type": "raw_text", "text": cell} for cell in cells)]
                )
                markdown.append(
                    "| " + " | ".join([f"[{number}]({url})", *cells]) + " |"
                )
    if planned:
        perf = (
            f"{completed}/{planned} cases completed · {chip}\n"
            f"Targets: {source['repo']} @ {source['commit']}"
        )
    else:
        perf = "CI target preparation failed: target snapshot is missing"
        degraded = True

    sys.path.insert(0, str(Path(__file__).with_name("sampler-perf")))
    from run_benchmarks import sampler_summary

    perf_text = "*Perf*\n" + perf
    details = {
        "perf": {
            "text": perf_text,
            "blocks": [
                {"type": "section", "text": {"type": "mrkdwn", "text": perf_text}}
            ],
        }
    }
    if planned:
        details["perf"]["blocks"].append(
            {
                "type": "table",
                "column_settings": [
                    {"align": "right"},
                    {"is_wrapped": True},
                    {"is_wrapped": True},
                    *({"align": "right"} for _ in range(4)),
                ],
                "rows": rows,
            }
        )
    sampler_path = results / "sampler-perf-results" / run_id / chip / "manifest.json"
    sampler_summaries = []
    sampler_annotations = []
    for enabled, name in ((False, "Off"), (True, "On")):
        sampler = None
        sampler_url = build_url
        sampler_status = "MISSING: Sampler Perf manifest was not uploaded."
        sampler_text = sampler_status
        if sampler_path.is_file():
            try:
                manifest = json.loads(sampler_path.read_text())
                if manifest["chip"] not in (None, chip):
                    raise ValueError(
                        "sampler result chip does not match nightly report"
                    )
                if source is None:
                    raise ValueError("CI targets snapshot is missing")
                matrix = sampler_summary(manifest, use_fused_sampling=enabled)
                if not matrix["rows"]:
                    raise ValueError(f"no Fused {name} cases in manifest")
            except (ValueError, KeyError, TypeError, OSError) as error:
                sampler_status = f"INVALID: Sampler Perf results ({error})."
                sampler_text = sampler_status
            else:
                sampler = matrix
                sampler_url = manifest.get("job_url") or build_url
                sampler_status = sampler["status"]
                sampler_text = (
                    sampler["text"]
                    + f"\nTargets: {source['repo']} @ {source['commit']}"
                )
        degraded |= sampler is None or sampler["degraded"]
        sampler_summaries.append(f"- Fused {name}: {sampler_status}")
        section = (
            f"*Sampler Perf (Fused {name})*\n{sampler_text}"
            f"\n<{sampler_url}|Job and details>"
        )
        payload = {
            "text": section,
            "blocks": [
                {"type": "section", "text": {"type": "mrkdwn", "text": section}}
            ],
        }
        if sampler is not None:
            payload["blocks"].append(
                {
                    "type": "table",
                    "column_settings": [
                        {"is_wrapped": True},
                        {"is_wrapped": True},
                        {"align": "right"},
                        {"is_wrapped": True},
                        *({"align": "right"} for _ in sampler["titles"][4:]),
                    ],
                    "rows": [
                        [{"type": "raw_text", "text": cell} for cell in row]
                        for row in [sampler["titles"], *sampler["rows"]]
                    ],
                }
            )
        details[f"sampler-{name.lower()}"] = payload
        sampler_annotations.append(
            f"**Sampler Perf (Fused {name})**\n{sampler_text}"
            f"\n\n[Job and details]({sampler_url})"
            + ("\n\n" + sampler["markdown"] if sampler is not None else "")
        )

    title = f"vllm-rbln {branch} nightly build"
    build_number = urlsplit(build_url).path.rstrip("/").rsplit("/", 1)[-1]
    sections = [
        "*Tests*\n" + "\n".join(tests),
        "*Examples*\n" + examples,
        "*Perf*\n" + (f"{completed}/{planned} cases completed" if planned else perf),
        "*Sampler Perf*\n" + "\n".join(sampler_summaries),
    ]
    text = (
        f"{'⚠️' if degraded else '✅'} *{title} <{build_url}|#{build_number}> "
        f"({commit[:8]})*\n\n" + "\n\n".join(sections)
    )
    payloads = {
        "summary": {
            "text": text,
            "blocks": [{"type": "section", "text": {"type": "mrkdwn", "text": text}}],
        },
        **details,
    }
    for payload in payloads.values():
        payload.update(unfurl_links=False, unfurl_media=False)
    annotation = (
        f"**{title} [#{build_number}]({build_url}) ({commit[:8]})**\n\n"
        + "\n\n".join(
            section.replace("*Tests*", "**Tests**")
            .replace("*Examples*", "**Examples**")
            .replace("*Perf*", "**Perf**")
            for section in (sections[0], sections[1], perf_text, sections[3])
        )
        + "\n\n"
        + "\n".join(markdown)
        + "\n\n"
        + "\n\n".join(sampler_annotations)
    )
    return (
        payloads,
        annotation,
        degraded,
    )


def slack_api(method, payload):
    request = Request(
        f"https://slack.com/api/{method}",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {os.environ['SLACK_BOT_TOKEN']}",
            "Content-Type": "application/json; charset=utf-8",
        },
    )
    while True:
        try:
            with urlopen(request, timeout=30) as response:
                result = json.load(response)
            break
        except HTTPError as error:
            if error.code != 429:
                raise
            time.sleep(float(error.headers["Retry-After"]))
    if not result["ok"]:
        raise RuntimeError(f"Slack {method} failed: {result['error']}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("nightly-report"))
    parser.add_argument("--chip", required=True)
    for flag, variable in (
        ("run-id", "BUILDKITE_BUILD_ID"),
        ("build-url", "BUILDKITE_BUILD_URL"),
        ("branch", "BUILDKITE_BRANCH"),
        ("commit", "BUILDKITE_COMMIT"),
    ):
        parser.add_argument(
            f"--{flag}",
            default=os.environ.get(variable),
            required=not os.environ.get(variable),
        )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    agent = agent_bin()
    if not args.dry_run and agent is None:
        parser.error("buildkite-agent is required to publish a report")
    with tempfile.TemporaryDirectory() as temporary:
        results = args.results_dir
        if args.download:
            if agent is None:
                parser.error("buildkite-agent is required to download artifacts")
            results = Path(temporary)
            for pattern in (
                "pytest-plan.json",
                "pytest-results/**/result.json",
                "example-results/**/results.json",
                f"perf-results/{args.run_id}/{args.chip}/**/result.json",
                f"sampler-perf-results/{args.run_id}/{args.chip}/manifest.json",
                f"ci-targets/{args.run_id}/snapshot.tar.gz",
            ):
                scope = []
                if pattern == "pytest-plan.json":
                    scope = ["--step", "nightly-pytest"]
                if pattern.startswith("ci-targets/"):
                    scope = ["--step", "nightly-targets-prepare"]
                found = subprocess.run(
                    [
                        agent,
                        "artifact",
                        "search",
                        pattern,
                        "--allow-empty-results",
                        *scope,
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                if found.stdout.strip():
                    subprocess.run(
                        [agent, "artifact", "download", pattern, str(results), *scope],
                        check=True,
                    )
        payloads, annotation, degraded = build_report(
            results,
            run_id=args.run_id,
            chip=args.chip,
            build_url=args.build_url,
            branch=args.branch,
            commit=args.commit,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "slack.json").write_text(
        json.dumps(payloads, ensure_ascii=False, indent=2) + "\n"
    )
    (args.output_dir / "summary.md").write_text(annotation + "\n")
    print(annotation)
    if args.dry_run:
        return 0
    annotation_result = subprocess.run(
        [
            agent,
            "annotate",
            "--context",
            "nightly-report",
            "--style",
            "warning" if degraded else "info",
        ],
        input=annotation,
        text=True,
    )
    channel = os.environ["SLACK_REPORT_CHANNEL"]
    if not channel:
        raise ValueError("SLACK_REPORT_CHANNEL must name a Slack channel")
    parent_ts = None
    for name, payload in payloads.items():
        payload = payload | {"channel": channel}
        key = f"nightly-slack-ts-{channel}" + ("" if name == "summary" else f"-{name}")
        timestamp = subprocess.run(
            [agent, "meta-data", "get", key, "--default", ""],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if timestamp:
            posted = slack_api("chat.update", payload | {"ts": timestamp})
        else:
            if name != "summary":
                assert parent_ts is not None
                payload["thread_ts"] = parent_ts
            posted = slack_api("chat.postMessage", payload)
        subprocess.run([agent, "meta-data", "set", key, posted["ts"]], check=True)
        if name == "summary":
            parent_ts = posted["ts"]
        print(f"Slack {name} report published to {posted['channel']} at {posted['ts']}")
    return annotation_result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
