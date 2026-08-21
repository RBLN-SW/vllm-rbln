# SPDX-License-Identifier: Apache-2.0
"""Greedy probe against a live serve, for diffing two KV addressing schemes.

Wrong KV addressing is silent and *reads faster*: token counts are fixed by
max_tokens, so every benchmark metric improves while the model emits garbage.
The only cheap detector is greedy text, so this sends fixed prompts at
temperature 0 and dumps them for a byte diff between configs.

Shaped to hit the cases that differ:
  round 1  cold, one long shared prefix
  round 2  same prefix, new questions       -> whole-block attach + partial tail
  round 3  prefix extended by one turn      -> adoption / copy-out
"""

import json
import os
import sys
import urllib.request

URL = os.environ.get("PROBE_URL", "http://localhost:8102/v1/completions")
MODEL = os.environ.get("PROBE_MODEL", "MiniMax")

# ~9k characters so the shared prefix spans more than one 8192-token kernel
# block once tokenized alongside the question.
PREFIX = (
    "The following is a technical reference on distributed storage systems.\n\n"
    + "\n".join(
        f"Section {i}. A log-structured merge tree defers writes into an "
        f"in-memory table and flushes sorted runs to durable storage. Level {i} "
        f"holds runs whose key ranges may overlap, and compaction rewrites them "
        f"into level {i + 1} where ranges are disjoint. Read amplification grows "
        f"with the number of overlapping runs; write amplification grows with the "
        f"number of times a key is rewritten during compaction. Bloom filters cut "
        f"read amplification for absent keys at the cost of memory proportional "
        f"to the key count."
        for i in range(1, 60)
    )
)

QUESTIONS = [
    "What causes write amplification in an LSM tree?",
    "How do Bloom filters affect read amplification?",
    "What is the difference between level 1 and level 2 key ranges?",
    "Summarize compaction in two sentences.",
]
FOLLOWUP = (
    "\n\nAddendum. Tiered compaction merges runs of similar size rather than "
    "into a fixed level, trading read amplification for write amplification."
)


def ask(prompt: str, max_tokens: int = 64) -> str:
    body = json.dumps(
        {
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 0,
        }
    ).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=1800) as response:
        return json.load(response)["choices"][0]["text"]


def main() -> None:
    out_path = sys.argv[1]
    rounds: dict[str, list[str]] = {}

    rounds["cold"] = [ask(f"{PREFIX}\n\nQ: {QUESTIONS[0]}\nA:")]
    rounds["shared_prefix"] = [ask(f"{PREFIX}\n\nQ: {q}\nA:") for q in QUESTIONS[1:]]
    rounds["extended_prefix"] = [
        ask(f"{PREFIX}{FOLLOWUP}\n\nQ: {q}\nA:") for q in QUESTIONS[:2]
    ]

    with open(out_path, "w") as f:
        json.dump(rounds, f, indent=2)
    for name, texts in rounds.items():
        for i, text in enumerate(texts):
            print(f"[{name}/{i}] {text[:120]!r}")


if __name__ == "__main__":
    main()
