# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import fire
from vllm import LLM

# Minimum gap by which a relevant pair must out-score an irrelevant one.
# Relational, not absolute: robust against precision / model-version drift
# in a way that hard-coded golden similarities are not.
RELEVANCE_MARGIN = 0.05
# How close self-similarity score(x, x) must be to the theoretical maximum.
SELF_SIM_FLOOR = 0.99
# Tolerance for the bi-encoder symmetry invariant score(a, b) == score(b, a).
SYMMETRY_TOL = 1e-3

# Self-contained correctness cases: (query, relevant_passage, irrelevant_passage).
# Each query must score higher against its relevant passage than the irrelevant
# one. Multilingual on purpose, since BGE-M3 is a multilingual model.
CASES = [
    (
        "What is the capital of China?",
        "Beijing is the capital of China.",
        "Gravity is a force that attracts two bodies toward each other.",
    ),
    (
        "How do plants make food?",
        "Photosynthesis lets plants convert sunlight into chemical energy.",
        "The stock market fell sharply after the interest rate hike.",
    ),
    (
        "대한민국의 수도는 어디인가요?",
        "서울은 대한민국의 수도이자 최대 도시입니다.",
        "고래는 바다에 사는 포유류입니다.",
    ),
]


def _scores(llm: LLM, data_1: list[str], data_2: list[str]) -> list[float]:
    # N -> N pairing: data_1[i] is scored against data_2[i].
    return [output.outputs.score for output in llm.score(data_1, data_2)]


def check_consistency(llm: LLM) -> None:
    queries = [q for q, _, _ in CASES]
    relevant = [r for _, r, _ in CASES]
    irrelevant = [i for _, _, i in CASES]

    rel_scores = _scores(llm, queries, relevant)
    irr_scores = _scores(llm, queries, irrelevant)
    self_scores = _scores(llm, relevant, relevant)
    swapped_scores = _scores(llm, relevant, queries)  # symmetry: score(r, q)

    failures: list[str] = []
    for idx, (q, _, _) in enumerate(CASES):
        s_rel = rel_scores[idx]
        s_irr = irr_scores[idx]
        s_self = self_scores[idx]
        s_swap = swapped_scores[idx]

        print(
            f"[{idx}] {q!r}\n"
            f"     relevant={s_rel:.4f}  irrelevant={s_irr:.4f}  "
            f"self={s_self:.4f}  swapped={s_swap:.4f}"
        )

        # 1) Relevance ordering: relevant must clearly beat irrelevant.
        if s_rel <= s_irr + RELEVANCE_MARGIN:
            failures.append(
                f"[{idx}] relevant ({s_rel:.4f}) did not beat irrelevant "
                f"({s_irr:.4f}) by margin {RELEVANCE_MARGIN}"
            )
        # 2) Self-similarity must sit at (near) the maximum.
        if s_self < SELF_SIM_FLOOR:
            failures.append(
                f"[{idx}] self-similarity {s_self:.4f} below floor {SELF_SIM_FLOOR}"
            )
        # 3) Self-similarity must dominate any cross-pair score.
        if s_self < s_rel:
            failures.append(
                f"[{idx}] self-similarity {s_self:.4f} below relevant {s_rel:.4f}"
            )
        # 4) Bi-encoder symmetry: score(q, r) ~= score(r, q).
        if abs(s_rel - s_swap) > SYMMETRY_TOL:
            failures.append(
                f"[{idx}] asymmetric score: score(q,r)={s_rel:.4f} vs "
                f"score(r,q)={s_swap:.4f} (tol {SYMMETRY_TOL})"
            )

    if failures:
        print("\nFAILED consistency checks:")
        for f in failures:
            print(f"  - {f}")
        exit(1)

    print(f"\nAll consistency checks passed over {len(CASES)} cases.")


def main(model_id: str):
    llm = LLM(model=model_id)
    check_consistency(llm)


def entry_point(
    model_id: str = "./bge-m3-1k-batch4",
):
    main(model_id=model_id)


if __name__ == "__main__":
    fire.Fire(entry_point)
