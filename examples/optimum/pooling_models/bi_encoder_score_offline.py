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

import fire
from vllm import LLM

# (text, text_pair) pairs to score. BGE-M3 is a bi-encoder, so score() embeds
# each side and returns their cosine similarity.
PAIRS = [
    (
        "What is the capital of China?",
        "Beijing is the capital of China.",
    ),
    (
        "How do plants make food?",
        "Photosynthesis lets plants convert sunlight into chemical energy.",
    ),
]


def main(
    model: str = "BAAI/bge-m3",
    max_num_seqs: int = 1,
    assert_ranking: bool = False,
    max_model_len: int = 4096,
    block_size: int = 4096,
):
    llm = LLM(
        model=model,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        block_size=block_size,
    )

    data_1 = [a for a, _ in PAIRS]
    data_2 = [b for _, b in PAIRS]

    # N -> N pairing: data_1[i] is scored against data_2[i] (its correct match).
    positives = llm.score(data_1, data_2)
    for idx, output in enumerate(positives):
        print(f"[{idx}] score={output.outputs.score:.4f}")

    # if assert_ranking:
    #     # Self-contained accuracy check (no external golden): score each query
    #     # against a mismatched document (the next pair's, rotated) and require
    #     # the correct pairing to score higher.
    #     neg_docs = data_2[1:] + data_2[:1]
    #     negatives = llm.score(data_1, neg_docs)
    #     for idx in range(len(PAIRS)):
    #         pos = positives[idx].outputs.score
    #         neg = negatives[idx].outputs.score
    #         print(f"[{idx}] positive={pos:.4f} negative={neg:.4f}")
    #         if not pos > neg:
    #             print(
    #                 f"ranking check FAILED at {idx}: positive {pos} <= negative {neg}"
    #             )
    #             exit(1)
    #     print("ranking check PASSED")


if __name__ == "__main__":
    fire.Fire(main)
