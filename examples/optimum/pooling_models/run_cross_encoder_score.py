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

# (query, document) pairs to score.
PAIRS = [
    (
        "what is panda?",
        "The giant panda (Ailuropoda melanoleuca), sometimes called a panda "
        "bear or simply panda, is a bear species endemic to China.",
    ),
    (
        "How do plants make food?",
        "Photosynthesis lets plants convert sunlight into chemical energy.",
    ),
]


def main(model_id: str):
    # Offline `LLM` exposes the native score API. For a cross-encoder it fuses
    # each (query, document) pair into one sequence and returns the sigmoid of
    # the classifier logit -- no manual activation needed.
    llm = LLM(model=model_id)

    data_1 = [a for a, _ in PAIRS]
    data_2 = [b for _, b in PAIRS]

    # N -> N pairing: data_1[i] is scored against data_2[i].
    outputs = llm.score(data_1, data_2)
    for idx, output in enumerate(outputs):
        print(f"[{idx}] score={output.outputs.score:.4f}")


def entry_point(
    model_id: str = "./bge-reranker-base",
):
    main(model_id=model_id)


if __name__ == "__main__":
    fire.Fire(entry_point)
