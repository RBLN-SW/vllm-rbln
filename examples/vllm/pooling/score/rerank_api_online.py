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

r"""Rank documents using a Qwen3 reranker server.

Start the server from the repository root:
    vllm serve Qwen/Qwen3-Reranker-0.6B --runner pooling --model-impl vllm \
        --hf-overrides '{"architectures": ["Qwen3ForSequenceClassification"],
            "classifier_from_token": ["no", "yes"],
            "is_original_qwen3_reranker": true}' \
        --chat-template examples/vllm/pooling/score/template/qwen3_reranker.jinja \
        --block-size 1024 --max-num-batched-tokens 512 --max-num-seqs 1
Run the client:
    python examples/vllm/pooling/score/rerank_api_online.py
"""

import json

import httpx


def main() -> None:
    with httpx.Client(base_url="http://localhost:8000", timeout=60.0) as client:
        response = client.get("/v1/models")
        response.raise_for_status()
        model = response.json()["data"][0]["id"]
        payload = {
            "model": model,
            "query": "What is the capital of France?",
            "documents": [
                "The capital of Brazil is Brasilia.",
                "The capital of France is Paris.",
                "Horses and cows are both animals",
            ],
        }
        response = client.post("/rerank", json=payload)
        response.raise_for_status()
        print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    main()
