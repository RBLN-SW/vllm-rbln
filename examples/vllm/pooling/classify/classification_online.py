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

r"""Classify sentiment using a Qwen3-based sequence classification model.

Start the server:
    vllm serve rd211/Qwen3-0.6B-Instruct --runner pooling --model-impl vllm \
        --block-size 1024 --max-num-batched-tokens 512 --max-num-seqs 1
Run the client:
    python examples/vllm/pooling/classify/classification_online.py
The response includes probabilities and the predicted negative/neutral/positive
label. Both text and token ID inputs are demonstrated.
"""

import json

import httpx


def main() -> None:
    prompts = [
        "I loved this movie. The acting was wonderful.",
        "The meeting starts at three o'clock.",
        "This product is terrible. I regret buying it.",
    ]
    with httpx.Client(base_url="http://localhost:8000", timeout=60.0) as client:
        response = client.get("/v1/models")
        response.raise_for_status()
        model = response.json()["data"][0]["id"]
        response = client.post("/classify", json={"model": model, "input": prompts})
        response.raise_for_status()
        print("Text inputs:", json.dumps(response.json(), indent=2))

        token_ids = []
        for prompt in prompts:
            response = client.post("/tokenize", json={"model": model, "prompt": prompt})
            response.raise_for_status()
            token_ids.append(response.json()["tokens"])
        response = client.post("/classify", json={"model": model, "input": token_ids})
        response.raise_for_status()
        print("Token ID inputs:", json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    main()
