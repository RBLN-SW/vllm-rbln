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

r"""Request Qwen3 embeddings through the OpenAI-compatible API.

Start the server:
    vllm serve Qwen/Qwen3-Embedding-0.6B --runner pooling --model-impl vllm \
        --block-size 1024 --max-num-batched-tokens 512 --max-num-seqs 1
Run the client:
    python examples/vllm/pooling/embed/openai_embedding_client.py
"""

from openai import OpenAI
from vllm.utils.print_utils import print_embeddings


def main() -> None:
    with OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1") as client:
        model = client.models.list().data[0].id
        prompts = [
            "Hello my name is",
            "The best thing about vLLM is that it supports many different models",
        ]
        response = client.embeddings.create(
            input=prompts, model=model, encoding_format="float"
        )
        for prompt, data in zip(prompts, response.data):
            print(f"Prompt: {prompt!r}")
            print_embeddings(data.embedding)
            print("-" * 60)


if __name__ == "__main__":
    main()
