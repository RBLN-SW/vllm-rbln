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

"""Generate Qwen3 embeddings at different Matryoshka dimensions on RBLN.

Qwen3-Embedding-0.6B supports dimensions from 32 to 1024. is_matryoshka
explicitly enables dimension selection because its HF config omits the flag.
"""

from vllm import LLM, PoolingParams
from vllm.utils.print_utils import print_embeddings


def main() -> None:
    llm = LLM(
        model="Qwen/Qwen3-Embedding-0.6B",
        model_impl="vllm",
        runner="pooling",
        hf_overrides={"is_matryoshka": True},
        block_size=1024,
        max_num_batched_tokens=512,
    )
    prompts = [
        "Follow the white rabbit.",
        "Sigue al conejo blanco.",
        "Suis le lapin blanc.",
        "跟着白兔走。",
        "اتبع الأرنب الأبيض.",
        "Folge dem weißen Kaninchen.",
    ]
    for dimensions in (32, 256, 1024):
        outputs = llm.embed(
            prompts, pooling_params=PoolingParams(dimensions=dimensions)
        )
        print(f"\nEmbedding dimensions: {dimensions}")
        print("-" * 60)
        for prompt, output in zip(prompts, outputs):
            embedding = output.outputs.embedding
            assert len(embedding) == dimensions, "Unexpected embedding dimensions."
            print(f"Prompt: {prompt!r}")
            print_embeddings(embedding)
            print("-" * 60)


if __name__ == "__main__":
    main()
