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

"""Evaluate answers with Skywork-Reward-V2-Qwen3-0.6B on RBLN.

Sequence rewards use the classify pooling task with activation disabled.
Rewards are raw scores, not probabilities; higher scores indicate preference
among answers to the same question.
"""

from vllm import LLM, PoolingParams


def main() -> None:
    llm = LLM(
        model="Skywork/Skywork-Reward-V2-Qwen3-0.6B",
        model_impl="vllm",
        runner="pooling",
        block_size=1024,
        max_num_batched_tokens=512,
    )
    question = "What is 12 minus 4, plus 1, divided by 3?"
    answers = [
        "First subtract: 12 - 4 = 8. Then add: 8 + 1 = 9. "
        "Finally divide: 9 / 3 = 3. The answer is 3.",
        "12 - 4 = 6, then 6 + 1 = 7. Dividing by 3 gives 4. The answer is 4.",
    ]
    tokenizer = llm.get_tokenizer()
    prompts = [
        {
            "prompt_token_ids": tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ],
                tokenize=True,
                add_generation_prompt=False,
            )
        }
        for answer in answers
    ]
    outputs = llm.encode(
        prompts,
        pooling_task="classify",
        pooling_params=PoolingParams(use_activation=False),
    )
    print("\nGenerated Outputs:\n" + "-" * 60)
    for answer, output in zip(answers, outputs):
        print(
            f"Question: {question!r}\n"
            f"Answer: {answer!r}\n"
            f"Reward: {output.outputs.data.item()}\n" + "-" * 60
        )


if __name__ == "__main__":
    main()
