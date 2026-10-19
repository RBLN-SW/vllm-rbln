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

r"""Evaluate answers using a Skywork sequence reward server.

Start the server:
    vllm serve Skywork/Skywork-Reward-V2-Qwen3-0.6B \
        --runner pooling --model-impl vllm --block-size 1024 \
        --max-num-batched-tokens 512 --max-num-seqs 1
Run the client:
    python examples/vllm/pooling/reward/sequence_reward_online.py
With use_activation=False, the response's probs contains raw reward scores,
not probabilities. Higher scores indicate preference for the same question.
"""

import httpx


def main() -> None:
    question = "What is 12 minus 4, plus 1, divided by 3?"
    answers = [
        "First subtract: 12 - 4 = 8. Then add: 8 + 1 = 9. "
        "Finally divide: 9 / 3 = 3. The answer is 3.",
        "12 - 4 = 6, then 6 + 1 = 7. Dividing by 3 gives 4. The answer is 4.",
    ]
    with httpx.Client(base_url="http://localhost:8000", timeout=60.0) as client:
        response = client.get("/v1/models")
        response.raise_for_status()
        model = response.json()["data"][0]["id"]
        print("\nGenerated Outputs:\n" + "-" * 60)
        for answer in answers:
            response = client.post(
                "/classify",
                json={
                    "model": model,
                    "messages": [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer},
                    ],
                    "add_generation_prompt": False,
                    "use_activation": False,
                },
            )
            response.raise_for_status()
            reward = response.json()["data"][0]["probs"][0]
            print(
                f"Question: {question!r}\n"
                f"Answer: {answer!r}\n"
                f"Reward: {reward}\n" + "-" * 60
            )


if __name__ == "__main__":
    main()
