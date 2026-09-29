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

"""Generate chat responses using the vllm model path on RBLN."""

import argparse

from vllm import LLM
from vllm.outputs import RequestOutput


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    sampling_group = parser.add_argument_group("Sampling parameters")
    sampling_group.add_argument("--max-tokens", type=int)
    sampling_group.add_argument("--temperature", type=float)
    sampling_group.add_argument("--top-p", type=float)
    sampling_group.add_argument("--top-k", type=int)
    args = parser.parse_args()

    # Reuse one engine for both single and batched conversations.
    llm = LLM(
        model=args.model,
        model_impl="vllm",
        block_size=1024,
        max_num_batched_tokens=512,
    )
    # Keep the model's sampling defaults unless a CLI option overrides them.
    sampling_params = llm.get_default_sampling_params()
    if args.max_tokens is not None:
        sampling_params.max_tokens = args.max_tokens
    if args.temperature is not None:
        sampling_params.temperature = args.temperature
    if args.top_p is not None:
        sampling_params.top_p = args.top_p
    if args.top_k is not None:
        sampling_params.top_k = args.top_k

    def print_outputs(outputs: list[RequestOutput], prompts: list):
        assert len(outputs) == len(prompts)
        print("\nGenerated Outputs:\n" + "-" * 80)
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompts[i]!r}\n")
            print(f"Generated text: {generated_text!r}")
            print("-" * 80)

    print("=" * 80)
    # Include the conversation history, ending with the next user turn.
    conversation = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hello! How can I assist you today?"},
        {
            "role": "user",
            "content": "Write an essay about the importance of higher education.",
        },
    ]
    # chat() applies the model's chat template before generating a reply.
    outputs = llm.chat(conversation, sampling_params, use_tqdm=False)
    print_outputs(outputs, [conversation])

    # Each conversation in the batch is an independent request.
    conversations = [conversation for _ in range(10)]
    outputs = llm.chat(conversations, sampling_params, use_tqdm=True)
    print_outputs(outputs, conversations)


if __name__ == "__main__":
    main()
