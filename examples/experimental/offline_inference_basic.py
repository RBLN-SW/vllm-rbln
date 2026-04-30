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

import argparse

from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-instruct")
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=40 * 1024)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--enable-expert-parallel", action="store_true")
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Pass user strings to the model verbatim (skip instruct chat formatting).",
    )
    return parser.parse_args()


def _format_prompts_for_model(tokenizer, user_turns: list[str], use_chat_template: bool) -> list[str]:
    if not use_chat_template:
        return user_turns
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            "Tokenizer has no chat_template; use --no-chat-template or an instruct-tuned checkpoint."
        )
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for text in user_turns
    ]


def main():
    args = parse_args()

    # Raw user messages (formatted with the model chat template unless --no-chat-template).
    user_turns = [
        "What is the capital of France?",
        "In one sentence, what do you think the future of AI looks like?",
        "Hello! Please reply with a brief friendly greeting.",
        "Who is the current president of the United States?",
    ]

    # Create an LLM.
    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        tensor_parallel_size=args.tensor_parallel_size,
        block_size=args.block_size,
        enable_chunked_prefill=True,
        max_num_batched_tokens=128,
        gpu_memory_utilization=0.9,
        enable_expert_parallel=args.enable_expert_parallel,
    )

    tokenizer = llm.get_tokenizer()
    prompts = _format_prompts_for_model(tokenizer, user_turns, use_chat_template=not args.no_chat_template)

    # Generate texts from the prompts. The output is a list of RequestOutput
    # objects that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, max_tokens=8))
    # Print the outputs.
    for prompt, output in zip(user_turns, outputs):
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == "__main__":
    main()
