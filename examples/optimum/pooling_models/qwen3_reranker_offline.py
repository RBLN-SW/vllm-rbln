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

import math

import fire
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def format_instruction(instruction, query, doc):
    return [
        {"role": "system", "content": instruction},
        {"role": "query", "content": query},
        {"role": "document", "content": doc},
    ]


def process_inputs(pairs, instruction, tokenizer) -> list[str]:
    messages = [format_instruction(instruction, query, doc) for query, doc in pairs]
    # The reranker chat template emits the judge system prompt and the trailing
    # assistant/think block on its own, so the rendered text is already the
    # complete prompt. vLLM tokenizes it back to the same ids.
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def get_input_prompts(tokenizer) -> list[str]:
    task = "Given a web search query, retrieve relevant passages that answer the query"
    queries = [
        "What is the capital of China?",
        "Explain gravity",
    ]
    documents = [
        "The capital of China is Beijing.",
        (
            "Gravity is a force that attracts two bodies towards each other. "
            "It gives weight to physical objects and "
            "is responsible for the movement of planets around the sun."
        ),
    ]

    pairs = list(zip(queries, documents))
    return process_inputs(pairs, task, tokenizer)


def compute_logits(outputs, true_token, false_token):
    scores = []
    for output in outputs:
        final_logits = output.outputs[0].logprobs[-1]
        true_logit = (
            final_logits[true_token].logprob if true_token in final_logits else -10
        )
        false_logit = (
            final_logits[false_token].logprob if false_token in final_logits else -10
        )
        true_score = math.exp(true_logit)
        false_score = math.exp(false_logit)
        scores.append(true_score / (true_score + false_score))
    return scores


def main(
    max_seq_len: int = 32768,
    num_input_prompt: int = 2,
    model: str = "Qwen/Qwen3-Reranker-0.6B",
):
    tokenizer = AutoTokenizer.from_pretrained(model)

    true_token = tokenizer("yes", add_special_tokens=False).input_ids[0]
    false_token = tokenizer("no", add_special_tokens=False).input_ids[0]

    llm = LLM(
        model=model,
        block_size=4096,
        max_model_len=max_seq_len,
    )
    prompts = get_input_prompts(tokenizer)
    prompts = prompts[:num_input_prompt]

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1,
        logprobs=20,
        allowed_token_ids=[true_token, false_token],
    )

    result = llm.generate(prompts, sampling_params)
    score = compute_logits(result, true_token, false_token)
    print(f"scores: {score}")


if __name__ == "__main__":
    fire.Fire(main)
