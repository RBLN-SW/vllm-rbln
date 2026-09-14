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

"""offline_inference_basic with an exact-length prompt.

The prompt is built from token ids, so its length is exactly

    prompt_len = num_prompt_blocks * prompt_block_size + prompt_extra

The three parameters place the sequence relative to the SWA kernel block
(= sliding window; 128 for gpt-oss): e.g. --num-prompt-blocks 2
--prompt-extra 18 starts decode at seq = 274, where the window spans two
blocks and the first one is already behind it.
"""

import argparse

from vllm import LLM, SamplingParams, TokensPrompt

PROMPT_TEXT = (
    "The sliding window attention mechanism restricts each token to attend "
    "only to a fixed number of preceding tokens, which keeps the key and "
    "value cache bounded regardless of how long the sequence grows. When a "
    "new token arrives, its keys and values are appended to the cache, and "
    "entries older than the window are no longer referenced by any query. "
    "Paged attention organizes this cache into fixed-size blocks, and a "
    "block table maps each logical position in the sequence to the physical "
    "block that actually holds its data, so blocks can be allocated and "
    "reused out of order. During decode, the kernel gathers only the blocks "
    "the window covers, computes the attention scores against them, and "
    "masks out any key that falls outside the window or beyond the current "
    "position. "
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="openai/gpt-oss-20b")
    parser.add_argument("--max-num-seqs", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--enable-expert-parallel", action="store_true")
    # prompt_len = num_prompt_blocks * prompt_block_size + prompt_extra
    parser.add_argument(
        "--num-prompt-blocks",
        type=int,
        default=2,
        help="whole kernel blocks the prompt fills",
    )
    parser.add_argument(
        "--prompt-block-size",
        type=int,
        default=128,
        help="kernel block size, i.e. the sliding window (128 for gpt-oss)",
    )
    parser.add_argument(
        "--prompt-extra",
        type=int,
        default=18,
        help="tokens past the last whole block",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32,
        help="decode steps to run past the prompt",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    prompt_len = args.num_prompt_blocks * args.prompt_block_size + args.prompt_extra
    assert prompt_len >= 1, "prompt must hold at least one token"
    assert prompt_len + args.max_tokens <= args.max_model_len, (
        f"prompt {prompt_len} + decode {args.max_tokens} exceeds "
        f"max_model_len {args.max_model_len}"
    )

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        tensor_parallel_size=args.tensor_parallel_size,
        block_size=args.block_size,
        enable_chunked_prefill=True,
        max_num_batched_tokens=128,
        num_gpu_blocks_override=8,
        enable_expert_parallel=args.enable_expert_parallel,
    )

    tokenizer = llm.get_tokenizer()
    text_ids = tokenizer.encode(PROMPT_TEXT, add_special_tokens=False)
    repeats = -(-prompt_len // len(text_ids))
    prompt = TokensPrompt(prompt_token_ids=(text_ids * repeats)[:prompt_len])
    print(
        f"prompt_len={prompt_len} "
        f"({args.num_prompt_blocks} x {args.prompt_block_size} "
        f"+ {args.prompt_extra}); decode runs seq "
        f"{prompt_len}..{prompt_len + args.max_tokens - 1}"
    )

    # ignore_eos so decode really runs max_tokens steps past the boundary.
    outputs = llm.generate(
        prompt,
        SamplingParams(temperature=0.0, max_tokens=args.max_tokens, ignore_eos=True),
    )
    for output in outputs:
        generated_text = output.outputs[0].text
        num_prompt = len(output.prompt_token_ids)
        num_generated = len(output.outputs[0].token_ids)
        print(
            f"prompt tokens: {num_prompt}, generated tokens: {num_generated}, "
            f"text: {generated_text!r}"
        )


if __name__ == "__main__":
    main()
