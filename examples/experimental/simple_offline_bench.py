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

import os

os.environ["RBLN_KERNEL_MODE"] = "triton"
os.environ["VLLM_RBLN_ENFORCE_MODEL_FP32"] = "1"
os.environ["VLLM_RBLN_COMPILE_STRICT_MODE"] = "1"
os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
os.environ["VLLM_USE_V1"] = "1"

import argparse
import time

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-instruct")
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-num-batched-tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=2 * 1024)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--num-requests", type=int, default=64)
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=1024)

    return parser.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    vocab = iter(tokenizer.vocab)
    single_token_string = next(vocab)
    while True:
        if (
            len(tokenizer.encode(single_token_string * 2, add_special_tokens=False))
            == 2
        ):
            break
        single_token_string = next(vocab)

    prompts = [single_token_string * (args.input_len - 1)] * args.num_requests

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        block_size=args.block_size,
        enable_chunked_prefill=True,
        gpu_memory_utilization=1,
        enable_prefix_caching=False,
    )

    _ = llm.generate(
        prompts[: min(len(prompts), 3)],
        SamplingParams(
            temperature=0.0,
            ignore_eos=True,
            max_tokens=args.output_len,
        ),
    )

    st = time.perf_counter()
    outputs = llm.generate(
        prompts,
        SamplingParams(
            temperature=0.0,
            ignore_eos=True,
            max_tokens=args.output_len,
        ),
    )
    total_run_time = time.perf_counter() - st

    print(f"total run time: {total_run_time} (sec)")
    assert all(
        output.prompt_token_ids and (len(output.prompt_token_ids) == args.input_len)
        for output in outputs
    )
    assert all(
        len(output.outputs[0].token_ids) == args.output_len for output in outputs
    )
    print(
        "output throughput: "
        f"{(args.num_requests * args.output_len) / total_run_time} (token/sec)"
    )


if __name__ == "__main__":
    main()
