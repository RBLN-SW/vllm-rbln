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
import os

os.environ["VLLM_RBLN_USE_VLLM_MODEL"] = "1"
os.environ["VLLM_RBLN_COMPILE_MODEL"] = "0"  # use cpu device
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=1024)
    return parser.parse_args()


def main():
    args = parse_args()

    kv_config = KVTransferConfig(
        kv_connector="RBLNLMCacheConnectorV1",
        kv_role="kv_both",
    )

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
        kv_transfer_config=kv_config,
    )

    _ = llm.generate(
        [
            "Qwen3 is the latest generation of large language models in Qwen series, "
            "offering a comprehensive suite of dense and mixture-of-experts",
        ],
        SamplingParams(temperature=0.0),
    )

    outputs = llm.generate(
        [
            "Qwen3 is the latest generation of large language models in Qwen series, "
            "offering a comprehensive suite of dense and "
            "mixture-of-experts (MoE) models",
        ],
        SamplingParams(temperature=0.0),
    )
    # Print the outputs.

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == "__main__":
    main()
