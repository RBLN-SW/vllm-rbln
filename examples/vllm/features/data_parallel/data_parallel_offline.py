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

"""Run offline data parallel inference with a MoE model on a single RBLN node.

Each DP rank processes four prompts with its own sampling parameters.
The script sets up DP communication; workers assign visible NPUs to each rank.
The default configuration requires four NPUs. TP increases the NPUs per DP rank,
and EP distributes experts across the combined DP and TP ranks when enabled.
"""

import argparse
import os
from multiprocessing import get_context
from time import sleep

from vllm import LLM, SamplingParams
from vllm.utils.network_utils import get_open_port


def run_rank(args: argparse.Namespace, dp_rank: int, dp_master_port: int) -> None:
    os.environ["VLLM_DP_RANK"] = str(dp_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(args.dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = "127.0.0.1"
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ] * args.dp_size
    prompts_per_rank = len(prompts) // args.dp_size
    start = dp_rank * prompts_per_rank
    prompts = prompts[start : start + prompts_per_rank]
    print(f"DP rank {dp_rank} needs to process {len(prompts)} prompts")

    sampling_params = SamplingParams(
        temperature=0.8, top_p=0.95, max_tokens=[16, 20][dp_rank % 2]
    )
    llm = LLM(
        model=args.model,
        model_impl="vllm",
        tensor_parallel_size=args.tp_size,
        enable_expert_parallel=args.enable_expert_parallel,
        block_size=8192,
        max_num_batched_tokens=512,
        max_num_seqs=1,
    )
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        print(
            f"DP rank {dp_rank}, Prompt: {output.prompt!r}, "
            f"Generated text: {output.outputs[0].text!r}"
        )

    # Let all engines pause their processing loops before any rank exits.
    sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="MiniMaxAI/MiniMax-M2.7")
    parser.add_argument("--dp-size", type=int, default=4)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--enable-expert-parallel", action="store_true")
    args = parser.parse_args()
    if args.dp_size < 1 or args.tp_size < 1:
        parser.error("DP and TP sizes must be positive")

    dp_master_port = get_open_port()
    context = get_context("spawn")
    procs = []
    for dp_rank in range(args.dp_size):
        proc = context.Process(target=run_rank, args=(args, dp_rank, dp_master_port))
        proc.start()
        procs.append(proc)

    exit_code = 0
    for proc in procs:
        proc.join()
        if proc.exitcode:
            exit_code = proc.exitcode
    raise SystemExit(exit_code)
