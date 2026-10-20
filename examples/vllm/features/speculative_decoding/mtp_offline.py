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

"""Run MTP speculative decoding with DeepSeek-V3.2 on RBLN.

MTP uses the target checkpoint's multi-token prediction layers; no separate
draft checkpoint is needed.
The fixed DP=8, TP=1 configuration enables EP and requires 8 NPUs.
Each rank processes one chat prompt and prints its draft acceptance metrics.
"""

import argparse
import os
from multiprocessing import get_context
from time import sleep

from vllm import LLM, SamplingParams
from vllm.utils.network_utils import get_open_port
from vllm.v1.metrics.reader import Counter

DP_SIZE = 8


def run_rank(args: argparse.Namespace, dp_rank: int, dp_master_port: int) -> None:
    os.environ["VLLM_DP_RANK"] = str(dp_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(DP_SIZE)
    os.environ["VLLM_DP_MASTER_IP"] = "127.0.0.1"
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)

    llm = LLM(
        model="deepseek-ai/DeepSeek-V3.2",
        model_impl="vllm",
        max_num_seqs=1,
        max_model_len=32768,
        block_size=8192,
        max_num_batched_tokens=512,
        tensor_parallel_size=1,
        enable_expert_parallel=True,
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": args.num_spec_tokens,
        },
        disable_log_stats=False,
    )
    prompts = [
        "Explain the first law of robotics.",
        "What is the capital of France?",
        "What might the future of AI look like?",
        "What is a good way to learn programming?",
    ]
    prompts = [prompts[dp_rank % len(prompts)]]
    conversations = [[{"role": "user", "content": prompt}] for prompt in prompts]
    outputs = llm.chat(
        conversations,
        sampling_params=SamplingParams(temperature=0.0, max_tokens=256),
    )
    for prompt, output in zip(prompts, outputs):
        print("-" * 50)
        print(
            f"[DP rank {dp_rank}] prompt: {prompt}\n"
            f"[DP rank {dp_rank}] generated text: {output.outputs[0].text}"
        )

    num_drafts = 0
    num_draft_tokens = 0
    num_accepted_tokens = 0
    for metric in llm.get_metrics():
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_draft_tokens":
            assert isinstance(metric, Counter)
            num_draft_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens":
            assert isinstance(metric, Counter)
            num_accepted_tokens += metric.value

    acceptance_length = 1 + num_accepted_tokens / num_drafts if num_drafts > 0 else 1
    print("-" * 50)
    print(f"[DP rank {dp_rank}] num_drafts: {num_drafts}")
    print(f"[DP rank {dp_rank}] num_draft_tokens: {num_draft_tokens}")
    print(f"[DP rank {dp_rank}] num_accepted_tokens: {num_accepted_tokens}")
    print(f"[DP rank {dp_rank}] mean acceptance length: {acceptance_length:.2f}")
    print("-" * 50)

    # Let all engines pause their processing loops before any rank exits.
    sleep(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-spec-tokens", type=int, default=2)
    args = parser.parse_args()

    dp_master_port = get_open_port()
    context = get_context("spawn")
    procs = []
    for dp_rank in range(DP_SIZE):
        proc = context.Process(target=run_rank, args=(args, dp_rank, dp_master_port))
        proc.start()
        procs.append(proc)

    exit_code = 0
    for proc in procs:
        proc.join()
        if proc.exitcode:
            exit_code = proc.exitcode
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
