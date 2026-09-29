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

"""Run speculative decoding using the vllm model path on RBLN.

By default, generate from four text prompts and print the results.
EAGLE, EAGLE3, and DFlash require a draft checkpoint compatible with the target.
N-gram and suffix drafting reuse token sequences; MTP uses the target's MTP layers.
Suffix requires arctic-inference and treats num-spec-tokens as an upper bound.
DFlash requires device tensors, which are enabled by default on RBLN.
"""

from argparse import BooleanOptionalAction

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.benchmarks.datasets import add_dataset_parser, get_samples
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.metrics.reader import Counter, Vector


def main() -> None:
    parser = FlexibleArgumentParser(description=__doc__)
    add_dataset_parser(parser)
    parser.set_defaults(dataset_name=None, num_prompts=4)
    parser.add_argument(
        "--prompts", nargs="+", help="Text prompts to use instead of a dataset."
    )
    parser.add_argument(
        "--method",
        choices=["ngram", "suffix", "eagle", "eagle3", "mtp", "dflash"],
        default="eagle",
    )
    parser.add_argument("--model-dir", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--eagle-dir", default=None)
    parser.add_argument("--dflash-dir", default=None)
    parser.add_argument(
        "--backend", choices=["openai", "openai-chat"], default="openai"
    )
    parser.add_argument("--num-spec-tokens", type=int, default=2)
    parser.add_argument("--prompt-lookup-max", type=int, default=5)
    parser.add_argument("--prompt-lookup-min", type=int, default=2)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--temp", type=float, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--print-output", action=BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.prompts is not None and args.dataset_name is not None:
        parser.error("--prompts and --dataset-name cannot be used together")
    args.enable_multimodal_chat = args.backend == "openai-chat"

    if args.dataset_name is None:
        prompts = args.prompts
        if prompts is None:
            prompts = [
                "A robot may not injure a human being",
                "The capital of France is",
                "The future of AI is",
                "A good way to learn programming is",
            ]
        if args.backend == "openai-chat":
            llm_prompts = [[{"role": "user", "content": prompt}] for prompt in prompts]
        else:
            llm_prompts = prompts
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_dir, trust_remote_code=args.trust_remote_code
        )
        samples = get_samples(args, tokenizer)
        prompts = [sample.prompt for sample in samples]
        if args.enable_multimodal_chat:
            llm_prompts = prompts
        else:
            # Dataset chat templates may already include the BOS token.
            llm_prompts = [
                {
                    "prompt_token_ids": tokenizer.encode(
                        sample.prompt, add_special_tokens=False
                    ),
                    "multi_modal_data": sample.multi_modal_data,
                }
                for sample in samples
            ]

    if args.method in ("eagle", "eagle3"):
        eagle_dir = args.eagle_dir
        if eagle_dir is None:
            eagle_dir = (
                "yuhuili/EAGLE-LLaMA3.1-Instruct-8B"
                if args.method == "eagle"
                else "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
            )
        speculative_config = {
            "method": args.method,
            "model": eagle_dir,
            "num_speculative_tokens": args.num_spec_tokens,
        }
    elif args.method == "ngram":
        speculative_config = {
            "method": "ngram",
            "num_speculative_tokens": args.num_spec_tokens,
            "prompt_lookup_max": args.prompt_lookup_max,
            "prompt_lookup_min": args.prompt_lookup_min,
        }
    elif args.method in ("suffix", "mtp"):
        speculative_config = {
            "method": args.method,
            "num_speculative_tokens": args.num_spec_tokens,
        }
    elif args.method == "dflash":
        dflash_dir = args.dflash_dir
        if dflash_dir is None:
            dflash_dir = "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat"
        speculative_config = {
            "method": "dflash",
            "model": dflash_dir,
            "num_speculative_tokens": args.num_spec_tokens,
        }
    else:
        raise ValueError(f"Unknown method: {args.method}")

    llm = LLM(
        model=args.model_dir,
        model_impl="vllm",
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=args.tp,
        pipeline_parallel_size=args.pp,
        block_size=1024,
        max_num_batched_tokens=512,
        max_model_len=args.max_model_len,
        speculative_config=speculative_config,
        disable_log_stats=False,
    )
    sampling_params = SamplingParams(
        temperature=args.temp,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.output_len,
    )
    if args.backend == "openai-chat":
        outputs = llm.chat(llm_prompts, sampling_params=sampling_params)
    else:
        outputs = llm.generate(llm_prompts, sampling_params=sampling_params)

    if args.print_output:
        for i, output in enumerate(outputs):
            print("-" * 50)
            print(f"prompt: {prompts[i]}")
            print(f"generated text: {output.outputs[0].text}")
            print("-" * 50)

    metrics = llm.get_metrics()
    total_num_output_tokens = sum(
        len(output.outputs[0].token_ids) for output in outputs
    )
    num_drafts = 0
    num_draft_tokens = 0
    num_accepted_tokens = 0
    acceptance_counts = [0] * args.num_spec_tokens
    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_draft_tokens":
            assert isinstance(metric, Counter)
            num_draft_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens":
            assert isinstance(metric, Counter)
            num_accepted_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            for pos in range(len(metric.values)):
                acceptance_counts[pos] += metric.values[pos]

    print("-" * 50)
    print(f"total_num_output_tokens: {total_num_output_tokens}")
    print(f"num_drafts: {num_drafts}")
    print(f"num_draft_tokens: {num_draft_tokens}")
    print(f"num_accepted_tokens: {num_accepted_tokens}")
    acceptance_length = 1 + (num_accepted_tokens / num_drafts) if num_drafts > 0 else 1
    print(f"mean acceptance length: {acceptance_length:.2f}")
    print("-" * 50)

    for i in range(len(acceptance_counts)):
        acceptance_rate = acceptance_counts[i] / num_drafts if num_drafts > 0 else 0
        print(f"acceptance at token {i}: {acceptance_rate:.2f}")


if __name__ == "__main__":
    main()
