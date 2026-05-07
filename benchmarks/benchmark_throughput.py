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
"""Benchmark offline inference throughput."""

import argparse
import dataclasses
import json
import os
import random
import time
import warnings
from typing import Any, Union

import numpy as np
import uvloop
from benchmark_dataset import (
    AIMODataset,
    BurstGPTDataset,
    ConversationDataset,
    InstructCoderDataset,
    RandomDataset,
    SampleRequest,
    ShareGPTDataset,
    SonnetDataset,
    VisionArenaDataset,
)
from benchmark_utils import (
    convert_to_pytorch_benchmark_format,
    write_to_json,
)
from transformers import AutoTokenizer
from typing_extensions import deprecated
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args,
)
from vllm.inputs import TextPrompt, TokensPrompt
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.sampling_params import BeamSearchParams
from vllm.utils import FlexibleArgumentParser, merge_async_iterators


def run_vllm(
    requests: list[SampleRequest],
    n: int,
    engine_args: EngineArgs,
    warmup_requests: int,
    disable_detokenize: bool = False,
) -> tuple[float, list[RequestOutput] | None]:
    llm = LLM(**dataclasses.asdict(engine_args))
    assert all(
        llm.llm_engine.model_config.max_model_len
        >= (request.prompt_len + request.expected_output_len)
        for request in requests
    ), (
        "Please ensure that max_model_len is greater than the sum of"
        " prompt_len and expected_output_len for all requests."
    )

    # Warm up the model
    if warmup_requests > 0:
        print(f"Warming up the model with {warmup_requests} requests")
        assert warmup_requests <= len(requests)
        warmup_prompts: list[Union[TextPrompt, TokensPrompt]] = []
        warmup_sampling_params: list[SamplingParams] = []
        for request in requests[-warmup_requests:]:
            warmup_prompts.append(
                TokensPrompt(
                    prompt_token_ids=request.prompt["prompt_token_ids"],
                    multi_modal_data=request.multi_modal_data,
                )
                if "prompt_token_ids" in request.prompt
                else TextPrompt(
                    prompt=request.prompt, multi_modal_data=request.multi_modal_data
                )
            )
            warmup_sampling_params.append(
                SamplingParams(
                    n=n,
                    temperature=1.0,
                    top_p=1.0,
                    ignore_eos=True,
                    max_tokens=request.expected_output_len,
                    detokenize=not disable_detokenize,
                )
            )
        warmup_lora_requests: list[LoRARequest] | None = None
        if engine_args.enable_lora:
            warmup_lora_requests = [request.lora_request for request in requests]

        _ = llm.generate(
            warmup_prompts,
            warmup_sampling_params,
            lora_request=warmup_lora_requests,
            use_tqdm=True,
        )

    # Add the requests to the engine.
    print(f"Running the model with {len(requests)} requests")
    prompts: list[Union[TextPrompt, TokensPrompt]] = []
    sampling_params: list[SamplingParams] = []
    for request in requests:
        prompts.append(
            TokensPrompt(
                prompt_token_ids=request.prompt["prompt_token_ids"],
                multi_modal_data=request.multi_modal_data,
            )
            if "prompt_token_ids" in request.prompt
            else TextPrompt(
                prompt=request.prompt, multi_modal_data=request.multi_modal_data
            )
        )
        sampling_params.append(
            SamplingParams(
                n=n,
                temperature=1.0,
                top_p=1.0,
                ignore_eos=True,
                max_tokens=request.expected_output_len,
                detokenize=not disable_detokenize,
            )
        )
    lora_requests: list[LoRARequest] | None = None
    if engine_args.enable_lora:
        lora_requests = [request.lora_request for request in requests]

    use_beam_search = False

    outputs = None
    if not use_beam_search:
        start = time.perf_counter()
        outputs = llm.generate(
            prompts, sampling_params, lora_request=lora_requests, use_tqdm=True
        )
        end = time.perf_counter()
    else:
        assert lora_requests is None, "BeamSearch API does not support LoRA"
        # output_len should be the same for all requests.
        output_len = requests[0].expected_output_len
        for request in requests:
            assert request.expected_output_len == output_len
        start = time.perf_counter()
        llm.beam_search(
            prompts,
            BeamSearchParams(
                beam_width=n,
                max_tokens=output_len,
                ignore_eos=True,
            ),
        )
        end = time.perf_counter()
    return end - start, outputs


def run_vllm_chat(
    requests: list[SampleRequest],
    n: int,
    engine_args: EngineArgs,
    disable_detokenize: bool = False,
) -> tuple[float, list[RequestOutput]]:
    """
    Run vLLM chat benchmark. This function is recommended ONLY for benchmarking
    multimodal models as it properly handles multimodal inputs and chat
    formatting. For non-multimodal models, use run_vllm() instead.
    """
    llm = LLM(**dataclasses.asdict(engine_args))

    assert all(
        llm.llm_engine.model_config.max_model_len
        >= (request.prompt_len + request.expected_output_len)
        for request in requests
    ), (
        "Please ensure that max_model_len is greater than the sum of "
        "prompt_len and expected_output_len for all requests."
    )

    prompts = []
    sampling_params: list[SamplingParams] = []
    for request in requests:
        prompts.append(request.prompt)
        sampling_params.append(
            SamplingParams(
                n=n,
                temperature=1.0,
                top_p=1.0,
                ignore_eos=True,
                max_tokens=request.expected_output_len,
                detokenize=not disable_detokenize,
            )
        )
    start = time.perf_counter()
    outputs = llm.chat(prompts, sampling_params, use_tqdm=True)
    end = time.perf_counter()
    return end - start, outputs


async def run_vllm_async(
    requests: list[SampleRequest],
    n: int,
    engine_args: AsyncEngineArgs,
    disable_frontend_multiprocessing: bool = False,
    disable_detokenize: bool = False,
) -> float:
    raise AssertionError("This function is not supported currently")
    async with build_async_engine_client_from_engine_args(
        engine_args,
        disable_frontend_multiprocessing=disable_frontend_multiprocessing,
    ) as llm:
        model_config = await llm.get_model_config()
        assert all(
            model_config.max_model_len
            >= (request.prompt_len + request.expected_output_len)
            for request in requests
        ), (
            "Please ensure that max_model_len is greater than the sum of"
            " prompt_len and expected_output_len for all requests."
        )

        # Add the requests to the engine.
        prompts: list[Union[TextPrompt, TokensPrompt]] = []
        sampling_params: list[SamplingParams] = []
        lora_requests: list[LoRARequest | None] = []
        for request in requests:
            prompts.append(
                TokensPrompt(
                    prompt_token_ids=request.prompt["prompt_token_ids"],
                    multi_modal_data=request.multi_modal_data,
                )
                if "prompt_token_ids" in request.prompt
                else TextPrompt(
                    prompt=request.prompt, multi_modal_data=request.multi_modal_data
                )
            )
            sampling_params.append(
                SamplingParams(
                    n=n,
                    temperature=1.0,
                    top_p=1.0,
                    ignore_eos=True,
                    max_tokens=request.expected_output_len,
                    detokenize=not disable_detokenize,
                )
            )
            lora_requests.append(request.lora_request)

        generators = []
        start = time.perf_counter()
        for i, (prompt, sp, lr) in enumerate(
            zip(prompts, sampling_params, lora_requests)
        ):
            generator = llm.generate(prompt, sp, lora_request=lr, request_id=f"test{i}")
            generators.append(generator)
        all_gens = merge_async_iterators(*generators)
        async for i, res in all_gens:
            pass
        end = time.perf_counter()
        return end - start


def save_to_pytorch_benchmark_format(
    args: argparse.Namespace, results: dict[str, Any]
) -> None:
    pt_records = convert_to_pytorch_benchmark_format(
        args=args,
        metrics={
            "requests_per_second": [results["requests_per_second"]],
            "total_tokens_per_second": [results["total_tokens_per_second"]],
            "output_tokens_per_second": [results["output_tokens_per_second"]],
        },
        extra_info={
            k: results[k] for k in ["elapsed_time", "num_requests", "total_num_tokens"]
        },
    )
    if pt_records:
        # Don't use json suffix here as we don't want CI to pick it up
        pt_file = f"{os.path.splitext(args.output_json)[0]}.pytorch.json"
        write_to_json(pt_file, pt_records)


def get_requests(args, tokenizer):
    # Common parameters for all dataset types.
    common_kwargs = {
        "dataset_path": args.dataset_path,
        "random_seed": args.seed,
    }
    sample_kwargs = {
        "tokenizer": tokenizer,
        "lora_path": args.lora_path,
        "max_loras": args.max_loras,
        "num_requests": args.num_prompts,
        "input_len": args.input_len,
        "output_len": args.output_len,
    }

    if args.dataset_path is None or args.dataset_name == "random":
        sample_kwargs["range_ratio"] = args.random_range_ratio
        sample_kwargs["prefix_len"] = args.prefix_len
        dataset_cls = RandomDataset
    elif args.dataset_name == "sharegpt":
        dataset_cls = ShareGPTDataset
        if args.backend == "vllm-chat":
            sample_kwargs["enable_multimodal_chat"] = True
    elif args.dataset_name == "sonnet":
        assert tokenizer.chat_template or tokenizer.default_chat_template, (
            "Tokenizer/model must have chat template for sonnet dataset."
        )
        dataset_cls = SonnetDataset
        sample_kwargs["prefix_len"] = args.prefix_len
        sample_kwargs["return_prompt_formatted"] = True
    elif args.dataset_name == "burstgpt":
        dataset_cls = BurstGPTDataset
    elif args.dataset_name == "hf":
        common_kwargs["no_stream"] = args.no_stream
        if args.dataset_path in VisionArenaDataset.SUPPORTED_DATASET_PATHS:
            dataset_cls = VisionArenaDataset
            common_kwargs["dataset_subset"] = None
            common_kwargs["dataset_split"] = "train"
            sample_kwargs["enable_multimodal_chat"] = True
        elif args.dataset_path in InstructCoderDataset.SUPPORTED_DATASET_PATHS:
            dataset_cls = InstructCoderDataset
            common_kwargs["dataset_split"] = "train"
        elif args.dataset_path in ConversationDataset.SUPPORTED_DATASET_PATHS:
            dataset_cls = ConversationDataset
            common_kwargs["dataset_subset"] = args.hf_subset
            common_kwargs["dataset_split"] = args.hf_split
            sample_kwargs["enable_multimodal_chat"] = True
        elif args.dataset_path in AIMODataset.SUPPORTED_DATASET_PATHS:
            dataset_cls = AIMODataset
            common_kwargs["dataset_subset"] = None
            common_kwargs["dataset_split"] = "train"
    else:
        raise ValueError(f"Unknown dataset name: {args.dataset_name}")
    # Remove None values
    sample_kwargs = {k: v for k, v in sample_kwargs.items() if v is not None}
    return dataset_cls(**common_kwargs).sample(**sample_kwargs)


@deprecated(
    "benchmark_throughput.py is deprecated and will be removed in a "
    "future version. Please use 'vllm bench throughput' instead.",
)
def main(args: argparse.Namespace):
    if args.seed is None:
        args.seed = 0
    print(args)
    random.seed(args.seed)
    # Sample the requests.
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code
    )
    requests = get_requests(args, tokenizer)
    is_multi_modal = any(request.multi_modal_data is not None for request in requests)
    request_outputs: list[RequestOutput] | None = None
    if args.backend == "vllm":
        if args.async_engine:
            elapsed_time = uvloop.run(
                run_vllm_async(
                    requests,
                    args.n,
                    AsyncEngineArgs.from_cli_args(args),
                    args.disable_frontend_multiprocessing,
                    args.disable_detokenize,
                )
            )
        else:
            elapsed_time, request_outputs = run_vllm(
                requests,
                args.n,
                EngineArgs.from_cli_args(args),
                args.warmup_requests,
                args.disable_detokenize,
            )
    elif args.backend == "vllm-chat":
        elapsed_time, request_outputs = run_vllm_chat(
            requests, args.n, EngineArgs.from_cli_args(args), args.disable_detokenize
        )
    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    median_ttft_ms, mean_ttft_ms, max_ttft_ms = 0, 0, 0
    median_tpot_ms, mean_tpot_ms, max_tpot_ms = 0, 0, 0
    if request_outputs:
        # Note: with the vllm and vllm-chat backends,
        # we have request_outputs, which we use to count tokens.
        total_prompt_tokens = 0
        total_output_tokens = 0

        ttfts: list[float] = []
        tpots: list[float] = []

        for ro in request_outputs:
            if not isinstance(ro, RequestOutput):
                continue
            total_prompt_tokens += (
                len(ro.prompt_token_ids) if ro.prompt_token_ids else 0
            )
            total_output_tokens += sum(len(o.token_ids) for o in ro.outputs if o)

            if ro.metrics is not None:
                num_output_tokens = sum(len(o.token_ids) for o in ro.outputs if o)
                ttfts.append(ro.metrics.first_token_time - ro.metrics.arrival_time)
                tpot = (ro.metrics.finished_time - ro.metrics.first_token_time) / (
                    num_output_tokens - 1
                )
                tpots.append(tpot)

        total_num_tokens = total_prompt_tokens + total_output_tokens
        if ttfts:
            median_ttft_ms = np.median(ttfts) * 1000
            mean_ttft_ms = np.mean(ttfts) * 1000
            max_ttft_ms = np.max(ttfts) * 1000
        if tpots:
            median_tpot_ms = np.median(tpots) * 1000
            mean_tpot_ms = np.mean(tpots) * 1000
            max_tpot_ms = np.max(tpots) * 1000
    else:
        total_num_tokens = sum(r.prompt_len + r.expected_output_len for r in requests)
        total_output_tokens = sum(r.expected_output_len for r in requests)
        total_prompt_tokens = total_num_tokens - total_output_tokens

    if is_multi_modal and args.backend != "vllm-chat":
        print(
            "\033[91mWARNING\033[0m: Multi-modal request with "
            f"{args.backend} backend detected. The "
            "following metrics are not accurate because image tokens are not"
            " counted. See vllm-project/vllm/issues/9778 for details."
        )
        # TODO(vllm-project/vllm/issues/9778): Count multi-modal token length.
        # vllm-chat backend counts the image tokens now

    print(
        f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
        f"{total_num_tokens / elapsed_time:.2f} total tokens/s, "
        f"{total_output_tokens / elapsed_time:.2f} output tokens/s"
    )
    print(f"Total num prompt tokens:  {total_prompt_tokens}")
    print(f"Total num output tokens:  {total_output_tokens}")
    print(
        f"Median TTFT: {median_ttft_ms:.2f} ms, ",
        f"Mean TTFT: {mean_ttft_ms:.2f} ms, Max TTFT: {max_ttft_ms:.2f} ms",
    )
    print(
        f"Median TPOT: {median_tpot_ms:.2f} ms, ",
        f"Mean TPOT: {mean_tpot_ms:.2f} ms, Max TPOT: {max_tpot_ms:.2f} ms",
    )

    # Output JSON results if specified
    if args.output_json:
        results = {
            "elapsed_time": elapsed_time,
            "num_requests": len(requests),
            "total_num_tokens": total_num_tokens,
            "total_input_tokens": total_prompt_tokens,
            "total_output_tokens": total_output_tokens,
            "requests_per_second": len(requests) / elapsed_time,
            "total_tokens_per_second": total_num_tokens / elapsed_time,
            "output_tokens_per_second": total_output_tokens / elapsed_time,
            "median_ttft_ms": median_ttft_ms,
            "mean_ttft_ms": mean_ttft_ms,
            "max_ttft_ms": max_ttft_ms,
            "median_tpot_ms": median_tpot_ms,
            "mean_tpot_ms": mean_tpot_ms,
            "max_tpot_ms": max_tpot_ms,
        }
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4)
        save_to_pytorch_benchmark_format(args, results)


def validate_args(args):
    """
    Validate command-line arguments.
    """

    # === Deprecation and Defaulting ===
    if args.dataset is not None:
        warnings.warn(
            "The '--dataset' argument will be deprecated in the next release. "
            "Please use '--dataset-name' and '--dataset-path' instead.",
            stacklevel=2,
        )
        args.dataset_path = args.dataset

    if not getattr(args, "tokenizer", None):
        args.tokenizer = args.model

    # === Backend Validation ===
    valid_backends = {"vllm", "vllm-chat"}
    if args.backend not in valid_backends:
        raise ValueError(f"Unsupported backend: {args.backend}")

    # === Dataset Configuration ===
    if not args.dataset and not args.dataset_path:
        print("When dataset path is not set, it will default to random dataset")
        args.dataset_name = "random"
        if args.input_len is None:
            raise ValueError("input_len must be provided for a random dataset")

    # === Dataset Name Specific Checks ===
    # --hf-subset and --hf-split: only used
    # when dataset_name is 'hf'
    if args.dataset_name != "hf" and (
        getattr(args, "hf_subset", None) is not None
        or getattr(args, "hf_split", None) is not None
    ):
        warnings.warn(
            "--hf-subset and --hf-split will be ignored \
                since --dataset-name is not 'hf'.",
            stacklevel=2,
        )
    elif args.dataset_name == "hf":
        if args.dataset_path in (
            VisionArenaDataset.SUPPORTED_DATASET_PATHS.keys()
            | ConversationDataset.SUPPORTED_DATASET_PATHS
        ):
            assert args.backend == "vllm-chat", (
                f"{args.dataset_path} needs to use vllm-chat as the backend."
            )  # noqa: E501
        elif args.dataset_path in (
            InstructCoderDataset.SUPPORTED_DATASET_PATHS
            | AIMODataset.SUPPORTED_DATASET_PATHS
        ):
            assert args.backend == "vllm", (
                f"{args.dataset_path} needs to use vllm as the backend."
            )  # noqa: E501
        else:
            raise ValueError(f"{args.dataset_path} is not supported by hf dataset.")

    # --random-range-ratio: only used when dataset_name is 'random'
    if args.dataset_name != "random" and args.random_range_ratio is not None:
        warnings.warn(
            "--random-range-ratio will be ignored since \
                --dataset-name is not 'random'.",
            stacklevel=2,
        )

    # --prefix-len: only used when dataset_name is 'random', 'sonnet', or not
    # set.
    if (
        args.dataset_name not in {"random", "sonnet", None}
        and args.prefix_len is not None
    ):
        warnings.warn(
            "--prefix-len will be ignored since --dataset-name\
                 is not 'random', 'sonnet', or not set.",
            stacklevel=2,
        )

    # === LoRA Settings ===
    if getattr(args, "enable_lora", False) and args.backend != "vllm":
        raise ValueError("LoRA benchmarking is only supported for vLLM backend")
    if getattr(args, "enable_lora", False) and args.lora_path is None:
        raise ValueError("LoRA path must be provided when enable_lora is True")

    # --data-parallel is not supported currently.
    # https://github.com/vllm-project/vllm/issues/16222
    if args.data_parallel_size > 1:
        raise ValueError(
            "Data parallel is not supported in offline benchmark, "
            "please use benchmark serving instead"
        )


def create_argument_parser():
    parser = FlexibleArgumentParser(description="Benchmark the throughput.")
    parser.add_argument(
        "--backend",
        type=str,
        choices=["vllm", "vllm-chat"],
        default="vllm",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        choices=["sharegpt", "random", "sonnet", "burstgpt", "hf"],
        help="Name of the dataset to benchmark on.",
        default="sharegpt",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Do not load the dataset in streaming mode.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to the ShareGPT dataset, will be deprecated in\
            the next release. The dataset is expected to "
        "be a json in form of list[dict[..., conversations: "
        "list[dict[..., value: <prompt_or_response>]]]]",
    )
    parser.add_argument(
        "--dataset-path", type=str, default=None, help="Path to the dataset"
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=None,
        help="Input prompt length for each request",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the "
        "output length from the dataset.",
    )
    parser.add_argument(
        "--n", type=int, default=1, help="Number of generated sequences per prompt."
    )
    parser.add_argument(
        "--num-prompts", type=int, default=1000, help="Number of prompts to process."
    )
    parser.add_argument(
        "--hf-max-batch-size",
        type=int,
        default=None,
        help="Maximum batch size for HF backend.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save the throughput results in JSON format.",
    )
    parser.add_argument(
        "--async-engine",
        action="store_true",
        default=False,
        help="Use vLLM async engine rather than LLM class.",
    )
    parser.add_argument(
        "--disable-frontend-multiprocessing",
        action="store_true",
        default=False,
        help="Disable decoupled async engine frontend.",
    )
    parser.add_argument(
        "--disable-detokenize",
        action="store_true",
        help=(
            "Do not detokenize the response (i.e. do not include "
            "detokenization time in the measurement)"
        ),
    )
    # LoRA
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="Path to the LoRA adapters to use. This can be an absolute path, "
        "a relative path, or a Hugging Face model identifier.",
    )
    parser.add_argument(
        "--prefix-len",
        type=int,
        default=None,
        help=f"Number of prefix tokens to be used in RandomDataset "
        "and SonnetDataset. For RandomDataset, the total input "
        "length is the sum of prefix-len (default: "
        f"{RandomDataset.DEFAULT_PREFIX_LEN}) and a random context length "
        "sampled from [input_len * (1 - range_ratio), "
        "input_len * (1 + range_ratio)]. For SonnetDataset, "
        f"prefix_len (default: {SonnetDataset.DEFAULT_PREFIX_LEN}) "
        "controls how much of the input is fixed lines versus "
        "random lines, but the total input length remains approximately "
        "input_len tokens.",
    )
    # random dataset
    parser.add_argument(
        "--random-range-ratio",
        type=float,
        default=None,
        help=f"Range ratio (default : {RandomDataset.DEFAULT_RANGE_RATIO}) "
        "for sampling input/output length, "
        "used only for RandomDataset. Must be in the range [0, 1) to "
        "define a symmetric sampling range "
        "[length * (1 - range_ratio), length * (1 + range_ratio)].",
    )

    # hf dataset
    parser.add_argument(
        "--hf-subset", type=str, default=None, help="Subset of the HF dataset."
    )
    parser.add_argument(
        "--hf-split", type=str, default=None, help="Split of the HF dataset."
    )

    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=1,
        help="Number of requests to warm up the model.",
    )

    parser = AsyncEngineArgs.add_cli_args(parser)

    return parser


if __name__ == "__main__":
    parser = create_argument_parser()
    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model
    validate_args(args)
    main(args)
