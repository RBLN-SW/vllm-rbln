#!/usr/bin/env python3
# Copyright 2026 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Measure one RBLNSampler case, including completion and token transfer to CPU."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import pickle
import platform
import statistics
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace


def write_result(path: Path, result: dict) -> None:
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(payload)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse one case without importing torch or initializing a device."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input-logits",
        type=Path,
        help="PyTorch .pt [B, V] Tensor or logits/metadata dictionary; loaded on CPU.",
    )
    source.add_argument(
        "--distribution", help="Preset from generate_logits.py; generate in memory."
    )
    parser.add_argument(
        "--case-id", default="single", help="Case identifier in the result."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="First B saved rows, or generated row count; defaults to all saved rows.",
    )
    parser.add_argument(
        "--vocab-size", type=int, help="Generated V; required for in-memory generation."
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16", "float16"),
        help="Generated dtype (default: bfloat16); "
        "saved input dtype must match if specified.",
    )
    parser.add_argument(
        "--input-seed",
        type=int,
        help="Generation seed (default: 0); does not seed sampler draws.",
    )
    parser.add_argument(
        "--placement",
        choices=("front", "back", "random"),
        help="Generation placement (default: random).",
    )
    parser.add_argument(
        "--candidate-count",
        type=int,
        help="Highest H generated logits placed in the front/back block.",
    )
    for name in ("variation", "head-mass", "dominant-mass", "exponent"):
        parser.add_argument(
            f"--{name}",
            type=float,
            help="Generation parameter; defaults and constraints "
            "follow generate_logits.py.",
        )
    parser.add_argument(
        "--head-tokens",
        type=int,
        help="Head token count for head presets (default: 32).",
    )
    for name in ("weights", "means", "stds"):
        parser.add_argument(
            f"--{name}",
            type=float,
            nargs="+",
            help="Gaussian mixture component values; "
            "defaults follow generate_logits.py.",
        )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use the dedicated argmax graph; "
        "incompatible with filters or nonzero temperature.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Sampling temperature (default: 1); zero selects greedy.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        help="K in [1, V]; omitted means None, K=V remains a tensor.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        help="P in (0, 1]; omitted means None, P=1 remains a tensor.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Positive repetition penalty; 1 disables penalties. Uses embedded "
        "row histories when present; preset-generated inputs use each row's top "
        "32 token IDs as fixed prompt history. Plain Tensors have no history and "
        "cannot enable penalties. History does not grow across calls.",
    )
    parser.add_argument(
        "--presence-penalty",
        type=float,
        default=0.0,
        help="Penalty in [-2, 2] for tokens present in embedded output history; "
        "0 disables it. Requires metadata rows with token histories.",
    )
    parser.add_argument(
        "--frequency-penalty",
        type=float,
        default=0.0,
        help="Penalty in [-2, 2] multiplied by each token's embedded output "
        "count; 0 disables it. Requires metadata rows with token histories.",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=0.0,
        help="Keep tokens with probability >= min-p times the maximum probability "
        "after temperature scaling; range [0, 1], 0 disables the processor.",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=0,
        help="Mask embedded EOS/stop IDs until this many output tokens exist; "
        "0 disables it. If a row's history length is >= this positive value, "
        "reset its benchmark output history to empty. Records the override; "
        "does not change the saved input or grow history across calls.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Extra warm-up calls, excluding the compile call "
        "and one mandatory preflight call with torch.rbln.explain().",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        required=True,
        help="Measured calls per round; no default until calibrated.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Measurement rounds; total samples = iterations * rounds.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Result JSON path, relative to the working directory if not absolute; "
        "replaces an existing file with mode 0600.",
    )
    args = parser.parse_args(argv)
    for name in (
        "batch_size",
        "vocab_size",
        "candidate_count",
        "top_k",
        "iterations",
        "rounds",
    ):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    if args.temperature is not None and (
        not math.isfinite(args.temperature) or args.temperature < 0
    ):
        parser.error("--temperature must be finite and nonnegative")
    if args.top_p is not None and (
        not math.isfinite(args.top_p) or not 0 < args.top_p <= 1
    ):
        parser.error("--top-p must be in (0, 1]")
    if not math.isfinite(args.repetition_penalty) or args.repetition_penalty <= 0:
        parser.error("--repetition-penalty must be finite and positive")
    for name in ("presence_penalty", "frequency_penalty"):
        value = getattr(args, name)
        if not math.isfinite(value) or not -2 <= value <= 2:
            parser.error(f"--{name.replace('_', '-')} must be in [-2, 2]")
    if not math.isfinite(args.min_p) or not 0 <= args.min_p <= 1:
        parser.error("--min-p must be in [0, 1]")
    if args.min_tokens < 0:
        parser.error("--min-tokens must be nonnegative")
    if args.greedy and args.temperature not in (None, 0):
        parser.error("--greedy conflicts with a nonzero --temperature")
    if (args.greedy or args.temperature == 0) and (
        args.top_k is not None or args.top_p is not None or args.min_p != 0
    ):
        parser.error("greedy sampling does not use --top-k, --top-p or --min-p")
    generation_fields = (
        "vocab_size",
        "input_seed",
        "placement",
        "candidate_count",
        "variation",
        "head_mass",
        "head_tokens",
        "dominant_mass",
        "exponent",
        "weights",
        "means",
        "stds",
    )
    if args.input_logits is not None:
        if not args.input_logits.is_file():
            parser.error(f"input file does not exist: {args.input_logits}")
        if any(getattr(args, name) is not None for name in generation_fields):
            parser.error("generation options cannot be combined with --input-logits")
    else:
        if args.batch_size is None or args.vocab_size is None:
            parser.error("generation requires --batch-size and --vocab-size")
        vocab = args.vocab_size
        if vocab < 2 or (args.top_k is not None and args.top_k > vocab):
            parser.error("V must be >= 2 and --top-k must be <= V")
    return args


def main() -> None:
    args = parse_args()
    result = {
        "schema_version": 1,
        "case_id": args.case_id,
        "status": "running",
        "config": {
            name: str(value) if isinstance(value, Path) else value
            for name, value in vars(args).items()
        },
    }
    try:
        import torch
        from generate_logits import generate_logits, load_logits, summarize_logits

        result["stage"] = "input_validation"
        if args.input_logits is not None:
            original, provenance = load_logits(args.input_logits)
            if args.batch_size is not None and args.batch_size > original.shape[0]:
                raise ValueError(
                    f"requested B={args.batch_size} exceeds saved B={original.shape[0]}"
                )
            if args.dtype is not None and args.dtype != str(
                original.dtype
            ).removeprefix("torch."):
                raise ValueError("--dtype does not match the input")
            original = original[: args.batch_size].contiguous()
        else:
            parameters = {
                name: getattr(args, name)
                for name in (
                    "variation",
                    "head_mass",
                    "head_tokens",
                    "dominant_mass",
                    "exponent",
                    "weights",
                    "means",
                    "stds",
                )
                if getattr(args, name) is not None
            }
            original, provenance = generate_logits(
                batch_size=args.batch_size,
                vocab_size=args.vocab_size,
                dtype=args.dtype or "bfloat16",
                distribution=args.distribution,
                parameters=parameters,
                seed=args.input_seed if args.input_seed is not None else 0,
                placement=args.placement or "random",
                candidate_count=args.candidate_count,
            )
        batch, vocab = original.shape
        if vocab < 2 or (args.top_k is not None and args.top_k > vocab):
            raise ValueError("V must be >= 2 and --top-k must be <= V")
        has_penalty = (
            args.repetition_penalty != 1.0
            or args.presence_penalty != 0
            or args.frequency_penalty != 0
        )
        reference = original
        prompt_ids = [[] for _ in range(batch)]
        output_ids = [[] for _ in range(batch)]
        stop_ids = [[] for _ in range(batch)]
        if has_penalty or args.min_tokens:
            if "rows" in provenance:
                rows = provenance["rows"][:batch]
                if len(rows) != batch:
                    raise ValueError("input metadata is missing row histories")
                if any(
                    not isinstance(row.get(name), list)
                    for row in rows
                    for name in ("prompt_token_ids", "context_generated_token_ids")
                ):
                    raise ValueError(
                        "each row requires prompt and output token histories"
                    )
                prompt_ids = [list(row["prompt_token_ids"]) for row in rows]
                output_ids = [list(row["context_generated_token_ids"]) for row in rows]
                history_source = "captured prompt and preceding generated tokens"
                if args.min_tokens:
                    for index, row in enumerate(rows):
                        ids = row.get("all_stop_token_ids", [])
                        if not isinstance(ids, list):
                            raise ValueError("all_stop_token_ids must be a list")
                        ids = list(ids)
                        if row.get("eos_token_id") is not None:
                            ids.append(row["eos_token_id"])
                        if not ids or any(
                            type(token) is not int or not 0 <= token < vocab
                            for token in ids
                        ):
                            raise ValueError(
                                "min-tokens requires valid EOS/stop IDs in each row"
                            )
                        stop_ids[index] = sorted(set(ids))
            elif (
                "generator_version" in provenance
                and not args.min_tokens
                and args.presence_penalty == 0
                and args.frequency_penalty == 0
            ):
                prompt_ids = original.topk(min(32, vocab), dim=-1).indices.tolist()
                history_source = "top min(32, V) input logits per row as prompt"
            else:
                raise ValueError(
                    "history-dependent sampling requires metadata rows with "
                    "prompt_token_ids and context_generated_token_ids; min-tokens "
                    "also requires EOS/stop IDs. Only repetition penalty supports "
                    "the top-32 prompt fallback for preset-generated inputs"
                )
            if any(
                type(token) is not int or not 0 <= token < vocab
                for ids in (*prompt_ids, *output_ids)
                for token in ids
            ):
                raise ValueError("history contains an invalid token ID")
            original_lengths = list(map(len, output_ids))
            reset_rows = []
            if args.min_tokens:
                for index, ids in enumerate(output_ids):
                    if len(ids) >= args.min_tokens:
                        output_ids[index] = []
                        reset_rows.append(index)
            result["history"] = {
                "source": history_source,
                "prompt_token_ids": prompt_ids,
                "original_output_lengths": original_lengths,
                "output_token_ids": output_ids,
                "effective_output_lengths": list(map(len, output_ids)),
                "min_tokens_reset_rows": reset_rows,
                "min_tokens_reset_rule": "reset to empty when length >= min_tokens"
                if args.min_tokens
                else None,
                "stop_token_ids": stop_ids,
                "fixed_across_calls": True,
            }
        if has_penalty:
            from vllm._custom_ops import apply_repetition_penalties_torch

            prompt_mask = torch.zeros((batch, vocab), dtype=torch.bool)
            output_counts = torch.zeros((batch, vocab), dtype=torch.int64)
            for row in range(batch):
                prompt_mask[row, prompt_ids[row]] = True
                output_counts[row] = torch.bincount(
                    torch.tensor(output_ids[row], dtype=torch.int64), minlength=vocab
                )
            output_mask = output_counts > 0
            reference = original.clone()
            apply_repetition_penalties_torch(
                reference,
                prompt_mask,
                output_mask,
                torch.full((batch,), args.repetition_penalty, dtype=torch.float32),
            )
            reference -= args.frequency_penalty * output_counts.float()
            reference -= args.presence_penalty * output_mask.float()
            result["history"]["changed_logits_per_row"] = (
                (reference != original).sum(-1).tolist()
            )
        greedy = args.greedy or args.temperature == 0
        temperature = (
            0.0 if greedy else (1.0 if args.temperature is None else args.temperature)
        )
        result["input"] = {
            "provenance": provenance,
            "shape": [batch, vocab],
            "dtype": str(original.dtype).removeprefix("torch."),
            "sha256": hashlib.sha256(
                original.view(torch.uint8).numpy().tobytes()
            ).hexdigest(),
            "row_slice": [0, batch],
        }
        result["sampling"] = {
            "greedy": greedy,
            "temperature": temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "max_num_logprobs": None,
            "no_penalties": not has_penalty,
            "repetition_penalty": args.repetition_penalty,
            "presence_penalty": args.presence_penalty,
            "frequency_penalty": args.frequency_penalty,
            "min_p": args.min_p,
            "min_tokens": args.min_tokens,
            "processors": [],
            "generators": {},
        }
        result["input"]["summary"] = summarize_logits(
            reference,
            temperature=1.0 if greedy else temperature,
            top_k=args.top_k if args.top_k is not None else 20,
            top_p=args.top_p if args.top_p is not None else 0.95,
        )
        result["input"]["summary"]["greedy_reference_at_temperature_one"] = greedy
        result["input"]["summary"]["logits_stage"] = (
            "after penalties, before min-tokens/min-p/top-k/top-p"
            if has_penalty
            else "input logits, before min-tokens/min-p/top-k/top-p"
        )
        if args.min_tokens:
            reference = reference.clone()
            for index, ids in enumerate(stop_ids):
                reference[index, ids] = -float("inf")
            if not torch.isfinite(reference).any(dim=-1).all():
                raise ValueError("min-tokens masks the entire vocabulary")
        scaled = (
            reference
            if greedy
            else reference / torch.tensor(temperature, dtype=original.dtype)
        )
        maximum = reference.max(dim=-1).values
        # The op filters logits-dtype probabilities, whose ties can span logits.
        reference_probs = scaled.softmax(dim=-1)
        min_p_allowed = None
        if args.min_p:
            threshold = reference_probs.amax(dim=-1, keepdim=True) * torch.tensor(
                args.min_p, dtype=original.dtype
            )
            min_p_allowed = reference_probs >= threshold
            reference_probs = scaled.masked_fill(~min_p_allowed, -float("inf")).softmax(
                dim=-1
            )
        boundary = (
            reference_probs.topk(args.top_k, dim=-1).values[:, -1]
            if args.top_k is not None
            else None
        )

        result["stage"] = "device_initialization"
        from vllm.v1.sample.logits_processor import LogitsProcessors
        from vllm.v1.sample.metadata import SamplingMetadata

        from vllm_rbln import envs
        from vllm_rbln.compilation import set_compile_stage
        from vllm_rbln.config import resolve_model_impl
        from vllm_rbln.platform import _apply_model_impl, use_device_tensor

        _apply_model_impl(resolve_model_impl(model_impl="vllm"))
        if not use_device_tensor():
            raise ValueError(
                "device tensors are required; check VLLM_RBLN_USE_DEVICE_TENSOR"
            )
        if envs.VLLM_RBLN_COMPILE_ONLY or torch.rbln.is_dummy_device():
            raise ValueError(
                "compile-only and dummy-device execution cannot measure NPU performance"
            )
        from vllm_rbln.v1.sample.rbln_sampler import RBLNSampler

        device = torch.device("rbln:0")
        torch.rbln.set_device(device)
        result["environment"] = {
            "device": str(device),
            "chip": torch.rbln.get_device_name(device),
            "device_properties": str(torch.rbln.get_device_properties(device)),
            "cpu": platform.processor(),
            "cpu_count": os.cpu_count(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "variables": {
                key: value
                for key, value in os.environ.items()
                if key.startswith(("VLLM_RBLN_", "RBLN_"))
                or key
                in (
                    "VLLM_DISABLE_COMPILE_CACHE",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                )
            },
            "python": platform.python_version(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in (
                    "torch",
                    "vllm",
                    "vllm-rbln",
                    "torch-rbln",
                    "rebel-compiler",
                )
            },
            "software_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[5],
                text=True,
            ).strip(),
            "source_sha256": {
                name: hashlib.sha256(
                    Path(__file__).with_name(name).read_bytes()
                ).hexdigest()
                for name in ("benchmark_sampler.py", "generate_logits.py")
            },
            "pid": os.getpid(),
        }
        with torch.inference_mode():
            pristine = original.to(device)
            work = torch.empty_like(pristine)
            logitsprocs = LogitsProcessors()
            if args.min_p or args.min_tokens:
                from vllm import SamplingParams
                from vllm.v1.sample.logits_processor import BatchUpdate

                from vllm_rbln.v1.sample.rbln_logits_processor import (
                    build_rbln_logitsprocs,
                )

                # Builtins need only batch capacity and speculative-mode status;
                # constructing VllmConfig would invoke model configuration hooks.
                processor_config = SimpleNamespace(
                    scheduler_config=SimpleNamespace(max_num_seqs=batch),
                    speculative_config=None,
                )
                logitsprocs = build_rbln_logitsprocs(
                    processor_config,
                    device,
                    is_pin_memory=False,
                    is_pooling_model=False,
                )
                update = BatchUpdate(
                    batch_size=batch,
                    removed=[],
                    moved=[],
                    added=[
                        (
                            index,
                            SamplingParams(
                                temperature=temperature,
                                min_p=args.min_p,
                                min_tokens=args.min_tokens,
                                max_tokens=max(16, args.min_tokens),
                                stop_token_ids=stop_ids[index],
                            ),
                            prompt_ids[index],
                            output_ids[index],
                        )
                        for index in range(batch)
                    ],
                )
                for processor in logitsprocs.all:
                    processor.update_state(update)
                result["sampling"]["processors"] = [
                    type(processor).__name__ for processor in logitsprocs.all
                ]
            metadata = SamplingMetadata(
                temperature=None
                if greedy
                else torch.full(
                    (batch,), temperature, dtype=torch.float32, device=device
                ),
                all_greedy=greedy,
                all_random=not greedy,
                top_k=None
                if args.top_k is None
                else torch.full((batch,), args.top_k, dtype=torch.int32, device=device),
                top_p=None
                if args.top_p is None
                else torch.full(
                    (batch,), args.top_p, dtype=torch.float32, device=device
                ),
                generators={},
                max_num_logprobs=None,
                no_penalties=not has_penalty,
                prompt_token_ids=torch.tensor(
                    [
                        ids + [vocab] * (max(map(len, prompt_ids)) - len(ids))
                        for ids in prompt_ids
                    ],
                    dtype=torch.int64,
                    device=device,
                )
                if has_penalty
                else None,
                frequency_penalties=torch.full(
                    (batch if has_penalty else 0,),
                    args.frequency_penalty,
                    dtype=torch.float32,
                    device=device,
                ),
                presence_penalties=torch.full(
                    (batch if has_penalty else 0,),
                    args.presence_penalty,
                    dtype=torch.float32,
                    device=device,
                ),
                repetition_penalties=torch.full(
                    (batch if has_penalty else 0,),
                    args.repetition_penalty,
                    dtype=torch.float32,
                    device=device,
                ),
                output_token_ids=output_ids,
                allowed_token_ids_mask=None,
                bad_words_token_ids={},
                logitsprocs=logitsprocs,
            )
            result["metadata"] = {
                "all_greedy": metadata.all_greedy,
                "all_random": metadata.all_random,
                "no_penalties": metadata.no_penalties,
                **{
                    name: None
                    if getattr(metadata, name) is None
                    else {
                        "values": getattr(metadata, name).cpu().tolist(),
                        "dtype": str(getattr(metadata, name).dtype),
                        "device": str(getattr(metadata, name).device),
                    }
                    for name in (
                        "temperature",
                        "top_k",
                        "top_p",
                        "repetition_penalties",
                        "presence_penalties",
                        "frequency_penalties",
                    )
                },
                "processors": [
                    {
                        "name": type(processor).__name__,
                        **(
                            {"min_p": processor.min_p.cpu().flatten().tolist()}
                            if hasattr(processor, "min_p") and args.min_p
                            else {}
                        ),
                        **(
                            {"active_min_tokens_rows": sorted(processor.min_toks)}
                            if hasattr(processor, "min_toks")
                            else {}
                        ),
                    }
                    for processor in logitsprocs.all
                ],
            }
            sampler = RBLNSampler()

            def restore() -> None:
                work.copy_(pristine)
                torch.rbln.synchronize(device)

            def invoke():
                output = sampler(work, metadata)
                tokens = output.sampled_token_ids.to("cpu")
                torch.rbln.synchronize(device)
                return output, tokens

            def validate(output, tokens) -> None:
                if (
                    output.sampled_token_ids.device.type != "rbln"
                    or output.logprobs_tensors is not None
                ):
                    raise ValueError("expected device tokens with logprobs disabled")
                if tokens.shape != (batch, 1) or tokens.dtype not in (
                    torch.int32,
                    torch.int64,
                ):
                    raise ValueError("invalid sampled token shape or dtype")
                ids = tokens[:, 0].long()
                if not ((ids >= 0) & (ids < vocab)).all():
                    raise ValueError("sampled token is outside the vocabulary")
                if greedy and not torch.equal(
                    reference[torch.arange(batch), ids], maximum
                ):
                    raise ValueError("greedy output is not a maximum-logit token")
                if args.min_tokens and any(
                    token in stop_ids[index] for index, token in enumerate(ids.tolist())
                ):
                    raise ValueError("sampled a stop token masked by min-tokens")
                if (
                    min_p_allowed is not None
                    and not min_p_allowed[torch.arange(batch), ids].all()
                ):
                    raise ValueError("sampled token is outside the CPU min-p reference")
                if (
                    boundary is not None
                    and not (
                        reference_probs[torch.arange(batch), ids] >= boundary
                    ).all()
                ):
                    raise ValueError("sampled token is outside the CPU top-k reference")

            result["stage"] = "compile"
            restore()
            started = time.perf_counter()
            with set_compile_stage("warmup"), torch.rbln.capture_programs() as compiled:
                output, tokens = invoke()
            result["compile_and_first_call_seconds"] = time.perf_counter() - started
            if not compiled or any(
                program.runtime is None or str(program.device) != str(device)
                for program in compiled
            ):
                raise RuntimeError(
                    "warm-up did not produce a real compiled program "
                    "on the requested NPU"
                )
            result["compiled_programs"] = [
                {
                    "name": program.name,
                    "device": str(program.device),
                    "inputs": [
                        {
                            "name": spec.name,
                            "shape": list(spec.shape),
                            "dtype": str(spec.dtype),
                        }
                        for spec in program.input_specs
                    ],
                    "outputs": [
                        {"shape": list(spec.shape), "dtype": str(spec.dtype)}
                        for spec in program.output_specs
                    ],
                }
                for program in compiled
            ]
            result["stage"] = "output_validation"
            validate(output, tokens)
            result["stage"] = "warmup"
            with (
                set_compile_stage("warmup"),
                torch.rbln.capture_programs() as additional,
            ):
                for _ in range(args.warmup):
                    restore()
                    output, tokens = invoke()
                    result["stage"] = "output_validation"
                    validate(output, tokens)
                    result["stage"] = "warmup"
                result["stage"] = "explain"
                restore()
                with torch.rbln.explain() as explanation:
                    output, tokens = invoke()
                result["explain"] = {
                    "phase": "post-warmup preflight",
                    "call_count": 1,
                    "included_in_measurement": False,
                    "scope": "sampler call through result completion on CPU; "
                    "excludes logits restoration and output validation",
                    "report": explanation.report(),
                    "data": explanation.dump(),
                }
                result["stage"] = "output_validation"
                validate(output, tokens)
            result["stage"] = "warmup"
            if additional:
                raise RuntimeError(
                    "additional compilation occurred after the first warm-up call"
                )
            result["warmup"] = {
                "compile_calls": 1,
                "additional_calls": args.warmup,
                "preflight_calls": 1,
            }

            durations = []
            result["stage"] = "measurement"
            with torch.rbln.capture_programs() as unexpected:
                for _ in range(args.rounds):
                    samples = []
                    for _ in range(args.iterations):
                        restore()
                        started_ns = time.perf_counter_ns()
                        output, tokens = invoke()
                        elapsed_ns = time.perf_counter_ns() - started_ns
                        result["stage"] = "output_validation"
                        validate(output, tokens)
                        result["stage"] = "measurement"
                        if unexpected:
                            raise RuntimeError(
                                "compilation occurred during measurement"
                            )
                        samples.append(elapsed_ns)
                    durations.append(samples)
            flat = [sample for samples in durations for sample in samples]
            ordered = sorted(flat)
            medians = [statistics.median(samples) for samples in durations]
            result["measurement"] = {
                "metric": "sampler call through result completion on CPU",
                "includes": [
                    "RBLNSampler.forward",
                    "sampled token transfer to CPU",
                    "device completion wait",
                ],
                "excludes": [
                    "input generation",
                    "logits restoration and its completion wait",
                    "compilation",
                    "explained preflight call",
                    "output validation",
                ],
                "sample_count": len(flat),
                "unit": "ns",
                "median": statistics.median(flat),
                "p90": ordered[math.ceil(0.9 * len(ordered)) - 1],
                "p90_method": "nearest rank",
                "round_medians": medians,
                "round_median_min": min(medians),
                "round_median_max": max(medians),
                "round_median_cv": statistics.pstdev(medians)
                / statistics.mean(medians),
                "samples_by_round": durations,
            }
            result["validation"] = {
                "status": "passed",
                "checks": [
                    "shape",
                    "integer dtype",
                    "token range",
                    "greedy maximum with ties",
                    "CPU reference top-k probability boundary with ties",
                    *(["CPU min-p support"] if args.min_p else []),
                    *(["min-tokens stop-token exclusion"] if args.min_tokens else []),
                ],
                "top_p_membership_checked": False,
                "additional_compilations": 0,
            }
            result["stage"] = "complete"
    except (
        ValueError,
        RuntimeError,
        OSError,
        AssertionError,
        EOFError,
        pickle.UnpicklingError,
    ) as error:
        result["status"] = (
            "validation_failed"
            if result.get("stage") in ("input_validation", "output_validation")
            else "execution_failed"
        )
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        write_result(args.output, result)
        raise
    result["status"] = "passed"
    write_result(args.output, result)
    print(
        f"{args.case_id}: passed; {result['measurement']['sample_count']} samples; "
        f"result: {args.output}"
    )
    measurement = result["measurement"]
    print(
        "Latency including CPU transfer: "
        f"median={measurement['median'] / 1_000_000:.6f} ms, "
        f"p90={measurement['p90'] / 1_000_000:.6f} ms; "
        f"round median CV={measurement['round_median_cv'] * 100:.2f}%"
    )
    print(
        "torch.rbln.explain(): one post-warmup preflight call; "
        "excluded from latency samples."
    )
    print(result["explain"]["report"])


if __name__ == "__main__":
    main()
