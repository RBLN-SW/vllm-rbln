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

"""Generate CPU sampler inputs, or inspect saved inputs without changing them."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch_rbln  # noqa: F401
import yaml

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
PRESETS = {
    "near_uniform": {"variation": 0.1},
    "head_concentrated": {
        "head_tokens": 32,
        "head_mass": 0.99,
        "variation": 0.1,
    },
    "head_with_broad_tail": {
        "head_tokens": 32,
        "head_mass": 0.5,
        "variation": 0.1,
    },
    "zipf": {"exponent": 1.0},
    "single_dominant": {"dominant_mass": 0.99, "variation": 0.1},
    "gaussian_mixture": {
        "weights": [0.7, 0.3],
        "means": [-2.0, 1.0],
        "stds": [1.5, 2.5],
    },
}
GENERATOR_VERSION = 2
REFERENCE_GENERATOR = "stratified_tail_quantiles"


def _number(name: str, value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _positive_int(name: str, value: Any) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _checksum(logits: torch.Tensor) -> str:
    return hashlib.sha256(
        logits.contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def _probabilities(
    logits: torch.Tensor, temperature: float
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        logits.device.type != "cpu"
        or logits.ndim != 2
        or min(logits.shape) < 1
        or logits.dtype not in DTYPES.values()
        or not torch.isfinite(logits).all()
    ):
        raise ValueError("logits must be a finite, nonempty CPU [B, V] float tensor")
    temperature = _number("temperature", temperature)
    if temperature <= 0:
        raise ValueError("temperature must be positive for distribution inspection")
    divisor = torch.tensor(temperature, dtype=logits.dtype, device="cpu")
    scaled = logits / divisor
    if not torch.isfinite(scaled).all():
        raise ValueError("temperature scaling produced non-finite logits")
    return scaled, scaled.double().softmax(dim=-1)


def generate_logits(
    *,
    batch_size: int,
    vocab_size: int,
    distribution: str,
    dtype: str = "float32",
    parameters: dict[str, Any] | None = None,
    placement: str = "random",
    candidate_count: int | None = None,
    seed: int = 0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return CPU [B, V] logits at temperature 1 and generation metadata.

    Gaussian mixtures generate raw logits; other presets generate log(q).
    Front/back place the highest candidate_count values in the first/last
    vocabulary block. Each row has a seeded order within its two blocks, shared
    by front/back. Random permutes the entire vocabulary. The probability RNG
    and placement RNG are independent; extending B preserves existing rows.
    """
    batch_size = _positive_int("batch_size", batch_size)
    vocab_size = _positive_int("vocab_size", vocab_size)
    if vocab_size < 2:
        raise ValueError("vocab_size must be at least 2")
    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {list(DTYPES)}")
    if distribution not in PRESETS:
        raise ValueError(f"unknown distribution: {distribution}")
    if placement not in ("front", "back", "random"):
        raise ValueError("placement must be front, back, or random")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be an integer in [0, 2**64)")
    params = dict(PRESETS[distribution])
    if parameters is not None:
        if not isinstance(parameters, dict) or parameters.keys() - params.keys():
            raise ValueError(f"parameters for {distribution} must use {list(params)}")
        params.update(parameters)
    if distribution == "gaussian_mixture":
        for name, values in params.items():
            if not isinstance(values, (list, tuple)) or not values:
                raise ValueError(f"{name} must be a nonempty sequence")
            params[name] = [_number(name, value) for value in values]
        if not len(params["weights"]) == len(params["means"]) == len(params["stds"]):
            raise ValueError("weights, means, and stds must have equal lengths")
        if any(weight <= 0 for weight in params["weights"]) or not math.isclose(
            math.fsum(params["weights"]), 1.0, rel_tol=0, abs_tol=1e-9
        ):
            raise ValueError("weights must be positive and sum to 1")
        if any(std <= 0 for std in params["stds"]):
            raise ValueError("stds must be positive")
        mixture_weights = torch.tensor(
            params["weights"], dtype=torch.float64, device="cpu"
        )
        mixture_means = torch.tensor(params["means"], dtype=torch.float64, device="cpu")
        mixture_stds = torch.tensor(params["stds"], dtype=torch.float64, device="cpu")
    else:
        params = {name: _number(name, value) for name, value in params.items()}
    variation = params.get("variation", 0.0)
    if not 0 <= variation < 1:
        raise ValueError("variation must be in [0, 1)")
    head = None
    mass = None
    if distribution in ("head_concentrated", "head_with_broad_tail"):
        raw_head = (parameters or {}).get(
            "head_tokens", PRESETS[distribution]["head_tokens"]
        )
        head = _positive_int("head_tokens", raw_head)
        if not 2 <= head < vocab_size:
            raise ValueError("head_tokens must be in [2, vocab_size)")
        mass = params["head_mass"]
        params["head_tokens"] = head
    elif distribution == "single_dominant":
        head, mass = 1, params["dominant_mass"]
    if head is not None:
        assert mass is not None
        if not head / vocab_size < mass < 1:
            raise ValueError("head mass must be in (head_tokens / vocab_size, 1)")
    if distribution == "zipf" and params["exponent"] <= 0:
        raise ValueError("exponent must be positive")
    if candidate_count is None:
        candidate_count = head
    if candidate_count is not None:
        candidate_count = _positive_int("candidate_count", candidate_count)
        if candidate_count >= vocab_size:
            raise ValueError("candidate_count must be less than vocab_size")
    if placement != "random" and candidate_count is None:
        raise ValueError("front/back require candidate_count for this distribution")

    distribution_rng = torch.Generator(device="cpu").manual_seed(seed)
    placement_seed = (seed + 1) % 2**64
    placement_rng = torch.Generator(device="cpu").manual_seed(placement_seed)
    logits = torch.empty((batch_size, vocab_size), dtype=DTYPES[dtype], device="cpu")
    for row in range(batch_size):
        if distribution == "gaussian_mixture":
            components = torch.multinomial(
                mixture_weights,
                vocab_size,
                replacement=True,
                generator=distribution_rng,
            )
            values = mixture_means[components] + mixture_stds[components] * torch.randn(
                vocab_size,
                dtype=torch.float64,
                device="cpu",
                generator=distribution_rng,
            )
            ranked = values.sort(descending=True, stable=True).values.to(DTYPES[dtype])
        else:
            if distribution == "zipf":
                q = torch.arange(
                    1, vocab_size + 1, dtype=torch.float64, device="cpu"
                ).pow(-params["exponent"])
                q /= q.sum()
            else:
                weights = 1 + variation * (
                    2
                    * torch.rand(
                        vocab_size,
                        dtype=torch.float64,
                        device="cpu",
                        generator=distribution_rng,
                    )
                    - 1
                )
                if head is None:
                    q = weights / weights.sum()
                else:
                    assert mass is not None
                    q = torch.cat(
                        (
                            mass * weights[:head] / weights[:head].sum(),
                            (1 - mass) * weights[head:] / weights[head:].sum(),
                        )
                    )
                    if q[:head].min() <= q[head:].max():
                        raise ValueError(
                            f"row {row}: head does not exceed tail; reduce variation "
                            "or increase head mass"
                        )
            if not torch.isfinite(q).all() or not (q > 0).all():
                raise ValueError("parameters produced non-finite or zero probabilities")
            ranked = q.sort(descending=True, stable=True).values.log().to(DTYPES[dtype])
        if not torch.isfinite(ranked).all():
            raise ValueError("dtype conversion produced non-finite logits")
        if distribution == "near_uniform" and variation > 0 and ranked[0] == ranked[-1]:
            raise ValueError("variation collapsed to uniform after dtype conversion")
        if placement == "random":
            order = torch.randperm(vocab_size, generator=placement_rng, device="cpu")
        else:
            assert candidate_count is not None
            first = torch.randperm(
                candidate_count, generator=placement_rng, device="cpu"
            )
            rest = candidate_count + torch.randperm(
                vocab_size - candidate_count, generator=placement_rng, device="cpu"
            )
            order = (
                torch.cat((first, rest))
                if placement == "front"
                else torch.cat((rest, first))
            )
        logits[row] = ranked[order]
    metadata = {
        "schema_version": 1,
        "generator_version": GENERATOR_VERSION,
        "torch_version": str(torch.__version__),
        "representation": (
            "raw Gaussian mixture logits before dtype conversion"
            if distribution == "gaussian_mixture"
            else "log(q), normalized in float64 at temperature 1 before cast"
        ),
        "config": {
            "batch_size": batch_size,
            "vocab_size": vocab_size,
            "distribution": distribution,
            "dtype": dtype,
            "parameters": params,
            "placement": placement,
            "candidate_count": candidate_count,
            "seed": seed,
        },
        "placement_seed": placement_seed,
        "shape": list(logits.shape),
        "dtype": dtype,
        "sha256": _checksum(logits),
    }
    return logits, metadata


def summarize_logits(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int = 20,
    top_p: float = 0.95,
) -> dict[str, Any]:
    """Describe a CPU reference distribution before top-k/top-p filtering."""
    scaled, probs = _probabilities(logits, temperature)
    top_k = _positive_int("top_k", top_k)
    top_p = _number("top_p", top_p)
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    k = min(top_k, logits.shape[1])
    ranked_probs = probs.sort(dim=-1, descending=True).values
    ranked_logits = scaled.sort(dim=-1, descending=True).values
    cumulative = ranked_probs.cumsum(dim=-1)
    # The reference distribution has unit mass, including round-off at its end.
    cumulative[:, -1] = 1.0
    counts = (cumulative < top_p).sum(dim=-1) + 1
    entropy = -torch.special.xlogy(probs, probs).sum(dim=-1)
    metrics = {
        "p_max": ranked_probs[:, 0],
        "top_k_mass": ranked_probs[:, :k].sum(dim=-1),
        "nucleus_candidates": counts,
        "entropy_nats": entropy,
        "effective_candidates": entropy.exp(),
        "positive_probability_tokens": (probs > 0).sum(dim=-1),
        "unique_scaled_logits": 1
        + (ranked_logits[:, 1:] != ranked_logits[:, :-1]).sum(-1),
        "top_k_boundary_ties": (scaled == ranked_logits[:, k - 1].unsqueeze(-1)).sum(
            -1
        ),
        "top_p_boundary_ties": (
            scaled == ranked_logits.gather(1, (counts - 1).unsqueeze(-1))
        ).sum(-1),
    }
    return {
        "stage": "CPU float64 softmax after logits-dtype temperature division, "
        "before filtering",
        "shape": list(logits.shape),
        "dtype": str(logits.dtype).removeprefix("torch."),
        "temperature": temperature,
        "top_k": k,
        "top_p": top_p,
        "rows": {name: values.tolist() for name, values in metrics.items()},
        "aggregate": {
            name: {
                "min": values.min().item(),
                "mean": values.double().mean().item(),
                "max": values.max().item(),
            }
            for name, values in metrics.items()
        },
    }


def _validate_artifact(logits: torch.Tensor, metadata: dict[str, Any]) -> None:
    _probabilities(logits, 1.0)
    if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
        raise ValueError("unsupported logits schema_version")
    if metadata.get("shape") != list(logits.shape):
        raise ValueError("logits shape does not match metadata")
    if metadata.get("dtype") != str(logits.dtype).removeprefix("torch."):
        raise ValueError("logits dtype does not match metadata")
    if metadata.get("sha256") != _checksum(logits):
        raise ValueError("logits checksum does not match metadata")
    if "rows" in metadata and (
        not isinstance(metadata["rows"], list)
        or len(metadata["rows"]) != logits.shape[0]
        or any(not isinstance(row, dict) for row in metadata["rows"])
    ):
        raise ValueError("metadata must describe every logits row")


def save_logits(
    path: Path,
    logits: torch.Tensor,
    metadata: dict[str, Any],
) -> None:
    """Save the final tensor and provenance, replacing an existing file."""
    path = Path(path)
    _validate_artifact(logits, metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"logits": logits.contiguous(), "metadata": metadata}, path)


def load_logits(path: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    """Load a torch.save Tensor or logits/metadata dictionary on CPU."""
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(artifact, torch.Tensor):
        logits = artifact.detach().contiguous()
        _probabilities(logits, 1.0)
        info = {
            "schema_version": 1,
            "source": "external_tensor",
            "shape": list(logits.shape),
            "dtype": str(logits.dtype).removeprefix("torch."),
            "sha256": _checksum(logits),
        }
    elif (
        isinstance(artifact, dict)
        and set(artifact) == {"logits", "metadata"}
        and isinstance(artifact["logits"], torch.Tensor)
    ):
        logits = artifact["logits"].detach().contiguous()
        info = artifact["metadata"]
        _validate_artifact(logits, info)
    else:
        raise ValueError("expected a Tensor or a dictionary with logits and metadata")
    return logits, info


def generate_from_reference(
    reference: torch.Tensor,
    metadata: dict[str, Any],
    *,
    batch_size: int,
    seed: int = 0,
    head_tokens: int = 256,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Generate rows from fitted tail quantiles and preserved prototype heads.

    Prototype rows cycle in input order. Their token rank placement and histories
    are retained. Generated files embed their fit for reuse without the original.
    """
    _validate_artifact(reference, metadata)
    _positive_int("batch_size", batch_size)
    _positive_int("head_tokens", head_tokens)
    if type(seed) is not int or not 0 <= seed < 2**63 - 1:
        raise ValueError("seed must be an integer in [0, 2**63 - 1)")
    vocab = reference.shape[1]
    if head_tokens >= vocab - 1:
        raise ValueError("head_tokens must leave at least two tail tokens")
    if metadata.get("generator") == REFERENCE_GENERATOR:
        if head_tokens != metadata["config"]["head_tokens"]:
            raise ValueError("head_tokens must match the embedded fit")
        fit = metadata["fit"]
        ranks = torch.tensor(fit["ranks"], dtype=torch.float64, device="cpu")
        knots = torch.tensor(fit["logits"], dtype=torch.float64, device="cpu")
        heads = torch.tensor(fit["head_logits"], dtype=torch.float64, device="cpu")
        token_ids = torch.tensor(fit["token_ids"], dtype=torch.int64, device="cpu")
        reference_checksum = metadata["config"]["reference_sha256"]
        reference_info = metadata["reference"]
    else:
        knot_count = min(1024, vocab - head_tokens)
        ranks = torch.linspace(
            head_tokens, vocab - 1, knot_count, dtype=torch.float64, device="cpu"
        )
        ranked, token_ids = reference.double().sort(
            dim=-1, descending=True, stable=True
        )
        lower = ranks.floor().long()
        upper = ranks.ceil().long()
        knots = ranked[:, lower].lerp(ranked[:, upper], ranks - lower)
        heads = ranked[:, :head_tokens]
        reference_checksum = metadata["sha256"]
        reference_info = {
            key: metadata[key] for key in ("source", "capture") if key in metadata
        }
    source_batch = heads.shape[0]
    knot_count = len(ranks)
    rng = torch.Generator(device="cpu").manual_seed(seed)
    generated = torch.empty((batch_size, vocab), dtype=reference.dtype, device="cpu")
    tail_ranks = torch.arange(head_tokens, vocab, dtype=torch.float64, device="cpu")
    prototype_rows = []
    for row in range(batch_size):
        prototype = row % source_batch
        samples = (
            tail_ranks
            + torch.rand(
                vocab - head_tokens, generator=rng, dtype=torch.float64, device="cpu"
            )
            - 0.5
        )
        samples.clamp_(head_tokens, vocab - 1)
        positions = (
            (samples - head_tokens) * (knot_count - 1) / (vocab - 1 - head_tokens)
        )
        left = positions.floor().long().clamp_max(knot_count - 2)
        values = knots[prototype, left].lerp(
            knots[prototype, left + 1], positions - left
        )
        values = torch.cat((heads[prototype], values))
        generated[row, token_ids[prototype]] = values.to(reference.dtype)
        prototype_rows.append(prototype)
    info = {
        "schema_version": 1,
        "source": "synthetic_calibrated",
        "generator": REFERENCE_GENERATOR,
        "generator_version": GENERATOR_VERSION,
        "torch_version": str(torch.__version__),
        "representation": "raw logits with preserved head and fitted tail quantiles",
        "config": {
            "batch_size": batch_size,
            "vocab_size": vocab,
            "dtype": str(reference.dtype).removeprefix("torch."),
            "seed": seed,
            "head_tokens": head_tokens,
            "tail_knots": knot_count,
            "placement": "prototype token ranks",
            "reference_sha256": reference_checksum,
            "prototype_rows": prototype_rows,
        },
        "fit": {
            "ranks": ranks.tolist(),
            "logits": knots.tolist(),
            "head_logits": heads.tolist(),
            "token_ids": token_ids.tolist(),
        },
        "reference": reference_info,
        "shape": list(generated.shape),
        "dtype": str(generated.dtype).removeprefix("torch."),
        "sha256": _checksum(generated),
    }
    if "rows" in metadata:
        info["rows"] = [
            dict(metadata["rows"][index], prototype_row=index)
            for index in prototype_rows
        ]
        for row in info["rows"]:
            if "sampled_token_id" in row:
                row["prototype_sampled_token_id"] = row.pop("sampled_token_id")
    return generated, info


def plot_logits(
    logits: torch.Tensor,
    path: Path,
    *,
    row: int = 0,
    temperature: float = 1.0,
    top_p: float = 0.95,
) -> None:
    """Save a PNG of one row; summaries cover all rows separately."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    scaled, probs = _probabilities(logits, temperature)
    top_p = _number("top_p", top_p)
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if type(row) is not int or not 0 <= row < logits.shape[0]:
        raise ValueError("plot row is outside the batch")
    values = scaled[row].double().numpy()
    ordered_values = scaled[row].sort().values.double().numpy()
    percentiles = torch.linspace(
        0, 100, logits.shape[1], dtype=torch.float64, device="cpu"
    ).numpy()
    positive_probs = probs[row][probs[row] > 0]
    ranked = probs[row].sort(descending=True).values.numpy()
    ranks = range(1, logits.shape[1] + 1)
    fig = Figure(figsize=(12, 8), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 2)
    axes[0, 0].hist(values, bins=50)
    axes[0, 0].set(
        title="Logit value distribution", xlabel="Scaled logit", ylabel="Token count"
    )
    axes[0, 1].plot(percentiles, ordered_values)
    axes[0, 1].set(title="Logit quantiles", xlabel="Percentile", ylabel="Scaled logit")
    axes[1, 0].plot(ranks, ranked.cumsum())
    axes[1, 0].axhline(
        top_p, color="tab:red", linestyle="--", label=f"Mass = {top_p:g}"
    )
    axes[1, 0].set(xlabel="Rank", ylabel="Cumulative probability", xscale="log")
    axes[1, 0].legend()
    axes[1, 1].hist(positive_probs.log10().numpy(), bins=50)
    axes[1, 1].set(
        title=f"Probability values (zeros: {logits.shape[1] - len(positive_probs)})",
        xlabel="Log10 probability",
        ylabel="Token count",
    )
    fig.suptitle(
        f"Row {row} of {logits.shape[0]} | V={logits.shape[1]} | "
        f"{logits.dtype} | T={temperature}\nCPU reference before filtering"
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        fig.savefig(stream, format="png", dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Use generate --help or inspect --help for options and defaults.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser(
        "generate",
        help="Generate and save CPU logits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Save CPU [B, V] logits and print their distribution summary.\n"
            "Explicit CLI values override one selected YAML input; omitted values\n"
            "keep YAML settings. Defaults below apply when neither supplies a value.\n"
            "Gaussian mixtures generate raw logits. Other presets generate log(q)."
        ),
        epilog="Preset parameter defaults:\n"
        + "\n".join(
            f"  {name}: {json.dumps(params)}" for name, params in PRESETS.items()
        ),
    )
    generate.add_argument(
        "--config",
        type=Path,
        metavar="YAML",
        help="Input definitions with schema_version: 1 and an inputs map "
        "(default: none).",
    )
    generate.add_argument(
        "--input-id",
        metavar="NAME",
        help="Select one YAML input; required for CLI overrides of a multi-input file "
        "(default: generate all configured inputs).",
    )
    generate.add_argument(
        "--reference",
        type=Path,
        metavar="FILE",
        help="Fit a saved .pt input instead of using a distribution preset. "
        "Preserves the head and token rank placement; resamples the tail. "
        "Generated files reuse their embedded fit. "
        "Requires batch size; accepts seed, dtype check and head-tokens "
        "(default: 256).",
    )
    generate.add_argument(
        "--output",
        type=Path,
        metavar="FILE",
        help="PyTorch .pt output with logits and metadata, replacing an existing file "
        "(required in CLI or YAML). "
        "CLI paths use the working directory; YAML paths use the YAML directory.",
    )
    generate.add_argument(
        "--batch-size",
        type=int,
        metavar="B",
        help="Number of independent rows, B >= 1 (required in CLI or YAML). "
        "A saved maximum batch can be sliced for smaller benchmark batches.",
    )
    generate.add_argument(
        "--vocab-size",
        type=int,
        metavar="V",
        help="Tokens per row, V >= 2 (required in CLI or YAML).",
    )
    generate.add_argument(
        "--seed",
        type=int,
        help="Input RNG seed in [0, 2**64); placement uses (seed + 1) modulo 2**64 "
        "(default: 0). Does not seed the sampler.",
    )
    generate.add_argument(
        "--candidate-count",
        type=int,
        metavar="H",
        help="Highest H values placed in the front/back block, 1 <= H < V. "
        "Default: head_tokens for head presets, 1 for single_dominant; otherwise "
        "required with front/back. Does not filter sampling candidates.",
    )
    generate.add_argument(
        "--distribution",
        choices=PRESETS,
        help="Distribution preset (required in CLI or YAML); defaults listed below.",
    )
    generate.add_argument(
        "--dtype",
        choices=DTYPES,
        help="Stored tensor dtype; values are generated in float64, then cast "
        "(default: float32).",
    )
    generate.add_argument(
        "--placement",
        choices=("front", "back", "random"),
        help="Place highest H values first/last, preserving block order between "
        "front/back, or shuffle all V positions (default: random).",
    )
    parameter_help = {
        "variation": "Relative uniform weight variation in [0, 1), default 0.1. "
        "Use 0 for intentional ties; not accepted by zipf or gaussian_mixture.",
        "head-mass": "Head probability mass before casting at T=1, in "
        "(head_tokens / V, 1). Defaults: 0.99 for head_concentrated, "
        "0.5 for head_with_broad_tail.",
        "dominant-mass": "Single-token probability mass before casting at T=1, "
        "in (1 / V, 1) (default: 0.99; single_dominant only).",
        "exponent": "Positive Zipf exponent s in q[r] proportional to r**(-s) "
        "(default: 1.0; zipf only).",
    }
    for name in ("variation", "head-mass", "dominant-mass", "exponent"):
        generate.add_argument(f"--{name}", type=float, help=parameter_help[name])
    generate.add_argument(
        "--head-tokens",
        type=int,
        help="Head size in [2, V), for the two head presets (default: 32).",
    )
    mixture_help = {
        "weights": "Positive component weights that sum to 1",
        "means": "Finite component means in raw logit units",
        "stds": "Positive component standard deviations in raw logit units",
    }
    for name in ("weights", "means", "stds"):
        generate.add_argument(
            f"--{name}",
            type=float,
            nargs="+",
            metavar="VALUE",
            help=f"{mixture_help[name]}; gaussian_mixture only, with equal-length "
            f"weights/means/stds (default: {PRESETS['gaussian_mixture'][name]}).",
        )
    inspect = commands.add_parser(
        "inspect",
        help="Summarize and plot saved logits",
        description="Verify saved logits and print their distribution summary. "
        "The logits file is read without modification.",
    )
    inspect.add_argument(
        "--input",
        type=Path,
        required=True,
        metavar="FILE",
        help="PyTorch .pt Tensor or logits/metadata dictionary (required).",
    )
    for command in (generate, inspect):
        command.add_argument(
            "--summary-top-k",
            type=int,
            default=20,
            metavar="K",
            help="Report the probability mass of the highest min(K, V) tokens, "
            "K >= 1 (default: 20). Statistics only; no filtering.",
        )
        command.add_argument(
            "--summary-top-p",
            type=float,
            default=0.95,
            metavar="P",
            help="Report the minimum ranked candidate count reaching mass P, "
            "0 < P <= 1 (default: 0.95). Also sets the plot's reference line; "
            "does not filter the logits.",
        )
        command.add_argument(
            "--plot",
            action="store_true",
            help="Save a PNG in addition to the summary (default: off). "
            "Imports matplotlib only when enabled; "
            "missing dependency raises ImportError.",
        )
        command.add_argument(
            "--plot-output",
            type=Path,
            metavar="PNG",
            help="PNG output path; requires --plot and replaces an existing image "
            "(default: logits path with its suffix replaced by .png).",
        )
        command.add_argument(
            "--plot-dir",
            type=Path,
            metavar="DIRECTORY",
            help="Save PNG images in a separate directory (default: beside input). "
            "Requires --plot; cannot be combined with --plot-output.",
        )
        command.add_argument(
            "--plot-row",
            type=int,
            default=0,
            metavar="ROW",
            help="Zero-based row to plot, 0 <= ROW < B (default: 0). "
            "The printed summary still covers all rows.",
        )
    args = parser.parse_args()
    if args.plot_output is not None and not args.plot:
        parser.error("--plot-output requires --plot")
    if args.plot_dir is not None and (not args.plot or args.plot_output is not None):
        parser.error("--plot-dir requires --plot and excludes --plot-output")
    if args.command == "inspect":
        entries = {args.input.stem: {"input": args.input}}
    else:
        config = {}
        if args.config is not None:
            config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
            if (
                not isinstance(config, dict)
                or set(config) != {"schema_version", "inputs"}
                or config["schema_version"] != 1
                or not isinstance(config["inputs"], dict)
                or not config["inputs"]
            ):
                parser.error(
                    "config requires schema_version: 1 and a nonempty inputs map"
                )
        if args.input_id is not None and args.config is None:
            parser.error("--input-id requires --config")
        entries = config.get("inputs", {"generated": {}})
        if args.input_id is not None:
            if args.input_id not in entries:
                parser.error(f"unknown input ID: {args.input_id}")
            entries = {args.input_id: entries[args.input_id]}
        option_names = (
            "batch_size",
            "vocab_size",
            "distribution",
            "dtype",
            "placement",
            "candidate_count",
            "seed",
            "output",
            "reference",
        )
        parameter_names = (
            "variation",
            "head_tokens",
            "head_mass",
            "dominant_mass",
            "exponent",
            "weights",
            "means",
            "stds",
        )
        if len(entries) > 1 and any(
            getattr(args, name) is not None
            for name in (*option_names, *parameter_names, "plot_output")
        ):
            parser.error("CLI overrides require selecting a single --input-id")
        entries = {name: dict(entry) for name, entry in entries.items()}
        for entry in entries.values():
            for name in option_names:
                if (value := getattr(args, name)) is not None:
                    entry[name] = value
            parameters = dict(entry.get("parameters", {}))
            for name in parameter_names:
                if (value := getattr(args, name)) is not None:
                    parameters[name] = value
            entry["parameters"] = parameters
            if args.config is not None and args.output is None and "output" in entry:
                entry["output"] = args.config.parent / entry["output"]
            if (
                args.config is not None
                and args.reference is None
                and "reference" in entry
            ):
                entry["reference"] = args.config.parent / entry["reference"]
    for input_id, entry in entries.items():
        if args.command == "generate":
            entry = dict(entry)
            if "output" not in entry:
                parser.error(f"{input_id}: output is required")
            path = Path(entry.pop("output"))
            if "batch_size" not in entry:
                parser.error(f"{input_id}: batch_size is required")
            if "reference" in entry:
                reference, reference_metadata = load_logits(entry.pop("reference"))
                dtype = entry.pop("dtype", None)
                if dtype is not None and dtype != str(reference.dtype).removeprefix(
                    "torch."
                ):
                    parser.error("reference dtype must match --dtype")
                parameters = entry.pop("parameters")
                if set(parameters) - {"head_tokens"} or set(entry) - {
                    "batch_size",
                    "seed",
                }:
                    parser.error(
                        "reference accepts only batch size, seed, dtype and head-tokens"
                    )
                logits, metadata = generate_from_reference(
                    reference, reference_metadata, **entry, **parameters
                )
            else:
                for name in ("batch_size", "vocab_size", "distribution"):
                    if name not in entry:
                        parser.error(f"{input_id}: {name} is required")
                logits, metadata = generate_logits(**entry)
        else:
            path = entry["input"]
            logits, metadata = load_logits(path)
        summary = summarize_logits(
            logits, top_k=args.summary_top_k, top_p=args.summary_top_p
        )
        image_path = args.plot_output or path.with_suffix(".png")
        if args.plot_dir is not None:
            image_path = args.plot_dir / image_path.name
        if args.plot:
            if image_path.resolve() == path.resolve():
                parser.error("plot path must differ from the logits file")
            if not 0 <= args.plot_row < logits.shape[0]:
                parser.error("--plot-row is outside the batch")
        if args.command == "generate":
            save_logits(path, logits, metadata)
        if args.plot:
            plot_logits(logits, image_path, row=args.plot_row, top_p=args.summary_top_p)
        print(
            json.dumps(
                {
                    "input_id": input_id,
                    "file": str(path),
                    "image": str(image_path) if args.plot else None,
                    "generation": {
                        key: value
                        for key, value in metadata.items()
                        if key not in ("fit", "rows")
                    },
                    "summary": {
                        key: value for key, value in summary.items() if key != "rows"
                    },
                },
                indent=2,
                allow_nan=False,
            )
        )


if __name__ == "__main__":
    main()
