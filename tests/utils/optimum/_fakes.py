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

"""Lightweight fakes for VllmConfig / hf_config used by converter tests.

The converter modules under test only touch a handful of fields on these
configs, so we avoid importing real vLLM dataclasses (slow import, strict
validation) and instead build SimpleNamespace stubs.
"""

from types import SimpleNamespace
from typing import Any


def make_hf_config(
    architectures: list[str] | None = None,
    **extra: Any,
) -> SimpleNamespace:
    """Build a minimal HF-style PretrainedConfig stub.

    ``architectures`` drives the ``is_*_arch`` predicates in
    :mod:`vllm_rbln.utils.optimum.registry`; everything else can be
    attached via kwargs (e.g. ``max_source_positions``, ``max_length``).
    """
    return SimpleNamespace(
        architectures=architectures if architectures is not None else [],
        **extra,
    )


def make_vllm_config(
    *,
    model: str = "facebook/opt-125m",
    max_model_len: int = 1024,
    runner_type: str = "generate",
    hf_config: SimpleNamespace | None = None,
    max_num_seqs: int = 4,
    max_num_batched_tokens: int = 1024,
    block_size: int = 128,
    enable_prefix_caching: bool = False,
    user_specified_block_size: bool = False,
    num_gpu_blocks: int | None = None,
    num_gpu_blocks_override: int | None = None,
    additional_config: dict[str, Any] | None = None,
    ec_transfer_config: Any = None,
) -> SimpleNamespace:
    """Build a minimal VllmConfig stub.

    All arguments are keyword-only with defaults that satisfy a generic
    decoder-only model. Override per-test as needed.
    """
    if hf_config is None:
        hf_config = make_hf_config(architectures=["LlamaForCausalLM"])

    model_config = SimpleNamespace(
        model=model,
        max_model_len=max_model_len,
        hf_config=hf_config,
        runner_type=runner_type,
    )
    scheduler_config = SimpleNamespace(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
    )
    cache_config = SimpleNamespace(
        block_size=block_size,
        enable_prefix_caching=enable_prefix_caching,
        user_specified_block_size=user_specified_block_size,
        num_gpu_blocks=num_gpu_blocks,
        num_gpu_blocks_override=num_gpu_blocks_override,
    )
    return SimpleNamespace(
        model_config=model_config,
        scheduler_config=scheduler_config,
        cache_config=cache_config,
        additional_config=additional_config if additional_config is not None else {},
        ec_transfer_config=ec_transfer_config,
    )
