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

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import vllm.forward_context as vfc
from vllm.config import ParallelConfig, VllmConfig
from vllm.forward_context import (
    DPMetadata,
    batchsize_logging_interval,
    create_forward_context,
    override_forward_context,
    track_batchsize,
)
from vllm.platforms import current_platform

from vllm_rbln import envs
from vllm_rbln.logger import init_logger

logger = init_logger(__name__)


@dataclass
class RBLNDPMetadata(DPMetadata):
    max_pads_across_dp: torch.Tensor | None = None

    @staticmethod
    def make(
        parallel_config: ParallelConfig,
        num_tokens: int,
        num_tokens_across_dp: torch.Tensor | None = None,
        num_padded_tokens: int | None = None,
    ) -> "RBLNDPMetadata":
        dp_size = parallel_config.data_parallel_size

        if dp_size > 1:
            assert num_tokens_across_dp is not None, (
                "num_tokens_across_dp should be applied for DP case"
            )
            assert num_padded_tokens is not None, (
                "num_padded_tokens should be applied for DP case"
            )
            num_tokens_across_dp_cpu = num_tokens_across_dp
            max_pad = num_padded_tokens

            max_pads_across_dp = torch.empty(max_pad, device="cpu")
        else:
            assert num_tokens_across_dp is None, (
                "num_tokens_across_dp should not be applied for non-DP case"
            )
            assert num_padded_tokens is None, (
                "num_padded_tokens should not be applied for non-DP case"
            )
            num_tokens_across_dp_cpu = torch.tensor(
                [num_tokens], device="cpu", dtype=torch.int32
            )
            max_pads_across_dp = None

        return RBLNDPMetadata(
            num_tokens_across_dp_cpu,
            max_pads_across_dp=max_pads_across_dp,
        )


@contextmanager
def set_forward_context(
    attn_metadata: Any,
    vllm_config: VllmConfig,
    num_tokens: int | None = None,
    num_tokens_across_dp: torch.Tensor | None = None,
    num_padded_tokens: int | None = None,
    **kwargs,
):
    """A context manager that stores the current forward context,
    can be attention metadata, etc.
    Here we can inject common logic for every model forward pass.
    """
    need_to_track_batchsize = track_batchsize and attn_metadata is not None
    if need_to_track_batchsize:
        vfc.forward_start_time = time.perf_counter()

    dp_metadata: DPMetadata | None = None
    if (
        vllm_config.parallel_config.data_parallel_size > 1
        or envs.VLLM_RBLN_USE_MOE_TOKENS_MASK
    ) and (attn_metadata is not None or num_tokens is not None):
        dp_metadata = RBLNDPMetadata.make(
            vllm_config.parallel_config,
            num_tokens or 0,
            num_tokens_across_dp,
            num_padded_tokens,
        )

    additional_kwargs = current_platform.set_additional_forward_context(**kwargs)

    forward_context = create_forward_context(
        attn_metadata,
        vllm_config,
        dp_metadata,
        additional_kwargs=additional_kwargs,
    )

    try:
        with override_forward_context(forward_context):
            yield
    finally:
        if need_to_track_batchsize:
            batchsize = num_tokens

            now = time.perf_counter()
            # time measurement is in milliseconds
            vfc.batchsize_forward_time[batchsize].append(
                (now - vfc.forward_start_time) * 1000
            )
            if now - vfc.last_logging_time > batchsize_logging_interval:
                vfc.last_logging_time = now
                forward_stats = []
                for bs, times in vfc.batchsize_forward_time.items():
                    if len(times) <= 1:
                        # can be cudagraph / profiling run
                        continue
                    medium = torch.quantile(torch.tensor(times), q=0.5).item()
                    medium = round(medium, 2)
                    forward_stats.append((bs, len(times), medium))
                forward_stats.sort(key=lambda x: x[1], reverse=True)
                if forward_stats:
                    logger.info(
                        (
                            "Batchsize forward time stats "
                            "(batchsize, count, median_time(ms)): %s"
                        ),
                        forward_stats,
                    )
