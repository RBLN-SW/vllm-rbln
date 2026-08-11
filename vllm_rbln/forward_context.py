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
import torch.distributed as dist
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


# Bit layout of the packed DP all_reduce payload (int32, low to high):
#   0..15 num_tokens (max 65535), 16..29 num_reqs (max 16383),
#   30 is_prefill flag
_DP_TOKEN_BITS = 16
_DP_REQ_BITS = 14
_DP_TOKEN_MASK = (1 << _DP_TOKEN_BITS) - 1
_DP_REQ_MASK_RAW = (1 << _DP_REQ_BITS) - 1
_DP_REQ_MASK_SHIFTED = _DP_REQ_MASK_RAW << _DP_TOKEN_BITS
_DP_PREFILL_FLAG = 1 << (_DP_TOKEN_BITS + _DP_REQ_BITS)


class DPTokensReduceHandle:
    """A DP num_tokens/num_reqs all_reduce, possibly still in flight.

    Issuing is split from consuming (`wait`) so the caller can run host input prep
    - and leave device work in flight - while the gloo collective progresses on the
    process group's own thread.
    """

    def __init__(self, encoded_across_dp: torch.Tensor, dp_size: int, work: Any | None):
        self._encoded_across_dp = encoded_across_dp
        self._dp_size = dp_size
        self._work = work

    def wait(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self._work is not None:
            self._work.wait()
            self._work = None

        encoded_across_dp = self._encoded_across_dp
        dp_size = self._dp_size

        prefill_mask = torch.tensor(
            [_DP_PREFILL_FLAG] * dp_size, device="cpu", dtype=torch.int32
        )
        any_prefill = bool((encoded_across_dp & prefill_mask).any().item())

        token_mask_t = torch.tensor(
            [_DP_TOKEN_MASK] * dp_size, device="cpu", dtype=torch.int32
        )
        num_tokens_across_dp_cpu = encoded_across_dp & token_mask_t

        if any_prefill:
            num_reqs_across_dp_cpu = None
        else:
            req_mask_t = torch.tensor(
                [_DP_REQ_MASK_SHIFTED] * dp_size, device="cpu", dtype=torch.int32
            )
            num_reqs_across_dp_cpu = (encoded_across_dp & req_mask_t) >> _DP_TOKEN_BITS

        return num_tokens_across_dp_cpu, num_reqs_across_dp_cpu


@dataclass
class RBLNDPMetadata(DPMetadata):
    max_pads_across_dp: torch.Tensor | None = None

    @staticmethod
    def start_num_tokens_and_reqs_across_dp(
        num_tokens: int,
        num_reqs: int,
        dp_size: int,
        dp_rank: int,
        is_prefill: bool,
        async_op: bool = False,
    ) -> DPTokensReduceHandle:
        """Issue the all-reduce of per-rank (num_tokens, num_reqs, is_prefill)
        across DP as a single bit-packed int32.

        Returns a handle; call `wait()` on it to get
        (num_tokens_across_dp_cpu, num_reqs_across_dp_cpu) - the latter is None
        if any rank is in prefill phase.
        """
        assert num_tokens <= _DP_TOKEN_MASK, (
            f"num_tokens={num_tokens} exceeds bit-packed limit {_DP_TOKEN_MASK}"
        )
        assert num_reqs <= _DP_REQ_MASK_RAW, (
            f"num_reqs={num_reqs} exceeds bit-packed limit {_DP_REQ_MASK_RAW}"
        )

        encoded = num_tokens | (num_reqs << _DP_TOKEN_BITS)
        if is_prefill:
            encoded |= _DP_PREFILL_FLAG

        num_tokens_across_dp = [0] * dp_size
        num_tokens_across_dp[dp_rank] = encoded
        encoded_across_dp = torch.tensor(
            num_tokens_across_dp, device="cpu", dtype=torch.int32
        )
        from vllm.distributed.parallel_state import get_dp_group

        work = dist.all_reduce(
            encoded_across_dp, group=get_dp_group().cpu_group, async_op=async_op
        )
        return DPTokensReduceHandle(
            encoded_across_dp, dp_size, work if async_op else None
        )

    @staticmethod
    def num_tokens_and_reqs_across_dp(
        num_tokens: int, num_reqs: int, dp_size: int, dp_rank: int, is_prefill: bool
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Blocking form of start_num_tokens_and_reqs_across_dp."""
        return RBLNDPMetadata.start_num_tokens_and_reqs_across_dp(
            num_tokens, num_reqs, dp_size, dp_rank, is_prefill, async_op=False
        ).wait()

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
