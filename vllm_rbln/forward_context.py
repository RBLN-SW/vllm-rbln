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
from vllm.config import CUDAGraphMode, ParallelConfig, VllmConfig
from vllm.forward_context import (
    BatchDescriptor,
    DPMetadata,
    batchsize_logging_interval,
    create_forward_context,
    override_forward_context,
    track_batchsize,
)
from vllm.v1.worker.ubatch_utils import UBatchSlices

import vllm_rbln.rbln_envs as envs
from vllm_rbln.logger import init_logger

logger = init_logger(__name__)


@dataclass
class RBLNDPMetadata(DPMetadata):
    """Cross-DP shape synchronization for MoE + speculative decoding.

    Motivation
    ----------
    MoE models with data parallelism require every DP rank to step with the
    same ``(batch, query_len)`` input shape: the MoE layers fire cross-rank
    all-reduce / all-to-all for expert dispatch, so any shape divergence
    either hangs the collective or triggers a hot-path recompile.
    Speculative decoding makes per-rank decisions diverge naturally -- a rank
    with drafts wants ``query_len = num_spec_tokens + 1`` while a rank with an
    ngram miss or a KV-block-boundary request would locally want ``1``.

    Key idea
    --------
    1. Bit-packed cross-DP all-reduce (``num_tokens_and_reqs_across_dp``):
       pack ``(is_prefill, num_reqs, num_tokens)`` into one int32 and run a
       single gloo all-reduce on the existing CPU group, so every rank learns
       the per-rank vectors and lifts shape decisions via MAX without extra
       collectives.
    2. Query backfill (see ``RBLNScheduler``) makes each rank's per-request
       query window uniformly ``num_spec_tokens + 1`` *before* communication,
       so the cross-DP MAX only has to resolve the batch dimension -- the
       query dimension is already an agreed invariant.
    """

    max_pads_across_dp: torch.Tensor | None = None

    @staticmethod
    def num_tokens_across_dp(
        num_tokens: int, dp_size: int, dp_rank: int
    ) -> torch.Tensor:
        """
        Gather the num_tokens across all DP ranks and return results in a
        CPU tensor of size dp_size.
        """
        num_tokens_across_dp = [0] * dp_size
        num_tokens_across_dp[dp_rank] = num_tokens
        num_tokens_tensor = torch.tensor(
            num_tokens_across_dp, device="cpu", dtype=torch.int32
        )
        from vllm.distributed.parallel_state import get_dp_group

        dist.all_reduce(num_tokens_tensor, group=get_dp_group().cpu_group)
        return num_tokens_tensor

    @staticmethod
    def num_tokens_and_reqs_across_dp(
        num_tokens: int,
        num_reqs: int,
        dp_size: int,
        dp_rank: int,
        is_prefill: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """All-reduce per-rank (num_tokens, num_reqs, is_prefill) across DP via
        a single bit-packed int32 and split the result back out.

        Bit layout (int32, low to high):
            bits  0..15  num_tokens (max 65535)
            bits 16..29  num_reqs   (max 16383)
            bit  30      is_prefill flag

        Returns:
            num_tokens_across_dp_cpu: per-rank num_tokens (size dp_size).
            num_reqs_across_dp_cpu: per-rank num_reqs (size dp_size), or None
                if any rank is in prefill phase.
        """
        token_bits = 16
        req_bits = 14
        token_mask = (1 << token_bits) - 1
        req_mask_raw = (1 << req_bits) - 1
        req_mask_shifted = req_mask_raw << token_bits
        prefill_flag = 1 << (token_bits + req_bits)

        assert num_tokens <= token_mask, (
            f"num_tokens={num_tokens} exceeds bit-packed limit {token_mask}"
        )
        assert num_reqs <= req_mask_raw, (
            f"num_reqs={num_reqs} exceeds bit-packed limit {req_mask_raw}"
        )

        encoded = num_tokens | (num_reqs << token_bits)
        if is_prefill:
            encoded |= prefill_flag

        encoded_across_dp = RBLNDPMetadata.num_tokens_across_dp(
            encoded, dp_size, dp_rank
        )

        prefill_mask = torch.tensor(
            [prefill_flag] * dp_size, device="cpu", dtype=torch.int32
        )
        any_prefill = bool((encoded_across_dp & prefill_mask).any().item())

        token_mask_t = torch.tensor(
            [token_mask] * dp_size, device="cpu", dtype=torch.int32
        )
        num_tokens_across_dp_cpu = encoded_across_dp & token_mask_t

        if any_prefill:
            num_reqs_across_dp_cpu = None
        else:
            req_mask_t = torch.tensor(
                [req_mask_shifted] * dp_size, device="cpu", dtype=torch.int32
            )
            num_reqs_across_dp_cpu = (encoded_across_dp & req_mask_t) >> token_bits

        return num_tokens_across_dp_cpu, num_reqs_across_dp_cpu

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

            max_tokens_across_dp_cpu = torch.max(num_tokens_across_dp_cpu)
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
            max_tokens_across_dp_cpu = num_tokens
            max_pads_across_dp = None

        return RBLNDPMetadata(
            max_tokens_across_dp_cpu,
            num_tokens_across_dp_cpu,
            max_pads_across_dp=max_pads_across_dp,
        )


@contextmanager
def _set_forward_context(
    attn_metadata: Any,
    vllm_config: VllmConfig,
    num_tokens: int | None = None,
    num_tokens_across_dp: torch.Tensor | None = None,
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    batch_descriptor: BatchDescriptor | None = None,
    ubatch_slices: UBatchSlices | None = None,
    num_padded_tokens: int | None = None,
    additional_kwargs: dict[str, Any] | None = None,
):
    """A context manager that stores the current forward context,
    can be attention metadata, etc.
    Here we can inject common logic for every model forward pass.
    """
    need_to_track_batchsize = track_batchsize and attn_metadata is not None
    if need_to_track_batchsize:
        vfc.forward_start_time = time.perf_counter()

    dp_metadata: DPMetadata | None = None
    enable_dp = vllm_config.parallel_config.data_parallel_size > 1
    use_moe_tokens_mask = envs.VLLM_RBLN_USE_MOE_TOKENS_MASK
    if (enable_dp or use_moe_tokens_mask) and (
        attn_metadata is not None or num_tokens is not None
    ):
        dp_metadata = RBLNDPMetadata.make(
            vllm_config.parallel_config,
            num_tokens or 0,
            num_tokens_across_dp,
            num_padded_tokens,
        )

    # NOTE: vLLM v0.22 removed the virtual_engine parameter
    forward_context = create_forward_context(
        attn_metadata,
        vllm_config,
        dp_metadata=dp_metadata,
        cudagraph_runtime_mode=cudagraph_runtime_mode,
        batch_descriptor=batch_descriptor,
        ubatch_slices=ubatch_slices,
    )
    if additional_kwargs:
        existing_additional_kwargs = getattr(forward_context, "additional_kwargs", None)
        if existing_additional_kwargs is None:
            forward_context.additional_kwargs = dict(additional_kwargs)
        else:
            existing_additional_kwargs.update(additional_kwargs)

    try:
        with override_forward_context(forward_context):
            yield
    finally:
        if need_to_track_batchsize:
            batchsize = num_tokens
            # we use synchronous scheduling right now,
            # adding a sync point here should not affect
            # scheduling of the next batch
            from vllm.platforms import current_platform

            synchronize = current_platform.synchronize
            if synchronize is not None:
                synchronize()
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


vfc.set_forward_context = _set_forward_context

# Importers should prefer this alias so the RBLN-specific kwargs are always
# accepted even if the monkey-patch above is bypassed by import ordering.
set_forward_context = _set_forward_context
