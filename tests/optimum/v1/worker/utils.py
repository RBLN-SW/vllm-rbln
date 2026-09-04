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

import os
from types import SimpleNamespace

import torch
import torch.nn as nn
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_rbln.model_executor.models.optimum.base import ModelInputForRBLN
from vllm_rbln.model_executor.models.optimum.model_base import (
    RBLNOptimumDecoderMixin,
)
from vllm_rbln.v1.core.optimum_scheduler import RBLNSchedulerOutput
from vllm_rbln.v1.worker.optimum_model_runner import RBLNOptimumModelRunner

MAX_NUM_SEQ = 2
MAX_MODEL_LEN = 64
OB_SIZE = 16
IB_SIZE = 4
NUM_BLOCKS = MAX_MODEL_LEN // OB_SIZE * MAX_NUM_SEQ + 1


class MockModelWrapper(nn.Module, RBLNOptimumDecoderMixin):
    """Stands in for an optimum decoder model: the runner reads the decode
    batch layout (decoder_batch_size, the bucket ladder, decode_batch_rows)
    from the real mixin methods; forward is faked per test."""

    class MockModel:
        def __init__(
            self, dtype: torch.dtype, decoder_batch_sizes: tuple[int, ...] | None
        ):
            self.rbln_config = SimpleNamespace(
                use_multiple_decoder=decoder_batch_sizes is not None, dtype=dtype
            )
            self.kv_block_adapter = SimpleNamespace(
                get_available_num_blocks=lambda: NUM_BLOCKS
            )

    def __init__(
        self,
        max_num_seqs: int,
        dtype: torch.dtype = torch.float32,
        decoder_batch_sizes: tuple[int, ...] | None = None,
    ):
        super().__init__()
        self.model = self.MockModel(dtype, decoder_batch_sizes)
        self.dtype = self.model.rbln_config.dtype
        self.decoder_batch_size = max_num_seqs
        self.use_multiple_decoder = decoder_batch_sizes is not None
        if decoder_batch_sizes is not None:
            # Ascending, like RBLNOptimumDecoderMixin.setup_decoder_mixin
            # stores the compiled decoder batch sizes.
            self.decoder_batch_sizes = decoder_batch_sizes

    def compute_logits(
        self, hidden_states: torch.Tensor, sampling_metadata: SamplingMetadata
    ) -> torch.Tensor:
        return hidden_states


def fake_load_model(
    runner: RBLNOptimumModelRunner,
    decoder_batch_sizes: tuple[int, ...] | None = None,
):
    model_dtype = runner.model_config.dtype

    def fake_forward(model_input: ModelInputForRBLN, **kwargs) -> torch.Tensor:
        current_num_reqs = runner.input_batch.num_reqs
        current_vocab_size = runner.model_config.get_vocab_size()

        return torch.randn(
            (current_num_reqs, 1, current_vocab_size),
            dtype=model_dtype,
            device=runner.device,
        )

    runner.model = MockModelWrapper(
        max_num_seqs=runner.scheduler_config.max_num_seqs,
        dtype=model_dtype,
        decoder_batch_sizes=decoder_batch_sizes,
    )
    runner.available_blocks = torch.arange(NUM_BLOCKS, dtype=torch.int16)
    runner.use_optimum_lora = False
    # Assign the fake forward function to the model
    runner.model.forward = fake_forward
    if runner.use_rbln_sampler:
        runner.prepare_rbln_sampler()
    warm_up = os.environ.get("VLLM_RBLN_ENABLE_WARM_UP", "False").lower() in [
        "true",
        "1",
    ]
    if warm_up:
        runner.dummy_sampler_run()


def _schedule_new_request(
    *req_ids: str,
    block_ids: tuple[list[int], ...],
    outer_block_ids: list[int],
    new_computed_tokens: int = 0,
    token_ids: list[int] | None = None,
    finished_req_ids: list[str] | None = None,
    new_computed_blocks: list[int] | None = None,
    preempted_req_ids: list[str] | None = None,
) -> RBLNSchedulerOutput:
    new_reqs = []
    num_scheduled_tokens = {}
    total_num_scheduled_tokens = 0
    if token_ids is None:
        token_ids = [1, 2, 3]
    outer_block_ids = torch.tensor(outer_block_ids)
    for req_id in req_ids:
        new_reqs.append(
            NewRequestData(
                req_id=req_id,
                prompt_token_ids=token_ids,
                mm_features=[],
                sampling_params=SamplingParams(),
                pooling_params=None,
                block_ids=block_ids,
                num_computed_tokens=new_computed_tokens,
                lora_request=None,
            )
        )
        num_scheduled_tokens[req_id] = len(token_ids)
        total_num_scheduled_tokens += num_scheduled_tokens[req_id]

    return RBLNSchedulerOutput(
        scheduled_new_reqs=new_reqs,
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=total_num_scheduled_tokens,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=0,
        finished_req_ids=set(finished_req_ids) if finished_req_ids else set(),
        free_encoder_mm_hashes=[],
        block_table_dict={req_id: outer_block_ids},
        cached_block_table=[],
        cached_length=[],
        dummy_block=None,
        cache_slot_id_dict={req_id: i for i, req_id in enumerate(req_ids)},
    )
