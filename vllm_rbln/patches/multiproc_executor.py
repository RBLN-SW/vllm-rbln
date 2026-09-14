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
"""Size the worker response queue from the responses a pipeline keeps unread.

A KV connector makes `collective_rpc` pass `output_rank=None`, so every worker
answers each RPC instead of only the output rank. Under pipeline parallelism the
first stage's answers stay unread until the output rank reports for the same
step, so `max_concurrent_batches * RPCS_PER_BATCH` of them are outstanding at
once. Upstream's fixed depth is smaller than that once the pipeline is deep
enough, and the first stage then blocks in `acquire_write` instead of taking its
next batch.
"""

from vllm.config import VllmConfig
from vllm.distributed.device_communicators.shm_broadcast import Handle, MessageQueue
from vllm.v1.executor.multiproc_executor import WorkerProc

from vllm_rbln.patches import register_patch

_init_message_queues_upstream = WorkerProc._init_message_queues

# `step_with_batch_queue` issues `execute_model` and then `sample_tokens` per
# step, and a KV connector routes both to every rank.
RPCS_PER_BATCH = 2


@register_patch(
    target="vllm.v1.executor.multiproc_executor.WorkerProc._init_message_queues",
    reason=(
        "Grow the worker response queue when a deep pipeline keeps more answers "
        "outstanding than upstream's fixed depth holds. A KV connector routes "
        "every RPC to every rank, and the first stage's answers stay unread until "
        "the output rank reports for the same step, so it blocks in acquire_write "
        "and the pipeline runs shallower than pipeline_parallel_size."
    ),
    key="vllm_rbln.patches.multiproc_executor.init_message_queues",
    owner_module="vllm_rbln.patches.multiproc_executor",
)
def patched_init_message_queues(
    self: WorkerProc, input_shm_handle: Handle, vllm_config: VllmConfig
) -> None:
    _init_message_queues_upstream(self, input_shm_handle, vllm_config)

    if vllm_config.parallel_config.nnodes_within_dp > 1:
        return

    max_chunks = vllm_config.max_concurrent_batches * RPCS_PER_BATCH
    if max_chunks > self.worker_response_mq.buffer.max_chunks:
        self.worker_response_mq = MessageQueue(1, 1, max_chunks=max_chunks)
