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
"""Size the worker response queue from the number of batches in flight.

A KV connector makes ``collective_rpc`` pass ``output_rank=None``, so every worker
enqueues a response each step rather than only the output rank. Under pipeline
parallelism the first stage runs ahead of the last one, and a step's future
resolves only once the output rank has reported for that same step, so the first
stage's responses stay unread for as many steps as the engine keeps in flight.
The upstream ring is a fixed ten slots, which is fewer than that once the pipeline
is deep enough: the first stage blocks in ``acquire_write``, and because
``worker_busy_loop`` is serial it stops draining ``rpc_broadcast_mq``, so the
engine cannot submit the next step either and the pipeline runs shallower than
``pipeline_parallel_size``.

Deriving the depth from ``max_concurrent_batches`` ties it to the quantity that
sets the lag. The lower bound keeps the upstream depth wherever the derived value
would be smaller, so deployments without pipeline parallelism are unchanged.
"""

from vllm.config import VllmConfig
from vllm.distributed.device_communicators.shm_broadcast import Handle, MessageQueue
from vllm.v1.executor.multiproc_executor import WorkerProc

from vllm_rbln.patches import register_patch

UPSTREAM_MAX_CHUNKS = 10


@register_patch(
    target="vllm.v1.executor.multiproc_executor.WorkerProc._init_message_queues",
    reason=(
        "Derive the worker response queue depth from max_concurrent_batches. A KV "
        "connector makes every pipeline stage enqueue a response per step, and the "
        "first stage runs that many steps ahead of the point where the engine reads "
        "them, so the fixed upstream depth leaves it blocked in acquire_write and "
        "the pipeline runs shallower than pipeline_parallel_size."
    ),
    key="vllm_rbln.patches.multiproc_executor.init_message_queues",
    owner_module="vllm_rbln.patches.multiproc_executor",
)
def patched_init_message_queues(
    self: WorkerProc, input_shm_handle: Handle, vllm_config: VllmConfig
) -> None:
    if vllm_config.parallel_config.nnodes_within_dp > 1:
        _init_message_queues_upstream(self, input_shm_handle, vllm_config)
        return

    self.rpc_broadcast_mq = MessageQueue.create_from_handle(
        input_shm_handle, self.worker.rank
    )
    max_chunks = max(UPSTREAM_MAX_CHUNKS, vllm_config.max_concurrent_batches * 2)
    self.worker_response_mq = MessageQueue(1, 1, max_chunks=max_chunks)
    self.peer_response_handles = []


_init_message_queues_upstream = WorkerProc._init_message_queues
