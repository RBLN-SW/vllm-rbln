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

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
    NixlPullConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlConnectorMetadata,
)
from vllm.v1.core.sched.output import SchedulerOutput

from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.base_scheduler import (
    RblnNixlSchedulerBase,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.metadata import (
    RblnNixlConnectorMetadata,
    connector_option,
)


class RblnNixlPullConnectorScheduler(RblnNixlSchedulerBase, NixlPullConnectorScheduler):
    """Scheduler side of the read path."""

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        # Upstream reads `remote_num_tokens` once for the match length and then
        # drops it. Snapshot it ahead of `super()`, which clears the requests it
        # came with. Zero is how a producer that kept no blocks reports itself.
        chunked = connector_option(self.vllm_config, "chunk_mode", False)
        valid_tokens = {
            req_id: req.kv_transfer_params["remote_num_tokens"]
            for req_id, (req, _) in self._reqs_need_recv.items()
            if chunked
            and req.kv_transfer_params
            and req.kv_transfer_params.get("remote_num_tokens")
        }
        base_meta = super().build_connector_meta(scheduler_output)
        assert isinstance(base_meta, NixlConnectorMetadata)
        meta = RblnNixlConnectorMetadata.promote(base_meta)
        meta.valid_tokens = valid_tokens
        return meta
