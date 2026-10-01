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


from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_transfer_topology import (
    RblnTransferTopology,
)
from vllm_rbln.patches import register_patch

register_patch(
    target="vllm.distributed.kv_transfer.kv_connector.utils.TransferTopology",
    reason=(
        "TransferTopology has no per-platform way to say how a block is laid "
        "out, and 0.30 dropped the three members that expressed it. RBLN "
        "needs them: the attention cache is K/V-first on rbln_triton_ops and "
        "blocks-first on rbln_custom_ops, so the two cut into regions "
        "differently. "
        "TODO(vllm-rbln): delete once the triton kernels read the packed "
        "layout too and vllm-rbln allocates only that one."
    ),
)(RblnTransferTopology)
