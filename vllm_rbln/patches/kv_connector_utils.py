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
        "0.26 asserts a blocks-first, K/V-packed cache in "
        "TransferTopology.__post_init__; RBLN's attention cache is K/V-first "
        "and fails it, so those models cannot register at all. Upstream has "
        "no per-platform way to say the layout differs, and it builds the "
        "class inside its own register_kv_caches, which the host-bounce path "
        "delegates to."
    ),
)(RblnTransferTopology)
