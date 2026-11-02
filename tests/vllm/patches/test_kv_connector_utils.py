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


"""The RBLN transfer topology has to reach upstream's own construction sites.

Upstream builds ``TransferTopology`` inside its NIXL ``register_kv_caches``,
which the host-bounce path delegates to, so the substitution has to be a patch
rather than a call this side owns.
"""

from __future__ import annotations

from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_transfer_topology import (
    RblnTransferTopology,
)
from vllm_rbln.patches.registry import get_registered_patch_descriptors


def test_one_registration_installs_it():
    # Duplicate targets raise, but two registrations of this class against
    # different targets would both apply and the later one would decide what
    # upstream reads.
    assert (
        sum(
            d.replacement is RblnTransferTopology
            for d in get_registered_patch_descriptors()
        )
        == 1
    )


def test_the_name_nixl_binds_later_is_the_patched_one():
    # The target is the module that defines the class, not the one that reads
    # it: upstream's worker binds it with `from ... import`, and what that name
    # ends up holding is the part the registry's identity check cannot see.
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
        base_worker as upstream_nixl,
    )

    assert upstream_nixl.TransferTopology is RblnTransferTopology
