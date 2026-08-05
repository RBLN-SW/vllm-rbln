# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Guard the vLLM KV-connector preemption contract RBLNModelRunner relies on.

``RBLNModelRunner.execute_model`` forwards
``scheduler_output.kv_connector_metadata`` to
``get_kv_transfer_group().handle_preemptions(...)`` (upstream gpu_model_runner
parity). Under the MultiConnector setup RBLN uses, that argument is a
``MultiKVConnectorMetadata``.

This pins the base-class contract that makes that call valid, so a breaking
change to vLLM's ``MultiConnector.handle_preemptions`` (removed / re-typed /
different dispatch) is caught here. Whether *our* call site passes the right
type is left to mypy; this covers the runtime dispatch contract mypy can't see
-- MultiConnector narrows to ``MultiKVConnectorMetadata`` in its body, below the
declared ``KVConnectorMetadata`` signature. Hardware-free (no NIXL / serve).
"""

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
    MultiKVConnectorMetadata,
)


class _DummyConn:
    def __init__(self) -> None:
        self.received = "UNSET"

    def handle_preemptions(self, arg) -> None:  # type: ignore[no-untyped-def]
        self.received = arg


class _DummyMeta(KVConnectorMetadata):
    pass


def test_multiconnector_dispatches_metadata_to_subconnectors():
    """The argument type RBLNModelRunner forwards (MultiKVConnectorMetadata) is
    accepted by MultiConnector.handle_preemptions and dispatched one entry per
    sub-connector. Fails if vLLM changes the base-class contract."""
    c0, c1 = _DummyConn(), _DummyConn()
    # Bypass __init__ (needs a full VllmConfig); handle_preemptions only reads
    # self._connectors.
    mc = MultiConnector.__new__(MultiConnector)
    mc._connectors = [c0, c1]

    m0, m1 = _DummyMeta(), _DummyMeta()
    mc.handle_preemptions(MultiKVConnectorMetadata(metadata=(m0, m1)))

    assert c0.received is m0
    assert c1.received is m1
