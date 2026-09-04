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

"""Helpers for driving RBLN connector hooks across (multi-)connectors.

Under ``MultiConnector`` these hooks sit on the nested children, so a hook driven
on the wrapper alone is silently skipped.
"""

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
)

from vllm_rbln.logger import init_logger

logger = init_logger(__name__)


@runtime_checkable
class SupportsKVCacheRegistrationFinalize(Protocol):
    """A connector that finalizes its KV-cache registration after warm-up.

    Implemented by connectors whose registration depends on the compiled
    model's KV cache physical views, which only exist once warm-up has run the
    compiled model.
    """

    def finalize_kv_cache_registration(self) -> None: ...


@runtime_checkable
class SupportsDeferredLoad(Protocol):
    """A connector that holds a KV load until a submission is in flight."""

    def flush_deferred_load(self) -> None: ...


def iter_kv_connectors(
    connector: KVConnectorBase_V1,
) -> Iterator[KVConnectorBase_V1]:
    """Yield leaf connectors, expanding ``MultiConnector`` children recursively."""
    if isinstance(connector, MultiConnector):
        for child in connector._connectors:
            yield from iter_kv_connectors(child)
    else:
        yield connector


def finalize_kv_cache_registrations(connector: KVConnectorBase_V1) -> None:
    """Finalize KV-cache registration on every connector that supports it.

    Recurses into ``MultiConnector`` children; connectors that do not implement
    the hook are skipped.
    """
    for child in iter_kv_connectors(connector):
        if isinstance(child, SupportsKVCacheRegistrationFinalize):
            logger.debug(
                "Finalizing KV cache registration for %s", type(child).__name__
            )
            child.finalize_kv_cache_registration()


def flush_deferred_loads(connector: KVConnectorBase_V1) -> None:
    """Issue the KV loads every connector that defers them is holding."""
    for child in iter_kv_connectors(connector):
        if isinstance(child, SupportsDeferredLoad):
            child.flush_deferred_load()
