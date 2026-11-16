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

"""NIXL VRAM registration shared by encoder-cache and KV-cache transfers."""

from collections.abc import Mapping, Sequence
from typing import Any


def register_device_regions(
    agent: Any,
    regions: Sequence[tuple[int, int]],
    *,
    device_id: int,
    rbln_ctx_ptr: int,
    backend_options: Mapping[str, Any] | None = None,
) -> Any:
    """Register physical RBLN areas through the owner's device context."""
    import nixl_rbln

    if device_id < 0:
        raise ValueError("NIXL VRAM registration requires an RBLN tensor")
    nixl_rbln.ensure_rbln_backend(
        agent,
        device_id,
        rbln_ctx_ptr=rbln_ctx_ptr,
        **(backend_options or {}),
    )
    return agent.register_memory(
        [(address, size, device_id, "") for address, size in regions],
        mem_type="VRAM",
        backends=["RBLN"],
    )
