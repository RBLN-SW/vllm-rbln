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

"""Make torch.accelerator.empty_cache() a no-op when there's no accelerator.

On shutdown, vLLM calls torch.accelerator.empty_cache() for any non-CPU
platform. RBLN is non-CPU but runs on a CPU-only torch build with no torch
accelerator (NPU memory belongs to the rebel runtime), so torch 2.11 raises
"Cannot access accelerator device when none is available" and the EngineCore
dies during cleanup.

We wrap empty_cache() to swallow that one case; other errors propagate.
Applied via register_ops() at plugin load.
"""

# NOTE(eunji.lee):
# required test: torch-rbln 2.11.0 + vllm 0.22.0
import torch

from vllm_rbln.logger import init_logger

logger = init_logger(__name__)

_PATCHED = False


def _safe_empty_cache(orig):
    def _wrapper(*args, **kwargs):
        try:
            return orig(*args, **kwargs)
        except RuntimeError as e:
            # Only swallow the "no accelerator at all" case; surface anything else.
            if "none is available" in str(e):
                return None
            raise

    return _wrapper


def _patch_accelerator_empty_cache() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    torch.accelerator.empty_cache = _safe_empty_cache(torch.accelerator.empty_cache)
    logger.info(
        "vllm_rbln: guarded torch.accelerator.empty_cache() against "
        "'no accelerator available' raise on CPU-only torch builds"
    )


_patch_accelerator_empty_cache()
