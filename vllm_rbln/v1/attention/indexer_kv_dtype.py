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
"""dtype of the sparse-attention (DSA) indexer K cache on RBLN.

Upstream's `--attention-config '{"indexer_kv_dtype": ...}'` picks it separately
from `--kv-cache-dtype`, which then covers only the MLA latent cache. "auto"
keeps the RBLN behavior from before the knob: the indexer follows
`--kv-cache-dtype` (fp8* gives an fp8 indexer, anything else bf16).
"""

from typing import Any

# The RBLN DSA indexer kernel reads a bf16 key cache or an fp8_e4m3 one with a
# per-token scale cache next to it. Upstream also lists mxfp4 and nvfp4.
RBLN_INDEXER_KV_DTYPES = ("bf16", "fp8")


def resolve_rbln_indexer_kv_dtype(
    cache_dtype: str | None, attention_config: Any
) -> str:
    """The DSA indexer K-cache dtype, "bf16" or "fp8"."""
    default = "fp8" if cache_dtype and cache_dtype.startswith("fp8") else "bf16"
    requested = getattr(attention_config, "indexer_kv_dtype", "auto")
    if requested == "auto":
        return default
    if requested not in RBLN_INDEXER_KV_DTYPES:
        raise ValueError(
            f"indexer_kv_dtype={requested!r} is not supported on RBLN; the DSA "
            f"indexer takes one of {('auto', *RBLN_INDEXER_KV_DTYPES)}."
        )
    return requested


def is_fp8_rbln_indexer(cache_dtype: str | None, attention_config: Any) -> bool:
    return resolve_rbln_indexer_kv_dtype(cache_dtype, attention_config) == "fp8"
