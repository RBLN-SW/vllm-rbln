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

"""Synchronize vLLM and optimum-rbln configuration."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig

__all__ = ["sync_vllm_and_optimum"]


def sync_vllm_and_optimum(vllm_config: "VllmConfig") -> None:
    """Select the compile or artifact settings and apply them to vLLM."""
    # config.py imports compile_spec before defining its configuration classes.
    # Load the coordinator only when those classes are available.
    from .sync import sync_vllm_and_optimum as sync_config

    sync_config(vllm_config)
