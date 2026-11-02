# Copyright 2026 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

from vllm_rbln.patches.axk2 import _skt_config
from vllm_rbln.patches.axk2.loader import alias_frozen_module

CANONICAL_NAME = "vllm.transformers_utils.configs.axk2"

# Imported by its real path instead of being loaded under CANONICAL_NAME: pickle
# names a class by the module it was loaded under, and an engine core started
# with VLLM_WORKER_MULTIPROC_METHOD=spawn unpickles this config before anything
# has imported vllm_rbln in that process, where only the real path resolves.
alias_frozen_module(CANONICAL_NAME, _skt_config)

AXK2Config = _skt_config.AXK2Config

__all__ = ["AXK2Config", "CANONICAL_NAME"]
