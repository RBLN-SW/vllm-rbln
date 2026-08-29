# Copyright 2026 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

from vllm_rbln.patches.axk2.loader import load_frozen_module

CANONICAL_NAME = "vllm.transformers_utils.configs.axk2"

_module = load_frozen_module(CANONICAL_NAME, "_skt_config.py")

AXK2Config = _module.AXK2Config

__all__ = ["AXK2Config", "CANONICAL_NAME"]
