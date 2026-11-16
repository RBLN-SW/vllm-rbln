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

# Import-free so the attention patch can read it without importing the drafter.
PASS_IDX = 0


def select(attn_metadata):
    """This pass's entry when an unrolled region passed one per pass."""
    if isinstance(attn_metadata, list):
        return attn_metadata[PASS_IDX]
    return attn_metadata
