# Copyright 2026 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Which metadata entry the attention layers read.

A module of its own, and deliberately free of imports: the drafter needs to
set this and the attention patch needs to read it, and having the drafter
import `vllm_rbln.patches.attention` for it would apply that module's patches
at drafter-import time instead of the point the patch registry chooses.

0 everywhere except inside the drafter's chained region, which advances it per
pass. A module global rather than an argument because threading it would mean
changing the signature of every forward between the drafter and the attention
layer, and it is only ever read at trace time.
"""

PASS_IDX = 0
