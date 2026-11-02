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

from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch

from vllm_rbln.patches.registry import register_patch

register_patch(
    target="vllm.v1.sample.ops.topk_topp_sampler.apply_top_k_top_p",
    reason=(
        "Upstream selects Triton filtering based on availability without "
        "a platform dispatch hook. RBLN cannot execute those kernels."
    ),
    key=f"{__name__}.apply_top_k_top_p",
    owner_module=__name__,
)(apply_top_k_top_p_pytorch)
