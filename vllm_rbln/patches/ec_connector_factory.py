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


from vllm_rbln.patches.registry import add_registration


@add_registration(
    reason=(
        "Register the vllm model path's EC connector. A registration rather "
        "than an import for its side effect, so that both apply sites cover it: "
        "the process that resolves the model path reaches no other one."
    )
)
def register_rbln_ec_connectors() -> None:
    from vllm.distributed.ec_transfer.ec_connector.factory import ECConnectorFactory

    ECConnectorFactory.register_connector(
        "RblnECCPUConnector",
        "vllm_rbln.distributed.ec_transfer.ec_connector.rbln_ec_cpu_connector",
        "RblnECCPUConnector",
    )
