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


from pathlib import Path


def fake_sysfs_net(tmp_path: Path, **operstate: str) -> Path:
    # The pod network is a veth with no `device`; RDMA NICs are physical.
    (tmp_path / "eth0").mkdir()
    (tmp_path / "eth0" / "operstate").write_text("up\n")
    for name, state in operstate.items():
        (tmp_path / name).mkdir()
        (tmp_path / name / "device").touch()
        (tmp_path / name / "operstate").write_text(f"{state}\n")
    return tmp_path
