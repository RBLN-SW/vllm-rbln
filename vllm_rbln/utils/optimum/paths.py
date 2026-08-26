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
import os

# The artefact optimum-rbln writes at the top level of a compiled model
# directory. A plain HuggingFace checkpoint never has one.
RBLN_CONFIG_FILE = "rbln_config.json"


def is_compiled_dir(path: str | None) -> bool:
    if not path:
        return False
    return os.path.isfile(os.path.join(path, RBLN_CONFIG_FILE))
