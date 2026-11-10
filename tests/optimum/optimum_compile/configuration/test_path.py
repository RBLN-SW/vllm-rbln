# Copyright 2025 Rebellions Inc. All rights reserved.
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

import json

from vllm_rbln.utils.optimum.configuration.readers import (
    load_compiled_rbln_config,
)
from vllm_rbln.utils.optimum.path import resolve_compiled_model_path


def test_metadata_loader_reads_only_the_selected_path(tmp_path):
    first = tmp_path / "source"
    second = tmp_path / "cache"
    for path, value in ((first, 1), (second, 2)):
        path.mkdir()
        (path / "rbln_config.json").write_text(json.dumps({"batch_size": value}))
    assert load_compiled_rbln_config(str(first)) == {"batch_size": 1}
    assert load_compiled_rbln_config(str(second)) == {"batch_size": 2}
    assert load_compiled_rbln_config(str(tmp_path / "absent")) is None


def test_explicit_compiled_directory_takes_precedence(tmp_path):
    source = tmp_path / "source"
    cache = tmp_path / "cache"
    for path in (source, cache):
        path.mkdir()
        (path / "rbln_config.json").write_text("{}")
    assert resolve_compiled_model_path(str(source), str(cache)) == str(source)
    assert resolve_compiled_model_path(str(tmp_path / "missing"), str(cache)) == str(
        cache
    )
