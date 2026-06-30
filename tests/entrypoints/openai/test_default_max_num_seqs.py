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

import pytest
import requests
from utils import RemoteOpenAIServer

MODEL_NAME = "facebook/opt-125m"
ARGS = ["--block-size", str(2048)]
# The RBLN default an unset max_num_seqs resolves to (see vllm_rbln.platform).
RBLN_DEFAULT_MAX_NUM_SEQS = 1
# opt_125m_batch2 is compiled for batch size 2, so the explicit value we set to
# check it is preserved must not exceed the compiled batch.
EXPLICIT_MAX_NUM_SEQS = 2

# VLLM_RBLN_USE_VLLM_MODEL selects the runtime backend: 0 = optimum path,
# 1 = vLLM-native model path. The default must hold for both.
# FIXME MODE=1 is skipped for now.
MODES = ["0"]


def _serve_env(mode: str) -> dict[str, str]:
    # VLLM_SERVER_DEV_MODE mounts the /server_info endpoint we read below.
    return {"VLLM_RBLN_USE_VLLM_MODEL": mode, "VLLM_SERVER_DEV_MODE": "1"}


def _served_max_num_seqs(server: RemoteOpenAIServer) -> int:
    """Read the running server's resolved max_num_seqs from /server_info."""
    resp = requests.get(server.url_for("server_info"), params={"config_format": "json"})
    resp.raise_for_status()
    return resp.json()["vllm_config"]["scheduler_config"]["max_num_seqs"]


@pytest.mark.parametrize("mode", MODES)
def test_serve_unset_max_num_seqs_defaults_to_one(mode):
    with RemoteOpenAIServer(MODEL_NAME, ARGS, env_dict=_serve_env(mode)) as server:
        resolved = _served_max_num_seqs(server)

    assert resolved == RBLN_DEFAULT_MAX_NUM_SEQS, (
        f"vllm serve with VLLM_RBLN_USE_VLLM_MODEL={mode} should default an unset "
        f"max_num_seqs to {RBLN_DEFAULT_MAX_NUM_SEQS}, got {resolved}"
    )


@pytest.mark.parametrize("mode", MODES)
def test_serve_explicit_max_num_seqs_is_preserved(mode):
    ARGS.extend(["--max-num-seqs", str(EXPLICIT_MAX_NUM_SEQS)])
    with RemoteOpenAIServer(MODEL_NAME, ARGS, env_dict=_serve_env(mode)) as server:
        resolved = _served_max_num_seqs(server)

    assert resolved == EXPLICIT_MAX_NUM_SEQS, (
        f"vllm serve with VLLM_RBLN_USE_VLLM_MODEL={mode} must preserve an explicit "
        f"max_num_seqs={EXPLICIT_MAX_NUM_SEQS}, got {resolved}"
    )
