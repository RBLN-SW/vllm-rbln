# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Console-script entry point that loads vllm_rbln plugins before dispatching
to the standard vllm CLI.

vLLM's CLI tools (notably `vllm bench serve`) do NOT trigger
`load_general_plugins()` because they never instantiate an `EngineCore`.
As a result, vllm_rbln's `register_ops()` — which applies our runtime
patches (including the `--trace` flag for `vllm bench serve`) — never runs
when users invoke `vllm bench serve` directly.

This shim is exposed as the `vllm-rbln` console script. It forces the
plugin registration step before handing control to vllm's CLI `main()`,
so commands like:

    vllm-rbln bench serve --trace <args>
    vllm-rbln serve <model> <args>

work the same as their `vllm <subcommand>` counterparts, but with the
RBLN tracing patches applied.
"""

from __future__ import annotations

import sys


def main() -> None:
    import vllm_rbln

    vllm_rbln.register_ops()

    from vllm.entrypoints.cli.main import main as _vllm_main

    sys.exit(_vllm_main())


if __name__ == "__main__":
    main()
