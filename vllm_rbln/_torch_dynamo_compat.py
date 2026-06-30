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

"""Lenient guard for torch 2.10's strict metrics_context enforcement.

torch 2.10 raises::

    RuntimeError: No toplevel event active. Please only call this
    function within a metrics context/dynamo_timed.

when ``CompileEventLogger.increment_toplevel`` /
``CompileEventLogger.add_to_set_toplevel`` is called outside a
``dynamo_timed`` block (``torch/_dynamo/utils.py`` ~lines 530, 580).

vllm-rbln's compile path enters ``torch.compile``-wrapped functions
(``rbln_model_runner.model_wrapper`` invoked via
``torch._dynamo.eval_frame.compile_wrapper``) during the
``compile_or_warm_up_model`` collective RPC, where the outermost
chromium event is not set up. Torch then raises and the EngineCore
fails to initialize KV caches.

This module replaces the two raise-on-missing-context entry points
with thin no-op-on-missing wrappers: if there's no toplevel event we
silently skip the metric update; everything else behaves identically.
We lose an optional metric increment, not correctness of the compile.

Imported by ``vllm_rbln.register_ops()`` at plugin load time so the
patch is in place before any vLLM worker starts compiling.
"""

import torch._dynamo.utils as _du

from vllm_rbln.logger import init_logger

logger = init_logger(__name__)

_PATCHED = False


def _silence_no_toplevel(orig):
    def _wrapper(*args, **kwargs):
        try:
            return orig(*args, **kwargs)
        except RuntimeError as e:
            if "No toplevel event active" in str(e):
                return None
            raise

    return _wrapper


def _patch_compile_event_logger() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    cel = _du.CompileEventLogger
    cel.increment_toplevel = staticmethod(_silence_no_toplevel(cel.increment_toplevel))
    cel.add_to_set_toplevel = staticmethod(
        _silence_no_toplevel(cel.add_to_set_toplevel)
    )
    logger.info(
        "vllm_rbln: silenced torch._dynamo CompileEventLogger "
        "'No toplevel event active' raises (torch 2.10 compat)"
    )


_patch_compile_event_logger()
