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

# A real RBLN NIXL worker, and a real peer to hand it metadata.
#
# A vllm import has to be function-local here: a conftest is imported before the
# native conftest's pytest_configure sets VLLM_RBLN_USE_VLLM_MODEL=1, and
# resolving RblnPlatform early would pin device_type for the session.

from __future__ import annotations

import contextlib
import sys
import threading
from types import SimpleNamespace
from typing import Any

import pytest


@pytest.fixture
def make_worker(monkeypatch, tmp_path_factory):
    """Build a real RblnNixl*ConnectorWorker; call it from the test body."""
    from vllm.config import set_current_vllm_config
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
        base_worker as up_worker,
    )

    from tests.native.distributed.kv_connector.utils import (
        FakeNixlAgent,
        KvGeometry,
        draft_model_dir,
        engine_config,
        fake_nixl_rbln,
        patch_in_package,
        set_connector_options,
    )
    from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.pull_worker import (  # noqa: E501
        RblnNixlPullConnectorWorker,
    )
    from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.push_worker import (  # noqa: E501
        RblnNixlPushConnectorWorker,
    )

    stack = contextlib.ExitStack()
    # Session-scoped so engine_config's cache is hit across tests in a run.
    draft_root = tmp_path_factory.getbasetemp()

    def _make(
        *,
        direction: str = "pull",
        draft_kv_heads: int | None = None,
        kv_cache: KvGeometry | None = None,
        swa_view_opt: bool = False,
        register: bool = True,
    ) -> Any:
        geometry = kv_cache or KvGeometry()
        config = engine_config(
            kv_buffer_device="rbln",
            speculative_model=(
                None
                if draft_kv_heads is None
                else draft_model_dir(draft_root, draft_kv_heads)
            ),
            block_size=geometry.block_size,
        )
        # Left open for the test's duration: get_current_attn_backends reads it
        # during __init__, and later production calls read it again.
        stack.enter_context(set_current_vllm_config(config))
        set_connector_options(monkeypatch, config, swa_view_opt=swa_view_opt)

        if geometry.draft_layers:
            assert config.speculative_config is not None, (
                "draft_layers needs a speculative config -- _layer_kv_heads "
                "recognises a head count only if the target or a declared "
                "draft model has it"
            )
            draft = config.speculative_config.draft_model_config
            geometry = geometry.with_draft_heads(draft.get_total_num_kv_heads())

        agent = FakeNixlAgent()
        monkeypatch.setattr(up_worker, "NixlWrapper", lambda *a, **k: agent)
        # Upstream builds an agent config only when this name is importable;
        # None takes the branch that passes no config at all.
        monkeypatch.setattr(up_worker, "nixl_agent_config", None)
        monkeypatch.setattr(up_worker, "get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(
            up_worker, "get_tensor_model_parallel_world_size", lambda: 1
        )
        # The other device-identity probe (see KvGeometry.kv_caches): the D2D
        # path asks rebel for the tensor's context pointer, which it hands
        # straight to the adapter. `aligned_tensor` is a separate name and
        # stays real, so the host-staging path still allocates for real.
        stack.enter_context(
            patch_in_package(
                rebel=SimpleNamespace(
                    context_of=lambda t: SimpleNamespace(rbln_ctx_ptr=0)
                )
            )
        )

        kv_caches = geometry.kv_caches()
        monkeypatch.setitem(
            sys.modules,
            "nixl_rbln",
            fake_nixl_rbln(geometry, kv_caches),
        )

        cls = (
            RblnNixlPullConnectorWorker
            if direction == "pull"
            else RblnNixlPushConnectorWorker
        )
        worker = cls(config, "local-engine", geometry.kv_cache_config())

        # The topology's layout answers all come from get_kv_cache_shape, and
        # upstream's FLASH_ATTN is 5-dim where RBLN's is 6-dim -- which flips
        # cross_layers_blocks. Assert the resolution rather than trust it.
        assert worker.attn_backends and worker.attn_backends[0].__module__.startswith(
            "vllm_rbln"
        ), (
            f"attn backend resolved to {worker.attn_backends} -- the native "
            "path's 6-dim RBLN backend must be registered before the worker is "
            "built, or every layout answer below is upstream's"
        )

        if register:
            worker.register_kv_caches(kv_caches)
            worker.finalize_kv_cache_registration()
        return worker

    try:
        yield _make
    finally:
        stack.close()


@pytest.fixture
def peer_listener():
    """A real ZMQ REP socket answering the side-channel handshake.

    The compat-hash gate and the engine-id check are then behaviour rather than
    a mocked return value, and the metadata crosses a real msgspec encode.
    """
    import msgspec
    import zmq
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
        GET_META_MSG,
    )

    started: list[tuple[Any, threading.Thread, Any, Any]] = []

    def _serve(payload_for):
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REP)
        port = sock.bind_to_random_port("tcp://127.0.0.1")
        stop = threading.Event()

        def loop():
            poller = zmq.Poller()
            poller.register(sock, zmq.POLLIN)
            decoder = msgspec.msgpack.Decoder()
            encoder = msgspec.msgpack.Encoder()
            while not stop.is_set():
                if not poller.poll(50):
                    continue
                kind, rank = decoder.decode(sock.recv())
                assert kind == GET_META_MSG, f"unexpected query {kind!r}"
                sock.send(encoder.encode(payload_for(rank)))

        t = threading.Thread(target=loop, daemon=True)
        t.start()
        started.append((stop, t, sock, ctx))
        return port

    try:
        yield _serve
    finally:
        for stop, t, sock, ctx in started:
            stop.set()
            t.join(timeout=2)
            # The bound port and the context's IO thread go with the socket; an
            # un-terminated context holding an open socket can block at exit.
            sock.close()
            ctx.term()
