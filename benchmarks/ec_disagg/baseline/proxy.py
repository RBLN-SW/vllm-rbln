# SPDX-License-Identifier: Apache-2.0
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

"""Round-robin streaming proxy for the local vLLM baseline replicas."""

import argparse
import itertools

from aiohttp import ClientError, ClientSession, ClientTimeout, TCPConnector, web
from multidict import CIMultiDict


def forward_headers(headers):
    hop_headers = {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
    hop_headers.update(
        name.strip().lower()
        for value in headers.getall("Connection", [])
        for name in value.split(",")
    )
    return CIMultiDict(
        (name, value)
        for name, value in headers.items()
        if name.lower() not in hop_headers
    )


def build_app(upstreams):
    replicas = itertools.cycle(upstreams)
    session_key = web.AppKey("upstream_session", ClientSession)

    async def upstream_session(app):
        async with ClientSession(
            auto_decompress=False,
            connector=TCPConnector(limit=0),
            timeout=ClientTimeout(total=None, connect=10),
        ) as session:
            app[session_key] = session
            yield

    async def forward(request):
        target = next(replicas) if request.method == "POST" else upstreams[0]
        headers = forward_headers(request.headers)
        headers.popall("Host", None)
        body = await request.read() if request.can_read_body else None
        headers.popall("Content-Length", None)
        try:
            upstream = await request.app[session_key].request(
                request.method,
                f"{target}{request.rel_url}",
                data=body,
                headers=headers,
                allow_redirects=False,
            )
        except (ClientError, TimeoutError) as exc:
            raise web.HTTPBadGateway(text=f"Replica unavailable: {target}") from exc

        async with upstream:
            response = web.StreamResponse(
                status=upstream.status,
                headers=forward_headers(upstream.headers),
            )
            await response.prepare(request)
            async for chunk in upstream.content.iter_any():
                await response.write(chunk)
            await response.write_eof()
            return response

    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.cleanup_ctx.append(upstream_session)
    app.router.add_get("/health", forward)
    app.router.add_get("/v1/models", forward)
    app.router.add_post("/v1/chat/completions", forward)
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("upstreams", nargs="+")
    args = parser.parse_args()
    web.run_app(
        build_app(args.upstreams), host=args.host, port=args.port, access_log=None
    )
