#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
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

"""
EC Disaggregated Encoder Proxy

Proxy that routes OpenAI-compatible "/v1/chat/completions" requests to two
clusters for EC (Encoder Cache) disaggregation:

  * encode  (encoder — runs the visual encoder, caches via NIXL)
  * decode  (llm — pulls encoder cache via NIXL, runs LLM)

For multimodal input we:
    1. Extract *every* image/audio item from the request.
    2. Fire N concurrent requests to the encoder cluster
       (one request per MM item, with **all text removed**).
    3. Wait for all of them to succeed (encoder caches are now available
       via NIXL on the encoder side).
    4. Forward the *original* request to a decode server (llm pulls
       the encoder cache via NIXL and runs prefill + decode).

Usage:
  python client_ec_disaggregated.py \
      --encode-servers-urls "http://127.0.0.1:8000,http://127.0.0.1:8001" \
      --decode-servers-urls "http://127.0.0.1:9000"

  # Then send requests to the proxy:
  curl http://127.0.0.1:1800/v1/chat/completions -d '{...}'
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import random
import uuid
from collections.abc import AsyncIterator

import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

###############################################################################
# FastAPI app & global state
###############################################################################

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger("ec_proxy")

app = FastAPI()
encode_session: aiohttp.ClientSession | None = None
decode_session: aiohttp.ClientSession | None = None

# Monotonic counter used to round-robin encoder targets across chat
# requests. Without this, every request starts its MM-item index at 0
# and always hits e_urls[0], serialising all encodes on a single
# encoder.
_encode_rr_counter: int = 0

###############################################################################
# Utils
###############################################################################

MM_TYPES = {"image_url", "audio_url", "input_audio"}


def extract_mm_items(request_data: dict) -> list[dict]:
    """
    Return *all* image/audio items that appear anywhere in `messages`.

    Each returned dict looks like:
        { "type": "image_url", "image_url": {...} }
    """
    items: list[dict] = []
    for msg in request_data.get("messages", []):
        content = msg.get("content")
        if not isinstance(content, list):
            continue

        for item in content:
            if item.get("type") in MM_TYPES:
                items.append(item)
    return items


async def fanout_encoder_primer(
    orig_request: dict,
    e_urls: list[str],
    req_id: str,
) -> None:
    """
    1. Build one request *per MM item* with all text removed.
    2. Send them concurrently to the encode cluster (round-robin).
    3. Raise if any of them fails.
    """
    logger.info("[%s] Processing multimodal items...", req_id)

    mm_items = extract_mm_items(orig_request)
    if not mm_items:
        logger.info("[%s] No multimodal items, skipping encoder", req_id)
        return

    logger.info("[%s] Got %d multimodal items", req_id, len(mm_items))

    tasks = []

    # Round-robin over encode servers, advancing a *global* counter so
    # that concurrent chat requests with a single MM item still spread
    # across encoders.
    global _encode_rr_counter
    start = _encode_rr_counter
    _encode_rr_counter = (start + len(mm_items)) % max(len(e_urls), 1)
    url_cycle = (e_urls[(start + i) % len(e_urls)] for i in range(len(mm_items)))

    for idx, (item, target_url) in enumerate(zip(mm_items, url_cycle)):
        child_req_id = f"{req_id}:{idx}:{uuid.uuid4().hex[:6]}"
        headers = {"x-request-id": child_req_id}

        encoder_req = {
            "model": orig_request.get("model"),
            "messages": [
                {"role": "user", "content": [item]},
            ],
            # Only need 1 token so the server actually runs the encoder path
            "max_tokens": 1,
            "stream": False,
        }
        tasks.append(
            encode_session.post(
                f"{target_url}/v1/chat/completions",
                json=encoder_req,
                headers=headers,
            )
        )

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Fail fast if any sub-request failed
    for idx, r in enumerate(results):
        if isinstance(r, Exception):
            logger.error(
                "[%s] Encoder request #%d raised exception: %s",
                req_id,
                idx,
                r,
                exc_info=r,
            )
            raise HTTPException(
                status_code=502, detail=f"Encoder request failed: {str(r)}"
            )
        if r.status != 200:
            try:
                detail = await r.text()
            except Exception:
                detail = "<unable to read body>"
            logger.error(
                "[%s] Encoder request #%d returned status %s: %s",
                req_id,
                idx,
                r.status,
                detail,
            )
            raise HTTPException(
                status_code=r.status,
                detail=f"Encoder request failed: {detail}",
            )

    logger.info(
        "[%s] All %d encoder requests completed successfully",
        req_id,
        len(mm_items),
    )


###############################################################################
# Middleware for request/response logging
###############################################################################


@app.middleware("http")
async def log_requests(request: Request, call_next):
    req_id = request.headers.get("x-request-id", str(uuid.uuid4()))

    logger.info(
        ">>> [%s] %s %s from %s",
        req_id,
        request.method,
        request.url.path,
        request.client.host if request.client else "unknown",
    )

    try:
        response = await call_next(request)
        logger.info(
            "<<< [%s] %s %s completed with status %d",
            req_id,
            request.method,
            request.url.path,
            response.status_code,
        )
        return response
    except Exception as e:
        logger.exception(
            "!!! [%s] %s %s failed with error: %s",
            req_id,
            request.method,
            request.url.path,
            str(e),
        )
        raise


###############################################################################
# FastAPI lifecycle
###############################################################################


@app.on_event("startup")
async def on_startup() -> None:
    global encode_session, decode_session
    timeout = aiohttp.ClientTimeout(total=100_000)
    # force_close=True: new TCP per request. Avoids occasional 502s when
    # the encoder server drops an idle keep-alive connection that the
    # proxy then reuses (ServerDisconnectedError at resp.start).
    # Overhead on localhost is negligible vs. encoder compute.
    enc_connector = aiohttp.TCPConnector(limit=0, force_close=True)
    dec_connector = aiohttp.TCPConnector(limit=0, force_close=True)
    encode_session = aiohttp.ClientSession(timeout=timeout, connector=enc_connector)
    decode_session = aiohttp.ClientSession(timeout=timeout, connector=dec_connector)


@app.on_event("shutdown")
async def on_shutdown() -> None:
    global encode_session, decode_session
    if encode_session:
        await encode_session.close()
    if decode_session:
        await decode_session.close()


###############################################################################
# Core forwarding
###############################################################################


async def forward_non_stream(
    req_data: dict,
    req_id: str,
    e_urls: list[str],
    d_url: str,
) -> dict:
    try:
        # Step 1: Fan-out to encoder cluster (produces NIXL caches)
        await fanout_encoder_primer(req_data, e_urls, req_id)

        # Step 2: Forward original request to decode cluster
        # (llm pulls encoder cache via NIXL)
        logger.info("[%s] Forwarding to decode: %s", req_id, d_url)
        headers = {"x-request-id": req_id}

        async with decode_session.post(
            f"{d_url}/v1/chat/completions", json=req_data, headers=headers
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("[%s] Error in forward_non_stream: %s", req_id, str(e))
        raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}") from e


async def forward_stream(
    req_data: dict,
    req_id: str,
    e_urls: list[str],
    d_url: str,
) -> AsyncIterator[str]:
    try:
        # Step 1: Fan-out to encoder cluster (produces NIXL caches)
        await fanout_encoder_primer(req_data, e_urls, req_id)

        # Step 2: Stream from decode cluster
        logger.info("[%s] Starting streaming from decode: %s", req_id, d_url)
        headers = {"x-request-id": req_id}

        async with decode_session.post(
            f"{d_url}/v1/chat/completions",
            json=req_data,
            headers=headers,
        ) as resp:
            resp.raise_for_status()
            async for chunk in resp.content.iter_chunked(1024):
                if chunk:
                    yield chunk.decode("utf-8", errors="ignore")

        logger.info("[%s] Streaming completed", req_id)

    except HTTPException:
        logger.exception("[%s] HTTPException in forward_stream", req_id)
        raise
    except Exception as e:
        logger.exception("[%s] Error in forward_stream: %s", req_id, str(e))
        raise HTTPException(
            status_code=500, detail=f"Proxy streaming error: {str(e)}"
        ) from e


###############################################################################
# Public routes
###############################################################################


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        req_data = await request.json()
        req_id = request.headers.get("x-request-id", str(uuid.uuid4()))

        e_urls = app.state.e_urls
        d_url = random.choice(app.state.d_urls)

        is_streaming = req_data.get("stream", False)

        if is_streaming:
            return StreamingResponse(
                forward_stream(req_data, req_id, e_urls, d_url),
                media_type="text/event-stream",
            )
        result = await forward_non_stream(req_data, req_id, e_urls, d_url)
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in chat_completions endpoint: %s", str(e))
        raise HTTPException(
            status_code=500, detail=f"Request processing error: {str(e)}"
        ) from e


@app.get("/v1/models")
async def list_models():
    async with decode_session.get(f"{app.state.d_urls[0]}/v1/models") as resp:
        resp.raise_for_status()
        return await resp.json()


@app.get("/health")
async def health_check():
    async def healthy(urls):
        if not urls:
            return "empty"
        for u in urls:
            try:
                async with encode_session.get(f"{u}/health") as resp:
                    resp.raise_for_status()
            except Exception:
                return "unhealthy"
        return "healthy"

    e_status, d_status = await asyncio.gather(
        healthy(app.state.e_urls),
        healthy(app.state.d_urls),
    )

    overall_healthy = all(status != "unhealthy" for status in (e_status, d_status))

    return JSONResponse(
        {
            "proxy": "healthy",
            "encode_cluster": e_status,
            "decode_cluster": d_status,
        },
        status_code=200 if overall_healthy else 503,
    )


###############################################################################
# Profiler fan-out
###############################################################################


async def _post_if_available(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict,
    headers: dict,
) -> dict | None:
    try:
        resp = await session.post(url, json=payload, headers=headers)
        if resp.status == 404:
            logger.warning("Profiling endpoint missing on %s", url)
            return None
        resp.raise_for_status()
        return await resp.json(content_type=None)
    except aiohttp.ClientResponseError as exc:
        if exc.status == 404:
            logger.warning("Profiling endpoint missing on %s", url)
            return None
        raise


async def _profile_cmd(cmd: str, payload: dict, e_url: str, d_url: str):
    headers = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', '')}"}

    encode_task = _post_if_available(
        encode_session, f"{e_url}/{cmd}_profile", payload, headers
    )
    decode_task = _post_if_available(
        decode_session, f"{d_url}/{cmd}_profile", payload, headers
    )

    encode_res, decode_res = await asyncio.gather(encode_task, decode_task)

    if encode_res is decode_res is None:
        raise HTTPException(
            status_code=503,
            detail="Profiling endpoints are disabled on all clusters",
        )

    return {
        "encode": encode_res,
        "decode": decode_res,
    }


@app.post("/start_profile")
async def start_profile(request: Request):
    body = await request.json()
    e_url = random.choice(app.state.e_urls)
    d_url = random.choice(app.state.d_urls)
    return await _profile_cmd("start", body, e_url, d_url)


@app.post("/stop_profile")
async def stop_profile(request: Request):
    body = await request.json()
    e_url = random.choice(app.state.e_urls)
    d_url = random.choice(app.state.d_urls)
    return await _profile_cmd("stop", body, e_url, d_url)


###############################################################################
# CLI
###############################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EC Disaggregated Encoder Proxy (E+PD mode)"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=1800)
    parser.add_argument(
        "--encode-servers-urls",
        required=True,
        help="Comma-separated encode (encoder) URLs "
        '("http://127.0.0.1:8000,http://127.0.0.1:8001")',
    )
    parser.add_argument(
        "--decode-servers-urls",
        required=True,
        help='Comma-separated decode (llm) URLs ("http://127.0.0.1:9000")',
    )

    args = parser.parse_args()
    app.state.e_urls = [
        u.strip() for u in args.encode_servers_urls.split(",") if u.strip()
    ]
    app.state.d_urls = [
        u.strip() for u in args.decode_servers_urls.split(",") if u.strip()
    ]

    logger.info("EC Proxy listening on %s:%s", args.host, args.port)
    logger.info("Encode servers (encoders): %s", app.state.e_urls)
    logger.info("Decode servers (llms): %s", app.state.d_urls)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        loop="uvloop",
        access_log=True,
    )
