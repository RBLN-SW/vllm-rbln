#!/usr/bin/env python3
"""
vLLM Prefill/Decode disaggregation proxy server (RblnNixlPullConnector).

Flow:
  1) client 요청 수신 (/v1/completions, /v1/chat/completions)
  2) prefill 서버로 max_tokens=1 + kv_transfer_params(do_remote_decode=True) 요청
  3) prefill 응답의 kv_transfer_params (remote_engine_id, remote_request_id,
     remote_block_ids, remote_host, remote_port, tp_size ...) 를 받아
  4) decode 서버로 원본 요청 + kv_transfer_params 전달 -> decode 가 prefill 에서 KV pull

Topology (default):
  prefill: 10.60.0.56:8100  (PP=4, DP 없음)
  decode : 10.60.0.40:8100  (DP=4, EP)

DP rank pinning:
  - 클라이언트의 `X-data-parallel-rank` 헤더는 decode 서버로만 forward
  - prefill 쪽 DP rank 는 `X-Prefill-Data-Parallel-Rank` 로 별도 지정 (prefill 은 DP=1 이라 보통 불필요)

Usage:
  python disagg.py --port 8800 \
      --prefill http://10.60.0.56:8100 \
      --decode  http://10.60.0.40:8100
"""

import argparse
import asyncio
import copy
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("pd_proxy")

PREFILL_BASE = os.environ.get("PREFILL_BASE", "http://10.60.0.56:8100")
DECODE_BASE = os.environ.get("DECODE_BASE", "http://10.60.0.40:8100")

DP_RANK_HEADER = "x-data-parallel-rank"
PREFILL_DP_RANK_HEADER = "x-prefill-data-parallel-rank"

EXCLUDED_HEADERS = {
    "host", "content-length", "content-type", "connection", "keep-alive",
    "transfer-encoding", "upgrade", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "accept-encoding", "expect",
    DP_RANK_HEADER, PREFILL_DP_RANK_HEADER,
}


def build_forward_headers(request: Request, target: str) -> dict:
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in EXCLUDED_HEADERS
    }
    headers["content-type"] = "application/json"

    if target == "decode":
        dp_rank = request.headers.get(DP_RANK_HEADER)
        if dp_rank is not None:
            headers[DP_RANK_HEADER] = dp_rank
    else:
        prefill_dp_rank = request.headers.get(PREFILL_DP_RANK_HEADER)
        if prefill_dp_rank is not None:
            headers[DP_RANK_HEADER] = prefill_dp_rank

    if "x-request-id" not in {k.lower() for k in headers}:
        headers["x-request-id"] = request.state.request_id
    return headers


def build_prefill_body(original: dict) -> dict:
    body = copy.deepcopy(original)
    body["stream"] = False
    body.pop("stream_options", None)
    body["max_tokens"] = 1
    if "max_completion_tokens" in body:
        body["max_completion_tokens"] = 1
    # min_tokens > max_tokens 이면 vLLM 이 요청을 거절함
    body.pop("min_tokens", None)
    body["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }
    return body


def build_decode_body(original: dict, kv_transfer_params: dict) -> dict:
    body = copy.deepcopy(original)
    # 클라이언트가 준 lane 지정(xfer_mode / xfer_rail)은 decode 에만 전달
    lane = {
        k: v
        for k, v in (original.get("kv_transfer_params") or {}).items()
        if k in ("xfer_mode", "xfer_rail")
    }
    body["kv_transfer_params"] = {**kv_transfer_params, **lane}
    return body


@asynccontextmanager
async def lifespan(app: FastAPI):
    limits = httpx.Limits(max_connections=2048, max_keepalive_connections=1024)
    app.state.client = httpx.AsyncClient(limits=limits,
                                         timeout=httpx.Timeout(None))
    logger.info("prefill=%s decode=%s", PREFILL_BASE, DECODE_BASE)
    yield
    await app.state.client.aclose()


app = FastAPI(lifespan=lifespan)


async def run_prefill(request: Request, path: str, body: dict) -> dict:
    client: httpx.AsyncClient = app.state.client
    headers = build_forward_headers(request, "prefill")
    resp = await client.post(f"{PREFILL_BASE}{path}",
                             json=build_prefill_body(body), headers=headers)
    resp.raise_for_status()
    kv_params = resp.json().get("kv_transfer_params")
    if not kv_params:
        raise RuntimeError(
            f"prefill response missing kv_transfer_params "
            f"(id={request.state.request_id}). "
            f"prefill 서버가 --kv-transfer-config 로 떠 있는지 확인하세요.")
    return kv_params


async def run_overlapping_decodes(client: httpx.AsyncClient, url: str,
                                  headers: dict, decode_body: dict,
                                  lanes: list[dict], rid: str,
                                  t1: float) -> Response:
    """lane 마다 decode 요청 하나씩, 동시에 보낸다. 첫 번째 응답을 돌려준다."""

    async def one(i: int, lane: dict):
        body = copy.deepcopy(decode_body)
        params = body["kv_transfer_params"]
        params.pop("xfer_mode", None)
        params.pop("xfer_rail", None)
        params.update({k: v for k, v in lane.items()
                       if k in ("xfer_mode", "xfer_rail")})
        # vLLM 이 x-request-id 로 request id 를 만들므로 겹치지 않게 분리
        hdrs = {**headers, "x-request-id": f"{rid}-lane{i}"}
        resp = await client.post(url, json=body, headers=hdrs)
        logger.info("[%s-lane%d] decode %s done in %.2fs [%s]", rid, i, lane,
                    time.perf_counter() - t1, resp.status_code)
        return resp

    try:
        resps = await asyncio.gather(
            *(one(i, lane) for i, lane in enumerate(lanes)))
    except Exception as e:
        logger.exception("[%s] overlapping decode error", rid)
        return JSONResponse({"error": f"decode stage failed: {e}"},
                            status_code=502)
    first = resps[0]
    return Response(content=first.content, status_code=first.status_code,
                    media_type=first.headers.get("content-type",
                                                 "application/json"))


async def proxy_generate(request: Request, path: str):
    request.state.request_id = request.headers.get("x-request-id",
                                                   uuid.uuid4().hex)
    rid = request.state.request_id
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)

    client: httpx.AsyncClient = app.state.client

    # ---- 1) prefill ----
    t0 = time.perf_counter()
    try:
        kv_params = await run_prefill(request, path, body)
    except httpx.HTTPStatusError as e:
        logger.error("[%s] prefill failed [%s]: %s", rid,
                     e.response.status_code, e.response.text[:500])
        return Response(content=e.response.content,
                        status_code=e.response.status_code,
                        media_type="application/json")
    except Exception as e:
        logger.exception("[%s] prefill error", rid)
        return JSONResponse({"error": f"prefill stage failed: {e}"},
                            status_code=502)
    t1 = time.perf_counter()
    logger.info("[%s] prefill done in %.2fs (remote_engine_id=%s, dp=%s)",
                rid, t1 - t0, kv_params.get("remote_engine_id"),
                request.headers.get(DP_RANK_HEADER))

    # ---- 2) decode ----
    decode_headers = build_forward_headers(request, "decode")
    decode_body = build_decode_body(body, kv_params)
    url = f"{DECODE_BASE}{path}"

    # 실험용: 같은 prefill KV 를 lane 설정만 바꿔 같은 DP rank 로 동시에 pull
    overlap_lanes = (body.get("kv_transfer_params") or {}).get("overlap_lanes")
    if overlap_lanes:
        if body.get("stream", False):
            return JSONResponse(
                {"error": "overlap_lanes supports stream=false only"},
                status_code=400)
        return await run_overlapping_decodes(
            client, url, decode_headers, decode_body, overlap_lanes, rid, t1)

    if body.get("stream", False):
        req = client.build_request("POST", url, json=decode_body,
                                   headers=decode_headers)
        try:
            resp = await client.send(req, stream=True)
        except Exception as e:
            logger.exception("[%s] decode connect error", rid)
            return JSONResponse({"error": f"decode stage failed: {e}"},
                                status_code=502)
        if resp.status_code != 200:
            err = await resp.aread()
            await resp.aclose()
            logger.error("[%s] decode failed [%s]: %s", rid,
                         resp.status_code, err[:500])
            return Response(content=err, status_code=resp.status_code,
                            media_type="application/json")

        async def stream():
            try:
                async for chunk in resp.aiter_bytes():
                    yield chunk
            except Exception:
                logger.exception("[%s] decode stream error", rid)
                yield b'data: {"error": "decode stage failed"}\n\n'
            finally:
                await resp.aclose()
                logger.info("[%s] decode done in %.2fs", rid,
                            time.perf_counter() - t1)

        return StreamingResponse(stream(), media_type="text/event-stream")

    try:
        resp = await client.post(url, json=decode_body, headers=decode_headers)
    except Exception as e:
        logger.exception("[%s] decode error", rid)
        return JSONResponse({"error": f"decode stage failed: {e}"},
                            status_code=502)
    logger.info("[%s] decode done in %.2fs [%s]", rid,
                time.perf_counter() - t1, resp.status_code)
    return Response(content=resp.content, status_code=resp.status_code,
                    media_type=resp.headers.get("content-type",
                                                "application/json"))


@app.post("/v1/completions")
async def completions(request: Request):
    return await proxy_generate(request, "/v1/completions")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await proxy_generate(request, "/v1/chat/completions")


@app.get("/v1/models")
async def models():
    resp = await app.state.client.get(f"{DECODE_BASE}/v1/models")
    return Response(content=resp.content, status_code=resp.status_code,
                    media_type="application/json")


@app.get("/health")
async def health():
    result = {}
    for name, base in (("prefill", PREFILL_BASE), ("decode", DECODE_BASE)):
        try:
            r = await app.state.client.get(f"{base}/health", timeout=5.0)
            result[name] = "ok" if r.status_code == 200 else f"http {r.status_code}"
        except Exception as e:
            result[name] = f"unreachable: {e}"
    status = 200 if all(v == "ok" for v in result.values()) else 503
    return JSONResponse(result, status_code=status)


def main():
    global PREFILL_BASE, DECODE_BASE
    parser = argparse.ArgumentParser(description="vLLM PD disaggregation proxy")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8800)
    parser.add_argument("--prefill", default=PREFILL_BASE,
                        help="prefill server base URL")
    parser.add_argument("--decode", default=DECODE_BASE,
                        help="decode server base URL")
    args = parser.parse_args()
    PREFILL_BASE = args.prefill.rstrip("/")
    DECODE_BASE = args.decode.rstrip("/")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
