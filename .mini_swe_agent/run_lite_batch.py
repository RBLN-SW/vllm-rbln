#!/usr/bin/env python3
"""mini-swe-agent 로 SWE-bench Lite 전체를 **우리 원격 샌드박스**에서 배치 실행.

agent(mini-swe-agent 루프 + LLM 호출)는 로컬에서 동시(worker 스레드)로 돌고, 각 인스턴스의
tool call(bash)은 RebelSandboxEnvironment → control plane → sandbox Pod 에서 실행된다.
harbor 없이 "agent 로컬 / tool 실행 원격" 을 데이터셋 전체에 대해 동시성 N 으로 돌리는 예시.

사전:
  pip install --extra-index-url https://nexus.mgmt.rbln.in/repository/pypi-rebel-sandbox-dev/simple \
    "rebel-sandbox-client[minisweagent]"
  export REBEL_SANDBOX_URL=https://rebel-sandbox.sandbox.udc.rbln.in
  export OPENAI_API_BASE=<모델 엔드포인트>  OPENAI_API_KEY=<키>

실행:
  python examples/mini_swe_agent/run_lite_batch.py --workers 50 --model openai/MiniMaxAI/MiniMax-M2.5
  # 결과(preds.json, mini-swe-agent/SWE-bench 포맷)를 --output 에 누적 저장(재실행 시 이어감).
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from minisweagent.agents.default import DefaultAgent
from minisweagent.config import get_config_from_spec
from minisweagent.models.litellm_model import LitellmModel
from minisweagent.utils.serialize import recursive_merge
from rebel_sandbox_client import SandboxClient
from rebel_sandbox_client import mlflow as mlflow_helper
from rebel_sandbox_client.minisweagent import RebelSandboxEnvironment

DATASET = os.getenv("SWEBENCH_DATASET", "princeton-nlp/SWE-bench_Lite")
SPLIT = os.getenv("SWEBENCH_SPLIT", "test")
EXPERIMENT = "rebel-sandbox/mini-swe-agent-batch"
_LOCK = threading.Lock()


def _mlflow_child(mlctx, iid, image, model, patch, ncalls, status, agent) -> str | None:
    """인스턴스별 child run 을 부모 아래에 기록(스레드 안전: MlflowClient 직접 사용).

    반환: child run_id (나중에 eval 결과 resolved 를 이 run 에 추가). 비활성/실패 시 None.
    """
    if not mlctx:
        return None
    client = mlctx["client"]
    try:
        run = client.create_run(
            experiment_id=mlctx["experiment_id"],
            run_name=iid,
            tags={"mlflow.parentRunId": mlctx["parent_run_id"]},
        )
        rid = run.info.run_id
        client.log_param(rid, "instance_id", iid)
        client.log_param(rid, "image", image)
        client.log_param(rid, "model", model)
        client.log_metric(rid, "api_calls", ncalls)
        client.log_metric(rid, "patch_nonempty", int(bool((patch or "").strip())))
        client.set_tag(rid, "exit_status", status)
        if patch:
            client.log_text(rid, patch, "patch.diff")
        msgs = getattr(agent, "messages", None)
        if msgs:
            client.log_text(
                rid, json.dumps(msgs, indent=2, default=str, ensure_ascii=False), "trajectory.json"
            )
        # 종료는 eval 결과까지 기록한 뒤(main)에서. 여기선 열어둔 채 rid 반환.
        return rid
    except Exception:  # noqa: BLE001 — 로깅 실패가 배치를 막지 않게
        return None


def _run_traced(mlctx, iid: str, agent, problem: str) -> dict:
    """agent.run 을 instance 단위 span 으로 감싸 하나의 multi-turn trace 로 만든다.

    litellm autolog 가 턴마다 만드는 LLM span 이 이 root span 아래 중첩되어, instance 당
    trace 1개(멀티턴)로 Traces 탭에 보인다. mlflow 비활성이면 그냥 실행.
    """
    mlf = mlctx.get("mlflow") if mlctx else None
    if mlf is None:
        return agent.run(problem)
    with mlf.start_span(name=f"instance:{iid}") as span:
        try:
            span.set_inputs({"instance_id": iid, "problem": problem[:2000]})
        except Exception:  # noqa: BLE001
            pass
        info = agent.run(problem)
        try:
            span.set_outputs(
                {
                    "exit_status": info.get("exit_status"),
                    "patch_nonempty": bool((info.get("submission") or "").strip()),
                    "api_calls": getattr(agent, "n_calls", None),
                }
            )
        except Exception:  # noqa: BLE001
            pass
        return info


def swebench_image(instance_id: str) -> str:
    iid = instance_id.replace("__", "_1776_")
    return f"docker.io/swebench/sweb.eval.x86_64.{iid}:latest".lower()


def write_pred(output: Path, iid: str, model_name: str, patch: str) -> None:
    with _LOCK:
        data = json.loads(output.read_text()) if output.exists() else {}
        data[iid] = {"model_name_or_path": model_name, "instance_id": iid, "model_patch": patch}
        output.write_text(json.dumps(data, indent=2))


def save_trajectory(traj_dir: Path, iid: str, agent) -> None:
    """sandbox 세션 로그 = agent 대화(각 스텝의 명령/출력/추론)를 인스턴스별 파일로 저장.

    sandbox Pod 자체는 sleep 이라 pod 로그가 비어 있고, 실제 sandbox 안 활동(실행한 bash
    명령과 그 출력)은 exec 응답=agent.messages 에 담긴다. 이를 저장해 각 sandbox 를 사후에
    그대로 들여다볼 수 있게 한다.
    """
    msgs = getattr(agent, "messages", None)
    if not msgs:
        return
    traj_dir.mkdir(parents=True, exist_ok=True)
    (traj_dir / f"{iid}.json").write_text(json.dumps(msgs, indent=2, default=str, ensure_ascii=False))


def solve(inst: dict, args, base_url: str, mlctx=None, evc: SandboxClient | None = None) -> dict:
    """한 인스턴스 실행 → dict(iid,status,api_calls,job_id,child_rid). 예외는 삼켜 배치가 안 죽게.

    patch 가 나오면 즉시 /eval 에 비동기 제출(job_id)하고, 결과는 main 이 나중에 폴링한다.
    """
    iid = inst["instance_id"]
    image = swebench_image(iid)
    env = agent = None
    patch, status, ncalls, job_id = "", "done", 0, ""
    try:
        config = recursive_merge(
            get_config_from_spec("swebench"),
            {
                "agent": {"step_limit": args.step_limit},
                "model": {
                    "model_name": args.model,
                    "model_kwargs": {"temperature": 0.0, "timeout": 300, "max_tokens": 2048},
                    # MiniMax 등은 litellm 가격맵에 없어 비용계산이 실패 → 에러 무시(비용 0 처리).
                    "cost_tracking": "ignore_errors",
                },
            },
        )
        env = RebelSandboxEnvironment(
            image=image,
            base_url=base_url,
            cwd="/testbed",
            env={"BASH_ENV": "/root/.bashrc", "PAGER": "cat"},
        )
        agent = DefaultAgent(LitellmModel(**config["model"]), env, **config["agent"])
        info = _run_traced(mlctx, iid, agent, inst["problem_statement"])
        patch = info.get("submission") or ""
        status = info.get("exit_status") or "done"
        ncalls = agent.n_calls
        write_pred(args.output, iid, args.model, patch)
        # patch 생성 직후 채점 비동기 제출(결과는 기다리지 않음 → main 이 폴링).
        if evc is not None and patch.strip():
            try:
                job_id = evc.eval(iid, patch, run_id=args.eval_run_id)
            except Exception as e:  # noqa: BLE001 — eval 제출 실패가 배치를 막지 않게
                print(f"[eval submit 실패] {iid}: {type(e).__name__}: {str(e)[:120]}", flush=True)
    except Exception as e:  # noqa: BLE001 — per-instance 격리
        status = f"ERROR: {type(e).__name__}: {str(e)[:200]}"
        write_pred(args.output, iid, args.model, "")
    finally:
        if agent is not None:
            try:
                save_trajectory(args.output.parent / "trajectories", iid, agent)
            except Exception:  # noqa: BLE001 — 로그 저장 실패가 배치를 막지 않게
                pass
        if env is not None:
            env.cleanup()
    # child run 생성(열어둠 → main 이 eval resolved 기록 후 종료).
    child_rid = _mlflow_child(mlctx, iid, image, args.model, patch, ncalls, status, agent)
    return {"iid": iid, "status": status, "ncalls": ncalls, "job_id": job_id, "child_rid": child_rid}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=int(os.getenv("WORKERS", "50")))
    ap.add_argument("--n", type=int, default=0, help="처음 N개만 (0=전체)")
    ap.add_argument("--model", default=os.getenv("MODEL_NAME", "openai/MiniMaxAI/MiniMax-M2.5"))
    ap.add_argument("--step-limit", type=int, default=int(os.getenv("STEP_LIMIT", "100")))
    ap.add_argument("--output", type=Path, default=Path(os.getenv("OUTPUT", "preds.json")))
    ap.add_argument(
        "--eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="patch 를 /eval 에 비동기 제출해 채점(기본 on). --no-eval 로 끔.",
    )
    ap.add_argument(
        "--eval-timeout",
        type=int,
        default=int(os.getenv("EVAL_TIMEOUT", "1800")),
        help="eval 결과 폴링 최대 대기(초)",
    )
    args = ap.parse_args()
    args.eval_run_id = os.getenv("EVAL_RUN_ID", "lite-batch")

    if not os.getenv("OPENAI_API_BASE"):
        ap.error("OPENAI_API_BASE 가 필요합니다 (모델 엔드포인트)")
    base_url = os.getenv("REBEL_SANDBOX_URL", "http://localhost:8080")
    evc = SandboxClient(base_url) if args.eval else None

    from datasets import load_dataset

    instances = [dict(r) for r in load_dataset(DATASET, split=SPLIT)]
    if args.n:
        instances = instances[: args.n]

    # 재개: 비어있지 않은 patch 가 이미 있는 인스턴스만 건너뜀(에러로 빈 patch 는 재시도).
    done = (
        {k for k, v in json.loads(args.output.read_text()).items() if (v.get("model_patch") or "").strip()}
        if args.output.exists()
        else set()
    )
    todo = [i for i in instances if i["instance_id"] not in done]
    print(
        f"[batch] dataset={DATASET}:{SPLIT} total={len(instances)} todo={len(todo)} "
        f"done={len(done)} workers={args.workers} model={args.model} step_limit={args.step_limit}",
        flush=True,
    )

    mlf = mlflow_helper.init(EXPERIMENT)
    t0 = time.time()
    submitted = errors = 0
    with mlflow_helper.run(mlf, run_name=f"lite-batch {args.model} n={len(todo)}") as parent:
        mlctx = None
        if mlf and parent is not None:
            from mlflow.tracking import MlflowClient

            mlf.log_params(
                {
                    "dataset": DATASET,
                    "split": SPLIT,
                    "workers": args.workers,
                    "model": args.model,
                    "step_limit": args.step_limit,
                    "n": args.n,
                    "total": len(instances),
                    "todo": len(todo),
                    "resumed_done": len(done),
                }
            )
            mlctx = {
                "client": MlflowClient(),
                "parent_run_id": parent.info.run_id,
                "experiment_id": parent.info.experiment_id,
                "mlflow": mlf,  # instance 별 multi-turn trace(span) 용
            }
        results = []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(solve, inst, args, base_url, mlctx, evc): inst["instance_id"] for inst in todo}
            for k, fut in enumerate(as_completed(futs), 1):
                res = fut.result()
                results.append(res)
                ok = not res["status"].startswith("ERROR")
                submitted += ok
                errors += not ok
                print(
                    f"[{k}/{len(todo)}] {res['iid']} :: {res['status']} (api_calls={res['ncalls']}) "
                    f"| ok={submitted} err={errors} | {time.time() - t0:.0f}s",
                    flush=True,
                )
        dur = time.time() - t0

        # ── eval 결과 비동기 폴링 → child run 에 resolved 기록 ──────────────────
        client = mlctx["client"] if mlctx else None
        resolved_count = eval_submitted = 0
        if evc is not None:
            pending = {r["job_id"]: r for r in results if r["job_id"]}
            eval_submitted = len(pending)
            print(f"[eval] {eval_submitted}건 제출 — 결과 폴링(최대 {args.eval_timeout}s)...", flush=True)
            end = time.time() + args.eval_timeout
            while pending and time.time() < end:
                for jid in list(pending):
                    try:
                        st = evc.eval_status(jid)
                    except Exception:  # noqa: BLE001 — 일시 오류는 다음 폴링에 재시도
                        continue
                    if st.get("state") in ("SUCCESS", "FAILURE", "REVOKED"):
                        r = pending.pop(jid)
                        resolved = bool(st.get("resolved"))
                        resolved_count += int(resolved)
                        print(f"[eval] {r['iid']} :: {st.get('state')} resolved={resolved}", flush=True)
                        if client and r["child_rid"]:
                            client.log_metric(r["child_rid"], "resolved", int(resolved))
                            client.set_tag(r["child_rid"], "eval_state", st.get("state") or "")
                if pending:
                    time.sleep(5)
            if pending:
                print(f"[eval] timeout — 미완 {len(pending)}건 (state 미확정)", flush=True)

        # child run 종료(전체). eval resolved 기록 후 마감.
        if client:
            for r in results:
                if r["child_rid"]:
                    client.set_terminated(
                        r["child_rid"], "FINISHED" if not r["status"].startswith("ERROR") else "FAILED"
                    )
        if mlf:
            mlf.log_metric("total_instances", len(todo))
            mlf.log_metric("submitted", submitted)
            mlf.log_metric("errors", errors)
            mlf.log_metric("submit_rate", round(submitted / max(len(todo), 1), 4))
            mlf.log_metric("duration_s", round(dur, 1))
            mlf.log_metric("throughput_per_min", round(len(todo) / max(dur, 1) * 60, 2))
            if evc is not None:
                mlf.log_metric("eval_submitted", eval_submitted)
                mlf.log_metric("resolved_count", resolved_count)
                # resolve 비율: 채점 제출 대비(=패치 생성분) 해결 비율.
                mlf.log_metric("resolve_rate", round(resolved_count / max(eval_submitted, 1), 4))
    tail = f" | eval resolved={resolved_count}/{eval_submitted}" if evc is not None else ""
    print(
        f"\n[done] {len(todo)} run in {time.time() - t0:.0f}s | submitted={submitted} errors={errors}"
        f"{tail} | preds → {args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
