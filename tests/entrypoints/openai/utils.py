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

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import anthropic
import httpx
import openai
import requests
from vllm.engine.arg_utils import AsyncEngineArgs
# from tests.models.utils import TextTextLogprobs
from vllm.entrypoints.cli.serve import ServeSubcommand
from vllm.model_executor.model_loader import get_model_loader
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.network_utils import get_open_port

VLLM_PATH = Path(__file__).parent.parent
"""Path to root of the vLLM repository."""


class RemoteOpenAIServer:
    DUMMY_API_KEY = "token-abc123"  # vLLM's OpenAI server does not need API key

    def _start_server(self, model: str, vllm_serve_args: list[str],
                      env_dict: dict[str, str] | None) -> None:
        """Subclasses override this method to customize server process launch"""
        env = os.environ.copy()
        # the current process might initialize cuda,
        # to be safe, we should use spawn method
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        if env_dict is not None:
            env.update(env_dict)
        serve_cmd = ["vllm", "serve", model, *vllm_serve_args]
        print(f"Launching RemoteOpenAIServer with: {' '.join(serve_cmd)}")
        self.proc: subprocess.Popen = subprocess.Popen(
            serve_cmd,
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

    def __init__(
        self,
        model: str,
        vllm_serve_args: list[str],
        *,
        env_dict: dict[str, str] | None = None,
        seed: int = 0,
        auto_port: bool = True,
        max_wait_seconds: float | None = None,
        override_hf_configs: dict[str, Any] | None = None,
    ) -> None:
        if auto_port:
            if "-p" in vllm_serve_args or "--port" in vllm_serve_args:
                raise ValueError("You have manually specified "
                                 "the port when `auto_port=True`.")

            # No need for a port if using unix sockets
            if "--uds" not in vllm_serve_args:
                # Don't mutate the input args
                vllm_serve_args = vllm_serve_args + [
                    "--port", str(get_open_port())
                ]
        if seed is not None:
            if "--seed" in vllm_serve_args:
                raise ValueError(
                    f"You have manually specified the seed when `seed={seed}`."
                )

            vllm_serve_args = vllm_serve_args + ["--seed", str(seed)]

        if override_hf_configs is not None:
            vllm_serve_args = vllm_serve_args + [
                "--hf-overrides",
                json.dumps(override_hf_configs),
            ]

        parser = FlexibleArgumentParser(
            description="vLLM's remote OpenAI server.")
        subparsers = parser.add_subparsers(required=False, dest="subparser")
        parser = ServeSubcommand().subparser_init(subparsers)
        args = parser.parse_args(["--model", model, *vllm_serve_args])
        self.uds = args.uds
        if args.uds:
            self.host = None
            self.port = None
        else:
            self.host = str(args.host or "127.0.0.1")
            self.port = int(args.port)

        self.show_hidden_metrics = \
            args.show_hidden_metrics_for_version is not None

        # download the model before starting the server to avoid timeout
        is_local = os.path.isdir(model)
        if not is_local:
            engine_args = AsyncEngineArgs.from_cli_args(args)
            model_config = engine_args.create_model_config()
            load_config = engine_args.create_load_config()

            model_loader = get_model_loader(load_config)
            model_loader.download_model(model_config)

        self._start_server(model, vllm_serve_args, env_dict)
        max_wait_seconds = max_wait_seconds or 240
        self._wait_for_server(url=self.url_for("health"),
                              timeout=max_wait_seconds)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.proc.terminate()
        try:
            self.proc.wait(8)
        except subprocess.TimeoutExpired:
            # force kill if needed
            self.proc.kill()

    def _poll(self) -> int | None:
        """Subclasses override this method to customize process polling"""
        return self.proc.poll()

    def _wait_for_server(self, *, url: str, timeout: float):
        # run health check
        start = time.time()
        client = (httpx.Client(transport=httpx.HTTPTransport(
            uds=self.uds)) if self.uds else requests)
        while True:
            try:
                if client.get(url).status_code == 200:
                    break
            except Exception:
                # this exception can only be raised by requests.get,
                # which means the server is not ready yet.
                # the stack trace is not useful, so we suppress it
                # by using `raise from None`.
                result = self._poll()
                if result is not None and result != 0:
                    raise RuntimeError("Server exited unexpectedly.") from None

                time.sleep(0.5)
                if time.time() - start > timeout:
                    raise RuntimeError(
                        "Server failed to start in time.") from None

    @property
    def url_root(self) -> str:
        return (f"http://{self.uds.split('/')[-1]}"
                if self.uds else f"http://{self.host}:{self.port}")

    def url_for(self, *parts: str) -> str:
        return self.url_root + "/" + "/".join(parts)

    def get_client(self, **kwargs):
        if "timeout" not in kwargs:
            kwargs["timeout"] = 600
        return openai.OpenAI(
            base_url=self.url_for("v1"),
            api_key=self.DUMMY_API_KEY,
            max_retries=0,
            **kwargs,
        )

    def get_async_client(self, **kwargs):
        if "timeout" not in kwargs:
            kwargs["timeout"] = 600
        return openai.AsyncOpenAI(
            base_url=self.url_for("v1"),
            api_key=self.DUMMY_API_KEY,
            max_retries=0,
            **kwargs,
        )

    def get_client_anthropic(self, **kwargs):
        if "timeout" not in kwargs:
            kwargs["timeout"] = 600
        return anthropic.Anthropic(
            base_url=self.url_for(),
            api_key=self.DUMMY_API_KEY,
            max_retries=0,
            **kwargs,
        )

    def get_async_client_anthropic(self, **kwargs):
        if "timeout" not in kwargs:
            kwargs["timeout"] = 600
        return anthropic.AsyncAnthropic(base_url=self.url_for(),
                                        api_key=self.DUMMY_API_KEY,
                                        max_retries=0,
                                        **kwargs)
