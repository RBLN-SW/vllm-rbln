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
"""Logging configuration for vLLM-RBLN Plugin."""

import json
import logging
from logging.config import dictConfig
from os import path
from typing import Any

from vllm import envs
from vllm.logger import _DATE_FORMAT, _use_color, init_logger

_FORMAT = "[vllm-rbln] %(levelname)s %(asctime)s [%(fileinfo)s:%(lineno)d] %(message)s"
DEFAULT_LOGGIN_CONFIG: dict[str, dict[str, Any] | Any] = {
    "formatters": {
        "vllm_rbln": {
            "class": "vllm.logging_utils.NewLineFormatter",
            "datefmt": _DATE_FORMAT,
            "format": _FORMAT,
        },
        "vllm_rbln_color": {
            "class": "vllm.logging_utils.ColoredFormatter",
            "datefmt": _DATE_FORMAT,
            "format": _FORMAT,
        },
    },
    "handlers": {
        "vllm_rbln": {
            "class": "logging.StreamHandler",
            # Choose formatter based on color setting.
            "formatter": "vllm_rbln_color" if _use_color() else "vllm_rbln",
            "level": envs.VLLM_LOGGING_LEVEL,
            "stream": envs.VLLM_LOGGING_STREAM,
        },
    },
    "loggers": {
        "vllm_rbln": {
            "handlers": ["vllm_rbln"],
            "level": envs.VLLM_LOGGING_LEVEL,
            "propagate": False,
        },
    },
    "version": 1,
    "disable_existing_loggers": False,
}


def _configure_vllm_root_logger() -> None:
    logging_config: dict[str, dict[str, Any] | Any] = {}

    if not envs.VLLM_CONFIGURE_LOGGING and envs.VLLM_LOGGING_CONFIG_PATH:
        raise RuntimeError(
            "VLLM_CONFIGURE_LOGGING evaluated to false, but "
            "VLLM_LOGGING_CONFIG_PATH was given. VLLM_LOGGING_CONFIG_PATH "
            "implies VLLM_CONFIGURE_LOGGING. Please enable "
            "VLLM_CONFIGURE_LOGGING or unset VLLM_LOGGING_CONFIG_PATH."
        )

    if envs.VLLM_CONFIGURE_LOGGING:
        logging_config = DEFAULT_LOGGIN_CONFIG

        vllm_handler = logging_config["handlers"]["vllm_rbln"]
        # Refresh these values in case env vars have changed.
        vllm_handler["level"] = envs.VLLM_LOGGING_LEVEL
        vllm_handler["stream"] = envs.VLLM_LOGGING_STREAM
        vllm_handler["formatter"] = "vllm_rbln_color" if _use_color() else "vllm_rbln"

        vllm_loggers = logging_config["loggers"]["vllm_rbln"]
        vllm_loggers["level"] = envs.VLLM_LOGGING_LEVEL

    if envs.VLLM_LOGGING_CONFIG_PATH:
        if not path.exists(envs.VLLM_LOGGING_CONFIG_PATH):
            raise RuntimeError(
                "Could not load logging config. File does not exist: %s",
                envs.VLLM_LOGGING_CONFIG_PATH,
            )
        with open(envs.VLLM_LOGGING_CONFIG_PATH, encoding="utf-8") as file:
            custom_config = json.loads(file.read())

        if not isinstance(custom_config, dict):
            raise ValueError(
                "Invalid logging config. Expected dict, got %s.",
                type(custom_config).__name__,
            )
        logging_config = custom_config

    for formatter in logging_config.get("formatters", {}).values():
        # This provides backwards compatibility after #10134.
        if formatter.get("class") == "vllm.logging.NewLineFormatter":
            formatter["class"] = "vllm.logging_utils.NewLineFormatter"

    if logging_config:
        dictConfig(logging_config)


# The root logger is initialized when the module is imported.
# This is thread-safe as the module is only imported once,
# guaranteed by the Python GIL.
_configure_vllm_root_logger()

# Transformers uses httpx to access the Hugging Face Hub. httpx is quite verbose,
# so we set its logging level to WARNING when vLLM's logging level is INFO.
if envs.VLLM_LOGGING_LEVEL == "INFO":
    logging.getLogger("httpx").setLevel(logging.WARNING)

logger = init_logger(__name__)
