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
from copy import deepcopy
from logging.config import dictConfig
from os import path
from typing import TYPE_CHECKING, Any
from weakref import WeakSet

import vllm.logger as vllm_logger
from vllm import envs
from vllm.logger import _DATE_FORMAT, _use_color, init_logger
from vllm.logging_utils import ColoredFormatter, NewLineFormatter

if TYPE_CHECKING:
    from vllm.config.logging import LoggingConfig

_FORMAT = "[vllm-rbln] %(levelname)s %(asctime)s [%(fileinfo)s:%(lineno)d] %(message)s"
DEFAULT_LOGGIN_CONFIG: dict[str, dict[str, Any] | Any] = {
    "formatters": {
        "vllm_rbln": {
            "()": "vllm.logging_utils.NewLineFormatter",
            "datefmt": _DATE_FORMAT,
            "format": _FORMAT,
            "log_level": envs.VLLM_LOGGING_LEVEL,
        },
        "vllm_rbln_color": {
            "()": "vllm.logging_utils.ColoredFormatter",
            "datefmt": _DATE_FORMAT,
            "format": _FORMAT,
            "log_level": envs.VLLM_LOGGING_LEVEL,
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

_console_handler: logging.Handler | None = None
_file_handlers: WeakSet[logging.FileHandler] = WeakSet()


def make_file_handler(file_path: str) -> logging.FileHandler:
    """Create a file handler using the same format as the stdout handler.

    Args:
        file_path: Destination path for the log file.

    Returns:
        A :class:`logging.FileHandler` whose output matches console output.

    Example:
        >>> logger.addHandler(make_file_handler("/tmp/metrics.log"))
    """
    handler = logging.FileHandler(file_path)
    level = logging.getLevelName(logging.getLogger("vllm_rbln").getEffectiveLevel())
    handler.setFormatter(NewLineFormatter(_FORMAT, log_level=level))
    _file_handlers.add(handler)
    return handler


def _configure_vllm_root_logger() -> None:
    global _console_handler
    logging_config: dict[str, dict[str, Any] | Any] = {}

    if not envs.VLLM_CONFIGURE_LOGGING and envs.VLLM_LOGGING_CONFIG_PATH:
        raise RuntimeError(
            "VLLM_CONFIGURE_LOGGING evaluated to false, but "
            "VLLM_LOGGING_CONFIG_PATH was given. VLLM_LOGGING_CONFIG_PATH "
            "implies VLLM_CONFIGURE_LOGGING. Please enable "
            "VLLM_CONFIGURE_LOGGING or unset VLLM_LOGGING_CONFIG_PATH."
        )

    if envs.VLLM_CONFIGURE_LOGGING:
        logging_config = deepcopy(DEFAULT_LOGGIN_CONFIG)

        vllm_handler = logging_config["handlers"]["vllm_rbln"]
        # Refresh these values in case env vars have changed.
        vllm_handler["level"] = envs.VLLM_LOGGING_LEVEL
        vllm_handler["stream"] = envs.VLLM_LOGGING_STREAM
        vllm_handler["formatter"] = "vllm_rbln_color" if _use_color() else "vllm_rbln"

        vllm_loggers = logging_config["loggers"]["vllm_rbln"]
        vllm_loggers["level"] = envs.VLLM_LOGGING_LEVEL
        for formatter in logging_config["formatters"].values():
            formatter["log_level"] = envs.VLLM_LOGGING_LEVEL

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
        _console_handler = (
            logging.getLogger("vllm_rbln").handlers[0]
            if not envs.VLLM_LOGGING_CONFIG_PATH
            else None
        )


def _sync_logging_config(config: "LoggingConfig | None") -> None:
    global _console_handler
    enabled = config.configure_logging if config else envs.VLLM_CONFIGURE_LOGGING
    config_file = (
        config.pylogging_config_file if config else envs.VLLM_LOGGING_CONFIG_PATH
    )
    if not enabled or config_file:
        return

    level = config.log_level if config else envs.VLLM_LOGGING_LEVEL
    root = logging.getLogger("vllm_rbln")
    root.setLevel(level)
    handler = _console_handler
    if handler is None or handler not in root.handlers:
        # Bootstrap can be disabled even when the parsed config enables logging.
        upstream_handler = logging.getLogger("vllm").handlers[0]
        assert isinstance(upstream_handler, logging.StreamHandler)
        handler = logging.StreamHandler(upstream_handler.stream)
        for installed in tuple(root.handlers):
            if installed not in _file_handlers:
                root.removeHandler(installed)
                installed.close()

    # dictConfig clears name registrations even for handlers it leaves attached.
    handler.name = "vllm_rbln"
    handler.setLevel(level)
    formatter_cls = ColoredFormatter if _use_color() else NewLineFormatter
    handler.setFormatter(formatter_cls(_FORMAT, datefmt=_DATE_FORMAT, log_level=level))
    root.addHandler(handler)
    root.propagate = False
    _console_handler = handler
    for file_handler in tuple(_file_handlers):
        file_handler.setFormatter(NewLineFormatter(_FORMAT, log_level=level))


# Platform discovery can run while this module is importing vLLM.
# Synchronize only after the RBLN logging functions and handlers are initialized.
_configure_vllm_root_logger()
_sync_logging_config(vllm_logger._last_configured_logging_config)

# Transformers uses httpx to access the Hugging Face Hub. httpx is quite verbose,
# so we set its logging level to WARNING when vLLM's logging level is INFO.
if envs.VLLM_LOGGING_LEVEL == "INFO":
    logging.getLogger("httpx").setLevel(logging.WARNING)

logger = init_logger(__name__)
