from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Mapping

import yaml


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


def _expand_string(value: str) -> str:
    """Expand ``${NAME}`` and ``${NAME:-default}`` without hiding missing vars."""

    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        if name in os.environ:
            return os.environ[name]
        if default is not None:
            return default
        raise KeyError(f"Required environment variable {name!r} is not set")

    return os.path.expanduser(_ENV_PATTERN.sub(replace, value))


def expand_config(value: Any) -> Any:
    if isinstance(value, str):
        return _expand_string(value)
    if isinstance(value, list):
        return [expand_config(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_config(item) for key, item in value.items()}
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, Mapping):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return expand_config(dict(config))
