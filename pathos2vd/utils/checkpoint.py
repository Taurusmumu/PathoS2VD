from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_state_dict(module: torch.nn.Module, path: str | Path, *, strict: bool = True) -> None:
    state = torch.load(str(path), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    module.load_state_dict(state, strict=strict)


def checkpoint_size_bytes(path: str | Path) -> int | None:
    target = Path(path)
    if not target.exists():
        return None
    if target.is_file():
        return target.stat().st_size
    return sum(item.stat().st_size for item in target.rglob("*") if item.is_file())


def save_metadata(path: str | Path, metadata: dict[str, Any]) -> None:
    import json

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
