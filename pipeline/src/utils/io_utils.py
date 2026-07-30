# -*- coding: utf-8 -*-
"""JSON/JSONL, logging, and resumable-checkpoint utilities."""
import json
import logging
import os
from pathlib import Path
from typing import Any


def setup_logging(level: int = logging.INFO, log_file: str | None = None):
    """Configure process-wide logging."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def read_jsonl(path: str | Path) -> list[dict]:
    """Read JSONL records; return an empty list for a missing file."""
    path = Path(path)
    if not path.exists():
        return []
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def write_jsonl(data: list[dict], path: str | Path, mode: str = "w"):
    """Write JSONL records, optionally in append mode."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode, encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def append_jsonl(data: list[dict], path: str | Path):
    """Append JSONL records."""
    write_jsonl(data, path, mode="a")


def read_json(path: str | Path) -> Any:
    """Read one JSON value."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Any, path: str | Path):
    """Write one pretty-printed JSON value."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


class Checkpoint:
    """Small JSON checkpoint supporting interrupted-run recovery."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._state: dict = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._state, f, ensure_ascii=False, indent=2)

    def get(self, key: str, default: Any = None) -> Any:
        return self._state.get(key, default)

    def set(self, key: str, value: Any):
        self._state[key] = value
        self.save()

    def increment(self, key: str, amount: int = 1) -> int:
        current = self._state.get(key, 0)
        self._state[key] = current + amount
        self.save()
        return self._state[key]
