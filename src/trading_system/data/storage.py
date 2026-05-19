from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from trading_system.config import settings


DatasetKind = Literal["market", "fundamentals", "signals", "backtests"]


@dataclass(frozen=True)
class LocalDataStore:
    """Simple filesystem-backed store for reproducible research datasets."""

    root: Path = settings.data_dir

    def save_frame(self, frame: pd.DataFrame, kind: DatasetKind, name: str) -> Path:
        path = self._dataset_path(kind, name, suffix=".parquet")
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path)
        return path

    def load_frame(self, kind: DatasetKind, name: str) -> pd.DataFrame:
        path = self._dataset_path(kind, name, suffix=".parquet")
        if not path.exists():
            raise FileNotFoundError(f"Dataset does not exist: {path}")
        return pd.read_parquet(path)

    def frame_exists(self, kind: DatasetKind, name: str) -> bool:
        return self._dataset_path(kind, name, suffix=".parquet").exists()

    def save_json(self, payload: dict[str, Any], kind: DatasetKind, name: str) -> Path:
        path = self._dataset_path(kind, name, suffix=".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def load_json(self, kind: DatasetKind, name: str) -> dict[str, Any]:
        path = self._dataset_path(kind, name, suffix=".json")
        if not path.exists():
            raise FileNotFoundError(f"JSON payload does not exist: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _dataset_path(self, kind: DatasetKind, name: str, suffix: str) -> Path:
        safe_name = name.replace("/", "_").replace("\\", "_").replace(":", "_")
        return self.root / kind / f"{safe_name}{suffix}"
