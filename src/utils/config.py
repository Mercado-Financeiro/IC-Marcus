from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass
class Settings:
    mlflow_uri: str = os.environ.get("MLFLOW_TRACKING_URI", "artifacts/mlruns")


settings = Settings()


def load_yaml(path: str | Path) -> dict:
    """Load YAML if PyYAML is available; else return empty dict.

    Falls back to empty dict so CLI can run offline in degraded mode.
    """
    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

