from __future__ import annotations

from typing import Optional


def try_import(name: str) -> Optional[object]:
    try:
        return __import__(name)
    except Exception:
        return None

