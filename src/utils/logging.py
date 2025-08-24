from __future__ import annotations

import json
import sys
from datetime import datetime, timezone


class _Logger:
    def _log(self, level: str, msg: str, **kw):
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "msg": msg,
            **kw,
        }
        sys.stdout.write(json.dumps(rec) + "\n")

    def debug(self, msg: str, **kw):
        self._log("DEBUG", msg, **kw)

    def info(self, msg: str, **kw):
        self._log("INFO", msg, **kw)

    def warning(self, msg: str, **kw):
        self._log("WARNING", msg, **kw)

    def error(self, msg: str, **kw):
        self._log("ERROR", msg, **kw)


log = _Logger()
