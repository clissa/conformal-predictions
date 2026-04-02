"""Utilities to mirror terminal output into per-run log files."""

from __future__ import annotations

import atexit
import sys
from pathlib import Path
from typing import TextIO


class _TeeStream:
    """Write stream content to both the terminal and a log file."""

    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self._streams)


_LOGGING_CONFIGURED = False


def configure_script_logging(
    config_path: str | Path,
    model_name: str,
    script_path: str | Path,
) -> Path:
    """Mirror stdout/stderr into ``logs/<config_folder>/<model_name>/<script>.log``."""
    global _LOGGING_CONFIGURED

    if _LOGGING_CONFIGURED:
        raise RuntimeError("Script logging has already been configured.")

    config_stem = Path(config_path).stem
    script_stem = Path(script_path).stem
    log_path = Path("logs") / config_stem / model_name / f"{script_stem}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    log_file = log_path.open("a", encoding="utf-8", buffering=1)
    atexit.register(log_file.close)

    sys.stdout = _TeeStream(sys.__stdout__, log_file)
    sys.stderr = _TeeStream(sys.__stderr__, log_file)
    _LOGGING_CONFIGURED = True

    print(f"[Logging] Mirroring terminal output to {log_path}")
    return log_path
