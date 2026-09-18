"""`ViewerManager` owns the one game-viewer child process (Phase 8) and the current view
settings. The viewer is its own process because `arcade` runs its event loop on the process's
main thread, which uvicorn already owns. See
`../../../docs/phases/phase-8-dashboard-game-viewer.md`."""

from __future__ import annotations

import subprocess
import sys
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

from neuroarena.render.view_settings import ViewSettings


class _Process(Protocol):
    def poll(self) -> int | None: ...
    def terminate(self) -> None: ...
    def kill(self) -> None: ...
    def wait(self, timeout: float | None = None) -> int: ...


def _spawn(args: list[str]) -> _Process:
    return subprocess.Popen(args)


class ViewerManager:
    def __init__(
        self,
        *,
        url: str,
        data_dir: Path,
        spawn: Callable[[list[str]], _Process] = _spawn,
        terminate_timeout_s: float = 5.0,
    ) -> None:
        self._url = url
        self._data_dir = data_dir
        self._spawn = spawn
        self._terminate_timeout_s = terminate_timeout_s
        self._lock = threading.Lock()
        self._process: _Process | None = None
        self._settings = ViewSettings()

    def open(self) -> bool:
        """Spawns the viewer; `False` (and does nothing) if one is already running."""
        with self._lock:
            if self._running_locked():
                return False
            self._process = self._spawn(
                [
                    sys.executable,
                    "-m",
                    "neuroarena.render.viewer",
                    "--url",
                    self._url,
                    "--data-dir",
                    str(self._data_dir),
                ]
            )
            return True

    def close(self) -> bool:
        """Terminates the viewer (kills it if it ignores `terminate`); `False` if none was
        running."""
        with self._lock:
            process = self._process
            self._process = None
            if process is None or process.poll() is not None:
                return False
            process.terminate()
            try:
                process.wait(timeout=self._terminate_timeout_s)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            return True

    def is_open(self) -> bool:
        with self._lock:
            return self._running_locked()

    def shutdown(self) -> None:
        self.close()

    def settings(self) -> ViewSettings:
        with self._lock:
            return self._settings

    def update_settings(self, partial: Mapping[str, Any]) -> ViewSettings:
        with self._lock:
            self._settings = self._settings.updated(partial)  # raises ValueError, unchanged
            return self._settings

    def _running_locked(self) -> bool:
        return self._process is not None and self._process.poll() is None
