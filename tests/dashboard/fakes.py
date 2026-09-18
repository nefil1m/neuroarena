"""Test doubles for the viewer child process — no real process is ever spawned in tests."""

from __future__ import annotations

import subprocess


class FakeProcess:
    def __init__(self, args: list[str], *, stubborn: bool = False) -> None:
        self.args = args
        self.terminated = False
        self.killed = False
        self._stubborn = stubborn
        self._returncode: int | None = None

    def poll(self) -> int | None:
        return self._returncode

    def terminate(self) -> None:
        self.terminated = True
        if not self._stubborn:
            self._returncode = -15

    def kill(self) -> None:
        self.killed = True
        self._returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        if self._returncode is None:
            raise subprocess.TimeoutExpired(cmd=self.args, timeout=timeout or 0.0)
        return self._returncode

    def die(self) -> None:
        """Simulates the window being closed by hand / the process crashing."""
        self._returncode = 0


class FakeSpawner:
    def __init__(self, *, stubborn: bool = False) -> None:
        self.processes: list[FakeProcess] = []
        self._stubborn = stubborn

    def __call__(self, args: list[str]) -> FakeProcess:
        process = FakeProcess(args, stubborn=self._stubborn)
        self.processes.append(process)
        return process
