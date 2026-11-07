"""Run generated AIDE programs in a constrained, secret-free Docker container."""

from __future__ import annotations

import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from aide.interpreter import ExecutionResult


@dataclass(frozen=True)
class ExecutorLimits:
    cpus: float = 2.0
    memory: str = "4g"
    pids: int = 256
    timeout_seconds: int = 120
    tmpfs_size: str = "512m"
    max_output_chars: int = 100_000


class DockerExecutor:
    def __init__(
        self,
        working_dir: str | Path,
        *,
        image: str = "nir-stage0-executor:latest",
        limits: ExecutorLimits | None = None,
    ) -> None:
        self.working_dir = Path(working_dir).resolve()
        if not self.working_dir.is_dir():
            raise ValueError(f"working directory does not exist: {self.working_dir}")
        self.image = image
        self.limits = limits or ExecutorLimits()

    def _command(self, container_name: str, script_name: str) -> list[str]:
        limits = self.limits
        return [
            "docker",
            "run",
            "--rm",
            "--name",
            container_name,
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            str(limits.pids),
            "--cpus",
            str(limits.cpus),
            "--memory",
            limits.memory,
            "--user",
            "1000:1000",
            "--tmpfs",
            f"/tmp:rw,nosuid,nodev,noexec,size={limits.tmpfs_size}",
            "--volume",
            f"{self.working_dir}:/workspace:rw",
            "--workdir",
            "/workspace",
            "--entrypoint",
            "python",
            self.image,
            script_name,
        ]

    @staticmethod
    def _text(value: str | bytes | None) -> str:
        if value is None:
            return ""
        return value.decode(errors="replace") if isinstance(value, bytes) else value

    def __call__(self, code: str, reset_session: bool = True) -> ExecutionResult:
        del reset_session  # every container is an intentionally fresh session
        run_id = uuid.uuid4().hex
        script_name = f".stage0-{run_id}.py"
        script_path = self.working_dir / script_name
        container_name = f"nir-stage0-{run_id[:16]}"
        script_path.write_text(code, encoding="utf-8")

        started = time.monotonic()
        try:
            completed = subprocess.run(
                self._command(container_name, script_name),
                capture_output=True,
                text=True,
                timeout=self.limits.timeout_seconds,
                check=False,
            )
            elapsed = time.monotonic() - started
            output = (completed.stdout or "") + (completed.stderr or "")
            output = output[-self.limits.max_output_chars :]
            if completed.returncode == 0:
                output += f"\nExecution time: {elapsed:.3f} seconds."
                return ExecutionResult([output], elapsed, None, None, [])
            output += f"\nContainer exited with status {completed.returncode}."
            return ExecutionResult(
                [output],
                elapsed,
                "ContainerExecutionError",
                {"returncode": completed.returncode},
                [],
            )
        except subprocess.TimeoutExpired as exc:
            elapsed = time.monotonic() - started
            subprocess.run(
                ["docker", "rm", "--force", container_name],
                capture_output=True,
                text=True,
                check=False,
            )
            output = self._text(exc.stdout) + self._text(exc.stderr)
            output = output[-self.limits.max_output_chars :]
            output += (
                f"\nTimeoutError: execution exceeded "
                f"{self.limits.timeout_seconds} seconds."
            )
            return ExecutionResult(
                [output],
                elapsed,
                "TimeoutError",
                {"timeout_seconds": self.limits.timeout_seconds},
                [],
            )
        finally:
            script_path.unlink(missing_ok=True)
