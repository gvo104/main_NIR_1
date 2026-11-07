from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from nir_stage0.docker_executor import DockerExecutor, ExecutorLimits

pytestmark = pytest.mark.docker


def _image_available(image):
    if shutil.which("docker") is None:
        return False
    result = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


@pytest.fixture
def image():
    name = os.environ.get("NIR_EXECUTOR_IMAGE", "nir-stage0-executor:latest")
    if not _image_available(name):
        pytest.skip(f"Docker image is unavailable: {name}")
    return name


def test_success_failure_secret_isolation_and_network_block(
    tmp_path, image, monkeypatch
):
    monkeypatch.setenv("lite_llm_token", "must-not-enter-container")
    executor = DockerExecutor(tmp_path, image=image)

    success = executor("""
import os
import socket

assert os.getenv("lite_llm_token") is None
assert os.getenv("OPENAI_API_KEY") is None
try:
    socket.create_connection(("1.1.1.1", 53), timeout=1)
except OSError:
    print("network blocked")
else:
    raise RuntimeError("network unexpectedly available")
""")
    assert success.exc_type is None
    assert "network blocked" in "".join(success.term_out)

    failure = executor("raise RuntimeError('expected failure')")
    assert failure.exc_type == "ContainerExecutionError"
    assert failure.exc_info["returncode"] != 0


def test_timeout_removes_container(tmp_path, image):
    executor = DockerExecutor(
        tmp_path,
        image=image,
        limits=ExecutorLimits(timeout_seconds=1),
    )

    result = executor("import time; time.sleep(30)")

    assert result.exc_type == "TimeoutError"
    assert result.exc_info == {"timeout_seconds": 1}
