from __future__ import annotations

import os

import pytest

from nir_stage0.live_runner import run_all_live

pytestmark = pytest.mark.live_llm


def test_live_api_tools_and_two_aide_steps():
    if os.environ.get("RUN_LIVE_LLM") != "1":
        pytest.skip("set RUN_LIVE_LLM=1 and enable VPN for live checks")

    result = run_all_live(
        env_file=os.environ.get("NIR_ENV_FILE", "../.env"),
        aide_max_tokens=int(os.environ.get("NIR_AIDE_MAX_TOKENS", "4096")),
        timeout=float(os.environ.get("NIR_REQUEST_TIMEOUT", "60")),
        image=os.environ.get("NIR_EXECUTOR_IMAGE", "nir-stage0-executor:latest"),
    )

    assert result["checks"]["chat_completion"]["status"] == "PASS"
    assert result["checks"]["tool_calling"]["status"] == "PASS"
    assert result["checks"]["aide_two_steps"]["status"] == "PASS"
