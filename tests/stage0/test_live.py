from __future__ import annotations

import os

import pytest

from nir_stage0.live_runner import run_all_live

pytestmark = pytest.mark.live_llm


def test_live_api_tools_and_two_aide_steps():
    if os.environ.get("RUN_LIVE_LLM") != "1":
        pytest.skip("set RUN_LIVE_LLM=1 and enable VPN for live checks")

    optional = {}
    if value := os.environ.get("NIR_AIDE_MAX_TOKENS"):
        optional["aide_max_tokens"] = int(value)
    if value := os.environ.get("NIR_REQUEST_TIMEOUT"):
        optional["timeout"] = float(value)
    if value := os.environ.get("NIR_EXECUTOR_IMAGE"):
        optional["image"] = value

    result = run_all_live(
        env_file=os.environ.get("NIR_ENV_FILE", "../.env"),
        settings_file=os.environ.get("NIR_SETTINGS_FILE", "config/stage0.toml"),
        **optional,
    )

    assert result["checks"]["chat_completion"]["status"] == "PASS"
    assert result["checks"]["tool_calling"]["status"] == "PASS"
    assert result["checks"]["aide_two_steps"]["status"] == "PASS"
