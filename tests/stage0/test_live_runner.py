import pytest

from nir_stage0.config import LLMConfig
from nir_stage0.live_runner import _live_experiment_name, _record_error, run_all_live


def test_live_experiment_names_do_not_reuse_partial_workspace():
    first = _live_experiment_name()
    second = _live_experiment_name()

    assert first.startswith("stage0-live-")
    assert second.startswith("stage0-live-")
    assert first != second


def test_error_record_redacts_endpoint_and_token_and_adds_category():
    config = LLMConfig(
        lite_llm_token="private-token",
        server_address="https://private.example/v1",
        llm_model="test-model",
        max_tokens=512,
        temperature=0.7,
    )
    result = {"checks": {}}
    error = ValueError(
        "bad response from https://private.example/v1 using private-token"
    )

    _record_error(result, "tool_calling", error, config)

    record = result["checks"]["tool_calling"]
    assert record["category"] == "response_format"
    assert record["message"] == "bad response from <server> using ***"


def test_live_runner_rejects_non_positive_timeout(tmp_path):
    with pytest.raises(ValueError, match="timeout must be positive"):
        run_all_live(env_file=tmp_path / "missing.env", timeout=0)
