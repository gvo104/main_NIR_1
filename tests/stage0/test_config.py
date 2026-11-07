from __future__ import annotations

import os

import pytest

from nir_stage0.config import ConfigError, load_llm_config

VALID_ENV = {
    "lite_llm_token": "secret-value",
    "server_address": "https://llm.example.test/v1/",
    "llm_model": "qwen3.8-27b",
    "max_tokens": "512",
    "temperature": "0.7",
}


def _write_env(path, values):
    path.write_text(
        "\n".join(f"{key}={value}" for key, value in values.items()) + "\n",
        encoding="utf-8",
    )


def test_loads_and_normalizes_explicit_env_file(tmp_path):
    env_file = tmp_path / "settings.env"
    _write_env(env_file, VALID_ENV)

    config = load_llm_config(env_file, environ={}, repo_root=tmp_path / "repo")

    assert config.server_address == "https://llm.example.test/v1"
    assert config.max_tokens == 512
    assert config.temperature == 0.7
    assert config.redacted_dict()["lite_llm_token"] == "***"
    assert "secret-value" not in repr(config)


def test_process_environment_overrides_dotenv(tmp_path):
    env_file = tmp_path / "settings.env"
    _write_env(env_file, VALID_ENV)
    process_env = dict(VALID_ENV, llm_model="override-model", max_tokens="1024")

    config = load_llm_config(env_file, environ=process_env, repo_root=tmp_path)

    assert config.llm_model == "override-model"
    assert config.max_tokens == 1024


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("server_address", "https://example.test/api", "/v1"),
        ("max_tokens", "0", "positive"),
        ("temperature", "2.1", "between 0 and 2"),
    ],
)
def test_rejects_invalid_values(tmp_path, key, value, message):
    values = dict(VALID_ENV, **{key: value})
    env_file = tmp_path / "settings.env"
    _write_env(env_file, values)

    with pytest.raises(ConfigError, match=message):
        load_llm_config(env_file, environ={}, repo_root=tmp_path)


def test_aide_environment_is_temporary(tmp_path, monkeypatch):
    env_file = tmp_path / "settings.env"
    _write_env(env_file, VALID_ENV)
    config = load_llm_config(env_file, environ={}, repo_root=tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "old-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    with config.aide_environment():
        assert os.environ["OPENAI_API_KEY"] == "secret-value"
        assert os.environ["OPENAI_BASE_URL"] == "https://llm.example.test/v1"

    assert os.environ["OPENAI_API_KEY"] == "old-key"
    assert "OPENAI_BASE_URL" not in os.environ
