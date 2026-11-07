from __future__ import annotations

import os

import pytest

from nir_stage0.config import ConfigError, load_llm_config

VALID_ENV = {
    "lite_llm_token": "secret-value",
    "server_address": "https://llm.example.test/v1/",
}


def _write_env(path, values):
    path.write_text(
        "\n".join(f"{key}={value}" for key, value in values.items()) + "\n",
        encoding="utf-8",
    )


def _write_settings(path, *, llm=None, runtime=None):
    llm_values = {
        "model": '"qwen3.8-27b"',
        "smoke_max_tokens": "4096",
        "aide_max_tokens": "32768",
        "temperature": "0.7",
        "reasoning_effort": '"low"',
    }
    runtime_values = {
        "request_timeout": "300",
        "executor_image": '"nir-stage0-executor:latest"',
    }
    llm_values.update(llm or {})
    runtime_values.update(runtime or {})
    path.write_text(
        "[llm]\n"
        + "\n".join(f"{key} = {value}" for key, value in llm_values.items())
        + "\n\n[runtime]\n"
        + "\n".join(f"{key} = {value}" for key, value in runtime_values.items())
        + "\n",
        encoding="utf-8",
    )


def _config_files(tmp_path):
    env_file = tmp_path / "secrets.env"
    settings_file = tmp_path / "stage0.toml"
    _write_env(env_file, VALID_ENV)
    _write_settings(settings_file)
    return env_file, settings_file


def test_loads_secrets_and_non_secret_settings(tmp_path):
    env_file, settings_file = _config_files(tmp_path)

    config = load_llm_config(
        env_file,
        settings_file,
        environ={},
        repo_root=tmp_path / "repo",
    )

    assert config.server_address == "https://llm.example.test/v1"
    assert config.llm_model == "qwen3.8-27b"
    assert config.max_tokens == 4096
    assert config.aide_max_tokens == 32768
    assert config.temperature == 0.7
    assert config.reasoning_effort == "low"
    assert config.request_timeout == 300
    assert config.executor_image == "nir-stage0-executor:latest"
    assert config.redacted_dict()["lite_llm_token"] == "***"
    assert "secret-value" not in repr(config)


def test_process_environment_overrides_dotenv_secrets(tmp_path):
    env_file, settings_file = _config_files(tmp_path)
    process_env = {
        "lite_llm_token": "override-secret",
        "server_address": "https://override.example.test/v1",
    }

    config = load_llm_config(
        env_file,
        settings_file,
        environ=process_env,
        repo_root=tmp_path,
    )

    assert config.lite_llm_token == "override-secret"
    assert config.server_address == "https://override.example.test/v1"


def test_legacy_non_secret_environment_values_do_not_override_toml(tmp_path):
    env_file, settings_file = _config_files(tmp_path)
    process_env = {
        "llm_model": "legacy-model",
        "max_tokens": "1",
        "temperature": "2.0",
    }

    config = load_llm_config(
        env_file,
        settings_file,
        environ=process_env,
        repo_root=tmp_path,
    )

    assert config.llm_model == "qwen3.8-27b"
    assert config.max_tokens == 4096
    assert config.temperature == 0.7


def test_rejects_invalid_server_address(tmp_path):
    env_file, settings_file = _config_files(tmp_path)
    _write_env(env_file, dict(VALID_ENV, server_address="https://example.test/api"))

    with pytest.raises(ConfigError, match="/v1"):
        load_llm_config(env_file, settings_file, environ={}, repo_root=tmp_path)


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("llm", "model", '""', "non-empty string"),
        ("llm", "smoke_max_tokens", "0", "positive"),
        ("llm", "aide_max_tokens", "0", "positive"),
        ("llm", "temperature", "2.1", "between 0 and 2"),
        ("llm", "reasoning_effort", '"unsupported"', "must be one of"),
        ("runtime", "request_timeout", "0", "positive"),
        ("runtime", "executor_image", '""', "non-empty string"),
    ],
)
def test_rejects_invalid_settings(tmp_path, section, key, value, message):
    env_file, settings_file = _config_files(tmp_path)
    overrides = {key: value}
    _write_settings(
        settings_file,
        llm=overrides if section == "llm" else None,
        runtime=overrides if section == "runtime" else None,
    )

    with pytest.raises(ConfigError, match=message):
        load_llm_config(env_file, settings_file, environ={}, repo_root=tmp_path)


def test_reasoning_can_be_disabled_with_none_effort(tmp_path):
    env_file, settings_file = _config_files(tmp_path)
    _write_settings(settings_file, llm={"reasoning_effort": '"none"'})

    config = load_llm_config(
        env_file,
        settings_file,
        environ={},
        repo_root=tmp_path,
    )

    assert config.reasoning_effort == "none"


def test_aide_environment_is_temporary(tmp_path, monkeypatch):
    env_file, settings_file = _config_files(tmp_path)
    config = load_llm_config(
        env_file,
        settings_file,
        environ={},
        repo_root=tmp_path,
    )
    monkeypatch.setenv("OPENAI_API_KEY", "old-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    with config.aide_environment():
        assert os.environ["OPENAI_API_KEY"] == "secret-value"
        assert os.environ["OPENAI_BASE_URL"] == "https://llm.example.test/v1"

    assert os.environ["OPENAI_API_KEY"] == "old-key"
    assert "OPENAI_BASE_URL" not in os.environ
