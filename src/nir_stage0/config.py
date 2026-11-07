"""Validated configuration for the OpenAI-compatible research endpoint."""

from __future__ import annotations

import os
import tomllib
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from dotenv import dotenv_values

REQUIRED_ENV_KEYS = ("lite_llm_token", "server_address")
REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}


class ConfigError(ValueError):
    """Raised when the LLM configuration is missing or invalid."""


@dataclass(frozen=True)
class LLMConfig:
    lite_llm_token: str = field(repr=False)
    server_address: str
    llm_model: str
    max_tokens: int
    temperature: float
    reasoning_effort: str = "low"
    aide_max_tokens: int = 32768
    request_timeout: float = 300.0
    executor_image: str = "nir-stage0-executor:latest"

    def redacted_dict(self) -> dict[str, str | int | float]:
        values = asdict(self)
        values["lite_llm_token"] = "***"
        return values

    @contextmanager
    def aide_environment(self) -> Iterator[None]:
        """Temporarily expose only the names required by AIDE's OpenAI backend."""

        keys = ("OPENAI_API_KEY", "OPENAI_BASE_URL")
        previous = {key: os.environ.get(key) for key in keys}
        os.environ["OPENAI_API_KEY"] = self.lite_llm_token
        os.environ["OPENAI_BASE_URL"] = self.server_address
        try:
            yield
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    return {
        key: value for key, value in dotenv_values(path).items() if value is not None
    }


def _read_settings_file(path: Path) -> dict:
    if not path.is_file():
        raise ConfigError(f"settings file does not exist: {path}")
    try:
        with path.open("rb") as file:
            return tomllib.load(file)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML settings file: {path}") from exc


def _settings_section(settings: dict, name: str) -> dict:
    section = settings.get(name)
    if not isinstance(section, dict):
        raise ConfigError(f"missing [{name}] section in settings file")
    return section


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be an integer")
    if value <= 0:
        raise ConfigError(f"{name} must be positive")
    return value


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be numeric")
    result = float(value)
    if result <= 0:
        raise ConfigError(f"{name} must be positive")
    return result


def _normalize_base_url(raw: str) -> str:
    value = raw.strip().rstrip("/")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ConfigError("server_address must be an absolute HTTP(S) URL")
    if parsed.query or parsed.fragment:
        raise ConfigError(
            "server_address must not contain query parameters or a fragment"
        )
    if parsed.path.rstrip("/") != "/v1":
        raise ConfigError(
            "server_address must point to the OpenAI-compatible /v1 base URL"
        )
    return urlunsplit((parsed.scheme, parsed.netloc, "/v1", "", ""))


def load_llm_config(
    env_file: str | Path | None = None,
    settings_file: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    repo_root: str | Path | None = None,
) -> LLMConfig:
    """Load secrets from dotenv and non-secret runtime settings from TOML."""

    root = Path(repo_root) if repo_root is not None else Path.cwd()
    root = root.resolve()
    if settings_file is None:
        settings_path = root / "config/stage0.toml"
    else:
        settings_path = Path(settings_file).expanduser()
        if not settings_path.is_absolute():
            settings_path = root / settings_path
        settings_path = settings_path.resolve()
    settings = _read_settings_file(settings_path)
    llm_settings = _settings_section(settings, "llm")
    runtime_settings = _settings_section(settings, "runtime")

    merged: dict[str, str] = {}

    for candidate in (root.parent / ".env", root / ".env"):
        merged.update(_read_env_file(candidate))

    if env_file is not None:
        explicit = Path(env_file).expanduser().resolve()
        if not explicit.is_file():
            raise ConfigError(f"env file does not exist: {explicit}")
        merged.update(_read_env_file(explicit))

    source_env = os.environ if environ is None else environ
    for key in REQUIRED_ENV_KEYS:
        if key in source_env:
            merged[key] = source_env[key]

    missing = [key for key in REQUIRED_ENV_KEYS if not merged.get(key, "").strip()]
    if missing:
        raise ConfigError(f"missing required configuration keys: {', '.join(missing)}")

    model = llm_settings.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ConfigError("llm.model must be a non-empty string")
    max_tokens = _positive_int(
        llm_settings.get("smoke_max_tokens"), "llm.smoke_max_tokens"
    )
    aide_max_tokens = _positive_int(
        llm_settings.get("aide_max_tokens"), "llm.aide_max_tokens"
    )
    temperature_value = llm_settings.get("temperature")
    if isinstance(temperature_value, bool) or not isinstance(
        temperature_value, (int, float)
    ):
        raise ConfigError("llm.temperature must be numeric")
    temperature = float(temperature_value)
    if not 0.0 <= temperature <= 2.0:
        raise ConfigError("llm.temperature must be between 0 and 2")
    reasoning_effort = llm_settings.get("reasoning_effort")
    if reasoning_effort not in REASONING_EFFORTS:
        allowed = ", ".join(sorted(REASONING_EFFORTS))
        raise ConfigError(f"llm.reasoning_effort must be one of: {allowed}")

    request_timeout = _positive_float(
        runtime_settings.get("request_timeout"), "runtime.request_timeout"
    )
    executor_image = runtime_settings.get("executor_image")
    if not isinstance(executor_image, str) or not executor_image.strip():
        raise ConfigError("runtime.executor_image must be a non-empty string")

    return LLMConfig(
        lite_llm_token=merged["lite_llm_token"].strip(),
        server_address=_normalize_base_url(merged["server_address"]),
        llm_model=model.strip(),
        max_tokens=max_tokens,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        aide_max_tokens=aide_max_tokens,
        request_timeout=request_timeout,
        executor_image=executor_image.strip(),
    )
