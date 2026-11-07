"""Validated configuration for the OpenAI-compatible research endpoint."""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from dotenv import dotenv_values

REQUIRED_KEYS = (
    "lite_llm_token",
    "server_address",
    "llm_model",
    "max_tokens",
    "temperature",
)


class ConfigError(ValueError):
    """Raised when the LLM configuration is missing or invalid."""


@dataclass(frozen=True)
class LLMConfig:
    lite_llm_token: str = field(repr=False)
    server_address: str
    llm_model: str
    max_tokens: int
    temperature: float

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
    *,
    environ: Mapping[str, str] | None = None,
    repo_root: str | Path | None = None,
) -> LLMConfig:
    """Load config with process env taking precedence over dotenv files."""

    root = Path(repo_root) if repo_root is not None else Path.cwd()
    root = root.resolve()
    merged: dict[str, str] = {}

    for candidate in (root.parent / ".env", root / ".env"):
        merged.update(_read_env_file(candidate))

    if env_file is not None:
        explicit = Path(env_file).expanduser().resolve()
        if not explicit.is_file():
            raise ConfigError(f"env file does not exist: {explicit}")
        merged.update(_read_env_file(explicit))

    source_env = os.environ if environ is None else environ
    for key in REQUIRED_KEYS:
        if key in source_env:
            merged[key] = source_env[key]

    missing = [key for key in REQUIRED_KEYS if not merged.get(key, "").strip()]
    if missing:
        raise ConfigError(f"missing required configuration keys: {', '.join(missing)}")

    try:
        max_tokens = int(merged["max_tokens"])
    except ValueError as exc:
        raise ConfigError("max_tokens must be an integer") from exc
    if max_tokens <= 0:
        raise ConfigError("max_tokens must be positive")

    try:
        temperature = float(merged["temperature"])
    except ValueError as exc:
        raise ConfigError("temperature must be numeric") from exc
    if not 0.0 <= temperature <= 2.0:
        raise ConfigError("temperature must be between 0 and 2")

    return LLMConfig(
        lite_llm_token=merged["lite_llm_token"].strip(),
        server_address=_normalize_base_url(merged["server_address"]),
        llm_model=merged["llm_model"].strip(),
        max_tokens=max_tokens,
        temperature=temperature,
    )
