"""OpenAI-compatible smoke checks shared by tests and the live runner."""

from __future__ import annotations

import json
from typing import Any

import httpx
from openai import OpenAI

from .config import LLMConfig

TOOL_NAME = "submit_smoke_result"
TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": "Return a short result proving that tool calling works.",
        "parameters": {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
    },
}


class ReasoningBudgetExhaustedError(ValueError):
    """Raised when a reasoning model emits no final answer before the limit."""


def _raise_if_reasoning_exhausted(response: Any) -> None:
    choice = response.choices[0]
    message = choice.message
    reasoning = getattr(message, "reasoning_content", None)
    if reasoning or choice.finish_reason == "length":
        raise ReasoningBudgetExhaustedError(
            "LLM exhausted its reasoning budget without returning final content"
        )


def create_openai_client(
    config: LLMConfig,
    *,
    timeout: float = 60.0,
    http_client: httpx.Client | None = None,
) -> OpenAI:
    return OpenAI(
        api_key=config.lite_llm_token,
        base_url=config.server_address,
        max_retries=0,
        timeout=timeout,
        http_client=http_client,
    )


def run_chat_smoke(client: OpenAI, config: LLMConfig) -> str:
    response = client.chat.completions.create(
        model=config.llm_model,
        messages=[
            {
                "role": "user",
                "content": "Hello! Identify yourself in one short sentence.",
            }
        ],
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        reasoning_effort=config.reasoning_effort,
    )
    content = response.choices[0].message.content
    if not content or not content.strip():
        _raise_if_reasoning_exhausted(response)
        raise ValueError("LLM returned an empty chat completion")
    return content.strip()


def run_tool_smoke(client: OpenAI, config: LLMConfig) -> dict[str, str]:
    response = client.chat.completions.create(
        model=config.llm_model,
        messages=[
            {
                "role": "user",
                "content": "Call submit_smoke_result exactly once with a short answer.",
            }
        ],
        tools=[TOOL_SCHEMA],
        tool_choice={"type": "function", "function": {"name": TOOL_NAME}},
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        reasoning_effort=config.reasoning_effort,
    )
    calls = response.choices[0].message.tool_calls or []
    if len(calls) != 1 or calls[0].function.name != TOOL_NAME:
        _raise_if_reasoning_exhausted(response)
        raise ValueError(f"expected exactly one {TOOL_NAME} tool call")
    arguments = json.loads(calls[0].function.arguments)
    if set(arguments) != {"answer"} or not isinstance(arguments["answer"], str):
        raise ValueError("tool arguments do not match the smoke-test schema")
    return arguments
