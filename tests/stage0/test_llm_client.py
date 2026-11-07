from __future__ import annotations

import json

import httpx

from nir_stage0.config import LLMConfig
from nir_stage0.llm_client import (
    TOOL_NAME,
    create_openai_client,
    run_chat_smoke,
    run_tool_smoke,
)

CONFIG = LLMConfig(
    lite_llm_token="test-token",
    server_address="https://llm.example.test/v1",
    llm_model="qwen3.8-27b",
    max_tokens=512,
    temperature=0.7,
)


def _response(message):
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": CONFIG.llm_model,
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        },
    )


def test_chat_request_contract():
    captured = {}

    def handler(request):
        captured["authorization"] = request.headers["authorization"]
        captured["body"] = json.loads(request.content)
        return _response({"role": "assistant", "content": "I am a test model."})

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = create_openai_client(CONFIG, http_client=http_client)

    assert run_chat_smoke(client, CONFIG) == "I am a test model."
    assert captured["authorization"] == "Bearer test-token"
    assert captured["body"]["model"] == CONFIG.llm_model
    assert captured["body"]["max_tokens"] == 512
    assert captured["body"]["temperature"] == 0.7
    assert captured["body"]["messages"][0]["role"] == "user"


def test_forced_tool_call_contract():
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return _response(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-test",
                        "type": "function",
                        "function": {
                            "name": TOOL_NAME,
                            "arguments": '{"answer":"tools work"}',
                        },
                    }
                ],
            }
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = create_openai_client(CONFIG, http_client=http_client)

    assert run_tool_smoke(client, CONFIG) == {"answer": "tools work"}
    assert captured["body"]["tool_choice"]["function"]["name"] == TOOL_NAME
    assert captured["body"]["tools"][0]["function"]["name"] == TOOL_NAME
