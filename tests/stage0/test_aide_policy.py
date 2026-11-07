from __future__ import annotations

from types import SimpleNamespace

import aide.agent
import pytest
from aide.interpreter import ExecutionResult
from aide.journal import Journal, Node
from aide.utils.metric import MetricValue

from nir_stage0.aide_policy import (
    USER_MESSAGE_FALLBACK,
    SelectableParentAgent,
    ensure_aide_user_message,
)


def _cfg(*, num_drafts=1, debug_prob=0.0):
    stage = SimpleNamespace(
        model="mock-model",
        provider="openai",
        temp=0.0,
        max_tokens=512,
        reasoning_effort=None,
    )
    agent = SimpleNamespace(
        search=SimpleNamespace(
            num_drafts=num_drafts,
            debug_prob=debug_prob,
            max_debug_depth=2,
        ),
        code=stage,
        feedback=stage,
        data_preview=False,
        k_fold_validation=1,
        expose_prediction=False,
    )
    return SimpleNamespace(
        agent=agent,
        exec=SimpleNamespace(timeout=30),
        workspace_dir="unused",
    )


def _good_node(code, metric, parent=None):
    node = Node(code=code, plan=f"plan {code}", parent=parent)
    node.is_buggy = False
    node.analysis = "ok"
    node.metric = MetricValue(metric, maximize=False)
    node._term_out = ["metric printed"]
    return node


def test_selector_can_choose_an_arbitrary_good_parent():
    journal = Journal(metric_maximize=False)
    first = _good_node("first", 1.0)
    second = _good_node("second", 2.0)
    journal.append(first)
    journal.append(second)

    agent = SelectableParentAgent(
        task_desc="test",
        cfg=_cfg(),
        journal=journal,
        parent_selector=lambda nodes: nodes[1],
    )

    assert agent.search_policy() is second


def test_selector_rejects_node_outside_journal():
    journal = Journal(metric_maximize=False)
    journal.append(_good_node("first", 1.0))
    outsider = _good_node("outsider", 0.5)
    agent = SelectableParentAgent(
        task_desc="test",
        cfg=_cfg(),
        journal=journal,
        parent_selector=lambda nodes: outsider,
    )

    with pytest.raises(ValueError, match="outside"):
        agent.search_policy()


def test_compatibility_adapter_adds_user_message_and_restores_query(monkeypatch):
    calls = []

    def fake_query(*args, **kwargs):
        calls.append((args, kwargs))
        return "ok"

    monkeypatch.setattr(aide.agent, "query", fake_query)
    original_query = aide.agent.query

    with ensure_aide_user_message():
        assert (
            aide.agent.query(
                system_message="system instruction",
                user_message=None,
                model="test-model",
            )
            == "ok"
        )

    assert calls[0][1]["user_message"] == USER_MESSAGE_FALLBACK
    assert aide.agent.query is original_query


def test_two_mocked_steps_preserve_generation_feedback_and_parent_link(monkeypatch):
    journal = Journal(metric_maximize=False)
    cfg = _cfg()
    agent = SelectableParentAgent(
        task_desc="test task",
        cfg=cfg,
        journal=journal,
        parent_selector=lambda nodes: nodes[0],
    )
    agent.data_preview = "fixture preview"
    monkeypatch.setattr(agent, "update_data_preview", lambda: None)
    generated = iter(
        [
            "Draft plan.\n```python\nprint('metric: 1.0')\n```",
            "Improve plan.\n```python\nprint('metric: 0.8')\n```",
        ]
    )

    def fake_query(*, func_spec=None, **kwargs):
        del kwargs
        if func_spec is None:
            return next(generated)
        return {
            "is_bug": False,
            "summary": "valid result",
            "metric": 1.0,
            "lower_is_better": True,
        }

    monkeypatch.setattr(aide.agent, "query", fake_query)

    def fake_exec(code, reset_session):
        assert reset_session is True
        assert "metric" in code
        return ExecutionResult(["metric: 1.0"], 0.01, None, None, [])

    agent.step(exec_callback=fake_exec)
    agent.step(exec_callback=fake_exec)

    assert len(journal) == 2
    assert journal[1].parent is journal[0]
    assert journal[1] in journal[0].children
    assert all(not node.is_buggy for node in journal)
