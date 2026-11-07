"""AIDE policy adapter used to prove that parent selection is replaceable."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from typing import Any, Iterator

import aide.agent as aide_agent
from aide.agent import Agent
from aide.journal import Node

from .llm_client import ReasoningBudgetExhaustedError

ParentSelector = Callable[[Sequence[Node]], Node]
USER_MESSAGE_FALLBACK = (
    "Follow the instructions in the system message and return the requested response."
)


def first_good_node(nodes: Sequence[Node]) -> Node:
    if not nodes:
        raise ValueError("cannot select a parent from an empty sequence")
    return nodes[0]


@contextmanager
def ensure_aide_user_message() -> Iterator[None]:
    """Add a user turn for endpoints that reject system-only AIDE requests.

    AIDE's upstream agent passes its complete prompt as ``system_message`` and
    leaves ``user_message`` empty for generation and feedback. Hosted vLLM
    rejects that otherwise valid Chat Completions shape. The adapter is scoped
    to the local live run and restores AIDE's original query function on exit.
    """

    original_query = aide_agent.query

    def query_with_user_message(*args: Any, **kwargs: Any):
        if kwargs.get("user_message") is None:
            kwargs["user_message"] = USER_MESSAGE_FALLBACK
        try:
            return original_query(*args, **kwargs)
        except ValueError as exc:
            if str(exc) == "OpenAI returned no text":
                raise ReasoningBudgetExhaustedError(
                    "LLM exhausted its reasoning budget without returning final content"
                ) from exc
            raise

    aide_agent.query = query_with_user_message
    try:
        yield
    finally:
        aide_agent.query = original_query


class SelectableParentAgent(Agent):
    """Keep AIDE draft/debug behavior while injecting the improve-node selector."""

    def __init__(self, *args, parent_selector: ParentSelector, **kwargs):
        super().__init__(*args, **kwargs)
        self.parent_selector = parent_selector

    def search_policy(self) -> Node | None:
        search_cfg = self.acfg.search

        if len(self.journal.draft_nodes) < search_cfg.num_drafts:
            return None

        if random.random() < search_cfg.debug_prob:
            debuggable_nodes = [
                node
                for node in self.journal.buggy_nodes
                if node.is_leaf and node.debug_depth < search_cfg.max_debug_depth
            ]
            if debuggable_nodes:
                return random.choice(debuggable_nodes)

        good_nodes = self.journal.good_nodes
        if not good_nodes:
            return None

        selected = self.parent_selector(tuple(good_nodes))
        if selected not in good_nodes:
            raise ValueError(
                "parent selector returned a node outside journal.good_nodes"
            )
        return selected

    @property
    def _prompt_environment(self):
        packages = (
            "numpy, pandas, scikit-learn, scipy, statsmodels, xgboost, "
            "lightgbm, matplotlib, seaborn"
        )
        return {
            "Installed packages": (
                f"The execution sandbox provides: {packages}. "
                "Use only the Python standard library and these packages."
            )
        }
