"""AIDE policy adapter used to prove that parent selection is replaceable."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence

from aide.agent import Agent
from aide.journal import Node

ParentSelector = Callable[[Sequence[Node]], Node]


def first_good_node(nodes: Sequence[Node]) -> Node:
    if not nodes:
        raise ValueError("cannot select a parent from an empty sequence")
    return nodes[0]


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
