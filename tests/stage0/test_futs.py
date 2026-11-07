from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO_ROOT / "third_party/era/implementation"))

from futs import (  # noqa: E402
    Node,
    Problem,
    Solution,
    backpropagate_visit,
    compute_pucts,
    compute_rank_scores,
    search,
)


def test_futs_deterministic_smoke_records_budget_and_edges():
    edges = []
    known = {"root"}

    def generate_fn(problem, parent_solution, parent_score):
        del problem, parent_score
        assert parent_solution.program in known
        child = Solution(program=f"{parent_solution.program}>{len(edges) + 1}")
        edges.append((parent_solution.program, child.program))
        known.add(child.program)
        return child

    def execute_fn(problem, solution):
        del problem
        return float(solution.program.count(">"))

    best, score = search(
        problem=Problem(description="deterministic stage0 smoke"),
        initial_solution=Solution(program="root"),
        initial_score=0.0,
        generate_fn=generate_fn,
        execute_fn=execute_fn,
        num_iterations=12,
        c_puct=1.0,
    )

    assert len(edges) == 12
    assert score == max(program.count(">") for program in known)
    assert best.program in known
    assert all(parent in known and child in known for parent, child in edges)


def test_rank_puct_and_visit_backpropagation():
    nodes = [
        Node(0, None, Solution("root"), 10.0),
        Node(1, 0, Solution("child"), 20.0),
        Node(2, 1, Solution("leaf"), 30.0),
    ]

    compute_rank_scores(nodes)
    assert [node.rank_score for node in nodes] == [0.0, 0.5, 1.0]

    backpropagate_visit(nodes, nodes[2])
    assert [node.num_visits for node in nodes] == [1, 1, 1]

    compute_pucts(nodes, c_puct=1.0)
    assert nodes[2].puct > nodes[1].puct > nodes[0].puct
