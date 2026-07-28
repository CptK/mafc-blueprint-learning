from __future__ import annotations

from mafc.blueprints.models import Blueprint
from mafc.blueprints.topology import longest_path_nodes
from mafc.learning.blueprint_contrast import enforce_path_budget


def _blueprint(nodes: list[dict], max_iterations: int = 2, start: str = "n0") -> Blueprint:
    return Blueprint.model_validate(
        {
            "name": "bp",
            "description": "d",
            "policy_constraints": {"max_iterations": max_iterations},
            "verification_graph": {"start_node": start, "nodes": nodes},
        }
    )


def _synth(node_id: str, *targets: str) -> dict:
    return {
        "id": node_id,
        "type": "synthesis",
        "transition": [{"if": "continue", "to": t} for t in targets],
    }


def _actions(node_id: str, *targets: str) -> dict:
    return {
        "id": node_id,
        "type": "actions",
        "actions": [{"action": "web_search", "intent": "look"}],
        "transition": [{"if": "continue", "to": t} for t in targets],
    }


def test_synthesis_nodes_count_toward_the_path():
    """The whole point: actions -> synthesis -> actions -> synthesis is 4, not 2."""
    bp = _blueprint(
        [
            _actions("n0", "n1"),
            _synth("n1", "n2"),
            _actions("n2", "n3"),
            _synth("n3", "finalize"),
        ]
    )
    assert longest_path_nodes(bp) == 4


def test_finalize_is_free():
    bp = _blueprint([_actions("n0", "finalize")])
    assert longest_path_nodes(bp) == 1


def test_longest_branch_wins_not_shortest():
    bp = _blueprint(
        [
            _synth("n0", "finalize", "n1"),
            _actions("n1", "n2"),
            _synth("n2", "finalize"),
        ]
    )
    assert longest_path_nodes(bp) == 3


def test_cycles_are_traversed_once():
    bp = _blueprint([_actions("n0", "n1"), _synth("n1", "n0", "finalize")])
    assert longest_path_nodes(bp) == 2


def test_budget_is_raised_to_the_longest_path():
    bp = _blueprint(
        [_actions("n0", "n1"), _synth("n1", "n2"), _actions("n2", "n3"), _synth("n3", "finalize")],
        max_iterations=2,
    )
    repaired = enforce_path_budget(bp)
    assert repaired.policy_constraints.max_iterations == 4


def test_sufficient_budget_is_left_alone():
    bp = _blueprint([_actions("n0", "finalize")], max_iterations=5)
    assert enforce_path_budget(bp) is bp


def test_budget_is_a_floor_never_lowered():
    bp = _blueprint([_actions("n0", "n1"), _synth("n1", "finalize")], max_iterations=9)
    assert enforce_path_budget(bp).policy_constraints.max_iterations == 9
