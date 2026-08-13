"""Tests for the synthetic refine node.

Refine changes when every blueprint stops, so the invariants that matter are the
guards: it must not fire when disabled, must not fire with nothing unresolved,
must not fire without budget, and must never outlive the iteration budget.
"""

from __future__ import annotations

import pytest

from mafc.agents.fact_check.agent import REFINE_NODE_ID, FactCheckAgent
from mafc.agents.fact_check.models import CheckStatus, FactCheckSessionState
from mafc.blueprints.models import (
    Blueprint,
    BlueprintEntryConditions,
    BlueprintPolicyConstraints,
    BlueprintRequiredCheck,
    BlueprintVerificationGraph,
)


def _blueprint(max_iterations: int = 9) -> Blueprint:
    return Blueprint(
        name="bp",
        description="d",
        entry_conditions=BlueprintEntryConditions(),
        policy_constraints=BlueprintPolicyConstraints(
            allowed_actions=["web_search"], max_iterations=max_iterations
        ),
        required_checks=[
            BlueprintRequiredCheck(id="on_path", description="activated on this path"),
            BlueprintRequiredCheck(id="other_lane", description="never activated"),
        ],
        verification_graph=BlueprintVerificationGraph(
            start_node="n0",
            nodes=[{"id": "n0", "type": "actions", "actions": [], "transition": []}],
        ),
    )


def _state(iteration: int = 1, statuses: dict[str, CheckStatus] | None = None) -> FactCheckSessionState:
    bp = _blueprint()
    return FactCheckSessionState(
        selected_blueprint=bp,
        current_node_id="n0",
        node_layers={"n0": 0},
        max_layer=0,
        iteration=iteration,
        required_check_status=dict(statuses or {"on_path": CheckStatus.UNCHECKED}),
    )


def _make(enabled: bool) -> FactCheckAgent:
    agent = FactCheckAgent.__new__(FactCheckAgent)
    agent.enable_refine_node = enabled
    return agent


def test_fires_when_budget_and_open_checks_remain():
    assert _make(True)._should_refine(_state(iteration=3)) is True


def test_disabled_by_default_flag():
    assert _make(False)._should_refine(_state(iteration=3)) is False


def test_does_not_fire_without_open_checks():
    """Nothing unresolved means no stated question more search could answer."""
    state = _state(iteration=3, statuses={"on_path": CheckStatus.SUPPORTED})
    assert _make(True)._should_refine(state) is False


def test_unclear_counts_as_open():
    state = _state(iteration=3, statuses={"on_path": CheckStatus.UNCLEAR})
    assert _make(True)._should_refine(state) is True


def test_does_not_fire_once_budget_is_spent():
    assert _make(True)._should_refine(_state(iteration=9)) is False
    assert _make(True)._should_refine(_state(iteration=12)) is False


def test_budget_never_goes_negative():
    assert _make(True)._refine_budget_remaining(_state(iteration=99)) == 0


@pytest.mark.parametrize("iteration,expected", [(1, 8), (5, 4), (8, 1), (9, 0)])
def test_budget_is_what_the_blueprint_left_unspent(iteration, expected):
    assert _make(True)._refine_budget_remaining(_state(iteration=iteration)) == expected


def test_refine_may_re_enter_itself_while_budget_lasts():
    """Refine spends the remaining budget across passes, not just one."""
    state = _state(iteration=4)
    state.current_node_id = REFINE_NODE_ID
    assert _make(True)._should_refine(state) is True


def test_refine_re_entry_stops_at_the_budget():
    state = _state(iteration=9)
    state.current_node_id = REFINE_NODE_ID
    assert _make(True)._should_refine(state) is False


def test_current_node_resolves_to_the_synthetic_node():
    """The refine node is not in the blueprint graph, so lookup must special-case it."""
    state = _state()
    state.current_node_id = REFINE_NODE_ID
    node = _make(True)._get_current_node(state)
    assert node.id == REFINE_NODE_ID
    assert [t.to for t in node.transition] == ["finalize"]
    assert node.activates_checks == []  # must not import obligations


def test_never_activated_checks_stay_out_of_the_ledger():
    """Path-scoping is deliberate: other lanes' checks may be unresolvable here."""
    state = _state()
    assert "other_lane" not in state.required_check_status
    assert "other_lane" not in state.open_check_ids()


def test_entering_refine_records_history_and_continues():
    class _Trace:
        def __init__(self):
            self.routed = []

        def record_auto_routing(self, target, iteration):
            self.routed.append((target, iteration))

    state, trace = _state(iteration=3), _Trace()
    assert _make(True)._enter_refine(state, trace) is False  # False => loop keeps running
    assert state.current_node_id == REFINE_NODE_ID
    assert state.node_history[-1] == REFINE_NODE_ID
    assert trace.routed == [(REFINE_NODE_ID, 3)]
    assert state.last_synthesis is None


def test_entering_refine_registers_it_in_the_layer_map():
    """build_system_prompt indexes node_layers by the current node every iteration.

    The synthetic node is absent from the blueprint topology that built the map,
    so failing to register it raised KeyError('refine') and lost the verdict.
    """

    class _Trace:
        def record_auto_routing(self, target, iteration):
            pass

    state = _state(iteration=3)
    assert REFINE_NODE_ID not in state.node_layers
    _make(True)._enter_refine(state, _Trace())
    assert state.node_layers[REFINE_NODE_ID] == state.max_layer + 1


def test_system_prompt_builds_at_the_refine_node():
    """End-to-end guard on the crash: the prompt must render once refine is entered.

    The position block moved to the runtime state block when the planner prompt was
    split for caching, so the node_layers lookup that crashed lives there now.
    """
    from mafc.agents.fact_check.prompts import build_runtime_state_block, build_system_prompt

    class _Trace:
        def record_auto_routing(self, target, iteration):
            pass

    state = _state(iteration=3)
    _make(True)._enter_refine(state, _Trace())
    text = build_runtime_state_block(state)
    assert "current node: refine" in text
    assert "stay_allowed: True" in text
    # The stable half must still render, and must not carry per-iteration state.
    system_text = build_system_prompt(state, "web_search")
    assert "current node:" not in system_text


def test_re_entering_refine_keeps_one_layer_entry():
    class _Trace:
        def record_auto_routing(self, target, iteration):
            pass

    agent, state, trace = _make(True), _state(iteration=3), _Trace()
    agent._enter_refine(state, trace)
    agent._enter_refine(state, trace)
    assert state.node_layers[REFINE_NODE_ID] == state.max_layer + 1
    assert state.node_history.count(REFINE_NODE_ID) == 2
