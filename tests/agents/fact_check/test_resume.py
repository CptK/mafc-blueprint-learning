"""Tests for rebuilding session state from a recorded trace.

Silent loss is the failure mode that matters: state that fails to rehydrate does
not raise, it just makes the resumed run look different from the original for
reasons unrelated to the change being measured. Referent status is the sharpest
case — dropping it removes the judge's referent block entirely.
"""

from __future__ import annotations

from mafc.agents.fact_check.models import CheckStatus
from mafc.agents.fact_check.resume import (
    RecordedAction,
    evidence_from_dict,
    evidences_from_trace,
    state_from_trace,
)
from mafc.blueprints.models import (
    Blueprint,
    BlueprintEntryConditions,
    BlueprintPolicyConstraints,
    BlueprintRequiredCheck,
    BlueprintVerificationGraph,
)


def _blueprint() -> Blueprint:
    return Blueprint(
        name="bp",
        description="d",
        entry_conditions=BlueprintEntryConditions(),
        policy_constraints=BlueprintPolicyConstraints(allowed_actions=["web_search"], max_iterations=9),
        required_checks=[
            BlueprintRequiredCheck(id="c_open", description="still open"),
            BlueprintRequiredCheck(id="c_done", description="settled"),
            BlueprintRequiredCheck(id="c_lane", description="never activated"),
        ],
        verification_graph=BlueprintVerificationGraph(
            start_node="n0",
            nodes=[
                {"id": "n0", "type": "actions", "actions": [], "transition": [{"if": "c", "to": "n1"}]},
                {"id": "n1", "type": "actions", "actions": [], "transition": []},
            ],
        ),
    )


def _evidence(source="https://a.example", raw="body", **kw):
    return {"source": source, "raw": raw, "takeaways": "t", "action": "inspect_web_source", **kw}


def _trace(**overrides) -> dict:
    trace = {
        "blueprint": {"name": "bp"},
        "iterations": [
            {
                "iteration": 1,
                "node_after": "n1",
                "delegated_tasks": [
                    {
                        "task_id": "t1",
                        "agent_type": "web_search",
                        "child_session_id": "s1",
                        "instruction": "find origin",
                        "child_trace": {"summary": {"result": {"evidences": [_evidence()]}}},
                    }
                ],
            },
            {"iteration": 2, "node_after": "n1", "delegated_tasks": []},
        ],
        "summary": {
            "required_checks": {"c_open": "unclear", "c_done": "supported"},
            "required_check_reasons": {"c_open": "no exact match found"},
            "node_history": ["n1"],
            "action_history": ["delegate: look for origin"],
        },
    }
    trace.update(overrides)
    return trace


def test_evidence_round_trips_with_referent():
    """Referent drives the judge's referent block; losing it changes the verdict."""
    e = evidence_from_dict(_evidence(referent="exact", preview="snippet"))
    assert e is not None
    assert e.source == "https://a.example"
    assert e.referent == "exact"
    assert e.preview == "snippet"
    assert str(e.takeaways) == "t"


def test_action_name_survives():
    """media_referent keys reverse-image-search detection off the action NAME."""
    e = evidence_from_dict(_evidence(action="reverse_image_search", action_repr="ris(x=1)"))
    assert isinstance(e.action, RecordedAction)
    assert e.action.name == "reverse_image_search"
    assert str(e.action) == "ris(x=1)"


def test_evidence_without_content_is_dropped():
    assert evidence_from_dict({"source": "x", "raw": None, "takeaways": None}) is None


def test_duplicate_sources_are_collapsed():
    """Sub-agents re-report their whole set each time they are consulted."""
    trace = _trace()
    trace["iterations"][1]["delegated_tasks"] = [
        {"task_id": "t2", "child_trace": {"summary": {"result": {"evidences": [_evidence(), _evidence()]}}}}
    ]
    assert len(evidences_from_trace(trace)) == 1


def test_distinct_sources_are_kept():
    trace = _trace()
    trace["iterations"][1]["delegated_tasks"] = [
        {
            "task_id": "t2",
            "child_trace": {"summary": {"result": {"evidences": [_evidence("https://b.example")]}}},
        }
    ]
    assert len(evidences_from_trace(trace)) == 2


def test_state_restores_the_check_ledger():
    state = state_from_trace(_trace(), _blueprint())
    assert state.required_check_status["c_open"] is CheckStatus.UNCLEAR
    assert state.required_check_status["c_done"] is CheckStatus.SUPPORTED
    assert state.open_check_ids() == ("c_open",)
    assert state.required_check_reasons["c_open"] == "no exact match found"


def test_never_activated_checks_stay_out():
    """Path-scoping must survive the round trip, or refine inherits dead obligations."""
    state = state_from_trace(_trace(), _blueprint())
    assert "c_lane" not in state.required_check_status
    assert "c_lane" not in state.required_check_defs


def test_iteration_resumes_at_the_recorded_count():
    """Budget left after resuming must equal what the original run had left."""
    state = state_from_trace(_trace(), _blueprint())
    assert state.iteration == 2
    assert state.selected_blueprint.policy_constraints.max_iterations - state.iteration == 7


def test_delegated_task_history_is_restored():
    """The planner sees prior tasks so refine does not re-issue the same work."""
    state = state_from_trace(_trace(), _blueprint())
    assert "t1" in state.delegated_tasks
    assert state.delegated_tasks["t1"].agent_type == "web_search"
    assert state.delegated_tasks["t1"].instruction == "find origin"


def test_history_and_position_are_restored():
    state = state_from_trace(_trace(), _blueprint())
    assert state.current_node_id == "n1"
    assert state.node_history == ["n1"]
    assert state.action_history == ["delegate: look for origin"]
    assert len(state.evidences) == 1


def test_unknown_check_status_degrades_to_unchecked():
    trace = _trace()
    trace["summary"]["required_checks"] = {"c_open": "bogus"}
    state = state_from_trace(trace, _blueprint())
    assert state.required_check_status["c_open"] is CheckStatus.UNCHECKED


def test_empty_trace_falls_back_to_the_start_node():
    state = state_from_trace({"summary": {}, "iterations": []}, _blueprint())
    assert state.current_node_id == "n0"
    assert state.iteration == 0
    assert state.evidences == []
