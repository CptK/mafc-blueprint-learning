from mafc.training.trace_io import normalise_trace


def _judge_run(label: str) -> dict:
    return {
        "decision": {"label": label, "justification": "It seems compromised."},
        "evidence_count": 2,
        "repair_response": None,
        "summary": {
            "total_output_tokens": 120,
            "errors": [],
            "result": {
                "evidences": [
                    {"source": "https://a.com/x", "takeaways": {"text": "found in 2025-07"}},
                    {"source": "https://b.com/y", "takeaways": None},
                ]
            },
        },
    }


def test_normalise_fact_check_trace() -> None:
    trace = {
        "judge_run": _judge_run("compromised (certain)"),
        "blueprint": {
            "name": "image_authenticity",
            "max_iterations": 2,
            "selection": {"claim_features": {"has_image": True, "text_length": 50}},
        },
        "iterations": [
            {"evidence_count_after": 4, "delegated_tasks": [{"task_id": "a"}]},
            {"evidence_count_after": 8, "delegated_tasks": [{"task_id": "b"}]},
        ],
        "summary": {
            "errors": ["Failed to retrieve content from https://x"],
            "runtime_seconds": 300.0,
            "total_calls": 9,
        },
    }
    norm = normalise_trace(trace, "c1", "fact_check")
    assert norm.judge_label == "compromised (certain)"
    assert norm.judge_direction is None  # direction coarsening happens in features
    assert norm.n_iterations == 2
    assert norm.hit_max_iterations is True
    assert norm.n_delegated_tasks == 2
    assert norm.evidence_growth == [4, 8]
    assert norm.retrieval_failures == 1
    assert norm.blueprint_name == "image_authenticity"
    assert norm.claim_features == {"has_image": True, "text_length": 50}
    assert len(norm.evidence) == 2
    assert norm.evidence[0].is_useful is True
    assert norm.evidence[1].is_useful is False


def test_normalise_strategy_trace() -> None:
    trace = {
        "judge_run": _judge_run("intact (rather certain)"),
        "rounds": [
            {"round": 1, "tool_calls": [{"tool": "web_search"}, {"tool": "media"}],
             "done": True, "evidence_count_after": 6},
        ],
        "summary": {"errors": [], "total_output_tokens": 1000},
    }
    norm = normalise_trace(trace, "c2", "strategy")
    assert norm.judge_label == "intact (rather certain)"
    assert norm.n_iterations == 1
    assert norm.n_delegated_tasks == 2
    assert norm.evidence_growth == [6]
    assert norm.retrieval_failures == 0
