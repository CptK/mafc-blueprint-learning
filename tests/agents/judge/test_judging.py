from __future__ import annotations

import pytest

from mafc.agents.common import AgentStatus
from mafc.agents.judge.agent import JudgeAgent
from mafc.common.evidence import Evidence
from mafc.common.modeling.prompt import Prompt

from tests.agents.judge.helpers import (
    CLASS_DEFINITIONS,
    DummyAction,
    DummyLabel,
    SequencedModel,
    make_evidence,
    make_session,
)


def _agent(outputs: list[str], *, extra_rules: str | None = None) -> JudgeAgent:
    return JudgeAgent(
        model=SequencedModel(outputs=outputs),
        class_definitions=CLASS_DEFINITIONS,
        extra_judge_rules=extra_rules,
    )


def test_predicts_label_and_sets_claim_fields() -> None:
    session = make_session()
    out = _agent(['{"label":"false","justification":"The evidence contradicts the claim."}']).run(session)

    assert out.result is not None
    assert out.session.status == AgentStatus.COMPLETED
    assert session.claim is not None
    assert session.claim.verdict == DummyLabel.FALSE
    assert session.claim.justification is not None
    assert "false" in str(out.result).lower()
    assert "contradicts" in str(out.result)


def test_result_contains_all_session_evidences() -> None:
    evidences = [make_evidence("first"), make_evidence("second")]
    session = make_session(evidences=evidences)
    out = _agent(['{"label":"true","justification":"Well supported."}']).run(session)

    assert out.session.status == AgentStatus.COMPLETED
    assert len(out.evidences) == 2


def test_repairs_non_json_first_response() -> None:
    # First response unparseable → repair call → valid JSON
    session = make_session()
    out = _agent(
        [
            "I think the label is false because evidence contradicts it.",
            '{"label":"false","justification":"Evidence contradicts."}',
        ]
    ).run(session)

    assert out.result is not None
    assert out.session.status == AgentStatus.COMPLETED
    assert session.claim is not None
    assert session.claim.verdict == DummyLabel.FALSE


def test_repair_prompt_is_sent_as_second_call() -> None:
    model = SequencedModel(
        outputs=[
            "not-json",
            '{"label":"true","justification":"Supported."}',
        ]
    )
    agent = JudgeAgent(model=model, class_definitions=CLASS_DEFINITIONS)
    agent.run(make_session())

    assert len(model.calls) == 2
    repair_call_text = str(model.calls[1][0].content)
    assert "Convert" in repair_call_text or "JSON" in repair_call_text


def test_extra_judge_rules_appear_in_prompt() -> None:
    rule = "Prefer UNCERTAIN when only one source is available."
    model = SequencedModel(outputs=['{"label":"uncertain","justification":"Only one source."}'])
    agent = JudgeAgent(model=model, class_definitions=CLASS_DEFINITIONS, extra_judge_rules=rule)
    agent.run(make_session())

    prompt_text = str(model.calls[0][0].content)
    assert rule in prompt_text


def test_evidence_takeaways_used_over_raw_in_prompt() -> None:
    evidence = Evidence(
        raw=Prompt(text="Raw text that should not appear"),
        action=DummyAction(),
        source="https://example.com",
        takeaways=Prompt(text="Takeaway text that should appear"),
    )
    model = SequencedModel(outputs=['{"label":"true","justification":"Supported."}'])
    agent = JudgeAgent(model=model, class_definitions=CLASS_DEFINITIONS)
    from mafc.agents.common import AgentSession

    session = AgentSession(
        id="judge:test",
        goal=Prompt(text="Judge."),
        claim=make_session().claim,
        evidences=[evidence],
    )
    agent.run(session)

    user_prompt_text = str(model.calls[0][1].content)
    assert "Takeaway text that should appear" in user_prompt_text
    assert "Raw text that should not appear" not in user_prompt_text


def test_evidence_raw_used_when_no_takeaways() -> None:
    evidence = Evidence(
        raw=Prompt(text="Only raw text available"),
        action=DummyAction(),
        source="https://example.com",
        takeaways=None,
    )
    model = SequencedModel(outputs=['{"label":"true","justification":"Supported."}'])
    agent = JudgeAgent(model=model, class_definitions=CLASS_DEFINITIONS)
    from mafc.agents.common import AgentSession

    session = AgentSession(
        id="judge:test",
        goal=Prompt(text="Judge."),
        claim=make_session().claim,
        evidences=[evidence],
    )
    agent.run(session)

    user_prompt_text = str(model.calls[0][1].content)
    assert "Only raw text available" in user_prompt_text


def test_synthesize_from_evidences_returns_formatted_string() -> None:
    agent = _agent(['{"label":"false","justification":"Contradicted."}'])
    evidences = [make_evidence("The event did not happen.")]

    result = agent.synthesize_from_evidences("Was the event real?", evidences)

    assert "false" in result.lower()
    assert "Contradicted" in result


def test_label_matching_is_case_insensitive() -> None:
    # Model returns label with mixed case (e.g. capitalised by the LLM)
    session = make_session()
    out = _agent(['{"label":"False","justification":"Evidence contradicts."}']).run(session)

    assert out.session.status == AgentStatus.COMPLETED
    assert session.claim is not None
    assert session.claim.verdict == DummyLabel.FALSE


def test_synthesize_from_evidences_falls_back_to_raw_response_on_parse_failure() -> None:
    raw_response = "I cannot determine a label."
    agent = _agent([raw_response])
    evidences = [make_evidence()]

    result = agent.synthesize_from_evidences("Was the event real?", evidences)

    assert result == raw_response


# ---------------------------------------------------------------------------
# n_samples > 1: sampled-label aggregation
# ---------------------------------------------------------------------------


def test_n_samples_majority_vote_without_numeric_mapping() -> None:
    model = SequencedModel(
        outputs=[
            '{"label":"false","justification":"Contradicted."}',
            '{"label":"true","justification":"Supported."}',
            '{"label":"false","justification":"Contradicted again."}',
        ]
    )
    agent = JudgeAgent(model=model, class_definitions=CLASS_DEFINITIONS, n_samples=3)
    session = make_session()
    out = agent.run(session)

    assert out.session.status == AgentStatus.COMPLETED
    assert len(model.calls) == 3
    assert session.claim is not None
    assert session.claim.verdict == DummyLabel.FALSE


def test_n_samples_mean_numeric_snaps_to_nearest_label() -> None:
    # true=1, uncertain=0, false=-1; samples true/true/uncertain -> mean 2/3 -> true
    numeric = {DummyLabel.TRUE: 1.0, DummyLabel.UNCERTAIN: 0.0, DummyLabel.FALSE: -1.0}
    model = SequencedModel(
        outputs=[
            '{"label":"true","justification":"Supported."}',
            '{"label":"true","justification":"Supported."}',
            '{"label":"uncertain","justification":"Thin evidence."}',
        ]
    )
    agent = JudgeAgent(model=model, class_definitions=CLASS_DEFINITIONS, n_samples=3, label_numeric=numeric)
    session = make_session()
    agent.run(session)

    assert session.claim is not None
    assert session.claim.verdict == DummyLabel.TRUE


def test_n_samples_mean_numeric_disagreement_lands_on_middle_label() -> None:
    # samples true/false/uncertain -> mean 0.0 -> uncertain
    numeric = {DummyLabel.TRUE: 1.0, DummyLabel.UNCERTAIN: 0.0, DummyLabel.FALSE: -1.0}
    model = SequencedModel(
        outputs=[
            '{"label":"true","justification":"Supported."}',
            '{"label":"false","justification":"Contradicted."}',
            '{"label":"uncertain","justification":"Mixed."}',
        ]
    )
    agent = JudgeAgent(model=model, class_definitions=CLASS_DEFINITIONS, n_samples=3, label_numeric=numeric)
    session = make_session()
    agent.run(session)

    assert session.claim is not None
    assert session.claim.verdict == DummyLabel.UNCERTAIN


def test_aggregate_score_is_kept_unsnapped_alongside_the_label() -> None:
    """The label is the verdict; the score preserves what the samples averaged to,
    so regression metrics are not charged for the discretization."""
    numeric = {DummyLabel.TRUE: 1.0, DummyLabel.UNCERTAIN: 0.0, DummyLabel.FALSE: -1.0}
    model = SequencedModel(
        outputs=[
            '{"label":"true","justification":"Supported."}',
            '{"label":"true","justification":"Supported."}',
            '{"label":"uncertain","justification":"Thin evidence."}',
        ]
    )
    agent = JudgeAgent(model=model, class_definitions=CLASS_DEFINITIONS, n_samples=3, label_numeric=numeric)
    session = make_session()
    out = agent.run(session)

    assert session.claim is not None
    assert session.claim.verdict == DummyLabel.TRUE
    assert session.claim.verdict_score == pytest.approx(2 / 3)
    assert (out.trace or {})["decision"]["score"] == pytest.approx(2 / 3)


def test_single_sample_score_equals_its_label_value() -> None:
    """Nothing is averaged, so snapped and un-snapped scoring must coincide."""
    numeric = {DummyLabel.TRUE: 1.0, DummyLabel.UNCERTAIN: 0.0, DummyLabel.FALSE: -1.0}
    model = SequencedModel(outputs=['{"label":"true","justification":"Supported."}'])
    agent = JudgeAgent(model=model, class_definitions=CLASS_DEFINITIONS, label_numeric=numeric)
    session = make_session()
    agent.run(session)

    assert session.claim is not None
    assert session.claim.verdict_score == pytest.approx(1.0)


def test_score_is_none_without_a_numeric_mapping() -> None:
    """Majority-vote benchmarks have no scale to report a score on."""
    model = SequencedModel(outputs=['{"label":"true","justification":"Supported."}'])
    agent = JudgeAgent(model=model, class_definitions=CLASS_DEFINITIONS)
    session = make_session()
    agent.run(session)

    assert session.claim is not None
    assert session.claim.verdict_score is None


def test_n_samples_tolerates_invalid_samples() -> None:
    # One unparseable sample (repair also fails) must not sink the aggregate.
    model = SequencedModel(
        outputs=[
            "not-json",
            "still not json",  # repair attempt for sample 1
            '{"label":"false","justification":"Contradicted."}',
            '{"label":"false","justification":"Contradicted."}',
        ]
    )
    agent = JudgeAgent(model=model, class_definitions=CLASS_DEFINITIONS, n_samples=3)
    session = make_session()
    out = agent.run(session)

    assert out.session.status == AgentStatus.COMPLETED
    assert session.claim is not None
    assert session.claim.verdict == DummyLabel.FALSE


def test_n_samples_all_invalid_aborts() -> None:
    model = SequencedModel(outputs=["junk"] * 6)  # 3 samples + 3 repair attempts
    agent = JudgeAgent(model=model, class_definitions=CLASS_DEFINITIONS, n_samples=3)
    out = agent.run(make_session())

    assert out.result is None
    assert out.errors
