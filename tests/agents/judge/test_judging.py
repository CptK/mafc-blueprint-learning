from __future__ import annotations


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
