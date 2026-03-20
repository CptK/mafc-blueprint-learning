from __future__ import annotations

from mafc.agents.fact_check.agent import FactCheckAgent
from mafc.common.evidence import Evidence
from ezmm import MultimodalSequence

from tests.agents.fact_check.helpers import (
    DummyAction,
    SequencedModel,
    make_registry,
    make_selector,
)


def test_synthesize_from_evidences_returns_string(tmp_path) -> None:
    registry = make_registry(tmp_path, include_media=False)
    synthesizer = SequencedModel(outputs=["The claim is supported by web evidence."])
    agent = FactCheckAgent(
        model=synthesizer,
        blueprint_selector=make_selector(registry),
    )
    evidences = [
        Evidence(
            raw=MultimodalSequence("Web article about the event."),
            action=DummyAction(),
            source="https://example.com/article",
            takeaways=MultimodalSequence("The event happened in 2023."),
        )
    ]

    result = agent.synthesize_from_evidences("When did the event happen?", evidences)

    assert isinstance(result, str)
    assert "supported" in result


def test_synthesize_from_evidences_returns_empty_for_no_evidences(tmp_path) -> None:
    registry = make_registry(tmp_path, include_media=False)
    agent = FactCheckAgent(
        model=SequencedModel(outputs=[]),
        blueprint_selector=make_selector(registry),
    )

    result = agent.synthesize_from_evidences("Any question?", [])

    assert result == ""
