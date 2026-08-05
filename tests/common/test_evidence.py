from ezmm import MultimodalSequence

from mafc.agents.tracing import serialize_evidence
from mafc.common.action import Action
from mafc.common.evidence import Evidence


class DummyAction(Action):
    name = "dummy"

    def __init__(self, value: int = 1):
        self._save_parameters(locals())
        self.value = value


def test_evidence_represents_one_source_backed_item() -> None:
    evidence = Evidence(
        raw=MultimodalSequence("Original article text"),
        action=DummyAction(),
        source="https://example.com/article",
        takeaways=MultimodalSequence("The article states the event occurred on March 2, 2024."),
    )

    assert evidence.is_useful() is True
    assert evidence.source == "https://example.com/article"
    assert "Evidence from `dummy`" in str(evidence)
    assert "March 2, 2024" in str(evidence)


def test_referent_defaults_to_unverified() -> None:
    """Unverified is the default, and is not the same as 'different media'."""
    evidence = Evidence(
        raw=MultimodalSequence("text"), action=DummyAction(), source="https://example.com/a"
    )
    assert evidence.referent is None


def test_referent_is_serialized_into_the_trace() -> None:
    """Persisting it is what makes rejudging an archived run deterministic."""
    evidence = Evidence(
        raw=MultimodalSequence("text"),
        action=DummyAction(),
        source="https://example.com/a",
        referent="exact",
    )
    assert serialize_evidence(evidence)["referent"] == "exact"
