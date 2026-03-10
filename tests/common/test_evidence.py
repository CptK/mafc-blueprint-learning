from ezmm import MultimodalSequence

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
