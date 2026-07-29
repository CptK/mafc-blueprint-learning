"""JudgeAgent must surface media referent status and per-source tags in its prompt.

Guards the evidence linkage for the wrong-referent flip class: the judge sees
which evidence pages are visually confirmed to show THIS media, so debunks from
unconfirmed pages can't silently flip authentic media (and vice versa).
"""

from __future__ import annotations

from mafc.agents.judge.agent import JudgeAgent
from mafc.common.action import Action
from mafc.common.evidence import Evidence
from mafc.common.modeling.prompt import Prompt

from .helpers import CLASS_DEFINITIONS, DummyAction, SequencedModel, make_session

_VALID_RESPONSE = '{"label": "true", "justification": "grounded"}'


class RisAction(Action):
    name = "reverse_image_search"

    def __init__(self):
        self._save_parameters(locals())


def _ris_evidence(text: str) -> Evidence:
    return Evidence(raw=Prompt(text=text), action=RisAction(), source="ris://query")


def _web_evidence(source: str, summary: str) -> Evidence:
    return Evidence(
        raw=Prompt(text=summary), action=DummyAction(), source=source, takeaways=Prompt(text=summary)
    )


def _user_prompt(model: SequencedModel) -> str:
    return str(model.calls[0][-1].content)


def test_judge_prompt_contains_referent_block_and_tags() -> None:
    model = SequencedModel([_VALID_RESPONSE])
    judge = JudgeAgent(model, CLASS_DEFINITIONS)
    evidences = [
        _ris_evidence(
            "Web Source https://confirmed.com/original\n"
            "Match type: EXACT copy of the media appears on this page."
        ),
        _web_evidence("https://www.confirmed.com/original?utm=1", "Original upload from 2024."),
        _web_evidence("https://factcheck.org/debunk", "This footage is old, says fact-check."),
    ]
    result = judge.run(make_session(evidences=evidences))
    assert result.result is not None

    prompt = _user_prompt(model)
    assert "Media referent status" in prompt
    assert "Binding rule" in prompt
    # The confirmed page's evidence block is tagged; the unconfirmed one is not.
    confirmed_block = prompt.split("https://www.confirmed.com/original?utm=1")[1].split("- Source:")[0]
    assert "SAME MEDIA CONFIRMED" in confirmed_block
    debunk_block = prompt.split("https://factcheck.org/debunk")[1].split("Return strict JSON")[0]
    assert "SAME MEDIA CONFIRMED" not in debunk_block


def test_judge_prompt_unchanged_without_ris_evidence() -> None:
    model = SequencedModel([_VALID_RESPONSE])
    judge = JudgeAgent(model, CLASS_DEFINITIONS)
    result = judge.run(make_session(evidences=[_web_evidence("https://factcheck.org/a", "Some finding.")]))
    assert result.result is not None
    prompt = _user_prompt(model)
    assert "Media referent status" not in prompt
    assert "Referent status:" not in prompt
