"""JudgeAgent must surface media referent status and per-source tags in its prompt.

Guards the evidence linkage for the wrong-referent flip class: the judge sees
which evidence pages are visually confirmed to show THIS media, so debunks from
unconfirmed pages can't silently flip authentic media (and vice versa).

Referent status is resolved at evidence assembly and stored on each item; the
judge reads it. Evidence written before that field existed is still parsed out of
the rendered RIS text, so archived runs stay rejudgeable — both paths are covered
here and must produce the same prompt.
"""

from __future__ import annotations

from mafc.agents.judge.agent import JudgeAgent
from mafc.common.action import Action
from mafc.common.evidence import Evidence
from mafc.common.media_referent import annotate_evidence_referents, extract_referent_digest
from mafc.common.modeling.prompt import Prompt

from .helpers import CLASS_DEFINITIONS, DummyAction, SequencedModel, make_session

_VALID_RESPONSE = '{"label": "true", "justification": "grounded"}'


class RisAction(Action):
    name = "reverse_image_search"

    def __init__(self):
        self._save_parameters(locals())


def _ris_evidence(text: str, source: str = "ris://query", referent: str | None = None) -> Evidence:
    return Evidence(raw=Prompt(text=text), action=RisAction(), source=source, referent=referent)


def _web_evidence(source: str, summary: str, referent: str | None = None) -> Evidence:
    return Evidence(
        raw=Prompt(text=summary),
        action=DummyAction(),
        source=source,
        takeaways=Prompt(text=summary),
        referent=referent,
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


def test_judge_reads_stored_referent_without_parsing_text() -> None:
    """The stored field alone drives the prompt — no 'Match type' prose present."""
    model = SequencedModel([_VALID_RESPONSE])
    judge = JudgeAgent(model, CLASS_DEFINITIONS)
    evidences = [
        _ris_evidence("Web Source https://confirmed.com/original", source="ris://query"),
        _web_evidence("https://www.confirmed.com/original?utm=1", "Original upload.", referent="exact"),
        _web_evidence("https://factcheck.org/debunk", "This footage is old.", referent="partial"),
    ]
    result = judge.run(make_session(evidences=evidences))
    assert result.result is not None

    prompt = _user_prompt(model)
    assert "Media referent status" in prompt
    # Stamped sources are listed in the summary block too, so scope the per-block
    # assertions to the evidence section.
    section = prompt.split("Accepted evidence:")[1]
    confirmed_block = section.split("https://www.confirmed.com/original?utm=1")[1].split("- Source:")[0]
    assert "SAME MEDIA CONFIRMED" in confirmed_block
    debunk_block = section.split("https://factcheck.org/debunk")[1].split("Return strict JSON")[0]
    assert "PARTIAL visual match only" in debunk_block


def test_stored_referent_takes_precedence_over_legacy_text() -> None:
    """A structured status wins over stale prose in the same evidence set."""
    evidences = [
        _ris_evidence(
            "Web Source https://page.com/a\nMatch type: PARTIAL match — a cropped version.",
            source="https://page.com/a",
            referent="exact",
        )
    ]
    digest = extract_referent_digest(evidences)
    assert digest.classify("https://page.com/a") == "exact"
    assert not digest.partial


def test_annotation_joins_ris_status_onto_other_sources() -> None:
    """The join the per-item action cannot express: RIS confirms a page, and the
    text-search item scraped from that same page inherits the confirmation."""
    ris = _ris_evidence(
        "Web Source https://news.com/story\nMatch type: EXACT copy of the media appears on this page.",
        source="https://news.com/story",
    )
    article = _web_evidence("https://www.news.com/story?ref=x", "This footage is from 2019.")
    unrelated = _web_evidence("https://other.com/post", "A different clip was debunked.")

    digest = annotate_evidence_referents(make_session().claim, [ris, article, unrelated], verify=False)

    assert digest.classify("https://news.com/story") == "exact"
    assert article.referent == "exact"
    assert unrelated.referent is None  # unconfirmed stays unconfirmed, never "different"


def test_annotation_is_idempotent() -> None:
    """Re-running over already-stamped evidence must not change any status."""
    evidences = [
        _ris_evidence(
            "Web Source https://news.com/story\nMatch type: EXACT copy of the media appears on this page.",
            source="https://news.com/story",
        ),
        _web_evidence("https://www.news.com/story?ref=x", "This footage is from 2019."),
    ]
    claim = make_session().claim
    annotate_evidence_referents(claim, evidences, verify=False)
    first = [e.referent for e in evidences]
    annotate_evidence_referents(claim, evidences, verify=False)
    assert [e.referent for e in evidences] == first == ["exact", "exact"]
