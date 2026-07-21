"""Sequential fold engine for the ``Strategy.md`` baseline.

A :class:`StrategySynthesizer` folds one batch of fact-check article analyses
into a running strategy document. The driver calls :meth:`fold` once per batch,
threading the returned document into the next call — a single sequential pass
with no merge step.

Design contract (the prompt is the algorithm here)
--------------------------------------------------
* The document is a *distillation of transferable methodology*, never a log of
  the claims seen. The prompt forbids recording claim-specific facts, names,
  verdicts, dates, or sources.
* Updates edit in place and generalize: the model is told to merge a new
  observation into existing guidance rather than append, and to respect a soft
  soft word target so the document stays distilled rather than accumulating.
* A stable section skeleton keeps successive folds editing the same structure
  rather than reinventing the layout each batch.

Output is delimited by sentinels (not JSON) because the payload is a large
free-text markdown document; sentinels avoid brittle escaping. A single
targeted repair call is issued if the sentinels are missing.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from mafc.common.logger import logger
from mafc.common.modeling import Message, MessageRole, Model, Prompt
from mafc.learning.models import ArticleAnalysis, ClaimLearningRecord

# Sentinels delimiting the two output sections. Kept verbose and unlikely to
# occur inside genuine document prose.
_STRATEGY_BEGIN = "===BEGIN STRATEGY==="
_STRATEGY_END = "===END STRATEGY==="
_CHANGELOG_BEGIN = "===BEGIN CHANGELOG==="
_CHANGELOG_END = "===END CHANGELOG==="

_STRATEGY_RE = re.compile(
    re.escape(_STRATEGY_BEGIN) + r"\s*(.*?)\s*" + re.escape(_STRATEGY_END),
    re.DOTALL,
)
_CHANGELOG_RE = re.compile(
    re.escape(_CHANGELOG_BEGIN) + r"\s*(.*?)\s*" + re.escape(_CHANGELOG_END),
    re.DOTALL,
)

# Recommended starting structure. Used both as guidance in the prompt and as the
# optional seed document the driver can write for an empty run.
DEFAULT_SKELETON = """\
# Strategy.md — Fact-Checking Playbook

> A living, general playbook distilled from professional fact-checks. It captures
> *how* experienced fact-checkers verify claims — reusable methods, not a record
> of individual claims. Use it as guidance, not a rigid script.

## General principles

## Approach by claim type
<!-- One short subsection per recurring claim type (e.g. media authenticity,
     quote attribution, statistics, event claims). Each: what to establish, what
     evidence settles it, where to look. -->

## Evidence techniques
<!-- Reusable techniques (reverse image search, geolocation, source/primary-record
     lookup, metadata checks, etc.): when each applies and what a strong vs weak
     result looks like. -->

## Counter-evidence and common pitfalls
<!-- Failure modes to actively guard against; when to deliberately search for
     contradicting evidence. -->

## When to stop
<!-- How to decide a verdict is sufficiently supported vs needs more digging. -->
"""

_SYSTEM_PROMPT = """\
You are an expert fact-checking methodologist maintaining a single living document, \
`Strategy.md`. This document is a distilled playbook that teaches a fact-checking agent \
*how* to verify claims well. It is the only fact-checking guidance the agent will receive.

CRITICAL — what the document is and is NOT:
- It IS a distillation of transferable, reusable methodology: how to approach each kind of \
claim, which evidence to seek and how to weigh it, which techniques apply when, common \
failure modes, and when a verdict is sufficiently supported.
- It is NOT a log, list, or summary of the specific claims you are shown. Never record \
individual claims, their verdicts, named people/places/sources, specific dates, URLs, or \
case-specific facts. If an example is genuinely illustrative, abstract it into a pattern \
(e.g. "for viral disaster footage, check whether the scene predates the claimed event") \
rather than naming the case.

HOW TO UPDATE (you are editing, not appending):
- Integrate the lessons from the new batch INTO the existing document. Prefer revising or \
merging existing guidance over adding new guidance. Two observations that say the same thing \
must collapse into one.
- Generalize aggressively. The Nth example of a technique should sharpen the existing \
guidance, not lengthen it. When a new case is just another instance of a pattern already \
present, subsume it under the general rule — do NOT append it as one more clause.
- Keep the section skeleton stable. Add a new subsection only for a genuinely new claim type \
or technique not already covered.
- Favor compressing weaker or redundant guidance over growing the document. A tight, high-signal \
document beats a long one.
- If the current document is empty, create it from scratch following the recommended skeleton.

KEEP IT SCANNABLE (this is as important as being correct — a wall of prose is unusable):
- NEVER let a sentence accrete into a long comma- or semicolon-separated list of parallel cases, \
tells, or techniques. The moment a rule would carry ~4+ parallel items, either GENERALIZE them into \
the single umbrella principle, or split them into a short bulleted list / named checklist — each item \
with a brief bold lead-in where it aids scanning. Adding the Nth case must never mean lengthening a \
comma-list; it means sharpening the umbrella rule or adding at most one scannable sub-item.
- De-duplicate ACROSS sections, not just within one. A claim-type subsection should name the decisive \
evidence and DEFER the mechanics to the techniques section rather than re-explaining a technique that \
lives there. State each method once, in its best home.
- Cut obvious, self-evident filler a competent fact-checker already knows; keep only non-obvious, \
decision-relevant judgment. Prefer a named checklist over a dense prose paragraph for any enumeration \
of failure modes or techniques.

RECOMMENDED SKELETON (adapt, don't pad):
{skeleton}

LENGTH TARGET: aim to keep the document around {max_words} words.

OUTPUT FORMAT — emit exactly two sentinel-delimited sections and nothing else:
{changelog_begin}
2-5 bullet points: what you changed and why, referencing the methodological lesson (not the \
specific claims). Note anything you compressed or removed to stay in budget.
{changelog_end}
{strategy_begin}
The complete, updated Strategy.md in markdown. Output the WHOLE document, not a diff.
{strategy_end}
"""

_USER_PROMPT = """\
Here is the current `Strategy.md`. Fold the lessons from the batch of fact-check analyses \
below into it, following all rules. Output the complete updated document.

{strategy_begin}
{current_doc}
{strategy_end}

---FACT-CHECK BATCH ({n} analyses)---
Each entry summarises a professional fact-check: the claim type, what evidence and \
investigative steps the checkers used, and how it resolved. Extract the *method*; discard \
the case specifics.

{batch_section}
---END BATCH---

The current document is {current_words} words (aim for roughly {max_words}). \
Remember: distill transferable methodology, edit in place, generalize rather than append, keep it \
scannable (bulleted, no run-on comma-lists of cases; state each method once in its best section), \
and emit the two sentinel sections only.
"""

_CONSOLIDATE_SYSTEM_PROMPT = """\
You are an expert fact-checking methodologist performing a periodic cleanup of a living `Strategy.md` \
playbook that has been built up incrementally, batch by batch. Incremental building makes the document \
verbose, repetitive, and hard to read: the same method gets restated across sections, obvious filler \
accumulates, and single sentences swell into long comma-separated lists of a dozen cases. Consolidate \
it for READABILITY and concision WITHOUT losing any distinct, transferable method.

Do:
- **De-duplicate across the whole document, not just within a section.** If a method, pitfall, or \
technique is already stated elsewhere, keep the single best home for it and remove the restatements. \
A claim-type subsection should name the decisive evidence and DEFER the mechanics to the techniques \
section rather than re-explaining them.
- **Cut obvious or self-evident guidance.** Remove generic filler a competent fact-checker already \
knows (e.g. "recover the original before judging" as a standalone principle) and keep only the \
non-obvious, decision-relevant judgment. Trimming an obvious *restatement* is not dropping a method.
- **Break run-on enumerations into scannable bullets.** A sentence cramming many parallel cases, tells, \
or techniques into comma/semicolon-separated clauses must become a short bulleted list (or a named \
one-line checklist), each item with a brief bold lead-in where it aids scanning. Prefer a named \
checklist over a dense prose paragraph for any list of failure modes or techniques.
- Tighten wording: collapse near-synonym slash-lists, and move long parenthetical case-lists in a \
heading into a short caption or sub-bullets.
- Keep the section skeleton coherent; a section that merely restates another in a different framing \
should be reduced to a compact cross-referencing checklist.

Do NOT:
- Drop any distinct technique, claim type, pitfall, or stopping rule — every distinct method must \
remain findable afterwards. (De-duplication removes *repetition* of a method; it never removes the \
method's single canonical statement.)
- Add new content, new examples, or any claim-specific facts (names, dates, verdicts, sources).

Aim to keep the document around {max_words} words or fewer, but never sacrifice a distinct method \
just to hit a number. Favor a shorter, well-structured, scannable document over a long prose one.

OUTPUT FORMAT — emit exactly two sentinel-delimited sections and nothing else:
{changelog_begin}
1-3 bullets: what you de-duplicated, cut as obvious, or restructured for readability.
{changelog_end}
{strategy_begin}
The complete consolidated Strategy.md in markdown. Output the WHOLE document.
{strategy_end}
"""

_CONSOLIDATE_USER_PROMPT = """\
Consolidate the living playbook below for quality, following all rules. It is currently \
{current_words} words. Output the complete consolidated document.

{strategy_begin}
{current_doc}
{strategy_end}
"""

_REPAIR_PROMPT = """\
Your previous response did not contain a parseable strategy document. It must include the \
updated document wrapped exactly between the sentinels {strategy_begin} and {strategy_end} \
(and ideally a {changelog_begin}/{changelog_end} section before it). Re-emit your answer in \
the required format. Output the COMPLETE document, not a diff, and nothing outside the \
sentinels.

---YOUR PREVIOUS RESPONSE---
{previous_response}
---END PREVIOUS RESPONSE---
"""

_RECORD_TEMPLATE = """\
[Analysis {i}] claim_type={claim_type} (process_richness={richness})
  claim (context only, do not copy): {claim_text}
  evidence_types: {evidence_types}
  investigative_steps: {steps}
  action -> finding: {links}
  search_queries: {queries}
  how it resolved (method cue, do not copy verdict): {verdict}\
"""


@dataclass
class StrategyFoldResult:
    """Outcome of folding one batch into the strategy document."""

    strategy_md: str
    """The complete updated document. Falls back to the unchanged input document
    when the fold could not be parsed even after a repair attempt."""

    changelog: str
    """Model's short explanation of what changed. Empty when not provided."""

    ok: bool
    """False when the response could not be parsed and the document was left
    unchanged. The driver can use this to flag a degraded batch."""

    llm_prompt: str | None = None
    llm_raw_response: str | None = None


def _truncate(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "…"


def _format_record(i: int, rec: ClaimLearningRecord) -> str:
    a: ArticleAnalysis | None = rec.article_analysis
    claim_text = _truncate(str(rec.claim), 200)
    if a is None:
        return (
            f"[Analysis {i}] (no article analysis available)\n"
            f"  claim (context only, do not copy): {claim_text}"
        )

    steps = "; ".join(a.investigative_steps) if a.investigative_steps else "not described"
    if a.action_evidence_links:
        links = "; ".join(f"{lnk.action} -> {_truncate(lnk.finding, 120)}" for lnk in a.action_evidence_links)
    else:
        links = "not described"
    queries = "; ".join(a.search_queries) if a.search_queries else "not described"

    return _RECORD_TEMPLATE.format(
        i=i,
        claim_type=a.claim_type,
        richness=a.process_richness,
        claim_text=claim_text,
        evidence_types=", ".join(a.evidence_types) if a.evidence_types else "none stated",
        steps=_truncate(steps, 500),
        links=_truncate(links, 600),
        queries=_truncate(queries, 300),
        verdict=_truncate(a.verdict_summary, 200),
    )


class StrategySynthesizer:
    """Folds batches of fact-check analyses into a single ``Strategy.md`` document.

    Stateless across calls: the driver owns the running document and passes it in
    on each :meth:`fold`. One LLM call per batch (plus at most one repair call).

    Args:
        model: LLM used for folding.
        max_words: Soft length target communicated to the model in both fold and
            consolidate prompts. Not enforced — the driver schedules consolidation
            passes on a fixed cadence rather than gating on length.
    """

    def __init__(
        self,
        model: Model,
        max_words: int = 2000,
        max_retries: int = 5,
        retry_base_delay: float = 2.0,
    ) -> None:
        self.model = model
        self.max_words = max_words
        self.max_retries = max_retries
        """Retries on a failed model call before giving up. A long sequential build
        will hit transient network/provider errors; without retries one blip aborts
        the whole run."""
        self.retry_base_delay = retry_base_delay
        """Base seconds for exponential backoff between retries (2,4,8,... )."""

    def _generate_text(self, messages: list[Message]) -> str:
        """Call the model with bounded exponential-backoff retry on any error.

        Transient failures (DNS/connection blips, timeouts, rate limits) are common
        over a multi-hundred-call run; retrying keeps one hiccup from aborting it.
        A persistent error still propagates after the retries are exhausted.
        """
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return self.model.generate(messages).text.strip()
            except Exception as exc:  # noqa: BLE001 — provider errors are heterogeneous
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                delay = self.retry_base_delay * (2**attempt)
                logger.warning(
                    f"[StrategySynthesizer] model.generate failed "
                    f"(attempt {attempt + 1}/{self.max_retries + 1}): "
                    f"{type(exc).__name__}: {exc}. Retrying in {delay:.0f}s."
                )
                time.sleep(delay)
        assert last_exc is not None
        logger.error(
            f"[StrategySynthesizer] model.generate failed after "
            f"{self.max_retries + 1} attempts: {type(last_exc).__name__}: {last_exc}"
        )
        raise last_exc

    def fold(self, strategy_md: str, records: list[ClaimLearningRecord]) -> StrategyFoldResult:
        """Fold one batch of records into the document and return the updated version.

        Args:
            strategy_md: Current document. Pass an empty string to build from scratch.
            records: Fact-check learning records for this batch. Records without an
                article_analysis are still rendered (as "no analysis available") —
                filtering is the driver's responsibility.

        Returns:
            A :class:`StrategyFoldResult`. On unrecoverable parse failure, ``ok`` is
            False and ``strategy_md`` is the unchanged input document.
        """
        if not records:
            logger.warning("[StrategySynthesizer] fold called with empty batch; returning unchanged.")
            return StrategyFoldResult(strategy_md=strategy_md, changelog="", ok=True)

        current_doc = strategy_md.strip() or "(empty — create the document from scratch)"
        batch_section = "\n\n".join(_format_record(i, r) for i, r in enumerate(records, start=1))

        system_prompt = _SYSTEM_PROMPT.format(
            skeleton=DEFAULT_SKELETON,
            max_words=self.max_words,
            changelog_begin=_CHANGELOG_BEGIN,
            changelog_end=_CHANGELOG_END,
            strategy_begin=_STRATEGY_BEGIN,
            strategy_end=_STRATEGY_END,
        )
        user_prompt = _USER_PROMPT.format(
            strategy_begin=_STRATEGY_BEGIN,
            strategy_end=_STRATEGY_END,
            current_doc=current_doc,
            n=len(records),
            batch_section=batch_section,
            max_words=self.max_words,
            current_words=len(strategy_md.split()),
        )

        return self._generate(system_prompt, user_prompt, fallback_doc=strategy_md)

    def consolidate(self, strategy_md: str) -> StrategyFoldResult:
        """Run a quality cleanup pass over the whole document.

        A dedicated call separate from :meth:`fold`, run by the driver on a fixed
        cadence (not budget-triggered): merge duplicate/overlapping guidance, remove
        redundancy, tighten wording, and improve organization while preserving every
        distinct method. On parse failure the document is returned unchanged
        (``ok=False``).
        """
        system_prompt = _CONSOLIDATE_SYSTEM_PROMPT.format(
            max_words=self.max_words,
            changelog_begin=_CHANGELOG_BEGIN,
            changelog_end=_CHANGELOG_END,
            strategy_begin=_STRATEGY_BEGIN,
            strategy_end=_STRATEGY_END,
        )
        user_prompt = _CONSOLIDATE_USER_PROMPT.format(
            strategy_begin=_STRATEGY_BEGIN,
            strategy_end=_STRATEGY_END,
            current_doc=strategy_md.strip(),
            current_words=len(strategy_md.split()),
        )
        return self._generate(system_prompt, user_prompt, fallback_doc=strategy_md)

    # ------------------------------------------------------------------

    def _generate(self, system_prompt: str, user_prompt: str, *, fallback_doc: str) -> StrategyFoldResult:
        """Run one generate + sentinel-parse, with a single targeted repair retry.

        Returns ``ok=False`` and ``fallback_doc`` unchanged when both attempts
        fail to yield a parseable strategy document.
        """
        messages = [
            Message(role=MessageRole.SYSTEM, content=Prompt(text=system_prompt)),
            Message(role=MessageRole.USER, content=Prompt(text=user_prompt)),
        ]

        raw_text = self._generate_text(messages)
        doc, changelog = _parse_response(raw_text)

        if doc is None:
            logger.debug("[StrategySynthesizer] Initial parse failed; issuing repair call.")
            repair_messages = [
                Message(
                    role=MessageRole.USER,
                    content=Prompt(
                        text=_REPAIR_PROMPT.format(
                            strategy_begin=_STRATEGY_BEGIN,
                            strategy_end=_STRATEGY_END,
                            changelog_begin=_CHANGELOG_BEGIN,
                            changelog_end=_CHANGELOG_END,
                            previous_response=raw_text,
                        )
                    ),
                ),
            ]
            repair_text = self._generate_text(repair_messages)
            doc, changelog = _parse_response(repair_text)
            if doc is None:
                logger.warning(
                    "[StrategySynthesizer] Could not parse strategy document after repair; "
                    "leaving document unchanged."
                )
                return StrategyFoldResult(
                    strategy_md=fallback_doc,
                    changelog="",
                    ok=False,
                    llm_prompt=user_prompt,
                    llm_raw_response=repair_text,
                )
            raw_text = repair_text

        return StrategyFoldResult(
            strategy_md=doc,
            changelog=changelog,
            ok=True,
            llm_prompt=user_prompt,
            llm_raw_response=raw_text,
        )


def _parse_response(text: str) -> tuple[str | None, str]:
    """Extract (strategy_doc, changelog) from a sentinel-delimited response.

    Returns (None, "") when the strategy sentinels are absent or empty.
    """
    m = _STRATEGY_RE.search(text)
    if m is None:
        return None, ""
    doc = m.group(1).strip()
    if not doc:
        return None, ""
    cm = _CHANGELOG_RE.search(text)
    changelog = cm.group(1).strip() if cm else ""
    return doc, changelog
