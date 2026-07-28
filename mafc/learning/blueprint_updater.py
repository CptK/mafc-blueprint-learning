"""Blueprint updater: improves an existing blueprint from a batch of labelled claims.

Given a blueprint and a set of ClaimLearningRecords (each carrying a fit assessment and optionally a
ground-truth article analysis), BlueprintUpdater asks an LLM to produce a revised version of the blueprint
as a nested JSON object. The LLM sees the current blueprint, all claims in the batch with their fit results
and article analyses, and a field reference explaining how each blueprint component affects runtime behaviour.
It returns a complete replacement blueprint together with a reasoning trace and a flag indicating whether the
batch signals that the blueprint should be split into two more specific ones.

This module is also used by NewBlueprintSynthesizer to create new blueprints from scratch: it passes the
generic blueprint as a template together with an extra hint that instructs the LLM to specialise rather than
refine.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from mafc.blueprints.models import Blueprint
from mafc.common.logger import logger
from mafc.common.modeling.message import Message, MessageRole
from mafc.common.modeling.model import Model
from mafc.common.modeling.prompt import Prompt
from mafc.learning.models import ArticleAnalysis, ClaimLearningRecord
from mafc.learning.outcomes import partition_by_outcome
from mafc.utils.parsing import extract_json_object, strip_json_fences

_BLUEPRINT_FIELD_REFERENCE = """\
BLUEPRINT FIELD REFERENCE

Each field has a specific runtime role. Understanding these helps you make targeted, \
effective updates.

name
  Unique identifier used in logging and registry lookups. Update when the blueprint's \
  scope changes substantially.

description
  Shown to an LLM during tie-break selection when multiple blueprints survive rule \
  filtering. Should concisely distinguish this blueprint from the others. Keep it \
  accurate as the blueprint evolves.

entry_conditions (all / any)
  Evaluated deterministically before any LLM call. Every blueprint whose conditions
  pass becomes a "survivor"; when two or more survive, an LLM tie-break picks between
  them. Entry conditions exist to ELIMINATE blueprints. Only 'all' can eliminate —
  'any' is an OR, so each entry you add there makes the blueprint match MORE claims,
  not fewer. A blueprint with conditions only in 'any' is effectively unrouted: it
  survives nearly every claim and leaves the decision entirely to the tie-break.
  Available boolean features: has_claim_text, has_image, has_video, is_multimodal,
  has_url, has_date, has_question, claim_has_author, claim_has_origin,
  claim_has_meta_info, claim_has_date_metadata.
  Available integer features: text_length, image_count, video_count.
  Available SEMANTIC features (what the claim is ABOUT, not what it contains):
  asserts_place_or_date, asserts_identity, asserts_synthetic_origin,
  asserts_recontextualization, is_document_screenshot, is_quote_attribution,
  is_statistical, is_scientific_medical.
  REQUIRED: put at least one SEMANTIC feature in 'all'. Structural features cannot
  discriminate — most claims in this corpus carry an image or a video, so gating on
  has_image or is_multimodal eliminates nothing. Semantic features are the only way
  to separate blueprints that would otherwise match the same claims. They are
  tri-state: when the extractor cannot determine one it does not eliminate the
  blueprint, so gating on them in 'all' is safe and cannot orphan a claim.
  Choose a SELECTIVE semantic feature — one true of the minority of claims that this
  blueprint is actually for. asserts_place_or_date and asserts_identity are true of
  roughly half of all claims and so discriminate poorly on their own; prefer
  asserts_synthetic_origin, asserts_recontextualization, is_quote_attribution,
  is_statistical, is_scientific_medical, or is_document_screenshot where they fit.
  Supported operators: ==, !=, <, <=, >, >=.
  REQUIRED STRUCTURE — each condition is an object with three keys: feature, op, value.
  Correct JSON structure:
    "entry_conditions": {
      "all": [
        {"feature": "asserts_synthetic_origin", "op": "==", "value": true}
      ],
      "any": [
        {"feature": "has_image", "op": "==", "value": true},
        {"feature": "has_video", "op": "==", "value": true}
      ]
    }
  Widening entry_conditions lets more claims in; narrowing them pushes claims to other
  blueprints or the generic fallback. Make sure conditions stay consistent with what
  the verification graph can actually handle.

selector_hints (positive / negative)
  Used only in LLM tie-break when multiple blueprints survive rule filtering.
  REQUIRED STRUCTURE — both positive and negative are objects with two keys:
    features: list of ClaimFeatures boolean/integer field names that characterise
              claims this blueprint is (or is not) a good fit for.
    examples: short, archetypal, generalised claim strings — not copies of batch
              claims. Derive the pattern from the batch, then write one concise
              illustrative example that would help an LLM selector recognise or
              reject future claims of that type.
  Do NOT put a top-level 'features' key directly inside selector_hints; features
  must always be nested inside positive or negative. Do NOT nest 'negative' inside
  'positive' (they are siblings, not parent/child).
  Correct JSON structure:
    "selector_hints": {
      "positive": {
        "features": ["has_image", "is_multimodal"],
        "examples": ["A viral photo claiming to show event X"]
      },
      "negative": {
        "features": ["has_question"],
        "examples": ["Did politician Y say Z?"]
      }
    }

policy_constraints
  allowed_actions: restricts which action types the planner may delegate. \
    Valid values: web_search, media.
  max_iterations: hard cap on iterations after which finalization is forced. \
    ONE ITERATION EXECUTES ONE NODE — and synthesis nodes cost an iteration exactly \
    like action nodes, because each one makes its own LLM call. So the budget must be \
    at least the number of nodes on the LONGEST path through the graph, counting \
    synthesis nodes. Count that path before setting this. A graph of \
    actions -> synthesis -> actions -> synthesis needs 4, not 2. Setting it lower does \
    not merely shorten runs: the deepest branch becomes unreachable, and the \
    required_checks attached to it never activate, so they report as UNCHECKED rather \
    than failing. Reaching 'finalize' is free and does not consume an iteration.
  require_counterevidence_search: if true, the planner is instructed to search \
    for counter-evidence before finalising. Set for claims where one-sided sourcing \
    is a known failure mode.

required_checks
  Each check (id + description) is shown to the planner at every action node \
  iteration, with a status of UNCHECKED / SUPPORTED / REFUTED / UNCLEAR. \
  The planner uses them as an explicit checklist: it tries to satisfy open checks \
  before finalising, and check statuses are shown in the final synthesis. \
  Add checks for investigative steps that are consistently missing; remove checks \
  that are consistently irrelevant to the claims you see.

verification_graph
  start_node: id of the first node to execute.

  Action nodes (type: "actions"):
    Each action has an 'intent' (what the planner should find) and 'query_guidance' \
    (how to search). These are shown verbatim to the planner as instructions. \
    The planner then decides what to delegate or whether to finalise immediately. \
    More specific intents and query guidance lead to better-targeted evidence.

  Synthesis nodes (type: "synthesis"):
    Execute automatically without an LLM delegation call — they aggregate \
    accumulated evidence into an intermediate state summary. Use them as reflection \
    points between action layers to give the router a clean evidence picture. \
    Synthesis nodes do NOT carry an 'actions' field — only action nodes do.

  Transitions:
    'if: "continue"' — always advances, no LLM routing call needed.
    'if: "<condition string>"' — shown to an LLM router which picks among competing \
      transitions. Condition strings should be concrete and mutually exclusive.
    'to: "finalize"' — stops execution and triggers verdict synthesis.
    Add graph layers when claims consistently need more investigative depth; \
    remove layers that are never reached or consistently skipped.

DO NOT invent fields outside this reference (e.g. no 'decision_logic', no top-level \
'features' inside selector_hints). The schema rejects unknown fields.\
"""

_VERIFICATION_STANCE_RULES = """\
VERIFICATION STANCE — mandatory for every blueprint you produce:

- A blueprint describes how to VERIFY a class of claims, not how to prove a suspected \
pathology. Never presuppose the verdict in the description, required checks, action \
intents, query guidance, or transition conditions. A blueprint framed around "exposing" \
its namesake failure mode (e.g. "expose the mismatch", "debunk the claim", "authentic \
media shared with a false context") measurably biases evidence collection and flips \
authentic claims to compromised. (Validated on VeriTaS: a neutral rewording of one such \
blueprint reduced its MSE by 21% on identical claims and tools.)
- Phrase every investigative step symmetrically. Write "establish the media's true \
origin and compare it with the claimed context — a MATCH is as valid a finding as a \
MISMATCH", never "find the original to expose the false context". Confirming the claim \
is always an equally acceptable outcome of every step.
- Evidence-referent discipline for media claims: instruct that a reverse-image match, \
fact-check, or "original" counts against the media only if it is shown to be THIS media \
(an EXACT match — same frames/image). Similar or PARTIAL matches, stock-site lookalikes, \
and debunks of OTHER videos/photos of the same event are not evidence about this media. \
Reverse-image results label each match EXACT or PARTIAL — blueprints should tell the \
planner to use that distinction.
- When a step cannot establish the origin or context, the guidance must say to report \
it as unverified — not to assume the suspected failure mode.\
"""

_SYSTEM_PROMPT = f"""\
You are an expert in fact-checking workflow design. Your task is to improve an existing \
fact-checking blueprint based on a batch of claims and their fit assessments.

A blueprint defines a complete fact-checking strategy: entry conditions for routing, \
required checks, verification graph nodes with action intents and query guidance, \
selector hints, and policy constraints.

Your goal is to produce an updated blueprint that better serves the claims assigned to it. \
Base changes on evidence across the entire batch:
- If claims consistently show the same missing capability, add it.
- If a required check is consistently flagged as irrelevant, narrow or remove it.
- If claims suggest contradictory changes (e.g. some need more depth, others less), \
  make the most conservative update — expand rather than narrow — and note the contradiction.
- If the batch clearly splits into two subgroups needing fundamentally different strategies, \
  set should_split to true and describe the subgroups.

You must output a complete, valid blueprint as a nested JSON object.

{_VERIFICATION_STANCE_RULES}

{_BLUEPRINT_FIELD_REFERENCE}\
"""

_USER_PROMPT_TEMPLATE = """\
The following blueprint is currently assigned to {n} claims. \
Based on the fit assessments and article analyses below, produce an improved version.

---CURRENT BLUEPRINT---
```json
{blueprint_json}
```
---END CURRENT BLUEPRINT---

---ASSIGNED CLAIMS ({n} total)---
{claims_section}
---END ASSIGNED CLAIMS---
{hint_section}
Return a single JSON object with this shape:

{{
  "reasoning": string,
  // What you changed and why. Note any contradictions across claims and how you resolved them.

  "should_split": boolean,
  // true if the batch splits into two distinct subgroups needing different strategies.

  "split_rationale": string | null,
  // If should_split is true: describe the two subgroups and how their needs differ.

  "updated_blueprint": object
  // The complete updated blueprint as a nested JSON object — NOT a string, NOT YAML.
}}

Example of the required output format (abbreviated):
```json
{{
  "reasoning": "Added reverse-image-search action because 12/15 claims required it.",
  "should_split": false,
  "split_rationale": null,
  "updated_blueprint": {{
    "name": "media_claim",
    "description": "Handles image/video claims",
    "entry_conditions": {{
      "any": [
        {{"feature": "has_image", "op": "==", "value": true}}
      ]
    }},
    "verification_graph": {{
      "start_node": "n1",
      "nodes": [
        {{
          "id": "n1",
          "type": "actions",
          "actions": [
            {{"action": "web_search", "intent": "...", "query_guidance": "..."}}
          ],
          "transition": [{{"if": "continue", "to": "finalize"}}]
        }}
      ]
    }}
  }}
}}
```

Return only the JSON object, no additional text, no markdown fences.\
"""

_REPAIR_PROMPT_TEMPLATE = """\
Your previous response failed validation. Fix the specific errors below and return \
a corrected JSON object with the same outer schema (reasoning, should_split, \
split_rationale, updated_blueprint).

---ERRORS---
{error_description}
---END ERRORS---

---YOUR PREVIOUS RESPONSE---
{previous_response}
---END PREVIOUS RESPONSE---

Return ONLY the corrected JSON object. No explanatory text, no markdown fences. \
Match the schema exactly — do not invent new fields, do not nest things differently. \
The 'updated_blueprint' value must be a nested JSON object, NOT a string and NOT YAML.\
"""

_CLAIM_TEMPLATE = """\
[Claim {i}]
Text: {claim_text}

Fit assessment:
  fit_level: {fit_level}
  missing_capabilities: {missing}
  covered_capabilities: {covered}
  reason: {reason}
{article_section}\
"""

# Outcome-aware claim template — adds the prior-run outcome fields
# alongside the existing fit/article information.
_OUTCOME_CLAIM_TEMPLATE = """\
[Claim {i}]
Text: {claim_text}
True label: {true_label}{outcome_extra}

Fit assessment:
  fit_level: {fit_level}
  missing_capabilities: {missing}
  covered_capabilities: {covered}
  reason: {reason}
{article_section}\
"""

# Headers for the three outcome-grouped sections in the outcomes-on prompt.
_OUTCOME_HEADERS = {
    "correct": "Claims where the prior fact-check produced the CORRECT verdict",
    "incorrect": "Claims where the prior fact-check produced the WRONG verdict",
    "unknown": "Claims with UNKNOWN prior outcome (execution not run or errored)",
}

_OUTCOME_DIRECTIVE = """\
The claims above are grouped by the prior fact-check's outcome with the current blueprint. \
Revise the blueprint to improve performance on the WRONG-verdict cases without regressing \
the CORRECT cases. If the failures share a root cause (e.g. missing action type, \
under-specified intent, weak required check), address it. If the failures look unrelated \
or the batch is dominated by correct outcomes, prefer a minimal revision over an aggressive \
one. If no concrete improvement is supported by the failures, leave the blueprint largely \
unchanged and explain why in 'reasoning'.\
"""

# Outcomes-on variant of the user prompt — the structured outcome sections
# replace the flat claim listing, and an extra directive is appended.
_USER_PROMPT_TEMPLATE_OUTCOMES = """\
The following blueprint is currently assigned to {n} claims. \
Based on the prior-run outcomes, fit assessments, and article analyses below, \
produce an improved version.

---CURRENT BLUEPRINT---
```json
{blueprint_json}
```
---END CURRENT BLUEPRINT---

---ASSIGNED CLAIMS BY OUTCOME ({n} total: {n_correct} correct, {n_incorrect} incorrect, {n_unknown} unknown)---
{outcome_sections}
---END ASSIGNED CLAIMS---

{outcome_directive}
{hint_section}
Return a single JSON object with this shape:

{{
  "reasoning": string,
  // What you changed and why. Reference specific WRONG-verdict claims that drove the change. Note if the batch supports no change.

  "should_split": boolean,
  // true if the batch splits into two distinct subgroups needing different strategies.

  "split_rationale": string | null,
  // If should_split is true: describe the two subgroups and how their needs differ.

  "updated_blueprint": object
  // The complete updated blueprint as a nested JSON object — NOT a string, NOT YAML.
}}

Return only the JSON object, no additional text, no markdown fences.\
"""

_ARTICLE_SECTION_TEMPLATE = """\
Article analysis (process_richness={richness}):
  claim_type: {claim_type}
  evidence_types: {evidence_types}
{steps_section}\
"""


class _LlmUpdateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning: str
    should_split: bool
    split_rationale: str | None = None
    updated_blueprint: dict[str, Any]


@dataclass
class BlueprintUpdateResult:
    """Outcome of one blueprint update call."""

    updated_blueprint: Blueprint | None
    """The parsed and validated updated blueprint. None only when should_split is true and
    no single blueprint covers all claims."""

    should_split: bool
    """Whether the LLM recommends splitting this blueprint into two more specific ones."""

    split_rationale: str | None
    """Description of the two subgroups when should_split is true."""

    reasoning: str
    """LLM explanation of what was changed and why."""

    llm_prompt: str | None = None
    llm_raw_response: str | None = None


def _format_article_section(analysis: ArticleAnalysis) -> str:
    """Render the article analysis fields relevant to blueprint updating as indented text."""
    steps: list[str] = []
    if analysis.investigative_steps:
        steps_yaml: str = yaml.dump(analysis.investigative_steps, default_flow_style=True).strip()
        steps.append(f"  investigative_steps: {steps_yaml}")
    if analysis.action_evidence_links:
        links: list[str] = [f"{lnk.action}: {lnk.finding}" for lnk in analysis.action_evidence_links]
        steps.append("  action_evidence_links:\n" + "\n".join(f"    - {link}" for link in links))
    steps_section: str = ("\n".join(steps) + "\n") if steps else ""

    return _ARTICLE_SECTION_TEMPLATE.format(
        richness=analysis.process_richness,
        claim_type=analysis.claim_type,
        evidence_types=", ".join(analysis.evidence_types) if analysis.evidence_types else "none",
        steps_section=steps_section,
    )


def _format_claim_section(records: list[ClaimLearningRecord]) -> str:
    """Format all claims in the batch into a numbered, human-readable block for the LLM prompt."""
    parts: list[str] = []
    for i, rec in enumerate(records, start=1):
        claim_text: str = str(rec.claim).strip()
        if len(claim_text) > 400:
            claim_text = claim_text[:400] + "…"

        if rec.fit_result is not None:
            fit_level: str = rec.fit_result.fit_level
            missing: str = ", ".join(rec.fit_result.missing_capabilities) or "none"
            covered: str = ", ".join(rec.fit_result.covered_capabilities) or "none"
            reason: str = rec.fit_result.reason
        else:
            fit_level = missing = covered = reason = "N/A"

        article_section: str = (
            _format_article_section(rec.article_analysis) + "\n" if rec.article_analysis is not None else ""
        )

        parts.append(
            _CLAIM_TEMPLATE.format(
                i=i,
                claim_text=claim_text,
                fit_level=fit_level,
                missing=missing,
                covered=covered,
                reason=reason,
                article_section=article_section,
            )
        )
    return "\n".join(parts)


def _format_outcome_claim(i: int, rec: ClaimLearningRecord, bucket: str) -> str:
    """Render one claim for the outcomes-on prompt, with outcome-specific extras.

    The 'incorrect' bucket gets the predicted label and judge reason added — these
    are the failure details the LLM needs to revise against. 'correct' and
    'unknown' buckets get only the true label as context.
    """
    claim_text: str = str(rec.claim).strip()
    if len(claim_text) > 400:
        claim_text = claim_text[:400] + "…"

    if rec.fit_result is not None:
        fit_level: str = rec.fit_result.fit_level
        missing = ", ".join(rec.fit_result.missing_capabilities) or "none"
        covered = ", ".join(rec.fit_result.covered_capabilities) or "none"
        reason = rec.fit_result.reason
    else:
        fit_level = missing = covered = reason = "N/A"

    er = rec.execution_result
    true_label = er.ground_truth if er is not None else "N/A"
    outcome_extra = ""
    if bucket == "incorrect" and er is not None:
        judge_reason = er.judge_reason or "(not recorded)"
        # Compact the judge reason — it can be very long.
        if len(judge_reason) > 600:
            judge_reason = judge_reason[:600] + "…"
        outcome_extra = f"\nPredicted label (wrong): {er.predicted_label}" f"\nJudge reason: {judge_reason}"

    article_section: str = (
        _format_article_section(rec.article_analysis) + "\n" if rec.article_analysis is not None else ""
    )

    return _OUTCOME_CLAIM_TEMPLATE.format(
        i=i,
        claim_text=claim_text,
        true_label=true_label,
        outcome_extra=outcome_extra,
        fit_level=fit_level,
        missing=missing,
        covered=covered,
        reason=reason,
        article_section=article_section,
    )


def _format_outcome_grouped_section(
    records: list[ClaimLearningRecord],
    error_threshold: float | None = None,
) -> tuple[str, int, int, int]:
    """Build the outcome-grouped claim section + return the bucket sizes.

    Only non-empty buckets are rendered; their order is fixed (incorrect first
    because it carries the actionable signal, then correct, then unknown).
    """
    correct, incorrect, unknown = partition_by_outcome(records, error_threshold=error_threshold)
    sections: list[str] = []
    # Stable ordering with explicit indexing across buckets so the LLM can
    # cross-reference [Claim N] in its reasoning unambiguously.
    counter = 1
    for bucket_name, bucket_records in (
        ("incorrect", incorrect),
        ("correct", correct),
        ("unknown", unknown),
    ):
        if not bucket_records:
            continue
        header = f"### {_OUTCOME_HEADERS[bucket_name]} ({len(bucket_records)} claim(s))"
        body = "\n".join(
            _format_outcome_claim(counter + j, r, bucket_name) for j, r in enumerate(bucket_records)
        )
        counter += len(bucket_records)
        sections.append(f"{header}\n\n{body}")
    return "\n\n".join(sections), len(correct), len(incorrect), len(unknown)


def _format_pydantic_errors(e: ValidationError) -> str:
    """Render a Pydantic ValidationError as a compact, model-friendly bullet list."""
    lines: list[str] = []
    for err in e.errors():
        loc = ".".join(str(x) for x in err.get("loc", ()))
        msg = err.get("msg", "validation error")
        err_type = err.get("type", "")
        suffix = f" [{err_type}]" if err_type else ""
        lines.append(f"  - at '{loc}': {msg}{suffix}")
    return "\n".join(lines) if lines else "  - (no detail)"


# ---------------------------------------------------------------------------
# Verification-stance (neutrality) lint
# ---------------------------------------------------------------------------

# High-precision markers of a blueprint that presupposes its verdict instead of
# verifying neutrally (see _VERIFICATION_STANCE_RULES). Matched case-insensitively
# against instruction-bearing text fields only.
_NEUTRALITY_PATTERNS: list[tuple[str, str]] = [
    (
        r"\bexpose\s+(?:the|a|any)\s+(?:mismatch|manipulat\w*|fabricat\w*|deception|false\w*)",
        "aims to 'expose' the suspected failure mode",
    ),
    (
        r"\bdebunk\s+(?:the|this)\s+(?:claim|media|video|image|photo)",
        "instructs to debunk rather than verify",
    ),
    (r"\bshared with a false\b", "presupposes the media context is false"),
    (
        r"\bto detect\s+(?:the\s+)?(?:recontextualiz\w*|miscontextualiz\w*|manipulat\w*|fabricat\w*)",
        "one-sided detection framing (no symmetric match outcome)",
    ),
    (
        r"\bprove\s+(?:the\s+|that\s+|it\s+)?(?:claim\s+|media\s+|video\s+|image\s+)?is\s+(?:false|fake|manipulated|fabricated|miscontextualized)",
        "aims to prove the claim false",
    ),
    (r"\bconfirm\s+(?:the\s+)?(?:hoax|fabrication|manipulation)\b", "presupposes the pathology to confirm"),
]

# Fields whose text is claim/routing material rather than investigator instructions.
_NEUTRALITY_SKIP_KEYS = {"name", "selector_hints", "entry_conditions"}


def _iter_text_fields(value: Any, path: str = "") -> "list[tuple[str, str]]":
    """Collect (path, string) pairs from a nested blueprint dump, skipping non-instruction fields."""
    out: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in _NEUTRALITY_SKIP_KEYS:
                continue
            out.extend(_iter_text_fields(child, f"{path}.{key}" if path else key))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            out.extend(_iter_text_fields(child, f"{path}[{i}]"))
    elif isinstance(value, str):
        out.append((path, value))
    return out


def check_blueprint_neutrality(blueprint: Blueprint) -> list[str]:
    """Return verification-stance violations as '<path>: <issue> ("<matched text>")' strings.

    Empty list means the blueprint passes the neutrality lint.
    """
    import re

    violations: list[str] = []
    for path, text in _iter_text_fields(blueprint.model_dump()):
        for pattern, issue in _NEUTRALITY_PATTERNS:
            m = re.search(pattern, text, flags=re.IGNORECASE)
            if m:
                violations.append(f'{path}: {issue} ("{m.group(0)}")')
    return violations


_NEUTRALITY_REWORD_TEMPLATE = """\
The blueprint you produced violates the mandatory verification stance at the \
locations listed below. Reword ONLY the offending text so every step is neutral and \
symmetric (confirming the claim is as valid an outcome as refuting it), keeping the \
blueprint's structure, checks, and coverage otherwise unchanged.

---VIOLATIONS---
{violations}
---END VIOLATIONS---

---YOUR BLUEPRINT---
```json
{blueprint_json}
```
---END BLUEPRINT---

Return the same outer JSON schema as before (reasoning, should_split, split_rationale, \
updated_blueprint) with should_split={should_split} and split_rationale copied unchanged. \
Return only the JSON object, no additional text, no markdown fences.\
"""


def _parse_update_response(text: str) -> BlueprintUpdateResult:
    """Parse the LLM JSON response and validate the embedded blueprint object.

    Raises ValueError with a descriptive message on any parse or validation failure.
    The message is suitable to feed back into a repair prompt verbatim.
    """
    cleaned = strip_json_fences(text).strip()

    try:
        raw = json.loads(extract_json_object(cleaned))
    except json.JSONDecodeError as e:
        raise ValueError(f"Response is not valid JSON: {e.msg} (line {e.lineno}, column {e.colno}).")

    if not isinstance(raw, dict):
        raise ValueError(f"Top-level JSON value must be an object, got {type(raw).__name__}.")

    try:
        wrapper = _LlmUpdateResponse.model_validate(raw)
    except ValidationError as e:
        raise ValueError(
            "Outer JSON object does not match the required schema "
            "(reasoning, should_split, split_rationale, updated_blueprint):\n"
            f"{_format_pydantic_errors(e)}"
        )

    if not isinstance(wrapper.updated_blueprint, dict):
        raise ValueError(
            f"'updated_blueprint' must be a nested JSON object, "
            f"got {type(wrapper.updated_blueprint).__name__}."
        )

    try:
        blueprint = Blueprint.model_validate(wrapper.updated_blueprint)
    except ValidationError as e:
        raise ValueError(
            "'updated_blueprint' does not match the Blueprint schema:\n" f"{_format_pydantic_errors(e)}"
        )

    return BlueprintUpdateResult(
        updated_blueprint=blueprint,
        should_split=wrapper.should_split,
        split_rationale=wrapper.split_rationale,
        reasoning=wrapper.reasoning,
    )


class BlueprintUpdater:
    """Improves a blueprint using a batch of claims and their fit assessments."""

    def __init__(
        self,
        model: Model,
        use_execution_outcomes: bool = False,
        outcome_error_threshold: float | None = None,
    ) -> None:
        self.model: Model = model
        self.use_execution_outcomes: bool = use_execution_outcomes
        """When true, the user-turn prompt partitions records by prior-run outcome
        (correct / incorrect / unknown) so the LLM can target failures specifically.
        Default false."""
        self.outcome_error_threshold: float | None = outcome_error_threshold
        """When set, the outcome partitioner uses score-error bucketing
        (``abs(predicted_score - gt_score) <= threshold``) instead of strict
        label equality. ``None`` keeps the legacy label-equality semantics."""

    def update(
        self,
        blueprint: Blueprint,
        records: list[ClaimLearningRecord],
        extra_user_hint: str | None = None,
    ) -> BlueprintUpdateResult | None:
        """Produce an updated blueprint from the assigned claim batch.

        Args:
            blueprint: The current blueprint to improve.
            records: Claims assigned to this blueprint, each with fit_result
                and optionally article_analysis populated.
            extra_user_hint: Optional text appended before the JSON schema block
                in the user prompt. Used by NewBlueprintSynthesizer to signal
                that the blueprint is a template for a new specialized one.

        Returns:
            A BlueprintUpdateResult, or None if the LLM response could not be
            parsed even after a targeted repair attempt.
        """
        if not records:
            logger.warning(f"[BlueprintUpdater] Called with empty records for '{blueprint.name}', skipping.")
            return None

        label: str = f"[BlueprintUpdater blueprint={blueprint.name} n={len(records)}]"
        prompt_text = self._build_prompt(blueprint, records, extra_user_hint)

        messages = [
            Message(role=MessageRole.SYSTEM, content=Prompt(text=_SYSTEM_PROMPT)),
            Message(role=MessageRole.USER, content=Prompt(text=prompt_text)),
        ]

        response = self.model.generate(messages)
        raw_text = response.text.strip()

        result, repair_text = self._parse_with_targeted_repair(raw_text, label)

        if result is None:
            return None

        result = self._enforce_neutrality(result, label)

        logger.debug(
            f"{label} should_split={result.should_split} "
            f"updated_name={result.updated_blueprint.name if result.updated_blueprint else 'N/A'}"
        )

        result.llm_prompt = prompt_text
        result.llm_raw_response = repair_text if repair_text is not None else raw_text
        return result

    def _enforce_neutrality(self, result: BlueprintUpdateResult, label: str) -> BlueprintUpdateResult:
        """Lint the produced blueprint against the verification-stance rules; on
        violations, run one rewording call. Falls back to the original result if the
        rewording fails, and warns if violations survive the rewording."""
        violations = check_blueprint_neutrality(result.updated_blueprint)
        if not violations:
            return result
        logger.warning(
            f"{label} Verification-stance violations, requesting rewording:\n  " + "\n  ".join(violations)
        )
        reword_prompt = _NEUTRALITY_REWORD_TEMPLATE.format(
            violations="\n".join(f"- {v}" for v in violations),
            blueprint_json=json.dumps(result.updated_blueprint.model_dump(), ensure_ascii=False, indent=2),
            should_split=json.dumps(result.should_split),
        )
        messages = [
            Message(role=MessageRole.SYSTEM, content=Prompt(text=_SYSTEM_PROMPT)),
            Message(role=MessageRole.USER, content=Prompt(text=reword_prompt)),
        ]
        raw_text = self.model.generate(messages).text.strip()
        reworded, _ = self._parse_with_targeted_repair(raw_text, f"{label}[neutrality-reword]")
        if reworded is None:
            logger.warning(f"{label} Neutrality rewording unparseable; keeping original blueprint.")
            return result
        remaining = check_blueprint_neutrality(reworded.updated_blueprint)
        if remaining:
            logger.warning(
                f"{label} Violations remain after rewording (accepting best effort):\n  "
                + "\n  ".join(remaining)
            )
        # Preserve the original deliberation; the reword call only touches wording.
        reworded.should_split = result.should_split
        reworded.split_rationale = result.split_rationale
        reworded.reasoning = result.reasoning
        return reworded

    def _parse_with_targeted_repair(
        self, raw_text: str, label: str
    ) -> tuple[BlueprintUpdateResult | None, str | None]:
        """Parse the response; on failure, issue one repair call that includes the actual error.

        Returns (result, repair_text). repair_text is None if the first parse succeeded,
        otherwise it is the verbatim text of the repair response (used downstream for logging).
        result is None when both parses fail.
        """
        try:
            return _parse_update_response(raw_text), None
        except ValueError as e:
            initial_error_msg = str(e)
            logger.debug(f"{label} Initial parse failed: {initial_error_msg}")

        repair_prompt = _REPAIR_PROMPT_TEMPLATE.format(
            error_description=initial_error_msg,
            previous_response=raw_text,
        )
        repair_messages = [
            Message(role=MessageRole.USER, content=Prompt(text=repair_prompt)),
        ]
        repair_response = self.model.generate(repair_messages)
        repair_text = repair_response.text.strip()

        try:
            return _parse_update_response(repair_text), repair_text
        except ValueError as repair_error:
            logger.warning(f"{label} Failed to parse update response after repair: {repair_error}")
            return None, repair_text

    def _build_prompt(
        self,
        blueprint: Blueprint,
        records: list[ClaimLearningRecord],
        extra_user_hint: str | None = None,
    ) -> str:
        """Render the full user-turn prompt from the current blueprint and claim batch.

        When ``use_execution_outcomes`` is true and at least one record carries
        execution-outcome data, the prompt switches to outcome-grouped sections;
        otherwise it falls back to the original flat claim listing.
        """
        blueprint_json = json.dumps(blueprint.model_dump(by_alias=True), indent=2, ensure_ascii=False)
        hint_section = f"\n{extra_user_hint}\n" if extra_user_hint else ""

        if self.use_execution_outcomes:
            outcome_sections, n_c, n_i, n_u = _format_outcome_grouped_section(
                records, error_threshold=self.outcome_error_threshold
            )
            # If we have no signal at all (everything in 'unknown'), the outcomes
            # framing adds noise without information. Fall back to the original
            # prompt so the LLM doesn't see a misleading "all unknown" categorisation.
            if n_c + n_i > 0:
                return _USER_PROMPT_TEMPLATE_OUTCOMES.format(
                    n=len(records),
                    n_correct=n_c,
                    n_incorrect=n_i,
                    n_unknown=n_u,
                    blueprint_json=blueprint_json,
                    outcome_sections=outcome_sections,
                    outcome_directive=_OUTCOME_DIRECTIVE,
                    hint_section=hint_section,
                )

        claims_section = _format_claim_section(records)
        return _USER_PROMPT_TEMPLATE.format(
            n=len(records),
            blueprint_json=blueprint_json,
            claims_section=claims_section,
            hint_section=hint_section,
        )
