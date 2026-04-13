from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from typing import Literal

import yaml
from ezmm import MultimodalSequence
from pydantic import BaseModel, ConfigDict

from mafc.blueprints.models import Blueprint, ClaimFeatures
from mafc.common.claim import Claim
from mafc.common.logger import logger
from mafc.common.modeling.message import Message, MessageRole
from mafc.common.modeling.model import Model
from mafc.common.modeling.prompt import Prompt
from mafc.learning.models import ArticleAnalysis
from mafc.utils.parsing import extract_json_object, strip_json_fences, try_parse_with_repair

_SYSTEM_PROMPT = """\
You are an expert in fact-checking methodology and workflow design.
Your task is to assess whether a given fact-checking blueprint is a good fit for verifying a specific claim, \
given what is known about the claim and — where available — how expert fact-checkers actually approached it.

A blueprint defines the analytical strategy for verifying a claim: the verification steps taken, their \
rationale and intent, the required checks that must be satisfied, the query guidance for each action, \
and the conditions under which different verification paths are followed.

Assess fit holistically: consider whether the blueprint's description, required checks, action intents, \
query guidance, and branching conditions match the nature of the claim and the evidence it requires.\
"""

_USER_PROMPT_TEMPLATE = """\
Assess whether the blueprint below is a good fit for verifying the given claim.

---CLAIM---
{claim_text}
---END CLAIM---

---CLAIM FEATURES---
```yaml
{claim_features_yaml}
```
---END CLAIM FEATURES---

---SELECTED BLUEPRINT---
```yaml
{blueprint_yaml}
```
---END SELECTED BLUEPRINT---

{article_analysis_section}\
Return a JSON object with this schema:

{{
  "fit_level": "good" | "partial" | "poor",
  // good    — the blueprint's approach, intents, and checks align well with what this claim requires
  // partial — the blueprint covers some of what is needed but misses important aspects
  // poor    — the blueprint's approach is fundamentally misaligned with this claim's requirements

  "needs_new_blueprint": boolean,
  // true if the existing blueprint pool likely cannot cover this claim well and a new blueprint
  // should be created. Set to false if this or another existing blueprint is probably sufficient.

  "covered_capabilities": [string, ...],
  // evidence types or analytical capabilities the blueprint handles that are relevant to this claim

  "missing_capabilities": [string, ...],
  // evidence types or analytical capabilities this claim requires that the blueprint does not address

  "reason": string
  // 2-3 sentences explaining your assessment
}}

Return only the JSON object, no additional text.\
"""

_ARTICLE_ANALYSIS_SECTION = """\
---ARTICLE ANALYSIS---
The following is a structured analysis of how expert fact-checkers actually verified this claim.
Fields with null values were not described in the article (see process_richness for how much \
investigative detail was available).
```yaml
{article_analysis_yaml}
```
---END ARTICLE ANALYSIS---

"""

_NO_ARTICLE_ANALYSIS_SECTION = """\
No ground-truth article analysis is available. \
Assess fit based on the claim, its features, and the blueprint alone.

"""

_REPAIR_PROMPT = """\
The previous response was not valid JSON or did not match the required schema.
Please return only a valid JSON object matching the required schema, with no additional text.\
"""


class _LlmFitResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fit_level: Literal["good", "partial", "poor"]
    needs_new_blueprint: bool
    covered_capabilities: list[str] = []
    missing_capabilities: list[str] = []
    reason: str


def _parse_fit_response(text: str) -> BlueprintFitResult | None:
    try:
        raw = json.loads(extract_json_object(strip_json_fences(text)))
        validated = _LlmFitResponse.model_validate(raw)
        return BlueprintFitResult(
            fit_level=validated.fit_level,
            needs_new_blueprint=validated.needs_new_blueprint,
            covered_capabilities=validated.covered_capabilities,
            missing_capabilities=validated.missing_capabilities,
            reason=validated.reason,
        )
    except Exception:
        return None


@dataclass
class BlueprintFitResult:
    """Structured assessment of how well a blueprint fits a given claim."""

    fit_level: Literal["good", "partial", "poor"]
    """Holistic fit rating.

    - 'good'    — blueprint approach, intents, and checks align well with the claim
    - 'partial' — blueprint covers some needs but misses important aspects
    - 'poor'    — blueprint is fundamentally misaligned with this claim's requirements
    """

    needs_new_blueprint: bool
    """True when no existing blueprint covers this claim well and a new one should be created."""

    covered_capabilities: list[str]
    """Evidence types or analytical capabilities the blueprint handles that are relevant."""

    missing_capabilities: list[str]
    """Evidence types or capabilities the claim requires that the blueprint does not address."""

    reason: str
    """2-3 sentence explanation of the assessment."""

    llm_prompt: str | None = None
    """The full prompt sent to the LLM, for inspection and logging."""

    llm_raw_response: str | None = None
    """The raw LLM response text that was parsed into this result."""


class BlueprintFitAssessor:
    """Assesses whether a selected blueprint is a good fit for a claim using an LLM."""

    def __init__(self, model: Model):
        self.model = model

    def assess(
        self,
        blueprint: Blueprint,
        claim: Claim | MultimodalSequence | str,
        claim_features: ClaimFeatures,
        article_analysis: ArticleAnalysis | None = None,
        claim_id: str | None = None,
    ) -> BlueprintFitResult | None:
        """Assess whether the blueprint is appropriate for the claim.

        Args:
            blueprint: The blueprint that was selected for this claim.
            claim: The claim to be verified.
            claim_features: Pre-extracted features of the claim.
            article_analysis: Optional ground-truth analysis of how fact-checkers
                actually verified this claim. When provided, gives the LLM a strong
                signal about what capabilities were truly needed.
            claim_id: Optional identifier used in log messages.

        Returns:
            A BlueprintFitResult, or None if the LLM response could not be parsed
            even after a repair attempt.
        """
        label = f"[BlueprintFitAssessor{f' claim={claim_id}' if claim_id else ''}]"

        prompt_text = self._build_prompt(claim, claim_features, blueprint, article_analysis)
        messages = [
            Message(role=MessageRole.SYSTEM, content=Prompt(text=_SYSTEM_PROMPT)),
            Message(role=MessageRole.USER, content=Prompt(text=prompt_text)),
        ]

        response = self.model.generate(messages)
        raw_text = response.text.strip()
        result, repair_text = try_parse_with_repair(
            response_text=raw_text,
            parse_fn=_parse_fit_response,
            model=self.model,
            repair_prompt_prefix=_REPAIR_PROMPT,
        )

        if result is None:
            logger.warning(f"{label} Failed to parse fit assessment after repair.")
            return None

        if repair_text is not None:
            logger.debug(f"{label} Repaired JSON parse.")

        logger.debug(
            f"{label} fit_level={result.fit_level} "
            f"needs_new_blueprint={result.needs_new_blueprint} "
            f"missing={result.missing_capabilities}"
        )

        result.llm_prompt = prompt_text
        result.llm_raw_response = repair_text if repair_text is not None else raw_text
        return result

    def _build_prompt(
        self,
        claim: Claim | MultimodalSequence | str,
        claim_features: ClaimFeatures,
        blueprint: Blueprint,
        article_analysis: ArticleAnalysis | None,
    ) -> str:
        claim_text = str(claim).strip()
        claim_features_yaml = yaml.dump(
            claim_features.model_dump(), default_flow_style=False, allow_unicode=True
        ).strip()
        blueprint_yaml = yaml.dump(
            blueprint.model_dump(by_alias=True), default_flow_style=False, allow_unicode=True
        ).strip()

        if article_analysis is not None:
            analysis_dict = dataclasses.asdict(article_analysis)
            article_analysis_yaml = yaml.dump(
                analysis_dict, default_flow_style=False, allow_unicode=True
            ).strip()
            article_analysis_section = _ARTICLE_ANALYSIS_SECTION.format(
                article_analysis_yaml=article_analysis_yaml
            )
        else:
            article_analysis_section = _NO_ARTICLE_ANALYSIS_SECTION

        return _USER_PROMPT_TEMPLATE.format(
            claim_text=claim_text,
            claim_features_yaml=claim_features_yaml,
            blueprint_yaml=blueprint_yaml,
            article_analysis_section=article_analysis_section,
        )
