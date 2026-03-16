from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum

from ezmm import MultimodalSequence
from pydantic import BaseModel, ConfigDict

from mafc.blueprints.features import evaluate_entry_conditions, extract_claim_features
from mafc.blueprints.models import Blueprint, ClaimFeatures
from mafc.blueprints.registry import BlueprintRegistry
from mafc.common.claim import Claim
from mafc.utils.parsing import extract_json_object
from mafc.common.modeling.message import Message, MessageRole
from mafc.common.modeling.model import Model
from mafc.common.modeling.prompt import Prompt


class BlueprintSelectionMode(Enum):
    """How the selector arrived at its final blueprint choice."""

    RULE_BASED = "rule_based"
    LLM_TIEBREAK = "llm_tiebreak"
    DEFAULT_FALLBACK = "default_fallback"


class LlmRejectedBlueprint(BaseModel):
    """Structured rejection explanation returned by the LLM tie-break."""

    model_config = ConfigDict(extra="forbid")

    name: str
    reason: str


class LlmTiebreakResponse(BaseModel):
    """Validated payload returned by the LLM tie-break prompt."""

    model_config = ConfigDict(extra="forbid")

    selected_blueprint: str
    reason: str | None = None
    rejected_blueprints: list[LlmRejectedBlueprint] = []


@dataclass
class BlueprintRejection:
    """Explanation for why one blueprint was not selected."""

    blueprint_name: str
    reason: str


@dataclass
class BlueprintSelectionResult:
    """Structured result returned by the blueprint selector."""

    selected_blueprint: Blueprint
    selection_mode: BlueprintSelectionMode
    claim_features: ClaimFeatures
    surviving_blueprints: list[str]
    rejected_blueprints: list[BlueprintRejection] = field(default_factory=list)
    reason: str | None = None
    all_blueprints: list[str] = field(default_factory=list)
    llm_prompt: str | None = None
    llm_raw_response: str | None = None


class BlueprintSelector:
    """Two-stage selector that filters by rules first, then uses an LLM tie-break."""

    def __init__(self, model: Model, registry: BlueprintRegistry, default_blueprint_name: str):
        """Initialize the selector with a registry, tie-break model, and default fallback blueprint."""
        self.model = model
        self.registry = registry
        self.default_blueprint_name = default_blueprint_name

    def select(
        self,
        claim: Claim | MultimodalSequence | str,
    ) -> BlueprintSelectionResult:
        """Select the best blueprint for a claim using filtering and optional LLM tie-break."""
        claim_features = extract_claim_features(claim)
        default_blueprint = self.registry.get(self.default_blueprint_name)
        blueprints = [
            blueprint
            for blueprint in self.registry.get_all()
            if blueprint.name != self.default_blueprint_name
        ]
        all_blueprint_names = [blueprint.name for blueprint in blueprints]

        survivors: list[Blueprint] = []
        rejected: list[BlueprintRejection] = []
        for blueprint in blueprints:
            matched, reasons = evaluate_entry_conditions(claim_features, blueprint.entry_conditions)
            if matched:
                survivors.append(blueprint)
            else:
                rejected.append(
                    BlueprintRejection(
                        blueprint_name=blueprint.name,
                        reason="; ".join(reasons),
                    )
                )

        if len(survivors) == 1:
            return BlueprintSelectionResult(
                selected_blueprint=survivors[0],
                selection_mode=BlueprintSelectionMode.RULE_BASED,
                claim_features=claim_features,
                surviving_blueprints=[survivors[0].name],
                rejected_blueprints=rejected,
                reason="Exactly one blueprint matched the rule-based entry conditions.",
                all_blueprints=all_blueprint_names,
            )

        if not survivors:
            return BlueprintSelectionResult(
                selected_blueprint=default_blueprint,
                selection_mode=BlueprintSelectionMode.DEFAULT_FALLBACK,
                claim_features=claim_features,
                surviving_blueprints=[],
                rejected_blueprints=rejected,
                reason="No blueprint matched the rule-based entry conditions.",
                all_blueprints=all_blueprint_names,
            )

        return self._select_with_llm(
            claim, claim_features, survivors, rejected, default_blueprint, all_blueprint_names
        )

    def _select_with_llm(
        self,
        claim: Claim | MultimodalSequence | str,
        claim_features: ClaimFeatures,
        survivors: list[Blueprint],
        rejected: list[BlueprintRejection],
        default_blueprint: Blueprint,
        all_blueprints: list[str],
    ) -> BlueprintSelectionResult:
        """Run the LLM tie-break over the surviving blueprints only."""
        llm_prompt = self._build_tiebreak_prompt(claim, claim_features, survivors)
        prompt = Prompt(text=llm_prompt)
        llm_raw_response: str | None = None
        parsed = None

        try:
            llm_raw_response = self.model.generate(
                [Message(role=MessageRole.USER, content=prompt)]
            ).text.strip()
            parsed = self._parse_tiebreak_response(llm_raw_response)
        except (json.JSONDecodeError, ValueError):
            pass

        if parsed is not None:
            selected_name = parsed.selected_blueprint
            selected_blueprint = next(
                (blueprint for blueprint in survivors if blueprint.name == selected_name), None
            )
            if selected_blueprint is not None:
                llm_rejections = list(rejected)
                for item in parsed.rejected_blueprints:
                    llm_rejections.append(
                        BlueprintRejection(
                            blueprint_name=item.name,
                            reason=item.reason,
                        )
                    )
                return BlueprintSelectionResult(
                    selected_blueprint=selected_blueprint,
                    selection_mode=BlueprintSelectionMode.LLM_TIEBREAK,
                    claim_features=claim_features,
                    surviving_blueprints=[blueprint.name for blueprint in survivors],
                    rejected_blueprints=llm_rejections,
                    reason=parsed.reason,
                    all_blueprints=all_blueprints,
                    llm_prompt=llm_prompt,
                    llm_raw_response=llm_raw_response,
                )

        fallback_rejections = list(rejected)
        fallback_rejections.extend(
            BlueprintRejection(
                blueprint_name=blueprint.name,
                reason="LLM tie-break did not return a valid survivor selection.",
            )
            for blueprint in survivors
        )
        return BlueprintSelectionResult(
            selected_blueprint=default_blueprint,
            selection_mode=BlueprintSelectionMode.DEFAULT_FALLBACK,
            claim_features=claim_features,
            surviving_blueprints=[blueprint.name for blueprint in survivors],
            rejected_blueprints=fallback_rejections,
            reason="Multiple blueprints survived rule filtering, but the LLM tie-break was invalid.",
            all_blueprints=all_blueprints,
            llm_prompt=llm_prompt,
            llm_raw_response=llm_raw_response,
        )

    def _build_tiebreak_prompt(
        self,
        claim: Claim | MultimodalSequence | str,
        claim_features: ClaimFeatures,
        survivors: list[Blueprint],
    ) -> str:
        """Build a compact selection prompt for the LLM tie-break."""
        claim_text = str(claim).strip()
        feature_lines = [f"- {key}: {value}" for key, value in sorted(claim_features.model_dump().items())]
        candidate_blocks = []
        for blueprint in survivors:
            positive = blueprint.selector_hints.positive
            negative = blueprint.selector_hints.negative
            candidate_blocks.append(
                "\n".join(
                    [
                        f"Name: {blueprint.name}",
                        f"Description: {blueprint.description}",
                        f"Positive features: {', '.join(positive.features) if positive.features else 'None'}",
                        f"Positive examples: {' | '.join(positive.examples) if positive.examples else 'None'}",
                        f"Negative features: {', '.join(negative.features) if negative.features else 'None'}",
                        f"Negative examples: {' | '.join(negative.examples) if negative.examples else 'None'}",
                    ]
                )
            )

        return (
            "You are selecting the most appropriate fact-check blueprint for a claim.\n"
            "Only choose from the provided candidate blueprints.\n"
            "Prefer the blueprint whose description and selector hints best match the claim.\n"
            "Return strict JSON only with schema:\n"
            '{"selected_blueprint":"name","reason":"short reason","rejected_blueprints":[{"name":"other","reason":"why not"}]}\n\n'
            f"Claim:\n{claim_text}\n\n"
            f"Extracted claim features:\n{chr(10).join(feature_lines)}\n\n"
            f"Candidate blueprints:\n\n{chr(10).join(candidate_blocks)}"
        )

    def _parse_tiebreak_response(self, response_text: str) -> LlmTiebreakResponse | None:
        """Parse and validate the LLM tie-break response payload."""
        text = response_text.strip()
        if text.startswith("```"):
            lines = [line for line in text.splitlines() if not line.startswith("```")]
            text = "\n".join(lines).strip()

        payload = json.loads(extract_json_object(text))
        try:
            return LlmTiebreakResponse.model_validate(payload)
        except Exception:
            return None
