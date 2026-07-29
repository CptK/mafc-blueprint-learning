from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from ezmm import MultimodalSequence
from pydantic import BaseModel, ConfigDict

from mafc.blueprints.features import evaluate_entry_conditions, extract_claim_features
from mafc.blueprints.models import Blueprint, ClaimFeatures
from mafc.blueprints.probe import BlueprintProbe, embed_claim
from mafc.blueprints.registry import BlueprintRegistry
from mafc.blueprints.semantic_features import SemanticFeatureExtractor
from mafc.common.logger import logger
from mafc.common.claim import Claim
from mafc.utils.parsing import extract_json_object
from mafc.common.modeling.message import Message, MessageRole
from mafc.common.modeling.model import Model
from mafc.common.modeling.prompt import Prompt
from mafc.learning.models import ArticleAnalysis

# Model.generate already retries transient API *exceptions*. These retries cover the
# other failure mode: a successful call whose body is empty, truncated, or not valid
# JSON. Without them the selector silently routes the claim to the generic blueprint.
_MAX_TIEBREAK_ATTEMPTS = 3

# Hybrid routing threshold. On the 2025 pool the probe was ~94% accurate above 0.8 and
# near chance below 0.6, so this keeps the confident majority and defers the rest.
DEFAULT_PROBE_CONFIDENCE_THRESHOLD = 0.8

_TIEBREAK_REPAIR_SUFFIX = """

Your previous response could not be parsed. Return ONLY the JSON object described \
above — no prose, no markdown fences, no trailing commentary — and make sure \
'selected_blueprint' is copied exactly from one of the candidate names listed above.\
"""


class BlueprintSelectionMethod(str, Enum):
    """Which mechanism decides between blueprints that survive the rule stage."""

    LLM_TIEBREAK = "llm_tiebreak"
    EMBEDDING_PROBE = "embedding_probe"
    # Probe when it is confident, LLM tie-break otherwise.
    HYBRID = "hybrid"


class BlueprintSelectionMode(Enum):
    """How the selector arrived at its final blueprint choice."""

    RULE_BASED = "rule_based"
    LLM_TIEBREAK = "llm_tiebreak"
    GT_INFORMED = "gt_informed"  # LLM tie-break with ground-truth article analysis
    EMBEDDING_PROBE = "embedding_probe"
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
    discriminator: str | None = None
    # No longer requested in the prompt (it dominated the output budget and caused
    # truncation), but still accepted so a volunteered list is not a parse failure.
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
    discriminator: str | None = None
    all_blueprints: list[str] = field(default_factory=list)
    llm_prompt: str | None = None
    llm_raw_response: str | None = None


class BlueprintSelector:
    """Two-stage selector that filters by rules first, then uses an LLM tie-break."""

    def __init__(
        self,
        model: Model,
        registry: BlueprintRegistry,
        default_blueprint_name: str,
        semantic_extractor: SemanticFeatureExtractor | None = None,
        selection_method: BlueprintSelectionMethod = BlueprintSelectionMethod.LLM_TIEBREAK,
        probe: BlueprintProbe | None = None,
        probe_confidence_threshold: float = DEFAULT_PROBE_CONFIDENCE_THRESHOLD,
    ):
        """Initialize the selector with a registry, tie-break model, and default fallback blueprint.

        Args:
            semantic_extractor: Optional extractor for tri-state semantic features.
                When omitted, semantic entry conditions stay undetermined and
                selection behaves exactly as before.
            selection_method: Which mechanism resolves two or more survivors.
            probe: Fitted embedding probe. Required by the probe and hybrid methods;
                without one they degrade to the LLM tie-break.
            probe_confidence_threshold: Hybrid only — below this the probe defers to
                the LLM tie-break.
        """
        self.model = model
        self.registry = registry
        self.default_blueprint_name = default_blueprint_name
        self.semantic_extractor = semantic_extractor
        self.selection_method = selection_method
        self.probe = probe
        self.probe_confidence_threshold = probe_confidence_threshold

        if probe is None and selection_method is not BlueprintSelectionMethod.LLM_TIEBREAK:
            logger.warning(
                f"Blueprint selection method '{selection_method.value}' needs a probe but none "
                f"was supplied; falling back to the LLM tie-break."
            )

    def select(
        self,
        claim: Claim | MultimodalSequence | str,
        article_analysis: ArticleAnalysis | None = None,
    ) -> BlueprintSelectionResult:
        """Select the best blueprint for a claim using filtering and optional LLM tie-break.

        Args:
            claim: The claim to select a blueprint for.
            article_analysis: Optional ground-truth article analysis. When provided,
                it is injected into the LLM tie-break prompt to improve selection
                accuracy. Has no effect on the rule-based hard filtering stage.
        """
        semantic_features = (
            self.semantic_extractor.extract(claim) if self.semantic_extractor is not None else None
        )
        claim_features = extract_claim_features(claim, semantic_features)
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

        probe_result = self._select_with_probe(
            claim, claim_features, survivors, rejected, all_blueprint_names
        )
        if probe_result is not None:
            return probe_result

        return self._select_with_llm(
            claim,
            claim_features,
            survivors,
            rejected,
            default_blueprint,
            all_blueprint_names,
            article_analysis,
        )

    def _select_with_probe(
        self,
        claim: Claim | MultimodalSequence | str,
        claim_features: ClaimFeatures,
        survivors: list[Blueprint],
        rejected: list[BlueprintRejection],
        all_blueprints: list[str],
    ) -> BlueprintSelectionResult | None:
        """Route via the embedding probe, or return None to defer to the LLM tie-break.

        Deferral is the safe direction: the probe is an optimization, so a missing
        artifact, a failed embedding, a low-confidence call, or a prediction naming a
        blueprint that did not survive the rule stage all fall through rather than fail.
        """
        if self.selection_method is BlueprintSelectionMethod.LLM_TIEBREAK or self.probe is None:
            return None

        embedding = embed_claim(str(claim).strip(), self.probe.embedding_model)
        if embedding is None:
            return None

        prediction = self.probe.predict(embedding)
        survivor_names = {blueprint.name for blueprint in survivors}

        if self.selection_method is BlueprintSelectionMethod.HYBRID:
            if prediction.confidence < self.probe_confidence_threshold:
                logger.debug(
                    f"Probe confidence {prediction.confidence:.2f} below "
                    f"{self.probe_confidence_threshold:.2f}; deferring to the LLM tie-break."
                )
                return None

        # The probe knows nothing about entry conditions, so its pick can name a
        # blueprint the rule stage already eliminated. Prefer the best surviving
        # candidate instead of overriding a deterministic gate.
        if prediction.blueprint_name not in survivor_names:
            ranked = sorted(
                (name for name in prediction.scores if name in survivor_names),
                key=lambda name: prediction.scores[name],
                reverse=True,
            )
            if not ranked:
                return None
            selected_name, confidence = ranked[0], prediction.scores[ranked[0]]
        else:
            selected_name, confidence = prediction.blueprint_name, prediction.confidence

        selected_blueprint = next(blueprint for blueprint in survivors if blueprint.name == selected_name)
        return BlueprintSelectionResult(
            selected_blueprint=selected_blueprint,
            selection_mode=BlueprintSelectionMode.EMBEDDING_PROBE,
            claim_features=claim_features,
            surviving_blueprints=[blueprint.name for blueprint in survivors],
            rejected_blueprints=list(rejected),
            reason=f"Embedding probe selected '{selected_name}' with confidence {confidence:.2f}.",
            discriminator=None,
            all_blueprints=all_blueprints,
        )

    def _select_with_llm(
        self,
        claim: Claim | MultimodalSequence | str,
        claim_features: ClaimFeatures,
        survivors: list[Blueprint],
        rejected: list[BlueprintRejection],
        default_blueprint: Blueprint,
        all_blueprints: list[str],
        article_analysis: ArticleAnalysis | None = None,
    ) -> BlueprintSelectionResult:
        """Run the LLM tie-break over the surviving blueprints only."""
        llm_prompt = self._build_tiebreak_prompt(claim, claim_features, survivors, article_analysis)
        selection_mode = (
            BlueprintSelectionMode.GT_INFORMED
            if article_analysis is not None
            else BlueprintSelectionMode.LLM_TIEBREAK
        )
        llm_raw_response: str | None = None
        parsed: LlmTiebreakResponse | None = None
        selected_blueprint: Blueprint | None = None

        for attempt in range(1, _MAX_TIEBREAK_ATTEMPTS + 1):
            prompt_text = llm_prompt if attempt == 1 else llm_prompt + _TIEBREAK_REPAIR_SUFFIX
            failure: str
            try:
                llm_raw_response = self.model.generate(
                    [Message(role=MessageRole.USER, content=Prompt(text=prompt_text))]
                ).text.strip()
                parsed = self._parse_tiebreak_response(llm_raw_response)
                failure = "response did not match the expected schema"
            except (json.JSONDecodeError, ValueError) as exc:
                parsed = None
                failure = f"{type(exc).__name__}: {exc}"

            if parsed is not None:
                selected_blueprint = next(
                    (blueprint for blueprint in survivors if blueprint.name == parsed.selected_blueprint),
                    None,
                )
                if selected_blueprint is not None:
                    break
                failure = f"selected unknown blueprint {parsed.selected_blueprint!r}"

            if attempt < _MAX_TIEBREAK_ATTEMPTS:
                logger.warning(
                    f"Blueprint tie-break unparseable (attempt {attempt}/"
                    f"{_MAX_TIEBREAK_ATTEMPTS}), retrying: {failure}"
                )

        if selected_blueprint is not None and parsed is not None:
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
                selection_mode=selection_mode,
                claim_features=claim_features,
                surviving_blueprints=[blueprint.name for blueprint in survivors],
                rejected_blueprints=llm_rejections,
                reason=parsed.reason,
                discriminator=parsed.discriminator,
                all_blueprints=all_blueprints,
                llm_prompt=llm_prompt,
                llm_raw_response=llm_raw_response,
            )

        logger.error(
            f"Blueprint tie-break failed after {_MAX_TIEBREAK_ATTEMPTS} attempts over "
            f"{len(survivors)} survivors; falling back to '{default_blueprint.name}'. "
            f"Last response: {(llm_raw_response or '')[:200]!r}"
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
        article_analysis: ArticleAnalysis | None = None,
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

        gt_section = ""
        if article_analysis is not None:
            gt_section = (
                "\nGround-truth fact-check analysis (use this to inform your selection):\n"
                f"- Claim type: {article_analysis.claim_type}\n"
                f"- Evidence types used: {', '.join(article_analysis.evidence_types) or 'none'}\n"
                f"- Process richness: {article_analysis.process_richness}\n"
                f"- Verdict summary: {article_analysis.verdict_summary}\n"
            )

        return (
            "You are selecting the most appropriate fact-check blueprint for a claim.\n"
            "Only choose from the provided candidate blueprints.\n"
            "Prefer the blueprint whose description and selector hints best match the claim.\n"
            "\n"
            "Every candidate below already passed the hard entry conditions, so 'it could\n"
            "apply' is not a reason to choose one. Select the NARROWEST blueprint that\n"
            "applies: the one whose description names the specific investigative question\n"
            "this claim raises. A blueprint broad enough to cover most claims of this kind\n"
            "is a fallback, not an answer — never prefer it over a narrower candidate that\n"
            "still fits. Choose the broad candidate only when no narrower one applies.\n"
            "\n"
            "In 'discriminator', name the concrete property of THIS claim that makes the\n"
            "chosen blueprint fit and the closest runner-up not fit. If you cannot name\n"
            "such a property, you are not discriminating between them — reconsider whether\n"
            "a narrower candidate actually applies.\n"
            "\n"
            # Keep the response short. Enumerating a reason per rejected candidate used
            # to overrun the output budget and truncate the JSON mid-string, which the
            # parser could only treat as a total failure.
            "Return strict JSON only, and keep it brief — one sentence per field:\n"
            '{"selected_blueprint":"name","reason":"short reason",'
            '"discriminator":"property of this claim separating it from the runner-up"}\n\n'
            f"Claim:\n{claim_text}\n\n"
            f"Extracted claim features:\n{chr(10).join(feature_lines)}\n"
            f"{gt_section}\n"
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
