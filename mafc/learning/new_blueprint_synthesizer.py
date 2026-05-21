"""New blueprint synthesizer: creates blueprints for claims that no existing blueprint covers.

Claims flagged with needs_new_blueprint=True during fit assessment are collected and passed to
NewBlueprintSynthesizer. It operates in two stages:

1. Clustering — a single LLM call groups the unmatched claims by the verification strategy they require (not
    by topic). Each cluster gets a label and a rationale describing the shared investigative approach. Claims
    that are genuine one-offs with no coherent peers are left ungrouped and discarded.

2. Synthesis — clusters that meet a minimum size threshold are each turned into a new blueprint by calling
    BlueprintUpdater with the generic blueprint as a template. An extra hint in the prompt tells the LLM to
    create a specialized blueprint rather than refine the generic one, including setting discriminating entry
    conditions.

The result is one BlueprintSynthesisResult per surviving cluster, each carrying the new blueprint, the cluster
metadata, and the full updater output for inspection.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, model_validator

from mafc.blueprints.models import Blueprint
from mafc.common.logger import logger
from mafc.common.modeling.message import Message, MessageRole
from mafc.common.modeling.model import Model
from mafc.common.modeling.prompt import Prompt
from mafc.learning.blueprint_updater import BlueprintUpdateResult, BlueprintUpdater
from mafc.learning.models import ClaimLearningRecord
from mafc.utils.parsing import extract_json_object, strip_json_fences, try_parse_with_repair

# ---------------------------------------------------------------------------
# Clustering prompts
# ---------------------------------------------------------------------------

_CLUSTER_SYSTEM_PROMPT = """\
You are an expert in fact-checking methodology. Your task is to group a set of claims \
that all failed to match any existing fact-checking blueprint, so that each group can \
become the basis for a new blueprint.

Group claims by the verification STRATEGY they require — not by topic or surface similarity. \
Two claims about completely different subjects belong in the same group if they would be \
verified using the same investigative approach and evidence types.

Bias strongly toward FEWER, BROADER clusters. A new blueprint is only justified when claims \
require a fundamentally different verification graph — different node types, different required \
checks, or a different evidence gathering sequence. Do NOT split claims into separate clusters \
just because they differ in topic, search queries, or specific details. If two groups of claims \
would produce nearly identical blueprint YAMLs, merge them into one cluster. \
When in doubt, merge rather than split.

Each group should be coherent enough that a single blueprint (with shared required checks, \
action intents, and graph structure) could serve all its members well. \
Claims that are genuine one-offs with no coherent peers should be left ungrouped.\
"""

_CLUSTER_USER_PROMPT_TEMPLATE = """\
The following {n} claims all lack a suitable fact-checking blueprint. \
Group them by the verification strategy they require.

---CLAIMS---
{claims_section}
---END CLAIMS---

Return a JSON object:

{{
  "clusters": [
    {{
      "label": string,
      // Short snake_case identifier for this cluster (e.g. "statistical_claim", "identity_claim").

      "rationale": string,
      // Why these claims form a coherent group: what shared verification strategy unites them.

      "claim_indices": [integer, ...]
      // 0-based indices of claims belonging to this cluster. Each claim may appear in at most one cluster.
    }}
  ]
}}

Omit claims from all clusters if they are genuine one-offs. \
Return only the JSON object, no additional text.\
"""

_CLUSTER_REPAIR_PROMPT = """\
The previous response was not valid JSON or did not match the required schema. \
Please return only a valid JSON object with a "clusters" array, no additional text.\
"""

_SYNTHESIS_HINT = """\
NOTE: The blueprint above (generic) is used as a TEMPLATE ONLY. \
Your goal is to create an entirely new, specialized blueprint for the claims shown — \
not to improve the generic blueprint itself. \
Set discriminating entry_conditions that distinguish these claims from what existing \
blueprints already cover, choose a descriptive name, and design a verification graph \
tailored to the shared investigative strategy these claims require.\
"""

# ---------------------------------------------------------------------------
# Pydantic models for clustering LLM output
# ---------------------------------------------------------------------------


class _LlmCluster(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    rationale: str
    claim_indices: list[int]


class _LlmClusterResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clusters: list[_LlmCluster]

    @model_validator(mode="after")
    def deduplicate_indices(self) -> _LlmClusterResponse:
        """Ensure each claim index appears in at most one cluster (first wins)."""
        seen: set[int] = set()
        for cluster in self.clusters:
            unique = [i for i in cluster.claim_indices if i not in seen]
            seen.update(unique)
            cluster.claim_indices = unique
        return self


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class BlueprintSynthesisResult:
    """One synthesized blueprint produced from a cluster of unmatched claims."""

    blueprint: Blueprint
    cluster_label: str
    cluster_rationale: str
    cluster_size: int
    update_result: BlueprintUpdateResult
    """Full updater output, including reasoning and should_split flag."""


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _format_claim_for_clustering(idx: int, rec: ClaimLearningRecord) -> str:
    claim_text = str(rec.claim).strip()
    if len(claim_text) > 300:
        claim_text = claim_text[:300] + "…"

    lines = [f"[{idx}] {claim_text}"]

    if rec.article_analysis is not None:
        lines.append(f"    claim_type: {rec.article_analysis.claim_type}")
        if rec.article_analysis.evidence_types:
            lines.append(f"    evidence_types: {', '.join(rec.article_analysis.evidence_types)}")

    if rec.fit_result is not None:
        if rec.fit_result.missing_capabilities:
            lines.append(f"    missing_capabilities: {', '.join(rec.fit_result.missing_capabilities)}")
        lines.append(f"    fit_level: {rec.fit_result.fit_level}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cluster response parsing
# ---------------------------------------------------------------------------


def _parse_cluster_response(text: str, n_records: int) -> list[_LlmCluster] | None:
    try:
        raw = json.loads(extract_json_object(strip_json_fences(text)))
        validated = _LlmClusterResponse.model_validate(raw)
        clusters = validated.clusters

        # Drop clusters with out-of-range indices
        valid_clusters = []
        for cluster in clusters:
            in_range = [i for i in cluster.claim_indices if 0 <= i < n_records]
            if len(in_range) != len(cluster.claim_indices):
                dropped = set(cluster.claim_indices) - set(in_range)
                logger.warning(
                    f"[NewBlueprintSynthesizer] Cluster '{cluster.label}' "
                    f"contained out-of-range indices {dropped}, dropping them."
                )
                cluster.claim_indices = in_range
            if cluster.claim_indices:
                valid_clusters.append(cluster)

        return valid_clusters
    except Exception as e:
        logger.debug(f"[NewBlueprintSynthesizer] Failed to parse cluster response: {e}")
        return None


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class NewBlueprintSynthesizer:
    """Clusters unmatched claims and synthesizes a new blueprint per cluster.

    Clustering is performed by an LLM. Synthesis reuses BlueprintUpdater with
    the generic blueprint as a starting template, with an added hint that the
    goal is to create a new specialized blueprint rather than refine the generic one.
    """

    def __init__(
        self,
        model: Model,
        updater: BlueprintUpdater,
        generic_blueprint: Blueprint,
        min_cluster_size: int = 3,
    ) -> None:
        self.model = model
        self.updater = updater
        self.generic_blueprint = generic_blueprint
        self.min_cluster_size = min_cluster_size

    def synthesize(self, records: list[ClaimLearningRecord]) -> list[BlueprintSynthesisResult]:
        """Cluster records and synthesize one new blueprint per surviving cluster.

        Args:
            records: Claims flagged with needs_new_blueprint=True. Each should
                have fit_result populated and optionally article_analysis.

        Returns:
            One BlueprintSynthesisResult per cluster that meets min_cluster_size.
            Empty list if no clusters are found or all are too small.
        """
        if not records:
            return []

        clusters = self._cluster(records)
        if not clusters:
            logger.info("[NewBlueprintSynthesizer] LLM produced no clusters.")
            return []

        results: list[BlueprintSynthesisResult] = []
        for cluster in clusters:
            if len(cluster.claim_indices) < self.min_cluster_size:
                logger.debug(
                    f"[NewBlueprintSynthesizer] Dropping cluster '{cluster.label}' "
                    f"(size {len(cluster.claim_indices)} < min {self.min_cluster_size})."
                )
                continue

            cluster_records = [records[i] for i in cluster.claim_indices]
            logger.info(
                f"[NewBlueprintSynthesizer] Synthesizing blueprint for cluster "
                f"'{cluster.label}' ({len(cluster_records)} claims)."
            )

            update_result = self.updater.update(
                self.generic_blueprint,
                cluster_records,
                extra_user_hint=_SYNTHESIS_HINT,
            )
            if update_result is None or update_result.updated_blueprint is None:
                logger.warning(
                    f"[NewBlueprintSynthesizer] Updater returned no blueprint "
                    f"for cluster '{cluster.label}', skipping."
                )
                continue

            results.append(
                BlueprintSynthesisResult(
                    blueprint=update_result.updated_blueprint,
                    cluster_label=cluster.label,
                    cluster_rationale=cluster.rationale,
                    cluster_size=len(cluster_records),
                    update_result=update_result,
                )
            )

        return results

    def _cluster(self, records: list[ClaimLearningRecord]) -> list[_LlmCluster]:
        claims_section = "\n\n".join(_format_claim_for_clustering(i, rec) for i, rec in enumerate(records))
        prompt_text = _CLUSTER_USER_PROMPT_TEMPLATE.format(
            n=len(records),
            claims_section=claims_section,
        )
        messages = [
            Message(role=MessageRole.SYSTEM, content=Prompt(text=_CLUSTER_SYSTEM_PROMPT)),
            Message(role=MessageRole.USER, content=Prompt(text=prompt_text)),
        ]

        response = self.model.generate(messages)
        raw_text = response.text.strip()

        n = len(records)
        result, repair_text = try_parse_with_repair(
            response_text=raw_text,
            parse_fn=lambda t: _parse_cluster_response(t, n),
            model=self.model,
            repair_prompt_prefix=_CLUSTER_REPAIR_PROMPT,
        )

        if result is None:
            logger.warning("[NewBlueprintSynthesizer] Failed to parse cluster response after repair.")
            return []

        if repair_text is not None:
            logger.debug("[NewBlueprintSynthesizer] Repaired cluster JSON parse.")

        logger.debug(
            f"[NewBlueprintSynthesizer] LLM produced {len(result)} cluster(s): "
            + ", ".join(f"'{c.label}' ({len(c.claim_indices)})" for c in result)
        )
        return result
