"""Blueprint consolidator: prunes and merges blueprints at the end of an epoch.

Two operations run in sequence:

1. Pruning — removes blueprints assigned fewer than `prune_threshold` claims in the epoch.
   A low-coverage blueprint is likely an over-specialised artefact from the synthesiser.
   Protected blueprints (e.g. "generic") are never pruned.

2. Merge detection & execution — a single LLM call reviews all remaining non-protected
   blueprints and identifies pairs whose verification strategies overlap enough to merge.
   Each identified pair is merged by calling BlueprintUpdater with the combined claim set,
   then removing the second blueprint from the registry.

Both operations receive the epoch's full list of ClaimLearningRecords (with assigned_blueprint
populated) so they can compute coverage and assemble claim sets without extra bookkeeping.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import yaml
from pydantic import BaseModel, ConfigDict, field_validator

from mafc.blueprints.models import Blueprint
from mafc.blueprints.registry import BlueprintRegistry
from mafc.common.logger import logger
from mafc.common.modeling.message import Message, MessageRole
from mafc.common.modeling.model import Model
from mafc.common.modeling.prompt import Prompt
from mafc.learning.blueprint_updater import BlueprintUpdater
from mafc.learning.models import ClaimLearningRecord
from mafc.utils.parsing import extract_json_object, strip_json_fences, try_parse_with_repair

# ---------------------------------------------------------------------------
# Merge-detection prompts
# ---------------------------------------------------------------------------

_MERGE_SYSTEM_PROMPT = """\
You are an expert in fact-checking workflow design. Your task is to identify which \
blueprints in a pool are redundant or similar enough that merging them would produce \
a better, more general blueprint without losing coverage.

Recommend merging a pair when:
- Their entry conditions overlap substantially (they would route similar claims).
- Their verification graphs follow the same investigative structure.
- Their required checks are nearly identical.
- A single merged blueprint would serve both claim sets at least as well as two separate ones.
- They are topical variants of the same strategy: if two blueprints follow the same \
investigative structure and differ mainly in subject domain (a specific country, region, \
religion, person, or community), they MUST be merged — topic-specific routing adds selector \
noise without any strategic gain. Blueprints are distinguished by HOW they verify, never by \
WHAT topic the claims are about.

Do NOT recommend merging blueprints that serve genuinely different verification strategies, \
even if they share some superficial similarities. When in doubt about strategy overlap, do \
not merge; when the only difference is topic, always merge.\
"""

_MERGE_USER_PROMPT_TEMPLATE = """\
Review the following {n} blueprints and identify any pairs that should be merged. \
Each entry shows the blueprint YAML and how many claims it was assigned in the last epoch, \
along with the claim types seen.

{blueprints_section}

Return a JSON object:

{{
  "merge_groups": [
    {{
      "blueprints": [string, string],
      // Exactly two blueprint names. The first will be kept as the merge base.

      "rationale": string
      // Why these blueprints are redundant or overlapping enough to merge.
    }}
  ]
}}

Return an empty merge_groups list if no merges are warranted. \
Return only the JSON object, no additional text.\
"""

_MERGE_REPAIR_PROMPT = """\
The previous response was not valid JSON or did not match the required schema. \
Please return only a valid JSON object with a "merge_groups" array, no additional text.\
"""

_MERGE_EXECUTION_HINT = """\
NOTE: You are merging two blueprints into one. The claims shown come from BOTH source \
blueprints. Produce a single blueprint that covers all of them well — broaden entry \
conditions if needed, combine required checks, and adjust the verification graph to \
handle the full range. Keep it as focused as possible; do not make it as generic as \
the fallback blueprint.\
"""

# ---------------------------------------------------------------------------
# Pydantic models for merge-detection LLM output
# ---------------------------------------------------------------------------


class _MergeGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    blueprints: list[str]
    rationale: str

    @field_validator("blueprints")
    @classmethod
    def exactly_two(cls, v: list[str]) -> list[str]:
        if len(v) != 2:
            raise ValueError(f"Each merge group must name exactly 2 blueprints, got {len(v)}.")
        if v[0] == v[1]:
            raise ValueError("A blueprint cannot be merged with itself.")
        return v


class _MergeDetectionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    merge_groups: list[_MergeGroup] = []


# ---------------------------------------------------------------------------
# Merge guards
# ---------------------------------------------------------------------------


def apply_merge_budget_guard(merged: Blueprint, base: Blueprint, removed: Blueprint) -> Blueprint:
    """Ensure a merged blueprint keeps at least its parents' investigation budget.

    A merge serves the union of both parents' traffic, so letting the LLM emit a
    smaller max_iterations than either parent silently downgrades investigation
    depth for all of it (the eom_new catch-all collapsed two max_iterations=4
    blueprints into a 3).
    """
    floor = max(
        base.policy_constraints.max_iterations,
        removed.policy_constraints.max_iterations,
    )
    if merged.policy_constraints.max_iterations >= floor:
        return merged
    logger.info(
        f"[BlueprintConsolidator] Merged '{merged.name}' came back with "
        f"max_iterations={merged.policy_constraints.max_iterations} < parents' "
        f"max {floor} — raising to {floor}."
    )
    constraints = merged.policy_constraints.model_copy(update={"max_iterations": floor})
    return merged.model_copy(update={"policy_constraints": constraints})


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ConsolidationResult:
    """Summary of what the consolidator changed in one run."""

    pruned: list[str] = field(default_factory=list)
    """Names of blueprints removed due to low coverage."""

    merged: list[tuple[str, str]] = field(default_factory=list)
    """(kept_name, removed_name) pairs for each executed merge."""

    merge_details: list[dict] = field(default_factory=list)
    """One dict per merge: {"base": ..., "removed": ..., "kept": ...} — ``kept`` is the
    merged blueprint's (possibly renamed) final name. Lets callers carry per-blueprint
    bookkeeping (e.g. cluster sizes) through merges."""

    @property
    def total_changes(self) -> int:
        return len(self.pruned) + len(self.merged)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _compute_coverage(
    records: list[ClaimLearningRecord],
) -> dict[str, list[ClaimLearningRecord]]:
    """Group epoch records by their assigned blueprint name."""
    groups: dict[str, list[ClaimLearningRecord]] = {}
    for rec in records:
        if rec.assigned_blueprint is not None:
            groups.setdefault(rec.assigned_blueprint, []).append(rec)
    return groups


def _sample_claim_types(records: list[ClaimLearningRecord], max_types: int = 6) -> list[str]:
    """Return a deduplicated sample of claim_type values from the records."""
    seen: list[str] = []
    for rec in records:
        if rec.article_analysis and rec.article_analysis.claim_type not in seen:
            seen.append(rec.article_analysis.claim_type)
        if len(seen) >= max_types:
            break
    return seen


def _format_blueprint_for_merge(
    bp: Blueprint,
    records: list[ClaimLearningRecord],
    size_override: int | None = None,
) -> str:
    blueprint_yaml = yaml.dump(
        bp.model_dump(by_alias=True), default_flow_style=False, allow_unicode=True
    ).strip()
    claim_types = _sample_claim_types(records)
    types_str = ", ".join(claim_types) if claim_types else "unknown"
    coverage = size_override if size_override is not None else len(records)
    return (
        f"[{bp.name}]  coverage: {coverage} claims  "
        f"claim types seen: {types_str}\n"
        f"```yaml\n{blueprint_yaml}\n```"
    )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse_merge_response(text: str) -> list[_MergeGroup] | None:
    try:
        raw = json.loads(extract_json_object(strip_json_fences(text)))
        validated = _MergeDetectionResponse.model_validate(raw)
        return validated.merge_groups
    except Exception as e:
        logger.debug(f"[BlueprintConsolidator] Failed to parse merge response: {e}")
        return None


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class BlueprintConsolidator:
    """Prunes low-coverage blueprints and merges overlapping ones.

    Args:
        model: LLM used for merge detection.
        updater: Used to produce the merged blueprint from a combined claim set.
        prune_threshold: Blueprints assigned fewer claims than this in the epoch
            are removed.
        protected_names: Blueprint names that are never pruned or merged.
            Defaults to {"generic"}.
    """

    def __init__(
        self,
        model: Model,
        updater: BlueprintUpdater,
        prune_threshold: int = 2,
        protected_names: set[str] | None = None,
        merge_size_lookup: dict[str, int] | None = None,
        max_merged_size: int | None = None,
    ) -> None:
        self.model = model
        self.updater = updater
        self.prune_threshold = prune_threshold
        self.protected_names: set[str] = protected_names if protected_names is not None else {"generic"}
        self.merge_size_lookup = merge_size_lookup
        """Optional blueprint-name -> claim-count map used for the merge size veto.
        In the offline script path the epoch coverage only counts representatives,
        so real cluster sizes must be supplied separately."""
        self.max_merged_size = max_merged_size
        """When set (with merge_size_lookup), merges whose combined claim count
        exceeds this are vetoed — merging must not recreate the mega-cluster
        catch-alls that the clustering size cap exists to prevent."""

    def consolidate(
        self,
        registry: BlueprintRegistry,
        epoch_records: list[ClaimLearningRecord],
    ) -> ConsolidationResult:
        """Run pruning then merge detection/execution against the current registry.

        Args:
            registry: Modified in place.
            epoch_records: All ClaimLearningRecords processed in the epoch, each
                with assigned_blueprint populated.

        Returns:
            A ConsolidationResult describing what changed.
        """
        result = ConsolidationResult()
        coverage = _compute_coverage(epoch_records)

        self._prune(registry, coverage, result)
        self._merge(registry, coverage, result)

        if result.total_changes:
            logger.info(
                f"[BlueprintConsolidator] pruned={len(result.pruned)} " f"merged={len(result.merged)}"
            )
        else:
            logger.debug("[BlueprintConsolidator] No changes.")

        return result

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------

    def _prune(
        self,
        registry: BlueprintRegistry,
        coverage: dict[str, list[ClaimLearningRecord]],
        result: ConsolidationResult,
    ) -> None:
        for bp in registry.get_all():
            if bp.name in self.protected_names:
                continue
            count = len(coverage.get(bp.name, []))
            if count < self.prune_threshold:
                registry.remove(bp.name)
                result.pruned.append(bp.name)
                logger.info(
                    f"[BlueprintConsolidator] Pruned '{bp.name}' "
                    f"(coverage={count} < threshold={self.prune_threshold})."
                )

    # ------------------------------------------------------------------
    # Merge detection
    # ------------------------------------------------------------------

    def _merge(
        self,
        registry: BlueprintRegistry,
        coverage: dict[str, list[ClaimLearningRecord]],
        result: ConsolidationResult,
    ) -> None:
        candidates = [bp for bp in registry.get_all() if bp.name not in self.protected_names]
        if len(candidates) < 2:
            return

        merge_groups = self._detect_merges(candidates, coverage)
        if not merge_groups:
            return

        # The detector reports pairs, but those pairs form cliques: when it judges A~B,
        # A~C and A~D it has identified one redundant family, not three coincidences.
        # Executing them as disjoint pairs would collapse such a family by a single
        # merge and abandon the rest, so fold each connected component whole.
        for component in self._merge_components(merge_groups):
            base_name = component[0]
            for other_name in component[1:]:
                if not registry.contains(base_name) or not registry.contains(other_name):
                    logger.debug(
                        f"[BlueprintConsolidator] Skipping merge ({base_name}, {other_name}): "
                        "one or both no longer in registry."
                    )
                    continue
                if self.max_merged_size is not None and self.merge_size_lookup is not None:
                    combined_size = self.merge_size_lookup.get(
                        base_name, 0
                    ) + self.merge_size_lookup.get(other_name, 0)
                    if combined_size > self.max_merged_size:
                        logger.info(
                            f"[BlueprintConsolidator] Vetoed merge ({base_name} + {other_name}): "
                            f"combined {combined_size} claims > cap {self.max_merged_size}."
                        )
                        continue

                merged_name = self._execute_merge(
                    registry, coverage, base_name, other_name, result
                )
                if merged_name is None:
                    continue
                # A merge renames the survivor, so the next fold must target the new
                # name and inherit the accumulated coverage and claim count.
                if merged_name != base_name:
                    coverage[merged_name] = coverage.get(base_name, []) + coverage.get(
                        other_name, []
                    )
                    if self.merge_size_lookup is not None:
                        self.merge_size_lookup[merged_name] = self.merge_size_lookup.get(
                            base_name, 0
                        ) + self.merge_size_lookup.get(other_name, 0)
                else:
                    coverage[base_name] = coverage.get(base_name, []) + coverage.get(
                        other_name, []
                    )
                    if self.merge_size_lookup is not None:
                        self.merge_size_lookup[base_name] = self.merge_size_lookup.get(
                            base_name, 0
                        ) + self.merge_size_lookup.get(other_name, 0)
                base_name = merged_name

    @staticmethod
    def _merge_components(merge_groups: list[_MergeGroup]) -> list[list[str]]:
        """Group the detected pairs into connected components, order preserved.

        Ordering follows first appearance so the blueprint the detector named first
        becomes the base of its component, keeping behaviour predictable.
        """
        parent: dict[str, str] = {}
        order: list[str] = []

        def find(name: str) -> str:
            parent.setdefault(name, name)
            while parent[name] != name:
                parent[name] = parent[parent[name]]
                name = parent[name]
            return name

        for group in merge_groups:
            for name in group.blueprints:
                if name not in parent:
                    parent[name] = name
                    order.append(name)
            first, second = find(group.blueprints[0]), find(group.blueprints[1])
            if first != second:
                parent[second] = first

        components: dict[str, list[str]] = {}
        for name in order:
            components.setdefault(find(name), []).append(name)
        return [members for members in components.values() if len(members) > 1]

    def _detect_merges(
        self,
        candidates: list[Blueprint],
        coverage: dict[str, list[ClaimLearningRecord]],
    ) -> list[_MergeGroup]:
        size_lookup = self.merge_size_lookup or {}
        blueprints_section = "\n\n".join(
            _format_blueprint_for_merge(bp, coverage.get(bp.name, []), size_lookup.get(bp.name))
            for bp in candidates
        )
        prompt_text = _MERGE_USER_PROMPT_TEMPLATE.format(
            n=len(candidates),
            blueprints_section=blueprints_section,
        )
        messages = [
            Message(role=MessageRole.SYSTEM, content=Prompt(text=_MERGE_SYSTEM_PROMPT)),
            Message(role=MessageRole.USER, content=Prompt(text=prompt_text)),
        ]

        response = self.model.generate(messages)
        raw_text = response.text.strip()

        groups, repair_text = try_parse_with_repair(
            response_text=raw_text,
            parse_fn=_parse_merge_response,
            model=self.model,
            repair_prompt_prefix=_MERGE_REPAIR_PROMPT,
        )

        if groups is None:
            logger.warning("[BlueprintConsolidator] Failed to parse merge detection response.")
            return []
        if repair_text is not None:
            logger.debug("[BlueprintConsolidator] Repaired merge detection JSON.")

        # Validate that all named blueprints are actually in the candidate set.
        candidate_names = {bp.name for bp in candidates}
        valid_groups = []
        for g in groups:
            if all(name in candidate_names for name in g.blueprints):
                valid_groups.append(g)
            else:
                unknown = [n for n in g.blueprints if n not in candidate_names]
                logger.warning(
                    f"[BlueprintConsolidator] Merge group references unknown blueprint(s) "
                    f"{unknown}, skipping."
                )

        logger.debug(
            f"[BlueprintConsolidator] Detected {len(valid_groups)} merge group(s): "
            + ", ".join(f"({g.blueprints[0]} + {g.blueprints[1]})" for g in valid_groups)
        )
        return valid_groups

    # ------------------------------------------------------------------
    # Merge execution
    # ------------------------------------------------------------------

    def _execute_merge(
        self,
        registry: BlueprintRegistry,
        coverage: dict[str, list[ClaimLearningRecord]],
        base_name: str,
        remove_name: str,
        result: ConsolidationResult,
    ) -> str | None:
        """Merge one blueprint into another. Returns the survivor's name, or None."""
        base_bp = registry.get(base_name)
        remove_bp = registry.get(remove_name)
        combined_records = coverage.get(base_name, []) + coverage.get(remove_name, [])

        update_result = self.updater.update(
            base_bp,
            combined_records,
            extra_user_hint=_MERGE_EXECUTION_HINT,
        )

        if update_result is None or update_result.updated_blueprint is None:
            logger.warning(
                f"[BlueprintConsolidator] Updater failed for merge "
                f"({base_name} + {remove_name}), skipping."
            )
            return None

        merged_bp = apply_merge_budget_guard(update_result.updated_blueprint, base_bp, remove_bp)

        registry.replace(base_name, merged_bp)
        registry.remove(remove_name)
        result.merged.append((merged_bp.name, remove_name))
        result.merge_details.append({"base": base_name, "removed": remove_name, "kept": merged_bp.name})
        logger.info(
            f"[BlueprintConsolidator] Merged '{remove_name}' into '{base_name}' "
            f"→ '{merged_bp.name}' "
            f"({len(combined_records)} combined claims)."
        )
        return merged_bp.name
