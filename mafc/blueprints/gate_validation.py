from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from mafc.blueprints.features import evaluate_entry_conditions, extract_claim_features
from mafc.blueprints.models import SEMANTIC_FEATURE_NAMES, Blueprint, BlueprintCondition
from mafc.blueprints.semantic_features import SemanticFeatureExtractor
from mafc.common.logger import logger

# A blueprint must be reachable by most of the claims it was synthesized from. Below
# this, an entry condition is excluding the blueprint from its own cluster.
DEFAULT_MIN_SELF_COVERAGE = 0.7


@dataclass
class GateValidationResult:
    """Outcome of validating one blueprint's entry conditions against its own claims."""

    blueprint: Blueprint
    coverage_before: float
    coverage_after: float
    dropped: list[BlueprintCondition] = field(default_factory=list)

    @property
    def repaired(self) -> bool:
        """Whether any condition had to be dropped."""
        return bool(self.dropped)


def _coverage(conditions: list[BlueprintCondition], feature_sets: list) -> float:
    """Fraction of the blueprint's own claims that these 'all' conditions admit."""
    if not feature_sets:
        return 1.0
    entry = type("_Entry", (), {"all": conditions, "any": []})()
    return sum(evaluate_entry_conditions(f, entry)[0] for f in feature_sets) / len(feature_sets)


def validate_entry_gates(
    blueprint: Blueprint,
    claim_texts: list[str],
    extractor: SemanticFeatureExtractor,
    min_self_coverage: float = DEFAULT_MIN_SELF_COVERAGE,
    workers: int = 8,
) -> GateValidationResult:
    """Drop entry conditions that exclude a blueprint from the claims it was built for.

    A gate on a feature that is false for most of the cluster (e.g. a statistics
    blueprint gated on asserts_synthetic_origin) makes the blueprint unreachable by its
    own traffic — strictly worse than having no gate, because the tie-break can no
    longer rescue it. Conditions are dropped lowest-coverage-first until the blueprint
    admits at least `min_self_coverage` of its own claims.
    """
    conditions = list(blueprint.entry_conditions.all)
    if not conditions or not claim_texts:
        return GateValidationResult(blueprint, 1.0, 1.0)

    texts = [text for text in claim_texts if text]
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(texts)))) as pool:
        semantic = list(pool.map(extractor.extract, texts))
    feature_sets = [extract_claim_features(text, features) for text, features in zip(texts, semantic)]
    coverage_before = _coverage(conditions, feature_sets)
    coverage = coverage_before
    dropped: list[BlueprintCondition] = []

    while coverage < min_self_coverage and conditions:
        # Drop whichever single condition is rejecting the most of our own claims.
        worst = min(conditions, key=lambda c: _coverage([c], feature_sets))
        conditions = [c for c in conditions if c is not worst]
        dropped.append(worst)
        coverage = _coverage(conditions, feature_sets)

    if not dropped:
        return GateValidationResult(blueprint, coverage_before, coverage)

    for condition in dropped:
        kind = "semantic" if condition.feature in SEMANTIC_FEATURE_NAMES else "structural"
        logger.warning(
            f"[{blueprint.name}] Dropping {kind} entry gate "
            f"'{condition.feature} {condition.op} {condition.value}': it admits only "
            f"{_coverage([condition], feature_sets):.0%} of this blueprint's own claims."
        )
    logger.info(
        f"[{blueprint.name}] Self-coverage repaired {coverage_before:.0%} -> {coverage:.0%} "
        f"after dropping {len(dropped)} condition(s)."
    )

    updated = blueprint.model_copy(deep=True)
    updated.entry_conditions.all = conditions
    return GateValidationResult(updated, coverage_before, coverage, dropped)
