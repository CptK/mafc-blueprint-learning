from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from ezmm import MultimodalSequence

from mafc.common.claim import Claim
from mafc.blueprints.models import ClaimFeatures

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")
MONTH_PATTERN = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
    flags=re.IGNORECASE,
)


def extract_claim_features(claim: Claim | MultimodalSequence | str) -> ClaimFeatures:
    """Extract a deterministic typed feature set from a claim-like input."""
    claim_sequence = _as_multimodal_sequence(claim)
    text = str(claim_sequence).strip()

    images = list(claim_sequence.images)
    videos = list(claim_sequence.videos)

    feature_values: dict[str, Any] = {
        "has_claim_text": bool(text),
        "text_length": len(text),
        "has_image": bool(images),
        "image_count": len(images),
        "has_video": bool(videos),
        "video_count": len(videos),
        "is_multimodal": bool(images or videos),
        "has_url": bool(URL_PATTERN.search(text)),
        "has_date": bool(YEAR_PATTERN.search(text) or MONTH_PATTERN.search(text)),
        "has_question": "?" in text,
    }

    if isinstance(claim, Claim):
        feature_values.update(
            {
                "claim_has_author": claim.author is not None,
                "claim_has_origin": claim.origin is not None,
                "claim_has_meta_info": claim.meta_info is not None,
                "claim_has_date_metadata": claim.date is not None,
            }
        )

    return ClaimFeatures.model_validate(feature_values)


def evaluate_entry_conditions(
    features: ClaimFeatures | Mapping[str, Any], entry_conditions: Any
) -> tuple[bool, list[str]]:
    """Evaluate blueprint entry conditions against a claim feature map."""
    feature_map = features.model_dump() if isinstance(features, ClaimFeatures) else dict(features)
    reasons: list[str] = []

    for condition in entry_conditions.all:
        if not _evaluate_condition(feature_map, condition):
            reasons.append(
                f"Required condition failed: {condition.feature} {condition.op} {condition.value!r}"
            )

    if entry_conditions.any:
        any_matches = any(_evaluate_condition(feature_map, condition) for condition in entry_conditions.any)
        if not any_matches:
            reasons.append("No optional any-condition matched.")

    return not reasons, reasons


def _as_multimodal_sequence(claim: Claim | MultimodalSequence | str) -> MultimodalSequence:
    """Normalize a claim-like input into a multimodal sequence."""
    if isinstance(claim, MultimodalSequence):
        return claim
    return MultimodalSequence(str(claim))


def _evaluate_condition(features: Mapping[str, Any], condition: Any) -> bool:
    """Evaluate one feature predicate against the extracted feature map."""
    actual = features.get(condition.feature)
    expected = condition.value
    operator = condition.op

    if operator == "==":
        return actual == expected
    if operator == "!=":
        return actual != expected
    if operator == ">":
        return actual is not None and actual > expected
    if operator == ">=":
        return actual is not None and actual >= expected
    if operator == "<":
        return actual is not None and actual < expected
    if operator == "<=":
        return actual is not None and actual <= expected
    if operator == "in":
        return actual in expected if expected is not None else False
    if operator == "contains":
        return expected in actual if actual is not None else False
    if operator == "not_contains":
        return expected not in actual if actual is not None else True
    raise ValueError(f"Unsupported blueprint condition operator: {operator}")
