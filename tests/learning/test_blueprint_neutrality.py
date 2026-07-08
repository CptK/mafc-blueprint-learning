"""Tests for the verification-stance (neutrality) lint on generated blueprints.

The lint guards against blueprints that presuppose their verdict — the failure
mode found in the learned `recontextualized_media` blueprint, whose "expose the
mismatch" framing flipped authentic media to compromised (neutral rewording cut
its MSE by 21% on identical claims).
"""

from __future__ import annotations

from mafc.blueprints.models import (
    Blueprint,
    BlueprintAction,
    BlueprintActionNode,
    BlueprintEntryConditions,
    BlueprintPolicyConstraints,
    BlueprintRequiredCheck,
    BlueprintSelectorHints,
    BlueprintVerificationGraph,
)
from mafc.learning.blueprint_updater import check_blueprint_neutrality


def _make_blueprint(
    description: str,
    intent: str = "Identify the original news reports describing the event in the media.",
    query_guidance: str = "Search for news coverage matching the visuals.",
    check_description: str = "The earliest upload of the media was located.",
) -> Blueprint:
    return Blueprint(
        name="recontextualized_media",
        description=description,
        entry_conditions=BlueprintEntryConditions(),
        selector_hints=BlueprintSelectorHints(),
        policy_constraints=BlueprintPolicyConstraints(
            allowed_actions=["web_search", "media"], max_iterations=3
        ),
        required_checks=[BlueprintRequiredCheck(id="media_origin_traced", description=check_description)],
        verification_graph=BlueprintVerificationGraph(
            start_node="n1",
            nodes=[
                BlueprintActionNode(
                    id="n1",
                    type="actions",
                    actions=[
                        BlueprintAction(action="web_search", intent=intent, query_guidance=query_guidance)
                    ],
                    transition=[],
                ),
            ],
        ),
    )


NEUTRAL_DESCRIPTION = (
    "Establishes the media's true origin and context via reverse-image-search and "
    "source lookup, then compares it against the claimed context. Confirming that the "
    "media genuinely shows the claimed event is just as valid an outcome as finding a mismatch."
)


def test_neutral_blueprint_passes() -> None:
    bp = _make_blueprint(description=NEUTRAL_DESCRIPTION)
    assert check_blueprint_neutrality(bp) == []


def test_original_recontextualized_media_wording_is_flagged() -> None:
    # Verbatim phrases from the learned blueprint that caused the 2026-07-02 regression.
    bp = _make_blueprint(
        description=(
            "Specialized strategy for viral video/image claims where authentic media "
            "is shared with a false location, date, or context (context_manipulation). "
            "Prioritizes reverse-image-search to find the original upload, then source "
            "lookup and date/location verification to expose the mismatch."
        ),
        check_description=(
            "The claimed date/location/context was compared against the verified "
            "original date/location to detect recontextualization."
        ),
    )
    violations = check_blueprint_neutrality(bp)
    joined = "\n".join(violations)
    assert any("description" in v for v in violations)
    assert "expose" in joined and "shared with a false" in joined
    assert any("required_checks" in v and "detection framing" in v for v in violations)


def test_presuppositional_intent_and_guidance_flagged() -> None:
    bp = _make_blueprint(
        description=NEUTRAL_DESCRIPTION,
        intent="debunk the claim by finding the real footage",
        query_guidance="search fact-checkers to prove the video is fake",
    )
    violations = check_blueprint_neutrality(bp)
    assert len(violations) == 2
    assert all("verification_graph" in v for v in violations)


def test_selector_hints_and_name_are_not_linted() -> None:
    # Claim examples legitimately describe pathologies; the name may reference one too.
    bp = _make_blueprint(description=NEUTRAL_DESCRIPTION)
    bp = bp.model_copy(
        update={
            "selector_hints": BlueprintSelectorHints.model_validate(
                {
                    "positive": {
                        "features": ["has_video"],
                        "examples": ["A video shared with a false location to expose the mismatch."],
                    }
                }
            )
        }
    )
    assert check_blueprint_neutrality(bp) == []
