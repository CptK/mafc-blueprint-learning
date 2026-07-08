"""Cross-blueprint contrast pass and iteration-budget guards.

Blueprints are synthesized one cluster at a time, so nothing ensures their
descriptions and selector hints partition the claim space. At eval time nearly
all claims are routed by an LLM tiebreak over these descriptions; when several
blueprints read alike, one broadly-worded catch-all can absorb half the traffic
(the eom_new regression: media_origin_context took 47% of claims and
under-investigated them). This module adds:

- ``BlueprintContrastPass`` — one LLM call over the *whole* pool that rewrites
  each blueprint's description and selector hints for mutual exclusivity,
  without touching strategy content. Revisions that fail the neutrality lint
  or arrive structurally invalid are dropped per-blueprint.
- ``enforce_iteration_floor`` — mechanical guard tying max_iterations to the
  traffic share a blueprint is expected to serve.
"""

from __future__ import annotations

import json
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

from mafc.blueprints.models import Blueprint, BlueprintSelectorHints
from mafc.common.logger import logger
from mafc.common.modeling.message import Message, MessageRole
from mafc.common.modeling.model import Model
from mafc.common.modeling.prompt import Prompt
from mafc.learning.blueprint_updater import check_blueprint_neutrality
from mafc.utils.parsing import extract_json_object, strip_json_fences, try_parse_with_repair

# ---------------------------------------------------------------------------
# Iteration-budget guard
# ---------------------------------------------------------------------------

HIGH_TRAFFIC_SHARE = 0.10
HIGH_TRAFFIC_ITERATION_FLOOR = 4


def enforce_iteration_floor(
    blueprint: Blueprint,
    expected_share: float,
    share_threshold: float = HIGH_TRAFFIC_SHARE,
    floor: int = HIGH_TRAFFIC_ITERATION_FLOOR,
) -> Blueprint:
    """Raise max_iterations for blueprints expected to serve a large traffic share.

    A blueprint born from a large cluster handles a proportionally large share of
    eval claims; giving it a smaller investigation budget than niche blueprints
    is exactly backwards (the 47%-traffic catch-all in eom_new ran with
    max_iterations=3 while 29-claim shards got 4).
    """
    if expected_share < share_threshold:
        return blueprint
    if blueprint.policy_constraints.max_iterations >= floor:
        return blueprint
    logger.info(
        f"[enforce_iteration_floor] '{blueprint.name}' expected share "
        f"{expected_share:.0%} >= {share_threshold:.0%} — raising max_iterations "
        f"{blueprint.policy_constraints.max_iterations} → {floor}."
    )
    constraints = blueprint.policy_constraints.model_copy(update={"max_iterations": floor})
    return blueprint.model_copy(update={"policy_constraints": constraints})


# ---------------------------------------------------------------------------
# Contrast pass
# ---------------------------------------------------------------------------

_CONTRAST_SYSTEM_PROMPT = """\
You are an expert in fact-checking workflow design. A pool of verification \
blueprints is routed at runtime by an LLM selector that reads ONLY each \
blueprint's description and selector hints. Your task is to revise those two \
fields across the whole pool so the blueprints partition the claim space as \
crisply as possible.

Rules:
- For each blueprint, the description must state (1) which claims it handles and \
(2) in one sentence, what distinguishes it from the most similar other blueprint \
in the pool. Selector-hint examples across blueprints must be mutually exclusive: \
a given claim should match exactly one blueprint's examples.
- Broadly-worded descriptions act as catch-alls and absorb traffic that better \
specialized blueprints should receive. Make each description as narrow as its \
actual specialty.
- Do NOT change verification strategy: you may only rewrite `description` and \
`selector_hints`. Never touch verification graphs, required checks, policy \
constraints, or entry conditions.
- Verification stance: descriptions must never presuppose a verdict. Never imply \
the media/claim is expected to be false, manipulated, or miscontextualized — \
confirming a claim is as valid an outcome as refuting it.
- If two blueprints cannot be meaningfully distinguished, still sharpen the wording \
as far as possible and note the overlap in `notes`.\
"""

_CONTRAST_USER_TEMPLATE = """\
Here are the {n} blueprints in the pool. `expected_share` is the fraction of \
runtime claims each is expected to serve (from training-cluster sizes).

{blueprints_section}

Return a JSON object:

{{
  "notes": string,
  // Brief observations about remaining overlaps, if any.

  "revisions": [
    {{
      "name": string,
      // Exact name of the blueprint being revised. Include every blueprint.

      "description": string,
      // Revised description: what it handles + what distinguishes it from its
      // nearest sibling in the pool.

      "selector_hints": {{
        "positive": {{"features": [string], "examples": [string]}},
        "negative": {{"features": [string], "examples": [string]}}
      }}
      // Revised hints. Positive examples must not plausibly match any other
      // blueprint in the pool; add negative examples naming the claims its
      // nearest sibling should take instead.
    }}
  ]
}}

Return only the JSON object, no additional text.\
"""

_CONTRAST_REPAIR_PROMPT = """\
The previous response was not valid JSON or did not match the required schema. \
Please return only a valid JSON object with "notes" and a "revisions" array, \
no additional text.\
"""


class _ContrastRevision(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    description: str
    selector_hints: dict[str, Any]


class _ContrastResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    notes: str = ""
    revisions: list[_ContrastRevision] = []


def _parse_contrast_response(text: str) -> _ContrastResponse | None:
    try:
        raw = json.loads(extract_json_object(strip_json_fences(text)))
        return _ContrastResponse.model_validate(raw)
    except Exception as e:
        logger.debug(f"[BlueprintContrastPass] Failed to parse response: {e}")
        return None


def _format_blueprint(bp: Blueprint, expected_share: float) -> str:
    routing_view = {
        "name": bp.name,
        "description": bp.description,
        "selector_hints": bp.selector_hints.model_dump(),
        "entry_conditions": bp.entry_conditions.model_dump(),
        "required_checks": [c.id for c in bp.required_checks],
        "allowed_actions": bp.policy_constraints.allowed_actions,
    }
    body = yaml.dump(routing_view, default_flow_style=False, allow_unicode=True, sort_keys=False).strip()
    return f"[{bp.name}]  expected_share: {expected_share:.1%}\n```yaml\n{body}\n```"


class BlueprintContrastPass:
    """Rewrites descriptions and selector hints across a blueprint pool for contrast.

    Args:
        model: LLM used for the single pool-wide revision call.
        protected_names: Blueprints excluded from revision (e.g. the generic
            fallback, whose description should stay broad).
    """

    def __init__(self, model: Model, protected_names: set[str] | None = None) -> None:
        self.model = model
        self.protected_names: set[str] = protected_names if protected_names is not None else {"generic"}

    def run(
        self,
        blueprints: list[Blueprint],
        expected_shares: dict[str, float],
    ) -> list[Blueprint]:
        """Return the pool with revised descriptions/selector hints.

        Any blueprint whose revision is missing, structurally invalid, or fails
        the neutrality lint keeps its original fields. Order is preserved.
        """
        candidates = [bp for bp in blueprints if bp.name not in self.protected_names]
        if len(candidates) < 2:
            return blueprints

        blueprints_section = "\n\n".join(
            _format_blueprint(bp, expected_shares.get(bp.name, 0.0)) for bp in candidates
        )
        messages = [
            Message(role=MessageRole.SYSTEM, content=Prompt(text=_CONTRAST_SYSTEM_PROMPT)),
            Message(
                role=MessageRole.USER,
                content=Prompt(
                    text=_CONTRAST_USER_TEMPLATE.format(
                        n=len(candidates), blueprints_section=blueprints_section
                    )
                ),
            ),
        ]

        response = self.model.generate(messages)
        parsed, _ = try_parse_with_repair(
            response_text=response.text.strip(),
            parse_fn=_parse_contrast_response,
            model=self.model,
            repair_prompt_prefix=_CONTRAST_REPAIR_PROMPT,
        )
        if parsed is None:
            logger.warning("[BlueprintContrastPass] Could not parse revision response — pool unchanged.")
            return blueprints
        if parsed.notes:
            logger.info(f"[BlueprintContrastPass] Notes: {parsed.notes}")

        revisions = {r.name: r for r in parsed.revisions}
        out: list[Blueprint] = []
        for bp in blueprints:
            out.append(self._apply_revision(bp, revisions.get(bp.name)))
        return out

    def _apply_revision(self, bp: Blueprint, revision: _ContrastRevision | None) -> Blueprint:
        if revision is None or bp.name in self.protected_names:
            if revision is None and bp.name not in self.protected_names:
                logger.warning(f"[BlueprintContrastPass] No revision returned for '{bp.name}' — kept as is.")
            return bp
        try:
            hints = BlueprintSelectorHints.model_validate(revision.selector_hints)
        except Exception as e:
            logger.warning(
                f"[BlueprintContrastPass] Invalid selector_hints for '{bp.name}' ({e}) — kept as is."
            )
            return bp

        revised = bp.model_copy(update={"description": revision.description, "selector_hints": hints})
        violations = check_blueprint_neutrality(revised)
        if violations:
            logger.warning(
                f"[BlueprintContrastPass] Revision for '{bp.name}' fails neutrality lint "
                f"({violations[0]}) — kept as is."
            )
            return bp
        return revised
