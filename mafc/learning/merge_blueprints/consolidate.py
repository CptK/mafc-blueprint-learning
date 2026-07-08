"""Action consolidation pass.

Merging accumulates actions: a hot spine node can end up with 30+ narrow,
overlapping sub-steps, none of them exact duplicates (so the union dedup never
removed them), but together far more verbose than any real fact-checker would
perform at one step. This pass is lossy by design — per action node it asks an
LLM to rewrite the action list into a small set of general, non-overlapping
actions with concise text, preserving investigative coverage.

It runs as a post-pass on the finished merged tree (like reconcile, but over
node payloads rather than branches).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from mafc.blueprints.models import BlueprintAction, BlueprintRequiredCheck
from mafc.common.logger import logger
from mafc.common.modeling.message import Message, MessageRole
from mafc.common.modeling.model import Model
from mafc.common.modeling.prompt import Prompt
from mafc.utils.parsing import (
    extract_json_object,
    strip_json_fences,
    try_parse_with_repair,
)


class _ConsolidatedAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: str
    intent: str
    query_guidance: str = ""


class _ConsolidationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actions: list[_ConsolidatedAction] = []


_SYSTEM = """\
You compress the action list of ONE step in a fact-checking workflow. After \
merging many strategies, this step accumulated many narrow, overlapping actions. \
Rewrite them into a SMALL set (at most {max_actions}) of general, non-overlapping \
actions that a real fact-checker would actually perform at a single step. Merge \
related sub-steps into one general action; preserve the overall investigative \
coverage; drop nothing essential.

Rules:
- Each action's "action" field must be one of the allowed executor types.
- Keep multiple actions only when they pursue genuinely different investigative \
goals (e.g. an open web search vs. a reverse-media search). Ensure every \
essential executor type stays represented.
- Write a concise one-sentence "intent" and a short "query_guidance". No long \
enumerations or run-on sentences.\
"""

_USER = """\
Allowed action types: {types}

Current actions ({n}):
{actions}

Return only a JSON object with at most {max_actions} actions:
{{"actions": [{{"action": <one of {types}>, "intent": string, "query_guidance": string}}]}}
"""

_REPAIR = "The previous response was not valid JSON matching the schema. Return only the JSON object."


def _describe(action: BlueprintAction) -> str:
    parts = [action.action]
    if action.intent:
        parts.append(f"intent: {action.intent}")
    if action.query_guidance:
        parts.append(f"guidance: {action.query_guidance}")
    return " | ".join(parts)


class ActionConsolidator:
    """Rewrites an over-stuffed action list into a concise, general one.

    Args:
        model: LLM used for the rewrite.
        max_actions: Upper bound on actions per node after consolidation.
        max_chars: An action node is consolidated when it has more than
            ``max_actions`` actions OR any action's text exceeds this length
            (so verbose-but-short lists still get tightened).
    """

    def __init__(self, model: Model, max_actions: int = 4, max_chars: int = 220) -> None:
        self.model = model
        self.max_actions = max_actions
        self.max_chars = max_chars

    def needs_consolidation(self, actions: list[BlueprintAction]) -> bool:
        if len(actions) > self.max_actions:
            return True
        return any(len(a.intent or "") + len(a.query_guidance or "") > self.max_chars for a in actions)

    def consolidate(self, actions: list[BlueprintAction]) -> list[BlueprintAction]:
        """Return a consolidated action list, or the original on failure."""
        if not actions:
            return actions

        allowed = sorted({a.action for a in actions})
        prompt = _USER.format(
            types=", ".join(allowed),
            n=len(actions),
            actions="\n".join(f"- {_describe(a)}" for a in actions),
            max_actions=self.max_actions,
        )
        messages = [
            Message(
                role=MessageRole.SYSTEM,
                content=Prompt(text=_SYSTEM.format(max_actions=self.max_actions)),
            ),
            Message(role=MessageRole.USER, content=Prompt(text=prompt)),
        ]
        response = self.model.generate(messages)
        parsed, _ = try_parse_with_repair(
            response_text=response.text.strip(),
            parse_fn=_parse,
            model=self.model,
            repair_prompt_prefix=_REPAIR,
        )
        if parsed is None or not parsed.actions:
            logger.warning("[TreeMerger] action consolidation failed; keeping original list.")
            return actions

        # Keep only allowed executor types, then clamp to the cap.
        allowed_set = set(allowed)
        result = [
            BlueprintAction(
                action=a.action,
                intent=a.intent or None,
                query_guidance=a.query_guidance or None,
            )
            for a in parsed.actions
            if a.action in allowed_set
        ]
        if not result:
            logger.warning("[TreeMerger] consolidation dropped all actions; keeping original list.")
            return actions
        return result[: self.max_actions]


def _parse(text: str) -> _ConsolidationResponse | None:
    try:
        return _ConsolidationResponse.model_validate_json(extract_json_object(strip_json_fences(text)))
    except Exception as e:  # noqa: BLE001 - best-effort with repair fallback
        logger.debug(f"[TreeMerger] failed to parse consolidation response: {e}")
        return None


# ---------------------------------------------------------------------------
# Required-check consolidation (conservative de-duplication)
# ---------------------------------------------------------------------------


class _CheckGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    merge_ids: list[str]
    canonical_id: str
    canonical_description: str


class _CheckConsolidationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    groups: list[_CheckGroup] = []


_CHECK_SYSTEM = """\
You de-duplicate the required checks of a fact-checking blueprint. Each check is a \
verification gate: an id and a description. After merging many blueprints, MANY \
checks are paraphrases of the SAME gate worded differently.

Group together ONLY checks that assert the SAME verification requirement — genuine \
paraphrases. Be conservative: required checks are correctness gates, so if two \
checks could plausibly be distinct requirements, do NOT group them. When in doubt, \
leave them separate.

For each group of two or more equivalent checks, pick the clearest id as the \
canonical_id and write one canonical_description that fully covers the group's \
intent. Do not list checks that have no duplicate.\
"""

_CHECK_USER = """\
Required checks ({n}):
{checks}

Return only a JSON object listing groups of 2+ genuinely equivalent checks:
{{"groups": [{{"merge_ids": [string, ...], "canonical_id": string, "canonical_description": string}}]}}
Return an empty groups list if nothing is safely mergeable.
"""


class CheckConsolidator:
    """Conservatively merges paraphrase-duplicate required checks into canonical ones.

    Only genuine paraphrases are collapsed; plausibly-distinct gates are kept.
    """

    def __init__(self, model: Model) -> None:
        self.model = model

    def consolidate(self, checks: list[BlueprintRequiredCheck]) -> list[BlueprintRequiredCheck]:
        if len(checks) < 2:
            return checks

        prompt = _CHECK_USER.format(
            n=len(checks),
            checks="\n".join(f"- {c.id}: {c.description}" for c in checks),
        )
        messages = [
            Message(role=MessageRole.SYSTEM, content=Prompt(text=_CHECK_SYSTEM)),
            Message(role=MessageRole.USER, content=Prompt(text=prompt)),
        ]
        response = self.model.generate(messages)
        parsed, _ = try_parse_with_repair(
            response_text=response.text.strip(),
            parse_fn=_parse_checks,
            model=self.model,
            repair_prompt_prefix=_REPAIR,
        )
        if parsed is None:
            logger.warning("[TreeMerger] check consolidation failed; keeping original checks.")
            return checks

        return _apply_check_groups(checks, parsed.groups)


def _parse_checks(text: str) -> _CheckConsolidationResponse | None:
    try:
        return _CheckConsolidationResponse.model_validate_json(extract_json_object(strip_json_fences(text)))
    except Exception as e:  # noqa: BLE001 - best-effort with repair fallback
        logger.debug(f"[TreeMerger] failed to parse check consolidation response: {e}")
        return None


def _apply_check_groups(
    checks: list[BlueprintRequiredCheck], groups: list[_CheckGroup]
) -> list[BlueprintRequiredCheck]:
    """Replace each valid group of duplicate checks with a single canonical check.

    Validates that group ids exist and no id is claimed by two groups; the merged
    check keeps the position of the group's first member.
    """
    valid_ids = {c.id for c in checks}
    id_to_group: dict[str, _CheckGroup] = {}
    for g in groups:
        members = [cid for cid in g.merge_ids if cid in valid_ids]
        if len(members) < 2 or any(cid in id_to_group for cid in members):
            continue  # skip invalid / overlapping groups
        g.merge_ids = members
        for cid in members:
            id_to_group[cid] = g

    result: list[BlueprintRequiredCheck] = []
    used_ids: set[str] = set()
    emitted_groups: set[int] = set()
    for check in checks:
        group = id_to_group.get(check.id)
        if group is None:
            result.append(check)
            used_ids.add(check.id)
            continue
        if id(group) in emitted_groups:
            continue  # a later member of an already-emitted group
        emitted_groups.add(id(group))
        canonical_id = _unique_id(group.canonical_id or check.id, used_ids)
        used_ids.add(canonical_id)
        result.append(
            BlueprintRequiredCheck(
                id=canonical_id, description=group.canonical_description or check.description
            )
        )
    return result


def _unique_id(candidate: str, taken: set[str]) -> str:
    if candidate not in taken:
        return candidate
    i = 2
    while f"{candidate}_{i}" in taken:
        i += 1
    return f"{candidate}_{i}"
