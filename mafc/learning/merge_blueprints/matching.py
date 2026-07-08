"""LLM seams for the tree merge.

All semantic judgments are batched per node — branching is small (synthesis
nodes have ~2 branches), so there is no candidate explosion and no need for an
embedding prefilter. One `match_branches` call aligns a whole node's outgoing
branches at once, which lets the model do the assignment globally instead of us
scoring pairs and running a separate best-match step.

Four seams:

* `match_branches`   — align base vs. signal branch sets (the core step).
* `refine_condition` — split an under-specified decision into two distinguishing
                       sub-conditions (the "alternative" resolution).
* `match_entry`      — route a blueprint to an existing router branch, or signal
                       a new one.
* `find_redundant_siblings` — used by the reconcile pass to merge sibling
                       branches a greedy seed order split apart.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict

from mafc.common.logger import logger
from mafc.common.modeling.message import Message, MessageRole
from mafc.common.modeling.model import Model
from mafc.common.modeling.prompt import Prompt
from mafc.learning.merge_blueprints.tree import (
    BlueprintEntryConditions,
    MergeEdge,
    MergeNode,
    describe_edge,
    describe_entry_conditions,
    describe_node,
)
from mafc.utils.parsing import (
    extract_json_object,
    strip_json_fences,
    try_parse_with_repair,
)


class Relation(str, Enum):
    """How a paired base branch and signal branch relate at their destination."""

    SAME = "SAME"
    SUBSET = "SUBSET"  # one step's work is contained in the other -> follow the superset
    OVERLAP_COMPLEMENTARY = "OVERLAP_COMPLEMENTARY"  # both should happen -> union the steps
    OVERLAP_ALTERNATIVE = "OVERLAP_ALTERNATIVE"  # do one or the other -> split the condition
    TYPE_MISMATCH = "TYPE_MISMATCH"  # action vs synthesis -> alignment, not a fork

    @property
    def is_mergeable(self) -> bool:
        """Whether the two destination nodes fold into one (vs. stay distinct)."""
        return self in {Relation.SAME, Relation.SUBSET, Relation.OVERLAP_COMPLEMENTARY}


# ---------------------------------------------------------------------------
# LLM response schemas
# ---------------------------------------------------------------------------


class _Pair(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_index: int
    signal_index: int
    relation: Relation
    rationale: str = ""


class _BranchAlignment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pairs: list[_Pair] = []
    unmatched_signal: list[int] = []


class _RefinedCondition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    existing_condition: str
    new_condition: str


class _EntryMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    match_index: int | None = None
    rationale: str = ""


class _SiblingPair(BaseModel):
    model_config = ConfigDict(extra="forbid")

    keep_index: int
    drop_index: int
    relation: Relation


class _SiblingMerges(BaseModel):
    model_config = ConfigDict(extra="forbid")

    merges: list[_SiblingPair] = []


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_MATCH_SYSTEM = """\
You align the outgoing branches of two fact-checking workflow nodes that are \
already known to be the same step. The BASE node belongs to the merged tree; the \
SIGNAL node comes from a blueprint being folded in. For each SIGNAL branch decide \
whether it corresponds to a BASE branch, and if so how their destination steps \
relate. A branch is a decision condition plus the step it leads to.

Relations (judge the DESTINATION steps, given the conditions already align):
- SAME: the same step, possibly phrased differently.
- SUBSET: one step's work is fully contained in the other.
- OVERLAP_COMPLEMENTARY: different work under the same trigger that SHOULD BOTH \
happen (e.g. reverse-image-search vs. publisher-investigation). These will be \
unioned into one richer step.
- OVERLAP_ALTERNATIVE: mutually exclusive — you would do one OR the other, never \
both. This means the shared condition is too coarse to tell them apart.
- TYPE_MISMATCH: one destination is an action step and the other is a \
synthesis/decision step (one workflow has an extra intermediate step).

Pair a SIGNAL branch with a BASE branch only when their conditions represent the \
same decision outcome. Each base branch and each signal branch may appear at most \
once. List SIGNAL branches with no corresponding BASE branch in unmatched_signal.\
"""

_MATCH_USER = """\
BASE node: {base_desc}
BASE branches:
{base_branches}

SIGNAL node: {signal_desc}
SIGNAL branches:
{signal_branches}

Return only a JSON object:
{{
  "pairs": [
    {{"base_index": int, "signal_index": int,
      "relation": "SAME|SUBSET|OVERLAP_COMPLEMENTARY|OVERLAP_ALTERNATIVE|TYPE_MISMATCH",
      "rationale": string}}
  ],
  "unmatched_signal": [int, ...]
}}
"""

_REFINE_SYSTEM = """\
A single decision condition routes to two mutually exclusive next steps, so the \
condition is too coarse to choose between them. Rewrite it as two precise, \
distinguishing sub-conditions — one selecting the existing step, one selecting the \
new step — so a reader can unambiguously tell which branch applies. Keep them \
mutually exclusive and faithful to the original intent.\
"""

_REFINE_USER = """\
Original (too-coarse) condition: "{condition}"

Existing step: {existing_node}
New step: {new_node}

Return only a JSON object:
{{"existing_condition": string, "new_condition": string}}
"""

_ENTRY_SYSTEM = """\
You route a fact-checking blueprint to an existing strategy branch. Given the new \
blueprint's entry conditions and a list of existing branches' entry conditions, \
return the index of the branch whose entry conditions overlap substantially (they \
would route similar claims), or null if none is a good fit and the blueprint needs \
its own branch. When in doubt, prefer null over a forced match.\
"""

_ENTRY_USER = """\
New blueprint entry conditions: {new_conditions}

Existing branches:
{existing}

Return only a JSON object: {{"match_index": int or null, "rationale": string}}
"""

_SIBLING_SYSTEM = """\
The following branches all leave the SAME node. A greedy merge may have left near- \
duplicate branches that should be a single branch. Identify pairs that should be \
merged because their decision and destination step are the same, a subset, or \
complementary work under the same trigger. Do NOT merge branches that are genuine \
alternatives. For each merge, keep_index is retained and drop_index is folded into \
it. Each index may appear in at most one merge.\
"""

_SIBLING_USER = """\
Branches leaving the node:
{branches}

Return only a JSON object:
{{"merges": [{{"keep_index": int, "drop_index": int,
  "relation": "SAME|SUBSET|OVERLAP_COMPLEMENTARY"}}]}}
"""

_REPAIR = "The previous response was not valid JSON matching the schema. Return only the JSON object."


def _enumerate(items: list[str]) -> str:
    return "\n".join(f"[{i}] {text}" for i, text in enumerate(items)) or "(none)"


class BranchMatcher:
    """Wraps a `Model` with the four semantic seams used by the merger."""

    def __init__(self, model: Model) -> None:
        self.model = model

    # ------------------------------------------------------------------

    def match_branches(
        self, base_node: MergeNode, base_edges: list[MergeEdge], signal_edges: list[MergeEdge]
    ) -> _BranchAlignment:
        prompt = _MATCH_USER.format(
            base_desc=describe_node(base_node),
            base_branches=_enumerate([describe_edge(e) for e in base_edges]),
            signal_desc="(same step as base)",
            signal_branches=_enumerate([describe_edge(e) for e in signal_edges]),
        )
        result = self._call(_MATCH_SYSTEM, prompt, _parse_alignment)
        if result is None:
            logger.warning("[TreeMerger] branch alignment failed; treating all signal branches as new.")
            return _BranchAlignment(unmatched_signal=list(range(len(signal_edges))))
        return _validate_alignment(result, len(base_edges), len(signal_edges))

    def refine_condition(self, condition: str, existing: MergeNode, new: MergeNode) -> tuple[str, str]:
        prompt = _REFINE_USER.format(
            condition=condition,
            existing_node=describe_node(existing),
            new_node=describe_node(new),
        )
        result = self._call(_REFINE_SYSTEM, prompt, _parse_refined)
        if result is None:
            # Fall back to keeping the coarse condition on the existing branch and
            # a negated marker on the new one — still avoids two identical edges.
            return condition, f"otherwise: {condition}"
        return result.existing_condition, result.new_condition

    def match_entry(
        self, new_conditions: BlueprintEntryConditions, existing: list[BlueprintEntryConditions]
    ) -> int | None:
        if not existing:
            return None
        prompt = _ENTRY_USER.format(
            new_conditions=describe_entry_conditions(new_conditions),
            existing=_enumerate([describe_entry_conditions(c) for c in existing]),
        )
        result = self._call(_ENTRY_SYSTEM, prompt, _parse_entry)
        if result is None or result.match_index is None:
            return None
        if 0 <= result.match_index < len(existing):
            return result.match_index
        return None

    def find_redundant_siblings(self, edges: list[MergeEdge]) -> list[_SiblingPair]:
        if len(edges) < 2:
            return []
        prompt = _SIBLING_USER.format(branches=_enumerate([describe_edge(e) for e in edges]))
        result = self._call(_SIBLING_SYSTEM, prompt, _parse_siblings)
        if result is None:
            return []
        return _validate_siblings(result, len(edges))

    # ------------------------------------------------------------------

    def _call(self, system: str, user: str, parse_fn):
        messages = [
            Message(role=MessageRole.SYSTEM, content=Prompt(text=system)),
            Message(role=MessageRole.USER, content=Prompt(text=user)),
        ]
        response = self.model.generate(messages)
        parsed, _ = try_parse_with_repair(
            response_text=response.text.strip(),
            parse_fn=parse_fn,
            model=self.model,
            repair_prompt_prefix=_REPAIR,
        )
        return parsed


# ---------------------------------------------------------------------------
# Parsing + validation
# ---------------------------------------------------------------------------


def _parse(model_cls, text: str):
    try:
        raw = extract_json_object(strip_json_fences(text))
        return model_cls.model_validate_json(raw)
    except Exception as e:  # noqa: BLE001 - parser is best-effort with repair fallback
        logger.debug(f"[TreeMerger] failed to parse {model_cls.__name__}: {e}")
        return None


def _parse_alignment(text: str) -> _BranchAlignment | None:
    return _parse(_BranchAlignment, text)


def _parse_refined(text: str) -> _RefinedCondition | None:
    return _parse(_RefinedCondition, text)


def _parse_entry(text: str) -> _EntryMatch | None:
    return _parse(_EntryMatch, text)


def _parse_siblings(text: str) -> _SiblingMerges | None:
    return _parse(_SiblingMerges, text)


def _validate_alignment(a: _BranchAlignment, n_base: int, n_signal: int) -> _BranchAlignment:
    """Drop out-of-range or non-one-to-one pairs so the merger can trust the result."""
    used_base: set[int] = set()
    used_signal: set[int] = set()
    pairs: list[_Pair] = []
    for p in a.pairs:
        if not (0 <= p.base_index < n_base and 0 <= p.signal_index < n_signal):
            continue
        if p.base_index in used_base or p.signal_index in used_signal:
            continue
        used_base.add(p.base_index)
        used_signal.add(p.signal_index)
        pairs.append(p)
    unmatched = [i for i in range(n_signal) if i not in used_signal]
    return _BranchAlignment(pairs=pairs, unmatched_signal=unmatched)


def _validate_siblings(s: _SiblingMerges, n: int) -> list[_SiblingPair]:
    used: set[int] = set()
    out: list[_SiblingPair] = []
    for m in s.merges:
        if not (0 <= m.keep_index < n and 0 <= m.drop_index < n):
            continue
        if m.keep_index == m.drop_index or m.keep_index in used or m.drop_index in used:
            continue
        if not m.relation.is_mergeable:
            continue
        used.add(m.keep_index)
        used.add(m.drop_index)
        out.append(m)
    return out
