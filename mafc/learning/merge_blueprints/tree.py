"""Mutable working representation for the blueprint tree merge.

The source blueprints expose immutable, strictly-validated pydantic graphs
(`BlueprintVerificationGraph`). Merging needs to mutate nodes and edges in place
— union action lists, re-point edges, splice subtrees — so we ingest each graph
into the lightweight `MergeNode`/`MergeEdge` structures below, do all the work
there, and emit a fresh `Blueprint` at the end.

Key conventions:

* Node ids are namespaced by blueprint (``media_claim/layer1_synthesis``) on
  ingest so non-matched nodes from different blueprints never collide.
* ``finalize`` is a single shared sentinel node per tree (the global sink). Any
  edge that terminated a source path points at this one object; on emit it is
  rendered back as the string target ``"finalize"`` rather than a real node.
* The router that dispatches on entry conditions is materialised only on emit,
  as a synthesis ``router`` node whose edges carry each entry's condition text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from mafc.blueprints.models import (
    Blueprint,
    BlueprintAction,
    BlueprintActionNode,
    BlueprintCondition,
    BlueprintEntryConditions,
    BlueprintPolicyConstraints,
    BlueprintRequiredCheck,
    BlueprintSynthesisNode,
    BlueprintTransition,
    BlueprintVerificationGraph,
)

NodeType = Literal["actions", "synthesis"]

FINALIZE_ID = "finalize"
ROUTER_ID = "router"


@dataclass
class MergeEdge:
    """A conditional branch: a free-text decision and the step it leads to."""

    condition: str
    child: "MergeNode"


@dataclass
class MergeNode:
    """A mutable verification step. ``actions`` is empty for synthesis nodes."""

    id: str
    type: NodeType
    actions: list[BlueprintAction] = field(default_factory=list)
    edges: list[MergeEdge] = field(default_factory=list)
    checks: list[BlueprintRequiredCheck] = field(default_factory=list)
    """Checks this node activates when execution reaches it. Source blueprints'
    checks attach to their lane-entry node so a claim only accumulates the
    checks of the path it takes — never the union of all lanes."""

    @property
    def is_finalize(self) -> bool:
        return self.id == FINALIZE_ID


@dataclass
class EntryBranch:
    """One router branch: an entry-condition gate and the strategy it routes to.

    ``description`` is the branch's ROUTING PROSE — what claims it handles and
    when to take it. It is seeded from the source blueprint's description and
    selector hints, updated on every fold, and emitted as the router edge's
    condition text. Boolean ``conditions`` remain only the permissive top-level
    gate of the merged blueprint; rendered as router text they are tautological
    (the eom_v3 merge produced `ANY(..., has_claim_text == True)` for every
    branch) and give the LLM router nothing to discriminate on.
    """

    label: str
    conditions: BlueprintEntryConditions
    start: MergeNode
    description: str = ""

    @property
    def is_fallback(self) -> bool:
        """A branch with no gate matches everything — the generic fallback."""
        return not (self.conditions.all or self.conditions.any)


@dataclass
class MergedStrategyTree:
    """The accumulating merged tree plus the cross-cutting fields it unions."""

    finalize: MergeNode = field(default_factory=lambda: MergeNode(FINALIZE_ID, "synthesis"))
    entries: list[EntryBranch] = field(default_factory=list)
    required_checks: list[BlueprintRequiredCheck] = field(default_factory=list)
    allowed_actions: list[str] = field(default_factory=list)
    max_iterations: int = 3
    require_counterevidence_search: bool = False

    # ------------------------------------------------------------------
    # Metadata accumulation
    # ------------------------------------------------------------------

    def absorb_metadata(self, bp: Blueprint) -> None:
        """Union a blueprint's policy into the tree.

        Required checks are NOT unioned here: they attach to the blueprint's
        lane-entry node (see the merger), so checks stay scoped to the paths
        that actually enter the lane. ``self.required_checks`` remains for
        deliberately-global checks only.
        """
        policy = bp.policy_constraints
        for action in policy.allowed_actions:
            if action not in self.allowed_actions:
                self.allowed_actions.append(action)
        self.max_iterations = max(self.max_iterations, policy.max_iterations)
        self.require_counterevidence_search = (
            self.require_counterevidence_search or policy.require_counterevidence_search
        )

    # ------------------------------------------------------------------
    # Emit
    # ------------------------------------------------------------------

    def to_blueprint(self, name: str, description: str) -> Blueprint:
        """Flatten the merged tree back into a single validated Blueprint.

        The router becomes the ``start_node``; its branches carry each entry's
        condition text. Entry conditions are unioned into ``any`` so the merged
        blueprint stays eligible for every source blueprint's claims.
        """
        nodes: dict[str, MergeNode] = {}
        for entry in self.entries:
            _collect_reachable(entry.start, nodes)

        # Check DEFINITIONS all live at the blueprint root (the contract every
        # consumer expects); nodes only REFERENCE the ids they activate. Global
        # checks (tree-level) come first and stay unreferenced. Id collisions
        # between lanes with materially different descriptions are renamed.
        check_defs: list[BlueprintRequiredCheck] = [c.model_copy(deep=True) for c in self.required_checks]
        defs_by_id: dict[str, BlueprintRequiredCheck] = {c.id: c for c in check_defs}
        node_check_refs: dict[str, list[str]] = {}
        for node in nodes.values():
            refs: list[str] = []
            for check in node.checks:
                existing = defs_by_id.get(check.id)
                if existing is None:
                    definition = check.model_copy(deep=True)
                elif existing.description == check.description:
                    refs.append(existing.id)
                    continue
                else:
                    renamed = _unique_check_id(check.id, set(defs_by_id))
                    definition = BlueprintRequiredCheck(id=renamed, description=check.description)
                defs_by_id[definition.id] = definition
                check_defs.append(definition)
                refs.append(definition.id)
            if refs:
                node_check_refs[node.id] = refs

        # Router edges carry each branch's routing prose (the LLM router reads
        # exactly these texts). Fallback branches go last so "take only if
        # nothing else fits" reads in order.
        router = MergeNode(ROUTER_ID, "synthesis")
        ordered_entries = [e for e in self.entries if not e.is_fallback] + [
            e for e in self.entries if e.is_fallback
        ]
        for entry in ordered_entries:
            text = entry.description or describe_entry_conditions(entry.conditions)
            if entry.is_fallback:
                text = (
                    f"Take this branch only if none of the other branches fits: {text}"
                    if entry.description
                    else ("Take this branch only if none of the other branches fits (generic fallback).")
                )
            router.edges.append(MergeEdge(text, entry.start))

        # A router over a single lane decides nothing, yet it costs an iteration and a
        # synthesis call on every run, and lengthens the longest path so the budget
        # guard raises max_iterations to pay for it. Enter the lane directly instead.
        # This is the shape a pairwise merge produces whenever the two blueprints align
        # onto one branch — i.e. exactly when the merge did its job.
        single_lane = len(ordered_entries) == 1
        start_node_id = ordered_entries[0].start.id if single_lane else ROUTER_ID

        graph_nodes: list = [] if single_lane else [_emit_node(router, node_check_refs)]
        for node in nodes.values():
            graph_nodes.append(_emit_node(node, node_check_refs))

        any_conditions: list[BlueprintCondition] = []
        seen_conditions: set[tuple] = set()
        for entry in self.entries:
            for cond in [*entry.conditions.all, *entry.conditions.any]:
                key = (cond.feature, cond.op, str(cond.value))
                if key not in seen_conditions:
                    any_conditions.append(cond.model_copy(deep=True))
                    seen_conditions.add(key)

        return Blueprint(
            name=name,
            description=description,
            entry_conditions=BlueprintEntryConditions(any=any_conditions),
            policy_constraints=BlueprintPolicyConstraints(
                allowed_actions=list(self.allowed_actions),
                max_iterations=self.max_iterations,
                require_counterevidence_search=self.require_counterevidence_search,
            ),
            required_checks=check_defs,
            verification_graph=BlueprintVerificationGraph(start_node=start_node_id, nodes=graph_nodes),
        )


# ---------------------------------------------------------------------------
# Ingest: BlueprintVerificationGraph -> MergeNode graph
# ---------------------------------------------------------------------------


def ingest_graph(graph: BlueprintVerificationGraph, namespace: str, finalize: MergeNode) -> MergeNode:
    """Build a fresh, namespaced MergeNode graph and return its start node.

    Edges that targeted ``finalize`` are wired to the shared sentinel.
    """
    built: dict[str, MergeNode] = {}
    for node in graph.nodes:
        ns_id = f"{namespace}/{node.id}"
        actions = (
            [a.model_copy(deep=True) for a in node.actions] if isinstance(node, BlueprintActionNode) else []
        )
        built[node.id] = MergeNode(id=ns_id, type=node.type, actions=actions)

    for node in graph.nodes:
        merge_node = built[node.id]
        for transition in node.transition:
            child = finalize if transition.to == FINALIZE_ID else built[transition.to]
            merge_node.edges.append(MergeEdge(transition.if_, child))

    return built[graph.start_node]


# ---------------------------------------------------------------------------
# Description helpers (used both on emit and to serialise nodes for the LLM)
# ---------------------------------------------------------------------------


def describe_node(node: MergeNode) -> str:
    """A compact natural-language summary of a node, for LLM prompts."""
    if node.is_finalize:
        return "FINALIZE (terminal — produce the verdict)"
    if node.type == "synthesis":
        return "synthesis/decision node (routes on accumulated evidence)"
    steps = "; ".join(f"{a.action}: {a.intent}" if a.intent else a.action for a in node.actions)
    return f"action node — steps: {steps}" if steps else "action node (no steps)"


def describe_edge(edge: MergeEdge) -> str:
    return f'if "{edge.condition}" -> {describe_node(edge.child)}'


def routing_description(bp: Blueprint) -> str:
    """The routing prose a blueprint contributes to its router branch.

    Combines the description (contrast-sharpened during generation) with a
    couple of positive selector examples — the same information the standalone
    selector used for its LLM tiebreak.
    """
    parts = [bp.description.strip()]
    examples = bp.selector_hints.positive.examples if bp.selector_hints else []
    if examples:
        parts.append("Typical claims: " + " | ".join(e.strip() for e in examples[:2]))
    return " ".join(p for p in parts if p)


def describe_entry_conditions(ec: BlueprintEntryConditions) -> str:
    def render(c: BlueprintCondition) -> str:
        return f"{c.feature} {c.op} {c.value}"

    parts: list[str] = []
    if ec.all:
        parts.append("ALL(" + ", ".join(render(c) for c in ec.all) + ")")
    if ec.any:
        parts.append("ANY(" + ", ".join(render(c) for c in ec.any) + ")")
    return " and ".join(parts) if parts else "always (fallback)"


# ---------------------------------------------------------------------------
# Internal emit helpers
# ---------------------------------------------------------------------------


def _collect_reachable(start: MergeNode, out: dict[str, MergeNode]) -> None:
    if start.is_finalize or start.id in out:
        return
    out[start.id] = start
    for edge in start.edges:
        _collect_reachable(edge.child, out)


def _unique_check_id(candidate: str, taken: set[str]) -> str:
    if candidate not in taken:
        return candidate
    i = 2
    while f"{candidate}_{i}" in taken:
        i += 1
    return f"{candidate}_{i}"


def _emit_node(node: MergeNode, node_check_refs: dict[str, list[str]] | None = None):
    transitions = [BlueprintTransition(**{"if": edge.condition, "to": edge.child.id}) for edge in node.edges]
    refs = (node_check_refs or {}).get(node.id, [])
    if node.type == "actions":
        return BlueprintActionNode(
            id=node.id, type="actions", actions=node.actions, transition=transitions, activates_checks=refs
        )
    return BlueprintSynthesisNode(id=node.id, type="synthesis", transition=transitions, activates_checks=refs)
