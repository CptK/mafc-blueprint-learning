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

    @property
    def is_finalize(self) -> bool:
        return self.id == FINALIZE_ID


@dataclass
class EntryBranch:
    """One router branch: an entry-condition gate and the strategy it routes to."""

    label: str
    conditions: BlueprintEntryConditions
    start: MergeNode


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
        """Union a blueprint's required checks and policy into the tree."""
        existing_check_ids = {c.id for c in self.required_checks}
        for check in bp.required_checks:
            if check.id not in existing_check_ids:
                self.required_checks.append(check.model_copy(deep=True))
                existing_check_ids.add(check.id)

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

        router = MergeNode(ROUTER_ID, "synthesis")
        for entry in self.entries:
            router.edges.append(MergeEdge(describe_entry_conditions(entry.conditions), entry.start))

        graph_nodes: list = [_emit_node(router)]
        for node in nodes.values():
            graph_nodes.append(_emit_node(node))

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
            required_checks=[c.model_copy(deep=True) for c in self.required_checks],
            verification_graph=BlueprintVerificationGraph(start_node=ROUTER_ID, nodes=graph_nodes),
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


def _emit_node(node: MergeNode):
    transitions = [BlueprintTransition(**{"if": edge.condition, "to": edge.child.id}) for edge in node.edges]
    if node.type == "actions":
        return BlueprintActionNode(id=node.id, type="actions", actions=node.actions, transition=transitions)
    return BlueprintSynthesisNode(id=node.id, type="synthesis", transition=transitions)
