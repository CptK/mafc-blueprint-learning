from __future__ import annotations

from dataclasses import dataclass

from mafc.blueprints.models import Blueprint
from mafc.common.logger import logger


@dataclass(frozen=True)
class BlueprintTopology:
    """Derived forward-only topology metadata computed from a blueprint graph."""

    node_layers: dict[str, int]
    max_layer: int


def longest_path_nodes(blueprint: Blueprint) -> int:
    """Return the most node visits any single run of this blueprint can make.

    One agent iteration executes exactly one node — synthesis nodes included, since
    each makes its own LLM call. So this is the minimum `max_iterations` at which the
    blueprint can still reach the end of its deepest branch; below it, that branch is
    dead and the checks attached to it never activate.

    Cycles are traversed at most once per path, making this the longest *simple* path.
    """
    nodes = {node.id: node for node in blueprint.verification_graph.nodes}
    start = blueprint.verification_graph.start_node

    def walk(node_id: str, on_path: frozenset[str]) -> int:
        # 'finalize' and dangling targets end the run without consuming an iteration.
        if node_id not in nodes or node_id in on_path:
            return 0
        targets = [transition.to for transition in nodes[node_id].transition]
        deepest = max((walk(t, on_path | {node_id}) for t in targets), default=0)
        return 1 + deepest

    return walk(start, frozenset())


def enforce_path_budget(blueprint: Blueprint) -> Blueprint:
    """Raise max_iterations so the blueprint can reach the end of its deepest branch.

    One iteration executes one node, and synthesis nodes consume one just like action
    nodes do. Authors reliably under-count this — budgets come out one short of the
    longest path — which silently strands the deepest branch: it is never reached, so
    the checks attached to it never activate and read as 'unchecked' rather than
    failing. This is a floor, not a target; early-exit transitions still end runs sooner.

    Applied on load (see ``mafc.blueprints.loader``) so hand-authored and generated
    blueprints are repaired on the same path.
    """
    required = longest_path_nodes(blueprint)
    current = blueprint.policy_constraints.max_iterations
    if current >= required:
        return blueprint

    logger.info(
        f"[enforce_path_budget] '{blueprint.name}' max_iterations {current} -> {required}: "
        f"its longest path visits {required} nodes, so the deepest branch was unreachable."
    )
    constraints = blueprint.policy_constraints.model_copy(update={"max_iterations": required})
    return blueprint.model_copy(update={"policy_constraints": constraints})


def analyze_blueprint_topology(blueprint: Blueprint) -> BlueprintTopology:
    """Analyze a blueprint graph into derived forward-layer metadata."""
    all_node_ids = {node.id for node in blueprint.verification_graph.nodes}
    transitions_by_node: dict[str, set[str]] = {}
    for node in blueprint.verification_graph.nodes:
        targets = {transition.to for transition in node.transition if transition.to in all_node_ids}
        transitions_by_node[node.id] = targets

    start_node = blueprint.verification_graph.start_node
    node_layers: dict[str, int] = {start_node: 0}
    queue: list[str] = [start_node]
    while queue:
        node_id = queue.pop(0)
        current_layer = node_layers[node_id]
        for target in transitions_by_node.get(node_id, set()):
            next_layer = current_layer + 1
            existing_layer = node_layers.get(target)
            if existing_layer is None or next_layer < existing_layer:
                node_layers[target] = next_layer
                queue.append(target)

    for node in blueprint.verification_graph.nodes:
        node_layers.setdefault(node.id, 0)

    return BlueprintTopology(
        node_layers=node_layers,
        max_layer=max(node_layers.values(), default=0),
    )
