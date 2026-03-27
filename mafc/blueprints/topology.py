from __future__ import annotations

from dataclasses import dataclass

from mafc.blueprints.models import Blueprint


@dataclass(frozen=True)
class BlueprintTopology:
    """Derived forward-only topology metadata computed from a blueprint graph."""

    node_layers: dict[str, int]
    max_layer: int


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
