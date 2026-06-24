"""Incremental tree merge: fold a set of blueprints into one strategy tree.

Algorithm (see also the module-level docs in `tree` and `matching`):

1. Seed with the most general blueprint, then fold the rest in one at a time.
2. For each blueprint, route it to an existing router branch (`match_entry`) or
   start a new one. When routed, co-traverse the two graphs from their start
   nodes, aligning branches level by level.
3. At each matched node pair, align outgoing branches (`match_branches`):
   * matched + destinations mergeable -> union the step, recurse;
   * matched + alternative/type-mismatch -> refine the condition into two
     distinguishing sub-conditions and keep both branches (never two identical
     edges);
   * unmatched signal branch -> graft its subtree as a new branch.
4. A final reconcile pass merges sibling branches a greedy seed order split apart.

The merge is greedy and order-dependent at borderline matches; seeding with the
most general blueprint and the reconcile pass keep it order-robust rather than
order-independent.
"""

from __future__ import annotations

from dataclasses import dataclass

from tqdm import tqdm

from mafc.blueprints.models import Blueprint, BlueprintAction
from mafc.common.logger import logger
from mafc.common.modeling.model import Model
from mafc.learning.merge_blueprints.matching import BranchMatcher
from mafc.learning.merge_blueprints.tree import (
    EntryBranch,
    MergedStrategyTree,
    MergeEdge,
    MergeNode,
    ingest_graph,
)


@dataclass
class TreeMergeResult:
    """The merged tree plus the blueprint it emits to."""

    tree: MergedStrategyTree
    blueprint: Blueprint


class BlueprintTreeMerger:
    """Folds many blueprints into one large strategy tree.

    Args:
        model: LLM backing the four matching seams.
        seed_first: Blueprint names to place first in the merge order (the
            "spine"). The first is the seed everything aligns against; default
            prefers ``generic`` as the most general.
        reconcile: Whether to run the sibling-reconciliation pass at the end.
    """

    def __init__(
        self,
        model: Model,
        seed_first: tuple[str, ...] = ("generic",),
        reconcile: bool = True,
    ) -> None:
        self.matcher = BranchMatcher(model)
        self.seed_first = seed_first
        self.reconcile = reconcile

    # ------------------------------------------------------------------

    def merge(
        self,
        blueprints: list[Blueprint],
        name: str = "merged",
        description: str = "Merged strategy tree.",
        progress: bool = False,
    ) -> TreeMergeResult:
        tree = MergedStrategyTree()

        ordered = self._order(blueprints)
        iterator = (
            tqdm(ordered, desc="Merging blueprints", unit="bp", dynamic_ncols=True)
            if progress
            else ordered
        )
        for bp in iterator:
            if progress:
                iterator.set_postfix_str(f"{bp.name} | {len(tree.entries)} branches")
            start = ingest_graph(bp.verification_graph, namespace=bp.name, finalize=tree.finalize)
            tree.absorb_metadata(bp)

            idx = self.matcher.match_entry(
                bp.entry_conditions, [e.conditions for e in tree.entries]
            )
            if idx is None:
                tree.entries.append(EntryBranch(bp.name, bp.entry_conditions, start))
                logger.debug(f"[TreeMerger] '{bp.name}' -> new router branch.")
            else:
                logger.debug(f"[TreeMerger] '{bp.name}' -> merged into branch '{tree.entries[idx].label}'.")
                self._merge_node(tree.entries[idx].start, start, tree, set())
                _union_entry_conditions(tree.entries[idx], bp)

        if self.reconcile:
            logger.info("[TreeMerger] Reconciling sibling branches...")
            self._reconcile(tree)

        return TreeMergeResult(tree=tree, blueprint=tree.to_blueprint(name, description))

    # ------------------------------------------------------------------
    # Core co-traversal
    # ------------------------------------------------------------------

    def _merge_node(
        self, base: MergeNode, signal: MergeNode, tree: MergedStrategyTree, visited: set[int]
    ) -> None:
        """Merge ``signal`` into ``base`` — precondition: they are the same step."""
        if base.is_finalize or signal.is_finalize or id(signal) in visited:
            return
        visited.add(id(signal))

        if base.type == "actions" and signal.type == "actions":
            _union_actions(base, signal)

        base_edges = list(base.edges)
        signal_edges = list(signal.edges)
        if not signal_edges:
            return

        alignment = self.matcher.match_branches(base, base_edges, signal_edges)
        matched_signal: set[int] = set()

        for pair in alignment.pairs:
            base_edge = base_edges[pair.base_index]
            signal_edge = signal_edges[pair.signal_index]
            matched_signal.add(pair.signal_index)
            base_edge.condition = _merge_condition(base_edge.condition, signal_edge.condition)
            self._resolve_children(base, base_edge, signal_edge, pair.relation, tree, visited)

        for i, signal_edge in enumerate(signal_edges):
            if i not in matched_signal:
                base.edges.append(MergeEdge(signal_edge.condition, signal_edge.child))  # graft

    def _resolve_children(
        self,
        base: MergeNode,
        base_edge: MergeEdge,
        signal_edge: MergeEdge,
        relation,
        tree: MergedStrategyTree,
        visited: set[int],
    ) -> None:
        child_base, child_signal = base_edge.child, signal_edge.child
        touches_finalize = child_base.is_finalize or child_signal.is_finalize

        if relation.is_mergeable and not touches_finalize:
            # Same / subset / complementary -> fold signal step into base step.
            self._merge_node(child_base, child_signal, tree, visited)
            return

        if child_base.is_finalize and child_signal.is_finalize:
            return  # both already terminate here

        # Alternative, type-mismatch, or one-side-finalize: the shared condition
        # cannot distinguish the two next steps. Split it rather than emit two
        # identical edges, and keep the signal branch.
        existing_cond, new_cond = self.matcher.refine_condition(
            base_edge.condition, child_base, child_signal
        )
        base_edge.condition = existing_cond
        base.edges.append(MergeEdge(new_cond, child_signal))

    # ------------------------------------------------------------------
    # Reconcile pass
    # ------------------------------------------------------------------

    def _reconcile(self, tree: MergedStrategyTree) -> None:
        """Merge near-duplicate sibling branches across the whole tree."""
        seen: set[int] = set()
        for entry in tree.entries:
            self._reconcile_node(entry.start, tree, seen)

    def _reconcile_node(self, node: MergeNode, tree: MergedStrategyTree, seen: set[int]) -> None:
        if node.is_finalize or id(node) in seen:
            return
        seen.add(id(node))

        merges = self.matcher.find_redundant_siblings(node.edges)
        # Apply highest drop_index first so earlier indices stay valid.
        for pair in sorted(merges, key=lambda m: m.drop_index, reverse=True):
            keep_edge = node.edges[pair.keep_index]
            drop_edge = node.edges[pair.drop_index]
            if not (keep_edge.child.is_finalize or drop_edge.child.is_finalize):
                self._merge_node(keep_edge.child, drop_edge.child, tree, set())
            del node.edges[pair.drop_index]

        for edge in node.edges:
            self._reconcile_node(edge.child, tree, seen)

    # ------------------------------------------------------------------
    # Ordering
    # ------------------------------------------------------------------

    def _order(self, blueprints: list[Blueprint]) -> list[Blueprint]:
        by_name = {bp.name: bp for bp in blueprints}
        ordered: list[Blueprint] = [by_name[n] for n in self.seed_first if n in by_name]
        seeded = {bp.name for bp in ordered}
        # Remaining blueprints, most general (most nodes) first.
        rest = [bp for bp in blueprints if bp.name not in seeded]
        rest.sort(key=lambda bp: len(bp.verification_graph.nodes), reverse=True)
        return ordered + rest


# ---------------------------------------------------------------------------
# Node/edge merge helpers
# ---------------------------------------------------------------------------


def _union_actions(base: MergeNode, signal: MergeNode) -> None:
    existing = {_action_key(a) for a in base.actions}
    for action in signal.actions:
        if _action_key(action) not in existing:
            base.actions.append(action)
            existing.add(_action_key(action))


def _action_key(a: BlueprintAction) -> tuple[str, str | None, str | None]:
    return (a.action, a.intent, a.query_guidance)


def _merge_condition(base_condition: str, signal_condition: str) -> str:
    """Keep the base phrasing; the conditions already matched semantically."""
    return base_condition


def _union_entry_conditions(entry: EntryBranch, bp: Blueprint) -> None:
    """Widen a router branch so it also routes the folded blueprint's claims."""
    existing = {(c.feature, c.op, str(c.value)) for c in entry.conditions.any}
    for cond in [*bp.entry_conditions.any, *bp.entry_conditions.all]:
        key = (cond.feature, cond.op, str(cond.value))
        if key not in existing:
            entry.conditions.any.append(cond.model_copy(deep=True))
            existing.add(key)
