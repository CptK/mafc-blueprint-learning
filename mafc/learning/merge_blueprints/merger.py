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
from mafc.learning.merge_blueprints.consolidate import ActionConsolidator, CheckConsolidator
from mafc.learning.merge_blueprints.matching import BranchMatcher
from mafc.learning.merge_blueprints.tree import (
    EntryBranch,
    MergedStrategyTree,
    MergeEdge,
    MergeNode,
    ingest_graph,
    routing_description,
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
        force_single_branch: Fold every blueprint into one lane, skipping the
            entry matcher. For consolidating a pair the merge detector has
            ALREADY judged redundant — it decided that on entry conditions,
            graphs, checks and claim types, so re-deciding it from routing
            prose alone only adds a way to get it wrong. Leave off when merging
            a pool of strategies that genuinely need separate lanes.
    """

    def __init__(
        self,
        model: Model,
        seed_first: tuple[str, ...] = ("generic",),
        reconcile: bool = True,
        consolidate: bool = True,
        max_actions: int = 4,
        sharpen: bool = True,
        force_single_branch: bool = False,
    ) -> None:
        self.matcher = BranchMatcher(model)
        self.seed_first = seed_first
        self.force_single_branch = force_single_branch
        self.reconcile = reconcile
        self.consolidate = consolidate
        self.sharpen = sharpen
        self.consolidator = ActionConsolidator(model, max_actions=max_actions)
        self.check_consolidator = CheckConsolidator(model)

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
            tqdm(ordered, desc="Merging blueprints", unit="bp", dynamic_ncols=True) if progress else ordered
        )
        for bp in iterator:
            if progress:
                iterator.set_postfix_str(f"{bp.name} | {len(tree.entries)} branches")
            start = ingest_graph(bp.verification_graph, namespace=bp.name, finalize=tree.finalize)
            # The blueprint's checks attach to its lane entry: they activate at
            # runtime only for claims whose path enters this lane.
            start.checks = [c.model_copy(deep=True) for c in bp.required_checks]
            tree.absorb_metadata(bp)

            # Branch identity lives in the routing DESCRIPTION (maintained tree
            # state): seeded from the blueprint's description + selector hints,
            # updated on every fold, sharpened once at the end. Matching on the
            # boolean gates dissolved semantically distinct lanes.
            bp_routing = routing_description(bp)
            # Fallback branches are excluded as fold targets: a specialized lane
            # matching "nothing specific" needs its own branch, not the generic one.
            candidates = [e for e in tree.entries if not e.is_fallback]
            if self.force_single_branch:
                idx = 0 if candidates else None
            else:
                idx = self.matcher.match_entry(bp_routing, [e.description for e in candidates])
            if idx is None:
                tree.entries.append(EntryBranch(bp.name, bp.entry_conditions, start, description=bp_routing))
                logger.debug(f"[TreeMerger] '{bp.name}' -> new router branch.")
            else:
                entry = candidates[idx]
                logger.debug(f"[TreeMerger] '{bp.name}' -> merged into branch '{entry.label}'.")
                self._merge_node(entry.start, start, tree, set())
                _union_entry_conditions(entry, bp)
                entry.description = self.matcher.fold_description(entry.description, bp_routing)

        if self.reconcile:
            logger.info("[TreeMerger] Reconciling sibling branches...")
            self._reconcile(tree)

        if self.consolidate:
            logger.info("[TreeMerger] Consolidating action nodes...")
            self._consolidate(tree, progress=progress)

        if self.sharpen:
            logger.info("[TreeMerger] Sharpening router branch descriptions...")
            self._sharpen_router(tree)

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
        _union_checks(base, signal)

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
        existing_cond, new_cond = self.matcher.refine_condition(base_edge.condition, child_base, child_signal)
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
    # Consolidation pass
    # ------------------------------------------------------------------

    def _consolidate(self, tree: MergedStrategyTree, progress: bool = False) -> None:
        """Rewrite over-stuffed/verbose action nodes into concise action lists."""
        nodes: dict[int, MergeNode] = {}
        for entry in tree.entries:
            _collect_action_nodes(entry.start, nodes, set())

        targets = [n for n in nodes.values() if self.consolidator.needs_consolidation(n.actions)]
        iterator = (
            tqdm(targets, desc="Consolidating actions", unit="node", dynamic_ncols=True)
            if progress
            else targets
        )
        for node in iterator:
            before = len(node.actions)
            node.actions = self.consolidator.consolidate(node.actions)
            if progress:
                iterator.set_postfix_str(f"{node.id}: {before}->{len(node.actions)}")

        # Checks are deduplicated PER ATTACHMENT NODE (small lists from folded
        # lanes — a far easier judgment than deduping the global union), plus
        # the global list if anything remained blueprint-level. Entry nodes can
        # be synthesis-type, so walk ALL nodes here, not just action nodes.
        all_nodes: dict[int, MergeNode] = {}
        for entry in tree.entries:
            _collect_nodes(entry.start, all_nodes, set())
        check_nodes = [n for n in all_nodes.values() if len(n.checks) >= 2]
        for node in check_nodes:
            before = len(node.checks)
            node.checks = self.check_consolidator.consolidate(node.checks)
            if len(node.checks) != before:
                logger.info(
                    f"[TreeMerger] checks at '{node.id}' consolidated: {before} -> {len(node.checks)}"
                )
        if len(tree.required_checks) >= 2:
            before_checks = len(tree.required_checks)
            tree.required_checks = self.check_consolidator.consolidate(
                tree.required_checks, add_applicability_escape=True
            )
            logger.info(
                f"[TreeMerger] global checks consolidated: {before_checks} -> {len(tree.required_checks)}"
            )

    # ------------------------------------------------------------------
    # Router sharpening
    # ------------------------------------------------------------------

    def _sharpen_router(self, tree: MergedStrategyTree) -> None:
        """One final contrast pass over the router branch descriptions.

        Runs after all folds so it sharpens complete, current state; nothing
        downstream regenerates descriptions, so timing stops mattering. The
        fallback branch is excluded — its "only if nothing else fits" text is
        emitted mechanically.
        """
        targets = [e for e in tree.entries if not e.is_fallback and e.description]
        if len(targets) < 2:
            return
        sharpened = self.matcher.sharpen_router([e.description for e in targets])
        for entry, description in zip(targets, sharpened):
            entry.description = description

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


def _collect_action_nodes(node: MergeNode, out: dict[int, MergeNode], seen: set[int]) -> None:
    if node.is_finalize or id(node) in seen:
        return
    seen.add(id(node))
    if node.type == "actions":
        out[id(node)] = node
    for edge in node.edges:
        _collect_action_nodes(edge.child, out, seen)


def _collect_nodes(node: MergeNode, out: dict[int, MergeNode], seen: set[int]) -> None:
    if node.is_finalize or id(node) in seen:
        return
    seen.add(id(node))
    out[id(node)] = node
    for edge in node.edges:
        _collect_nodes(edge.child, out, seen)


def _union_actions(base: MergeNode, signal: MergeNode) -> None:
    existing = {_action_key(a) for a in base.actions}
    for action in signal.actions:
        if _action_key(action) not in existing:
            base.actions.append(action)
            existing.add(_action_key(action))


def _union_checks(base: MergeNode, signal: MergeNode) -> None:
    """Move the signal node's attached checks onto the base node (dedup by id)."""
    existing = {c.id for c in base.checks}
    for check in signal.checks:
        if check.id not in existing:
            base.checks.append(check)
            existing.add(check.id)


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
