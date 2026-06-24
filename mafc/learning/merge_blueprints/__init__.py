"""Merge a set of blueprints into one large strategy tree.

Public entry point: `BlueprintTreeMerger`. It ingests each blueprint's
verification graph, folds them together by recursively aligning branches level
by level, and emits a single merged `Blueprint`.
"""

from mafc.learning.merge_blueprints.matching import BranchMatcher, Relation
from mafc.learning.merge_blueprints.merger import BlueprintTreeMerger, TreeMergeResult
from mafc.learning.merge_blueprints.tree import (
    EntryBranch,
    MergedStrategyTree,
    MergeEdge,
    MergeNode,
)

__all__ = [
    "BlueprintTreeMerger",
    "TreeMergeResult",
    "BranchMatcher",
    "Relation",
    "MergedStrategyTree",
    "MergeNode",
    "MergeEdge",
    "EntryBranch",
]
