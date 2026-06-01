"""Blueprint-execution infrastructure for the learning loop.

Lets the learner run a real fact-check with a *forced* blueprint and reuse the
result across epochs via a disk-backed cache.

Three pieces:

- ``ExecutionResult``: outcome of one fact-check (label, cost, iterations,
  required-check statuses, visited graph nodes, judge reasoning).
- ``BlueprintExecutionCache``: disk-backed store keyed by
  ``(agent_fingerprint, blueprint_content_hash, claim_id)``. Auto-invalidates
  when a blueprint or the agent configuration changes.
- ``BlueprintExecutor``: a thread-safe wrapper around ``FactCheckAgent`` that
  forces a chosen blueprint per call and consults the cache before running.

The executor reuses ``mafc.eval.single.run_fact_check`` so its results are
identical in shape to what the standalone benchmark runner produces.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from mafc.agents.fact_check.agent import FactCheckAgent
from mafc.blueprints.features import extract_claim_features
from mafc.blueprints.models import Blueprint, ClaimFeatures
from mafc.blueprints.registry import BlueprintRegistry
from mafc.blueprints.selector import (
    BlueprintSelectionMode,
    BlueprintSelectionResult,
)
from mafc.common.claim import Claim
from mafc.common.logger import logger
from mafc.eval.single import run_fact_check

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class ExecutionResult:
    """Outcome of executing one blueprint on one claim.

    Captures everything the learner needs to judge blueprint quality from a
    real run — verdict correctness, resource use, and per-graph-node
    attribution material for later phases.
    """

    claim_id: str
    blueprint_name: str
    ground_truth: str
    predicted_label: str | None
    correct: bool
    n_iterations: int
    cost_usd: float
    duration_ms: int
    required_check_statuses: dict[str, str] = field(default_factory=dict)
    visited_node_ids: list[str] = field(default_factory=list)
    judge_reason: str | None = None
    errors: list[str] = field(default_factory=list)
    trace_path: str | None = None
    gt_score: float | None = None
    """Continuous ground-truth score for ordinal benchmarks (e.g. VeriTaS
    integrity in [-1, +1]). Populated from ``benchmark.sample_extra_fields``
    when the benchmark exposes one. ``None`` for purely categorical benchmarks."""
    predicted_score: float | None = None
    """Numeric mapping of ``predicted_label`` for ordinal benchmarks. Set by
    ``BlueprintExecutor`` after the fact-check returns, using the
    ``label_to_numeric`` mapping supplied at construction. ``None`` when no
    mapping is configured or the predicted label isn't in the mapping."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExecutionResult:
        return cls(
            claim_id=d["claim_id"],
            blueprint_name=d["blueprint_name"],
            ground_truth=d["ground_truth"],
            predicted_label=d.get("predicted_label"),
            correct=bool(d.get("correct", False)),
            n_iterations=int(d.get("n_iterations", 0)),
            cost_usd=float(d.get("cost_usd", 0.0)),
            duration_ms=int(d.get("duration_ms", 0)),
            required_check_statuses=dict(d.get("required_check_statuses") or {}),
            visited_node_ids=list(d.get("visited_node_ids") or []),
            judge_reason=d.get("judge_reason"),
            errors=list(d.get("errors") or []),
            trace_path=d.get("trace_path"),
            gt_score=d.get("gt_score"),
            predicted_score=d.get("predicted_score"),
        )

    @classmethod
    def from_result_dict(cls, result: dict[str, Any]) -> ExecutionResult:
        """Build from the dict shape returned by ``run_fact_check``."""
        cost = result.get("cost") or {}
        # ``run_fact_check`` already merges ``benchmark.sample_extra_fields``
        # into the result dict; for VeriTaS that includes ``gt_integrity_score``.
        gt_score = result.get("gt_integrity_score")
        try:
            gt_score = float(gt_score) if gt_score is not None else None
        except (TypeError, ValueError):
            gt_score = None
        return cls(
            claim_id=str(result["claim_id"]),
            blueprint_name=result.get("blueprint_name") or "unknown",
            ground_truth=str(result.get("ground_truth", "")),
            predicted_label=result.get("predicted"),
            correct=bool(result.get("correct", False)),
            n_iterations=int(result.get("n_iterations", 0)),
            cost_usd=float(cost.get("cost_usd", 0.0)),
            duration_ms=int(result.get("duration_ms", 0)),
            required_check_statuses=dict(result.get("required_checks") or {}),
            visited_node_ids=list(result.get("node_history") or []),
            judge_reason=result.get("judge_reason"),
            errors=list(result.get("errors") or []),
            trace_path=result.get("trace_path"),
            gt_score=gt_score,
            predicted_score=None,  # Filled in by BlueprintExecutor.run if mapping configured.
        )


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


_HASH_LEN = 16
_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._-]")


def _sanitize(s: str) -> str:
    return _SANITIZE_RE.sub("_", s)[:200]


def _hash_blueprint(blueprint: Blueprint) -> str:
    """Stable short hash of a blueprint's content.

    Uses ``model_dump_json(by_alias=True)`` with sorted keys so semantically
    identical blueprints produce identical hashes regardless of field order.
    """
    payload = json.dumps(
        blueprint.model_dump(by_alias=True),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:_HASH_LEN]


class BlueprintExecutionCache:
    """Disk-backed cache of ``ExecutionResult`` objects.

    Layout: ``<root>/<agent_fp>/<blueprint_hash>/<claim_id>.json``.

    Changing the blueprint content or the agent configuration changes the path,
    so stale results are simply unreachable rather than incorrect. Concurrent
    writes are safe (atomic ``os.replace`` after writing to a sibling tmp file).
    """

    def __init__(self, root: Path, agent_fingerprint: str) -> None:
        self.root = Path(root)
        self.fingerprint = agent_fingerprint
        self.root.mkdir(parents=True, exist_ok=True)
        self._fp_dir = self.root / self.fingerprint
        self._fp_dir.mkdir(parents=True, exist_ok=True)
        self._stats_lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def _path(self, blueprint: Blueprint, claim_id: str) -> Path:
        bp_hash = _hash_blueprint(blueprint)
        bp_dir = self._fp_dir / bp_hash
        return bp_dir / f"{_sanitize(claim_id)}.json"

    def get(self, blueprint: Blueprint, claim_id: str) -> ExecutionResult | None:
        path = self._path(blueprint, claim_id)
        if not path.exists():
            with self._stats_lock:
                self._misses += 1
            return None
        try:
            with open(path) as f:
                raw = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"[BlueprintExecutionCache] Skipping corrupt cache entry {path}: {e}")
            with self._stats_lock:
                self._misses += 1
            return None
        with self._stats_lock:
            self._hits += 1
        return ExecutionResult.from_dict(raw)

    def put(self, blueprint: Blueprint, claim_id: str, result: ExecutionResult) -> None:
        path = self._path(blueprint, claim_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.{threading.get_ident()}")
        try:
            with open(tmp, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass

    def stats(self) -> dict[str, int]:
        with self._stats_lock:
            return {"hits": self._hits, "misses": self._misses}


# ---------------------------------------------------------------------------
# Forced-blueprint selector
# ---------------------------------------------------------------------------


class _MutableSingleBlueprintSelector:
    """Drop-in replacement for ``BlueprintSelector`` that always returns one blueprint.

    ``BlueprintExecutor`` calls ``set(blueprint)`` from the executing thread
    immediately before invoking the agent, so concurrent threads each own their
    own selector instance and never race.
    """

    def __init__(self, registry: BlueprintRegistry, default_blueprint_name: str) -> None:
        self.registry = registry
        self.default_blueprint_name = default_blueprint_name
        self._forced: Blueprint | None = None

    def set(self, blueprint: Blueprint) -> None:
        self._forced = blueprint

    def select(
        self,
        claim: Claim,
        article_analysis: Any | None = None,
    ) -> BlueprintSelectionResult:
        if self._forced is None:
            raise RuntimeError(
                "_MutableSingleBlueprintSelector.select called before set() — "
                "BlueprintExecutor must set the forced blueprint each run."
            )
        bp = self._forced
        features: ClaimFeatures = extract_claim_features(claim)
        return BlueprintSelectionResult(
            selected_blueprint=bp,
            selection_mode=BlueprintSelectionMode.RULE_BASED,
            claim_features=features,
            surviving_blueprints=[bp.name],
            rejected_blueprints=[],
            reason="Forced by BlueprintExecutor.",
            all_blueprints=[b.name for b in self.registry.get_all()],
        )


# ---------------------------------------------------------------------------
# Internal sample shim
# ---------------------------------------------------------------------------


@dataclass
class _LabelShim:
    value: str


@dataclass
class _ExecutorSample:
    """Minimal duck-typed sample for ``run_fact_check``."""

    id: str
    input: Claim
    label: _LabelShim


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


AgentFactory = Callable[[_MutableSingleBlueprintSelector], FactCheckAgent]


class BlueprintExecutor:
    """Runs a fact-check on demand with a forced blueprint, with cache lookup first.

    Per-thread agent instances mirror the pattern in ``mafc/eval/runner.py`` so
    each worker has its own web-search agent / event loop. The factory is
    called lazily on first use within each thread.
    """

    def __init__(
        self,
        agent_factory: AgentFactory,
        registry: BlueprintRegistry,
        cache: BlueprintExecutionCache,
        default_blueprint_name: str = "generic",
        label_to_numeric: dict[str, float] | None = None,
    ) -> None:
        self._agent_factory = agent_factory
        self._registry = registry
        self._cache = cache
        self._default = default_blueprint_name
        self._label_to_numeric = label_to_numeric
        """Optional mapping from predicted-label strings to scalar values for
        ordinal benchmarks. When provided, ``run()`` annotates the returned
        ``ExecutionResult.predicted_score`` so downstream MSE-based gating and
        error-magnitude outcome bucketing have a numeric prediction. ``None``
        for purely categorical benchmarks."""
        self._thread_local = threading.local()
        self._total_runs = 0
        self._cache_hits = 0
        self._executed = 0
        self._stats_lock = threading.Lock()

    def _get_thread_agent(self) -> tuple[_MutableSingleBlueprintSelector, FactCheckAgent]:
        if not hasattr(self._thread_local, "agent"):
            selector = _MutableSingleBlueprintSelector(self._registry, self._default)
            agent = self._agent_factory(selector)
            self._thread_local.selector = selector
            self._thread_local.agent = agent
        return self._thread_local.selector, self._thread_local.agent

    def run(
        self,
        claim: Claim,
        blueprint: Blueprint,
        *,
        true_label: str,
        claim_id: str | None = None,
        gt_score: float | None = None,
    ) -> ExecutionResult:
        cid = claim_id or claim.id
        if cid is None:
            raise ValueError(
                "BlueprintExecutor.run requires a stable claim id — either set claim.id "
                "or pass claim_id explicitly."
            )

        cached = self._cache.get(blueprint, cid)
        with self._stats_lock:
            self._total_runs += 1
        if cached is not None:
            # Old cache entries may pre-date the score plumbing; lazily fill in
            # predicted_score (from the configured mapping) and gt_score (from
            # the caller) so the rest of the pipeline sees consistent data.
            if (
                cached.predicted_score is None
                and self._label_to_numeric
                and cached.predicted_label is not None
            ):
                cached.predicted_score = self._label_to_numeric.get(cached.predicted_label)
            if cached.gt_score is None and gt_score is not None:
                cached.gt_score = gt_score
            with self._stats_lock:
                self._cache_hits += 1
            return cached

        selector, agent = self._get_thread_agent()
        selector.set(blueprint)
        sample = _ExecutorSample(id=cid, input=claim, label=_LabelShim(true_label))
        result_dict = run_fact_check(sample, agent)
        if gt_score is not None:
            # Surface the continuous ground-truth score via the same field
            # ``VeriTaS.sample_extra_fields`` would have populated when a
            # benchmark instance is passed to ``run_fact_check`` directly.
            result_dict["gt_integrity_score"] = gt_score
        exec_result = ExecutionResult.from_result_dict(result_dict)
        if self._label_to_numeric and exec_result.predicted_label is not None:
            exec_result.predicted_score = self._label_to_numeric.get(exec_result.predicted_label)
        # Cache any run that produced a verdict. Non-fatal errors (e.g. a
        # ScrapeMM failure on one source while the agent still finalises from
        # others) are kept on the result but don't disqualify the cache entry —
        # the runner counts these as "completed" too. Only runs that never
        # produced a label are left uncached so transient infrastructure
        # failures can be retried.
        if exec_result.predicted_label is not None:
            self._cache.put(blueprint, cid, exec_result)
        with self._stats_lock:
            self._executed += 1
        return exec_result

    def stats(self) -> dict[str, int]:
        with self._stats_lock:
            return {
                "total_runs": self._total_runs,
                "cache_hits": self._cache_hits,
                "executed": self._executed,
            }
