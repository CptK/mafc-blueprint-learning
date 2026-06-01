"""Pydantic schema for the blueprint learning run configuration (loaded from YAML)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from mafc.eval.run_config import AgentsConfig, BlueprintsConfig as EvalBlueprintsConfig, RunConfig


class DataConfig(BaseModel):
    data_path: str
    split: str = "2025_q4"
    label_scheme: int = 3
    train_fraction: float = 0.8
    seed: int = 42
    first_n: int | None = None
    dev_fraction: float = 0.0
    """Fraction of the (shuffled) training samples held out for per-epoch evaluation via
    ``BlueprintExecutor``. 0.0 disables dev eval entirely. Requires ``execution.enabled``."""


class ModelConfig(BaseModel):
    name: str
    temperature: float = 0.7
    max_response_length: int = 8192


class BlueprintsConfig(BaseModel):
    config_dir: str = "config/blueprints"
    default_blueprint: str = "generic"
    selector_max_response_length: int = 8192


class LearningConfig(BaseModel):
    max_epochs: int = 5
    minibatch_size: int = 20
    update_threshold: int = 3
    consolidate_every: int = 1
    use_article_analysis_for_selection: bool = False
    workers: int = 4
    freeze_all_blueprints: bool = False
    """If true, every blueprint in the registry is treated as frozen so the
    updater never mutates them. Used by Phase-0 smoke runs to verify the
    execution cache: a second invocation must produce identical
    (claim, assigned_blueprint) pairs and therefore 100% cache hits.
    """
    rollback_on_regression: bool = False
    """When true, the script takes a registry snapshot at the
    start of each epoch and restores it after the epoch if dev_macro_f1
    regresses below the running best by more than ``rollback_margin``.
    Requires ``data.dev_fraction > 0`` and ``execution.enabled = true``.
    Default ``false`` preserves the Phase-0/1 behaviour for ablations.
    """
    rollback_margin: float = 0.0
    """Tolerance band around the running-best gate metric. Direction depends
    on ``gate_metric``: for ``macro_f1`` a rollback fires when
    ``dev_macro_f1 < best_so_far - margin``; for ``mse`` it fires when
    ``dev_mse > best_so_far + margin``. Set to a small positive value to
    absorb stochastic noise on small dev sets.
    """
    gate_metric: str = "macro_f1"
    """Which dev metric the rollback wrapper consults. ``"macro_f1"`` keeps
    the Phase-2 behaviour for backwards compatibility. ``"mse"`` switches to
    mean squared error against the continuous ground-truth integrity score
    (lower is better) — the right choice when the eval metric is MSE on
    ordinal labels (e.g. VeriTaS 7-class). MSE mode requires the executor's
    ``label_to_numeric`` mapping to be populated and each sample to carry a
    continuous ground-truth score.
    """
    outcome_error_threshold: float | None = None
    """Phase-4 outcome bucketing mode. ``None`` uses strict label equality
    (the original behaviour: ``correct`` iff predicted_label == ground_truth).
    A float value switches to score-error bucketing: a record is ``correct``
    iff ``abs(predicted_score - gt_score) <= threshold``. For VeriTaS 7-class
    on [-1, +1], 1/3 ≈ 0.333 is a natural threshold (off-by-one-bin counts
    as a near miss; anything bigger is a miss).
    """
    use_execution_outcomes: bool = False
    """Phase-4 gate. When true, BlueprintUpdater and NewBlueprintSynthesizer
    read ``ClaimLearningRecord.execution_result`` and partition records by
    outcome (correct / incorrect / unknown) when building their prompts.
    Requires ``execution.enabled = true``. Default ``false`` preserves the
    Phase 0-2 behaviour and produces byte-identical updater/synthesizer
    prompts for ablations.
    """


class SynthesizerConfig(BaseModel):
    min_cluster_size: int = 3


class ConsolidatorConfig(BaseModel):
    enabled: bool = True
    prune_threshold: int = 2
    protected_names: list[str] = Field(default_factory=lambda: ["generic"])


class OutputConfig(BaseModel):
    dir: str = "out/learning"
    save_epoch_snapshots: bool = True
    log_level: str = "INFO"
    """Console log level for the slurm stdout file. Default INFO so per-iteration
    agent prompts and responses (emitted via ``logger.debug``) stay out of the
    multi-megabyte run logs. Set to ``DEBUG`` to surface them again."""


class ExecutionConfig(BaseModel):
    """Optional Phase-0+ execution feedback configuration.

    When ``enabled`` is true the learning loop runs a real fact-check (with the
    selected blueprint forced) on each training record and stores the outcome
    on the ``ClaimLearningRecord``. Phase 0 uses this only as a side effect
    (recorded into the scorecard); later phases will gate updater decisions on
    these results.

    The ``agents``/``blueprints``/``run`` sections mirror ``BenchmarkRunConfig``
    so existing eval configs can be lifted in with minimal duplication.
    """

    enabled: bool = False
    cache_dir: str | None = None
    """Disk location for the execution cache. Defaults to ``<output.dir>/execution_cache``."""
    write_traces: bool = False
    """Write per-claim fact-check traces under ``<run_dir>/execution_traces/``."""
    agents: AgentsConfig | None = None
    blueprints: EvalBlueprintsConfig | None = None
    run: RunConfig = RunConfig()


class LearningRunConfig(BaseModel):
    data: DataConfig
    model: ModelConfig
    blueprints: BlueprintsConfig = BlueprintsConfig()
    learning: LearningConfig = LearningConfig()
    synthesizer: SynthesizerConfig = SynthesizerConfig()
    consolidator: ConsolidatorConfig = ConsolidatorConfig()
    execution: ExecutionConfig = ExecutionConfig()
    output: OutputConfig = OutputConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> LearningRunConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)
