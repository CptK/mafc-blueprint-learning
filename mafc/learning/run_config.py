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


class SynthesizerConfig(BaseModel):
    min_cluster_size: int = 3


class ConsolidatorConfig(BaseModel):
    enabled: bool = True
    prune_threshold: int = 2
    protected_names: list[str] = Field(default_factory=lambda: ["generic"])


class OutputConfig(BaseModel):
    dir: str = "out/learning"
    save_epoch_snapshots: bool = True


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
