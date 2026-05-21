"""Pydantic schema for the blueprint learning run configuration (loaded from YAML)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    data_path: str
    split: str = "2025_q4"
    label_scheme: int = 3
    train_fraction: float = 0.8
    seed: int = 42
    first_n: int | None = None


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


class SynthesizerConfig(BaseModel):
    min_cluster_size: int = 3


class ConsolidatorConfig(BaseModel):
    enabled: bool = True
    prune_threshold: int = 2
    protected_names: list[str] = Field(default_factory=lambda: ["generic"])


class OutputConfig(BaseModel):
    dir: str = "out/learning"
    save_epoch_snapshots: bool = True


class LearningRunConfig(BaseModel):
    data: DataConfig
    model: ModelConfig
    blueprints: BlueprintsConfig = BlueprintsConfig()
    learning: LearningConfig = LearningConfig()
    synthesizer: SynthesizerConfig = SynthesizerConfig()
    consolidator: ConsolidatorConfig = ConsolidatorConfig()
    output: OutputConfig = OutputConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> LearningRunConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)
