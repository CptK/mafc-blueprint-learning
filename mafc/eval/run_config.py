"""Pydantic schema for benchmark run configuration (loaded from YAML)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, model_validator


class BenchmarkConfig(BaseModel):
    name: str = "veritas"
    split: str = "2025_q4"
    data_path: str | None = None  # overrides default data/<split> path resolution
    label_scheme: int = 3
    sample_ids: list[str] | None = None
    first_n: int | None = None

    @model_validator(mode="after")
    def _validate_sample_selection(self) -> "BenchmarkConfig":
        if self.sample_ids is not None and self.first_n is not None:
            raise ValueError("only one of sample_ids or first_n can be set, not both")
        return self


class AgentModelConfig(BaseModel):
    model: str
    temperature: float = 1.0
    max_response_length: int = 64000


class FactCheckAgentConfig(AgentModelConfig):
    workers: int = 4  # internal task parallelism within a single fact-check run


class WebSearchAgentConfig(AgentModelConfig):
    workers: int = 4
    max_iterations: int = 2
    max_results_per_query: int = 4


class AgentsConfig(BaseModel):
    fact_check: FactCheckAgentConfig
    web_search: WebSearchAgentConfig
    media: AgentModelConfig
    judge: AgentModelConfig


class BlueprintsConfig(BaseModel):
    selector_model: str
    config_dir: str = "config/blueprints"


class RunConfig(BaseModel):
    concurrency: int = 1
    timeout_per_sample: int | None = None  # seconds; None means no timeout
    traces: bool = True
    log_level: str = "INFO"


class BenchmarkRunConfig(BaseModel):
    benchmark: BenchmarkConfig
    agents: AgentsConfig
    blueprints: BlueprintsConfig
    run: RunConfig = RunConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BenchmarkRunConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)
