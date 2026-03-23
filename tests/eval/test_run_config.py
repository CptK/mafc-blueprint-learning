import pytest
import yaml

from mafc.eval.run_config import (
    AgentModelConfig,
    AgentsConfig,
    BenchmarkConfig,
    BenchmarkRunConfig,
    BlueprintsConfig,
    FactCheckAgentConfig,
    RunConfig,
    WebSearchAgentConfig,
)

_MODEL = "OPENAI:gpt-4o"


def _minimal_config_dict() -> dict:
    return {
        "benchmark": {"name": "veritas", "split": "2025_q4"},
        "agents": {
            "fact_check": {"model": _MODEL},
            "web_search": {"model": _MODEL},
            "media": {"model": _MODEL},
            "judge": {"model": _MODEL},
        },
        "blueprints": {"selector_model": _MODEL},
    }


def _make_run_config(concurrency: int = 1) -> BenchmarkRunConfig:
    return BenchmarkRunConfig(
        benchmark=BenchmarkConfig(),
        agents=AgentsConfig(
            fact_check=FactCheckAgentConfig(model=_MODEL),
            web_search=WebSearchAgentConfig(model=_MODEL),
            media=AgentModelConfig(model=_MODEL),
            judge=AgentModelConfig(model=_MODEL),
        ),
        blueprints=BlueprintsConfig(selector_model=_MODEL),
        run=RunConfig(concurrency=concurrency),
    )


# ---------------------------------------------------------------------------
# BenchmarkConfig validation
# ---------------------------------------------------------------------------


def test_benchmark_config_rejects_both_sample_ids_and_first_n():
    with pytest.raises(ValueError, match="only one"):
        BenchmarkConfig(sample_ids=["a", "b"], first_n=5)


def test_benchmark_config_allows_sample_ids_alone():
    cfg = BenchmarkConfig(sample_ids=["a", "b"])
    assert cfg.sample_ids == ["a", "b"]
    assert cfg.first_n is None


def test_benchmark_config_allows_first_n_alone():
    cfg = BenchmarkConfig(first_n=10)
    assert cfg.first_n == 10
    assert cfg.sample_ids is None


def test_benchmark_config_allows_neither():
    cfg = BenchmarkConfig()
    assert cfg.sample_ids is None
    assert cfg.first_n is None


def test_benchmark_config_defaults():
    cfg = BenchmarkConfig()
    assert cfg.name == "veritas"
    assert cfg.split == "2025_q4"
    assert cfg.label_scheme == 3
    assert cfg.data_path is None


# ---------------------------------------------------------------------------
# RunConfig defaults
# ---------------------------------------------------------------------------


def test_run_config_defaults():
    cfg = RunConfig()
    assert cfg.concurrency == 1
    assert cfg.timeout_per_sample is None
    assert cfg.traces is True
    assert cfg.log_level == "INFO"


# ---------------------------------------------------------------------------
# BenchmarkRunConfig defaults
# ---------------------------------------------------------------------------


def test_benchmark_run_config_run_defaults():
    cfg = _make_run_config()
    assert cfg.run.concurrency == 1
    assert cfg.run.traces is True
    assert cfg.run.timeout_per_sample is None


def test_agent_model_config_defaults():
    cfg = AgentModelConfig(model=_MODEL)
    assert cfg.temperature == 1.0
    assert cfg.max_response_length == 64000


def test_fact_check_agent_config_defaults():
    cfg = FactCheckAgentConfig(model=_MODEL)
    assert cfg.workers == 4


def test_web_search_agent_config_defaults():
    cfg = WebSearchAgentConfig(model=_MODEL)
    assert cfg.workers == 4
    assert cfg.max_iterations == 2
    assert cfg.max_results_per_query == 4


def test_blueprints_config_defaults():
    cfg = BlueprintsConfig(selector_model=_MODEL)
    assert cfg.config_dir == "config/blueprints"


# ---------------------------------------------------------------------------
# BenchmarkRunConfig.from_yaml
# ---------------------------------------------------------------------------


def test_from_yaml_loads_minimal_config(tmp_path):
    yaml_path = tmp_path / "run.yaml"
    yaml_path.write_text(yaml.dump(_minimal_config_dict()), encoding="utf-8")
    cfg = BenchmarkRunConfig.from_yaml(yaml_path)
    assert cfg.benchmark.name == "veritas"
    assert cfg.agents.fact_check.model == _MODEL
    assert cfg.blueprints.selector_model == _MODEL


def test_from_yaml_overrides_defaults(tmp_path):
    data = _minimal_config_dict()
    data["benchmark"]["first_n"] = 5
    data["run"] = {"concurrency": 8, "traces": False}
    yaml_path = tmp_path / "run.yaml"
    yaml_path.write_text(yaml.dump(data), encoding="utf-8")
    cfg = BenchmarkRunConfig.from_yaml(yaml_path)
    assert cfg.benchmark.first_n == 5
    assert cfg.run.concurrency == 8
    assert cfg.run.traces is False
