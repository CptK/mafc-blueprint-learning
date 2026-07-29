from __future__ import annotations

import numpy as np
import pytest

from mafc.blueprints.loader import load_blueprints
from mafc.blueprints.probe import PROBE_FILENAME, BlueprintProbe
from mafc.blueprints.registry import BlueprintRegistry
from mafc.blueprints.selector import (
    BlueprintSelectionMethod,
    BlueprintSelectionMode,
    BlueprintSelector,
)
from mafc.common.modeling.message import Message
from mafc.common.modeling.model import Model, Response

DIM = 8


class CountingModel(Model):
    """Records how often the LLM tie-break was consulted."""

    def __init__(self, output: str = '{"selected_blueprint":"alpha","reason":"r"}'):
        super().__init__(specifier="OPENAI:gpt-5-mini-2025-08-07")
        self.output = output
        self.calls = 0

    def _do_generate(self, messages: list[Message]) -> Response:
        self.calls += 1
        return Response(text=self.output, total_cost=0.0)


def _registry(tmp_path) -> BlueprintRegistry:
    for name in ("alpha", "beta", "generic"):
        (tmp_path / f"{name}.yaml").write_text(
            f"""
name: {name}
description: Blueprint {name}.
verification_graph:
  start_node: s
  nodes:
    - id: s
      type: synthesis
      transition: []
""".strip(),
            encoding="utf-8",
        )
    return BlueprintRegistry(load_blueprints(tmp_path))


def _probe(confident_for: str = "beta", scale: float = 12.0) -> BlueprintProbe:
    """Probe that fires strongly on the first embedding dimension."""
    classes = ["alpha", "beta"]
    coefficients = np.zeros((2, DIM), dtype=np.float32)
    target = classes.index(confident_for)
    coefficients[target, 0] = scale
    coefficients[1 - target, 0] = -scale
    return BlueprintProbe(classes, coefficients, np.zeros(2, dtype=np.float32))


@pytest.fixture
def patched_embed(monkeypatch):
    """Return a fixed embedding, avoiding any network call."""
    vector = np.zeros(DIM, dtype=np.float32)
    vector[0] = 1.0
    monkeypatch.setattr("mafc.blueprints.selector.embed_claim", lambda text, model: vector)
    return vector


def test_llm_tiebreak_is_the_default(tmp_path, patched_embed):
    model = CountingModel()
    selector = BlueprintSelector(model=model, registry=_registry(tmp_path), default_blueprint_name="generic")

    result = selector.select("A claim.")

    assert result.selection_mode == BlueprintSelectionMode.LLM_TIEBREAK
    assert model.calls == 1


def test_probe_method_routes_without_calling_the_llm(tmp_path, patched_embed):
    model = CountingModel()
    selector = BlueprintSelector(
        model=model,
        registry=_registry(tmp_path),
        default_blueprint_name="generic",
        selection_method=BlueprintSelectionMethod.EMBEDDING_PROBE,
        probe=_probe("beta"),
    )

    result = selector.select("A claim.")

    assert result.selected_blueprint.name == "beta"
    assert result.selection_mode == BlueprintSelectionMode.EMBEDDING_PROBE
    assert model.calls == 0


def test_hybrid_defers_to_llm_when_probe_is_unsure(tmp_path, patched_embed):
    model = CountingModel()
    selector = BlueprintSelector(
        model=model,
        registry=_registry(tmp_path),
        default_blueprint_name="generic",
        selection_method=BlueprintSelectionMethod.HYBRID,
        probe=_probe("beta", scale=0.01),  # near 50/50
        probe_confidence_threshold=0.8,
    )

    result = selector.select("A claim.")

    assert result.selection_mode == BlueprintSelectionMode.LLM_TIEBREAK
    assert model.calls == 1


def test_hybrid_uses_probe_when_confident(tmp_path, patched_embed):
    model = CountingModel()
    selector = BlueprintSelector(
        model=model,
        registry=_registry(tmp_path),
        default_blueprint_name="generic",
        selection_method=BlueprintSelectionMethod.HYBRID,
        probe=_probe("beta"),
        probe_confidence_threshold=0.8,
    )

    assert selector.select("A claim.").selection_mode == BlueprintSelectionMode.EMBEDDING_PROBE
    assert model.calls == 0


def test_missing_probe_degrades_to_llm(tmp_path, patched_embed):
    """A routing optimization must never be able to fail a run."""
    model = CountingModel()
    selector = BlueprintSelector(
        model=model,
        registry=_registry(tmp_path),
        default_blueprint_name="generic",
        selection_method=BlueprintSelectionMethod.EMBEDDING_PROBE,
        probe=None,
    )

    assert selector.select("A claim.").selection_mode == BlueprintSelectionMode.LLM_TIEBREAK


def test_failed_embedding_degrades_to_llm(tmp_path, monkeypatch):
    monkeypatch.setattr("mafc.blueprints.selector.embed_claim", lambda text, model: None)
    model = CountingModel()
    selector = BlueprintSelector(
        model=model,
        registry=_registry(tmp_path),
        default_blueprint_name="generic",
        selection_method=BlueprintSelectionMethod.EMBEDDING_PROBE,
        probe=_probe("beta"),
    )

    assert selector.select("A claim.").selection_mode == BlueprintSelectionMode.LLM_TIEBREAK


def test_probe_never_overrides_the_rule_stage(tmp_path, patched_embed):
    """A probe pick eliminated by entry conditions falls back to the best survivor."""
    registry = _registry(tmp_path)
    (tmp_path / "beta.yaml").write_text(
        """
name: beta
description: Blueprint beta.
entry_conditions:
  all:
    - feature: has_image
      op: "=="
      value: true
verification_graph:
  start_node: s
  nodes:
    - id: s
      type: synthesis
      transition: []
""".strip(),
        encoding="utf-8",
    )
    registry = BlueprintRegistry(load_blueprints(tmp_path))
    selector = BlueprintSelector(
        model=CountingModel(),
        registry=registry,
        default_blueprint_name="generic",
        selection_method=BlueprintSelectionMethod.EMBEDDING_PROBE,
        probe=_probe("beta"),
    )

    # Text-only claim: 'beta' is gated out, so only 'alpha' survives the rule stage.
    result = selector.select("A text-only claim.")

    assert result.selected_blueprint.name == "alpha"


def test_probe_artifact_round_trips(tmp_path):
    probe = _probe("beta")
    path = tmp_path / PROBE_FILENAME
    probe.save(path)
    reloaded = BlueprintProbe.load(path)

    vector = np.zeros(DIM, dtype=np.float32)
    vector[0] = 1.0
    assert reloaded.predict(vector).blueprint_name == probe.predict(vector).blueprint_name
    assert reloaded.classes == probe.classes


def test_probe_artifact_is_not_loaded_as_a_blueprint(tmp_path):
    """The artifact lives in the blueprint dir and must not break the loader."""
    _registry(tmp_path)
    _probe("beta").save(tmp_path / PROBE_FILENAME)

    names = {bp.name for bp in load_blueprints(tmp_path)}

    assert names == {"alpha", "beta", "generic"}
