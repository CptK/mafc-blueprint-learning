from __future__ import annotations

import pytest
from pydantic import ValidationError

from mafc.blueprints import BlueprintRegistry, load_blueprint, load_blueprints

VALID_BLUEPRINT_YAML = """
name: media_location
description: Verify a claim that depends on image geolocation.
entry_conditions:
  all:
    - feature: has_image
      op: "=="
      value: true
selector_hints:
  positive:
    features: [has_image, has_location]
    exaples:
      - "This photo was taken in Athens."
  negative:
    features: [has_video]
    exaples:
      - "This video shows protests in Paris."
policy_constraints:
  allowed_actions: [media_agent, web_search_agent]
  max_iterations: 3
  require_counterevidence_search: true
required_checks:
  - id: image_real
    description: Retrieved information indicates the image is authentic.
verification_graph:
  start_node: iter1_search
  nodes:
    - id: iter1_search
      type: actions
      actions:
        - action: media_agent
          intent: find likely location
          query_guidance: use visible landmarks
      transition:
        - if: found new evidence
          to: verdict_gate
    - id: verdict_gate
      type: gate
      rules:
        support_conditions: [image_real]
        refute_conditions: [image_real_failed]
        if_fail: return unknown
"""


def test_load_blueprint_from_yaml_file(tmp_path) -> None:
    blueprint_path = tmp_path / "media_location.yaml"
    blueprint_path.write_text(VALID_BLUEPRINT_YAML, encoding="utf-8")

    blueprint = load_blueprint(blueprint_path)

    assert blueprint.name == "media_location"
    assert blueprint.selector_hints.positive.examples == ["This photo was taken in Athens."]
    assert blueprint.policy_constraints.allowed_actions == ["media_agent", "web_search_agent"]
    assert blueprint.verification_graph.start_node == "iter1_search"
    assert blueprint.verification_graph.nodes[0].id == "iter1_search"


def test_load_blueprints_from_directory_returns_sorted_results(tmp_path) -> None:
    second_path = tmp_path / "b.yaml"
    second_path.write_text(VALID_BLUEPRINT_YAML.replace("media_location", "b_blueprint"), encoding="utf-8")
    first_path = tmp_path / "a.json"
    first_path.write_text(
        """
{
  "name": "a_blueprint",
  "description": "JSON blueprint",
  "verification_graph": {
    "start_node": "node1",
    "nodes": [
      {"id": "node1", "type": "synthesis", "transition": []}
    ]
  }
}
""".strip(),
        encoding="utf-8",
    )

    blueprints = load_blueprints(tmp_path)

    assert [blueprint.name for blueprint in blueprints] == ["a_blueprint", "b_blueprint"]


def test_load_blueprint_rejects_missing_transition_target(tmp_path) -> None:
    blueprint_path = tmp_path / "broken.yaml"
    blueprint_path.write_text(
        VALID_BLUEPRINT_YAML.replace("to: verdict_gate", "to: missing_node"),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_blueprint(blueprint_path)


def test_registry_rejects_duplicate_blueprint_names(tmp_path) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text(VALID_BLUEPRINT_YAML, encoding="utf-8")
    second.write_text(
        VALID_BLUEPRINT_YAML.replace("Verify a claim that depends on image geolocation.", "Duplicate name"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="already registered"):
        BlueprintRegistry.from_path(tmp_path)


def test_load_blueprint_rejects_non_mapping_top_level(tmp_path) -> None:
    blueprint_path = tmp_path / "invalid.yaml"
    blueprint_path.write_text("- not-a-mapping", encoding="utf-8")

    with pytest.raises(ValueError, match="mapping at top level"):
        load_blueprint(blueprint_path)
